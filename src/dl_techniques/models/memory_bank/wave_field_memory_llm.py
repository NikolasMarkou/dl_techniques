"""WaveFieldMemoryLLM — sibling-stack memory-augmented LM.

Mirrors :class:`WaveFieldLLM` but inserts two taps into the
``WaveFieldDecoderBlock`` list:

- After block ``L_write - 1``: :class:`MemoryWriteController` projects the
  pre-block hidden state into right-padded ``(K_wm, V_wm)``.
- After block ``L_read - 1``: :class:`MemoryReadController` queries against
  ``[M_LT ; M_WM]``, performs ST top-K retrieval, and adds a gated
  injection (and 4 anti-collapse aux losses in Phase 2+).

Key implementation notes (per F-002, F-004, LESSONS):

- Existing :class:`WaveFieldDecoderBlock` is reused **verbatim** — zero
  modification.
- Phase counter and global step live as ``add_weight(trainable=False)`` so
  they survive ``model.save`` / ``load_model`` round-trips.
- The whole curriculum reads off that phase counter **inside the traced
  graph**: memory injection, the anti-collapse aux losses and the
  phase-2 backbone freeze are all multiplicative gates derived from
  ``current_phase``. Nothing about the curriculum is a Python attribute
  a callback flips — ``fit()`` traces ``train_function`` before
  ``on_train_begin`` and rebuilds it only from ``compile()``, so a
  Python flip after the first batch never reaches the graph.
- Custom :meth:`train_step` splits trainable variables by the leading
  component of ``Variable.path`` — the Keras 3 layer-qualified property —
  falling back to the bare ``Variable.name`` for weights the model creates
  on itself (``memory_`` / ``gate_`` -> memory optimizer; everything else
  -> backbone optimizer). ``Variable.name`` alone is **not** the
  layer-qualified name under Keras 3; see
  :func:`split_trainable_by_prefix`.
- :meth:`compile` accepts both ``backbone_optimizer`` and
  ``memory_optimizer`` and registers the backbone with Keras while
  keeping the memory optimizer as a model attribute.
- :meth:`warmup_memory_keys` runs offline ``MiniBatchKMeans`` on hidden
  states at the read tap and seeds ``K_lt``.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)
from dl_techniques.initializers.identity_plus_noise import (
    IdentityPlusNoise,
)
from dl_techniques.models.wave_field.model import (
    WaveFieldDecoderBlock,
)
from dl_techniques.models.memory_bank.memory_banks import (
    LongTermMemoryBank,
    WorkingMemoryBank,
)
from dl_techniques.models.memory_bank.write_controller import (
    MemoryWriteController,
)
from dl_techniques.models.memory_bank.read_controller import (
    MemoryReadController,
    _DEAD_INFONCE_TEMPERATURE_KEY,
)
from dl_techniques.models.memory_bank.phase_scheduler import (
    PHASE_WARMUP,
    PHASE_FREEZE_BACKBONE,
)


# ---------------------------------------------------------------------


def linear_top_k_anneal(
    start: int, end: int, end_step: int,
) -> Callable[[int], int]:
    """O7 helper: linear anneal of `top_k` from `start` to `end` over the
    first `end_step` training steps. After `end_step`, returns `end`.
    Returns a callable suitable for `WaveFieldMemoryLLM(top_k_schedule=...)`.
    """
    if end_step <= 0:
        raise ValueError(f"end_step must be positive, got {end_step}")

    def schedule(step: int) -> int:
        if step >= end_step:
            return int(end)
        frac = float(step) / float(end_step)
        return int(round(start + (end - start) * frac))

    return schedule


def _scale_gradient(grad: Any, scale: Any) -> Any:
    """Multiply ``grad`` by scalar ``scale``, preserving sparsity.

    The token-embedding gradient arrives as a ``tf.IndexedSlices``; the
    plain ``grad * scale`` spelling densifies it, which for a
    ``(vocab_size, embed_dim)`` table is a per-step allocation of tens of
    millions of floats. Scaling ``.values`` keeps the sparse form.

    :param grad: A dense gradient tensor or a ``tf.IndexedSlices``.
    :param scale: Scalar tensor; cast to the gradient's dtype.
    :returns: The scaled gradient, same kind as the input.
    """
    if isinstance(grad, tf.IndexedSlices):
        return tf.IndexedSlices(
            grad.values * ops.cast(scale, grad.values.dtype),
            grad.indices,
            grad.dense_shape,
        )
    return grad * ops.cast(scale, grad.dtype)


def split_trainable_by_prefix(
    variables: List[Any],
    memory_prefixes: Tuple[str, ...] = ("memory_", "gate_"),
) -> Tuple[List[Any], List[Any]]:
    """Partition ``variables`` into ``(memory_vars, backbone_vars)``.

    Under Keras 3 a variable carries **two** names, and they are not the
    same string: ``Variable.name`` is the bare weight name handed to
    ``add_weight`` (``"kernel"``, ``"bias"``, ``"gamma"``), while the
    layer-qualified hierarchy lives in the separate ``Variable.path``
    property (``"memory_read_controller/gate_W_g/kernel"``). Matching
    ``.name`` therefore sees only weights created by a literal
    ``add_weight(name="memory_...")`` call and never sees a sublayer's
    weights, whatever its parent layer is called.

    A variable is routed to memory when **either**:

    - the **leading component of** ``.path`` starts with a prefix — the
      memory subtrees (``memory_lt_bank``, ``memory_read_controller``,
      ``memory_write_controller``) are top-level attributes of the model,
      so every weight beneath them, including the read gate
      ``memory_read_controller/gate_W_g/kernel``, matches here; or
    - the bare ``.name`` starts with a prefix — this covers weights the
      model creates on **itself** via ``add_weight``, whose path is
      prefixed by the model's own name (``wave_field_memory_llm/
      memory_global_step``) and so has a non-matching leading component.

    Matching *any* path component is deliberately **not** done: the
    backbone's :class:`WaveFieldAttention` contains a ``gate_proj`` Dense
    in every block, so an any-component rule would hand 2 variables per
    block to the memory optimizer.

    :param variables: Keras 3 variables to partition.
    :param memory_prefixes: Prefixes marking the memory partition.
    :returns: ``(memory_vars, backbone_vars)``. Variables with an empty or
        missing path and name are routed to backbone (defensive).
    """
    memory_vars: List[Any] = []
    backbone_vars: List[Any] = []
    for v in variables:
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-015 — match on
        # `.path` (leading component) plus the bare `.name`. Do NOT go
        # back to `.name` alone: under Keras 3 that is the bare weight
        # name, so every Dense/LayerNormalization weight in the memory
        # controllers — the read gate above all — silently trained on the
        # backbone optimizer. And do NOT relax to an any-component path
        # match: `block_*/attention/gate_proj/*` would be captured.
        path = getattr(v, "path", "") or ""
        name = getattr(v, "name", "") or ""
        head = path.split("/", 1)[0]
        if any(head.startswith(p) or name.startswith(p)
               for p in memory_prefixes):
            memory_vars.append(v)
        else:
            backbone_vars.append(v)
    return memory_vars, backbone_vars


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldMemoryLLM(keras.Model):
    """Memory-augmented WaveFieldLLM with dual-tap topology.

    Memory hyperparameters scale per variant (see :data:`MODEL_VARIANTS`).

    .. note::

       **Incremental decoding (KV-cache) is not supported.**

       DECISION plan_2026-05-09_0f39a086/D-002 — the audit's O1 ("add a
       KV cache to support incremental decoding") was investigated and
       deferred. The backbone uses :class:`WaveFieldAttention`, which is
       FFT-based and recomputes the full sequence per call. Adding
       single-token incremental decoding requires modifying
       :class:`WaveFieldDecoderBlock` (or replacing the attention layer
       with a streaming variant), which is outside the scope of the
       memory-bank package. Two future paths:

       (a) Add a streaming variant of WaveFieldAttention that maintains
           a rolling spectral cache.

       (b) Replace the backbone with a standard MHA decoder for serving
           and keep the wave-field backbone for training only.

       The memory-bank read/write controllers themselves are
       incremental-friendly (single-token retrieval is O(top_k) against
       a static M_static), but they cannot be exercised incrementally
       without an incremental backbone.
    """

    DEFAULT_VOCAB_SIZE = 50261
    DEFAULT_INITIALIZER_RANGE = 0.02
    DEFAULT_LAYER_NORM_EPS = 1e-5

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "embed_dim": 256, "depth": 4, "num_heads": 4,
            "max_seq_len": 512, "field_size": 1024,
            "d_k": 64, "d_v": 128, "s_lt": 4096,
            "description": "WaveFieldMemoryLLM Tiny",
        },
        "small": {
            "embed_dim": 768, "depth": 12, "num_heads": 12,
            "max_seq_len": 1024, "field_size": 2048,
            "d_k": 128, "d_v": 256, "s_lt": 16384,
            "description": "WaveFieldMemoryLLM Small",
        },
        "medium": {
            "embed_dim": 1024, "depth": 24, "num_heads": 16,
            "max_seq_len": 1024, "field_size": 2048,
            "d_k": 128, "d_v": 512, "s_lt": 32768,
            "description": "WaveFieldMemoryLLM Medium",
        },
        "large": {
            "embed_dim": 1280, "depth": 36, "num_heads": 20,
            "max_seq_len": 1024, "field_size": 2048,
            "d_k": 128, "d_v": 512, "s_lt": 65536,
            "description": "WaveFieldMemoryLLM Large",
        },
        "xl": {
            "embed_dim": 1600, "depth": 48, "num_heads": 25,
            "max_seq_len": 1024, "field_size": 2048,
            "d_k": 128, "d_v": 512, "s_lt": 65536,
            "description": "WaveFieldMemoryLLM XL",
        },
    }

    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 1024,
        field_size: Optional[int] = None,
        # Memory hyperparameters.
        d_k: int = 128,
        d_v: int = 256,
        s_lt: int = 16384,
        top_k: int = 32,
        gate_init_bias: float = -3.0,
        # Aux loss coefficients.
        lambda_gate_entropy: float = 1e-3,
        lambda_load_balance: float = 1e-2,
        lambda_z_loss: float = 1e-3,
        lambda_diversity: float = 1e-3,
        lambda_infonce: float = 5e-3,
        lambda_v_diversity: float = 1e-3,
        diversity_subsample: int = 1024,
        infonce_negatives: int = 256,
        # O6 — opt-in V_lt diversity aux loss. Default False so existing
        # variants and tests are unaffected.
        enable_v_diversity: bool = False,
        # C-28 — emit the four anti-collapse aux losses into the graph.
        # They are then scaled by the `current_phase` gate, so Phase 1
        # contributes exactly 0.0 and Phase 2+ contributes the real term.
        # `False` removes them from the graph entirely and no curriculum
        # can bring them back (see `MemoryReadController`'s class
        # docstring).
        enable_aux_losses: bool = True,
        # O7 — optional schedule for `read_controller.top_k`. A callable
        # `step -> int` that returns the new top_k for a given training
        # step. Applied by `PhaseScheduler.on_train_batch_begin` only on
        # phase transitions (cheap retrace boundary). NOT serialized
        # (callables can't round-trip via get_config).
        top_k_schedule: Optional[Callable[[int], int]] = None,
        # O4 — opt-in per-head keys/values. Default False keeps MQA
        # behavior bit-exact.
        multi_head_keys: bool = False,
        # Common transformer dropout / norm params.
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = DEFAULT_INITIALIZER_RANGE,
        layer_norm_eps: float = DEFAULT_LAYER_NORM_EPS,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if field_size is None:
            field_size = 2 * max_seq_len

        # Compute and validate L_write / L_read.
        l_write = max(1, depth // 3)
        l_read = max(l_write + 1, (2 * depth) // 3)
        if not (l_write < l_read < depth):
            raise ValueError(
                f"Invalid tap topology: L_write={l_write}, L_read={l_read}, "
                f"depth={depth}. Need L_write < L_read < depth."
            )
        if d_v >= embed_dim:
            raise ValueError(f"d_v ({d_v}) must be < embed_dim ({embed_dim})")
        if d_k == d_v:
            raise ValueError("d_k must differ from d_v")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.field_size = field_size
        self.d_k = d_k
        self.d_v = d_v
        self.s_lt = s_lt
        self.top_k = top_k
        self.gate_init_bias = gate_init_bias
        self.lambda_gate_entropy = lambda_gate_entropy
        self.lambda_load_balance = lambda_load_balance
        self.lambda_z_loss = lambda_z_loss
        self.lambda_diversity = lambda_diversity
        self.lambda_infonce = lambda_infonce
        self.lambda_v_diversity = lambda_v_diversity
        self.diversity_subsample = diversity_subsample
        self.infonce_negatives = infonce_negatives
        self.enable_v_diversity = enable_v_diversity
        self.enable_aux_losses = enable_aux_losses
        self.top_k_schedule = top_k_schedule
        self.multi_head_keys = multi_head_keys
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        self.L_write = l_write
        self.L_read = l_read

        self._build_architecture()

        # Phase counter + global step (survive save/load). Stored as
        # float32 so they live on the same device (GPU) as the rest of
        # the train_step graph — int32/int64 variables go to CPU by
        # default since most int kernels are CPU-only, and Keras 3 / TF
        # 2.18 errors on cross-device resource access from the compiled
        # multi_step_on_iterator. Callers that need integer values cast
        # via `int(self.current_phase.numpy())`.
        self.current_phase = self.add_weight(
            name="memory_current_phase",
            shape=(),
            initializer=keras.initializers.Constant(1.0),
            trainable=False,
            dtype="float32",
        )
        self._global_step = self.add_weight(
            name="memory_global_step",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )

        # Optimizers (set by compile()).
        self.backbone_optimizer = None
        self.memory_optimizer = None
        self._backbone_base_lr: Optional[float] = None

        logger.info(
            f"Created WaveFieldMemoryLLM: depth={depth}, embed_dim={embed_dim}, "
            f"L_write={l_write}, L_read={l_read}, d_k={d_k}, d_v={d_v}, "
            f"s_lt={s_lt}, top_k={top_k}"
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_architecture(self) -> None:
        kernel_init = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range,
        )

        self.token_embeddings = keras.layers.Embedding(
            self.vocab_size, self.embed_dim,
            embeddings_initializer=kernel_init, name="token_embeddings",
        )
        self.position_embeddings = keras.layers.Embedding(
            self.max_seq_len, self.embed_dim,
            embeddings_initializer=kernel_init, name="position_embeddings",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="embed_dropout",
        )

        self.blocks = [
            WaveFieldDecoderBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                max_seq_len=self.max_seq_len,
                field_size=self.field_size,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                initializer_range=self.initializer_range,
                name=f"block_{i}",
            )
            for i in range(self.depth)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, name="final_norm",
        )

        if not self.tie_word_embeddings:
            self.lm_head = keras.layers.Dense(
                self.vocab_size, use_bias=False,
                kernel_initializer=kernel_init, name="lm_head",
            )
        else:
            self.lm_head = None

        # Memory components. O4 plumbing: pass num_heads + multi_head_keys
        # so the LT bank, write controller (and its inner WM bank), and
        # read controller all agree on shape semantics.
        self.lt_memory = LongTermMemoryBank(
            s_lt=self.s_lt, d_k=self.d_k, d_v=self.d_v,
            initializer_range=self.initializer_range,
            num_heads=self.num_heads,
            multi_head_keys=self.multi_head_keys,
            name="memory_lt_bank",
        )
        self.write_controller = MemoryWriteController(
            d_k=self.d_k, d_v=self.d_v, embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            initializer_range=self.initializer_range,
            num_heads=self.num_heads,
            multi_head_keys=self.multi_head_keys,
            name="memory_write_controller",
        )
        self.read_controller = MemoryReadController(
            embed_dim=self.embed_dim, num_heads=self.num_heads,
            d_k=self.d_k, d_v=self.d_v,
            s_lt=self.s_lt, max_seq_len=self.max_seq_len,
            top_k=self.top_k,
            initializer_range=self.initializer_range,
            gate_init_bias=self.gate_init_bias,
            layer_norm_eps=self.layer_norm_eps,
            lambda_gate_entropy=self.lambda_gate_entropy,
            lambda_load_balance=self.lambda_load_balance,
            lambda_z_loss=self.lambda_z_loss,
            lambda_diversity=self.lambda_diversity,
            lambda_infonce=self.lambda_infonce,
            lambda_v_diversity=self.lambda_v_diversity,
            diversity_subsample=self.diversity_subsample,
            infonce_negatives=self.infonce_negatives,
            enable_v_diversity=self.enable_v_diversity,
            # DECISION plan-2026-08-14T233721-d4f9beb2/D-016 — the aux
            # losses are enabled STATICALLY here, at construction, because
            # the flags decide graph shape. The curriculum switches them
            # at runtime with the `aux_scale` tensor `call` passes down.
            # Do NOT hand these flags to `PhaseScheduler` to flip.
            enable_gate_entropy=self.enable_aux_losses,
            enable_load_balance=self.enable_aux_losses,
            enable_z_loss=self.enable_aux_losses,
            enable_diversity=self.enable_aux_losses,
            enable_infonce=self.enable_aux_losses,
            multi_head_keys=self.multi_head_keys,
            name="memory_read_controller",
        )

        # Eagerly build sub-layers to pin variable creation to model
        # construction time (parity with WaveFieldLLM).
        block_input_shape: Tuple[Optional[int], ...] = (
            None, self.max_seq_len, self.embed_dim,
        )
        self.token_embeddings.build((None, self.max_seq_len))
        self.position_embeddings.build((self.max_seq_len,))
        self.embed_norm.build(block_input_shape)
        for block in self.blocks:
            block.build(block_input_shape)
        self.final_norm.build(block_input_shape)
        if self.lm_head is not None:
            self.lm_head.build(block_input_shape)

        self.lt_memory.build()
        self.write_controller.build(block_input_shape)
        self.read_controller.build(block_input_shape)

        # Convenience alias: PhaseScheduler reads `wm_memory` to flip
        # the working-memory bank's trainable flag.
        self.wm_memory = self.write_controller.wm_bank

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Union[Any, Dict[str, Any]],
        attention_mask: Optional[Any] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("dict input must contain 'input_ids'")
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(seq_len, dtype="int32")

        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        x = token_emb + pos_emb
        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        # Phase 1 (PHASE_WARMUP) disables memory entirely.
        # current_phase is float32; cast PHASE_WARMUP to match.
        #
        # R1 design note: the plan considered guarding the read pass
        # under `not training` with `ops.cond(memory_active, ...)` to
        # skip retrieval at eval time in P1. Two reasons we keep the
        # multiply-by-zero pattern instead:
        #   (1) `keras.ops.cond` inside a tf.function-compiled graph
        #       traces BOTH branches in TF backend Keras 3. The
        #       "skip retrieval" branch would still pay the trace cost
        #       and would not actually skip the kernel launch.
        #   (2) The retrieval kernels are small relative to backbone
        #       attention; the savings are not worth the divergence
        #       between training and eval graphs (which would also
        #       complicate save/load by changing call-time behavior).
        # Keep the gate-by-zero. P1 add_loss calls are gated by the
        # `if training` block and the per-flag enables in
        # `MemoryReadController._maybe_add_aux_losses`, so eval-time
        # forward in P1 is correct even with retrieval running.
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-016 — one gate, read
        # from the `current_phase` Variable on every traced step, drives
        # BOTH the injection and the anti-collapse aux losses. The two were
        # separate mechanisms before: the injection used this gate (and
        # worked), while the aux losses used Python bools a callback
        # flipped after the trace (and never fired once). Do not
        # reintroduce a Python-attribute switch for either.
        memory_active = ops.not_equal(
            self.current_phase, ops.cast(PHASE_WARMUP, "float32"),
        )
        aux_scale = ops.cast(memory_active, "float32")

        k_wm = None
        v_wm = None
        wm_mask = None

        for i, block in enumerate(self.blocks):
            if i == self.L_write:
                k_wm, v_wm, wm_mask = self.write_controller(x, training=training)
            if i == self.L_read:
                # Always run the read pass (to keep the graph static); then
                # gate the injection by `memory_active`.
                k_lt, v_lt = self.lt_memory(None)
                injection = self.read_controller(
                    x, k_lt, v_lt, k_wm, v_wm, wm_mask,
                    aux_scale=aux_scale, training=training,
                )
                injection = injection * ops.cast(memory_active, injection.dtype)
                x = x + injection
            x = block(x, attention_mask=attention_mask, training=training)

        x = self.final_norm(x)

        if self.tie_word_embeddings:
            embedding_weights = self.token_embeddings.embeddings
            logits = ops.matmul(x, ops.transpose(embedding_weights))
        else:
            logits = self.lm_head(x)

        return {"logits": logits, "last_hidden_state": x}

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {
            "logits": (*input_shape, self.vocab_size),
            "last_hidden_state": (*input_shape, self.embed_dim),
        }

    # ------------------------------------------------------------------
    # Compile / train_step (dual optimizer)
    # ------------------------------------------------------------------

    def compile(
        self,
        backbone_optimizer: Optional[keras.optimizers.Optimizer] = None,
        memory_optimizer: Optional[keras.optimizers.Optimizer] = None,
        **kwargs: Any,
    ) -> None:
        """Register the backbone optimizer with Keras and store the memory
        optimizer for manual application inside :meth:`train_step`."""
        if backbone_optimizer is None:
            raise ValueError("backbone_optimizer must be provided")
        if memory_optimizer is None:
            raise ValueError("memory_optimizer must be provided")
        self.backbone_optimizer = backbone_optimizer
        self.memory_optimizer = memory_optimizer
        # Remembered so `set_backbone_optimizer_active(True)` can restore
        # the rate the caller asked for after a phase-2 freeze.
        self._backbone_base_lr: Optional[float] = None
        lr = getattr(backbone_optimizer, "_learning_rate", None)
        if isinstance(lr, keras.Variable):
            self._backbone_base_lr = float(ops.convert_to_numpy(lr))
        super().compile(optimizer=backbone_optimizer, **kwargs)

    def set_backbone_optimizer_active(self, active: bool) -> None:
        """Gate the backbone optimizer's learning-rate ``Variable``.

        This is the eager half of the phase-2 backbone freeze; the in-graph
        half is the gradient mask in :meth:`train_step`. The mask stops new
        gradient from reaching the backbone, but a gradient of exactly zero
        is **not** a freeze under Adam/AdamW: the moment estimates carried
        over from phase 1 keep producing a non-zero
        ``lr * m_hat / (sqrt(v_hat) + eps)`` update, and AdamW's decay term
        keeps shrinking the weights. Zeroing the learning rate closes both,
        exactly (every term of both updates is multiplied by ``lr``).

        The learning rate is a ``keras.Variable`` read inside the traced
        step, so assigning it here takes effect on the very next batch
        without a retrace.

        :param active: ``True`` restores the rate passed to
            :meth:`compile`; ``False`` sets it to ``0.0``.
        :raises ValueError: If the backbone optimizer was built with a
            ``LearningRateSchedule`` (or any non-``Variable`` rate), which
            cannot be assigned and therefore cannot be frozen.
        """
        opt = self.backbone_optimizer
        if opt is None:
            return
        lr = getattr(opt, "_learning_rate", None)
        if not isinstance(lr, keras.Variable):
            raise ValueError(
                "PhaseScheduler's phase-2 backbone freeze needs an "
                "assignable learning rate, but this model's "
                f"backbone_optimizer carries {type(lr).__name__}. Compile "
                "the backbone optimizer with a float learning_rate (drive "
                "any schedule from a callback instead); a "
                "LearningRateSchedule cannot be zeroed, so the backbone "
                "would keep training through phase 2."
            )
        lr.assign(self._backbone_base_lr if active else 0.0)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        # DECISION plan-2026-08-17T183311-79c63e38/D-034
        # NO `optimizer.scale_loss(loss)` call here, DELIBERATELY -- unlike
        # every other custom `train_step` in this repo, where omitting it is a
        # ~32768x silent under-update under `mixed_float16`. This model is the
        # measured exception, for two independent reasons:
        #   1. Gradients are applied through `self.backbone_optimizer` and
        #      `self.memory_optimizer` -- the RAW optimizer objects. Keras'
        #      `auto_scale_loss` only wraps `self.optimizer`, so neither of
        #      these is ever a `LossScaleOptimizer`, neither unscales, and
        #      calling `scale_loss` on them is an unconditional no-op. Adding
        #      the call would look like a fix and change nothing.
        #   2. The model cannot execute under `mixed_float16` at all today:
        #      `MemoryReadController.call()` raises `InvalidArgumentError` on a
        #      dtype mismatch before a single step completes (measured).
        # Making fp16 correct here means wrapping BOTH optimizers by hand and
        # reconciling their two independent dynamic scales -- a design change,
        # not a one-line fix, and unverifiable until (2) is resolved. Flagged
        # as a scoped follow-up in decisions.md D-034; do not "just add
        # scale_loss".
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # R3+R4: drop None-gradient pairs first, then route via
        # `split_trainable_by_prefix` (leading `.path` component, or the
        # bare `.name` for model-owned weights). Keeps routing logic in
        # one place; train_step no longer encodes the prefix policy
        # inline.
        live = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if live:
            live_grads = [g for g, _ in live]
            live_vars = [v for _, v in live]
            mem_vars, bb_vars = split_trainable_by_prefix(live_vars)
            mem_set = {id(v) for v in mem_vars}
            memory_pairs = [
                (g, v) for g, v in zip(live_grads, live_vars) if id(v) in mem_set
            ]
            backbone_pairs = [
                (g, v) for g, v in zip(live_grads, live_vars)
                if id(v) not in mem_set
            ]
        else:
            memory_pairs, backbone_pairs = [], []

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-016 — the phase-2
        # backbone freeze is this mask, read from `current_phase` on every
        # traced step, NOT a `layer.trainable = False` flip in the
        # curriculum callback. `self.trainable_variables` above is a Python
        # list evaluated once when `train_function` is traced, so a
        # `.trainable` flip at a later phase boundary cannot remove
        # anything from it. Keeping the mask here (rather than only zeroing
        # the backbone learning rate) makes the freeze a property of the
        # SAVED phase counter: a model reloaded mid-curriculum freezes
        # correctly with no callback attached. See decisions.md D-016.
        backbone_active = ops.cast(
            ops.not_equal(
                self.current_phase,
                ops.cast(PHASE_FREEZE_BACKBONE, "float32"),
            ),
            "float32",
        )
        backbone_pairs = [
            (_scale_gradient(g, backbone_active), v)
            for g, v in backbone_pairs
        ]

        if backbone_pairs:
            self.backbone_optimizer.apply_gradients(backbone_pairs)
        if memory_pairs:
            self.memory_optimizer.apply_gradients(memory_pairs)

        self._global_step.assign_add(tf.constant(1.0, dtype="float32"))

        # B5: dict-keyed forward + dict-keyed compile must work for
        # non-loss metrics. The Keras `CompileMetrics` container expects
        # update_state(y, y_pred) with the same dict structure compile()
        # received. The "loss" tracker takes the scalar loss. After
        # updating state, flatten CompileMetrics.result() (which returns
        # a dict of inner-metric-name -> tensor) into the top-level
        # output so e.g. 'acc' appears in `history.history`.
        for metric in self.metrics:
            mname = getattr(metric, "name", "")
            if mname == "loss":
                metric.update_state(loss)
            else:
                # CompileMetrics handles dict-keyed routing internally;
                # any other Metric instance gets the (y, y_pred) raw and
                # is expected to handle dicts (most don't, so users
                # should compile via metrics={"logits": [...]} which
                # routes through CompileMetrics).
                metric.update_state(y, y_pred)

        out: Dict[str, Any] = {}
        for m in self.metrics:
            r = m.result()
            if isinstance(r, dict):
                out.update(r)
            else:
                out[m.name] = r
        return out

    # ------------------------------------------------------------------
    # KMeans warmup
    # ------------------------------------------------------------------

    def warmup_memory_keys(
        self,
        dataset: Any,
        num_batches: int = 64,
    ) -> None:
        """Seed ``K_lt`` from offline ``MiniBatchKMeans`` on hidden states.

        Collects the read-tap hidden state on ``num_batches`` batches with
        the model in Phase-1 mode (memory bypassed), runs
        ``sklearn.cluster.MiniBatchKMeans`` with ``n_clusters=s_lt``, and
        calls :meth:`LongTermMemoryBank.assign_keys_from_kmeans`.

        On any exception the warmup falls back to leaving ``K_lt`` at its
        ``RandomNormal`` init (logged warning).
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            logger.warning(
                "warmup_memory_keys: scikit-learn not available; "
                "K_lt remains at RandomNormal init"
            )
            return

        # Run the warmup with current_phase forced to PHASE_WARMUP so the
        # read tap is bypassed (we still build all variables; we just skip
        # the memory contribution).
        prev_phase = int(self.current_phase.numpy())
        self.current_phase.assign(PHASE_WARMUP)

        try:
            hiddens: List[np.ndarray] = []
            count = 0
            for batch in dataset.take(num_batches):
                if isinstance(batch, (tuple, list)):
                    x_batch = batch[0]
                else:
                    x_batch = batch
                # Forward through embed + first L_read blocks to get the
                # hidden state at the read tap.
                h = self._hidden_at_read_tap(x_batch)
                hiddens.append(np.asarray(h).reshape(-1, self.embed_dim))
                count += 1

            if not hiddens:
                logger.warning(
                    "warmup_memory_keys: dataset yielded zero batches; skipping"
                )
                return

            stacked = np.concatenate(hiddens, axis=0)
            n_clusters = self.s_lt
            if stacked.shape[0] < n_clusters:
                logger.warning(
                    f"warmup_memory_keys: only {stacked.shape[0]} hiddens for "
                    f"{n_clusters} clusters; tiling input"
                )
                reps = (n_clusters + stacked.shape[0] - 1) // stacked.shape[0]
                stacked = np.tile(stacked, (reps, 1))[: n_clusters * 2]

            # B1 — KMeans must condition K_lt against the actual query
            # direction the read controller will project hiddens into,
            # otherwise the centroids are essentially random in d_k space
            # and the warmup adds no information. Project hiddens through
            # the head-averaged W_Q kernel BEFORE clustering, then run
            # KMeans in d_k space directly.
            #
            # W_Q.kernel shape: (embed_dim, num_heads * d_k).
            # Reshape to (D, H, d_k) and mean over heads -> (D, d_k).
            # This requires read_controller.W_Q.built (which holds since
            # `_build_architecture` eagerly builds all sublayers).
            wq_kernel = np.asarray(self.read_controller.W_Q.kernel)
            assert wq_kernel.shape == (self.embed_dim, self.num_heads * self.d_k), (
                f"unexpected W_Q kernel shape {wq_kernel.shape}"
            )
            wq_avg = (
                wq_kernel
                .reshape(self.embed_dim, self.num_heads, self.d_k)
                .mean(axis=1)  # (D, d_k) — head-averaged Q projection
                .astype(np.float32)
            )
            stacked_dk = stacked @ wq_avg  # (N, d_k)

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(4096, max(256, stacked_dk.shape[0] // 4)),
                n_init=1,
                max_iter=20,
                random_state=0,
            )
            kmeans.fit(stacked_dk)
            centroids_dk = kmeans.cluster_centers_.astype(np.float32)
            self.lt_memory.assign_keys_from_kmeans(centroids_dk)
            logger.info(
                f"warmup_memory_keys: seeded K_lt with {n_clusters} centroids "
                f"from {stacked.shape[0]} hidden states"
            )
        except Exception as exc:
            logger.warning(
                f"warmup_memory_keys: failed ({exc}); K_lt remains at "
                f"RandomNormal init"
            )
        finally:
            self.current_phase.assign(prev_phase)

    def _hidden_at_read_tap(self, input_ids: Any) -> Any:
        """Compute hidden state at the read tap (read-only, no aux losses)."""
        if isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids", input_ids)
        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(seq_len, dtype="int32")
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embed_norm(x)
        for i, block in enumerate(self.blocks):
            if i == self.L_read:
                return x
            x = block(x, attention_mask=None, training=False)
        return x

    # ------------------------------------------------------------------
    # O3 — reset memory state
    # ------------------------------------------------------------------

    def reset_memory(self, seed: Optional[int] = None) -> None:
        """Re-initialize ``K_lt`` and ``V_lt`` from
        ``RandomNormal(stddev=initializer_range)``, set
        ``current_phase`` to ``PHASE_WARMUP`` and ``_global_step`` to
        zero. Useful for restarting curriculum or running ablations
        without rebuilding the model.

        :param seed: Optional seed for the random init. If ``None``,
            the model's existing init RNG state is used (Keras
            generates a fresh seed each call).
        """
        gen = keras.random.SeedGenerator(seed=seed) if seed is not None else None

        def _normal(shape):
            kwargs = {"stddev": self.initializer_range}
            if gen is not None:
                kwargs["seed"] = gen
            return keras.random.normal(shape, **kwargs)

        # K_lt and V_lt live on the LongTermMemoryBank.
        self.lt_memory.K_lt.assign(_normal(self.lt_memory.K_lt.shape))
        self.lt_memory.V_lt.assign(_normal(self.lt_memory.V_lt.shape))

        self.current_phase.assign(float(PHASE_WARMUP))
        self._global_step.assign(0.0)

        logger.info(
            f"reset_memory: K_lt/V_lt re-initialized; phase->{PHASE_WARMUP}, "
            f"step->0"
        )

    # ------------------------------------------------------------------
    # Config + factory
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "field_size": self.field_size,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "s_lt": self.s_lt,
            "top_k": self.top_k,
            "gate_init_bias": self.gate_init_bias,
            "lambda_gate_entropy": self.lambda_gate_entropy,
            "lambda_load_balance": self.lambda_load_balance,
            "lambda_z_loss": self.lambda_z_loss,
            "lambda_diversity": self.lambda_diversity,
            "lambda_infonce": self.lambda_infonce,
            "lambda_v_diversity": self.lambda_v_diversity,
            "diversity_subsample": self.diversity_subsample,
            "infonce_negatives": self.infonce_negatives,
            "enable_v_diversity": self.enable_v_diversity,
            "enable_aux_losses": self.enable_aux_losses,
            "multi_head_keys": self.multi_head_keys,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
        })
        return config

    @classmethod
    def from_variant(
        cls,
        variant: str,
        **overrides: Any,
    ) -> "WaveFieldMemoryLLM":
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(overrides)
        return cls(**config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WaveFieldMemoryLLM":
        """Rebuild from a ``get_config()`` dict.

        Every value ``get_config`` emits is a plain scalar, so this is a
        straight ``cls(**config)``. It is spelled out rather than inherited to
        record the one constructor argument that deliberately does NOT survive
        a round-trip: ``top_k_schedule`` is a Python callable, is therefore
        absent from ``get_config``, and a reloaded model runs with the fixed
        ``top_k``. Re-attach it after loading if you were using one.

        It also drops ``infonce_temperature`` from configs saved before
        2026-08-19, when that dead argument was removed (see
        :data:`~dl_techniques.models.memory_bank.read_controller._DEAD_INFONCE_TEMPERATURE_KEY`).
        """
        if _DEAD_INFONCE_TEMPERATURE_KEY in config:
            config = {k: v for k, v in config.items()
                      if k != _DEAD_INFONCE_TEMPERATURE_KEY}
            logger.warning(
                f"WaveFieldMemoryLLM: ignoring legacy config key "
                f"'{_DEAD_INFONCE_TEMPERATURE_KEY}'; the InfoNCE temperature "
                f"is the learned `log_temp_nce` weight and always was."
            )
        return cls(**config)


# ---------------------------------------------------------------------


# DECISION plan-2026-08-10T130454-3649c19e/D-014
# Do NOT key this dict by bare class name (`"WaveFieldAttention": ...`), which
# is what it did before. Keras looks a serialized CLASS up by its
# `registered_name` field, not by `class_name`
# (`keras/src/saving/serialization_lib.py::_retrieve_class_or_fn` calls
# `get_registered_object(registered_name, custom_objects=...)`). Every class
# here is decorated with a bare `@keras.saving.register_keras_serializable()`,
# so its registered name is `"Custom>ClassName"` and a bare-name key can never
# match. Deriving the keys with `keras.saving.get_registered_name` keeps the
# registration decorator as the single source of that fact, so adding a
# `package=`/`name=` argument to any of these classes cannot silently
# desynchronize this dict. See decisions.md D-014.
def memory_llm_custom_objects() -> Dict[str, Any]:
    """Return the ``custom_objects`` dict needed by ``keras.models.load_model``
    to deserialize a saved :class:`WaveFieldMemoryLLM`.

    Keys are the Keras *registered* names (``"Custom>WaveFieldAttention"``,
    ...), derived from each class rather than hard-coded, because that is the
    string Keras actually looks up when deserializing a class.
    """
    from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
    classes = (
        MaskedCausalLMLoss,
        FocalCausalLMLoss,
        WaveFieldMemoryLLM,
        WaveFieldDecoderBlock,
        WaveFieldAttention,
        IdentityPlusNoise,
        LongTermMemoryBank,
        WorkingMemoryBank,
        MemoryWriteController,
        MemoryReadController,
    )
    return {keras.saving.get_registered_name(cls): cls for cls in classes}


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_wave_field_memory_llm(
    variant: str = "small",
    **overrides: Any,
) -> WaveFieldMemoryLLM:
    """Create a :class:`WaveFieldMemoryLLM` from a named variant.

    :param variant: One of ``WaveFieldMemoryLLM.MODEL_VARIANTS``
        ("tiny", "small", "medium", "large", "xl").
    :param overrides: Constructor arguments overriding the variant's entries.
    :returns: A configured :class:`WaveFieldMemoryLLM`.
    :raises ValueError: If ``variant`` is not a known variant name.

    Example::

        model = create_wave_field_memory_llm("tiny", vocab_size=128)
        logits = model(token_ids)
    """
    return WaveFieldMemoryLLM.from_variant(variant, **overrides)

# ---------------------------------------------------------------------
