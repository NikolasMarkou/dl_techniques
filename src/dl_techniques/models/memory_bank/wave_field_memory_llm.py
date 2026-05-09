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
- Custom :meth:`train_step` splits trainable variables by name prefix
  (``memory_`` / ``gate_`` -> memory optimizer; everything else ->
  backbone optimizer).
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
    _IdentityPlusNoise,
)
from dl_techniques.models.wave_field_llm.wave_field_llm import (
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
)
from dl_techniques.models.memory_bank.phase_scheduler import (
    PHASE_WARMUP,
    PHASE_FREEZE_BACKBONE,
    PHASE_FULL,
    PHASE_EXTEND,
)


# ---------------------------------------------------------------------


def split_trainable_by_prefix(
    variables: List[Any],
    memory_prefixes: Tuple[str, ...] = ("memory_", "gate_"),
) -> Tuple[List[Any], List[Any]]:
    """Partition ``variables`` into ``(memory_vars, backbone_vars)`` by
    matching the leading path component of each variable's ``.name`` (or
    its full name) against ``memory_prefixes``.

    Variables whose name contains any of the prefixes are routed to the
    memory optimizer; everything else to the backbone optimizer.
    """
    memory_vars: List[Any] = []
    backbone_vars: List[Any] = []
    for v in variables:
        name = getattr(v, "name", "") or ""
        if any(p in name for p in memory_prefixes):
            memory_vars.append(v)
        else:
            backbone_vars.append(v)
    return memory_vars, backbone_vars


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WaveFieldMemoryLLM(keras.Model):
    """Memory-augmented WaveFieldLLM with dual-tap topology.

    Memory hyperparameters scale per variant (see :data:`MODEL_VARIANTS`).
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
        diversity_subsample: int = 1024,
        infonce_negatives: int = 256,
        infonce_temperature: float = 0.1,
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
        self.diversity_subsample = diversity_subsample
        self.infonce_negatives = infonce_negatives
        self.infonce_temperature = infonce_temperature
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        self.L_write = l_write
        self.L_read = l_read

        self._build_architecture()

        # Phase counter + global step (survive save/load).
        self.current_phase = self.add_weight(
            name="memory_current_phase",
            shape=(),
            initializer=keras.initializers.Constant(1),
            trainable=False,
            dtype="int32",
        )
        self._global_step = self.add_weight(
            name="memory_global_step",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="int64",
        )

        # Optimizers (set by compile()).
        self.backbone_optimizer = None
        self.memory_optimizer = None

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

        # Memory components.
        self.lt_memory = LongTermMemoryBank(
            s_lt=self.s_lt, d_k=self.d_k, d_v=self.d_v,
            initializer_range=self.initializer_range,
            name="memory_lt_bank",
        )
        self.write_controller = MemoryWriteController(
            d_k=self.d_k, d_v=self.d_v, embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            initializer_range=self.initializer_range,
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
            diversity_subsample=self.diversity_subsample,
            infonce_negatives=self.infonce_negatives,
            infonce_temperature=self.infonce_temperature,
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
        memory_active = ops.not_equal(self.current_phase, PHASE_WARMUP)

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
                    x, k_lt, v_lt, k_wm, v_wm, wm_mask, training=training,
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
        super().compile(optimizer=backbone_optimizer, **kwargs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        memory_pairs = []
        backbone_pairs = []
        for g, v in zip(grads, trainable_vars):
            if g is None:
                continue
            name = getattr(v, "name", "") or ""
            if "memory_" in name or "gate_" in name:
                memory_pairs.append((g, v))
            else:
                backbone_pairs.append((g, v))

        if backbone_pairs:
            self.backbone_optimizer.apply_gradients(backbone_pairs)
        if memory_pairs:
            self.memory_optimizer.apply_gradients(memory_pairs)

        self._global_step.assign_add(tf.constant(1, dtype="int64"))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

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

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(4096, max(256, stacked.shape[0] // 4)),
                n_init=1,
                max_iter=20,
                random_state=0,
            )
            kmeans.fit(stacked)
            centroids = kmeans.cluster_centers_.astype(np.float32)
            # KMeans runs on hidden-state space (D); we need K_lt at d_k.
            # Project via the read controller's W_Q-based key estimator:
            # for warmup we project centroids through a freshly-initialized
            # random Gaussian projection D -> d_k.
            rng = np.random.RandomState(0)
            proj = rng.randn(self.embed_dim, self.d_k).astype(np.float32) / np.sqrt(self.embed_dim)
            centroids_dk = centroids @ proj
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
            "diversity_subsample": self.diversity_subsample,
            "infonce_negatives": self.infonce_negatives,
            "infonce_temperature": self.infonce_temperature,
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


# ---------------------------------------------------------------------


def memory_llm_custom_objects() -> Dict[str, Any]:
    """Return the ``custom_objects`` dict needed by ``keras.models.load_model``
    to deserialize a saved :class:`WaveFieldMemoryLLM`.
    """
    from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
    return {
        "MaskedCausalLMLoss": MaskedCausalLMLoss,
        "FocalCausalLMLoss": FocalCausalLMLoss,
        "WaveFieldMemoryLLM": WaveFieldMemoryLLM,
        "WaveFieldDecoderBlock": WaveFieldDecoderBlock,
        "WaveFieldAttention": WaveFieldAttention,
        "_IdentityPlusNoise": _IdentityPlusNoise,
        "LongTermMemoryBank": LongTermMemoryBank,
        "WorkingMemoryBank": WorkingMemoryBank,
        "MemoryWriteController": MemoryWriteController,
        "MemoryReadController": MemoryReadController,
    }
