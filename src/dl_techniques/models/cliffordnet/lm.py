"""An autoregressive language model whose sequence mixer is a geometric product.

This is the CliffordNet backbone with its spatial machinery reinterpreted along a
sequence axis. The premise carries over unchanged: instead of an attention block
followed by an FFN, one bilinear operation — the Clifford geometric product
between a pointwise *detail* stream and a locally aggregated *context* stream —
performs both token mixing and channel mixing, so there is no feed-forward
network anywhere in the stack. The product keeps both the symmetric inner part
(coherence, the quantity an attention score measures) and the antisymmetric wedge
part (structural divergence, which attention discards), and only a few diagonals
of the channel interaction matrix are sampled, by cyclic channel rolls at the
offsets in `shifts`. Cost is `O(seq_len * D * |shifts|)`: linear in sequence
length, with no `seq_len^2` term and no KV cache.

The consequence for causality is structural rather than arithmetic, and it is the
most important thing to understand about this model. There is no attention matrix
here, so there is no causal mask to build, get the polarity of, or forget under a
new code path. The context stream is a pair of depthwise convolutions padded on
the left only, so position `i` mathematically cannot read position `i + 1`. The
optional global-context branch is the one place where a future leak could enter —
a global average over the sequence would see everything — and it is replaced in
the causal blocks by a cumulative mean over positions `0..i`, a summary that
grows as decoding proceeds instead of being constant across the sequence.

Blocks are transform-only. Each returns just its gated geometric update, and this
model performs the residual add and the stochastic-depth gate itself:
`x = x + drop_path(block(x))`. Rewriting that as `x = block(x)` would not simply
drop a skip connection — with `layer_scale_init` at `1e-5` the signal would decay
by orders of magnitude per block while every shape and every finiteness check
still passed. The per-block rates are the shared linear ramp, and
`StochasticDepth(0.0)` is exactly the identity.

`global_context_period` overrides the model-level `use_global_context` on a
periodic schedule rather than replacing it: at a period of `n`, blocks at
1-indexed positions `n, 2n, ...` always get the branch, and every other block
still follows the model-level flag. This gives the cheap interleaving of local
blocks with occasional global ones without a per-block list. `-1` is accepted as
a "disabled" sentinel and normalized to `None`; any other value below 1, or a
non-integer, raises at construction.

Normalization defaults to `zero_centered_rms_norm`, which is the block's own
sequence-mode default and deliberately not the `BatchNormalization` the image
model gets. A norm that reduces over the sequence axis lets every position's
statistics see the whole sequence, which leaks the future past the convolutional
padding that was supposed to prevent it; the causal block rejects those types
outright rather than trusting the caller, so the default here is per-position by
construction.

Weight tying is on by default: the LM head reuses the transposed token-embedding
matrix, saving `vocab_size * channels` parameters, and when `use_bias` is set an
explicit `output_bias` variable supplies the bias that the absent `Dense` would
have carried. This follows the Press & Wolf recipe used by GPT-2 and by small
modern LMs; at large scale the current preference runs the other way, hence the
`tie_word_embeddings=False` path with its own `Dense`.

`call()` returns a dict, `{"logits": (B, seq_len, vocab_size)}`, not a bare
tensor — losses and trainers must index it. Positions are looked up as
`arange(seq_len)` against a `max_seq_length`-sized embedding table with no
guard of its own, so a longer batch fails inside the embedding rather than with a
message from this class.

The variant ladder grows `channels` and `depth` together, widens `shifts` as
capacity grows so that deeper models can exploit longer-range channel mixing, and
raises stochastic depth with depth (`0.05` at `nano` through `0.25` at `xl`).

References:
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Press & Wolf, 2017. Using the Output Embedding to Improve Language Models.
      (https://arxiv.org/abs/1608.05859)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CausalCliffordNetBlock,
)
from dl_techniques.utils.model_build import materialize_sublayers

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


@keras.saving.register_keras_serializable()
class CliffordNetLM(keras.Model):
    """CliffordNet language model for causal language modeling.

    Adapts the isotropic CliffordNet vision backbone for sequence modeling.
    Token sequences are embedded, processed as ``(B, seq_len, D)`` by
    :class:`CausalCliffordNetBlock` layers, then projected to vocabulary
    logits.

    The causal depthwise convolutions in each block use left-only padding so
    position *i* can only see positions ``<= i``, preserving autoregressive
    causality.

    :param vocab_size: Vocabulary size (including special tokens).
    :param max_seq_length: Maximum sequence length for positional embeddings.
    :param channels: Feature dimensionality D (constant throughout blocks).
    :param depth: Number of CausalCliffordNetBlock layers.
    :param shifts: Channel-shift offsets for sparse rolling product.
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context calculation mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add causal cumulative-mean context branch.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param tie_word_embeddings: If True, the LM head reuses the (transposed)
        token embedding matrix instead of an independent Dense projection.
        Saves ``vocab_size * channels`` parameters and matches the Press &
        Wolf (2017) recipe used in GPT-2 / small modern LMs. For
        large-scale models the modern preference is untying. Default: True.
    :param use_bias: Whether Dense/projection layers use bias.
    :param kernel_initializer: Kernel initializer for all dense layers.
    :param bias_initializer: Bias initializer for all dense layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.

    Example:
        .. code-block:: python

            model = CliffordNetLM.from_variant("nano", vocab_size=50261)
            input_ids = keras.random.randint((2, 64), 0, 50261, dtype="int32")
            outputs = model(input_ids)
            print(outputs["logits"].shape)  # (2, 64, 50261)
    """

    LAYERNORM_EPSILON: float = 1e-6

    # Pre-defined variant configurations for NLP.
    # Scaling ladder: channels x depth grows roughly 1.5x per step.
    # Shifts widen as capacity grows so deeper blocks can exploit
    # multi-scale Clifford products. Stochastic depth scales with depth.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            channels=128,
            depth=12,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "mini": dict(
            channels=192,
            depth=12,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            channels=384,
            depth=18,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "large": dict(
            channels=512,
            depth=20,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.2,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "xl": dict(
            channels=768,
            depth=28,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.25,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        channels: int = 128,
        depth: int = 12,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        global_context_period: Optional[int] = None,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.0,
        tie_word_embeddings: bool = True,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        normalization_type: str = "zero_centered_rms_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.channels = channels
        self.depth = depth
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        # Normalize global_context_period: -1 is sentinel for "disabled" (=> None).
        # Validate: must be None, -1, or int >= 1. Other values raise.
        if global_context_period is None or global_context_period == -1:
            self.global_context_period = None
        elif isinstance(global_context_period, bool) or not isinstance(
            global_context_period, int
        ):
            raise ValueError(
                f"global_context_period must be None, -1, or an int >= 1; "
                f"got {global_context_period!r}"
            )
        elif global_context_period < 1:
            raise ValueError(
                f"global_context_period must be None, -1, or an int >= 1; "
                f"got {global_context_period}"
            )
        else:
            self.global_context_period = global_context_period
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.tie_word_embeddings = tie_word_embeddings
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.normalization_type = normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})

        # --- Embeddings ---
        self.token_embedding = keras.layers.Embedding(
            vocab_size, channels, name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            max_seq_length, channels, name="position_embedding",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            dropout_rate, name="embed_dropout",
        )

        # --- CliffordNet blocks ---
        drop_rates = linear_drop_path_rates(depth, stochastic_depth_rate)
        _block_kw: Dict[str, Any] = dict(
            channels=channels,
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            normalization_type=self.normalization_type,
            normalization_kwargs=dict(self.normalization_kwargs),
        )

        def _block_uses_global_ctx(idx: int) -> bool:
            # DECISION plan_2026-05-15_fe237831/D-001: when global_context_period
            # is set, periodic positions (1-indexed n, 2n, ...) force True
            # regardless of model-level use_global_context. Non-periodic positions
            # respect the model-level flag. -1 is normalized to None at __init__.
            if (
                self.global_context_period is not None
                and (idx + 1) % self.global_context_period == 0
            ):
                return True
            return use_global_context

        self.clifford_blocks = [
            CausalCliffordNetBlock(
                name=f"clifford_block_{i}",
                use_global_context=_block_uses_global_ctx(i),
                **_block_kw,
            )
            for i in range(depth)
        ]
        # External residual + drop_path (blocks are transform-only now):
        # x = x + StochasticDepth(rate)(block(x)). Built here (not in call())
        # so the sub-layers serialize with the model.
        self.drop_paths = [
            StochasticDepth(
                drop_path_rate=drop_rates[i],
                name=f"clifford_drop_path_{i}",
            )
            for i in range(depth)
        ]

        # --- Output head ---
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm",
        )
        self.head_dropout = (
            keras.layers.Dropout(dropout_rate, name="head_dropout")
            if dropout_rate > 0.0
            else None
        )
        if tie_word_embeddings:
            self.output_proj = None
            self.output_bias = (
                self.add_weight(
                    name="output_bias",
                    shape=(vocab_size,),
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                    trainable=True,
                )
                if use_bias
                else None
            )
        else:
            self.output_proj = keras.layers.Dense(
                vocab_size,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="output_proj",
            )
            self.output_bias = None

        logger.info(
            f"Created CliffordNetLM (vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, channels={channels}, "
            f"depth={depth}, shifts={self.shifts}, cli_mode={cli_mode}, "
            f"ctx_mode={ctx_mode}, global_ctx={use_global_context}, "
            f"global_context_period={self.global_context_period}, "
            f"tie_word_embeddings={tie_word_embeddings})"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method CliffordNetLM inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param input_ids: Token IDs ``(B, seq_len)``.
        :param training: Whether in training mode.
        :return: Dict with ``"logits"`` key: ``(B, seq_len, vocab_size)``.
        """
        seq_len = keras.ops.shape(input_ids)[1]
        positions = keras.ops.arange(seq_len)

        # Embed tokens + positions
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_norm(x, training=training)
        x = self.embed_dropout(x, training=training)

        # Apply CausalCliffordNet blocks (external residual + drop_path).
        # ``x`` stays ``(B, seq_len, D)`` — see ``layers/geometric/clifford_block.py``.
        for block, drop_path in zip(self.clifford_blocks, self.drop_paths):
            x = x + drop_path(block(x, training=training), training=training)

        # Output projection
        x = self.head_norm(x, training=training)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)
        if self.tie_word_embeddings:
            logits = keras.ops.matmul(
                x, keras.ops.transpose(self.token_embedding.embeddings),
            )
            if self.output_bias is not None:
                logits = logits + self.output_bias
        else:
            logits = self.output_proj(x)

        return {"logits": logits}

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        return {"logits": (input_shape[0], input_shape[1], self.vocab_size)}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "channels": self.channels,
            "depth": self.depth,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "global_context_period": self.global_context_period,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "normalization_type": self.normalization_type,
            "normalization_kwargs": dict(self.normalization_kwargs),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetLM":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        max_seq_length: int = 512,
        **kwargs: Any,
    ) -> "CliffordNetLM":
        """Create a CliffordNetLM from a predefined variant.

        :param variant: One of ``"nano"``, ``"mini"``, ``"base"``, ``"large"``, ``"xl"``.
        :param vocab_size: Vocabulary size.
        :param max_seq_length: Maximum sequence length.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNetLM` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordNetLM-{variant.upper()}")
        return cls(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            **defaults,
        )
