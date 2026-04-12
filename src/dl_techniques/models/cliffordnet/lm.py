"""CliffordNet causal language model.

Adapts the isotropic CliffordNet geometric algebra backbone
(arXiv:2601.06793v2) for autoregressive sequence modeling.  Token sequences
are embedded, reshaped to 4-D ``(B, 1, seq_len, D)`` for
:class:`CausalCliffordNetBlock` processing, then projected to vocabulary
logits.

The :class:`CausalCliffordNetBlock` layers use left-only padded depthwise
convolutions so that each position can only attend to current and past
positions, preserving strict autoregressive causality.

Architecture:

.. code-block:: text

    Input IDs (B, seq_len)
         │
         ▼
    Token Embedding + Positional Embedding
    ─► LayerNorm ─► Dropout
         │
         ▼
    Reshape to (B, 1, seq_len, D)
         │
         ▼
    CausalCliffordNetBlock × depth
    (causal depthwise conv context + Clifford products + GGR)
         │
         ▼
    Reshape to (B, seq_len, D)
         │
         ▼
    LayerNorm ─► Dropout ─► Dense(vocab_size)
         │
         ▼
    Logits (B, seq_len, vocab_size)

References:
    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra. arXiv:2601.06793v2.
"""

import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CausalCliffordNetBlock,
)

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to ``max_rate``."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


@keras.saving.register_keras_serializable()
class CliffordNetLM(keras.Model):
    """CliffordNet language model for causal language modeling.

    Adapts the isotropic CliffordNet vision backbone for sequence modeling.
    Token sequences are embedded, reshaped to 4D ``(B, 1, seq_len, D)`` for
    :class:`CausalCliffordNetBlock` processing, then projected to vocabulary
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
    :param use_bias: Whether Dense/projection layers use bias.
    :param kernel_initializer: Kernel initializer for all dense layers.
    :param bias_initializer: Bias initializer for all dense layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.

    Example:
        .. code-block:: python

            model = CliffordNetLM.from_variant("nano", vocab_size=50261)
            input_ids = keras.random.uniform((2, 64), 0, 50261, dtype="int32")
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
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
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
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

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
        drop_rates = _linear_drop_path_rates(depth, stochastic_depth_rate)
        _block_kw: Dict[str, Any] = dict(
            channels=channels,
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.clifford_blocks = [
            CausalCliffordNetBlock(
                drop_path_rate=drop_rates[i],
                name=f"clifford_block_{i}",
                **_block_kw,
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
        self.output_proj = keras.layers.Dense(
            vocab_size,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="output_proj",
        )

        logger.info(
            f"Created CliffordNetLM (vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, channels={channels}, "
            f"depth={depth}, shifts={self.shifts}, cli_mode={cli_mode}, "
            f"ctx_mode={ctx_mode}, global_ctx={use_global_context})"
        )

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

        # Reshape to 4D: (B, seq_len, D) -> (B, 1, seq_len, D)
        x = keras.ops.expand_dims(x, axis=1)

        # Apply CausalCliffordNet blocks
        for block in self.clifford_blocks:
            x = block(x, training=training)

        # Reshape back to 3D: (B, 1, seq_len, D) -> (B, seq_len, D)
        x = keras.ops.squeeze(x, axis=1)

        # Output projection
        x = self.head_norm(x, training=training)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)
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
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
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
