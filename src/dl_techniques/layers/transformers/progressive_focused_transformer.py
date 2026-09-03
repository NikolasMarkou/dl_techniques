"""The Progressive Focused Transformer (PFT) block.

Implements :class:`PFTBlock`, a pre-norm transformer block (norm then
attention, norm then FFN, both with residuals and optional stochastic
depth) built around Progressive Focused Attention. Each block receives the
attention map produced by the previous block and uses it to refine which
window regions the current block focuses on, so focus sharpens across
depth instead of each layer starting from scratch. Attention runs over
non-overlapping windows, with alternating blocks shifting the window
partition (SW-MSA) so information can cross window boundaries. The FFN
type, normalization type, and stochastic depth rate are all configurable
via factory arguments.

``x' = x + DropPath(PFA(Norm1(x), prev_attn_map))``
``y  = x' + DropPath(FFN(Norm2(x')))``

References:
    - Long et al., 2025. Progressive Focused Transformer for Single Image
      Super-Resolution. (CVPR)
"""

import keras
from typing import Optional, Tuple, Literal, Union, Dict, Any

from ..ffn.factory import create_ffn_layer
from ..norms import create_normalization_layer
from ..stochastic_depth import StochasticDepth
from ..attention.progressive_focused_attention import ProgressiveFocusedAttention
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

NormalizationType = Literal[
    'layer_norm', 'rms_norm', 'zero_centered_rms_norm',
    'band_rms', 'adaptive_band_rms', 'dynamic_tanh'
]
FFNType = Literal[
    'mlp', 'swiglu', 'geglu', 'glu', 'swin_mlp',
    'differential', 'residual', 'orthoglu'
]


@register_dl_technique("dl_techniques.layers.transformers.progressive_focused_transformer")
class PFTBlock(keras.layers.Layer):
    """
    Progressive Focused Transformer Block.

    Combines progressive focused attention, pre-normalization, a configurable
    FFN, residual connections, and stochastic depth into a single building
    block for the PFT-SR architecture. Each layer receives the attention map
    from the previous layer to hierarchically refine focus on relevant
    features. Alternating blocks use shifted windows (SW-MSA) for
    cross-window information flow.

    ``x' = x + DropPath(PFA(Norm1(x), prev_attn_map))``
    ``y  = x' + DropPath(FFN(Norm2(x')))``

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────────┐
        │  Input (B, H, W, dim)                  │
        │  + prev_attn_map (optional)            │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Norm1 ─► Progressive Focused Attn     │
        │  ─► [StochasticDepth] ─► + Residual    │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Norm2 ─► FFN (configurable type)      │
        │  ─► [StochasticDepth] ─► + Residual    │
        └──────────────────┬─────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │  Output (B, H, W, dim)                 │
        │  + attn_map (for next block)           │
        └────────────────────────────────────────┘

    :param dim: Embedding dimension (number of channels).
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param window_size: Attention window size. Default: 8.
    :type window_size: int
    :param shift_size: Cyclic shift for SW-MSA. Default: 0.
    :type shift_size: int
    :param mlp_ratio: FFN expansion ratio. Default: 4.0.
    :type mlp_ratio: float
    :param qkv_bias: Whether QKV projections use bias. Default: True.
    :type qkv_bias: bool
    :param attention_dropout_rate: Attention weight dropout. Default: 0.0.
    :type attention_dropout_rate: float
    :param projection_dropout_rate: Projection / FFN dropout. Default: 0.0.
    :type projection_dropout_rate: float
    :param drop_path_rate: Stochastic depth rate. Default: 0.0.
    :type drop_path_rate: float
    :param norm_type: Normalization layer type. Default: ``'layer_norm'``.
    :type norm_type: NormalizationType
    :param norm_kwargs: Extra kwargs for the normalization factory.
    :type norm_kwargs: Optional[Dict[str, Any]]
    :param ffn_type: FFN architecture type. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param ffn_kwargs: Extra kwargs for the FFN factory.
    :type ffn_kwargs: Optional[Dict[str, Any]]
    :param ffn_activation: FFN activation function. Default: ``'gelu'``.
    :type ffn_activation: str
    :param use_lepe: Enable locally-enhanced positional encoding.
    :type use_lepe: bool
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            attention_dropout_rate: float = 0.0,
            projection_dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_type: NormalizationType = 'layer_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            ffn_type: FFNType = 'mlp',
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            ffn_activation: str = 'gelu',
            use_lepe: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self._dim = dim
        self._num_heads = num_heads
        self._window_size = window_size
        self._shift_size = shift_size
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._attention_dropout_rate = attention_dropout_rate
        self._projection_dropout_rate = projection_dropout_rate
        self._drop_path_rate = drop_path_rate
        self._norm_type = norm_type
        self._norm_kwargs = norm_kwargs or {}
        self._ffn_type = ffn_type
        self._ffn_kwargs = ffn_kwargs or {}
        self._ffn_activation = deserialize_activation(ffn_activation)
        self._use_lepe = use_lepe

        self._validate_config()

        # All sublayers depend only on config params, not on input_shape.
        mlp_hidden_dim = int(self._dim * self._mlp_ratio)

        self._norm1 = create_normalization_layer(
            normalization_type=self._norm_type,
            name="attention_norm",
            **self._norm_kwargs
        )

        self._norm2 = create_normalization_layer(
            normalization_type=self._norm_type,
            name="ffn_norm",
            **self._norm_kwargs
        )

        self._attn = ProgressiveFocusedAttention(
            dim=self._dim,
            num_heads=self._num_heads,
            window_size=self._window_size,
            shift_size=self._shift_size,
            qkv_bias=self._qkv_bias,
            attention_dropout_rate=self._attention_dropout_rate,
            projection_dropout_rate=self._projection_dropout_rate,
            use_lepe=self._use_lepe,
            name="progressive_focused_attention"
        )

        self._ffn = self._build_ffn(mlp_hidden_dim)

        if self._drop_path_rate > 0.0:
            self._drop_path = StochasticDepth(
                drop_path_rate=self._drop_path_rate,
                name="stochastic_depth"
            )
        else:
            self._drop_path = None

    def _validate_config(self) -> None:
        """Validate layer configuration parameters.

        :raises ValueError: If any parameters are invalid or incompatible.
        """
        # Validate shift_size constraints
        if self._shift_size >= self._window_size:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be less than "
                f"window_size ({self._window_size}). "
                f"For shifted windows, typically use shift_size = window_size // 2."
            )

        if self._shift_size < 0:
            raise ValueError(
                f"shift_size ({self._shift_size}) must be non-negative"
            )

        # Validate dimension divisibility for multi-head attention
        if self._dim % self._num_heads != 0:
            raise ValueError(
                f"dim ({self._dim}) must be divisible by "
                f"num_heads ({self._num_heads}). "
                f"Got head_dim = {self._dim / self._num_heads}"
            )

        # Validate dropout rates
        if self._drop_path_rate < 0.0 or self._drop_path_rate > 1.0:
            raise ValueError(
                f"drop_path_rate ({self._drop_path_rate}) must be "
                f"between 0.0 and 1.0"
            )

        if self._attention_dropout_rate < 0.0 or self._attention_dropout_rate > 1.0:
            raise ValueError(
                f"attention_dropout_rate ({self._attention_dropout_rate}) must be "
                f"between 0.0 and 1.0"
            )

        if self._projection_dropout_rate < 0.0 or self._projection_dropout_rate > 1.0:
            raise ValueError(
                f"projection_dropout_rate ({self._projection_dropout_rate}) must be "
                f"between 0.0 and 1.0"
            )

        # Validate MLP ratio
        if self._mlp_ratio <= 0.0:
            raise ValueError(
                f"mlp_ratio ({self._mlp_ratio}) must be positive"
            )

    def build(self, input_shape: Union[tuple, list]) -> None:
        """Build all sub-layers with correct shapes.

        :param input_shape: Single shape tuple or list of
            ``[x_shape, attn_map_shape]``.
        :type input_shape: Union[tuple, list]
        """
        if self.built:
            return

        # DECISION plan_2026-06-15_2a23a001/D-002: a bare single-input shape is
        # a tuple of ints, so only treat input_shape as [x, attn_map] when its first element is itself a shape sequence. See decisions.md.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # A None or non-subscriptable shape means build_from_config loaded
        # with no shape; derive a minimal one from config so sublayers can still build.
        if not hasattr(x_shape, '__len__') or x_shape is None:
            x_shape = (None, None, None, self._dim)

        # Deserialization passes shapes back as JSON lists; the concatenation
        # below requires a tuple.
        x_shape = tuple(x_shape)

        norm_shape = (None,) + x_shape[1:]

        self._norm1.build(norm_shape)
        self._norm2.build(norm_shape)
        self._attn.build(x_shape)
        self._ffn.build(norm_shape)

        if self._drop_path is not None:
            self._drop_path.build(norm_shape)

        super().build(input_shape)

    def _build_ffn(self, hidden_dim: int) -> keras.layers.Layer:
        """Build the Feed-Forward Network using factory pattern.

        :param hidden_dim: Hidden dimension (``dim * mlp_ratio``).
        :type hidden_dim: int
        :return: Configured FFN layer.
        :rtype: keras.layers.Layer
        """
        ffn_config = self._ffn_kwargs.copy()

        if self._ffn_type == 'mlp':
            return create_ffn_layer(
                ffn_type='mlp',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout_rate,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'swiglu':
            # Uses its own expansion factor instead of hidden_dim.
            return create_ffn_layer(
                ffn_type='swiglu',
                output_dim=self._dim,
                ffn_expansion_factor=self._mlp_ratio,
                dropout_rate=self._projection_dropout_rate,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'geglu':
            return create_ffn_layer(
                ffn_type='geglu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                dropout_rate=self._projection_dropout_rate,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'glu':
            return create_ffn_layer(
                ffn_type='glu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout_rate,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'swin_mlp':
            return create_ffn_layer(
                ffn_type='swin_mlp',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout_rate,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'orthoglu':
            return create_ffn_layer(
                ffn_type='orthoglu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'differential':
            return create_ffn_layer(
                ffn_type='differential',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'residual':
            return create_ffn_layer(
                ffn_type='residual',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                name="ffn",
                **ffn_config
            )

        else:
            return create_ffn_layer(
                ffn_type=self._ffn_type,
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                name="ffn",
                **ffn_config
            )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass of the PFT block.

        :param inputs: Single tensor ``(B, H, W, dim)`` or tuple
            ``(x, prev_attn_map)`` where ``prev_attn_map`` is the
            attention map from the preceding block (or ``None``).
        :type inputs: Union[keras.KerasTensor, Tuple]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Tuple ``(output, attn_map)`` where ``output`` has shape
            ``(B, H, W, dim)`` and ``attn_map`` is passed to the next block.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if isinstance(inputs, (list, tuple)):
            x, prev_attn_map = inputs
        else:
            x = inputs
            prev_attn_map = None

        shortcut = x
        x_norm = self._norm1(x)
        attn_output, attn_map = self._attn(
            x_norm,
            prev_attn_map=prev_attn_map,
            training=training
        )
        if self._drop_path is not None:
            attn_output = self._drop_path(attn_output, training=training)
        x = shortcut + attn_output

        shortcut = x
        x_norm = self._norm2(x)
        ffn_output = self._ffn(x_norm, training=training)
        # The same stochastic-depth layer is reused here, so it drops both
        # sub-blocks together or neither.
        if self._drop_path is not None:
            ffn_output = self._drop_path(ffn_output, training=training)
        x = shortcut + ffn_output

        return x, attn_map

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self._dim,
            "num_heads": self._num_heads,
            "window_size": self._window_size,
            "shift_size": self._shift_size,
            "mlp_ratio": self._mlp_ratio,
            "qkv_bias": self._qkv_bias,
            "attention_dropout_rate": self._attention_dropout_rate,
            "projection_dropout_rate": self._projection_dropout_rate,
            "drop_path_rate": self._drop_path_rate,
            "norm_type": self._norm_type,
            "norm_kwargs": self._norm_kwargs,
            "ffn_type": self._ffn_type,
            "ffn_kwargs": self._ffn_kwargs,
            "ffn_activation": serialize_activation(self._ffn_activation),
            "use_lepe": self._use_lepe,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PFTBlock":
        """Create layer from a configuration dictionary.

        :param config: Configuration from ``get_config``.
        :type config: Dict[str, Any]
        :return: New ``PFTBlock`` instance.
        :rtype: PFTBlock
        """
        return cls(**config)

    def compute_output_shape(
            self,
            input_shape: Union[tuple, list]
    ) -> Tuple[tuple, tuple]:
        """Compute output shapes for feature tensor and attention map.

        :param input_shape: Single shape or list of shapes.
        :type input_shape: Union[tuple, list]
        :return: Tuple ``(output_shape, attn_map_shape)``.
        :rtype: Tuple[tuple, tuple]
        """
        # See D-002 in build(): only index input_shape[0] when it is itself a
        # shape sequence, not a bare tuple of ints.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        output_shape = x_shape
        batch = x_shape[0]
        h, w = x_shape[1], x_shape[2]

        if h is not None and w is not None:
            num_windows = (h // self._window_size) * (w // self._window_size)
            window_area = self._window_size * self._window_size

            if batch is not None:
                attn_batch = batch * num_windows
            else:
                attn_batch = None
        else:
            attn_batch = None
            window_area = self._window_size * self._window_size

        attn_map_shape = (attn_batch, self._num_heads, window_area, window_area)

        return output_shape, attn_map_shape