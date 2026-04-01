"""
Progressive Focused Transformer (PFT) Block Module.

This module implements a complete PFT transformer block that combines:
- Progressive Focused Attention (PFA) for hierarchical attention refinement
- Configurable Feed-Forward Network (FFN) via factory pattern
- Pre-normalization architecture for training stability
- Residual connections for gradient flow
- Stochastic depth regularization to prevent overfitting

The block follows the pre-normalization architecture (norm -> attention/FFN -> residual)
which has been shown to improve training stability in deep transformer networks compared
to post-normalization.

Architecture Flow:
-----------------
Input (B, H, W, C)
    ↓
[Norm1] → [Progressive Focused Attention] → [+residual]
    ↓
[Norm2] → [Feed-Forward Network] → [+residual]
    ↓
Output (B, H, W, C) + Attention Map

Both attention and FFN outputs can optionally go through stochastic depth,
which randomly drops entire layers during training for regularization.

Key Features:
------------
1. **Progressive Focused Attention**: Uses attention maps from previous layers
   to progressively refine focus on relevant features
2. **Windowed Attention**: Efficient computation through non-overlapping windows
3. **Shifted Windows**: Alternating shift patterns enable cross-window connections
4. **Factory Patterns**: Easy experimentation with different normalization and FFN types
5. **Stochastic Depth**: Layer-wise dropout for improved regularization

References
----------
    Long, Wei, et al. "Progressive Focused Transformer for Single Image Super-Resolution."
    CVPR 2025.
"""

import keras
from typing import Optional, Tuple, Literal, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn.factory import create_ffn_layer
from ..norms import create_normalization_layer
from ..stochastic_depth import StochasticDepth
from ..attention.progressive_focused_attention import ProgressiveFocusedAttention

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'rms_norm', 'zero_centered_rms_norm',
    'band_rms', 'adaptive_band_rms', 'dynamic_tanh'
]
FFNType = Literal[
    'mlp', 'swiglu', 'geglu', 'glu', 'swin_mlp',
    'differential', 'residual', 'orthoglu'
]

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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

    **Architecture Overview:**

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
    :param attention_dropout: Attention weight dropout. Default: 0.0.
    :type attention_dropout: float
    :param projection_dropout: Projection / FFN dropout. Default: 0.0.
    :type projection_dropout: float
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
            attention_dropout: float = 0.0,
            projection_dropout: float = 0.0,
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

        # ============ Store Configuration Parameters ============
        self._dim = dim
        self._num_heads = num_heads
        self._window_size = window_size
        self._shift_size = shift_size
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._attention_dropout = attention_dropout
        self._projection_dropout = projection_dropout
        self._drop_path_rate = drop_path_rate
        self._norm_type = norm_type
        self._norm_kwargs = norm_kwargs or {}
        self._ffn_type = ffn_type
        self._ffn_kwargs = ffn_kwargs or {}
        self._ffn_activation = ffn_activation
        self._use_lepe = use_lepe

        # ============ Validate Configuration ============
        # Check for invalid parameter combinations early
        self._validate_config()

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

        if self._attention_dropout < 0.0 or self._attention_dropout > 1.0:
            raise ValueError(
                f"attention_dropout ({self._attention_dropout}) must be "
                f"between 0.0 and 1.0"
            )

        if self._projection_dropout < 0.0 or self._projection_dropout > 1.0:
            raise ValueError(
                f"projection_dropout ({self._projection_dropout}) must be "
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
        # ============ Extract Input Shape ============
        # Handle both single tensor and tuple inputs
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # ============ Create Normalization Layers ============
        # Using factory pattern for flexibility in normalization choice
        # Pre-normalization: norm is applied before attention/FFN
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

        # ============ Create Progressive Focused Attention ============
        # This is the core innovation of PFT
        self._attn = ProgressiveFocusedAttention(
            dim=self._dim,
            num_heads=self._num_heads,
            window_size=self._window_size,
            shift_size=self._shift_size,
            qkv_bias=self._qkv_bias,
            attention_dropout=self._attention_dropout,
            projection_dropout=self._projection_dropout,
            use_lepe=self._use_lepe,
            name="progressive_focused_attention"
        )

        # ============ Create Feed-Forward Network ============
        # Using factory pattern for flexibility in FFN architecture
        mlp_hidden_dim = int(self._dim * self._mlp_ratio)
        self._ffn = self._build_ffn(mlp_hidden_dim)

        # ============ Create Stochastic Depth ============
        # Stochastic depth (layer dropout) for regularization
        # Applied to both attention and FFN outputs before residual connection
        if self._drop_path_rate > 0.0:
            self._drop_path = StochasticDepth(
                drop_rate=self._drop_path_rate,
                name="stochastic_depth"
            )
        else:
            self._drop_path = None

        # ============ Explicitly Build Sub-layers ============
        # This ensures all weights are created and properly initialized
        # Shape: (batch, H, W, dim)
        norm_shape = (None,) + x_shape[1:]

        # Build normalization layers
        self._norm1.build(norm_shape)
        self._norm2.build(norm_shape)

        # Build attention layer
        self._attn.build(x_shape)

        # Build FFN layer
        self._ffn.build(norm_shape)

        # Build stochastic depth if present
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
        # Prepare FFN configuration by copying user-provided kwargs
        ffn_config = self._ffn_kwargs.copy()

        # ============ Configure FFN Based on Type ============

        if self._ffn_type == 'mlp':
            # Standard MLP: Linear -> Activation -> Dropout -> Linear
            return create_ffn_layer(
                ffn_type='mlp',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'swiglu':
            # SwiGLU: State-of-the-art gated activation
            # Uses its own expansion factor instead of hidden_dim
            return create_ffn_layer(
                ffn_type='swiglu',
                output_dim=self._dim,
                ffn_expansion_factor=self._mlp_ratio,
                dropout_rate=self._projection_dropout,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'geglu':
            # GeGLU: Alternative gated activation (GELU-based)
            return create_ffn_layer(
                ffn_type='geglu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                dropout_rate=self._projection_dropout,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'glu':
            # GLU: Gated Linear Unit (flexible activation)
            return create_ffn_layer(
                ffn_type='glu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'swin_mlp':
            # Swin MLP: Optimized for vision transformers
            return create_ffn_layer(
                ffn_type='swin_mlp',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                dropout_rate=self._projection_dropout,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'orthoglu':
            # OrthoGLU: GLU with orthogonal regularization
            return create_ffn_layer(
                ffn_type='orthoglu',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                activation=self._ffn_activation,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'differential':
            # Differential FFN: Uses difference computations
            return create_ffn_layer(
                ffn_type='differential',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                name="ffn",
                **ffn_config
            )

        elif self._ffn_type == 'residual':
            # Residual FFN: Additional residual connections within FFN
            return create_ffn_layer(
                ffn_type='residual',
                hidden_dim=hidden_dim,
                output_dim=self._dim,
                name="ffn",
                **ffn_config
            )

        else:
            # Fallback: use factory with the specified type
            # This allows for future FFN types without code changes
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
        # ============ Unpack Inputs ============
        # Handle both single tensor and tuple inputs
        if isinstance(inputs, (list, tuple)):
            x, prev_attn_map = inputs
        else:
            x = inputs
            prev_attn_map = None

        # ============ Attention Sub-block ============
        # Store input for residual connection
        shortcut = x

        # Pre-normalization: normalize before attention
        x_norm = self._norm1(x)

        # Progressive Focused Attention
        # Passes previous attention map for hierarchical focusing
        attn_output, attn_map = self._attn(
            x_norm,
            prev_attn_map=prev_attn_map,
            training=training
        )

        # Apply stochastic depth to attention output
        # During training, randomly drops the entire attention transformation
        if self._drop_path is not None:
            attn_output = self._drop_path(attn_output, training=training)

        # First residual connection: add attention output to input
        # This preserves gradient flow and enables very deep networks
        x = shortcut + attn_output

        # ============ FFN Sub-block ============
        # Store current state for second residual connection
        shortcut = x

        # Pre-normalization: normalize before FFN
        x_norm = self._norm2(x)

        # Feed-Forward Network
        # Point-wise transformation to increase model capacity
        ffn_output = self._ffn(x_norm, training=training)

        # Apply stochastic depth to FFN output
        # Same stochastic depth layer is reused (drops both or neither)
        if self._drop_path is not None:
            ffn_output = self._drop_path(ffn_output, training=training)

        # Second residual connection: add FFN output to input
        x = shortcut + ffn_output

        # Return transformed features and attention map for next layer
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
            "attention_dropout": self._attention_dropout,
            "projection_dropout": self._projection_dropout,
            "drop_path_rate": self._drop_path_rate,
            "norm_type": self._norm_type,
            "norm_kwargs": self._norm_kwargs,
            "ffn_type": self._ffn_type,
            "ffn_kwargs": self._ffn_kwargs,
            "ffn_activation": self._ffn_activation,
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
        # Extract x shape
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # Output shape is same as input x shape
        output_shape = x_shape

        # Compute attention map shape
        batch = x_shape[0]
        h, w = x_shape[1], x_shape[2]

        # Calculate number of windows and tokens per window
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


# ---------------------------------------------------------------------