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

    A complete transformer block featuring progressive focused attention,
    pre-normalization, configurable FFN, residual connections, and stochastic depth.
    This block is the fundamental building unit of the PFT-SR architecture.

    **Design Philosophy:**

    The PFT block is designed for vision tasks, particularly image super-resolution,
    where spatial relationships and progressive feature refinement are critical.
    Unlike standard transformers that treat all tokens equally, PFT progressively
    focuses attention on increasingly relevant features across layers.

    **Architecture Details:**

    - **Pre-Normalization**: Applies normalization before attention/FFN rather than
      after. This improves gradient flow and training stability in deep networks.

    - **Windowed Attention**: Reduces computational complexity while maintaining
      local spatial relationships. Window size is typically 7-16 for vision tasks.

    - **Shifted Windows**: Alternating blocks use shifted windows to enable
      cross-window connections, preventing information isolation within windows.

    - **Progressive Focusing**: Each layer receives attention maps from the previous
      layer, creating a hierarchical refinement of attention patterns.

    **Usage Pattern:**

    Typical usage involves alternating between regular windowed attention (W-MSA)
    and shifted window attention (SW-MSA) in consecutive blocks:

    - Block 0: shift_size=0 (W-MSA)
    - Block 1: shift_size=window_size//2 (SW-MSA)
    - Block 2: shift_size=0 (W-MSA)
    - Block 3: shift_size=window_size//2 (SW-MSA)
    - ...

    Parameters
    ----------
    dim : int
        The embedding dimension (number of channels). Must be divisible
        by num_heads.
    num_heads : int
        Number of attention heads. Typical values: 3, 4, 6, 8, 12.
    window_size : int, optional
        Size of the attention window. Attention is computed within
        non-overlapping windows of this size. Typical values: 7, 8, 16.
        Larger windows capture more global context but increase computation.
        Default is 8.
    shift_size : int, optional
        Shift size for shifted window attention (SW-MSA). Use 0 for regular
        W-MSA, and ``window_size // 2`` for SW-MSA. Shifted blocks should
        alternate with non-shifted blocks. Default is 0.
    mlp_ratio : float, optional
        Expansion ratio for the FFN hidden dimension.
        hidden_dim = dim * mlp_ratio. Typical values: 2.0-4.0.
        Higher values increase model capacity but also computation.
        Default is 4.0.
    qkv_bias : bool, optional
        Whether to include bias terms in QKV projections. Including bias
        can improve performance but adds parameters. Default is True.
    attention_dropout : float, optional
        Dropout rate for attention weights after softmax. Helps prevent
        overfitting to specific attention patterns. Default is 0.0.
    projection_dropout : float, optional
        Dropout rate for attention output projection and FFN output.
        Default is 0.0.
    drop_path_rate : float, optional
        Stochastic depth rate. Probability of dropping the entire layer
        during training. This provides strong regularization in deep networks.
        Typically increases with layer depth (e.g., 0.0 → 0.5 from shallow
        to deep layers). Default is 0.0.
    norm_type : NormalizationType, optional
        Type of normalization to use. Supported options:

        - ``'layer_norm'``: Standard Layer Normalization (default, stable)
        - ``'rms_norm'``: RMS Normalization (faster, often equivalent)
        - ``'zero_centered_rms_norm'``: Zero-centered RMS (improved stability)
        - ``'band_rms'``: Band-RMS Normalization (vision-specific)
        - ``'adaptive_band_rms'``: Adaptive Band-RMS (learnable bands)
        - ``'dynamic_tanh'``: Dynamic tanh normalization

        Default is ``'layer_norm'``.
    norm_kwargs : dict, optional
        Additional keyword arguments passed to the normalization layer factory.
        For example, {'epsilon': 1e-6} for LayerNorm, or
        {'max_band_width': 0.1} for BandRMS. Default is None.
    ffn_type : FFNType, optional
        Type of Feed-Forward Network to use. Supported options:

        - ``'mlp'``: Standard MLP with activation (default, stable)
        - ``'swiglu'``: SwiGLU (state-of-the-art for many tasks)
        - ``'geglu'``: GeGLU (alternative gated activation)
        - ``'glu'``: GLU (Gated Linear Unit)
        - ``'swin_mlp'``: Swin Transformer MLP variant
        - ``'differential'``: Differential FFN
        - ``'residual'``: Residual FFN
        - ``'orthoglu'``: Orthogonal GLU

        Default is ``'mlp'``.
    ffn_kwargs : dict, optional
        Additional keyword arguments passed to the FFN factory.
        Default is None.
    ffn_activation : str, optional
        Activation function for FFN (when using ``'mlp'`` type).
        Common choices: 'gelu', 'relu', 'swish', 'mish'.
        Default is ``'gelu'``.
    use_lepe : bool, optional
        Whether to use Locally-Enhanced Positional Encoding in the
        attention mechanism. LePE provides position information through
        depthwise convolution, which is more efficient than absolute
        positional embeddings for vision tasks. Recommended: True.
        Default is True.
    **kwargs
        Additional keyword arguments for the Layer base class
        (e.g., name, dtype, trainable).

    Attributes
    ----------
    _norm1 : keras.layers.Layer
        First normalization layer (before attention).
    _norm2 : keras.layers.Layer
        Second normalization layer (before FFN).
    _attn : ProgressiveFocusedAttention
        Progressive focused attention layer.
    _ffn : keras.layers.Layer
        Feed-forward network layer.
    _drop_path : StochasticDepth or None
        Stochastic depth layer if drop_path_rate > 0.

    Notes
    -----
    **Input Shape:**
        Tuple or single tensor:

        - If tuple: ``(x, prev_attn_map)`` where:
            - x: ``(batch_size, height, width, dim)``
            - prev_attn_map: Attention map from previous block
        - If single: ``(batch_size, height, width, dim)``
          (prev_attn_map assumed to be None)

    **Output Shape:**
        Tuple of:

        - output: ``(batch_size, height, width, dim)`` - transformed features
        - attn_map: Attention map for next block

    **Important Constraints:**
        - Height and width must be divisible by window_size
        - dim must be divisible by num_heads
        - shift_size must be less than window_size
        - For shifted blocks (shift_size > 0), typically use window_size // 2

    **Memory Considerations:**
        - Attention complexity: O(H*W*window_size²) per block
        - FFN complexity: O(H*W*dim*mlp_ratio*dim)
        - Peak memory: During backpropagation through attention

    Examples
    --------
    Basic usage with default settings (W-MSA):

    >>> import keras
    >>> x = keras.random.normal((2, 64, 64, 96))
    >>> block = PFTBlock(dim=96, num_heads=3, window_size=8, shift_size=0)
    >>> output, attn_map = block((x, None))
    >>> print(output.shape)
    (2, 64, 64, 96)

    Shifted window block (SW-MSA) for alternating layers:

    >>> shifted_block = PFTBlock(
    ...     dim=96, num_heads=3, window_size=8, shift_size=4
    ... )
    >>> output, attn_map = shifted_block((output, attn_map))
    >>> print(output.shape)
    (2, 64, 64, 96)

    Advanced configuration with SwiGLU FFN and RMSNorm:

    >>> block = PFTBlock(
    ...     dim=96,
    ...     num_heads=3,
    ...     window_size=8,
    ...     mlp_ratio=4.0,
    ...     norm_type='rms_norm',
    ...     ffn_type='swiglu',
    ...     drop_path_rate=0.1,
    ...     attention_dropout=0.1,
    ...     projection_dropout=0.1
    ... )

    Custom normalization with specific parameters:

    >>> block = PFTBlock(
    ...     dim=96,
    ...     num_heads=3,
    ...     norm_type='band_rms',
    ...     norm_kwargs={'max_band_width': 0.1, 'epsilon': 1e-7}
    ... )

    Stack of blocks for deep network:

    >>> import keras
    >>> def build_pft_stage(dim, num_heads, depth, window_size):
    ...     blocks = []
    ...     for i in range(depth):
    ...         # Alternate between W-MSA and SW-MSA
    ...         shift = 0 if (i % 2 == 0) else window_size // 2
    ...         # Increase drop_path_rate with depth
    ...         drop_path = 0.1 * (i / depth)
    ...         blocks.append(PFTBlock(
    ...             dim=dim,
    ...             num_heads=num_heads,
    ...             window_size=window_size,
    ...             shift_size=shift,
    ...             drop_path_rate=drop_path
    ...         ))
    ...     return blocks
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
        """Initialize PFTBlock with given configuration."""
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
        """
        Validate layer configuration parameters.

        Performs comprehensive validation of all configuration parameters
        to catch errors early before layer building. This helps provide
        clear error messages for common misconfigurations.

        Raises
        ------
        ValueError
            If any configuration parameters are invalid or incompatible.
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
        """
        Build layer components.

        Creates all sub-layers: normalization layers, attention, FFN, and
        stochastic depth. This method follows Keras 3 best practices by
        explicitly building all sub-layers to ensure proper weight
        initialization and serialization.

        **Build Order:**

        1. Create normalization layers (Norm1, Norm2)
        2. Create Progressive Focused Attention layer
        3. Create Feed-Forward Network layer
        4. Create Stochastic Depth layer (if needed)
        5. Explicitly build all sub-layers with correct shapes

        Parameters
        ----------
        input_shape : tuple or list
            Shape tuple or list of shape tuples for inputs.
            Can be either:

            - Single tuple: ``(batch, height, width, dim)`` for x
            - List of tuples: ``[(x_shape), (attn_map_shape)]``

        Notes
        -----
        Explicit sub-layer building is critical for:

        - Proper weight initialization
        - Model serialization/deserialization
        - Mixed precision training
        - Distributed training
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
        """
        Build the Feed-Forward Network using factory pattern.

        Constructs the appropriate FFN architecture based on the specified
        ffn_type. Different FFN types have different strengths:

        - **MLP**: Standard and stable, good default choice
        - **SwiGLU**: State-of-the-art for many tasks, more parameters
        - **GeGLU**: Alternative to SwiGLU with similar performance
        - **GLU**: Gated activation for better expressiveness
        - **Swin MLP**: Optimized for vision tasks
        - **OrthoGLU**: Uses orthogonal regularization

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension for the FFN, computed as ``dim * mlp_ratio``.

        Returns
        -------
        keras.layers.Layer
            The configured FFN layer.

        Notes
        -----
        The factory handles all FFN-specific parameter requirements.
        Some FFN types ignore certain parameters (e.g., SwiGLU has its
        own expansion logic and ignores the hidden_dim parameter).
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
        """
        Forward pass of PFT block.

        Implements the complete transformer block with pre-normalization,
        progressive focused attention, FFN, and residual connections.

        **Computation Flow:**

        1. **Attention Sub-block:**
           - Normalize input (Norm1)
           - Apply Progressive Focused Attention with previous attention map
           - Apply stochastic depth (optional)
           - Add residual connection

        2. **FFN Sub-block:**
           - Normalize attention output (Norm2)
           - Apply Feed-Forward Network
           - Apply stochastic depth (optional)
           - Add residual connection

        **Mathematical Formulation:**

        .. code-block:: text

            # Attention sub-block
            x_norm = Norm1(x)
            attn_out, attn_map = PFA(x_norm, prev_attn_map)
            x = x + DropPath(attn_out)

            # FFN sub-block
            x_norm = Norm2(x)
            ffn_out = FFN(x_norm)
            x = x + DropPath(ffn_out)

        Parameters
        ----------
        inputs : tensor or tuple
            Either a single tensor or a tuple of ``(x, prev_attn_map)`` where:

            - **x**: Input tensor of shape ``(batch_size, height, width, dim)``.
              The feature map to transform.
            - **prev_attn_map**: Optional previous attention map for progressive
              focusing. If None, standard attention is used (first layer behavior).

        training : bool, optional
            Whether in training mode. Affects:

            - Dropout behavior (active only during training)
            - Stochastic depth (active only during training)
            - Batch normalization (if used in custom components)

            Default is None (uses Keras learning phase).

        Returns
        -------
        tuple
            A tuple of ``(output, attention_map)`` where:

            - **output**: Transformed feature tensor of shape
              ``(batch_size, height, width, dim)``. Has the same spatial
              dimensions as input but with refined features.
            - **attention_map**: Attention weights from this block of shape
              ``(batch*num_windows, num_heads, window_area, window_area)``.
              Pass this to the next block as prev_attn_map for progressive focusing.

        Notes
        -----
        **Pre-Normalization Benefits:**

        - Better gradient flow to earlier layers
        - More stable training in deep networks
        - Less sensitive to learning rate
        - Enables training of very deep transformers (50+ layers)

        **Stochastic Depth:**

        When drop_path_rate > 0, entire layers are randomly dropped during
        training. This:

        - Reduces overfitting in deep networks
        - Encourages feature reuse across layers
        - Acts as implicit ensemble of shallower networks
        - Drop probability typically increases with layer depth

        Examples
        --------
        Basic forward pass:

        >>> import keras
        >>> x = keras.random.normal((2, 64, 64, 96))
        >>> block = PFTBlock(dim=96, num_heads=3, window_size=8)
        >>> output, attn_map = block(x, training=True)

        With previous attention map:

        >>> output, new_attn_map = block((output, attn_map), training=True)

        Sequential blocks with progressive focusing:

        >>> blocks = [
        ...     PFTBlock(96, 3, 8, shift_size=0),
        ...     PFTBlock(96, 3, 8, shift_size=4),
        ...     PFTBlock(96, 3, 8, shift_size=0),
        ... ]
        >>> x = keras.random.normal((2, 64, 64, 96))
        >>> attn_map = None
        >>> for block in blocks:
        ...     x, attn_map = block((x, attn_map), training=True)
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
        """
        Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all parameters needed
            to reconstruct this layer via ``from_config``.

        Notes
        -----
        This method is called during model saving to serialize the layer.
        All parameters needed to reconstruct the layer must be included.
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
        """
        Create layer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from ``get_config``.

        Returns
        -------
        PFTBlock
            New instance created from configuration.

        Notes
        -----
        This method is called during model loading to deserialize the layer.
        """
        return cls(**config)

    def compute_output_shape(
            self,
            input_shape: Union[tuple, list]
    ) -> Tuple[tuple, tuple]:
        """
        Compute output shape of the layer.

        Parameters
        ----------
        input_shape : tuple or list
            Input shape(s). Either:

            - Single tuple: ``(batch, height, width, dim)`` for x
            - List of tuples: ``[(x_shape), (attn_map_shape)]``

        Returns
        -------
        tuple
            Output shapes as ``(output_shape, attn_map_shape)`` where:

            - **output_shape**: Same as input x shape
              ``(batch_size, height, width, dim)``.
              The block preserves spatial dimensions.
            - **attn_map_shape**: Shape of attention weights
              ``(batch*num_windows, num_heads, window_area, window_area)``.
              Used as input to next block.

        Notes
        -----
        The output spatial dimensions are always the same as input.
        This makes PFT blocks stackable without dimension changes.
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