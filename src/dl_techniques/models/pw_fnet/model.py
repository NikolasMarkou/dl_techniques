"""
PW-FNet: Pyramid Wavelet-Fourier Network for Image Restoration.

This module implements the complete PW-FNet architecture with configurable
normalization and feed-forward network components. It replaces self-attention
with Fourier-based token mixing for efficient image restoration tasks such as
deraining, deblurring, and dehazing.

**Key Components**:
- FFTLayer & IFFTLayer: Frequency domain transformations
- PW_FNet_Block: Core building block with configurable norm and FFN
- Downsample & Upsample: Spatial resolution scaling layers
- PW_FNet: Complete model with multi-scale supervision

"""

import keras
from keras import ops
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.fft_layers import FFTLayer, IFFTLayer
from dl_techniques.layers.norms import create_normalization_layer


# ---------------------------------------------------------------------
# Core PW-FNet Block
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PW_FNet_Block(keras.layers.Layer):
    """
    Pyramid Wavelet-Fourier Network (PW-FNet) building block with configurable components.

    This layer is the core component of the PW-FNet architecture, replacing
    computationally expensive self-attention with an efficient frequency-domain
    token mixer and a feed-forward network. It supports configurable normalization
    and FFN types through factory patterns while maintaining the original spatial
    FFN as the default for optimal performance.

    **Intent**: To provide an efficient and effective feature processing block
    that captures global context through Fourier transforms while maintaining
    low computational overhead. Achieves O(N log N) complexity compared to O(N²)
    for standard attention. Now supports experimentation with different
    normalization and FFN architectures.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    Norm1 (configurable) ───────────┐
       ↓                            │
    Token Mixer                     │ (Residual)
       ├─ PointwiseConv(→hidden_dim)
       ├─ FFT2D (→ 2*hidden_dim channels)
       ├─ PointwiseConv (on real+imag parts)
       ├─ GELU
       ├─ IFFT2D (→ hidden_dim channels)
       └─ PointwiseConv(→C)
       ↓                            │
    Add ←───────────────────────────┘
       ↓
    Norm2 (configurable) ───────────┐
       ↓                            │
    FFN (configurable)              │ (Residual)
       ├─ Spatial FFN (default):    │
       │  ├─ PointwiseConv(→hidden_dim)
       │  ├─ DepthwiseConv(3×3)
       │  ├─ GELU
       │  └─ PointwiseConv(→C)
       │                            │
       └─ Factory FFN (optional):   │
          └─ Dense-based FFN layers
       ↓                            │
    Add ←───────────────────────────┘
       ↓
    Output(shape=[batch, H, W, C])
    ```

    Args:
        dim: Number of input and output channels. Must be positive.
        ffn_expansion_factor: Expansion factor for the hidden dimension
            in the FFN and token mixer. Determines computational cost and
            capacity. Defaults to 2.0 (2x expansion).
        normalization_type: Type of normalization to use. Supports all types
            from the normalization factory: 'layer_norm', 'rms_norm',
            'zero_centered_rms_norm', 'band_rms', etc. Defaults to 'layer_norm'.
        norm1_kwargs: Optional dictionary of custom arguments for the first
            normalization layer (after token mixer). These will be passed to
            the normalization factory.
        norm2_kwargs: Optional dictionary of custom arguments for the second
            normalization layer (after FFN). These will be passed to the
            normalization factory.
        use_spatial_ffn: If True, uses the spatial FFN with depthwise convolution
            (original architecture, recommended for image restoration). If False,
            uses a factory-based FFN. Defaults to True.
        ffn_type: Type of FFN to use when use_spatial_ffn=False. Options include:
            'mlp', 'swiglu', 'geglu', 'glu', 'differential', 'residual', etc.
            Ignored when use_spatial_ffn=True. Must be specified if use_spatial_ffn=False.
        ffn_kwargs: Optional dictionary of custom arguments for the factory FFN.
            Only used when use_spatial_ffn=False. These will be passed to the
            FFN factory.
        **kwargs: Additional arguments for the Layer base class.

    Raises:
        ValueError: If dim is not positive, or if use_spatial_ffn=False but
            ffn_type is not specified.

    Example:
        >>> # Default configuration (original architecture)
        >>> block = PW_FNet_Block(dim=64)
        >>>
        >>> # With RMS normalization
        >>> block = PW_FNet_Block(
        ...     dim=64,
        ...     normalization_type='rms_norm',
        ...     norm1_kwargs={'epsilon': 1e-6, 'use_scale': True}
        ... )
        >>>
        >>> # With factory FFN
        >>> block = PW_FNet_Block(
        ...     dim=64,
        ...     use_spatial_ffn=False,
        ...     ffn_type='swiglu',
        ...     ffn_kwargs={'dropout_rate': 0.1}
        ... )
    """

    def __init__(
            self,
            dim: int,
            ffn_expansion_factor: float = 2.0,
            normalization_type: str = 'layer_norm',
            norm1_kwargs: Optional[Dict[str, Any]] = None,
            norm2_kwargs: Optional[Dict[str, Any]] = None,
            use_spatial_ffn: bool = True,
            ffn_type: Optional[str] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the PW-FNet block with configurable components.

        Args:
            dim: Number of input/output channels.
            ffn_expansion_factor: Expansion factor for hidden dimensions.
            normalization_type: Type of normalization layer to use.
            norm1_kwargs: Custom arguments for first normalization layer.
            norm2_kwargs: Custom arguments for second normalization layer.
            use_spatial_ffn: Whether to use spatial FFN (True) or factory FFN (False).
            ffn_type: Type of factory FFN to use (required if use_spatial_ffn=False).
            ffn_kwargs: Custom arguments for factory FFN.
            **kwargs: Additional Layer arguments.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(**kwargs)

        # Validate and store configuration
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ffn_expansion_factor <= 0:
            raise ValueError(
                f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}"
            )
        if not use_spatial_ffn and ffn_type is None:
            raise ValueError(
                "ffn_type must be specified when use_spatial_ffn=False"
            )

        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        self.norm1_kwargs = norm1_kwargs or {}
        self.norm2_kwargs = norm2_kwargs or {}
        self.use_spatial_ffn = use_spatial_ffn
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs or {}

        hidden_dim = int(dim * ffn_expansion_factor)
        self._hidden_dim = hidden_dim  # Store for introspection

        # CREATE all sub-layers in __init__ (they are unbuilt)

        # Normalization layers using factory
        self.norm1 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm1",
            **self.norm1_kwargs
        )
        self.norm2 = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="norm2",
            **self.norm2_kwargs
        )

        # Token Mixer sub-layers (unchanged)
        self.token_mixer_expand = keras.layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            use_bias=True,
            name="token_mixer_expand"
        )
        self.fft = FFTLayer(name="fft")
        self.freq_conv = keras.layers.Conv2D(
            hidden_dim * 2,  # Operates on concatenated real/imag parts
            kernel_size=1,
            use_bias=True,
            name="freq_conv"
        )
        self.ifft = IFFTLayer(name="ifft")
        self.token_mixer_project = keras.layers.Conv2D(
            dim,
            kernel_size=1,
            use_bias=True,
            name="token_mixer_project"
        )

        # Feed-Forward Network sub-layers (configurable)
        if self.use_spatial_ffn:
            self._setup_spatial_ffn(hidden_dim)
        else:
            self._setup_factory_ffn(hidden_dim)

    def _setup_spatial_ffn(self, hidden_dim: int) -> None:
        """
        Setup the spatial FFN with depthwise convolution (original architecture).

        This is the default FFN configuration that uses spatial operations
        optimized for image restoration tasks.

        Args:
            hidden_dim: Hidden dimension for FFN expansion.
        """
        self.ffn_expand = keras.layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            use_bias=True,
            name="ffn_expand"
        )
        self.ffn_depthwise = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=True,
            name="ffn_depthwise"
        )
        self.ffn_project = keras.layers.Conv2D(
            self.dim,
            kernel_size=1,
            use_bias=True,
            name="ffn_project"
        )

    def _setup_factory_ffn(self, hidden_dim: int) -> None:
        """
        Setup a factory-based FFN (experimental).

        Uses the FFN factory to create a configurable feed-forward network.
        Note: Factory FFNs are Dense-based and may not preserve spatial structure
        as well as the spatial FFN for image tasks.

        Args:
            hidden_dim: Hidden dimension for FFN expansion.
        """
        self.ffn = create_ffn_layer(
            ffn_type=self.ffn_type,
            hidden_dim=hidden_dim,
            output_dim=self.dim,
            name="ffn",
            **self.ffn_kwargs
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables are created before weight loading.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Token Mixer Path
        self.norm1.build(input_shape)
        self.token_mixer_expand.build(input_shape)
        expanded_shape = self.token_mixer_expand.compute_output_shape(input_shape)

        self.fft.build(expanded_shape)
        fft_output_shape = self.fft.compute_output_shape(expanded_shape)

        self.freq_conv.build(fft_output_shape)
        freq_conv_output_shape = self.freq_conv.compute_output_shape(
            fft_output_shape
        )

        self.ifft.build(freq_conv_output_shape)
        ifft_output_shape = self.ifft.compute_output_shape(
            freq_conv_output_shape
        )

        self.token_mixer_project.build(ifft_output_shape)

        # FFN Path
        self.norm2.build(input_shape)

        if self.use_spatial_ffn:
            self.ffn_expand.build(input_shape)
            ffn_expanded_shape = self.ffn_expand.compute_output_shape(input_shape)
            self.ffn_depthwise.build(ffn_expanded_shape)
            self.ffn_project.build(ffn_expanded_shape)
        else:
            self.ffn.build(input_shape)

        super().build(input_shape)

    def _token_mixer_forward(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Forward pass through the token mixer (frequency domain processing).

        Args:
            x: Input tensor of shape (batch, height, width, dim).

        Returns:
            Token-mixed features of shape (batch, height, width, dim).
        """
        # Expand channels for frequency domain processing
        x_expanded = self.token_mixer_expand(x)

        # Frequency domain token mixing: FFT -> Conv -> GELU -> IFFT
        x_fft = self.fft(x_expanded)
        x_freq = self.freq_conv(x_fft)
        x_freq = keras.activations.gelu(x_freq, approximate=False)
        x_ifft = self.ifft(x_freq)

        # Project back to original dimension
        x_token_mixed = self.token_mixer_project(x_ifft)

        return x_token_mixed

    def _spatial_ffn_forward(
            self,
            x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Forward pass through the spatial FFN.

        Args:
            x: Input tensor of shape (batch, height, width, dim).

        Returns:
            FFN output of shape (batch, height, width, dim).
        """
        # FFN: Expand -> Depthwise Conv -> GELU -> Project
        x_ffn_expanded = self.ffn_expand(x)
        x_ffn_depthwise = self.ffn_depthwise(x_ffn_expanded)
        x_ffn_depthwise = keras.activations.gelu(x_ffn_depthwise, approximate=False)
        x_ffn_projected = self.ffn_project(x_ffn_depthwise)

        return x_ffn_projected

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the PW-FNet block.

        Args:
            inputs: Input tensor of shape (batch, height, width, dim).
            training: Boolean indicating training mode (for potential dropout).

        Returns:
            Output tensor of shape (batch, height, width, dim).
        """
        # -- Token Mixer Stage --
        # Normalize input
        x_norm1 = self.norm1(inputs)

        # Apply token mixer
        x_token_mixed = self._token_mixer_forward(x_norm1)

        # First residual connection
        x = inputs + x_token_mixed

        # -- Feed-Forward Network Stage --
        # Normalize features
        x_norm2 = self.norm2(x)

        # Apply FFN (spatial or factory-based)
        if self.use_spatial_ffn:
            x_ffn = self._spatial_ffn_forward(x_norm2)
        else:
            x_ffn = self.ffn(x_norm2, training=training)

        # Second residual connection
        return x + x_ffn

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "normalization_type": self.normalization_type,
            "norm1_kwargs": self.norm1_kwargs,
            "norm2_kwargs": self.norm2_kwargs,
            "use_spatial_ffn": self.use_spatial_ffn,
            "ffn_type": self.ffn_type,
            "ffn_kwargs": self.ffn_kwargs,
        })
        return config


# ---------------------------------------------------------------------
# Scaling Layers
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Downsample(keras.layers.Layer):
    """
    Trainable downsampling layer using strided convolution.

    This layer reduces the spatial resolution of feature maps by a factor of 2
    while increasing the channel dimension. It uses a strided convolution with
    a 4×4 kernel to learn optimal downsampling patterns for the specific task.

    Args:
        dim: Number of output channels. Must be positive.
        **kwargs: Additional arguments for the Layer base class.
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """
        Initialize the downsample layer.

        Args:
            dim: Number of output channels.
            **kwargs: Additional Layer arguments.

        Raises:
            ValueError: If dim is not positive.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim

        # Create convolution layer for learnable downsampling
        self.conv = keras.layers.Conv2D(
            dim,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=True,
            name="down_conv"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the internal convolution layer."""
        super().build(input_shape)
        if not self.conv.built:
            self.conv.build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply downsampling to input tensor.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).

        Returns:
            Downsampled tensor of shape (batch, height//2, width//2, dim).
        """
        return self.conv(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape after downsampling.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple with halved spatial dimensions.
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing the dim parameter.
        """
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


@keras.saving.register_keras_serializable()
class Upsample(keras.layers.Layer):
    """
    Trainable upsampling layer using transposed convolution.

    This layer increases the spatial resolution of feature maps by a factor of 2
    while reducing the channel dimension. It uses a transposed convolution (also
    known as deconvolution) to learn optimal upsampling patterns for the specific task.

    Args:
        dim: Number of output channels. Must be positive.
        **kwargs: Additional arguments for the Layer base class.
    """

    def __init__(self, dim: int, **kwargs: Any) -> None:
        """
        Initialize the upsample layer.

        Args:
            dim: Number of output channels.
            **kwargs: Additional Layer arguments.

        Raises:
            ValueError: If dim is not positive.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim

        # Create transposed convolution for learnable upsampling
        self.conv_transpose = keras.layers.Conv2DTranspose(
            dim,
            kernel_size=2,
            strides=2,
            padding="same",
            use_bias=True,
            name="up_conv_transpose"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the internal transposed convolution layer."""
        super().build(input_shape)
        if not self.conv_transpose.built:
            self.conv_transpose.build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply upsampling to input tensor.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).

        Returns:
            Upsampled tensor of shape (batch, height*2, width*2, dim).
        """
        return self.conv_transpose(inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape after upsampling.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple with doubled spatial dimensions.
        """
        # Ensure the output is a tuple to satisfy strict tests
        return tuple(self.conv_transpose.compute_output_shape(input_shape))

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing the dim parameter.
        """
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# ---------------------------------------------------------------------
# PW-FNet Main Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PW_FNet(keras.Model):
    """
    Complete Pyramid Wavelet-Fourier Network (PW-FNet) model for image restoration.

    This model implements a 3-level U-Net architecture with configurable PW-FNet
    blocks for efficient and effective image restoration. It supports multi-scale
    outputs to enable hierarchical supervision during training, improving
    convergence and final performance.

    **Configurability**: This implementation supports configurable normalization
    and FFN components through factory patterns, enabling experimentation with
    different architectural choices while maintaining backward compatibility.

    Args:
        img_channels: Number of channels in input/output images (e.g., 3 for RGB).
            Must be positive.
        width: Base channel width of the model. Controls model capacity and
            computational cost. Typical values: 32-64. Must be positive.
        middle_blk_num: Number of PW-FNet blocks in the bottleneck. More blocks
            capture more complex patterns. Typical values: 4-12. Must be non-negative.
        enc_blk_nums: List of block counts for each encoder level. Length determines
            number of scales. Typical: [2, 2] for 2 levels. Must be non-empty.
        dec_blk_nums: List of block counts for each decoder level. Should match
            enc_blk_nums length. Typical: [2, 2]. Must be non-empty.
        normalization_type: Type of normalization to use throughout the model.
            Supports all types from the normalization factory: 'layer_norm',
            'rms_norm', 'zero_centered_rms_norm', 'band_rms', 'dynamic_tanh', etc.
            Defaults to 'layer_norm' (original behavior).
        norm_kwargs: Optional dictionary of custom arguments to pass to all
            normalization layers. Applied globally unless overridden.
        use_spatial_ffn: If True, uses spatial FFN with depthwise convolution
            in all blocks (original architecture, recommended). If False, uses
            factory-based FFN. Defaults to True.
        ffn_type: Type of factory FFN to use when use_spatial_ffn=False. Options
            include: 'mlp', 'swiglu', 'geglu', 'glu', 'differential', etc.
            Ignored when use_spatial_ffn=True.
        ffn_kwargs: Optional dictionary of custom arguments for factory FFN.
            Only used when use_spatial_ffn=False.
        **kwargs: Additional arguments for the Model base class.

    Example:
        >>> # Default configuration (original architecture)
        >>> model = PW_FNet(img_channels=3, width=32)
        >>>
        >>> # With RMS normalization
        >>> model = PW_FNet(
        ...     img_channels=3,
        ...     width=32,
        ...     normalization_type='rms_norm',
        ...     norm_kwargs={'epsilon': 1e-6, 'use_scale': True}
        ... )
        >>>
        >>> # With factory FFN
        >>> model = PW_FNet(
        ...     img_channels=3,
        ...     width=32,
        ...     use_spatial_ffn=False,
        ...     ffn_type='swiglu',
        ...     ffn_kwargs={'dropout_rate': 0.1}
        ... )
    """

    def __init__(
            self,
            img_channels: int = 3,
            width: int = 32,
            middle_blk_num: int = 4,
            enc_blk_nums: Optional[List[int]] = None,
            dec_blk_nums: Optional[List[int]] = None,
            normalization_type: str = 'layer_norm',
            norm_kwargs: Optional[Dict[str, Any]] = None,
            use_spatial_ffn: bool = True,
            ffn_type: Optional[str] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the PW-FNet model.

        Args:
            img_channels: Number of image channels.
            width: Base channel width.
            middle_blk_num: Number of bottleneck blocks.
            enc_blk_nums: Block counts for encoder levels.
            dec_blk_nums: Block counts for decoder levels.
            normalization_type: Type of normalization to use.
            norm_kwargs: Custom arguments for normalization layers.
            use_spatial_ffn: Whether to use spatial FFN or factory FFN.
            ffn_type: Type of factory FFN (required if use_spatial_ffn=False).
            ffn_kwargs: Custom arguments for factory FFN.
            **kwargs: Additional Model arguments.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(**kwargs)

        # Set defaults
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2]
        if norm_kwargs is None:
            norm_kwargs = {}
        if ffn_kwargs is None:
            ffn_kwargs = {}

        # Validate parameters
        if img_channels <= 0:
            raise ValueError(f"img_channels must be positive, got {img_channels}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if middle_blk_num < 0:
            raise ValueError(
                f"middle_blk_num must be non-negative, got {middle_blk_num}"
            )
        if not enc_blk_nums:
            raise ValueError("enc_blk_nums cannot be empty")
        if not dec_blk_nums:
            raise ValueError("dec_blk_nums cannot be empty")
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError(
                f"enc_blk_nums and dec_blk_nums must have same length, "
                f"got {len(enc_blk_nums)} and {len(dec_blk_nums)}"
            )
        if any(n < 0 for n in enc_blk_nums):
            raise ValueError("All values in enc_blk_nums must be non-negative")
        if any(n < 0 for n in dec_blk_nums):
            raise ValueError("All values in dec_blk_nums must be non-negative")
        if not use_spatial_ffn and ffn_type is None:
            raise ValueError(
                "ffn_type must be specified when use_spatial_ffn=False"
            )

        # Store configuration
        self.img_channels = img_channels
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums
        self.normalization_type = normalization_type
        self.norm_kwargs = norm_kwargs
        self.use_spatial_ffn = use_spatial_ffn
        self.ffn_type = ffn_type
        self.ffn_kwargs = ffn_kwargs

        # CREATE all sub-layers in __init__ (unbuilt)
        # Introduction convolution
        self.intro = keras.layers.Conv2D(
            width,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="intro_conv"
        )

        # -- Encoder --
        self.encoder_level1 = [
            self._create_block(width, f"enc_l1_blk_{i}")
            for i in range(enc_blk_nums[0])
        ]
        self.down1 = Downsample(width * 2, name="down1")

        self.encoder_level2 = [
            self._create_block(width * 2, f"enc_l2_blk_{i}")
            for i in range(enc_blk_nums[1])
        ]
        self.down2 = Downsample(width * 4, name="down2")

        # -- Bottleneck --
        self.bottleneck = [
            self._create_block(width * 4, f"middle_blk_{i}")
            for i in range(middle_blk_num)
        ]

        # -- Decoder --
        self.up2 = Upsample(width * 2, name="up2")
        self.reduce_conv2 = keras.layers.Conv2D(
            width * 2,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="reduce2"
        )
        self.decoder_level2 = [
            self._create_block(width * 2, f"dec_l2_blk_{i}")
            for i in range(dec_blk_nums[0])
        ]

        self.up1 = Upsample(width, name="up1")
        self.reduce_conv1 = keras.layers.Conv2D(
            width,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="reduce1"
        )
        self.decoder_level1 = [
            self._create_block(width, f"dec_l1_blk_{i}")
            for i in range(dec_blk_nums[1])
        ]

        # -- Multi-scale Output Heads --
        self.output_l2 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l2"
        )
        self.output_l1 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l1"
        )
        self.output_l0 = keras.layers.Conv2D(
            img_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="output_l0"
        )

    def _create_block(self, dim: int, name: str) -> PW_FNet_Block:
        """
        Create a PW-FNet block with the model's configuration.

        Args:
            dim: Number of channels for the block.
            name: Name for the block.

        Returns:
            Configured PW_FNet_Block instance.
        """
        return PW_FNet_Block(
            dim=dim,
            normalization_type=self.normalization_type,
            norm1_kwargs=self.norm_kwargs,
            norm2_kwargs=self.norm_kwargs,
            use_spatial_ffn=self.use_spatial_ffn,
            ffn_type=self.ffn_type,
            ffn_kwargs=self.ffn_kwargs,
            name=name
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Forward pass through the PW-FNet model.

        Args:
            inputs: Input tensor of shape (batch, height, width, img_channels).
            training: Boolean indicating training mode (for potential dropout/BN).

        Returns:
            List of three restored images at different scales:
            [full_resolution, half_resolution, quarter_resolution].
        """
        # Downsample original input for multi-scale supervision targets
        input_l1 = keras.layers.AveragePooling2D(pool_size=2)(inputs)
        input_l2 = keras.layers.AveragePooling2D(pool_size=2)(input_l1)

        # -- Encoder Path --
        # Initial feature extraction
        feat = self.intro(inputs)

        # Level 1: Full resolution processing
        feat_l1 = feat
        for blk in self.encoder_level1:
            feat_l1 = blk(feat_l1, training=training)
        skip1 = feat_l1

        # Level 2: Half resolution processing
        feat_l2 = self.down1(skip1)
        for blk in self.encoder_level2:
            feat_l2 = blk(feat_l2, training=training)
        skip2 = feat_l2

        # Bottleneck: Quarter resolution processing
        bottleneck_feat = self.down2(skip2)
        for blk in self.bottleneck:
            bottleneck_feat = blk(bottleneck_feat, training=training)

        # -- Decoder Path & Multi-scale Outputs --

        # Quarter-resolution output (from bottleneck features)
        res_l2 = self.output_l2(bottleneck_feat)
        out_l2 = input_l2 + res_l2

        # Level 2 Decoder: Reconstruct half resolution
        dec_feat_l2 = self.up2(bottleneck_feat)
        dec_feat_l2 = ops.concatenate([dec_feat_l2, skip2], axis=-1)
        dec_feat_l2 = self.reduce_conv2(dec_feat_l2)
        for blk in self.decoder_level2:
            dec_feat_l2 = blk(dec_feat_l2, training=training)

        # Half-resolution output (from first decoder stage)
        res_l1 = self.output_l1(dec_feat_l2)
        out_l1 = input_l1 + res_l1

        # Level 1 Decoder: Reconstruct full resolution
        dec_feat_l1 = self.up1(dec_feat_l2)
        dec_feat_l1 = ops.concatenate([dec_feat_l1, skip1], axis=-1)
        dec_feat_l1 = self.reduce_conv1(dec_feat_l1)
        for blk in self.decoder_level1:
            dec_feat_l1 = blk(dec_feat_l1, training=training)

        # Full-resolution output (from final decoder stage)
        res_l0 = self.output_l0(dec_feat_l1)
        out_l0 = inputs + res_l0

        # Return multi-scale outputs: [full, half, quarter]
        return [out_l0, out_l1, out_l2]

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = {
            "img_channels": self.img_channels,
            "width": self.width,
            "middle_blk_num": self.middle_blk_num,
            "enc_blk_nums": self.enc_blk_nums,
            "dec_blk_nums": self.dec_blk_nums,
            "normalization_type": self.normalization_type,
            "norm_kwargs": self.norm_kwargs,
            "use_spatial_ffn": self.use_spatial_ffn,
            "ffn_type": self.ffn_type,
            "ffn_kwargs": self.ffn_kwargs,
        }
        return config

# ---------------------------------------------------------------------