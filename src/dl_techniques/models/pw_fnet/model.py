"""
PW-FNet: Pyramid Wavelet-Fourier Network for Image Restoration.

This module implements the complete PW-FNet architecture, an efficient U-Net variant
that replaces self-attention with Fourier-based token mixing for image restoration
tasks such as deraining, deblurring, and dehazing.

**Key Components**:
- FFTLayer & IFFTLayer: Frequency domain transformations
- PW_FNet_Block: Core building block with token mixer and FFN
- Downsample & Upsample: Spatial resolution scaling layers
- PW_FNet: Complete model with multi-scale supervision
"""

import keras
from keras import ops
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------
# 1. Utility FFT Layers
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTLayer(keras.layers.Layer):
    """
    Applies 2D Fast Fourier Transform and outputs concatenated real/imag parts.

    This layer transforms real-valued spatial domain features into the frequency
    domain. To interface with standard real-valued Keras layers (like Conv2D),
    it outputs a single real-valued tensor where the real and imaginary components
    of the FFT are concatenated along the channel axis.

    **Intent**: To provide a modular, serializable Keras layer for frequency domain
    analysis that is compatible with standard convolutional pipelines. This is a
    core component of the Fourier-based token mixer in PW-FNet.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    Transpose: [batch, C, H, W]
           ↓
    Create (real, imag) tuple where imag is zeros
           ↓
    FFT2D along spatial dims → (fft_real, fft_imag)
           ↓
    Transpose both back: [batch, H, W, C]
           ↓
    Concatenate: [fft_real, fft_imag] along channels
           ↓
    Output(float32, shape=[batch, H, W, 2*C])
    ```

    **Input shape**:
        4D tensor: `(batch_size, height, width, channels)`.

    **Output shape**:
        4D tensor: `(batch_size, height, width, 2 * channels)` with `float32`
        dtype, representing concatenated real and imaginary parts.

    **Example**:
        >>> fft_layer = FFTLayer()
        >>> spatial_features = ops.random.normal((2, 32, 32, 64))
        >>> freq_features = fft_layer(spatial_features)
        >>> print(freq_features.shape, freq_features.dtype)
        (2, 32, 32, 128) float32
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the FFT layer.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply 2D FFT to input tensor.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).

        Returns:
            Real tensor of shape (batch, height, width, 2*channels)
            representing concatenated real and imaginary frequency components.
        """
        # Permute: (batch, height, width, channels) -> (batch, channels, height, width)
        x_permuted = keras.ops.transpose(inputs, [0, 3, 1, 2])

        # Keras FFT requires a tuple of (real, imag)
        # Input is real, so imag part is zero
        real_part = x_permuted
        imag_part = keras.ops.zeros_like(real_part)

        # Apply 2D FFT along spatial dimensions (last two)
        fft_real, fft_imag = keras.ops.fft2((real_part, imag_part))

        # Permute back: (batch, channels, height, width) -> (batch, height, width, channels)
        fft_real_permuted = keras.ops.transpose(fft_real, [0, 2, 3, 1])
        fft_imag_permuted = keras.ops.transpose(fft_imag, [0, 2, 3, 1])

        # Concatenate real and imaginary parts along the channel axis
        return keras.ops.concatenate(
            [fft_real_permuted, fft_imag_permuted], axis=-1
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (channels are doubled).

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple.
        """
        batch, height, width, channels = input_shape
        if channels is not None:
            output_channels = channels * 2
        else:
            output_channels = None
        return batch, height, width, output_channels

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config


@keras.saving.register_keras_serializable()
class IFFTLayer(keras.layers.Layer):
    """
    Applies 2D Inverse FFT to concatenated real/imag parts.

    This layer transforms frequency domain features (represented as a real tensor
    with concatenated real and imaginary parts) back into the spatial domain.
    It performs the inverse operation of `FFTLayer`.

    **Intent**: To reconstruct spatial features from the frequency domain after
    processing, completing the Fourier-based token mixer block in PW-FNet.

    **Architecture**:
    ```
    Input(float32, shape=[batch, H, W, 2*C])
           ↓
    Split into (real, imag) parts along channels
           ↓
    Transpose both: [batch, C, H, W]
           ↓
    IFFT2D on (real, imag) tuple → (ifft_real, ifft_imag)
           ↓
    Take ifft_real and transpose back: [batch, H, W, C]
           ↓
    Output(float32, shape=[batch, H, W, C])
    ```

    **Input shape**:
        4D tensor: `(batch_size, height, width, 2 * channels)` with `float32`
        dtype. Channel dimension must be even.

    **Output shape**:
        4D tensor: `(batch_size, height, width, channels)` with `float32` dtype.

    **Example**:
        >>> # Create layers
        >>> fft_layer = FFTLayer()
        >>> ifft_layer = IFFTLayer()
        >>> # Round-trip transformation
        >>> spatial_features = ops.random.normal((2, 32, 32, 64))
        >>> freq_features = fft_layer(spatial_features) # shape (2,32,32,128)
        >>> reconstructed = ifft_layer(freq_features)   # shape (2,32,32,64)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the IFFT layer.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply 2D IFFT to input tensor and extract real part.

        Args:
            inputs: Real tensor with concatenated real/imag parts, shape
                    (batch, height, width, 2*channels).

        Returns:
            Real tensor of shape (batch, height, width, channels) in spatial domain.
        """
        # Input is a concatenation of real and imaginary parts
        real_part, imag_part = keras.ops.split(inputs, 2, axis=-1)

        # Permute: (batch, height, width, channels) -> (batch, channels, height, width)
        real_permuted = keras.ops.transpose(real_part, [0, 3, 1, 2])
        imag_permuted = keras.ops.transpose(imag_part, [0, 3, 1, 2])

        # Apply 2D IFFT along spatial dimensions (last two)
        ifft_real, _ = keras.ops.ifft2((real_permuted, imag_permuted))

        # Permute back: (batch, channels, height, width) -> (batch, height, width, channels)
        ifft_permuted = keras.ops.transpose(ifft_real, [0, 2, 3, 1])

        return ifft_permuted

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (channels are halved).

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple.
        """
        batch, height, width, channels = input_shape
        if channels is not None:
            if channels % 2 != 0:
                raise ValueError(
                    f"Input channels for IFFTLayer must be even, got {channels}"
                )
            output_channels = channels // 2
        else:
            output_channels = None
        return batch, height, width, output_channels

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config


# ---------------------------------------------------------------------
# 2. Core PW-FNet Block
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PW_FNet_Block(keras.layers.Layer):
    """
    Pyramid Wavelet-Fourier Network (PW-FNet) building block.

    This layer is the core component of the PW-FNet architecture, replacing
    computationally expensive self-attention with an efficient frequency-domain
    token mixer and a feed-forward network. It follows modern Keras best
    practices for composite layers, ensuring robust serialization.

    **Intent**: To provide an efficient and effective feature processing block
    that captures global context through Fourier transforms while maintaining
    low computational overhead. Achieves O(N log N) complexity compared to O(N²)
    for standard attention.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    LayerNorm ──────────────┐
       ↓                    │
    Token Mixer             │ (Residual)
       ├─ PointwiseConv(→hidden_dim)
       ├─ FFT2D (→ 2*hidden_dim channels)
       ├─ PointwiseConv (on real+imag parts)
       ├─ GELU
       ├─ IFFT2D (→ hidden_dim channels)
       └─ PointwiseConv(→C)
       ↓                    │
    Add ←───────────────────┘
       ↓
    LayerNorm ──────────────┐
       ↓                    │
    Feed-Forward Network    │ (Residual)
       ├─ PointwiseConv(→hidden_dim)
       ├─ DepthwiseConv(3×3)
       ├─ GELU
       └─ PointwiseConv(→C)
       ↓                    │
    Add ←───────────────────┘
       ↓
    Output(shape=[batch, H, W, C])
    ```

    Args:
        dim: Number of input and output channels. Must be positive.
        ffn_expansion_factor: Expansion factor for the hidden dimension
            in the FFN and token mixer. Determines computational cost and
            capacity. Defaults to 2.0 (2x expansion).
        **kwargs: Additional arguments for the Layer base class.

    Raises:
        ValueError: If dim is not positive.
    """

    def __init__(
            self,
            dim: int,
            ffn_expansion_factor: float = 2.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the PW-FNet block.

        Args:
            dim: Number of input/output channels.
            ffn_expansion_factor: Expansion factor for hidden dimensions.
            **kwargs: Additional Layer arguments.

        Raises:
            ValueError: If dim is not positive.
        """
        super().__init__(**kwargs)

        # Validate and store configuration
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ffn_expansion_factor <= 0:
            raise ValueError(
                f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}"
            )

        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor
        hidden_dim = int(dim * ffn_expansion_factor)
        self._hidden_dim = hidden_dim  # Store for introspection

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Normalization layers
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1"
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm2"
        )

        # Token Mixer sub-layers
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

        # Feed-Forward Network sub-layers
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
            dim,
            kernel_size=1,
            use_bias=True,
            name="ffn_project"
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
        self.ffn_expand.build(input_shape)
        ffn_expanded_shape = self.ffn_expand.compute_output_shape(input_shape)
        self.ffn_depthwise.build(ffn_expanded_shape)
        self.ffn_project.build(ffn_expanded_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the PW-FNet block.

        Args:
            inputs: Input tensor of shape (batch, height, width, dim).
            training: Boolean indicating training mode (unused, for API consistency).

        Returns:
            Output tensor of shape (batch, height, width, dim).
        """
        # -- Token Mixer Stage --
        # Normalize input
        x_norm1 = self.norm1(inputs)

        # Expand channels for frequency domain processing
        x_expanded = self.token_mixer_expand(x_norm1)

        # Frequency domain token mixing: FFT -> Conv -> GELU -> IFFT
        x_fft = self.fft(x_expanded)
        x_freq = self.freq_conv(x_fft)
        x_freq = keras.activations.gelu(x_freq, approximate=False)
        x_ifft = self.ifft(x_freq)

        # Project back to original dimension
        x_token_mixed = self.token_mixer_project(x_ifft)

        # First residual connection
        x = inputs + x_token_mixed

        # -- Feed-Forward Network Stage --
        # Normalize features
        x_norm2 = self.norm2(x)

        # FFN: Expand -> Depthwise Conv -> GELU -> Project
        x_ffn_expanded = self.ffn_expand(x_norm2)
        x_ffn_depthwise = self.ffn_depthwise(x_ffn_expanded)
        x_ffn_depthwise = keras.activations.gelu(x_ffn_depthwise, approximate=False)
        x_ffn_projected = self.ffn_project(x_ffn_depthwise)

        # Second residual connection
        return x + x_ffn_projected

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
        })
        return config


# ---------------------------------------------------------------------
# 3. Scaling Layers
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
# 4. PW-FNet Main Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PW_FNet(keras.Model):
    """
    Complete Pyramid Wavelet-Fourier Network (PW-FNet) model for image restoration.

    This model implements a 3-level U-Net architecture with PW-FNet blocks
    for efficient and effective image restoration. It supports multi-scale
    outputs to enable hierarchical supervision during training, improving
    convergence and final performance.

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
        **kwargs: Additional arguments for the Model base class.
    """

    def __init__(
            self,
            img_channels: int = 3,
            width: int = 32,
            middle_blk_num: int = 4,
            enc_blk_nums: Optional[List[int]] = None,
            dec_blk_nums: Optional[List[int]] = None,
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

        # Store configuration
        self.img_channels = img_channels
        self.width = width
        self.middle_blk_num = middle_blk_num
        self.enc_blk_nums = enc_blk_nums
        self.dec_blk_nums = dec_blk_nums

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
            PW_FNet_Block(width, name=f"enc_l1_blk_{i}")
            for i in range(enc_blk_nums[0])
        ]
        self.down1 = Downsample(width * 2, name="down1")

        self.encoder_level2 = [
            PW_FNet_Block(width * 2, name=f"enc_l2_blk_{i}")
            for i in range(enc_blk_nums[1])
        ]
        self.down2 = Downsample(width * 4, name="down2")

        # -- Bottleneck --
        self.bottleneck = [
            PW_FNet_Block(width * 4, name=f"middle_blk_{i}")
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
            PW_FNet_Block(width * 2, name=f"dec_l2_blk_{i}")
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
            PW_FNet_Block(width, name=f"dec_l1_blk_{i}")
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
        }
        return config