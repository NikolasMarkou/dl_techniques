import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

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

# ---------------------------------------------------------------------

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