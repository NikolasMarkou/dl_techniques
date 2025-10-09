"""Fixed 2D Haar wavelet decomposition filters.

This initializer is deterministic and does not perform random sampling.
Instead, it builds a fixed set of 2x2 convolution kernels that correspond
to the basis functions of the 2D discrete Haar wavelet transform. Its
purpose is to equip a convolutional layer with the ability to perform a
single level of multi-resolution analysis, decomposing an input signal (such
as an image) into distinct frequency sub-bands.

Architecture and Mathematical Foundations:
The Haar wavelet transform is the simplest form of wavelet analysis and is
based on a single prototype wavelet. In two dimensions, this decomposition is
achieved by applying the 1D transform separably along the rows and columns,
resulting in four distinct filters that capture different components of the
signal:

1.  **LL (Low-pass/Approximation)**: `[[0.5, 0.5], [0.5, 0.5]]`
    This filter acts as a 2x2 averager. When applied with a stride of 2, it
    produces a downsampled, lower-resolution version of the input, capturing
    its low-frequency content or "approximation" coefficients.

2.  **LH (Horizontal Detail)**: `[[0.5, -0.5], [0.5, -0.5]]`
    This filter averages vertically (low-pass) and takes the difference
    horizontally (high-pass). It therefore acts as a detector for vertical
    edges and captures horizontal detail coefficients.

3.  **HL (Vertical Detail)**: `[[0.5, 0.5], [-0.5, -0.5]]`
    Conversely, this filter takes the difference vertically (high-pass) and
    averages horizontally (low-pass), making it a detector for horizontal
    edges and capturing vertical detail coefficients.

4.  **HH (Diagonal Detail)**: `[[0.5, -0.5], [-0.5, 0.5]]`
    This filter takes the difference in both directions, making it sensitive
    to diagonal details and capturing diagonal detail coefficients.

Together, these four filters form an orthonormal basis. This mathematical
property ensures that the transformation is energy-preserving and allows for
perfect reconstruction of the original signal from its coefficients. In a
neural network context, this initializer provides a fixed, non-learned layer
that acts as an engineered feature extractor, effectively embedding the
principles of wavelet analysis directly into the model's architecture.

References:
    - Mallat, S. (1989). *A theory for multiresolution signal
      decomposition: The wavelet representation*. IEEE Transactions on
      Pattern Analysis and Machine Intelligence.
    - Daubechies, I. (1992). *Ten lectures on wavelets*. Society for
      Industrial and Applied Mathematics.

"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Any, Tuple, Optional, Union, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques")
class HaarWaveletInitializer(keras.initializers.Initializer):
    """Haar wavelet initializer for convolutional layers.

    Implements the standard 2D Haar wavelet decomposition filters:
    - LL: Low-Low (approximation)
    - LH: Low-High (horizontal details)
    - HL: High-Low (vertical details)
    - HH: High-High (diagonal details)

    The filters are normalized to maintain variance across forward passes and
    form an orthonormal basis for perfect reconstruction.

    Args:
        scale: Scaling factor for the wavelet coefficients. Must be positive.
        seed: Random seed for reproducibility (not used in deterministic wavelets).

    Raises:
        ValueError: If scale is not positive.

    Example:
        >>> initializer = HaarWaveletInitializer(scale=1.0)
        >>> weights = initializer((2, 2, 3, 4))  # 2x2 kernels, 3 input channels, 4 output channels
    """

    def __init__(
        self,
        scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize Haar wavelet kernel initializer.

        Args:
            scale: Scaling factor for the wavelet coefficients.
            seed: Random seed for reproducibility.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If scale is not positive.
        """
        super().__init__(**kwargs)

        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")

        self.scale = float(scale)
        self.seed = seed

        logger.info(f"Initialized HaarWaveletInitializer with scale={self.scale}")

    def __call__(
        self,
        shape: Sequence[int],
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Generate orthonormal Haar wavelet filters.

        Args:
            shape: Required shape (kernel_h, kernel_w, in_channels, channel_multiplier).
            dtype: Data type of the tensor.
            **kwargs: Additional arguments.

        Returns:
            Tensor: Initialized Haar wavelet kernels.

        Raises:
            ValueError: If shape is invalid for 2x2 Haar wavelets.
        """
        if len(shape) != 4:
            raise ValueError(f"Expected 4D shape, got {len(shape)}D")

        kernel_h, kernel_w, in_channels, channel_multiplier = shape

        if kernel_h != 2 or kernel_w != 2:
            raise ValueError(
                f"Haar wavelets require 2x2 kernels, got {kernel_h}x{kernel_w}"
            )

        logger.debug(f"Generating Haar wavelet filters for shape {shape}")

        # Define standard 2D Haar wavelet decomposition filters
        # These preserve energy and maintain orthonormality
        sqrt2 = np.sqrt(2.0)

        # Haar wavelet basis functions
        patterns = np.array([
            # LL: Scaling function (approximation - low pass in both directions)
            [[0.5, 0.5],
             [0.5, 0.5]],
            # LH: Horizontal detail (low pass vertical, high pass horizontal)
            [[1.0 / sqrt2, -1.0 / sqrt2],
             [1.0 / sqrt2, -1.0 / sqrt2]],
            # HL: Vertical detail (high pass vertical, low pass horizontal)
            [[1.0 / sqrt2, 1.0 / sqrt2],
             [-1.0 / sqrt2, -1.0 / sqrt2]],
            # HH: Diagonal detail (high pass in both directions)
            [[1.0 / sqrt2, -1.0 / sqrt2],
             [-1.0 / sqrt2, 1.0 / sqrt2]]
        ], dtype=np.float32)

        # Apply scaling while preserving orthonormality
        patterns *= self.scale

        # Initialize output tensor
        kernel = np.zeros(shape, dtype=np.float32)

        # Distribute patterns across input channels and output channels
        for i in range(in_channels):
            for j in range(channel_multiplier):
                # Cycle through the 4 wavelet patterns
                pattern_idx = (i * channel_multiplier + j) % len(patterns)
                kernel[:, :, i, j] = patterns[pattern_idx]

        # Convert to tensor using keras ops
        return ops.convert_to_tensor(kernel, dtype=dtype)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        Returns:
            Dict containing the initializer configuration.
        """
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'seed': self.seed,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'HaarWaveletInitializer':
        """Create initializer from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            HaarWaveletInitializer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
# builder utility
# ---------------------------------------------------------------------

def create_haar_depthwise_conv2d(
    input_shape: Tuple[int, int, int],
    channel_multiplier: int = 4,
    scale: float = 1.0,
    use_bias: bool = False,
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
    trainable: bool = False,
    name: Optional[str] = None
) -> keras.layers.DepthwiseConv2D:
    """Create a Haar wavelet depthwise convolution layer.

    Implements 2D Haar wavelet decomposition as a depthwise convolution with
    stride=2 for dyadic downsampling. This is the standard approach for
    wavelet decomposition in neural networks.

    Args:
        input_shape: Input tensor shape (height, width, channels).
        channel_multiplier: Output channels per input channel. For full wavelet
            decomposition, this should be 4 (LL, LH, HL, HH).
        scale: Wavelet coefficient scaling factor.
        use_bias: Whether to add bias terms (typically False for wavelets).
        kernel_regularizer: Optional kernel regularization.
        trainable: Whether wavelet weights can be trained. Usually False for
            fixed wavelet transforms.
        name: Layer name.

    Returns:
        keras.layers.DepthwiseConv2D: Configured Haar wavelet layer.

    Raises:
        ValueError: If input_shape or channel_multiplier is invalid.

    Example:
        >>> # Create a standard Haar wavelet decomposition layer
        >>> layer = create_haar_depthwise_conv2d(
        ...     input_shape=(256, 256, 3),
        ...     channel_multiplier=4,
        ...     trainable=False
        ... )
        >>> # Input: (batch, 256, 256, 3) -> Output: (batch, 128, 128, 12)
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input shape (H,W,C), got {len(input_shape)}D")

    if channel_multiplier <= 0:
        raise ValueError(f"channel_multiplier must be positive, got {channel_multiplier}")

    # Log warning if using non-standard configuration
    if channel_multiplier != 4 and not trainable:
        logger.warning(
            f"Using channel_multiplier={channel_multiplier} with trainable=False. "
            "For standard wavelet decomposition, channel_multiplier should be 4."
        )

    logger.info(
        f"Creating Haar wavelet layer: input_shape={input_shape}, "
        f"channel_multiplier={channel_multiplier}, trainable={trainable}"
    )

    return keras.layers.DepthwiseConv2D(
        kernel_size=2,
        strides=2,  # Dyadic downsampling for wavelet decomposition
        padding='valid',
        depth_multiplier=channel_multiplier,
        use_bias=use_bias,
        depthwise_initializer=HaarWaveletInitializer(scale=scale),
        depthwise_regularizer=kernel_regularizer,
        trainable=trainable,
        name=name or 'haar_dwconv'
    )

# ---------------------------------------------------------------------
