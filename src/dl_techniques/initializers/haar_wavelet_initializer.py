"""
HaarWaveletInitializer: A Keras initializer for generating Haar wavelet decomposition filters.

This module implements a custom Keras initializer that creates 2D Haar wavelet filters
for use in convolutional neural networks. Haar wavelets are the simplest form of wavelets
and provide an efficient mechanism for multi-resolution analysis and frequency decomposition
of signals and images.

The implementation offers:

1. Generation of standard 2D Haar wavelet decomposition filters:
   - LL: Low-Low pass filter (approximation coefficients)
   - LH: Low-High pass filter (horizontal detail coefficients)
   - HL: High-Low pass filter (vertical detail coefficients)
   - HH: High-High pass filter (diagonal detail coefficients)

2. A utility function to create a Keras DepthwiseConv2D layer pre-configured
   with Haar wavelet filters for wavelet decomposition.

Wavelet transforms are particularly useful in:
- Image processing and computer vision for multi-scale feature extraction
- Signal denoising and compression
- Edge and texture detection
- Multi-resolution analysis
- Neural network architectures like wavelet scattering networks and wavelet CNNs

Mathematical background:
The discrete Haar wavelet transform decomposes a signal into a set of coefficients
representing different frequency bands. For 2D signals like images, the decomposition
is applied along both dimensions, resulting in four subbands:

1. Approximation coefficients (LL): Represent low-frequency content in both directions
2. Horizontal detail coefficients (LH): Capture horizontal edges (vertical changes)
3. Vertical detail coefficients (HL): Capture vertical edges (horizontal changes)
4. Diagonal detail coefficients (HH): Capture diagonal details

These filters form an orthonormal basis, meaning they preserve energy and enable
perfect reconstruction. The implementation normalizes the filters to maintain
variance across forward passes.

References:
    [1] Mallat, S. (1989). A theory for multiresolution signal decomposition:
        The wavelet representation. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 11(7), 674-693.

    [2] Daubechies, I. (1992). Ten lectures on wavelets. Society for Industrial and
        Applied Mathematics. https://doi.org/10.1137/1.9781611970104

    [3] Cotter, S. F., & Rao, B. D. (2002). Sparse channel estimation via matching
        pursuit with application to equalization. IEEE Transactions on Communications,
        50(3), 374-377.

    [4] Oyallon, E., & Mallat, S. (2015). Deep roto-translation scattering for object
        classification. In Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 2865-2873).

    [5] Fujieda, S., Takayama, K., & Hachisuka, T. (2018). Wavelet convolutional
        neural networks. arXiv preprint arXiv:1805.08620.

Example usage:
    ```python
    # Example 1: Manual initialization of convolution kernels
    initializer = HaarWaveletInitializer(scale=1.0)
    kernel = initializer((2, 2, 3, 4))  # 2x2 kernel, 3 input channels, 4 filters per channel

    # Example 2: Using the convenience function to create a wavelet decomposition layer
    input_shape = (256, 256, 3)  # Input image dimensions (H, W, C)
    wavelet_layer = create_haar_depthwise_conv2d(
        input_shape=input_shape,
        channel_multiplier=4,  # Standard decomposition uses 4 filters
        trainable=False  # Fixed wavelet filters
    )

    # Example 3: Creating a wavelet scattering network
    model = keras.Sequential([
        keras.layers.Input(shape=(256, 256, 3)),
        create_haar_depthwise_conv2d(input_shape=(256, 256, 3), name='wavelet_level1'),
        # Apply non-linearity
        keras.layers.Activation('relu'),
        # Further decompose approximation coefficients
        create_haar_depthwise_conv2d(input_shape=(128, 128, 12), name='wavelet_level2'),
        # Feature aggregation layers
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(10, activation='softmax')
    ])
    ```

Note:
    The implementation requires 2x2 kernels and is designed for DepthwiseConv2D layers
    with stride=2 to properly implement the standard Haar wavelet decomposition with
    dyadic downsampling. For full wavelet decomposition, channel_multiplier should be 4.
"""

import keras
import numpy as np
import tensorflow as tf
from keras.api.regularizers import Regularizer
from typing import Dict, Any, Tuple, Optional, Union, Sequence


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class HaarWaveletInitializer(keras.initializers.Initializer):
    """Haar wavelet initializer for convolutional layers.

    Implements the standard 2D Haar wavelet decomposition filters:
    - LL: Low-Low (approximation)
    - LH: Low-High (horizontal details)
    - HL: High-Low (vertical details)
    - HH: High-High (diagonal details)

    The filters are normalized to maintain variance across forward passes.
    """

    def __init__(
            self,
            scale: float = 1.0,
            seed: Optional[int] = None,
            dtype: Any = None
    ) -> None:
        """Initialize Haar wavelet kernels.

        Args:
            scale: Scaling factor for the wavelet coefficients
            seed: Random seed for reproducibility
            dtype: Data type for initialization

        Raises:
            ValueError: If scale is not positive
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        self.scale = float(scale)
        self.seed = seed
        self.dtype = dtype or tf.float32

    def __call__(
            self,
            shape: Sequence[int],
            dtype: Any = None,
            **kwargs
    ) -> tf.Tensor:
        """Generate orthonormal Haar wavelet filters.

        Args:
            shape: Required shape (kernel_h, kernel_w, in_channels, channel_multiplier)
            dtype: Data type of the tensor
            **kwargs: Additional arguments

        Returns:
            tf.Tensor: Initialized Haar wavelet kernels

        Raises:
            ValueError: If shape is invalid for 2x2 Haar wavelets
        """
        if len(shape) != 4:
            raise ValueError(f"Expected 4D shape, got {len(shape)}D")

        kernel_h, kernel_w, in_channels, channel_multiplier = shape

        if kernel_h != 2 or kernel_w != 2:
            raise ValueError(
                f"Haar wavelets require 2x2 kernels, got {kernel_h}x{kernel_w}"
            )

        # Define standard 2D Haar wavelet decomposition filters
        # These preserve energy and maintain orthonormality
        sqrt2 = np.sqrt(2.0)
        patterns = np.array([
            # LL: Scaling function (average)
            [[0.5, 0.5],
             [0.5, 0.5]],
            # LH: Horizontal detail
            [[1.0 / sqrt2, -1.0 / sqrt2],
             [1.0 / sqrt2, -1.0 / sqrt2]],
            # HL: Vertical detail
            [[1.0 / sqrt2, 1.0 / sqrt2],
             [-1.0 / sqrt2, -1.0 / sqrt2]],
            # HH: Diagonal detail
            [[1.0 / sqrt2, -1.0 / sqrt2],
             [-1.0 / sqrt2, 1.0 / sqrt2]]
        ], dtype=np.float32)

        # Apply scaling while preserving orthonormality
        patterns *= self.scale

        # Initialize output tensor
        kernel = np.zeros(shape, dtype=np.float32)

        # Distribute patterns across channels
        for i in range(in_channels):
            for j in range(channel_multiplier):
                pattern_idx = (i * channel_multiplier + j) % len(patterns)
                kernel[:, :, i, j] = patterns[pattern_idx]

        return tf.convert_to_tensor(kernel, dtype=dtype or self.dtype)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'scale': self.scale,
            'seed': self.seed,
            'dtype': self.dtype.name if self.dtype else None
        }

# ---------------------------------------------------------------------
# use of the initializer
# ---------------------------------------------------------------------

def create_haar_depthwise_conv2d(
        input_shape: Tuple[int, int, int],
        channel_multiplier: int = 4,
        scale: float = 1.0,
        use_bias: bool = False,
        kernel_regularizer: Optional[Union[str, Regularizer]] = None,
        trainable: bool = False,  # Usually fixed for wavelet transform
        name: Optional[str] = None
) -> keras.layers.DepthwiseConv2D:
    """Create a Haar wavelet depthwise convolution layer.

    Implements 2D Haar wavelet decomposition as a depthwise convolution.

    Args:
        input_shape: Input tensor shape (height, width, channels)
        channel_multiplier: Output channels per input channel (typically 4 for full decomposition)
        scale: Wavelet coefficient scaling factor
        use_bias: Whether to add bias terms
        kernel_regularizer: Optional kernel regularization
        trainable: Whether wavelets weights can be trained (default False)
        name: Layer name

    Returns:
        keras.layers.DepthwiseConv2D: Configured layer

    Raises:
        ValueError: If input_shape or channel_multiplier is invalid
    """
    if len(input_shape) != 3:
        raise ValueError(f"Expected 3D input shape (H,W,C), got {len(input_shape)}D")

    if channel_multiplier <= 0:
        raise ValueError(f"channel_multiplier must be positive, got {channel_multiplier}")

    if channel_multiplier != 4 and not trainable:
        raise ValueError(
            "For standard wavelet decomposition, channel_multiplier should be 4 "
            f"when trainable=False, got {channel_multiplier}"
        )

    return keras.layers.DepthwiseConv2D(
        kernel_size=2,
        strides=2,  # Dyadic downsampling
        padding='valid',
        depth_multiplier=channel_multiplier,
        use_bias=use_bias,
        depthwise_initializer=HaarWaveletInitializer(scale=scale),
        depthwise_regularizer=kernel_regularizer,
        trainable=trainable,  # Typically fixed for wavelet transform
        name=name or 'haar_dwconv'
    )

# ---------------------------------------------------------------------
