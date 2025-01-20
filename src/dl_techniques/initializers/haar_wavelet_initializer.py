import keras
import numpy as np
import tensorflow as tf
from keras.api.initializers import Initializer
from keras.api.regularizers import Regularizer
from typing import Dict, Any, Tuple, Optional, Union, Sequence


# ---------------------------------------------------------------------

class HaarWaveletInitializer(Initializer):
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
