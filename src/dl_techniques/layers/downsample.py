"""
Downsampling Module for Neural Networks.

This module provides various downsampling strategies for convolutional neural networks
implemented in Keras/TensorFlow. It supports multiple downsampling methods including
convolution-based, pooling-based, and hybrid approaches.

Key Features:
    - Multiple downsampling strategies (Conv2D, MaxPool, strided operations)
    - Support for batch normalization and layer normalization
    - Orthonormal initialization options
    - Configurable convolution parameters

Dependencies:
    - TensorFlow 2.18.0
    - Keras 3.8.0
    - Python 3.11

Note:
    All convolution operations use 'same' padding by default to maintain
    spatial dimensions consistency except for explicit strided operations.

Example:
    >>> input_layer = keras.layers.Input(shape=(64, 64, 3))
    >>> conv_params = {"filters": 64, "activation": "relu"}
    >>> downsampled = downsample(input_layer, "conv2d_2x2", conv_params)
"""

import copy
import keras
from keras.api.layers import Layer
from typing import Dict, Optional, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv2d_builder import ConvType, conv2d_wrapper
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------

# Constants
DEFAULT_SOFTORTHONORMAL_STDDEV: float = 0.02

DownsampleType = Literal[
    "conv2d_2x2",
    "conv2d_3x3",
    "maxpool",
    "maxpool_2x2_conv2d_1x1_orthonormal",
    "strides",
    "conv2d_1x1_orthonormal"
]

# ---------------------------------------------------------------------


def downsample(
        input_layer: Layer,
        downsample_type: DownsampleType,
        conv_params: Optional[Dict[str, Union[int, str, float]]] = None,
        bn_params: Optional[Dict[str, Union[float, bool]]] = None,
        ln_params: Optional[Dict[str, Union[float, bool]]] = None
) -> Layer:
    """
    Applies downsampling operation to the input layer based on specified strategy.

    This function supports various downsampling methods including convolution-based
    downsampling, max pooling, and strided operations. It can also apply batch
    normalization and layer normalization after downsampling.

    Args:
        input_layer: Input Keras layer to be downsampled
        downsample_type: Type of downsampling operation to apply
        conv_params: Dictionary containing convolution parameters such as:
            - filters: Number of output filters
            - activation: Activation function to use
            - kernel_initializer: Weight initialization method
            - kernel_regularizer: Weight regularization method
        bn_params: Batch normalization parameters (optional)
        ln_params: Layer normalization parameters (optional)

    Returns:
        Downsampled Keras layer

    Raises:
        ValueError: If downsample_type is None, empty, or unsupported

    Example:
        >>> conv_params = {
        ...     "filters": 64,
        ...     "activation": "relu",
        ...     "kernel_initializer": "he_normal"
        ... }
        >>> x = downsample(input_layer, "conv2d_2x2", conv_params)
    """
    if not downsample_type:
        raise ValueError("downsample_type cannot be None or empty")

    downsample_type = downsample_type.lower().strip()
    x = input_layer
    params = copy.deepcopy(conv_params) if conv_params else {}

    # Method 1: 2x2 Convolution Downsampling
    # This method uses a 2x2 convolution with stride 2 to reduce spatial dimensions
    # Advantages:
    # - Learnable parameters for downsampling
    # - Maintains spatial relationship between features
    # - Can adjust number of filters
    if downsample_type == "conv2d_2x2":
        params.update({
            "kernel_size": (2, 2),
            "strides": (2, 2),
            "padding": "same"
        })
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params
        )

    # Method 2: 3x3 Convolution Downsampling
    # Uses a larger kernel for downsampling, capturing more spatial context
    # Advantages:
    # - Larger receptive field
    # - Better feature extraction during downsampling
    # - More parameters for learning complex patterns
    elif downsample_type == "conv2d_3x3":
        params.update({
            "kernel_size": (3, 3),
            "strides": (2, 2),
            "padding": "same"
        })
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params
        )

    # Method 3: MaxPool Downsampling
    # Traditional max pooling followed by optional 1x1 convolution
    # Advantages:
    # - Translation invariance
    # - No learnable parameters in pooling
    # - Reduces spatial dimensions while preserving important features
    elif downsample_type == "maxpool":
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same",
            strides=(2, 2)
        )(x)
        if params:
            params.update({
                "kernel_size": (1, 1),
                "strides": (1, 1)
            })
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params
            )

    # Method 4: MaxPool with Orthonormal 1x1 Convolution
    # Combines max pooling with orthonormally initialized 1x1 convolution
    # Advantages:
    # - Stable gradient flow due to orthonormal initialization
    # - Combines benefits of pooling and learnable feature transformation
    # - Better preservation of feature magnitudes
    elif downsample_type == "maxpool_2x2_conv2d_1x1_orthonormal":
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same",
            strides=(2, 2)
        )(x)
        if params:
            params.update({
                "kernel_size": (1, 1),
                "strides": (1, 1),
                "kernel_initializer": keras.initializers.truncated_normal(
                    mean=0.0,
                    seed=0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV
                ),
                "kernel_regularizer": SoftOrthonormalConstraintRegularizer()
            })
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params,
                conv_type=ConvType.CONV2D
            )

    # Method 5: Strided Downsampling
    # Simple strided slicing or 1x1 convolution with stride 2
    # Advantages:
    # - Computationally efficient
    # - No additional parameters if using slicing
    # - Maintains feature channels without transformation
    elif downsample_type == "strides":
        if params:
            params.update({
                "kernel_size": (1, 1),
                "strides": (2, 2),
                "padding": "same"
            })
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params
            )
        else:
            x = x[:, ::2, ::2, :]

    # Method 6: 1x1 Orthonormal Convolution
    # Uses 1x1 convolution with orthonormal initialization for downsampling
    # Advantages:
    # - Minimal spatial context mixing
    # - Stable gradient flow
    # - Efficient parameter usage
    elif downsample_type == "conv2d_1x1_orthonormal":
        if params:
            params.update({
                "kernel_size": (1, 1),
                "strides": (2, 2),
                "padding": "same",
                "kernel_initializer": keras.initializers.truncated_normal(
                    mean=0.0,
                    seed=0,
                    stddev=DEFAULT_SOFTORTHONORMAL_STDDEV
                ),
                "kernel_regularizer": SoftOrthonormalConstraintRegularizer()
            })
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params,
                conv_type=ConvType.CONV2D
            )
        else:
            x = x[:, ::2, ::2, :]

    else:
        raise ValueError(f"Unsupported downsample_type: [{downsample_type}]")

    return x

# ---------------------------------------------------------------------