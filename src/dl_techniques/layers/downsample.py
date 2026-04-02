"""
Downsampling module for neural networks.

This module provides various downsampling strategies for convolutional neural
networks implemented in Keras/TensorFlow. It supports multiple downsampling
methods including convolution-based, pooling-based, and hybrid approaches.
All convolution operations use 'same' padding by default to maintain spatial
dimensions consistency except for explicit strided operations.
"""

import copy
import keras
from keras.api.layers import Layer
from typing import Dict, Optional, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv2d_builder import ConvType, conv2d_wrapper
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

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
    Apply downsampling operation to the input layer based on specified strategy.

    Supports various downsampling methods including convolution-based
    downsampling (2x2, 3x3), max pooling, strided operations, and hybrid
    approaches with orthonormal initialization.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────┐
        │  Input [batch, H, W, C]                   │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌──────────┬───────┴────────┬───────────────┐
        │ conv2d   │ maxpool        │ strides       │
        │ (2x2/3x3)│ ──▶ opt 1x1    │ ──▶ opt 1x1   │
        └────┬─────┴───────┬────────┴───────┬───────┘
             ▼             ▼               ▼
        ┌───────────────────────────────────────────┐
        │  Output [batch, H/2, W/2, C']             │
        └───────────────────────────────────────────┘

    :param input_layer: Input Keras layer to be downsampled.
    :type input_layer: Layer
    :param downsample_type: Type of downsampling operation to apply.
    :type downsample_type: DownsampleType
    :param conv_params: Dictionary containing convolution parameters such as
        filters, activation, kernel_initializer, kernel_regularizer.
    :type conv_params: Optional[Dict[str, Union[int, str, float]]]
    :param bn_params: Batch normalization parameters.
    :type bn_params: Optional[Dict[str, Union[float, bool]]]
    :param ln_params: Layer normalization parameters.
    :type ln_params: Optional[Dict[str, Union[float, bool]]]
    :return: Downsampled Keras layer.
    :rtype: Layer
    :raises ValueError: If downsample_type is None, empty, or unsupported.
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
