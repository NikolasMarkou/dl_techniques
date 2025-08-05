"""
Enhanced upsampling module for neural networks.

This module provides various upsampling strategies for convolutional neural networks
implemented in Keras/TensorFlow. It supports multiple upsampling methods including
transposed convolution, bilinear interpolation, and nearest neighbor approaches.

Key Features:
    - Multiple upsampling strategies (Conv2DTranspose, Bilinear, Nearest Neighbor)
    - Support for batch normalization and layer normalization
    - Orthonormal initialization options
    - Configurable convolution parameters

Note:
    All convolution operations use 'same' padding by default to maintain
    spatial dimensions consistency except for explicit strided operations.

Example:
    >>> input_layer = keras.layers.Input(shape=(32, 32, 64))
    >>> conv_params = {"filters": 32, "activation": "relu"}
    >>> upsampled = upsample(input_layer, "conv2d_transpose", conv_params)
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

UpsampleType = Literal[
    "conv2d_transpose",
    "bilinear_conv2d_3x3",
    "nearest_conv2d_3x3",
    "conv2d_1x1_nearest",
    "bilinear_conv2d_v1",
    "laplacian_conv2d",
    "nearest",
    "bilinear"
]

# ---------------------------------------------------------------------

def upsample(
        input_layer: Layer,
        upsample_type: UpsampleType,
        conv_params: Optional[Dict[str, Union[int, str, float]]] = None,
        bn_params: Optional[Dict[str, Union[float, bool]]] = None,
        ln_params: Optional[Dict[str, Union[float, bool]]] = None
) -> Layer:
    """
    Applies upsampling operation to the input layer based on specified strategy.

    This function supports various upsampling methods including transposed convolution,
    bilinear interpolation, and nearest neighbor approaches. It can also apply batch
    normalization and layer normalization after upsampling.

    Args:
        input_layer: Input Keras layer to be upsampled
        upsample_type: Type of upsampling operation to apply
        conv_params: Dictionary containing convolution parameters such as:
            - filters: Number of output filters
            - activation: Activation function to use
            - kernel_initializer: Weight initialization method
            - kernel_regularizer: Weight regularization method
        bn_params: Batch normalization parameters (optional)
        ln_params: Layer normalization parameters (optional)

    Returns:
        Upsampled Keras layer

    Raises:
        ValueError: If upsample_type is None, empty, or unsupported

    Example:
        >>> conv_params = {
        ...     "filters": 32,
        ...     "activation": "relu",
        ...     "kernel_initializer": "he_normal"
        ... }
        >>> x = upsample(input_layer, "conv2d_transpose", conv_params)
    """
    if not upsample_type:
        raise ValueError("upsample_type cannot be None or empty")

    upsample_type = upsample_type.lower().strip()
    x = input_layer
    params = copy.deepcopy(conv_params) if conv_params else {}

    # Method 1: Transposed Convolution Upsampling
    # This method uses transposed convolution (deconvolution) to learn the upsampling
    # Advantages:
    # - Learnable parameters for upsampling
    # - Can learn complex upsampling patterns
    # - End-to-end trainable
    if upsample_type == "conv2d_transpose":
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params,
            conv_type=ConvType.CONV2D_TRANSPOSE
        )

    # Method 2: Bilinear Upsampling with 3x3 Convolution
    # Uses bilinear interpolation followed by 3x3 convolution for feature refinement
    # Advantages:
    # - Smooth upsampling with bilinear interpolation
    # - Additional feature processing with 3x3 conv
    # - Better handling of fine details
    elif upsample_type == "bilinear_conv2d_3x3":
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear"
        )(x)
        params.update({
            "kernel_size": (3, 3),
            "strides": (1, 1),
            "padding": "same"
        })
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params
        )

    # Method 3: Nearest Neighbor Upsampling with 3x3 Convolution
    # Uses nearest neighbor interpolation followed by 3x3 convolution
    # Advantages:
    # - Fast and simple upsampling with nearest neighbor
    # - Feature refinement with 3x3 conv
    # - Preserves sharp edges
    elif upsample_type == "nearest_conv2d_3x3":
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        )(x)
        params.update({
            "kernel_size": (3, 3),
            "strides": (1, 1),
            "padding": "same"
        })
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params
        )

    # Method 4: 1x1 Convolution with Nearest Neighbor Upsampling
    # Applies 1x1 conv with orthonormal initialization followed by nearest neighbor
    # Advantages:
    # - Channel-wise feature processing before upsampling
    # - Stable gradients with orthonormal initialization
    # - Memory efficient
    elif upsample_type == "conv2d_1x1_nearest":
        params.update({
            "kernel_size": (1, 1),
            "strides": (1, 1),
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
            conv_params=params
        )
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        )(x)

    # Method 5: Bilinear Upsampling with 1x1 Convolution
    # Applies 1x1 conv with orthonormal regularization then bilinear upsampling
    # Advantages:
    # - Channel mixing before smooth upsampling
    # - Controlled feature transformation
    # - Good for general-purpose upsampling
    elif upsample_type == "bilinear_conv2d_v1":
        params.update({
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same",
            "kernel_regularizer": SoftOrthonormalConstraintRegularizer()
        })
        x = conv2d_wrapper(
            input_layer=x,
            bn_params=bn_params,
            ln_params=ln_params,
            conv_params=params
        )
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear"
        )(x)

    # Method 6: Laplacian Network Upsampling
    # Specialized upsampling for Laplacian pyramidal networks
    # Advantages:
    # - Optimized for multi-scale feature processing
    # - Efficient handling of activation functions
    # - Flexible operation ordering
    elif upsample_type == "laplacian_conv2d":
        params.update({
            "kernel_size": (1, 1),
            "strides": (1, 1),
            "padding": "same"
        })
        # Optimize operation order for linear activations
        if params.get("activation", "linear") == "linear":
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params
            )
            x = keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear"
            )(x)
        else:
            x = keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="bilinear"
            )(x)
            x = conv2d_wrapper(
                input_layer=x,
                bn_params=bn_params,
                ln_params=ln_params,
                conv_params=params
            )

    # Method 7: Simple Nearest Neighbor Upsampling
    # Basic nearest neighbor interpolation without additional processing
    # Advantages:
    # - Fast and memory efficient
    # - Preserves hard edges
    # - No additional parameters
    elif upsample_type in ["nn", "nearest"]:
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        )(x)

    # Method 8: Simple Bilinear Upsampling
    # Basic bilinear interpolation without additional processing
    # Advantages:
    # - Smooth upsampling
    # - No additional parameters
    # - Good default choice for general use
    elif upsample_type == "bilinear":
        x = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear"
        )(x)

    else:
        raise ValueError(f"Unsupported upsample_type: [{upsample_type}]")

    return x

# ---------------------------------------------------------------------
