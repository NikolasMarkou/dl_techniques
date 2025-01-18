"""
Advanced Layer and Activation Function Wrappers
============================================

This module provides custom wrappers and utilities for deep learning model construction,
with a focus on convolution operations and activation functions in Keras.

Key Components:
--------------
1. Activation Functions:
   - Custom implementation of advanced activation functions including Mish, ScaledMish
   - Parameterized activation wrappers for LeakyReLU and PReLU
   - Flexible activation selection through string identifiers

2. Multi-scale Feature Generation:
   - Generates multi-scale feature representations using pooling operations
   - Supports both max and average pooling strategies
   - Optional value normalization, clipping, and rounding
   - TensorFlow function compilation for performance optimization

3. Convolution Operations:
   - Enum-based convolution type selection (Conv2D, DepthwiseConv2D, etc.)
   - Comprehensive wrapper for convolution layers with:
     * Batch normalization
     * Layer normalization
     * Dropout (spatial and regular)
     * Flexible activation functions
   - Parameter validation and deep copying for safety

Usage:
------
```python
# Example activation usage
layer = activation_wrapper("mish")(input_tensor)

# Example convolution usage
conv_params = {
    "filters": 64,
    "kernel_size": (3, 3),
    "activation": "relu",
    "padding": "same"
}
layer = conv2d_wrapper(
    input_layer=prev_layer,
    conv_params=conv_params,
    bn_params={"momentum": 0.99},
    conv_type=ConvType.CONV2D
)
"""

import copy
import keras
from enum import Enum
import tensorflow as tf
from keras.api.layers import Layer
from typing import List, Tuple, Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mish import Mish, ScaledMish


# ---------------------------------------------------------------------

def activation_wrapper(activation: Union[Layer, str] = "linear") -> keras.layers.Layer:
    """
    Creates and returns a Keras activation layer based on the specified activation type.

    This wrapper supports both standard Keras activations and custom implementations
    including Mish, ScaledMish, and various LeakyReLU variants. It provides flexible
    configuration of advanced activation functions with specific parameters for
    improved training dynamics.

    Args:
        activation (Union[Layer, str]): Activation specification. Can be either:
            - A string identifying the activation function
            - A pre-configured Keras Layer instance
            Default is "linear" (no activation)

    Supported activation types:
        - "mish": Self-regularizing non-monotonic activation (2020)
        - "scaled_mish": Mish variant with saturation, alpha=2.0
        - "leakyrelu"/"leaky_relu": LeakyReLU with alpha=0.3
        - "leakyrelu_01"/"leaky_relu_01": LeakyReLU with alpha=0.1
        - "leaky_relu_001"/"leakyrelu_001": LeakyReLU with alpha=0.01
        - "prelu": Parametric ReLU with:
            * Constrained alpha in [0,1]
            * L1 regularization (1e-3)
            * Shared parameters across spatial dimensions
        - Any valid Keras activation name (e.g., "relu", "tanh", etc.)

    Returns:
        keras.layers.Layer: Configured activation layer ready for use in a model

    Examples:
        >>> # Using string identifier
        >>> layer = activation_wrapper("mish")
        >>> # Using pre-configured layer
        >>> custom_activation = keras.layers.PReLU()
        >>> layer = activation_wrapper(custom_activation)

    Note:
        When using PReLU, the alpha parameter is:
        - Initialized to 0.1
        - Constrained between 0 and 1
        - Regularized with L1 (1e-3)
        - Shared across spatial dimensions (1,2)
    """
    # If activation is already a Layer instance, return it as is
    if not isinstance(activation, str):
        return activation

    # Normalize activation string
    activation = activation.lower().strip()

    # Select appropriate activation implementation
    if activation in ["mish"]:
        x = Mish()
    elif activation in ["scaled_mish"]:
        x = ScaledMish(alpha=2.0)
    elif activation in ["leakyrelu", "leaky_relu"]:
        x = keras.layers.LeakyReLU(alpha=0.3)
    elif activation in ["leakyrelu_01", "leaky_relu_01"]:
        x = keras.layers.LeakyReLU(alpha=0.1)
    elif activation in ["leaky_relu_001", "leakyrelu_001"]:
        x = keras.layers.LeakyReLU(alpha=0.01)
    elif activation in ["prelu"]:
        constraint = keras.constraints.MinMaxNorm(
            min_value=0.0,
            max_value=1.0,
            rate=1.0,
            axis=0
        )
        x = keras.layers.PReLU(
            alpha_initializer=0.1,
            alpha_regularizer=keras.regularizers.l1(1e-3),
            alpha_constraint=constraint,
            shared_axes=[1, 2]
        )
    else:
        x = keras.layers.Activation(activation)

    return x


# ---------------------------------------------------------------------

def multiscales_generator_fn(
        shape: List[int],
        no_scales: int,
        kernel_size: Tuple[int, int] = (3, 3),
        use_max_pool: bool = False,
        clip_values: bool = False,
        round_values: bool = False,
        normalize_values: bool = False,
        concrete_functions: bool = False,
        jit_compile: bool = False):
    def multiscale_fn(n: tf.Tensor) -> List[tf.Tensor]:
        n_scale = n
        scales = [n_scale]

        for _ in range(no_scales):
            # downsample, clip and round
            if use_max_pool:
                n_scale = \
                    tf.nn.max_pool2d(
                        input=n_scale,
                        ksize=kernel_size,
                        padding="SAME",
                        strides=(2, 2))
            else:
                n_scale = \
                    tf.nn.avg_pool2d(
                        input=n_scale,
                        ksize=kernel_size,
                        padding="SAME",
                        strides=(2, 2))

            # clip values
            if clip_values:
                n_scale = tf.clip_by_value(n_scale,
                                           clip_value_min=0.0,
                                           clip_value_max=255.0)
            # round values
            if round_values:
                n_scale = tf.round(n_scale)

            # normalize (sum of channel dim equals 1)
            if normalize_values:
                n_scale += 1e-5
                n_scale = \
                    n_scale / \
                    tf.reduce_sum(n_scale, axis=-1, keepdims=True)
            scales.append(n_scale)

        return scales

    result = tf.function(
        func=multiscale_fn,
        input_signature=[
            tf.TensorSpec(shape=shape, dtype=tf.float32),
        ],
        jit_compile=jit_compile,
        reduce_retracing=True)

    if concrete_functions:
        return result.get_concrete_function()

    return result


# ---------------------------------------------------------------------


class ConvType(Enum):
    CONV2D = 0

    CONV2D_DEPTHWISE = 1

    CONV2D_TRANSPOSE = 2

    CONV2D_SEPARABLE = 3

    @staticmethod
    def from_string(type_str: str) -> "ConvType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")

        # --- clean string and get
        return ConvType[type_str]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


def conv2d_wrapper(
        input_layer: keras.layers.Layer,
        conv_params: Dict[str, Any],
        bn_params: Optional[Dict[str, Any]] = None,
        ln_params: Optional[Dict[str, Any]] = None,
        dropout_params: Optional[Dict[str, Any]] = None,
        dropout_2d_params: Optional[Dict[str, Any]] = None,
        conv_type: Union[ConvType, str] = ConvType.CONV2D
) -> keras.layers.Layer:
    """
    Creates a wrapped convolution layer with optional normalization, activation, and regularization.

    This wrapper provides a flexible way to construct complex convolution blocks with:
    - Multiple types of convolution (standard, depthwise, transpose, separable)
    - Batch normalization
    - Layer normalization
    - Dropout (both standard and spatial)
    - Custom activations

    The layer ordering is:
    1. Convolution operation
    2. Normalization (batch and/or layer)
    3. Activation
    4. Dropout (standard and/or spatial)

    Args:
        input_layer (keras.layers.Layer): Input tensor or layer
        conv_params (Dict[str, Any]): Convolution parameters dictionary including:
            - filters: Number of output filters
            - kernel_size: Size of convolution kernel
            - strides: Convolution stride
            - padding: Padding type ('valid' or 'same')
            - kernel_initializer: Weight initialization method
            - kernel_regularizer: Weight regularization method
            - Other valid Conv2D parameters
        bn_params (Optional[Dict[str, Any]]): Batch normalization parameters.
            If None, batch normalization is not applied. Default: None
        ln_params (Optional[Dict[str, Any]]): Layer normalization parameters.
            If None, layer normalization is not applied. Default: None
        dropout_params (Optional[Dict[str, Any]]): Standard dropout parameters.
            If None, dropout is not applied. Default: None
        dropout_2d_params (Optional[Dict[str, Any]]): Spatial dropout parameters.
            If None, spatial dropout is not applied. Default: None
        conv_type (Union[ConvType, str]): Type of convolution to use.
            Can be either a ConvType enum or a string. Default: ConvType.CONV2D

    Returns:
        keras.layers.Layer: The constructed layer stack

    Raises:
        ValueError: If input_layer is None
        ValueError: If conv_params is None
        ValueError: If conv_type is invalid

    Examples:
        >>> # Basic Conv2D with batch normalization
        >>> layer = conv2d_wrapper(
        ...     input_layer=prev_layer,
        ...     conv_params={
        ...         "filters": 64,
        ...         "kernel_size": (3, 3),
        ...         "padding": "same",
        ...         "activation": "relu"
        ...     },
        ...     bn_params={"momentum": 0.99}
        ... )

        >>> # Depthwise convolution with dropout
        >>> layer = conv2d_wrapper(
        ...     input_layer=prev_layer,
        ...     conv_params={
        ...         "kernel_size": (3, 3),
        ...         "depth_multiplier": 1,
        ...         "padding": "same"
        ...     },
        ...     dropout_params={"rate": 0.1},
        ...     conv_type=ConvType.CONV2D_DEPTHWISE
        ... )

    Notes:
        - Convolution type can be automatically adjusted based on parameters:
          * If 'depth_multiplier' is in conv_params, switches to CONV2D_DEPTHWISE
          * If 'dilation_rate' is in conv_params, switches to CONV2D_TRANSPOSE
        - Activation is applied after normalization layers
        - Both standard and spatial dropout can be applied simultaneously
        - Parameters are deep copied to prevent unexpected modifications
    """
    # Argument validation
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # Prepare flags and parameters
    use_ln = ln_params is not None
    use_bn = bn_params is not None
    use_dropout = dropout_params is not None
    use_dropout_2d = dropout_2d_params is not None

    # Deep copy conv_params to prevent modifications
    conv_params = copy.deepcopy(conv_params)
    conv_activation = conv_params.get("activation", "linear")
    conv_params["activation"] = "linear"

    # Handle convolution type
    if isinstance(conv_type, str):
        conv_type = ConvType.from_string(conv_type)
    if "depth_multiplier" in conv_params and conv_type != ConvType.CONV2D_DEPTHWISE:
        conv_type = ConvType.CONV2D_DEPTHWISE
    if "dilation_rate" in conv_params and conv_type != ConvType.CONV2D_TRANSPOSE:
        conv_type = ConvType.CONV2D_TRANSPOSE

    # Build layer stack
    x = input_layer

    # Apply convolution
    if conv_type == ConvType.CONV2D:
        x = keras.layers.Conv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_DEPTHWISE:
        x = keras.layers.DepthwiseConv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_TRANSPOSE:
        x = keras.layers.Conv2DTranspose(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_SEPARABLE:
        x = keras.layers.SeparableConv2D(**conv_params)(x)
    else:
        raise ValueError(f"Unsupported convolution type: [{conv_type}]")

    # Apply normalization
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)
    if use_ln:
        x = keras.layers.LayerNormalization(**ln_params)(x)

    # Apply activation
    if conv_activation is not None and conv_activation != "linear":
        x = activation_wrapper(conv_activation)(x)

    # Apply dropout
    if use_dropout:
        x = keras.layers.Dropout(**dropout_params)(x)
    if use_dropout_2d:
        x = keras.layers.SpatialDropout2D(**dropout_2d_params)(x)

    return x

# ---------------------------------------------------------------------
