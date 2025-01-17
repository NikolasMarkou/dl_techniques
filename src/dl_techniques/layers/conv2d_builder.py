import copy
import keras
from enum import Enum
import tensorflow as tf
from typing import List, Tuple, Iterable, Union, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mish import Mish, ScaledMish


# ---------------------------------------------------------------------

def activation_wrapper(
        activation: Union[keras.layers.Layer, str] = "linear") -> keras.layers.Layer:
    if not isinstance(activation, str):
        return activation

    activation = activation.lower().strip()

    if activation in ["mish"]:
        # Mish: A Self Regularized Non-Monotonic Activation Function (2020)
        x = Mish()
    elif activation in ["scaled_mish"]:
        # scaled mish: mish that saturates
        x = ScaledMish(alpha=2.0)
    elif activation in ["leakyrelu", "leaky_relu"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = keras.layers.LeakyReLU(alpha=0.3)
    elif activation in ["leakyrelu_01", "leaky_relu_01"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = keras.layers.LeakyReLU(alpha=0.1)
    elif activation in ["leaky_relu_001", "leakyrelu_001"]:
        # leaky relu, practically same us Relu
        # with very small negative slope to allow gradient flow
        x = keras.layers.LeakyReLU(alpha=0.01)
    elif activation in ["prelu"]:
        # parametric Rectified Linear Unit
        constraint = \
            keras.constraints.MinMaxNorm(
                min_value=0.0, max_value=1.0, rate=1.0, axis=0)
        x = keras.layers.PReLU(
            alpha_initializer=0.1,
            # very small l1
            alpha_regularizer=keras.regularizers.l1(1e-3),
            alpha_constraint=constraint,
            shared_axes=[1, 2])
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
        input_layer,
        conv_params: Dict,
        bn_params: Dict = None,
        ln_params: Dict = None,
        dropout_params: Dict = None,
        dropout_2d_params: Dict = None,
        conv_type: Union[ConvType, str] = ConvType.CONV2D):
    """
    wraps a conv2d with a preceding normalizer

    if bn_post_params force a conv(linear)->bn->activation setup

    :param input_layer: the layer to operate on
    :param conv_params: conv2d parameters
    :param bn_params: batchnorm parameters before the conv, None to disable bn
    :param ln_params: layer normalization parameters before the conv, None to disable ln
    :param dropout_params: dropout parameters after the conv, None to disable it
    :param dropout_2d_params: dropout parameters after the conv, None to disable it
    :param conv_type: if true use depthwise convolution,

    :return: transformed input
    """
    # --- argument checking
    if input_layer is None:
        raise ValueError("input_layer cannot be None")
    if conv_params is None:
        raise ValueError("conv_params cannot be None")

    # --- prepare arguments
    use_ln = ln_params is not None
    use_bn = bn_params is not None
    use_dropout = dropout_params is not None
    use_dropout_2d = dropout_2d_params is not None
    conv_params = copy.deepcopy(conv_params)
    conv_activation = conv_params.get("activation", "linear")
    conv_params["activation"] = "linear"

    # TODO restructure this
    if isinstance(conv_type, str):
        conv_type = ConvType.from_string(conv_type)
    if "depth_multiplier" in conv_params:
        if conv_type != ConvType.CONV2D_DEPTHWISE:
            conv_type = ConvType.CONV2D_DEPTHWISE
    if "dilation_rate" in conv_params:
        if conv_type != ConvType.CONV2D_TRANSPOSE:
            conv_type = ConvType.CONV2D_TRANSPOSE

    # --- set up stack of operation
    x = input_layer

    # --- convolution
    if conv_type == ConvType.CONV2D:
        x = keras.layers.Conv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_DEPTHWISE:
        x = keras.layers.DepthwiseConv2D(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_TRANSPOSE:
        x = keras.layers.Conv2DTranspose(**conv_params)(x)
    elif conv_type == ConvType.CONV2D_SEPARABLE:
        x = keras.layers.SeparableConv2D(**conv_params)(x)
    else:
        raise ValueError(f"don't know how to handle this [{conv_type}]")

    # --- perform post convolution normalizations and activation
    if use_bn:
        x = keras.layers.BatchNormalization(**bn_params)(x)
    if use_ln:
        x = keras.layers.LayerNormalization(**ln_params)(x)

    # --- perform activation post normalization
    if (conv_activation is not None and
            conv_activation != "linear"):
        x = activation_wrapper(conv_activation)(x)

    # --- dropout
    if use_dropout:
        x = keras.layers.Dropout(**dropout_params)(x)

    if use_dropout_2d:
        x = keras.layers.SpatialDropout2D(**dropout_2d_params)(x)

    return x

# ---------------------------------------------------------------------
