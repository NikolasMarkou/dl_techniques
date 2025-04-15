"""ConvNeXt V2 models for Keras.

References:
- [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
"""


import keras
import tensorflow as tf
from keras import utils, backend
from keras.api.models import Sequential
from keras.api.applications import imagenet_utils
from typing import List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.convnext_v2_block import ConvNextV2Block, ConvNextV2Config
from dl_techniques.layers.stochastic_depth import StochasticDepth

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_EPSILON = 1e-6
DEFAULT_KERNEL_INITIALIZER = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
DEFAULT_KERNEL_SIZE = 7
DEFAULT_STEM_KERNEL_SIZE = 4
DEFAULT_STEM_STRIDE = 4
DEFAULT_DOWNSAMPLE_KERNEL_SIZE = 2
DEFAULT_DOWNSAMPLE_STRIDE = 2
DEFAULT_PADDING = "same"
DEFAULT_BIAS_INITIALIZER = keras.initializers.Zeros()

# Model variant configurations
MODEL_CONFIGS = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "dims": [40, 80, 160, 320],
        "default_size": 224,
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "dims": [48, 96, 192, 384],
        "default_size": 224,
    },
    "pico": {
        "depths": [2, 2, 6, 2],
        "dims": [64, 128, 256, 512],
        "default_size": 224,
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "dims": [80, 160, 320, 640],
        "default_size": 224,
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
        "default_size": 224,
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
        "default_size": 224,
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "dims": [192, 384, 768, 1536],
        "default_size": 224,
    },
    "huge": {
        "depths": [3, 3, 27, 3],
        "dims": [352, 704, 1408, 2816],
        "default_size": 224,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  References:
    - [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders]
      (https://arxiv.org/abs/2301.00808)

  For image classification use cases, see
  [this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).
  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: Each Keras Application expects a specific kind of input preprocessing.
  For ConvNeXt V2, preprocessing is included in the model using a `Normalization`
  layer. ConvNeXt V2 models expect their inputs to be float or uint8 tensors of
  pixels with values in the [0-255] range.

  When calling the `summary()` method after instantiating a ConvNeXt V2 model,
  prefer setting the `expand_nested` argument `summary()` to `True` to better
  investigate the instantiated model.

  Args:
    include_top: Whether to include the fully-connected
      layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
      or the path to the weights file to be loaded. Defaults to `None`.
    input_tensor: Optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
      if `include_top` is False.
      It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
      when `include_top` is `False`. Defaults to None.
      - `None` means that the output of the model will be
        the 4D tensor output of the last convolutional layer.
      - `avg` means that global average pooling
        will be applied to the output of the
        last convolutional layer, and thus
        the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
        be applied.
    classes: Optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified. Defaults to 1000 (number of
      ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      Defaults to `"softmax"`.
    kernel_regularizer: Optional regularizer for the kernel weights.

  Returns:
    A `keras.Model` instance.
"""


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def PreStem(name: Optional[str] = None) -> callable:
    """Normalizes inputs with ImageNet-1k mean and std.

    Args:
        name: Name prefix for the layers.

    Returns:
        A function that applies normalization to inputs.
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(x):
        x = keras.layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=name + "_prestem_normalization",
        )(x)
        return x

    return apply


def Head(
        num_classes: int = 1000,
        classifier_activation: Optional[Union[str, callable]] = None,
        head_init_scale: float = 1.0,
        name: Optional[str] = None
) -> callable:
    """Implementation of classification head for ConvNeXt V2.

    Args:
        num_classes: Number of classes for the classification layer.
        classifier_activation: Activation function for the classification layer.
        head_init_scale: Scaling factor for classification layer initialization.
        name: Name prefix for the layers.

    Returns:
        A function that applies the classification head to inputs.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = keras.layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = keras.layers.LayerNormalization(
            epsilon=DEFAULT_EPSILON, name=name + "_head_layernorm"
        )(x)
        x = keras.layers.Dense(
            num_classes,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=0.02 * head_init_scale
            ),
            bias_initializer=DEFAULT_BIAS_INITIALIZER,
            activation=classifier_activation,
            name=name + "_head_dense",
        )(x)
        return x

    return apply


def ConvNextV2BlockFunc(
        projection_dim: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        drop_path_rate: float = 0.0,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None
) -> callable:
    """ConvNeXt V2 block function.

    Args:
        projection_dim: Number of filters for convolution layers.
        kernel_size: Size of the depthwise convolution kernel.
        drop_path_rate: Stochastic depth probability.
        kernel_regularizer: Optional regularizer for kernel weights.
        name: Optional name for the block.

    Returns:
        A function representing a ConvNeXtV2Block.
    """
    if name is None:
        name = "convnextv2_block_" + str(backend.get_uid("convnextv2_block"))

    def apply(inputs):
        # Store input before block for residual connection
        residual = inputs

        # Create block configuration
        block_config = ConvNextV2Config(
            kernel_size=kernel_size,
            filters=projection_dim,
            kernel_regularizer=kernel_regularizer,
        )

        # Apply ConvNextV2Block
        x = ConvNextV2Block(
            conv_config=block_config,
            dropout_rate=0.0,  # We handle dropout with StochasticDepth
            name=name
        )(inputs)

        # Apply stochastic depth if needed
        if drop_path_rate > 0:
            x = StochasticDepth(
                drop_path_rate=drop_path_rate,
                name=name + "_stochastic_depth"
            )(x)

        # Add residual connection
        x = keras.layers.Add(name=name + "_residual")([residual, x])
        return x

    return apply


# ---------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------

def ConvNextV2(
        depths: List[int],
        dims: List[int],
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        default_size: int = 224,
        model_name: str = "convnextv2",
        include_preprocessing: bool = True,
        include_top: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """Instantiates the ConvNeXt V2 architecture.

    Args:
        depths: Number of blocks in each stage.
        dims: Feature dimensions for each stage.
        drop_path_rate: Stochastic depth rate.
        head_init_scale: Initial scaling for classification head.
        default_size: Default input image size.
        model_name: Name of the model.
        include_preprocessing: Whether to include preprocessing normalization.
        include_top: Whether to include classification head.
        weights: Path to weights file to load.
        input_tensor: Optional input tensor.
        input_shape: Optional input shape tuple.
        pooling: Feature pooling mode when include_top is False.
        classes: Number of output classes.
        classifier_activation: Activation for classification layer.
        kernel_regularizer: Weight regularization.

    Returns:
        A Keras model instance.
    """
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = inputs
    if include_preprocessing:
        channel_axis = 3 if backend.image_data_format() == "channels_last" else 1
        num_channels = input_shape[channel_axis - 1]
        if num_channels == 3:
            x = PreStem(name=model_name)(x)

    # Stem block
    stem = Sequential(
        [
            keras.layers.Conv2D(
                dims[0],
                kernel_size=DEFAULT_STEM_KERNEL_SIZE,
                strides=DEFAULT_STEM_STRIDE,
                padding=DEFAULT_PADDING,
                kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
                kernel_regularizer=kernel_regularizer,
                use_bias=True,
                name=model_name + "_stem_conv",
            ),
            keras.layers.LayerNormalization(
                epsilon=DEFAULT_EPSILON, name=model_name + "_stem_layernorm"
            ),
        ],
        name=model_name + "_stem",
    )

    # Downsampling blocks
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = 3
    for i in range(num_downsample_layers):
        downsample_layer = Sequential(
            [
                keras.layers.LayerNormalization(
                    epsilon=DEFAULT_EPSILON,
                    name=model_name + "_downsampling_layernorm_" + str(i),
                ),
                keras.layers.Conv2D(
                    dims[i + 1],
                    kernel_size=DEFAULT_DOWNSAMPLE_KERNEL_SIZE,
                    strides=DEFAULT_DOWNSAMPLE_STRIDE,
                    padding=DEFAULT_PADDING,
                    kernel_initializer=DEFAULT_KERNEL_INITIALIZER,
                    kernel_regularizer=kernel_regularizer,
                    use_bias=True,
                    name=model_name + "_downsampling_conv_" + str(i),
                ),
            ],
            name=model_name + "_downsampling_block_" + str(i),
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth rates
    depth_drop_rates = tf.linspace(0.0, drop_path_rate, sum(depths))
    depth_drop_rates = tf.split(depth_drop_rates, depths)

    # First apply downsampling blocks and then apply ConvNeXt stages
    cur = 0
    num_convnext_blocks = 4

    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNextV2BlockFunc(
                projection_dim=dims[i],
                kernel_size=DEFAULT_KERNEL_SIZE,
                drop_path_rate=float(depth_drop_rates[i][j]),
                kernel_regularizer=kernel_regularizer,
                name=model_name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    if include_top:
        x = Head(
            num_classes=classes,
            classifier_activation=classifier_activation,
            head_init_scale=head_init_scale,
            name=model_name,
        )(x)
    else:
        if pooling == "avg":
            x = keras.layers.GlobalAveragePooling2D(name=model_name + "_global_avg_pool")(x)
        elif pooling == "max":
            x = keras.layers.GlobalMaxPooling2D(name=model_name + "_global_max_pool")(x)
        x = keras.layers.LayerNormalization(
            epsilon=DEFAULT_EPSILON, name=model_name + "_final_layernorm"
        )(x)

    # Create model
    model = keras.api.Model(inputs=inputs, outputs=x, name=model_name)

    # Load weights if provided
    if weights is not None:
        model.load_weights(weights)

    return model


# ---------------------------------------------------------------------
# Model Variants
# ---------------------------------------------------------------------

def ConvNextV2Atto(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Atto model with 3.7M parameters."""
    config = MODEL_CONFIGS["atto"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_atto",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Femto(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Femto model with 5.2M parameters."""
    config = MODEL_CONFIGS["femto"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_femto",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Pico(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Pico model with 9.1M parameters."""
    config = MODEL_CONFIGS["pico"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_pico",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Nano(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Nano model with 15.6M parameters."""
    config = MODEL_CONFIGS["nano"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_nano",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Tiny(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Tiny model with 28.6M parameters."""
    config = MODEL_CONFIGS["tiny"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_tiny",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Base(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Base model with 89M parameters."""
    config = MODEL_CONFIGS["base"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_base",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Large(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Large model with 198M parameters."""
    config = MODEL_CONFIGS["large"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_large",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


def ConvNextV2Huge(
        include_top: bool = True,
        include_preprocessing: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[keras.KerasTensor] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        pooling: Optional[str] = None,
        classes: int = 1000,
        classifier_activation: Union[str, callable, None] = "softmax",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
) -> keras.Model:
    """ConvNeXt V2 Huge model with 660M parameters."""
    config = MODEL_CONFIGS["huge"]
    return ConvNextV2(
        depths=config["depths"],
        dims=config["dims"],
        default_size=config["default_size"],
        model_name="convnextv2_huge",
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        kernel_regularizer=kernel_regularizer,
    )


# Add docstrings to all model variants
ConvNextV2Atto.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Atto")
ConvNextV2Femto.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Femto")
ConvNextV2Pico.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Pico")
ConvNextV2Nano.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Nano")
ConvNextV2Tiny.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Tiny")
ConvNextV2Base.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Base")
ConvNextV2Large.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Large")
ConvNextV2Huge.__doc__ = BASE_DOCSTRING.format(name="ConvNextV2Huge")


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the convnext model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
        x: A floating point `numpy.array` or a `tf.Tensor`.
        data_format: Optional data format of the image tensor/array.

    Returns:
        Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


def decode_predictions(preds, top=5):
    """Decodes the prediction of a ConvNextV2 model.

    Args:
        preds: Numpy array encoding a batch of predictions.
        top: Integer, how many top-guesses to return.

    Returns:
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.
    """
    return imagenet_utils.decode_predictions(preds, top=top)