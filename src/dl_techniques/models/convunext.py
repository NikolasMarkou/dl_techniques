"""ConvUNext models for Keras.

References:
- [SKOOTS: Skeleton oriented object segmentation for mitochondria, 2023]
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
"""

import copy
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import tensorflow as tf
import keras
from keras import utils, backend

from dl_techniques.layers.conv2d_builder import conv2d_wrapper
# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils import logger
from dl_techniques.regularizers.soft_orthogonal import (
    DEFAULT_SOFTORTHOGONAL_STDDEV,
    SoftOrthonormalConstraintRegularizer
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

INPUT_TENSOR_STR = "input_tensor"
MASK_TENSOR_STR = "mask_tensor"
DEFAULT_LN_EPSILON = 1e-6
DEFAULT_BN_MOMENTUM = 0.9
DEFAULT_BN_EPSILON = 1e-5

# ---------------------------------------------------------------------
# Base Model Configuration
# ---------------------------------------------------------------------

class ConvUNextConfig:
    """Configuration for ConvUNext model."""

    def __init__(
        self,
        depth: int = 5,
        width: int = 1,
        encoder_kernel_size: int = 5,
        decoder_kernel_size: int = 3,
        filters: int = 32,
        max_filters: int = -1,
        filters_level_multiplier: float = 2.0,
        activation: str = "leaky_relu_01",
        upsample_type: str = "bilinear",
        use_ln: bool = True,
        use_gamma: bool = True,
        use_global_gamma: bool = True,
        use_bias: bool = False,
        use_concat: bool = True,
        use_half_resolution: bool = False,
        use_self_attention: bool = False,
        use_attention_gates: bool = False,
        use_soft_orthogonal_regularization: bool = False,
        use_soft_orthonormal_regularization: bool = False,
        use_output_normalization: bool = False,
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = "l2",
        kernel_initializer: str = "glorot_normal",
        dropout_rate: float = -1,
        depth_drop_rate: float = 0.0,
        spatial_dropout_rate: float = -1,
        multiple_scale_outputs: bool = True,
        convolutional_self_attention_dropout_rate: float = 0.0,
        convolutional_self_attention_strides: int = 1,
        convolutional_self_attention_blocks: int = -1,
        name: str = "convunext",
    ):
        """Initialize ConvUNext configuration.

        Args:
            depth: Number of levels to go down.
            width: Number of horizontal nodes.
            encoder_kernel_size: Kernel size for encoder convolutional layers.
            decoder_kernel_size: Kernel size for decoder convolutional layers.
            filters: Number of filters for base convolutional layer.
            max_filters: Maximum number of filters.
            filters_level_multiplier: Filter multiplication factor per level.
            activation: Activation function.
            upsample_type: Type of upsampling ('bilinear', 'nearest', etc.).
            use_ln: Whether to use layer normalization.
            use_gamma: Whether to use gamma learning in ConvNeXt blocks.
            use_global_gamma: Whether to use global gamma learning.
            use_bias: Whether to use bias in convolutional layers.
            use_concat: Whether to concatenate skip connections (True) or add them (False).
            use_half_resolution: Whether to downsample input and upsample output.
            use_self_attention: Whether to use self-attention at bottom layer.
            use_attention_gates: Whether to use attention gates between depths.
            use_soft_orthogonal_regularization: Whether to use soft orthogonal regularization.
            use_soft_orthonormal_regularization: Whether to use soft orthonormal regularization.
            use_output_normalization: Whether to normalize outputs.
            kernel_regularizer: Regularization for kernel weights.
            kernel_initializer: Initialization for kernel weights.
            dropout_rate: Dropout rate (-1 to disable).
            depth_drop_rate: Stochastic depth drop rate.
            spatial_dropout_rate: Spatial dropout rate (-1 to disable).
            multiple_scale_outputs: Whether to output at multiple scales.
            convolutional_self_attention_dropout_rate: Dropout rate for self-attention.
            convolutional_self_attention_strides: Strides for self-attention.
            convolutional_self_attention_blocks: Number of self-attention blocks.
            name: Name prefix for the model.
        """
        self.depth = depth
        self.width = width
        self.encoder_kernel_size = encoder_kernel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.filters = filters
        self.max_filters = max_filters
        self.filters_level_multiplier = filters_level_multiplier
        self.activation = activation
        self.upsample_type = upsample_type
        self.use_ln = use_ln
        self.use_gamma = use_gamma
        self.use_global_gamma = use_global_gamma
        self.use_bias = use_bias
        self.use_concat = use_concat
        self.use_half_resolution = use_half_resolution
        self.use_self_attention = use_self_attention
        self.use_attention_gates = use_attention_gates
        self.use_soft_orthogonal_regularization = use_soft_orthogonal_regularization
        self.use_soft_orthonormal_regularization = use_soft_orthonormal_regularization
        self.use_output_normalization = use_output_normalization
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.depth_drop_rate = depth_drop_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.multiple_scale_outputs = multiple_scale_outputs
        self.convolutional_self_attention_dropout_rate = convolutional_self_attention_dropout_rate
        self.convolutional_self_attention_strides = convolutional_self_attention_strides
        self.convolutional_self_attention_blocks = convolutional_self_attention_blocks
        self.name = name

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def _validate_config(config: ConvUNextConfig) -> None:
    """Validates the ConvUNext configuration.

    Args:
        config: The ConvUNext configuration to validate.

    Raises:
        ValueError: If any configuration parameters are invalid.
    """
    if config.filters <= 0:
        raise ValueError("filters must be > 0")

    if config.width is None or config.width <= 0:
        config.width = 1

    if config.depth <= 0 or config.width <= 0:
        raise ValueError("depth and width must be > 0")

    if config.convolutional_self_attention_blocks <= 0 and config.use_self_attention:
        config.convolutional_self_attention_blocks = config.width

    if (config.convolutional_self_attention_dropout_rate < 0 or
            config.convolutional_self_attention_dropout_rate > 1):
        raise ValueError("convolutional_self_attention_dropout_rate must be >= 0 and <= 1")

    if (config.use_soft_orthonormal_regularization and
            config.use_soft_orthogonal_regularization):
        raise ValueError(
            "only one use_soft_orthonormal_regularization or "
            "use_soft_orthogonal_regularization must be turned on")

def _create_layer_params(config: ConvUNextConfig) -> Tuple[Dict, List, List, List, List, List, List, List]:
    """Creates parameters for various layers in the model.

    Args:
        config: The ConvUNext configuration.

    Returns:
        A tuple containing parameters for different types of layers.
    """
    # Setup layer normalization parameters
    ln_params = dict(
        scale=True,
        center=config.use_bias,
        epsilon=DEFAULT_LN_EPSILON
    )

    # Setup dropout parameters
    dropout_params = None
    if config.dropout_rate > 0.0:
        dropout_params = {"rate": config.dropout_rate}

    dropout_2d_params = None
    if config.spatial_dropout_rate > 0.0:
        dropout_2d_params = {"rate": config.spatial_dropout_rate}

    # Calculate stochastic depth drop rates
    depth_drop_rates = list(np.linspace(
        start=max(0.0, config.depth_drop_rate/config.width),
        stop=max(0.0, config.depth_drop_rate),
        num=config.width
    ))

    # Base convolution parameters
    base_conv_params = dict(
        kernel_size=config.encoder_kernel_size,
        filters=config.filters,
        strides=(1, 1),
        padding="same",
        use_bias=config.use_bias,
        activation=config.activation,
        kernel_regularizer=config.kernel_regularizer,
        kernel_initializer=config.kernel_initializer
    )

    # Initialize parameter lists for different layers
    conv_params = []
    conv_params_up = []
    conv_params_down = []
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    # Calculate base filters
    base_filters = config.filters
    if config.use_half_resolution:
        base_filters *= 2

    # Create layer parameters for each depth level
    for d in range(config.depth):
        filters_level = int(round(base_filters * max(1, config.filters_level_multiplier ** d)))
        if config.max_filters > 0:
            filters_level = min(config.max_filters, filters_level)

        filters_level_next = int(round(base_filters * max(1, config.filters_level_multiplier ** (d + 1))))
        if config.max_filters > 0:
            filters_level_next = min(config.max_filters, filters_level_next)

        # Default conv parameters
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = 3
        params["activation"] = "linear"
        params["use_bias"] = config.use_bias
        conv_params.append(params)

        # First residual conv parameters (depthwise)
        conv_params_res_1.append(dict(
            kernel_size=config.encoder_kernel_size,
            depth_multiplier=1,
            strides=(1, 1),
            padding="same",
            use_bias=config.use_bias,
            activation="linear",
            depthwise_regularizer=config.kernel_regularizer,
            depthwise_initializer=config.kernel_initializer
        ))

        # Second residual conv parameters (pointwise 1)
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["use_bias"] = config.use_bias
        params["activation"] = config.activation
        params["filters"] = filters_level * 4
        params["kernel_initializer"] = config.kernel_initializer
        params["kernel_regularizer"] = config.kernel_regularizer
        conv_params_res_2.append(params)

        # Third residual conv parameters (pointwise 2)
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["use_bias"] = config.use_bias
        params["activation"] = "linear"
        params["filters"] = filters_level
        params["kernel_initializer"] = config.kernel_initializer
        params["kernel_regularizer"] = config.kernel_regularizer
        conv_params_res_3.append(params)

        # Downsampling conv parameters
        params = copy.deepcopy(base_conv_params)
        params["use_bias"] = config.use_bias
        params["filters"] = filters_level_next
        params["activation"] = "linear"  # Downsampling activation
        conv_params_down.append(params)

        # Upsampling conv parameters
        params = copy.deepcopy(base_conv_params)
        params["use_bias"] = config.use_bias
        params["filters"] = filters_level
        params["activation"] = "linear"  # Upsampling activation
        conv_params_up.append(params)

    return (ln_params, dropout_params, dropout_2d_params, depth_drop_rates,
            conv_params, conv_params_up, conv_params_down,
            conv_params_res_1, conv_params_res_2, conv_params_res_3)

def _create_node_dependencies(depth: int) -> Tuple[Dict, List, set]:
    """Creates dependency mappings for the U-Net structure.

    Args:
        depth: The depth of the U-Net.

    Returns:
        A tuple containing node dependencies, nodes to visit, and visited nodes.
    """
    # Create node dependency mapping
    nodes_dependencies = {}
    for d in range(0, depth, 1):
        if d == (depth - 1):
            # Add only left dependency at bottom level
            nodes_dependencies[(d, 1)] = [(d, 0)]
        else:
            # Add left and bottom dependency
            nodes_dependencies[(d, 1)] = [(d, 0), (d + 1, 1)]

    # Initialize tracking variables
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set([(depth - 1, 0), (depth - 1, 1)])

    return nodes_dependencies, nodes_to_visit, nodes_visited

def _apply_stem(
    x: tf.Tensor,
    mask_layer: tf.Tensor,
    config: ConvUNextConfig
) -> tf.Tensor:
    """Applies the stem block to the input tensor.

    Args:
        x: Input tensor.
        mask_layer: Mask tensor.
        config: Model configuration.

    Returns:
        Processed tensor after stem block.
    """
    # Apply mask
    x = x * (1.0 - mask_layer)

    # Apply stem convolution
    x = conv2d_wrapper(
        input_layer=x,
        ln_params=None,
        bn_params=dict(
            scale=True,
            center=config.use_bias,
            momentum=DEFAULT_BN_MOMENTUM,
            epsilon=DEFAULT_BN_EPSILON
        ),
        conv_params=dict(
            kernel_size=(5, 5),
            filters=config.filters,
            strides=(1, 1),
            padding="same",
            use_bias=config.use_bias,
            activation=config.activation,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer
        ))

    # Apply optional half-resolution downsampling
    if config.use_half_resolution:
        x = conv2d_wrapper(
            input_layer=x,
            ln_params=None,
            bn_params=dict(
                scale=True,
                center=config.use_bias,
                momentum=DEFAULT_BN_MOMENTUM,
                epsilon=DEFAULT_BN_EPSILON
            ),
            conv_params=dict(
                kernel_size=(2, 2),
                filters=config.filters * 2,  # Double filters in half resolution
                strides=(2, 2),
                padding="same",
                use_bias=config.use_bias,
                activation="linear",
                kernel_regularizer=config.kernel_regularizer,
                kernel_initializer=config.kernel_initializer
            ))

    return x

def _create_masks_and_coords(
    mask_layer: tf.Tensor,
    coords_layer: tf.Tensor,
    depth: int,
    use_half_resolution: bool
) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Creates downsampled masks and coordinate layers.

    Args:
        mask_layer: Input mask tensor.
        coords_layer: Input coordinates tensor.
        depth: Model depth.
        use_half_resolution: Whether to use half resolution.

    Returns:
        Lists of mask and coordinate tensors at different scales.
    """
    if use_half_resolution:
        masks = [
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(mask_layer)
        ]
        coords = [
            keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(coords_layer)
        ]
    else:
        masks = [mask_layer]
        coords = [coords_layer]

    # Create downsampled versions for each depth
    for d in range(depth - 1):
        m = masks[-1]
        m = keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2), padding="same")(m)
        masks.append(m)

        c = coords[-1]
        c = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(c)
        coords.append(c)

    return masks, coords

def _apply_encoder_block(
    x: tf.Tensor,
    d: int,
    w: int,
    depth_drop_rates: List[float],
    conv_params_res_1: List[Dict],
    conv_params_res_2: List[Dict],
    conv_params_res_3: List[Dict],
    ln_params: Dict,
    dropout_params: Optional[Dict],
    dropout_2d_params: Optional[Dict],
    width: int,
    config: ConvUNextConfig
) -> tf.Tensor:
    """Applies an encoder block to the input tensor.

    Args:
        x: Input tensor.
        d: Current depth level.
        w: Current width index.
        depth_drop_rates: List of stochastic depth rates.
        conv_params_res_1: Parameters for first residual convolution.
        conv_params_res_2: Parameters for second residual convolution.
        conv_params_res_3: Parameters for third residual convolution.
        ln_params: Layer normalization parameters.
        dropout_params: Dropout parameters.
        dropout_2d_params: Spatial dropout parameters.
        width: Model width.
        config: Model configuration.

    Returns:
        Processed tensor after encoder block.
    """
    # Store input for residual connection
    x_skip = x

    # Apply transformer block if using self-attention at the bottom level
    if config.use_self_attention and d == config.depth - 1:
        x = ConvolutionalTransformerBlock(
            dim=conv_params_res_3[d]["filters"],
            num_heads=8,
            mlp_ratio=4,
            use_bias=config.use_bias,
            dropout_rate=0.25,
            attention_dropout=config.convolutional_self_attention_dropout_rate,
            activation=config.activation,
            use_gamma=config.use_gamma,
            use_global_gamma=config.use_global_gamma,
            use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization,
            name=f"encoder_{d}_{w}"
        )(x)
    else:
        # Apply ConvNext block
        x = ConvNextBlock(
            name=f"encoder_{d}_{w}",
            conv_params_1=conv_params_res_1[d],
            conv_params_2=conv_params_res_2[d],
            conv_params_3=conv_params_res_3[d],
            ln_params=ln_params,
            bn_params=None,
            dropout_params=dropout_params,
            use_gamma=config.use_gamma,
            use_global_gamma=config.use_global_gamma,
            dropout_2d_params=dropout_2d_params,
            use_soft_orthogonal_regularization=config.use_soft_orthogonal_regularization,
            use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization
        )(x)

    # Apply residual connection with stochastic depth if shapes match
    if (x_skip.shape[-1] == x.shape[-1] and
            not (config.use_self_attention and d == config.depth - 1)):
        if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
            x = StochasticDepth(depth_drop_rates[w])(x)
        x = keras.layers.Add()([x_skip, x])

    return x

def _apply_decoder_block(
    x: tf.Tensor,
    d: int,
    w: int,
    depth_drop_rates: List[float],
    conv_params_res_1: List[Dict],
    conv_params_res_2: List[Dict],
    conv_params_res_3: List[Dict],
    ln_params: Dict,
    dropout_params: Optional[Dict],
    dropout_2d_params: Optional[Dict],
    width: int,
    decoder_kernel_size: int,
    config: ConvUNextConfig
) -> tf.Tensor:
    """Applies a decoder block to the input tensor.

    Args:
        x: Input tensor.
        d: Current depth level.
        w: Current width index.
        depth_drop_rates: List of stochastic depth rates.
        conv_params_res_1: Parameters for first residual convolution.
        conv_params_res_2: Parameters for second residual convolution.
        conv_params_res_3: Parameters for third residual convolution.
        ln_params: Layer normalization parameters.
        dropout_params: Dropout parameters.
        dropout_2d_params: Spatial dropout parameters.
        width: Model width.
        decoder_kernel_size: Kernel size for decoder.
        config: Model configuration.

    Returns:
        Processed tensor after decoder block.
    """
    # Store input for residual connection
    x_skip = x

    # Create decoder params based on encoder params but with decoder kernel size
    params = copy.deepcopy(conv_params_res_1[d])
    params["kernel_size"] = decoder_kernel_size

    # Apply ConvNext block
    x = ConvNextBlock(
        name=f"decoder_{d}_{w}",
        conv_params_1=params,
        conv_params_2=conv_params_res_2[d],
        conv_params_3=conv_params_res_3[d],
        ln_params=ln_params,
        bn_params=None,
        use_gamma=config.use_gamma,
        use_global_gamma=config.use_global_gamma,
        dropout_params=dropout_params,
        dropout_2d_params=dropout_2d_params,
        use_soft_orthogonal_regularization=config.use_soft_orthogonal_regularization,
        use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization
    )(x)

    # Apply residual connection with stochastic depth if shapes match
    if x_skip.shape[-1] == x.shape[-1]:
        if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
            x = StochasticDepth(depth_drop_rates[w])(x)
        x = keras.layers.Add()([x_skip, x])

    return x

def _finalize_outputs(
    output_layers: List[tf.Tensor],
    ln_params: Dict,
    filters: int,
    kernel_regularizer: Union[str, keras.regularizers.Regularizer],
    kernel_initializer: str,
    use_bias: bool,
    use_half_resolution: bool,
    use_output_normalization: bool
) -> List[tf.Tensor]:
    """Finalizes output layers with upsampling or normalization as needed.

    Args:
        output_layers: List of output tensors.
        ln_params: Layer normalization parameters.
        filters: Number of filters.
        kernel_regularizer: Kernel regularizer.
        kernel_initializer: Kernel initializer.
        use_bias: Whether to use bias.
        use_half_resolution: Whether using half resolution.
        use_output_normalization: Whether to normalize outputs.

    Returns:
        List of finalized output tensors.
    """
    # Apply upsampling for half-resolution mode
    if use_half_resolution:
        for i, layer in enumerate(output_layers):
            # Upsample back to original resolution
            layer = keras.layers.UpSampling2D(
                size=(2, 2),
                interpolation="nearest"
            )(layer)

            # Apply convolution to refine features
            layer = conv2d_wrapper(
                input_layer=layer,
                bn_params=None,
                ln_params=ln_params if use_output_normalization else None,
                conv_params=dict(
                    kernel_size=(3, 3),
                    filters=filters,
                    strides=(1, 1),
                    padding="same",
                    use_bias=use_bias,
                    activation="linear",
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer
                )
            )
            output_layers[i] = layer
    # Apply normalization if requested
    elif use_output_normalization:
        for i in range(len(output_layers)):
            output_layers[i] = keras.layers.LayerNormalization(
                **ln_params, name=f"norm_output_{i}"
            )(output_layers[i])

    # Name output layers
    for i in range(len(output_layers)):
        output_layers[i] = keras.layers.Layer(
            name=f"decoder_output_{i}"
        )(output_layers[i])

    return output_layers

# ---------------------------------------------------------------------
# Main Model Builder
# ---------------------------------------------------------------------

def ConvUNext(
    input_dims: Tuple[int, ...],
    depth: int = 5,
    width: int = 1,
    encoder_kernel_size: int = 5,
    decoder_kernel_size: int = 3,
    filters: int = 32,
    max_filters: int = -1,
    filters_level_multiplier: float = 2.0,
    activation: str = "leaky_relu_01",
    upsample_type: str = "bilinear",
    use_ln: bool = True,
    use_gamma: bool = True,
    use_global_gamma: bool = True,
    use_bias: bool = False,
    use_concat: bool = True,
    use_half_resolution: bool = False,
    use_self_attention: bool = False,
    use_attention_gates: bool = False,
    use_soft_orthogonal_regularization: bool = False,
    use_soft_orthonormal_regularization: bool = False,
    use_output_normalization: bool = False,
    kernel_regularizer: Union[str, keras.regularizers.Regularizer] = "l2",
    kernel_initializer: str = "glorot_normal",
    dropout_rate: float = -1,
    depth_drop_rate: float = 0.0,
    spatial_dropout_rate: float = -1,
    multiple_scale_outputs: bool = True,
    convolutional_self_attention_dropout_rate: float = 0.0,
    convolutional_self_attention_strides: int = 1,
    convolutional_self_attention_blocks: int = -1,
    name: str = "convunext",
    **kwargs
) -> Tuple[keras.Model, None, None]:
    """Builds a ConvUNext model with ConvNeXt blocks and U-Net architecture.

    Args:
        input_dims: Model input dimensions (H, W, C).
        depth: Number of levels to go down.
        width: Number of horizontal nodes.
        encoder_kernel_size: Kernel size for encoder convolutional layers.
        decoder_kernel_size: Kernel size for decoder convolutional layers.
        filters: Number of filters for base convolutional layer.
        max_filters: Maximum number of filters.
        filters_level_multiplier: Filter multiplication factor per level.
        activation: Activation function.
        upsample_type: Type of upsampling ('bilinear', 'nearest', etc.).
        use_ln: Whether to use layer normalization.
        use_gamma: Whether to use gamma learning in ConvNeXt blocks.
        use_global_gamma: Whether to use global gamma learning.
        use_bias: Whether to use bias in convolutional layers.
        use_concat: Whether to concatenate skip connections (True) or add them (False).
        use_half_resolution: Whether to downsample input and upsample output.
        use_self_attention: Whether to use self-attention at bottom layer.
        use_attention_gates: Whether to use attention gates between depths.
        use_soft_orthogonal_regularization: Whether to use soft orthogonal regularization.
        use_soft_orthonormal_regularization: Whether to use soft orthonormal regularization.
        use_output_normalization: Whether to normalize outputs.
        kernel_regularizer: Regularization for kernel weights.
        kernel_initializer: Initialization for kernel weights.
        dropout_rate: Dropout rate (-1 to disable).
        depth_drop_rate: Stochastic depth drop rate.
        spatial_dropout_rate: Spatial dropout rate (-1 to disable).
        multiple_scale_outputs: Whether to output at multiple scales.
        convolutional_self_attention_dropout_rate: Dropout rate for self-attention.
        convolutional_self_attention_strides: Strides for self-attention.
        convolutional_self_attention_blocks: Number of self-attention blocks.
        name: Name prefix for the model.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the model and two None values (for compatibility).
    """
    # Log build information and check unused arguments
    logger.info("Building ConvUNext backbone")
    if len(kwargs) > 0:
        logger.info(f"Parameters not used: {kwargs}")

    # Create and validate configuration
    config = ConvUNextConfig(
        depth=depth,
        width=width,
        encoder_kernel_size=encoder_kernel_size,
        decoder_kernel_size=decoder_kernel_size,
        filters=filters,
        max_filters=max_filters,
        filters_level_multiplier=filters_level_multiplier,
        activation=activation,
        upsample_type=upsample_type.strip().lower(),
        use_ln=use_ln,
        use_gamma=use_gamma,
        use_global_gamma=use_global_gamma,
        use_bias=use_bias,
        use_concat=use_concat,
        use_half_resolution=use_half_resolution,
        use_self_attention=use_self_attention,
        use_attention_gates=use_attention_gates,
        use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
        use_soft_orthonormal_regularization=use_soft_orthonormal_regularization,
        use_output_normalization=use_output_normalization,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer.strip().lower(),
        dropout_rate=dropout_rate,
        depth_drop_rate=depth_drop_rate,
        spatial_dropout_rate=spatial_dropout_rate,
        multiple_scale_outputs=multiple_scale_outputs,
        convolutional_self_attention_dropout_rate=convolutional_self_attention_dropout_rate,
        convolutional_self_attention_strides=convolutional_self_attention_strides,
        convolutional_self_attention_blocks=convolutional_self_attention_blocks,
        name=name
    )
    _validate_config(config)

    # Create layer parameters
    (ln_params, dropout_params, dropout_2d_params, depth_drop_rates,
     conv_params, conv_params_up, conv_params_down,
     conv_params_res_1, conv_params_res_2, conv_params_res_3) = _create_layer_params(config)

    # Create node dependencies for U-Net structure
    nodes_dependencies, nodes_to_visit, nodes_visited = _create_node_dependencies(depth)
    nodes_output = {}

    # Define input layers
    input_layer = keras.Input(name=INPUT_TENSOR_STR, shape=input_dims)
    mask_layer = keras.Input(name=MASK_TENSOR_STR, shape=input_dims[:-1] + [1,])
    coords_layer = SpatialLayer()(mask_layer)

    # Create masks and coordinate tensors at different scales
    masks, coords = _create_masks_and_coords(
        mask_layer, coords_layer, depth, config.use_half_resolution)

    # Apply stem block
    x = _apply_stem(input_layer, mask_layer, config)

    # Build encoder path
    for d in range(depth):
        # Handle self-attention subsampling if needed
        if (config.use_self_attention and
                d == depth - 1 and
                config.convolutional_self_attention_strides > 1):
            subsampling_d = conv_params_res_1[d].copy()
            subsampling_d["kernel_size"] = config.convolutional_self_attention_strides
            subsampling_d["strides"] = (config.convolutional_self_attention_strides,
                                       config.convolutional_self_attention_strides)
            x = ConvNextBlock(
                name=f"subsampling_{d}",
                conv_params_1=subsampling_d,
                conv_params_2=conv_params_res_2[d],
                conv_params_3=conv_params_res_3[d],
                ln_params=ln_params,
                bn_params=None,
                dropout_params=dropout_params,
                use_gamma=config.use_gamma,
                use_global_gamma=config.use_global_gamma,
                dropout_2d_params=dropout_2d_params,
                use_soft_orthogonal_regularization=config.use_soft_orthogonal_regularization,
                use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization)(x)

        # Determine width at current level
        tmp_width = width
        if config.use_self_attention and d == depth - 1:
            tmp_width = config.convolutional_self_attention_blocks

        # Apply encoder blocks at current level
        for w in range(tmp_width):
            x = _apply_encoder_block(
                x, d, w, depth_drop_rates,
                conv_params_res_1, conv_params_res_2, conv_params_res_3,
                ln_params, dropout_params, dropout_2d_params, width, config)

        # Handle self-attention upsampling if needed
        if (config.use_self_attention and
                d == depth - 1 and
                config.convolutional_self_attention_strides > 1):
            x = conv2d_wrapper(
                input_layer=x,
                ln_params=None,
                bn_params=None,
                conv_type=ConvType.CONV2D_TRANSPOSE,
                conv_params=dict(
                    kernel_size=config.convolutional_self_attention_strides,
                    filters=conv_params_res_3[d]["filters"],
                    strides=config.convolutional_self_attention_strides,
                    padding="same",
                    use_bias=config.use_bias,
                    activation="linear",
                    kernel_regularizer=config.kernel_regularizer,
                    kernel_initializer=config.kernel_initializer
                ))

        # Store output at current level
        node_level = (d, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

        # Apply downsampling if not at bottom level
        if d != (depth - 1):
            # Concatenate with spatial information
            x = keras.layers.Concatenate(axis=-1)([x, coords[d], masks[d]])

            # Create downsampling block
            down_1 = copy.deepcopy(conv_params_res_1[d])
            down_1["strides"] = (2, 2)
            down_1["kernel_size"] = (4, 4)
            down_2 = copy.deepcopy(conv_params_res_2[d])
            down_3 = copy.deepcopy(conv_params_res_3[d])
            down_3["filters"] = conv_params_down[d]["filters"]

            # Apply downsampling
            x = ConvNextBlock(
                name=f"downsample_{d}",
                conv_params_1=down_1,
                conv_params_2=down_2,
                conv_params_3=down_3,
                ln_params=ln_params,
                bn_params=None,
                dropout_params=dropout_params,
                use_gamma=config.use_gamma,
                use_global_gamma=config.use_global_gamma,
                dropout_2d_params=dropout_2d_params,
                use_soft_orthogonal_regularization=config.use_soft_orthogonal_regularization,
                use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization)(x)

    # Add bottom level output to decoder path
    nodes_output[(depth - 1, 1)] = nodes_output[(depth - 1, 0)]

    # Build decoder path based on dependencies
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"Node: [{node}], "
                   f"nodes_visited: {nodes_visited}, "
                   f"nodes_to_visit: {nodes_to_visit}, "
                   f"dependencies: {nodes_dependencies[node]}")

        # Skip already processed nodes
        if node in nodes_visited:
            logger.info(f"Node: [{node}] already processed")
            continue

        # Make sure all dependencies are available
        dependencies = nodes_dependencies[node]
        dependencies_matched = all([
            (d in nodes_output) and (d in nodes_visited or d == node)
            for d in dependencies
        ])

        if not dependencies_matched:
            logger.info(f"Node: [{node}] dependencies not matched, continuing")
            nodes_to_visit.append(node)
            continue

        # Sort dependencies by depth (ascending)
        dependencies = sorted(list(dependencies),
                             key=lambda d: d[0],
                             reverse=False)

        logger.info(f"Processing node: {node}, "
                   f"dependencies: {dependencies}, "
                   f"nodes_output: {list(nodes_output.keys())}")

        # Collect input features from dependencies
        x_input = []
        for dependency in dependencies:
            logger.debug(f"Processing dependency: {dependency}")
            x = nodes_output[dependency]

            # Handle upsampling if coming from lower depth
            if dependency[0] == node[0]:
                pass
            elif dependency[0] > node[0]:
                logger.info("Upsampling here")
                x = upsample(
                    input_layer=x,
                    upsample_type=config.upsample_type,
                    ln_params=ln_params,
                    bn_params=None,
                    conv_params=conv_params_up[node[0]])
            else:
                raise ValueError(f"Node: {node}, dependencies: {dependencies}, "
                               f"should not be here")

            x_input.append(x)

        # Apply attention gates if requested
        if config.use_attention_gates and len(x_input) == 2:
            logger.debug(f"Adding AttentionGate at depth: [{node[0]}]")
            x_input[0] = AdditiveAttentionGate(
                use_bias=config.use_bias,
                use_bn=None,
                use_ln=config.use_ln,
                use_soft_orthogonal_regularization=config.use_soft_orthogonal_regularization,
                use_soft_orthonormal_regularization=config.use_soft_orthonormal_regularization,
                attention_channels=conv_params_res_3[node[0]]["filters"],
                kernel_initializer=config.kernel_initializer
            )(x_input)

        # Combine inputs
        d = node[0]
        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 1:
            if config.use_concat:
                # Concatenate features and spatial information
                x = keras.layers.Concatenate(axis=-1)(x_input)
                x = keras.layers.Concatenate(axis=-1)([x, coords[d], masks[d]])

                # Project features with 1x1 convolution
                params = copy.deepcopy(conv_params_res_3[node[0]])
                params["kernel_size"] = (1, 1)
                params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()
                params["kernel_initializer"] = keras.initializers.TruncatedNormal(
                    stddev=DEFAULT_SOFTORTHOGONAL_STDDEV, seed=1)

                x = conv2d_wrapper(
                    input_layer=x,
                    bn_params=None,
                    ln_params=ln_params,
                    conv_params=params)
            else:
                # Add features
                x = keras.layers.Add()(x_input)
        else:
            raise ValueError("Empty input list - this should never happen")

        # Apply decoder blocks
        for w in range(width):
            x = _apply_decoder_block(
                x, d, w, depth_drop_rates,
                conv_params_res_1, conv_params_res_2, conv_params_res_3,
                ln_params, dropout_params, dropout_2d_params,
                width, decoder_kernel_size, config)

        # Store output for this node
        nodes_output[node] = x
        nodes_visited.add(node)

    # Collect outputs at multiple scales if requested
    output_layers = []
    if multiple_scale_outputs:
        tmp_output_layers = []
        for d in range(1, depth, 1):
            if d < 0 or 1 < 0:
                logger.error(f"There is no node[{d},1] - please check your assumptions")
                continue
            x = nodes_output[(d, 1)]
            tmp_output_layers.append(x)

        # Reverse order so deeper levels come first
        output_layers += tmp_output_layers[::-1]

    # Add top-level output
    output_layers.append(nodes_output[(0, 1)])

    # Ensure deepest outputs come first (important for training)
    output_layers = output_layers[::-1]

    # Finalize outputs with upsampling/normalization as needed
    output_layers = _finalize_outputs(
        output_layers, ln_params, filters,
        kernel_regularizer, kernel_initializer,
        use_bias, use_half_resolution, use_output_normalization)

    # Create the model
    model = keras.Model(
        name=name,
        trainable=True,
        inputs=[input_layer, mask_layer],
        outputs=output_layers
    )

    return model, None, None

# ---------------------------------------------------------------------
# Model Variants
# ---------------------------------------------------------------------

def ConvUNextSmall(
    input_dims: Tuple[int, ...],
    **kwargs
) -> Tuple[keras.Model, None, None]:
    """Creates a small ConvUNext model with 32 base filters.

    Args:
        input_dims: Input dimensions (H, W, C).
        **kwargs: Additional arguments to pass to ConvUNext.

    Returns:
        A tuple containing the model and two None values.
    """
    return ConvUNext(
        input_dims=input_dims,
        depth=4,
        width=2,
        filters=32,
        name="convunext_small",
        **kwargs
    )

def ConvUNextMedium(
    input_dims: Tuple[int, ...],
    **kwargs
) -> Tuple[keras.Model, None, None]:
    """Creates a medium ConvUNext model with 48 base filters.

    Args:
        input_dims: Input dimensions (H, W, C).
        **kwargs: Additional arguments to pass to ConvUNext.

    Returns:
        A tuple containing the model and two None values.
    """
    return ConvUNext(
        input_dims=input_dims,
        depth=5,
        width=2,
        filters=48,
        name="convunext_medium",
        **kwargs
    )

def ConvUNextLarge(
    input_dims: Tuple[int, ...],
    **kwargs
) -> Tuple[keras.Model, None, None]:
    """Creates a large ConvUNext model with 64 base filters.

    Args:
        input_dims: Input dimensions (H, W, C).
        **kwargs: Additional arguments to pass to ConvUNext.

    Returns:
        A tuple containing the model and two None values.
    """
    return ConvUNext(
        input_dims=input_dims,
        depth=5,
        width=3,
        filters=64,
        name="convunext_large",
        **kwargs
    )

def ConvUNextXLarge(
    input_dims: Tuple[int, ...],
    **kwargs
) -> Tuple[keras.Model, None, None]:
    """Creates an extra large ConvUNext model with 96 base filters.

    Args:
        input_dims: Input dimensions (H, W, C).
        **kwargs: Additional arguments to pass to ConvUNext.

    Returns:
        A tuple containing the model and two None values.
    """
    return ConvUNext(
        input_dims=input_dims,
        depth=6,
        width=3,
        filters=96,
        name="convunext_xlarge",
        **kwargs
    )

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def preprocess_input(x: Union[np.ndarray, tf.Tensor], data_format: Optional[str] = None) -> Union[np.ndarray, tf.Tensor]:
    """Preprocesses input for ConvUNext models.

    Args:
        x: Input tensor or array.
        data_format: Data format ('channels_first' or 'channels_last').

    Returns:
        Preprocessed input (normalized to [0, 1]).
    """
    # ConvUNext expects inputs in [0, 1]
    return x / 255.0 if x.dtype != tf.float32 and x.dtype != np.float32 else x

def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """Returns metadata for predefined ConvUNext model variants.

    Args:
        model_name: Name of the model variant.

    Returns:
        Dictionary of model metadata.
    """
    metadata = {
        "convunext_small": {
            "depth": 4,
            "width": 2,
            "filters": 32,
            "params": "6.7M"
        },
        "convunext_medium": {
            "depth": 5,
            "width": 2,
            "filters": 48,
            "params": "16.2M"
        },
        "convunext_large": {
            "depth": 5,
            "width": 3,
            "filters": 64,
            "params": "30.8M"
        },
        "convunext_xlarge": {
            "depth": 6,
            "width": 3,
            "filters": 96,
            "params": "62.4M"
        }
    }

    return metadata.get(model_name, {"unknown": True})

# ---------------------------------------------------------------------