"""
unet with convnext blocks
convnext blocks: https://github.com/keras-team/keras/blob/v3.6.0/keras/src/applications/convnext.py#L550
unet with convnext : SKOOTS: Skeleton oriented object segmentation for mitochondria, 2023
"""

import copy
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils import logger
from dl_techniques.layers import (
    upsample,
    ConvType,
    SpatialLayer,
    ConvNextBlock,
    conv2d_wrapper,
    StochasticDepth,
    AdditiveAttentionGate,
    ConvolutionalTransformerBlock
)
from dl_techniques.regularizers.soft_orthogonal import (
    DEFAULT_SOFTORTHOGONAL_STDDEV,
    SoftOrthonormalConstraintRegularizer
)

# ---------------------------------------------------------------------


def builder(
        input_dims,
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
        kernel_regularizer="l2",
        kernel_initializer="glorot_normal",
        dropout_rate: float = -1,
        depth_drop_rate: float = 0.0,
        spatial_dropout_rate: float = -1,
        multiple_scale_outputs: bool = True,
        convolutional_self_attention_dropout_rate: float = 0.0,
        convolutional_self_attention_strides: int = 1,
        convolutional_self_attention_blocks: int = -1,
        name="convunext",
        **kwargs) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    builds a modified unet model that uses convnext blocks and laplacian downsampling

    :param input_dims: Models input dimensions
    :param depth: number of levels to go down
    :param width: number of horizontals nodes, if -1 it gets set to depth
    :param encoder_kernel_size: kernel size of encoder convolutional layer
    :param decoder_kernel_size: kernel size of decoder convolutional layer
    :param filters_level_multiplier: every down level increase the number of filters by a factor of
    :param filters: filters of base convolutional layer
    :param max_filters: max number of filters
    :param activation: activation of the first 1x1 kernel
    :param use_bn: use batch normalization
    :param use_ln: use layer normalization
    :param use_gamma: if True (True by default) use gamma learning in convnext
    :param use_bias: use bias (bias free means this should be off)
    :param use_attention_gates: if True add attention gates between depths
    :param use_half_resolution: if True subsample at the top and upsample at the bottom
    :param use_self_attention: if True add a convolutional self-attention element at the bottom layer
    :param use_concat: if True concatenate otherwise add skip layers (True by default)
    :param use_soft_orthogonal_regularization: if True use soft orthogonal regularization on the 1x1 kernels
    :param use_soft_orthonormal_regularization: if true use soft orthonormal regularization on the 1x1 middle kernels
    :param kernel_regularizer: Kernel weight regularizer
    :param kernel_initializer: Kernel weight initializer
    :param dropout_rate: probability of dropout, negative to turn off
    :param spatial_dropout_rate: probability of spatial dropout, negative to turn off
    :param depth_drop_rate: probability of residual block dropout, negative or zero to turn off
    :param multiple_scale_outputs: if True for each scale give an output
    :param convolutional_self_attention_dropout_rate:
    :param convolutional_self_attention_dropout_rate:
    :param convolutional_self_attention_strides:
    :param name: name of the model

    :return: unet with convnext blocks model
    """
    # --- argument checking
    logger.info("building convunext backbone")
    if len(kwargs) > 0:
        logger.info(f"parameters not used: {kwargs}")
    if filters <= 0:
        raise ValueError("filters must be > 0")
    if width is None or width <= 0:
        width = 1
    if depth <= 0 or width <= 0:
        raise ValueError("depth and width must be > 0")
    if convolutional_self_attention_blocks <= 0 and use_self_attention:
        convolutional_self_attention_blocks = width

    if convolutional_self_attention_dropout_rate < 0 or convolutional_self_attention_dropout_rate > 1:
        raise ValueError("convolutional_self_attention_dropout_rate must be >= 0 and <= 1")

    downsample_activation = "linear"
    upsample_activation = "linear"
    upsample_type = upsample_type.strip().lower()
    kernel_initializer = kernel_initializer.strip().lower()

    if (use_soft_orthonormal_regularization and
            use_soft_orthogonal_regularization):
        raise ValueError(
            "only one use_soft_orthonormal_regularization or "
            "use_soft_orthogonal_regularization must be turned on")

    # --- setup parameters
    ln_params = \
        dict(
            scale=True,
            center=use_bias,
            epsilon=DEFAULT_LN_EPSILON
        )

    dropout_params = None
    if dropout_rate > 0.0:
        dropout_params = {"rate": dropout_rate}

    dropout_2d_params = None
    if spatial_dropout_rate > 0.0:
        dropout_2d_params = {"rate": spatial_dropout_rate}

    depth_drop_rates = (
        list(np.linspace(
            start=max(0.0, depth_drop_rate/width),
            stop=max(0.0, depth_drop_rate),
            num=width)
        )
    )

    base_conv_params = dict(
        kernel_size=encoder_kernel_size,
        filters=filters,
        strides=(1, 1),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )

    conv_params = []
    conv_params_up = []
    conv_params_down = []
    conv_params_res_1 = []
    conv_params_res_2 = []
    conv_params_res_3 = []

    # with half resolution, double the filters after the stem
    base_filters = filters
    if use_half_resolution:
        base_filters *= 2

    for d in range(depth):
        filters_level = \
            int(round(base_filters * max(1, filters_level_multiplier ** d)))
        if max_filters > 0:
            filters_level = min(max_filters, filters_level)
        filters_level_next = \
            int(round(base_filters * max(1, filters_level_multiplier ** (d + 1))))
        if max_filters > 0:
            filters_level_next = min(max_filters, filters_level_next)

        # default conv
        params = copy.deepcopy(base_conv_params)
        params["filters"] = filters_level
        params["kernel_size"] = 3
        params["activation"] = "linear"
        params["use_bias"] = use_bias
        conv_params.append(params)

        # 1st residual conv
        conv_params_res_1.append(dict(
            kernel_size=encoder_kernel_size,
            depth_multiplier=1,
            strides=(1, 1),
            padding="same",
            use_bias=use_bias,
            activation="linear",
            depthwise_regularizer=kernel_regularizer,
            depthwise_initializer=kernel_initializer
        ))

        # 2nd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["use_bias"] = use_bias
        params["activation"] = activation
        params["filters"] = filters_level * 4
        params["kernel_initializer"] = kernel_initializer
        params["kernel_regularizer"] = kernel_regularizer
        conv_params_res_2.append(params)

        # 3rd residual conv
        params = copy.deepcopy(base_conv_params)
        params["kernel_size"] = 1
        params["use_bias"] = use_bias
        params["activation"] = "linear"
        params["filters"] = filters_level
        params["kernel_initializer"] = kernel_initializer
        params["kernel_regularizer"] = kernel_regularizer
        conv_params_res_3.append(params)

        # conv2d params when moving down the scale
        params = copy.deepcopy(base_conv_params)
        params["use_bias"] = use_bias
        params["filters"] = filters_level_next
        params["activation"] = downsample_activation
        conv_params_down.append(params)

        # conv2d params when moving up the scale
        params = copy.deepcopy(base_conv_params)
        params["use_bias"] = use_bias
        params["filters"] = filters_level
        params["activation"] = upsample_activation
        conv_params_up.append(params)

    # --- book keeping
    nodes_dependencies = {}
    for d in range(0, depth, 1):
        if d == (depth - 1):
            # add only left dependency
            nodes_dependencies[(d, 1)] = [(d, 0)]
        else:
            # add left and bottom dependency
            nodes_dependencies[(d, 1)] = [(d, 0), (d + 1, 1)]

    nodes_output = {}
    nodes_to_visit = list(nodes_dependencies.keys())
    nodes_visited = set([(depth - 1, 0), (depth - 1, 1)])

    # --- build model
    # set input
    input_layer = \
        tf.keras.Input(
            name=INPUT_TENSOR_STR,
            shape=input_dims)
    # Where mask is 1.0 the input is masked
    # Where mask is 0.0 the input is not-masked
    mask_layer = \
        tf.keras.Input(
            name=MASK_TENSOR_STR,
            shape=input_dims[:-1] + [1,])
    coords_layer = SpatialLayer()(mask_layer)

    x = input_layer

    if use_half_resolution:
        masks = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(mask_layer)
        ]
        coords = [
            tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2, 2), padding="same")(coords_layer)
        ]
    else:
        masks = [mask_layer]
        coords = [coords_layer]

    for d in range(depth-1):
        m = masks[-1]
        m = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2), padding="same")(m)
        masks.append(m)
        c = coords[-1]
        c = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(c)
        coords.append(c)

    # --- apply mask
    x = x * (1.0 - mask_layer)

    # --- stem
    x = \
        conv2d_wrapper(
            input_layer=x,
            ln_params=None,
            bn_params=dict(
                scale=True,
                center=use_bias,
                momentum=DEFAULT_BN_MOMENTUM,
                epsilon=DEFAULT_BN_EPSILON
            ),
            conv_params=dict(
                kernel_size=(5, 5),
                filters=filters,
                strides=(1, 1),
                padding="same",
                use_bias=use_bias,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            ))

    # --- downsample
    if use_half_resolution:
        x = \
            conv2d_wrapper(
                input_layer=x,
                ln_params=None,
                bn_params=dict(
                    scale=True,
                    center=use_bias,
                    momentum=DEFAULT_BN_MOMENTUM,
                    epsilon=DEFAULT_BN_EPSILON
                ),
                conv_params=dict(
                    kernel_size=(2, 2),
                    filters=base_filters,
                    strides=(2, 2),
                    padding="same",
                    use_bias=use_bias,
                    activation=downsample_activation,
                    kernel_regularizer=kernel_regularizer,
                    kernel_initializer=kernel_initializer
                ))

    # --- build backbone
    for d in range(depth):
        if (use_self_attention and
                d == depth - 1 and
                convolutional_self_attention_strides > 1):
            subsampling_d = conv_params_res_1[d]
            subsampling_d["kernel_size"] = convolutional_self_attention_strides
            subsampling_d["strides"] = (convolutional_self_attention_strides, convolutional_self_attention_strides)
            x = \
                ConvNextBlock(
                    name=f"subsampling_{d}",
                    conv_params_1=conv_params_res_1[d],
                    conv_params_2=conv_params_res_2[d],
                    conv_params_3=conv_params_res_3[d],
                    ln_params=ln_params,
                    bn_params=None,
                    dropout_params=dropout_params,
                    use_gamma=use_gamma,
                    use_global_gamma=use_global_gamma,
                    dropout_2d_params=dropout_2d_params,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)

        tmp_width = width
        if use_self_attention and d == depth-1:
            tmp_width = convolutional_self_attention_blocks

        for w in range(tmp_width):
            # get skip for residual
            x_skip = x

            if use_self_attention and d == depth-1:
                x = (
                    ConvolutionalTransformerBlock(
                        dim=conv_params_res_3[d]["filters"],
                        num_heads=8,
                        mlp_ratio=4,
                        use_bias=use_bias,
                        dropout_rate=0.25,
                        attention_dropout=convolutional_self_attention_dropout_rate,
                        activation=activation,
                        use_gamma=use_gamma,
                        use_global_gamma=use_global_gamma,
                        use_soft_orthonormal_regularization=use_soft_orthonormal_regularization,
                        name=f"encoder_{d}_{w}")(x)
                )
            else:
                x = \
                    ConvNextBlock(
                        name=f"encoder_{d}_{w}",
                        conv_params_1=conv_params_res_1[d],
                        conv_params_2=conv_params_res_2[d],
                        conv_params_3=conv_params_res_3[d],
                        ln_params=ln_params,
                        bn_params=None,
                        dropout_params=dropout_params,
                        use_gamma=use_gamma,
                        use_global_gamma=use_global_gamma,
                        dropout_2d_params=dropout_2d_params,
                        use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                        use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)

            if x_skip.shape[-1] == x.shape[-1] and not (use_self_attention and d == depth - 1):
                if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
                    x = StochasticDepth(depth_drop_rates[w])(x)
                # transformers do not need another skip connection
                x = tf.keras.layers.Add()([x_skip, x])

        if (use_self_attention and
                d == depth - 1 and
                convolutional_self_attention_strides > 1):
            x = \
                conv2d_wrapper(
                    input_layer=x,
                    ln_params=None,
                    bn_params=None,
                    conv_type=ConvType.CONV2D_TRANSPOSE,
                    conv_params=dict(
                        kernel_size=convolutional_self_attention_strides,
                        filters=conv_params_res_3[d]["filters"],
                        strides=convolutional_self_attention_strides,
                        padding="same",
                        use_bias=use_bias,
                        activation=upsample_activation,
                        kernel_regularizer=kernel_regularizer,
                        kernel_initializer=kernel_initializer
                    ))

        node_level = (d, 0)
        nodes_visited.add(node_level)
        nodes_output[node_level] = x

        if d != (depth - 1):
            x = tf.keras.layers.Concatenate(axis=-1)([x, coords[d], masks[d]])

            down_1 = copy.deepcopy(conv_params_res_1[d])
            down_1["strides"] = (2, 2)
            down_1["kernel_size"] = (4, 4)
            down_2 = copy.deepcopy(conv_params_res_2[d])
            down_3 = copy.deepcopy(conv_params_res_3[d])
            down_3["filters"] = conv_params_down[d]["filters"]

            x = \
                ConvNextBlock(
                    name=f"downsample_{d}",
                    conv_params_1=down_1,
                    conv_params_2=down_2,
                    conv_params_3=down_3,
                    ln_params=ln_params,
                    bn_params=None,
                    dropout_params=dropout_params,
                    use_gamma=use_gamma,
                    use_global_gamma=use_global_gamma,
                    dropout_2d_params=dropout_2d_params,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)
            del down_1, down_2, down_3
    del x, x_skip

    # --- VERY IMPORTANT
    # add this, so it works correctly
    nodes_output[(depth - 1, 1)] = nodes_output[(depth - 1, 0)]

    # --- build the encoder side based on dependencies
    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop(0)
        logger.info(f"node: [{node}, "
                    f"nodes_visited: {nodes_visited}, "
                    f"nodes_to_visit: {nodes_to_visit}, "
                    f"dependencies: {nodes_dependencies[node]}")
        # make sure a node is not visited twice
        if node in nodes_visited:
            logger.info(f"node: [{node}] already processed")
            continue
        # make sure that all the dependencies for a node are matched
        dependencies = nodes_dependencies[node]
        dependencies_matched = \
            all([
                (d in nodes_output) and (d in nodes_visited or d == node)
                for d in dependencies
            ])
        if not dependencies_matched:
            logger.info(f"node: [{node}] dependencies not matched, continuing")
            nodes_to_visit.append(node)
            continue

        # sort it so all same level dependencies are first and added
        # as residual before finally concatenating the previous scale
        dependencies = \
            sorted(list(dependencies),
                   key=lambda d: d[0],
                   reverse=False)
        logger.info(f"processing node: {node}, "
                    f"dependencies: {dependencies}, "
                    f"nodes_output: {list(nodes_output.keys())}")

        x_input = []

        logger.debug(f"node: [{node}], dependencies: {dependencies}")
        for dependency in dependencies:
            logger.debug(f"processing dependency: {dependency}")
            x = nodes_output[dependency]

            if dependency[0] == node[0]:
                pass
            elif dependency[0] > node[0]:
                logger.info("upsampling here")
                x = \
                    upsample(
                        input_layer=x,
                        upsample_type=upsample_type,
                        ln_params=ln_params,
                        bn_params=None,
                        conv_params=conv_params_up[node[0]])
            else:
                raise ValueError(f"node: {node}, dependencies: {dependencies}, "
                                 f"should not supposed to be here")

            x_input.append(x)

        # add attention gates,
        # first input is assumed to be the higher depth
        # and the second input is assumed to be the lower depth
        if use_attention_gates and len(x_input) == 2:
            logger.debug(f"adding AttentionGate at depth: [{node[0]}]")

            x_input[0] = (
                AdditiveAttentionGate(
                    use_bias=use_bias,
                    use_bn=None,
                    use_ln=use_ln,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization,
                    attention_channels=conv_params_res_3[node[0]]["filters"],
                    kernel_initializer=kernel_initializer
                )(x_input))

        d = node[0]
        if len(x_input) == 1:
            x = x_input[0]
        elif len(x_input) > 1:
            if use_concat:
                x = tf.keras.layers.Concatenate(axis=-1)(x_input)
                x = tf.keras.layers.Concatenate(axis=-1)([x, coords[d], masks[d]])
                # project the concatenated result using a convolution
                # https://www.researchgate.net/figure/UNet-Architecture-with-ConvNext-computational-blocks-offers-superior-accuracy-per_fig2_370621145
                params = copy.deepcopy(conv_params_res_3[node[0]])
                params["kernel_size"] = (1, 1)
                params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()
                params["kernel_initializer"] = tf.keras.initializers.truncated_normal(
                    stddev=DEFAULT_SOFTORTHOGONAL_STDDEV, seed=1)
                x = conv2d_wrapper(
                    input_layer=x,
                    bn_params=None,
                    ln_params=ln_params,
                    conv_params=params)
            else:
                x = tf.keras.layers.Add()(x_input)
        else:
            raise ValueError("this must never happen")

        # --- convnext block
        for w in range(width):
            x_skip = x

            params = copy.deepcopy(conv_params_res_1[d])
            params["kernel_size"] = decoder_kernel_size
            x = \
                ConvNextBlock(
                    name=f"decoder_{node[0]}_{w}",
                    conv_params_1=params,
                    conv_params_2=conv_params_res_2[node[0]],
                    conv_params_3=conv_params_res_3[node[0]],
                    ln_params=ln_params,
                    bn_params=None,
                    use_gamma=use_gamma,
                    use_global_gamma=use_global_gamma,
                    dropout_params=dropout_params,
                    dropout_2d_params=dropout_2d_params,
                    use_soft_orthogonal_regularization=use_soft_orthogonal_regularization,
                    use_soft_orthonormal_regularization=use_soft_orthonormal_regularization)(x)

            if x_skip.shape[-1] == x.shape[-1]:
                if len(depth_drop_rates) <= width and depth_drop_rates[w] > 0.0:
                    x = StochasticDepth(depth_drop_rates[w])(x)
                x = tf.keras.layers.Add()([x_skip, x])

        nodes_output[node] = x
        nodes_visited.add(node)

    # --- output layer here
    output_layers = []

    # depth outputs
    if multiple_scale_outputs:
        tmp_output_layers = []
        for d in range(1, depth, 1):
            d = d
            w = 1

            if d < 0 or w < 0:
                logger.error(f"there is no node[{d},{w}] please check your assumptions")
                continue
            x = nodes_output[(d, w)]
            tmp_output_layers.append(x)
        # reverse here so deeper levels come on top
        output_layers += tmp_output_layers[::-1]

    # add as last the best output
    output_layers.append(nodes_output[(0, 1)])

    # !!! IMPORTANT !!!
    # reverse it so the deepest output is first
    # otherwise we will get the most shallow output
    output_layers = output_layers[::-1]

    # upsample and/or normalize output
    if use_half_resolution:
        for i, layer in enumerate(output_layers):
            layer = (
                tf.keras.layers.UpSampling2D(
                    size=(2, 2),
                    interpolation="nearest")(layer)
            )
            kernel_size = (3, 3)

            # bring back to filters output
            layer = (
                conv2d_wrapper(
                    input_layer=layer,
                    bn_params=None,
                    ln_params=ln_params if use_output_normalization else None,
                    conv_params=dict(
                        kernel_size=kernel_size,
                        filters=filters,
                        strides=(1, 1),
                        padding="same",
                        use_bias=use_bias,
                        activation=upsample_activation,
                        kernel_regularizer=kernel_regularizer,
                        kernel_initializer=kernel_initializer
                    ))
            )
            output_layers[i] = layer
    elif use_output_normalization:
        for i in range(len(output_layers)):
            x = output_layers[i]
            output_layers[i] = tf.keras.layers.LayerNormalization(**ln_params, name=f"norm_output_{i}")(x)

    for i in range(len(output_layers)):
        x = output_layers[i]
        output_layers[i] = (
            tf.keras.layers.Layer(
                name=f"decoder_output_{i}")(x))

    end_to_end_model = (
        tf.keras.Model(
            name=name,
            trainable=True,
            inputs=[input_layer, mask_layer],
            outputs=output_layers
        )
    )

    return (
        end_to_end_model,
        None,
        None
    )

# ---------------------------------------------------------------------
