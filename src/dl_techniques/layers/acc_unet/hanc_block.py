"""HANCBlock, a hierarchical-context building block from ACC-UNet.

HANCBlock is a Keras layer that maps a feature map to ``filters`` channels.
It wraps hierarchical neighborhood aggregation in an inverted bottleneck: a
1x1 expansion, a depthwise 3x3 convolution, a HANCLayer, a 1x1 projection,
and Squeeze-and-Excitation recalibration. HANCLayer stands in for
self-attention, which scores every pixel pair and costs O(N^2) in the pixel
count N. Instead it average- and max-pools the feature map at k scales,
concatenates the results onto the full-resolution map along the channel
axis, and mixes them with a learned 1x1 convolution, so cost grows with k
rather than with N squared. The input channel count is a constructor
argument and is checked against the input shape at build time. The residual
connection is present only when ``input_channels`` equals ``filters``. Every
convolution here runs without a bias term, so ``bias_initializer`` and
``bias_regularizer`` are stored for serialization and have no effect.

References:
    - Yan et al., 2023. ACC-UNet: An adaptive context and contrast-aware UNet
      for seismic facies identification.
    - Sandler et al., 2018. MobileNetV2: Inverted Residuals and Linear
      Bottlenecks.
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.conv_blocks.squeeze_excitation import SqueezeExcitation

from .hanc_layer import HANCLayer

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.acc_unet.hanc_block")
class HANCBlock(keras.layers.Layer):
    """Aggregate neighborhood context at k scales inside an inverted bottleneck.

    HANC stands for hierarchical aggregation of neighborhood context. The
    block expands the input channels, processes them depthwise, aggregates
    context at ``k`` pooling scales, projects to ``filters`` channels, and
    rescales channels with Squeeze-and-Excitation. One LeakyReLU instance is
    reused after each of the three normalized convolutions.

    Architecture:

    .. code-block:: text

        input [B, H, W, C_in]
              │
              ├─────────────────────────┐
              ▼                         │
        ┌───────────────────────┐       │
        │ expand conv 1x1       │       │
        │ bn, leaky relu        │       │
        └───────────────────────┘       │
              │ [B, H, W, C_in*inv]     │
              ▼                         │
        ┌───────────────────────┐       │
        │ depthwise conv 3x3    │       │
        │ bn, leaky relu        │       │
        └───────────────────────┘       │
              │ [B, H, W, C_in*inv]     │
              ▼                         │
        ┌───────────────────────┐       │
        │ hanc, k scales        │       │
        └───────────────────────┘       │
              │ [B, H, W, C_in]         │
              ▼                         │
             (+)◄───────────────────────┘  (optional)
              │
              ▼
        ┌───────────────────────┐  (optional)
        │ batch norm            │
        └───────────────────────┘
              │ [B, H, W, C_in]
              ▼
        ┌───────────────────────┐
        │ project conv 1x1      │
        │ bn, leaky relu        │
        └───────────────────────┘
              │ [B, H, W, filters]
              ▼
        ┌───────────────────────┐
        │ squeeze-excitation    │
        └───────────────────────┘
              │
              ▼
        output [B, H, W, filters]

    The skip and its batch norm run only when ``input_channels == filters``.

    Input shape:
        4D tensor ``(batch, height, width, input_channels)``.

    Output shape:
        4D tensor ``(batch, height, width, filters)``.

    :param filters: Number of output filters. Must be positive.
    :type filters: int
    :param input_channels: Number of input channels. Must be positive and must
        match the last dimension of the input shape.
    :type input_channels: int
    :param k: Hierarchical levels for HANC operation (1-5 supported).
        Determines the granularity of context aggregation.
    :type k: int
    :param inv_factor: Inverted bottleneck expansion factor.
        Determines the channel width of the internal processing.
    :type inv_factor: int
    :param kernel_initializer: Initializer for convolution kernels.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors. Kept in the config
        for serialization; the convolutions here use no bias.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for convolution kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias vectors. Kept in the
        config for serialization; the convolutions here use no bias.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    :ivar expanded_channels: Channel width inside the bottleneck,
        ``input_channels * inv_factor``.
    :vartype expanded_channels: int
    :ivar use_residual: Whether the skip connection and its batch norm are
        active, true when ``input_channels == filters``.
    :vartype use_residual: bool

    :raises ValueError: If filters, input_channels, or inv_factor are not positive.
    :raises ValueError: If k is not between 1 and 5.
    """

    def __init__(
        self,
        filters: int,
        input_channels: int,
        k: int = 3,
        inv_factor: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if k < 1 or k > 5:
            raise ValueError(f"k must be between 1 and 5, got {k}")
        if inv_factor <= 0:
            raise ValueError(f"inv_factor must be positive, got {inv_factor}")

        self.filters = filters
        self.input_channels = input_channels
        self.k = k
        self.inv_factor = inv_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.expanded_channels = self.input_channels * self.inv_factor
        self.use_residual = (self.input_channels == self.filters)

        self.expand_conv = keras.layers.Conv2D(
            filters=self.expanded_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='expand_conv'
        )
        self.expand_bn = keras.layers.BatchNormalization(name='expand_bn')

        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='depthwise_conv'
        )
        self.depthwise_bn = keras.layers.BatchNormalization(name='depthwise_bn')

        # HANC returns to input_channels so the skip tensor lines up unchanged.
        self.hanc_layer = HANCLayer(
            in_channels=self.expanded_channels,
            out_channels=self.input_channels,
            k=self.k,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc'
        )

        if self.use_residual:
            self.residual_bn = keras.layers.BatchNormalization(name='residual_bn')
        else:
            self.residual_bn = None

        self.output_conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_conv'
        )
        self.output_bn = keras.layers.BatchNormalization(name='output_bn')

        self.squeeze_excitation = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='se'
        )

        # Stateless, so one instance serves all three activation sites.
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers in computational order.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input shape is not 4D, or if its last
            dimension differs from ``input_channels``.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] != self.input_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.input_channels}, "
                f"got {input_shape[-1]}"
            )

        self.expand_conv.build(input_shape)
        expand_output_shape = self.expand_conv.compute_output_shape(input_shape)
        self.expand_bn.build(expand_output_shape)

        self.depthwise_conv.build(expand_output_shape)
        depthwise_output_shape = self.depthwise_conv.compute_output_shape(expand_output_shape)
        self.depthwise_bn.build(depthwise_output_shape)

        self.hanc_layer.build(depthwise_output_shape)
        hanc_output_shape = self.hanc_layer.compute_output_shape(depthwise_output_shape)

        if self.residual_bn is not None:
            self.residual_bn.build(hanc_output_shape)

        # HANCLayer restores input_channels, so its output shape feeds output_conv directly.
        output_input_shape = hanc_output_shape
        self.output_conv.build(output_input_shape)
        output_conv_shape = self.output_conv.compute_output_shape(output_input_shape)
        self.output_bn.build(output_conv_shape)

        self.squeeze_excitation.build(output_conv_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Input tensor of shape ``(batch, height, width, input_channels)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for batch normalization.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, height, width, filters)``.
        :rtype: keras.KerasTensor
        """
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = self.activation(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.activation(x)

        x = self.hanc_layer(x, training=training)

        # The skip carries the block input, so the add sits before projection.
        if self.use_residual and self.residual_bn is not None:
            x = x + inputs
            x = self.residual_bn(x, training=training)

        x = self.output_conv(x)
        x = self.output_bn(x, training=training)
        x = self.activation(x)

        x = self.squeeze_excitation(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Input shape with the channel dimension replaced by ``filters``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If the input shape is not 4D.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.filters])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'input_channels': self.input_channels,
            'k': self.k,
            'inv_factor': self.inv_factor,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------