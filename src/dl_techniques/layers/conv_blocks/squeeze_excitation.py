"""
Squeeze-and-Excitation block, built by the ``SqueezeExcitation`` class.

A convolutional block treats every channel equally regardless of how useful
its content is for the task. This layer instead computes a per-channel gate:
it squeezes each channel's spatial extent down to one number with global
average pooling, passes that channel vector through a small bottleneck MLP
(1x1 convolutions) to produce a sigmoid weight per channel, then rescales the
original input by those weights. The bottleneck width is controlled by
``reduction_ratio``. The layer accepts 2D, 3D or 4D input, internally
expanding to 4D for the convolutional gate and squeezing back afterward.

References:
    - Hu et al., 2018. Squeeze-and-Excitation Networks.
"""

import keras
from typing import Dict, Optional, Tuple, Union, Callable, Any

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.conv_blocks.squeeze_excitation")
class SqueezeExcitation(keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.

    This layer implements the Squeeze-and-Excitation mechanism that adaptively
    recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels. Given input ``X``, the SE block
    computes: ``z = GAP(X)``, ``s = sigmoid(W2 * act(W1 * z))``,
    ``output = X * s``, where ``W1`` reduces channels by ``reduction_ratio``
    and ``W2`` restores the original channel count. The layer supports 2D, 3D,
    and 4D inputs by internally expanding to 4D for the convolutional
    infrastructure.

    Architecture:

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input [B, ..., C]                 │
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Squeeze: GlobalAvgPool → [B,1,1,C]│
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Excitation:                       │
        │    Conv1x1(C→C*r) → Activation     │
        │    Conv1x1(C*r→C) → Sigmoid        │
        │    → attention weights [B,1,1,C]   │
        └──────────────┬─────────────────────┘
                       ▼
        ┌────────────────────────────────────┐
        │  Scale: Input * attention_weights  │
        │  → Output [B, ..., C]              │
        └────────────────────────────────────┘

    :param reduction_ratio: Float in ``(0, 1]`` determining the bottleneck width.
        Defaults to 0.25.
    :type reduction_ratio: float
    :param activation: Activation function for the reduction layer. String identifier
        or callable. Final activation is always sigmoid. Defaults to ``'relu'``.
    :type activation: str or callable
    :param use_bias: Whether convolution layers use bias vectors. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernel weights.
        Defaults to ``'glorot_normal'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: str or keras.regularizers.Regularizer or None
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: str or keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
        self,
        reduction_ratio: float = 0.25,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not 0 < reduction_ratio <= 1.0:
            raise ValueError(
                f"reduction_ratio must be in range (0, 1], got {reduction_ratio}"
            )

        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.reduction_activation = keras.activations.get(activation)

        # Set in build(), since they depend on input_shape.
        self.input_channels: Optional[int] = None
        self.bottleneck_channels: Optional[int] = None
        self.global_pool: Optional[keras.layers.GlobalAveragePooling2D] = None
        self.conv_reduce: Optional[keras.layers.Conv2D] = None
        self.conv_restore: Optional[keras.layers.Conv2D] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create all sub-layers.

        :param input_shape: Shape of the input tensor in format
            ``(batch, [spatial_dims...], channels)``.
        :type input_shape: tuple
        """
        if len(input_shape) not in (2, 3, 4):
            raise ValueError(
                f"Expected 2D (B, C), 3D (B, S, C), or 4D (B, H, W, C) input shape, "
                f"got {len(input_shape)}D: {input_shape}"
            )

        self.input_channels = input_shape[-1]
        if self.input_channels is None:
            raise ValueError("Last dimension (channels) of input must be defined")

        self.bottleneck_channels = max(1, int(round(
            self.input_channels * self.reduction_ratio
        )))

        logger.info(
            f"Building SqueezeExcitation: input_channels={self.input_channels}, "
            f"bottleneck_channels={self.bottleneck_channels}"
        )

        self.global_pool = keras.layers.GlobalAveragePooling2D(
            keepdims=True,
            name='global_pool'
        )

        self.conv_reduce = keras.layers.Conv2D(
            filters=self.bottleneck_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name='conv_reduce'
        )

        self.conv_restore = keras.layers.Conv2D(
            filters=self.input_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            name='conv_restore'
        )

        if len(input_shape) == 2:
            internal_shape = (input_shape[0], 1, 1, self.input_channels)
        elif len(input_shape) == 3:
            internal_shape = (input_shape[0], input_shape[1], 1, self.input_channels)
        else:
            internal_shape = input_shape

        self.global_pool.build(internal_shape)

        # GAP output is always (B, 1, 1, C).
        pooled_shape = (input_shape[0], 1, 1, self.input_channels)
        self.conv_reduce.build(pooled_shape)

        reduced_shape = (pooled_shape[0], 1, 1, self.bottleneck_channels)
        self.conv_restore.build(reduced_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SE block.

        :param inputs: Input tensor of shape ``(B, C)``, ``(B, S, C)``, or ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode.
        :type training: bool or None
        :return: Output tensor with same shape as input after SE recalibration.
        :rtype: keras.KerasTensor
        """
        if (self.global_pool is None or
            self.conv_reduce is None or
            self.conv_restore is None):
            raise RuntimeError(
                "Layer must be built before calling. "
                "This usually happens automatically on first call."
            )

        x = inputs
        input_rank = len(inputs.shape)

        if input_rank == 2:
            x = keras.ops.expand_dims(x, axis=1)
            x = keras.ops.expand_dims(x, axis=1)
        elif input_rank == 3:
            x = keras.ops.expand_dims(x, axis=2)

        squeezed = self.global_pool(x)

        excited = self.conv_reduce(squeezed, training=training)
        excited = self.reduction_activation(excited)
        excited = self.conv_restore(excited, training=training)
        attention_weights = keras.activations.sigmoid(excited)

        output = keras.ops.multiply(x, attention_weights)

        if input_rank == 2:
            output = keras.ops.squeeze(output, axis=[1, 2])
        elif input_rank == 3:
            output = keras.ops.squeeze(output, axis=2)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'activation': keras.activations.serialize(self.reduction_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config
