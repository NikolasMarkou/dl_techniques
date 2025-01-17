import keras
import dataclasses
import tensorflow as tf
from keras import layers
from typing import Any, Dict, Tuple, Optional, Literal

# ---------------------------------------------------------------------


@dataclasses.dataclass
class NonLocalAttentionConfig:
    """Configuration for Non-Local Attention Layer.

    Attributes:
        attention_channels: Number of channels in attention mechanism
        kernel_size: Size of depthwise convolution kernel
        use_bias: Whether to use bias in convolution layers
        normalization: Type of normalization to use ('batch', 'layer', or None)
        intermediate_activation: Activation function for intermediate layers
        output_activation: Activation function for output
        output_channels: Number of output channels (-1 for same as input)
        dropout_rate: Dropout rate (0 to disable)
        attention_mode: Type of attention mechanism ('gaussian', 'dot_product')
    """
    attention_channels: int
    kernel_size: Tuple[int, int] = (7, 7)
    use_bias: bool = False
    normalization: Optional[Literal['batch', 'layer']] = 'batch'
    intermediate_activation: str = 'relu'
    output_activation: str = 'linear'
    output_channels: int = -1
    dropout_rate: float = 0.0
    attention_mode: Literal['gaussian', 'dot_product'] = 'gaussian'

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NonLocalAttention(layers.Layer):
    """Non-local Self Attention Layer.

    Implementation of the self-attention mechanism from:
    "Non-local Neural Networks" (Wang et al., 2018)
    https://arxiv.org/abs/1711.07971

    The layer implements a space-time non-local block that captures long-range
    dependencies in feature representations through self-attention mechanisms.

    Args:
        config: NonLocalAttentionConfig containing layer parameters
        **kwargs: Additional layer arguments

    Input shape:
        4D tensor with shape: (batch_size, height, width, channels)

    Output shape:
        4D tensor with shape: (batch_size, height, width, output_channels)
        where output_channels is specified in config or same as input if -1
    """

    def __init__(
            self,
            config: NonLocalAttentionConfig,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self._validate_config()

        # Initialize layer components
        self.depthwise_conv = None
        self.normalization = None
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.attention = None
        self.output_conv = None
        self.dropout = None

        # Build normalization if specified
        if config.normalization:
            self.normalization = self._build_normalization()

        # Build dropout if specified
        if config.dropout_rate > 0:
            self.dropout = layers.Dropout(config.dropout_rate)

    def _validate_config(self) -> None:
        """Validates configuration parameters."""
        if self.config.attention_channels <= 0:
            raise ValueError("attention_channels must be positive")
        if self.config.dropout_rate < 0 or self.config.dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")

    def _build_normalization(self) -> layers.Layer:
        """Builds normalization layer based on config."""
        if self.config.normalization == 'batch':
            return layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5
            )
        elif self.config.normalization == 'layer':
            return layers.LayerNormalization(
                epsilon=1e-5
            )
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization}")

    def build(self, input_shape: tf.TensorShape) -> None:
        """Builds the layer's weights.

        Args:
            input_shape: TensorShape of input tensor
        """
        channels = input_shape[-1]
        output_channels = (
            channels if self.config.output_channels <= 0
            else self.config.output_channels
        )

        # Depthwise convolution for spatial information
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=self.config.kernel_size,
            padding='same',
            use_bias=self.config.use_bias,
            activation=self.config.intermediate_activation,
            depthwise_initializer='glorot_normal'
        )

        # Query, Key, Value projections
        conv_params = {
            'kernel_size': (1, 1),
            'strides': (1, 1),
            'padding': 'same',
            'use_bias': self.config.use_bias,
            'kernel_initializer': 'glorot_normal'
        }

        self.query_conv = layers.Conv2D(
            filters=self.config.attention_channels,
            **conv_params
        )

        key_value_channels = (
            self.config.attention_channels
            if self.config.attention_mode == 'dot_product'
            else self.config.attention_channels // 8
        )

        self.key_conv = layers.Conv2D(
            filters=key_value_channels,
            activation=self.config.intermediate_activation,
            **conv_params
        )

        self.value_conv = layers.Conv2D(
            filters=key_value_channels,
            **conv_params
        )

        # Attention mechanism
        self.attention = layers.Attention(
            use_scale=self.config.attention_mode == 'dot_product',
            score_mode='dot',
            dropout=self.config.dropout_rate if self.config.dropout_rate > 0 else None
        )

        # Output projection
        self.output_conv = layers.Conv2D(
            filters=output_channels,
            activation=self.config.output_activation,
            **conv_params
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor
            training: Whether in training mode
            **kwargs: Additional arguments

        Returns:
            Output tensor after applying non-local attention
        """
        x = self.depthwise_conv(inputs)

        if self.normalization is not None:
            x = self.normalization(x, training=training)

        # Generate query, key, value projections
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # Reshape for attention
        shape = tf.shape(query)
        query_reshaped = tf.reshape(query, [shape[0], -1, shape[-1]])
        key_reshaped = tf.reshape(key, [shape[0], -1, shape[-1]])
        value_reshaped = tf.reshape(value, [shape[0], -1, shape[-1]])

        # Apply attention
        attention_output = self.attention(
            [query_reshaped, value_reshaped, key_reshaped],
            training=training
        )

        # Reshape back to spatial dimensions
        attention_output = tf.reshape(attention_output, shape)

        # Output projection
        output = self.output_conv(attention_output)

        if self.dropout is not None and training:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(
            self,
            input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        """Computes output shape from input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        output_shape = list(input_shape)
        if self.config.output_channels > 0:
            output_shape[-1] = self.config.output_channels
        return tf.TensorShape(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns layer configuration.

        Returns:
            Dictionary containing configuration
        """
        config = super().get_config()
        config.update({
            'config': dataclasses.asdict(self.config)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NonLocalAttention":
        """Creates layer from configuration.

        Args:
            config: Layer configuration dictionary

        Returns:
            Instantiated layer
        """
        layer_config = NonLocalAttentionConfig(**config['config'])
        return cls(layer_config)

# ---------------------------------------------------------------------
