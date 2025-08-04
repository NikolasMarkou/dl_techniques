"""
This module implements the Non-local Attention layer, a self-attention mechanism
for computer vision tasks, based on the influential paper "Non-local Neural Networks".

Standard convolutional layers operate on a small, local neighborhood of pixels,
limiting their receptive field. In contrast, this Non-local Attention layer captures
long-range dependencies by computing the response at a position as a weighted sum of
features at *all* positions in the input feature map. This allows the model to
build relationships between distant pixels, which is crucial for understanding
complex scenes and objects.

It functions as a self-attention block tailored for 4D image-like tensors
(batch, height, width, channels).

Architectural Breakdown:

1.  **Spatial Pre-processing (Optional):**
    -   An optional `DepthwiseConv2D` is first applied to the input. This step can
        capture local spatial context and enrich the features before the global
        attention mechanism is applied.
    -   This is followed by an optional normalization layer (`BatchNormalization` or
        `LayerNormalization`) to stabilize the activations.

2.  **Query, Key, and Value Projection:**
    -   The pre-processed feature map is then projected into three distinct
        representations using 1x1 convolutions: Query (Q), Key (K), and Value (V).
        This is a standard pattern in attention mechanisms.

3.  **Attention Computation:**
    -   The core of the layer. The 4D feature maps (Q, K, V) are flattened into
        sequences of "pixels", effectively treating the entire spatial grid as a
        sequence.
    -   The attention mechanism then calculates a similarity score between every
        Query pixel and every Key pixel. This score matrix determines how much
        "attention" each position should pay to every other position.
    -   These attention scores are used to compute a weighted sum of the Value pixels,
        producing an output where each pixel's representation is a mixture of
        information from all other pixels in the input.
    -   The layer supports two operational modes inspired by the original paper:
        'dot_product' (standard scaled dot-product attention) and 'gaussian'
        (which is approximated by adjusting channel sizes as described in the paper).

4.  **Output Transformation:**
    -   The attended features, now rich with global context, are reshaped back into
        their original 4D spatial format.
    -   A final 1x1 convolution transforms these features into the desired output
        channel dimension.
    -   Optional dropout is applied for regularization.

This layer is typically used as a block within a larger network (e.g., a ResNet),
often with a residual connection around it (`output = inputs + NonLocalAttention(inputs)`),
to augment standard convolutions with global reasoning capabilities.

Reference:
-   "Non-local Neural Networks" (https://arxiv.org/abs/1711.07971)
"""

import keras
from typing import Any, Dict, Tuple, Optional, Literal, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NonLocalAttention(keras.layers.Layer):
    """Non-local Self Attention Layer.

    Implementation of the self-attention mechanism from:
    "Non-local Neural Networks" (Wang et al., 2018)
    https://arxiv.org/abs/1711.07971

    The layer implements a space-time non-local block that captures long-range
    dependencies in feature representations through self-attention mechanisms.

    Args:
        attention_channels: Number of channels in attention mechanism.
        kernel_size: Size of depthwise convolution kernel. Defaults to (7, 7).
        use_bias: Whether to use bias in convolution layers. Defaults to False.
        normalization: Type of normalization to use ('batch', 'layer', or None).
            Defaults to 'batch'.
        intermediate_activation: Activation function for intermediate layers.
            Defaults to 'relu'.
        output_activation: Activation function for output. Defaults to 'linear'.
        output_channels: Number of output channels (-1 for same as input).
            Defaults to -1.
        dropout_rate: Dropout rate (0 to disable). Defaults to 0.0.
        attention_mode: Type of attention mechanism ('gaussian', 'dot_product').
            Defaults to 'gaussian'.
        kernel_initializer: Initializer for the kernel weights.
            Defaults to 'glorot_normal'.
        bias_initializer: Initializer for the bias vector. Defaults to 'zeros'.
        kernel_regularizer: Regularizer function applied to kernel weights.
            Defaults to None.
        bias_regularizer: Regularizer function applied to bias vector.
            Defaults to None.
        activity_regularizer: Regularizer function applied to output.
            Defaults to None.
        **kwargs: Additional layer arguments.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, output_channels)`
        where output_channels is specified or same as input if -1.

    Raises:
        ValueError: If attention_channels <= 0 or dropout_rate not in [0, 1).
    """

    def __init__(
            self,
            attention_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = (7, 7),
            use_bias: bool = False,
            normalization: Optional[Literal['batch', 'layer']] = 'batch',
            intermediate_activation: Union[str, callable] = 'relu',
            output_activation: Union[str, callable] = 'linear',
            output_channels: int = -1,
            dropout_rate: float = 0.0,
            attention_mode: Literal['gaussian', 'dot_product'] = 'gaussian',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_normal',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if attention_channels <= 0:
            raise ValueError("attention_channels must be positive")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("dropout_rate must be in [0, 1)")

        # Store configuration
        self.attention_channels = attention_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.use_bias = use_bias
        self.normalization = normalization
        self.intermediate_activation = intermediate_activation
        self.output_activation = output_activation
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate
        self.attention_mode = attention_mode

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Initialize sublayers (will be built in build())
        self.depthwise_conv = None
        self.normalization_layer = None
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.attention = None
        self.output_conv = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the layer's weights.

        Args:
            input_shape: Shape tuple of input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        channels = input_shape[-1]
        actual_output_channels = (
            channels if self.output_channels <= 0
            else self.output_channels
        )

        # Build depthwise convolution for spatial information
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            activation=self.intermediate_activation,
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer
        )

        # Build normalization if specified
        if self.normalization == 'batch':
            self.normalization_layer = keras.layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5
            )
        elif self.normalization == 'layer':
            self.normalization_layer = keras.layers.LayerNormalization(
                epsilon=1e-5
            )

        # Common convolution parameters
        conv_params = {
            'kernel_size': (1, 1),
            'strides': (1, 1),
            'padding': 'same',
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer
        }

        # Build Query, Key, Value projections
        self.query_conv = keras.layers.Conv2D(
            filters=self.attention_channels,
            **conv_params
        )

        # Adjust key/value channels based on attention mode
        key_value_channels = (
            self.attention_channels
            if self.attention_mode == 'dot_product'
            else self.attention_channels // 8
        )

        self.key_conv = keras.layers.Conv2D(
            filters=key_value_channels,
            activation=self.intermediate_activation,
            **conv_params
        )

        self.value_conv = keras.layers.Conv2D(
            filters=key_value_channels,
            **conv_params
        )

        # Build attention mechanism
        self.attention = keras.layers.Attention(
            use_scale=self.attention_mode == 'dot_product',
            score_mode='dot',
            dropout=self.dropout_rate if self.dropout_rate > 0 else None
        )

        # Build output projection
        self.output_conv = keras.layers.Conv2D(
            filters=actual_output_channels,
            activation=self.output_activation,
            **conv_params
        )

        # Build dropout if specified
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.
            **kwargs: Additional arguments.

        Returns:
            Output tensor after applying non-local attention.
        """
        # Apply depthwise convolution for spatial processing
        x = self.depthwise_conv(inputs, training=training)

        # Apply normalization if specified
        if self.normalization_layer is not None:
            x = self.normalization_layer(x, training=training)

        # Generate query, key, value projections
        query = self.query_conv(x, training=training)
        key = self.key_conv(x, training=training)
        value = self.value_conv(x, training=training)

        # Reshape for attention computation
        shape = keras.ops.shape(query)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]

        query_reshaped = keras.ops.reshape(query, [batch_size, -1, channels])
        key_reshaped = keras.ops.reshape(key, [batch_size, -1, keras.ops.shape(key)[-1]])
        value_reshaped = keras.ops.reshape(value, [batch_size, -1, keras.ops.shape(value)[-1]])

        # Apply attention mechanism
        attention_output = self.attention(
            [query_reshaped, value_reshaped, key_reshaped],
            training=training
        )

        # Reshape back to spatial dimensions
        attention_output = keras.ops.reshape(
            attention_output,
            [batch_size, height, width, keras.ops.shape(attention_output)[-1]]
        )

        # Apply output projection
        output = self.output_conv(attention_output, training=training)

        # Apply dropout if specified and training
        if self.dropout is not None and training:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Computes output shape from input shape.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple of output tensor.
        """
        output_shape = list(input_shape)
        if self.output_channels > 0:
            output_shape[-1] = self.output_channels
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Returns layer configuration.

        Returns:
            Dictionary containing configuration.
        """
        config = super().get_config()
        config.update({
            'attention_channels': self.attention_channels,
            'kernel_size': self.kernel_size,
            'use_bias': self.use_bias,
            'normalization': self.normalization,
            'intermediate_activation': self.intermediate_activation,
            'output_activation': self.output_activation,
            'output_channels': self.output_channels,
            'dropout_rate': self.dropout_rate,
            'attention_mode': self.attention_mode,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NonLocalAttention":
        """Creates layer from configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            Instantiated layer.
        """
        return cls(**config)

# ---------------------------------------------------------------------
