"""
A self-attention mechanism for computer vision tasks,
based on the influential paper "Non-local Neural Networks".

Standard convolutional layers operate on a small, local neighborhood of pixels,
limiting their receptive field. In contrast, this Non-local Attention layer captures
long-range dependencies by computing the response at a position as a weighted sum of
features at *all* positions in the input feature map. This allows the model to
build relationships between distant pixels, which is crucial for understanding
complex scenes and objects.

It functions as a self-attention block tailored for 4D image-like tensors
(batch, height, width, channels).

Reference:
-   "Non-local Neural Networks" (https://arxiv.org/abs/1711.07971)
"""

import keras
from keras import ops
from typing import Any, Dict, Tuple, Optional, Literal, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NonLocalAttention(keras.layers.Layer):
    """Non-local self-attention layer for capturing long-range spatial dependencies.

    Implements the self-attention mechanism from "Non-local Neural Networks"
    (Wang et al., 2018) that enables convolutional networks to capture global
    spatial dependencies by computing attention between all spatial positions in
    a 4D feature map. The input is first spatially pre-processed with an optional
    depthwise convolution and normalization, then projected into query, key, and
    value representations via 1x1 convolutions. The spatial dimensions are flattened
    into sequences for attention computation: ``A = Attention(Q, K, V)`` using either
    scaled dot-product (``score = Q K^T / sqrt(d_k)``) or Gaussian mode (with reduced
    key/value channels ``d_kv = d_attn / 8``). The attended output is reshaped back to
    spatial format and projected to the desired output channels.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────┐
        │   Input [B, H, W, C]        │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ DepthwiseConv2D(kernel_size) │
        │ + Normalization (optional)   │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  1x1 Conv2D Projections     │
        ├─────────┬─────────┬─────────┤
        │ Q_proj  │ K_proj  │ V_proj  │
        └────┬────┴────┬────┴────┬────┘
             ▼         ▼         ▼
        ┌────────────────────────────┐
        │ Reshape (H,W) → (H*W)     │
        │ Q [B, H*W, d_attn]        │
        │ K [B, H*W, d_kv]          │
        │ V [B, H*W, d_kv]          │
        └─────────────┬──────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Attention(Q, V, K)          │
        │ → [B, H*W, d_kv]           │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Reshape → [B, H, W, d_kv]  │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Output Conv2D 1x1           │
        │ + Dropout (optional)        │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │ Output [B, H, W, out_ch]    │
        └─────────────────────────────┘

    :param attention_channels: Number of channels in the attention mechanism.
        Must be positive.
    :type attention_channels: int
    :param kernel_size: Size of the depthwise convolution kernel.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param use_bias: Whether to use bias in convolution layers.
    :type use_bias: bool
    :param normalization: Type of normalization (``'batch'``, ``'layer'``, or ``None``).
    :type normalization: Optional[Literal['batch', 'layer']]
    :param intermediate_activation: Activation function for intermediate layers.
    :type intermediate_activation: Union[str, callable]
    :param output_activation: Activation function for the output projection.
    :type output_activation: Union[str, callable]
    :param output_channels: Number of output channels (``-1`` to match input).
    :type output_channels: int
    :param dropout_rate: Dropout rate between 0.0 and 1.0.
    :type dropout_rate: float
    :param attention_mode: Attention type (``'gaussian'`` or ``'dot_product'``).
    :type attention_mode: Literal['gaussian', 'dot_product']
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer for layer activity.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer parent class.
    :type kwargs: Any

    :raises ValueError: If attention_channels <= 0.
    :raises ValueError: If dropout_rate not in [0, 1).
    :raises ValueError: If normalization not in ['batch', 'layer', None].
    :raises ValueError: If attention_mode not in ['gaussian', 'dot_product'].
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
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(attention_channels, dropout_rate, normalization, attention_mode)

        # Store ALL configuration parameters
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

        # Common convolution parameters for reuse
        self._conv_params = {
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

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            activation=self.intermediate_activation,
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name='depthwise_conv'
        )

        # Create normalization layer if specified
        if self.normalization == 'batch':
            self.normalization_layer = keras.layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5,
                name='batch_norm'
            )
        elif self.normalization == 'layer':
            self.normalization_layer = keras.layers.LayerNormalization(
                epsilon=1e-5,
                name='layer_norm'
            )
        else:
            self.normalization_layer = None

        # Create Query, Key, Value projection layers
        self.query_conv = keras.layers.Conv2D(
            filters=self.attention_channels,
            name='query_conv',
            **self._conv_params
        )

        # Adjust key/value channels based on attention mode
        # Gaussian mode uses fewer channels as per original paper
        self.key_value_channels = (
            self.attention_channels
            if self.attention_mode == 'dot_product'
            else self.attention_channels // 8
        )

        self.key_conv = keras.layers.Conv2D(
            filters=self.key_value_channels,
            activation=self.intermediate_activation,
            name='key_conv',
            **self._conv_params
        )

        self.value_conv = keras.layers.Conv2D(
            filters=self.key_value_channels,
            name='value_conv',
            **self._conv_params
        )

        # Create attention mechanism
        self.attention = keras.layers.Attention(
            use_scale=self.attention_mode == 'dot_product',
            score_mode='dot',
            dropout=self.dropout_rate if self.dropout_rate > 0 else None,
            name='attention'
        )

        # Create dropout layer if specified
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name='dropout')
        else:
            self.dropout = None

        # Note: output_conv will be created in build() since we need input channels

    def _validate_inputs(
        self,
        attention_channels: int,
        dropout_rate: float,
        normalization: Optional[str],
        attention_mode: str
    ) -> None:
        """Validate initialization parameters.

        :param attention_channels: Number of attention channels to validate.
        :type attention_channels: int
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float
        :param normalization: Normalization type to validate.
        :type normalization: Optional[str]
        :param attention_mode: Attention mode to validate.
        :type attention_mode: str

        :raises ValueError: If any parameter is invalid.
        """
        if attention_channels <= 0:
            raise ValueError(f"attention_channels must be positive, got {attention_channels}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if normalization not in ['batch', 'layer', None]:
            raise ValueError(f"normalization must be 'batch', 'layer', or None, got {normalization}")
        if attention_mode not in ['gaussian', 'dot_product']:
            raise ValueError(f"attention_mode must be 'gaussian' or 'dot_product', got {attention_mode}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        channels = input_shape[-1]
        actual_output_channels = (
            channels if self.output_channels <= 0
            else self.output_channels
        )

        # Create output projection layer (needs input channels)
        self.output_conv = keras.layers.Conv2D(
            filters=actual_output_channels,
            activation=self.output_activation,
            name='output_conv',
            **self._conv_params
        )

        # Build all sub-layers in computational order for serialization robustness
        self.depthwise_conv.build(input_shape)

        # Depthwise conv doesn't change shape, so normalization uses same shape
        if self.normalization_layer is not None:
            self.normalization_layer.build(input_shape)

        # Query, Key, Value projections all use the same input shape
        self.query_conv.build(input_shape)
        self.key_conv.build(input_shape)
        self.value_conv.build(input_shape)

        # Attention layer build - it doesn't have explicit weights but we build for consistency
        # The attention expects [query, value, key] inputs as sequences
        batch_size = input_shape[0] if input_shape[0] is not None else 1
        height = input_shape[1] if input_shape[1] is not None else 32
        width = input_shape[2] if input_shape[2] is not None else 32
        seq_len = height * width

        query_seq_shape = (batch_size, seq_len, self.attention_channels)
        key_value_seq_shape = (batch_size, seq_len, self.key_value_channels)

        self.attention.build([query_seq_shape, key_value_seq_shape, key_value_seq_shape])

        # Output conv processes the attention output
        attention_output_shape = (batch_size, height, width, self.key_value_channels)
        self.output_conv.build(attention_output_shape)

        # Dropout doesn't need explicit building but we do it for consistency
        if self.dropout is not None:
            self.dropout.build(attention_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Apply non-local attention to input features.

        :param inputs: Input tensor of shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask tensor.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Affects dropout and normalization.
        :type training: Optional[bool]
        :param kwargs: Additional arguments (unused, kept for compatibility).
        :type kwargs: Any

        :return: Tensor of shape ``(batch_size, height, width, output_channels)``
            with spatially attended features incorporating long-range dependencies.
        :rtype: keras.KerasTensor
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

        # Reshape for attention computation: (B, H, W, C) -> (B, H*W, C)
        shape = ops.shape(query)
        batch_size, height, width = shape[0], shape[1], shape[2]

        query_reshaped = ops.reshape(query, [batch_size, -1, self.attention_channels])
        key_reshaped = ops.reshape(key, [batch_size, -1, self.key_value_channels])
        value_reshaped = ops.reshape(value, [batch_size, -1, self.key_value_channels])

        # Apply attention mechanism: [query, value, key] format for keras.layers.Attention
        attention_output = self.attention(
            [query_reshaped, value_reshaped, key_reshaped],
            mask=attention_mask,
            training=training
        )

        # Reshape back to spatial dimensions: (B, H*W, C) -> (B, H, W, C)
        attention_output = ops.reshape(
            attention_output,
            [batch_size, height, width, self.key_value_channels]
        )

        # Apply output projection
        output = self.output_conv(attention_output, training=training)

        # Apply dropout if specified and in training mode
        if self.dropout is not None:
            output = self.dropout(output, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple with spatial dimensions preserved.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        if self.output_channels > 0:
            output_shape[-1] = self.output_channels
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all parameters required to recreate this layer.
        :rtype: Dict[str, Any]
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

# ---------------------------------------------------------------------
