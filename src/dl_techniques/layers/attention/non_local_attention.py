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
from keras import ops
from typing import Any, Dict, Tuple, Optional, Literal, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NonLocalAttention(keras.layers.Layer):
    """
    Non-local Self Attention Layer for computer vision tasks.

    Implementation of the self-attention mechanism from "Non-local Neural Networks"
    (Wang et al., 2018) that captures long-range dependencies in feature
    representations through self-attention mechanisms optimized for 4D image tensors.

    **Intent**: Provide a production-ready non-local attention block that enables
    convolutional networks to capture long-range spatial dependencies by computing
    attention between all spatial positions, overcoming the limited receptive field
    of standard convolutions.

    **Architecture**:
    ```
    Input [B, H, W, C]
           ↓
    DepthwiseConv2D(7×7) → Normalization (optional)
           ↓
    Q_proj → Q [B, H, W, attention_channels]
    K_proj → K [B, H, W, key_value_channels]
    V_proj → V [B, H, W, key_value_channels]
           ↓
    Reshape → Q [B, H*W, attention_channels]
           → K [B, H*W, key_value_channels]
           → V [B, H*W, key_value_channels]
           ↓
    Attention(Q, K, V) → [B, H*W, key_value_channels]
           ↓
    Reshape → [B, H, W, key_value_channels]
           ↓
    Output_proj → Output [B, H, W, output_channels]
           ↓
    Dropout (optional) → Final Output [B, H, W, output_channels]
    ```

    **Mathematical Operations**:
    1. **Spatial Pre-processing**: X' = DepthwiseConv2D(X), optionally normalized
    2. **Projections**: Q = X' W_q, K = X' W_k, V = X' W_v
    3. **Spatial Flattening**: Reshape (H,W) → (H*W) for sequence processing
    4. **Attention**: A = Attention(Q, K, V) using scaled dot-product or Gaussian
    5. **Spatial Reconstruction**: Reshape (H*W) → (H,W)
    6. **Output**: O = A W_o, optionally with dropout

    Args:
        attention_channels: Integer, number of channels in attention mechanism.
            Must be positive.
        kernel_size: Integer or tuple, size of depthwise convolution kernel.
            Defaults to (7, 7).
        use_bias: Boolean, whether to use bias in convolution layers.
            Defaults to False.
        normalization: String, type of normalization to use ('batch', 'layer', or None).
            Defaults to 'batch'.
        intermediate_activation: String or callable, activation function for
            intermediate layers. Defaults to 'relu'.
        output_activation: String or callable, activation function for output.
            Defaults to 'linear'.
        output_channels: Integer, number of output channels (-1 for same as input).
            Defaults to -1.
        dropout_rate: Float, dropout rate (0 to disable). Must be between 0.0 and 1.0.
            Defaults to 0.0.
        attention_mode: String, type of attention mechanism ('gaussian', 'dot_product').
            Defaults to 'gaussian'.
        kernel_initializer: String or initializer instance, initializer for the
            kernel weights. Defaults to 'glorot_normal'.
        bias_initializer: String or initializer instance, initializer for the
            bias vector. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        activity_regularizer: Optional regularizer for layer activity.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, output_channels)`
        where output_channels is specified or same as input if -1.

    Call arguments:
        inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        training: Boolean indicating training or inference mode. Affects dropout
            and normalization behavior.

    Returns:
        Output tensor with shape `(batch_size, height, width, output_channels)`
        containing spatially attended features with long-range dependencies.

    Example:
        ```python
        # Basic usage for feature map attention
        inputs = keras.Input(shape=(32, 32, 256))
        attention_layer = NonLocalAttention(attention_channels=128)
        outputs = attention_layer(inputs)
        print(outputs.shape)  # (None, 32, 32, 256)

        # With residual connection (common pattern)
        def residual_nonlocal_block(x):
            attended = NonLocalAttention(
                attention_channels=x.shape[-1] // 2,
                dropout_rate=0.1
            )(x)
            return x + attended

        # Advanced configuration for high-resolution inputs
        nonlocal_layer = NonLocalAttention(
            attention_channels=64,
            kernel_size=(5, 5),
            normalization='layer',
            attention_mode='dot_product',
            output_channels=512,
            dropout_rate=0.2,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a ResNet-style architecture
        def nonlocal_resnet_block(x, filters):
            # Standard conv block
            conv_out = keras.layers.Conv2D(filters, 3, padding='same')(x)
            conv_out = keras.layers.BatchNormalization()(conv_out)
            conv_out = keras.layers.ReLU()(conv_out)

            # Non-local attention
            attended = NonLocalAttention(
                attention_channels=filters // 2
            )(conv_out)

            # Residual connection
            return conv_out + attended
        ```

    Raises:
        ValueError: If attention_channels <= 0.
        ValueError: If dropout_rate not in [0, 1).
        ValueError: If normalization not in ['batch', 'layer', None].
        ValueError: If attention_mode not in ['gaussian', 'dot_product'].

    Note:
        This implementation follows modern Keras 3 patterns where all sub-layers
        are created in __init__ and explicitly built in build() method for robust
        serialization support. The layer is optimized for 4D image tensors and
        includes spatial pre-processing for enhanced feature quality.
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
        """
        Validate initialization parameters.

        Args:
            attention_channels: Number of attention channels to validate.
            dropout_rate: Dropout rate to validate.
            normalization: Normalization type to validate.
            attention_mode: Attention mode to validate.

        Raises:
            ValueError: If any parameter is invalid.
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
        """
        Build the layer and all its sub-layers.

        This method explicitly builds each sub-layer for robust serialization
        support, ensuring all weight variables exist before weight restoration.

        Args:
            input_shape: Shape tuple of the input tensor.
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
        training: Optional[bool] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """
        Apply non-local attention to input features.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean, whether in training mode. Affects dropout and
                normalization behavior.
            **kwargs: Additional arguments (unused but kept for compatibility).

        Returns:
            Output tensor of shape (batch_size, height, width, output_channels)
            with spatially attended features incorporating long-range dependencies.
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
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Spatial dimensions remain the same, only channels may change.
        """
        output_shape = list(input_shape)
        if self.output_channels > 0:
            output_shape[-1] = self.output_channels
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL configuration parameters passed to __init__ for proper
        serialization and deserialization.

        Returns:
            Dictionary containing the layer configuration with all parameters
            required to recreate this layer.
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

