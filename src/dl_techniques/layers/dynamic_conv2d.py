"""
2D convolution with input-dependent dynamic kernel aggregation.

This layer enhances the representational power of a standard convolution by
dynamically adapting its behavior to each input instance. Instead of employing a
single, static convolutional kernel, it maintains a set of K parallel "expert"
kernels. For each input feature map, a lightweight attention mechanism generates
mixing coefficients used to compute a weighted sum of expert outputs:
y_dynamic = sum_k pi_k(x) * conv(x, W_k) where pi_k(x) = softmax(SE(GAP(x))/tau).

References:
    - Chen, Y., et al. (2019). Dynamic Convolution: Attention over
      Convolution Kernels. *CVPR*.
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict, List

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DynamicConv2D(keras.layers.Layer):
    """
    Dynamic 2D Convolution with attention over convolution kernels.

    Implements dynamic convolution as described in Chen et al. (2019), where
    K parallel convolution experts are aggregated using input-dependent attention
    weights from a squeeze-and-excitation gating network. The attention uses
    temperature-controlled softmax: pi(x) = softmax(SE(GAP(x)) / tau), and the
    final output is y = sum_k pi_k(x) * conv(x, W_k). This provides increased
    representation power without increasing network depth or width, with only
    approximately 4% additional FLOPs overhead.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────┐
        │  Input [batch, H, W, C]                         │
        └──────────┬──────────────────────┬───────────────┘
                   ▼                      ▼
        ┌─────────────────────┐  ┌────────────────────────┐
        │  Attention Network  │  │  K Parallel Conv2D     │
        │  ┌───────────────┐  │  │  Expert_0 ──▶ y_0     │
        │  │  GAP           │  │  │  Expert_1 ──▶ y_1     │
        │  └───────┬───────┘  │  │  ...                   │
        │          ▼          │  │  Expert_K ──▶ y_K      │
        │  ┌───────────────┐  │  └────────────┬───────────┘
        │  │ Dense(C//4,   │  │               ▼
        │  │   ReLU)       │  │    Stack: [y_0, ..., y_K]
        │  └───────┬───────┘  │               │
        │          ▼          │               │
        │  ┌───────────────┐  │               │
        │  │ Dense(K)      │  │               │
        │  └───────┬───────┘  │               │
        │          ▼          │               │
        │  ┌───────────────┐  │               │
        │  │ Softmax(tau)  │  │               │
        │  └───────┬───────┘  │               │
        └──────────┼──────────┘               │
                   ▼                          ▼
        ┌─────────────────────────────────────────────────┐
        │  Weighted Sum: sum_k pi_k * y_k                 │
        └──────────────────────┬──────────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  Activation (optional)                          │
        └──────────────────────┬──────────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  Output [batch, H', W', filters]                │
        └─────────────────────────────────────────────────┘

    :param filters: Dimensionality of the output space (number of output filters).
        Must be positive.
    :type filters: int
    :param kernel_size: Height and width of the 2D convolution window.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param num_kernels: Number of parallel convolution kernels to aggregate.
        Must be at least 2. Defaults to 4.
    :type num_kernels: int
    :param temperature: Temperature parameter for softmax attention. Higher values
        create more uniform attention. Defaults to 30.0.
    :type temperature: float
    :param attention_reduction_ratio: Reduction ratio for attention mechanism.
        Defaults to 4.
    :type attention_reduction_ratio: int
    :param strides: Strides of the convolution. Defaults to (1, 1).
    :type strides: Union[int, Tuple[int, int]]
    :param padding: Either "valid" or "same". Defaults to "valid".
    :type padding: str
    :param dilation_rate: Dilation rate for convolution. Defaults to (1, 1).
    :type dilation_rate: Union[int, Tuple[int, int]]
    :param groups: Number of groups for grouped convolution. Defaults to 1.
    :type groups: int
    :param activation: Activation function. Defaults to None.
    :type activation: Optional[Union[str, keras.layers.Activation]]
    :param use_bias: Whether the layer uses bias vectors. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for bias vectors.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Regularizer for layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kernel_constraint: Constraint for kernel weights.
    :type kernel_constraint: Optional[keras.constraints.Constraint]
    :param bias_constraint: Constraint for bias vectors.
    :type bias_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]],
            num_kernels: int = 4,
            temperature: float = 30.0,
            attention_reduction_ratio: int = 4,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = "valid",
            dilation_rate: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            activation: Optional[Union[str, keras.layers.Activation]] = None,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs
    ):
        """Initialize the DynamicConv2D layer."""
        super().__init__(**kwargs)

        # Validate core parameters
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if num_kernels < 2:
            raise ValueError(f"num_kernels must be at least 2, got {num_kernels}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if attention_reduction_ratio <= 0:
            raise ValueError(f"attention_reduction_ratio must be positive, got {attention_reduction_ratio}")
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")

        # Store configuration
        self.filters = filters
        self.num_kernels = num_kernels
        self.temperature = float(temperature)
        self.attention_reduction_ratio = attention_reduction_ratio

        # Normalize kernel_size to tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        # Normalize strides to tuple
        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = tuple(strides)

        # Normalize dilation_rate to tuple
        if isinstance(dilation_rate, int):
            self.dilation_rate = (dilation_rate, dilation_rate)
        else:
            self.dilation_rate = tuple(dilation_rate)

        # Validate padding
        if padding.lower() not in ['valid', 'same']:
            raise ValueError(f"padding must be 'valid' or 'same', got {padding}")
        self.padding = padding.lower()

        self.groups = groups
        self.use_bias = use_bias

        # Handle activation
        if activation is not None:
            self.activation = keras.activations.get(activation)
        else:
            self.activation = None

        # Store initializers, regularizers, and constraints
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        # Initialize sub-layers as None - they will be created in build()
        self.conv_layers: List[keras.layers.Conv2D] = []
        self.gap: Optional[keras.layers.GlobalAveragePooling2D] = None
        self.attention_dense1: Optional[keras.layers.Dense] = None
        self.attention_dense2: Optional[keras.layers.Dense] = None

        # Pre-create GAP layer since it doesn't depend on input shape
        self.gap = keras.layers.GlobalAveragePooling2D(
            data_format='channels_last',
            keepdims=False,
            name='gap'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating all sub-layers.

        :param input_shape: Shape tuple with format (batch_size, height, width, channels).
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), got {input_shape}"
            )

        # Extract input channels (channels_last format)
        batch_size, height, width, input_channels = input_shape

        if input_channels is None:
            raise ValueError("Channel dimension of input must be defined at build time")

        # Validate groups compatibility
        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by groups ({self.groups})"
            )
        if self.filters % self.groups != 0:
            raise ValueError(
                f"Filters ({self.filters}) must be divisible by groups ({self.groups})"
            )

        # Create K parallel convolution layers
        self.conv_layers = []
        for k in range(self.num_kernels):
            conv_layer = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format='channels_last',
                dilation_rate=self.dilation_rate,
                groups=self.groups,
                activation=None,  # Activation applied after aggregation
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=None,  # Activity regularizer applied on final output
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                name=f'conv_expert_{k}'
            )
            self.conv_layers.append(conv_layer)

        # Create attention mechanism layers
        # Calculate reduced channels for first dense layer
        attention_channels = max(1, input_channels // self.attention_reduction_ratio)

        self.attention_dense1 = keras.layers.Dense(
            attention_channels,
            activation='relu',
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='attention_dense1'
        )

        self.attention_dense2 = keras.layers.Dense(
            self.num_kernels,
            activation=None,  # No activation, softmax applied separately
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='attention_dense2'
        )

        # Build all sub-layers explicitly for proper serialization
        # This ensures weight variables exist before weight restoration
        self.gap.build(input_shape)

        # GAP output shape is always (batch_size, channels)
        gap_output_shape = (input_shape[0], input_channels)
        self.attention_dense1.build(gap_output_shape)

        # First dense output shape
        dense1_output_shape = (input_shape[0], attention_channels)
        self.attention_dense2.build(dense1_output_shape)

        # Build all convolution layers
        for conv_layer in self.conv_layers:
            conv_layer.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass with dynamic kernel aggregation.

        :param inputs: Input tensor with shape (batch_size, height, width, channels).
        :type inputs: keras.KerasTensor
        :param training: Boolean or None, indicates whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor with shape (batch_size, new_height, new_width, filters).
        :rtype: keras.KerasTensor
        """
        # Step 1: Compute attention weights
        # Global Average Pooling to compress spatial dimensions
        pooled = self.gap(inputs, training=training)  # Shape: (batch_size, input_channels)

        # First dense layer with ReLU activation
        attention_hidden = self.attention_dense1(pooled, training=training)
        # Shape: (batch_size, attention_channels)

        # Second dense layer to get attention logits
        attention_logits = self.attention_dense2(attention_hidden, training=training)
        # Shape: (batch_size, num_kernels)

        # Apply softmax with temperature to get normalized attention weights
        attention_weights = keras.ops.softmax(attention_logits / self.temperature)
        # Shape: (batch_size, num_kernels)

        # Step 2: Apply all convolution kernels in parallel
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(inputs, training=training)
            conv_outputs.append(conv_output)

        # Stack all convolution outputs: (num_kernels, batch_size, H', W', filters)
        stacked_outputs = keras.ops.stack(conv_outputs, axis=0)

        # Step 3: Aggregate outputs using attention weights
        # Reshape attention weights for broadcasting: (num_kernels, batch_size, 1, 1, 1)
        # We need to expand dimensions to match spatial and channel dimensions
        expanded_attention = keras.ops.expand_dims(attention_weights, axis=-1)  # (batch_size, num_kernels, 1)
        expanded_attention = keras.ops.expand_dims(expanded_attention, axis=-1)  # (batch_size, num_kernels, 1, 1)
        expanded_attention = keras.ops.expand_dims(expanded_attention, axis=-1)  # (batch_size, num_kernels, 1, 1, 1)

        # Transpose to match stacked_outputs: (num_kernels, batch_size, 1, 1, 1)
        expanded_attention = keras.ops.transpose(expanded_attention, (1, 0, 2, 3, 4))

        # Element-wise multiplication and sum over kernel dimension
        # stacked_outputs: (num_kernels, batch_size, ...)
        # expanded_attention: (num_kernels, batch_size, 1, 1, 1) -> broadcasts
        weighted_outputs = stacked_outputs * expanded_attention

        # Sum over kernel dimension to get final aggregated output
        aggregated_output = keras.ops.sum(weighted_outputs, axis=0)
        # Shape: (batch_size, H', W', filters)

        # Apply activation if specified
        if self.activation is not None:
            aggregated_output = self.activation(aggregated_output)

        return aggregated_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple (batch_size, height, width, channels).
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (batch_size, new_height, new_width, filters).
        :rtype: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        # If the layer is built, delegate to the first conv sub-layer for accuracy.
        if self.conv_layers:
            return self.conv_layers[0].compute_output_shape(input_shape)

        # Manual calculation if the layer is not yet built.
        # This mirrors Keras's internal logic for robustness.
        batch_size, height, width, _ = input_shape

        def conv_output_length(input_length, kernel_size, padding, stride, dilation):
            """Calculate output length for convolution dimension."""
            if input_length is None:
                return None
            if padding == "same":
                return (input_length + stride - 1) // stride

            # 'valid' padding
            dilated_kernel_size = (kernel_size - 1) * dilation + 1
            output_length = (input_length - dilated_kernel_size) // stride + 1
            return output_length

        out_height = conv_output_length(
            height, self.kernel_size[0], self.padding, self.strides[0], self.dilation_rate[0]
        )
        out_width = conv_output_length(
            width, self.kernel_size[1], self.padding, self.strides[1], self.dilation_rate[1]
        )

        return (batch_size, out_height, out_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'num_kernels': self.num_kernels,
            'temperature': self.temperature,
            'attention_reduction_ratio': self.attention_reduction_ratio,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        })
        return config

# ---------------------------------------------------------------------
