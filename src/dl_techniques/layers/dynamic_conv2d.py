"""
2D convolution with input-dependent dynamic kernel aggregation.

This layer enhances the representational power of a standard convolution by
dynamically adapting its behavior to each input instance. Instead of employing a
single, static convolutional kernel, it maintains a set of `K` parallel
"expert" kernels. For each input feature map, a lightweight attention mechanism
generates a set of mixing coefficients, which are then used to compute a
weighted sum of the outputs from the expert kernels.

Architectural and Mathematical Underpinnings:

The core principle is to replace the static linear transformation of a
standard convolution with an input-dependent, non-linear function. This is
achieved by aggregating multiple linear functions (the expert convolutions)
with weights determined by the input itself.

1.  **Parallel Convolution Experts**: The layer maintains `K` separate `Conv2D`
    operations, each with its own learnable kernel `W_k` and bias `b_k`. For a
    given input `x`, each expert computes a separate output feature map:

        y_k = conv(x, W_k) + b_k

2.  **Input-Dependent Attention Mechanism**: A small gating network computes `K`
    attention scores, `π_k(x)`, that determine how to combine the expert
    outputs. This network follows a Squeeze-and-Excitation architecture:
    a.  **Squeeze**: The input `x` is spatially compressed into a channel
        descriptor vector via Global Average Pooling (GAP). This vector serves
        as a compact, global summary of the input's content.
    b.  **Excitation**: This summary vector is passed through a two-layer
        fully-connected network. This network learns a non-linear mapping
        from the input's global features to the optimal mixing strategy for
        the kernels.
    c.  **Weight Generation**: The final layer of the attention network outputs
        `K` logits, which are converted into a probability distribution over
        the kernels using a temperature-controlled softmax function:

            π(x) = softmax(attention_network(GAP(x)) / τ)

        The temperature `τ` is a crucial hyperparameter that controls the
        sparsity of the attention weights. High temperatures encourage smoother,
        more uniform weights (useful for stabilizing early training), while
        low temperatures lead to a "harder" selection where only a few expert
        kernels are chosen.

3.  **Dynamic Output Aggregation**: The final output of the layer is the
    weighted sum of the individual expert outputs, using the dynamically
    computed attention scores `π_k(x)` as the weights:

        y_dynamic = Σ_{k=1 to K} π_k(x) * y_k

This entire process is differentiable, allowing the kernel weights `W_k`, bias
terms `b_k`, and the parameters of the attention network to be jointly
optimized via backpropagation. The layer learns not only a set of specialized
feature detectors (the kernels) but also the logic for when and how to combine
them, effectively increasing the model's capacity without a significant increase
in width or depth.

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
    Dynamic 2D Convolution with Attention over Convolution Kernels.

    This layer implements dynamic convolution as described in "Dynamic Convolution:
    Attention over Convolution Kernels" (Chen et al., 2019). Instead of using a
    single convolution kernel, it aggregates multiple parallel convolution kernels
    dynamically based on input-dependent attention weights.

    **Intent**: Increase model representation capability without increasing network
    depth or width by using input-dependent kernel selection and aggregation.

    **Mathematical Formulation**:
    Instead of standard convolution: y = conv(x, W) + b
    Dynamic convolution computes:
    ```
    π_k(x) = softmax(attention_network(GAP(x)) / τ)
    y_k = conv(x, W_k) + b_k  for each kernel k
    y = Σ_k π_k(x) * y_k      (weighted combination of outputs)
    ```

    Where:
    - π_k(x) are input-dependent attention weights summing to 1
    - W_k, b_k are the k-th convolution kernel and bias
    - τ is the temperature parameter for softmax
    - GAP is Global Average Pooling

    **Architecture**:
    ```
    Input Feature Map
           ↓
    ┌─────────────────┐  ┌─────────────────────────┐
    │ Global Avg Pool │  │ K Parallel Convolutions │
    │        ↓        │  │   W_1×x    W_2×x  ...   │
    │ Dense(C//4,ReLU)│  │   y_1      y_2    ...   │
    │        ↓        │  └─────────────────────────┘
    │ Dense(K)        │              ↓
    │        ↓        │    [y_1, y_2, ..., y_K]
    │ Softmax(τ)      │              ↓
    └─────────────────┘      Weighted Aggregation
           ↓                  π_1×y_1 + π_2×y_2 + ...
    [π_1, π_2, ..., π_K]            ↓
           ↓                   Final Output
           └─────→ Weights ─────────→
    ```

    **Key Benefits**:
    - Increased representation power through non-linear kernel aggregation
    - Input-dependent adaptation without architectural changes
    - Computationally efficient (~4% FLOPs overhead)
    - Drop-in replacement for standard Conv2D layers

    Args:
        filters: Integer, dimensionality of the output space (number of output filters).
            Must be positive.
        kernel_size: An integer or tuple/list of 2 integers, specifying the height
            and width of the 2D convolution window. Must be positive.
        num_kernels: Integer, number of parallel convolution kernels to aggregate.
            More kernels provide higher representation power but increase computational
            cost and training difficulty. Must be at least 2. Defaults to 4.
        temperature: Float, temperature parameter for softmax attention. Higher values
            create more uniform attention (helpful for training), lower values create
            more sparse attention. Must be positive. Defaults to 30.0 as recommended.
        attention_reduction_ratio: Integer, reduction ratio for attention mechanism.
            The first dense layer reduces input channels by this factor. Must be positive.
            Defaults to 4.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution. Defaults to (1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: An integer or tuple/list of 2 integers, specifying dilation rate.
            Defaults to (1, 1).
        groups: A positive integer specifying number of groups for grouped convolution.
            Must divide both input and output channels evenly. Defaults to 1.
        activation: Activation function. If None, no activation is applied. Defaults to None.
        use_bias: Boolean, whether the layer uses bias vectors. Defaults to True.
        kernel_initializer: Initializer for convolution kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Regularizer function for kernel weights. Defaults to None.
        bias_regularizer: Regularizer function for bias vectors. Defaults to None.
        activity_regularizer: Regularizer function for layer output. Defaults to None.
        kernel_constraint: Constraint function for kernel weights. Defaults to None.
        bias_constraint: Constraint function for bias vectors. Defaults to None.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: (batch_size, height, width, channels) in channels_last format.

    Output shape:
        4D tensor with shape: (batch_size, new_height, new_width, filters) in channels_last format.

    Example:
        ```python
        # Basic usage as drop-in replacement for Conv2D
        x = keras.random.normal((32, 64, 64, 3))
        layer = DynamicConv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        y = layer(x)  # Shape: (32, 64, 64, 64)

        # With advanced configuration
        layer = DynamicConv2D(
            filters=128,
            kernel_size=(3, 3),
            num_kernels=6,
            temperature=15.0,
            strides=2,
            padding='same',
            activation='gelu'
        )
        y = layer(x)  # Shape: (32, 32, 32, 128)

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        x = DynamicConv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        x = DynamicConv2D(128, 3, padding='same', activation='relu')(x)
        outputs = keras.layers.GlobalAveragePooling2D()(x)
        model = keras.Model(inputs, outputs)
        ```

    Notes:
        - This layer assumes channels_last data format (batch, height, width, channels)
        - The attention mechanism uses a squeeze-and-excitation pattern which adds
          minimal computational overhead (~4% FLOPs)
        - Temperature parameter τ is crucial: start with default (30.0) and adjust if needed
        - For best results, use K=4 kernels as recommended in the paper
        - All convolution experts share the same parameters (strides, padding, etc.)
          but have independent weights
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
        """
        Build the layer by creating all sub-layers.

        Args:
            input_shape: Shape tuple with format (batch_size, height, width, channels).
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
        """
        Forward pass with dynamic kernel aggregation.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean or None, indicates whether in training mode.

        Returns:
            Output tensor with shape (batch_size, new_height, new_width, filters).
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
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple (batch_size, height, width, channels).

        Returns:
            Output shape tuple (batch_size, new_height, new_width, filters).
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
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
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