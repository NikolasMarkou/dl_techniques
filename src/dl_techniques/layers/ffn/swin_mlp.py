"""
# SwinMLP Layer
A Keras layer implementing the MLP (Multi-Layer Perceptron) module used in Swin Transformer
architectures, providing a flexible and efficient feedforward network with configurable
activation, dropout, and regularization.

## Conceptual Overview
The SwinMLP layer is a core component of the Swin Transformer architecture, serving as the
feedforward network (FFN) within each transformer block. Unlike standard transformer MLPs
that use ReLU activation, SwinMLP defaults to GELU activation for improved gradient flow
and representation learning.

This implementation provides a two-layer feedforward network with an expansion-contraction
pattern, where the hidden dimension is typically larger than the input/output dimensions,
allowing the model to learn complex non-linear transformations in a higher-dimensional space
before projecting back to the original dimension.

### Architecture Pattern:
The layer follows the standard transformer MLP pattern:
1. **Expansion**: Linear projection to higher-dimensional space (hidden_dim)
2. **Activation**: Non-linear transformation (default: GELU)
3. **Regularization**: Optional dropout for generalization
4. **Contraction**: Linear projection back to output dimension
5. **Final Regularization**: Optional dropout before residual connection

### Mathematical Description:
For input tensor `x` with shape `(..., input_dim)`:

#### Standard Operation (out_dim = None):
```
h = Dense₁(x)           # x ∈ ℝ^(input_dim) → h ∈ ℝ^(hidden_dim)
h = GELU(h)             # Non-linear activation
h = Dropout(h)          # Optional regularization
y = Dense₂(h)           # h ∈ ℝ^(hidden_dim) → y ∈ ℝ^(input_dim)
y = Dropout(y)          # Optional regularization
```

#### With Custom Output Dimension:
```
h = Dense₁(x)           # x ∈ ℝ^(input_dim) → h ∈ ℝ^(hidden_dim)
h = GELU(h)             # Non-linear activation
h = Dropout(h)          # Optional regularization
y = Dense₂(h)           # h ∈ ℝ^(hidden_dim) → y ∈ ℝ^(out_dim)
y = Dropout(y)          # Optional regularization
```

Where:
- `Dense₁` and `Dense₂` are linear transformations with learnable weights and biases
- `GELU(x) = x * Φ(x)` where Φ is the cumulative distribution function of standard normal
- `Dropout` applies stochastic regularization during training only

### Key Benefits:
1. **Improved Gradient Flow**: GELU activation provides smoother gradients compared to ReLU
2. **Flexible Dimensionality**: Configurable hidden and output dimensions for various architectures
3. **Regularization Control**: Built-in dropout with separate rates for different stages
4. **Memory Efficiency**: Optional output dimension allows for dimension reduction/expansion
5. **Training Stability**: Proper weight initialization and regularization options
6. **Activation Flexibility**: Support for any Keras activation function or custom callables

The layer can be used as a drop-in replacement for custom MLP implementations in transformer
architectures, or as a standalone feedforward network in any neural architecture requiring
non-linear transformations with regularization.

### Usage Patterns:
- **Swin Transformer Blocks**: As the FFN component with 4x expansion ratio
- **Vision Transformers**: For patch embedding and feature transformation
- **General Purpose**: Any architecture requiring configurable feedforward networks
- **Dimension Adaptation**: Converting between different feature dimensions in networks
"""

import keras
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SwinMLP(keras.layers.Layer):
    """MLP module for Swin Transformer with configurable activation and regularization.

    This layer implements a two-layer MLP with configurable hidden dimension,
    activation function, dropout, and regularization. It's designed to be used
    within Swin Transformer blocks but can be used as a general-purpose MLP.

    Args:
        hidden_dim: Dimension of the hidden layer.
        use_bias: Whether to use bias or not
        out_dim: Dimension of the output layer. If None, uses input dimension.
        activation: Activation function name or callable. Defaults to "gelu".
        dropout_rate: Dropout rate. Defaults to 0.0.
        kernel_initializer: Initializer for the kernel weights. Defaults to "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for the kernel weights.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        **kwargs: Additional keyword arguments for Layer base class.
    """

    def __init__(
            self,
            hidden_dim: int,
            use_bias: bool = True,
            out_dim: Optional[int] = None,
            activation: Union[str, callable] = "gelu",
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"drop rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Initialize layers to None - will be created in build()
        self.fc1 = None
        self.act = None
        self.drop1 = None
        self.fc2 = None
        self.drop2 = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sublayers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Determine output dimension
        input_dim = input_shape[-1]
        out_dim = self.out_dim or input_dim

        # Create first dense layer
        self.fc1 = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc1"
        )

        # Create activation layer
        self.act = keras.layers.Activation(self.activation, name="act")

        # Create dropout layers if needed
        if self.dropout_rate > 0.0:
            self.drop1 = keras.layers.Dropout(self.dropout_rate, name="drop1")
            self.drop2 = keras.layers.Dropout(self.dropout_rate, name="drop2")

        # Create second dense layer
        self.fc2 = keras.layers.Dense(
            out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            name="fc2"
        )

        # Build sublayers
        self.fc1.build(input_shape)

        # Compute intermediate shape for fc2
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        self.fc2.build(tuple(intermediate_shape))

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the SwinMLP layer.

        Args:
            x: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (..., out_dim) after MLP transformation.
        """
        # First dense layer
        x = self.fc1(x)

        # Activation
        x = self.act(x)

        # First dropout
        if self.drop1 is not None:
            x = self.drop1(x, training=training)

        # Second dense layer
        x = self.fc2(x)

        # Second dropout
        if self.drop2 is not None:
            x = self.drop2(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Set output dimension
        if self.out_dim is not None:
            input_shape_list[-1] = self.out_dim
        # If out_dim is None, output shape is same as input shape

        # Return as tuple
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "activation": self.activation,
            "drop": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
