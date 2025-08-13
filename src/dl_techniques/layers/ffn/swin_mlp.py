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
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SwinMLP(keras.layers.Layer):
    """
    MLP module for Swin Transformer with configurable activation and regularization.

    This layer implements a two-layer MLP with configurable hidden dimension,
    activation function, dropout, and regularization. It's designed to be used
    within Swin Transformer blocks but can be used as a general-purpose MLP.

    The layer follows the expansion-contraction pattern typical in transformer
    architectures:
    1. Dense layer projecting to hidden_dim
    2. Non-linear activation (default GELU)
    3. Optional dropout for regularization
    4. Dense layer projecting to output dimension
    5. Optional final dropout

    Args:
        hidden_dim: Integer, dimension of the hidden layer. Must be positive.
            This is typically larger than the input dimension (e.g., 4x expansion).
        use_bias: Boolean, whether to use bias in dense layers. Defaults to True.
        out_dim: Optional integer, dimension of the output layer. If None, uses
            input dimension (auto-detected during first call). Must be positive if specified.
        activation: Union[str, callable], activation function name or callable.
            Can be string name ('gelu', 'relu') or callable. Defaults to 'gelu'.
        dropout_rate: Float, dropout rate for regularization. Must be between 0 and 1.
            Applied after both the activation and final projection. Defaults to 0.0.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer for
            kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer for
            bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]], regularizer
            for kernel weights.
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]], regularizer
            for bias weights.
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]], regularizer
            for layer activations.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common case is 3D input: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., out_dim)`.
        If out_dim is None, output shape is: `(batch_size, ..., input_dim)`.

    Raises:
        ValueError: If hidden_dim is not positive.
        ValueError: If out_dim is specified and not positive.
        ValueError: If dropout_rate is not between 0 and 1.

    Example:
        ```python
        # Basic usage for Swin Transformer (4x expansion)
        layer = SwinMLP(hidden_dim=3072)  # For 768-dim input

        # With custom output dimension and dropout
        layer = SwinMLP(
            hidden_dim=2048,
            out_dim=512,
            dropout_rate=0.1,
            activation='swish'
        )

        # With regularization
        layer = SwinMLP(
            hidden_dim=1024,
            dropout_rate=0.15,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            activity_regularizer=keras.regularizers.L1(1e-5)
        )

        # In a model
        inputs = keras.Input(shape=(196, 768))  # Vision Transformer patch tokens
        x = SwinMLP(hidden_dim=3072, dropout_rate=0.1)(inputs)
        model = keras.Model(inputs, x)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and avoids common build errors.
    """

    def __init__(
            self,
            hidden_dim: int,
            use_bias: bool = True,
            out_dim: Optional[int] = None,
            activation: Union[str, Callable] = "gelu",
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim is not None and out_dim <= 0:
            raise ValueError(f"out_dim must be positive when specified, got {out_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store ALL configuration parameters as instance attributes
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.out_dim = out_dim
        self.activation_fn = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Store original activation for serialization
        self.activation = activation

        # CREATE all sub-layers in __init__ (MODERN PATTERN)
        # First dense layer (expansion)
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

        # Dropout layers (created even if rate is 0 for consistency)
        self.drop1 = keras.layers.Dropout(self.dropout_rate, name="drop1")
        self.drop2 = keras.layers.Dropout(self.dropout_rate, name="drop2")

        # Second dense layer will be created when we know the output dimension
        # We defer this to the first call since out_dim might be None
        self.fc2 = None
        self._output_dim = None

    def _create_second_layer(self, input_dim: int) -> None:
        """Create the second dense layer when output dimension is known."""
        if self.fc2 is None:
            # Determine actual output dimension
            self._output_dim = self.out_dim if self.out_dim is not None else input_dim

            # Create second dense layer (contraction)
            self.fc2 = keras.layers.Dense(
                self._output_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                name="fc2"
            )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the SwinMLP layer.

        Args:
            x: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Affects dropout behavior.

        Returns:
            Output tensor of shape (..., out_dim) after MLP transformation.
        """
        # Create second layer if not yet created
        if self.fc2 is None:
            input_dim = x.shape[-1]
            self._create_second_layer(input_dim)

        # First dense layer (expansion)
        x = self.fc1(x, training=training)

        # Activation
        x = self.activation_fn(x)

        # First dropout (after activation)
        x = self.drop1(x, training=training)

        # Second dense layer (contraction)
        x = self.fc2(x, training=training)

        # Second dropout (final regularization)
        x = self.drop2(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation
        output_shape = list(input_shape)

        # Set output dimension
        if self.out_dim is not None:
            output_shape[-1] = self.out_dim
        # If out_dim is None, output shape is same as input shape

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        This method returns ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "use_bias": self.use_bias,
            "out_dim": self.out_dim,
            "activation": self.activation,  # Store original activation for serialization
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    # NOTE: get_build_config() and build_from_config() are REMOVED
    # These are deprecated methods that cause serialization issues in Keras 3
    # The modern pattern handles building automatically

# ---------------------------------------------------------------------