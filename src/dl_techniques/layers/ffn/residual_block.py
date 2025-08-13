"""Defines a custom Keras Residual Block layer.

This module provides a custom Keras layer, `ResidualBlock`, which implements a
residual connection around a two-layer feed-forward network. The block is designed
to be a flexible and reusable component in deep learning models.

The main path consists of two dense layers with a configurable activation function
and dropout. The residual (or skip) path contains a separate dense layer to project
the input to the same dimensionality as the output of the main path. This design
allows the block to be used even when the input and output dimensions differ.

The layer is registered as a Keras serializable object, allowing models that
use it to be saved and loaded seamlessly using `keras.models.save_model` and
`keras.models.load_model`.

Example:
    ```python
    import keras
    import numpy as np

    # Create a simple model using the ResidualBlock
    inputs = keras.Input(shape=(32,))
    x = ResidualBlock(hidden_dim=64, output_dim=16, dropout_rate=0.1)(inputs)
    x = keras.layers.LayerNormalization()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    # Print model summary
    model.summary()

    # Test with dummy data
    dummy_data = np.random.rand(10, 32)
    predictions = model.predict(dummy_data)
    print(f"Output shape: {predictions.shape}")
    # Output shape: (10, 1)
    ```

"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ResidualBlock(keras.layers.Layer):
    """
    Residual block with linear transformations and configurable activation.

    This layer applies a residual connection around a two-layer MLP with configurable activation.
    The architecture follows the pattern:

        output = output_layer(dropout(activation(hidden_layer(x)))) + residual_layer(x)

    Where:
    - Main path: x → Dense(hidden_dim) → Activation → Dropout → Dense(output_dim)
    - Residual path: x → Dense(output_dim)
    - Final output: main_path + residual_path

    Args:
        hidden_dim: Integer, dimensionality of the hidden layer. Must be positive.
        output_dim: Integer, dimensionality of the output space. Must be positive.
        dropout_rate: Float, dropout rate for regularization. Must be between 0 and 1.
            Defaults to 0.0 (no dropout).
        activation: String or callable, activation function. Defaults to 'relu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: String or initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance for bias vector.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case is 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., output_dim)`.
        For 2D input: `(batch_size, output_dim)`.

    Raises:
        ValueError: If hidden_dim or output_dim are not positive integers.
        ValueError: If dropout_rate is not between 0 and 1.

    Example:
        ```python
        # Basic usage
        layer = ResidualBlock(hidden_dim=128, output_dim=64)

        # Advanced configuration
        layer = ResidualBlock(
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.2,
            activation='gelu',
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(784,))
        x = ResidualBlock(hidden_dim=512, output_dim=256)(inputs)
        x = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where sub-layers
        are created in __init__ and Keras handles the build lifecycle automatically.
        This ensures proper serialization and avoids common build errors.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            dropout_rate: float = 0.0,
            activation: Union[str, callable] = 'relu',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate input parameters
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters as instance attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers here in __init__ (MODERN KERAS 3 PATTERN)

        # Hidden transformation layer with activation
        self.hidden_layer = keras.layers.Dense(
            self.hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="hidden_layer"
        )

        # Output transformation layer (no activation - applied after residual)
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_layer"
        )

        # Residual connection layer
        self.residual_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="residual_layer"
        )

        # Dropout layer (only create if needed)
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout = None

        # No custom weights to create, so no build() method is needed
        # Keras will automatically handle building of sub-layers

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass with residual connection.

        Args:
            inputs: Input tensor of shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (..., output_dim).
        """
        # Main path: hidden layer with activation
        hidden = self.hidden_layer(inputs, training=training)

        # Apply dropout if configured
        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)

        # Output layer
        output = self.output_layer(hidden, training=training)

        # Residual path: direct projection
        residual = self.residual_layer(inputs, training=training)

        # Combine main path and residual path
        return output + residual

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    # DELETED: get_build_config() and build_from_config() methods
    # These are deprecated in Keras 3 and cause serialization issues
    # Keras handles the build lifecycle automatically with the modern pattern

# ---------------------------------------------------------------------