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
    >>> import keras
    >>> import numpy as np
    >>>
    >>> # Create a simple model using the ResidualBlock
    >>> inputs = keras.Input(shape=(32,))
    >>> x = ResidualBlock(hidden_dim=64, output_dim=16, dropout_rate=0.1)(inputs)
    >>> x = keras.layers.LayerNormalization()(x)
    >>> outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    >>> model = keras.Model(inputs, outputs)
    >>>
    >>> # Print model summary
    >>> model.summary()
    >>>
    >>> # Test with dummy data
    >>> dummy_data = np.random.rand(10, 32)
    >>> predictions = model.predict(dummy_data)
    >>> print(f"Output shape: {predictions.shape}")
    Output shape: (10, 1)

"""

import keras
from typing import Optional, Union, Any, Tuple, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ResidualBlock(keras.layers.Layer):
    """
    Residual block with linear transformations and configurable activation.

    This layer implements a residual connection around a two-layer MLP. The main path
    consists of:
    1. Dense layer with configurable activation
    2. Optional dropout for regularization
    3. Dense layer (linear output)

    The residual path uses a separate dense layer to project the input to match
    the output dimensionality, enabling residual connections even when input and
    output dimensions differ.

    Mathematical formulation:
        hidden = activation(input @ W1 + b1)
        hidden = dropout(hidden) if dropout_rate > 0
        main_output = hidden @ W2 + b2
        residual_output = input @ W_res + b_res
        output = main_output + residual_output

    Args:
        hidden_dim: Integer, dimensionality of the hidden layer. Must be positive.
        output_dim: Integer, dimensionality of the output space. Must be positive.
        dropout_rate: Float, dropout rate for regularization between 0 and 1.
            Defaults to 0.0 (no dropout).
        activation: String or callable, activation function for hidden layer.
            Accepts string names ('relu', 'gelu') or callable functions.
            Defaults to 'relu'.
        use_bias: Boolean, whether to use bias in all dense layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights. Accepts string
            names ('l1', 'l2') or Regularizer instances. Defaults to None.
        bias_regularizer: Optional regularizer for bias weights. Accepts string
            names ('l1', 'l2') or Regularizer instances. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tensor with shape `(batch_size, ..., input_dim)` where input_dim is the
        size of the last dimension.

    Output shape:
        Tensor with shape `(batch_size, ..., output_dim)` where all dimensions
        except the last are preserved from the input.

    Attributes:
        hidden_layer: Dense layer for hidden transformation.
        output_layer: Dense layer for output transformation.
        residual_layer: Dense layer for residual projection.
        dropout: Dropout layer if dropout_rate > 0, otherwise None.

    Example:
        ```python
        # Basic usage
        block = ResidualBlock(hidden_dim=128, output_dim=64)

        # With dropout and custom activation
        block = ResidualBlock(
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.2,
            activation='gelu'
        )

        # With regularization
        block = ResidualBlock(
            hidden_dim=512,
            output_dim=256,
            kernel_regularizer='l2',
            bias_regularizer=keras.regularizers.L1(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(784,))
        x = ResidualBlock(hidden_dim=512, output_dim=256)(inputs)
        x = keras.layers.LayerNormalization()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If hidden_dim or output_dim is not positive.
        ValueError: If dropout_rate is not between 0 and 1.

    Note:
        This implementation creates a learnable residual projection, making it suitable
        for cases where input and output dimensions differ. For same-dimension cases,
        this adds parameters but maintains architectural flexibility.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.0,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)
        # Hidden transformation with activation
        self.hidden_layer = keras.layers.Dense(
            units=self.hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="hidden_layer"
        )

        # Output transformation (linear)
        self.output_layer = keras.layers.Dense(
            units=self.output_dim,
            activation=None,  # Linear output
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_layer"
        )

        # Residual projection layer
        self.residual_layer = keras.layers.Dense(
            units=self.output_dim,
            activation=None,  # Linear projection
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="residual_layer"
        )

        # Optional dropout layer
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Build sub-layers in computational order for robust serialization
        self.hidden_layer.build(input_shape)

        # Compute intermediate shape after hidden layer
        hidden_output_shape = list(input_shape)
        hidden_output_shape[-1] = self.hidden_dim
        hidden_output_shape = tuple(hidden_output_shape)

        # Dropout doesn't change shape, so use hidden_output_shape
        if self.dropout is not None:
            self.dropout.build(hidden_output_shape)

        self.output_layer.build(hidden_output_shape)

        # Residual layer takes original input shape
        self.residual_layer.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with residual connection.

        Args:
            inputs: Input tensor of shape (batch_size, ..., input_dim).
            training: Boolean indicating training mode for dropout.

        Returns:
            Output tensor of shape (batch_size, ..., output_dim).
        """
        # Main path: input -> hidden -> dropout -> output
        hidden = self.hidden_layer(inputs, training=training)

        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)

        main_output = self.output_layer(hidden, training=training)

        # Residual path: input -> projection
        residual_output = self.residual_layer(inputs, training=training)

        # Combine main path and residual path
        return main_output + residual_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple with last dimension set to output_dim.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct this layer.
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

# ---------------------------------------------------------------------