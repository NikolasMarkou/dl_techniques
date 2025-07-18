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

Classes:
    ResidualBlock: A Keras layer implementing a residual block with dense layers.

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
from typing import Optional, Union, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ResidualBlock(keras.layers.Layer):
    """
    Residual block with linear transformations and ReLU activation.

    This layer applies a residual connection around a two-layer MLP with ReLU activation.

    Args:
        hidden_dim: Integer, dimensionality of the hidden layer.
        output_dim: Integer, dimensionality of the output space.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        activation: String or callable, activation function. Defaults to 'relu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_regularizer: Regularizer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
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

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Layers will be initialized in build()
        self.hidden_layer = None
        self.output_layer = None
        self.residual_layer = None
        self.dropout = None
        self.activation_fn = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer weights."""
        self._build_input_shape = input_shape

        # Hidden transformation
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

        # Output transformation
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_layer"
        )

        # Residual connection
        self.residual_layer = keras.layers.Dense(
            self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="residual_layer"
        )

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Build sublayers
        self.hidden_layer.build(input_shape)

        # Calculate intermediate shape after hidden layer
        hidden_output_shape = list(input_shape)
        hidden_output_shape[-1] = self.hidden_dim
        self.output_layer.build(tuple(hidden_output_shape))

        self.residual_layer.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with residual connection."""
        # Main path
        hidden = self.hidden_layer(inputs, training=training)
        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)
        output = self.output_layer(hidden, training=training)

        # Residual path
        residual = self.residual_layer(inputs, training=training)

        # Combine
        return output + residual

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        """Get layer configuration."""
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

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
