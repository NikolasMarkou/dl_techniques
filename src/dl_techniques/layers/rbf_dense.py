import keras
import tensorflow as tf
from typing import Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .rbf import RBFLayer


# ---------------------------------------------------------------------

class RBFDenseBlock(keras.layers.Layer):
    """A reusable block combining RBF and Dense layers with proper normalization.

    This block implements the following architecture:
    Input → RBF → LayerNorm → Dense → LayerNorm → ReLU → Dense (output)

    Args:
        rbf_units: Number of RBF centers.
        hidden_units: Number of units in the hidden dense layer.
        output_units: Number of units in the output dense layer.
        rbf_initializer: Initializer for RBF parameters. Defaults to 'uniform'.
        kernel_initializer: Initializer for dense layers. Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer for dense layers.
        use_bias: Whether to use bias in dense layers. Defaults to True.
        activation: Optional activation for output layer. Defaults to None.
        epsilon: Small constant for layer normalization. Defaults to 1e-6.

    Input shape:
        (batch_size, input_dim)

    Output shape:
        (batch_size, output_units)
    """

    def __init__(
            self,
            rbf_units: int,
            hidden_units: int,
            output_units: int,
            rbf_initializer: Union[str, keras.initializers.Initializer] = 'uniform',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            activation: Optional[Union[str, keras.layers.Activation]] = None,
            hidden_activation: Union[str, keras.layers.Activation] = 'gelu',
            epsilon: float = 1e-6,
            **kwargs: Any
    ) -> None:
        """Initialize the RBFDenseBlock."""
        super().__init__(**kwargs)

        # Store configuration
        self.rbf_units = rbf_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.rbf_initializer = keras.initializers.get(rbf_initializer)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.hidden_activation = keras.activations.get(hidden_activation)
        self.epsilon = epsilon

        # Create layers
        self.rbf_layer = RBFLayer(
            units=rbf_units,
            center_initializer=rbf_initializer
        )

        self.norm1 = keras.layers.LayerNormalization(epsilon=epsilon)

        self.dense1 = keras.layers.Dense(
            units=hidden_units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        self.norm2 = keras.layers.LayerNormalization(epsilon=epsilon)
        self.hidden_act = keras.layers.Activation(self.hidden_activation)

        self.dense2 = keras.layers.Dense(
            units=output_units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            activation=self.activation
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the block.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            training: Boolean indicating whether in training mode

        Returns:
            Output tensor of shape (batch_size, output_units)
        """
        # RBF and first normalization
        x = self.rbf_layer(inputs, training=training)
        x = self.norm1(x, training=training)

        # First dense block
        x = self.dense1(x)
        x = self.norm2(x, training=training)
        x = self.hidden_act(x)

        # Output layer
        return self.dense2(x)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the block."""
        config = super().get_config()
        config.update({
            'rbf_units': self.rbf_units,
            'hidden_units': self.hidden_units,
            'output_units': self.output_units,
            'rbf_initializer':
                keras.initializers.serialize(self.rbf_initializer),
            'kernel_initializer':
                keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer':
                keras.regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation),
            'hidden_activation': keras.activations.serialize(self.hidden_activation),
            'epsilon': self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RBFDenseBlock':
        """Create a block from its config."""
        return cls(**config)
