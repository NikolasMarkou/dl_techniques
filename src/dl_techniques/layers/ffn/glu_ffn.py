"""
GLU Feed Forward Network Implementation
======================================

This module implements the Gated Linear Unit (GLU) Feed Forward Network variants as described
in the paper "GLU Variants Improve Transformer" by Noam Shazeer (2020).

This implementation provides a Dense-based implementation (GLUFFN) that follows the
architecture described in the paper:
- Input is projected twice to create gate and value paths
- The gate projection is passed through a customizable activation function
- Gate and value are multiplied element-wise
- The result is projected to the output dimension

References:
----------
Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint.
"""

import keras
import tensorflow as tf
from keras import layers, initializers, regularizers, activations
from typing import Callable, Optional, Union, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GLUFFN(keras.layers.Layer):
    """
    Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer" (Shazeer, 2020).

    This implementation uses Dense layers and supports various GLU variants through different activation functions.

    Args:
        hidden_dim: int, dimension of the hidden layer.
        output_dim: int, dimension of the output.
        activation: str or Callable, activation function to use in the gate.
            Defaults to 'sigmoid'.
        dropout_rate: float, dropout rate. Defaults to 0.0.
        kernel_initializer: str or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional Regularizer for kernel weights. Defaults to None.
        bias_initializer: str or Initializer, initializer for bias. Defaults to 'zeros'.
        use_bias: bool, whether to use bias. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: Union[str, Callable] = 'sigmoid',
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            use_bias: bool = True,
            **kwargs
    ):
        """Initialize the GLU FFN layer."""
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # FIX: Resolve activation string to a callable function immediately upon initialization.
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = use_bias

        # Create all sub-layers in __init__
        self.gate_proj = layers.Dense(
            hidden_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="gate_proj"
        )

        self.value_proj = layers.Dense(
            hidden_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="value_proj"
        )

        self.output_proj = layers.Dense(
            output_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="output_proj"
        )

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass for the GLU FFN."""
        gate = self.gate_proj(inputs)
        value = self.value_proj(inputs)

        # This now correctly calls the resolved activation function
        gate = self.activation(gate)

        hidden = gate * value
        hidden = self.dropout(hidden, training=training)
        output = self.output_proj(hidden)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            # FIX: Serialize the activation function so it can be saved as a string.
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'use_bias': self.use_bias,
        })
        return config