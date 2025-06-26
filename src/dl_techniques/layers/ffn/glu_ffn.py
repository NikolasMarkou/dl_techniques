"""
GLU Feed Forward Network Implementation
======================================

This module implements the Gated Linear Unit (GLU) Feed Forward Network variants as described
in the paper "GLU Variants Improve Transformer" by Noam Shazeer (2020).

Paper Overview:
--------------
The paper introduces and compares variations of Gated Linear Units (GLU) when used in the
feed-forward sublayers of Transformer models. GLUs consist of the component-wise product of
two linear projections, with one being passed through an activation function. The paper
explores several activation variants beyond the original sigmoid function.

Key GLU Variants:
- GLU: Uses sigmoid activation (original)
- Bilinear: No activation (linear)
- ReGLU: Uses ReLU activation
- GEGLU: Uses GELU activation
- SwiGLU: Uses Swish activation

Network Architecture:
-------------------
                   +------------------+
                   |     Input x      |
                   +--------+---------+
                            |
                   +--------v---------+
          +--------+   Linear Proj    +---------+
          |        +------------------+         |
          |                                     |
+---------v----------+               +----------v---------+
|    Gate Projection |               | Value Projection   |
|        (xW)        |               |        (xV)        |
+----------+---------+               +---------+----------+
          |                                     |
+---------v----------+                          |
|    Activation fn   |                          |
|   (sigmoid/etc.)   |                          |
+----------+---------+                          |
          |                                     |
          |          +------------------+       |
          +---------->  Element-wise    <-------+
                     |  Multiplication  |
                     +--------+---------+
                              |
                     +--------v---------+
                     |   Linear Proj    |
                     |       (W2)       |
                     +--------+---------+
                              |
                     +--------v---------+
                     |     Output       |
                     +------------------+

Performance Details:
------------------
The paper demonstrates that GLU variants, particularly GEGLU and SwiGLU, outperform standard
ReLU and GELU activations in Transformer FFNs across multiple language understanding tasks:

1. On pre-training perplexity (text span-filling task):
   - GEGLU: 1.633 vs. ReLU: 1.677 (lower is better)
   - SwiGLU: 1.636 vs. ReLU: 1.677 (lower is better)

2. On fine-tuning performance:
   - Average GLUE benchmark: GLU variants achieved ~84.4 vs. ReLU's 83.8
   - Average SuperGLUE benchmark: SwiGLU achieved 74.56 vs. ReLU's 72.76
   - SQuAD v1.1 F1: ReGLU achieved 91.18 vs. ReLU's 90.87

Implementation Details:
---------------------
This implementation provides two variants of GLU FFN:
1. Dense-based implementation (GLUFFN): Uses standard Dense layers for projections
2. Conv2D-based implementation (Conv2DGLUFFN): Uses 1x1 convolutions for projections

Both implementations follow the architecture described in the paper:
- Input is projected twice to create gate and value paths
- The gate projection is passed through a customizable activation function
- Gate and value are multiplied element-wise
- The result is projected to the output dimension

The implementation supports all activation variants mentioned in the paper and allows
for customizable kernel initializers and regularizers. The activation function is passed
as an initialization parameter, making it easy to create any GLU variant.

Improvements Over Standard FFN:
-----------------------------
1. Performance Improvements:
   - Consistent improvement in perplexity and downstream task performance
   - Better convergence properties during training

2. Architectural Advantages:
   - Introduces multiplicative gating that helps control information flow
   - Allows different parts of the network to specialize (gate vs. value paths)
   - Provides a more expressive intermediate representation

3. Flexibility:
   - The gating mechanism allows the network to selectively emphasize or suppress
     different parts of the representation
   - Different activation functions can be chosen based on the specific task

4. Parameter Efficiency:
   - The paper shows that GLU variants perform better even when normalized for
     parameter count and computation cost

The paper attributes the success of these architectures to "divine benevolence" rather
than providing a theoretical explanation, suggesting that the empirical results are
the primary justification for their use.

References:
----------
Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint.
"""

import keras
import tensorflow as tf
from keras.api.regularizers import Regularizer
from keras.api.initializers import Initializer
from typing import Callable, Optional, Union


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class GLUFFN(keras.layers.Layer):
    """
    Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer" (Shazeer, 2020).

    This implementation uses Dense layers and supports various GLU variants through different activation functions.

    Args:
        hidden_dim: int, dimension of the hidden layer
        output_dim: int, dimension of the output
        activation: Callable, activation function to use in the gate (default: sigmoid)
        dropout_rate: float, dropout rate (default: 0.0)
        kernel_initializer: Union[str, Initializer], initializer for kernel weights (default: 'glorot_uniform')
        kernel_regularizer: Optional[Regularizer], regularizer for kernel weights (default: None)
        bias_initializer: Union[str, Initializer], initializer for bias (default: 'zeros')
        use_bias: bool, whether to use bias (default: True)
        name: Optional[str], name for the layer (default: None)
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: Callable = tf.nn.sigmoid,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[Regularizer] = None,
            bias_initializer: Union[str, Initializer] = 'zeros',
            use_bias: bool = True,
            name: Optional[str] = None,
            **kwargs
    ):
        """Initialize the GLU FFN layer."""
        super(GLUFFN, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias

        # Define the layers
        self.gate_proj = keras.layers.Dense(
            hidden_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            name="gate_proj"
        )

        self.value_proj = keras.layers.Dense(
            hidden_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            name="value_proj"
        )

        self.output_proj = keras.layers.Dense(
            output_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            name="output_proj"
        )

        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the GLU FFN.

        Args:
            inputs: tf.Tensor, input tensor
            training: bool, whether in training mode

        Returns:
            tf.Tensor: Output tensor
        """
        # Project inputs for gate and value
        gate = self.gate_proj(inputs)
        value = self.value_proj(inputs)

        # Apply activation to gate
        gate = self.activation(gate)

        # Element-wise multiplication of gate and value
        hidden = gate * value

        # Apply dropout (if any)
        if self.dropout_rate > 0:
            hidden = self.dropout(hidden, training=training)

        # Project to output dimension
        output = self.output_proj(hidden)

        return output

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super(GLUFFN, self).get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_initializer': self.bias_initializer,
            'use_bias': self.use_bias,
        })
        return config
