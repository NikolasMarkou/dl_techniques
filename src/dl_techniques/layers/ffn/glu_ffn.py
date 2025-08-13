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

References:
----------
Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint.
"""

import keras
from typing import Optional, Union, Tuple, Any, Callable
from keras import ops, layers, initializers, regularizers, activations


@keras.saving.register_keras_serializable()
class GLUFFN(keras.layers.Layer):
    """
    Gated Linear Unit Feed Forward Network as described in "GLU Variants Improve Transformer" (Shazeer, 2020).

    This layer implements the GLU mechanism where the input is projected to two parallel paths:
    a gate path and a value path. The gate path is passed through an activation function,
    and then element-wise multiplied with the value path before final projection to output.

    The GLU mechanism provides better control over information flow compared to standard
    feed-forward networks and has been shown to improve performance in Transformer models.

    Mathematical formulation:
        gate = activation(input @ W_gate + b_gate)
        value = input @ W_value + b_value
        hidden = gate ⊙ value  # Element-wise multiplication
        output = hidden @ W_out + b_out

    Where ⊙ denotes element-wise multiplication.

    Args:
        hidden_dim: Integer, dimension of the hidden layer (intermediate dimension).
            Must be positive. This determines the size of both gate and value projections.
        output_dim: Integer, dimension of the output layer. Must be positive.
            This is the final output dimension after the second linear projection.
        activation: Callable or string, activation function to apply to the gate projection.
            Common choices:
            - keras.ops.sigmoid: Original GLU variant
            - keras.ops.relu: ReGLU variant
            - keras.ops.gelu: GEGLU variant
            - keras.ops.silu: SwiGLU variant
            Defaults to keras.ops.sigmoid.
        dropout_rate: Float, dropout rate applied to the hidden representation.
            Must be between 0.0 and 1.0. Defaults to 0.0 (no dropout).
        use_bias: Boolean, whether to use bias terms in linear projections.
            Defaults to True.
        kernel_initializer: String or keras.initializers.Initializer, initializer for
            the kernel weights of all linear projections. Defaults to 'glorot_uniform'.
        bias_initializer: String or keras.initializers.Initializer, initializer for
            the bias weights of all linear projections. Defaults to 'zeros'.
        kernel_regularizer: Optional keras.regularizers.Regularizer, regularizer applied
            to all kernel weights. Defaults to None.
        bias_regularizer: Optional keras.regularizers.Regularizer, regularizer applied
            to all bias weights. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Most common case is 3D input: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        N-D tensor with shape: `(..., output_dim)`.
        For 3D input: `(batch_size, sequence_length, output_dim)`.

    Example:
        ```python
        # Basic GLU (with sigmoid activation)
        layer = GLUFFN(hidden_dim=2048, output_dim=768)

        # GEGLU variant (with GELU activation)
        layer = GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.gelu,
            dropout_rate=0.1
        )

        # SwiGLU variant (with Swish/SiLU activation)
        layer = GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.silu,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # ReGLU variant (with ReLU activation)
        layer = GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.relu,
            use_bias=False
        )

        # In a Transformer model
        inputs = keras.Input(shape=(128, 768))
        x = GLUFFN(hidden_dim=3072, output_dim=768, activation=keras.ops.gelu)(inputs)
        model = keras.Model(inputs, x)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where sub-layers
        are created in __init__ and Keras handles building automatically. This
        ensures proper serialization and eliminates common build errors.

        The different activation functions create different GLU variants:
        - sigmoid: Original GLU
        - relu: ReGLU (often performs well)
        - gelu: GEGLU (excellent for language tasks)
        - silu/swish: SwiGLU (state-of-the-art performance)

    References:
        Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint arXiv:2002.05202.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = keras.ops.sigmoid,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
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

        # Store ALL configuration arguments as instance attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)  # Convert string to callable if needed
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        self.gate_proj = layers.Dense(
            units=hidden_dim,
            activation=None,  # We apply activation manually in call()
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate_proj"
        )

        self.value_proj = layers.Dense(
            units=hidden_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="value_proj"
        )

        self.output_proj = layers.Dense(
            units=output_dim,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_proj"
        )

        # Create dropout layer if needed
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(rate=dropout_rate, name="dropout")
        else:
            self.dropout = None

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the GLU FFN.

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with shape (..., output_dim).
        """
        # Project inputs to gate and value paths
        gate = self.gate_proj(inputs, training=training)
        value = self.value_proj(inputs, training=training)

        # Apply activation to gate (this creates the GLU mechanism)
        if self.activation is not None:
            gate = self.activation(gate)

        # Element-wise multiplication of gate and value (core GLU operation)
        hidden = ops.multiply(gate, value)

        # Apply dropout if configured
        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)

        # Project to output dimension
        output = self.output_proj(hidden, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple. Same as input shape except last dimension
            becomes output_dim.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config


def create_glu_variants() -> dict[str, GLUFFN]:
    """
    Create common GLU variants as described in the paper.

    Returns:
        Dictionary mapping variant names to configured GLUFFN instances.

    Example:
        ```python
        variants = create_glu_variants()

        # Use GEGLU for language modeling
        geglu_layer = variants['geglu']

        # Use SwiGLU for best performance
        swiglu_layer = variants['swiglu']
        ```
    """
    return {
        'glu': GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.sigmoid,
            dropout_rate=0.1
        ),
        'bilinear': GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=None,  # Linear (no activation)
            dropout_rate=0.1
        ),
        'reglu': GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.relu,
            dropout_rate=0.1
        ),
        'geglu': GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.gelu,
            dropout_rate=0.1
        ),
        'swiglu': GLUFFN(
            hidden_dim=2048,
            output_dim=768,
            activation=keras.ops.silu,  # Swish/SiLU activation
            dropout_rate=0.1
        ),
    }