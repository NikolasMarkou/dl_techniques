"""
Neural network layer module implementing the Differential Feed-Forward Network (DifferentialFFN).

This module provides an implementation of a specialized feed-forward network that leverages
a dual-pathway architecture with explicit positive and negative branches. The DifferentialFFN
is inspired by biological neural systems which utilize both excitatory and inhibitory signals
to process information.

The key innovation in this layer is the computation of a differential representation
between positive and negative pathways, allowing the network to model complex relationships
by explicitly separating enhancing and suppressing factors. This approach can be particularly
effective in attention mechanisms, feature discrimination tasks, and scenarios where nuanced
signal processing is required.

Features:
- Dual pathway processing (positive and negative branches)
- Branch normalization before activation for more stable training
- Differential computation with layer normalization
- Customizable hidden and output dimensions
- Support for dropout regularization
- Configurable weight initialization and regularization

Example usage:
```python
# Create a model using DifferentialFFN
inputs = keras.Input(shape=(input_dim,))
x = DifferentialFFN(
    hidden_dim=128,
    output_dim=64,
    dropout_rate=0.1
)(inputs)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)
```

This layer is particularly useful in models requiring:
- Attention mechanisms with fine-grained control
- Complex feature interaction modeling
- Enhanced gradient flow through explicit pathway separation
- Bidirectional signal processing (amplification and attenuation)
"""

import keras
from typing import Callable, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentialFFN(keras.layers.Layer):
    """
    Differential Feed-Forward Network layer that implements a gating mechanism with separate positive and negative pathways.

    This layer creates a specialized feed-forward network architecture with dual pathways:
    1. A positive branch with: Dense → Normalization → Activation → Sigmoid Gate
    2. A negative branch with: Dense → Normalization → Activation → Sigmoid Gate

    The final output is computed as the difference between these gated pathways (positive - negative),
    which is then normalized and projected to the output dimension. This differential architecture
    enables the network to learn more nuanced representations by explicitly modeling positive and
    negative contributions separately, similar to how biological neural systems use excitatory and
    inhibitory signals.

    This approach can be particularly effective for:
    - Capturing subtle differences in feature importance
    - Implementing a form of attention mechanism through the gating
    - Improving gradient flow during backpropagation
    - Enhancing feature discrimination capabilities

    The layer applies layer normalization in several steps to stabilize training:
    1. After each branch's initial dense layer (before activation)
    2. To the differential representation before the final projection

    Args:
        hidden_dim: Integer, dimension of the hidden layer. Must be positive and even
            (since it's split into two halves for positive/negative projections).
        output_dim: Integer, dimension of the output. Must be positive.
        branch_activation: Union[str, Callable], activation function used in the branches.
            Can be string name ('gelu', 'relu') or callable. Defaults to 'gelu'.
        gate_activation: Union[str, Callable], activation function to use in the gate.
            Can be string name ('sigmoid', 'tanh') or callable. Defaults to 'sigmoid'.
        dropout_rate: Float, dropout rate for regularization. Must be between 0 and 1.
            Defaults to 0.0.
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer for
            kernel weights. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
            regularizer for kernel weights. If None, uses SoftOrthonormalConstraintRegularizer.
        bias_initializer: Union[str, keras.initializers.Initializer], initializer for bias.
            Defaults to 'zeros'.
        use_bias: Boolean, whether to use bias in dense layers. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common case is 2D input: `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        For 2D input: `(batch_size, output_dim)`.

    Raises:
        ValueError: If hidden_dim is not positive or not even.
        ValueError: If output_dim is not positive.
        ValueError: If dropout_rate is not between 0 and 1.

    Example:
        ```python
        # Basic usage
        layer = DifferentialFFN(hidden_dim=128, output_dim=64)

        # With custom configuration
        layer = DifferentialFFN(
            hidden_dim=256,
            output_dim=128,
            branch_activation='swish',
            gate_activation='tanh',
            dropout_rate=0.15,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(784,))
        x = DifferentialFFN(256, 128, dropout_rate=0.1)(inputs)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and avoids common build errors.
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            branch_activation: Union[str, Callable] = "gelu",
            gate_activation: Union[str, Callable] = "sigmoid",
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            use_bias: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even (divisible by 2), got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration arguments as instance attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.branch_activation = keras.activations.get(branch_activation)
        self.gate_activation = keras.activations.get(gate_activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = (
            keras.regularizers.get(kernel_regularizer)
            if kernel_regularizer is not None
            else SoftOrthonormalConstraintRegularizer()
        )
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.use_bias = use_bias

        # CREATE all sub-layers here in __init__ (MODERN PATTERN)
        # Positive pathway components
        self.positive_dense = keras.layers.Dense(
            self.hidden_dim,
            activation=None,  # Applied after normalization
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="positive_dense"
        )

        self.layer_norm_pos = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_positive"
        )

        self.positive_proj = keras.layers.Dense(
            self.hidden_dim // 2,
            activation=None,  # Will apply gate_activation in call()
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="positive_proj"
        )

        # Negative pathway components
        self.negative_dense = keras.layers.Dense(
            self.hidden_dim,
            activation=None,  # Applied after normalization
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="negative_dense"
        )

        self.layer_norm_neg = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_negative"
        )

        self.negative_proj = keras.layers.Dense(
            self.hidden_dim // 2,
            activation=None,  # Will apply gate_activation in call()
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="negative_proj"
        )

        # Final processing components
        self.layer_norm_diff = keras.layers.LayerNormalization(
            center=False,
            scale=True,
            name="layer_norm_diff"
        )

        self.output_proj = keras.layers.Dense(
            self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="output_proj"
        )

        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Differential FFN layer.

        The computation follows these steps:
        1. Process inputs through separate positive and negative branches:
           - Dense projection
           - Layer normalization
           - Activation
           - Projection with gate activation
        2. Compute the difference (positive - negative)
        3. Normalize the differential representation
        4. Apply dropout for regularization
        5. Project to output dimension

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Affects dropout and normalization behavior.

        Returns:
            Output tensor with shape (..., output_dim).
        """
        # Positive branch: Dense → Normalization → Activation → Projection → Gate
        positive_dense = self.positive_dense(inputs, training=training)
        positive_norm = self.layer_norm_pos(positive_dense, training=training)
        positive_act = self.branch_activation(positive_norm)
        positive_proj = self.positive_proj(positive_act, training=training)
        positive_gate = self.gate_activation(positive_proj)

        # Negative branch: Dense → Normalization → Activation → Projection → Gate
        negative_dense = self.negative_dense(inputs, training=training)
        negative_norm = self.layer_norm_neg(negative_dense, training=training)
        negative_act = self.branch_activation(negative_norm)
        negative_proj = self.negative_proj(negative_act, training=training)
        negative_gate = self.gate_activation(negative_proj)

        # Compute differential representation (positive - negative)
        diff = positive_gate - negative_gate
        diff = self.layer_norm_diff(diff, training=training)

        # Apply dropout for regularization
        diff = self.dropout(diff, training=training)

        # Project to output dimension
        output = self.output_proj(diff, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple, which is the input shape with the last dimension
            replaced by output_dim.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
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
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'branch_activation': keras.activations.serialize(self.branch_activation),
            'gate_activation': keras.activations.serialize(self.gate_activation),
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'use_bias': self.use_bias,
        })
        return config

    # NOTE: get_build_config() and build_from_config() are REMOVED
    # These are deprecated methods that cause serialization issues in Keras 3
    # The modern pattern handles building automatically