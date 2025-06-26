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
from keras.api.regularizers import Regularizer
from keras.api.initializers import Initializer
from typing import Callable, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentialFFN(keras.layers.Layer):
    """Differential Feed-Forward Network layer that implements a gating mechanism with separate positive and negative pathways.

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
        hidden_dim: int, dimension of the hidden layer
        output_dim: int, dimension of the output
        branch_activation: Union[str, Callable], activation function used in the branches (default: "gelu")
        gate_activation: Union[str, Callable], activation function to use in the gate (default: "sigmoid")
        dropout_rate: float, dropout rate (default: 0.0)
        kernel_initializer: Union[str, Initializer], initializer for kernel weights (default: 'glorot_uniform')
        kernel_regularizer: Optional[Union[str, Regularizer]], regularizer for kernel weights
                           (default: SoftOrthonormalConstraintRegularizer)
        bias_initializer: Union[str, Initializer], initializer for bias (default: 'zeros')
        use_bias: bool, whether to use bias (default: True)
        name: Optional[str], name for the layer (default: None)
        **kwargs: Additional keyword arguments for the base Layer class
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            branch_activation: Union[str, Callable] = "gelu",
            gate_activation: Union[str, Callable] = "sigmoid",
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[Union[str, Regularizer]] = None,
            bias_initializer: Union[str, Initializer] = 'zeros',
            use_bias: bool = True,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.branch_activation = keras.activations.get(branch_activation)
        self.gate_activation = keras.activations.get(gate_activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer) or SoftOrthonormalConstraintRegularizer()
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.use_bias = use_bias

        # These will be created in build()
        self.positive_dense = None
        self.layer_norm_pos = None
        self.positive_proj = None

        self.negative_dense = None
        self.layer_norm_neg = None
        self.negative_proj = None

        self.layer_norm_diff = None
        self.output_proj = None
        self.dropout = None

        # Store build shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple) -> None:
        """Create the layer's weights and sublayers based on input shape.

        This method instantiates all the sublayers with the appropriate configurations
        when the shape of the input is known.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples,
                indicating the input shape of the layer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create sublayers
        # Positive path: Dense -> Normalization -> Activation -> Projection
        self.positive_dense = keras.layers.Dense(
            self.hidden_dim,
            activation=None,  # No activation here, will be applied after normalization
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
            activation=self.gate_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="positive_proj"
        )

        # Negative path: Dense -> Normalization -> Activation -> Projection
        self.negative_dense = keras.layers.Dense(
            self.hidden_dim,
            activation=None,  # No activation here, will be applied after normalization
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
            activation=self.gate_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="negative_proj"
        )

        # Final normalization and projection
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

        self.dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs:keras.KerasTensor, training: bool = None) -> keras.KerasTensor:
        """Forward pass through the Differential FFN layer.

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
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode (affects dropout behavior)

        Returns:
            Output tensor with shape [..., output_dim]
        """
        # Positive branch: Dense -> Normalization -> Activation -> Projection
        positive_dense = self.positive_dense(inputs, training=training)
        positive_norm = self.layer_norm_pos(positive_dense, training=training)
        positive_act = self.branch_activation(positive_norm)
        positive_gate = self.positive_proj(positive_act, training=training)

        # Negative branch: Dense -> Normalization -> Activation -> Projection
        negative_dense = self.negative_dense(inputs, training=training)
        negative_norm = self.layer_norm_neg(negative_dense, training=training)
        negative_act = self.branch_activation(negative_norm)
        negative_gate = self.negative_proj(negative_act, training=training)

        # Differential and attenuation
        diff = positive_gate - negative_gate
        diff = self.layer_norm_diff(diff, training=training)

        # Apply dropout to the normalized differential features
        diff = self.dropout(diff, training=training)

        # Project to output dimension
        output = self.output_proj(diff, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Output shape tuple, which is the input shape with the last dimension
            replaced by output_dim
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Set the last dimension to output_dim
        output_shape = input_shape_list[:-1] + [self.output_dim]

        # Return as tuple for consistency
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing all parameters needed to instantiate this layer
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get configuration needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])