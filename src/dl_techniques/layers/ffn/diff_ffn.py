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
    Differential Feed-Forward Network layer implementing dual-pathway processing.

    This layer creates a specialized feed-forward network architecture with dual pathways
    that process information through separate positive and negative branches. Each branch
    consists of: Dense → LayerNormalization → Activation → Dense(gate). The final output
    is computed as the difference between these gated pathways (positive - negative),
    which is then normalized and projected to the output dimension.

    This differential architecture enables the network to learn nuanced representations
    by explicitly modeling positive and negative contributions separately, similar to how
    biological neural systems use excitatory and inhibitory signals.

    Mathematical formulation:
        pos_branch = gate_activation(Dense(branch_activation(LayerNorm(Dense(x)))))
        neg_branch = gate_activation(Dense(branch_activation(LayerNorm(Dense(x)))))
        diff = pos_branch - neg_branch
        output = Dense(Dropout(LayerNorm(diff)))

    This approach can be particularly effective for:
    - Capturing subtle differences in feature importance
    - Implementing attention-like gating mechanisms
    - Improving gradient flow during backpropagation
    - Enhancing feature discrimination capabilities

    Args:
        hidden_dim: Integer, dimension of the hidden layer in each branch. Must be positive
            and divisible by 2 for proper gating projection.
        output_dim: Integer, dimension of the output. Must be positive.
        branch_activation: String or callable, activation function used in the branches.
            Accepts standard activation names ('gelu', 'relu', 'swish') or callables.
            Defaults to 'gelu'.
        gate_activation: String or callable, activation function used in the gate projections.
            Typically 'sigmoid' for proper gating behavior. Defaults to 'sigmoid'.
        dropout_rate: Float, dropout rate applied to the differential features.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in dense layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer for kernel weights.
            If None, uses SoftOrthonormalConstraintRegularizer for better training stability.
        bias_regularizer: Optional Regularizer, regularizer for bias weights.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tensor with shape: `(batch_size, ..., input_dim)`
        Where input_dim is the last dimension of the input tensor.

    Output shape:
        Tensor with shape: `(batch_size, ..., output_dim)`
        Same as input shape except the last dimension is replaced by output_dim.

    Attributes:
        positive_dense: Dense layer for positive branch initial projection.
        layer_norm_pos: LayerNormalization for positive branch.
        positive_proj: Dense layer for positive branch gating.
        negative_dense: Dense layer for negative branch initial projection.
        layer_norm_neg: LayerNormalization for negative branch.
        negative_proj: Dense layer for negative branch gating.
        layer_norm_diff: LayerNormalization for differential features.
        dropout: Dropout layer for regularization.
        output_proj: Final dense projection to output dimension.

    Example:
        ```python
        # Basic usage
        layer = DifferentialFFN(hidden_dim=128, output_dim=64)

        # Advanced configuration
        layer = DifferentialFFN(
            hidden_dim=256,
            output_dim=128,
            branch_activation='swish',
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(512,))
        x = DifferentialFFN(hidden_dim=256, output_dim=128)(inputs)
        outputs = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If hidden_dim is not positive or not divisible by 2.
        ValueError: If output_dim is not positive.
        ValueError: If dropout_rate is not between 0.0 and 1.0.

    Note:
        The hidden_dim should be divisible by 2 as each branch projects to hidden_dim // 2
        before the differential computation. This ensures balanced pathway processing.

    References:
        - Inspired by biological excitatory/inhibitory neural processing
        - Related to differential attention mechanisms in modern transformers
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        branch_activation: Union[str, Callable] = "gelu",
        gate_activation: Union[str, Callable] = "sigmoid",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be divisible by 2, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.branch_activation = keras.activations.get(branch_activation)
        self.gate_activation = keras.activations.get(gate_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Handle regularizer - use default if None provided
        if kernel_regularizer is None:
            self.kernel_regularizer = SoftOrthonormalConstraintRegularizer()
        else:
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # Positive branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.positive_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,  # Applied after normalization
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="positive_dense"
        )

        self.layer_norm_pos = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_positive"
        )

        self.positive_proj = keras.layers.Dense(
            units=self.hidden_dim // 2,
            activation=self.gate_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="positive_proj"
        )

        # Negative branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.negative_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,  # Applied after normalization
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="negative_dense"
        )

        self.layer_norm_neg = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_negative"
        )

        self.negative_proj = keras.layers.Dense(
            units=self.hidden_dim // 2,
            activation=self.gate_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="negative_proj"
        )

        # Differential processing layers
        self.layer_norm_diff = keras.layers.LayerNormalization(
            center=False,  # No centering for differential features
            scale=True,
            name="layer_norm_diff"
        )

        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )

        self.output_proj = keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_proj"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This follows Pattern 2: Composite Layer from the modern Keras 3 guide.
        """
        # Build sub-layers in computational order to ensure proper weight creation

        # Positive branch layers
        self.positive_dense.build(input_shape)

        # Compute intermediate shape after positive_dense
        dense_output_shape = self.positive_dense.compute_output_shape(input_shape)

        self.layer_norm_pos.build(dense_output_shape)

        # After normalization and activation, shape is preserved
        self.positive_proj.build(dense_output_shape)

        # Negative branch layers (same shapes as positive)
        self.negative_dense.build(input_shape)
        self.layer_norm_neg.build(dense_output_shape)
        self.negative_proj.build(dense_output_shape)

        # Differential processing layers
        # After projection, shape is (batch, ..., hidden_dim // 2)
        proj_output_shape = self.positive_proj.compute_output_shape(dense_output_shape)

        self.layer_norm_diff.build(proj_output_shape)
        self.dropout.build(proj_output_shape)
        self.output_proj.build(proj_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Differential FFN layer.

        The computation follows these steps:
        1. Process inputs through separate positive and negative branches:
           - Dense projection to hidden_dim
           - Layer normalization
           - Branch activation
           - Dense projection to hidden_dim // 2 with gate activation
        2. Compute the difference (positive - negative)
        3. Normalize the differential representation
        4. Apply dropout for regularization
        5. Project to output dimension

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode (applies dropout) or inference mode. If None,
                uses the learning phase.

        Returns:
            Output tensor with shape (..., output_dim).
        """
        # Positive branch: Dense -> LayerNorm -> Activation -> Gate
        pos_hidden = self.positive_dense(inputs, training=training)
        pos_normed = self.layer_norm_pos(pos_hidden, training=training)
        pos_activated = self.branch_activation(pos_normed)
        pos_gated = self.positive_proj(pos_activated, training=training)

        # Negative branch: Dense -> LayerNorm -> Activation -> Gate
        neg_hidden = self.negative_dense(inputs, training=training)
        neg_normed = self.layer_norm_neg(neg_hidden, training=training)
        neg_activated = self.branch_activation(neg_normed)
        neg_gated = self.negative_proj(neg_activated, training=training)

        # Compute differential representation
        differential = pos_gated - neg_gated

        # Normalize differential features
        diff_normed = self.layer_norm_diff(differential, training=training)

        # Apply regularization
        diff_dropped = self.dropout(diff_normed, training=training)

        # Final projection to output dimension
        output = self.output_proj(diff_dropped, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as input shape except the last dimension
            is replaced by output_dim.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all parameters needed to reconstruct this layer.
            Includes all __init__ parameters with proper serialization of complex objects.
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'branch_activation': keras.activations.serialize(self.branch_activation),
            'gate_activation': keras.activations.serialize(self.gate_activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config