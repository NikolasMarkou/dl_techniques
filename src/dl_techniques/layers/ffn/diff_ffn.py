"""
Neural network layer module implementing the Differential Feed-Forward Network (DifferentialFFN).

This module provides an implementation of a specialized feed-forward network that leverages
a dual-pathway architecture with explicit positive and negative branches. The DifferentialFFN
is inspired by biological neural systems which utilize both excitatory and inhibitory signals
to process information.

The key innovation is the initial splitting of the input tensor into positive and negative
components, which are then processed by separate pathways. The final output is computed
from the difference between these pathways, allowing the network to model complex relationships
by explicitly separating enhancing and suppressing factors. This approach can be particularly
effective in attention mechanisms, feature discrimination, and scenarios where nuanced
signal processing is required.
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

    This layer creates a specialized feed-forward network with dual pathways that process
    positive and negative components of the input separately. The architecture demonstrates
    Pattern 2: Composite Layer from modern Keras 3 patterns, using explicit sub-layer
    building for robust serialization.

    **Intent**: Implement a biologically-inspired dual-pathway architecture where excitatory
    and inhibitory signals are processed separately, enabling more nuanced feature learning
    and improved gradient flow through explicit positive/negative component separation.

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Split: [ReLU(x), ReLU(-x)]
           ↓
    Positive Branch: Dense(hidden_dim) → LayerNorm → Activation → Dense(hidden_dim//2, gate)
           ↓
    Negative Branch: Dense(hidden_dim) → LayerNorm → Activation → Dense(hidden_dim//2, gate)
           ↓
    Differential: pos_branch - neg_branch
           ↓
    LayerNorm → Dropout → Dense(output_dim)
           ↓
    Output(shape=[..., output_dim])
    ```

    **Mathematical Operations**:
    1. **Input Splitting**: pos = ReLU(x), neg = ReLU(-x)
    2. **Branch Processing**: Each branch applies Dense → LayerNorm → Activation → Gate
    3. **Differential Computation**: diff = pos_gated - neg_gated
    4. **Output Projection**: output = Dense(LayerNorm(Dropout(diff)))

    This dual-pathway approach enables explicit modeling of enhancing vs. suppressing
    contributions, improving feature discrimination and gradient flow.

    Args:
        hidden_dim: Integer, dimension of the hidden layer in each branch. Must be positive
            and divisible by 2 for proper gating projection.
        output_dim: Integer, dimension of the output. Must be positive.
        branch_activation: String or callable, activation function used in the branches.
            Accepts standard activation names ('gelu', 'relu', 'swish') or callables.
            Defaults to 'gelu'.
        gate_activation: String or callable, activation function used in the gate projections.
            Typically 'sigmoid' for proper gating behavior. Defaults to 'sigmoid'.
        dropout_rate: Float between 0.0 and 1.0, dropout rate applied to differential features.
            Provides regularization to prevent overfitting. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in dense layers. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer, regularizer for kernel weights.
            If None, uses SoftOrthonormalConstraintRegularizer for stability.
        bias_regularizer: Optional Regularizer, regularizer for bias weights.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common: 2D tensor with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        Same rank as input, last dimension becomes `output_dim`.

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
        inputs = keras.Input(shape=(512,))
        outputs = layer(inputs)  # Shape: (batch, 64)

        # Advanced configuration with custom activations
        layer = DifferentialFFN(
            hidden_dim=256,
            output_dim=128,
            branch_activation='swish',
            gate_activation='sigmoid',
            dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer-style model
        inputs = keras.Input(shape=(seq_len, embed_dim))
        x = DifferentialFFN(
            hidden_dim=embed_dim * 4,
            output_dim=embed_dim,
            branch_activation='gelu',
            dropout_rate=0.1
        )(inputs)
        model = keras.Model(inputs, x)
        ```

    Raises:
        ValueError: If `hidden_dim` is not positive or not divisible by 2.
        ValueError: If `output_dim` is not positive.
        ValueError: If `dropout_rate` is not between 0.0 and 1.0.

    Note:
        This implementation follows Pattern 2 (Composite Layer) from modern Keras 3 patterns.
        The `hidden_dim` must be divisible by 2 because each branch's gating projection
        maps to `hidden_dim // 2` dimensions, ensuring the final differential features
        have consistent dimensionality.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        branch_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        gate_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "sigmoid",
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

        # CREATE all sub-layers in __init__ (Pattern 2: Composite Layer)
        # Following modern Keras 3 pattern - create but don't build here

        # Positive branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.positive_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
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
            activation=None,  # Activation applied separately for clarity
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
            activation=None,
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
            activation=None,  # Activation applied separately for clarity
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
        This follows Pattern 2: Composite Layer from modern Keras 3 patterns.
        """
        # Build positive branch sub-layers in computational order
        self.positive_dense.build(input_shape)
        dense_output_shape = self.positive_dense.compute_output_shape(input_shape)
        self.layer_norm_pos.build(dense_output_shape)
        self.positive_proj.build(dense_output_shape)

        # Build negative branch sub-layers
        self.negative_dense.build(input_shape)
        # Note: negative branch has same architecture as positive
        self.layer_norm_neg.build(dense_output_shape)
        self.negative_proj.build(dense_output_shape)

        # Build differential processing layers
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

        Args:
            inputs: Input tensor with shape (..., input_dim).
            training: Boolean indicating whether the layer should behave in
                training mode (applies dropout) or inference mode.

        Returns:
            Output tensor with shape (..., output_dim).
        """
        # Split input into positive and negative components
        inputs_positive = keras.ops.relu(inputs)
        inputs_negative = keras.ops.relu(-inputs)

        # Positive branch processing
        pos_hidden = self.positive_dense(inputs_positive)
        pos_normed = self.layer_norm_pos(pos_hidden, training=training)
        pos_activated = self.branch_activation(pos_normed)
        pos_projected = self.positive_proj(pos_activated)
        pos_gated = self.gate_activation(pos_projected)

        # Negative branch processing
        neg_hidden = self.negative_dense(inputs_negative)
        neg_normed = self.layer_norm_neg(neg_hidden, training=training)
        neg_activated = self.branch_activation(neg_normed)
        neg_projected = self.negative_proj(neg_activated)
        neg_gated = self.gate_activation(neg_projected)

        # Compute differential representation
        differential = pos_gated - neg_gated

        # Process differential features
        diff_normed = self.layer_norm_diff(differential, training=training)
        diff_dropped = self.dropout(diff_normed, training=training)

        # Final projection to output dimension
        output = self.output_proj(diff_dropped)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple with last dimension as output_dim.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

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
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
