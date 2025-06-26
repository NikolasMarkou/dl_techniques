"""
Neural network layer module implementing the Differential Feed-Forward Network (DifferentialFFN).

This module provides an implementation of a specialized feed-forward network that leverages
a dual-pathway architecture with explicit positive and negative branches. The DifferentialFFN
is inspired by biological neural systems which utilize both excitatory and inhibitory signals
to process information.
"""

import keras
from keras import ops
from typing import Callable, Optional, Union, Tuple, Dict, Any

# Type aliases for clearer type hints
TensorLike = Union[keras.KerasTensor, ops.Tensor]


@keras.saving.register_keras_serializable()
class DifferentialFFN(keras.layers.Layer):
    """Differential Feed-Forward Network layer that implements a gating mechanism with separate positive and negative pathways.

    This layer creates a specialized feed-forward network architecture with dual pathways:
    1. A positive branch with ReLU activation followed by a sigmoid gate
    2. A negative branch with ReLU activation followed by a sigmoid gate

    The final output is computed as the difference between these gated pathways (positive - negative),
    which is then normalized and projected to the output dimension. This differential architecture
    enables the network to learn more nuanced representations by explicitly modeling positive and
    negative contributions separately, similar to how biological neural systems use excitatory and
    inhibitory signals.

    Features:
    - Dual pathway processing (positive and negative branches)
    - Differential computation with layer normalization
    - Customizable hidden and output dimensions
    - Support for dropout regularization
    - Configurable weight initialization and regularization
    - Optional weight sharing between feature branches (feature extraction kernels and biases when
      use_bias=True are tied, while gate projections remain independent to ensure meaningful
      differential computation)
    - Option to disable differential processing
    - Optional return of differential representation
    - Customizable gate temperature and activation function

    This approach can be particularly effective for:
    - Capturing subtle differences in feature importance
    - Implementing a form of attention mechanism through the gating
    - Improving gradient flow during backpropagation
    - Enhancing feature discrimination capabilities

    Args:
        hidden_dim: int, dimension of the hidden layer
        output_dim: int, dimension of the output
        activation: Union[str, Callable], activation function to use in the gate (default: "sigmoid")
        dropout_rate: float, dropout rate using Keras convention (fraction to drop, default: 0.0)
        kernel_initializer: Union[str, keras.initializers.Initializer], initializer for kernel weights (default: 'glorot_uniform')
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]], regularizer for kernel weights (default: None)
        bias_initializer: Union[str, keras.initializers.Initializer], initializer for bias (default: 'zeros')
        use_bias: bool, whether to use bias (default: True)
        share_weights: bool, whether to share weights between positive and negative branches (default: False).
                       Only the feature extraction kernels are shared, not the gate projection weights.
        use_diff: bool, whether to use differential processing (if False, only positive branch is used) (default: True)
        return_diff: bool, whether to return differential representation along with output (default: False)
        gate_temperature: float, temperature parameter for gate activation (lower values make gates sharper, default: 1.0)
                          must be positive.
        gate_activation: Union[str, Callable], activation function to use for gating (default: "sigmoid").
                         Typically sigmoid-like functions (sigmoid, tanh) are most appropriate for gating.
        name: Optional[str], name for the layer (default: None)
        **kwargs: Additional keyword arguments for the base Layer class
    """

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            activation: Union[str, Callable] = "relu",
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            use_bias: bool = True,
            share_weights: bool = False,
            use_diff: bool = True,
            return_diff: bool = False,
            gate_temperature: float = 1.0,
            gate_activation: Union[str, Callable] = "sigmoid",
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Validate parameters
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if gate_temperature <= 0:
            raise ValueError(f"gate_temperature must be positive, got {gate_temperature}")

        # Check activation serializability
        if callable(activation) and not isinstance(activation, str) and not hasattr(activation, '__name__'):
            raise ValueError("Activation must be a string, a named function, or a Keras-serializable callable")
        if callable(gate_activation) and not isinstance(gate_activation, str) and not hasattr(gate_activation,
                                                                                              '__name__'):
            raise ValueError("Gate activation must be a string, a named function, or a Keras-serializable callable")

        # Set supports_masking to True to properly handle masks
        self.supports_masking = True

        # Store configuration parameters
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.share_weights = share_weights
        self.use_diff = use_diff
        self.return_diff = return_diff
        self.gate_temperature = gate_temperature

        # Process activations
        self._activation_name = None
        if callable(activation) and hasattr(activation, '__name__'):
            self._activation_name = activation.__name__
            self.activation = activation
        else:
            self._activation_name = activation if isinstance(activation, str) else None
            self.activation = keras.activations.get(activation)

        self._gate_activation_name = None
        if callable(gate_activation) and hasattr(gate_activation, '__name__'):
            self._gate_activation_name = gate_activation.__name__
            self.gate_activation = gate_activation
        else:
            self._gate_activation_name = gate_activation if isinstance(gate_activation, str) else None
            self.gate_activation = keras.activations.get(gate_activation)

        # Check if gate activation is sigmoidal
        sigmoidal_gates = ["sigmoid", "hard_sigmoid", "tanh", "swish", "silu"]
        if (isinstance(gate_activation, str) and gate_activation not in sigmoidal_gates) or \
                (self._gate_activation_name and self._gate_activation_name not in sigmoidal_gates):
            print("Warning: Non-sigmoidal gate activation may affect the interpretability "
                  "of the differential representation. Recommended: sigmoid, tanh.")

        # Process initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # These will be created in build()
        self.positive_branch = None
        self.positive_proj = None
        self.negative_branch = None
        self.negative_proj = None
        self.layer_norm = None
        self.output_proj = None
        self.dropout = None

        # Store build shape for serialization
        self._build_input_shape = None

        # Check for return_diff with use_diff=False during initialization
        if self.return_diff and not self.use_diff:
            raise ValueError("return_diff=True requires use_diff=True")

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

        # Create positive branch
        self.positive_branch = keras.layers.Dense(
            self.hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="positive_branch"
        )

        self.positive_proj = keras.layers.Dense(
            self.hidden_dim,
            activation=None,  # We'll apply gate activation with temperature manually
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            name="positive_proj"
        )

        # Create negative branch if differential is supported
        if self.use_diff:
            # Always create separate layer instances for negative branch
            self.negative_branch = keras.layers.Dense(
                self.hidden_dim,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_initializer=self.bias_initializer,
                name="negative_branch"
            )

            self.negative_proj = keras.layers.Dense(
                self.hidden_dim,
                activation=None,  # We'll apply gate activation with temperature manually
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_initializer=self.bias_initializer,
                name="negative_proj"
            )

        # Layer normalization for feature representation
        self.layer_norm = keras.layers.LayerNormalization(
            center=False,
            scale=True,
            name="layer_norm"
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

        # Build sublayers with input shape
        self.positive_branch.build(input_shape)

        # Calculate intermediate shapes for subsequent layers
        pos_branch_output_shape = list(input_shape)
        pos_branch_output_shape[-1] = self.hidden_dim
        pos_branch_output_shape = tuple(pos_branch_output_shape)

        self.positive_proj.build(pos_branch_output_shape)

        if self.use_diff:
            self.negative_branch.build(input_shape)
            self.negative_proj.build(pos_branch_output_shape)

            # If weight sharing is enabled, tie only the feature extraction weights
            if self.share_weights:
                # Tie only the feature branch kernels, not the projection kernels
                self._tie_feature_weights()

        self.layer_norm.build(pos_branch_output_shape)
        self.dropout.build(pos_branch_output_shape)
        self.output_proj.build(pos_branch_output_shape)

        # Call super().build() after setting all attributes
        super().build(input_shape)

    def _tie_feature_weights(self) -> None:
        """Tie weights between positive and negative feature branches when share_weights is True.

        This creates a weight sharing relationship for feature extraction only,
        while keeping gate projections independent to ensure meaningful differential computation.
        """
        if not self.use_diff or not self.share_weights:
            return

        # Store original references for cleanup
        old_kernel = self.negative_branch.kernel
        old_bias = self.negative_branch.bias if self.use_bias else None

        # Tie only the feature branch kernels
        self.negative_branch.kernel = self.positive_branch.kernel

        # Tie the feature branch bias if used
        if self.use_bias:
            self.negative_branch.bias = self.positive_branch.bias

        # Clean up trainable_weights list to avoid duplicates
        for weight_list in (self.negative_branch._trainable_weights,
                            self.negative_branch._non_trainable_weights):
            # Remove old kernel and bias from weight lists
            weight_list[:] = [w for w in weight_list
                              if w is not old_kernel and (old_bias is None or w is not old_bias)]

    def call(self, inputs: TensorLike, training: Optional[bool] = None, mask: Optional[TensorLike] = None) -> Union[
        TensorLike, Tuple[TensorLike, TensorLike]]:
        """Forward pass through the Differential FFN layer.

        The computation follows these steps:
        1. Process inputs through separate positive and negative branches
        2. Apply gate activation to both branches with temperature scaling
        3. Compute the difference (positive - negative) if differential is enabled
        4. Normalize the representation
        5. Apply dropout for regularization
        6. Project to output dimension

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode (affects dropout behavior)
            mask: Input mask tensor (will be properly propagated)

        Returns:
            If return_diff=False: Output tensor with shape [..., output_dim]
            If return_diff=True: Tuple of (output tensor, normalized feature tensor)
        """
        positive_branch = self.positive_branch(inputs)
        positive_proj = self.positive_proj(positive_branch)
        # Apply gate activation with temperature scaling
        positive_gate = self.gate_activation(positive_proj / self.gate_temperature)

        if self.use_diff:
            negative_branch = self.negative_branch(inputs)
            negative_proj = self.negative_proj(negative_branch)
            # Apply gate activation with temperature scaling
            negative_gate = self.gate_activation(negative_proj / self.gate_temperature)
            # Compute differential representation
            features = positive_gate - negative_gate
        else:
            # Use only positive gate when differential is disabled
            features = positive_gate

        # Apply layer normalization
        normed_features = self.layer_norm(features)

        # Apply dropout to the normalized features
        dropout_features = self.dropout(normed_features, training=training)

        # Project to output dimension
        output = self.output_proj(dropout_features)

        if self.return_diff:
            return output, normed_features
        return output

    def compute_mask(self, inputs: TensorLike, mask: Optional[TensorLike] = None) -> Optional[TensorLike]:
        """Computes an output mask tensor.

        Args:
            inputs: Input tensor
            mask: Input mask tensor

        Returns:
            Output mask tensor (same as input mask)
        """
        return mask

    def compute_output_shape(self, input_shape: Tuple) -> Union[Tuple, Tuple[Tuple, Tuple]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            If return_diff=False: Output shape tuple with the last dimension as output_dim
            If return_diff=True and use_diff=True: Tuple of (output shape, differential shape)
            If return_diff=True but use_diff=False: Just output shape (no differential is computed)
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Set the last dimension to output_dim for the main output
        output_shape = tuple(input_shape_list[:-1] + [self.output_dim])

        # Only return the differential shape if both return_diff and use_diff are True
        if self.return_diff and self.use_diff:
            # Set the last dimension to hidden_dim for the differential output
            diff_shape = tuple(input_shape_list[:-1] + [self.hidden_dim])
            return output_shape, diff_shape

        return output_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing all parameters needed to instantiate this layer
        """
        config = super().get_config()

        # Handle the activation serialization
        activation = self._activation_name
        if activation is None and callable(self.activation):
            # If activation is a callable with __name__, use that
            if hasattr(self.activation, '__name__'):
                activation = self.activation.__name__
            else:
                # This should never happen due to validation in __init__
                activation = "unknown"

        # Handle gate activation serialization
        gate_activation = self._gate_activation_name
        if gate_activation is None and callable(self.gate_activation):
            if hasattr(self.gate_activation, '__name__'):
                gate_activation = self.gate_activation.__name__
            else:
                # This should never happen due to validation in __init__
                gate_activation = "unknown"

        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'activation': activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'use_bias': self.use_bias,
            'share_weights': self.share_weights,
            'use_diff': self.use_diff,
            'return_diff': self.return_diff,
            'gate_temperature': self.gate_temperature,
            'gate_activation': gate_activation,
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

    @classmethod
    def from_attention(cls, query: TensorLike, key: TensorLike, value: TensorLike, **kwargs) -> 'DifferentialFFN':
        """Helper method to create a DifferentialFFN layer with dimensions matching attention components.

        Creates a layer with hidden_dim=4*query_dim and output_dim=value_dim by default,
        but explicit kwargs for these parameters will override the defaults.

        Args:
            query: Query tensor from attention mechanism
            key: Key tensor from attention mechanism
            value: Value tensor from attention mechanism
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            Configured DifferentialFFN layer with dimensions matching the attention components
        """
        # Get the last dimension of each tensor
        query_dim = query.shape[-1]
        value_dim = value.shape[-1]

        # Create default parameters that can be overridden by kwargs
        params = {
            'hidden_dim': query_dim * 4,  # Common practice to use 4x the query dim for FFN
            'output_dim': value_dim,  # Match output dimension to value dimension
        }

        # Explicit kwargs override the defaults
        params.update(kwargs)

        # Create the layer with appropriate dimensions
        return cls(**params)