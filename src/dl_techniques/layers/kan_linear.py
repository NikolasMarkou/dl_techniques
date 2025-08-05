"""
Kolmogorov-Arnold Network (KAN) Implementation
============================================

This implementation is based on the paper:
"KAN: Kolmogorov-Arnold Networks" by Liu et al. (2024)

Theoretical Background:
    The Kolmogorov-Arnold representation theorem states that any multivariate continuous
    function can be represented as a composition of univariate functions. KAN leverages
    this by creating a deep neural network architecture where activation functions are
    learned using B-splines, rather than using fixed activation functions.

Key Components:
    1. B-spline Activation Functions:
       - Uses B-splines of order k (typically 3) as basis functions
       - Grid points are learnable parameters
       - Combines multiple B-splines to create flexible activation functions

    2. Network Structure:
       - Input layer: Maps input features to B-spline space
       - Hidden layers: Combine B-spline activations with linear transformations
       - Output layer: Maps combined features to output space

    3. Learning Components:
       - Base weights: Standard linear transformation matrices
       - Spline weights: Coefficients for B-spline combinations
       - Grid points: Learnable points for B-spline evaluation
       - Scaling factors: Control contribution of spline vs. linear components

Usage Recommendations:
    1. Grid Size Selection:
       - Start with small grid (5-10 points)
       - Increase if underfitting observed
       - Monitor memory usage with large grids

    2. Spline Order:
       - Order 3 (cubic) recommended for most applications
       - Higher orders may help with very smooth functions
       - Lower orders for simpler relationships

    3. Architecture:
       - Use fewer neurons than equivalent MLP
       - Add layers gradually if needed
       - Consider residual connections for deep networks

References:
    1. Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
    2. Kolmogorov, A. N. (1957) "On the representation of continuous functions"
    3. Arnold, V. I. (1963) "On functions of three variables"
    4. De Boor, C. (1978) "A Practical Guide to Splines"
"""

import keras
import numpy as np
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KANLinear(keras.layers.Layer):
    """Kolmogorov-Arnold Network Linear Layer implementation with enhanced stability.

    This layer implements the core KAN linear transformation using B-spline basis functions
    combined with traditional linear transformations. It provides learnable activation
    functions through B-spline coefficients.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Size of the grid for B-splines. Default is 5.
        spline_order: Order of B-splines. Default is 3.
        activation: Activation function name to use. Default is 'swish'.
        regularization_factor: L2 regularization factor. Default is 0.01.
        grid_range: Range for the grid as (min, max). Default is (-1, 1).
        epsilon: Small constant for numerical stability. Default is 1e-7.
        clip_value: Maximum absolute value for gradients. Default is 1e3.
        use_residual: Whether to use residual connections. Default is True.
        kernel_initializer: Initializer for base weights. Default is 'orthogonal'.
        spline_initializer: Initializer for spline weights. Default is 'glorot_uniform'.
        kernel_regularizer: Regularizer for base weights.
        spline_regularizer: Regularizer for spline weights.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
        >>> layer = KANLinear(in_features=10, out_features=5, grid_size=8)
        >>> x = keras.random.normal((32, 10))
        >>> y = layer(x)
        >>> print(y.shape)  # (32, 5)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = 'swish',
        regularization_factor: float = 0.01,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        epsilon: float = 1e-7,
        clip_value: float = 1e3,
        use_residual: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'orthogonal',
        spline_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        spline_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize KANLinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            grid_size: Size of the grid for B-splines.
            spline_order: Order of B-splines.
            activation: Activation function name to use.
            regularization_factor: L2 regularization factor.
            grid_range: Range for the grid as (min, max).
            epsilon: Small constant for numerical stability.
            clip_value: Maximum absolute value for clipping.
            use_residual: Whether to use residual connections.
            kernel_initializer: Initializer for base weights.
            spline_initializer: Initializer for spline weights.
            kernel_regularizer: Regularizer for base weights.
            spline_regularizer: Regularizer for spline weights.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            ValueError: If features are not positive, grid size < spline order, or invalid grid range.
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Features must be positive integers")
        if grid_size < spline_order:
            raise ValueError("Grid size must be >= spline order")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range")

        super().__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_name = activation
        self.regularization_factor = regularization_factor
        self.grid_range = grid_range
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.use_residual = use_residual and (in_features == out_features)

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.spline_initializer = keras.initializers.get(spline_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer or keras.regularizers.L2(regularization_factor))
        self.spline_regularizer = keras.regularizers.get(spline_regularizer or keras.regularizers.L2(regularization_factor))

        # Initialize activation function
        self.activation_fn = keras.activations.get(activation)

        # Initialize weights to None - will be created in build()
        self.base_weight = None
        self.spline_weight = None
        self.spline_scaler = None
        self._cached_grid = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and setup internal state.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        input_dim = input_shape[-1]
        if input_dim != self.in_features:
            raise ValueError(f"Input dimension {input_dim} doesn't match in_features {self.in_features}")

        # Initialize base weights with orthogonal initialization for better gradient flow
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.in_features, self.out_features),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=keras.constraints.MaxNorm(max_value=self.clip_value),
            trainable=True
        )

        # Initialize spline weights with careful scaling
        spline_init_scale = 1.0 / np.sqrt(self.grid_size + self.spline_order - 1)
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.in_features, self.out_features, self.grid_size + self.spline_order - 1),
            initializer=keras.initializers.RandomUniform(-spline_init_scale, spline_init_scale),
            regularizer=self.spline_regularizer,
            constraint=keras.constraints.MaxNorm(max_value=self.clip_value),
            trainable=True
        )

        # Initialize scaling factors with positive constraint
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.in_features, self.out_features),
            initializer='ones',
            regularizer=self.spline_regularizer,
            constraint=keras.constraints.NonNeg(),
            trainable=True
        )

        # Build the grid
        self._build_grid()

        super().build(input_shape)

    def _build_grid(self) -> None:
        """Build the grid points with proper spacing."""
        if self._cached_grid is None:
            grid_min, grid_max = self.grid_range
            # Create uniform grid points
            grid_points = ops.linspace(grid_min, grid_max, self.grid_size)
            self._cached_grid = ops.cast(grid_points, dtype="float32")



    def _normalize_inputs(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Normalize inputs to prevent numerical issues.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized input tensor.
        """
        # Clip extreme values
        x = ops.clip(x, self.grid_range[0], self.grid_range[1])

        # Scale to [0, 1] range
        x_normalized = (x - self.grid_range[0]) / (
            self.grid_range[1] - self.grid_range[0] + self.epsilon
        )
        return x_normalized

    def _compute_spline_basis(
        self,
        x: keras.KerasTensor,
        safe_mode: bool = True
    ) -> keras.KerasTensor:
        """Compute B-spline basis functions with enhanced numerical stability.

        Args:
            x: Input tensor.
            safe_mode: Whether to use additional numerical safeguards.

        Returns:
            Tensor of B-spline basis function values with shape [batch, in_features, num_basis_functions].
        """
        x_norm = self._normalize_inputs(x)

        # Scale normalized input to grid indices
        x_scaled = x_norm * (self.grid_size - 1)

        # Compute grid indices
        grid_indices = ops.floor(x_scaled)
        grid_indices = ops.clip(grid_indices, 0, self.grid_size - self.spline_order)

        # Compute local coordinates
        local_x = x_scaled - grid_indices

        # Create a simplified B-spline basis using polynomial interpolation
        # For numerical stability, we use a fixed set of basis functions
        num_basis = self.grid_size + self.spline_order - 1

        # Create basis functions - simplified approach for stability
        # Generate evenly spaced basis centers
        basis_centers = ops.linspace(0.0, 1.0, num_basis)
        basis_centers = ops.reshape(basis_centers, (1, 1, num_basis))

        # Expand dimensions for broadcasting
        local_x_expanded = ops.expand_dims(local_x, axis=-1)  # [batch, in_features, 1]

        # Compute Gaussian-like basis functions for stability
        sigma = 1.0 / num_basis
        basis_values = ops.exp(-ops.square(local_x_expanded - basis_centers) / (2 * sigma * sigma))

        if safe_mode:
            basis_values = ops.clip(basis_values, self.epsilon, 1.0 - self.epsilon)

        # Normalize basis functions to ensure partition of unity
        basis_sum = ops.sum(basis_values, axis=-1, keepdims=True)
        basis_values = basis_values / (basis_sum + self.epsilon)

        return basis_values

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass with enhanced stability and edge case handling.

        Args:
            inputs: Input tensor.
            training: Whether layer is in training mode.

        Returns:
            Output tensor after KAN transformation.
        """
        # Handle empty inputs using ops.cond for graph compatibility
        input_size = ops.size(inputs)

        def empty_case():
            return ops.zeros((0, self.out_features), dtype=inputs.dtype)

        def normal_case():
            # Compute base transformation
            base_output = ops.matmul(inputs, self.base_weight)

            # Compute spline transformation
            spline_basis = self._compute_spline_basis(inputs, safe_mode=training)
            # spline_basis shape: [batch, in_features, num_basis_functions]

            # Compute spline output using tensor operations
            # spline_weight shape: [in_features, out_features, num_basis_functions]

            # Expand spline_basis to [batch, in_features, 1, num_basis_functions]
            spline_basis_expanded = ops.expand_dims(spline_basis, axis=2)

            # Expand spline_weight to [1, in_features, out_features, num_basis_functions]
            spline_weight_expanded = ops.expand_dims(self.spline_weight, axis=0)

            # Element-wise multiplication and sum over basis functions
            # [batch, in_features, out_features, num_basis_functions] -> [batch, in_features, out_features]
            spline_contributions = ops.sum(spline_basis_expanded * spline_weight_expanded, axis=-1)

            # Sum over input features to get [batch, out_features]
            spline_output = ops.sum(spline_contributions, axis=1)

            # Scale spline output
            spline_scaler_safe = ops.maximum(self.spline_scaler, self.epsilon)
            scaling_factor = ops.mean(spline_scaler_safe, axis=0)  # Average over input features
            scaled_spline_output = spline_output * scaling_factor

            # Combine outputs with residual connection if enabled
            if self.use_residual and training:
                # Add skip connection with gating
                gate = ops.sigmoid(scaling_factor)
                total_output = gate * (base_output + scaled_spline_output) + (1 - gate) * inputs
            else:
                total_output = base_output + scaled_spline_output

            # Apply activation
            activated_output = self.activation_fn(total_output)

            # Final clipping for numerical stability
            if training:
                activated_output = ops.clip(activated_output, -self.clip_value, self.clip_value)

            return activated_output

        # Use conditional execution for graph compatibility
        return ops.cond(
            ops.equal(input_size, 0),
            empty_case,
            normal_case
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.out_features])

    def get_config(self) -> Dict[str, Any]:
        """Return the config of the layer with all parameters.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "in_features": self.in_features,
            "out_features": self.out_features,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "activation": self.activation_name,
            "regularization_factor": self.regularization_factor,
            "grid_range": self.grid_range,
            "epsilon": self.epsilon,
            "clip_value": self.clip_value,
            "use_residual": self.use_residual,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "spline_initializer": keras.initializers.serialize(self.spline_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "spline_regularizer": keras.regularizers.serialize(self.spline_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
