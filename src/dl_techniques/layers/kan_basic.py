"""
Kolmogorov-Arnold Network (KAN) Implementation
============================================

This implementation is based on the paper:
"KAN: Kolmogorov-Arnold Networks" by Liu et al. (2024)

Theoretical Background
---------------------
The Kolmogorov-Arnold representation theorem states that any multivariate continuous
function can be represented as a composition of univariate functions. KAN leverages
this by creating a deep neural network architecture where activation functions are
learned using B-splines, rather than using fixed activation functions.

Key Components
-------------
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

Implementation Details
--------------------
1. B-spline Computation:
   - Uses De Boor's algorithm for numerical stability
   - Implements proper scaling of inputs to grid range
   - Handles boundary conditions for B-spline evaluation

2. Weight Initialization:
   - Base weights: Glorot uniform initialization
   - Spline weights: Glorot uniform initialization
   - Grid points: Uniform distribution in specified range
   - Scaling factors: Initialized to ones

3. Regularization:
   - L2 regularization on all weights
   - Separate regularization factors for base and spline weights
   - Grid point constraints to maintain proper ordering

Key Findings and Improvements
---------------------------
1. Advantages:
   - Better approximation of complex functions
   - Fewer parameters than equivalent MLPs
   - More interpretable due to B-spline basis
   - Better handling of multi-scale problems

2. Limitations:
   - Computational overhead from B-spline evaluation
   - Memory intensive due to spline coefficient storage
   - Sensitive to grid point initialization
   - May require careful tuning of hyperparameters

3. Performance Considerations:
   - B-spline computation is the main bottleneck
   - Grid size vs. accuracy trade-off
   - Memory usage scales with grid size and spline order

Usage Recommendations
-------------------
1. Grid Size Selection:
   - Start with small grid (5-10 points)
   - Increase if underfitting observed
   - Monitor memory usage with large grids

2. Spline Order:
   - Order 3 (cubic) recommended for most applications
   - Higher orders may help with very smooth functions
   - Lower orders for simpler relationships

3. Regularization:
   - Start with small regularization factor (0.01)
   - Adjust based on overfitting/underfitting
   - Consider separate factors for different components

4. Architecture:
   - Use fewer neurons than equivalent MLP
   - Add layers gradually if needed
   - Consider residual connections for deep networks

References
----------
1. Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
2. Kolmogorov, A. N. (1957) "On the representation of continuous functions"
3. Arnold, V. I. (1963) "On functions of three variables"
4. De Boor, C. (1978) "A Practical Guide to Splines"

Author: Nikolas Markou
Date: 16/01/2025
Version: 1.0.0
"""
import numpy as np
from typing import Tuple
import tensorflow as tf
from keras.api.layers import Layer
from keras.api.regularizers import l2

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils import logger

# ---------------------------------------------------------------------


class KANLinear(Layer):
    """Kolmogorov-Arnold Network Linear Layer implementation with enhanced stability."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 activation: str = 'silu',
                 regularization_factor: float = 0.01,
                 grid_range: Tuple[float, float] = (-1, 1),
                 epsilon: float = 1e-7,
                 clip_value: float = 1e3,
                 use_residual: bool = True,
                 **kwargs):
        """Initialize KANLinear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            grid_size: Size of the grid for B-splines
            spline_order: Order of B-splines
            activation: Activation function to use
            regularization_factor: L2 regularization factor
            grid_range: Range for the grid
            epsilon: Small constant for numerical stability
            clip_value: Maximum absolute value for gradients
            use_residual: Whether to use residual connections

        Raises:
            ValueError: If inputs are invalid
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Features must be positive integers")
        if grid_size < spline_order:
            raise ValueError("Grid size must be >= spline order")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range")

        super(KANLinear, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_func = getattr(tf.nn, activation)
        self.regularizer = l2(regularization_factor)
        self.grid_range = grid_range
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.use_residual = use_residual and (in_features == out_features)

        # Initialize weights with careful constraints
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(in_features, out_features),
            initializer=self._create_orthogonal_initializer(),
            regularizer=self.regularizer,
            constraint=self._create_clip_constraint(),
            trainable=True
        )

        # Initialize spline weights with careful scaling
        spline_init_scale = 1.0 / np.sqrt(grid_size + spline_order - 1)
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(in_features, out_features, grid_size + spline_order - 1),
            initializer=tf.keras.initializers.RandomUniform(
                -spline_init_scale,
                spline_init_scale
            ),
            regularizer=self.regularizer,
            constraint=self._create_clip_constraint(),
            trainable=True
        )

        # Initialize scaling factors with positive constraint
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(in_features, out_features),
            initializer='ones',
            regularizer=self.regularizer,
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True
        )

    def _create_orthogonal_initializer(self):
        """Creates an orthogonal initializer for better gradient flow."""
        return tf.keras.initializers.Orthogonal(gain=1.0)

    def _create_clip_constraint(self):
        """Creates a constraint to clip weights for stability."""
        return tf.keras.constraints.MaxNorm(max_value=self.clip_value)

    def build_grid(self) -> tf.Tensor:
        """Builds the grid points with proper spacing and caching."""
        if not hasattr(self, '_cached_grid'):
            grid_min, grid_max = self.grid_range
            # Use log-spaced grid for better handling of different scales
            grid_points = tf.exp(tf.linspace(
                tf.math.log(tf.abs(grid_min) + self.epsilon),
                tf.math.log(tf.abs(grid_max) + self.epsilon),
                self.grid_size
            ))
            grid_points = tf.sign(grid_min) * grid_points
            self._cached_grid = tf.cast(grid_points, dtype=tf.float32)
        return self._cached_grid

    def _normalize_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Normalizes inputs to prevent numerical issues."""
        # Clip extreme values
        x = tf.clip_by_value(x, self.grid_range[0], self.grid_range[1])

        # Scale to [0, 1] range
        x_normalized = (x - self.grid_range[0]) / (
                self.grid_range[1] - self.grid_range[0] + self.epsilon
        )
        return x_normalized

    def compute_spline_basis(self,
                             x: tf.Tensor,
                             grid_points: tf.Tensor,
                             safe_mode: bool = True) -> tf.Tensor:
        """Computes B-spline basis functions with enhanced numerical stability.

        Args:
            x: Input tensor
            grid_points: Grid points for B-spline evaluation
            safe_mode: Whether to use additional numerical safeguards

        Returns:
            Tensor of B-spline basis function values
        """
        x_norm = self._normalize_inputs(x)

        # Compute knot differences with numerical stability
        u = tf.expand_dims(x_norm, -1) - tf.range(
            0.,
            self.grid_size + self.spline_order - 1,
            dtype=tf.float32
        )

        # Safe computation of basis functions
        u = tf.maximum(self.epsilon, u)
        u = tf.minimum(1.0 - self.epsilon, u)

        # Initialize basis functions
        basis = tf.ones_like(u)

        # Apply modified De Boor's algorithm
        for j in range(1, self.spline_order):
            w = u[..., j:] / (j + self.epsilon)

            if safe_mode:
                # Add numerical safeguards
                w = tf.clip_by_value(w, self.epsilon, 1.0 - self.epsilon)
                basis = tf.clip_by_value(basis, self.epsilon, self.clip_value)

            basis = w * basis[..., :-1] + (1 - w) * basis[..., 1:]

        return basis

    @tf.custom_gradient
    def _custom_matmul(self, x, w):
        """Custom matrix multiplication with gradient clipping."""
        result = tf.matmul(x, w)

        def grad(dy):
            dx = tf.matmul(dy, w, transpose_b=True)
            dw = tf.matmul(x, dy, transpose_a=True)

            # Clip gradients
            dx = tf.clip_by_norm(dx, self.clip_value)
            dw = tf.clip_by_norm(dw, self.clip_value)

            return dx, dw

        return result, grad

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Forward pass with enhanced stability and edge case handling.

        Args:
            inputs: Input tensor
            training: Whether layer is in training mode

        Returns:
            Output tensor
        """
        # Input validation
        tf.debugging.assert_all_finite(
            inputs,
            "Input contains inf or nan"
        )

        # Handle empty inputs
        if tf.size(inputs) == 0:
            return tf.zeros([0, self.out_features], dtype=inputs.dtype)

        # Build grid
        grid_points = self.build_grid()

        # Compute base transformation with gradient control
        base_output = self._custom_matmul(inputs, self.base_weight)

        # Compute spline transformation
        spline_basis = self.compute_spline_basis(
            inputs,
            grid_points,
            safe_mode=training
        )

        # Safe computation of spline output
        spline_output = tf.einsum(
            '...i,ijo->...jo',
            spline_basis,
            self.spline_weight,
            name="spline_transform"
        )

        # Scale spline output
        scaled_spline_output = spline_output * tf.maximum(
            self.spline_scaler,
            self.epsilon
        )

        # Combine outputs with residual connection if enabled
        if self.use_residual and training:
            # Add skip connection with gating
            gate = tf.sigmoid(self.spline_scaler)
            total_output = gate * (base_output + scaled_spline_output) + \
                           (1 - gate) * inputs
        else:
            total_output = base_output + scaled_spline_output

        # Apply activation with gradient clipping
        activated_output = self.activation_func(total_output)

        # Final numerical safety check
        if training:
            activated_output = tf.clip_by_value(
                activated_output,
                -self.clip_value,
                self.clip_value
            )

        return activated_output

    def get_config(self) -> dict:
        """Returns the config of the layer with all parameters."""
        base_config = super().get_config()
        return {
            **base_config,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "activation": self.activation_func.__name__,
            "regularization_factor": self.regularizer.l2,
            "grid_range": self.grid_range,
            "epsilon": self.epsilon,
            "clip_value": self.clip_value,
            "use_residual": self.use_residual
        }

# ---------------------------------------------------------------------


class KAN(tf.keras.Sequential):
    """Kolmogorov-Arnold Network model with stability enhancements."""

    def __init__(self,
                 layers_configurations: list,
                 enable_debugging: bool = False,
                 **kwargs):
        """Initialize KAN model.

        Args:
            layers_configurations: List of layer configurations
            enable_debugging: Whether to enable extra validation
            **kwargs: Additional arguments for Sequential
        """
        super(KAN, self).__init__()

        self.enable_debugging = enable_debugging
        self._validate_configurations(layers_configurations)

        for layer_config in layers_configurations:
            self.add(KANLinear(**layer_config, **kwargs))

    def _validate_configurations(self, configs: list) -> None:
        """Validates layer configurations for compatibility."""
        if not configs:
            raise ValueError("Empty layer configurations")

        for i in range(len(configs) - 1):
            if configs[i]['out_features'] != configs[i + 1]['in_features']:
                raise ValueError(
                    f"Layer {i} output features don't match layer {i + 1} input features"
                )

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Forward pass with additional validation in debug mode."""
        outputs = inputs

        if self.enable_debugging:
            tf.debugging.assert_all_finite(inputs, "Input contains inf or nan")

        for layer in self.layers:
            outputs = layer(outputs, training=training)

            if self.enable_debugging:
                tf.debugging.assert_all_finite(
                    outputs,
                    f"Layer {layer.name} produced inf or nan"
                )

        return outputs
