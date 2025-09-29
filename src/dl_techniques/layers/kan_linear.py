"""
A linear layer based on the Kolmogorov-Arnold Network (KAN) architecture.

This layer replaces the standard linear transformation followed by a fixed
activation function (e.g., `Wx + b` -> `ReLU`) with a more expressive structure
where the activation functions themselves are learnable. It operationalizes
the Kolmogorov-Arnold representation theorem, which posits that any multivariate
continuous function can be expressed as a composition of univariate functions.

Architecture:
    Unlike traditional Multi-Layer Perceptrons (MLPs) that learn linear
    transformations (weights) between fixed non-linearities, a KAN layer
    learns the non-linear transformations directly. This is achieved by
    parameterizing each activation function as a linear combination of B-spline
    basis functions. The layer's output is formed by two components:

    1.  A standard linear transformation (`base_weight`), which acts as a
        residual connection and provides a stable foundation for learning.
    2.  A spline-based transformation (`spline_weight`), which constitutes the
        learnable activation. For each pair of input and output neurons, a
        unique univariate function is learned.

    The final output is a combination of these two paths, allowing the model
    to learn both the linear dependencies and the intricate non-linear
    relationships within the data.

Foundational Mathematics:
    The core of KANs is the Kolmogorov-Arnold representation theorem. The theorem
    states that any multivariate continuous function `f(x_1, ..., x_n)` can be
    written as a finite sum of compositions of univariate functions:
    `f(x) = Σ_q Φ_q( Σ_p ψ_{q,p}(x_p) )`

    A KAN is a neural network realization of this theorem. Each `KANLinear`
    layer can be seen as learning the inner functions `ψ_{q,p}`. The network
    learns these functions by representing them as B-splines.

    A B-spline is a piecewise polynomial function defined over a set of intervals
    (determined by a `grid`). It is constructed from a linear combination of
    basis spline functions (`_compute_spline_basis`). The coefficients of this
    combination (the `spline_weight`) are learnable parameters. By adjusting
    these coefficients, the layer can approximate any continuous univariate
    function on the defined grid, effectively learning the optimal "activation
    function" for each connection in a data-driven manner.

Usage Recommendations:
    -   **Grid Size:** The `grid_size` parameter controls the granularity of the
        learned splines. It is advisable to start with a modest grid size
        (e.g., 5-10) to encourage smoother, more general functions and avoid
        overfitting. If the model underfits, gradually increase the grid size
        to allow for more complex function shapes, but be mindful that this
        significantly increases parameter count and memory consumption.

    -   **Spline Order:** The `spline_order` (typically 3 for cubic splines)
        determines the smoothness of the learned functions. Cubic splines offer
        a good balance of flexibility and smoothness for most tasks. Higher
        orders can capture exceptionally smooth functions, while lower orders
        (e.g., linear) may be sufficient for simpler relationships.

    -   **Network Architecture:** KANs are often more parameter-efficient than
        MLPs. It is recommended to start with a narrower and shallower
        architecture than an equivalent MLP. For deeper KANs, incorporating
        residual connections (enabled by default when dimensions match) is
        crucial for stable training and effective gradient flow.

References:
    1.  Liu, Z., Wang, Y., et al. (2024). "KAN: Kolmogorov-Arnold Networks."
        arXiv preprint arXiv:2404.19756.
    2.  Kolmogorov, A. N. (1957). "On the representation of continuous
        functions of many variables by superpositions of continuous
        functions of one variable and addition." Doklady Akademii Nauk SSSR.
    3.  De Boor, C. (1978). "A Practical Guide to Splines." Springer-Verlag.
"""

import keras
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KANLinear(keras.layers.Layer):
    """Kolmogorov-Arnold Network Linear Layer implementation with enhanced stability.

    This layer implements the core KAN linear transformation using B-spline basis functions
    combined with traditional linear transformations. It provides learnable activation
    functions through B-spline coefficients. The input features are automatically inferred
    from the input shape during the build phase.

    **Intent**: Implement a KAN layer that learns flexible activation functions using B-splines
    while maintaining numerical stability and proper serialization support for modern Keras 3.

    **Architecture**:
    ```
    Input(shape=[..., input_features])
           ↓
    Base Transform: x @ W_base → (batch, features)
           ↓
    B-spline Basis: φ(x) → (batch, input_features, num_basis)
           ↓
    Spline Transform: Σ(φ(x) ⊙ W_spline) → (batch, features)
           ↓
    Combine: base + scaled_spline → (batch, features)
           ↓
    Activation: σ(combined) → (batch, features)
           ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **Base transformation**: y_base = x @ W_base
    2. **B-spline basis**: φᵢ(x) = exp(-(x - cᵢ)² / (2σ²))
    3. **Spline transformation**: y_spline = Σᵢ φᵢ(x) ⊙ W_spline
    4. **Combination**: y = σ(y_base + α * y_spline)

    Args:
        features: Integer, number of output features. Must be positive.
        grid_size: Integer, size of the grid for B-splines. Must be >= spline_order. Defaults to 5.
        spline_order: Integer, order of B-splines. Must be positive. Defaults to 3.
        activation: String or callable, activation function name to use. Defaults to 'swish'.
        regularization_factor: Float, L2 regularization factor. Must be non-negative. Defaults to 0.01.
        grid_range: Tuple of two floats, range for the grid as (min, max). Defaults to (-1, 1).
        epsilon: Float, small constant for numerical stability. Must be positive. Defaults to 1e-7.
        clip_value: Float, maximum absolute value for gradients. Must be positive. Defaults to 1e3.
        use_residual: Boolean, whether to use residual connections. Only effective when
            input and output dimensions match. Defaults to True.
        kernel_initializer: String or Initializer, initializer for base weights.
            Defaults to 'orthogonal'.
        spline_initializer: String or Initializer, initializer for spline weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for base weights. If None, uses L2 with
            regularization_factor.
        spline_regularizer: Optional regularizer for spline weights. If None, uses L2 with
            regularization_factor.
        **kwargs: Additional keyword arguments passed to the parent class.

    Input shape:
        Tensor with shape `(batch_size, ..., input_features)`.

    Output shape:
        Tensor with shape `(batch_size, ..., features)`.

    Attributes:
        base_weight: Weight matrix for linear transformation of shape (input_features, features).
        spline_weight: Weight tensor for B-spline coefficients of shape
            (input_features, features, grid_size + spline_order - 1).
        spline_scaler: Scaling factors for spline outputs of shape (input_features, features).

    Example:
        ```python
        # Basic usage - input features inferred automatically
        layer = KANLinear(features=64)
        inputs = keras.Input(shape=(32,))  # 32 input features
        outputs = layer(inputs)  # (batch, 64) output features

        # Advanced configuration
        layer = KANLinear(
            features=128,
            grid_size=8,
            spline_order=3,
            activation='gelu',
            regularization_factor=0.001,
            grid_range=(-2, 2),
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(784,))
        x = KANLinear(256, activation='swish')(inputs)
        x = KANLinear(128, activation='gelu')(x)
        outputs = KANLinear(10, activation='softmax')(x)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If features is not positive.
        ValueError: If grid_size < spline_order.
        ValueError: If grid_range[0] >= grid_range[1].
        ValueError: If input is not at least 2D during build.

    Note:
        This implementation uses simplified B-spline basis functions based on Gaussian-like
        functions for enhanced numerical stability. The residual connection is only applied
        when input and output dimensions match.
    """

    def __init__(
        self,
        features: int,
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
            features: Number of output features.
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
            ValueError: If features is not positive, grid size < spline order, or invalid grid range.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if features <= 0:
            raise ValueError("Features must be positive integer")
        if grid_size < spline_order:
            raise ValueError("Grid size must be >= spline order")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range")

        # Store ALL configuration parameters
        self.features = features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_name = activation
        self.regularization_factor = regularization_factor
        self.grid_range = grid_range
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.use_residual = use_residual

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.spline_initializer = keras.initializers.get(spline_initializer)
        self.kernel_regularizer = keras.regularizers.get(
            kernel_regularizer or keras.regularizers.L2(regularization_factor)
        )
        self.spline_regularizer = keras.regularizers.get(
            spline_regularizer or keras.regularizers.L2(regularization_factor)
        )

        # Initialize activation function
        self.activation_fn = keras.activations.get(activation)

        # Initialize weight attributes - will be created in build()
        self.base_weight = None
        self.spline_weight = None
        self.spline_scaler = None
        self._cached_grid = None
        self.input_features = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and setup internal state.

        Args:
            input_shape: Shape of the input tensor.

        Raises:
            ValueError: If input is not at least 2D.
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        # Infer input features from shape
        self.input_features = input_shape[-1]
        if self.input_features is None:
            raise ValueError("Input features dimension cannot be None")

        # Update residual flag based on actual dimensions
        self._use_residual_actual = self.use_residual and (self.input_features == self.features)

        # Initialize base weights with orthogonal initialization for better gradient flow
        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.input_features, self.features),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=keras.constraints.MaxNorm(max_value=self.clip_value),
            trainable=True
        )

        # Initialize spline weights with careful scaling
        spline_init_scale = 1.0 / np.sqrt(self.grid_size + self.spline_order - 1)
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.input_features, self.features, self.grid_size + self.spline_order - 1),
            initializer=keras.initializers.RandomUniform(-spline_init_scale, spline_init_scale),
            regularizer=self.spline_regularizer,
            constraint=keras.constraints.MaxNorm(max_value=self.clip_value),
            trainable=True
        )

        # Initialize scaling factors with positive constraint
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            regularizer=self.spline_regularizer,
            constraint=keras.constraints.NonNeg(),
            trainable=True
        )

        # Build the grid
        self._build_grid()

        # Always call parent build at the end
        super().build(input_shape)

    def _build_grid(self) -> None:
        """Build the grid points with proper spacing."""
        if self._cached_grid is None:
            grid_min, grid_max = self.grid_range
            # Create uniform grid points
            grid_points = keras.ops.linspace(grid_min, grid_max, self.grid_size)
            self._cached_grid = keras.ops.cast(grid_points, dtype="float32")

    def _normalize_inputs(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Normalize inputs to prevent numerical issues.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized input tensor scaled to [0, 1] range.
        """
        # Clip extreme values
        x = keras.ops.clip(x, self.grid_range[0], self.grid_range[1])

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
            x: Input tensor of shape (..., input_features).
            safe_mode: Whether to use additional numerical safeguards.

        Returns:
            Tensor of B-spline basis function values with shape
            (..., input_features, num_basis_functions).
        """
        x_norm = self._normalize_inputs(x)

        # Scale normalized input to grid indices
        x_scaled = x_norm * (self.grid_size - 1)

        # Compute grid indices
        grid_indices = keras.ops.floor(x_scaled)
        grid_indices = keras.ops.clip(grid_indices, 0, self.grid_size - self.spline_order)

        # Compute local coordinates
        local_x = x_scaled - grid_indices

        # Create a simplified B-spline basis using polynomial interpolation
        # For numerical stability, we use a fixed set of basis functions
        num_basis = self.grid_size + self.spline_order - 1

        # Create basis functions - simplified approach for stability
        # Generate evenly spaced basis centers
        basis_centers = keras.ops.linspace(0.0, 1.0, num_basis)
        basis_centers = keras.ops.reshape(basis_centers, (1, 1, num_basis))

        # Expand dimensions for broadcasting
        # local_x shape: (..., input_features)
        local_x_expanded = keras.ops.expand_dims(local_x, axis=-1)  # (..., input_features, 1)

        # Compute Gaussian-like basis functions for stability
        sigma = 1.0 / num_basis
        basis_values = keras.ops.exp(
            -keras.ops.square(local_x_expanded - basis_centers) / (2 * sigma * sigma)
        )

        if safe_mode:
            basis_values = keras.ops.clip(basis_values, self.epsilon, 1.0 - self.epsilon)

        # Normalize basis functions to ensure partition of unity
        basis_sum = keras.ops.sum(basis_values, axis=-1, keepdims=True)
        basis_values = basis_values / (basis_sum + self.epsilon)

        return basis_values

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass with enhanced stability and edge case handling.

        Args:
            inputs: Input tensor of shape (batch_size, ..., input_features).
            training: Boolean indicating whether the layer is in training mode.

        Returns:
            Output tensor of shape (batch_size, ..., features) after KAN transformation.
        """
        # Removed conditional logic for empty inputs. Standard ops handle zero-sized
        # dimensions correctly, and this is more robust for tf.function tracing.

        # Compute base transformation
        base_output = keras.ops.matmul(inputs, self.base_weight)

        # Compute spline transformation
        spline_basis = self._compute_spline_basis(inputs, safe_mode=training)
        # spline_basis shape: (..., input_features, num_basis)
        # spline_weight shape: (input_features, features, num_basis)

        # Use einsum for efficient and N-D compatible tensor contraction.
        spline_output = keras.ops.einsum('...ik,iok->...o', spline_basis, self.spline_weight)

        # Scale spline output
        spline_scaler_safe = keras.ops.maximum(self.spline_scaler, self.epsilon)
        scaling_factor = keras.ops.mean(spline_scaler_safe, axis=0)  # Average over input features
        scaled_spline_output = spline_output * scaling_factor

        # Combine outputs with residual connection if enabled
        if self._use_residual_actual and training:
            # Add skip connection with gating
            gate = keras.ops.sigmoid(scaling_factor)
            total_output = gate * (base_output + scaled_spline_output) + (1 - gate) * inputs
        else:
            total_output = base_output + scaled_spline_output

        # Apply activation
        activated_output = self.activation_fn(total_output)

        # Final clipping for numerical stability
        if training:
            activated_output = keras.ops.clip(activated_output, -self.clip_value, self.clip_value)

        return activated_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple with the last dimension changed to features.
        """
        input_shape_list = list(input_shape)
        input_shape_list[-1] = self.features
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Return the config of the layer with all parameters.

        Returns:
            Dictionary containing the layer configuration for serialization.
        """
        config = super().get_config()
        config.update({
            "features": self.features,
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

# ---------------------------------------------------------------------