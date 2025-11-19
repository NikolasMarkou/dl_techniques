"""
Kolmogorov-Arnold Network (KAN) linear layer.

This layer serves as a more expressive alternative to the standard `Dense` layer,
grounded in the principles of the Kolmogorov-Arnold representation theorem.
The theorem posits that any multivariate continuous function can be represented
as a finite composition of univariate functions and the binary operation of
addition. This layer provides a practical, learnable approximation of this
theorem.

Architecturally, a traditional dense layer computes `y = activation(W @ x + b)`,
applying a single, fixed activation function after a linear transformation.
In contrast, a KAN layer replaces this structure by learning a unique,
univariate activation function `phi_ij` for each connection between an
input neuron `i` and an output neuron `j`. The layer's output is the sum
of these individually activated connections, `y_j = Σ_i phi_ij(x_i)`. This
design moves the learning complexity from the linear weights (as in MLPs) to
the activation functions themselves, enabling the network to capture more
intricate and non-linear relationships with fewer parameters and layers.

The core mathematical challenge is parameterizing the learnable activation
functions `phi_ij(x)` in a differentiable and efficient manner. This
implementation achieves this by representing each `phi_ij` as a B-spline.
A B-spline is a piecewise polynomial function constructed as a linear
combination of basis spline functions (`B_k`) over a specified grid:

`spline_ij(x) = Σ_k c_ijk * B_k(x)`

The coefficients `c_ijk` are the primary learnable parameters of the layer,
controlling the shape of the spline. The basis functions `B_k(x)` are
efficiently computed using the Cox-de Boor recursion formula.

To combine the representational power of splines with the favorable
optimization properties of common activations, each `phi_ij` is a sum of
the learnable B-spline and a fixed base activation `b(x)` (e.g., SiLU),
each with its own learnable scalar weight:

`phi_ij(x) = w_base_ij * b(x) + w_spline_ij * spline_ij(x)`

This composite structure allows the layer to learn both global trends through
the base function and fine-grained, localized features through the spline,
adapting its functional form to the data distribution.

References:
    - Liu, Z., Wang, Y., et al. (2024). "KAN: Kolmogorov-Arnold Networks."
      arXiv preprint arXiv:2404.19756.

"""

import keras
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KANLinear(keras.layers.Layer):
    """
    Kolmogorov-Arnold Network (KAN) linear layer with learnable activation functions.

    This layer replaces the standard Dense layer's operation `activation(x @ W + b)`
    with a more expressive structure where each connection from an input feature to an
    output feature has its own learnable activation function `phi_ij`. Unlike traditional
    layers with fixed, predefined activations, KAN layers learn optimal activation
    functions for each input-output connection in a data-driven manner through
    parameterized B-splines combined with base activation functions.

    **Intent**: Provide a Keras-native, serializable implementation of a KAN layer
    that offers greater expressivity than traditional Dense layers by learning
    connection-specific activation functions. This enables more parameter-efficient
    architectures that can capture complex input-output relationships with fewer
    layers and neurons.

    **Architecture**:
    ```
    Input(shape=[..., input_features])
              ↓
    For each (i, j) pair:
         ↓
    Base Path: b(x_i) * w_base_ij
         ↓
    Spline Path: spline_ij(x_i) * w_spline_ij
         ↓
    Combine: phi_ij(x_i) = base + spline
         ↓
    Sum over i: y_j = Σ_i phi_ij(x_i)
         ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **Per-Connection Activation**: phi_ij(x) = w_base_ij * b(x) + w_spline_ij * spline_ij(x)
       - b(x) is a fixed base activation (e.g., SiLU/Swish)
       - spline_ij(x) is a learnable B-spline: Σ_k c_ijk * B_k(x)
       - w_base_ij and w_spline_ij are learnable scalar weights
    2. **Output Aggregation**: y_j = Σ_i phi_ij(x_i)
    3. **B-spline Basis**: B_k(x) computed via Cox-de Boor recursion on grid

    Where:
    - x_i is the i-th input feature
    - y_j is the j-th output feature
    - B_k(x) are B-spline basis functions of specified order
    - c_ijk are learned spline coefficients

    **Usage Recommendations**:
    - **Grid Updates**: Call `update_grid_from_samples(x)` periodically during
      training with representative data to adapt the B-spline grid to the actual
      input distribution for optimal expressivity.
    - **Network Depth**: KANs are more parameter-efficient than MLPs. Start with
      narrower and shallower architectures.
    - **Residual Connections**: For deep KANs, use standard residual connections:
      `y = x + KANLinear(features)(x)` to facilitate gradient flow.

    References:
        1. Liu, Z., Wang, Y., et al. (2024). "KAN: Kolmogorov-Arnold Networks."
           arXiv preprint arXiv:2404.19756.

    Args:
        features: Integer, number of output features. Must be positive.
            This determines the width of the layer's output.
        grid_size: Integer, number of intervals in the B-spline grid. Must be positive.
            Controls the resolution of learnable activation functions. Larger values
            enable more complex activation shapes but increase parameters.
            Defaults to 5.
        spline_order: Integer, order (degree) of the B-spline basis functions.
            Must be non-negative. Higher orders produce smoother activations.
            Common values: 1 (linear), 2 (quadratic), 3 (cubic). Defaults to 3.
        grid_range: Tuple of two floats, (min, max) range for the initial B-spline grid.
            Should approximately cover the expected range of input activations.
            Can be updated later via `update_grid_from_samples()`.
            Defaults to (-2.0, 2.0).
        activation: String name or callable for the base activation function b(x).
            Applied uniformly to all connections. Common choices: 'swish', 'silu',
            'relu', 'gelu'. Defaults to 'swish'.
        base_trainable: Boolean, whether the base activation scaling weights
            (base_scaler) are trainable. When True, the layer can learn to emphasize
            or de-emphasize the base activation path per connection. Defaults to True.
        spline_trainable: Boolean, whether the spline activation scaling weights
            (spline_scaler) are trainable. When True, the layer can learn to emphasize
            or de-emphasize the spline path per connection. Defaults to True.
        kernel_initializer: Initializer for the spline coefficient weights. Can be
            string name ('glorot_uniform', 'he_normal') or Initializer instance.
            Defaults to 'glorot_uniform'.
        epsilon: Small float added for numerical stability in division operations
            during B-spline basis computation. Defaults to 1e-7.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_features)`.
        The layer accepts any number of leading batch dimensions.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., features)`.
        Only the final dimension changes from input_features to features.
    """

    def __init__(
            self,
            features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            grid_range: Tuple[float, float] = (-2.0, 2.0),
            activation: Union[str, Callable] = 'swish',
            base_trainable: bool = True,
            spline_trainable: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if features <= 0:
            raise ValueError("Features must be a positive integer.")
        if grid_size <= 0:
            raise ValueError("Grid size must be a positive integer.")
        if spline_order < 0:
            raise ValueError("Spline order must be a non-negative integer.")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range: min must be less than max.")

        # Store configuration
        self.features = features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.base_activation_name = activation
        self.base_trainable = base_trainable
        self.spline_trainable = spline_trainable
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.epsilon = epsilon
        self.base_activation_fn = keras.activations.get(activation)

        # Attributes initialized in build()
        self.input_features: Optional[int] = None
        self.grid: Optional[keras.Variable] = None
        self.spline_weight: Optional[keras.Variable] = None
        self.spline_scaler: Optional[keras.Variable] = None
        self.base_scaler: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create layer weights and B-spline grid based on input shape.

        Args:
            input_shape: Shape tuple of the input tensor. Must be at least 2D.
        """
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        self.input_features = input_shape[-1]
        if self.input_features is None:
            raise ValueError("Input features dimension cannot be None.")

        # Number of B-spline basis functions = grid_size + spline_order
        num_basis_fns = self.grid_size + self.spline_order

        # 1. Spline Coefficients (Control Points)
        # Shape: (input_features, output_features, num_basis_fns)
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.input_features, self.features, num_basis_fns),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # 2. Spline Scaler
        # Shape: (input_features, output_features)
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            trainable=self.spline_trainable,
        )

        # 3. Base Scaler
        # Shape: (input_features, output_features)
        self.base_scaler = self.add_weight(
            name="base_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            trainable=self.base_trainable,
        )

        # 4. Grid (Knot Sequence)
        # The grid determines the shape of the basis functions. It is a non-trainable
        # weight so it persists during saving/loading but isn't updated by gradient descent.
        # Size: grid_size + 1 (intervals) + 2 * spline_order (padding)
        grid_length = self.grid_size + 2 * self.spline_order + 1
        self.grid = self.add_weight(
            name="grid",
            shape=(grid_length,),
            initializer="zeros",  # Initialized properly immediately below
            trainable=False,
            dtype=self.dtype,
        )

        # Initialize the grid based on the configured range
        self._set_grid_from_range(self.grid_range[0], self.grid_range[1])

        super().build(input_shape)

    def _compute_grid_values(self, start: float, stop: float) -> keras.KerasTensor:
        """
        Compute the extended B-spline knot sequence values.

        Generates a uniform grid between [start, stop] and extends it on both
        ends to satisfy B-spline boundary conditions.

        Args:
            start: Range minimum.
            stop: Range maximum.

        Returns:
            Tensor containing the complete knot sequence.
        """
        # Create uniform grid points within the specified range
        # Shape: (grid_size + 1,)
        grid_points = keras.ops.linspace(
            start, stop, self.grid_size + 1, dtype=self.dtype
        )

        # Calculate grid spacing step size
        h = (grid_points[1] - grid_points[0])

        # Generate index ranges for extensions
        start_indices = keras.ops.arange(-self.spline_order, 0, dtype=self.dtype)
        end_indices = keras.ops.arange(1, self.spline_order + 1, dtype=self.dtype)

        # Extend grid on the left
        extended_knots_start = start_indices * h + grid_points[0]

        # Extend grid on the right
        extended_knots_end = end_indices * h + grid_points[-1]

        # Concatenate to form complete knot sequence
        return keras.ops.concatenate(
            [extended_knots_start, grid_points, extended_knots_end], axis=0
        )

    def _set_grid_from_range(self, start: float, stop: float) -> None:
        """Helper to calculate and assign grid values to the state variable."""
        grid_values = self._compute_grid_values(start, stop)
        self.grid.assign(grid_values)

    def _compute_bspline_basis(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute B-spline basis functions using Cox-de Boor recursion formula.

        Args:
            x: Input tensor, shape (..., input_features).

        Returns:
            Basis function values, shape (..., input_features, num_basis_fns).
        """
        # Add dimension for broadcasting with grid: (..., input_features, 1)
        x = keras.ops.expand_dims(x, axis=-1)

        # Access the grid weight
        grid = self.grid

        # Base case k=0: piecewise constant basis functions
        # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
        # Grid slice logic: we compare against all intervals simultaneously
        grid_left = grid[:-1]
        grid_right = grid[1:]

        basis = keras.ops.cast(
            keras.ops.logical_and(x >= grid_left, x < grid_right),
            dtype=self.dtype,
        )

        # Iteratively compute higher-order B-splines via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            # Grid indices for term 1: t_i to t_{i+k}
            # Denominator: t_{i+k} - t_i
            d1 = grid[k:-1] - grid[:-(k + 1)]
            # Numerator: x - t_i
            n1 = x - grid[:-(k + 1)]

            # Term 1 calculation with stability epsilon
            term1 = keras.ops.divide(n1, d1 + self.epsilon) * basis[..., :-1]

            # Grid indices for term 2: t_{i+1} to t_{i+k+1}
            # Denominator: t_{i+k+1} - t_{i+1}
            d2 = grid[k + 1:] - grid[1:-k]
            # Numerator: t_{i+k+1} - x
            n2 = grid[k + 1:] - x

            # Term 2 calculation
            term2 = keras.ops.divide(n2, d2 + self.epsilon) * basis[..., 1:]

            # Combine terms
            basis = term1 + term2

        return basis

    def call(
            self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: compute output using learned activation functions.

        Args:
            inputs: Input tensor, shape (batch_size, ..., input_features).
            training: Unused.

        Returns:
            Output tensor, shape (batch_size, ..., features).
        """
        # Path 1: Base Activation
        # Shape: (..., input_features)
        base_val = self.base_activation_fn(inputs)

        # Path 2: B-spline Activation
        # Evaluate basis functions: (..., input_features, num_basis_fns)
        spline_basis = self._compute_bspline_basis(inputs)

        # Linear combination of basis functions with learned weights
        # Tensor contraction:
        # spline_basis:  [..., i, k]
        # spline_weight: [i, o, k]
        # Result:        [..., i, o]
        spline_val = keras.ops.einsum(
            '...ik,iok->...io', spline_basis, self.spline_weight
        )

        # Combine Paths
        # Expand base_val to (..., input_features, 1) for broadcasting over output features
        # Then scale by base_scaler (input_features, features)
        phi_base = keras.ops.expand_dims(base_val, axis=-1) * self.base_scaler

        # Scale spline path
        phi_spline = spline_val * self.spline_scaler

        # Sum components to get activation function output per connection
        phi = phi_base + phi_spline

        # Aggregate inputs: y_j = sum_i(phi_{ij}(x_i))
        # Sum over input_features dimension (axis -2)
        output = keras.ops.sum(phi, axis=-2)

        return output

    def update_grid_from_samples(self, x: Union[keras.KerasTensor, Any]) -> None:
        """
        Update B-spline grid based on input data statistics.

        This method updates the `grid` weight to cover the range of the input data `x`.
        It maintains a uniform grid distribution but adapts the min/max boundaries.

        Args:
            x: Input data tensor, shape (batch_size, input_features).

        Raises:
            ValueError: If input is not 2D.
        """
        # Ensure input is a tensor
        x = keras.ops.convert_to_tensor(x, dtype=self.dtype)

        if len(keras.ops.shape(x)) != 2:
            raise ValueError("Input 'x' for grid update must be 2D (batch, features).")

        # Sort each feature column to find distribution boundaries
        # Shape: (batch_size, input_features)
        x_sorted = keras.ops.sort(x, axis=0)
        batch_size = keras.ops.shape(x)[0]

        # Determine indices for uniform quantile selection
        # We select grid_size + 1 points to estimate the range
        indices = keras.ops.cast(
            keras.ops.linspace(0, batch_size - 1, self.grid_size + 1),
            dtype="int32"
        )

        # Gather values at these indices
        # Shape: (grid_size + 1, input_features)
        grid_points_per_feature = keras.ops.take(x_sorted, indices, axis=0)

        # Average across features to find a unified range for the layer
        # Shape: (grid_size + 1,)
        new_grid_points = keras.ops.mean(grid_points_per_feature, axis=1)

        # Update the grid range configuration
        new_min = float(keras.ops.convert_to_numpy(new_grid_points[0]))
        new_max = float(keras.ops.convert_to_numpy(new_grid_points[-1]))
        self.grid_range = (new_min, new_max)

        # Re-calculate and assign the grid weight values
        self._set_grid_from_range(new_min, new_max)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape from input shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config.update({
            "features": self.features,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "activation": self.base_activation_name,
            "base_trainable": self.base_trainable,
            "spline_trainable": self.spline_trainable,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
