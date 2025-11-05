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
from typing import Tuple, Optional, Dict, Any, Union

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

    Attributes:
        features: Number of output features (from constructor).
        grid_size: Number of B-spline grid intervals (from constructor).
        spline_order: Order of B-spline basis functions (from constructor).
        grid_range: Current (min, max) range of the B-spline grid.
        base_activation_name: String name of the base activation function.
        base_trainable: Whether base scalers are trainable (from constructor).
        spline_trainable: Whether spline scalers are trainable (from constructor).
        kernel_initializer: Initializer instance for spline weights.
        epsilon: Numerical stability constant (from constructor).
        base_activation_fn: Callable activation function instance.
        input_features: Integer, number of input features (set during build).
        grid: KerasTensor, extended B-spline knot sequence including boundary knots.
        spline_weight: KerasTensor, learnable coefficients for B-spline basis,
            shape (input_features, features, grid_size + spline_order).
        spline_scaler: KerasTensor, learnable scaling weights for spline path,
            shape (input_features, features).
        base_scaler: KerasTensor, learnable scaling weights for base activation path,
            shape (input_features, features).

    Methods:
        update_grid_from_samples(x): Update B-spline grid based on input data quantiles.

    Example:
        ```python
        # Basic usage: replace Dense layer
        layer = KANLinear(features=64)
        inputs = keras.Input(shape=(32,))
        outputs = layer(inputs)  # Shape: (batch, 64)

        # Custom configuration with finer grid
        layer = KANLinear(
            features=128,
            grid_size=10,  # More fine-grained grid
            spline_order=2,  # Quadratic splines
            grid_range=(-3.0, 3.0),
            activation='gelu'
        )

        # Build a simple KAN network
        inputs = keras.Input(shape=(784,))
        x = KANLinear(128)(inputs)
        x = KANLinear(64)(x)
        outputs = KANLinear(10)(x)
        model = keras.Model(inputs, outputs)

        # Training with grid updates (recommended)
        model.compile(optimizer='adam', loss='mse')
        for epoch in range(num_epochs):
            model.fit(x_train, y_train)
            # Update grids periodically (e.g., every 5 epochs)
            if epoch % 5 == 0:
                for layer in model.layers:
                    if isinstance(layer, KANLinear):
                        layer.update_grid_from_samples(x_train[:100])
        ```

    Note:
        The layer's expressivity comes from learning both the shape (spline coefficients)
        and importance (scalers) of activation functions for each input-output connection.
        This is more flexible than traditional layers but requires more parameters:
        O(input_features * features * grid_size) vs O(input_features * features) for Dense.
        The trade-off often favors KAN layers in terms of model depth and total parameters.
    """

    def __init__(
            self,
            features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            grid_range: Tuple[float, float] = (-2.0, 2.0),
            activation: str = 'swish',
            base_trainable: bool = True,
            spline_trainable: bool = True,
            kernel_initializer: Union[
                str, keras.initializers.Initializer
            ] = 'glorot_uniform',
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

        # Store configuration for serialization and reference
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

        # Attributes to be created in build() - initialized as None
        self.input_features = None
        self.grid = None
        self.spline_weight = None
        self.spline_scaler = None
        self.base_scaler = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create layer weights and B-spline grid based on input shape.

        This method is called automatically when the layer is first used with a specific
        input shape. It creates all trainable weights (spline coefficients and scalers)
        and initializes the B-spline knot sequence.

        Args:
            input_shape: Shape tuple of the input tensor. Must be at least 2D.
                The last dimension is used as input_features.

        Raises:
            ValueError: If input shape is less than 2D or input_features is None.
        """
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        self.input_features = input_shape[-1]
        if self.input_features is None:
            raise ValueError("Input features dimension cannot be None.")

        # Number of B-spline basis functions = grid_size + spline_order
        # This follows from the Cox-de Boor formula for B-splines
        num_basis_fns = self.grid_size + self.spline_order

        # Create spline coefficient weight: one set per (input, output, basis) triple
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.input_features, self.features, num_basis_fns),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        # Create scaling weights for the spline path
        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            trainable=self.spline_trainable,
        )

        # Create scaling weights for the base activation path
        self.base_scaler = self.add_weight(
            name="base_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            trainable=self.base_trainable,
        )

        # Initialize the B-spline grid (knot sequence)
        self._build_grid()
        super().build(input_shape)

    def _build_grid(self) -> None:
        """
        Build the extended B-spline knot sequence from the grid range.

        Creates a uniform grid over [grid_range[0], grid_range[1]] and extends it
        with additional knots on both ends to support B-spline basis functions of
        the specified order. The extended knots ensure basis functions are properly
        defined at the boundaries.

        The knot sequence structure:
        [extended_left ... grid_points ... extended_right]
        where extended regions have spline_order knots each.
        """
        # Create uniform grid points within the specified range
        grid_points = keras.ops.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size + 1
        )

        # Calculate grid spacing for uniform extension
        h = (grid_points[1] - grid_points[0])

        # Extend grid on the left: create spline_order knots before grid_points[0]
        # Cast integer range to float before multiplication for type consistency
        extended_knots_start = (
                keras.ops.cast(keras.ops.arange(-self.spline_order, 0), dtype=self.dtype)
                * h + grid_points[0]
        )

        # Extend grid on the right: create spline_order knots after grid_points[-1]
        extended_knots_end = (
                keras.ops.cast(keras.ops.arange(1, self.spline_order + 1), dtype=self.dtype)
                * h + grid_points[-1]
        )

        # Concatenate to form complete knot sequence
        self.grid = keras.ops.concatenate(
            [extended_knots_start, grid_points, extended_knots_end], axis=0
        )

    def _compute_bspline_basis(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute B-spline basis functions using Cox-de Boor recursion formula.

        Evaluates all B-spline basis functions at input points x using the iterative
        Cox-de Boor formula. Starts with piecewise constant basis (order 0) and
        recursively builds up to the desired spline order.

        Cox-de Boor formula:
        B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
        B_{i,k}(x) = ((x - t_i) / (t_{i+k} - t_i)) * B_{i,k-1}(x) +
                     ((t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})) * B_{i+1,k-1}(x)

        Args:
            x: Input tensor, shape (..., input_features). Values to evaluate basis at.

        Returns:
            Basis function values, shape (..., input_features, num_basis_functions).
            Each slice [..., i, :] contains evaluations of all basis functions for
            the i-th input feature.
        """
        # Add dimension for broadcasting with grid
        x = keras.ops.expand_dims(x, axis=-1)
        grid = self.grid

        # Base case k=0: piecewise constant basis functions
        # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
        basis = keras.ops.cast(
            keras.ops.logical_and(x >= grid[:-1], x < grid[1:]),
            dtype=self.dtype,
        )

        # Iteratively compute higher-order B-splines via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            # First term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
            term1_num = x - grid[:-(k + 1)]
            term1_den = grid[k:-1] - grid[:-(k + 1)]
            term1 = keras.ops.divide(term1_num, term1_den + self.epsilon) * basis[..., :-1]

            # Second term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            term2_num = grid[k + 1:] - x
            term2_den = grid[k + 1:] - grid[1:-k]
            term2 = keras.ops.divide(term2_num, term2_den + self.epsilon) * basis[..., 1:]

            # Combine terms for next order
            basis = term1 + term2

        return basis

    def call(
            self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: compute output using learned activation functions.

        Processes inputs through two parallel paths (base activation and B-spline),
        combines them with learned scaling weights, and aggregates across input features.

        Args:
            inputs: Input tensor, shape (batch_size, ..., input_features).
            training: Optional boolean, whether the call is in training mode.
                Not used in this layer but kept for API consistency.

        Returns:
            Output tensor, shape (batch_size, ..., features).
        """
        # Path 1: Apply base activation function to all inputs
        # Shape: (..., input_features)
        base_val = self.base_activation_fn(inputs)

        # Path 2: Compute B-spline activation functions
        # Evaluate B-spline basis functions at input values
        # Shape: (..., input_features, num_basis_fns)
        spline_basis = self._compute_bspline_basis(inputs)

        # Linear combination of basis functions using learned coefficients
        # Einsum performs: for each input feature i and output feature o,
        # spline_val[..., i, o] = sum_k spline_basis[..., i, k] * spline_weight[i, o, k]
        # Shape: (..., input_features, features)
        spline_val = keras.ops.einsum(
            '...ik,iok->...io', spline_basis, self.spline_weight
        )

        # Combine base and spline paths to form phi_ij(x_i)
        # Expand base_val to match scaler dimensions and apply learned scaling
        # Shape: (..., input_features, features)
        phi_base = keras.ops.expand_dims(base_val, axis=-1) * self.base_scaler
        phi_spline = spline_val * self.spline_scaler
        phi = phi_base + phi_spline

        # Aggregate over input features: sum_i phi_ij(x_i)
        # This produces the final output for each output feature j
        # Shape: (..., features)
        output = keras.ops.sum(phi, axis=-2)

        return output

    def update_grid_from_samples(self, x: keras.KerasTensor) -> None:
        """
        Update B-spline grid based on quantiles of input data.

        Adapts the B-spline knot positions to match the actual distribution of input
        values, potentially improving the layer's expressivity. This should be called
        periodically during training (e.g., every few epochs) with representative
        input data.

        Args:
            x: Input data tensor, shape (batch_size, input_features). Should be
                representative of the training distribution.

        Raises:
            ValueError: If input is not 2D.
        """
        if not isinstance(x, keras.KerasTensor):
            x = keras.ops.convert_to_tensor(x, dtype=self.dtype)

        if len(x.shape) != 2:
            raise ValueError("Input 'x' for grid update must be 2D.")

        # Sort each feature column to compute quantiles
        batch_size, num_features = keras.ops.shape(x)
        x_sorted = keras.ops.sort(x, axis=0)

        # Select grid_size+1 evenly spaced samples from sorted data
        # These samples become the new grid points
        indices = keras.ops.cast(
            keras.ops.linspace(0, batch_size - 1, self.grid_size + 1),
            dtype='int32'
        )
        grid_points_per_feature = keras.ops.take(x_sorted, indices, axis=0)

        # Average grid points across features to create unified grid
        # (Alternative: maintain per-feature grids for more flexibility)
        new_grid_points = keras.ops.mean(grid_points_per_feature, axis=1)

        # Update grid range and rebuild knot sequence
        self.grid_range = (new_grid_points[0], new_grid_points[-1])
        self._build_grid()

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape from input shape.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Shape tuple of output tensor, with last dimension changed to features.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        Returns serialization-compatible configuration containing all constructor
        arguments needed to recreate the layer. This enables saving and loading
        models containing this layer.

        Returns:
            Dictionary mapping constructor argument names to their values.
        """
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
