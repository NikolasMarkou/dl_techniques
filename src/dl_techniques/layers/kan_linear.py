"""
A linear layer based on the Kolmogorov-Arnold Network (KAN) architecture.

This layer replaces the standard linear transformation followed by a fixed
activation function with a more expressive structure where the activation
functions themselves are learnable. It operationalizes the Kolmogorov-Arnold
representation theorem by parameterizing each activation as a combination of
basis functions (in this case, Gaussian RBFs).

Architecture:
    The layer's output is formed by two main components:
    1.  A standard linear transformation (`base_weight`), which provides a
        stable linear foundation for the mapping.
    2.  A non-linear transformation based on basis functions (`spline_weight`),
        which constitutes the learnable "activation".

    These two components are combined. If the input and output dimensions match,
    a gated residual connection is also added, which is crucial for stable
    training of deep KAN-based networks.

Foundational Mathematics:
    The core of KANs is the Kolmogorov-Arnold representation theorem. A KAN
    is a neural network realization of this theorem. This layer learns the
    univariate functions by representing them as a linear combination of basis
    functions. The coefficients of this combination are the learnable parameters,
    allowing the layer to approximate the optimal "activation function" for
    each connection in a data-driven manner.

Usage Recommendations:
    -   **Grid Size:** Start with a modest grid size (e.g., 5-10) to encourage
        smoother functions and avoid overfitting. Increase `grid_size` if the
        model underfits, but be mindful of the increased parameter count.
    -   **Network Architecture:** KANs are often more parameter-efficient than
        MLPs. It is recommended to start with a narrower and shallower
        architecture. For deeper KANs, ensuring residual connections are enabled
        is critical for effective gradient flow.

References:
    1.  Liu, Z., Wang, Y., et al. (2024). "KAN: Kolmogorov-Arnold Networks."
        arXiv preprint arXiv:2404.19756.
"""

import keras
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KANLinear(keras.layers.Layer):
    """Kolmogorov-Arnold Network Linear Layer with a stable architecture.

    This layer implements the core KAN linear transformation using a learned
    combination of basis functions, augmented with a traditional linear path
    and a robust residual connection.

    **Intent**: Implement a KAN-style layer that learns flexible activation
    functions while maintaining numerical stability, architectural consistency
    between training and inference, and proper Keras 3 serialization support.

    **Architecture**:
    ```
    Input(shape=[..., input_features])
           ↓
    ┌───────────────────┴───────────────────┐
    │              Spline Path              │      Linear Path
    │                  ↓                    │          ↓
    │   B-spline Basis: φ(x)                │   x @ W_base
    │                  ↓                    │          ↓
    │   Spline Transform: Σ(φ(x) ⊙ W_spline)│
    └──────────────────┬────────────────────┘
                       ↓
              Combine: base + scaled_spline
                       ↓
        (Optional) Gated Residual: + (1-g)*x
                       ↓
                 Activation: σ(combined)
                       ↓
           Output(shape=[..., features])
    ```
    The residual connection is only active if `use_residual=True` and
    input/output feature dimensions are the same.

    Args:
        features: Integer, number of output features.
        grid_size: Integer, number of grid points for the basis functions.
            Defaults to 5.
        spline_order: Integer, order of the basis functions. Defaults to 3.
        activation: String or callable, activation function. Defaults to 'swish'.
        regularization_factor: Float, L2 regularization factor. Defaults to 0.01.
        grid_range: Tuple of two floats, (min, max) range for the grid.
            Defaults to (-1, 1).
        use_residual: Boolean, whether to use a residual connection if input
            and output dimensions match. Defaults to True.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = 'swish',
        regularization_factor: float = 0.01,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        use_residual: bool = True,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = 'orthogonal',
        spline_initializer: Union[
            str, keras.initializers.Initializer
        ] = 'glorot_uniform',
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        spline_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        epsilon: float = 1e-7,
        clip_value: float = 1e3,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if features <= 0:
            raise ValueError("Features must be a positive integer.")
        if grid_size < spline_order:
            raise ValueError("Grid size must be >= spline order.")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("Invalid grid range: min must be less than max.")

        # Store ALL configuration parameters for serialization
        self.features = features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation_name = activation
        self.regularization_factor = regularization_factor
        self.grid_range = grid_range
        self.use_residual = use_residual
        self.epsilon = epsilon
        self.clip_value = clip_value

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.spline_initializer = keras.initializers.get(spline_initializer)
        self.kernel_regularizer = keras.regularizers.get(
            kernel_regularizer or keras.regularizers.L2(regularization_factor)
        )
        self.spline_regularizer = keras.regularizers.get(
            spline_regularizer or keras.regularizers.L2(regularization_factor)
        )
        self.activation_fn = keras.activations.get(activation)

        # Attributes to be created in build()
        self.input_features = None
        self.base_weight = None
        self.spline_weight = None
        self.spline_scaler = None
        self._cached_grid = None
        self._use_residual_actual = False

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and setup internal state."""
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got shape {input_shape}"
            )

        self.input_features = input_shape[-1]
        if self.input_features is None:
            raise ValueError("Input features dimension cannot be None.")

        # Determine if the residual connection can be used
        self._use_residual_actual = (
            self.use_residual and self.input_features == self.features
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=(self.input_features, self.features),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        num_basis_fns = self.grid_size + self.spline_order - 1
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=(self.input_features, self.features, num_basis_fns),
            initializer=self.spline_initializer,
            regularizer=self.spline_regularizer,
            trainable=True,
        )

        self.spline_scaler = self.add_weight(
            name="spline_scaler",
            shape=(self.input_features, self.features),
            initializer='ones',
            regularizer=self.spline_regularizer,
            constraint=keras.constraints.NonNeg(),
            trainable=True,
        )

        self._build_grid()
        super().build(input_shape)

    def _build_grid(self) -> None:
        """Build the grid points for basis functions."""
        if self._cached_grid is None:
            grid_min, grid_max = self.grid_range
            grid_points = keras.ops.linspace(
                grid_min, grid_max, self.grid_size
            )
            self._cached_grid = keras.ops.cast(grid_points, dtype=self.dtype)

    def _normalize_inputs(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Normalize inputs to the range [0, 1] for basis function mapping."""
        x = keras.ops.clip(x, self.grid_range[0], self.grid_range[1])
        x_normalized = (x - self.grid_range[0]) / (
            self.grid_range[1] - self.grid_range[0] + self.epsilon
        )
        return x_normalized

    def _compute_spline_basis(
        self, x: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute Gaussian RBF basis functions for numerical stability."""
        x_norm = self._normalize_inputs(x)
        x_expanded = keras.ops.expand_dims(x_norm, axis=-1)

        num_basis = self.grid_size + self.spline_order - 1
        basis_centers = keras.ops.linspace(0.0, 1.0, num_basis)
        basis_centers = keras.ops.reshape(basis_centers, (1, 1, num_basis))

        sigma = 1.0 / num_basis  # Heuristic for width
        basis_values = keras.ops.exp(
            -keras.ops.square(x_expanded - basis_centers)
            / (2 * sigma * sigma)
        )

        # Normalize basis functions to sum to 1 (partition of unity)
        basis_sum = keras.ops.sum(basis_values, axis=-1, keepdims=True)
        return basis_values / (basis_sum + self.epsilon)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for the KANLinear layer."""
        # Linear path
        base_output = keras.ops.matmul(inputs, self.base_weight)

        # Non-linear "spline" path
        spline_basis = self._compute_spline_basis(inputs)

        # FIX: The scaler should modulate the spline weights *before* the einsum.
        # This scales the contribution of each input-output spline function.
        scaled_spline_weight = self.spline_weight * keras.ops.expand_dims(
            self.spline_scaler, axis=-1
        )
        spline_output = keras.ops.einsum(
            '...ik,iok->...o', spline_basis, scaled_spline_weight
        )

        # Combine linear and non-linear paths
        total_output = base_output + spline_output

        # **CORRECTED RESIDUAL LOGIC**
        # Apply gated residual connection if dimensions match and it's enabled.
        if self._use_residual_actual:
            # FIX: The gate should be per-output-feature. We average the scalers
            # over the input dimension (axis=0) to get a gate for each output feature.
            gate = keras.ops.sigmoid(
                keras.ops.mean(self.spline_scaler, axis=0, keepdims=True)
            )
            total_output = gate * total_output + (1 - gate) * inputs

        # Apply final activation
        activated_output = self.activation_fn(total_output)

        # Apply gradient clipping only during training for stability
        if training:
            activated_output = keras.ops.clip(
                activated_output, -self.clip_value, self.clip_value
            )

        return activated_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the config of the layer for serialization."""
        config = super().get_config()
        config.update({
            "features": self.features,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "activation": self.activation_name,
            "regularization_factor": self.regularization_factor,
            "grid_range": self.grid_range,
            "use_residual": self.use_residual,
            "epsilon": self.epsilon,
            "clip_value": self.clip_value,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "spline_initializer": keras.initializers.serialize(
                self.spline_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "spline_regularizer": keras.regularizers.serialize(
                self.spline_regularizer
            ),
        })
        return config