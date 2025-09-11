import keras
from keras import ops, layers
from typing import List, Dict, Any, Tuple, Optional


@keras.saving.register_keras_serializable()
class ApproximatedLNNLayer(keras.layers.Layer):
    """
    Gradient-tape-free approximation of Lagrangian Neural Network dynamics.

    This layer provides a computationally efficient alternative to traditional
    Lagrangian Neural Networks by directly learning the components required for
    the Euler-Lagrange equation without relying on automatic differentiation.
    It uses separate neural networks to approximate the gradient of the Lagrangian,
    the inverse Hessian matrix, and mixed partial derivatives, then assembles
    these components to solve for system accelerations.

    **Intent**: Deliver performance improvements over gradient-tape-based LNNs,
    especially on hardware where higher-order automatic differentiation is not
    optimized, while maintaining the physics-informed structure that preserves
    energy conservation properties.

    **Architecture**:
    ```
    Inputs: q (coordinates), q_dot (velocities)
              ↓
    Concatenate: [q, q_dot] → shared_input
              ↓
    ┌─ MLP_grad_L_q → ∂L/∂q
    ├─ MLP_inv_hessian → (∂²L/∂q̇²)⁻¹
    └─ MLP_mixed_derivs → ∂/∂q(∂L/∂q̇)
              ↓
    Euler-Lagrange Assembly:
    q̈ = (∂²L/∂q̇²)⁻¹ · [∂L/∂q - ∂/∂q(∂L/∂q̇) · q̇]
              ↓
    Output: q_ddot (accelerations)
    ```

    **Mathematical Operations**:
    The layer directly approximates the components of the Euler-Lagrange equation:

    1. **Gradient approximation**: `∂L/∂q ≈ MLP_grad(q, q̇)`
    2. **Inverse Hessian approximation**: `(∂²L/∂q̇²)⁻¹ ≈ MLP_inv_hess(q, q̇)`
    3. **Mixed derivatives**: `∂/∂q(∂L/∂q̇) ≈ MLP_mixed(q, q̇)`
    4. **Assembly**: `q̈ = H⁻¹ · [∇_q L - J_q(∇_{q̇} L) · q̇]`

    Where each MLP learns its respective component directly from training data,
    avoiding the computational overhead of nested automatic differentiation.

    Args:
        hidden_dims: List of integers specifying the number of units in each
            hidden layer for all internal MLPs. Must be non-empty with positive
            values. All three approximation networks share this architecture.
        activation: Activation function for hidden layers across all MLPs.
            Defaults to 'softplus' for consistency with Lagrangian mechanics
            where second derivatives should be well-defined.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        A list/tuple of two tensors: `[q, q_dot]`.
        - `q`: Generalized coordinates, shape `(batch_size, ..., coord_dim)`
        - `q_dot`: Generalized velocities, shape `(batch_size, ..., coord_dim)`
        Both tensors must have identical shapes.

    Output shape:
        Generalized accelerations `q_ddot` with same shape as inputs:
        `(batch_size, ..., coord_dim)`.

    Attributes:
        grad_L_q_mlp: Sequential model approximating ∂L/∂q.
        inverse_hessian_mlp: Sequential model approximating (∂²L/∂q̇²)⁻¹.
        jac_q_grad_q_dot_mlp: Sequential model approximating ∂/∂q(∂L/∂q̇).

    Example:
        ```python
        # Create approximated LNN for 3D system
        approx_lnn = ApproximatedLNNLayer(
            hidden_dims=[256, 128, 64],
            activation='softplus'
        )

        # Define system inputs
        q = keras.Input(shape=(3,), name="coordinates")
        q_dot = keras.Input(shape=(3,), name="velocities")

        # Compute approximated accelerations
        q_ddot = approx_lnn([q, q_dot])

        # Build and compile model
        model = keras.Model(inputs=[q, q_dot], outputs=q_ddot)
        model.compile(optimizer='adam', loss='mse')
        ```

    Performance Notes:
        - Significantly faster than gradient-tape-based LNNs during inference
        - Requires more parameters (3 × MLP parameters vs 1 × MLP)
        - Training may require careful initialization and loss weighting
        - Best suited for systems where gradient computation is expensive

    Raises:
        ValueError: If hidden_dims is empty or contains non-positive integers.
        ValueError: If input shapes are incompatible during build/call.
    """

    def __init__(
            self,
            hidden_dims: List[int],
            activation: str = 'softplus',
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate and store configuration
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError(
                f"hidden_dims must be a non-empty list of positive integers, "
                f"got {hidden_dims}"
            )

        self.hidden_dims = hidden_dims
        self.activation = activation

        # Create all three MLP approximators in __init__ (modern Keras pattern)
        # Each MLP will be completed with appropriate output layers in build()

        # MLP to approximate ∂L/∂q (gradient w.r.t. coordinates)
        self.grad_L_q_mlp = self._create_complete_mlp(
            name="grad_L_q_mlp",
            output_units=None  # Will be set in build()
        )

        # MLP to approximate (∂²L/∂q̇²)⁻¹ (inverse Hessian w.r.t. velocities)
        self.inverse_hessian_mlp = self._create_complete_mlp(
            name="inverse_hessian_mlp",
            output_units=None  # Will be set in build()
        )

        # MLP to approximate ∂/∂q(∂L/∂q̇) (mixed partial derivatives)
        self.jac_q_grad_q_dot_mlp = self._create_complete_mlp(
            name="jac_q_grad_q_dot_mlp",
            output_units=None  # Will be set in build()
        )

    def _create_complete_mlp(
            self,
            name: str,
            output_units: Optional[int]
    ) -> keras.Sequential:
        """
        Create a complete MLP with hidden layers and optional output layer.

        Args:
            name: Name for the Sequential model.
            output_units: Number of units in output layer. If None, no output
                layer is added (will be added later in build()).

        Returns:
            Sequential model with specified architecture.
        """
        mlp_layers = []

        # Add hidden layers
        for units in self.hidden_dims:
            mlp_layers.append(
                layers.Dense(units, activation=self.activation)
            )

        # Add output layer if units specified
        if output_units is not None:
            mlp_layers.append(
                layers.Dense(output_units, name=f"{name}_output")
            )

        return keras.Sequential(mlp_layers, name=name)

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build all internal MLP sub-layers with proper output dimensions.

        This method completes the MLP architectures by adding output layers
        with appropriate dimensions based on the coordinate space, then
        explicitly builds all sub-layers for robust serialization.

        Args:
            input_shape: List of two shape tuples for [q, q_dot] inputs.

        Raises:
            ValueError: If input_shape format is incorrect or coordinate
                dimensions don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "Input to ApproximatedLNNLayer must be a list of "
                f"two tensors: [q, q_dot], got {len(input_shape)} inputs"
            )

        q_shape, q_dot_shape = input_shape

        if q_shape[-1] != q_dot_shape[-1]:
            raise ValueError(
                f"The last dimension (coord_dim) of q and q_dot must be equal, "
                f"but got {q_shape[-1]} and {q_dot_shape[-1]}"
            )

        coord_dim = q_shape[-1]
        mlp_input_shape = (None, 2 * coord_dim)

        # Complete and build the gradient MLP (outputs coord_dim values)
        if not self.grad_L_q_mlp.built:
            self.grad_L_q_mlp.add(
                layers.Dense(coord_dim, name="grad_L_q_output")
            )
            self.grad_L_q_mlp.build(mlp_input_shape)

        # Complete and build the inverse Hessian MLP (outputs coord_dim² values)
        if not self.inverse_hessian_mlp.built:
            self.inverse_hessian_mlp.add(
                layers.Dense(coord_dim * coord_dim, name="inverse_hessian_output")
            )
            self.inverse_hessian_mlp.build(mlp_input_shape)

        # Complete and build the mixed derivatives MLP (outputs coord_dim² values)
        if not self.jac_q_grad_q_dot_mlp.built:
            self.jac_q_grad_q_dot_mlp.add(
                layers.Dense(coord_dim * coord_dim, name="jac_q_grad_q_dot_output")
            )
            self.jac_q_grad_q_dot_mlp.build(mlp_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Forward pass computing accelerations from approximated components.

        Assembles the Euler-Lagrange equation using the three learned
        approximations without requiring gradient computation.

        Args:
            inputs: List containing [q, q_dot] where:
                - q: Generalized coordinates
                - q_dot: Generalized velocities

        Returns:
            q_ddot: Computed generalized accelerations

        Note:
            All matrix operations use batched operations for efficiency.
            The inverse Hessian and Jacobian outputs are reshaped from
            flat vectors to matrices before use.
        """
        q, q_dot = inputs
        coord_dim = ops.shape(q)[-1]

        # Prepare shared input for all MLPs
        coords_and_velocities = ops.concatenate([q, q_dot], axis=-1)

        # 1. Approximate all required components using the MLPs

        # Approximate ∂L/∂q (gradient w.r.t. coordinates)
        grad_L_q = self.grad_L_q_mlp(coords_and_velocities)

        # Approximate (∂²L/∂q̇²)⁻¹ (inverse Hessian matrix)
        inverse_hessian_flat = self.inverse_hessian_mlp(coords_and_velocities)
        inverse_hessian = ops.reshape(
            inverse_hessian_flat,
            (-1, coord_dim, coord_dim)
        )

        # Approximate ∂/∂q(∂L/∂q̇) (mixed partial derivatives)
        jac_q_grad_q_dot_flat = self.jac_q_grad_q_dot_mlp(coords_and_velocities)
        jac_q_grad_q_dot = ops.reshape(
            jac_q_grad_q_dot_flat,
            (-1, coord_dim, coord_dim)
        )

        # 2. Assemble the Euler-Lagrange equation
        # Compute the mixed term: ∂/∂q(∂L/∂q̇) · q̇
        q_dot_expanded = ops.expand_dims(q_dot, axis=-1)  # (batch, coord_dim, 1)
        mixed_term = ops.matmul(jac_q_grad_q_dot, q_dot_expanded)
        mixed_term_squeezed = ops.squeeze(mixed_term, axis=-1)  # (batch, coord_dim)

        # Compute the "force" term: F = ∂L/∂q - ∂/∂q(∂L/∂q̇) · q̇
        force = grad_L_q - mixed_term_squeezed
        force_expanded = ops.expand_dims(force, axis=-1)  # (batch, coord_dim, 1)

        # 3. Solve for accelerations: q̈ = (∂²L/∂q̇²)⁻¹ · F
        q_ddot_expanded = ops.matmul(inverse_hessian, force_expanded)
        q_ddot = ops.squeeze(q_ddot_expanded, axis=-1)  # (batch, coord_dim)

        return q_ddot

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for accelerations.

        Args:
            input_shape: List of input shapes [q_shape, q_dot_shape].

        Returns:
            Output shape tuple, identical to coordinate input shape.
        """
        q_shape = input_shape[0]
        return q_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters needed
            to recreate this layer instance.
        """
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
        })
        return config