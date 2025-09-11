import keras
from keras import ops, layers
from typing import List, Dict, Any, Tuple, Optional
import tensorflow as tf  # Using TF backend for GradientTape


@keras.saving.register_keras_serializable()
class LagrangianNeuralNetworkLayer(keras.layers.Layer):
    """
    Physics-informed layer modeling system dynamics through learned Lagrangian mechanics.

    This layer implements the principles of "Lagrangian Neural Networks" (LNNs)
    as described in Cranmer et al. (arXiv:2003.04630). It uses an internal neural
    network to approximate the scalar Lagrangian `L` of a physical system from its
    generalized coordinates `q` and velocities `q_dot`, then computes accelerations
    `q_ddot` by numerically solving the Euler-Lagrange equation using automatic
    differentiation.

    **Intent**: Provide a physics-informed layer that preserves energy conservation
    by structure while learning system dynamics. Particularly useful when canonical
    momenta are unknown or difficult to compute, addressing limitations of
    Hamiltonian Neural Networks.

    **Architecture**:
    ```
    Inputs: q (coordinates), q_dot (velocities)
              ↓
    Concatenate: [q, q_dot]
              ↓
    MLP: [Hidden Layers] → Dense(1) → L (scalar Lagrangian)
              ↓
    Automatic Differentiation:
    - ∂L/∂q, ∂L/∂q̇
    - ∂²L/∂q̇² (Hessian)
    - ∂/∂q(∂L/∂q̇) (mixed derivatives)
              ↓
    Euler-Lagrange Solver: q̈ = H⁻¹[∂L/∂q - ∂/∂q(∂L/∂q̇)·q̇]
              ↓
    Output: q_ddot (accelerations)
    ```

    **Mathematical Operations**:
    The layer solves the vectorized Euler-Lagrange equation:

    ```
    q̈ = [∂²L/∂q̇²]⁻¹ · [∂L/∂q - ∂/∂q(∂L/∂q̇) · q̇]
    ```

    Where:
    - `L(q, q̇)` is the learned scalar Lagrangian
    - `∂L/∂q` is the gradient w.r.t. coordinates
    - `∂²L/∂q̇²` is the Hessian matrix w.r.t. velocities
    - `∂/∂q(∂L/∂q̇)` represents mixed partial derivatives
    - `·` denotes matrix multiplication
    - `⁻¹` denotes the pseudoinverse for numerical stability

    Args:
        hidden_dims: List of integers specifying units in each hidden layer
            of the internal MLP that approximates the Lagrangian. Must be
            non-empty with positive values.
        activation: Activation function for hidden layers. The paper recommends
            'softplus' as its second derivative is non-zero, unlike 'relu'.
            Defaults to 'softplus'.
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
        mlp: Internal Sequential model learning the scalar Lagrangian.

    Example:
        ```python
        # Create layer for 2D system
        lnn_layer = LagrangianNeuralNetworkLayer(
            hidden_dims=[128, 64, 32],
            activation='softplus'
        )

        # Define model inputs
        q = keras.Input(shape=(2,), name="coordinates")
        q_dot = keras.Input(shape=(2,), name="velocities")

        # Compute accelerations
        q_ddot = lnn_layer([q, q_dot])

        # Build complete model
        model = keras.Model(inputs=[q, q_dot], outputs=q_ddot)
        model.compile(optimizer='adam', loss='mse')
        ```

    Raises:
        ValueError: If hidden_dims is empty or contains non-positive integers.
        ValueError: If input shapes are incompatible during build/call.

    References:
        Cranmer, M., et al. "Lagrangian Neural Networks."
        arXiv preprint arXiv:2003.04630 (2020).
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

        # Create the internal MLP in __init__ (following modern Keras patterns)
        # This MLP learns the scalar Lagrangian L(q, q_dot)
        mlp_layers = []
        for units in self.hidden_dims:
            mlp_layers.append(
                layers.Dense(units, activation=self.activation)
            )
        # Final layer outputs single scalar value for the Lagrangian
        mlp_layers.append(
            layers.Dense(1, name="lagrangian_output")
        )

        self.mlp = keras.Sequential(mlp_layers, name="lagrangian_mlp")

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the internal MLP sub-layer with explicit shape information.

        This method follows modern Keras best practices by explicitly building
        sub-layers to ensure robust serialization and weight management.

        Args:
            input_shape: List of two shape tuples for [q, q_dot] inputs.

        Raises:
            ValueError: If input_shape format is incorrect or coordinate
                dimensions don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "Input to LagrangianNeuralNetworkLayer must be a list of "
                f"two tensors: [q, q_dot], got {len(input_shape)} inputs"
            )

        q_shape, q_dot_shape = input_shape

        if q_shape[-1] != q_dot_shape[-1]:
            raise ValueError(
                f"The last dimension (coord_dim) of q and q_dot must be equal, "
                f"but got {q_shape[-1]} and {q_dot_shape[-1]}"
            )

        # The MLP takes concatenated coordinates and velocities as input
        coord_dim = q_shape[-1]
        mlp_input_shape = (None, 2 * coord_dim)

        # Explicitly build the sub-layer (CRITICAL for serialization)
        if not self.mlp.built:
            self.mlp.build(mlp_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Forward pass computing accelerations from learned Lagrangian.

        Implements numerical solution to the Euler-Lagrange equation using
        automatic differentiation to compute required derivatives of the
        learned Lagrangian function.

        Args:
            inputs: List containing [q, q_dot] where:
                - q: Generalized coordinates
                - q_dot: Generalized velocities

        Returns:
            q_ddot: Computed generalized accelerations

        Note:
            Uses persistent GradientTape for multiple derivative computations.
            The tape is properly cleaned up after use to prevent memory leaks.
        """
        q, q_dot = inputs

        # Use TensorFlow's GradientTape for automatic differentiation
        # Persistent=True allows multiple gradient computations
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
            tape2.watch(q)
            tape2.watch(q_dot)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(q_dot)

                # 1. Compute the scalar Lagrangian L(q, q_dot)
                coords_and_velocities = ops.concatenate([q, q_dot], axis=-1)
                lagrangian = self.mlp(coords_and_velocities)
                lagrangian = ops.squeeze(lagrangian, axis=-1)  # Shape: (batch,)

            # 2. Compute first-order derivatives
            # ∂L/∂q_dot (gradient w.r.t. velocities)
            grad_L_q_dot = tape1.gradient(lagrangian, q_dot)

        # 3. Compute second-order derivatives using the persistent tape
        # ∂²L/∂q_dot² (Hessian matrix w.r.t. velocities)
        hessian_L_q_dot = tape2.jacobian(grad_L_q_dot, q_dot)

        # ∂L/∂q (gradient w.r.t. coordinates)
        grad_L_q = tape2.gradient(lagrangian, q)

        # ∂/∂q(∂L/∂q_dot) (mixed partial derivatives)
        jac_q_grad_q_dot = tape2.jacobian(grad_L_q_dot, q)

        # Clean up persistent tape to prevent memory leaks
        del tape2

        # 4. Solve the Euler-Lagrange equation
        # q̈ = H⁻¹ · [∂L/∂q - ∂/∂q(∂L/∂q̇) · q̇]

        # Compute pseudoinverse of Hessian for numerical stability
        inverse_hessian = tf.linalg.pinv(hessian_L_q_dot)

        # Compute the "force" term: F = ∂L/∂q - ∂/∂q(∂L/∂q̇) · q̇
        q_dot_expanded = ops.expand_dims(q_dot, axis=-1)  # (batch, coord_dim, 1)
        mixed_term = ops.matmul(jac_q_grad_q_dot, q_dot_expanded)
        mixed_term_squeezed = ops.squeeze(mixed_term, axis=-1)  # (batch, coord_dim)

        force = grad_L_q - mixed_term_squeezed
        force_expanded = ops.expand_dims(force, axis=-1)  # (batch, coord_dim, 1)

        # 5. Solve for accelerations: q̈ = H⁻¹ · F
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