import keras
from keras import ops, layers
from typing import List, Dict, Any, Tuple, Optional
import tensorflow as tf  # Using TF backend for GradientTape


@keras.saving.register_keras_serializable()
class LagrangianNeuralNetworkLayer(keras.layers.Layer):
    """
    Physics-informed layer modeling dynamics through learned Lagrangian mechanics.

    Implements "Lagrangian Neural Networks" (Cranmer et al., arXiv:2003.04630).
    An internal MLP approximates the scalar Lagrangian ``L(q, q_dot)`` from
    generalized coordinates and velocities. Accelerations are computed by
    numerically solving the Euler-Lagrange equation using automatic differentiation:
    ``q_ddot = [d^2L/dq_dot^2]^{-1} * [dL/dq - d/dq(dL/dq_dot) * q_dot]``.
    The pseudoinverse is used for numerical stability.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────────┐
        │       LagrangianNeuralNetworkLayer                │
        │                                                   │
        │  q, q_dot ──► Concatenate([q, q_dot])             │
        │                    │                              │
        │                    ▼                              │
        │              ┌───────────┐                        │
        │              │ MLP → L   │  (scalar Lagrangian)   │
        │              └─────┬─────┘                        │
        │                    │                              │
        │                    ▼                              │
        │         Automatic Differentiation                 │
        │         ├─ dL/dq        (gradient)                │
        │         ├─ d²L/dq_dot²  (Hessian)                 │
        │         └─ d/dq(dL/dq_dot) (mixed)                │
        │                    │                              │
        │                    ▼                              │
        │         Euler-Lagrange Solver                     │
        │         q_ddot = H^{-1} * [dL/dq - J*q_dot]       │
        │                    │                              │
        │                    ▼                              │
        │              Output: q_ddot                       │
        └───────────────────────────────────────────────────┘

    :param hidden_dims: List of integers specifying units in each hidden layer
        of the internal MLP. Must be non-empty with positive values.
    :type hidden_dims: List[int]
    :param activation: Activation function for hidden layers. ``'softplus'``
        recommended as its second derivative is non-zero. Defaults to ``'softplus'``.
    :type activation: str
    :param kwargs: Additional arguments for the Layer base class.
    :type kwargs: Any
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

        :param input_shape: List of two shape tuples for ``[q, q_dot]`` inputs.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If input_shape format is incorrect or dimensions don't match.
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

        :param inputs: List containing ``[q, q_dot]``.
        :type inputs: List[keras.KerasTensor]
        :return: Computed generalized accelerations ``q_ddot``.
        :rtype: keras.KerasTensor
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

        :param input_shape: List of input shapes ``[q_shape, q_dot_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape tuple, identical to coordinate input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        q_shape = input_shape[0]
        return q_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
        })
        return config