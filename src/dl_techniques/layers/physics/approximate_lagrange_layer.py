import keras
from keras import ops, layers
from typing import List, Dict, Any, Tuple, Optional


@keras.saving.register_keras_serializable()
class ApproximatedLNNLayer(keras.layers.Layer):
    """
    Gradient-tape-free approximation of Lagrangian Neural Network dynamics.

    Provides a computationally efficient alternative to gradient-tape-based LNNs
    by directly learning the three components of the Euler-Lagrange equation
    with separate MLPs: ``dL/dq``, ``(d^2L/dq_dot^2)^{-1}``, and
    ``d/dq(dL/dq_dot)``. These are assembled as
    ``q_ddot = H^{-1} * [dL/dq - J * q_dot]`` without nested automatic
    differentiation, trading increased parameter count (3x MLP) for significantly
    faster inference.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │          ApproximatedLNNLayer                │
        │                                              │
        │  q, q_dot ──► Concatenate([q, q_dot])        │
        │                    │                         │
        │           ┌───────┼───────┐                  │
        │           ▼       ▼       ▼                  │
        │     ┌─────────┐┌──────┐┌─────────┐          │
        │     │MLP_grad ││MLP_H ││MLP_mixed│          │
        │     │ dL/dq   ││H^{-1}││ J_mixed │          │
        │     └────┬────┘└──┬───┘└────┬────┘          │
        │          │        │         │                │
        │          ▼        ▼         ▼                │
        │     Euler-Lagrange Assembly:                 │
        │     q_ddot = H^{-1} * [dL/dq - J * q_dot]   │
        │                    │                         │
        │                    ▼                         │
        │              Output: q_ddot                  │
        └──────────────────────────────────────────────┘

    :param hidden_dims: List of integers specifying units in each hidden layer
        for all three internal MLPs. Must be non-empty with positive values.
    :type hidden_dims: List[int]
    :param activation: Activation function for hidden layers. Defaults to ``'softplus'``.
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

        :param name: Name for the Sequential model.
        :type name: str
        :param output_units: Number of units in output layer. None defers to ``build()``.
        :type output_units: Optional[int]
        :return: Sequential model with specified architecture.
        :rtype: keras.Sequential
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

        :param input_shape: List of two shape tuples for ``[q, q_dot]`` inputs.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If input_shape format is incorrect or dimensions don't match.
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

        :param inputs: List containing ``[q, q_dot]``.
        :type inputs: List[keras.KerasTensor]
        :return: Computed generalized accelerations ``q_ddot``.
        :rtype: keras.KerasTensor
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