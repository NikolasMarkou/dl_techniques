"""
Radial Basis Function Layer with Enhanced Center Repulsion
======================================================

Theory and Mathematical Foundation
-------------------------------
1. Basic RBF Network:
   A Radial Basis Function (RBF) network is a type of neural network that uses
   radial basis functions as activation functions. For an input x, each RBF unit
   computes:
       φᵢ(x) = exp(-γᵢ ||x - cᵢ||²)
   where:
       - cᵢ is the center vector for unit i
       - γᵢ is the width parameter (precision = 1/2σ²)
       - ||·||² denotes squared Euclidean distance

2. Enhanced Center Repulsion:
   To prevent center collapse and ensure better space coverage, we use an enhanced
   repulsion mechanism with adaptive strength:
       Vᵣₑₚ(cᵢ, cⱼ) = α * D * max(0, d_min * (1 + μ) - ||cᵢ - cⱼ||)²
   where:
       - α is the base repulsion strength
       - D is the input dimensionality (for scaling)
       - μ is a safety margin (typically 0.2)
       - d_min is the minimum desired distance between centers
       - ||cᵢ - cⱼ|| is the Euclidean distance between centers

3. Total Loss Function:
   The network is trained with a combined loss:
       L_total = L_task + α * D * L_repulsion
   where L_repulsion is the sum of repulsion potentials between all center pairs.

Implementation Details
--------------------
1. Numerical Stability:
   - Use squared distances with epsilon for numerical stability
   - Scale repulsion strength with input dimensionality
   - Add safety margin to minimum distance
   - Apply bounded constraints to parameters

2. Initialization:
   - Centers: Uniform distribution over expected data range
   - Widths: Constant value with non-negativity constraint
   - Repulsion parameters: Scaled by input dimension

3. Performance Optimization:
   - Efficient distance computation using broadcasting
   - Vectorized repulsion computation with masking
   - Carefully tuned hyperparameters for stability

References
---------
[1] Moody, J., & Darken, C. J. (1989). Fast learning in networks of
    locally-tuned processing units.
[2] Schwenker, F., Kestler, H. A., & Palm, G. (2001). Three learning phases
    for radial-basis-function networks.
[3] Bishop, C. M. (1995). Neural Networks for Pattern Recognition.
"""


import keras
from keras import ops
from typing import Any, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RBFLayer(keras.layers.Layer):
    """
    Enhanced Radial Basis Function Layer with Stable Center Repulsion.

    This layer implements RBF units with an improved repulsive force mechanism
    between centers to ensure better coverage of the input space. The implementation
    includes numerical stability improvements and adaptive repulsion strength.

    Args:
        units: Number of RBF units in the layer.
        gamma_init: Initial value for the width parameter (1/2σ²). Defaults to 1.0.
        repulsion_strength: Base weight of the repulsion term (α). Defaults to 0.1.
        min_center_distance: Minimum desired distance between centers. Defaults to 1.0.
        center_initializer: Initializer for centers. Defaults to uniform distribution.
        center_constraint: Optional constraint for centers.
        trainable_gamma: Whether γ is trainable. Defaults to True.
        safety_margin: Additional margin for minimum distance. Defaults to 0.2.

    Input shape:
        (batch_size, input_dim) or (batch_size, timesteps, input_dim)

    Output shape:
        (batch_size, units) or (batch_size, timesteps, units)
    """

    def __init__(
            self,
            units: int,
            gamma_init: float = 1.0,
            repulsion_strength: float = 0.1,
            min_center_distance: float = 1.0,
            center_initializer: Union[str, keras.initializers.Initializer] = 'uniform',
            center_constraint: Optional[keras.constraints.Constraint] = None,
            trainable_gamma: bool = True,
            safety_margin: float = 0.2,
            **kwargs: Any
    ):
        """Initialize the RBF layer with enhanced parameters."""
        # Input validation
        if units < 1:
            raise ValueError(f"Number of units must be positive, got {units}")
        if gamma_init <= 0:
            raise ValueError(f"gamma_init must be positive, got {gamma_init}")
        if repulsion_strength < 0:
            raise ValueError(
                f"repulsion_strength must be non-negative, got {repulsion_strength}"
            )
        if min_center_distance <= 0:
            raise ValueError(
                f"min_center_distance must be positive, got {min_center_distance}"
            )
        if safety_margin < 0:
            raise ValueError(
                f"safety_margin must be non-negative, got {safety_margin}"
            )

        super().__init__(**kwargs)

        # Store layer parameters
        self.units = units
        self.gamma_init = gamma_init
        self.repulsion_strength = repulsion_strength
        self.min_center_distance = min_center_distance
        self.safety_margin = safety_margin
        self.center_initializer = keras.initializers.get(center_initializer)
        self.center_constraint = keras.constraints.get(center_constraint)
        self.trainable_gamma = trainable_gamma

        # Initialize weights as None (will be created in build())
        self.centers = None
        self.gamma = None
        self._feature_dim = None  # Store dimensionality for scaling

    def build(self, input_shape):
        """
        Create the layer's weights with improved initialization.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.

        Raises:
            ValueError: If input_shape has less than 2 dimensions.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {len(input_shape)}"
            )

        # Store feature dimension for scaling
        self._feature_dim = input_shape[-1]

        # Create centers with shape (n_units, feature_dim)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self._feature_dim),
            initializer=self.center_initializer,
            constraint=self.center_constraint,
            trainable=True,
        )

        # Create gamma (precision) parameter with non-negativity constraint
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.units,),
            initializer=keras.initializers.Constant(self.gamma_init),
            constraint=ValueRangeConstraint(min_value=1e-5, max_value=None, clip_gradients=False),
            regularizer=keras.regularizers.L2(1e-5),
            trainable=self.trainable_gamma,
        )

        self.built = True

    def _compute_pairwise_distances(
            self,
            x,
            y,
            epsilon: float = 1e-12
    ):
        """
        Compute pairwise squared Euclidean distances with improved stability.

        Args:
            x: Tensor of shape (n, d)
            y: Tensor of shape (m, d)
            epsilon: Small constant for numerical stability

        Returns:
            Tensor of shape (n, m) containing squared distances.
        """
        # Compute squared norms
        x_norm = ops.sum(ops.square(x), axis=-1)
        y_norm = ops.sum(ops.square(y), axis=-1)

        # Reshape for broadcasting
        x_norm = ops.reshape(x_norm, [-1, 1])  # Shape: (n, 1)
        y_norm = ops.reshape(y_norm, [1, -1])  # Shape: (1, m)

        # Compute cross term
        cross_term = 2 * ops.matmul(x, ops.transpose(y))

        # Combine terms with numeric stability
        distances = x_norm - cross_term + y_norm + epsilon

        return ops.maximum(distances, 0.0)  # Ensure non-negative

    def _compute_repulsion(self, centers):
        """
        Compute the enhanced repulsion regularization term.

        Features improved stability and scaling with input dimensionality.

        Args:
            centers: Tensor of shape (n_units, feature_dim)

        Returns:
            Scalar tensor containing the total repulsion energy.
        """
        # Add safety margin to minimum distance
        effective_min_dist = self.min_center_distance * (1.0 + self.safety_margin)

        # Compute pairwise distances with stability
        distances = self._compute_pairwise_distances(centers, centers)

        # Create mask to exclude self-distances (diagonal elements)
        mask = ops.ones_like(distances) - ops.eye(self.units)

        # Compute repulsion with improved formula
        sqrt_dist = ops.sqrt(distances)
        repulsion = ops.square(ops.maximum(0.0, effective_min_dist - sqrt_dist))

        # Scale repulsion by dimensionality
        dim_scale = ops.cast(self._feature_dim, 'float32')
        scaled_repulsion = self.repulsion_strength * dim_scale * \
                           ops.mean(repulsion * mask)

        return scaled_repulsion

    def call(self, inputs, training=None):
        """
        Forward pass with improved numerical stability.

        Args:
            inputs: Input tensor of shape (batch_size, feature_dim)
            training: Boolean indicating training mode

        Returns:
            Tensor of shape (batch_size, units) containing RBF activations
        """
        # Compute distances with numeric stability
        distances = self._compute_pairwise_distances(inputs, self.centers)

        # Compute RBF activations with stable exponentiation
        # First expand gamma to match the shape needed for broadcasting
        gamma_expanded = ops.expand_dims(self.gamma, 0)  # Shape: (1, units)

        # Cap the exponent to prevent numerical overflow
        bounded_exponent = ops.minimum(gamma_expanded * distances, 50.0)

        # Calculate activations: exp(-γ||x-c||²)
        activations = ops.exp(-bounded_exponent)

        # Add repulsion loss in training mode
        if training:
            self.add_loss(self._compute_repulsion(self.centers))

        return activations

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape based on input shape.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Output shape with last dimension replaced by units
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'gamma_init': self.gamma_init,
            'repulsion_strength': self.repulsion_strength,
            'min_center_distance': self.min_center_distance,
            'safety_margin': self.safety_margin,
            'center_initializer':
                keras.initializers.serialize(self.center_initializer),
            'center_constraint':
                keras.constraints.serialize(self.center_constraint),
            'trainable_gamma': self.trainable_gamma,
        })
        return config

    @property
    def center_positions(self):
        """Get current positions of RBF centers."""
        return self.centers

    @property
    def width_values(self):
        """Get current width (gamma) values."""
        return self.gamma