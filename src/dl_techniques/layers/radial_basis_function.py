# Copyright (c) 2025. All rights reserved.
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

import tensorflow as tf
import keras
from typing import Tuple, Any, Optional, Union
from keras.api.layers import Layer


class RBFLayer(Layer):
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
    ) -> None:
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

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Create the layer's weights with improved initialization.

        Args:
            input_shape: Tuple of integers, the shape of input tensor.

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
            dtype=self.dtype
        )

        # Create gamma (precision) parameter with non-negativity constraint
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.units,),
            initializer=tf.constant_initializer(self.gamma_init),
            constraint=keras.constraints.NonNeg(),
            trainable=self.trainable_gamma,
            dtype=self.dtype
        )

        super().build(input_shape)

    def _compute_pairwise_distances(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            epsilon: float = 1e-12
    ) -> tf.Tensor:
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
        x_norm = tf.reduce_sum(tf.square(x), axis=-1)
        y_norm = tf.reduce_sum(tf.square(y), axis=-1)

        # Reshape for broadcasting
        x_norm = tf.reshape(x_norm, [-1, 1])  # Shape: (n, 1)
        y_norm = tf.reshape(y_norm, [1, -1])  # Shape: (1, m)

        # Compute cross term
        cross_term = 2 * tf.matmul(x, y, transpose_b=True)

        # Combine terms with numeric stability
        distances = x_norm - cross_term + y_norm + epsilon

        return tf.maximum(distances, 0.0)  # Ensure non-negative

    def _compute_repulsion(self, centers: tf.Tensor) -> tf.Tensor:
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

        # Create mask to exclude self-distances
        mask = tf.ones_like(distances) - tf.eye(self.units)

        # Compute repulsion with improved formula
        sqrt_dist = tf.sqrt(distances)
        repulsion = tf.square(tf.maximum(0.0, effective_min_dist - sqrt_dist))

        # Scale repulsion by dimensionality
        dim_scale = tf.cast(self._feature_dim, tf.float32)
        scaled_repulsion = self.repulsion_strength * dim_scale * \
                           tf.reduce_mean(repulsion * mask)

        return scaled_repulsion

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
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
        gamma_expanded = tf.expand_dims(self.gamma, 0)  # Shape: (1, units)
        activations = tf.exp(-tf.minimum(gamma_expanded * distances, 50.0))

        # Add repulsion loss in training mode
        if training:
            self.add_loss(self._compute_repulsion(self.centers))

        return activations

    def get_config(self) -> dict:
        """Return layer configuration."""
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
    def center_positions(self) -> tf.Tensor:
        """Get current positions of RBF centers."""
        return tf.identity(self.centers)

    @property
    def width_values(self) -> tf.Tensor:
        """Get current width (gamma) values."""
        return tf.identity(self.gamma)