"""
A Radial Basis Function (RBF) layer with center repulsion.

This layer implements a layer of Radial Basis Function units, which are
powerful for function approximation and pattern recognition tasks. Unlike
standard sigmoidal neurons, RBF units respond to localized regions of the
input space, making them effective at learning local features.

Architecture and Mathematical Foundation:
    The core of the RBF layer is a set of units, each with a 'center'
    vector that has the same dimensionality as the input. The activation of
    each unit is determined by the proximity of the input vector to its
    center. This relationship is formalized by the Gaussian RBF function:

    φᵢ(x) = exp(-γᵢ ||x - cᵢ||²)

    Where:
    - `x` is the input vector.
    - `cᵢ` is the center vector of the i-th RBF unit.
    - `γᵢ` is the trainable width (or precision) parameter for the i-th unit.
      It controls the radius of influence. A larger gamma results in a more
      localized, narrower response.
    - `||·||²` denotes the squared Euclidean distance.

    The output of the layer is a vector where each element is the activation
    `φᵢ(x)` from the corresponding RBF unit.

Enhanced Center Repulsion:
    To mitigate "center collapse" (where multiple centers converge to the
    same location), this implementation includes an adaptive repulsion
    mechanism.

    During training, a penalty term is added to the model's loss:

    V_rep(cᵢ, cⱼ) = α · D · max(0, d_min·(1 + μ) - ||cᵢ - cⱼ||)²

    This force ensures centers maintain a minimum separation, maximizing
    the coverage of the input space.

References:
    - Moody, J., & Darken, C. J. (1989). "Fast learning in networks of
      locally-tuned processing units."
    - Bishop, C. M. (1995). "Neural Networks for Pattern Recognition."
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RBFLayer(keras.layers.Layer):
    """
    Radial Basis Function layer with stable center repulsion mechanism.

    This layer implements RBF units with an improved repulsive force mechanism
    between centers to ensure better coverage of the input space. It utilizes
    broadcasting for distance calculations to support inputs of arbitrary rank
    (e.g., (batch, dim) or (batch, time, dim)) numerically stably.

    Attributes:
        units (int): Number of RBF units.
        gamma_init (float): Initial value for width parameter (1/2σ²).
        repulsion_strength (float): Strength of the repulsion penalty.
        min_center_distance (float): Minimum desired distance between centers.
        safety_margin (float): Margin added to minimum distance for repulsion.
        centers (keras.Variable): Weight matrix of center positions.
        gamma_raw (keras.Variable): Raw width parameters (pre-softplus).

    Args:
        units: Integer, number of RBF units in the layer. Must be positive.
        gamma_init: Float, initial value for the width parameter.
            Defaults to 1.0.
        repulsion_strength: Float, strength of center repulsion.
            Defaults to 0.1.
        min_center_distance: Float, minimum distance threshold for centers.
            Defaults to 1.0.
        center_initializer: Initializer for RBF centers.
            Defaults to 'uniform'.
        center_constraint: Constraint for center positions.
            Defaults to None.
        trainable_gamma: Boolean, whether width parameters are trainable.
            Defaults to True.
        safety_margin: Float, margin for repulsion calculation.
            Defaults to 0.2.
        kernel_regularizer: Regularizer for center weights.
            Defaults to None.
        gamma_regularizer: Regularizer for width parameters.
            Defaults to None.
        **kwargs: Standard Layer keyword arguments.
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
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        gamma_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if gamma_init <= 0:
            raise ValueError(f"gamma_init must be positive, got {gamma_init}")
        if repulsion_strength < 0:
            raise ValueError(f"repulsion_strength must be non-negative, got {repulsion_strength}")
        if min_center_distance <= 0:
            raise ValueError(f"min_center_distance must be positive, got {min_center_distance}")
        if safety_margin < 0:
            raise ValueError(f"safety_margin must be non-negative, got {safety_margin}")

        self.units = units
        self.gamma_init = gamma_init
        self.repulsion_strength = repulsion_strength
        self.min_center_distance = min_center_distance
        self.safety_margin = safety_margin
        self.trainable_gamma = trainable_gamma

        self.center_initializer = keras.initializers.get(center_initializer)
        self.center_constraint = keras.constraints.get(center_constraint)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)

        # State definitions
        self.centers: Optional[keras.Variable] = None
        self.gamma_raw: Optional[keras.Variable] = None
        self._feature_dim: int = 0

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create layer weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions, got {len(input_shape)}"
            )

        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError("The last dimension of the input must be defined.")

        self._feature_dim = feature_dim

        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self._feature_dim),
            initializer=self.center_initializer,
            constraint=self.center_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Calculate inverse softplus for initialization
        # softplus(x) = log(1 + exp(x)) -> x = log(exp(y) - 1)
        # We use numpy for stable constant calculation
        if self.gamma_init > 20.0:
            # For large values, softplus is approximately linear
            init_val = self.gamma_init
        else:
            init_val = np.log(np.exp(self.gamma_init) - 1.0)

        self.gamma_raw = self.add_weight(
            name='gamma_raw',
            shape=(self.units,),
            initializer=keras.initializers.Constant(init_val),
            regularizer=self.gamma_regularizer,
            trainable=self.trainable_gamma,
        )

        super().build(input_shape)

    @property
    def gamma(self) -> keras.KerasTensor:
        """
        Effective positive gamma values via softplus transformation.

        Returns:
            A tensor containing the strictly positive width parameters.
        """
        return keras.activations.softplus(self.gamma_raw)

    def _compute_repulsion_loss(self) -> keras.KerasTensor:
        """
        Compute the center repulsion loss.

        Calculates pairwise distances between centers and applies a penalty
        if they are closer than `min_center_distance * (1 + safety_margin)`.

        Returns:
            A scalar tensor representing the regularization loss.
        """
        # centers shape: (units, feature_dim)
        # Expand for broadcasting:
        # c1: (units, 1, feature_dim)
        # c2: (1, units, feature_dim)
        c1 = ops.expand_dims(self.centers, axis=1)
        c2 = ops.expand_dims(self.centers, axis=0)

        # Squared Euclidean distance between all pairs
        diff = c1 - c2
        # shape: (units, units)
        dist_sq = ops.sum(ops.square(diff), axis=-1)

        # Safe sqrt for gradient stability (avoid sqrt(0))
        dist = ops.sqrt(dist_sq + 1e-7)

        # Effective threshold
        threshold = self.min_center_distance * (1.0 + self.safety_margin)

        # Penalty: max(0, threshold - distance)^2
        penalty = ops.square(ops.maximum(0.0, threshold - dist))

        # Mask the diagonal (distance to self is 0, which would cause max penalty)
        eye_mask = ops.eye(self.units, dtype=self.compute_dtype)
        # Invert mask: 1.0 for off-diagonal, 0.0 for diagonal
        off_diag_mask = 1.0 - eye_mask

        masked_penalty = penalty * off_diag_mask

        # Average penalty over all pairs
        # We normalize by units^2 - units (number of off-diagonal elements)
        # or just mean over all and let the weight handle scaling.
        # Following original logic: scale by dim and strength.
        mean_penalty = ops.mean(masked_penalty)

        dim_scale = ops.cast(self._feature_dim, dtype=self.compute_dtype)

        return self.repulsion_strength * dim_scale * mean_penalty

    def call(
        self,
        inputs: keras.KerasTensor,
        training: bool = False
    ) -> keras.KerasTensor:
        """
        Forward pass of the RBF Layer.

        Args:
            inputs: Input tensor of shape `(batch_size, ... , dim)`.
            training: Boolean indicating whether the layer is in training mode.

        Returns:
            Output tensor of shape `(batch_size, ..., units)`.
        """
        # Inputs shape: (batch, ..., dim)
        # Centers shape: (units, dim)

        # We broaden inputs to (batch, ..., 1, dim) to broadcast against centers
        # This works for 2D inputs (batch, dim) -> (batch, 1, dim)
        # And 3D inputs (batch, time, dim) -> (batch, time, 1, dim)
        inputs_expanded = ops.expand_dims(inputs, axis=-2)

        # Squared difference: (batch, ..., units, dim)
        diff = inputs_expanded - self.centers

        # Squared Euclidean distance: (batch, ..., units)
        dist_sq = ops.sum(ops.square(diff), axis=-1)

        # Gamma broadcasting: (units,)
        # dist_sq is (batch, ..., units), gamma broadcasts automatically to last dim

        # Bounded exponent to prevent numerical underflow/overflow
        # exp(-50) is effectively 0
        exponent = ops.minimum(dist_sq * self.gamma, 50.0)

        output = ops.exp(-exponent)

        if training and self.units > 1 and self.repulsion_strength > 0:
            repulsion_loss = self._compute_repulsion_loss()
            self.add_loss(repulsion_loss)

        return output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples.

        Returns:
            Output shape tuple.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the config of the layer.

        Returns:
            A Python dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'gamma_init': self.gamma_init,
            'repulsion_strength': self.repulsion_strength,
            'min_center_distance': self.min_center_distance,
            'center_initializer': keras.initializers.serialize(self.center_initializer),
            'center_constraint': keras.constraints.serialize(self.center_constraint),
            'trainable_gamma': self.trainable_gamma,
            'safety_margin': self.safety_margin,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
        })
        return config


    # Convenience properties for inspection
    @property
    def center_positions(self) -> Optional[keras.KerasTensor]:
        """Get current positions of RBF centers."""
        return self.centers

    @property
    def width_values(self) -> Optional[keras.KerasTensor]:
        """Get current effective width (gamma) values."""
        return self.gamma if self.built else None

# ---------------------------------------------------------------------
