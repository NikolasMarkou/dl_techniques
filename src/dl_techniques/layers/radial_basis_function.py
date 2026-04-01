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
    """Radial Basis Function layer with adaptive center repulsion.

    Each of the ``units`` RBF neurons computes a Gaussian activation
    ``phi_i(x) = exp(-gamma_i * ||x - c_i||^2)`` measuring the proximity
    of the input ``x`` to a learnable center ``c_i``. The width parameter
    ``gamma_i`` is stored in raw (pre-softplus) form to guarantee
    positivity. During training an auxiliary repulsive penalty
    ``V_rep = alpha * D * max(0, d_min*(1+mu) - ||c_i - c_j||)^2``
    discourages centre collapse, ensuring broad coverage of the input
    space. Broadcasting-based distance computation supports inputs of
    arbitrary rank (2-D, 3-D, etc.).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [..., dim]                │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Expand dims → [..., 1, dim]     │
        │  Broadcast against centers       │
        │  [units, dim]                    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Squared Euclidean distance      │
        │  ||x - c_i||^2  → [..., units]  │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Gaussian activation             │
        │  exp(-gamma_i * dist^2)          │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Output [..., units]             │
        └──────────────────────────────────┘
        (+ center repulsion loss during training)

    :param units: Number of RBF units. Must be positive.
    :type units: int
    :param gamma_init: Initial value for the width parameter.
    :type gamma_init: float
    :param repulsion_strength: Strength of the center repulsion penalty.
    :type repulsion_strength: float
    :param min_center_distance: Minimum desired distance between centres.
    :type min_center_distance: float
    :param center_initializer: Initializer for RBF center positions.
    :type center_initializer: Union[str, keras.initializers.Initializer]
    :param center_constraint: Optional constraint for center positions.
    :type center_constraint: Optional[keras.constraints.Constraint]
    :param trainable_gamma: Whether the width parameters are trainable.
    :type trainable_gamma: bool
    :param safety_margin: Margin added to minimum distance threshold.
    :type safety_margin: float
    :param kernel_regularizer: Optional regularizer for center weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param gamma_regularizer: Optional regularizer for width parameters.
    :type gamma_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

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
        """Create layer weights (centers and raw gamma values).

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
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
        """Effective positive gamma values via softplus transformation.

        :return: Strictly positive width parameters.
        :rtype: keras.KerasTensor"""
        return keras.activations.softplus(self.gamma_raw)

    def _compute_repulsion_loss(self) -> keras.KerasTensor:
        """Compute the pairwise center repulsion regularisation loss.

        :return: Scalar regularisation loss tensor.
        :rtype: keras.KerasTensor"""
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
        """Forward pass computing Gaussian RBF activations.

        :param inputs: Input tensor of shape ``(batch, ..., dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool
        :return: RBF activations of shape ``(batch, ..., units)``.
        :rtype: keras.KerasTensor"""
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
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
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
        """Get current positions of RBF centers.

        :return: Center weight tensor or ``None`` if not built.
        :rtype: Optional[keras.KerasTensor]"""
        return self.centers

    @property
    def width_values(self) -> Optional[keras.KerasTensor]:
        """Get current effective width (gamma) values.

        :return: Effective gamma tensor or ``None`` if not built.
        :rtype: Optional[keras.KerasTensor]"""
        return self.gamma if self.built else None

# ---------------------------------------------------------------------
