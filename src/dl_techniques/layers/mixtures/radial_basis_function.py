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
from typing import Literal, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------

from ...utils.tensors import resolve_training_factor

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
        │  ||x - c_i||^2  → [..., units]   │
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
    :param output_mode: Output normalization mode. ``'basis'`` (the default)
        returns the raw unnormalized Gaussian activations
        ``phi_k = exp(-gamma_k * ||x - c_k||^2)``, with the exponent clipped at
        50.0.

        .. warning::

           **The 50.0 clip is a hazard on the ``'basis'`` path, not a
           safeguard.** It does two things, and only the first is desirable: it
           keeps ``exp`` from underflowing (the floor is ``exp(-50) ~ 1.93e-22``
           rather than ``0.0``), AND it freezes learning once saturated, because
           ``ops.minimum`` has a structurally ZERO gradient in its saturated
           branch. Since ``E[||x - c||^2] ~ D`` for standardized inputs, every
           unit saturates at the same 50.0 once ``D * gamma >~ 50``, and the
           layer becomes a constant ``exp(-50)`` with gradient exactly 0.0 on
           both ``centers`` and ``gamma_raw``. Measured with stock defaults
           (``units=8``, ``gamma_init=1.0``) on ``normal(0, 1)`` input: gradmax
           is ``0.0 / 0.0`` at ``D >= 64``, and a 40-epoch binary fit of
           ``Sequential([RBFLayer, Dense(1)])`` at ``D = 128`` leaves the loss at
           chance (``0.693``) with ``d|gamma_raw|`` exactly ``0.0``. This is
           long-standing behavior, deliberately NOT changed here (changing the
           default arm of a shipped layer needs its own plan); it is pinned by
           ``test_basis_mode_is_dead_at_realistic_dimension``, an
           ``xfail(strict=True)`` test that will FAIL loudly the moment someone
           fixes it. Prefer ``'normalized'``, or keep ``D * gamma`` well below
           50, if you need this layer to train.

        ``'normalized'`` returns the Normalized RBF (NRBF) activations
        ``phi_k / sum_j phi_j``, which sum to 1.0 along the last axis. The
        normalized arm is computed as a softmax over the **unclipped** exponent
        and is therefore exactly the textbook NRBF: far from every center it
        selects the nearest one, rather than degenerating to a uniform
        ``1/units``. It carries no clip and therefore no gradient plateau, but
        it does expect approximately **standardized inputs**: because softmax
        saturates to one-hot once the gap between the two nearest units exceeds
        ~88, large ``gamma`` or large input scale collapses its gradients by
        orders of magnitude without ever reaching exact zero (measured at
        ``D = 128``: gradmax ``4.3e-01`` at ``gamma=20, scale=10`` versus
        ``5.3e-04`` at ``gamma=20, scale=30``). It also has one true failure
        point: the unclipped exponent overflows float32 when
        ``D * gamma * max|x|^2 > 3.4e38``, making the row all ``-inf`` and the
        softmax NaN. Note the threshold is a function of ``D`` and ``gamma``,
        NOT a fixed input magnitude — at ``D = 1024`` or ``gamma = 100`` it is
        reached a decade earlier in ``|x|`` than at ``D = 128, gamma = 1``. No
        plausible input reaches it (``normal(0, 1) * 1e7`` is finite), and
        re-clipping to prevent it would reintroduce exactly the plateau
        described above, so this is disclosed rather than repaired (D-008).

        Note this vocabulary is deliberately DISJOINT
        from ``GMMLayer``/``KMeansLayer``'s ``{'assignments', 'mixture'}``: RBF
        has no reconstruction-mode analogue, and its normalized output is a
        normalized basis activation, not a posterior or cluster assignment.
    :type output_mode: Literal['basis', 'normalized']
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
        output_mode: Literal['basis', 'normalized'] = 'basis',
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
        if output_mode not in ['basis', 'normalized']:
            raise ValueError(
                f"output_mode must be 'basis' or 'normalized', got {output_mode}"
            )

        self.units = units
        self.output_mode = output_mode
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

        # Mixed-precision: autocast=False keeps centers in variable_dtype (float32) inside
        # call() under a mixed_float16 policy, so the distance / exp math runs in float32
        # (matching the float32 inputs cast) and the output is cast to compute_dtype on
        # return. Uniform with GMMLayer / KMeansLayer. No-op under the float32 policy.
        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self._feature_dim),
            initializer=self.center_initializer,
            constraint=self.center_constraint,
            regularizer=self.kernel_regularizer,
            trainable=True,
            autocast=False,
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
            autocast=False,  # mixed-precision: keep float32 for the kernel math
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

        # Mask the diagonal (distance to self is 0, which would cause max penalty).
        # variable_dtype (float32) to match the autocast=False centers under mixed precision.
        eye_mask = ops.eye(self.units, dtype=self.variable_dtype)
        # Invert mask: 1.0 for off-diagonal, 0.0 for diagonal
        off_diag_mask = 1.0 - eye_mask

        masked_penalty = penalty * off_diag_mask

        # Average penalty over all pairs
        # We normalize by units^2 - units (number of off-diagonal elements)
        # or just mean over all and let the weight handle scaling.
        # Following original logic: scale by dim and strength.
        mean_penalty = ops.mean(masked_penalty)

        dim_scale = ops.cast(self._feature_dim, dtype=self.variable_dtype)

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
        :return: RBF activations of shape ``(batch, ..., units)``. Unnormalized
            under ``output_mode='basis'``; summing to 1.0 along the last axis
            under ``output_mode='normalized'``.
        :rtype: keras.KerasTensor"""
        # Inputs shape: (batch, ..., dim)
        # Centers shape: (units, dim)

        # Cast inputs to variable_dtype (float32) so the distance / exp math runs in full
        # precision and matches the autocast=False centers under a mixed_float16 policy.
        # The output is cast back to compute_dtype before returning (no-op under float32).
        inputs = ops.cast(inputs, self.variable_dtype)

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

        # log(phi_k) = -gamma_k * ||x - c_k||^2, UNCLIPPED. The 50.0 clip belongs to
        # the 'basis' arm alone (applied below, at the ops.exp) -- see the D-002 anchor.
        scaled_dist_sq = dist_sq * self.gamma

        # DECISION plan-2026-07-20T160907-7de371a1/D-002: NRBF is a softmax over the
        # UNCLIPPED, PRE-exp `-scaled_dist_sq` (which IS log(phi)), computed here in
        # variable_dtype (float32) and strictly ABOVE the ops.cast(output,
        # self.compute_dtype) below. Three rewrites are real bugs, not style choices:
        #
        # 1. Do NOT feed `ops.minimum(scaled_dist_sq, 50.0)` into the softmax. The clip
        #    exists only to keep ops.exp from underflowing in the 'basis' arm; softmax
        #    never calls exp on the raw value and is internally shift-stabilized, so it
        #    needs no clip. Feeding it the CLIPPED value makes this arm a DEAD LAYER at
        #    ordinary feature dimensions: E[dist_sq] ~ D, so once D*gamma >~ 50 every
        #    unit saturates at the same 50.0, softmax sees a constant vector and returns
        #    uniform 1/units -- and ops.minimum has a STRUCTURAL ZERO gradient in the
        #    saturated branch, so `centers` and `gamma_raw` both get gradient exactly
        #    0.0. Measured at D=128 with stock defaults: output [0.16667]*6, gradmax
        #    0.0/0.0. It also destroys NRBF's defining property -- selecting the nearest
        #    center far from the data -- which is the whole reason to prefer it over
        #    'basis'. This shipped once (D-008); the test that should have caught it ran
        #    at dim=4, below the saturation threshold.
        # 2. Do NOT rewrite as `output / ops.sum(output, axis=-1, keepdims=True)`, and
        # 3. do NOT move the normalization below the cast.
        #    (2) and (3) are the same NaN: phi underflows to EXACT 0.0 in float16 for
        #    ordinary inputs (normal(0,1) in 16 dims already gives phi ~ 1.1e-7), so
        #    under a mixed_float16 policy a post-cast division is 0/0 -> NaN. Reproduced
        #    live; see findings/rbf-normalization.md F8 and D-007. softmax is
        #    shift-invariant -- its largest term is always exp(0)=1, so its denominator
        #    is always >= 1 and cannot vanish -- and over the unclipped exponent it is
        #    EXACTLY phi_k/sum_j phi_j, not an approximation.
        #
        # `training` is not consulted: normalization is identical in train and inference.
        if self.output_mode == 'normalized':
            output = ops.softmax(-scaled_dist_sq, axis=-1)
        else:
            # DECISION plan-2026-07-20T160907-7de371a1/D-012: this 50.0 clip is a
            # KNOWN DEFECT, knowingly left in place. Do NOT read it as protection.
            # It bounds the exponent so ops.exp cannot underflow -- and it is also
            # the SAME structural zero-gradient plateau that D-002/D-008 removed
            # from the 'normalized' arm directly above. ops.minimum passes zero
            # gradient through its saturated branch, so once D * gamma >~ 50 (i.e.
            # at ORDINARY feature dimensions, since E[||x - c||^2] ~ D) every unit
            # pins to the constant exp(-50) = 1.929e-22 and both `centers` and
            # `gamma_raw` receive gradient EXACTLY 0.0. Measured, stock defaults,
            # normal(0,1) input: gradmax 0.0/0.0 at D in {64, 128, 256}; a 40-epoch
            # binary fit at D=128 sits at chance loss 0.693 with d|gamma_raw| = 0.0.
            #
            # Why it is still here: 'basis' is the DEFAULT mode of a shipped layer
            # and this behavior is long-standing and byte-identical to what shipped
            # before D-008 -- so removing the clip is a production behavior change
            # to every existing caller, not a bug fix, and it needs its own plan,
            # pre-mortem and review. Do NOT "just" delete the minimum() here.
            #
            # It is pinned, not merely documented: see
            # tests/test_layers/test_mixtures/test_radial_basis_function.py
            # ::test_basis_mode_is_dead_at_realistic_dimension, an
            # xfail(strict=True) end-to-end training test that FAILS the moment
            # this is fixed. When you fix it, that test is your checklist.
            output = ops.exp(-ops.minimum(scaled_dist_sq, 50.0))

        # DECISION plan_2026-06-14_5e80bd3e/D-001: gate on a graph-safe training factor so
        # the repulsion loss fires for a symbolic training=True tensor (custom @tf.function
        # loop) and is a zero contribution under symbolic-False, never coercing a tensor to
        # a bool. python-True keeps the exact unmasked add_loss; symbolic path multiplies by
        # the 0/1 factor.
        if self.units > 1 and self.repulsion_strength > 0:
            # variable_dtype factor so the masked loss stays float32-consistent under
            # a mixed_float16 policy (matches the autocast=False weights).
            training_factor = resolve_training_factor(training, self.variable_dtype)
            if training_factor is not None:
                repulsion_loss = self._compute_repulsion_loss()
                self.add_loss(
                    repulsion_loss if isinstance(training_factor, float)
                    else training_factor * repulsion_loss
                )

        # Cast to compute_dtype so the layer emits the policy's compute dtype
        # (float16 under mixed precision; no-op under float32).
        return ops.cast(output, self.compute_dtype)

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
            'output_mode': self.output_mode,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RBFLayer":
        """Create a layer instance from its serialized configuration.

        :param config: Configuration dictionary produced by ``get_config``.
        :type config: Dict[str, Any]
        :return: Reconstructed layer instance.
        :rtype: RBFLayer
        """
        config = dict(config)
        if "center_initializer" in config and not isinstance(config["center_initializer"], str):
            config["center_initializer"] = keras.initializers.deserialize(
                config["center_initializer"]
            )
        if "center_constraint" in config:
            config["center_constraint"] = keras.constraints.deserialize(config["center_constraint"])
        if "kernel_regularizer" in config:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "gamma_regularizer" in config:
            config["gamma_regularizer"] = keras.regularizers.deserialize(
                config["gamma_regularizer"]
            )
        return cls(**config)


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
