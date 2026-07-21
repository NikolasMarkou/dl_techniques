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

from ...utils.tensors import resolve_training_factor, pairwise_squared_distance

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

    **Known limits of the default ``'basis'`` arm, at a glance.** Read the
    ``gamma_init`` and ``output_mode`` parameter docs below before using it.

    * It trains AT ALL, not WELL. At equal budget ``'normalized'`` reaches
      loss 0.176 where ``'basis'`` sits at 0.6899 (chance).
    * It has a **ceiling in the feature dimension**: the resolved gamma is
      ``1/D``, so the ``centers`` gradient falls below this suite's
      ``MIN_USEFUL_GRADMAX = 1e-4`` liveness floor from ``D ~ 512`` upward.
      ``'basis'`` is usefully trainable only to roughly ``D ~ 400``;
      ``D = 784`` (flattened MNIST) is already past it.
    * It requires **approximately standardized input**. The resolved exponent
      is ~``scale^2 + mean^2``, so an input scale or mean near ``sqrt(50) ~ 7``
      re-saturates the 50.0 clip and returns the gradient to exactly ``0.0``
      at ANY ``D``.

    If you need this layer to train, use ``output_mode='normalized'``.

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
    :param gamma_init: Initial value for the width parameter. ``None`` (the
        default) selects a **mode-dependent** value, resolved in ``build()``
        once the input's last dimension ``D`` is known: ``1.0 / D`` under
        ``output_mode='basis'``, and ``1.0`` under ``output_mode='normalized'``.
        Pass a float to pin an explicit, dimension-blind value; the explicit
        value always wins over both defaults.

        Why ``1/D`` for ``'basis'``: ``||x - c||^2`` grows approximately
        linearly with ``D`` for standardized input (measured ratio
        ``E[||x-c||^2] / D`` = 1.00 at ``D`` = 4 / 128 / 1024), so a fixed
        ``gamma`` that is sensible at ``D = 4`` drives the exponent
        ``dist_sq * gamma`` past the arm's 50.0 clip at ``D >= 64`` and
        saturates every unit into a constant with exactly zero gradient.

        **The ``1/D`` law has a ceiling, and it is not far away.** Because the
        resolved gamma shrinks as ``1/D``, so does the ``centers`` gradient.
        Measured, ``units=8``, standardized input, stock defaults --
        ``max|d loss / d centers|``: ``3.56e-04`` (``D`` = 128), ``1.50e-04``
        (256), ``1.14e-04`` (384), ``8.52e-05`` (512), ``5.68e-05`` (784),
        ``3.78e-05`` (1024). Against the suite's own usefulness floor of
        ``1e-4`` that puts the ceiling at roughly ``D ~ 400``: from ``D = 512``
        upward ``'basis'`` is NOT usefully trainable, and ``D = 784``
        (flattened MNIST) is already past it. Pinned by
        ``test_basis_mode_gradient_falls_below_floor_at_high_dimension``
        (``xfail(strict=True)``). Do NOT try to buy headroom with a larger
        constant ``c/D`` -- it is strictly worse (measured at ``D = 512``:
        ``c=1`` -> ``8.3e-05``, ``c=4`` -> ``9.6e-07``, ``c=8`` -> ``1.5e-09``).
        A different parameterization, not a different constant, is what a
        high-``D`` ``'basis'`` arm needs.

        **The ``1/D`` law also assumes approximately standardized input**,
        since it relies on ``E[||x-c||^2] ~ D``. The resolved exponent is then
        ~``scale^2 + mean^2`` -- dimension-free, which is a real improvement
        over the old ``D * gamma`` trigger but not a removal. Measured at
        ``D = 128``: ``scale=2`` -> ``3.14e-06`` (already below the floor),
        ``scale=4`` -> ``1.46e-13``, ``scale=10`` -> EXACTLY ``0.0`` with
        forward std exactly ``0.0``; ``mean=7.1, scale=1`` -> EXACTLY ``0.0``.
        Being dimension-free this reaches ``D = 4`` as well (``1.35e-06`` at
        ``scale=10``). Pinned by
        ``test_basis_mode_gradient_collapses_at_non_standard_input_scale``.

        Why NOT ``1/D`` for ``'normalized'``: that arm is a softmax over
        ``-dist_sq * gamma`` and is shift-invariant, so only the BETWEEN-UNIT
        logit gaps carry signal. Those gaps are set by the center spread rather
        than by ``D``, so a ``1/D`` gamma shrinks them ~``D``-fold and collapses
        the output toward a uniform ``1/units`` (measured at ``D = 128``: mean
        logit spread 1.767 at ``gamma=1.0`` versus 0.0149 at ``gamma=1/128``).
        ``'normalized'`` therefore keeps the historical ``1.0``.

        ``get_config()`` emits this constructor argument verbatim, so a config
        carrying an explicit float -- including every artifact saved before this
        default existed, which froze a concrete ``1.0`` -- deserializes to
        exactly its previous numerics.
    :type gamma_init: Optional[float]
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

           **``'basis'`` trains at all, but it does not train well. Prefer
           ``'normalized'``.**

           *What is guaranteed.* With ``gamma_init=None`` (the default), gamma
           resolves on this arm to ``1/feature_dim``, which keeps the exponent
           ``O(1)`` at every dimension. The 50.0 clip below is RETAINED, but at stock
           defaults it no longer engages, so ``ops.minimum``'s structurally zero
           saturated-branch gradient is no longer reachable. Measured,
           ``units=8``, ``normal(0, 1)`` input, 3 seeds: gradmax on
           ``(centers, gamma_raw)`` goes from EXACTLY ``0.0 / 0.0`` to
           ``3.0e-04 / 3.3e-02`` at ``D = 128`` and ``1.4e-04 / 3.4e-02`` at
           ``D = 256``; forward output std goes from ``0.0`` (the constant
           ``exp(-50) = 1.93e-22``) to ``2.7e-02 .. 4.9e-02``.

           *The guarantee has TWO stated boundaries, both measured.* It holds
           for ``D <~ 400`` and for approximately standardized input only.

           1. **Ceiling in ``D``.** The resolved gamma is ``1/D``, so the
              ``centers`` gradient shrinks with ``D``: ``3.56e-04`` (128),
              ``1.50e-04`` (256), ``1.14e-04`` (384), ``8.52e-05`` (512),
              ``5.68e-05`` (784), ``3.78e-05`` (1024). From ``D = 512`` up it
              is BELOW the ``1e-4`` usefulness floor this suite judges
              liveness by, so ``'basis'`` is not usefully trainable there --
              including at ``D = 784``, flattened MNIST. A larger constant
              ``c/D`` makes it strictly worse, not better. Pinned by
              ``test_basis_mode_gradient_falls_below_floor_at_high_dimension``.
           2. **Standardized input required.** The resolved exponent is
              ~``scale^2 + mean^2`` and is dimension-free, so an input scale or
              mean near ``sqrt(50) ~ 7`` re-saturates the 50.0 clip and D-012
              recurs in full at ANY ``D``: at ``D = 128``, ``scale=2`` gives
              ``3.14e-06`` and ``scale=10`` gives EXACTLY ``0.0``. Pinned by
              ``test_basis_mode_gradient_collapses_at_non_standard_input_scale``.

           *What is NOT guaranteed -- the residual limitation.* Live gradients
           are not fast convergence. At identical hyperparameters
           (``units=8``, ``lr=1e-2``, 40 epochs, ``D = 128``, a linearly
           separable binary fit) ``'normalized'`` reaches loss **0.176** while
           fixed ``'basis'`` sits at **0.690**, still chance. Sweeping epochs
           alone (3 seeds, ``units`` and ``lr`` held fixed) does not clear
           ``0.5`` at any budget up to 400 epochs (worst seed at 400: 0.527).
           The binding constraint is the raw-``exp`` parameterization, not the
           initialization: squaring an ``exp(-large)`` activation destroys the
           signal, which is why softmax's shift-invariance makes ``'normalized'``
           depend only on RELATIVE distances and therefore immune. This is a
           property of the mode, not a remaining bug, and is pinned by
           ``test_basis_mode_fit_is_slow_at_realistic_dimension``
           (``xfail(strict=True)``).

           If you need this layer to train efficiently, use ``'normalized'``.

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
        gamma_init: Optional[float] = None,
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
        if gamma_init is not None and gamma_init <= 0:
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
        # Build-time state, NOT a constructor parameter: never enters get_config().
        self._gamma_init_resolved: Optional[float] = None

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

        # DECISION plan-2026-07-20T175634-f3aca1ff/D-001: gamma's default MUST be
        # resolved HERE, against the real feature dimension, and NOT in __init__.
        # E[||x - c||^2] ~ D exactly for standardized input (measured ratio 1.00 at
        # D = 4 / 128 / 1024), so a dimension-blind gamma makes the exponent
        # dist_sq * gamma scale linearly in D; past D * gamma >~ 50 the 'basis' arm
        # saturates and BOTH trainable weights receive gradient exactly 0.0 (D-012).
        # Do NOT "simplify" any of the following:
        #   - Do NOT move this into __init__ or default the kwarg to a float there.
        #     D is not knowable before build(), so the whole fix disappears.
        #   - Do NOT use 1/sqrt(D). The distance is SQUARED; the scale law is linear
        #     in D, not in sqrt(D). 1/sqrt(D) still saturates at large D.
        #   - Do NOT let this value reach get_config(). It is build-time state; a
        #     D-dependent config would be non-portable across input shapes, and an
        #     explicit float in an existing artifact must keep winning outright.
        #   - Do NOT "fix" this by reshaping the 50.0 clip in call() instead. A soft
        #     clamp is measurably INERT: at gamma_init=1.0 / D=128 the true float64
        #     gradient is 7.28e-46, below float32's subnormal floor (~1.4e-45), so it
        #     underflows to bit-identical exact 0.0 whatever the clamp's derivative.
        #
        # DECISION plan-2026-07-20T175634-f3aca1ff/D-008: the default is PER-MODE,
        # and collapsing it to one dimension-aware value for both arms is a REGRESSION,
        # not a simplification. The two arms consume the exponent differently:
        #   - 'basis' returns exp(-gamma*d2), an ABSOLUTE magnitude. It needs
        #     gamma*d2 = O(1), hence gamma ~ 1/D.
        #   - 'normalized' is a softmax over -gamma*d2 and is SHIFT-INVARIANT, so only
        #     the BETWEEN-UNIT logit gaps carry signal. Those gaps are set by the
        #     center spread, not by D, so dividing gamma by D shrinks them ~D-fold and
        #     drives the softmax toward uniform 1/units -- the same dead-layer output
        #     D-008/plan-...-7de371a1 removed, arrived at from the opposite direction.
        #     Measured at D=128: mean per-sample logit spread 1.767 (gamma=1.0) vs
        #     0.0149 (gamma=1/128), and the end-to-end fit regresses from loss 0.176
        #     to 0.692 (chance). Caught by
        #     ::test_normalized_model_learns_at_realistic_dimension.
        # So 'normalized' keeps the historical 1.0 and is byte-identical to its
        # pre-change behavior at every configuration. Do NOT "unify" these branches.
        # See decisions.md D-001 / D-008 and the D-002 anchor at the clip in call().
        if self.gamma_init is not None:
            self._gamma_init_resolved = self.gamma_init
        elif self.output_mode == 'normalized':
            self._gamma_init_resolved = 1.0
        else:
            self._gamma_init_resolved = 1.0 / float(self._feature_dim)

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
        if self._gamma_init_resolved > 20.0:
            # For large values, softplus is approximately linear
            init_val = self._gamma_init_resolved
        else:
            init_val = np.log(np.exp(self._gamma_init_resolved) - 1.0)

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
        # R3 (D-002): pairwise center-to-center squared distance via the shared helper.
        # centers (units, feature_dim) x centers -> (units, units). Same result as the
        # prior inline expand-axis-1/0 broadcast.
        dist_sq = pairwise_squared_distance(self.centers, self.centers)

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

        # R3 (D-002): shared pairwise squared-distance helper. inputs (batch, ..., dim)
        # x centers (units, dim) -> (batch, ..., units). The helper uses the same
        # expand-axis=-2 broadcast this code did inline, so it supports arbitrary
        # input rank identically.
        dist_sq = pairwise_squared_distance(inputs, self.centers)

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
            # DECISION plan-2026-07-20T175634-f3aca1ff/D-001: THE 50.0 CLIP STAYS,
            # AND IT IS NOT THE DEFECT. It supersedes the former D-012 anchor here.
            # (Anchored to D-001, whose Reasoning holds the retain-the-clip and
            # soft-clamp-is-inert measurements. plan.md step 3 said "D-002"; that was
            # a drafting slip copied from the sibling 7de371a1/D-002 anchor 30 lines
            # above -- THIS plan's D-002 is the unrelated output_mode-swap rejection.)
            #
            # History: D-012 filed this clip as the cause of 'basis' being dead at
            # ordinary D (gradmax exactly 0.0/0.0 at D >= 64, output pinned to the
            # constant exp(-50) = 1.929e-22). That diagnosis was wrong about the
            # mechanism. The cause was the DIMENSION-BLIND gamma default; the clip
            # was only where the symptom surfaced. With gamma resolving to 1/D in
            # build(), the exponent stays O(1) and this minimum() no longer engages
            # at stock defaults -- measured gradmax 3.0e-04 / 3.3e-02 at D=128.
            #
            # Do NOT delete the minimum(): it is the underflow floor. Unclipped,
            # exp(-128) underflows float32 to EXACT 0.0, which is strictly worse
            # than exp(-50) -- that was already disproven under D-012.
            #
            # Do NOT replace it with a soft/smooth clamp either. That is measurably
            # INERT, not merely unnecessary: in the saturated regime the true
            # float64 gradient is 7.28e-46, an order of magnitude BELOW float32's
            # subnormal floor (~1.4e-45), so it underflows to bit-identical exact
            # 0.0 regardless of the clamp's derivative shape. Squaring an exp(-~50)
            # output is what kills the gradient, not minimum()'s saturated branch.
            #
            # The live guard against regression is the D-scaled gamma_init in
            # build() (see the D-001 anchor there), pinned by
            # tests/test_layers/test_mixtures/test_radial_basis_function.py
            # ::test_basis_mode_gradients_are_live_at_realistic_dimension.
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
