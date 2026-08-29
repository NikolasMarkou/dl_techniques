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

    V_rep(cᵢ, cⱼ) = α · mean_{i≠j} max(0, d_min·(1 + μ) - ||cᵢ - cⱼ||)²

    This force ensures centers maintain a minimum separation, maximizing
    the coverage of the input space.

References:
    - Moody, J., & Darken, C. J. (1989). "Fast learning in networks of
      locally-tuned processing units."
    - Bishop, C. M. (1995). "Neural Networks for Pattern Recognition."
"""

import keras
import numpy as np
from typing import Literal, Optional, Union, Tuple, Dict, Any, ClassVar, FrozenSet

# ---------------------------------------------------------------------

from ...utils.tensors import resolve_training_factor, pairwise_squared_distance
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# Named numeric constants; naming them changed NO number.

# Upper bound on the RBF exponent in the 'basis' arm. Load-bearing; the argument for
# retaining it is at the use site in `call()`, under the f3aca1ff/D-001 anchor there.
_EXP_CLIP_MAX = 50.0

# DECISION plan-2026-08-26T061816-c515641a/D-014: additive epsilon under the repulsion
# loss's sqrt, so d(sqrt)/dx is finite at dist_sq == 0 (the diagonal, which the eye mask
# then zeroes anyway). DELIBERATELY NOT `keras.backend.epsilon()`: that returns 1e-7
# today, so the substitution looks inert, but it is a MUTABLE PROCESS GLOBAL -- any
# caller of `keras.backend.set_epsilon()` would silently move this layer's gradients.
_REPULSION_SQRT_EPSILON = 1e-7

# Above this gamma, softplus is linear to float32 precision, so the inverse-softplus
# initialisation `log(expm1(gamma))` is replaced by `gamma` itself.
_SOFTPLUS_LINEAR_THRESHOLD = 20.0

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.mixtures.radial_basis_function")
class RBFLayer(keras.layers.Layer):
    """Radial Basis Function layer with adaptive center repulsion.

    Each of the ``units`` RBF neurons computes a Gaussian activation
    ``phi_i(x) = exp(-gamma_i * ||x - c_i||^2)`` measuring the proximity
    of the input ``x`` to a learnable center ``c_i``. The width parameter
    ``gamma_i`` is stored in raw (pre-softplus) form to guarantee
    positivity. During training an auxiliary repulsive penalty
    ``V_rep = alpha * mean_{i!=j} max(0, d_min*(1+mu) - ||c_i - c_j||)^2``
    discourages centre collapse, ensuring broad coverage of the input
    space. Broadcasting-based distance computation supports inputs of
    arbitrary rank (2-D, 3-D, etc.).

    **Known limits of the default ``'basis'`` arm, at a glance:** it trains AT ALL,
    not WELL (loss 0.690 vs ``'normalized'``'s 0.176 at equal budget); it has a
    ceiling near ``D ~ 400`` in the feature dimension; and it requires approximately
    standardized input. All three are measured, with numbers, under ``output_mode``
    below -- read that before using it. **If you need this layer to train, use
    ``output_mode='normalized'``.**

    *Masks are ignored.* The layer declares no ``supports_masking`` and no
    ``compute_mask``, and nothing in the package does, so padded positions produce
    ordinary activations that flow downstream. Strip padding before this layer.

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

        The ``1/D`` law has two boundaries -- a ceiling near ``D ~ 400`` and a
        requirement of approximately standardized input. Both are measured, with
        their tables, under ``output_mode`` below; they are not repeated here. One
        fact that belongs to gamma alone: do NOT try to buy high-``D`` headroom with
        a larger constant ``c/D``, which is strictly worse (measured at ``D = 512``:
        ``c=1`` -> ``8.3e-05``, ``c=4`` -> ``9.6e-07``, ``c=8`` -> ``1.5e-09``). A
        different parameterization, not a different constant, is what a high-``D``
        ``'basis'`` arm needs.

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
    :param repulsion_strength: Strength of the center repulsion penalty. It scales an
        added scalar LOSS -- the mean squared threshold violation over the
        ``units*(units-1)`` off-diagonal center pairs -- whereas
        ``KMeansLayer.repulsion_strength`` scales a centroid displacement VECTOR. The
        two are different quantities, but **the knob now carries no per-dimension
        factor in either layer**, so a given value means a comparable amount of
        "repulsion relative to the threshold" in both (this was not true before the
        ``dim_scale = feature_dim`` multiplier was removed; see
        ``_compute_repulsion_loss``).

        Measured initial value of this loss at the shipped defaults (``units=16``,
        ``center_initializer='uniform'``, ``min_distance=1.0``,
        ``safety_margin=0.2``; mean over 8 seeds)::

            D     4     16    32    64    96    128   256   512   784   1024
            loss  0.126 0.108 0.094 0.077 0.064 0.054 0.030 0.008 0.000 0.000

        The curve is a few percent of a cross-entropy-scale task loss across the whole
        range and decays monotonically: a ``RandomUniform`` center's vector norm grows
        as ``~0.05*sqrt(D/3)``, so at large ``D`` centers clear the
        ``min_distance*(1+safety_margin)=1.2`` threshold unaided and the loss
        reaches exactly 0.0. The curve was measured at this default initializer only;
        a different ``center_initializer`` moves it.

        The default still separates centers. Measured at ``D=128``, ``units=16``,
        200 Adam steps on a binary task (mean over 3 seeds), MINIMUM pairwise center
        distance after training: 0.93 with repulsion off, **2.24 at this default**,
        2.09 at the old ``dim_scale``-inflated effective strength -- i.e. the removed
        factor bought no extra separation while costing ~250x in loss magnitude.
    :type repulsion_strength: float
    :param min_distance: Minimum desired distance between centres.
    :type min_distance: float
    :param center_initializer: Initializer for RBF center positions.
    :type center_initializer: Union[str, keras.initializers.Initializer]
    :param center_constraint: Optional constraint for center positions.
    :type center_constraint: Optional[keras.constraints.Constraint]
    :param trainable_gamma: Whether the width parameters are trainable.
    :type trainable_gamma: bool
    :param safety_margin: Margin added to minimum distance threshold.
    :type safety_margin: float
    :param center_regularizer: Optional regularizer for center weights.
    :type center_regularizer: Optional[keras.regularizers.Regularizer]
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
           defaults it no longer engages, so ``keras.ops.minimum``'s structurally zero
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

    #: Legal ``output_mode`` values, declared once on the owning class. Deliberately
    #: DISJOINT from ``GMMLayer``/``KMeansLayer``'s set -- see the D-003 amendment in
    #: ``factory.validate_mixture_config``.
    VALID_OUTPUT_MODES: ClassVar[FrozenSet[str]] = frozenset({'basis', 'normalized'})

    def __init__(
        self,
        units: int,
        gamma_init: Optional[float] = None,
        repulsion_strength: float = 0.1,
        min_distance: float = 1.0,
        center_initializer: Union[str, keras.initializers.Initializer] = 'uniform',
        center_constraint: Optional[keras.constraints.Constraint] = None,
        trainable_gamma: bool = True,
        safety_margin: float = 0.2,
        center_regularizer: Optional[keras.regularizers.Regularizer] = None,
        gamma_regularizer: Optional[keras.regularizers.Regularizer] = None,
        output_mode: Literal['basis', 'normalized'] = 'basis',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # `isinstance(units, bool)` is load-bearing, not redundant: `isinstance(True, int)`
        # is True and `True <= 0` is False, so a config `units: true` would otherwise build a
        # 1-unit layer or die later with `Cannot convert '(True, 8)' to a shape`. Same defect
        # and same rationale as kmeans.py's D-008 anchor. No int check is added here on
        # purpose -- `isinstance(np.int64(4), int)` is False, and numpy counts are accepted.
        if isinstance(units, bool) or units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if gamma_init is not None and gamma_init <= 0:
            raise ValueError(f"gamma_init must be positive, got {gamma_init}")
        if repulsion_strength < 0:
            raise ValueError(f"repulsion_strength must be non-negative, got {repulsion_strength}")
        if min_distance <= 0:
            raise ValueError(f"min_distance must be positive, got {min_distance}")
        if safety_margin < 0:
            raise ValueError(f"safety_margin must be non-negative, got {safety_margin}")
        if output_mode not in self.VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {sorted(self.VALID_OUTPUT_MODES)}, "
                f"got '{output_mode}'"
            )

        self.units = units
        self.output_mode = output_mode
        self.gamma_init = gamma_init
        self.repulsion_strength = repulsion_strength
        self.min_distance = min_distance
        self.safety_margin = safety_margin
        self.trainable_gamma = trainable_gamma

        self.center_initializer = keras.initializers.get(center_initializer)
        self.center_constraint = keras.constraints.get(center_constraint)
        self.center_regularizer = keras.regularizers.get(center_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)

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
        #     drives the softmax toward a dead uniform 1/units. Measured at D=128: mean
        #     per-sample logit spread 1.767 (gamma=1.0) vs 0.0149 (gamma=1/128), and the
        #     end-to-end fit regresses from loss 0.176 to 0.692 (chance). Pinned by
        #     ::test_normalized_model_learns_at_realistic_dimension.
        # So 'normalized' keeps 1.0. Do NOT "unify" these branches.
        # See decisions.md D-001 / D-008 and the D-002 anchor at the clip in call().
        if self.gamma_init is not None:
            self._gamma_init_resolved = self.gamma_init
        elif self.output_mode == 'normalized':
            self._gamma_init_resolved = 1.0
        else:
            self._gamma_init_resolved = 1.0 / float(self._feature_dim)

        # autocast=False / explicit dtype. RBFLayer is deliberately NOT a BaseMixtureLayer
        # subclass, but follows the same rule; the rationale is written once, in
        # `base.BaseMixtureLayer`'s "Mixed-precision contract" docstring paragraph.
        self.centers = self.add_weight(
            name='centers',
            shape=(self.units, self._feature_dim),
            initializer=self.center_initializer,
            constraint=self.center_constraint,
            regularizer=self.center_regularizer,
            trainable=True,
            dtype=self.dtype,  # R5: explicit, matches kmeans/gmm (autocast=False already)
            autocast=False,
        )

        # Inverse softplus: softplus(x) = log(1+exp(x)) -> x = log(expm1(y)). `expm1` is
        # the idiom, NOT a precision fix -- measured bit-identical to `exp(y) - 1.0` after
        # the float32 cast at every gamma this layer resolves (1/D for D in 64/784/4096).
        if self._gamma_init_resolved > _SOFTPLUS_LINEAR_THRESHOLD:
            init_val = self._gamma_init_resolved
        else:
            init_val = np.log(np.expm1(self._gamma_init_resolved))

        self.gamma_raw = self.add_weight(
            name='gamma_raw',
            shape=(self.units,),
            initializer=keras.initializers.Constant(init_val),
            regularizer=self.gamma_regularizer,
            trainable=self.trainable_gamma,
            dtype=self.dtype,  # R5: explicit, matches kmeans/gmm
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
        # R3 (D-002): shared helper, centers x centers -> (units, units).
        dist_sq = pairwise_squared_distance(self.centers, self.centers)

        # Safe sqrt for gradient stability (avoid sqrt(0))
        dist = keras.ops.sqrt(dist_sq + _REPULSION_SQRT_EPSILON)

        threshold = self.min_distance * (1.0 + self.safety_margin)

        # Penalty: max(0, threshold - distance)^2
        penalty = keras.ops.square(keras.ops.maximum(0.0, threshold - dist))

        # Mask the diagonal (distance to self is 0, which would cause max penalty).
        # variable_dtype (float32) to match the autocast=False centers under mixed precision.
        eye_mask = keras.ops.eye(self.units, dtype=self.variable_dtype)
        off_diag_mask = 1.0 - eye_mask

        masked_penalty = penalty * off_diag_mask

        # DECISION plan-2026-08-26T061816-c515641a/D-018: MEAN over the units*(units-1)
        # off-diagonal pairs, NO per-dimension factor. Do not reintroduce the removed
        # `dim_scale = feature_dim` multiplier -- it made a default-on aux loss reach ~3x a
        # cross-entropy task loss at D~256 (7.22 measured at the 0.1 default) and bought no
        # extra separation -- and do not divide by units**2, which is not the term count.
        n_pairs = float(self.units * (self.units - 1))
        if n_pairs == 0.0:
            return keras.ops.zeros((), dtype=self.variable_dtype)

        return self.repulsion_strength * keras.ops.sum(masked_penalty) / n_pairs

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computing Gaussian RBF activations.

        :param inputs: Input tensor of shape ``(batch, ..., dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode. ``None`` defers to the
            enclosing Keras call context.
        :type training: Optional[bool]
        :return: RBF activations of shape ``(batch, ..., units)``. Unnormalized
            under ``output_mode='basis'``; summing to 1.0 along the last axis
            under ``output_mode='normalized'``.
        :rtype: keras.KerasTensor"""
        # Cast in / cast out: see `base.BaseMixtureLayer`'s mixed-precision contract.
        inputs = keras.ops.cast(inputs, self.variable_dtype)

        # R3 (D-002): shared helper. (batch, ..., dim) x (units, dim) -> (batch, ..., units),
        # for arbitrary input rank.
        dist_sq = pairwise_squared_distance(inputs, self.centers)

        # gamma is (units,) and broadcasts against dist_sq's last axis. This is
        # -log(phi_k), UNCLIPPED; the clip belongs to the 'basis' arm alone, below.
        scaled_dist_sq = dist_sq * self.gamma

        # DECISION plan-2026-07-20T160907-7de371a1/D-002: NRBF is a softmax over the
        # UNCLIPPED, PRE-exp `-scaled_dist_sq` (which IS log(phi)), computed here in
        # variable_dtype (float32) and strictly ABOVE the keras.ops.cast(output,
        # self.compute_dtype) below. Three rewrites are real bugs, not style choices:
        #
        # 1. Do NOT feed `keras.ops.minimum(scaled_dist_sq, 50.0)` into the softmax. The clip
        #    exists only to keep keras.ops.exp from underflowing in the 'basis' arm; softmax
        #    never calls exp on the raw value and is internally shift-stabilized, so it
        #    needs no clip. Feeding it the CLIPPED value makes this arm a DEAD LAYER at
        #    ordinary feature dimensions: E[dist_sq] ~ D, so once D*gamma >~ 50 every
        #    unit saturates at the same 50.0, softmax sees a constant vector and returns
        #    uniform 1/units -- and keras.ops.minimum has a STRUCTURAL ZERO gradient in the
        #    saturated branch, so `centers` and `gamma_raw` both get gradient exactly
        #    0.0. Measured at D=128 with stock defaults: output [0.16667]*6, gradmax
        #    0.0/0.0. It also destroys NRBF's defining property -- selecting the nearest
        #    center far from the data -- which is the whole reason to prefer it over
        #    'basis'.
        # 2. Do NOT rewrite as `output / keras.ops.sum(output, axis=-1, keepdims=True)`, and
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
            output = keras.ops.softmax(-scaled_dist_sq, axis=-1)
        else:
            # DECISION plan-2026-07-20T175634-f3aca1ff/D-001: THE CLIP STAYS, AND IT IS
            # NOT THE DEFECT. (Spelled `_EXP_CLIP_MAX` at module scope; the NAME changed,
            # the VALUE did not.)
            #
            # Do NOT delete the minimum(): it is the underflow FLOOR. Unclipped,
            # exp(-128) underflows float32 to EXACT 0.0, strictly worse than exp(-50).
            #
            # Do NOT replace it with a soft/smooth clamp either -- measurably INERT, not
            # merely unnecessary: in the saturated regime the true float64 gradient is
            # 7.28e-46, an order of magnitude BELOW float32's subnormal floor (~1.4e-45),
            # so it underflows to bit-identical exact 0.0 whatever the clamp's derivative.
            # Squaring an exp(-~50) output kills the gradient, not minimum()'s branch.
            #
            # 'basis' being dead at ordinary D was caused by the DIMENSION-BLIND gamma
            # default, not by this clip; with gamma resolving to 1/D in build() the
            # exponent stays O(1) and this minimum() no longer engages at stock defaults
            # (measured gradmax 3.0e-04 / 3.3e-02 at D=128). The live guard is therefore
            # the D-scaled gamma_init in build() (see the D-001 anchor there), pinned by
            # ::test_basis_mode_gradients_are_live_at_realistic_dimension.
            output = keras.ops.exp(-keras.ops.minimum(scaled_dist_sq, _EXP_CLIP_MAX))

        # DECISION plan_2026-06-14_5e80bd3e/D-001: gate on a graph-safe training factor so
        # the repulsion loss fires for a symbolic training=True tensor (custom @tf.function
        # loop) and is a zero contribution under symbolic-False, never coercing a tensor to
        # a bool. python-True keeps the exact unmasked add_loss; the symbolic path
        # multiplies by the 0/1 factor, built in variable_dtype per the mixed-precision
        # contract.
        if self.units > 1 and self.repulsion_strength > 0:
            training_factor = resolve_training_factor(training, self.variable_dtype)
            if training_factor is not None:
                repulsion_loss = self._compute_repulsion_loss()
                self.add_loss(
                    repulsion_loss if isinstance(training_factor, float)
                    else training_factor * repulsion_loss
                )

        return keras.ops.cast(output, self.compute_dtype)

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
            'min_distance': self.min_distance,
            'center_initializer': keras.initializers.serialize(self.center_initializer),
            'center_constraint': keras.constraints.serialize(self.center_constraint),
            'trainable_gamma': self.trainable_gamma,
            'safety_margin': self.safety_margin,
            'center_regularizer': keras.regularizers.serialize(self.center_regularizer),
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
        if "center_regularizer" in config:
            config["center_regularizer"] = keras.regularizers.deserialize(
                config["center_regularizer"]
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
