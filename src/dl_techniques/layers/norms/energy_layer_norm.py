"""Energy Transformer layer normalization for Keras 3.

Implements equations (1)-(2) of Hoover, Liang, Pham, Panda, Strobelt, Zaki, Chau and
Krotov, "Energy Transformer", NeurIPS 2023 (https://arxiv.org/abs/2302.07253).

The difference from ``keras.layers.LayerNormalization`` is the parameterization. Here
the scale ``gamma`` is a SCALAR and the offset ``delta`` is a VECTOR of dimension ``D``.
Stock ``LayerNormalization`` has a per-feature vector gamma. That form does not match
the Energy Transformer Lagrangian, so it does not give the positive-semi-definite
Hessian the block's energy-descent guarantee rests on.
"""

import keras
from keras import ops, initializers, constraints
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint

# ---------------------------------------------------------------------

# DECISION plan_2026-07-13_57c9833e/D-010
# The default lower bound on `gamma`. `gamma > 0` is the precondition for the PSD
# Hessian dg/dx, which is in turn the precondition for the Energy Transformer's
# descent guarantee: dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0.
# Keep the floor STRICTLY positive. Do NOT swap it for keras.constraints.NonNeg().
# NonNeg permits gamma == 0 exactly (measured: NonNeg()(0.0) == 0.0, while
# ValueRangeConstraint(1e-3)(0.0) == 0.001), and at gamma == 0 the Hessian is the ZERO
# matrix (measured). That is still PSD, so the guarantee is not violated - it just
# stops saying anything, because g collapses to the constant delta and the descent
# test stays green on a dead layer.
# Do NOT remove this default to "match the paper". The paper never trains gamma
# negative; this layer is trainable and ships to people who will. Measured at HEAD with
# the constraint overridden: gamma = -1.0 flips the sign of every NONZERO eigenvalue of
# dg/dx (see the sibling D-010 anchor in __init__ - the ZERO eigenvalue of the constant
# direction survives at every gamma), and an EnergyTransformer block then RISES in
# energy instead of falling. Protocol, since the magnitudes move with the draw:
# embed_dim 32, 4 heads, head_dim 8, hopfield_dim 64, 12 steps, step_size 0.1,
# eps 1e-5, weights and input held IDENTICAL across the two gamma arms; at seeds
# 0/5/42 the energy is non-increasing at gamma = +1.0 and increasing at gamma = -1.0
# in 3 of 3. The SIGN is the claim; the magnitudes are ~2e-02 either way and are not
# reproducible to more digits than that.
# The constraint is overridable - pass gamma_constraint=None - but that must be an
# explicit choice, never a silent one.
# The originating plan directory is gone; this comment is the record.
_GAMMA_FLOOR = 1e-3


@keras.saving.register_keras_serializable()
class EnergyLayerNorm(keras.layers.Layer):
    """Energy Transformer layer normalization, with a scalar gamma and a vector delta.

    The layer produces the ``g = dL/dx`` "activation function" of the Energy
    Transformer. Its Lagrangian ``L`` has a PSD Hessian, which is what turns the
    block's closed-form update into a provable descent direction on the block's
    scalar energy.

    Statistics are taken over the LAST axis only, per token and per sample. No
    token is mixed with another.

    .. code-block:: text

        xbar = mean_j(x_j)                                  # scalar per token
        var  = mean_j((x_j - xbar)^2)                       # scalar per token
        g_i  = gamma * (x_i - xbar) / sqrt(var + eps) + delta_i

    This ``g`` is the gradient of the Lagrangian

    .. code-block:: text

        L(x) = D * gamma * sqrt(var + eps) + sum_j delta_j * x_j
        g    = dL/dx

    Its Hessian ``dg/dx`` is PSD for ``gamma > 0``, hence for the ET energy ``E``:

    .. code-block:: text

        dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0

    That inequality is why the parameterization is a scalar gamma and a vector
    delta. The BULK eigenvalues of ``dg/dx`` -- ``D - 2`` of them, degenerate --
    equal ``gamma / sqrt(var + eps)``, so their magnitude moves with the DRAW's
    variance and is not a function of ``gamma`` and ``D`` alone. The other two
    are one float32-zero eigenvalue and one small RADIAL eigenvalue; the
    three-part spectrum is stated in full further down. Measured on rank-3
    inputs with ``D = 8``, ``eps = 1e-5``, ``keras.random.normal((1, 1, 8),
    seed=s)`` for ``s`` in 0-4: at ``gamma = 1.7`` the eigenvalues span
    ``[0.0000, 5.6069]`` across those five draws; at ``gamma = 0.0`` they are all
    exactly ``0.0``; at ``gamma = -1.0`` they span ``[-3.2982, 0.0000]``, so the
    Hessian is no longer PSD. The sign structure holds at every draw; the
    endpoints do not. The Jacobian is symmetric to float32: measured
    ``max|J - J.T| <= 1.192e-07`` over the same five draws.

    **Shape contract**: gamma is a SCALAR (``shape=()``); delta is a VECTOR
    (``shape=(D,)``). A per-feature gamma would break the ``g = dL/dx`` identity.

    **Architecture Overview:**

    .. code-block:: text

                      inputs: x   (..., D)
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ x_bar = mean(x, axis=-1)        (..., 1)       │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
        ┌────────────────────────────────────────────────┐
        │ centered = x - x_bar            (..., D)       │
        └───────────┬───────────────────────┬────────────┘
                    │                       │
                    │                       ▼
                    │           ┌───────────────────────────────────┐
                    │           │ variance = mean(centered^2,       │
                    │           │            axis=-1)   (..., 1)    │
                    │           └───────────┬───────────────────────┘
                    │                       │
                    │                       ▼
                    │           ┌───────────────────────────────────┐
                    │           │ inv_std = rsqrt(variance + eps)   │
                    │           └───────────┬───────────────────────┘
                    │                       │
                    ▼                       ▼
        ┌───────────────────────────────────────────────────────────┐
        │ gamma * centered * inv_std + delta                        │
        │ gamma is a SCALAR ()   delta is a VECTOR (D,)             │
        └───────────────────────┬───────────────────────────────────┘
                                │
                                ▼
                      output: g   (..., D)   SAME shape as x

    **Numerical note: the constant-token cliff.** ``eps`` sits inside the
    ``sqrt``, so as a token approaches constant (``var -> 0``) the Jacobian
    ``dg/dx`` does not merely grow. It saturates at a ceiling set by ``eps``:

    .. code-block:: text

        var >> eps :  BULK eigenvalues of dg/dx  ~   1.52 .. 2.03
        var == 0   :  BULK eigenvalues of dg/dx  ->  gamma / sqrt(eps)
                                                 =   1.7 / sqrt(1e-5) = 537.6

    The spectrum has THREE parts at every draw, not two. One eigenvalue sits at
    float32 zero (measured ``|lambda| <= 8e-08``): the constant direction that
    mean-subtraction removes. One is a small RADIAL eigenvalue
    ``gamma * eps / (var + eps)^(3/2)``. The remaining ``D - 2`` are degenerate
    BULK eigenvalues at ``gamma / sqrt(var + eps)``. Both lines of the block
    above quote the BULK part, and the amplification below is the bulk-to-cliff
    ratio.

    All figures measured at ``gamma = 1.7``, ``eps = 1e-5``, over 20 tokens
    drawn as ``keras.random.normal((1, 1, D), seed=s)`` for ``s`` in 0-19:

    .. code-block:: text

        D = 64 :  BULK [1.5199, 2.0295]   RADIAL [1.224e-05, 2.882e-05]
        D =  8 :  BULK [1.2589, 5.6069]   RADIAL [7.055e-06, 6.105e-04]

    So the BULK range narrows as ``D`` grows, because ``var`` concentrates. The
    cliff eigenvalue at ``var == 0`` is ``537.5872``, matching
    ``gamma / sqrt(eps)`` to all printed digits, which is a **265x to 354x
    amplification** of the bulk gain at ``D = 64`` over those 20 draws. The
    amplification is draw-dependent; the ceiling ``gamma / sqrt(eps)`` is not.

    A constant token is not exotic. An ``Embedding`` PAD row, an all-zero conv
    cell, and a collapsed early-training activation are all exactly ``var = 0``.

    Two things the cliff is NOT:

    * **It is not a broken guarantee.** ``dg/dx`` stays PSD across the cliff
      (at ``var == 0``, ``D = 64``, ``gamma = 1.7`` the smallest eigenvalue
      measures ``2.003e-05``; it is the numerically-zero constant direction, so
      the digits are float32 noise and the claim is only its sign). The energy
      descent still holds. This is a conditioning problem, not a correctness one.
    * **It is not a forward flush-to-zero bug.** Under ``mixed_float16``,
      ``eps = 1e-5`` is subnormal but representable (``float16(1e-5)`` measures
      ``1.001358e-05``, and the fp16 minimum subnormal is about ``6e-8``), so
      ``sqrt(0 + eps)`` does not become ``sqrt(0)``. The forward pass is finite.

    **The BACKWARD pass used to be a different matter under** ``mixed_float16``
    **— FIXED 2026-08-28; do not re-derive the historical numbers below against
    the current code, they will not reproduce.** The gradient of ``rsqrt``
    carries ``(var + eps)^(-3/2)``, which overflows fp16 for a near-constant
    token. Until 2026-08-28 ``call()`` ran that arithmetic in the compute dtype,
    so under ``mixed_float16`` the layer returned a FINITE forward and a **NaN
    input gradient**. Measured then on a ``(1, 1, 8)`` input filled with ``3.0``
    and one element raised by ``1e-3`` (fp16 variance ``4.172e-07``): the forward
    output was finite at every epsilon tried, but the gradient was **NaN** for
    ``epsilon <= 1.481e-05`` and finite for ``epsilon >= 1.482e-05`` — and the
    shipped default ``epsilon=1e-5`` sat inside the NaN region. The float32
    control on the same input was finite at ``epsilon=1e-6``
    (``max|grad| = 1.4219e+03``), which is what identified it as an fp16 RANGE
    limit rather than a mathematical singularity, and therefore as a dtype bug.

    ``call()`` now computes the statistics in
    ``keras.backend.result_type(inputs.dtype, "float32")`` and casts the result
    back, the same template as ``rms_norm.py``. Measured before vs after on a
    ``(1, 2, 8)`` input whose token 0 is ``[3.0] * 7 + [3.001]`` (variance
    ``1.09375e-07``, small but NONZERO — an exactly constant token measures the
    SAFE case), taking the gradient of ``sum(y ** 2)`` w.r.t. the input, GPU,
    TF32 disabled:

    .. code-block:: text

        arm                            grad finite BEFORE   AFTER    max|grad| AFTER
        mixed_float16 @ eps=1e-5       False (8 of 16 NaN)  True     3.1475e+02
        float32       @ eps=1e-5       True                 True     1.7122e+02
        mixed_float16 @ eps=1e-3       True                 True     3.4160e+00

    The 8 NaNs were exactly token 0's 8 components. The float32 forward output is
    BITWISE identical before and after (max abs diff ``0.0``): this was a dtype
    fix, not a change of the layer's math.

    The training-path consequence was the reason it mattered. Under
    ``mixed_float16``, ``fit()`` wraps the optimizer in a ``LossScaleOptimizer``,
    whose job is to SKIP a non-finite step — so a NaN gradient produced no error
    and no warning, just a model that did not train. Measured on
    ``Input(2, 8) -> Dense(8, identity, no bias) -> EnergyLayerNorm(eps=1e-5)``
    with ``LossScaleOptimizer(SGD(0.1))``: BEFORE, the gradient was NaN at every
    one of 30 steps, the dynamic loss scale decayed ``32768 -> 0.0``, the loss
    was pinned at ``0.5228752493858337`` for the first 12 steps and ended HIGHER
    than it started. AFTER, the gradient is merely too large for fp16 at the
    initial loss scale — ``inf``, not ``NaN``, which is exactly the condition
    ``LossScaleOptimizer`` exists to handle — so the scale backs off
    ``32768 -> 1024`` over 5 steps and the model then trains normally:
    ``0.5198928713798523`` at step 0 down to ``5.235e-05`` at step 29, total
    ``|delta W| = 95.80``. A 5-step trace is therefore too short to observe
    recovery; that is loss-scale warm-up, not the defect.

    An ordinary token was never affected, before or after: a standard-normal
    ``(2, 4, 8)`` input at ``epsilon=1e-5`` gave a finite gradient at 20 of 20
    ``keras.random.normal`` seeds, with ``max|grad|`` in
    ``[3.876e-03, 1.523e-02]`` under the same ``sum(y ** 2)`` loss as the float32
    control above; at ``numpy.random.default_rng(0)`` it measured ``7.5684e-03``,
    and so did ``seed=11``. The loss is part of the claim: a plain ``sum(y)``
    loss makes the input gradient cancel and reads ``0.0`` at ``seed=2``, and it
    does not reproduce the ``1.4219e+03`` float32 control above.

    **Mitigation for a caller who sees a training-stability cliff** (loss spikes
    on a batch with heavy padding, or in the first steps before activations
    spread out): raise ``norm_epsilon`` / ``epsilon``. The ceiling is
    ``gamma / sqrt(eps)``, so moving eps from ``1e-5`` to ``1e-3`` cuts the
    worst-case gain 10x. Note this is a CONDITIONING mitigation only — it is no
    longer needed for fp16 finiteness, and it is NOT the way to fix a NaN,
    because a larger epsilon trains a different network. The alternative is to
    mask PAD tokens so they never reach the norm, which is what
    ``EnergyTransformer`` already does for the Hopfield energy.

    :param epsilon: Positive constant added inside the sqrt. Defaults to ``1e-5``.
    :type epsilon: float
    :param gamma_initializer: Initializer for the scalar ``gamma``. Defaults to
        ``'ones'``. The paper requires ``gamma > 0`` for the PSD Hessian.
    :type gamma_initializer: Union[str, initializers.Initializer]
    :param delta_initializer: Initializer for the ``(D,)`` offset ``delta``.
        Defaults to ``'zeros'``.
    :type delta_initializer: Union[str, initializers.Initializer]
    :param gamma_constraint: Constraint applied to ``gamma`` after every optimizer
        step. Defaults to a strictly-positive floor,
        ``ValueRangeConstraint(min_value=1e-3)``, because ``gamma > 0`` is the
        precondition for the PSD Hessian that makes the descent guarantee true.
        Without it the guarantee is silently false: a trained ``gamma < 0``
        flips the sign of every nonzero eigenvalue of ``dg/dx``, so the block
        performs energy ASCENT with no error, no NaN and no failing test.
        ``dg/dx`` goes negative SEMI-definite, not negative-definite -- the
        layer subtracts the mean, so the constant direction stays an exact null
        direction at every ``gamma``. Measured on a 12-step
        ``EnergyTransformer`` (embed_dim 32, 4 heads, head_dim 8, hopfield_dim
        64, step 0.1, eps 1e-5), with weights and input held identical between
        the two arms: energy ASCENDS at ``gamma = -1.0`` and DESCENDS at
        ``gamma = +1.0``, 3 of 3 seeds each (0, 5, 42), magnitudes around 2e-02
        either way. No single figure is quoted because two matched-weight
        protocols disagree in the third digit; the sign flip is the claim.
        Pass ``None`` to disable it, which is a legitimate thing to want but
        must be an explicit choice. See the D-010 anchor above.
    :type gamma_constraint: Optional[constraints.Constraint]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar epsilon: The configured epsilon, stored as a float.
    :vartype epsilon: float
    :ivar gamma_initializer: The resolved initializer object for ``gamma``.
    :vartype gamma_initializer: initializers.Initializer
    :ivar delta_initializer: The resolved initializer object for ``delta``.
    :vartype delta_initializer: initializers.Initializer
    :ivar gamma_constraint: The resolved constraint, or ``None`` if the caller
        passed ``None`` explicitly.
    :vartype gamma_constraint: Optional[constraints.Constraint]
    :ivar gamma: The scalar scale, or ``None`` until ``build()`` runs.
    :vartype gamma: Optional[keras.Variable]
    :ivar delta: The ``(D,)`` offset, or ``None`` until ``build()`` runs.
    :vartype delta: Optional[keras.Variable]

    :raises ValueError: If ``epsilon`` is not a positive number. Raised in
        ``__init__``.
    :raises ValueError: If the last axis of the input shape is undefined. Raised
        in ``build()``.

    Input shape:
        Tensor of shape ``(..., D)``; normalization is over the last axis.
        Typically ``(batch, num_tokens, embed_dim)``.

    Output shape:
        Identical to the input shape.

    Example:
        >>> layer = EnergyLayerNorm(epsilon=1e-5)
        >>> g = layer(keras.random.normal((2, 16, 64)))
        >>> g.shape
        (2, 16, 64)

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (1)-(2).
    """

    # DECISION plan_2026-07-13_57c9833e/D-005
    # Do NOT add a `lagrangian()` or an `energy()` method here. It looks like an omission:
    # the Lagrangian L is written out in the docstring above, and the sibling ET layers
    # (energy_attention.py, energy_transformer.py) both expose an energy()/update() pair.
    # It is not. The ET block's reported energy is E_ATT + E_HN only, and the LayerNorm
    # Lagrangian is not a term in it, so a method here would have zero call sites. Summing
    # it into the block's energy would make the descent test assert on the wrong quantity.
    # Omitted per the use-before-reuse / earned-abstraction rule.
    # The originating plan directory is gone; this comment is the record.

    # Sentinel: distinguishes "caller said nothing" (-> apply the default positivity floor)
    # from "caller explicitly said None" (-> an unconstrained gamma, chosen on purpose).
    # Without it, `gamma_constraint=None` could not turn the constraint OFF.
    _DEFAULT_CONSTRAINT = "__default__"

    def __init__(
        self,
        epsilon: float = 1e-5,
        gamma_initializer: Union[str, initializers.Initializer] = 'ones',
        delta_initializer: Union[str, initializers.Initializer] = 'zeros',
        gamma_constraint: Any = _DEFAULT_CONSTRAINT,
        **kwargs: Any
    ) -> None:
        """Validate ``epsilon``, resolve the initializers and the constraint, and store them.

        No weight is created here. ``gamma`` and ``delta`` need the feature
        dimension, so they are created in ``build()``.

        :param epsilon: Positive constant added inside the sqrt.
        :type epsilon: float
        :param gamma_initializer: Initializer for the scalar ``gamma``.
        :type gamma_initializer: Union[str, initializers.Initializer]
        :param delta_initializer: Initializer for the ``(D,)`` offset ``delta``.
        :type delta_initializer: Union[str, initializers.Initializer]
        :param gamma_constraint: Constraint for ``gamma``. Leave it unset for the
            positivity floor; pass ``None`` to run without any constraint.
        :type gamma_constraint: Any
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``epsilon`` is not a positive number.
        """
        super().__init__(**kwargs)

        # ----- validation -----
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError(f"epsilon must be a positive number, got {epsilon}")

        # ----- store ALL configuration -----
        self.epsilon = float(epsilon)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.delta_initializer = initializers.get(delta_initializer)

        # DECISION plan_2026-07-13_57c9833e/D-010
        # ON BY DEFAULT. Do NOT flip this default to `None` "because the paper does not
        # constrain gamma": the paper never trains gamma negative, and this layer is
        # trainable. `gamma < 0` flips the sign of every NONZERO eigenvalue of the
        # Hessian dg/dx, so dg/dx stops being PSD and the block performs energy ASCENT
        # while still running, still training and still emitting finite output. It goes
        # negative SEMI-definite, NOT negative-definite: dg/dx always has an exact null
        # direction because the layer subtracts the mean, so g(x + c) == g(x) for any
        # constant c (measured D=8, eps=1e-5, float64: |J @ 1| = 2.2e-16,
        # |det J| = 2.3e-20). Do NOT "prove" the sign by asserting all eigenvalues are
        # negative - that zero eigenvalue lands on either side of 0 at rounding level
        # and the assertion passes by luck. The sibling D-010 anchor at _GAMMA_FLOOR
        # turns on the same null direction: at gamma == 0 it is the WHOLE space and
        # dg/dx is the zero matrix, PSD but vacuous.
        # Reused, not re-written, from `dl_techniques.constraints.value_range_constraint`.
        # The originating plan directory is gone; this comment is the record.
        self.gamma_constraint = (
            ValueRangeConstraint(min_value=_GAMMA_FLOOR)
            if gamma_constraint is self._DEFAULT_CONSTRAINT
            else constraints.get(gamma_constraint)
        )

        # ----- weights are created in build() -----
        self.gamma: Optional[keras.Variable] = None
        self.delta: Optional[keras.Variable] = None

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyLayerNorm with "
            f"epsilon={self.epsilon}, "
            f"gamma_initializer={gamma_initializer}, "
            f"delta_initializer={delta_initializer}"
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``gamma`` and the ``(D,)`` vector ``delta``.

        :param input_shape: Shape of the input tensor; the last axis is the feature
            axis ``D``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is undefined.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError(
                "The last axis of the input shape must be defined, got "
                f"input_shape={input_shape}"
            )

        # gamma is a SCALAR (paper eq. 1). This is not a bug and must NOT be "fixed" to a
        # per-feature vector: a vector gamma breaks g = dL/dx. It is not merely a different
        # parameterization - it makes the Jacobian dg/dx ASYMMETRIC, so dg/dx stops being
        # the Hessian of any scalar Lagrangian and the descent guarantee goes away. That is
        # guarded behaviourally by test_jacobian_is_symmetric in
        # tests/test_layers/test_norms/test_energy_layer_norm.py, not only by the shape
        # assertion below.
        #
        # The constraint below applies the positivity floor by default (D-010).
        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer=self.gamma_initializer,
            constraint=self.gamma_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        # delta is a VECTOR of dim D (the Lagrangian's linear term).
        self.delta = self.add_weight(
            name="delta",
            shape=(int(feature_dim),),
            initializer=self.delta_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the ET layer norm: ``gamma * (x - xbar) / sqrt(var + eps) + delta``.

        Measured against the closed form on a ``(2, 16, 64)`` input at
        ``gamma = 1.7`` and ``eps = 1e-5``: ``max|layer - closed form| =
        9.537e-07``.

        The statistics are computed in
        ``keras.backend.result_type(inputs.dtype, "float32")`` and the result is
        cast back, so the returned tensor carries the layer's own compute dtype.
        This is not decoration: computing them in the compute dtype gave a NaN
        INPUT gradient under ``mixed_float16`` at the shipped default
        ``epsilon=1e-5`` (8 of 16 components, exactly the near-constant token's,
        at token variance ``1.09375e-07``) while the FORWARD pass stayed finite —
        so a forward-only check cannot see it. Fixed 2026-08-28; the class
        docstring carries the full before/after trace, including the
        ``LossScaleOptimizer`` run that turned it into a silently non-training
        model. The float32 forward output is bitwise unchanged by the fix
        (max abs diff ``0.0``), so this is a dtype fix and not a math change.

        :param inputs: Input tensor of shape ``(..., D)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; present for interface consistency.
        :type training: Optional[bool]

        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-28T122601-61a91416/D-008
        # The statistics MUST be computed in at least float32, and the result cast back.
        # Do NOT "simplify" this by dropping the upcast and running the arithmetic in the
        # compute dtype, and do NOT fix the fp16 gradient NaN by raising `epsilon` to 1e-3
        # instead: a larger epsilon trains a DIFFERENT network (the Jacobian ceiling
        # gamma/sqrt(eps) drops 10x), a substitution two prior plans already refused.
        # `rsqrt`'s gradient carries (var + eps)^(-3/2), which overflows fp16 for a
        # near-constant token; computing it in fp16 gave a FINITE forward and a NaN input
        # gradient at the shipped default eps=1e-5. See the call() docstring for the
        # measured before/after and decisions.md D-008.
        # Template: rms_norm.py:362-372 (the in-package upcast pattern).
        original_dtype = inputs.dtype
        stat_dtype = keras.backend.result_type(original_dtype, "float32")
        x = ops.cast(inputs, stat_dtype)

        # Statistics over the LAST axis only — per token, per sample. No token mixing.
        x_bar = ops.mean(x, axis=-1, keepdims=True)
        centered = x - x_bar
        variance = ops.mean(ops.square(centered), axis=-1, keepdims=True)

        inv_std = ops.rsqrt(variance + self.epsilon)

        # gamma is a scalar => broadcasts over everything; delta is (D,) => broadcasts
        # over the leading (batch, token) axes.
        gamma = ops.cast(self.gamma, stat_dtype)
        delta = ops.cast(self.delta, stat_dtype)
        outputs = gamma * centered * inv_std + delta

        # Cast back: a Keras layer returns its own COMPUTE dtype.
        return ops.cast(outputs, original_dtype)

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape. The layer is shape-preserving, so this is the identity.

        Uses only the passed shape and stored config, never a weight shape, so it is
        valid on an UNBUILT layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        ``gamma_constraint`` is serialized EXPLICITLY (D-010). If it were dropped
        here, a saved model would reload with an unconstrained gamma and could
        then train itself into energy ascent, which is the defect the constraint
        exists to stop. ``constraints.get()`` in ``__init__`` accepts the
        serialized dict as-is, so the base ``Layer.from_config`` (a plain
        ``cls(**config)``) rebuilds it exactly; this class does not override
        ``from_config``. A serialized ``None`` reloads as an explicit ``None``,
        meaning an unconstrained gamma, and NOT as the default floor. The
        ``_DEFAULT_CONSTRAINT`` sentinel only covers a caller who never mentioned
        the argument. Both facts are pinned by
        ``tests/test_layers/test_norms/test_the_base_from_config_round_trips.py``.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'delta_initializer': initializers.serialize(self.delta_initializer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
        })
        return config


# ---------------------------------------------------------------------
