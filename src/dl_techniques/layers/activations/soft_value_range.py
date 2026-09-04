"""
Soft value-range map: a smooth, strictly monotone stand-in for hard clipping.

``keras.ops.clip`` is the usual way to force a tensor into ``[lo, hi]``, and it has
one structural defect: its derivative is exactly ``0`` everywhere outside the
interval. Anything the clip pushes back onto a bound stops receiving gradient, so
the value that was too large has no way to learn to be smaller. The classic case is
the WGAN critic, whose weights are clipped to a small box after every step.

This module ships the softplus composition that removes that structural zero::

    sp(u) = softplus(beta * u) / beta
    y     = lo + sp(x - lo)          # lower bound, always applied
    y     = hi - sp(hi - y)          # upper bound, only when max_value is given

``sp`` is a smooth lower-bounded ramp: it is close to ``u`` for ``u >> 1/beta`` and
decays smoothly to ``0`` for ``u << -1/beta``, so the composition is a smoothed
``clip`` whose derivative is a product of sigmoids -- never structurally zero, and
strictly positive as a real-valued function.

Guarantees, by regime
---------------------
The distinction below is not pedantry; it is the difference between a docstring
this module's own test suite confirms and one it disproves.

**True unconditionally, in every dtype, for every finite input:**

* *Upper feasibility.* ``y <= hi`` whenever ``max_value`` is given, structurally:
  ``sp >= 0``, so ``hi - sp(...)`` can never exceed ``hi``. Holds at ``+-1e6``.
* *Lower feasibility in one-sided mode.* ``y = lo + sp(x - lo) >= lo``, structurally.
* *Monotone non-decreasing* in ``x``.
* *Interior bias is bounded by* ``log(2) / beta``, and the bound is tight. Measured
  in float64 at ``lo=-1, hi=1, sharpness=50`` relative (so ``beta = 25``):
  ``max|y - x|`` over 2001 interior points is ``0.027725887222397994``, against
  ``log(2)/25 = 0.027725887222397813``. In float32 the same probe reads
  ``0.027725934982299805`` -- the bound plus ``4.8e-08``, which is rounding of the
  result at ``|y| ~ 1`` (local spacing ``1.19e-07``), not a wider bias.

**Two float-level caveats that are NOT regime hedges but exact statements:**

* *Lower feasibility in TWO-sided mode is bounded, not exact, at low sharpness.*
  The upper branch pulls the already-lifted value back down, and it overshoots by at
  most ``log(1 + exp(-beta * (hi - lo))) / beta``. MEASURED at ``lo=-1, hi=1`` over
  20001 points spanning ``+-50``, undershoot below ``lo``:

  =====================  ==================  ====================
  ``sharpness`` (rel.)   measured            predicted by the formula
  =====================  ==================  ====================
  1.0                    ``6.265e-01``       ``6.265e-01``
  2.0                    ``1.269e-01``       ``1.269e-01``
  5.0                    ``2.686e-03``       ``2.686e-03``
  10.0                   ``9.060e-06``       ``9.080e-06``
  20.0                   ``0.0`` (exact)     ``2.061e-10``
  50.0 (the default)     ``0.0`` (exact)     ``7.715e-24``
  =====================  ==================  ====================

  So at the default ``sharpness=50`` the lower bound is exact in float32, and the
  undershoot only becomes observable below ``beta * (hi - lo) ~ 20``. A caller who
  needs an EXACT lower bound at low sharpness wants the constraint wrapper's hard
  guard, not this function -- the smooth map deliberately never clips.
* *After the cast back to a narrower dtype, the bound is only as exact as the
  dtype's representation of it.* In float16, ``0.01`` is not representable and
  rounds to ``0.010002136230468750``, so a saturated output compares ``> 0.01`` in
  float64 while being exactly ``cast(0.01, float16)``. Compare against the bound
  CAST TO THE OUTPUT DTYPE, not against the Python float.

**True of the REAL-VALUED map, but only in-regime in float32:**

* *Strict interiority* (``lo < y < hi``). The gap ``exp(-beta*d)/beta`` at distance
  ``d`` outside the interval rounds to exactly zero once it drops below the local
  float spacing. Measured at ``beta = 25``: the gap is ``3.16e-03 / 2.22e-05 /
  2.38e-07`` at ``d = 0.1 / 0.3 / 0.5``, and ``y == lo`` EXACTLY for ``d >= 0.8``
  (float32 spacing near 1.0 is ``1.19e-07``). Do not rely on strict interiority as a
  global property; rely on feasibility, which is global.
* *Nonzero gradient outside the interval* -- the defining advantage over hard
  clipping. It survives only while ``exp(-beta*d)`` is representable, i.e. roughly
  ``d < 88/beta`` in float32. Measured at ``lo=-1, hi=1``: ``beta=25, d=1.0`` gives
  ``1.39e-11`` where ``keras.ops.clip`` gives exactly ``0.0``; but ``beta=100,
  d=1.0`` and ``beta=25, d=1e6`` both underflow to exactly ``0.0``. Larger
  ``sharpness`` buys a harder knee and costs regime width; that is the trade.

Choosing ``sharpness``
----------------------
``relative_sharpness=True`` (the default) sets ``beta = sharpness / (hi - lo)``, so
``sharpness`` is measured in interval widths and transfers across differently scaled
bounds. It is ignored in one-sided mode (``max_value is None``), where there is no
width to divide by, and ``beta = sharpness`` in the caller's data units.

======================================  ===============================================
Role                                    Suggested ``sharpness``
======================================  ===============================================
Weight projection (e.g. a WGAN critic)  50 - 200, relative. Tight box, small interior bias.
Bounded output activation               5 - 15, relative. A gentle knee trains better.
Non-negativity floor                    ``max_value=None`` with
                                        ``relative_sharpness=False``, 1 - 20 in DATA units.
======================================  ===============================================

**If you want a saturating bounded head, this is usually the wrong tool.**
``min + (max - min) * sigmoid(x)`` is smooth, bounded, has no interior bias to
speak of at the centre, and saturates gracefully. The soft value-range map earns
its keep where the output should be *the identity in the interior* and merely
softly constrained at the edges -- a projection, not a squashing nonlinearity.

Numerical note
--------------
The compute dtype WIDENS only when the input is narrower than float32
(float16/bfloat16 -> float32), then the result is cast back to the input dtype.
``beta * x`` overflows float16 at any realistic ``beta``. The widening is
deliberately not an unconditional cast to ``"float32"``: that silently narrows a
float64 caller (repo decision ``plan-2026-07-29T070705-9bfc04c5/D-007``, idiom at
``sparsemax.py:232-241``).

This function NEVER hard-clips. An exact bound guard belongs in the constraint
wrapper, where it is a projection detail, not in the differentiable map.

Using it in an ``activation=`` slot
-----------------------------------
The function takes required parameters beyond ``x``, so it cannot be handed to an
``activation=`` slot directly. **Do not bind them with ``functools.partial``**: Keras
cannot serialize a bare ``partial``, so the model saves but reloads with the
activation missing or as a broken reference. Pass the registered
:class:`SoftValueRange` LAYER instead -- it is callable, carries its own
``get_config``, and survives a ``.keras`` round trip::

    keras.layers.Dense(16, activation=SoftValueRange(min_value=-1.0, max_value=1.0))

or, equivalently, apply the layer as its own step in the graph.

References:
    - Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein GAN."
      https://arxiv.org/abs/1701.07875 -- the weight-clipping critic whose
      zero-gradient boundary motivates this map.
    - Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
      "Improved Training of Wasserstein GANs." https://arxiv.org/abs/1704.00028 --
      documents the pathologies of hard weight clipping directly.
    - Bertsekas, D. P. (1999). "Nonlinear Programming" (2nd ed.), Athena Scientific
      -- smooth exact-penalty / barrier reformulations of box constraints, of which
      the softplus composition is the elementwise case.

"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------


def _validated_bounds(
        min_value: float,
        max_value: Optional[float],
        sharpness: float,
) -> Tuple[float, Optional[float]]:
    """Validate the three range parameters and return them as plain floats.

    ONE validator, two call sites -- :func:`soft_value_range` and
    :class:`SoftValueRange`. Duplicating the checks would let the Layer and the
    function drift into rejecting different inputs while both test suites stayed
    green, which is the same defect class the single-formula rule exists to prevent.

    :param min_value: Lower bound ``lo``.
    :type min_value: float
    :param max_value: Upper bound ``hi``, or ``None`` for one-sided mode.
    :type max_value: Optional[float]
    :param sharpness: Knee steepness. Must be strictly positive.
    :type sharpness: float
    :return: ``(lo, hi)`` as Python floats, with ``hi`` left as ``None`` in
        one-sided mode.
    :rtype: Tuple[float, Optional[float]]
    :raises ValueError: If ``sharpness <= 0``, or if ``max_value < min_value``. The
        message names the offending value(s).
    """
    if sharpness <= 0.0:
        raise ValueError(
            f"sharpness must be strictly positive, got {sharpness}. It is the knee "
            f"steepness of softplus(beta * u) / beta; a non-positive value inverts "
            f"or annihilates the map."
        )

    lo = float(min_value)
    hi = None if max_value is None else float(max_value)

    if hi is not None and hi < lo:
        raise ValueError(
            f"max_value must be >= min_value, got min_value={lo} and max_value={hi}. "
            f"The interval [{lo}, {hi}] is empty."
        )

    return lo, hi


# ---------------------------------------------------------------------
# Standalone activation functions
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.soft_value_range")
def soft_value_range(
        x: keras.KerasTensor,
        min_value: float,
        max_value: Optional[float] = None,
        sharpness: float = 50.0,
        relative_sharpness: bool = True,
) -> keras.KerasTensor:
    """Softly map ``x`` into ``[min_value, max_value]`` without hard clipping.

    Computes ``y = lo + sp(x - lo)`` and then, when ``max_value`` is given,
    ``y = hi - sp(hi - y)``, where ``sp(u) = softplus(beta * u) / beta``. The result
    is smooth, monotone non-decreasing, and never crosses a bound.

    See the module docstring for which guarantees are unconditional (``y <= hi``,
    monotonicity, the ``log(2)/beta`` interior-bias bound), which are real-valued
    properties that float32 realises only in-regime (strict interiority; a nonzero
    gradient outside the interval, which underflows once ``d > ~88/beta``), and for
    the two exact float-level caveats: the two-sided lower bound may undershoot by up
    to ``log(1 + exp(-beta*(hi - lo))) / beta`` (measured ``0.0`` at the default
    ``sharpness=50``, ``6.27e-01`` at ``sharpness=1``), and a narrow output dtype can
    only represent the bound to its own resolution.

    :param x: Input tensor of any shape.
    :type x: keras.KerasTensor
    :param min_value: Lower bound ``lo``. Always applied.
    :type min_value: float
    :param max_value: Upper bound ``hi``. ``None`` selects one-sided mode -- a smooth
        floor at ``lo`` and no ceiling.
    :type max_value: Optional[float]
    :param sharpness: Knee steepness. Larger means closer to ``keras.ops.clip`` and a
        narrower regime in which the outside gradient is representable. Must be
        strictly positive.
    :type sharpness: float
    :param relative_sharpness: When ``True`` (default) ``beta = sharpness / (hi - lo)``
        so ``sharpness`` is expressed in interval widths. Ignored -- not an error --
        when ``max_value is None``, where ``beta = sharpness`` in data units.
    :type relative_sharpness: bool
    :return: Tensor of the same shape and dtype as ``x``.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``sharpness <= 0``, or if ``max_value < min_value``.
    """
    lo, hi = _validated_bounds(min_value, max_value, sharpness)

    # Degenerate interval: the two branches collapse onto the single point `lo`, so
    # short-circuit rather than divide by a zero width in the relative-sharpness rule.
    # No `logger.warning` here on purpose (R-033): `min_value`/`max_value` are
    # constructor-time constants for every caller in this tree, so
    # `SoftValueRange.__init__` warns once at construction instead of on
    # every call through this forward-path helper.
    if hi is not None and hi == lo:
        return keras.ops.full_like(x, lo)

    # DECISION plan-2026-09-01T175024-5a32e889/D-001
    # WIDEN ONLY when the input is narrower than float32; never cast unconditionally
    # to "float32". `beta * x` overflows float16 at any realistic beta (the
    # mixed_float16 regression arm runs at beta = 10000), so the widening is
    # mandatory -- but an unconditional "float32" would silently NARROW a float64
    # caller and move its error by ~7 orders of magnitude with every test still
    # green. Idiom and prior measurement: `sparsemax.py:232-241`, repo decision
    # plan-2026-07-29T070705-9bfc04c5/D-007.
    # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
    # a Keras-2 residue banned across all of `src/`. Do NOT reduce it to a bare
    # `str(d)` -- a `tf.DType` stringifies as "<dtype: 'float32'>". D-007.
    input_dtype = getattr(x.dtype, "name", None) or str(x.dtype)
    compute_dtype = (
        "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
    )
    x_wide = keras.ops.cast(x, compute_dtype)

    # `relative_sharpness` has no meaning without a width, so one-sided mode ignores
    # it instead of raising: the flag is a scale convention, not a mode switch.
    if hi is None or not relative_sharpness:
        beta = float(sharpness)
    else:
        beta = float(sharpness) / (hi - lo)

    def _soft_ramp(u: keras.KerasTensor) -> keras.KerasTensor:
        """``sp(u) = softplus(beta * u) / beta``, via the backend's own softplus.

        Do NOT hand-roll this as ``log1p(exp(beta * u))``: that overflows for
        ``beta * u`` above ~88 in float32, where ``keras.ops.softplus`` is
        implemented with the standard ``max(v, 0) + log1p(exp(-|v|))`` stabilisation
        and returns the correct large value.
        """
        return keras.ops.softplus(u * beta) / beta

    y = lo + _soft_ramp(x_wide - lo)
    if hi is not None:
        # DECISION plan-2026-09-01T175024-5a32e889/D-002
        # The upper branch reads the ALREADY-lifted `y`, not `x_wide`. Composing the
        # two branches this way makes `y <= hi` structural, but it costs an exact
        # lower bound: this line pulls the value back down by up to
        # log(1 + exp(-beta * (hi - lo))) / beta. MEASURED at lo=-1, hi=1: undershoot
        # 6.265e-01 / 1.269e-01 / 2.686e-03 / 9.060e-06 at relative sharpness
        # 1 / 2 / 5 / 10, and exactly 0.0 from sharpness 20 upward. Do NOT "fix" this
        # by clamping here -- a hard bound reintroduces the exactly-zero gradient this
        # whole module exists to avoid. The exact guard belongs in
        # `SoftValueRangeConstraint(enforce_hard_bounds=True)`, where the value is
        # assigned outside any tape and no gradient is at stake (decisions.md D-002).
        y = hi - _soft_ramp(hi - y)

    return keras.ops.cast(y, input_dtype)


# ---------------------------------------------------------------------
# Keras layer implementations
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.soft_value_range")
class SoftValueRange(keras.layers.Layer):
    """Softly map activations into ``[min_value, max_value]`` without hard clipping.

    A thin, stateless :class:`keras.layers.Layer` wrapper over the module-level
    :func:`soft_value_range`. It owns no weights and re-derives no math: ``call``
    delegates, so the Layer and the function can never disagree.

    Use it where the output should be *the identity in the interior* and only softly
    constrained at the edges -- a smooth projection. It is a poor choice as a general
    saturating head; ``min + (max - min) * sigmoid(x)`` is better there. See the
    module docstring for the full comparison and for which guarantees are
    unconditional (``y <= hi``, monotonicity, the ``log(2)/beta`` interior bias)
    versus which are real-valued properties float32 realises only in-regime (strict
    interiority; a nonzero gradient outside the interval, which underflows once the
    distance exceeds roughly ``88/beta``).

    **Architecture Overview:**

    .. code-block:: text

                             x  [B, ..., F]
                                     |
                                     v
                 +-------------------------------------------+
                 | y = lo + softplus(beta*(x - lo)) / beta    |   lower knee
                 +--------------------+----------------------+
                                     |
                                     v   (only when max_value is not None)
                 +-------------------------------------------+
                 | y = hi - softplus(beta*(hi - y)) / beta    |   upper knee
                 +--------------------+----------------------+
                                     |
                                     v
                             y  [B, ..., F]

    The two knees are composed, not applied independently: the upper branch reads the
    already-lifted value. That is what makes ``y <= hi`` structural, and it is also
    the source of the small low-sharpness lower-bound undershoot documented in the
    module docstring.

    Choosing ``sharpness``
    ^^^^^^^^^^^^^^^^^^^^^^

    ======================================  ===============================================
    Role                                    Suggested ``sharpness``
    ======================================  ===============================================
    Weight projection (e.g. a WGAN critic)  50 - 200, relative. Tight box, small interior bias.
    Bounded output activation               5 - 15, relative. A gentle knee trains better.
    Non-negativity floor                    ``max_value=None`` with
                                            ``relative_sharpness=False``, 1 - 20 in DATA units.
    ======================================  ===============================================

    Larger ``sharpness`` buys a harder knee and costs regime width -- the outside
    gradient, the whole reason to prefer this over ``keras.ops.clip``, underflows to
    exactly zero sooner. That is the trade, not a free knob.

    **Mixed precision.** :func:`soft_value_range` already widens float16/bfloat16
    inputs to float32 before touching ``beta * x`` (which overflows fp16 at any
    realistic ``beta``) and casts the result back, so under a ``mixed_float16``
    global policy this layer is safe as written and its output stays float16 like
    every other layer's. If you want the OUTPUT itself kept at full precision -- for
    instance a bounded head whose values feed a loss -- say so explicitly with
    ``SoftValueRange(..., dtype="float32")`` rather than relying on the internal
    upcast, which is a numerical guard and not a dtype promise.

    Input shape:
        Arbitrary. Use the keyword argument ``input_shape`` when using this layer as
        the first layer in a model.

    Output shape:
        Same shape as the input.

    :param min_value: Lower bound ``lo``. Always applied.
    :type min_value: float
    :param max_value: Upper bound ``hi``. ``None`` (the default) selects one-sided
        mode -- a smooth floor at ``lo`` and no ceiling.
    :type max_value: Optional[float]
    :param sharpness: Knee steepness. Must be strictly positive. Defaults to 50.0.
    :type sharpness: float
    :param relative_sharpness: When ``True`` (default) ``beta = sharpness / (hi - lo)``
        so ``sharpness`` is expressed in interval widths. Ignored -- not an error --
        when ``max_value is None``.
    :type relative_sharpness: bool
    :param kwargs: Additional keyword arguments passed to the Layer base class, such
        as ``name``, ``dtype``, ``trainable``, etc.

    :raises ValueError: If ``sharpness <= 0``, or if ``max_value < min_value``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.activations.soft_value_range import SoftValueRange

        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(16)(inputs)
        outputs = SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=10.0)(x)
        model = keras.Model(inputs, outputs)

    Note:
        There is deliberately NO ``enforce_hard_bounds`` option here. An exact
        ``maximum``/``minimum`` guard on a forward-pass output would reintroduce the
        exactly-zero gradient this map exists to remove, which is a defect in an
        activation, not a feature. The hard guard exists only in the weight-constraint
        wrapper, where the value is assigned outside any tape and no gradient is at
        stake.
    """

    def __init__(
            self,
            min_value: float,
            max_value: Optional[float] = None,
            sharpness: float = 50.0,
            relative_sharpness: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the range parameters and store them.

        :param min_value: Lower bound ``lo``.
        :type min_value: float
        :param max_value: Upper bound ``hi``, or ``None`` for one-sided mode.
        :type max_value: Optional[float]
        :param sharpness: Knee steepness. Must be strictly positive.
        :type sharpness: float
        :param relative_sharpness: Whether ``sharpness`` is expressed in interval
            widths.
        :type relative_sharpness: bool
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``sharpness <= 0``, or if ``max_value < min_value``.
        """
        super().__init__(**kwargs)

        # ONE validator, shared with the function -- see `_validated_bounds`. The
        # checks are NOT restated here: a second copy is a second thing to drift.
        lo, hi = _validated_bounds(min_value, max_value, sharpness)

        # The degenerate-interval warning lives HERE, not inside `call()`'s
        # `soft_value_range()` helper: `lo`/`hi` are constructor-time
        # constants, so the condition is knowable now, once, rather than on
        # every forward pass -- a `logger.*` call reachable from `call()`
        # runs once at trace time and never again under `tf.function` (R-033).
        if hi is not None and hi == lo:
            logger.warning(
                f"SoftValueRange constructed with min_value == max_value == {lo}; "
                f"the interval has zero width, so the output is the constant {lo} "
                f"everywhere and carries no gradient. `sharpness` and "
                f"`relative_sharpness` are ignored in this case."
            )

        self.min_value = lo
        self.max_value = hi
        self.sharpness = float(sharpness)
        self.relative_sharpness = bool(relative_sharpness)

        # Elementwise and shape-preserving, so a mask passes straight through.
        self.supports_masking = True

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Mark the layer built. It is stateless -- there is nothing to create.

        There is no shape contract to assert either: the map is elementwise and
        accepts any rank. Deliberately NO weight-creation guard is paired with this
        in the tests, because a ``len(weights) > 0`` assertion on a genuinely
        weightless layer is vacuous.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``None``.
        :rtype: None
        """
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply the soft value-range map element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused; the map has no
            training-dependent behaviour. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return soft_value_range(
            inputs,
            min_value=self.min_value,
            max_value=self.max_value,
            sharpness=self.sharpness,
            relative_sharpness=self.relative_sharpness,
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        Reads only stored configuration, so it works on an UNBUILT instance.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        :return: The base Layer config plus all four range parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "min_value": self.min_value,
            "max_value": self.max_value,
            "sharpness": self.sharpness,
            "relative_sharpness": self.relative_sharpness,
        })
        return config

    def __repr__(self) -> str:
        """Return a short representation naming the interval and the sharpness.

        :return: A string such as
            ``SoftValueRange(min_value=-1.0, max_value=1.0, sharpness=50.0)``.
        :rtype: str
        """
        return (
            f"SoftValueRange(min_value={self.min_value}, "
            f"max_value={self.max_value}, sharpness={self.sharpness}, "
            f"relative_sharpness={self.relative_sharpness}, name='{self.name}')"
        )
