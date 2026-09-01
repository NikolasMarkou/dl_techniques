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
from typing import Optional

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


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

    # Degenerate interval: the two branches collapse onto the single point `lo`, so
    # short-circuit rather than divide by a zero width in the relative-sharpness rule.
    if hi is not None and hi == lo:
        logger.warning(
            f"soft_value_range called with min_value == max_value == {lo}; the "
            f"interval has zero width, so the output is the constant {lo} everywhere "
            f"and carries no gradient. `sharpness` and `relative_sharpness` are "
            f"ignored in this case."
        )
        return keras.ops.full_like(x, lo)

    # DECISION plan-2026-09-01T175024-5a32e889/D-001
    # WIDEN ONLY when the input is narrower than float32; never cast unconditionally
    # to "float32". `beta * x` overflows float16 at any realistic beta (the
    # mixed_float16 regression arm runs at beta = 10000), so the widening is
    # mandatory -- but an unconditional "float32" would silently NARROW a float64
    # caller and move its error by ~7 orders of magnitude with every test still
    # green. Idiom and prior measurement: `sparsemax.py:232-241`, repo decision
    # plan-2026-07-29T070705-9bfc04c5/D-007.
    input_dtype = keras.backend.standardize_dtype(x.dtype)
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
