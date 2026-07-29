"""Tests for the Sparsemax activation layer.

.. note::

   **Defects A, B, C, D and E are all CLOSED and this file now asserts it
   directly.**  Defect A (the ``0 * -inf = NaN`` arithmetic gather) was closed
   first; B, C, D and E — which shared one root cause, a reduction (ramp,
   cumsum, support test, ``k_z`` count) running in a compute dtype with
   neither the range nor the integer precision the algorithm needs — were
   closed by subtracting the row max, building the ``arange`` ramp in
   ``float32``, and running the reduction in a NEVER-NARROWED dtype.

   :class:`TestSparsemaxClosedDefects` below is the executable record.  Those
   cases were previously ``xfail(strict=True)`` pins on OPEN defects; the
   markers are gone and every one of them is now a **live** assertion of the
   correct behaviour, so a regression on any of the four families fails this
   file loudly.  See the ``# DECISION plan-2026-07-28T134123-420f6ccb/D-017``
   anchor in ``src/dl_techniques/layers/activations/sparsemax.py``.
"""

import ast
import inspect
import os
import re
import tempfile
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import ml_dtypes
import numpy as np
import keras
from keras import ops
import pytest
import tensorflow as tf

from dl_techniques.layers.activations.sparsemax import Sparsemax
from dl_techniques.layers.attention.multi_head_cross_attention import (
    MultiHeadCrossAttention,
)
from dl_techniques.losses.sparsemax_loss import SparsemaxLoss


def _x() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.standard_normal((4, 8)).astype("float32")


# ---------------------------------------------------------------------
# float64 reference oracle for `-inf`-masked sparsemax.
#
# Test-local helper on purpose: it is an independent re-implementation of the
# Martins & Astudillo projection used to CHECK the layer, so it must NOT share
# code with `src/`. It is not an abstraction the library gains.
# ---------------------------------------------------------------------


def _sparsemax_reference(z: np.ndarray) -> np.ndarray:
    """Sparsemax projection over the FINITE entries of each row, in float64.

    Masked (non-finite) positions are exactly ``0.0`` by construction; finite
    positions receive the standard sparsemax projection computed over ONLY the
    finite sub-vector.

    :param z: 2-D array of logits, possibly containing ``-inf`` / ``+inf``.
    :type z: np.ndarray

    :return: float64 array of the same shape as ``z``.
    :rtype: np.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    out = np.zeros_like(z)
    for i, row in enumerate(z):
        idx = np.where(finite[i])[0]
        assert idx.size > 0, f"row {i} is fully masked; that case is out of scope"
        vals = row[idx]
        sorted_vals = np.sort(vals)[::-1]
        k = np.arange(1, sorted_vals.size + 1)
        cssv = np.cumsum(sorted_vals)
        cond = 1.0 + k * sorted_vals > cssv
        k_z = int(np.count_nonzero(cond))
        tau = (cssv[k_z - 1] - 1.0) / k_z
        out[i, idx] = np.maximum(vals - tau, 0.0)
    return out


# ---------------------------------------------------------------------
# Oracle tolerance for assertion (b) — DERIVED, not fitted.
#
# WHY THIS EXISTS.  Iteration 1 of this test file used a FLAT
# ``atol = rtol = 1e-3`` for all three dtype policies, justified as "the
# documented TF32 noise floor".  That is a float32-derived number, and applying
# it unexamined to ``mixed_float16`` demands precision BELOW float16's own
# representable resolution: the bar could never be met, so it measured nothing.
# (It shipped unvalidated because the property test aborts at its first grid
# point, so assertion (b) never executed in the intended-RED run.)
#
# ERROR PROPAGATION — RE-DERIVED FOR THE SHIFTED, WIDENED LAYER.
# ---------------------------------------------------------------------
# The previous derivation (kept in git history) charged five roundings, each at
# ``ulp(M)`` with ``M = max|z_finite|``, and summed to ``_ULP_BUDGET = 4``.  It
# described a layer that never shifted its input and ran the whole reduction in
# the compute dtype.  Both premises are gone: the layer now subtracts the row
# max and runs the ramp / cumsum / support test / ``k_z`` count in a
# NEVER-NARROWED reduction dtype (float32 for float16 / bfloat16, the compute
# dtype itself for float32 / float64 — see the ``D-007`` anchor in
# ``sparsemax.py``).  Every term below is the SAME term, re-charged; no term was
# replaced by a fitted constant.
#
# THE LEMMA THAT SETS THE SCALE (new, and it is what makes this bound tight).
# Write ``m = max(z_finite)`` and ``s_i = z_i - m``, so ``max(s) = 0``.  The
# maximum entry is always in the support, hence ``tau <= 0``.  The outputs are
# non-negative and sum to 1, and the largest of them is ``0 - tau``, hence
# ``tau >= -1``.  Therefore:
#
#     tau in [-1, 0]   and   every SUPPORTED s_i lies in [tau, tau + 1] c [-1, 0]
#
# so **every quantity that reaches a non-zero output is at scale <= 1**, and it
# is also at scale <= the row spread ``d = max(z_finite) - min(z_finite)``
# (trivially, since ``|s_i| <= d``).  The ulp argument is therefore
#
#     D = min(d, 1.0)
#
# — the spread, as this step set out to use, CAPPED by the lemma.  Entries
# outside the support are clamped to exactly ``0.0``; a perturbation can only
# lift one across ``tau`` by an amount of the same size as the perturbation, so
# they need no separate term.
#
# THE TERMS.  ``u_c(x) = ulp(x)`` in the compute (= output) dtype;
# ``u_r(x) = ulp(x)`` in the reduction dtype.
#
#   0.0 u_c(D)   INPUT CAST.  The float32 fixture is rounded to the compute
#                dtype at the layer boundary — but the oracle is now fed those
#                SAME post-cast bits (`_to_compute_dtype`), so this rounding is
#                shared by both sides and cancels identically.  It was charged
#                at 0.5 ulp before because the call sites fed the oracle
#                PRE-cast values; measured cost of that bug on this file's own
#                grid under fp16 was 5.4e-3, i.e. larger than the whole
#                ``_TF32_ATOL_FLOOR``.  This term returns the moment any call
#                site regresses to passing a raw float32 fixture.
#   0.5 u_c(D)   THE SHIFT, on ``z_i``.  ``row_max`` is a selection and is
#                exact; ``fl(z_i - row_max)`` rounds relative to its own result,
#                which for a supported ``i`` is ``s_i`` with ``|s_i| <= D``.
#                (NEW TERM — there was no shift to charge before.)
#   0.5 u_c(D)   THE SHIFT, inherited by ``tau``.  ``tau`` is
#                ``(sum_{j<=k_z} s_j - 1) / k_z``, i.e. it carries the MEAN of
#                ``k_z`` shift roundings, each ``<= 0.5 u_c(D)``; the mean of
#                bounded terms obeys the same bound.  (NEW TERM.)
#   0.0          CAST INTO THE REDUCTION DTYPE.  Exact, by construction:
#                ``D-007`` makes the reduction dtype never narrow.  If that
#                decision is ever reverted, this term stops being zero.
#   1.0 u_r(1)   ONE CUMSUM PARTIAL.  Partial sums up to index ``k_z`` are sums
#                of values in ``[-1, 0]``, so ``|partial| <= k_z`` and its
#                spacing is ``<= 2 * k_z * u_r(1)`` (the 2 is binade
#                misalignment); ``tau`` then DIVIDES by ``k_z``, restoring the
#                scale: ``0.5 * 2 * u_r(1) = 1.0 u_r(1)``.  Note the change of
#                dtype AND of scale: this used to be ``1.0 ulp_compute(M)``.
#   1.0 u_r(1)   ``S - 1.0``, by the same magnitude-and-divide argument
#                (``|S - 1| <= k_z + 1``).  Same dtype/scale change.
#   0.5 u_c(1)   THE FINAL ``shifted - tau``.  IEEE rounds relative to the
#                RESULT, and the result is an output value in ``[0, 1]`` — so
#                this term is charged at ``u_c(1)``, NOT at ``u_c(D)``.  When
#                the spread is small (``D << 1``) the output can still be ~1
#                (e.g. two near-equal logits give ``[0.5, 0.5]``), which is
#                exactly why it cannot be folded into the ``D`` terms.
#   (the ``/ k_z`` division's own rounding is subsumed: it is charged inside the
#    two ``u_r(1)`` terms, whose "divide by k_z" step is what sets their scale.)
#
# Sum:
#
#     atol_derived = 1.0 * u_c(D) + 0.5 * u_c(1) + 2.0 * u_r(1)
#
# spelled below as ``_ULP_BUDGET_SHIFT`` / ``_ULP_BUDGET_OUTPUT`` /
# ``_ULP_BUDGET_REDUCE``.  ``sort``, ``flip``, ``ops.max`` and the one-hot
# selection are exact and contribute nothing.
#
# WHAT IS DELIBERATELY NOT CHARGED (unchanged in spirit from the old block).
# (i) A rigorous forward bound on a ``k_z``-term cumsum is ``O(k_z)`` ulp, which
# at K = 512 would permit a completely wrong answer and would be a vacuous
# assertion.  We charge ONE representative accumulation rounding instead, on the
# basis that the per-step errors partially cancel rather than add coherently.
# The consequence is that this test FAILS if the layer's accumulation error ever
# starts growing with ``k`` — which is the point.  (ii) A DISCRETE
# mis-selection of ``k_z`` at an exact tie is not an ulp-sized event and is not
# charged here; it is pinned separately by the exact-rational oracle tests,
# which compare ``k_z`` itself.
#
# DERIVED vs MEASURED (checked, NOT fitted).  The constants above were fixed by
# the accounting FIRST; the numbers below are what the fixed layer then measured
# over the whole committed property grid (4 widths x 4 mask fractions x 8 seeds)
# PLUS every `_attack_corpus()` row, replayed under each policy, against the
# EXACT-RATIONAL oracle.  "derived" excludes `_TF32_ATOL_FLOOR` so the derivation
# is checked on its own terms; "worst ratio" is `max(err / returned atol)`, so
# anything < 1.0 is a pass.
#
#   policy           derived range          measured worst   worst ratio  viol.
#   float32          2.98e-07 .. 4.17e-07   1.987e-08        0.056  (18x)  0/134
#   mixed_float16    4.89e-04 .. 1.465e-03  1.628e-04        0.167  ( 6x)  0/138
#   mixed_bfloat16   3.91e-03 .. 1.172e-02  1.302e-03        0.167  ( 6x)  0/131
#   float64          5.55e-16 .. 7.77e-16   5.551e-17        0.071  (14x)  0/128
#
# PROOF THAT THE TIGHTENED BOUND STILL BITES (executed, not argued).  Three
# injections, none committed:
#   * disabling the row-max shift in `sparsemax.py` (the mechanism this budget
#     was re-derived for) turns `test_sparsemax_property_random_masks` RED at
#     `mixed_float16` immediately (max abs difference 2.60e-03 vs atol 1.00e-03);
#   * a uniform +1.2e-03 added to every positive output is caught at float32
#     (134/134), float64 (128/128) and `mixed_float16` (26/138) — bfloat16
#     correctly needs ~5e-03, which is its own representable resolution;
#   * re-introducing the (A) oracle-input bug is caught at 69/138 (fp16) and
#     72/131 (bf16) points.  The OLD `4 * ulp(max|z|)` budget ABSORBED that same
#     bug silently (3.13e-02 fp16 / 2.50e-01 bf16 at `max|z| = 10`, versus a
#     5.45e-03 / 3.92e-02 measured error) — which is why (A) had to be fixed at
#     the call sites rather than paid for out of the budget.
#
# WHICH TERM GOVERNS.  ``_TF32_ATOL_FLOOR`` still dominates for float32 and
# float64 (4.2e-7 and 7.8e-16 are both far under 1e-3), exactly as before; the
# derived term governs float16 and bfloat16, whose ``u_c(1)`` alone exceeds the
# floor.  The floor is RETAINED and must not be deleted: it is not slack for
# this layer, it absorbs the ~1000x float32 measurement swing caused by
# `test_linear_attention.py:37` disabling TF32 process-globally at import, so
# the same assertion runs in two different regimes depending on what pytest
# co-collected.
#
# NET EFFECT vs THE OLD BUDGET.  On the Defect-B spread row (``d = 16.95``,
# fp16) the old ``4 * ulp(max|z|)`` returned ``0.0625`` for an assertion about
# an output value in ``[0, 1]`` — 64x looser than fp16's own resolution at 1.0.
# The lemma's ``min(d, 1.0)`` cap makes that row return ``1.465e-3`` instead, a
# 43x tightening, WITHOUT introducing a second output-scale tolerance function:
# the cap is derived, so it applies everywhere rather than at hand-picked call
# sites.
# ---------------------------------------------------------------------

#: Flat absolute floor, retained for float32/float64: the documented TF32
#: precision floor (`test_linear_attention.py:37` disables TF32
#: process-globally at import, so the same assertion runs in two regimes).
_TF32_ATOL_FLOOR = 1e-3

#: Ulp of ``D = min(row spread, 1.0)`` in the OUTPUT dtype: the two row-max
#: shift roundings (on ``z_i``, and inherited by ``tau``). Derived above.
_ULP_BUDGET_SHIFT = 1.0

#: Ulp of ``1.0`` in the OUTPUT dtype: the final ``shifted - tau``, whose
#: result is an output value in ``[0, 1]``. Derived above.
_ULP_BUDGET_OUTPUT = 0.5

#: Ulp of ``1.0`` in the REDUCTION dtype: one cumsum partial plus ``S - 1.0``.
#: Negligible for float16/bfloat16 (the reduction runs in float32), leading for
#: float32/float64. Derived above.
_ULP_BUDGET_REDUCE = 2.0

#: Compute-dtype name -> numpy dtype. `bfloat16` comes from `ml_dtypes` (a
#: hard TensorFlow dependency, already in the environment); `np.spacing` is
#: defined on it, so `_oracle_atol` works for bfloat16 too. The `None` guard
#: below is retained for any future name that has no numpy counterpart.
_NUMPY_DTYPE = {
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "float32": np.float32,
    "float64": np.float64,
}


# DECISION plan-2026-07-29T110112-09832856/D-008
# DO NOT re-inline this rule at a call site, and do NOT "just restate it, it is
# three lines". That was tried: `_grad_atol` shipped a third hand-written copy
# of it, and this helper is the reversal of that. A rule restated in N places is
# a hand-maintained lockstep invariant, i.e. a latent defect — this exact
# problem family already drifted one restated value to three different wrong
# values in a single plan (`plans/LESSONS.md:20`). ONE home, the derivation
# command printed beside it, every other statement DELETED. Importing the rule
# from `sparsemax.py` is also wrong (an instrument must not share code with the
# thing it measures) — see the docstring.
def _reduction_dtype(input_dtype: str) -> str:
    """The dtype the layer runs its sort/cumsum reduction in — the D-007 rule.

    THE SINGLE SOURCE OF TRUTH IS
    ``src/dl_techniques/layers/activations/sparsemax.py:225-227``.
    This function is its ONE mirror on the test side. There are exactly two
    statements of this rule in the repository: the source and this helper.

    LOCKSTEP OBLIGATION. If the source rule moves, THIS MOVES. Every tolerance
    in this module charges a ``u_r`` (reduction-dtype ulp) term through this
    function; if the two ever disagree, every such term is charged against the
    wrong dtype and every tolerance here silently mis-measures — the guards go
    quietly vacuous rather than red. Find both sites with::

        grep -nE '^ *"float32" if' \\
            src/dl_techniques/layers/activations/sparsemax.py \\
            tests/test_layers/test_activations/test_sparsemax.py

    That command must return exactly TWO lines — one per site, and it does not
    match itself. A third means the rule has been re-duplicated and the
    duplicate must be routed through this helper instead
    (`plans/LESSONS.md:20` — one home for the rule, every other statement
    DELETED, not corrected). ``test_reduction_dtype_mirror_agrees_with_the_
    source_rule`` executes both that count and a functional comparison against
    the source's own expression.

    It is a mirror rather than an import because an instrument must not share
    code with the thing it measures — the same reason the exact-rational
    oracles in this module are test-local. Importing ``sparsemax``'s own rule
    would make a re-narrowing of the source invisible to these tolerances by
    construction.

    :param input_dtype: Name of the dtype the layer receives / emits, e.g.
        ``'float16'``. For a tolerance this is the OUTPUT dtype, which under
        every policy in use here equals the layer's compute dtype.
    :type input_dtype: str

    :return: ``'float32'`` for the two narrow dtypes, ``input_dtype`` itself
        otherwise (float64 is NEVER narrowed — that is the whole of D-007).
    :rtype: str
    """
    return (
        "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
    )


def _oracle_atol(z: np.ndarray, output_dtype: str) -> float:
    """Absolute tolerance for the layer-vs-oracle comparison on inputs ``z``.

    Returns ``max(_TF32_ATOL_FLOOR, 1.0 * u_c(D) + 0.5 * u_c(1) + 2.0 * u_r(1))``
    where ``D = min(row spread, 1.0)``, ``u_c`` is the ulp of the layer's
    OUTPUT dtype and ``u_r`` the ulp of its reduction dtype. See the
    term-by-term derivation above.

    ``z`` should be the bits the LAYER RECEIVES, i.e. already cast through
    :func:`_to_compute_dtype` under a narrow policy — the spread is a property
    of the received row, not of the test's float32 fixture.

    :param z: The logit batch handed to the layer (may contain ``-inf``).
    :type z: np.ndarray
    :param output_dtype: Name of the layer's output dtype, e.g. ``'float16'``.
    :type output_dtype: str

    :return: Absolute tolerance for ``np.testing.assert_allclose``.
    :rtype: float
    """
    np_dtype = _NUMPY_DTYPE[output_dtype]
    assert np_dtype is not None, f"no numpy dtype for {output_dtype!r}"

    # The reduction dtype the layer will use — the D-007 rule, stated once in
    # `_reduction_dtype` and nowhere else on this side. See that helper's
    # docstring for the lockstep obligation against `sparsemax.py:225-227`.
    np_reduction = _NUMPY_DTYPE[_reduction_dtype(output_dtype)]

    finite = np.asarray(z)[np.isfinite(z)]
    assert finite.size > 0, "batch is fully masked; that case is out of scope"

    # Widest per-row spread in the batch (the tolerance is applied batch-wide).
    zf = np.asarray(z, dtype=np.float64)
    masked = np.where(np.isfinite(zf), zf, np.nan)
    spread = float(np.nanmax(np.nanmax(masked, axis=-1) - np.nanmin(masked, axis=-1)))

    # The lemma: tau in [-1, 0] and every supported entry lies in [-1, 0], so
    # the spread never has to be charged above 1.0.
    scale = min(spread, 1.0)

    u_c_scale = float(np.spacing(np.asarray(scale, dtype=np_dtype)))
    u_c_one = float(np.spacing(np.asarray(1.0, dtype=np_dtype)))
    u_r_one = float(np.spacing(np.asarray(1.0, dtype=np_reduction)))

    derived = (
        _ULP_BUDGET_SHIFT * u_c_scale
        + _ULP_BUDGET_OUTPUT * u_c_one
        + _ULP_BUDGET_REDUCE * u_r_one
    )
    return max(_TF32_ATOL_FLOOR, derived)


# ---------------------------------------------------------------------
# GRADIENT tolerance — the SIBLING of `_oracle_atol`, not a generalization.
#
# DECISION plan-2026-07-29T110112-09832856/D-002
# Do NOT merge this with `_oracle_atol`, and do NOT add a third tolerance
# function.  The two take genuinely different scale arguments — `_oracle_atol`
# is keyed to `min(row spread, 1.0)`, the FORWARD output's scale, while a
# gradient's scale is `max(|v|, 1)`, the UPSTREAM's — so a merged version would
# be a parameterized union, not a shared abstraction.  See `D-002` of
# `plan-2026-07-29T110112-09832856`.
#
# THE REDUCTION-DTYPE RULE IS NOT RESTATED HERE.  There are exactly TWO
# statements of it in the repository: `sparsemax.py:225-227` (the SOURCE OF
# TRUTH) and `_reduction_dtype` above (its one test-side mirror).  Both this
# function and `_oracle_atol` call that helper; neither carries a copy.  A third
# copy is a defect, not a style choice — see `_reduction_dtype`'s docstring for
# the lockstep obligation and the grep that finds both sites, and `D-008` of
# `plan-2026-07-29T110112-09832856` for why the duplicate was extracted rather
# than restated a third time.
# ---------------------------------------------------------------------
# THE TERMS.  ``u_c(x) = ulp(x)`` in the compute (= output) dtype, ``u_r(x) =
# ulp(x)`` in the reduction dtype, both evaluated at ``S = max(|v|, 1.0)`` —
# the upstream's magnitude, floored at 1 because the analytic VJP
# ``grad_i = v_i - mean_S(v)`` is a difference of quantities at scale ``|v|``
# and the layer's own internal quantities are at scale <= 1 (the `tau in [-1,0]`
# lemma above).  Charged in the same style as the forward derivation:
#
#   1.0 u_c(S)   THE MEAN ``mean_S(v)``.  A ``k``-term sum then divided by
#                ``k``: ``|partial| <= k*S`` so its spacing is ``<= 2*k*u_c(S)``
#                (the 2 is binade misalignment), and the ``/ k`` restores the
#                scale.  Same magnitude-and-divide argument as `_oracle_atol`'s
#                cumsum term.
#   1.0 u_c(S)   THE SUBTRACTION ``v_i - mean``, rounded relative to its own
#                result, which lies at scale ``<= 2S``.
#   1.0 u_c(S)   ONE REPRESENTATIVE ACCUMULATION ROUNDING in the backward's
#                reversed cumsum (the adjoint of `ops.cumsum`).  As forward,
#                ONE is charged, not ``O(k)``: an ``O(k)`` bound would permit a
#                completely wrong answer at K=4096 and measure nothing.  The
#                consequence is that this bound FAILS if the backward error ever
#                starts growing with ``k`` — which is the point.
#   1.0 u_c(S)   THE CAST BACK.  ``ops.cast(shifted, reduction_dtype)`` at
#                `sparsemax.py:228` is differentiable, so the gradient crosses
#                back from the reduction dtype into the compute dtype and is
#                rounded once on the way.
#   2.0 u_r(S)   THE TWO ROUNDINGS THAT HAPPEN INSIDE the never-narrowed
#                reduction dtype (the cumsum partial and the ``/ k_z`` of the
#                ``tau`` route).  Negligible for float16/bfloat16 (the reduction
#                runs in float32) and leading for float32/float64, exactly as
#                `_ULP_BUDGET_REDUCE` is forward.
#
# Sum: ``atol = 4 * u_c(S) + 2 * u_r(S)``.  The permutation scatter (`ops.sort`
# / `ops.flip`), the `ops.where` selection and the integer `one_hot` path are
# exact and contribute nothing; `k_z` is piecewise-constant and has NO gradient.
#
# DERIVED vs MEASURED (checked, NOT fitted; GPU, `CUDA_VISIBLE_DEVICES=1`).
# The constants were fixed by the accounting first; the numbers below are what
# the layer then measured over the whole committed `_attack_corpus()` — the
# corpus `test_gradient_matches_analytic_vjp` actually runs on — at 4 policies:
#
#   policy           atol at S=1.0   worst measured   in ulp_out   slack
#   float32          7.152557e-07    2.012e-07        1.69         3.55x
#   mixed_float16    3.906488e-03    1.726e-03        0.88         4.55x
#   mixed_bfloat16   3.125024e-02    2.365e-02        1.51         2.65x
#   float64          1.332268e-15    2.776e-16        0.62         9.68x
#
# DECISION plan-2026-07-29T110112-09832856/D-007
# DO NOT TIGHTEN THIS BUDGET.  The real headroom is **2.65x at its tightest**
# (bfloat16), NOT the ~5x an earlier record claimed: that 5x came from quoting
# the RANDOM-K-GRID worst (0.73 ulp) as if it were the corpus worst (1.69 ulp).
# Name the grid when quoting either.  A 2-ulp floor would leave bfloat16 at
# 1.32x and float32 at 1.18x — thin enough that ordinary reduction-order drift
# turns this guard permanently red for no signal, which is a guard that measures
# nothing (`plans/LESSONS.md:47`).  See `D-007` of
# `plan-2026-07-29T110112-09832856`.
# ---------------------------------------------------------------------

#: Ulp of ``S = max(|v|, 1)`` in the OUTPUT dtype, for the gradient budget:
#: the mean, the subtraction, one accumulation and the cast back. Derived above.
_GRAD_ULP_BUDGET_COMPUTE = 4.0

#: Ulp of ``S`` in the REDUCTION dtype: the two roundings taken inside the
#: never-narrowed reduction. Derived above.
_GRAD_ULP_BUDGET_REDUCE = 2.0


def _grad_atol(v: np.ndarray, output_dtype: str) -> float:
    """Absolute tolerance for a layer-gradient vs analytic-VJP comparison.

    Returns ``4 * u_c(S) + 2 * u_r(S)`` where ``S = max(|v|, 1.0)``, ``u_c`` is
    the ulp of the layer's OUTPUT dtype and ``u_r`` the ulp of its reduction
    dtype. See the term-by-term derivation above.

    This is the SIBLING of :func:`_oracle_atol`, deliberately not a
    generalization of it: that function is keyed to the forward output's scale
    (``min(row spread, 1.0)``) while a gradient is keyed to the upstream's
    (``max(|v|, 1)``), so a merged function would be a parameterized union
    rather than a shared abstraction. Both functions take their reduction dtype
    from :func:`_reduction_dtype`, the D-007 rule's single test-side home;
    neither carries a copy of it. There is no
    ``_TF32_ATOL_FLOOR`` here: the layer contains no matmul or convolution for
    TF32 to bind to, and the gradient path was MEASURED regime-insensitive.

    ``v`` should be the UPSTREAM the tape actually received, i.e. already cast
    through :func:`_to_compute_dtype` under a narrow policy — the same
    post-cast-bits rule the oracles enforce.

    :param v: The upstream cotangent ``dL/dp`` handed to the tape.
    :type v: np.ndarray
    :param output_dtype: Name of the layer's output dtype, e.g. ``'float16'``.
    :type output_dtype: str

    :return: Absolute tolerance for the gradient comparison.
    :rtype: float
    """
    np_dtype = _NUMPY_DTYPE[output_dtype]
    assert np_dtype is not None, f"no numpy dtype for {output_dtype!r}"

    # The D-007 reduction-dtype rule, via its single test-side home.
    np_reduction = _NUMPY_DTYPE[_reduction_dtype(output_dtype)]

    v64 = np.asarray(v, dtype=np.float64)
    assert np.isfinite(v64).all(), "the upstream cotangent must be finite"

    # The gradient's scale is the UPSTREAM's magnitude, floored at 1.0: the
    # layer's own internal quantities never exceed scale 1 (the `tau in [-1, 0]`
    # lemma), so below |v| = 1 the budget must not shrink with the upstream.
    scale = max(float(np.max(np.abs(v64))), 1.0)

    u_c = float(np.spacing(np.asarray(scale, dtype=np_dtype)))
    u_r = float(np.spacing(np.asarray(scale, dtype=np_reduction)))

    return _GRAD_ULP_BUDGET_COMPUTE * u_c + _GRAD_ULP_BUDGET_REDUCE * u_r


def _output_dtype() -> str:
    """The dtype a default-constructed layer emits under the global policy."""
    return keras.mixed_precision.global_policy().compute_dtype


def _partially_masked_row_batch(
    width: int,
    masked_fraction: float,
    seed: int,
    rows: int = 4,
) -> np.ndarray:
    """Build a seeded ``(rows, width)`` logit batch with `-inf` at a random subset.

    Includes deliberate exact-tie patterns, and NEVER masks an entire row
    (fully-masked rows are out of scope: `apply_attention_mask`'s rescue_axis
    handles them upstream and they never reach Sparsemax).

    :param width: Row width.
    :type width: int
    :param masked_fraction: Fraction of each row to set to ``-inf`` (``0.0``
        yields the no-``inf`` control).
    :type masked_fraction: float
    :param seed: RNG seed.
    :type seed: int
    :param rows: Number of rows in the batch.
    :type rows: int

    :return: float32 array of shape ``(rows, width)``.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    scale = float(rng.choice([0.5, 1.0, 2.0]))
    z = (rng.uniform(-5.0, 5.0, size=(rows, width)) * scale).astype(np.float32)

    # Deliberate exact ties: duplicate one value into a couple of positions.
    if width >= 3:
        for r in range(rows):
            src = int(rng.integers(width))
            dst = rng.choice(width, size=min(2, width - 1), replace=False)
            z[r, dst] = z[r, src]

    n_masked = int(round(masked_fraction * width))
    # Precondition: never mask an entire row.
    n_masked = min(n_masked, width - 1)
    assert n_masked < width, (
        f"generator would mask an entire row (width={width}, "
        f"masked_fraction={masked_fraction})"
    )
    if n_masked > 0:
        for r in range(rows):
            pos = rng.choice(width, size=n_masked, replace=False)
            z[r, pos] = -np.inf

    assert np.isfinite(z).any(axis=-1).all(), "a generated row is fully masked"
    return z


def _finite_mask_row(width: int) -> np.ndarray:
    """``[2.0, 1.0, -1e4, -1e4, ...]`` — the large-finite-negative mask idiom.

    All entries are finite, so this row is NOT a ``-inf`` mask; but at
    ``width >= ~7`` its float16 cumsum overflows. The correct answer is
    ``[1, 0, 0, ...]``: the strict argmax takes the whole mass.

    :param width: Row width.
    :type width: int
    :return: float32 array of shape ``(1, width)``.
    :rtype: np.ndarray
    """
    z = np.full((1, width), -1e4, dtype=np.float32)
    z[0, 0] = 2.0
    z[0, 1] = 1.0
    return z


def _large_value_row(width: int) -> np.ndarray:
    """``[400.0, 300.0, 300.0, ...]`` — no mask of any kind, just large values.

    ``300 * 256 = 76800`` exceeds float16's ``65504``, so the cumsum overflows
    while the answer (``[1, 0, 0, ...]``) is completely unaffected.

    :param width: Row width.
    :type width: int
    :return: float32 array of shape ``(1, width)``.
    :rtype: np.ndarray
    """
    z = np.full((1, width), 300.0, dtype=np.float32)
    z[0, 0] = 400.0
    return z


# ---------------------------------------------------------------------
# EXACT-RATIONAL ORACLE (`fractions.Fraction`) — the instrument of record for
# the LARGE-MAGNITUDE / OVERFLOW attack corpus below.
#
# WHY A SECOND ORACLE.  `_sparsemax_reference` above is float64 and uses the
# same `vals - tau` formulation as the layer, so it shares the layer's
# cancellation structure.  That is fine at the magnitudes of the property grid
# (measured: it agrees with the exact oracle to <= 8.9e-16 there, and EXACTLY
# at every Defect-D onset), but "the oracle is the same species as the code"
# is precisely how a wrong fix gets validated.  Fractions have no rounding at
# all, so agreement with them is evidence rather than coincidence.
#
# THE ONE RULE THAT MATTERS: FEED IT THE BITS THE LAYER RECEIVED.
# ---------------------------------------------------------------------
# Under `mixed_float16` the layer never sees the float32 array the test built;
# it sees that array rounded to float16 AT THE LAYER BOUNDARY.  An oracle fed
# the *intended* Python literals is answering a different question, and its
# disagreement with the layer is an artifact of the test, not a defect in the
# code.  This has already happened TWICE in this file's history: a predecessor
# plan produced 319 false violations that way, and this plan's own exploration
# produced a phantom `err = 0.5` "knife-edge residual" (a 1-unit gap at
# float32 1.68e7 is below that scale's ulp, so the two entries arrive
# BIT-IDENTICAL and the exact answer really is the tie).  Measured cost of
# getting this wrong on the ordinary property grid under fp16: up to
# 5.4e-3 of pure oracle-input error, i.e. larger than the entire `1e-3`
# tolerance floor the tests assert with.
#
# The rule is therefore CODIFIED, not documented: `_exact_sparsemax` takes an
# array that is ALREADY in the policy's compute dtype and ASSERTS it, and
# `_to_compute_dtype` is the only sanctioned way to produce one.  Never call
# the oracle on a raw float32 test fixture while a narrow policy is active.
# ---------------------------------------------------------------------


def _to_compute_dtype(z: np.ndarray) -> np.ndarray:
    """Round ``z`` to the ACTIVE policy's compute dtype — the layer boundary.

    :param z: Logit batch as the test constructed it (typically float32).
    :type z: np.ndarray

    :return: The same batch in the compute dtype, i.e. the bits the layer
        will actually receive.
    :rtype: np.ndarray
    """
    np_dtype = _NUMPY_DTYPE[_output_dtype()]
    assert np_dtype is not None, f"no numpy dtype for {_output_dtype()!r}"
    return np.asarray(z).astype(np_dtype)


def _exact_sparsemax(z_compute: np.ndarray) -> List[Dict[str, Any]]:
    """Sparsemax over each row's FINITE entries in EXACT rational arithmetic.

    Every float is a dyadic rational, so ``Fraction(*x.as_integer_ratio())`` is
    a lossless transcription of the received bits and the whole projection
    (sort, cumsum, support test, ``k_z``, ``tau``, ``max(z - tau, 0)``) is then
    computed without a single rounding.

    :param z_compute: 2-D batch ALREADY cast to the active policy's compute
        dtype — use :func:`_to_compute_dtype`. Passing a wider array is the
        oracle-input bug this assertion exists to stop.
    :type z_compute: np.ndarray

    :return: One dict per row with keys ``k_z`` (int), ``tau`` (Fraction),
        ``out`` (list of Fraction, full row width, exactly ``0`` at masked
        positions) and ``total`` (Fraction, the exact row sum).
    :rtype: list[dict]

    :raises AssertionError: if ``z_compute`` is not in the compute dtype, is
        not 2-D, or contains a fully-masked row.
    """
    expected = np.dtype(_NUMPY_DTYPE[_output_dtype()])
    assert z_compute.dtype == expected, (
        f"exact oracle fed dtype {z_compute.dtype} while the policy's compute "
        f"dtype is {expected} — the oracle MUST consume the bits the layer "
        "receives, not the test's intended literals. Use _to_compute_dtype()."
    )
    assert z_compute.ndim == 2, f"expected a 2-D batch, got {z_compute.shape}"

    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(z_compute):
        finite_idx = [j for j, v in enumerate(row) if np.isfinite(float(v))]
        assert finite_idx, f"row {i} is fully masked; that case is out of scope"

        vals = [Fraction(*float(row[j]).as_integer_ratio()) for j in finite_idx]
        sorted_vals = sorted(vals, reverse=True)

        cumsum, acc = [], Fraction(0)
        for v in sorted_vals:
            acc += v
            cumsum.append(acc)

        one = Fraction(1)
        k_z = sum(
            1
            for kk in range(1, len(sorted_vals) + 1)
            if one + kk * sorted_vals[kk - 1] > cumsum[kk - 1]
        )
        assert k_z >= 1, f"row {i}: exact support is empty, which is impossible"

        tau = (cumsum[k_z - 1] - one) / k_z
        out = [Fraction(0)] * row.size
        for j, v in zip(finite_idx, vals):
            out[j] = max(v - tau, Fraction(0))

        rows.append(
            {"k_z": k_z, "tau": tau, "out": out, "total": sum(out)}
        )
    return rows


def _exact_rows_as_float64(z_compute: np.ndarray) -> np.ndarray:
    """:func:`_exact_sparsemax`'s answer as a float64 array, for `allclose`.

    :param z_compute: 2-D batch already in the compute dtype.
    :type z_compute: np.ndarray
    :return: float64 array of the same shape.
    :rtype: np.ndarray
    """
    rows = _exact_sparsemax(z_compute)
    return np.array(
        [[float(v) for v in r["out"]] for r in rows], dtype=np.float64
    )


# ---------------------------------------------------------------------
# THE ATTACK CORPUS.
#
# Shape convention is `_partially_masked_row_batch`'s, deliberately: seeded,
# 2-D `(rows, width)` float32, deliberate exact ties, and NEVER a fully-masked
# row (that case is rescued upstream by `apply_attention_mask(rescue_axis=...)`
# and is out of scope). This is an EXTENSION of that generator's corpus, not a
# second convention: the random property rows come from the generator itself,
# and the adversarial rows below are hand-built because a random draw does not
# reach them.
#
# Each entry carries the policy under which its defect was MEASURED. That is a
# hint about provenance, not a restriction: the corpus rows are ordinary
# `float32` arrays and are meant to be replayed under every policy.
# ---------------------------------------------------------------------

#: `(label, z_float32, measured_under_policy)`.
_AttackCase = Tuple[str, np.ndarray, str]


def _plateau_row(width: int, n_finite: int, level: float = 20.0) -> np.ndarray:
    """``n_finite`` entries at ``level``, the rest ``-inf`` (D-017 clause (d)).

    The exact answer is uniform ``1 / n_finite`` over the finite entries. Under
    fp16 that mass is not representable at scale ``level``, which is the
    documented cancellation route to a silently un-normalised row.

    :param width: Row width ``K``.
    :type width: int
    :param n_finite: Number of unmasked entries (``< width``).
    :type n_finite: int
    :param level: The plateau value.
    :type level: float
    :return: float32 array of shape ``(1, width)``.
    :rtype: np.ndarray
    """
    assert 0 < n_finite < width, "a plateau row must be partially masked"
    z = np.full((1, width), -np.inf, dtype=np.float32)
    z[:, :n_finite] = level
    return z


def _attack_corpus() -> List[_AttackCase]:
    """The measured adversarial rows, plus the seeded property-grid controls.

    Contents, all reproduced verbatim from where they were measured:

    * the four ``TestSparsemaxClosedDefects`` pin inputs (Defects B/C/D/E — the
      5th pin shares Defect B's input);
    * the ``D-017`` clause (d) plateau rows (fp16, K=512 and K=1024);
    * the three Defect-D onsets (float32 ``|z| >= 1.68e7``, fp16 ``>= 2048``,
      bf16 ``>= 300``), each in its all-finite and its 2-of-4-masked form;
    * the two all-finite cumsum-overflow families that a "loudness guard"
      destroyed (they are the CONTROL that must stay correct);
    * a seeded sample of `_partially_masked_row_batch` rows — the ordinary,
      small-magnitude control.

    :return: List of ``(label, float32 batch, policy it was measured under)``.
    :rtype: list[tuple[str, np.ndarray, str]]
    """
    cases: List[_AttackCase] = []

    # --- the four pin inputs, verbatim (test_sparsemax.py:507-724) ---
    z = np.full((1, 4096), -16.95, dtype=np.float32)
    z[:, 0] = 0.0
    cases.append(("pin_defect_b_spread16.95_K4096", z, "mixed_float16"))

    rng = np.random.default_rng(7)
    cases.append(
        (
            "pin_defect_c_bf16_K256",
            rng.uniform(-5.0, 5.0, size=(2, 256)).astype(np.float32),
            "mixed_bfloat16",
        )
    )

    cases.append(
        (
            "pin_defect_d_f32_1.68e7_K4",
            np.full((1, 4), 1.68e7, dtype=np.float32),
            "float32",
        )
    )

    z = np.full((1, 4096), -np.inf, dtype=np.float32)
    z[:, :2051] = -31.921875  # exactly representable in float16
    cases.append(("pin_defect_e_fp16_m2051_K4096", z, "mixed_float16"))

    # --- D-017 clause (d): the fp16 plateau rows ---
    for width, n_finite in ((512, 256), (512, 257), (1024, 1023), (1024, 259)):
        cases.append(
            (
                f"plateau_K{width}_m{n_finite}",
                _plateau_row(width, n_finite),
                "mixed_float16",
            )
        )

    # --- the three Defect-D onsets, all-finite and partially masked ---
    for policy, magnitude in (
        ("float32", 1.68e7),
        ("mixed_float16", 2048.0),
        ("mixed_bfloat16", 300.0),
    ):
        cases.append(
            (
                f"onset_d_{policy}_all_finite",
                np.full((1, 4), magnitude, dtype=np.float32),
                policy,
            )
        )
        z = np.full((1, 4), magnitude, dtype=np.float32)
        z[:, 2:] = -np.inf  # the 2-of-4-masked NaN route
        cases.append((f"onset_d_{policy}_masked_2_of_4", z, policy))

    # --- controls that must NOT break (the loudness guard destroyed these) ---
    cases.append(("control_finite_mask_row_K256", _finite_mask_row(256), "mixed_float16"))
    cases.append(("control_large_value_row_K256", _large_value_row(256), "mixed_float16"))

    # --- ordinary, small-magnitude control rows from the shipped generator ---
    for width, frac, seed in ((17, 0.0, 1), (128, 0.5, 3), (512, 0.9, 5)):
        cases.append(
            (
                f"property_w{width}_f{frac}_s{seed}",
                _partially_masked_row_batch(width, frac, seed),
                "float32",
            )
        )

    return cases


class TestSparsemax:

    def test_construction(self) -> None:
        assert Sparsemax(axis=-1).axis == -1

    def test_invalid_axis(self) -> None:
        with pytest.raises(ValueError):
            Sparsemax(axis=1.5)

    def test_forward_sums_to_one(self) -> None:
        y = ops.convert_to_numpy(Sparsemax()(_x()))
        assert y.shape == (4, 8)
        np.testing.assert_allclose(y.sum(axis=-1), np.ones(4), atol=1e-5)
        assert np.all(y >= -1e-6)

    def test_compute_output_shape(self) -> None:
        layer = Sparsemax()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = Sparsemax()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sparsemax.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )

    # -----------------------------------------------------------------
    # `-inf` (attention-mask) coverage.
    #
    # Assertion (b)'s absolute tolerance comes from `_oracle_atol` — a DERIVED,
    # dtype-aware bound (see the error-propagation derivation at module scope).
    # It is `max(1e-3, u_c(min(spread, 1)) + 0.5 u_c(1) + 2 u_r(1))`: float32
    # and float64 keep exactly the historical 1e-3 TF32 floor; float16 and
    # bfloat16 get the ulp-scaled term, because a flat 1e-3 demands precision
    # below their representable resolution and can therefore never pass.
    # Both the oracle and the tolerance are fed `_to_compute_dtype(z)`.
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("compute_dtype", ["float32", "mixed_float16"])
    @pytest.mark.parametrize("n", [4, 512, 4096])
    def test_partial_mask_neg_inf_no_nan(self, n: int, compute_dtype: str) -> None:
        """A partially `-inf`-masked row must stay finite and match the oracle."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(compute_dtype)
        try:
            z = np.full((2, n), -np.inf, dtype=np.float32)
            # Three surviving keys (not one): a single survivor is degenerate.
            z[:, 0] = 2.0
            z[:, 1] = 1.0
            z[:, 2] = 0.5

            out = ops.convert_to_numpy(Sparsemax()(z))

            # (a) THE primary, intended-RED assertion. Must be first.
            nan_count = int(np.isnan(out).sum())
            assert not np.isnan(out).any(), (
                "Sparsemax produced NaN for a partially-masked -inf row "
                f"(n={n}, dtype={compute_dtype}); nan_count={nan_count}/{out.size}"
            )

            # (b) Secondary correctness check against the float64 oracle, at a
            #     tolerance DERIVED from the output dtype's resolution.
            #     BOTH the oracle and the tolerance consume the POST-CAST bits:
            #     under `mixed_float16` the layer never sees the float32 array
            #     built above, and feeding the oracle the pre-cast fixture
            #     charges the comparison an input-rounding term worth up to
            #     5.4e-3 that the derived budget does not (and must not) carry.
            z_compute = _to_compute_dtype(z)
            ref = _sparsemax_reference(z_compute)
            atol = _oracle_atol(z_compute, _output_dtype())
            np.testing.assert_allclose(
                out.astype(np.float64),
                ref,
                atol=atol,
                rtol=1e-3,
                err_msg=(
                    f"sparsemax != float64 oracle (n={n}, dtype={compute_dtype}, "
                    f"derived atol={atol:.3e})"
                ),
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_sparsemax_property_random_masks(self, dtype_policy: str) -> None:
        """Randomized mask/width/tie grid, in every supported dtype policy.

        Uses the shared `dtype_policy` fixture from `tests/test_layers/conftest.py`
        (float32 / mixed_float16 / float64), which restores the process-global
        policy in its teardown.

        The ``masked_fraction == 0.0`` grid points carry NO ``inf`` at all and are
        the deliberate no-`inf` control inside this property test.

        Assertion (a) is checked over the WHOLE grid before any (b) is checked, so
        the NaN assertion is what fires first regardless of grid order.
        """
        widths = (3, 17, 128, 512)
        fractions = (0.0, 0.25, 0.5, 0.9)
        seeds = tuple(range(8))

        collected = []
        for width in widths:
            for frac in fractions:
                for seed in seeds:
                    z = _partially_masked_row_batch(width, frac, seed)
                    out = ops.convert_to_numpy(Sparsemax()(z))
                    label = (
                        f"policy={dtype_policy} width={width} "
                        f"masked_fraction={frac} seed={seed}"
                    )

                    # (a) primary, intended-RED assertion — first, for every point.
                    nan_count = int(np.isnan(out).sum())
                    assert not np.isnan(out).any(), (
                        f"Sparsemax produced NaN ({label}); "
                        f"nan_count={nan_count}/{out.size}"
                    )
                    # Post-cast bits, captured while the policy is still
                    # active: both the oracle and the tolerance must see the
                    # row the LAYER received, not the float32 fixture.
                    collected.append((label, out.astype(np.float64), _to_compute_dtype(z)))

        # (b) secondary correctness check, only reachable once (a) held everywhere.
        #     Tolerance is DERIVED per grid point from the output dtype's
        #     resolution — see `_oracle_atol` and its derivation.
        out_dtype = _output_dtype()
        for label, out, z_compute in collected:
            atol = _oracle_atol(z_compute, out_dtype)
            np.testing.assert_allclose(
                out,
                _sparsemax_reference(z_compute),
                atol=atol,
                rtol=1e-3,
                err_msg=(
                    f"sparsemax != float64 oracle ({label}, "
                    f"derived atol={atol:.3e})"
                ),
            )

    def test_attention_integration_sparsemax_fp16_no_nan(self) -> None:
        """End-to-end repro: sparsemax attention under fp16 with a partial mask."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            rng = np.random.default_rng(0)
            x = rng.standard_normal((2, 4, 16)).astype("float32")
            mask = np.ones((2, 4, 4), dtype="float32")
            mask[:, :, 2:] = 0.0

            layer = MultiHeadCrossAttention(
                dim=16, num_heads=2, probability_type="sparsemax"
            )
            out = ops.convert_to_numpy(layer(x, attention_mask=mask))

            nan_count = int(np.isnan(out).sum())
            assert not np.isnan(out).any(), (
                "MultiHeadCrossAttention(probability_type='sparsemax') produced NaN "
                f"under mixed_float16 with a partial mask; "
                f"nan_count={nan_count}/{out.size}"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize(
        "label,builder",
        [
            # The LARGE-FINITE-NEGATIVE mask convention: `-1e4` in place of
            # `-inf`. This is what an attention mask bias degrades to once it
            # is cast to float16, so it is an ordinary production shape.
            ("large_finite_negative_mask", lambda k: _finite_mask_row(k)),
            # No mask anywhere: 256 entries near 300 overflow the fp16 cumsum
            # (`300 * 256 = 76800 > 65504`) purely by being large.
            ("no_mask_large_values", lambda k: _large_value_row(k)),
        ],
    )
    def test_finite_cumsum_overflow_rows_are_correct_not_nan(
        self, label: str, builder
    ) -> None:
        """An ALL-FINITE fp16 row whose cumsum overflows must still be correct.

        This is the SECOND defect family closed by the ``ops.where`` selection,
        found only by adversarial review. The ``-inf`` these rows put into
        ``z_cumsum`` is born from OVERFLOW, not from the input, but it reached
        the same ``-inf * 0.0`` product and poisoned the whole row. Measured on
        the pre-fix bytes (``87aa809c^``): ``nan=256/256, sum=0``. Here the
        answer is exact.

        Crucially, ``k_z`` is selected LONG BEFORE the overflow position, so the
        overflow never touches the result — which is why a guard keyed on
        ``isfinite(z_cumsum)`` (written, measured and reverted; see clause (e)
        of the ``D-017`` anchor) destroys these rows for nothing.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            k = 256
            z = builder(k)
            assert np.isfinite(z).all(), "premise: the INPUT is entirely finite"

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            expected = np.zeros((1, k), dtype=np.float64)
            expected[0, 0] = 1.0  # float64 oracle: the strict argmax takes all

            assert np.isfinite(out).all(), (
                f"{label}: an all-finite row whose fp16 cumsum overflows returned "
                f"a non-finite answer (nan={int(np.isnan(out).sum())}, "
                f"inf={int(np.isinf(out).sum())}) — the pre-fix behaviour"
            )
            total = float(out.sum())
            assert abs(total - 1.0) <= 1e-3, (
                f"{label}: sum(out) = {total}, expected 1.0"
            )
            np.testing.assert_array_equal(
                out,
                expected,
                err_msg=f"{label}: sparsemax != float64 oracle (expected err 0.0)",
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)


class TestExactRationalOracle:
    """Validate the INSTRUMENT, before it is used to judge the layer.

    None of these tests touch ``Sparsemax``. They exist because an oracle that
    is never itself checked is an assertion generator, not evidence — and the
    two oracle failures already in this problem's history (319 false
    violations, then a phantom ``err = 0.5``) were both oracle-INPUT bugs that
    a layer-vs-oracle test cannot distinguish from a code defect.
    """

    def test_oracle_refuses_input_that_was_not_cast_to_the_compute_dtype(
        self,
    ) -> None:
        """The F-15 discipline is enforced by code, not by comment.

        Under ``mixed_float16`` the layer receives float16 bits. Handing the
        oracle the test's float32 fixture asks it a different question, and
        that must fail loudly rather than produce a plausible wrong number.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            # fp16's own Defect-D onset. NOT float32's 1.68e7: that OVERFLOWS
            # to `+inf` under fp16, so the oracle would refuse it as a
            # (differently) out-of-scope row and this test would pass for the
            # wrong reason. A corpus row is only meaningful under a policy
            # that can represent it.
            z32 = np.full((1, 4), 2048.0, dtype=np.float32)

            with pytest.raises(AssertionError, match="_to_compute_dtype"):
                _exact_sparsemax(z32)

            # ...and the sanctioned route is accepted.
            rows = _exact_sparsemax(_to_compute_dtype(z32))
            assert rows[0]["total"] == Fraction(1)
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_exact_oracle_is_exactly_normalised_on_the_whole_attack_corpus(
        self,
    ) -> None:
        """Every corpus row has an exact answer that sums to exactly 1.

        This pins the CORPUS as much as the oracle: a row whose exact sum is
        not ``1`` would mean the fixture is degenerate (e.g. accidentally
        fully masked), and every later layer-vs-oracle comparison built on it
        would be meaningless.
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            for label, z, policy in _attack_corpus():
                keras.mixed_precision.set_global_policy(policy)

                assert z.dtype == np.float32, f"{label}: corpus rows are float32"
                assert z.ndim == 2, f"{label}: corpus rows are 2-D batches"
                assert np.isfinite(z).any(axis=-1).all(), (
                    f"{label}: a corpus row is fully masked, which is out of scope"
                )

                z_compute = _to_compute_dtype(z)
                for i, row in enumerate(_exact_sparsemax(z_compute)):
                    assert row["total"] == Fraction(1), (
                        f"{label} row {i} (policy={policy}): exact sum is "
                        f"{row['total']}, not 1 — the fixture is degenerate"
                    )
                    assert 1 <= row["k_z"] <= z.shape[-1], (
                        f"{label} row {i}: exact k_z = {row['k_z']} is out of range"
                    )
                    masked = ~np.isfinite(z_compute[i].astype(np.float64))
                    for j in np.flatnonzero(masked):
                        assert row["out"][int(j)] == Fraction(0), (
                            f"{label} row {i}: masked position {j} is not exactly 0"
                        )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_float64_reference_agrees_with_the_exact_oracle(
        self, dtype_policy: str
    ) -> None:
        """Cross-check the two oracles against each other on the property grid.

        The retained float64 ``_sparsemax_reference`` shares the layer's
        ``vals - tau`` formulation, so it could in principle inherit the
        layer's cancellation. On this grid it does not: measured worst-case
        disagreement is ``7.9e-16`` (float32 policy), i.e. float64 round-off.

        Both oracles are fed the SAME post-cast bits — the comparison is
        oracle-vs-oracle, so any policy-dependent input rounding must be
        applied once and shared, not applied to one side only.
        """
        widths = (3, 17, 128, 512)
        fractions_masked = (0.0, 0.25, 0.5, 0.9)

        worst = 0.0
        worst_label = ""
        for width in widths:
            for frac in fractions_masked:
                for seed in range(8):
                    z = _partially_masked_row_batch(width, frac, seed)
                    z_compute = _to_compute_dtype(z)
                    ref = _sparsemax_reference(z_compute.astype(np.float64))
                    exact = _exact_rows_as_float64(z_compute)
                    delta = float(np.max(np.abs(ref - exact)))
                    if delta > worst:
                        worst = delta
                        worst_label = (
                            f"policy={dtype_policy} width={width} "
                            f"masked_fraction={frac} seed={seed}"
                        )

        assert worst <= 1e-12, (
            "the float64 reference oracle disagrees with EXACT rational "
            f"arithmetic by {worst:.3e} on the property grid ({worst_label}); "
            "it can no longer be trusted as the instrument for that grid"
        )

    def test_float64_reference_is_still_exact_at_the_defect_d_magnitudes(
        self,
    ) -> None:
        """The float64 reference must survive the LARGE-magnitude corpus too.

        This is the specific cross-check step 4 depends on before it keeps
        using ``_sparsemax_reference`` anywhere near these inputs. Measured:
        the two oracles agree EXACTLY (delta ``0.0``) at every Defect-D onset
        and to ``<= 8.9e-16`` on the plateau rows.
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            worst = 0.0
            worst_label = ""
            for label, z, policy in _attack_corpus():
                if not (label.startswith("onset_d_") or label.startswith("plateau_")):
                    continue
                keras.mixed_precision.set_global_policy(policy)
                z_compute = _to_compute_dtype(z)
                ref = _sparsemax_reference(z_compute.astype(np.float64))
                exact = _exact_rows_as_float64(z_compute)
                delta = float(np.max(np.abs(ref - exact)))
                if delta > worst:
                    worst, worst_label = delta, f"{label} (policy={policy})"

            assert worst <= 1e-12, (
                "the float64 reference oracle disagrees with EXACT rational "
                f"arithmetic by {worst:.3e} at {worst_label}; it must not be "
                "used as the instrument at these magnitudes"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# REGRESSION GUARDS for the four CLOSED defects B/C/D/E.
#
# These five cases were `xfail(strict=True)` pins while B/C/D/E were open.
# Each already asserted the CORRECT behaviour, so closing the defects turned
# every one of them into an XPASS — and `strict=True` converted that XPASS
# into a hard FAILURE, which is exactly what forced the markers to be deleted
# rather than letting the improvement be absorbed silently. The markers are
# now gone and the bodies are LIVE assertions.
#
# Do not weaken these back into `xfail`/`skip`, and do not delete them: an
# anchor without a test that can fail is a comment, not a guard. Each case
# is the only committed repro of its input family, at the magnitude where
# the defect was originally measured.
# ---------------------------------------------------------------------


class TestSparsemaxClosedDefects:
    """Executable record of the four defects ``Sparsemax`` no longer has.

    Defect **A** (the ``0 * -inf = NaN`` gather) and the all-finite
    cumsum-overflow family were closed first. Defects **B**, **C**, **D** and
    **E** below shared one root cause — the reduction (ramp, cumsum, support
    test, ``k_z`` count) ran in the COMPUTE dtype, which under ``float16`` /
    ``bfloat16`` has neither the range nor the integer precision the algorithm
    needs, and which under ``float32`` fails once magnitudes reach ~1.7e7.
    They are closed by three mechanisms: subtracting the row max before the
    projection, building the ``arange`` ramp in ``float32``, and running the
    reduction in a never-narrowed dtype (``float32`` for ``float16`` /
    ``bfloat16``, the compute dtype itself for ``float32`` / ``float64``).

    The fifth case used to record something different in kind: an accepted
    REGRESSION where, on Defect-B inputs, the failure had moved from loud
    (``NaN``) to silent (a finite, un-normalised row). That premise is
    **eliminated**, not merely fixed — the row is now simply correct — so the
    case was retargeted onto the strictly stronger exact-answer claim on the
    same input rather than deleted (which would have left the family with only
    the weaker ``sum(out)`` check).

    See the ``# DECISION plan-2026-07-28T134123-420f6ccb/D-017`` anchor in
    ``src/dl_techniques/layers/activations/sparsemax.py``.
    """

    def test_reduction_dtype_mirror_agrees_with_the_source_rule(self) -> None:
        """The test-side D-007 mirror must be the SAME FUNCTION as the source.

        ``_reduction_dtype`` (this module) mirrors the reduction-dtype rule at
        ``sparsemax.py:225-227``. Every ``u_r`` term in ``_oracle_atol`` and
        ``_grad_atol`` is charged through that mirror, so if the two ever
        disagree the tolerances are computed against the wrong dtype and the
        forward/backward guards go quietly VACUOUS rather than red. A comment
        naming the obligation is not a guard (`plans/LESSONS.md:11`); this is
        the guard.

        It compares the two as FUNCTIONS, not as text: the source's own
        ``reduction_dtype = ...`` expression is extracted with :mod:`ast` and
        EVALUATED over the four dtype names, then compared to the mirror's
        output. Reformatting, renaming the local, or moving the lines leaves
        this green; changing what the rule COMPUTES (e.g. re-narrowing float64
        to ``"float32"``) turns it red. ``G7`` covers the complementary
        direction — a source re-narrowing that a lockstep-updated mirror would
        hide from the tolerances is still caught there, numerically.

        The trailing check forbids RESTATEMENT (`plans/LESSONS.md:20`): the
        rule may be spelled in exactly two places, the source and the mirror.
        A third copy is what `D-008` of ``plan-2026-07-29T110112-09832856``
        was raised and ruled on.
        """
        source_path = inspect.getsourcefile(Sparsemax)
        assert source_path is not None, "cannot locate sparsemax.py"
        source_text = open(source_path, "r", encoding="utf-8").read()

        # Extract every `reduction_dtype = <expr>` assignment in `Sparsemax`.
        tree = ast.parse(source_text)
        exprs = [
            node.value
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == "Sparsemax"
            for node in ast.walk(cls)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "reduction_dtype"
                for t in node.targets
            )
        ]
        assert len(exprs) == 1, (
            "expected exactly one `reduction_dtype = ...` assignment in "
            f"`Sparsemax`, found {len(exprs)} in {source_path}. The D-007 rule "
            "has moved; `_reduction_dtype` must move with it."
        )
        expr = exprs[0]

        # The expression's single free variable is the dtype it switches on.
        free = sorted({n.id for n in ast.walk(expr) if isinstance(n, ast.Name)})
        assert len(free) == 1, (
            f"the source rule reads {free!r}; this guard can only evaluate a "
            "rule that is a pure function of the input dtype name"
        )
        code = compile(ast.Expression(expr), "<sparsemax reduction rule>", "eval")

        for dtype in ("float16", "bfloat16", "float32", "float64"):
            source_answer = eval(code, {"__builtins__": {}}, {free[0]: dtype})
            assert source_answer == _reduction_dtype(dtype), (
                f"D-007 LOCKSTEP BROKEN for {dtype!r}: "
                f"{source_path}:{expr.lineno} computes {source_answer!r} but "
                f"`_reduction_dtype` mirrors {_reduction_dtype(dtype)!r}. "
                "Move the mirror, or revert the source."
            )

        # No third statement of the rule: this is the grep printed in
        # `_reduction_dtype`'s docstring, executed. The pattern is anchored so
        # that it matches neither the docstring line that prints it nor this
        # line — the two hits are the two real homes and nothing else.
        pattern = re.compile(r'^ *"float32" if', re.MULTILINE)
        test_path = os.path.abspath(__file__)
        counts = {
            path: len(pattern.findall(open(path, "r", encoding="utf-8").read()))
            for path in (source_path, test_path)
        }
        assert counts == {source_path: 1, test_path: 1}, (
            f"the D-007 rule is stated {counts} times (expected exactly one "
            "home per file: `sparsemax.py`'s rule and `_reduction_dtype`). A "
            "further copy must be routed through `_reduction_dtype` instead; "
            "see D-008 of plan-2026-07-29T110112-09832856."
        )

    def test_defect_b_closed_overflow_born_inf_excluded_from_support(self) -> None:
        """fp16 cumsum overflow must not inflate the support set.

        Defect B (``sparsemax.py:~220``, CLOSED): an overflow-born non-finite
        ``z_cumsum`` used to be ADMITTED to the support, because
        ``support = 1 + finite - (-inf) = +inf > 0``. Measured at spread 16.95,
        ``mixed_float16``, K=4096, with NO ``-inf`` anywhere in the input:
        ``k_z = 1863`` where the exact answer is ``1``, giving an entirely
        FINITE output (nan=0, inf=0) with ``sum(out) = 16.9375``. The cumsum no
        longer overflows because the row max is subtracted first and the
        reduction runs in float32.

        An interim attempt at a "loudness guard" made this row fail on the
        FINITENESS assertion (nan=4096) instead; it was measured to destroy
        correct answers on ordinary rows and was REVERTED — see clause (e) of
        the D-017 anchor. Do not re-derive it.

        The exact answer on this row is pinned, more strongly, by
        ``test_defect_b_closed_max_entry_takes_all_mass_on_the_spread_row``.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            k = 4096
            z = np.full((1, k), -16.95, dtype=np.float32)
            z[:, 0] = 0.0

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            assert np.isfinite(out).all(), (
                "Defect B: sparsemax output is not finite "
                f"(nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())})"
            )
            total = float(out.sum())
            assert abs(total - 1.0) <= 1e-2, (
                f"Defect B: sum(out) = {total}, expected 1.0 "
                "(the single largest entry should take all the mass)"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_defect_b_closed_max_entry_takes_all_mass_on_the_spread_row(self) -> None:
        """The Defect-B spread row must be EXACTLY one-hot on its maximum.

        Same input as
        ``test_defect_b_closed_overflow_born_inf_excluded_from_support``, but a
        strictly stronger claim: not merely finite-and-normalised, but the
        exact answer. With a spread of 16.95 over K=4096 the projection puts
        all the mass on the single largest entry, so ``out[0, 0] == 1.0`` and
        every other entry is EXACTLY ``0.0``.

        This case used to be ``test_defect_b_loud_to_silent_conversion_is_
        accepted``, an ``xfail(strict=True)`` pin asserting only ``normalised
        or loud`` — a deliberately accepted regression where the row was
        neither correct (``sum(out) = 16.9375``) nor alarming (nan=0, inf=0).
        That premise is ELIMINATED, not merely fixed, so the case was
        retargeted rather than deleted; deleting it would have left this input
        family with only the weaker ``sum(out)`` check above. See D-004 of
        ``plan-2026-07-29T070705-9bfc04c5``.

        The ``0.0`` assertion is exact on purpose: it is what distinguishes
        "the right answer" from "an answer that happens to sum to 1".
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            k = 4096
            z = np.full((1, k), -16.95, dtype=np.float32)
            z[:, 0] = 0.0
            assert np.isfinite(z).all(), "premise: the INPUT is entirely finite"

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            assert np.isfinite(out).all(), (
                "the row is not finite "
                f"(nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())})"
            )

            # `_oracle_atol` is keyed to `min(row spread, 1.0)`, so on this row
            # (spread 16.95) it returns fp16's OUTPUT-scale budget, ~1.5e-3 —
            # not the 0.0625 that `4 * ulp(max|z|)` used to return for a claim
            # about a value in [0, 1]. See the derivation's closing paragraph.
            atol = _oracle_atol(_to_compute_dtype(z), _output_dtype())
            assert abs(float(out[0, 0]) - 1.0) <= atol, (
                f"the strict maximum must take ALL the mass: out[0, 0] = "
                f"{float(out[0, 0])}, expected 1.0 (derived atol={atol:.3e})"
            )

            rest = out[0, 1:]
            nonzero = int((rest != 0.0).sum())
            assert nonzero == 0, (
                f"{nonzero} of {rest.size} non-maximal entries are not exactly "
                f"0.0; largest is {float(np.max(np.abs(rest)))}"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_defect_c_closed_bfloat16_ramp_builds_and_runs_at_k256(self) -> None:
        """``Sparsemax()(z)`` must not raise under ``mixed_bfloat16`` at K=256.

        Defect C (``sparsemax.py:~210``, CLOSED): the ramp was built as
        ``ops.arange(1, k + 1, dtype=inputs.dtype)``, which cannot represent
        the integers ``1..K`` in a narrow dtype, so the layer RAISED. The break
        was NON-MONOTONE in K — it raised at ``mixed_bfloat16`` K=256/257 but
        was fine at K=512, and raised at ``mixed_float16`` K=2048 but was fine
        at K=4096 — which is why this case pins the exact K where it broke.
        Closed by building the ramp in ``float32`` unconditionally.

        This test was named ``test_defect_c_bfloat16_arange_ramp_raises``
        while it asserted, under ``xfail``, that the raise happened. It no
        longer does, so the name no longer says it does.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        try:
            rng = np.random.default_rng(7)
            z = rng.uniform(-5.0, 5.0, size=(2, 256)).astype(np.float32)

            # The assertion IS that this call completes. Any exception is the
            # defect; it must not be swallowed into a softer claim.
            out = ops.convert_to_numpy(Sparsemax()(z))
            assert out.shape == (2, 256)
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_defect_d_closed_float32_large_magnitude_still_projects(self) -> None:
        """float32 magnitudes at the 1.0-swamping onset must still project.

        Defect D (``sparsemax.py:~220``, CLOSED): round-off used to absorb the
        literal ``1.0`` in ``1.0 + k*z - cumsum``, so ``support == 0``
        everywhere, ``k_z == 0``, ``one_hot(-1)`` was all-zero, ``tau = -inf``
        and the output was all ``+inf``. Measured float32 onsets: 1.68e7 (K=4)
        and 7.77e7 (K=512). Closed by subtracting the row max, which moves the
        arithmetic from scale ``max|z|`` down to the row SPREAD (here exactly
        ``0.0``), so the literal ``1.0`` is never swamped.

        The name previously read ``..._swamps_the_literal_one``, which asserted
        the defect; it no longer happens.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            z = np.full((1, 4), 1.68e7, dtype=np.float32)

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            assert np.isfinite(out).all(), (
                "Defect D: sparsemax output is not finite "
                f"(nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())})"
            )
            total = float(out.sum())
            assert abs(total - 1.0) <= 1e-2, (
                f"Defect D: sum(out) = {total}, expected 1.0 "
                "(all entries equal -> uniform 0.25 each)"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_defect_e_closed_fp16_support_count_2051_is_counted_exactly(self) -> None:
        """An fp16 support of exactly 2051 must not select a masked position.

        Defect E (``sparsemax.py:~225``, CLOSED): ``k_z = ops.sum(support_mask)``
        accumulated in the compute dtype and hit float16's integer wall —
        measured on the TF/GPU tree reduction: 2049 -> 2048, 2051 -> 2052,
        4095 -> 4096 (2050 / 3000 / 4094 are exact). The 2051 -> 2052 overshoot
        selected a MASKED position whose ``z_cumsum`` is ``-inf``, so
        ``tau = -inf`` and the row died (nan=2045, inf=2051 at K=4096). The
        ``ops.where`` selection that closed Defect A could NOT help here: the
        ``-inf`` sat at the SELECTED index, not at a masked-out one. Closed by
        running the count in a never-narrowed reduction dtype, where 2051 is
        exact.

        (An earlier revision of this text claimed 4095 -> 4096 indexes OUT OF
        RANGE for depth 4096. That claim is FALSE — 4095 is a valid index —
        and was deleted; no end-to-end input reaching an out-of-range one-hot
        was constructible, since every such K raised Defect C first.)

        The name previously read ``..._overshoots_into_a_masked_index``, which
        asserted the defect; the count is now exact.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            k = 4096
            m = 2051  # support size that fp16 accumulation rounds to 2052
            z = np.full((1, k), -np.inf, dtype=np.float32)
            z[:, :m] = -31.921875  # exactly representable in float16
            masked = ~np.isfinite(z)

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            assert np.isfinite(out).all(), (
                "Defect E: sparsemax output is not finite "
                f"(nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())})"
            )
            assert np.all(out[masked] == 0.0), (
                "Defect E: masked positions are not exactly 0.0; "
                f"{int((out[masked] != 0.0).sum())} of {int(masked.sum())} are non-zero"
            )
            total = float(out.sum())
            assert abs(total - 1.0) <= 1e-2, (
                f"Defect E: sum(out) = {total}, expected 1.0 "
                f"(uniform 1/{m} over the finite entries)"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# XLA CAPABILITY GATE.
#
# This is a NEW capability, not a regression check: before the reduction was
# widened, `Sparsemax` could not even TRACE under `tf.function` at
# `mixed_float16` K=512, because `ops.arange(1, k + 1, dtype=inputs.dtype)`
# lowers to `Range[Tidx=DT_HALF]`, for which no XLA kernel exists. There is
# therefore no baseline to preserve here — every point below is newly bought
# by the float32 ramp.
#
# WHY THE FULL K GRID, INCLUDING 257 AND 2048.  The eager break was
# NON-MONOTONE in K: `mixed_bfloat16` raised at K=256 and K=257 but not at
# K=512, and `mixed_float16` raised at K=2048 but not at K=4096. There is no
# threshold to sample around, so a "representative" grid is exactly the grid
# that misses the defect. Do not thin this parametrization on the assumption
# that a monotone onset exists.
#
# WHAT IS ASSERTED.  "It compiles" alone is a test that passes the moment it
# stops raising, which is far weaker than the property that matters. Each
# point also asserts finiteness, exact `0.0` at every masked position, a
# normalised row sum, and agreement with the EAGER answer within
# `_oracle_atol` computed on the POST-CAST bits (never a raw float32 fixture:
# feeding the tolerance pre-cast bits is the same oracle-input bug documented
# at the top of this file, worth up to 5.4e-3 under fp16).
# ---------------------------------------------------------------------

#: The four policies the layer must compile under. Wider than
#: `tests/test_layers/conftest.py`'s `dtype_policy` fixture, which omits
#: `mixed_bfloat16` — the policy Defect C actually raised under at K=256.
_XLA_POLICIES = ("float32", "mixed_float16", "mixed_bfloat16", "float64")

#: `K` values. 257 and 2048 are load-bearing (see the note above); 8 is the
#: small control and 4096 the largest measured width.
_XLA_WIDTHS = (8, 256, 257, 512, 2048, 4096)


class TestSparsemaxXLACapability:
    """`Sparsemax` compiles and runs under `tf.function(jit_compile=True)`.

    24 grid points: ``K`` in ``{8, 256, 257, 512, 2048, 4096}`` times
    ``{float32, mixed_float16, mixed_bfloat16, float64}``.
    """

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    @pytest.mark.parametrize("width", _XLA_WIDTHS)
    def test_jit_compiled_sparsemax_matches_eager(
        self, width: int, policy: str
    ) -> None:
        """Compile at ``(width, policy)``, run, and compare against eager.

        :param width: Row width ``K``.
        :type width: int
        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            # Seeded, partially masked, never fully masked (the generator
            # asserts that); the mask is what makes `-inf` reach the reduction.
            z = _partially_masked_row_batch(width, 0.25, seed=width)
            # Both paths are fed the bits the layer would receive anyway, so
            # eager-vs-XLA is a comparison of ARITHMETIC, not of rounding at
            # the layer boundary.
            z_compute = _to_compute_dtype(z)
            masked = ~np.isfinite(z_compute)

            layer = Sparsemax()
            eager = ops.convert_to_numpy(layer(z_compute))

            @tf.function(jit_compile=True)
            def _compiled(t):
                return layer(t)

            # `jit_compile=True` RAISES if XLA cannot lower the graph, so the
            # call itself is the compilation assertion. Do not wrap it.
            compiled_out = _compiled(tf.convert_to_tensor(z_compute))
            xla = ops.convert_to_numpy(compiled_out)

            assert keras.backend.standardize_dtype(compiled_out.dtype) == (
                _output_dtype()
            ), (
                f"XLA output dtype {compiled_out.dtype} != compute dtype "
                f"{_output_dtype()} under {policy}"
            )

            eager64 = eager.astype(np.float64)
            xla64 = xla.astype(np.float64)

            assert np.isfinite(xla64).all(), (
                f"XLA output not finite at K={width} under {policy} "
                f"(nan={int(np.isnan(xla64).sum())}, "
                f"inf={int(np.isinf(xla64).sum())})"
            )
            assert np.all(xla64[masked] == 0.0), (
                f"XLA output is non-zero at {int((xla64[masked] != 0.0).sum())} "
                f"masked positions at K={width} under {policy}"
            )

            sums = xla64.sum(axis=-1)
            worst = float(np.max(np.abs(sums - 1.0)))
            assert worst <= 1e-2, (
                f"XLA row sums are not normalised at K={width} under "
                f"{policy}: worst |sum - 1| = {worst}"
            )

            atol = _oracle_atol(z_compute, _output_dtype())
            np.testing.assert_allclose(
                xla64,
                eager64,
                atol=atol,
                rtol=0.0,
                err_msg=(
                    f"XLA disagrees with eager at K={width} under {policy} "
                    f"(derived atol={atol})"
                ),
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# THE BACKWARD PASS.
#
# Before this class the file had ZERO gradient coverage of any kind (one grep
# hit, the word "degrades" inside a comment), while nine sibling files in this
# directory do have `GradientTape` tests. Their house shape is
# `assert gradients is not None` + `is_finite`, which would NOT catch the one
# real defect measured here, so these guards compare against the ANALYTIC VJP
# instead.
#
# `Sparsemax` has no `custom_gradient` and no `stop_gradient`: the backward pass
# is pure autodiff through `max` / `sort` / `cumsum` / `where` / `maximum`. From
# Martins & Astudillo 2016 Prop. 1, with support `S = {i : p_i > 0}`, `k = |S|`:
#
#     J_ij = [i in S][j in S] * (delta_ij - 1/k)        (0 elsewhere)
#     grad_z_i = [i in S] * ( v_i - (1/k) * sum_{j in S} v_j )
#
# `S` and `k` come from the EXACT-RATIONAL oracle fed POST-CAST bits, never from
# the layer's own output and never from an intended Python literal — the same
# rule the forward oracles enforce, for the same reason.
#
# WHY BOUNDARY-FREE ROWS ONLY.  On a support-BOUNDARY row (some `z_i == tau`
# exactly, so `p_i == 0` yet `shifted - tau == 0`), `ops.maximum` at
# `sparsemax.py:556` routes a full gradient to an entry that `k_z` excludes.
# The result is not even a Clarke subgradient (228/300 random directions give a
# mixing coefficient outside [0, 1]), with errors up to 3.24. That is an OPEN
# defect, deliberately not fixed here, and it is pinned separately rather than
# absorbed into a widened tolerance. Excluding those rows is what keeps this
# guard a measurement of the other 46; it is NOT a claim that they are correct.
# ---------------------------------------------------------------------


class TestSparsemaxBackwardPass:
    """`Sparsemax`'s gradient vs the analytic VJP, on boundary-free rows.

    Measured on GPU at every point below (`CUDA_VISIBLE_DEVICES=1`): 46 / 43 /
    45 / 46 boundary-free rows checked under float32 / mixed_float16 /
    mixed_bfloat16 / float64, zero failures, worst error 1.69 ulp of the output
    dtype against a 4-ulp budget.
    """

    @staticmethod
    def _layer_gradient(
        z_compute: np.ndarray, v_compute: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backprop ``sum(sparsemax(z) * v)`` through a default `Sparsemax`.

        Internal to this class. Both arguments must ALREADY be in the active
        policy's compute dtype (:func:`_to_compute_dtype`), so that the tape
        sees the bits the layer receives and no cast happens inside the taped
        region — a cast there would add a rounding the derived budget does not
        carry.

        :param z_compute: 2-D logit batch in the compute dtype.
        :type z_compute: np.ndarray
        :param v_compute: Upstream ``dL/dp``, same shape and dtype as ``z``.
        :type v_compute: np.ndarray

        :return: ``(grad_z, forward_output)``, both as float64 arrays.
        :rtype: tuple[np.ndarray, np.ndarray]

        :raises AssertionError: if the tape returns no gradient at all, which
            would make every downstream comparison vacuous.
        """
        assert z_compute.dtype == v_compute.dtype, (
            f"z is {z_compute.dtype} but v is {v_compute.dtype}; both must be "
            "post-cast so no rounding happens inside the taped region"
        )
        layer = Sparsemax()
        x = tf.convert_to_tensor(z_compute)
        v = tf.convert_to_tensor(v_compute)
        with tf.GradientTape() as tape:
            tape.watch(x)
            p = layer(x)
            loss = tf.reduce_sum(p * tf.cast(v, p.dtype))
        grad = tape.gradient(loss, x)
        assert grad is not None, (
            "the tape returned no gradient for the input — Sparsemax has "
            "become non-differentiable and every assertion here is vacuous"
        )
        return (
            np.asarray(grad, dtype=np.float64),
            np.asarray(p, dtype=np.float64),
        )

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_gradient_matches_analytic_vjp(self, policy: str) -> None:
        """``grad_z == s * (v - mean_S(v))`` on every boundary-free row.

        Runs the whole committed `_attack_corpus()` PLUS one random batch per
        `_XLA_WIDTHS` (which is what keeps the non-monotone K points 256/257 in
        the gradient grid too), under each of the four policies.

        The tolerance is `_grad_atol`, i.e. ``4 * u_c(S) + 2 * u_r(S)`` at
        ``S = max(|v|, 1)`` — see its derivation block. It must NOT be
        tightened: measured slack over this corpus is 2.65x at its tightest
        (bfloat16, 1.51 of 4.00 ulp), and a 2-ulp floor would leave float32 at
        1.18x, i.e. red on ordinary reduction-order drift.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            out_dtype = _output_dtype()
            rng = np.random.default_rng(9091)

            batches: List[Tuple[str, np.ndarray]] = [
                (label, z) for label, z, _ in _attack_corpus()
            ]
            batches += [
                (
                    f"random_K{width}",
                    (rng.standard_normal((3, width)) * 3.0).astype(np.float32),
                )
                for width in _XLA_WIDTHS
            ]

            checked = 0
            boundary_rows = 0
            skipped: List[str] = []

            for label, z in batches:
                z_compute = _to_compute_dtype(z)

                # The `any(isfinite)` predicate the exact oracle already uses:
                # under `mixed_float16` the literal 1.68e7 casts to `inf`, so
                # three corpus batches have no finite entry left and are out of
                # the oracle's scope. They are SKIPPED here, never deleted from
                # the corpus — they remain in scope for their own policies.
                if not np.isfinite(
                    np.asarray(z_compute, dtype=np.float64)
                ).any(axis=-1).all():
                    skipped.append(label)
                    continue

                v = rng.standard_normal(z.shape).astype(np.float32)
                v_compute = _to_compute_dtype(v)
                grad, _ = self._layer_gradient(z_compute, v_compute)

                v64 = np.asarray(v_compute, dtype=np.float64)
                atol = _grad_atol(v_compute, out_dtype)

                for i, row in enumerate(_exact_sparsemax(z_compute)):
                    tau = row["tau"]
                    # A row is a BOUNDARY row iff some finite entry equals
                    # `tau` EXACTLY, in exact rational arithmetic — that is the
                    # `ops.maximum` tie that F-02 mis-differentiates. Excluded
                    # here, pinned separately.
                    if any(
                        np.isfinite(float(val))
                        and Fraction(*float(val).as_integer_ratio()) == tau
                        for val in z_compute[i]
                    ):
                        boundary_rows += 1
                        continue

                    support = np.array(
                        [float(o) > 0.0 for o in row["out"]], dtype=bool
                    )
                    k = int(support.sum())
                    assert k >= 1, f"{label} row {i}: empty support is impossible"

                    reference = np.zeros(v64.shape[-1], dtype=np.float64)
                    reference[support] = (
                        v64[i][support] - v64[i][support].sum() / k
                    )

                    err = float(np.max(np.abs(grad[i] - reference)))
                    checked += 1
                    assert err <= atol, (
                        f"gradient != analytic VJP at {label} row {i} "
                        f"(policy={policy}, K={z.shape[-1]}, k_z={k}): "
                        f"max abs error {err:.6e} > derived atol {atol:.6e} "
                        f"({err / atol:.2f}x the 4-ulp budget)"
                    )

            # Anti-vacuity: measured 46 / 43 / 45 / 46 boundary-free rows per
            # policy (the fp16 figure is lower because of the three skips).
            # A collapse below this floor means the corpus, the skip predicate
            # or the boundary filter silently ate the grid.
            assert checked >= 40, (
                f"only {checked} boundary-free rows were checked under "
                f"{policy} (expected >= 40; boundary rows excluded="
                f"{boundary_rows}, batches skipped={skipped}) — this guard has "
                "gone vacuous"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_rowmax_shift_is_gradient_neutral(self, policy: str) -> None:
        """``grad(z)`` is BIT-IDENTICAL to ``grad(z - rowmax(z))``.

        The committed guard for the ``ops.max`` route at `sparsemax.py:205-206`
        — a gradient path the layer did not have before the row-max shift was
        added to close Defects B/D. Because every row of the sparsemax Jacobian
        sums to exactly 0, the upstream reaching `ops.max` sums to 0 too, so
        the route contributes EXACTLY nothing.

        Asserted at ZERO tolerance (`assert_array_equal`, not `assert_allclose`)
        because the claim is exact and was measured exact — 0.000e+00 at every
        point, all four policies. Do not weaken this to `allclose`: an
        approximate version would pass even if the route started contributing.

        The pre-shift is done in NUMPY at the layer boundary, in the compute
        dtype, so no source edit is needed and both calls receive representable
        bits.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            rng = np.random.default_rng(2026)
            for width in _XLA_WIDTHS:
                for frac in (0.0, 0.25):
                    z = _partially_masked_row_batch(
                        width, frac, seed=width + int(frac * 100)
                    )
                    z_compute = _to_compute_dtype(z)
                    v_compute = _to_compute_dtype(
                        rng.standard_normal(z.shape).astype(np.float32)
                    )

                    # Row max over the FINITE entries only; `-inf` positions
                    # stay `-inf` under the shift, so the mask survives it.
                    z64 = np.asarray(z_compute, dtype=np.float64)
                    row_max = np.max(
                        np.where(np.isfinite(z64), z64, -np.inf),
                        axis=-1,
                        keepdims=True,
                    )
                    shifted = (
                        z_compute - row_max.astype(z_compute.dtype)
                    ).astype(z_compute.dtype)

                    grad_raw, _ = self._layer_gradient(z_compute, v_compute)
                    grad_shifted, _ = self._layer_gradient(shifted, v_compute)

                    np.testing.assert_array_equal(
                        grad_raw,
                        grad_shifted,
                        err_msg=(
                            f"the row-max shift is NOT gradient-neutral at "
                            f"K={width} masked_fraction={frac} under {policy}: "
                            f"worst |delta| = "
                            f"{float(np.max(np.abs(grad_raw - grad_shifted)))}"
                        ),
                    )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_masked_gradients_are_exactly_zero_and_never_nan(
        self, policy: str
    ) -> None:
        """`-inf` positions get gradient exactly ``0.0``; nothing is NaN or inf.

        Three claims, at all four policies (the last two were measured at only
        two policies before this plan's GPU round extended them):

        1. **Non-vacuity control first.** A HALF-masked row must produce
           NON-ZERO gradient at its support, otherwise "the masked slots are
           zero" would be true of every input and would measure nothing. The
           row and upstream are chosen so the answer is exactly representable
           in every dtype: support ``{0, 1}``, ``k = 2``, upstream
           ``[1, -1, 0, ...]``, so ``grad == [1, -1, 0, 0, 0, 0, 0, 0]``
           EXACTLY — measured identical under all four policies.
        2. A FULLY `-inf`-masked row makes the forward all-NaN (the known,
           documented behaviour: such rows are rescued upstream by
           `apply_attention_mask(rescue_axis=...)` and are out of the forward
           oracle's scope) — but the GRADIENT is exactly ``0.0``, not NaN. That
           asymmetry is the whole point of the case.
        3. Over the WHOLE `_attack_corpus()`, including the three rows that
           cast to non-finite under `mixed_float16`, the gradient is finite and
           is exactly ``0.0`` at every non-finite input position. Nothing is
           skipped here: unlike the exact oracle, this claim is well-defined on
           an out-of-scope row, and it was measured to hold on all of them.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            # --- (1) the non-vacuity control ---------------------------------
            z = np.full((1, 8), -np.inf, dtype=np.float32)
            z[0, :4] = np.array([2.0, 1.5, 0.5, -1.0], dtype=np.float32)
            v = np.zeros((1, 8), dtype=np.float32)
            v[0, 0], v[0, 1] = 1.0, -1.0

            grad, _ = self._layer_gradient(
                _to_compute_dtype(z), _to_compute_dtype(v)
            )
            expected = np.zeros((1, 8), dtype=np.float64)
            expected[0, 0], expected[0, 1] = 1.0, -1.0
            np.testing.assert_array_equal(
                grad,
                expected,
                err_msg=(
                    f"CONTROL ({policy}): a half-masked row's gradient is "
                    f"{grad.tolist()}, expected {expected.tolist()} — if it is "
                    "all zeros, the assertions below measure nothing"
                ),
            )

            # --- (2) the fully-masked row: NaN forward, ZERO gradient ---------
            z_all_masked = _to_compute_dtype(
                np.full((1, 8), -np.inf, dtype=np.float32)
            )
            grad, forward = self._layer_gradient(
                z_all_masked, _to_compute_dtype(np.ones((1, 8), dtype=np.float32))
            )
            assert int(np.isnan(forward).sum()) == forward.size, (
                f"premise ({policy}): a fully -inf-masked row's FORWARD is "
                f"documented as all-NaN, but {int(np.isnan(forward).sum())} of "
                f"{forward.size} entries are NaN — the case has changed shape "
                "and claim (2) below is no longer the asymmetry it pins"
            )
            assert np.all(grad == 0.0), (
                f"a fully -inf-masked row's gradient is not exactly 0.0 under "
                f"{policy}: nan={int(np.isnan(grad).sum())}, "
                f"inf={int(np.isinf(grad).sum())}, grad={grad.tolist()}"
            )

            # --- (3) the whole corpus: finite, and exactly 0 where masked -----
            rng = np.random.default_rng(4242)
            for label, z, _measured_under in _attack_corpus():
                z_compute = _to_compute_dtype(z)
                v_compute = _to_compute_dtype(
                    rng.standard_normal(z.shape).astype(np.float32)
                )
                grad, _ = self._layer_gradient(z_compute, v_compute)

                assert np.isfinite(grad).all(), (
                    f"gradient is not finite on corpus row {label} under "
                    f"{policy}: nan={int(np.isnan(grad).sum())}, "
                    f"inf={int(np.isinf(grad).sum())} of {grad.size}"
                )

                masked = ~np.isfinite(np.asarray(z_compute, dtype=np.float64))
                if masked.any():
                    nonzero = int((grad[masked] != 0.0).sum())
                    assert nonzero == 0, (
                        f"{nonzero} of {int(masked.sum())} masked positions "
                        f"have a non-zero gradient on corpus row {label} under "
                        f"{policy}; largest is "
                        f"{float(np.max(np.abs(grad[masked])))}"
                    )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# `SparsemaxLoss` x the global dtype policy  (MEASURED 2026-07-29)
# ---------------------------------------------------------------------
# WHAT WAS PROBED.  `losses/sparsemax_loss.py:225` builds its inner
# `Sparsemax()` EAGERLY in `__init__`, so that sub-layer's dtype policy is
# frozen at LOSS-CONSTRUCTION time. The open question was what happens when the
# ambient policy differs at CALL time.
#
# WHAT WAS MEASURED (both orders, plus same-policy controls, on GPU/TF):
#
#   construct   call        loss.dtype  inner compute  result
#   ---------   ---------   ----------  -------------  --------------------
#   float32     float32     float32     float32        OK   -> float32
#   float32     fp16        float32     float32        OK   -> float32
#   float32     float64     float32     float32        OK   -> float32
#   fp16        {any}       float32     float16        RAISE InvalidArgument
#   bf16        {any}       float32     bfloat16       RAISE InvalidArgument
#   float64     {any}       float32     float64        RAISE InvalidArgument
#
# THE RULE, and why the call-time policy turned out to be the wrong axis.
# `keras.losses.Loss` takes its dtype from `backend.floatx()`, NOT from the
# global mixed-precision policy — so `loss.dtype` is `float32` under EVERY
# policy above, and `Loss.__call__` casts both `y_true` and `y_pred` to
# float32 before `call()` runs. The inner layer, by contrast, honours the
# policy captured at construction and returns ITS compute dtype. When the two
# disagree, `y_pred - p` at `sparsemax_loss.py:258` is float32 minus
# float16/bfloat16/float64 and TensorFlow raises. The ambient policy at call
# time is therefore IRRELEVANT: the loss is a pure function of the policy in
# force when it was CONSTRUCTED.
#
# RULING: (a) a loud dtype mismatch, not (b) a silently-narrow reduction.
# `D-007` made the reduction dtype derive from `inputs.dtype` rather than
# `self.compute_dtype`, which is exactly why route (b) is closed here.
#
# THIS IS A PRE-EXISTING GAP IN `losses/sparsemax_loss.py`, NOT A REGRESSION
# FROM THIS PLAN.  Measured on `ba9efff0` (pre-fix HEAD): `Sparsemax` already
# returned float16/bfloat16/float64 under the corresponding policies. This
# plan's `D-003` deliberately PRESERVED that dtype contract (`tau` is cast back
# so the observable output dtype stays the compute dtype), so the fix neither
# introduced nor removed the mismatch.
#
# THE TESTS BELOW PIN CURRENT BEHAVIOUR, INCLUDING THE DEFECT.  Fixing it
# means one line in `losses/sparsemax_loss.py` (cast `p` to `y_pred.dtype`
# before the subtraction), and that file is outside this plan's Files To
# Modify. `test_loss_constructed_under_a_narrow_policy_raises` therefore
# asserts the RAISE — it is a guard against silent drift, not an endorsement.
# When the loss is fixed, that test will fail loudly and must be rewritten to
# assert the working behaviour; that is the intended handoff.
# ---------------------------------------------------------------------

#: Policies whose compute dtype differs from `backend.floatx()`, i.e. every
#: policy under which constructing a `SparsemaxLoss` currently poisons it.
_LOSS_BROKEN_POLICIES = ("mixed_float16", "mixed_bfloat16", "float64")


class TestSparsemaxLossDtypePolicy:
    """`SparsemaxLoss`'s eager inner `Sparsemax()` vs. the global policy.

    Probe of `losses/sparsemax_loss.py:225` (plan step 6). That file is
    PROBED, NOT MODIFIED — see the note above for the measured rule and for
    why the defect these tests pin is escalated rather than fixed here.
    """

    @pytest.mark.parametrize("construct_policy", ("float32",) + _LOSS_BROKEN_POLICIES)
    def test_inner_sparsemax_policy_is_frozen_at_construction(
        self, construct_policy: str
    ) -> None:
        """The sub-layer keeps its construction-time policy across a later change.

        This is the MECHANISM the other two tests rest on: changing the global
        policy after `__init__` does not reach the already-built sub-layer.

        :param construct_policy: Policy in force when the loss is constructed.
        :type construct_policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy(construct_policy)
            loss = SparsemaxLoss(from_logits=True)
            frozen = loss.sparsemax.compute_dtype
            assert frozen == keras.mixed_precision.global_policy().compute_dtype

            # Flip to the opposite policy AFTER construction.
            other = "float32" if construct_policy != "float32" else "mixed_float16"
            keras.mixed_precision.set_global_policy(other)

            assert loss.sparsemax.compute_dtype == frozen, (
                f"inner Sparsemax followed the ambient policy: expected the "
                f"construction-time {frozen!r}, got "
                f"{loss.sparsemax.compute_dtype!r} after switching to {other}"
            )
            # `Loss` itself ignores the policy entirely - it tracks
            # `backend.floatx()`. This is the other half of the mismatch.
            assert loss.dtype == keras.backend.floatx()
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize("call_policy", ("float32", "mixed_float16"))
    @pytest.mark.parametrize("construct_policy", _LOSS_BROKEN_POLICIES)
    def test_loss_constructed_under_a_narrow_policy_raises(
        self, construct_policy: str, call_policy: str
    ) -> None:
        """PINS A KNOWN GAP in `losses/sparsemax_loss.py` (do not "fix" this test).

        A `SparsemaxLoss` built under any policy whose compute dtype differs
        from `backend.floatx()` is unusable: `Loss.__call__` casts `y_pred` to
        float32 while the frozen sub-layer returns its own dtype, so
        `y_pred - p` raises. The call-time policy does not change this, which
        is why it is parametrized here.

        :param construct_policy: Policy in force at construction.
        :type construct_policy: str
        :param call_policy: Policy in force at call time (measured irrelevant).
        :type call_policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy(construct_policy)
            loss = SparsemaxLoss(from_logits=True)

            keras.mixed_precision.set_global_policy(call_policy)
            # `masked_fraction=0.0` - the no-`inf` control. `SparsemaxLoss`'s
            # `0.5*||z - p||^2 - z^T y` is undefined on `-inf` logits, so the
            # masked corpus belongs to the LAYER's tests, not the loss's.
            z = _partially_masked_row_batch(16, 0.0, seed=606)
            z_compute = _to_compute_dtype(z)
            y = np.zeros_like(z_compute)
            y[:, 0] = 1.0

            with pytest.raises(tf.errors.InvalidArgumentError, match="Sub"):
                loss(ops.convert_to_tensor(y), ops.convert_to_tensor(z_compute))
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_loss_built_under_float32_survives_a_later_fp16_policy(self) -> None:
        """The one order that works: build under float32, then switch to fp16.

        Asserts the observable consequences: the loss still runs, it returns
        **float32** (not float16 - it is not policy-aware at all), and its
        value matches the Fenchel-Young formula evaluated on the EXACT-rational
        sparsemax of the bits the layer actually received.
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy("float32")
            loss = SparsemaxLoss(from_logits=True)

            keras.mixed_precision.set_global_policy("mixed_float16")
            # No-`inf` control: see the note in the raising test above.
            z = _partially_masked_row_batch(16, 0.0, seed=607)
            # The fp16 bits the caller hands in; `Loss.__call__` upcasts these
            # to float32 losslessly, so these ARE the received bits.
            z_compute = _to_compute_dtype(z)
            y = np.zeros_like(z_compute)
            y[:, 0] = 1.0

            out = loss(
                ops.convert_to_tensor(y), ops.convert_to_tensor(z_compute)
            )
            assert keras.backend.standardize_dtype(out.dtype) == "float32", (
                f"loss dtype {out.dtype} - expected float32 even under "
                f"mixed_float16 (keras.losses.Loss tracks backend.floatx())"
            )

            # Reference: L = 0.5 * ||z - p||^2 - z^T y, reduced by the loss's
            # default `sum_over_batch_size`.
            z64 = np.asarray(z_compute, dtype=np.float64)
            p64 = _exact_rows_as_float64(z_compute)
            y64 = np.asarray(y, dtype=np.float64)
            per_sample = 0.5 * np.sum((z64 - p64) ** 2, axis=-1) - np.sum(
                z64 * y64, axis=-1
            )
            expected = float(np.mean(per_sample))

            # First-order propagation of the per-element sparsemax tolerance
            # through the loss: dL/dp_j = -(z_j - p_j), so an elementwise error
            # of `eps` costs at most `sum_j |z_j - p_j| * eps`. `_TF32_ATOL_FLOOR`
            # covers the float32 summation itself.
            eps = _oracle_atol(z_compute, _output_dtype())
            atol = _TF32_ATOL_FLOOR + eps * float(
                np.max(np.sum(np.abs(z64 - p64), axis=-1))
            )
            actual = float(ops.convert_to_numpy(out))
            assert abs(actual - expected) <= atol, (
                f"loss {actual} != Fenchel-Young reference {expected} "
                f"(derived atol={atol})"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
