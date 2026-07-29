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
from dl_techniques.layers.attention.capsule_routing_attention import (
    CapsuleRoutingSelfAttention,
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
# charged here.  NO committed test compares the LAYER's ``k_z`` to the oracle's:
# ``k_z`` is internal to ``call()`` and is not observable from the output.  The
# risk was measured benign — ``tau`` is invariant across an exact tie, so a
# mis-selection at a tie cannot change the answer — but that is a MEASUREMENT,
# not a guard.
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
#
# DECISION plan-2026-07-29T110112-09832856/D-018
# DO NOT CITE `sparsemax.py` BY LINE NUMBER FROM THIS MODULE. Not here, not in
# a docstring, not inside an `xfail` reason string. Cite the SYMBOL (the
# function plus the expression) and, where one exists, the anchored grep that
# finds it. This is not style: step 9 of this plan inserted 47 lines into
# `sparsemax.py` and every one of the 13 `sparsemax.py:NNN` citations in this
# file silently came to name a different line — including all four copies of
# the lockstep pointer just above, which came to name unrelated prose about
# 5-D video transformers. A file:line citation is a hand-maintained lockstep
# invariant with no guard, i.e. a latent defect (`plans/LESSONS.md:20`), and
# renumbering them would only reset the clock.
def _reduction_dtype(input_dtype: str) -> str:
    """The dtype the layer runs its sort/cumsum reduction in — the D-007 rule.

    THE SINGLE SOURCE OF TRUTH IS the ``reduction_dtype = ...`` assignment
    inside ``Sparsemax.call``
    (``src/dl_techniques/layers/activations/sparsemax.py``).
    This function is its ONE mirror on the test side. There are exactly two
    statements of this rule in the repository: the source and this helper.

    CITED BY SYMBOL, NEVER BY LINE NUMBER. This pointer used to read
    ``sparsemax.py:225-227``; step 9 of
    ``plan-2026-07-29T110112-09832856`` inserted 47 lines above it and every
    copy of that pointer silently came to name unrelated prose about 5-D video
    transformers. A file:line citation is a hand-maintained lockstep
    invariant, i.e. a latent defect (`plans/LESSONS.md:20`). Use the grep
    below, which survives insertion.

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


# DECISION plan-2026-07-29T110112-09832856/D-016
# `apply_tf32_floor=False` EXISTS SO THE DERIVATION BELOW HAS EXACTLY ONE HOME.
# Do NOT "just write the three terms out" at a caller that wants the unfloored
# value — that was the suggested remedy for the loss guard's floored `eps`, and
# taking it would have created a SECOND hand-maintained copy of this derivation
# in a module whose whole D-008 history is the deletion of exactly that pattern
# (`plans/LESSONS.md:20`). Do NOT delete or gate `_TF32_ATOL_FLOOR` itself
# either: several forward guards (G6, the whole G8 grid) depend on the floored
# value, and lowering it for them is a separate, unmeasured change.
# The default is `True`, so every pre-existing caller is BIT-IDENTICAL: the
# `True` branch is the same `max(...)` expression, unmoved.
def _oracle_atol(
    z: np.ndarray, output_dtype: str, *, apply_tf32_floor: bool = True
) -> float:
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
    :param apply_tf32_floor: Keyword-only, default ``True`` (the historical
        behaviour, bit-identical). Pass ``False`` to get the DERIVED term
        alone, without the ``_TF32_ATOL_FLOOR`` clamp. Only a caller that can
        argue TF32 cannot reach its measurement may do so — the floor is a
        float32 matmul/convolution belt and this layer has neither (the same
        argument :func:`_grad_atol` makes for carrying no floor at all). At
        float64 and float32 the derived term is ~7.8e-16 and ~4.2e-07, so the
        clamp is 12 and 3 orders of magnitude wide there and silently dominates
        anything built on top of it.
    :type apply_tf32_floor: bool

    :return: Absolute tolerance for ``np.testing.assert_allclose``.
    :rtype: float
    """
    np_dtype = _NUMPY_DTYPE[output_dtype]
    assert np_dtype is not None, f"no numpy dtype for {output_dtype!r}"

    # The reduction dtype the layer will use — the D-007 rule, stated once in
    # `_reduction_dtype` and nowhere else on this side. See that helper's
    # docstring for the lockstep obligation against the `reduction_dtype = ...`
    # assignment in `Sparsemax.call` (cited by symbol, not by line: see D-018).
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
    return max(_TF32_ATOL_FLOOR, derived) if apply_tf32_floor else derived


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
# statements of it in the repository: the `reduction_dtype = ...` assignment in
# `Sparsemax.call` (the SOURCE OF TRUTH, cited by symbol — see D-018 for why not
# by line) and `_reduction_dtype` above (its one test-side mirror).  Both this
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
#   1.0 u_c(S)   THE CAST BACK.  `Sparsemax.call`'s
#                ``ops.cast(shifted, reduction_dtype)`` is differentiable, so
#                the gradient crosses back from the reduction dtype into the
#                compute dtype and is rounded once on the way.
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
# DECISION plan-2026-07-29T110112-09832856/D-017
# DERIVED vs MEASURED — THE ONE HOME FOR THESE NUMBERS.
#
# Do NOT restate the slack figure anywhere else (not in `_grad_atol`'s
# docstring, not in `TestSparsemaxBackwardPass`'s, not in a plan file). Two
# earlier restatements of it drifted to two different wrong values and a third
# value fell out of dividing the two columns of the table that stated them,
# because the atol column was quoted at `S = 1.0` while the error column was
# measured at the corpus's real `S ~ 2.5-3.4` (`plans/LESSONS.md:20`). Every
# figure below is quoted AT ITS OWN WORST POINT'S `S`, so `err / atol` is an
# arithmetic identity a reader can check on the line.
#
# GRID (name it, per the rule two paragraphs down): the loop of
# `test_gradient_matches_analytic_vjp` exactly as shipped — the whole committed
# `_attack_corpus()` PLUS one `rng(9091)` batch per `_XLA_WIDTHS`, with the
# `any(isfinite)` skip and the exact-rational boundary-row filter applied.
# The per-policy boundary-free ROW COUNTS this reproduced (which is how the
# grid was confirmed to be the shipped one) are stated once, beside the
# anti-vacuity floor they justify inside that test — not restated here.
# Measured on GPU (`CUDA_VISIBLE_DEVICES=1`, RTX 4070), serial, two independent
# runs at `iter-1/step-13`, agreeing to every digit.  The constants were fixed
# by the accounting FIRST; these are what the layer then measured.
#
#   policy          worst point               S       atol=4u_c+2u_r  worst err     err/atol  slack   err in u_c(S)
#   float32         random_K256 row 0         2.863578  1.430511e-06  1.577040e-07  0.1102    9.07x   0.66
#   mixed_float16   property_w512_f0.9_s5 r1  3.380859  7.812977e-03  3.515625e-03  0.4500    2.22x   1.80
#   mixed_bfloat16  random_K256 row 0         2.859375  6.250048e-02  9.602865e-03  0.1536    6.51x   0.62
#   float64         property_w17_f0.0_s1 r2   2.577674  2.664535e-15  2.220446e-16  0.0833    12.00x  0.50
#
# DECISION plan-2026-07-29T110112-09832856/D-007
# DO NOT TIGHTEN THIS BUDGET.  The real headroom is **2.22x at its tightest, and
# the tightest policy is `mixed_float16`** — not bfloat16, and not the ~5x an
# earlier record claimed.  Two corrections are folded in here and both were
# quoting-errors of the same species, so read the rule before the numbers:
#   (i)  the ~5x came from quoting the RANDOM-K-GRID worst (0.73 ulp) as if it
#        were the corpus worst.  NAME THE GRID WHEN QUOTING EITHER.
#   (ii) the 2.65x-at-bfloat16 that replaced it came from dividing an atol
#        quoted at `S = 1.0` by an error measured at the corpus's real `S`.
#        NAME THE SCALE `S` WHEN QUOTING EITHER.  At a common `S` bfloat16 has
#        6.51x, i.e. 2.9x MORE headroom than that figure claimed, while
#        `mixed_float16` has 2.22x, i.e. 2.0x LESS.
# Halving the budget to `2 * u_c(S)` would leave `mixed_float16` at **1.11x**
# (3.906250e-03 against a measured 3.515625e-03) — thin enough that ordinary
# reduction-order drift turns this guard permanently red for no signal, which is
# a guard that measures nothing (`plans/LESSONS.md:47`).  Re-measure on the
# target device before touching any constant here.  See `D-007` (superseded
# figures) and `D-017` of `plan-2026-07-29T110112-09832856`.
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


def _exact_rows_along_axis(z_compute: np.ndarray, axis: int) -> np.ndarray:
    """:func:`_exact_rows_as_float64` for an N-D batch reduced along ``axis``.

    THE WHOLE ADAPTER IS THREE LINES, AND THAT IS THE POINT: no second oracle
    exists for the moved-axis / rank>2 / dynamic-K surface, and none may be
    written. ``np.moveaxis`` to last, ``reshape(-1, K)``, run the ONE exact
    rational oracle, reshape and move back. The layer does structurally the
    same thing internally (``Sparsemax.call``'s transpose/reshape shim moves
    the reduction axis to last and reshapes to ``(-1, k)``) — but the two
    agree only because
    both are correct, not because they share code: this adapter is pure numpy
    index bookkeeping and imports nothing from the layer.

    ``np.moveaxis`` accepts negative ``axis`` with the ordinary Python
    convention, so ``-2`` on a rank-4 batch is ``ndim - 2`` — the same
    normalisation ``axis = self.axis if self.axis >= 0 else ndim + self.axis``
    in ``Sparsemax.call`` performs. It also RAISES ``AxisError``
    on an out-of-range value, which is exactly what the layer does not do
    (F-06); out-of-range axes are out of scope here and are step 9's subject.

    :param z_compute: Batch of any rank >= 1, ALREADY cast to the active
        policy's compute dtype — use :func:`_to_compute_dtype`. The dtype
        assertion lives in :func:`_exact_sparsemax` and fires through here.
    :type z_compute: np.ndarray
    :param axis: The reduction axis, either sign, IN RANGE.
    :type axis: int

    :return: float64 array of the SAME shape as ``z_compute``, holding the
        exact rational projection taken along ``axis``.
    :rtype: np.ndarray
    """
    moved = np.moveaxis(np.asarray(z_compute), axis, -1)
    exact = _exact_rows_as_float64(moved.reshape(-1, moved.shape[-1]))
    return np.moveaxis(exact.reshape(moved.shape), -1, axis)


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

    # --- the four pin inputs, verbatim from the `test_defect_{b,c,d,e}_closed_*`
    # methods of `TestSparsemaxClosedDefects` (cited by symbol, not by line —
    # see D-018) ---
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


#: The four policies the layer must compile and project under. Wider than
#: `tests/test_layers/conftest.py`'s `dtype_policy` fixture, which omits
#: `mixed_bfloat16` — the policy Defect C actually raised under at K=256.
#: Defined here rather than beside `_XLA_WIDTHS` because `TestSparsemax`'s G9
#: capsule-integration case parametrizes over it and decorators run at
#: class-definition time.
_XLA_POLICIES = ("float32", "mixed_float16", "mixed_bfloat16", "float64")


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

    # -----------------------------------------------------------------
    # G9 — `axis` RANGE validation (F-06). See the
    # `# DECISION plan-2026-07-29T110112-09832856/D-014` anchor in
    # `src/dl_techniques/layers/activations/sparsemax.py`'s `call()`.
    #
    # WHY THESE CASES ARE SAFE TO RUN IN-PROCESS NOW, AND WERE NOT BEFORE.
    # Pre-fix, `ndim + axis in [-ndim, -2]` reached a TF C++ `LOG(FATAL)`
    # (`tensor_shape.cc:356 Check failed: d >= 0`) and ABORTED THE
    # INTERPRETER (SIGABRT, exit 134). No `pytest.raises`, no `except`, no
    # `pytest.warns` can contain that — the exploration had to run one case
    # per process to measure the band at all. The range check rejects in
    # PYTHON, strictly before `list.pop` and `ops.transpose`, so TF never
    # sees the bad permutation and the whole band is ordinary
    # `pytest.raises(ValueError)` territory. Re-verified by running this
    # class in-process (no abort) AND, with the check commented out, by
    # `subprocess.run` returning `-6` / 134 — that subprocess exit code IS
    # this band's RED proof, because an aborted process cannot report a
    # failed assertion.
    #
    # PROVEN RED (GPU, `CUDA_VISIBLE_DEVICES=1`), validation commented out
    # IN PLACE in `call()` and restored:
    #   * `test_out_of_range_axis_raises_value_error`, silent-wrong-shape
    #     band: `DID NOT RAISE <class 'ValueError'>` — the `pytest.raises`
    #     context is the assertion that fires, at every rank-2/3/4 case.
    #   * same test, former-SIGABRT band: cannot be proven RED in-pytest at
    #     all, because the process dies. Proven instead by exit code in an
    #     isolated subprocess (returncode 134 / -6 vs 0 with the check in
    #     place), which is a STRICTLY stronger statement than a failed
    #     assertion.
    #   * `test_bool_axis_is_rejected`: `DID NOT RAISE` with the
    #     `isinstance(axis, bool)` clause removed — `Sparsemax(axis=True)`
    #     constructs and behaves as `axis=1`.
    #   * `test_every_in_range_axis_is_accepted_at_ranks_1_to_4` is the
    #     OVER-REJECTION control and goes red the other way: narrowing the
    #     predicate to `0 <= self.axis < ndim` fires its `ValueError` at
    #     every negative axis (10 of the 20 (rank, axis) points).
    # -----------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape,axis,band",
        [
            # BAND 1 — `ndim + axis == -1`: pre-fix this returned a SILENTLY
            # TRANSPOSED, non-normalised answer (`(2,4,5)` -> `(2,5,4)`,
            # last-axis sums `[0.1687, 1.3085, 1.0049, 0.7024]`) while
            # `compute_output_shape` kept declaring the INPUT shape.
            ((4, 5), -3, "silent_wrong_shape"),
            ((2, 4, 5), -4, "silent_wrong_shape"),
            ((2, 3, 4, 5), -5, "silent_wrong_shape"),
            # BAND 2 — `ndim + axis in [-ndim, -2]`: pre-fix an UNCATCHABLE
            # process abort (SIGABRT, exit 134).
            ((4, 5), -4, "former_sigabrt"),
            ((2, 4, 5), -5, "former_sigabrt"),
            ((2, 4, 5), -6, "former_sigabrt"),
            ((2, 3, 4, 5), -6, "former_sigabrt"),
            ((2, 3, 4, 5), -8, "former_sigabrt"),
            # BAND 3 — `ndim + axis < -ndim` or `axis >= ndim`: pre-fix an
            # `IndexError` from `list.pop`, loud but naming neither the
            # offending axis nor the rank.
            ((2, 4, 5), 3, "former_index_error"),
            ((2, 4, 5), -7, "former_index_error"),
            ((4, 5), 9, "former_index_error"),
            # Rank-1 boundaries. Not part of the exploration's measured band
            # table (which swept ranks 1-4 only for the two defect bands);
            # included because rank-1 is a real call shape for this layer.
            ((5,), 1, "rank1_out_of_range"),
            ((5,), -2, "rank1_out_of_range"),
        ],
    )
    def test_out_of_range_axis_raises_value_error(
        self, shape: Tuple[int, ...], axis: int, band: str
    ) -> None:
        """An out-of-range ``axis`` must raise ``ValueError`` at call time.

        The message must name the offending axis, the input rank and the legal
        range — the pre-fix `IndexError` ("pop index out of range") named none
        of the three, and the other two bands said nothing at all because one
        returned a wrong answer and the other killed the process.

        :param shape: Input shape handed to the layer.
        :type shape: tuple[int, ...]
        :param axis: The out-of-range axis under test.
        :type axis: int
        :param band: Which pre-fix failure mode this case used to hit.
        :type band: str
        """
        ndim = len(shape)
        assert not -ndim <= axis < ndim, (
            f"premise: axis={axis} must be OUT of range for rank {ndim}"
        )

        z = np.random.default_rng(0).standard_normal(shape).astype("float32")
        layer = Sparsemax(axis=axis)

        with pytest.raises(ValueError) as excinfo:
            layer(z)

        message = str(excinfo.value)
        assert f"axis={axis}" in message, (
            f"[{band}] the message does not name the offending axis: {message}"
        )
        assert f"rank {ndim}" in message, (
            f"[{band}] the message does not name the input rank: {message}"
        )
        assert f"[{-ndim}, {ndim - 1}]" in message, (
            f"[{band}] the message does not name the legal range "
            f"[{-ndim}, {ndim - 1}]: {message}"
        )

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_axis_is_rejected(self, value: bool) -> None:
        """`bool` is a subclass of `int`, so it slipped through `isinstance`.

        `Sparsemax(axis=True)` used to construct happily and behave as
        ``axis=1``; `Sparsemax(axis=False)` as ``axis=0``. Neither is anything
        a caller could have meant, and both are reachable through
        `ProbabilityOutput(**type_config)` from a config dict — where `True`
        and `1` are the same JSON-ish value away from each other.

        :param value: The bool under test.
        :type value: bool
        """
        with pytest.raises(ValueError, match="axis must be an integer"):
            Sparsemax(axis=value)

    def test_every_in_range_axis_is_accepted_at_ranks_1_to_4(self) -> None:
        """OVER-REJECTION control for the range check: nothing in range raises.

        The whole risk of F-06's fix is that it rejects too much — the check
        sits on the forward path of 18 of the 31 attention registry keys via
        `layers/activations/probability_output.py:184`. This walks every axis
        of both signs at ranks 1-4 (20 ``(rank, axis)`` points) and asserts the
        layer still runs, keeps its shape, and returns a distribution along the
        requested axis.

        The ORACLE comparison for these axes lives in
        `TestSparsemaxAgainstExactOracle`; this method is deliberately a
        cheap, exhaustive no-raise + shape + normalisation sweep, so a check
        that over-rejects at one odd rank cannot hide behind that class's
        condensed 9-case grid.
        """
        shapes = {1: (7,), 2: (3, 7), 3: (2, 3, 7), 4: (2, 3, 4, 7)}
        rng = np.random.default_rng(9)

        points = 0
        for ndim, shape in shapes.items():
            z = rng.standard_normal(shape).astype("float32")
            for axis in range(-ndim, ndim):
                out = ops.convert_to_numpy(Sparsemax(axis=axis)(z))
                points += 1

                assert out.shape == shape, (
                    f"rank {ndim}, axis {axis}: output shape {out.shape} != "
                    f"input shape {shape} — the permutation did not "
                    "round-trip (this is exactly the `norm == -1` signature)"
                )
                assert np.isfinite(out).all(), (
                    f"rank {ndim}, axis {axis}: non-finite output"
                )
                sums = out.sum(axis=axis)
                np.testing.assert_allclose(
                    sums,
                    np.ones_like(sums),
                    atol=1e-5,
                    err_msg=(
                        f"rank {ndim}, axis {axis}: rows do not sum to 1 "
                        f"along the requested axis (worst deviation "
                        f"{float(np.max(np.abs(sums - 1.0))):.3e})"
                    ),
                )

        # ANTI-VACUITY: 2 + 4 + 6 + 8 axes over ranks 1..4.
        assert points == 20, f"swept {points} (rank, axis) points, expected 20"

    @pytest.mark.parametrize("axis", [-1, 0, 1, -2, 3])
    def test_config_round_trip_preserves_axis(self, axis: int) -> None:
        """`get_config`/`from_config` still carry `axis` verbatim.

        The `__init__` validation must not normalise, clamp or otherwise
        rewrite `axis` — a layer that silently turned an out-of-range value
        into an in-range one would defeat the call-time check, and one that
        rewrote ``-1`` into ``ndim - 1`` would not survive a round trip at a
        different rank. ``axis=3`` here is deliberately out of range for most
        ranks: construction and serialization must still accept it, because
        the RANGE is a property of the call, not of the config.

        :param axis: The axis to round-trip.
        :type axis: int
        """
        config = Sparsemax(axis=axis).get_config()
        assert config["axis"] == axis and type(config["axis"]) is int
        assert Sparsemax.from_config(config).axis == axis

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_capsule_routing_axis_minus_2_site_still_forwards(
        self, policy: str
    ) -> None:
        """I-8: the repo's ONLY live non-default-axis sparsemax site.

        `capsule_routing_attention.py:311-319`'s ``_site_config(-2)``
        OVERRIDES any caller-supplied axis and builds ``Sparsemax(axis=-2)``
        on RANK-4 routing logits (measured shapes ``(2, 4, n, m)`` inside the
        routing loop). ``-2`` is in range there, so the F-06 check must be
        invisible to it — this is A-3 asserted by EXECUTION rather than by
        reading the code, and it is the assumption the plan declares falsified
        the moment this raises.

        Nothing here is sparsemax-specific beyond the `probability_type`: the
        point is the end-to-end path, config -> `ProbabilityOutput` ->
        `Sparsemax(axis=-2)` -> `call()`'s validation branch.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            layer = CapsuleRoutingSelfAttention(
                num_heads=4, key_dim=8, probability_type="sparsemax"
            )
            x = np.random.default_rng(4).standard_normal((2, 5, 32)).astype(
                "float32"
            )
            out = ops.convert_to_numpy(layer(x))

            # PREMISE (anti-vacuity): a site really is configured at axis=-2.
            # Without this the test would still pass on a build that had
            # quietly dropped the override, i.e. it would guard nothing.
            axes = sorted(
                sub.axis
                for sub in layer._flatten_layers(include_self=False)
                if isinstance(sub, Sparsemax)
            )
            assert -2 in axes, (
                f"no Sparsemax(axis=-2) sub-layer was built (axes={axes}); "
                "the capsule routing axis override is gone and this guard is "
                "no longer watching the site it claims to"
            )

            assert out.shape == (2, 5, 32), f"unexpected output shape {out.shape}"
            assert np.isfinite(out).all(), (
                f"CapsuleRoutingSelfAttention(probability_type='sparsemax') "
                f"returned non-finite output under {policy}: "
                f"nan={int(np.isnan(out).sum())}, inf={int(np.isinf(out).sum())}"
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

        ``_reduction_dtype`` (this module) mirrors the ``reduction_dtype =
        ...`` assignment in ``Sparsemax.call``. Every ``u_r`` term in
        ``_oracle_atol`` and
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

    # DECISION plan-2026-07-29T110112-09832856/D-003
    # THE TOLERANCE BELOW IS A HARD-CODED LITERAL, AND THAT IS THE POINT.
    # Do NOT "make it consistent with the rest of the file" by routing it
    # through `_oracle_atol`. That was measured: `_oracle_atol` returns
    # `max(_TF32_ATOL_FLOOR = 1e-3, derived)`, and float64's derived term is
    # ~7.8e-16, so EVERY float64 assertion in this module runs at `atol = 1e-3`.
    # The regression this guard exists to catch moves the error only to
    # 1.99e-08 — five orders UNDER that floor — so a `_oracle_atol`-toleranced
    # version of this test is green on the broken layer and measures nothing.
    # That is not hypothetical: it is exactly why 63 committed parametrizations
    # stayed green while D-007 was reverted (`plans/LESSONS.md:11`).
    # Do not widen `1e-12` either; it is the only value in the record sitting
    # strictly between the two MEASURED errors and above float64 accumulation
    # noise. If a future device moves either number, RE-DERIVE the literal from
    # a fresh A/B on that device — never fall back to a derived helper.
    def test_float64_reduction_is_never_narrowed_to_float32(self) -> None:
        """float64 must keep float64 through the sort/cumsum reduction (D-007).

        The layer's reduction dtype WIDENS the two narrow dtypes and leaves
        everything else alone; the one spelling that must never return is an
        unconditional ``"float32"``, which silently degrades the most precise
        policy. See the ``D-007`` anchor in
        ``src/dl_techniques/layers/activations/sparsemax.py``.

        MEASURED SEPARATION (GPU, `CUDA_VISIBLE_DEVICES=1`, RTX 4070; the CPU
        figures are identical to four digits) on exactly this input, which is
        the whole attack corpus's worst row for this defect::

            shipped (never-narrow)        max abs err = 3.469447e-17
            reverted (unconditional f32)  max abs err = 1.986821e-08
            degradation factor                          5.727e+08

        ``3.469447e-17 < 1e-12 < 1.986821e-08``, with margins of 2.882e+04x
        below and 1.987e+04x above — roughly symmetric in log space, which is
        what makes ``1e-12`` the right literal rather than a lucky one.

        THIS TEST WAS PROVEN RED BY EXECUTION, not by derivation: the source's
        reduction rule was reverted in place to unconditional ``"float32"``,
        this assertion fired at ``1.986821e-08 > 1e-12`` (1.99e+04x over), and
        the whole ``test_activations`` directory was then run in that broken
        state (``3 failed, 591 passed, 4 skipped, 4 xfailed``). EXACTLY THREE
        tests went red, and all three were added by
        ``plan-2026-07-29T110112-09832856`` — every case that predates this
        plan, including the four-policy property grid and the ``-inf`` mask
        coverage, stayed GREEN. That is the blind spot this guard closes, now
        executed rather than derived::

            test_float64_reduction_is_never_narrowed_to_float32  (this test)
            test_reduction_dtype_mirror_agrees_with_the_source_rule
            test_gradient_matches_analytic_vjp[float64]

        The second is EXPECTED and correct: it compares the mirror to the
        source rule as FUNCTIONS, so re-narrowing the source necessarily breaks
        the lockstep. The third was NOT predicted by the plan — the backward
        pass inherits the narrowed reduction too, and G1's float64
        parametrization catches it at ``6.386212e-09`` against a
        ``2.664535e-15`` budget (2.40e+06x) on ``pin_defect_c_bf16_K256``.

        None of the three subsumes the others. The mirror test catches a
        re-narrowing even if nobody re-runs any arithmetic, but is blind to a
        source break that someone "helpfully" mirrors in lockstep. G1 catches
        the backward-side consequence, but only via a gradient and only on the
        corpus rows it keeps. This test is the direct FORWARD numeric claim on
        the corpus's worst row for this defect.
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float64")
        try:
            # Premises. The fixture is the no-`inf` control row set (masked
            # fraction 0.0): 4 x 17 = 68 finite entries, deterministic seed.
            z = _partially_masked_row_batch(17, 0.0, 1)
            assert z.shape == (4, 17), f"fixture shape moved: {z.shape}"
            assert np.isfinite(z).all(), (
                "premise: this fixture carries no masked entry at all — the "
                "defect is in the reduction's PRECISION, not in its masking"
            )
            assert _output_dtype() == "float64", (
                f"premise: the policy must be float64, got {_output_dtype()!r}"
            )

            z_compute = _to_compute_dtype(z)
            layer = Sparsemax()
            raw = layer(z_compute)

            # I-1, on the tensor: a float64 policy must not observably emit
            # float32. Checked before `convert_to_numpy` widens anything back.
            assert keras.backend.standardize_dtype(raw.dtype) == "float64", (
                f"layer returned {raw.dtype} under a float64 policy (I-1)"
            )

            out64 = ops.convert_to_numpy(raw).astype(np.float64)
            exact64 = _exact_rows_as_float64(z_compute)
            err = float(np.max(np.abs(out64 - exact64)))

            # HARD-CODED, DELIBERATELY. See the block comment above this method
            # for why no derived tolerance may be substituted here.
            atol = 1e-12
            assert err <= atol, (
                "D-007 RE-NARROWED: the float64 reduction is no longer running "
                f"in float64. max abs error vs the exact rational oracle is "
                f"{err:.6e} against a hard-coded atol of {atol:.1e} "
                f"({err / atol:.3e}x over). The shipped layer measures "
                "3.469447e-17 here; an unconditional `\"float32\"` reduction "
                "dtype measures 1.986821e-08. Check the `reduction_dtype` "
                "expression in `sparsemax.py` before looking anywhere else."
            )

            # ANTI-VACUITY: the comparison must actually have compared
            # something. A zero-size or all-zero pair would satisfy the bound
            # above for free.
            assert out64.shape == (4, 17) and exact64.shape == (4, 17), (
                f"compared shapes {out64.shape} vs {exact64.shape}"
            )
            assert int((out64 > 0.0).sum()) >= 4, (
                "every row must have at least one entry in the support; got "
                f"{int((out64 > 0.0).sum())} positive entries in total"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

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

#: `_XLA_POLICIES` (the four-policy tuple) is defined ABOVE `TestSparsemax`,
#: not here: `TestSparsemax`'s G9 capsule-integration case parametrizes over it
#: too, and a decorator is evaluated at class-definition time. One home for the
#: tuple, defined before its first use.

#: `K` values. 257 and 2048 are load-bearing (see the note above); 8 is the
#: small control and 4096 the largest measured width.
_XLA_WIDTHS = (8, 256, 257, 512, 2048, 4096)


class TestSparsemaxXLACapability:
    """`Sparsemax` compiles and runs under `tf.function(jit_compile=True)`.

    24 grid points per method: ``K`` in ``{8, 256, 257, 512, 2048, 4096}`` times
    ``{float32, mixed_float16, mixed_bfloat16, float64}`` — once for the
    forward output and once for the backward pass, which XLA lowers separately.
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

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    @pytest.mark.parametrize("width", _XLA_WIDTHS)
    def test_jit_compiled_backward_matches_eager(
        self, width: int, policy: str
    ) -> None:
        """The XLA-compiled BACKWARD pass is bit-identical to the eager one.

        The forward sibling above only proves the compiled FORWARD agrees.
        `Sparsemax` has no `custom_gradient`, so its backward pass is whatever
        XLA lowers `max` / `sort` / `cumsum` / `where` / `maximum` to — a
        different set of kernels from the eager ones, with a different reduction
        order available to it. That is exactly the thing measured to move fp16
        and bf16 magnitudes elsewhere in this file, so it is asserted rather
        than assumed.

        NOT BIT-IDENTICAL — and this is the measurement, not a concession.
        The plan for this guard specified `assert_array_equal` at zero
        tolerance, on the strength of a `max|xla - eager| = 0.000e+00` reading.
        **That reading does not reproduce on GPU** (RTX 4070, TF 2.18): the
        delta is non-zero at 22 of these 24 points, worst 9.537e-07 (float32,
        K=4096) / 2.441e-03 (fp16, K=256) / 1.563e-02 (bf16, K=2048) /
        5.551e-16 (float64, K=2048). The forward sibling above is unaffected,
        so it is the BACKWARD lowering that differs.

        It is not an artefact of this harness. Moving the tape OUTSIDE the
        compiled function (compiled forward, eager backward) reproduces the
        same delta to the digit at 21 of 24 points, so the difference is XLA's
        gradient kernels, not where the tape is written.

        The tolerance is therefore `_grad_atol(v_compute, _output_dtype())` —
        the gradient-scale budget `4*u_c(S) + 2*u_r(S)` at ``S = max(|v|, 1)``,
        NOT `_oracle_atol`, whose scale argument is the forward row spread and
        which would be charging the wrong magnitude here. Worst measured ratio
        over the whole grid is **0.333** of that budget (float32, K=4096), so
        the guard keeps ~3x headroom and still bites: an injection scaling the
        compiled path's upstream by 1.5 fails it by 5-6 orders of magnitude.
        Do NOT widen it further, and do not re-tighten to `assert_array_equal`
        without re-measuring on the target device.

        The eager side reuses
        `TestSparsemaxBackwardPass._layer_gradient`, the file's single eager
        backward harness (its 4th call site). The compiled side cannot call it:
        the tape must be TRACED INSIDE the `jit_compile=True` function for the
        backward to be compiled at all, and `_layer_gradient` crosses the numpy
        boundary. The compiled body below is therefore a deliberate mirror of
        it — if one changes, both change.

        :param width: Row width ``K``.
        :type width: int
        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            z = _partially_masked_row_batch(width, 0.25, seed=width)
            # Post-cast bits on BOTH sides, so this compares ARITHMETIC and not
            # a rounding at the layer boundary (same rule as the forward
            # sibling and as `_layer_gradient`'s own precondition).
            z_compute = _to_compute_dtype(z)
            rng = np.random.default_rng(4041 + width)
            v_compute = _to_compute_dtype(
                rng.standard_normal(z.shape).astype(np.float32)
            )
            masked = ~np.isfinite(np.asarray(z_compute, dtype=np.float64))

            eager_grad, _ = TestSparsemaxBackwardPass._layer_gradient(
                z_compute, v_compute
            )

            layer = Sparsemax()
            v_tensor = tf.convert_to_tensor(v_compute)

            @tf.function(jit_compile=True)
            def _compiled_grad(t):
                # Mirror of `_layer_gradient`'s taped region. The tape lives
                # INSIDE the compiled function on purpose: that is what makes
                # XLA lower the backward too, which is the property under test.
                with tf.GradientTape() as tape:
                    tape.watch(t)
                    p = layer(t)
                    loss = tf.reduce_sum(p * tf.cast(v_tensor, p.dtype))
                return tape.gradient(loss, t)

            # `jit_compile=True` RAISES if XLA cannot lower forward OR backward,
            # so this call is itself the compilation assertion. Do not wrap it.
            xla_raw = _compiled_grad(tf.convert_to_tensor(z_compute))
            assert xla_raw is not None, (
                "the compiled tape returned no gradient — every assertion "
                "below would be vacuous"
            )
            xla_grad = np.asarray(xla_raw, dtype=np.float64)

            # ANTI-VACUITY. An all-zero gradient would satisfy bit-identity
            # trivially, and a NaN-vs-NaN pair compares EQUAL under
            # `assert_array_equal` — so both are excluded before the comparison
            # is trusted.
            assert np.isfinite(xla_grad).all(), (
                f"XLA gradient is not finite at K={width} under {policy} "
                f"(nan={int(np.isnan(xla_grad).sum())}, "
                f"inf={int(np.isinf(xla_grad).sum())})"
            )
            nonzero = int(np.count_nonzero(xla_grad))
            assert nonzero > 0, (
                f"XLA gradient is identically zero at K={width} under "
                f"{policy} — bit-identity would be vacuous"
            )
            assert np.all(xla_grad[masked] == 0.0), (
                f"XLA gradient is non-zero at "
                f"{int((xla_grad[masked] != 0.0).sum())} masked positions at "
                f"K={width} under {policy}"
            )

            atol = _grad_atol(v_compute, _output_dtype())
            np.testing.assert_allclose(
                xla_grad,
                eager_grad,
                atol=atol,
                rtol=0.0,
                err_msg=(
                    f"XLA backward disagrees with eager backward at K={width} "
                    f"under {policy}: worst |delta| = "
                    f"{float(np.max(np.abs(xla_grad - eager_grad))):.6e} > "
                    f"derived atol {atol:.6e} "
                    f"({float(np.max(np.abs(xla_grad - eager_grad))) / atol:.3f}"
                    f"x the budget; worst measured on this grid is 0.333x; "
                    f"{nonzero} non-zero gradient entries)"
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
# exactly, so `p_i == 0` yet `shifted - tau == 0`), the
# `ops.maximum(shifted - tau, 0.0)` in `Sparsemax.call` routes a full gradient
# to an entry that `k_z` excludes.
# The result is not even a Clarke subgradient (228/300 random directions give a
# mixing coefficient outside [0, 1]), with errors up to 3.24. That is an OPEN
# defect, deliberately not fixed here, and it is pinned separately rather than
# absorbed into a widened tolerance. Excluding those rows is what keeps this
# guard a measurement of the other 46; it is NOT a claim that they are correct.
# ---------------------------------------------------------------------


class TestSparsemaxBackwardPass:
    """`Sparsemax`'s gradient vs the analytic VJP, on boundary-free rows.

    Measured on GPU at every point below (`CUDA_VISIBLE_DEVICES=1`), zero
    failures. The boundary-free row COUNTS are stated where they are
    load-bearing — beside the anti-vacuity floor they justify, inside
    `test_gradient_matches_analytic_vjp`.

    THE WORST-ERROR AND SLACK FIGURES ARE DELIBERATELY NOT RESTATED HERE.
    Their one home is the `DERIVED vs MEASURED` table above `_grad_atol`,
    which quotes each policy at its own worst point's `S`. This docstring used
    to carry its own copy ("worst error 1.69 ulp ... against a 4-ulp budget"),
    which was the float32 number presented as the global one and was wrong
    twice over — see `D-017` of `plan-2026-07-29T110112-09832856`.
    """

    @staticmethod
    def _layer_gradient(
        z_compute: np.ndarray, v_compute: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backprop ``sum(sparsemax(z) * v)`` through a default `Sparsemax`.

        Internal to this class. Both arguments must ALREADY be in the active
        policy's compute dtype (:func:`_to_compute_dtype`), so that the tape
        sees the bits the layer receives and the ``tf.cast(v, p.dtype)`` inside
        the taped region is a NO-OP — a cast that actually converted there
        would add a rounding the derived budget does not carry.

        That no-op is guaranteed by the ``z.dtype == v.dtype`` assert below
        plus the fact that a default `Sparsemax` emits its input's dtype, and
        it is EXECUTED, not assumed: measured at all four `_XLA_POLICIES`,
        ``p.dtype == v.dtype`` and the cast is bit-identity
        (`iter-1/step-13`). This docstring used to claim outright that "no cast
        happens inside the taped region", which the line below contradicts —
        the load-bearing statement is that the cast is a no-op BECAUSE of the
        assert, not that it is absent.

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
        ``S = max(|v|, 1)``. It must NOT be tightened; the measured per-policy
        slack, each figure quoted at its own worst point's ``S``, lives in the
        ``DERIVED vs MEASURED`` table above `_grad_atol` and NOWHERE ELSE. Go
        read it there rather than trusting a number restated here — a copy in
        this docstring is exactly what drifted (`D-017`).

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

        The committed guard for the ``row_max = ops.max(inputs_2d, ...)`` /
        ``shifted = inputs_2d - row_max`` route in ``Sparsemax.call``
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

    # DECISION plan-2026-07-29T110112-09832856/D-010
    # POLARITY.  This pin asserts the CORRECT (k-branch) gradient and marks it
    # `xfail(strict=True)`, so it reports `xfailed` while F-02 is open and
    # `xpassed` -> FAILED the moment F-02 is fixed. Do NOT "simplify" it into
    # the obvious-looking alternative — asserting the MEASURED values
    # `[-1, 1]` / `[-5, 5]` and marking THAT `xfail` — which is the same
    # mechanism run backwards: it would go red on the FIX and stay green
    # forever while the defect is open, i.e. it would guard nothing.
    # PORTABILITY.  Do not reintroduce the TF literals in any form.
    # `ops.maximum`'s tie routing (`x >= y` routes to `x`) is a TF
    # gradient-registration detail; JAX's `jnp.maximum` splits 50/50 at exact
    # equality and would give a DIFFERENT (still non-Clarke) boundary
    # gradient. An exact-value assertion is a backend pin wearing a
    # correctness pin's name. See `D-001`/`D-010` of
    # `plan-2026-07-29T110112-09832856`.
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OPEN DEFECT F-02: ops.maximum(shifted - tau, 0.0) in "
            "Sparsemax.call hands a full gradient to a support-BOUNDARY "
            "entry (z_i == tau exactly, so p_i == 0 but shifted - tau == 0), "
            "while k_z correctly excludes it. User ruling for this plan: "
            "GUARD, do not fix."
        ),
    )
    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_open_defect_f02_boundary_gradient_should_equal_the_k_branch(
        self, policy: str
    ) -> None:
        """OPEN DEFECT F-02 — the support-boundary gradient is wrong.

        `z = [[1.0, 0.0]]` is the minimal reproducer: the top-2 gap is exactly
        1.0, the knife edge of sparsemax, so `k_z == 1`, `p == [1, 0]` and the
        second entry sits EXACTLY on `tau`. The analytic VJP over the support
        `S = {0}` is therefore ``v_0 - v_0/1 == 0`` — the true gradient is the
        zero vector for EVERY upstream `v`. The layer instead routes a full
        gradient through `ops.maximum`'s tie, and the result is not even a
        Clarke subgradient: solving ``layer = l*J(k=2) + (1-l)*J(k=1)`` over
        300 random directions put `l` outside `[0, 1]` in 228 of them.

        Measured identically under all four policies, so this is an
        op-semantics defect, not a precision one. Incidence on ORDINARY
        `N(0, 2)` K=64 logits is 0/1000 (float32, float64) but **6/1000**
        (mixed_float16) and **26/1000** (mixed_bfloat16) — narrow dtypes
        manufacture the tie by quantization, so this is not measure-zero.

        WHAT IS ASSERTED, AND WHY IT LOOKS INVERTED. This test asserts the
        CORRECT behaviour (gradient == the k-branch, within `_grad_atol`) and
        is marked `xfail(strict=True)`. It therefore reports `xfailed` today.
        It deliberately does NOT assert the measured wrong values: see the
        POLARITY and PORTABILITY notes above the marker.

        WHAT AN XPASS OBLIGES. If this test XPASSes, `strict=True` turns that
        into a hard FAILURE, and that failure is the intended signal: someone
        has added a `custom_gradient` or a `stop_gradient`ed support mask to
        `sparsemax.py` (both are STOP tripwires of the plan that wrote this
        pin, so it should have been a deliberate act). The obligation is then,
        in the same commit:

        1. remove the `xfail` marker — a marker over a closed defect is a
           false claim;
        2. RENAME this test, dropping `open_defect_f02` — a name asserting a
           closed defect is itself a false claim
           (`plans/LESSONS.md:44`); grep the whole file for the old name;
        3. update the sparsemax entry in `plans/SYSTEM.md` (currently
           `:101-107`), which records F-02 as open;
        4. re-enable the boundary rows that
           `test_gradient_matches_analytic_vjp` currently EXCLUDES — that
           guard's boundary filter exists only because of this defect.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            out_dtype = _output_dtype()
            z = np.array([[1.0, 0.0]], dtype=np.float32)
            z_compute = _to_compute_dtype(z)

            # --- premise: this row really is a support-boundary row ----------
            # If any of these fire, the reproducer has stopped reproducing and
            # the assertion below would be xfailing for the WRONG reason. They
            # are stated separately so `-rx` names which one it was.
            row = _exact_sparsemax(z_compute)[0]
            support = np.array([float(o) > 0.0 for o in row["out"]], dtype=bool)
            k = int(support.sum())
            assert k == 1, (
                f"premise ({policy}): expected k_z == 1 on z=[[1.0, 0.0]], got "
                f"{k} — the reproducer is no longer the knife-edge row"
            )
            assert any(
                np.isfinite(float(val))
                and Fraction(*float(val).as_integer_ratio()) == row["tau"]
                for val in z_compute[0]
            ), (
                f"premise ({policy}): no entry of z=[[1.0, 0.0]] equals tau "
                f"exactly in rational arithmetic — this is not a boundary row "
                "and F-02's mechanism is not being exercised"
            )

            # --- the pin -----------------------------------------------------
            # Several upstreams, because the defect is DIRECTION-dependent:
            # under TF, `v = [1, 0]` happens to give the right answer while
            # `[0, 1]` and `[3, 5]` do not. A single-direction pin could go
            # green on a backend whose tie routing differs without the defect
            # being fixed at all. The strongest direction is checked first so
            # the reported xfail reason is the substantive one.
            for v in ([0.0, 1.0], [3.0, 5.0], [1.0, -1.0], [1.0, 0.0]):
                v_compute = _to_compute_dtype(
                    np.array([v], dtype=np.float32)
                )
                grad, _ = self._layer_gradient(z_compute, v_compute)

                v64 = np.asarray(v_compute, dtype=np.float64)
                reference = np.zeros_like(v64)
                reference[0][support] = (
                    v64[0][support] - v64[0][support].sum() / k
                )

                atol = _grad_atol(v_compute, out_dtype)
                err = float(np.max(np.abs(grad[0] - reference[0])))
                assert err <= atol, (
                    f"OPEN DEFECT F-02 ({policy}): the boundary row's gradient "
                    f"disagrees with the k={k} branch for upstream v={v}: max "
                    f"abs error {err:.6e} > derived atol {atol:.6e} "
                    f"({err / atol:.2f}x). This assertion failing is the "
                    "EXPECTED state while F-02 is open; it passing everywhere "
                    "means the defect is fixed and this test's xfail marker "
                    "and name must both go — see the docstring."
                )
        finally:
            keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# THE LAYER vs THE EXACT-RATIONAL ORACLE.
#
# WHY THIS CLASS EXISTS.  Until it was added, `_attack_corpus()` had exactly
# two consumers, both inside `TestExactRationalOracle`, whose own docstring
# says "None of these tests touch ``Sparsemax``". So the corpus and the oracle
# only ever checked EACH OTHER: no committed test compared the LAYER to the
# exact-rational oracle at all. Everything layer-side compared against the
# float64 `_sparsemax_reference` (which shares the layer's `vals - tau`
# cancellation structure), or against the layer's own eager answer (the XLA
# class), or against the analytic VJP (the backward class). An instrument that
# is validated but never used is not evidence.
#
# WHAT IS ASSERTED, per `(corpus row, policy)` pair:
#   * the output dtype IS the compute dtype (I-1: `Sparsemax` never returns
#     float32 under a narrow policy — asserted on the TENSOR, before any
#     numpy conversion can launder it);
#   * exact `0.0` at every masked position, with `== 0.0`, not `allclose`;
#   * agreement with `_exact_rows_as_float64` at `_oracle_atol(z_compute,
#     _output_dtype())`, `rtol=0.0`, fed the POST-CAST bits on BOTH sides.
#
# WHY ONE TEST CASE PER `(row, policy)` PAIR rather than a loop inside four
# policy cases: the three oracle-out-of-scope pairs must be REPORTED as pytest
# skips (`-rs` names which and why), not silently `continue`d past. A `continue`
# inside a loop is invisible to the runner, and "which rows did this guard
# actually look at" is precisely the question a vacuous guard answers wrongly.
# The corpus length is READ (`len(_attack_corpus())`), never hard-coded — the
# record has carried it as both 19 and 21 (`plans/LESSONS.md:20`).
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# THE `axis != -1` / rank>2 GRID  (G8).
#
# WHY THESE NINE CASES AND NOT A CROSS-PRODUCT.  The exploration ran 302
# oracle comparisons over ranks 1-4 x axes {0, 1, -2, -1} x the six `_XLA_WIDTHS`
# x 4 policies, masked and unmasked, and every one passed. A committed replay of
# that cross-product would be ~1500 cases of exact RATIONAL arithmetic at K up to
# 4096 — minutes of gate time to re-derive a grid whose result is already known.
# What a committed guard has to preserve is COVERAGE OF THE DEGREES OF FREEDOM,
# not the cardinality: each rank, each axis and each width appears at least once,
# and the widths stay exactly `_XLA_WIDTHS` because the historical defect onsets
# were NON-MONOTONE in K (I-7 — 256 and 257 are load-bearing, do not thin them).
# `test_the_axis_grid_covers_every_declared_rank_axis_and_width` asserts that
# coverage property directly, so thinning this tuple goes RED rather than
# quietly shrinking the guard.
#
# THE NON-REDUCED DIMENSIONS ARE DELIBERATELY TINY (1..3). They cost oracle time
# linearly and buy nothing: the layer flattens every one of them into the batch
# dimension of `Sparsemax.call`'s `ops.reshape(inputs_permuted, (-1, k))`, so a
# size-8 leading dim
# exercises the identical code path as a size-2 one.
#
# `rank4_K4096_axism2` IS THE LIVE PRODUCTION SHAPE, not a synthetic corner.
# `capsule_routing_attention.py:331` builds `Sparsemax(axis=-2)` whenever
# `probability_type='sparsemax'`, and a `call()` spy measured it receiving rank-4
# tensors of shape `(2, 4, K, 1)` from the routing loop (`:797-803`). It is the
# repo's ONLY live non-default-axis sparsemax site and it is reachable from 18 of
# the 31 attention registry keys (I-8).
# ---------------------------------------------------------------------

#: `(case_id, input shape, axis)`. ``K = shape[axis]``.
_AXIS_SHAPE_CASES: Tuple[Tuple[str, Tuple[int, ...], int], ...] = (
    ("rank1_K8_axis0", (8,), 0),
    ("rank1_K257_axism1", (257,), -1),
    ("rank2_K256_axis0", (256, 3), 0),
    ("rank2_K512_axism1", (2, 512), -1),
    ("rank3_K8_axis0", (8, 2, 3), 0),
    ("rank3_K257_axis1", (2, 257, 3), 1),
    ("rank3_K2048_axism2", (2, 2048, 2), -2),
    # F-07's LIVE shape: capsule routing, `axis=-2` on rank-4, trailing 1.
    ("rank4_K4096_axism2", (2, 1, 4096, 1), -2),
    ("rank4_K256_axis1", (2, 256, 2, 2), 1),
)


class TestSparsemaxAgainstExactOracle:
    """`Sparsemax`'s FORWARD output vs `fractions.Fraction` arithmetic.

    Measured on GPU (`CUDA_VISIBLE_DEVICES=1`, RTX 4070) before being
    committed: 0 violations at 147 points per policy, worst ratio
    ``err / derived`` = 0.0556 / 0.1666 / 0.1667 / 0.0714 for float32 /
    mixed_float16 / mixed_bfloat16 / float64 — identical in BOTH TF32 regimes
    (standalone and co-collected with `test_linear_attention.py`, which
    disables TF32 process-globally at import). A green result here is
    therefore expected; the value of this class is that the comparison is
    COMMITTED, not that it is surprising.

    This class is the home for every layer-vs-exact-oracle comparison. Add to
    it rather than starting a sibling class.
    """

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    @pytest.mark.parametrize(
        "case_index",
        # READ the corpus length; do not assume it. Ids are the corpus labels
        # so a skip or a failure names the row it happened on.
        range(len(_attack_corpus())),
        ids=[label for label, _, _ in _attack_corpus()],
    )
    def test_layer_matches_the_exact_oracle_on_the_attack_corpus(
        self, case_index: int, policy: str
    ) -> None:
        """One corpus row under one policy, layer vs exact rational arithmetic.

        :param case_index: Index into `_attack_corpus()`.
        :type case_index: int
        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            label, z, _measured_under = _attack_corpus()[case_index]
            z_compute = _to_compute_dtype(z)
            z64 = np.asarray(z_compute, dtype=np.float64)

            # The `any(isfinite)` predicate the exact oracle already applies.
            # Under `mixed_float16` the float32 literal 1.68e7 casts to `inf`,
            # leaving three corpus batches with no finite entry at all — the
            # projection is undefined there, for the layer and the oracle
            # alike. Those pairs are SKIPPED, never deleted from the corpus:
            # they remain in scope under the policies that can represent them.
            if not np.isfinite(z64).any(axis=-1).all():
                # ANTI-VACUITY: a skip is only ever legitimate for these three
                # pairs. If the predicate starts firing anywhere else it is a
                # bug in the predicate (or a corpus change), and this guard
                # must go RED rather than quietly shrink its own grid.
                assert policy == "mixed_float16" and label in (
                    "pin_defect_d_f32_1.68e7_K4",
                    "onset_d_float32_all_finite",
                    "onset_d_float32_masked_2_of_4",
                ), (
                    f"{label} has no finite entry under {policy}, which is not "
                    "one of the three known oracle-out-of-scope (row, policy) "
                    "pairs — the skip predicate has started eating rows it "
                    "should be checking"
                )
                pytest.skip(
                    f"{label} under {policy}: every entry casts to a "
                    "non-finite value (float32 1.68e7 -> fp16 inf), so the "
                    "projection is undefined and the exact-rational oracle "
                    "refuses the row by the same predicate. Out of scope for "
                    "THIS policy only."
                )

            layer = Sparsemax()
            raw = layer(z_compute)

            # I-1, on the tensor: the observable output dtype is the compute
            # dtype. Checked before `convert_to_numpy`, which would happily
            # widen a narrow answer and hide a regression.
            assert keras.backend.standardize_dtype(raw.dtype) == (
                _output_dtype()
            ), (
                f"{label}: layer returned {raw.dtype} under {policy}, but the "
                f"compute dtype is {_output_dtype()} (I-1)"
            )

            out64 = ops.convert_to_numpy(raw).astype(np.float64)
            exact64 = _exact_rows_as_float64(z_compute)

            masked = ~np.isfinite(z64)
            assert np.all(out64[masked] == 0.0), (
                f"{label} under {policy}: "
                f"{int((out64[masked] != 0.0).sum())} of "
                f"{int(masked.sum())} masked positions are not EXACTLY 0.0 "
                f"(worst {float(np.max(np.abs(out64[masked]))) if masked.any() else 0.0:.3e})"
            )

            atol = _oracle_atol(z_compute, _output_dtype())
            err = float(np.max(np.abs(out64 - exact64)))
            np.testing.assert_allclose(
                out64,
                exact64,
                atol=atol,
                rtol=0.0,
                err_msg=(
                    f"layer != EXACT rational oracle at {label} under "
                    f"{policy} (K={z.shape[-1]}): max abs error {err:.6e} vs "
                    f"derived atol {atol:.6e} ({err / atol:.3f}x)"
                ),
            )

            # Per-case floor: EVERY row of this batch was compared. Stated as
            # an equality rather than a `>= n` floor because the count is known
            # exactly here — one comparison per row, no filtering of any kind
            # (unlike the backward guard, which legitimately drops boundary
            # rows and therefore can only assert a floor).
            assert out64.shape[0] == z_compute.shape[0] >= 1, (
                f"{label}: compared {out64.shape[0]} rows of "
                f"{z_compute.shape[0]}"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_the_corpus_grid_is_not_silently_skipped(self) -> None:
        """Grid-level anti-vacuity: the skip predicate eats 3 rows, not more.

        The per-case guard above cannot see a predicate bug that skips
        EVERYTHING — pytest would report 76 skips and 0 failures, which is a
        green run. This method counts, over the whole
        ``_attack_corpus() x _XLA_POLICIES`` grid, how many ROWS the predicate
        would let through, and pins the exact set it turns away.

        THE FLOOR. Measured: 29 rows per policy x 4 policies = 116, minus the
        3 single-row batches the fp16 predicate refuses = **113**. The floor is
        **100**, chosen so that losing any ENTIRE policy (-29 or -27 -> 84..87)
        or the whole seeded property subset (-48 -> 65) goes red, while
        ordinary corpus growth does not. It is the analogue of the backward
        guard's ``checked >= 40`` against a measured 43..46.
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            rows_compared = 0
            skipped: List[Tuple[str, str]] = []
            for policy in _XLA_POLICIES:
                keras.mixed_precision.set_global_policy(policy)
                for label, z, _measured_under in _attack_corpus():
                    z64 = np.asarray(_to_compute_dtype(z), dtype=np.float64)
                    if not np.isfinite(z64).any(axis=-1).all():
                        skipped.append((label, policy))
                    else:
                        rows_compared += int(z64.shape[0])

            assert rows_compared >= 100, (
                f"only {rows_compared} corpus rows would be compared across "
                f"the whole 4-policy grid (expected >= 100, measured 113); "
                f"{len(skipped)} (row, policy) pairs were turned away by the "
                f"predicate, e.g. {sorted(skipped)[:5]} — this guard has gone "
                "vacuous"
            )
            assert sorted(skipped) == sorted(
                [
                    ("pin_defect_d_f32_1.68e7_K4", "mixed_float16"),
                    ("onset_d_float32_all_finite", "mixed_float16"),
                    ("onset_d_float32_masked_2_of_4", "mixed_float16"),
                ]
            ), (
                "the set of oracle-out-of-scope (row, policy) pairs has "
                f"changed: {sorted(skipped)}. Exactly three pairs are known "
                "out of scope (float32 1.68e7 -> fp16 inf); any other pair "
                "being skipped means rows are being silently dropped, and any "
                "of these three no longer skipping means the corpus changed."
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    # -----------------------------------------------------------------
    # G8 — `axis != -1`, rank > 2, dynamic `K`.
    #
    # The two staticmethods below are NOT a fifth module-level abstraction;
    # they follow `TestSparsemaxBackwardPass._layer_gradient`'s precedent
    # (a helper scoped to the class that needs it). `_exact_rows_along_axis`
    # IS the step's one budgeted module-level abstraction, and with it the
    # plan's 4/4 abstraction budget is spent.
    # -----------------------------------------------------------------

    @staticmethod
    def _moved_flat(z: np.ndarray, axis: int) -> np.ndarray:
        """The ``(rows, K)`` view of ``z`` with ``axis`` moved to last.

        `_oracle_atol` derives its scale from each row's spread along the LAST
        numpy axis, so it must be fed this view and never the raw N-D batch:
        on `(2, 257, 3)` with `axis=1` the raw batch's last-axis spread is
        taken over 3 unrelated entries and the resulting tolerance is a
        different number derived from a different row.

        :param z: Batch of any rank >= 1.
        :type z: np.ndarray
        :param axis: The reduction axis, either sign, in range.
        :type axis: int
        :return: 2-D view/copy of shape ``(z.size // K, K)``.
        :rtype: np.ndarray
        """
        moved = np.moveaxis(np.asarray(z), axis, -1)
        return moved.reshape(-1, moved.shape[-1])

    @staticmethod
    def _axis_batch(
        shape: Tuple[int, ...], axis: int, masked: bool, seed: int
    ) -> np.ndarray:
        """Seeded float32 logits of ``shape``, optionally ``-inf``-masked.

        Built in the MOVED layout and moved back, so the mask is written
        through a contiguous ``(rows, K)`` view — writing through
        ``moveaxis(...).reshape(...)`` of an already-built array can silently
        hit a copy and drop every mask, which would leave the "masked" half of
        this grid a duplicate of the unmasked half.

        Each row carries a deliberate exact tie (entry ``K//2`` copied from
        entry ``0``); ties are the pattern this family's defects live on.
        A quarter of each row is masked, so no row is ever fully masked
        (``K >= 8`` throughout) — fully-masked rows are out of the oracle's
        scope by the same ``any(isfinite)`` predicate used above.

        :param shape: The layer's input shape.
        :type shape: tuple[int, ...]
        :param axis: The reduction axis, either sign, in range.
        :type axis: int
        :param masked: Whether to write ``-inf`` into a quarter of each row.
        :type masked: bool
        :param seed: RNG seed.
        :type seed: int
        :return: float32 array of shape ``shape``.
        :rtype: np.ndarray
        """
        ndim = len(shape)
        ax = axis % ndim
        k = shape[ax]
        moved_shape = shape[:ax] + shape[ax + 1:] + (k,)

        rng = np.random.default_rng(seed)
        moved = rng.uniform(-5.0, 5.0, size=moved_shape).astype(np.float32)
        flat = moved.reshape(-1, k)
        flat[:, k // 2] = flat[:, 0]

        if masked:
            n_masked = max(1, k // 4)
            assert n_masked < k, f"would mask an entire row (K={k})"
            for r in range(flat.shape[0]):
                pos = rng.choice(k, size=n_masked, replace=False)
                flat[r, pos] = -np.inf
            assert np.isfinite(flat).any(axis=-1).all(), "a row is fully masked"

        return np.ascontiguousarray(np.moveaxis(moved, -1, ax))

    # -----------------------------------------------------------------
    # PROVEN RED (GPU, `CUDA_VISIBLE_DEVICES=1`), each injection reverted in
    # place. The assertion that fired is the final `assert_allclose` below in
    # every case:
    #
    #   (A) the oracle applied along the WRONG axis (`_exact_rows_along_axis(
    #       z_compute, -1)` while the layer reduces `axis=1`), rank-3 K=257:
    #       err = 1.000000e+00 at all 4 policies, vs atol 1.000e-03 (float32,
    #       1000x) / 1.465e-03 (fp16, 683x) / 1.172e-02 (bf16, 85x) / 1.000e-03
    #       (float64, 1000x). This reproduces the exploration's own harness
    #       control to the digit.
    #   (B) a broken projection (`out * 0.5`): RED at 17/17 of the dynamic-K,
    #       jit, save/load and rank-4-K4096 cases, worst 3.9x (bf16) — i.e.
    #       none of those four is a shape-only test.
    #   (C) a CALIBRATED additive `+1.2e-3` is RED on float32 and float64 only
    #       (1.20x over the `_TF32_ATOL_FLOOR`) and GREEN on fp16 and bf16,
    #       whose derived tolerances are 1.465e-03 and 1.172e-02. That is not a
    #       weakness of this grid: it is D-012's measured fact that the narrow
    #       policies' real sensitivity is ~1 output ulp, and 1.2e-3 is BELOW one
    #       bfloat16 ulp at 1.0 (7.8e-3), so no representable output can move.
    #       Do NOT quote (C) as a uniform proof; (B) is the uniform one.
    # -----------------------------------------------------------------

    def _assert_matches_exact_oracle_along_axis(
        self, z_compute: np.ndarray, out: Any, axis: int, what: str
    ) -> int:
        """Shared body of the four G8 comparisons; returns rows compared.

        :param z_compute: The POST-CAST bits handed to the layer.
        :type z_compute: np.ndarray
        :param out: The layer's raw output tensor (not yet numpy).
        :type out: Any
        :param axis: The reduction axis.
        :type axis: int
        :param what: Identifier for the failure messages.
        :type what: str
        :return: Number of ``(row along axis)`` projections compared.
        :rtype: int
        """
        policy = keras.mixed_precision.global_policy().name

        # I-1, on the TENSOR, before `convert_to_numpy` can launder a narrow
        # answer into float64.
        assert keras.backend.standardize_dtype(out.dtype) == _output_dtype(), (
            f"{what}: layer returned {out.dtype} under {policy}, but the "
            f"compute dtype is {_output_dtype()} (I-1)"
        )

        out64 = ops.convert_to_numpy(out).astype(np.float64)
        assert out64.shape == z_compute.shape, (
            f"{what}: output shape {out64.shape} != input shape "
            f"{z_compute.shape} — the axis permutation did not round-trip"
        )

        exact64 = _exact_rows_along_axis(z_compute, axis)
        flat = self._moved_flat(z_compute, axis)

        masked = ~np.isfinite(np.asarray(z_compute, dtype=np.float64))
        assert np.all(out64[masked] == 0.0), (
            f"{what} under {policy}: "
            f"{int((out64[masked] != 0.0).sum())} of {int(masked.sum())} "
            "masked positions are not EXACTLY 0.0"
        )

        atol = _oracle_atol(flat, _output_dtype())
        err = float(np.max(np.abs(out64 - exact64)))
        np.testing.assert_allclose(
            out64,
            exact64,
            atol=atol,
            rtol=0.0,
            err_msg=(
                f"layer != EXACT rational oracle at {what} under {policy} "
                f"(axis={axis}, K={z_compute.shape[axis]}): max abs error "
                f"{err:.6e} vs derived atol {atol:.6e} ({err / atol:.3f}x)"
            ),
        )
        return int(flat.shape[0])

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    @pytest.mark.parametrize("masked", (False, True), ids=("unmasked", "masked"))
    @pytest.mark.parametrize(
        "case_index",
        range(len(_AXIS_SHAPE_CASES)),
        ids=[case_id for case_id, _, _ in _AXIS_SHAPE_CASES],
    )
    def test_moved_axis_and_rank_match_the_exact_oracle(
        self, case_index: int, masked: bool, policy: str
    ) -> None:
        """One ``(rank, axis, K)`` point vs exact rational arithmetic.

        Covers ranks 1-4 and axes ``{0, 1, -2, -1}`` at every width in
        ``_XLA_WIDTHS``. ``rank4_K4096_axism2`` is F-07's LIVE shape:
        `capsule_routing_attention.py:331` builds ``Sparsemax(axis=-2)`` on
        rank-4 routing logits whenever ``probability_type='sparsemax'``, which
        18 of the 31 attention registry keys can reach — so this is the one
        parametrization in the file that guards a shipped configuration rather
        than a synthetic one.

        Only IN-RANGE axes appear here. Out-of-range ``axis`` is an OPEN defect
        (F-06: a silently transposed non-distribution at ``axis == -(ndim+1)``,
        and an uncatchable TF C++ SIGABRT for ``ndim + axis in [-ndim, -2]``
        that no ``except`` can contain) and is guarded separately; do NOT add
        an out-of-range value to `_AXIS_SHAPE_CASES` to "round out the grid".

        :param case_index: Index into `_AXIS_SHAPE_CASES`.
        :type case_index: int
        :param masked: Whether a quarter of each row is ``-inf``.
        :type masked: bool
        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            case_id, shape, axis = _AXIS_SHAPE_CASES[case_index]
            z = self._axis_batch(shape, axis, masked, seed=1000 + case_index)
            z_compute = _to_compute_dtype(z)

            layer = Sparsemax(axis=axis)
            out = layer(z_compute)

            rows = self._assert_matches_exact_oracle_along_axis(
                z_compute, out, axis, case_id
            )

            # ANTI-VACUITY, per case: the number of projections compared is
            # known exactly here (no row is ever filtered), so it is asserted
            # as an equality rather than a floor.
            expected = int(np.prod(shape)) // shape[axis]
            assert rows == expected >= 1, (
                f"{case_id}: compared {rows} rows, expected {expected}"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_dynamic_k_matches_the_exact_oracle(self, policy: str) -> None:
        """``input_shape[axis] is None`` — the reduction dim itself unknown.

        ``Sparsemax.call``'s ``if input_shape[axis] is not None`` branch is the
        layer's ONLY static-shape read: it takes
        ``k`` from ``input_shape[axis]`` when that is known and from the
        backend shape tensor otherwise, and four downstream consumers
        (``reshape``, ``arange``, ``reshape``, ``one_hot``) then have to accept
        a TENSOR ``k``. This exercises that branch on the harder spelling —
        the UNKNOWN dimension is the reduction axis itself, not a free batch
        dim — and retraces the same model at two widths, so a ``k`` frozen at
        first trace produces a wrong projection at the second.

        A shape-only version of this test would pass on a completely broken
        projection; every width here is compared against the exact rational
        oracle.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            axis = 1
            inp = keras.Input(shape=(None, 6))
            model = keras.Model(inp, Sparsemax(axis=axis)(inp))

            # PRECONDITION (anti-vacuity): the branch under test is selected by
            # `input_shape[axis] is None`. If this ever becomes a static width
            # the test silently degrades into a duplicate of the static grid.
            assert inp.shape[axis] is None, (
                f"the reduction axis is static ({inp.shape[axis]}); this test "
                "would then not exercise the dynamic-k branch at all"
            )

            rows_compared = 0
            for k in (8, 257):
                z = self._axis_batch((2, k, 6), axis, masked=True, seed=k)
                z_compute = _to_compute_dtype(z)
                rows_compared += self._assert_matches_exact_oracle_along_axis(
                    z_compute, model(z_compute), axis, f"dynamic_k={k}"
                )

            # 2 widths x (2 * 6) rows each.
            assert rows_compared == 24, (
                f"dynamic-K guard compared {rows_compared} rows, expected 24"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.parametrize("policy", _XLA_POLICIES)
    def test_jit_compiled_moved_axis_matches_the_exact_oracle(
        self, policy: str
    ) -> None:
        """The single ``jit_compile=True`` point of the moved-axis grid.

        One ``(rank, axis, K)`` point — rank-3, ``axis=1``, K=257 — at all four
        policies. The width grid under XLA is already swept at ``axis=-1`` by
        `TestSparsemaxXLACapability`; what is unique here is that XLA lowers the
        TRANSPOSE/reshape shim of `Sparsemax.call` as well as the
        projection, and that the comparison is against the EXACT oracle rather
        than against the layer's own eager answer.

        :param policy: Global Keras dtype policy name.
        :type policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(policy)
        try:
            axis = 1
            z = self._axis_batch((2, 257, 3), axis, masked=True, seed=257)
            z_compute = _to_compute_dtype(z)

            layer = Sparsemax(axis=axis)

            @tf.function(jit_compile=True)
            def _compiled(t):
                return layer(t)

            # `jit_compile=True` raises if XLA cannot lower the graph, so the
            # call itself is the compilation assertion. Do not wrap it.
            out = _compiled(tf.convert_to_tensor(z_compute))

            rows = self._assert_matches_exact_oracle_along_axis(
                z_compute, out, axis, "jit_axis1_K257"
            )
            assert rows == 6, f"compared {rows} rows, expected 6"
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_save_load_round_trip_preserves_axis_1(self) -> None:
        """`axis` survives `.keras` serialization, and the answer is unchanged.

        `get_config` carrying `axis` is necessary but not sufficient: this
        asserts the RELOADED model's output is BIT-IDENTICAL to the original's
        (`assert_array_equal`, zero tolerance) and that both match the exact
        rational oracle along ``axis=1``. A round-trip that silently reverted
        to the default ``axis=-1`` would produce a well-formed, normalised,
        completely wrong distribution — which a shape or a sum-to-one check
        cannot see, since both hold along either axis.
        """
        axis = 1
        z = self._axis_batch((2, 257, 3), axis, masked=True, seed=11)
        z_compute = _to_compute_dtype(z)

        inp = keras.Input(shape=(257, 3))
        model = keras.Model(inp, Sparsemax(axis=axis)(inp))
        y0 = ops.convert_to_numpy(model(z_compute))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sparsemax_axis1.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        reloaded_axes = [
            layer.axis for layer in loaded.layers if isinstance(layer, Sparsemax)
        ]
        assert reloaded_axes == [axis], (
            f"reloaded model's Sparsemax axes are {reloaded_axes}, expected "
            f"[{axis}] — `axis` did not survive the round trip"
        )

        reloaded_out = loaded(z_compute)
        np.testing.assert_array_equal(
            ops.convert_to_numpy(reloaded_out),
            y0,
            err_msg="reloaded model is not bit-identical to the original",
        )
        rows = self._assert_matches_exact_oracle_along_axis(
            z_compute, reloaded_out, axis, "save_load_axis1"
        )
        assert rows == 6, f"compared {rows} rows, expected 6"

    def test_the_axis_grid_covers_every_declared_rank_axis_and_width(
        self,
    ) -> None:
        """Grid-level anti-vacuity for G8: the coverage property, asserted.

        `_AXIS_SHAPE_CASES` is condensed from the exploration's 302-point
        cross-product, so the thing that must not silently erode is COVERAGE.
        This pins the exact degrees of freedom the condensation preserves —
        ranks 1-4, both signs of axis including ``0``, ``1``, ``-2`` and
        ``-1``, every width in ``_XLA_WIDTHS`` (I-7: 256 and 257 are
        load-bearing, the historical onsets were non-monotone in K), and the
        presence of F-07's live rank-4 ``axis=-2`` shape.

        THE FLOOR. Measured: 33 rows per (mask variant, policy) x 2 variants x
        4 policies = **264** projections. The floor is **200**, chosen so that
        losing an entire policy (-66) or the masked half (-132) goes red while
        ordinary shrinkage of a leading dimension does not. Same construction
        as `test_the_corpus_grid_is_not_silently_skipped`'s ``>= 100``.
        """
        ranks = {len(shape) for _, shape, _ in _AXIS_SHAPE_CASES}
        assert ranks == {1, 2, 3, 4}, f"ranks covered: {sorted(ranks)}"

        axes = {axis for _, _, axis in _AXIS_SHAPE_CASES}
        assert axes == {0, 1, -2, -1}, f"axes covered: {sorted(axes)}"

        widths = {shape[axis] for _, shape, axis in _AXIS_SHAPE_CASES}
        assert widths == set(_XLA_WIDTHS), (
            f"widths covered: {sorted(widths)}, expected exactly "
            f"{sorted(_XLA_WIDTHS)} — the non-monotone points 256 and 257 are "
            "load-bearing (I-7) and must not be thinned"
        )

        assert any(
            len(shape) == 4 and axis == -2
            for _, shape, axis in _AXIS_SHAPE_CASES
        ), (
            "F-07's live shape (rank-4, axis=-2, "
            "`capsule_routing_attention.py:331`) is no longer in the grid"
        )

        rows = sum(
            int(np.prod(shape)) // shape[axis]
            for _, shape, axis in _AXIS_SHAPE_CASES
        )
        total = rows * 2 * len(_XLA_POLICIES)
        assert total >= 200, (
            f"the moved-axis grid would compare only {total} projections "
            "across (cases x masked/unmasked x 4 policies); expected >= 200, "
            "measured 264 — this guard has gone vacuous"
        )


# ---------------------------------------------------------------------
# `SparsemaxLoss` x the global dtype policy  (MEASURED 2026-07-29; FIXED here)
# ---------------------------------------------------------------------
# WHAT WAS PROBED.  `losses/sparsemax_loss.py` builds its inner `Sparsemax()`
# EAGERLY in `__init__`, so that sub-layer's dtype policy is frozen at
# LOSS-CONSTRUCTION time. The open question was what happens when the ambient
# policy differs at CALL time.
#
# WHAT WAS MEASURED — the full 4x4 CONSTRUCT x CALL policy grid, GPU/TF:
# 4/4 float32-CONSTRUCTION points worked and 12/12 non-float32-construction
# points raised, REGARDLESS of the call policy. The call-time policy is not
# the axis; the construction-time policy is.
#
# THE MECHANISM.  `keras.losses.Loss` takes its dtype from `backend.floatx()`,
# NOT from the global mixed-precision policy — so `loss.dtype` is float32
# under EVERY policy, and `Loss.__call__` casts both `y_true` and `y_pred` to
# it before `call()` runs. The inner layer, by contrast, honours the policy
# captured at construction and returns ITS compute dtype. When the two
# disagreed, `y_pred - p` was float32 minus float16/bfloat16/float64 and
# TensorFlow raised.
#
# THE FIX THAT SHIPPED (`plan-2026-07-29T110112-09832856` step 10, `D-015`):
# ONE line in `losses/sparsemax_loss.py` — `p = keras.ops.cast(
# self.sparsemax(y_pred), self.dtype)`. THE FIX IS IN THE LOSS, NOT IN THE
# LAYER. Making `Sparsemax` return float32 under a narrow policy is prohibited
# by that layer's own cast-back anchor and by user ruling; it would change the
# dtype contract for every other consumer. `D-007`'s never-narrow reduction
# rule is a separate, already-closed matter and is not what fixed this.
#
# WHY THE PROJECTION IS STILL COMPUTED NARROW.  The cast reconciles dtypes at
# the boundary; it does not widen the projection. Under fp16/bf16 construction
# the sparsemax runs in fp16/bf16 (measured loss error 1.56e-05 / 7.89e-05
# against the exact-rational Fenchel-Young reference), which is why the
# tolerance below is derived on the CONSTRUCTION policy's compute dtype and
# NOT on float32 — a float32-derived atol would be ~1e2..1e4x too tight at
# bf16 and the guard would be permanently red, or (if floored) vacuous.
# The rejected alternative, `Sparsemax(dtype=self.dtype)` in `__init__`
# (spelling D), would have kept float32 accuracy at every policy but changed
# the sub-layer's advertised policy contract and broken
# `test_inner_sparsemax_policy_is_frozen_at_construction`, which under the
# shipped spelling stays valid UNCHANGED.
#
# THE TESTS BELOW ASSERT THE WORKING BEHAVIOUR, and they were proven RED with
# the `sparsemax_loss.py` cast reverted IN PLACE (never `git stash`): all 6
# parametrizations of
# `test_loss_constructed_under_a_narrow_policy_matches_the_reference` fail.
# NO NEGATIVE CONTROL ASSERTING A RAISE SURVIVES HERE, deliberately: the old
# one used `pytest.raises(tf.errors.InvalidArgumentError, match="Sub")`, which
# only ever covered the EAGER path — the graph / `model.fit` path raised
# `TypeError`. If a raise-asserting control is ever re-added for some other
# reason, it must accept BOTH exception types.
# ---------------------------------------------------------------------

#: Policies whose compute dtype differs from `backend.floatx()`, i.e. every
#: policy under which the inner `Sparsemax` is frozen at a dtype the `Loss`
#: itself does not use. Constructing under these USED to poison the loss;
#: since `D-015` the boundary cast reconciles them, so these are the NARROW
#: construction policies, not broken ones.
_LOSS_NARROW_CONSTRUCT_POLICIES = ("mixed_float16", "mixed_bfloat16", "float64")


class TestSparsemaxLossDtypePolicy:
    """`SparsemaxLoss`'s eager inner `Sparsemax()` vs. the global policy.

    Guards `losses/sparsemax_loss.py`'s construction-time dtype freeze and the
    `D-015` boundary cast that reconciles it with `keras.losses.Loss`'s own
    `backend.floatx()` dtype — see the note above for the measured 4x4 grid,
    the mechanism, and why the fix lives in the loss rather than in the layer.
    """

    # STILL VALID UNCHANGED UNDER `D-015`, and that is a deliberate property of
    # the spelling that shipped: the boundary cast leaves the sub-layer's
    # construction-time policy alone. Spelling D (`Sparsemax(dtype=self.dtype)`
    # in `__init__`) would have made this test RED at its 3 narrow
    # parametrizations — the inner layer would report float32 instead of the
    # construction policy's compute dtype. Do not "modernize" this test to
    # accommodate a float32 inner layer; that would silently endorse the
    # rejected fix.
    @pytest.mark.parametrize(
        "construct_policy", ("float32",) + _LOSS_NARROW_CONSTRUCT_POLICIES
    )
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
    @pytest.mark.parametrize("construct_policy", _LOSS_NARROW_CONSTRUCT_POLICIES)
    def test_loss_constructed_under_a_narrow_policy_matches_the_reference(
        self, construct_policy: str, call_policy: str
    ) -> None:
        """`D-015`: a loss built under ANY policy runs and is numerically right.

        This is the guard for the `losses/sparsemax_loss.py` boundary cast. It
        replaced a test that asserted the RAISE these 6 grid points used to
        produce; that name and that assertion became false claims the moment
        the cast landed (`plans/LESSONS.md:44`).

        Asserts three things per grid point: the loss RUNS, it returns
        `backend.floatx()` (asserted as `floatx()`, never as the literal
        `"float32"` — `Loss` tracks the backend default, and a hard-coded
        literal would make this an environment pin), and its value matches the
        Fenchel-Young formula `0.5*||z - p||^2 - z^T y` evaluated on the
        EXACT-rational sparsemax of the bits the frozen sub-layer really
        received.

        The call-time policy stays parametrized because "the call policy is
        irrelevant" is the measured 4x4 finding this class exists to keep true
        — it is now asserted in the working direction instead of the raising
        one.

        RED PROOF (`plan-2026-07-29T110112-09832856` step 10): with the cast at
        the `p = self.sparsemax(y_pred)` site reverted IN PLACE (never
        `git stash`), all 6 parametrizations fail on the `loss(...)` call
        itself, before any assertion runs.

        :param construct_policy: Policy in force at construction — the axis
            that actually matters.
        :type construct_policy: str
        :param call_policy: Policy in force at call time (measured irrelevant).
        :type call_policy: str
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy(construct_policy)
            loss = SparsemaxLoss(from_logits=True)
            inner_dtype = loss.sparsemax.compute_dtype
            # Premise: the sub-layer really is frozen at the NARROW dtype, so
            # this grid point exercises the reconciliation rather than a
            # degenerate all-float32 path.
            assert inner_dtype != keras.backend.floatx(), (
                f"inner Sparsemax reports {inner_dtype!r} under "
                f"{construct_policy!r}; this case can no longer see the "
                "construct-time dtype freeze it exists to guard"
            )

            keras.mixed_precision.set_global_policy(call_policy)
            # `masked_fraction=0.0` - the no-`inf` control. `SparsemaxLoss`'s
            # `0.5*||z - p||^2 - z^T y` is undefined on `-inf` logits, so the
            # masked corpus belongs to the LAYER's tests, not the loss's.
            z = _partially_masked_row_batch(16, 0.0, seed=606)
            z_compute = _to_compute_dtype(z)
            y = np.zeros_like(z_compute)
            y[:, 0] = 1.0

            out = loss(
                ops.convert_to_tensor(y), ops.convert_to_tensor(z_compute)
            )
            assert keras.backend.standardize_dtype(out.dtype) == (
                keras.backend.floatx()
            ), (
                f"loss dtype {out.dtype} - expected backend.floatx()="
                f"{keras.backend.floatx()} under construct={construct_policy}, "
                f"call={call_policy} (keras.losses.Loss is not policy-aware)"
            )

            # THE BITS THAT REACH THE PROJECTION. `Loss.__call__` upcasts the
            # caller's bits to `floatx()` losslessly, then the frozen sub-layer
            # casts THAT down to its own compute dtype. Restoring the
            # construction policy here is not cosmetic: `_exact_sparsemax`
            # asserts it is fed the ACTIVE policy's compute dtype, which is
            # precisely the "feed the oracle the layer's real bits" guard
            # (`plans/LESSONS.md:42`), and here the layer in question is the
            # frozen sub-layer, not the ambient one.
            keras.mixed_precision.set_global_policy(construct_policy)
            assert _output_dtype() == inner_dtype
            z_outer = np.asarray(z_compute, dtype=np.float64)
            z_inner = _to_compute_dtype(z_compute)
            p64 = _exact_rows_as_float64(z_inner)
            y64 = np.asarray(y, dtype=np.float64)
            per_sample = 0.5 * np.sum((z_outer - p64) ** 2, axis=-1) - np.sum(
                z_outer * y64, axis=-1
            )
            expected = float(np.mean(per_sample))

            # DERIVED ON THE CONSTRUCTION POLICY'S COMPUTE DTYPE, not float32.
            # `eps` is the per-element projection tolerance for a layer running
            # in `inner_dtype`; first-order propagation through the loss gives
            # `dL/dp_j = -(z_j - p_j)`, so an elementwise error of `eps` costs
            # at most `sum_j |z_j - p_j| * eps`. `_TF32_ATOL_FLOOR` covers the
            # float32 summation the loss itself performs. Charging `eps` at
            # float32 instead would under-budget bf16 by ~1e4x and this guard
            # would be permanently red.
            #
            # `apply_tf32_floor=False` IS LOAD-BEARING; DO NOT DROP IT.
            # `_oracle_atol`'s clamp is a TF32 belt, and TF32 is a float32
            # matmul/convolution feature: neither `Sparsemax` nor
            # `SparsemaxLoss` contains a matmul or a convolution, so nothing on
            # this path can bind it (the identical argument `_grad_atol`'s
            # docstring makes for carrying no floor at all). Charging the
            # CLAMPED value here multiplied 1e-3 by `max_j sum|z - p| = 93.27`
            # and made the float64 arm 94x looser than derivable: MEASURED at
            # `iter-1/step-13` on GPU, a systematic 1% shrink of the projection
            # (`p -> 0.99p`, moving the loss by 7.42e-02) PASSED it, and 0.1%
            # passed all six arms. Unfloored, the float64 arms run at atol
            # 1.000e-03 against measured errors 1.006e-05 / 3.897e-06 — still
            # 99x / 257x margin — and both go red on 1% AND on 0.1%. The fp16
            # and bf16 arms are BYTE-UNCHANGED (their derived `eps`, 1.4651e-03
            # and 1.1719e-02, already exceeds the floor), so this tightening
            # buys the float64 third of the grid and costs the other two
            # nothing.
            eps = _oracle_atol(z_inner, inner_dtype, apply_tf32_floor=False)
            atol = _TF32_ATOL_FLOOR + eps * float(
                np.max(np.sum(np.abs(z_outer - p64), axis=-1))
            )
            actual = float(ops.convert_to_numpy(out))
            assert abs(actual - expected) <= atol, (
                f"loss {actual} != Fenchel-Young reference {expected} "
                f"(construct={construct_policy}, call={call_policy}, "
                f"inner={inner_dtype}, derived atol={atol})"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_loss_built_under_float32_matches_the_reference(self) -> None:
        """The float32 NUMERIC ANCHOR: build under float32, then switch to fp16.

        Near-vacuous as a *policy* test since `D-015` — every construction
        policy now works, so "the one order that works" is no longer
        distinguishing and the name that said so was retired. It is kept, and
        renamed, for what it still uniquely pins: the float32 arm of the
        Fenchel-Young numerics, which is the arm the `D-015` cast is
        BIT-IDENTICAL on (measured, all three reductions, `np.array_equal`) and
        therefore the arm that must never move.

        Asserts the observable consequences: the loss still runs, it returns
        `backend.floatx()` (not float16 - it is not policy-aware at all), and
        its value matches the Fenchel-Young formula evaluated on the
        EXACT-rational sparsemax of the bits the layer actually received.
        """
        previous = keras.mixed_precision.global_policy().name
        try:
            keras.mixed_precision.set_global_policy("float32")
            loss = SparsemaxLoss(from_logits=True)
            inner_dtype = loss.sparsemax.compute_dtype

            keras.mixed_precision.set_global_policy("mixed_float16")
            # No-`inf` control: see the note in the narrow-policy test above.
            z = _partially_masked_row_batch(16, 0.0, seed=607)
            # The fp16 bits the caller hands in; `Loss.__call__` upcasts these
            # to float32 losslessly, so these ARE the received bits.
            z_compute = _to_compute_dtype(z)
            y = np.zeros_like(z_compute)
            y[:, 0] = 1.0

            out = loss(
                ops.convert_to_tensor(y), ops.convert_to_tensor(z_compute)
            )
            assert keras.backend.standardize_dtype(out.dtype) == (
                keras.backend.floatx()
            ), (
                f"loss dtype {out.dtype} - expected backend.floatx()="
                f"{keras.backend.floatx()} even under mixed_float16 "
                f"(keras.losses.Loss is not policy-aware)"
            )

            # Reference: L = 0.5 * ||z - p||^2 - z^T y, reduced by the loss's
            # default `sum_over_batch_size`. Restoring the CONSTRUCTION policy
            # is load-bearing for the same reason as in the narrow-policy test:
            # `_exact_sparsemax` asserts it is fed the active policy's compute
            # dtype, and the layer whose bits matter is the FROZEN sub-layer.
            keras.mixed_precision.set_global_policy("float32")
            z_inner = _to_compute_dtype(z_compute)
            z64 = np.asarray(z_compute, dtype=np.float64)
            p64 = _exact_rows_as_float64(z_inner)
            y64 = np.asarray(y, dtype=np.float64)
            per_sample = 0.5 * np.sum((z64 - p64) ** 2, axis=-1) - np.sum(
                z64 * y64, axis=-1
            )
            expected = float(np.mean(per_sample))

            # First-order propagation of the per-element sparsemax tolerance
            # through the loss: dL/dp_j = -(z_j - p_j), so an elementwise error
            # of `eps` costs at most `sum_j |z_j - p_j| * eps`. `_TF32_ATOL_FLOOR`
            # covers the float32 summation itself. `eps` is charged at the
            # CONSTRUCTION policy's compute dtype (`float32` here) because that
            # is what the frozen sub-layer runs in - charging it at the ambient
            # fp16 would make this float32 anchor 1.47x looser than the arm it
            # claims to pin.
            #
            # `apply_tf32_floor=False` for the same reason as in the
            # narrow-policy test above (no matmul, no convolution, nothing for
            # TF32 to bind to) - see the long note there. The reviewer that
            # raised this only measured the float64 arms; this float32 anchor
            # was floored too, and gains just as much. MEASURED at
            # `iter-1/step-13` on GPU: atol 1.005e-01 -> 1.042e-03, a 96.5x
            # tightening, against a measured error of 1.091e-05 (95x margin
            # retained), and it now goes RED on both a 1% and a 0.1% systematic
            # shrink of the projection, which it did not before.
            eps = _oracle_atol(z_inner, inner_dtype, apply_tf32_floor=False)
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
