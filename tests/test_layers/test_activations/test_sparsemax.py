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

import os
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

    # The reduction dtype the layer will use, re-derived here rather than
    # imported: `D-007` in `sparsemax.py` (`float32` for the two narrow dtypes,
    # the compute dtype itself otherwise). Deliberately duplicated for the same
    # reason the oracles are test-local — an instrument must not share code
    # with the thing it measures. If this ever disagrees with the layer, the
    # `u_r` terms below are being charged against the wrong dtype.
    reduction_dtype = (
        "float32" if output_dtype in ("float16", "bfloat16") else output_dtype
    )
    np_reduction = _NUMPY_DTYPE[reduction_dtype]

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
