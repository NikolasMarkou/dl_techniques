"""Tests for the Sparsemax activation layer.

.. warning::

   **THIS FILE BEING GREEN DOES NOT MEAN SPARSEMAX IS NUMERICALLY CORRECT.**
   The work that produced these tests closed **Defect A only** — the
   ``0 * -inf = NaN`` arithmetic gather that made a partially ``-inf``-masked
   row return all-``NaN``.  Four further defects (**B**, **C**, **D**, **E**)
   were MEASURED and remain **OPEN and UNGUARDED** inside ``Sparsemax.call()``;
   see the ``# DECISION plan-2026-07-28T134123-420f6ccb/D-017`` anchor in
   ``src/dl_techniques/layers/activations/sparsemax.py``.  They are pinned by
   :class:`TestSparsemaxOpenDefects` below as ``xfail(strict=True)`` cases —
   pinned, not fixed.  A fifth pin there records a REGRESSION the same work
   knowingly shipped: on Defect-B inputs the failure moved from loud (``NaN``)
   to silent (a finite, un-normalised row).
"""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

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
# ERROR PROPAGATION.  The layer computes ``out_i = max(z_i - tau, 0)`` with
# ``tau = (S - 1) / k_z`` and ``S = cumsum(sort(z))[k_z - 1]``, all in the
# compute dtype.  ``out_i`` is a difference of two quantities of magnitude up to
# ``M = max|z_finite|`` (``|z_i| <= M`` and ``tau <= max(z) <= M``), so AS THE
# LAYER IS CURRENTLY WRITTEN its absolute resolution is ``ulp(M)``.
#
# THAT IS A PROPERTY OF THIS FORMULATION, NOT A LAW.  ``M`` here is the
# magnitude of the raw logits only because the layer never shifts them.
# Sparsemax is shift-invariant, so subtracting the row max before the cumsum is
# EXACT, and it replaces ``M`` with the row SPREAD ``max(z) - min(z)`` (0 for a
# plateau row), collapsing the floor by orders of magnitude.  Measured: that one
# change takes this file's own grid from 86/128 to 0/128 violations, with
# ``max_err`` 0.013165 -> 0.000628 — i.e. 12x BELOW ``ulp(M = 10)``.  **beta
# MUST re-tighten ``_ULP_BUDGET`` (and re-derive the ``ulp`` argument against
# the row spread rather than ``max|z|``) once row-max subtraction lands.**  Do
# not read the budget below as a permanent ceiling.
#
# Counting the roundings that survive to the output of the CURRENT
# formulation, each bounded by half the local spacing:
#
#   0.5 ulp(M)  rounding ``z_i`` into the compute dtype (the input is float32);
#   1.0 ulp(M)  one rounding of a cumsum partial sum.  A partial sum has
#               magnitude up to ``k_z * M``, so its own spacing is up to
#               ``2 * k_z * ulp(M)`` (the factor 2 is binade misalignment
#               between ``k_z * M`` and ``M``); ``tau`` then DIVIDES by
#               ``k_z``, restoring the scale, so the contribution is
#               ``0.5 * 2 * ulp(M) = 1.0 ulp(M)``;
#   1.0 ulp(M)  rounding of ``S - 1.0``, by the same magnitude-and-divide
#               argument;
#   0.5 ulp(M)  rounding of the division ``/ k_z`` (result ``|tau| <= M``);
#   0.5 ulp(M)  rounding of the subtraction ``z_i - tau`` (result ``<= M``).
#
# Sum = 3.5 ulp(M); rounded up to the next integer, ``_ULP_BUDGET = 4``.
# ``sort``, ``flip`` and the one-hot selection are exact and contribute nothing.
#
# THIS BOUND IS DELIBERATELY STRICTER THAN WORST-CASE THEORY.  A rigorous
# forward bound on a ``k_z``-term cumsum is ``O(k_z)`` ulp, which at K = 512
# would permit a completely wrong answer and would be a vacuous assertion.  We
# charge ONE representative accumulation rounding instead, on the basis that the
# per-step errors partially cancel rather than add coherently.  The consequence
# is that this test FAILS if the layer's accumulation error ever starts growing
# with ``k`` — which is the point.  The constant is derived from the arithmetic
# above; it is NOT fitted to observed error (measured worst case over this
# file's grid is 1.685 ulp under ``mixed_float16``, i.e. the bound carries a
# 2.4x margin).
#
# float32 and float64 are NOT loosened: ``_TF32_ATOL_FLOOR`` is a FLOOR, and at
# these grids ``4 * ulp(M)`` is ~4e-6 (float32) / ~7e-15 (float64), far below
# it, so both keep exactly today's 1e-3 bar.  Only float16, whose ``4 * ulp(M)``
# exceeds 1e-3 for every ``M >= 0.5``, is governed by the derived term.
# ---------------------------------------------------------------------

#: Flat absolute floor, retained for float32/float64: the documented TF32
#: precision floor (`test_linear_attention.py:37` disables TF32
#: process-globally at import, so the same assertion runs in two regimes).
_TF32_ATOL_FLOOR = 1e-3

#: Ulp of ``max|z_finite|`` allowed between layer and oracle. Derived above.
_ULP_BUDGET = 4.0

_NUMPY_DTYPE = {
    "float16": np.float16,
    "bfloat16": None,  # np has no bfloat16; not used by this file's grid.
    "float32": np.float32,
    "float64": np.float64,
}


def _oracle_atol(z: np.ndarray, output_dtype: str) -> float:
    """Absolute tolerance for the layer-vs-oracle comparison on inputs ``z``.

    Returns ``max(_TF32_ATOL_FLOOR, _ULP_BUDGET * ulp(max|z_finite|))``
    evaluated in the layer's OUTPUT dtype. See the derivation above.

    :param z: The logit batch handed to the layer (may contain ``-inf``).
    :type z: np.ndarray
    :param output_dtype: Name of the layer's output dtype, e.g. ``'float16'``.
    :type output_dtype: str

    :return: Absolute tolerance for ``np.testing.assert_allclose``.
    :rtype: float
    """
    np_dtype = _NUMPY_DTYPE[output_dtype]
    assert np_dtype is not None, f"no numpy dtype for {output_dtype!r}"
    finite = np.asarray(z)[np.isfinite(z)]
    assert finite.size > 0, "batch is fully masked; that case is out of scope"
    max_abs = float(np.max(np.abs(finite)))
    ulp = float(np.spacing(np.asarray(max_abs, dtype=np_dtype)))
    return max(_TF32_ATOL_FLOOR, _ULP_BUDGET * ulp)


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
    # It is `max(1e-3, 4 * ulp(max|z_finite|))` in the OUTPUT dtype: float32
    # and float64 keep exactly the historical 1e-3 TF32 floor; float16 gets the
    # ulp-scaled term, because a flat 1e-3 demands precision below float16's
    # representable resolution and can therefore never pass.
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
            ref = _sparsemax_reference(z)
            atol = _oracle_atol(z, _output_dtype())
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
                    collected.append((label, out.astype(np.float64), z))

        # (b) secondary correctness check, only reachable once (a) held everywhere.
        #     Tolerance is DERIVED per grid point from the output dtype's
        #     resolution — see `_oracle_atol` and its derivation.
        out_dtype = _output_dtype()
        for label, out, z in collected:
            atol = _oracle_atol(z, out_dtype)
            np.testing.assert_allclose(
                out,
                _sparsemax_reference(z),
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


# ---------------------------------------------------------------------
# SCOPE PINS: FIVE cases — the four OPEN defects B/C/D/E, plus the accepted
# loud -> silent conversion this plan knowingly ships on Defect-B inputs.
#
# Each case below asserts the CORRECT behaviour and is expected to FAIL.
# `strict=True` is the whole point: when the follow-up plan fixes one of these,
# the case XPASSes, `strict=True` turns that XPASS into a hard FAILURE, and
# whoever fixed it is FORCED to delete the marker rather than have the
# improvement absorbed silently. Do not relax these to non-strict xfail, to
# `skip`, or delete them: an anchor without a test that can fail is a comment,
# not a guard.
#
# An `xpassed` here CONTRADICTS a measured defect. That is a FINDING to
# investigate, never a win to bank.
# ---------------------------------------------------------------------


class TestSparsemaxOpenDefects:
    """Executable record of what ``Sparsemax`` still gets WRONG.

    **A green test file does NOT mean sparsemax is numerically correct.** The
    change that landed alongside these tests closed **Defect A** (the
    ``0 * -inf = NaN`` gather) and, as a bonus, the all-finite cumsum-overflow
    family — and nothing else. Defects **B**, **C**, **D** and **E** below are
    MEASURED, OPEN, UNGUARDED and deferred to a follow-up plan. They share one
    root cause: the reduction (ramp, cumsum, support test, ``k_z`` count) runs
    in the COMPUTE dtype, which under ``float16`` / ``bfloat16`` has neither
    the range nor the integer precision the algorithm needs — and under
    ``float32`` fails once magnitudes reach ~1.7e7.

    A **fifth** pin records something different in kind: not a defect this
    change failed to fix, but a REGRESSION it knowingly introduced — on
    Defect-B inputs the failure moved from loud (NaN) to silent (a finite,
    un-normalised row). It is accepted deliberately; see clause (e) of the
    anchor.

    See the ``# DECISION plan-2026-07-28T134123-420f6ccb/D-017`` anchor in
    ``src/dl_techniques/layers/activations/sparsemax.py``.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect B (sparsemax.py:~220, OPEN, deferred to the follow-up "
            "'beta' plan): an overflow-born non-finite z_cumsum is ADMITTED to "
            "the support because support = 1 + finite - (-inf) = +inf > 0. "
            "Measured at spread 16.95, mixed_float16, K=4096, with NO -inf in "
            "the input: k_z = 1863 where the exact answer is 1. This case "
            "fails on the NORMALISATION assertion: the output is entirely "
            "FINITE (nan=0, inf=0) and sum(out) = 16.9375 where 1.0 is "
            "correct. (An interim revision of this plan added a 'loudness "
            "guard' that made it fail on the FINITENESS assertion with "
            "nan=4096 instead; the guard was measured to destroy correct "
            "answers on ordinary rows and was REVERTED — see clause (e) of the "
            "D-017 anchor. Do not re-derive it.) The silence itself is pinned "
            "separately by test_defect_b_loud_to_silent_conversion_is_"
            "accepted. Remove this marker when beta fixes it."
        ),
    )
    def test_defect_b_overflow_born_inf_admitted_to_support(self) -> None:
        """fp16 cumsum overflow must not inflate the support set."""
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "ACCEPTED LOUD -> SILENT CONVERSION on Defect-B inputs (OPEN, "
            "deferred to the follow-up 'beta' plan). This pin is about the "
            "ALARM, not the answer: on this all-finite fp16 row the layer "
            "SHOULD either be normalised or fail visibly, and it does NEITHER "
            "— it returns a finite, plausible-looking sum(out) = 16.9375 where "
            "1.0 is correct. Before this plan the same input returned "
            "nan=4096, because the `-inf * 0.0` product at the NON-SELECTED "
            "overflowed positions manufactured a NaN that INCIDENTALLY masked "
            "Defect B's wrong k_z. The ops.where selection removes that "
            "product, so the pre-existing wrong answer is now quiet. This is "
            "DELIBERATE: the alternative — a guard keyed on cumsum finiteness "
            "— was implemented, measured and REVERTED because it destroyed "
            "exactly-correct answers on two ordinary input families (see "
            "test_finite_cumsum_overflow_rows_are_correct_not_nan). Full "
            "reasoning: clause (e) of the "
            "`# DECISION plan-2026-07-28T134123-420f6ccb/D-017` anchor in "
            "sparsemax.py. Beta closes this with a widened reduction plus an "
            "OUTPUT-side predicate (|sum(out) - 1| > tol); remove this marker "
            "then."
        ),
    )
    def test_defect_b_loud_to_silent_conversion_is_accepted(self) -> None:
        """A Defect-B row must be either normalised OR loud. It is neither."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            k = 4096
            z = np.full((1, k), -16.95, dtype=np.float32)
            z[:, 0] = 0.0
            assert np.isfinite(z).all(), "premise: the INPUT is entirely finite"

            out = ops.convert_to_numpy(Sparsemax()(z)).astype(np.float64)

            total = float(np.sum(out[np.isfinite(out)]))
            normalised = abs(total - 1.0) <= 1e-2
            loud = not np.isfinite(out).all()
            assert normalised or loud, (
                "accepted loud->silent conversion: the row is neither correct "
                f"nor alarming — finite everywhere, sum(out) = {total}, "
                "expected 1.0"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect C (sparsemax.py:~210, OPEN, deferred to the follow-up "
            "'beta' plan): ops.arange(1, k+1, dtype=inputs.dtype) cannot "
            "represent the ramp in a narrow dtype, so the layer RAISES. The "
            "break is NON-MONOTONE in K: it raises at mixed_bfloat16 K=256/257 "
            "but is fine at K=512, and raises at mixed_float16 K=2048 but is "
            "fine at K=4096. Remove this marker when beta fixes it."
        ),
    )
    def test_defect_c_bfloat16_arange_ramp_raises(self) -> None:
        """``Sparsemax()(z)`` must not raise under ``mixed_bfloat16`` at K=256."""
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect D (sparsemax.py:~220, OPEN, deferred to the follow-up "
            "'beta' plan): round-off absorbs the literal 1.0 in "
            "`1.0 + k*z - cumsum`, so support == 0 everywhere, k_z == 0, "
            "one_hot(-1) is all-zero, tau = -inf and the output is all +inf. "
            "Measured float32 onset 1.68e7 (K=4) / 7.77e7 (K=512). "
            "Remove this marker when beta fixes it."
        ),
    )
    def test_defect_d_float32_large_magnitude_swamps_the_literal_one(self) -> None:
        """float32 magnitudes at the 1.0-swamping onset must still project."""
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect E (sparsemax.py:~225, OPEN, deferred to the follow-up "
            "'beta' plan): k_z = ops.sum(support_mask) accumulates in the "
            "compute dtype and hits float16's integer wall — measured on the "
            "TF/GPU tree reduction: 2049 -> 2048, 2051 -> 2052, 4095 -> 4096 "
            "(2050 / 3000 / 4094 are exact). The 2051 -> 2052 overshoot selects "
            "a MASKED position whose z_cumsum is -inf, so tau = -inf and the row "
            "dies (nan=2045, inf=2051 at K=4096). The M2 ops.where fix CANNOT "
            "help here: the -inf sits at the SELECTED index, not at a masked-out "
            "one, so no spelling of that selection recovers tau. An earlier "
            "version of this string "
            "claimed 4095 -> 4096 indexes OUT OF RANGE for depth 4096; that claim "
            "is FALSE (4095 is a valid index) and was deleted — no end-to-end "
            "input reaching an out-of-range one-hot was constructible, since "
            "every such K raises Defect C first. Remove this marker when beta "
            "fixes it."
        ),
    )
    def test_defect_e_fp16_support_count_overshoots_into_a_masked_index(self) -> None:
        """An fp16 support of exactly 2051 must not select a masked position."""
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
