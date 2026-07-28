"""
Unit tests for `dl_techniques.layers.attention.common`.

Scope: the shared, class-free primitives of the ``layers/attention`` package —
with the emphasis squarely on :func:`apply_attention_mask`, the fp16-safe additive
mask-bias builder that ten masked attention layers are about to adopt.

**Why these tests are shaped the way they are.**

1.  ``MASK_BIAS_VALUE = -1e9`` is NOT dtype-independent: ``np.float16(-1e9)`` is
    ``-inf``. Every fp16 assertion below is therefore preceded (in
    :class:`TestMaskBiasValueIsTheActualHazard`) by an explicit **anti-vacuity**
    check that the constant really does overflow to ``-inf`` in fp16. Without it a
    "passing" fp16 test proves nothing — it might be passing because the hazard was
    never reached (`plans/LESSONS.md`: an ``N=7`` test once hid an fp16 ``-inf``
    that only appeared at ``N >= 512``).
2.  Every fp16 assertion has a ``float32`` (and ``float64``) **control on the same
    input**, supplied by the shared ``dtype_policy`` fixture in
    ``tests/test_layers/conftest.py``. "fp16 is noisy" must never be able to
    masquerade as "the NaN bug is detected".
3.  Sizes are realistic (``_N = 512`` keys), not toy, for anything that feeds a
    softmax.

The specific defect under guard is the arithmetic form
``logits + (1 - keep) * MASK_BIAS_VALUE`` evaluated in the COMPUTE dtype: under
``mixed_float16`` that is ``0 * -inf = NaN`` at every **unmasked** position, i.e. it
destroys the whole batch even when nothing is masked. ``apply_attention_mask`` must
be structurally immune (``ops.where`` in ``mask_dtype(...)``), and these tests were
observed FAILING against exactly that arithmetic form injected into the helper body.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.attention.common import (
    MASK_BIAS_VALUE,
    apply_attention_mask,
    compute_attention_scale,
    mask_dtype,
    validate_head_divisibility,
)

# ---------------------------------------------------------------------
# Shapes. `_N` is deliberately large: a masked-softmax hazard that only bites at
# a realistic key count is exactly the kind this suite exists to catch.
# ---------------------------------------------------------------------

_B, _H, _N = 2, 4, 512


def _compute_dtype() -> str:
    """Compute dtype implied by the CURRENT global policy (``float16`` under mixed)."""
    return keras.mixed_precision.global_policy().compute_dtype


def _logits(seed: int = 0) -> np.ndarray:
    """A ``(B, H, N, N)`` block of float32 logits, deterministic."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_B, _H, _N, _N)).astype("float32") * 4.0


def _as_compute(x: np.ndarray):
    """Round-trip ``x`` through the compute dtype and return (tensor, float32 view).

    Returning the round-tripped float32 view is what lets the expected-value
    reference be built WITHOUT any fp16 rounding slack: the reference is computed
    from the very same bits the helper receives.
    """
    cd = _compute_dtype()
    x_c = x.astype(cd)
    return ops.convert_to_tensor(x_c), x_c.astype("float32")


# ---------------------------------------------------------------------
# Anti-vacuity: prove the hazard is real before testing the defence against it.
# ---------------------------------------------------------------------


class TestMaskBiasValueIsTheActualHazard:
    """The constant must be finite in float32 and ``-inf`` in float16.

    If either of these ever stops holding, every fp16 test below silently stops
    testing anything and must be re-derived.
    """

    def test_fp16_cast_of_mask_bias_value_is_negative_infinity(self):
        with np.errstate(over="ignore"):
            in_fp16 = np.float16(MASK_BIAS_VALUE)
        assert np.isneginf(in_fp16), (
            f"anti-vacuity FAILED: np.float16({MASK_BIAS_VALUE}) == {in_fp16}, not -inf. "
            "The fp16 tests in this file are then vacuous and must be redesigned."
        )

    def test_mask_bias_value_is_finite_in_float32_and_float64(self):
        assert np.isfinite(np.float32(MASK_BIAS_VALUE))
        assert np.isfinite(np.float64(MASK_BIAS_VALUE))

    def test_zero_times_fp16_bias_is_nan(self):
        """The exact arithmetic that the helper must never perform."""
        with np.errstate(over="ignore", invalid="ignore"):
            product = np.float16(0.0) * np.float16(MASK_BIAS_VALUE)
        assert np.isnan(product), (
            "anti-vacuity FAILED: `0 * float16(MASK_BIAS_VALUE)` is not NaN, so the "
            "`(1 - keep) * MASK_BIAS_VALUE` failure mode is not reproducible here."
        )


# ---------------------------------------------------------------------
# The dtype floor.
# ---------------------------------------------------------------------


class TestMaskDtype:
    """``mask_dtype`` is a pure selector: at least float32, float64 honored."""

    @pytest.mark.parametrize(
        "compute, expected",
        [
            ("float16", "float32"),
            ("bfloat16", "float32"),
            ("float32", "float32"),
            ("float64", "float64"),
        ],
    )
    def test_floor(self, compute, expected):
        assert mask_dtype(compute) == expected


# ---------------------------------------------------------------------
# apply_attention_mask
# ---------------------------------------------------------------------


class TestApplyAttentionMaskDtypeContract:
    """Return dtype: ``mask_dtype`` by default, the requested dtype on request."""

    def test_default_out_dtype_is_the_mask_dtype(self, dtype_policy):
        logits, _ = _as_compute(_logits())
        keep = ops.ones((_B, 1, 1, _N), dtype="float32")

        out = apply_attention_mask(logits, keep)

        expected = mask_dtype(_compute_dtype())
        assert keras.backend.standardize_dtype(out.dtype) == expected, (
            f"default out_dtype must be mask_dtype({_compute_dtype()!r}) == {expected!r}, "
            f"got {out.dtype!r}. Returning the compute dtype by default would re-create "
            "`-inf` in fp16 at the caller."
        )

    def test_explicit_out_dtype_is_honored(self, dtype_policy):
        cd = _compute_dtype()
        logits, logits_f32 = _as_compute(_logits())
        keep = ops.ones((_B, 1, 1, _N), dtype="float32")

        out = apply_attention_mask(logits, keep, out_dtype=cd)

        assert keras.backend.standardize_dtype(out.dtype) == cd
        # An all-ones keep masks nothing, so the cast back is lossless-by-construction
        # in every dtype: the result must be the input, exactly.
        np.testing.assert_allclose(
            ops.convert_to_numpy(out).astype("float32"), logits_f32, rtol=0, atol=0
        )


class TestApplyAttentionMaskFiniteness:
    """The core guard: no NaN / no Inf under ANY global dtype policy."""

    def test_all_ones_keep_masks_nothing_and_stays_finite(self, dtype_policy):
        """The hopfield catastrophe: an all-ones mask NaN-ed the entire batch.

        Under the arithmetic form this is ``0 * -inf`` at EVERY position.
        """
        logits, logits_f32 = _as_compute(_logits(seed=1))
        keep = ops.ones((_B, 1, 1, _N), dtype="float32")

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        assert np.isnan(out).sum() == 0, f"{np.isnan(out).sum()}/{out.size} NaN"
        assert np.isinf(out).sum() == 0, f"{np.isinf(out).sum()}/{out.size} Inf"
        np.testing.assert_allclose(out, logits_f32, rtol=0, atol=0)

    def test_partial_mask_matches_the_reference_bias(self, dtype_policy):
        logits, logits_f32 = _as_compute(_logits(seed=2))
        keep_np = np.ones((_B, 1, 1, _N), dtype="float32")
        keep_np[:, :, :, _N // 2:] = 0.0                       # mask the second half
        keep = ops.convert_to_tensor(keep_np)

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        assert np.all(np.isfinite(out)), (
            f"{(~np.isfinite(out)).sum()}/{out.size} non-finite entries under policy "
            f"{dtype_policy!r}"
        )
        ref = np.where(keep_np > 0.0, logits_f32, logits_f32 + MASK_BIAS_VALUE)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

    def test_causal_mask_stays_finite(self, dtype_policy):
        logits, _ = _as_compute(_logits(seed=3))
        causal = np.tril(np.ones((_N, _N), dtype="float32"))[None, None, :, :]
        keep = ops.convert_to_tensor(causal)

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        assert np.all(np.isfinite(out))

    def test_fully_masked_row_is_finite(self, dtype_policy):
        """A row that keeps NOTHING must still be finite, not NaN.

        Two independent defences make this true and this test passes under either, so
        it is deliberately a pure finiteness/normalization assertion: the ``out_dtype``
        default keeps the row in ``mask_dtype`` where ``-1e9`` is finite, and (since
        step 4c) the ``rescue_axis`` default means the all-``-1e9`` row is not even
        FORMED. What the rescue specifically changes — the row's VALUES — is asserted
        in :class:`TestApplyAttentionMaskRescueAxis`, not here.
        """
        logits, _ = _as_compute(_logits(seed=4))
        keep_np = np.ones((_B, 1, _N, _N), dtype="float32")
        keep_np[:, :, 0, :] = 0.0                              # query row 0 keeps nothing
        keep = ops.convert_to_tensor(keep_np)

        biased = apply_attention_mask(logits, keep)
        probs = ops.convert_to_numpy(ops.softmax(biased, axis=-1)).astype("float32")

        assert np.all(np.isfinite(probs)), (
            f"{(~np.isfinite(probs)).sum()} non-finite softmax entries for a "
            f"fully-masked row under policy {dtype_policy!r}"
        )
        np.testing.assert_allclose(probs[:, :, 0, :].sum(axis=-1), 1.0, rtol=1e-5,
                                   atol=1e-5)

    def test_masked_positions_receive_no_softmax_mass(self, dtype_policy):
        """Semantic guard: the bias must actually suppress the masked positions."""
        logits, _ = _as_compute(_logits(seed=5))
        keep_np = np.ones((_B, 1, 1, _N), dtype="float32")
        keep_np[:, :, :, ::2] = 0.0                            # mask every other key
        keep = ops.convert_to_tensor(keep_np)

        probs = ops.convert_to_numpy(
            ops.softmax(apply_attention_mask(logits, keep), axis=-1)
        ).astype("float32")

        assert np.all(np.isfinite(probs))
        masked_mass = probs[:, :, :, ::2].sum()
        assert masked_mass == pytest.approx(0.0, abs=1e-6), (
            f"masked positions carry {masked_mass} probability mass; the additive bias "
            "did not suppress them"
        )
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5)


class TestApplyAttentionMaskKeepDtypes:
    """`keep` arrives as bool, int32, float32 or the compute dtype across the sites."""

    @pytest.mark.parametrize("keep_dtype", ["bool", "int32", "float32", "compute"])
    def test_keep_dtype_is_accepted_and_gives_the_same_answer(
        self, dtype_policy, keep_dtype
    ):
        logits, logits_f32 = _as_compute(_logits(seed=6))
        keep_np = np.ones((_B, 1, 1, _N), dtype="float32")
        keep_np[:, :, :, -8:] = 0.0

        resolved = _compute_dtype() if keep_dtype == "compute" else keep_dtype
        keep = ops.convert_to_tensor(keep_np.astype(resolved))

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        assert np.all(np.isfinite(out))
        ref = np.where(keep_np > 0.0, logits_f32, logits_f32 + MASK_BIAS_VALUE)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


class TestApplyAttentionMaskBroadcasting:
    """`keep` may arrive at any rank BROADCASTABLE against `logits`; result is logits-shaped.

    The helper does no ``expand_dims`` / ``reshape`` / ``repeat`` of its own (invariant
    I3: each of the ten call sites owns — and keeps — its own broadcast and cast order).
    So the ranks exercised here are the *already-broadcastable* ones a site produces,
    and the last test pins the consequence: a raw ``(B, N)`` padding mask is NOT
    broadcastable against ``(B, H, N, N)`` and raises loudly rather than mis-masking.
    """

    @pytest.mark.parametrize("rank", [2, 3, 4])
    def test_broadcast_from_rank(self, dtype_policy, rank):
        logits, logits_f32 = _as_compute(_logits(seed=7))

        base = np.ones((_N,), dtype="float32")
        base[-16:] = 0.0
        if rank == 2:
            keep_np = np.tril(np.ones((_N, _N), dtype="float32"))      # (N, N) causal
        elif rank == 3:
            keep_np = base[None, None, :]                              # (1, 1, N)
        else:
            keep_np = np.broadcast_to(base, (_B, _N)).copy()[:, None, None, :]

        broadcast = np.broadcast_to(keep_np, (_B, _H, _N, _N))

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, ops.convert_to_tensor(keep_np))
        ).astype("float32")

        assert out.shape == (_B, _H, _N, _N)
        assert np.all(np.isfinite(out)), (
            f"non-finite result for rank-{rank} keep under policy {dtype_policy!r}"
        )
        ref = np.where(broadcast > 0.0, logits_f32, logits_f32 + MASK_BIAS_VALUE)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)

    def test_unbroadcastable_keep_raises_rather_than_mis_masking(self, dtype_policy):
        """A raw ``(B, N)`` mask must FAIL, not silently align on the wrong axes.

        This is the invariant-I3 contract made observable: the site is responsible for
        expanding ``(B, N)`` to ``(B, 1, 1, N)``, and forgetting to do so is a loud
        error rather than a mask applied to the query axis.
        """
        logits, _ = _as_compute(_logits(seed=71))
        keep = ops.convert_to_tensor(np.ones((_B, _N), dtype="float32"))

        with pytest.raises(Exception, match="(?i)broadcast"):
            ops.convert_to_numpy(apply_attention_mask(logits, keep))


class TestApplyAttentionMaskDoesNoPolarityInference:
    """Invariant I2: `keep` is taken EXACTLY as given — no guessing, ever."""

    def test_inverting_the_predicate_inverts_the_masking(self, dtype_policy):
        logits, logits_f32 = _as_compute(_logits(seed=8))
        keep_np = np.zeros((_B, 1, 1, _N), dtype="float32")
        keep_np[:, :, :, : _N // 4] = 1.0
        keep = ops.convert_to_tensor(keep_np)
        inverted = ops.convert_to_tensor(1.0 - keep_np)

        a = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")
        b = ops.convert_to_numpy(apply_attention_mask(logits, inverted)).astype("float32")

        np.testing.assert_allclose(
            a, np.where(keep_np > 0.0, logits_f32, logits_f32 + MASK_BIAS_VALUE),
            rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            b, np.where(keep_np > 0.0, logits_f32 + MASK_BIAS_VALUE, logits_f32),
            rtol=1e-6, atol=1e-6,
        )

    def test_boolean_predicate_is_used_directly(self, dtype_policy):
        """`capsule_routing_attention.py` passes a raw boolean — no `> 0` at the site."""
        logits, logits_f32 = _as_compute(_logits(seed=9))
        keep_np = np.zeros((_B, 1, 1, _N), dtype="bool")
        keep_np[:, :, :, 3:] = True

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, ops.convert_to_tensor(keep_np))
        ).astype("float32")

        ref = np.where(keep_np, logits_f32, logits_f32 + MASK_BIAS_VALUE)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


class TestApplyAttentionMaskRescueAxis:
    """The fully-masked-row rescue, which is ON BY DEFAULT (step 4c, decisions.md D-009).

    ``rescue_axis`` hoists into this helper the predicate-level rescue that
    ``capsule_routing_attention.py`` used to own alone (D-006): a slice of ``keep``
    that keeps NOTHING is treated as keeping EVERYTHING, so an all-``MASK_BIAS_VALUE``
    row — ``-inf`` in fp16, hence ``softmax(all -inf) = 0/0 = NaN`` — is never FORMED.

    Step 4b landed it as an opt-in defaulting to ``None`` so un-migrated sites stayed
    byte-identical; step 4c flipped that on the user's direction ("I care about
    correctness, not backwards compatibility"). Two properties matter as much as the
    rescue itself and are asserted here:

    *   the DEFAULT rescues — a call that passes no ``rescue_axis`` at all gets the
        finite, uniform answer rather than an all-``-inf`` row
        (``test_the_default_rescues``), and ``None`` is now the explicit OPT-OUT
        (``test_the_explicit_opt_out_does_not_rescue``);
    *   the axis is **honored, not hardcoded** — ``test_a_non_last_rescue_axis_is_honored``
        rescues along ``-2`` and would pass just as happily if the implementation
        ignored the argument and always used ``-1``... which is exactly why it asserts
        the ``-1`` answer is DIFFERENT.
    """

    @staticmethod
    def _keep_with_a_dead_row():
        """``(B, 1, N, N)`` float32 keep mask whose query row 0 keeps nothing."""
        keep_np = np.ones((_B, 1, _N, _N), dtype="float32")
        keep_np[:, :, 0, :] = 0.0
        return keep_np

    def test_the_default_rescues(self, dtype_policy):
        """The step-4c flip: NO ``rescue_axis`` argument still rescues the dead row.

        Observed FAILING on the step-4b code (where the default was ``None``): the
        dead row came back as ``logits + MASK_BIAS_VALUE`` everywhere, so the
        ``assert_allclose`` against the UNBIASED logits fired with a ~1e9 mismatch.
        """
        logits, logits_f32 = _as_compute(_logits(seed=11))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        assert np.all(np.isfinite(out)), (
            f"{(~np.isfinite(out)).sum()}/{out.size} non-finite entries under "
            f"{dtype_policy!r}"
        )
        np.testing.assert_allclose(
            out[:, :, 0, :],
            np.broadcast_to(logits_f32[:, :, 0, :], out[:, :, 0, :].shape),
            rtol=0,
            atol=0,
            err_msg=(
                "a default-argument call did NOT rescue the fully-masked row: the "
                "rescue must be the DEFAULT, with `rescue_axis=None` the opt-out"
            ),
        )

    def test_the_default_is_byte_identical_to_the_explicit_rescue_axis(
        self, dtype_policy
    ):
        """The default is exactly ``rescue_axis=-1``, not merely 'close to' it."""
        logits, _ = _as_compute(_logits(seed=12))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        default = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        explicit = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        )
        np.testing.assert_array_equal(default, explicit)

    def test_the_explicit_opt_out_does_not_rescue(self, dtype_policy):
        """``rescue_axis=None`` is the documented OPT-OUT and must still work.

        A site that genuinely wants a fully-masked row to stay fully biased (so a bad
        mask is a loud NaN rather than finite garbage) must be able to ask for it.
        This is the same assertion the pre-4c ``test_the_default_does_not_rescue``
        made, re-aimed at the explicit opt-out instead of at the default.
        """
        logits, logits_f32 = _as_compute(_logits(seed=11))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=None)
        ).astype("float32")

        dead = out[:, :, 0, :]
        expected = np.broadcast_to(
            logits_f32[:, :, 0, :] + MASK_BIAS_VALUE, dead.shape
        )
        np.testing.assert_allclose(dead, expected, rtol=1e-6, atol=1e-6)

    def test_a_rescued_row_receives_no_bias_at_all(self, dtype_policy):
        logits, logits_f32 = _as_compute(_logits(seed=13))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        ).astype("float32")

        assert np.all(np.isfinite(out)), (
            f"{(~np.isfinite(out)).sum()}/{out.size} non-finite entries under "
            f"{dtype_policy!r} — the rescue did not run"
        )
        np.testing.assert_allclose(
            out[:, :, 0, :],
            np.broadcast_to(logits_f32[:, :, 0, :], out[:, :, 0, :].shape),
            rtol=0,
            atol=0,
        )

    def test_a_rescued_row_softmaxes_finitely_in_every_dtype(self, dtype_policy):
        """The whole point: this is the assertion that is NaN without the rescue."""
        logits, _ = _as_compute(_logits(seed=14))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        probs = ops.convert_to_numpy(
            ops.softmax(
                apply_attention_mask(
                    logits,
                    keep,
                    out_dtype=_compute_dtype(),      # the UNSAFE cast-back, on purpose
                    rescue_axis=-1,
                ),
                axis=-1,
            )
        ).astype("float32")

        assert np.all(np.isfinite(probs)), (
            f"{(~np.isfinite(probs)).sum()} non-finite softmax entries for a "
            f"fully-masked row under {dtype_policy!r}, WITH the rescue and the "
            "compute-dtype cast-back that every batch-A site uses"
        )
        np.testing.assert_allclose(
            probs[:, :, 0, :].sum(axis=-1), 1.0, rtol=1e-3, atol=1e-3
        )

    def test_rows_that_keep_something_are_untouched_by_the_rescue(self, dtype_policy):
        """The rescue must be invisible to every non-degenerate row (bit-exact)."""
        logits, _ = _as_compute(_logits(seed=15))
        causal = np.tril(np.ones((_N, _N), dtype="float32"))[None, None, :, :]
        keep = ops.convert_to_tensor(causal)     # row 0 keeps exactly one key: alive

        without = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=None)
        )
        with_rescue = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        np.testing.assert_array_equal(without, with_rescue)

    def test_a_non_last_rescue_axis_is_honored(self, dtype_policy):
        """The axis is caller-supplied, not inferred — proved by using a different one.

        ``keep`` here has a dead COLUMN (key 0 kept by no query) and a dead ROW
        (query 0 keeps no key). ``rescue_axis=-1`` must revive the row and leave the
        column masked; ``rescue_axis=-2`` must do the opposite. An implementation
        that ignored the argument would make these two identical.
        """
        logits, logits_f32 = _as_compute(_logits(seed=16))
        keep_np = np.ones((_B, 1, _N, _N), dtype="float32")
        keep_np[:, :, 0, :] = 0.0                 # dead query row
        keep_np[:, :, :, 1] = 0.0                 # dead key column
        keep = ops.convert_to_tensor(keep_np)

        by_row = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        ).astype("float32")
        by_col = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-2)
        ).astype("float32")

        # Row rescue: query row 0 is unbiased everywhere.
        np.testing.assert_allclose(
            by_row[:, :, 0, :],
            np.broadcast_to(logits_f32[:, :, 0, :], by_row[:, :, 0, :].shape),
            rtol=0, atol=0,
        )
        # ... while the dead column stays masked for every OTHER query.
        assert np.all(by_row[:, :, 1:, 1] < MASK_BIAS_VALUE / 2.0)

        # Column rescue: the mirror image.
        np.testing.assert_allclose(
            by_col[:, :, :, 1],
            np.broadcast_to(logits_f32[:, :, :, 1], by_col[:, :, :, 1].shape),
            rtol=0, atol=0,
        )
        assert np.all(by_col[:, :, 0, 2:] < MASK_BIAS_VALUE / 2.0)

        assert not np.allclose(by_row, by_col), (
            "anti-vacuity FAILED: rescue_axis=-1 and rescue_axis=-2 gave the same "
            "answer, so this test cannot tell whether the axis is honored at all"
        )

    def test_the_rescue_operates_on_keeps_own_shape_before_broadcasting(
        self, dtype_policy
    ):
        """A rank-4 ``(B, 1, 1, N)`` key mask that keeps nothing revives entirely."""
        logits, logits_f32 = _as_compute(_logits(seed=17))
        keep = ops.zeros((_B, 1, 1, _N), dtype="float32")

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        ).astype("float32")

        np.testing.assert_allclose(out, logits_f32, rtol=0, atol=0)

    @pytest.mark.parametrize("keep_dtype", ["bool", "int32", "float32"])
    def test_the_rescue_accepts_every_keep_dtype(self, dtype_policy, keep_dtype):
        logits, logits_f32 = _as_compute(_logits(seed=18))
        keep_np = self._keep_with_a_dead_row()
        keep = ops.convert_to_tensor(keep_np.astype(keep_dtype))

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        ).astype("float32")

        np.testing.assert_allclose(
            out[:, :, 0, :],
            np.broadcast_to(logits_f32[:, :, 0, :], out[:, :, 0, :].shape),
            rtol=0, atol=0,
        )

    def test_the_rescue_is_graph_safe_under_tf_function_and_jit(self, dtype_policy):
        """A rescue that only works eagerly is not a fix.

        The rescue must contain NO data-dependent Python branch. This traces it under
        ``tf.function`` with ``jit_compile=True`` and requires the compiled answer to
        match the eager one exactly.
        """
        import tensorflow as tf

        logits, _ = _as_compute(_logits(seed=19))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        eager = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        ).astype("float32")

        @tf.function(jit_compile=True)
        def traced(x, k):
            return apply_attention_mask(x, k, rescue_axis=-1)

        compiled = ops.convert_to_numpy(traced(logits, keep)).astype("float32")

        assert np.all(np.isfinite(compiled))
        np.testing.assert_allclose(compiled, eager, rtol=1e-6, atol=1e-6)


class TestApplyAttentionMaskRejectsASizeOneRescueAxis:
    """A ``keep`` that is CONSTANT along the softmax-reduced axis is rejected (D-017).

    This class exists because of a MEASURED silent-un-masking report: with a
    ``(B, H, Q, 1)`` keep predicate the rescue
    ``kept OR NOT any(kept, axis=-1, keepdims=True)`` is all-``True``, so the mask has
    no effect whatsoever and a ``GatedAttention`` forward came back **bit-identical
    (maxdiff 0.0) to a no-op all-ones mask**.

    The diagnosis matters more than the symptom, and it is asserted rather than
    asserted-about in :meth:`test_a_mask_constant_along_the_reduced_axis_cannot_mask`:
    softmax is invariant to a CONSTANT shift of the row it reduces over, so an
    additive bias that is constant along that axis is mathematically a NO-OP. The
    all-``True`` rescue is not "the identity by accident" — it is the correct rescue
    (every row keeps nothing, therefore every row keeps everything). What is wrong is
    the CALLER: such a mask can never mask anything, in any implementation, so it is
    a shape-error-free mistake (a query-axis mask passed where a key-axis mask was
    expected, or a mask that was never broadcast). Since step 10 the helper says so.
    """

    def test_a_size_one_rescue_axis_raises(self):
        """The C1 guard: a keep whose rescue axis is statically 1 is a hard error."""
        logits = ops.convert_to_tensor(
            np.arange(6, dtype="float32").reshape(2, 3)
        )
        keep = ops.convert_to_tensor(np.array([[1.0], [0.0]], dtype="float32"))

        with pytest.raises(ValueError, match=r"rescue_axis.*size 1"):
            apply_attention_mask(logits, keep)

    def test_a_rank_4_query_only_mask_raises(self):
        """The exact layout the reviewer measured: ``(B, H, Q, 1)``."""
        logits, _ = _as_compute(_logits(seed=41))
        keep_np = np.ones((_B, _H, _N, 1), dtype="float32")
        keep_np[:, :, 5:, :] = 0.0

        with pytest.raises(ValueError, match=r"rescue_axis"):
            apply_attention_mask(logits, ops.convert_to_tensor(keep_np))

    def test_a_mask_constant_along_the_reduced_axis_cannot_mask(self):
        """WHY the raise is correct, measured — not asserted as taste.

        Softmax is shift-invariant, so biasing a whole reduced row by a constant is a
        mathematical no-op. With a REPRESENTABLE constant that is exactly what
        happens (identical probabilities). With ``MASK_BIAS_VALUE`` the row's own
        differences are annihilated by float32 rounding — the ulp at ``1e9`` is 64 —
        so the pre-step-10 output was not "masking", it was a uniform distribution,
        i.e. finite garbage either way.
        """
        row = np.array([[0.0, 1.0, 2.0]], dtype="float32")
        base = ops.convert_to_numpy(
            keras.activations.softmax(ops.convert_to_tensor(row))
        )
        shifted = ops.convert_to_numpy(
            keras.activations.softmax(ops.convert_to_tensor(row - 30.0))
        )
        saturated = ops.convert_to_numpy(
            keras.activations.softmax(
                ops.convert_to_tensor(row + np.float32(MASK_BIAS_VALUE))
            )
        )

        np.testing.assert_allclose(shifted, base, rtol=0, atol=0)
        np.testing.assert_allclose(
            saturated, np.full_like(saturated, 1.0 / 3.0), rtol=0, atol=1e-6
        )

    def test_the_explicit_opt_out_still_accepts_a_size_one_axis(self):
        """``rescue_axis=None`` names NO softmax axis, so the helper cannot judge.

        This is what keeps ``ring_attention.py``'s per-tile opt-out (D-011) working:
        the guard fires only when the caller has told the helper which axis its
        softmax reduces over.
        """
        logits = ops.convert_to_tensor(
            np.arange(6, dtype="float32").reshape(2, 3)
        )
        keep = ops.convert_to_tensor(np.array([[1.0], [0.0]], dtype="float32"))

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=None)
        )
        np.testing.assert_allclose(out[0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            out[1], np.full(3, 3.0 + MASK_BIAS_VALUE, dtype="float32"), rtol=1e-6
        )

    def test_a_size_one_axis_that_is_not_the_rescue_axis_is_accepted(self):
        """Only the NAMED axis is checked — broadcasting over heads stays legal."""
        logits, _ = _as_compute(_logits(seed=42))
        keep = ops.convert_to_tensor(np.ones((_B, 1, _N, _N), dtype="float32"))

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        assert np.all(np.isfinite(out))

    def test_a_genuinely_length_one_axis_is_not_rejected(self):
        """A single-token sequence is ordinary input, not a caller mistake.

        The rejected condition is BROADCAST (size 1 in ``keep`` while ``logits`` is
        longer), not size. A first draft of this guard tested size alone and broke
        ``tests/test_layers/test_transformers/test_text_decoder.py::
        TestTextDecoder::test_single_token_input``, where a one-token decode builds a
        ``(1, 1, 1)`` causal mask against ``(1, H, 1, 1)`` logits — caught by the
        consumer gate, not by this file, which is why the case is pinned here now.
        """
        logits = ops.convert_to_tensor(np.zeros((1, 4, 1, 1), dtype="float32"))
        keep = ops.convert_to_tensor(np.ones((1, 1, 1, 1), dtype="float32"))

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, np.zeros((1, 4, 1, 1)), rtol=0, atol=0)

    def test_a_masked_single_key_is_rescued_rather_than_rejected(self):
        """The same shape with the single key MASKED: rescued, still not an error."""
        logits = ops.convert_to_tensor(np.full((1, 2, 1, 1), 3.0, dtype="float32"))
        keep = ops.convert_to_tensor(np.zeros((1, 1, 1, 1), dtype="float32"))

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        np.testing.assert_allclose(out, np.full((1, 2, 1, 1), 3.0), rtol=0, atol=0)

    def test_a_dynamic_rescue_axis_length_is_not_rejected(self):
        """A ``None`` (trace-time-unknown) axis length cannot be judged, so it passes.

        The guard is STATIC by construction: it must never become a data-dependent
        branch (D-008 forbids that — the rescue has to stay graph-safe).
        """
        import tensorflow as tf

        @tf.function(
            input_signature=[
                tf.TensorSpec([2, None], tf.float32),
                tf.TensorSpec([2, None], tf.float32),
            ]
        )
        def traced(x, k):
            return apply_attention_mask(x, k)

        out = ops.convert_to_numpy(
            traced(
                tf.constant(np.arange(6, dtype="float32").reshape(2, 3)),
                tf.constant(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], "float32")),
            )
        )
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[1], [3.0, 4.0, 5.0], rtol=1e-6)


class TestApplyAttentionMaskAcceptsAMultiAxisRescueAxis:
    """A TUPLE ``rescue_axis`` is a legal softmax axis and must not be a ``TypeError``.

    ``keras.layers.Softmax`` accepts a tuple ``axis`` and ``ProbabilityOutput``
    forwards ``type_config`` verbatim (``activations/probability_output.py:180``), so
    the eight sites that DERIVE ``rescue_axis`` from their own ``probability_config``
    (D-017) can hand this function a tuple. Before this class existed, that reached the
    D-017 bounds check as a tuple and raised, verbatim::

        TypeError: '<=' not supported between instances of 'int' and 'tuple'

    from ``common.py``'s ``-len(keep_shape) <= rescue_axis`` — a config that had run
    before step 10 (the rescue axis was hard-coded ``-1`` then). A ``TypeError`` from an
    internal comparison is the one outcome that is definitely wrong: it names neither
    the parameter nor the cause.

    The shipped answer GENERALIZES rather than rejects, because the generalization is
    exact: a softmax over axes ``(1, 2)`` reduces over the JOINT block, so
    "a slice that keeps nothing keeps everything" means the joint block, and
    "the predicate is constant along the reduced axis" means constant along EVERY
    named axis. See D-018.
    """

    def test_a_tuple_rescue_axis_is_not_a_type_error(self):
        """RED before D-018: this raised ``TypeError`` from the bounds comparison."""
        logits = ops.convert_to_tensor(
            np.zeros((2, 3, 4, 5), dtype="float32")
        )
        keep = ops.convert_to_tensor(np.ones((2, 3, 4, 5), dtype="float32"))

        out = apply_attention_mask(logits, keep, rescue_axis=(1, 2))
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_a_tuple_axis_rescues_over_the_joint_block(self):
        """The rescue is JOINT, not per-axis: a dead ``(axis1, axis2)`` block revives.

        Batch item 0's whole ``(3, 4)`` block at key 0 is masked, so it keeps nothing
        under a softmax reducing over ``(1, 2)`` and must receive NO bias. Batch item 1
        keeps one position in that block, so the rest of it stays masked.
        """
        logits = ops.convert_to_tensor(np.zeros((2, 3, 4, 2), dtype="float32"))
        keep_np = np.ones((2, 3, 4, 2), dtype="float32")
        keep_np[0, :, :, 0] = 0.0            # batch 0, key 0: the whole block is dead
        keep_np[1, :, :, 0] = 0.0
        keep_np[1, 0, 0, 0] = 1.0            # batch 1, key 0: one survivor
        keep = ops.convert_to_tensor(keep_np)

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=(1, 2))
        )

        # Rescued: the dead block gets no bias at all.
        np.testing.assert_allclose(out[0, :, :, 0], np.zeros((3, 4)), rtol=0, atol=0)
        # Not rescued: only the single survivor is unbiased.
        assert out[1, 0, 0, 0] == 0.0
        assert np.all(out[1, 1:, :, 0] == np.float32(MASK_BIAS_VALUE))
        # The unreduced key axis is untouched in both.
        np.testing.assert_allclose(out[:, :, :, 1], np.zeros((2, 3, 4)), rtol=0, atol=0)

    def test_a_per_axis_rescue_would_over_unmask_and_is_not_what_ships(self):
        """Anti-vacuity for the joint rescue: ``rescue_axis=-1`` gives a DIFFERENT answer.

        Without this control the joint-rescue test could pass for the wrong reason.
        Here ONE ``(head, query)`` position masks both of its keys while the joint
        ``(1, 2)`` block still keeps plenty elsewhere. The joint rescue therefore
        leaves that position masked — correctly, the block is not dead — whereas a
        naive last-axis rescue un-masks it. That is the silent OVER-permission the
        joint form avoids.
        """
        logits = ops.convert_to_tensor(np.zeros((2, 3, 4, 2), dtype="float32"))
        keep_np = np.ones((2, 3, 4, 2), dtype="float32")
        keep_np[1, 1, 2, :] = 0.0
        keep = ops.convert_to_tensor(keep_np)

        joint = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=(1, 2))
        )
        per_axis = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        )
        assert not np.allclose(joint, per_axis)

    def test_a_list_axis_is_accepted_like_a_tuple(self):
        """``[1, 2]`` and ``(1, 2)`` are the same softmax axis and must agree."""
        logits = ops.convert_to_tensor(np.zeros((2, 3, 4, 2), dtype="float32"))
        keep_np = np.ones((2, 3, 4, 2), dtype="float32")
        keep_np[0, :, :, 0] = 0.0
        keep = ops.convert_to_tensor(keep_np)

        as_tuple = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=(1, 2))
        )
        as_list = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=[1, 2])
        )
        np.testing.assert_allclose(as_list, as_tuple, rtol=0, atol=0)

    def test_a_single_element_tuple_matches_the_bare_int(self):
        """``(-1,)`` must trace to the same numbers as ``-1``."""
        logits, _ = _as_compute(_logits(seed=77))
        keep_np = np.ones((_B, _H, _N, _N), dtype="float32")
        keep_np[:, :, :, 300:] = 0.0
        keep = ops.convert_to_tensor(keep_np)

        bare = ops.convert_to_numpy(apply_attention_mask(logits, keep, rescue_axis=-1))
        tup = ops.convert_to_numpy(apply_attention_mask(logits, keep, rescue_axis=(-1,)))
        np.testing.assert_allclose(tup, bare, rtol=0, atol=0)

    def test_broadcast_across_every_named_axis_raises_a_named_value_error(self):
        """The D-017 rejection generalizes: constant along ALL reduced axes is a no-op.

        This is the exact consumer layout the review measured — a rank-2 key-padding
        mask expanded to ``(B, 1, 1, N)`` at a site whose softmax reduces over
        ``(1, 2)``. The bias is then constant across the whole reduced block, so it
        cannot mask; the error must NAME ``rescue_axis``, not be a ``TypeError``.
        """
        logits = ops.convert_to_tensor(np.zeros((2, 3, 4, 5), dtype="float32"))
        keep = ops.convert_to_tensor(np.ones((2, 1, 1, 5), dtype="float32"))

        with pytest.raises(ValueError, match=r"rescue_axis=\(1, 2\).*size 1"):
            apply_attention_mask(logits, keep, rescue_axis=(1, 2))

    def test_broadcast_across_only_some_named_axes_is_accepted(self):
        """Size 1 along ONE reduced axis still masks, so it must NOT be rejected.

        ``keep`` is size 1 over heads (axis 1) but varies over queries (axis 2), so the
        bias is not constant over the joint block and the mask does real work. A guard
        that fired per-axis would reject a legitimate caller here.
        """
        logits = ops.convert_to_tensor(np.zeros((2, 3, 4, 5), dtype="float32"))
        keep_np = np.ones((2, 1, 4, 5), dtype="float32")
        keep_np[:, :, 2:, :] = 0.0
        keep = ops.convert_to_tensor(keep_np)

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=(1, 2))
        )
        assert np.all(out[:, :, :2, :] == 0.0)
        assert np.all(out[:, :, 2:, :] == np.float32(MASK_BIAS_VALUE))

    def test_a_non_integer_axis_is_a_named_value_error(self):
        """A junk axis must be diagnosed by name, never by an internal ``TypeError``."""
        logits = ops.convert_to_tensor(np.zeros((2, 3), dtype="float32"))
        keep = ops.convert_to_tensor(np.ones((2, 3), dtype="float32"))

        with pytest.raises(ValueError, match=r"rescue_axis"):
            apply_attention_mask(logits, keep, rescue_axis="last")
        with pytest.raises(ValueError, match=r"rescue_axis"):
            apply_attention_mask(logits, keep, rescue_axis=())

    def test_the_consumer_path_no_longer_type_errors(self):
        """End to end at a DERIVING site (D-017 (b)), which is how the crash was found.

        ``GatedAttention(probability_config={"axis": (1, 2)})`` forwards the tuple to
        ``ProbabilityOutput`` AND derives ``rescue_axis`` from it. With a rank-2 mask
        (expanded to ``(B, 1, 1, N)``) the predicate is constant over the reduced
        block, so the right answer is the named ``ValueError`` — previously a
        ``TypeError`` from a comparison inside ``common.py``.
        """
        from dl_techniques.layers.attention.gated_attention import GatedAttention

        layer = GatedAttention(
            dim=32, num_heads=4, probability_config={"axis": (1, 2)}
        )
        x = ops.convert_to_tensor(
            np.random.default_rng(3).standard_normal((2, 6, 32)).astype("float32")
        )
        mask = ops.convert_to_tensor(np.ones((2, 6), dtype="float32"))

        with pytest.raises(ValueError, match=r"rescue_axis"):
            layer(x, attention_mask=mask)


class TestApplyAttentionMaskAssumesABinaryKeep:
    """``keep`` is a BINARY predicate; a fractional value means KEEP (W3).

    The replaced arithmetic form ``logits + (1 - m) * MASK_BIAS_VALUE`` interpolated:
    ``m = 0.5`` produced ``-5e8``, which is "masked" for every practical purpose.
    ``ops.where(cast(keep) > 0, ...)`` does not interpolate, so the same input is now
    FULL KEEP. MEASURED at ``GatedAttention`` in float32 with a masked half of 0.5:
    the new output equals the ALL-ONES output exactly, the old one equalled the HARD
    mask exactly, and the two differ by 1.81 — while the binary control is
    bit-identical. That is a deliberate, documented semantics change and NOT a
    soft-mask feature; these tests pin it so it cannot drift back silently.
    """

    def test_a_fractional_keep_is_full_keep(self):
        logits, logits_f32 = _as_compute(_logits(seed=43))
        keep = ops.convert_to_tensor(
            np.full((_B, 1, _N, _N), 0.5, dtype="float32")
        )

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype(
            "float32"
        )
        np.testing.assert_allclose(
            out,
            np.broadcast_to(logits_f32, out.shape),
            rtol=0,
            atol=0,
            err_msg=(
                "a fractional keep value must be FULL KEEP (unbiased); it must "
                "never be interpolated into a partial bias"
            ),
        )

    def test_only_exactly_zero_and_negative_values_mask(self):
        logits = ops.convert_to_tensor(
            np.zeros((1, 4), dtype="float32")
        )
        keep = ops.convert_to_tensor(
            np.array([[1.0, 0.5, 0.0, -1.0]], dtype="float32")
        )

        out = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=None)
        )
        np.testing.assert_allclose(
            out, [[0.0, 0.0, MASK_BIAS_VALUE, MASK_BIAS_VALUE]], rtol=1e-6
        )

    def test_the_binary_precondition_is_documented(self):
        """The precondition must be stated where a caller reads it, not only here."""
        doc = apply_attention_mask.__doc__ or ""
        assert "binary" in doc.lower(), (
            "`apply_attention_mask` must state its BINARY-keep precondition in its "
            "own docstring — a graded mask is silently treated as full keep"
        )


# ---------------------------------------------------------------------
# The two pre-existing exports — regression cover only (they had no test file).
# ---------------------------------------------------------------------


class TestTheAdopterCountsAreMechanical:
    """The counts in ``common.py``'s docstring are DERIVED, not remembered.

    History: the same two numbers drifted three ways at once — the module docstring
    said "Seven of the ten adopters", the live ``D-009`` anchor said "Six of them",
    and the plan's own decision log said "seven", while the source said **eight**.
    The commit that claimed to fix the stale count reintroduced it. Prose cannot be
    kept in lockstep by hand, so the count now has exactly one home and this test
    re-derives it from the source on every run.

    The predicates are the ones written verbatim in the docstring, so a reader can
    reproduce them with ``grep`` in one command.
    """

    @staticmethod
    def _attention_sources():
        import pathlib

        import dl_techniques.layers.attention.common as common_mod

        pkg = pathlib.Path(common_mod.__file__).parent
        return sorted(p for p in pkg.glob("*.py") if p.name != "__init__.py")

    def _counts(self):
        import re

        call = re.compile(r"(^|[^._0-9A-Za-z])apply_attention_mask\(")
        adopters, derivers = [], []
        for path in self._attention_sources():
            if path.name == "common.py":
                continue
            text = path.read_text(encoding="utf-8")
            if any(call.search(line) for line in text.splitlines()):
                adopters.append(path.name)
            if "rescue_axis=(self.probability_config" in text:
                derivers.append(path.name)
        return adopters, derivers

    def test_there_are_exactly_ten_adopters_and_eight_derive_the_axis(self):
        adopters, derivers = self._counts()
        assert len(adopters) == 10, f"adopters drifted: {adopters}"
        assert len(derivers) == 8, f"derivers drifted: {derivers}"
        assert set(derivers) <= set(adopters)

    def test_the_two_non_deriving_adopters_are_the_documented_ones(self):
        """``capsule_routing`` PINS its axis; ``ring`` OPTS OUT. Nothing else."""
        adopters, derivers = self._counts()
        assert set(adopters) - set(derivers) == {
            "capsule_routing_attention.py",
            "ring_attention.py",
        }

    def test_the_module_docstring_states_the_derived_numbers(self):
        """The prose must agree with the source, or this fails loudly.

        This is the anti-drift half: the docstring is the single home of the count,
        so it is the thing that has to be checked against the mechanical derivation.
        """
        import dl_techniques.layers.attention.common as common_mod

        adopters, derivers = self._counts()
        doc = common_mod.__doc__
        words = {6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN"}
        assert f"**{words[len(derivers)]} of the {words[len(adopters)]} adopters**" in doc, (
            "common.py's module docstring no longer states the mechanically-derived "
            f"counts ({len(derivers)} of {len(adopters)}). Update the "
            "'MECHANICAL' paragraph — and only that paragraph."
        )

    def test_no_live_decision_anchor_restates_a_count(self):
        """A stale number inside a ``# DECISION`` anchor is the worst place for one.

        The D-009 anchor used to say "Six of them build that softmax from a
        user-supplied ``probability_config``" while the truth was eight. Anchors now
        cite the docstring paragraph instead of repeating a number.
        """
        import pathlib
        import re

        import dl_techniques.layers.attention.common as common_mod

        source = pathlib.Path(common_mod.__file__).read_text(encoding="utf-8")
        anchor_region = source.split("def apply_attention_mask(")[0]
        comment_lines = [
            ln for ln in anchor_region.splitlines() if ln.lstrip().startswith("#")
        ]
        offenders = [
            ln
            for ln in comment_lines
            if re.search(
                r"\b(six|seven|eight|nine|ten)\b\s+(of\s+them|modules|adopters|sites)",
                ln,
                re.IGNORECASE,
            )
        ]
        assert not offenders, (
            "a DECISION anchor restates an adopter count; cite the docstring's "
            f"MECHANICAL paragraph instead: {offenders}"
        )


class TestPreExistingExports:
    """`common.py` had no test module; these pin the other two helpers."""

    def test_validate_head_divisibility_accepts_and_rejects(self):
        assert validate_head_divisibility(64, 8) is None
        with pytest.raises(ValueError, match="must be divisible by"):
            validate_head_divisibility(65, 8)

    def test_validate_head_divisibility_message_names_the_caller_arguments(self):
        with pytest.raises(ValueError, match=r"hidden_size \(65\).*num_kv_heads \(8\)"):
            validate_head_divisibility(
                65, 8, dim_name="hidden_size", num_heads_name="num_kv_heads"
            )

    @pytest.mark.parametrize("head_dim", [1, 32, 64, 128])
    def test_compute_attention_scale_is_a_python_float(self, head_dim):
        scale = compute_attention_scale(head_dim)
        assert isinstance(scale, float)
        assert scale == pytest.approx(1.0 / np.sqrt(head_dim), rel=1e-12)
