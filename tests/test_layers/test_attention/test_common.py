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
        """A row that keeps NOTHING must still be finite (uniform garbage, not NaN).

        The helper's DEFAULT (stay in ``mask_dtype``) is what makes this true: the
        row is all ``-1e9``, which is finite in float32, and its softmax is a finite
        uniform distribution rather than ``NaN``.
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
    """The opt-in fully-masked-row rescue (step 4b, decisions.md D-008).

    ``rescue_axis`` hoists into this helper the predicate-level rescue that
    ``capsule_routing_attention.py`` used to own alone (D-006): a slice of ``keep``
    that keeps NOTHING is treated as keeping EVERYTHING, so an all-``MASK_BIAS_VALUE``
    row — ``-inf`` in fp16, hence ``softmax(all -inf) = 0/0 = NaN`` — is never FORMED.

    Two properties matter as much as the rescue itself and are asserted here:

    *   it is **opt-in**: ``rescue_axis=None`` (the default) must be byte-identical to
        the pre-4b behavior, so no un-migrated call site changes silently;
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

    def test_the_default_does_not_rescue(self, dtype_policy):
        """Opt-in, part 1: without ``rescue_axis`` the dead row is still fully biased."""
        logits, logits_f32 = _as_compute(_logits(seed=11))
        keep_np = self._keep_with_a_dead_row()
        keep = ops.convert_to_tensor(keep_np)

        out = ops.convert_to_numpy(apply_attention_mask(logits, keep)).astype("float32")

        dead = out[:, :, 0, :]
        expected = np.broadcast_to(
            logits_f32[:, :, 0, :] + MASK_BIAS_VALUE, dead.shape
        )
        np.testing.assert_allclose(dead, expected, rtol=1e-6, atol=1e-6)

    def test_the_default_is_byte_identical_to_the_explicit_no_rescue(self, dtype_policy):
        """Opt-in, part 2: ``None`` is not merely 'close to' the old behavior."""
        logits, _ = _as_compute(_logits(seed=12))
        keep = ops.convert_to_tensor(self._keep_with_a_dead_row())

        default = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        explicit = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=None)
        )
        np.testing.assert_array_equal(default, explicit)

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

        without = ops.convert_to_numpy(apply_attention_mask(logits, keep))
        with_rescue = ops.convert_to_numpy(
            apply_attention_mask(logits, keep, rescue_axis=-1)
        )
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


# ---------------------------------------------------------------------
# The two pre-existing exports — regression cover only (they had no test file).
# ---------------------------------------------------------------------


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
