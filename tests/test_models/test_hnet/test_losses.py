"""Guards for H-Net's ratio (load-balancing) loss.

What this file pins
-------------------
1. **Parity with the step-3 oracle.** ``ratio_loss`` is measured against
   ``tests/test_layers/test_dynamic_chunking/hnet_reference_numpy.ratio_loss_reference``
   -- the float64 transcription of ``hnet/utils/train.py:13-40`` that was itself
   RED-proven in step 3. No second reference is written here; a loss with two rival
   references has no reference.
2. **The two divergences from upstream are deliberate and separately guarded**: the
   optional padding mask (upstream runs on a packed layout and has none), and the refusal
   of ``target_ratio <= 1`` (upstream divides by zero -- the oracle's own
   ``test_ratio_loss_rejects_N_equal_to_one_by_upstreams_own_precondition`` records that
   behaviour, so this port's ``raise`` is a documented difference rather than an
   accident).
3. **The sum over chunking levels is a SUM**, not a mean: adding a hierarchy level must
   add its own pressure rather than dilute the levels above it.

Tolerances
----------
Every ``assert_allclose`` passes ``rtol=0`` and derives its bound at the call site from
the reduction length and the dtype's own epsilon -- see :func:`parity_atol`.
``tests/numerics.reassociation_atol`` is deliberately NOT used: D-012 measured it
UNDER-counting against a float64 oracle, and this whole file compares against a float64
oracle.

All CPU, all tiny.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hnet.losses import (
    DEFAULT_TARGET_RATIO,
    ratio_loss,
    total_ratio_loss,
)

from ...test_layers.test_dynamic_chunking.hnet_reference_numpy import (
    ratio_loss_reference,
)


# ---------------------------------------------------------------------
# Tolerance derivation
# ---------------------------------------------------------------------


def parity_atol(n_elements: int, dtype: str, target_ratio: float) -> float:
    """Derive an absolute bound for one ``ratio_loss`` vs oracle comparison.

    The two implementations differ only in how they reduce: this port computes
    ``sum(x) / count`` while the oracle calls NumPy's pairwise ``mean``. So the whole
    disagreement is the summation-order error of two means, propagated through the loss.

    * A naive sum of ``n`` terms each bounded by ``1`` carries at most
      ``(n - 1) * eps * n`` absolute error, i.e. ``(n - 1) * eps`` once divided by ``n``.
      Take that as the error of each of ``true_ratio`` and ``average_prob``.
    * The loss is ``((1 - a)(1 - b) + a * b * (N - 1)) * N / (N - 1)``. On ``a, b`` in
      ``[0, 1]`` every partial derivative is bounded by ``max(1, N - 1) * N / (N - 1)``,
      and both means carry error, so multiply by two.

    :param n_elements: Number of positions each mean reduces over.
    :type n_elements: int
    :param dtype: The accumulation dtype, ``"float32"`` or ``"float64"``.
    :type dtype: str
    :param target_ratio: ``N``.
    :type target_ratio: float
    :returns: The absolute bound.
    :rtype: float
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    mean_error = (n_elements - 1) * eps
    sensitivity = max(1.0, target_ratio - 1.0) * target_ratio / (target_ratio - 1.0)
    return 2.0 * sensitivity * mean_error + 4.0 * eps


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def routing_draw(batch=3, length=17, seed=0):
    """A random routing record: ``p`` in ``[0, 1)``, the hard mask, and a padding mask.

    The hard predicate is ``p > 0.5`` (H2, corrected at step 3 -- ``argmax`` breaks the
    exact-0.5 tie toward "not a boundary"), matching ``RoutingModule``.
    """
    rng = np.random.default_rng(seed)
    p = rng.random((batch, length))
    p[:, 0] = 1.0                       # `dc.py:96` forces a boundary at position 0
    mask = p > 0.5
    two_class = np.stack([1.0 - p, p], axis=-1)
    return p, two_class, mask


# ---------------------------------------------------------------------
# 1. Parity with the step-3 oracle
# ---------------------------------------------------------------------


class TestRatioLossParity:
    """`ratio_loss` reproduces `train.py:13-40` as the step-3 oracle transcribes it."""

    @pytest.mark.parametrize("target_ratio", [2.0, 4.0, 6.0, 9.0])
    def test_float64_parity_against_the_oracle(self, target_ratio):
        """The float64 arm, where the oracle's own dtype is the comparison dtype."""
        p, two_class, mask = routing_draw()
        expected = ratio_loss_reference(two_class, mask, target_ratio=target_ratio)

        got = float(
            ratio_loss(
                keras.ops.convert_to_tensor(two_class, dtype="float64"),
                keras.ops.convert_to_tensor(mask),
                target_ratio=target_ratio,
            )
        )

        atol = parity_atol(p.size, "float64", target_ratio)
        assert atol > 0.0
        np.testing.assert_allclose(got, expected, rtol=0, atol=atol)

    def test_float32_parity_against_the_float64_oracle(self):
        """The float32 arm. The bound is float32's, derived the same way."""
        p, two_class, mask = routing_draw(seed=1)
        expected = ratio_loss_reference(two_class, mask, target_ratio=6.0)

        got = float(
            ratio_loss(
                keras.ops.convert_to_tensor(two_class.astype("float32")),
                keras.ops.convert_to_tensor(mask),
                target_ratio=6.0,
            )
        )

        atol = parity_atol(p.size, "float32", 6.0)
        np.testing.assert_allclose(got, expected, rtol=0, atol=atol)

    def test_the_two_class_and_one_class_forms_agree(self):
        """`boundary_prob` may arrive as `(B, L, 2)` or `(B, L)`; `[..., -1]` is `p`.

        The twin is implicit and strong: taking `[..., 0]` instead would read `1 - p`,
        which this equality forbids because the two differ here by construction.
        """
        p, two_class, mask = routing_draw(seed=2)

        from_two = float(ratio_loss(two_class, mask, target_ratio=6.0))
        from_one = float(ratio_loss(p, mask, target_ratio=6.0))

        assert from_two == from_one, "the last class must be p, exactly"

        wrong_class = float(ratio_loss(two_class[..., 0], mask, target_ratio=6.0))
        assert abs(wrong_class - from_two) > 1e-3, (
            "reading the FIRST class would be 1 - p; if that were equal here the "
            "equality above would prove nothing"
        )

    def test_the_hand_pin_travels_through_the_keras_implementation(self):
        """The step-3 hand pin (`1.07`) recomputed through this function.

        `p = [1.00, 0.20, 0.02, 0.50]`, `mask = [T, F, F, F]`, `N = 2`::

            true_ratio   = 1 / 4                       = 0.25
            average_prob = (1.00+0.20+0.02+0.50) / 4   = 0.43
            (1-0.25)*(1-0.43) + 0.25*0.43*1            = 0.4275 + 0.1075 = 0.535
            * 2 / 1                                    = 1.07
        """
        p = np.array([[1.00, 0.20, 0.02, 0.50]])
        mask = np.array([[True, False, False, False]])

        got = float(
            ratio_loss(
                keras.ops.convert_to_tensor(p, dtype="float64"), mask, target_ratio=2.0
            )
        )

        atol = parity_atol(4, "float64", 2.0)
        np.testing.assert_allclose(got, 1.07, rtol=0, atol=atol)


# ---------------------------------------------------------------------
# 2. The padding-mask divergence
# ---------------------------------------------------------------------


class TestPaddingMask:
    """`mask=None` is the reference; a mask restricts both means to valid positions."""

    def test_an_all_true_mask_reproduces_the_unmasked_value_exactly(self):
        """The "nothing changed" arm.

        Both paths are literally the same expression (`sum / count`), so the bound is an
        exact `0.0` and `0.0` is attained -- not a tolerance, an identity.
        """
        p, two_class, mask = routing_draw(seed=3)
        all_true = np.ones_like(mask)

        unmasked = float(ratio_loss(two_class, mask, target_ratio=6.0))
        masked = float(ratio_loss(two_class, mask, mask=all_true, target_ratio=6.0))

        assert masked == unmasked

    def test_a_restrictive_mask_moves_it(self):
        """The "something changed" twin the arm above owes.

        Masking off the second half of every row changes both means, so the loss must
        move by more than the float32 noise floor.
        """
        p, two_class, mask = routing_draw(seed=3)
        partial = np.ones_like(mask)
        partial[:, p.shape[1] // 2:] = False

        unmasked = float(ratio_loss(two_class, mask, target_ratio=6.0))
        masked = float(ratio_loss(two_class, mask, mask=partial, target_ratio=6.0))

        assert abs(masked - unmasked) > 1e-3, (
            f"a mask that hides half of every row must change the measured ratio; "
            f"got {masked} vs {unmasked}"
        )

    def test_the_mask_is_equivalent_to_deleting_the_masked_positions(self):
        """The masked loss on a padded row equals the unmasked loss on the short row.

        This is what "excluded from the loss" must mean, and it is the claim a mask that
        merely zeroed the contributions (without shrinking the denominator) would fail.
        """
        p = np.array([[0.9, 0.1, 0.8, 0.0, 0.0, 0.0]])
        mask = p > 0.5
        valid = np.array([[True, True, True, False, False, False]])

        padded = float(ratio_loss(p, mask, mask=valid, target_ratio=3.0))
        short = float(ratio_loss(p[:, :3], mask[:, :3], target_ratio=3.0))

        atol = parity_atol(6, "float64", 3.0)
        np.testing.assert_allclose(padded, short, rtol=0, atol=atol)


# ---------------------------------------------------------------------
# 3. The N > 1 precondition
# ---------------------------------------------------------------------


class TestTargetRatioPrecondition:
    """`N > 1` is upstream's precondition; this port enforces it instead of dividing."""

    @pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -3.0])
    def test_a_target_ratio_of_at_most_one_raises(self, bad):
        with pytest.raises(ValueError, match="target_ratio must be > 1"):
            ratio_loss(np.zeros((1, 4)), np.zeros((1, 4), dtype=bool), target_ratio=bad)

    def test_upstream_would_have_returned_a_non_finite_number(self):
        """The divergence, measured rather than asserted from memory.

        `hnet_reference_numpy.ratio_loss_reference` transcribes upstream faithfully and
        returns a non-finite value at `N = 1`. This port raising instead is therefore a
        documented difference, not a re-derivation.
        """
        p = np.array([[0.9, 0.1]])
        mask = p > 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            assert not np.isfinite(ratio_loss_reference(p, mask, target_ratio=1.0))

    def test_the_default_target_ratio_is_a_legal_one(self):
        assert DEFAULT_TARGET_RATIO > 1.0

    def test_the_default_target_ratio_is_pinned_to_its_documented_value(self):
        """The VALUE, against a literal written here rather than imported.

        MEASURED 2026-09-09: `DEFAULT_TARGET_RATIO 6.0 -> 3.0` -- a halving of the
        compression every default run optimises toward, since `HNet.__init__` turns
        `target_ratios=None` into `(DEFAULT_TARGET_RATIO,) * n_levels` -- survived all
        426 tests of `test_models/test_hnet/` AND all 239 of
        `test_train/test_hnet/ + test_byte_lm.py`. `> 1.0` above is a precondition
        check, not a pin: `3.0` satisfies it. This is the spelling half; the arm below
        is the behavioural half, and neither is sufficient alone.
        """
        assert DEFAULT_TARGET_RATIO == 6.0

    def test_the_default_target_ratio_is_OBSERVABLE_in_the_loss(self):
        """The EFFECT half: omitting `target_ratio` must weight by 6.0 exactly.

        `ratio_loss` is called WITHOUT the argument and compared against the literal
        `6.0`, so the default path itself is graded rather than the constant's
        spelling. The `3.0` arm is the DIFFER twin: without it, two readings of one
        mutated constant would agree perfectly.
        """
        p, two_class, mask = routing_draw(seed=71)
        default = float(ratio_loss(two_class, mask))
        pinned = float(ratio_loss(two_class, mask, target_ratio=6.0))
        halved = float(ratio_loss(two_class, mask, target_ratio=3.0))

        np.testing.assert_allclose(default, pinned, rtol=0, atol=0.0)
        assert abs(halved - default) > 1e-3, (
            f"the instrument cannot tell N=3 from N=6: {halved} vs {default}"
        )


# ---------------------------------------------------------------------
# 4. The sum over chunking levels
# ---------------------------------------------------------------------


class TestTotalRatioLoss:
    """One term per record, summed in record order."""

    @staticmethod
    def record(seed):
        p, two_class, mask = routing_draw(seed=seed)
        return {
            "boundary_prob": two_class,
            "boundary_mask": mask,
            "padding_mask": np.ones_like(mask),
        }

    def test_it_is_the_sum_of_the_per_level_terms(self):
        records = [self.record(10), self.record(11)]
        targets = (6.0, 3.0)

        total = float(total_ratio_loss(records, targets))
        parts = [
            float(ratio_loss(r["boundary_prob"], r["boundary_mask"],
                             mask=r["padding_mask"], target_ratio=n))
            for r, n in zip(records, targets)
        ]

        atol = parity_atol(records[0]["boundary_mask"].size, "float64", 3.0)
        np.testing.assert_allclose(total, sum(parts), rtol=0, atol=atol)

    def test_it_is_not_the_MEAN_of_the_per_level_terms(self):
        """The twin that separates a sum from an average.

        With two levels a mean would read exactly half the sum; the two terms here are
        both far from zero, so the separation is real rather than a rounding artefact.
        """
        records = [self.record(10), self.record(11)]
        total = float(total_ratio_loss(records, (6.0, 3.0)))
        assert total > 0.5, "both terms must be substantial for this twin to bite"

        one_level = float(total_ratio_loss(records[:1], (6.0,)))
        assert total > one_level, "adding a level must ADD pressure"

    def test_each_target_ratio_is_read_at_its_own_level(self):
        """Two IDENTICAL records with DIFFERENT targets must give two different terms.

        The obvious form of this test -- swapping the targets between two *different*
        records and asserting the total moves -- was MEASURED not to kill the mutation
        it exists for: an implementation that applies `target_ratios[0]` everywhere still
        reads (6, 6) forward and (2, 2) swapped, and those differ. Holding the record
        fixed removes that escape: under the mutation both terms would be
        `ratio_loss(r, 6)`, so the total would be `2 * ratio_loss(r, 6)` and this
        equality fails.
        """
        record = self.record(10)
        total = float(total_ratio_loss([record, record], (6.0, 2.0)))

        first = float(ratio_loss(record["boundary_prob"], record["boundary_mask"],
                                 mask=record["padding_mask"], target_ratio=6.0))
        second = float(ratio_loss(record["boundary_prob"], record["boundary_mask"],
                                  mask=record["padding_mask"], target_ratio=2.0))

        atol = parity_atol(record["boundary_mask"].size, "float64", 2.0)
        np.testing.assert_allclose(total, first + second, rtol=0, atol=atol)
        assert abs(first - second) > 1e-3, (
            "the two targets must give visibly different terms, or the equality above "
            "is satisfied by an implementation that ignores target_ratios[1]"
        )
        assert abs(total - 2.0 * first) > 1e-3

    def test_swapping_targets_between_different_records_also_moves_it(self):
        """A weaker, independent arm kept for coverage of the ordering itself."""
        records = [self.record(10), self.record(12)]
        forward = float(total_ratio_loss(records, (6.0, 2.0)))
        swapped = float(total_ratio_loss(records, (2.0, 6.0)))
        assert abs(forward - swapped) > 1e-3

    def test_a_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="one target ratio is needed per chunking"):
            total_ratio_loss([self.record(10)], (6.0, 3.0))

    def test_no_chunking_levels_is_exactly_zero(self):
        """A 1-stage-only (never chunking) layout contributes no ratio loss at all."""
        assert float(total_ratio_loss([], ())) == 0.0


# ---------------------------------------------------------------------
# 5. D-031: the record's `padding_mask` is actually PICKED UP
# ---------------------------------------------------------------------


class TestTotalRatioLossReadsThePaddingMask:
    """Surviving mutation S-5: `mask=record.get("padding_mask")` -> `mask=None` was
    GREEN across all 426 model tests.

    Not because the masked mean is untested -- ``TestPaddingMask`` above tests
    :func:`ratio_loss`'s ``mask`` argument thoroughly -- but because every routing
    record in every fixture in this repository carried an ALL-TRUE ``padding_mask``,
    so the `sum/count` expression gives the identical number whether the key is read
    or dropped. A mask that masks nothing cannot detect a mask that is ignored.

    Every fixture below therefore masks something.
    """

    @staticmethod
    def masked_record(seed, keep_fraction=0.5):
        """A record whose ``padding_mask`` genuinely removes trailing positions."""
        p, two_class, boundary = routing_draw(seed=seed)
        valid = np.ones_like(boundary)
        cut = int(p.shape[1] * keep_fraction)
        valid[:, cut:] = False
        assert not valid.all() and valid.any(), "the fixture must mask SOMETHING"
        return {
            "boundary_prob": two_class,
            "boundary_mask": boundary,
            "padding_mask": valid,
        }

    def test_a_restrictive_padding_mask_changes_the_total(self):
        """MAIN, and the direct kill for S-5.

        The same records, once through :func:`total_ratio_loss` (which must read
        ``padding_mask``) and once through :func:`ratio_loss` with ``mask=None``
        (which is what S-5 turned the former into). They must differ.
        """
        records = [self.masked_record(21), self.masked_record(22)]
        targets = (6.0, 3.0)

        with_mask = float(total_ratio_loss(records, targets))
        ignoring_mask = sum(
            float(ratio_loss(r["boundary_prob"], r["boundary_mask"], target_ratio=n))
            for r, n in zip(records, targets)
        )

        assert abs(with_mask - ignoring_mask) > 1e-3, (
            f"total_ratio_loss ignored the records' padding_mask: {with_mask} == "
            f"{ignoring_mask}"
        )

    def test_it_equals_the_per_record_masked_terms(self):
        """The POSITIVE statement of the same claim: the total is the masked sum.

        The arm above says "not the unmasked value"; this says "IS the masked value",
        so a third behaviour (dropping the record entirely, or masking with the wrong
        mask) fails one of the two.
        """
        records = [self.masked_record(21), self.masked_record(22)]
        targets = (6.0, 3.0)

        total = float(total_ratio_loss(records, targets))
        parts = sum(
            float(ratio_loss(r["boundary_prob"], r["boundary_mask"],
                             mask=r["padding_mask"], target_ratio=n))
            for r, n in zip(records, targets)
        )

        atol = parity_atol(records[0]["boundary_mask"].size, "float64", 3.0)
        np.testing.assert_allclose(total, parts, rtol=0, atol=atol)

    def test_a_record_with_no_padding_mask_key_still_works(self):
        """``.get`` is deliberate: a record without the key means "no mask".

        This is the behaviour S-5 made unconditional, and it must remain available --
        the guard above forbids it being the ONLY behaviour, not its existence.
        """
        record = self.masked_record(21)
        del record["padding_mask"]
        assert np.isfinite(float(total_ratio_loss([record], (6.0,))))
