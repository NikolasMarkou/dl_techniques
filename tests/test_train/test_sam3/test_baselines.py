"""Guards for ``src/train/sam3/baselines.py`` -- the non-model baseline family.

Three things are being protected, in descending order of how quietly they fail.

1. **The instrument.** ``score_prior`` reproduces ``evaluate_sam3``'s box-IoU
   arithmetic, and a wrong Hungarian assignment is INVISIBLE: it returns a
   finite, plausible number. So the guard is a value known IN ADVANCE -- the
   GT-oracle arm must read EXACTLY 1.0 -- and that guard is RED-proven here
   against a DEAD-COMPONENT injection (``TestTheOracleGuardIsRedProven``):
   replacing ``loss.matcher`` with a no-op that assigns every query to target 0
   makes ``_assert_oracle_reads_exactly_one`` fire.
2. **The seed separation.** The prior is fitted on the TRAIN split and scored
   on the VAL split. A seed mismatch leaks the scoring split into the fit and
   nothing raises. ``TestTheFitNeverReadsTheScoringSplit`` asserts it
   STRUCTURALLY, with a recording fake in place of ``build_sam3_dataset``: the
   fit path is proven never to construct the ``seed + VAL_SEED_OFFSET`` split
   at all, rather than proven to give a plausible answer.
3. **The fit's free choices.** ``n_init`` and ``random_state`` were MEASURED in
   a prior plan to move the prior by up to 0.016 IoU and to flip a per-seed
   sign. ``TestKmeansPriorPinsItsFreeChoices`` compares against a ``KMeans``
   constructed with those two values spelled out at the call site, so a change
   of default fails by value.

Device: CPU-cheap. Only ``TestTheFamilyEndToEnd`` and the oracle guards build a
model, and they build the ``tiny`` variant with a 4-sample split.
"""

from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from train.sam3 import baselines
from train.sam3.baselines import (
    DEGENERATE_ARM,
    GRID_ARM,
    ORACLE_ARM,
    KMEANS_N_INIT,
    VAL_SEED_OFFSET,
    build_context,
    build_split_dataset,
    degenerate_prior,
    evaluate_family,
    family_max,
    fit_kmeans_prior,
    fixed_grid_prior,
    gt_oracle_predictor,
    kmeans_arm,
    pool_train_gt,
    score_prior,
    tile_to_queries,
)

#: A `tiny`-variant split small enough to build and score in a test.
TINY_SPLIT: Dict[str, Any] = {
    "variant": "tiny",
    "batch_size": 2,
    "num_train_samples": 4,
    "num_val_samples": 4,
    "max_instances": 3,
    "max_per_category": 2,
}

#: `k` for the tiny end-to-end family: the 4-sample train pool holds only a
#: handful of boxes, and `fit_kmeans_prior` refuses `k > len(pool)` on purpose.
TINY_K: int = 2


def _assert_oracle_reads_exactly_one(value: float) -> None:
    """THE liveness assertion, in one place so the RED proof can name it.

    Both the live guard and ``TestTheOracleGuardIsRedProven`` call this, so the
    mutation record can say WHICH assertion fired rather than "the suite went
    red".

    Args:
        value: The GT-oracle arm's box IoU.

    Returns:
        None.

    Raises:
        AssertionError: If the oracle is not exactly 1.0.
    """
    assert value == pytest.approx(1.0, abs=1e-6), (
        f"GT-oracle arm read {value!r}, not 1.0. The oracle contains every "
        f"valid GT box of its own image and the matcher may pick any query "
        f"per target, so anything but 1.0 means the IoU instrument is broken "
        f"and no other number it produces may be believed.")


@pytest.fixture(scope="module")
def tiny_context() -> Tuple[Any, Any]:
    """A seed-pinned `tiny` model and its compiled loss."""
    keras.utils.set_random_seed(1234)
    return build_context(seed=7, split=TINY_SPLIT)


@pytest.fixture(scope="module")
def tiny_val_dataset(tiny_context: Tuple[Any, Any]) -> Any:
    """The `tiny` SCORING split -- seed ``7 + VAL_SEED_OFFSET``."""
    model, _loss = tiny_context
    return build_split_dataset(model, seed=7, train=False, split=TINY_SPLIT)


class TestTheFixedGridPrior:
    """Shape, determinism and a hand-computed value oracle."""

    def test_shape_is_side_squared_by_four(self) -> None:
        assert fixed_grid_prior(side=5).shape == (25, 4)
        assert fixed_grid_prior(side=3).shape == (9, 4)

    def test_dtype_is_float32(self) -> None:
        assert fixed_grid_prior().dtype == np.float32

    def test_it_is_deterministic(self) -> None:
        assert np.array_equal(fixed_grid_prior(), fixed_grid_prior())

    def test_two_centers_hand_computed(self) -> None:
        """Row-major over ``(cy, cx)``: row ``r*5+c`` is center ``((c+.5)/5,
        (r+.5)/5)``.

        Hand-computed, not read off the implementation: row 0 is the top-left
        cell ``(0.1, 0.1)`` and row 6 is ``r=1, c=1``, i.e. ``(0.3, 0.3)``.
        """
        grid = fixed_grid_prior(side=5, box_size=0.2)
        np.testing.assert_allclose(grid[0], [0.1, 0.1, 0.2, 0.2], atol=1e-7)
        np.testing.assert_allclose(grid[6], [0.3, 0.3, 0.2, 0.2], atol=1e-7)
        # The last cell, for the corner the two above do not pin.
        np.testing.assert_allclose(grid[24], [0.9, 0.9, 0.2, 0.2], atol=1e-7)

    def test_every_center_is_inside_the_unit_square(self) -> None:
        grid = fixed_grid_prior()
        assert float(grid[:, :2].min()) > 0.0
        assert float(grid[:, :2].max()) < 1.0

    def test_box_size_is_honoured(self) -> None:
        grid = fixed_grid_prior(side=2, box_size=0.5)
        np.testing.assert_allclose(grid[:, 2:], 0.5, atol=1e-7)

    @pytest.mark.parametrize("side,box_size", [(0, 0.2), (-1, 0.2)])
    def test_it_refuses_a_degenerate_side(self, side: int,
                                          box_size: float) -> None:
        with pytest.raises(ValueError, match="side must be"):
            fixed_grid_prior(side=side, box_size=box_size)

    @pytest.mark.parametrize("box_size", [0.0, -0.1, 1.5])
    def test_it_refuses_a_degenerate_box_size(self, box_size: float) -> None:
        with pytest.raises(ValueError, match="box_size must be"):
            fixed_grid_prior(box_size=box_size)


class TestTileToQueries:
    """The prior -> query-slot expansion every image-independent arm goes through."""

    def test_it_cycles_the_prior(self) -> None:
        prior = np.asarray([[0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.3, 0.3]],
                           dtype=np.float32)
        tiled = tile_to_queries(prior, 5)
        assert tiled.shape == (5, 4)
        for index in range(5):
            np.testing.assert_allclose(tiled[index], prior[index % 2])

    def test_it_truncates_a_prior_larger_than_q(self) -> None:
        tiled = tile_to_queries(fixed_grid_prior(), 8)
        assert tiled.shape == (8, 4)
        np.testing.assert_allclose(tiled, fixed_grid_prior()[:8])

    def test_it_refuses_an_empty_prior(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            tile_to_queries(np.zeros((0, 4), np.float32), 4)

    def test_it_refuses_a_non_box_prior(self) -> None:
        with pytest.raises(ValueError, match=r"\(P, 4\)"):
            tile_to_queries(np.zeros((3, 5), np.float32), 4)


class TestKmeansPriorPinsItsFreeChoices:
    """`n_init` and `random_state` are pinned, and the fit is deterministic."""

    @staticmethod
    def _pool(seed: int = 0, n: int = 200) -> np.ndarray:
        rng = np.random.default_rng(seed)
        centers = np.asarray([[0.2, 0.2, 0.1, 0.1], [0.8, 0.3, 0.2, 0.2],
                              [0.5, 0.7, 0.3, 0.15]], dtype=np.float32)
        picks = centers[rng.integers(0, len(centers), n)]
        return (picks + rng.normal(0.0, 0.01, picks.shape)).astype(np.float32)

    def test_it_is_deterministic_at_a_pinned_seed(self) -> None:
        pool = self._pool()
        first = fit_kmeans_prior(pool, k=3, seed=11)
        second = fit_kmeans_prior(pool, k=3, seed=11)
        assert np.array_equal(first, second)

    def test_it_matches_kmeans_with_both_choices_spelled_out(self) -> None:
        """The value oracle for the pinned free choices.

        Built from ``sklearn`` directly with ``n_init`` and ``random_state``
        written at the call site, so changing either default in the module
        fails HERE, by value -- which is the failure mode the prior plan
        measured at up to 0.016 IoU and one flipped sign.
        """
        from sklearn.cluster import KMeans

        pool = self._pool()
        expected = KMeans(n_clusters=3, n_init=KMEANS_N_INIT,
                          random_state=11).fit(pool).cluster_centers_
        np.testing.assert_allclose(
            fit_kmeans_prior(pool, k=3, seed=11), expected, atol=1e-6)

    def test_the_random_state_is_the_seed_argument(self) -> None:
        """A different seed is a different `random_state`, hence a fit that
        may differ -- and the module must not silently pin one seed for all."""
        from sklearn.cluster import KMeans

        pool = self._pool()
        for seed in (0, 3, 11):
            expected = KMeans(n_clusters=4, n_init=KMEANS_N_INIT,
                              random_state=seed).fit(pool).cluster_centers_
            np.testing.assert_allclose(
                fit_kmeans_prior(pool, k=4, seed=seed), expected, atol=1e-6)

    def test_shape_is_k_by_four(self) -> None:
        assert fit_kmeans_prior(self._pool(), k=6, seed=1).shape == (6, 4)

    def test_it_refuses_k_larger_than_the_pool(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            fit_kmeans_prior(self._pool(n=5), k=8, seed=1)

    def test_it_refuses_a_degenerate_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            fit_kmeans_prior(self._pool(), k=0, seed=1)


class TestTheFitNeverReadsTheScoringSplit:
    """Seed separation, asserted BY CONSTRUCTION rather than by plausibility."""

    @staticmethod
    def _recorder(monkeypatch: pytest.MonkeyPatch) -> List[int]:
        """Replace ``build_sam3_dataset`` with a fake that records its seed."""
        seen: List[int] = []

        def fake(**kwargs: Any) -> List[Any]:
            seen.append(int(kwargs["seed"]))
            return []

        monkeypatch.setattr(baselines, "build_sam3_dataset", fake)
        return seen

    def test_fit_kmeans_prior_constructs_no_dataset_at_all(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The strongest form: the fit cannot reach ANY split.

        ``build_sam3_dataset`` is replaced by a sentinel that raises. The fit
        succeeding proves it is pure in its array argument, so the only split
        it can ever see is the one ``pool_train_gt`` hands it.
        """
        def explode(**_kwargs: Any) -> None:
            raise AssertionError(
                "fit_kmeans_prior reached build_sam3_dataset; it must be pure "
                "in its array argument.")

        monkeypatch.setattr(baselines, "build_sam3_dataset", explode)
        pool = np.random.default_rng(0).uniform(
            0.1, 0.9, (50, 4)).astype(np.float32)
        assert fit_kmeans_prior(pool, k=4, seed=1).shape == (4, 4)

    def test_pool_train_gt_reads_the_fit_seed_and_only_it(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._recorder(monkeypatch)
        pool = pool_train_gt(seed=5, model=object(), split=TINY_SPLIT)
        assert seen == [5], (
            f"pool_train_gt constructed splits {seen}; it must construct the "
            f"TRAIN split (seed=5) and nothing else. "
            f"{5 + VAL_SEED_OFFSET} would be the SCORING split.")
        assert 5 + VAL_SEED_OFFSET not in seen
        assert pool.shape == (0, 4)

    def test_the_two_seeds_differ_by_the_offset_and_are_both_logged(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both seeds must appear in the OUTPUT, not only in a comment."""
        self._recorder(monkeypatch)
        lines: List[str] = []

        class _Recorder:
            @staticmethod
            def info(fmt: str, *args: Any) -> None:
                lines.append(fmt % args)

        monkeypatch.setattr(baselines, "logger", _Recorder)
        pool_train_gt(seed=5, model=object(), split=TINY_SPLIT)
        assert any("seed=5" in line and "seed=10005" in line
                   for line in lines), lines
        assert VAL_SEED_OFFSET == 10_000

    def test_build_split_dataset_offsets_only_the_val_split(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._recorder(monkeypatch)
        build_split_dataset(object(), seed=5, train=True, split=TINY_SPLIT)
        build_split_dataset(object(), seed=5, train=False, split=TINY_SPLIT)
        assert seen == [5, 5 + VAL_SEED_OFFSET]


class TestTheDegenerateAndOraclePriors:
    """The two arms whose values are known before the instrument runs."""

    def test_the_degenerate_prior_is_one_centered_box(self) -> None:
        np.testing.assert_allclose(degenerate_prior(), [[0.5, 0.5, 0.2, 0.2]])

    def test_the_oracle_tiles_each_image_s_own_gt(self) -> None:
        boxes = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 100.0
        valid = np.ones((2, 3), dtype=np.float32)
        tiled = gt_oracle_predictor(boxes, valid, num_queries=5)
        assert tiled.shape == (2, 5, 4)
        for image in range(2):
            for slot in range(5):
                np.testing.assert_allclose(tiled[image, slot],
                                           boxes[image, slot % 3])

    def test_the_oracle_is_image_dependent(self) -> None:
        """A liveness check on the liveness arm: it must NOT be constant."""
        boxes = np.random.default_rng(0).uniform(
            0.2, 0.8, (4, 3, 4)).astype(np.float32)
        tiled = gt_oracle_predictor(boxes, np.ones((4, 3), np.float32), 5)
        assert float(tiled.std(axis=0).mean()) > 1e-3


class TestScorePriorLiveness:
    """``score_prior`` on values known IN ADVANCE, on a real `tiny` split."""

    def test_the_gt_oracle_arm_reads_exactly_one(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        _assert_oracle_reads_exactly_one(
            score_prior(gt_oracle_predictor, model, loss, tiny_val_dataset))

    def test_the_split_has_matched_pairs_so_the_oracle_is_not_vacuous(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """Guards the guard: 1.0 out of ZERO matched pairs would be 0.0, but a
        near-empty split would still make the oracle uninformative."""
        model, loss = tiny_context
        _iou, matched = baselines._score(
            gt_oracle_predictor, model, loss, tiny_val_dataset)
        assert matched > 0.0

    def test_the_degenerate_arm_reads_below_five_percent(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        value = score_prior(degenerate_prior(), model, loss, tiny_val_dataset)
        assert 0.0 < value < 0.05, (
            f"the one-box arm read {value:.4f}; an instrument that scores the "
            f"crudest possible non-reader near the grid's ~0.35 is broken.")

    def test_the_grid_beats_the_degenerate_arm(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        grid = score_prior(fixed_grid_prior(), model, loss, tiny_val_dataset)
        one = score_prior(degenerate_prior(), model, loss, tiny_val_dataset)
        assert grid > one

    def test_every_arm_shares_one_matched_denominator(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """``is_matched`` depends only on ``target_valid``, so no arm can win
        by matching fewer pairs."""
        model, loss = tiny_context
        totals = {
            baselines._score(prior, model, loss, tiny_val_dataset)[1]
            for prior in (gt_oracle_predictor, degenerate_prior(),
                          fixed_grid_prior())}
        assert len(totals) == 1, totals


class TestTheOracleGuardIsRedProven:
    """SC-D: a DEAD-COMPONENT injection must make the oracle guard FAIL.

    TWO mutations, because the first one alone is too easy to survive.

    * **M1 -- the no-op ASSIGNMENT, denominator kept live.** ``loss.matcher``
      still computes the real ``is_matched`` but returns a constant assignment
      (every query mapped to target slot 0). Same shapes, same dtypes, same
      denominator, no raise -- and a FINITE, PLAUSIBLE reading. This is the
      injection that matters: a wrong Hungarian assignment is invisible to
      every other check in this file.
    * **M2 -- the fully dead matcher.** Assignment AND ``is_matched`` are
      constants. Recorded separately because it reads ``nan``, i.e. it is
      caught by arithmetic rather than by the guard's threshold, which is
      exactly why it cannot stand in for M1.
    """

    @staticmethod
    def _no_op_assignment(loss: Any) -> Any:
        """M1: keep the real ``is_matched``, kill only the assignment."""
        live = loss.matcher

        def mutant(pred_logits: Any, pred_boxes: Any, target_boxes: Any,
                   target_valid: Any) -> Any:
            from keras import ops

            _assignment, is_matched = live(pred_logits, pred_boxes,
                                           target_boxes, target_valid)
            return ops.zeros_like(_assignment), is_matched

        return mutant

    def test_m1_a_no_op_assignment_makes_the_oracle_assertion_fire(
            self, tiny_context: Tuple[Any, Any], tiny_val_dataset: Any,
            monkeypatch: pytest.MonkeyPatch) -> None:
        model, loss = tiny_context
        monkeypatch.setattr(loss, "matcher", self._no_op_assignment(loss))
        value = score_prior(gt_oracle_predictor, model, loss,
                            tiny_val_dataset)
        assert np.isfinite(value) and 0.0 <= value < 1.0, (
            f"M1 must produce a PLAUSIBLE reading, not an obviously broken "
            f"one; got {value!r}. A mutation caught by arithmetic does not "
            f"prove the guard is what catches it.")
        with pytest.raises(AssertionError, match="not 1.0"):
            _assert_oracle_reads_exactly_one(value)

    def test_m2_a_fully_dead_matcher_makes_the_oracle_assertion_fire(
            self, tiny_context: Tuple[Any, Any], tiny_val_dataset: Any,
            monkeypatch: pytest.MonkeyPatch) -> None:
        from keras import ops

        model, loss = tiny_context

        def dead_matcher(pred_logits: Any, pred_boxes: Any,
                         target_boxes: Any, target_valid: Any) -> Any:
            del pred_logits, target_boxes, target_valid
            shape = (ops.shape(pred_boxes)[0], ops.shape(pred_boxes)[1])
            return (ops.zeros(shape, dtype="int32"),
                    ops.ones(shape, dtype="float32"))

        monkeypatch.setattr(loss, "matcher", dead_matcher)
        value = score_prior(gt_oracle_predictor, model, loss,
                            tiny_val_dataset)
        with pytest.raises(AssertionError, match="not 1.0"):
            _assert_oracle_reads_exactly_one(value)

    def test_the_guard_is_green_again_once_the_matcher_is_restored(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """The control for the mutation above -- otherwise the RED reading
        could be an artifact of the fixture rather than of the injection."""
        model, loss = tiny_context
        _assert_oracle_reads_exactly_one(
            score_prior(gt_oracle_predictor, model, loss, tiny_val_dataset))


class TestTheFamilyEndToEnd:
    """``evaluate_family`` / ``family_max`` on a `tiny` split."""

    @pytest.fixture(scope="class")
    def tiny_family(self) -> Dict[str, Dict[str, float]]:
        keras.utils.set_random_seed(1234)
        return evaluate_family(seed=7, ks=(TINY_K,), split=TINY_SPLIT)

    def test_both_liveness_arms_are_present_and_read_their_known_values(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        _assert_oracle_reads_exactly_one(tiny_family[ORACLE_ARM]["box_iou"])
        assert tiny_family[DEGENERATE_ARM]["box_iou"] < 0.05

    def test_the_meta_block_carries_both_seeds(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        meta = tiny_family["_meta"]
        assert meta["fit_seed"] == 7.0
        assert meta["score_seed"] == 7.0 + VAL_SEED_OFFSET
        assert meta["kmeans_n_init"] == float(KMEANS_N_INIT)
        assert meta["kmeans_random_state"] == 7.0

    def test_the_grid_and_kmeans_arms_are_both_present(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        assert GRID_ARM in tiny_family
        assert kmeans_arm(TINY_K) in tiny_family

    def test_family_max_excludes_the_liveness_arms(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        """The bar is ``{grid} u {k-means}``. Including the ORACLE would make
        it 1.0 and nothing could ever clear it; including DEGENERATE would make
        it a floor rather than a bar."""
        name, value = family_max(tiny_family)
        assert name in (GRID_ARM, kmeans_arm(TINY_K))
        assert value < 1.0
        assert value == max(tiny_family[GRID_ARM]["box_iou"],
                            tiny_family[kmeans_arm(TINY_K)]["box_iou"])

    def test_family_max_refuses_an_empty_family(self) -> None:
        with pytest.raises(ValueError, match="not the family"):
            family_max({ORACLE_ARM: {"box_iou": 1.0, "matched": 1.0}})
