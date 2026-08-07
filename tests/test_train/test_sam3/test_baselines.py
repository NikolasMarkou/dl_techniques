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

4. **The image-reading arm's liveness.** ``connected_components_predictor`` is
   NOT a family member -- it is a zero-parameter detector that reads the pixels
   and, on this generator, BEATS the trained model. A detector that quietly
   stopped reading the image would still return finite, plausible boxes, so
   ``_assert_the_detector_reads_the_pixels`` pins both an across-image spread
   and "it must beat the fixed grid", and ``TestTheDetectorGuardIsRedProven``
   fires it with a dead-component injection (a detector that ignores its
   ``image`` argument and returns the grid). ``TestTheDetectorIsNotAFamilyMember``
   pins that ``family_max`` never quotes it as the bar.

Device: CPU-cheap. Only ``TestTheFamilyEndToEnd`` and the oracle guards build a
model, and they build the ``tiny`` variant with a 4-sample split.
"""

from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from train.sam3 import baselines
from train.sam3.baselines import (
    CC_THRESHOLD,
    CONNECTED_COMPONENTS_ARM,
    DEGENERATE_ARM,
    GRID_ARM,
    ORACLE_ARM,
    KMEANS_N_INIT,
    PROMPT_KEYS,
    PROMPT_SWAP_SHIFTS,
    VAL_SEED_OFFSET,
    build_context,
    build_parser,
    build_split_dataset,
    connected_components_boxes,
    connected_components_predictor,
    degenerate_prior,
    evaluate_family,
    family_max,
    fit_kmeans_prior,
    fixed_grid_prior,
    gt_oracle_predictor,
    kmeans_arm,
    pool_train_gt,
    prompt_swap_retention,
    score_prior,
    swap_batch_prompts,
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


def _assert_the_detector_reads_the_pixels(spread: float, detector: float,
                                          grid: float) -> None:
    """THE liveness assertion for the image-reading arm, in ONE place.

    Both the live guard and ``TestTheDetectorGuardIsRedProven`` call it, so a
    mutation record can name WHICH assertion fired. A detector that has stopped
    reading the pixels still returns ``(B, Q, 4)`` of finite boxes and still
    scores a plausible number -- only these two facts separate it from a fixed
    prior.

    Args:
        spread: Across-IMAGE std of the emitted boxes.
        detector: The detector's box IoU on the scoring split.
        grid: The fixed 5x5 grid's box IoU on the SAME split.

    Returns:
        None.

    Raises:
        AssertionError: If the boxes are image-independent, or if an
            image-reading detector fails to beat a predictor that reads
            nothing.
    """
    assert spread > 1e-3, (
        f"the detector's across-image box spread is {spread:.3e}: it is "
        f"emitting (nearly) the same boxes for every image, i.e. it is a "
        f"fixed prior wearing a detector's name.")
    assert detector > grid, (
        f"the detector read {detector:.4f} against the fixed 5x5 grid's "
        f"{grid:.4f}. A predictor that looks at the pixels of a generator "
        f"which draws bright non-overlapping shapes on a dark canvas cannot "
        f"lose to one that looks at nothing; this reading means it is not "
        f"reading them.")


def _assert_the_prompt_swap_is_live(changed_fraction: float,
                                    rel_delta_logits: float) -> None:
    """THE liveness assertion for the prompt-swap diagnostic, in ONE place.

    A ``retained`` of 1.00 has TWO possible causes and they are opposite in
    meaning: the model's boxes really do ignore the text prompt, or the
    instrument never swapped anything. Only these two numbers separate them, so
    no retention figure may be quoted without them.

    Args:
        changed_fraction: Fraction of images whose ``token_ids`` actually
            differ after the rotation.
        rel_delta_logits: Max change in ``pred_logits`` under the swap, over
            that tensor's own std across the split.

    Returns:
        None.

    Raises:
        AssertionError: If no prompt changed, or if the swap moved no model
            output at all.
    """
    assert changed_fraction > 0.0, (
        f"the prompt swap changed the prompt on {changed_fraction:.4f} of "
        f"images: NOTHING was swapped, so any retention it reports is a "
        f"property of the instrument and not of the model.")
    assert rel_delta_logits > 0.0, (
        f"the swap moved pred_logits by a relative {rel_delta_logits:.3e}, "
        f"i.e. the swapped prompt reached NO model output at all. A retention "
        f"measured through a dead text path says nothing about the boxes.")


def _canvas_with_rectangles(size: int, boxes: List[Tuple[int, int, int, int]],
                            ) -> np.ndarray:
    """A synthetic ``[0, 255]`` canvas: dark background, bright rectangles.

    Args:
        size: Side of the square image.
        boxes: ``(row0, row1, col0, col1)`` half-open pixel extents.

    Returns:
        ``(size, size, 3)`` float32.
    """
    image = np.full((size, size, 3), 30.0, dtype=np.float32)
    for row0, row1, col0, col1 in boxes:
        image[row0:row1, col0:col1] = 200.0
    return image


def _detector_spread(model: Any, dataset: Any) -> float:
    """Across-IMAGE std of the detector's emitted boxes on ``dataset``.

    Goes through ``baselines.connected_components_predictor``, so a
    monkeypatched detector is measured rather than bypassed.

    Args:
        model: A built ``Sam3TrainingModel`` -- supplies ``num_queries``.
        dataset: The scoring split.

    Returns:
        Mean over ``(query, coordinate)`` of the std over the IMAGE axis,
        pooled over the whole split.
    """
    emitted = []
    for inputs, _y_true in dataset:
        images = np.asarray(inputs["image"], dtype=np.float32)
        emitted.append(baselines.connected_components_predictor(
            images, np.zeros((images.shape[0], 1, 4), np.float32),
            np.ones((images.shape[0], 1), np.float32),
            int(model.num_queries)))
    return float(np.concatenate(emitted).std(axis=0).mean())


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


class TestTheConnectedComponentsDetector:
    """The image-reading arm: geometry oracle, blindness to labels, edges.

    This arm exists because the pre-registered family had NO image-reading
    member, so "the model beats the family" could not distinguish "the model
    learned detection" from "this generator's box task is solvable by any rule
    that looks at the pixels". It reads ~0.94 where the trained arm reads
    ~0.84, which is the finding, not a bug.
    """

    def test_one_rectangle_gives_its_exact_normalized_box(self) -> None:
        """A hand-computed value oracle, not a self-consistency check."""
        image = _canvas_with_rectangles(64, [(10, 20, 5, 25)])
        boxes = connected_components_boxes(image, num_queries=1)
        assert boxes.shape == (1, 4)
        np.testing.assert_allclose(
            boxes[0], [15.0 / 64.0, 15.0 / 64.0, 20.0 / 64.0, 10.0 / 64.0],
            atol=1e-6)

    def test_components_are_ranked_by_pixel_count_descending(self) -> None:
        small = (2, 6, 2, 6)          # 16 px
        large = (30, 50, 30, 60)      # 600 px
        image = _canvas_with_rectangles(64, [small, large])
        boxes = connected_components_boxes(image, num_queries=2)
        # Row 0 must be the LARGE component: its width is 30/64, the small
        # one's is 4/64.
        assert boxes[0][2] == pytest.approx(30.0 / 64.0)
        assert boxes[1][2] == pytest.approx(4.0 / 64.0)

    def test_fewer_components_than_queries_are_tiled(self) -> None:
        image = _canvas_with_rectangles(32, [(4, 8, 4, 8)])
        boxes = connected_components_boxes(image, num_queries=5)
        assert boxes.shape == (5, 4)
        for slot in range(5):
            np.testing.assert_allclose(boxes[slot], boxes[0])

    def test_an_empty_canvas_falls_back_to_one_centered_box(self) -> None:
        """A zero-instance image is 25% of this generator's prompts."""
        image = np.full((16, 16, 3), 10.0, dtype=np.float32)
        boxes = connected_components_boxes(image, num_queries=3)
        assert boxes.shape == (3, 4)
        np.testing.assert_allclose(boxes[0], degenerate_prior()[0])

    def test_the_threshold_is_what_separates_shapes_from_the_canvas(
            self) -> None:
        """`data.py` draws shapes at 140..255 on a canvas at 13..64, so the
        cut must sit strictly between them."""
        assert 64.0 < CC_THRESHOLD < 140.0
        below = np.full((16, 16, 3), CC_THRESHOLD - 1.0, dtype=np.float32)
        assert connected_components_boxes(below, 1)[0][2] == pytest.approx(0.2)

    def test_a_non_image_input_raises(self) -> None:
        with pytest.raises(ValueError, match=r"must be \(S, S, C\)"):
            connected_components_boxes(np.zeros((8, 8), np.float32), 1)

    def test_the_predictor_is_blind_to_the_labels_it_is_scored_against(
            self) -> None:
        """Not a comment, a measurement: the targets are deleted, so garbage
        targets and real targets give bit-identical boxes."""
        images = np.stack([_canvas_with_rectangles(32, [(4, 12, 4, 12)]),
                           _canvas_with_rectangles(32, [(16, 28, 16, 30)])])
        real = connected_components_predictor(
            images, np.zeros((2, 3, 4), np.float32),
            np.ones((2, 3), np.float32), 4)
        garbage = connected_components_predictor(
            images, np.full((2, 3, 4), 9.9, np.float32),
            np.zeros((2, 3), np.float32), 4)
        np.testing.assert_array_equal(real, garbage)
        assert real.shape == (2, 4, 4)

    def test_the_predictor_is_image_dependent(self) -> None:
        images = np.stack([_canvas_with_rectangles(32, [(2, 10, 2, 10)]),
                           _canvas_with_rectangles(32, [(18, 30, 18, 30)])])
        emitted = connected_components_predictor(
            images, np.zeros((2, 1, 4), np.float32),
            np.ones((2, 1), np.float32), 3)
        assert float(emitted.std(axis=0).mean()) > 1e-2

    def test_it_carries_the_reads_image_marker(self) -> None:
        """`_score` dispatches on this attribute. Without it the arm is called
        with the label signature and scored blind to the pixels."""
        assert getattr(connected_components_predictor, "reads_image", False)
        assert not getattr(gt_oracle_predictor, "reads_image", False)
        assert not getattr(fixed_grid_prior, "reads_image", False)


class TestTheDetectorOnARealSplit:
    """The detector scored through the SAME `score_prior` path as the family."""

    def test_it_beats_the_fixed_grid_and_is_image_dependent(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        detector = score_prior(connected_components_predictor, model, loss,
                               tiny_val_dataset)
        grid = score_prior(fixed_grid_prior(), model, loss, tiny_val_dataset)
        spread = _detector_spread(model, tiny_val_dataset)
        _assert_the_detector_reads_the_pixels(spread, detector, grid)

    def test_it_shares_the_family_s_matched_denominator(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """It cannot win by matching fewer pairs: `is_matched` depends only on
        `target_valid`."""
        model, loss = tiny_context
        totals = {
            baselines._score(prior, model, loss, tiny_val_dataset)[1]
            for prior in (connected_components_predictor, fixed_grid_prior(),
                          gt_oracle_predictor)}
        assert len(totals) == 1, totals

    def test_it_stays_below_the_gt_oracle(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """A detector reading 1.0 would mean it is somehow seeing the labels."""
        model, loss = tiny_context
        detector = score_prior(connected_components_predictor, model, loss,
                               tiny_val_dataset)
        assert detector < 1.0


class TestTheDetectorGuardIsRedProven:
    """SC-D: a DEAD-COMPONENT injection must make the detector guard FAIL.

    The injection replaces :func:`connected_components_boxes` with one that
    ignores its ``image`` argument and returns the fixed 5x5 grid instead. Same
    shape, same dtype, finite, plausible -- and the ONLY thing that separates
    it from the live detector is that its boxes stop depending on the image.
    """

    @staticmethod
    def _blind_detector(image, num_queries, threshold=CC_THRESHOLD):
        del image, threshold
        return tile_to_queries(fixed_grid_prior(), num_queries)

    def test_a_blind_detector_fires_the_liveness_assertion(
            self, monkeypatch: pytest.MonkeyPatch,
            tiny_context: Tuple[Any, Any], tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        monkeypatch.setattr(baselines, "connected_components_boxes",
                            self._blind_detector)
        detector = score_prior(connected_components_predictor, model, loss,
                               tiny_val_dataset)
        grid = score_prior(fixed_grid_prior(), model, loss, tiny_val_dataset)
        spread = _detector_spread(model, tiny_val_dataset)
        with pytest.raises(AssertionError,
                           match="fixed prior wearing a detector's name"):
            _assert_the_detector_reads_the_pixels(spread, detector, grid)

    def test_the_live_detector_passes_the_same_assertion(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """The control beside the mutation: without the injection the SAME
        call passes, so the RED above is the injection's doing."""
        model, loss = tiny_context
        detector = score_prior(connected_components_predictor, model, loss,
                               tiny_val_dataset)
        grid = score_prior(fixed_grid_prior(), model, loss, tiny_val_dataset)
        _assert_the_detector_reads_the_pixels(
            _detector_spread(model, tiny_val_dataset), detector, grid)


class TestTheDetectorIsNotAFamilyMember:
    """`plans/SYSTEM.md:220` names the family. The detector is not in it."""

    @pytest.fixture(scope="class")
    def tiny_family(self) -> Dict[str, Dict[str, float]]:
        keras.utils.set_random_seed(1234)
        return evaluate_family(seed=7, ks=(TINY_K,), split=TINY_SPLIT)

    def test_the_detector_arm_is_reported(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        assert CONNECTED_COMPONENTS_ARM in tiny_family

    def test_family_max_excludes_it_even_when_it_wins(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        """The whole point of the separation: the detector normally SCORES
        HIGHEST, and quoting it as the family max would silently redefine the
        bar `SYSTEM.md:220` fixes."""
        name, value = family_max(tiny_family)
        assert name != CONNECTED_COMPONENTS_ARM
        assert value == max(tiny_family[GRID_ARM]["box_iou"],
                            tiny_family[kmeans_arm(TINY_K)]["box_iou"])

    def test_its_name_cannot_be_picked_up_by_the_family_filter(self) -> None:
        assert not CONNECTED_COMPONENTS_ARM.startswith(
            ("FIXED-GRID", "KMEANS-PRIOR"))


class TestTheSwapTouchesOnlyThePrompt:
    """`swap_batch_prompts` is the diagnostic's one moving part."""

    @staticmethod
    def _batch() -> Dict[str, Any]:
        return {
            "image": np.arange(3 * 2 * 2 * 3, dtype=np.float32).reshape(
                3, 2, 2, 3),
            "token_ids": np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32),
            "token_padding_mask": np.array(
                [[1, 0], [1, 1], [0, 0]], dtype=np.int32),
        }

    def test_it_rotates_both_prompt_tensors_by_the_shift(self) -> None:
        """A hand-computed value oracle: shift 1 moves row i to row i+1."""
        swapped = swap_batch_prompts(self._batch(), 1)
        np.testing.assert_array_equal(
            np.asarray(swapped["token_ids"]),
            np.array([[3, 3], [1, 1], [2, 2]], dtype=np.int32))
        np.testing.assert_array_equal(
            np.asarray(swapped["token_padding_mask"]),
            np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32))

    def test_the_mask_travels_with_the_ids(self) -> None:
        """Rotating the ids alone would feed a prompt under ANOTHER prompt's
        mask -- a third thing, not a swap. The four category phrases have
        different word counts, so the masks genuinely differ."""
        batch = self._batch()
        swapped = swap_batch_prompts(batch, 2)
        for index in range(3):
            source = (index - 2) % 3
            np.testing.assert_array_equal(
                np.asarray(swapped["token_ids"])[index],
                batch["token_ids"][source])
            np.testing.assert_array_equal(
                np.asarray(swapped["token_padding_mask"])[index],
                batch["token_padding_mask"][source])

    def test_the_image_is_the_callers_object_untouched(self) -> None:
        batch = self._batch()
        swapped = swap_batch_prompts(batch, 1)
        assert swapped["image"] is batch["image"]
        assert set(swapped) == set(batch)

    def test_shift_zero_is_a_no_op_and_is_therefore_vacuous(self) -> None:
        """The vacuity mode the liveness assertion exists to catch."""
        batch = self._batch()
        np.testing.assert_array_equal(
            np.asarray(swap_batch_prompts(batch, 0)["token_ids"]),
            batch["token_ids"])


class TestPromptSwapRetentionOnARealSplit:
    """The diagnostic run end to end against a real model and split."""

    @pytest.fixture(scope="class")
    def swap(self, tiny_context: Tuple[Any, Any],
             tiny_val_dataset: Any) -> Dict[str, float]:
        model, loss = tiny_context
        return prompt_swap_retention(model, loss, tiny_val_dataset)

    def test_it_returns_the_declared_keys_as_floats(
            self, swap: Dict[str, float]) -> None:
        assert set(swap) == {
            "box_iou_true", "box_iou_worst_wrong_prompt", "retained",
            "prompt_changed_fraction", "rel_delta_pred_boxes",
            "rel_delta_pred_logits"}
        assert all(isinstance(value, float) for value in swap.values())

    def test_the_instrument_is_live(self, swap: Dict[str, float]) -> None:
        """The GREEN half of the RED proof below: on a real split the swap
        really does change prompts and really does reach an output."""
        _assert_the_prompt_swap_is_live(swap["prompt_changed_fraction"],
                                        swap["rel_delta_pred_logits"])

    def test_the_wrong_prompt_arm_is_the_worst_of_the_shifts(
            self, swap: Dict[str, float]) -> None:
        assert swap["box_iou_worst_wrong_prompt"] <= swap["box_iou_true"] or (
            swap["retained"] >= 1.0)
        assert swap["retained"] == pytest.approx(
            swap["box_iou_worst_wrong_prompt"] / swap["box_iou_true"])

    def test_the_true_arm_reproduces_score_prior_s_reduction(
            self, tiny_context: Tuple[Any, Any], tiny_val_dataset: Any,
            swap: Dict[str, float]) -> None:
        """`_matched_iou` has ONE home: the oracle scored through the prior
        path still reads exactly 1.0, so the shared reduction did not drift."""
        model, loss = tiny_context
        _assert_oracle_reads_exactly_one(
            score_prior(gt_oracle_predictor, model, loss, tiny_val_dataset))
        assert 0.0 <= swap["box_iou_true"] <= 1.0

    def test_the_targets_are_never_swapped(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """Every arm is scored against each image's OWN ground truth, so the
        matched-pair denominator cannot move between arms."""
        model, loss = tiny_context
        totals = {baselines._score(prior, model, loss, tiny_val_dataset)[1]
                  for prior in (gt_oracle_predictor, fixed_grid_prior())}
        assert len(totals) == 1, totals


class TestThePromptSwapGuardIsRedProven:
    """SC-D: a DEAD-SWAP injection must make the liveness assertion FAIL.

    The injection replaces :func:`swap_batch_prompts` with one that returns the
    caller's inputs unchanged. The diagnostic still runs, still returns six
    finite floats, and still reports ``retained == 1.0000`` -- which is exactly
    the headline number. The ONLY thing separating that vacuous 1.0000 from the
    real measurement is ``prompt_changed_fraction``.
    """

    @staticmethod
    def _dead_swap(inputs, shift):
        del shift
        return dict(inputs)

    def test_a_dead_swap_fires_the_liveness_assertion(
            self, monkeypatch: pytest.MonkeyPatch,
            tiny_context: Tuple[Any, Any], tiny_val_dataset: Any) -> None:
        model, loss = tiny_context
        monkeypatch.setattr(baselines, "swap_batch_prompts", self._dead_swap)
        swap = prompt_swap_retention(model, loss, tiny_val_dataset)
        assert swap["retained"] == pytest.approx(1.0), (
            "the dead swap must still LOOK like the finding")
        with pytest.raises(AssertionError, match="NOTHING was swapped"):
            _assert_the_prompt_swap_is_live(swap["prompt_changed_fraction"],
                                            swap["rel_delta_pred_logits"])

    def test_the_live_swap_passes_the_same_assertion(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """The control beside the mutation."""
        model, loss = tiny_context
        swap = prompt_swap_retention(model, loss, tiny_val_dataset)
        _assert_the_prompt_swap_is_live(swap["prompt_changed_fraction"],
                                        swap["rel_delta_pred_logits"])


class TestThePromptSwapIsNotAFamilyMember:
    """It is a QUALIFIER on the model's number, not a bar the model clears."""

    def test_the_shift_set_is_pinned_and_excludes_the_vacuous_zero(
            self) -> None:
        assert 0 not in PROMPT_SWAP_SHIFTS
        assert len(PROMPT_SWAP_SHIFTS) >= 2, (
            "one rotation can land on the same category by chance")

    def test_only_the_two_prompt_keys_are_named(self) -> None:
        assert PROMPT_KEYS == ("token_ids", "token_padding_mask")

    def test_the_cli_exposes_it_and_it_is_off_by_default(self) -> None:
        args = build_parser().parse_args([])
        assert args.prompt_swap is None
        typed = build_parser().parse_args(
            ["--prompt-swap", "results/x_seed{seed}/best_model.keras"])
        assert typed.prompt_swap.format(seed=2) == (
            "results/x_seed2/best_model.keras")
