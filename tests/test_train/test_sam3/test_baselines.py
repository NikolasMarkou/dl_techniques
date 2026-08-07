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

5. **The distractor gap's ability to SEPARATE.** ``distractor_gap`` scores the
   same boxes against the prompted category's GT and against every other
   category's, and the whole claim is that a category-blind detector cannot win
   it. Both ends are pinned on the REAL 64-image split, at values known before
   the instrument runs: the GT oracle reads ``1.0 / 0.0 / gap 1.0`` BY
   CONSTRUCTION (the generator places instances NON-OVERLAPPING, so a box that
   is a prompted instance has zero IoU with every other category's), and the
   connected-components detector -- which BEATS the trained checkpoint on raw
   ``box_iou`` -- reads a gap of ~0. ``TestTheDistractorGapIsRedProven`` fires
   both ends with dead-component injections, one of which (feeding the SAME
   target dict twice) is the sharpest guard in this file: it is the only check
   that proves the two ``_matched_iou`` calls really do receive DIFFERENT
   target sets.

Device: CPU-cheap. ``TestTheFamilyEndToEnd`` and the oracle guards build the
``tiny`` variant with a 4-sample split; the distractor-gap liveness classes
build the ``small`` variant on the REAL 64-image split (~10 s), because a
near-zero statistic measured on 4 images is not a measurement -- and because
the 0.02 ceiling the detector must sit under was fixed on that split.
"""

import hashlib
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest

from tests.test_train.test_sam3.parser_help_guard import (
    assert_no_bare_percent_help,
)
from train.sam3 import baselines
from train.sam3.baselines import (
    CC_THRESHOLD,
    CONNECTED_COMPONENTS_ARM,
    DEGENERATE_ARM,
    DISTRACTOR_GAP_ARM,
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
    distractor_gap,
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

#: The ceiling a CATEGORY-BLIND arm's distractor gap must sit under, on the
#: real split. Pre-registered as SC-C before it was measured; the connected-
#: components detector reads -0.0029 and the one-box degenerate prior -0.0085,
#: both an order of magnitude below it, against the oracle's 1.0. The bound is
#: on the ABSOLUTE gap: a blind arm can land on either side of zero (it does --
#: both blind arms here score the distractor set slightly HIGHER than the
#: prompted one), and a one-sided `gap <= 0.02` would be satisfied by an arm
#: that had collapsed to a large NEGATIVE gap, which is not "cannot win" but a
#: different defect.
CATEGORY_BLIND_GAP_CEILING: float = 0.02


def _image_key(images: Any) -> str:
    """A content hash of one batch's images.

    Used to hand a baseline predictor the ground truth of the image it is
    currently being asked about, since :func:`distractor_gap` passes its
    stand-in checkpoint only the model INPUTS. Hashing rather than relying on
    iteration order means a dataset that replayed its batches in a different
    order would raise instead of silently scoring the wrong labels.

    Args:
        images: ``(B, S, S, 3)`` image tensor or array.

    Returns:
        The sha256 hex digest of the batch's float32 bytes.
    """
    array = np.ascontiguousarray(np.asarray(images, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _degenerate_as_predictor(target_boxes: np.ndarray,
                             target_valid: np.ndarray,
                             num_queries: int) -> np.ndarray:
    """:data:`baselines.Predictor` view of the one-centered-box prior.

    ``distractor_gap`` scores a CHECKPOINT, not a prior, so the array-shaped
    arms of the family reach it through the same predictor signature the GT
    oracle uses.

    Args:
        target_boxes: ``(B, N_max, 4)`` -- read for the batch size only.
        target_valid: Unused.
        num_queries: The model's ``Q``.

    Returns:
        ``(B, num_queries, 4)`` float32, identical for every image.
    """
    del target_valid
    tiled = tile_to_queries(degenerate_prior(), num_queries)
    return np.tile(tiled[None], (target_boxes.shape[0], 1, 1))


class _PredictorAsCheckpoint:
    """A stand-in checkpoint whose ``pred_boxes`` come from a baseline arm.

    ``distractor_gap`` takes a MODEL -- it calls ``model.sam3(inputs,
    training=False)`` -- but the two arms whose readings are known IN ADVANCE
    (the GT oracle at 1.0, the category-blind detector at ~0) are PREDICTORS,
    scored elsewhere in this module through :func:`baselines._score`. This
    adapter is what lets the same two arms run through the diagnostic under
    test, so its liveness is pinned against values that were not read off its
    own output.

    It is deliberately thin: it forwards the batch to the predictor and emits
    ZEROED ``pred_logits`` exactly as :func:`baselines._score` does (a prior
    makes no class claim, so the matcher's class cost is constant). That the
    adapter is faithful is not asserted by inspection -- it is proven by value,
    in ``test_the_detector_arm_agrees_with_score_prior_digit_for_digit``.

    Attributes:
        include_masks: Copied from the real model, for ``unpack_targets``.
        num_queries: Copied from the real model.
    """

    def __init__(self, model: Any, predictor: Any, dataset: Any) -> None:
        """Prime the image-hash -> ground-truth table by one pass over ``dataset``.

        Args:
            model: A built ``Sam3TrainingModel`` -- supplies the two geometry
                attributes only; its weights are never read.
            predictor: A :data:`baselines.Predictor`, or an
                :data:`baselines.ImagePredictor` carrying ``reads_image``.
            dataset: The scoring split, built with
                ``include_all_instances=True``.
        """
        self.include_masks = model.include_masks
        self.num_queries = int(model.num_queries)
        self._predictor = predictor
        self._reads_image = bool(getattr(predictor, "reads_image", False))
        self._ground_truth: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for inputs, y_true, _all_instances in dataset:
            targets = baselines.unpack_targets(
                baselines.ops.cast(y_true, "float32"), self.include_masks)
            self._ground_truth[_image_key(inputs["image"])] = (
                np.asarray(baselines.ops.convert_to_numpy(baselines.ops.cast(
                    targets["target_boxes"], "float32")), dtype=np.float32),
                np.asarray(baselines.ops.convert_to_numpy(baselines.ops.cast(
                    targets["target_valid"], "float32")), dtype=np.float32))

    def sam3(self, inputs: Dict[str, Any], training: bool = False
             ) -> Dict[str, Any]:
        """Emit the predictor's boxes for this batch.

        Args:
            inputs: One batch of the dataset's input dict.
            training: Accepted for signature compatibility; ignored.

        Returns:
            ``{"pred_boxes": (B, Q, 4), "pred_logits": (B, Q, 1)}``.

        Raises:
            AssertionError: If the batch's images were not seen while priming,
                i.e. the adapter is being asked about ground truth it does not
                hold.
        """
        del training
        images = np.asarray(inputs["image"], dtype=np.float32)
        key = _image_key(images)
        assert key in self._ground_truth, (
            "the stand-in checkpoint was handed a batch of images it never "
            "saw while priming, so it has no ground truth for them; any "
            "number derived from this batch would describe other images.")
        target_boxes, target_valid = self._ground_truth[key]
        boxes = (self._predictor(images, target_boxes, target_valid,
                                 self.num_queries) if self._reads_image
                 else self._predictor(target_boxes, target_valid,
                                      self.num_queries))
        boxes = np.asarray(boxes, dtype=np.float32)
        return {
            "pred_boxes": baselines.ops.convert_to_tensor(boxes),
            "pred_logits": baselines.ops.zeros(
                (boxes.shape[0], self.num_queries, 1), dtype="float32"),
        }


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


def _assert_the_oracle_separates_the_two_target_sets(
        result: Dict[str, float]) -> None:
    """THE liveness assertion for ``distractor_gap``, in ONE place.

    All three readings are known BEFORE the instrument runs, from the
    STRUCTURE of the metric rather than from any run of it. A predictor that
    emits exactly the prompted category's ground-truth boxes must score the
    prompted target set 1.0 (the matcher is free to pick any query per target
    and every target is present among the queries); and it must score the
    distractor target set 0.0, because ``data.py``'s generator places every
    instance NON-OVERLAPPING -- so a box that IS a prompted instance has zero
    intersection with every other category's instance in that image. The gap
    is their difference, so it is 1.0.

    The three are asserted separately, with distinct messages, so a RED proof
    can name WHICH one fired: the two mutations this file ships fire different
    ones (a broken matcher fires the prompted assertion; scoring the same
    target set twice fires the distractor assertion while the prompted one
    stays green).

    Args:
        result: A :func:`distractor_gap` result computed for the GT-oracle arm.

    Returns:
        None.

    Raises:
        AssertionError: If any of the three known readings is wrong.
    """
    assert result["box_iou_prompted"] == pytest.approx(1.0, abs=1e-6), (
        f"the oracle's prompted arm read {result['box_iou_prompted']!r}, not "
        f"1.0. The oracle emits the prompted category's own GT boxes, so "
        f"anything else means the matched-IoU reduction under the prompted "
        f"target set is broken and no gap derived from it may be believed.")
    assert result["box_iou_distractor"] == pytest.approx(0.0, abs=1e-6), (
        f"the oracle's distractor arm read {result['box_iou_distractor']!r}, "
        f"not 0.0. The generator places instances NON-OVERLAPPING, so the "
        f"prompted category's own boxes cannot overlap another category's. A "
        f"non-zero reading means the distractor arm is not being handed a "
        f"DIFFERENT target set -- the failure mode that makes every gap "
        f"number in this module a plausible-looking lie.")
    assert result["gap"] == pytest.approx(1.0, abs=1e-6), (
        f"the oracle's gap read {result['gap']!r}, not 1.0. The instrument's "
        f"ceiling is not where it is claimed to be, so no arm's distance from "
        f"it is interpretable.")


def _assert_a_category_blind_arm_cannot_win_the_gap(
        name: str, result: Dict[str, float]) -> None:
    """THE liveness assertion for the metric's FLOOR, in ONE place.

    This is the whole point of the diagnostic: an arm that cannot read the
    text prompt at all -- because it has no text input -- must read ~0 here NO
    MATTER HOW HIGH its raw ``box_iou`` is. The connected-components detector
    scores 0.937 prompted, ABOVE the trained checkpoint, and still reads a gap
    of -0.003. An instrument on which a blind arm posts a real gap is
    measuring something other than category selectivity.

    Args:
        name: The arm's name, for the failure message.
        result: That arm's :func:`distractor_gap` result.

    Returns:
        None.

    Raises:
        AssertionError: If the blind arm's absolute gap clears the ceiling.
    """
    assert abs(result["gap"]) <= CATEGORY_BLIND_GAP_CEILING, (
        f"{name} -- an arm with NO text input at all -- read a distractor gap "
        f"of {result['gap']:.4f}, outside the pre-registered "
        f"+/-{CATEGORY_BLIND_GAP_CEILING} band (prompted "
        f"{result['box_iou_prompted']:.4f}, distractor "
        f"{result['box_iou_distractor']:.4f}). A category-BLIND arm posting a "
        f"gap means the gap is not measuring category selectivity, and every "
        f"trained arm's reading on it is uninterpretable.")


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


@pytest.fixture(scope="module")
def tiny_all_instance_dataset(tiny_context: Tuple[Any, Any]) -> Any:
    """The SAME scoring split, carrying the eval-only all-category geometry.

    One dataset object, not two zipped ones: the prompted targets and the
    all-instance geometry come out of the SAME record, so there is no batch
    alignment to get wrong.
    """
    model, _loss = tiny_context
    return build_split_dataset(model, seed=7, train=False, split=TINY_SPLIT,
                               include_all_instances=True)


@pytest.fixture(scope="module")
def real_context() -> Tuple[Any, Any]:
    """The REAL run geometry -- ``baselines.SPLIT``, ``small``, 64 val images.

    The near-zero readings this file pins for the blind arms are statistics
    over ~200 matched pairs; measured on the 4-image ``tiny`` split they would
    be noise quoted to four decimals. Cheap despite the name: no arm here runs
    the model's forward pass, so only the build (~1.6 s) is paid.
    """
    keras.utils.set_random_seed(1234)
    return build_context(seed=7)


@pytest.fixture(scope="module")
def real_val_dataset(real_context: Tuple[Any, Any]) -> Any:
    """The real scoring split in its ordinary 2-tuple form."""
    model, _loss = real_context
    return build_split_dataset(model, seed=7, train=False)


@pytest.fixture(scope="module")
def real_all_instance_dataset(real_context: Tuple[Any, Any]) -> Any:
    """The SAME real scoring split, carrying the all-category geometry."""
    model, _loss = real_context
    return build_split_dataset(model, seed=7, train=False,
                               include_all_instances=True)


@pytest.fixture(scope="module")
def oracle_gap(real_context: Tuple[Any, Any],
               real_all_instance_dataset: Any) -> Dict[str, float]:
    """``distractor_gap`` of the GT-oracle arm on the real split."""
    model, loss = real_context
    return distractor_gap(
        _PredictorAsCheckpoint(model, gt_oracle_predictor,
                               real_all_instance_dataset),
        loss, real_all_instance_dataset)


@pytest.fixture(scope="module")
def detector_gap(real_context: Tuple[Any, Any],
                 real_all_instance_dataset: Any) -> Dict[str, float]:
    """``distractor_gap`` of the connected-components detector."""
    model, loss = real_context
    return distractor_gap(
        _PredictorAsCheckpoint(model, connected_components_predictor,
                               real_all_instance_dataset),
        loss, real_all_instance_dataset)


@pytest.fixture(scope="module")
def degenerate_gap(real_context: Tuple[Any, Any],
                   real_all_instance_dataset: Any) -> Dict[str, float]:
    """``distractor_gap`` of the one-centered-box prior."""
    model, loss = real_context
    return distractor_gap(
        _PredictorAsCheckpoint(model, _degenerate_as_predictor,
                               real_all_instance_dataset),
        loss, real_all_instance_dataset)


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
            "prompt_changed_fraction", "matched_pairs", "rel_delta_pred_boxes",
            "rel_delta_pred_logits"}
        assert all(isinstance(value, float) for value in swap.values())

    def test_it_reports_the_denominator_the_family_is_scored_over(
            self, swap: Dict[str, float], tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """`retained` divides two independently pooled ratios.

        The matcher keeps a pair only where its cost clears
        `VALID_COST_THRESHOLD`, and that cost includes a class term computed
        from `pred_logits` -- which DO move under a swap. So an equal
        denominator is a fact to be checked, not a construction. Here it is
        also checked to be the SAME denominator every prior arm is scored
        over, so the model's number and the family's are like for like.
        """
        model, loss = tiny_context
        assert swap["matched_pairs"] > 0.0
        assert swap["matched_pairs"] == pytest.approx(
            baselines._score(gt_oracle_predictor, model, loss,
                             tiny_val_dataset)[1])

    def test_the_unequal_denominator_raise_is_red_proven(
            self, monkeypatch: pytest.MonkeyPatch,
            tiny_context: Tuple[Any, Any], tiny_val_dataset: Any) -> None:
        """Make ONE swapped arm match fewer pairs and the ratio must REFUSE.

        The injection wraps `_matched_iou` and drops one matched pair from the
        first wrong-prompt call only -- exactly what a future change to the
        class cost could do silently, since `retained` would keep returning a
        finite, plausible number.
        """
        model, loss = tiny_context
        real = baselines._matched_iou
        seen: List[int] = []

        def _one_arm_short(boxes, logits, targets, loss_):
            total, pairs = real(boxes, logits, targets, loss_)
            seen.append(1)
            return (total, pairs - 1.0) if len(seen) == 2 else (total, pairs)

        monkeypatch.setattr(baselines, "_matched_iou", _one_arm_short)
        with pytest.raises(ValueError, match="DIFFERENT numbers of"):
            prompt_swap_retention(model, loss, tiny_val_dataset)

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


class TestDistractorGapOnARealSplit:
    """The diagnostic run end to end against a real model and split.

    The liveness half (ORACLE reads 1.0, the category-blind detector reads
    ~0.005) and its RED proofs are a separate concern and live in their own
    classes; what is pinned here is that the plumbing is real -- the declared
    keys, the two DIFFERENT denominators, and the arithmetic relating them.
    """

    @pytest.fixture(scope="class")
    def gap(self, tiny_context: Tuple[Any, Any],
            tiny_all_instance_dataset: Any) -> Dict[str, float]:
        model, loss = tiny_context
        return distractor_gap(model, loss, tiny_all_instance_dataset)

    def test_it_returns_the_declared_keys_as_floats(
            self, gap: Dict[str, float]) -> None:
        assert set(gap) == {
            "box_iou_prompted", "box_iou_distractor", "gap", "relative_gap",
            "matched_pairs_prompted", "matched_pairs_distractor",
            "images_with_distractor"}
        assert all(isinstance(value, float) for value in gap.values())

    def test_the_gap_is_the_difference_of_the_two_arms(
            self, gap: Dict[str, float]) -> None:
        assert gap["gap"] == pytest.approx(
            gap["box_iou_prompted"] - gap["box_iou_distractor"])
        assert gap["relative_gap"] == pytest.approx(
            gap["gap"] / gap["box_iou_prompted"])

    def test_both_denominators_are_reported_and_neither_is_asserted_equal(
            self, gap: Dict[str, float]) -> None:
        """The two arms score DIFFERENT target sets, by design.

        ``prompt_swap_retention`` RAISES when its two arms match different
        pair counts, because there the two arms score the SAME targets and a
        differing denominator means the ratio compares two populations. Here
        the populations differ ON PURPOSE -- ``zero_instance_rate`` gives some
        images an absent prompted category, which contributes zero prompted
        pairs and non-zero distractor pairs -- so copying that raise would
        make the diagnostic refuse to run on its own intended input.
        """
        assert gap["matched_pairs_prompted"] > 0.0
        assert gap["matched_pairs_distractor"] > 0.0

    def test_the_prompted_arm_is_the_family_s_own_denominator(
            self, gap: Dict[str, float], tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """The prompted arm must be like-for-like with every family arm, so
        the gap qualifies the SAME number the family bars."""
        model, loss = tiny_context
        assert gap["matched_pairs_prompted"] == pytest.approx(
            baselines._score(gt_oracle_predictor, model, loss,
                             tiny_val_dataset)[1])

    def test_the_image_count_is_reported_not_absorbed(
            self, gap: Dict[str, float]) -> None:
        """An image with only ONE category drawn has no distractor at all."""
        assert 0.0 < gap["images_with_distractor"] <= float(
            TINY_SPLIT["num_val_samples"])

    def test_it_refuses_a_dataset_built_without_the_all_category_export(
            self, tiny_context: Tuple[Any, Any],
            tiny_val_dataset: Any) -> None:
        """The silent failure this raise exists to prevent: scoring the
        prompted targets twice reads gap == 0.0 for every checkpoint."""
        model, loss = tiny_context
        with pytest.raises(ValueError, match="include_all_instances=True"):
            distractor_gap(model, loss, tiny_val_dataset)


class TestTheDistractorTargetsAreTheSameImageSTargets:
    """A-2, executed rather than inferred.

    The plan's assumption was that a SIBLING distractor dataset would zip
    batch-for-batch with the val split. It is retired by construction here --
    both target sets come out of ONE record -- but the property is still
    ASSERTED in code before any number is quoted, because a misaligned pair
    would produce entirely plausible garbage.
    """

    @staticmethod
    def _one_batch(dataset: Any, model: Any) -> Tuple[Dict[str, Any],
                                                      Dict[str, Any]]:
        for _inputs, y_true, all_instances in dataset:
            return (dict(all_instances),
                    baselines.unpack_targets(
                        baselines.ops.cast(y_true, "float32"),
                        model.include_masks))
        raise AssertionError("the fixture dataset yielded no batch")

    def test_the_all_instance_set_contains_every_prompted_target(
            self, tiny_context: Tuple[Any, Any],
            tiny_all_instance_dataset: Any) -> None:
        """The GREEN control beside the mutation below."""
        model, _loss = tiny_context
        all_instances, prompted = self._one_batch(tiny_all_instance_dataset,
                                                  model)
        targets = baselines._distractor_targets(all_instances, prompted)
        assert set(targets) == {"target_boxes", "target_valid"}

    def test_a_displaced_box_fires_the_alignment_assertion(
            self, tiny_context: Tuple[Any, Any],
            tiny_all_instance_dataset: Any) -> None:
        """RED proof: move the all-instance geometry off the record it came
        from and the assertion must REFUSE before any IoU is computed.

        The firing assertion is the ``ValueError`` in
        ``_distractor_targets``: 'has a valid prompted target box ... with no
        counterpart among the ... all-instance row(s) of its prompted
        category'.
        """
        model, _loss = tiny_context
        all_instances, prompted = self._one_batch(tiny_all_instance_dataset,
                                                  model)
        displaced = np.asarray(baselines.ops.convert_to_numpy(
            all_instances[baselines.RECORD_ALL_BOXES]), dtype=np.float32)
        displaced[..., 0] += 0.25
        all_instances[baselines.RECORD_ALL_BOXES] = displaced
        with pytest.raises(ValueError, match="no counterpart"):
            baselines._distractor_targets(all_instances, prompted)

    def test_the_distractor_rows_exclude_the_prompted_category(
            self, tiny_context: Tuple[Any, Any],
            tiny_all_instance_dataset: Any) -> None:
        """The whole point of the arm: it must not be scoring the prompted
        instances under another name."""
        model, _loss = tiny_context
        all_instances, prompted = self._one_batch(tiny_all_instance_dataset,
                                                  model)
        targets = baselines._distractor_targets(all_instances, prompted)
        ids = np.asarray(baselines.ops.convert_to_numpy(
            all_instances[baselines.RECORD_ALL_CATEGORY_IDS]))
        prompt_id = np.asarray(baselines.ops.convert_to_numpy(
            all_instances[baselines.RECORD_PROMPT_ID]))
        valid = np.asarray(baselines.ops.convert_to_numpy(
            targets["target_valid"]))
        assert valid.sum() > 0.0, "no distractor row at all is a dead arm"
        assert not np.any(valid[ids == prompt_id[:, None]] > 0.0)


class TestTheDistractorGapCli:
    """The flag is wired end to end and is OFF by default."""

    def test_the_cli_exposes_it_and_it_is_off_by_default(self) -> None:
        args = build_parser().parse_args([])
        assert args.distractor_gap is None
        typed = build_parser().parse_args(
            ["--distractor-gap", "results/x_seed{seed}/final_model.keras"])
        assert typed.distractor_gap.format(seed=2) == (
            "results/x_seed2/final_model.keras")


class TestTheDistractorGapSeparatesTheOracleFromABlindArm:
    """SC-C: the instrument's CEILING and FLOOR, on the REAL 64-image split.

    ``box_iou`` alone settles nothing on this generator -- the zero-parameter
    connected-components detector scores ABOVE the trained checkpoint. The
    claim ``distractor_gap`` makes is that the SAME arm cannot win the gap. It
    is pinned at both ends here, at values fixed before the instrument ran:
    the oracle at exactly ``1.0 / 0.0 / 1.0`` by construction, the blind arms
    inside +/-0.02.

    The degenerate arm is here for a specific reason: it is blind AND bad
    (0.025 prompted). Together with the detector (blind and EXCELLENT, 0.937
    prompted) it separates "the gap is near zero" from "the prompted IoU is
    near zero", so nobody can read this metric as a quality score.
    """

    def test_the_oracle_reads_one_zero_and_a_gap_of_one(
            self, oracle_gap: Dict[str, float]) -> None:
        _assert_the_oracle_separates_the_two_target_sets(oracle_gap)

    def test_the_oracle_arm_is_not_vacuous(
            self, oracle_gap: Dict[str, float]) -> None:
        """A ceiling of 1.0 read over zero pairs, or with no distractor row on
        any image, would be an arithmetic artifact rather than a measurement."""
        assert oracle_gap["matched_pairs_prompted"] > 0.0
        assert oracle_gap["matched_pairs_distractor"] > 0.0
        assert oracle_gap["images_with_distractor"] > 0.0

    def test_the_detector_wins_box_iou_and_still_loses_the_gap(
            self, detector_gap: Dict[str, float]) -> None:
        """The plan's whole thesis in one assertion pair.

        Both halves are load-bearing. Without the first, a detector that had
        collapsed to boxing nothing would satisfy the second and look like a
        pass; without the second, the arm is just another ``box_iou`` row.
        """
        prompted = detector_gap["box_iou_prompted"]
        assert prompted >= 0.9, (
            f"the category-blind detector read {prompted:.4f} on the prompted "
            f"target set. Its near-zero GAP is only meaningful while its raw "
            f"box_iou is HIGH -- a detector scoring ~0 everywhere would post "
            f"a near-zero gap too, and prove nothing.")
        _assert_a_category_blind_arm_cannot_win_the_gap(
            CONNECTED_COMPONENTS_ARM, detector_gap)

    def test_the_degenerate_arm_is_blind_and_bad_and_still_reads_no_gap(
            self, degenerate_gap: Dict[str, float]) -> None:
        """Stops the gap being read as a quality metric: this arm is terrible
        at the task and reads the same near-zero gap as the excellent one."""
        assert degenerate_gap["box_iou_prompted"] < 0.05
        _assert_a_category_blind_arm_cannot_win_the_gap(
            DEGENERATE_ARM, degenerate_gap)

    def test_the_oracle_beats_both_blind_arms_by_two_orders_of_magnitude(
            self, oracle_gap: Dict[str, float],
            detector_gap: Dict[str, float],
            degenerate_gap: Dict[str, float]) -> None:
        """The separation itself, stated as one comparison."""
        assert oracle_gap["gap"] > 50.0 * max(
            abs(detector_gap["gap"]), abs(degenerate_gap["gap"]))

    def test_the_detector_arm_agrees_with_score_prior_digit_for_digit(
            self, real_context: Tuple[Any, Any], real_val_dataset: Any,
            detector_gap: Dict[str, float]) -> None:
        """Proves the stand-in checkpoint is FAITHFUL rather than convenient.

        ``_PredictorAsCheckpoint`` is test-side machinery, and a wrong one
        would produce entirely plausible numbers. Its prompted arm must
        reproduce :func:`score_prior`'s published reduction -- the same number
        every other detector row in this file quotes -- to full float
        precision, through a completely different call path.
        """
        model, loss = real_context
        assert detector_gap["box_iou_prompted"] == pytest.approx(
            score_prior(connected_components_predictor, model, loss,
                        real_val_dataset), abs=1e-12)

    def test_every_arm_shares_both_denominators(
            self, oracle_gap: Dict[str, float],
            detector_gap: Dict[str, float],
            degenerate_gap: Dict[str, float]) -> None:
        """``is_matched`` depends only on ``target_valid``, so no arm can win
        the gap by matching fewer pairs on one of the two target sets."""
        for key in ("matched_pairs_prompted", "matched_pairs_distractor",
                    "images_with_distractor"):
            assert len({oracle_gap[key], detector_gap[key],
                        degenerate_gap[key]}) == 1, key

    def test_the_two_denominators_differ_and_that_is_the_design(
            self, oracle_gap: Dict[str, float]) -> None:
        """``zero_instance_rate`` gives ~25% of images an ABSENT prompted
        category: zero prompted pairs, non-zero distractor pairs. This is why
        ``prompt_swap_retention``'s equal-denominator raise is NOT copied."""
        assert (oracle_gap["matched_pairs_distractor"]
                > oracle_gap["matched_pairs_prompted"])


class TestTheZeroInstanceImagesAreScoredNotRefused:
    """The edge case that would make an inherited raise fire on real input.

    ``prompt_swap_retention`` RAISES on unequal denominators. Copying that
    into ``distractor_gap`` would make it refuse the split it is built for --
    so the SPLIT is first shown to actually contain such an image, and only
    then is the non-refusal asserted. Without the first half the second is
    vacuous.
    """

    def test_the_split_really_contains_an_absent_prompted_category(
            self, real_context: Tuple[Any, Any],
            real_all_instance_dataset: Any) -> None:
        model, _loss = real_context
        empty_prompted = 0
        with_distractor = 0
        for _inputs, y_true, all_instances in real_all_instance_dataset:
            prompted = baselines.unpack_targets(
                baselines.ops.cast(y_true, "float32"), model.include_masks)
            targets = baselines._distractor_targets(all_instances, prompted)
            prompted_rows = np.asarray(baselines.ops.convert_to_numpy(
                baselines.ops.cast(prompted["target_valid"], "float32"))
            ).sum(axis=-1)
            distractor_rows = np.asarray(baselines.ops.convert_to_numpy(
                targets["target_valid"])).sum(axis=-1)
            empty_prompted += int(np.sum(
                (prompted_rows == 0.0) & (distractor_rows > 0.0)))
            with_distractor += int(np.sum(distractor_rows > 0.0))
        assert empty_prompted > 0, (
            "no image on this split has an absent prompted category, so the "
            "unequal-denominator edge case is NOT exercised and the test "
            "below proves nothing.")
        assert with_distractor > empty_prompted

    def test_the_diagnostic_reports_both_denominators_without_raising(
            self, oracle_gap: Dict[str, float]) -> None:
        """The result exists at all -- the fixture would have raised."""
        assert oracle_gap["matched_pairs_prompted"] > 0.0
        assert (oracle_gap["images_with_distractor"]
                == pytest.approx(float(baselines.SPLIT["num_val_samples"])))


class TestTheDistractorGapIsRedProven:
    """SC-C's RED half: TWO dead-component injections, on the real split.

    * **M-A -- the no-op ASSIGNMENT.** ``loss.matcher`` still computes the real
      ``is_matched`` (both denominators stay live and correct) but returns a
      constant assignment mapping every query to target slot 0. Same shapes,
      same dtypes, no raise, and a finite plausible reading -- prompted 0.4808,
      distractor 0.2075, gap 0.2732. This is the same injection
      ``TestTheOracleGuardIsRedProven`` uses on ``score_prior``, applied to the
      new reduction, and it fires the PROMPTED assertion.
    * **M-B -- the SAME target dict twice.** ``_distractor_targets`` is
      replaced by one that hands back the PROMPTED targets. This is the
      sharpest guard in the file: it is the only check that can tell whether
      the two ``_matched_iou`` calls really receive DIFFERENT target sets.
      Under it the prompted assertion stays GREEN (1.0, still correct) and the
      DISTRACTOR one fires at 1.0 instead of 0.0, driving the gap to exactly
      0.0 -- which is precisely what a checkpoint with no category selectivity
      would be reported as, so the failure is invisible to every other test
      here.
    """

    @staticmethod
    def _no_op_assignment(loss: Any) -> Any:
        """M-A: keep the real ``is_matched``, kill only the assignment."""
        live = loss.matcher

        def mutant(pred_logits: Any, pred_boxes: Any, target_boxes: Any,
                   target_valid: Any) -> Any:
            from keras import ops

            assignment, is_matched = live(pred_logits, pred_boxes,
                                          target_boxes, target_valid)
            return ops.zeros_like(assignment), is_matched

        return mutant

    @staticmethod
    def _the_prompted_targets_again(all_instances: Dict[str, Any],
                                    prompted: Dict[str, Any]
                                    ) -> Dict[str, Any]:
        """M-B: the distractor set IS the prompted set."""
        del all_instances
        return {"target_boxes": prompted["target_boxes"],
                "target_valid": prompted["target_valid"]}

    def test_m_a_a_no_op_assignment_fires_the_prompted_assertion(
            self, real_context: Tuple[Any, Any],
            real_all_instance_dataset: Any,
            monkeypatch: pytest.MonkeyPatch) -> None:
        model, loss = real_context
        stand_in = _PredictorAsCheckpoint(model, gt_oracle_predictor,
                                          real_all_instance_dataset)
        monkeypatch.setattr(loss, "matcher", self._no_op_assignment(loss))
        result = distractor_gap(stand_in, loss, real_all_instance_dataset)
        assert all(np.isfinite(value) for value in result.values()), (
            "M-A must produce a PLAUSIBLE reading, not an obviously broken "
            f"one; got {result!r}. A mutation caught by arithmetic does not "
            "prove the guard is what catches it.")
        assert 0.0 < result["gap"] < 1.0, (
            f"M-A read a gap of {result['gap']!r} -- a value no reader would "
            f"question. That is exactly why the guard, not the eye, has to "
            f"catch it.")
        with pytest.raises(AssertionError, match="prompted arm read"):
            _assert_the_oracle_separates_the_two_target_sets(result)

    def test_m_b_the_same_target_dict_twice_fires_the_distractor_assertion(
            self, real_context: Tuple[Any, Any],
            real_all_instance_dataset: Any,
            monkeypatch: pytest.MonkeyPatch) -> None:
        model, loss = real_context
        monkeypatch.setattr(baselines, "_distractor_targets",
                            self._the_prompted_targets_again)
        result = distractor_gap(
            _PredictorAsCheckpoint(model, gt_oracle_predictor,
                                   real_all_instance_dataset),
            loss, real_all_instance_dataset)
        assert result["gap"] == 0.0, (
            f"scoring the prompted targets twice must drive the gap to "
            f"EXACTLY 0.0; got {result['gap']!r}. If it does not, this "
            f"mutation is not the one it claims to be.")
        assert result["box_iou_prompted"] == pytest.approx(1.0, abs=1e-6), (
            "M-B must leave the PROMPTED arm correct -- that is what makes it "
            "sharper than M-A: only the distractor half is wrong.")
        with pytest.raises(AssertionError, match="distractor arm read"):
            _assert_the_oracle_separates_the_two_target_sets(result)

    def test_m_b_also_collapses_the_second_denominator_onto_the_first(
            self, real_context: Tuple[Any, Any],
            real_all_instance_dataset: Any, oracle_gap: Dict[str, float],
            monkeypatch: pytest.MonkeyPatch) -> None:
        """An independent structural signature of the same defect, so the RED
        does not rest on the IoU value alone: the two denominators DIFFER by
        construction on this split and become equal the moment one target set
        is scored twice."""
        model, loss = real_context
        monkeypatch.setattr(baselines, "_distractor_targets",
                            self._the_prompted_targets_again)
        result = distractor_gap(
            _PredictorAsCheckpoint(model, gt_oracle_predictor,
                                   real_all_instance_dataset),
            loss, real_all_instance_dataset)
        assert (result["matched_pairs_distractor"]
                == result["matched_pairs_prompted"])
        assert (oracle_gap["matched_pairs_distractor"]
                != oracle_gap["matched_pairs_prompted"])

    def test_the_guard_is_green_again_once_both_are_restored(
            self, real_context: Tuple[Any, Any],
            real_all_instance_dataset: Any) -> None:
        """The control beside the two mutations -- otherwise the RED readings
        could be an artifact of the fixture rather than of the injections."""
        model, loss = real_context
        _assert_the_oracle_separates_the_two_target_sets(distractor_gap(
            _PredictorAsCheckpoint(model, gt_oracle_predictor,
                                   real_all_instance_dataset),
            loss, real_all_instance_dataset))


class TestTheDistractorGapIsNotAFamilyMember:
    """SC-D. `plans/SYSTEM.md:220` names the family; this diagnostic is not it.

    A gap is not a ``box_iou`` and must never be quotable as the bar an
    accuracy claim clears. Two independent locks are pinned: the arm never
    enters ``evaluate_family``'s results at all, and even if it were injected
    there its NAME cannot pass ``family_max``'s allowlist filter.
    """

    @pytest.fixture(scope="class")
    def tiny_family(self) -> Dict[str, Dict[str, float]]:
        keras.utils.set_random_seed(1234)
        return evaluate_family(seed=7, ks=(TINY_K,), split=TINY_SPLIT)

    def test_it_never_enters_the_family_results_at_all(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        assert DISTRACTOR_GAP_ARM not in tiny_family
        assert not any("gap" in row for row in tiny_family.values())

    def test_its_name_cannot_be_picked_up_by_the_family_filter(self) -> None:
        assert not DISTRACTOR_GAP_ARM.startswith(
            ("FIXED-GRID", "KMEANS-PRIOR"))

    def test_family_max_ignores_it_even_when_it_would_win(
            self, tiny_family: Dict[str, Dict[str, float]]) -> None:
        """The GT oracle's gap is 1.0 -- higher than any family member's
        ``box_iou`` can ever be. Filing it under the family would silently
        replace the pre-registered bar with an unreachable one."""
        before = family_max(tiny_family)
        injected = dict(tiny_family)
        injected[DISTRACTOR_GAP_ARM] = {"box_iou": 1.0}
        after = family_max(injected)
        assert after == before
        assert after[0] != DISTRACTOR_GAP_ARM

    def test_the_allowlist_filter_itself_is_unchanged(self) -> None:
        """I-6, read off the source rather than off behaviour: the two
        prefixes are the whole allowlist, and this plan widened neither."""
        import inspect

        source = inspect.getsource(family_max)
        assert 'startswith(("FIXED-GRID", "KMEANS-PRIOR"))' in source
        assert source.count("startswith") == 1


class TestThisParserSHelpDoesNotCrash:
    """``--help`` is the ONLY path that formats a ``help=`` string.

    A bare ``%`` in one of them makes ``python -m train.sam3.baselines --help``
    exit 1 (``TypeError: %o format: an integer is required, not dict``) while
    every other test in this file passes -- measured, on this very parser. The
    check is the same one ``test_train_sam3.py`` runs on the trainer's parser;
    it lives in ``parser_help_guard`` so there is ONE implementation and each
    parser's own test file calls it, rather than a copy per file that a new
    parser can be added without.
    """

    def test_no_help_string_carries_a_bare_percent(self) -> None:
        assert_no_bare_percent_help(build_parser(), "baselines.build_parser")

    def test_help_exits_zero(self) -> None:
        """The end-to-end proof: argparse actually FORMATS every help string."""
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["--help"])
        assert exit_info.value.code == 0


#: The stand-in ``distractor_gap`` reading the JSON-payload guard looks for.
#: Sentinel values, not plausible ones: a number that could have come from
#: anywhere proves nothing about WHICH dict reached the file.
_GAP_SENTINEL: Dict[str, float] = {
    "box_iou_prompted": 0.111111,
    "box_iou_distractor": 0.222222,
    "gap": -0.111111,
    "relative_gap": -1.0,
    "matched_pairs_prompted": 11.0,
    "matched_pairs_distractor": 22.0,
    "images_with_distractor": 33.0,
}

#: The stand-in ``prompt_swap_retention`` reading, same reasoning.
_SWAP_SENTINEL: Dict[str, float] = {
    "box_iou_true": 0.333333,
    "box_iou_worst_wrong_prompt": 0.444444,
    "retained": 1.333332,
    "matched_pairs": 44.0,
    "prompt_changed_fraction": 0.55,
    "rel_delta_pred_boxes": 6.6,
    "rel_delta_pred_logits": 7.7,
}


class TestTheJsonPayloadCarriesTheDiagnosticsAndNotOnlyTheFamily:
    """``--json`` must write the ``--distractor-gap`` / ``--prompt-swap``
    numbers, not only the family table.

    This is a WIRING guard for a defect that actually shipped and was caught in
    review: ``main`` built its payload from ``evaluate_family``'s results alone,
    so a run invoked as ``--distractor-gap ... --json out.json`` wrote a file
    with SEVEN family arms and ZERO gap numbers, while the plan naming that file
    as the gap's machine-readable evidence read as satisfied. The numbers
    survived only in the stdout log, which nothing parses. Nothing in the
    module raised, and every other test in this file stayed green -- the only
    observable is the CONTENT of the written file, which is what this pins.

    RED-proof: reverting ``main``'s ``if args.json`` block to the pre-fix
    ``payload = {str(seed): per_seed[seed] for seed in seeds}`` (the two loops'
    dicts never folded in) makes
    ``test_the_distractor_gap_block_writes_into_the_json`` fire at
    ``assert "distractor_gap" in payload["1"]``, and the prompt-swap twin fire
    at ``assert "prompt_swap" in payload["1"]``. Both were executed RED before
    this test was committed.

    Everything expensive is replaced: the family evaluation, the context, the
    split builder, ``keras.models.load_model``, and both diagnostics. That is
    deliberate -- the claim under test is "the value this function returned
    reached the file", and a real forward pass would only make it slower and
    make the sentinel unrecognisable.
    """

    @pytest.fixture
    def written_payload(self, tmp_path: Any,
                        monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
        """Run ``main`` end to end with every heavy call replaced, and return
        the parsed JSON it wrote."""
        import json as json_module

        family = {
            "_meta": {"pool_size": 10.0, "num_queries": 4.0},
            ORACLE_ARM: {"box_iou": 1.0, "matched": 5.0},
            DEGENERATE_ARM: {"box_iou": 0.01, "matched": 5.0},
            CONNECTED_COMPONENTS_ARM: {"box_iou": 0.9, "matched": 5.0},
            f"{GRID_ARM} 5x5 wh0.2": {"box_iou": 0.2, "matched": 5.0},
        }
        monkeypatch.setattr(
            baselines, "evaluate_family",
            lambda seed, ks: {k: dict(v) for k, v in family.items()})
        monkeypatch.setattr(
            baselines, "build_context", lambda seed: (object(), object()))
        monkeypatch.setattr(
            baselines, "build_split_dataset",
            lambda model, seed, train=False, include_all_instances=False:
            object())
        monkeypatch.setattr(baselines.keras.models, "load_model",
                            lambda path, compile=False: object())
        monkeypatch.setattr(baselines, "distractor_gap",
                            lambda *a, **kw: dict(_GAP_SENTINEL))
        monkeypatch.setattr(baselines, "prompt_swap_retention",
                            lambda *a, **kw: dict(_SWAP_SENTINEL))

        checkpoint = tmp_path / "ckpt_seed1.keras"
        checkpoint.write_text("not a real checkpoint -- never opened")
        out = tmp_path / "payload.json"
        template = str(tmp_path / "ckpt_seed{seed}.keras")

        code = baselines.main([
            "--seeds", "1",
            "--distractor-gap", template,
            "--prompt-swap", template,
            "--json", str(out)])
        assert code == 0
        return json_module.loads(out.read_text())

    def test_the_distractor_gap_block_writes_into_the_json(
            self, written_payload: Dict[str, Any]) -> None:
        """The firing assertion of the RED proof."""
        assert "distractor_gap" in written_payload["1"]
        assert written_payload["1"]["distractor_gap"] == _GAP_SENTINEL

    def test_the_prompt_swap_block_writes_into_the_json(
            self, written_payload: Dict[str, Any]) -> None:
        """The firing assertion of the RED proof, prompt-swap twin."""
        assert "prompt_swap" in written_payload["1"]
        assert written_payload["1"]["prompt_swap"] == _SWAP_SENTINEL

    def test_the_gap_is_keyed_per_seed_and_not_at_the_top_level(
            self, written_payload: Dict[str, Any]) -> None:
        """Ambiguity is the failure mode a flat key would reintroduce: three
        seeds' gaps under one key cannot be told apart."""
        assert "distractor_gap" not in written_payload
        assert "prompt_swap" not in written_payload
        assert set(written_payload) == {"1", "_family_max"}

    def test_the_family_rows_are_undisturbed(
            self, written_payload: Dict[str, Any]) -> None:
        """The fix is ADDITIVE: every key the file carried before is still
        there, with the same value."""
        row = written_payload["1"]
        assert row["_meta"] == {"pool_size": 10.0, "num_queries": 4.0}
        assert row[ORACLE_ARM] == {"box_iou": 1.0, "matched": 5.0}
        assert row[CONNECTED_COMPONENTS_ARM] == {"box_iou": 0.9,
                                                 "matched": 5.0}
        assert written_payload["_family_max"]["1"] == {
            "arm": f"{GRID_ARM} 5x5 wh0.2", "box_iou": 0.2}

    def test_the_two_diagnostics_never_reach_family_max(
            self, written_payload: Dict[str, Any]) -> None:
        """I-6 at the payload level: the bar quoted in the file is still the
        family's, and neither new key is arm-shaped enough to be picked up."""
        assert written_payload["_family_max"]["1"]["arm"].startswith(GRID_ARM)
        for key in ("distractor_gap", "prompt_swap"):
            assert not key.startswith(("FIXED-GRID", "KMEANS-PRIOR"))

    def test_without_the_two_flags_the_payload_is_exactly_as_before(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """No flag, no key -- an absent diagnostic must not become a null row
        that a consumer would read as 'measured, and it was nothing'."""
        import json as json_module

        monkeypatch.setattr(
            baselines, "evaluate_family",
            lambda seed, ks: {
                "_meta": {"pool_size": 10.0, "num_queries": 4.0},
                ORACLE_ARM: {"box_iou": 1.0, "matched": 5.0},
                DEGENERATE_ARM: {"box_iou": 0.01, "matched": 5.0},
                CONNECTED_COMPONENTS_ARM: {"box_iou": 0.9, "matched": 5.0},
                f"{GRID_ARM} 5x5 wh0.2": {"box_iou": 0.2, "matched": 5.0}})
        out = tmp_path / "family_only.json"
        assert baselines.main(["--seeds", "1", "--json", str(out)]) == 0
        payload = json_module.loads(out.read_text())
        assert "distractor_gap" not in payload["1"]
        assert "prompt_swap" not in payload["1"]
