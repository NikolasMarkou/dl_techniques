"""
Guards for `src/train/sam/data.py` -- the per-instance sources and the
``tf.data`` assembly.

This module exists so that an I/O or data-shape problem can never be mistaken
for a model problem: the synthetic source needs no COCO on disk and no
``pycocotools``, so a failing ``fit()`` over it is a failure of the model or
the loss, and nothing else.
"""

import os
from typing import Any, Dict, List, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.SAM.SAM1 import SAMTrainingModel
from dl_techniques.models.SAM.SAM1.training_model import (
    INPUT_BOXES,
    INPUT_GT_MASK,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
    IOU_SUPERVISION,
    LOW_RES_LOGITS,
    TRAINING_REFINEMENT_ROUNDS,
)
from dl_techniques.losses.sam_mask_loss import SAMIoULoss, SAMMaskLoss

from train.sam.data import (
    DATA_SOURCES,
    MASK_DIVISOR,
    MAX_JITTER_PIXELS,
    MIN_BOX_SIDE,
    MIN_MASK_PIXELS,
    PADDING_LABEL,
    RECORD_BOX,
    RECORD_IMAGE,
    RECORD_MASK,
    _box_from_mask,
    build_sam_dataset,
    coco_instance_samples,
    jitter_box,
    sample_point_in_mask,
    sample_point_outside_mask,
    synthetic_instance_samples,
)

from tests.test_models.test_sam.test_correctness import (
    GRID_SIZE,
    IMG_SIZE,
    build_reduced_sam,
    seed_nonzero_weights,
)


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `trainers/epoch_iterator.py:151`. These tests run the REAL trainer over
# a deliberately tiny synthetic corpus while `steps_per_epoch` comes from the
# shipped config, so the iterator is legitimately exhausted before the epoch
# ends. Padding the corpus to match would change what the test measures (the
# config -> `fit()` wiring), so the advisory is suppressed HERE only; a real
# starved input in any other module still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Your input ran out of data:UserWarning"),
]

LOW_RES = 4 * GRID_SIZE


class TestSyntheticSource:
    """The source's record contract, and that its masks are per-instance."""

    def test_record_shapes_and_dtypes(self) -> None:
        records = list(
            synthetic_instance_samples(4, image_size=IMG_SIZE, seed=0)
        )
        assert len(records) == 4
        for record in records:
            assert record[RECORD_IMAGE].shape == (IMG_SIZE, IMG_SIZE, 3)
            assert record[RECORD_IMAGE].dtype == np.float32
            assert record[RECORD_MASK].shape == (
                IMG_SIZE // MASK_DIVISOR,
                IMG_SIZE // MASK_DIVISOR,
            )
            assert record[RECORD_BOX].shape == (4,)

    def test_the_image_domain_is_0_255_not_0_1(self) -> None:
        """
        ``SAM.preprocess`` normalizes by ImageNet statistics in the 0-255
        domain. A [0, 1] image is a silent 255x under-exposure: every shape
        assertion passes and the model sees near-black.
        """
        records = list(
            synthetic_instance_samples(6, image_size=IMG_SIZE, seed=1)
        )
        peak = max(float(r[RECORD_IMAGE].max()) for r in records)
        assert peak > 100.0, f"brightest pixel over 6 samples was {peak}"
        assert all(float(r[RECORD_IMAGE].min()) >= 0.0 for r in records)
        assert all(float(r[RECORD_IMAGE].max()) <= 255.0 for r in records)

    def test_every_emitted_mask_is_non_empty(self) -> None:
        """The empty-after-resize edge case is dropped, never emitted."""
        records = list(
            synthetic_instance_samples(30, image_size=IMG_SIZE, seed=2)
        )
        sizes = [float(r[RECORD_MASK].sum()) for r in records]
        assert min(sizes) >= MIN_MASK_PIXELS, f"smallest mask was {min(sizes)}"

    def test_masks_are_per_instance_not_one_merged_foreground(self) -> None:
        """
        The probe a collapsed foreground map could not satisfy. Consecutive
        records from the same image must have DIFFERENT masks; a merged
        foreground would hand back the same mask for every instance.
        """
        records = list(
            synthetic_instance_samples(
                40, image_size=IMG_SIZE, max_instances=3, seed=3
            )
        )
        pairs = [
            (a, b)
            for a, b in zip(records, records[1:])
            if a[RECORD_IMAGE] is b[RECORD_IMAGE]
        ]
        assert pairs, "no image contributed two instances -- probe is vacuous"
        distinct = sum(
            1 for a, b in pairs if not np.array_equal(a[RECORD_MASK], b[RECORD_MASK])
        )
        assert distinct == len(pairs), (
            f"{len(pairs) - distinct} of {len(pairs)} same-image instance pairs "
            f"shared an identical mask -- these are not per-instance masks"
        )

    def test_the_box_bounds_its_own_mask(self) -> None:
        mask = np.zeros((32, 32), dtype="uint8")
        mask[5:12, 20:27] = 1
        assert list(_box_from_mask(mask)) == [20.0, 5.0, 27.0, 12.0]

    def test_an_empty_mask_has_no_box(self) -> None:
        with pytest.raises(ValueError, match="EMPTY mask"):
            _box_from_mask(np.zeros((8, 8), dtype="uint8"))

    def test_the_same_seed_reproduces_the_same_records(self) -> None:
        first = list(synthetic_instance_samples(5, image_size=IMG_SIZE, seed=7))
        second = list(synthetic_instance_samples(5, image_size=IMG_SIZE, seed=7))
        other = list(synthetic_instance_samples(5, image_size=IMG_SIZE, seed=8))
        assert all(
            np.array_equal(a[RECORD_MASK], b[RECORD_MASK])
            for a, b in zip(first, second)
        )
        assert not all(
            np.array_equal(a[RECORD_MASK], b[RECORD_MASK])
            for a, b in zip(first, other)
        )

    def test_a_mask_size_that_does_not_divide_the_image_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive divisor"):
            next(
                synthetic_instance_samples(
                    1, image_size=IMG_SIZE, mask_size=IMG_SIZE // 3 + 1
                )
            )


class TestInitialPointPrompt:
    """The pipeline's point prompt lands inside the mask it was drawn from."""

    def test_the_point_is_inside_the_mask(self) -> None:
        mask = np.zeros((LOW_RES, LOW_RES), dtype="float32")
        mask[10:30, 40:55] = 1.0
        coords, labels = sample_point_in_mask(
            tf.constant(mask), image_size=IMG_SIZE
        )
        scale = IMG_SIZE / LOW_RES
        x, y = np.asarray(coords)[0]
        col = int(round((x + 0.5) / scale - 0.5))
        row = int(round((y + 0.5) / scale - 0.5))
        assert mask[row, col] > 0
        assert int(np.asarray(labels)[0]) == 1

    def test_an_empty_mask_yields_a_padding_label(self) -> None:
        _, labels = sample_point_in_mask(
            tf.zeros((LOW_RES, LOW_RES)), image_size=IMG_SIZE
        )
        assert int(np.asarray(labels)[0]) == PADDING_LABEL

    def test_a_non_empty_mask_does_not(self) -> None:
        """Control: a sampler labelling everything -1 would pass the test
        above."""
        mask = np.zeros((LOW_RES, LOW_RES), dtype="float32")
        mask[0:4, 0:4] = 1.0
        _, labels = sample_point_in_mask(tf.constant(mask), image_size=IMG_SIZE)
        assert int(np.asarray(labels)[0]) == 1


class TestDatasetAssembly:
    """The dataset is shaped exactly as ``SAMTrainingModel.fit`` wants it."""

    def test_batch_structure_and_shapes(self) -> None:
        dataset = build_sam_dataset(
            num_samples=6, image_size=IMG_SIZE, batch_size=2
        )
        inputs, targets = next(iter(dataset))
        assert set(targets) == {LOW_RES_LOGITS, IOU_SUPERVISION}
        assert tuple(inputs[INPUT_IMAGE].shape) == (2, IMG_SIZE, IMG_SIZE, 3)
        assert tuple(inputs[INPUT_POINT_COORDS].shape) == (2, 1, 2)
        assert tuple(inputs[INPUT_POINT_LABELS].shape) == (2, 1)
        assert tuple(inputs[INPUT_GT_MASK].shape) == (2, 1, LOW_RES, LOW_RES)
        assert tuple(targets[LOW_RES_LOGITS].shape) == (2, 1, LOW_RES, LOW_RES)
        assert tuple(targets[IOU_SUPERVISION].shape) == (2, 1, 2)

    def test_the_gt_target_is_single_instance_whatever_the_round_count(
        self,
    ) -> None:
        """
        The pipeline must NOT know about ``num_refinement_rounds``. Its target
        mask axis stays 1; ``SAMMaskLoss`` repeats it.
        """
        dataset = build_sam_dataset(
            num_samples=4, image_size=IMG_SIZE, batch_size=2
        )
        _, targets = next(iter(dataset))
        assert tuple(targets[LOW_RES_LOGITS].shape)[1] == 1
        assert TRAINING_REFINEMENT_ROUNDS > 1

    def test_the_dataset_yields_the_number_of_records_requested(self) -> None:
        dataset = build_sam_dataset(
            num_samples=7, image_size=IMG_SIZE, batch_size=1
        )
        assert sum(1 for _ in dataset) == 7


class TestEndToEndFitOverTheSyntheticSource:
    """
    A real ``fit()``. A green test suite is not evidence a training path works
    (LESSONS: 249 tests green while both trainers were broken); this class runs
    the actual thing, at both round counts.
    """

    @staticmethod
    def _model(rounds: int) -> SAMTrainingModel:
        keras.utils.set_random_seed(5)
        model = SAMTrainingModel(
            build_reduced_sam(), num_refinement_rounds=rounds
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
            loss_weights={LOW_RES_LOGITS: 1.0, IOU_SUPERVISION: 1.0},
        )
        return model

    @pytest.mark.parametrize("rounds", [1, TRAINING_REFINEMENT_ROUNDS])
    def test_fit_runs_and_the_loss_is_finite(self, rounds: int) -> None:
        model = self._model(rounds)
        dataset = build_sam_dataset(
            num_samples=8, image_size=IMG_SIZE, batch_size=2, seed=11
        )
        history = model.fit(dataset, epochs=2, verbose=0)
        losses = history.history["loss"]
        assert len(losses) == 2
        assert all(np.isfinite(losses)), losses

    def test_the_dead_component_probe_turns_the_dataset_path_red(self) -> None:
        """
        The green above is not vacuous: with every output detached, the same
        dataset must produce ``No gradients provided for any variable``.
        """
        from tests.test_models.test_sam.dead_component_oracle import (
            NO_GRADIENTS_MESSAGE,
            outputs_stop_gradient,
        )

        model = self._model(1)
        dataset = build_sam_dataset(
            num_samples=4, image_size=IMG_SIZE, batch_size=2, seed=12
        )
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                model.fit(dataset, epochs=1, verbose=0)

    def test_the_loss_decreases_over_a_short_overfit_run(self) -> None:
        """
        Not an accuracy claim -- a wiring claim. Repeatedly fitting ONE tiny
        batch must reduce its own loss; a pipeline that fed mismatched
        image/mask pairs would keep the loss flat while every shape assertion
        still passed.
        """
        model = self._model(1)
        dataset = build_sam_dataset(
            num_samples=2, image_size=IMG_SIZE, batch_size=2, seed=13
        ).cache().repeat(12)
        history = model.fit(dataset, epochs=1, verbose=0)
        first = float(history.history["loss"][0])
        after = float(model.evaluate(dataset.take(1), verbose=0)[0])
        assert after < first, f"loss did not fall: {first} -> {after}"


# ---------------------------------------------------------------------------
# Plan step 6 -- the COCO arm.
# ---------------------------------------------------------------------------
COCO_ANNOTATIONS = os.path.join(
    "/media/arxwn/data0_4tb/datasets/coco_2017", "annotations",
    "instances_val2017.json",
)
requires_coco = pytest.mark.skipif(
    not os.path.exists(COCO_ANNOTATIONS),
    reason=f"local COCO 2017 not found at {COCO_ANNOTATIONS}",
)


class TestSourceRegistry:
    """Both sources are reachable by name, and an unknown name is refused."""

    def test_both_sources_are_registered(self) -> None:
        assert set(DATA_SOURCES) == {"synthetic", "coco"}

    def test_an_unknown_source_is_refused_by_name(self) -> None:
        """
        A silently-defaulted source is how a smoke number gets quoted as a
        real-data number.
        """
        with pytest.raises(ValueError, match="unknown data source"):
            build_sam_dataset(
                num_samples=1, image_size=IMG_SIZE, batch_size=1, source="cocoa"
            )


@requires_coco
class TestCocoSource:
    """The COCO arm emits the SAME record contract as the synthetic one."""

    def test_records_match_the_shared_contract(self) -> None:
        records = list(
            coco_instance_samples(
                6, image_size=IMG_SIZE, split="val2017", max_images=32, seed=0
            )
        )
        assert len(records) == 6
        for record in records:
            assert record[RECORD_IMAGE].shape == (IMG_SIZE, IMG_SIZE, 3)
            assert record[RECORD_MASK].shape == (
                IMG_SIZE // MASK_DIVISOR,
                IMG_SIZE // MASK_DIVISOR,
            )
            assert record[RECORD_MASK].sum() > 0
            assert record[RECORD_BOX].shape == (4,)

    def test_the_image_domain_is_0_255_not_the_loader_default_0_1(self) -> None:
        """
        The loader's own default is ``pixel_scale = 1/255``. ``SAM.preprocess``
        normalizes in the 0-255 domain, so inheriting that default would hand
        the model a 255x under-exposed image with every shape assertion green.
        """
        records = list(
            coco_instance_samples(
                4, image_size=IMG_SIZE, split="val2017", max_images=32, seed=0
            )
        )
        assert max(float(r[RECORD_IMAGE].max()) for r in records) > 100.0

    def test_the_box_is_in_image_pixels_not_normalized(self) -> None:
        """
        ``_build_instances`` returns a box normalized to ``[0, 1]`` (the
        detection head's frame); a SAM prompt lives in image pixels. A missed
        rescale would put every box in the top-left 1x1 pixel, invisibly to
        every shape assertion.
        """
        records = list(
            coco_instance_samples(
                8, image_size=IMG_SIZE, split="val2017", max_images=32, seed=0
            )
        )
        widest = max(
            float(r[RECORD_BOX][2] - r[RECORD_BOX][0]) for r in records
        )
        assert widest > 1.0, f"widest box was {widest} px -- still normalized?"
        assert all(float(r[RECORD_BOX].max()) <= IMG_SIZE + 1e-3 for r in records)

    def test_a_fit_step_runs_over_the_coco_arm(self) -> None:
        keras.utils.set_random_seed(5)
        model = SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=1)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
        )
        dataset = build_sam_dataset(
            num_samples=4,
            image_size=IMG_SIZE,
            batch_size=2,
            source="coco",
            source_kwargs={"split": "val2017", "max_images": 32},
        )
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][-1])


# ---------------------------------------------------------------------------
# Plan step 7 -- prompt-sampling PROPERTIES over >=200 samples.
#
# "0 violations over N rows" is worthless if the corpus was swept rather than
# attacked, and every containment check here is also satisfied by a CONSTANT
# sampler -- so each property class ships a degeneracy check and a deliberately
# broken control beside it.
# ---------------------------------------------------------------------------
PROPERTY_SAMPLES = 240


def _property_masks(count: int, seed: int) -> List[np.ndarray]:
    """Real instance masks from the synthetic source, not hand-drawn boxes."""
    return [
        record[RECORD_MASK]
        for record in synthetic_instance_samples(
            count, image_size=IMG_SIZE, max_instances=3, seed=seed
        )
    ]


def _cell_of(coord: np.ndarray) -> Tuple[int, int]:
    """Invert the pipeline's coordinate convention back to a mask cell."""
    scale = IMG_SIZE / LOW_RES
    x, y = float(coord[0]), float(coord[1])
    return int(round((y + 0.5) / scale - 0.5)), int(round((x + 0.5) / scale - 0.5))


class TestPointPromptProperties:
    """Foreground points inside, background points outside, over 240 masks."""

    def test_every_foreground_point_is_inside_its_own_mask(self) -> None:
        masks = _property_masks(PROPERTY_SAMPLES, seed=21)
        violations = []
        cells = set()
        for index, mask in enumerate(masks):
            coords, labels = sample_point_in_mask(
                tf.constant(mask), image_size=IMG_SIZE
            )
            row, col = _cell_of(np.asarray(coords)[0])
            cells.add((row, col))
            if mask[row, col] <= 0 or int(np.asarray(labels)[0]) != 1:
                violations.append((index, row, col))
        assert len(masks) == PROPERTY_SAMPLES
        assert violations == [], f"{len(violations)} of {PROPERTY_SAMPLES}: {violations[:5]}"
        # Degeneracy check: a sampler always returning one pixel satisfies the
        # containment assertion above for any mask that happens to contain it.
        assert len(cells) > PROPERTY_SAMPLES // 4, (
            f"only {len(cells)} distinct cells over {PROPERTY_SAMPLES} masks"
        )

    def test_every_background_point_is_outside_its_own_mask(self) -> None:
        masks = _property_masks(PROPERTY_SAMPLES, seed=22)
        violations = []
        cells = set()
        for index, mask in enumerate(masks):
            coords, labels = sample_point_outside_mask(
                tf.constant(mask), image_size=IMG_SIZE
            )
            row, col = _cell_of(np.asarray(coords)[0])
            cells.add((row, col))
            if mask[row, col] > 0 or int(np.asarray(labels)[0]) != 0:
                violations.append((index, row, col))
        assert violations == [], f"{len(violations)} of {PROPERTY_SAMPLES}: {violations[:5]}"
        assert len(cells) > PROPERTY_SAMPLES // 4

    def test_a_deliberately_broken_sampler_DOES_violate(self) -> None:
        """
        The control that makes the two properties above mean something. A
        sampler ignoring the mask (uniform over the whole grid) must be caught
        by the very same check, or the check proves nothing.
        """
        masks = _property_masks(PROPERTY_SAMPLES, seed=23)
        rng = np.random.default_rng(0)
        violations = 0
        for mask in masks:
            row = int(rng.integers(0, LOW_RES))
            col = int(rng.integers(0, LOW_RES))
            if mask[row, col] <= 0:
                violations += 1
        assert violations > PROPERTY_SAMPLES // 2, (
            f"the mask-ignoring sampler violated only {violations} times -- the "
            f"masks are so large that containment is nearly free, so the "
            f"properties above are weak evidence"
        )

    def test_a_fully_covered_mask_yields_a_padding_background_label(self) -> None:
        """The mirror of the empty-mask edge case, on the background side."""
        _, labels = sample_point_outside_mask(
            tf.ones((LOW_RES, LOW_RES)), image_size=IMG_SIZE
        )
        assert int(np.asarray(labels)[0]) == PADDING_LABEL


class TestBoxJitterProperties:
    """Bounded, non-inverted, in-frame -- over 240 real instance boxes."""

    @staticmethod
    def _boxes(count: int, seed: int) -> List[np.ndarray]:
        return [
            record[RECORD_BOX]
            for record in synthetic_instance_samples(
                count, image_size=IMG_SIZE, max_instances=3, seed=seed
            )
        ]

    def test_the_ground_truth_box_contains_its_own_mask(self) -> None:
        """
        Checked BEFORE jitter, because jitter deliberately breaks containment --
        that is what makes it noise. Asserting containment on the jittered box
        would be asserting something false.
        """
        records = list(
            synthetic_instance_samples(
                PROPERTY_SAMPLES, image_size=IMG_SIZE, max_instances=3, seed=24
            )
        )
        scale = IMG_SIZE / LOW_RES
        violations = []
        for index, record in enumerate(records):
            x1, y1, x2, y2 = record[RECORD_BOX]
            rows = np.flatnonzero(record[RECORD_MASK].any(axis=1))
            cols = np.flatnonzero(record[RECORD_MASK].any(axis=0))
            # +/- one downsampled cell of slack: the box is measured at full
            # resolution, the mask at `low_res_logits` resolution.
            if not (
                cols[0] * scale >= x1 - scale
                and (cols[-1] + 1) * scale <= x2 + scale
                and rows[0] * scale >= y1 - scale
                and (rows[-1] + 1) * scale <= y2 + scale
            ):
                violations.append((index, record[RECORD_BOX]))
        assert len(records) == PROPERTY_SAMPLES
        assert violations == [], f"{len(violations)}: {violations[:3]}"

    def test_jitter_stays_within_the_cap_and_never_inverts(self) -> None:
        boxes = self._boxes(PROPERTY_SAMPLES, seed=25)
        # Derived bound: each coordinate is offset by at most
        # MAX_JITTER_PIXELS, then the far corner may be raised by at most
        # MIN_BOX_SIDE more to enforce ordering.
        cap = MAX_JITTER_PIXELS + MIN_BOX_SIDE
        worst = 0.0
        moved = 0
        for box in boxes:
            jittered = np.asarray(jitter_box(tf.constant(box), IMG_SIZE))
            deviation = float(np.max(np.abs(jittered - box)))
            worst = max(worst, deviation)
            moved += int(deviation > 0.0)
            assert deviation <= cap + 1e-4, f"{box} -> {jittered} ({deviation})"
            assert jittered[2] > jittered[0], f"inverted in x: {jittered}"
            assert jittered[3] > jittered[1], f"inverted in y: {jittered}"
            assert float(jittered.min()) >= 0.0
            assert float(jittered.max()) <= IMG_SIZE + 1e-4
        # Non-degeneracy: a jitter that returned the box unchanged would pass
        # every assertion above.
        assert moved == PROPERTY_SAMPLES, f"{moved}/{PROPERTY_SAMPLES} moved"
        assert worst > 1.0, f"largest deviation over {PROPERTY_SAMPLES} was {worst}"

    def test_the_std_scales_with_the_box_side(self) -> None:
        """
        The 10%-of-side rule is observable, not merely written down: a large
        box must be jittered more than a tiny one. A constant-std implementation
        passes every bound assertion above.
        """
        small = np.asarray([10.0, 10.0, 14.0, 14.0], dtype="float32")
        large = np.asarray(
            [10.0, 10.0, 10.0 + IMG_SIZE * 0.7, 10.0 + IMG_SIZE * 0.7],
            dtype="float32",
        )

        def spread(box: np.ndarray) -> float:
            draws = [
                np.asarray(jitter_box(tf.constant(box), IMG_SIZE)) - box
                for _ in range(200)
            ]
            return float(np.std(np.stack(draws)))

        small_spread, large_spread = spread(small), spread(large)
        assert large_spread > 3.0 * small_spread, (
            f"small-box spread {small_spread:.4f} vs large-box {large_spread:.4f}"
        )

    def test_the_cap_actually_binds_on_a_huge_box(self) -> None:
        """
        Attack the cap rather than sweep it: at 10% of a 224 px side the
        unclipped std would be 22.4 px, so the cap must bite.
        """
        box = np.asarray([0.0, 0.0, float(IMG_SIZE), float(IMG_SIZE)], dtype="float32")
        deviations = [
            float(np.max(np.abs(np.asarray(jitter_box(tf.constant(box), IMG_SIZE)) - box)))
            for _ in range(200)
        ]
        assert max(deviations) <= MAX_JITTER_PIXELS + MIN_BOX_SIDE + 1e-4
        assert max(deviations) > 10.0, "the cap was never approached"


class TestBoxAndBackgroundPointsReachTheModel:
    """The pipeline options are wired through, not merely implemented."""

    def test_the_dataset_emits_a_box_and_background_points_when_asked(
        self,
    ) -> None:
        dataset = build_sam_dataset(
            num_samples=4,
            image_size=IMG_SIZE,
            batch_size=2,
            num_background_points=2,
            include_box=True,
            seed=31,
        )
        inputs, _ = next(iter(dataset))
        assert tuple(inputs[INPUT_POINT_COORDS].shape) == (2, 3, 2)
        assert tuple(inputs[INPUT_BOXES].shape) == (2, 1, 4)
        labels = np.asarray(inputs[INPUT_POINT_LABELS])
        assert list(labels[0]) == [1, 0, 0]

    def test_the_defaults_emit_neither(self) -> None:
        """Control: without it, the test above could pass on a pipeline that
        always emitted them."""
        dataset = build_sam_dataset(
            num_samples=2, image_size=IMG_SIZE, batch_size=2, seed=31
        )
        inputs, _ = next(iter(dataset))
        assert INPUT_BOXES not in inputs
        assert tuple(inputs[INPUT_POINT_COORDS].shape) == (2, 1, 2)

    def test_a_fit_step_runs_with_a_box_and_background_points(self) -> None:
        keras.utils.set_random_seed(5)
        model = SAMTrainingModel(build_reduced_sam(), num_refinement_rounds=1)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
        )
        dataset = build_sam_dataset(
            num_samples=4,
            image_size=IMG_SIZE,
            batch_size=2,
            num_background_points=1,
            include_box=True,
            seed=32,
        )
        history = model.fit(dataset, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"][-1])
