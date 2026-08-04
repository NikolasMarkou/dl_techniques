"""
Guards for `src/train/sam/data.py` -- the per-instance sources and the
``tf.data`` assembly.

This module exists so that an I/O or data-shape problem can never be mistaken
for a model problem: the synthetic source needs no COCO on disk and no
``pycocotools``, so a failing ``fit()`` over it is a failure of the model or
the loss, and nothing else.
"""

from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.models.sam import SAMTrainingModel
from dl_techniques.models.sam.training_model import (
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
    MASK_DIVISOR,
    MIN_MASK_PIXELS,
    PADDING_LABEL,
    RECORD_BOX,
    RECORD_IMAGE,
    RECORD_MASK,
    _box_from_mask,
    build_sam_dataset,
    sample_point_in_mask,
    synthetic_instance_samples,
)

from tests.test_models.test_sam.test_correctness import (
    GRID_SIZE,
    IMG_SIZE,
    build_reduced_sam,
    seed_nonzero_weights,
)

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
