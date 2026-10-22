"""SAM 1's ``build_sam_dataset`` states its length and redraws its prompts every pass.

The same two defects plan-2026-10-14T062616-81938e81 found in the SAM 2 pipeline
(D-258, D-259; ``tests/test_train/test_sam2/test_the_pipeline_states_its_length_and_redraws_its_prompts.py``),
in the pipeline SAM 2's prompt samplers come from (D-262):

1. The source is a ``from_generator`` dataset of unknown cardinality, so Keras 3.8
   logs every loss as ``n / (n + 1)`` of the mean over ``n`` batches.
2. The point is drawn by a stateful op in the ``map``; under the trainer's global
   seed it restarts on every iterator, so every epoch draws the same points.
"""

import warnings
from typing import Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.sam_mask_loss import SAMIoULoss, SAMMaskLoss
from dl_techniques.models.vision_language.sam.sam1.training_model import (
    INPUT_GT_MASK,
    INPUT_POINT_COORDS,
    IOU_SUPERVISION,
    LOW_RES_LOGITS,
)
from train.sam.data import build_sam_dataset

IMAGE_SIZE = 64
MASK_SIZE = 16
CONSTANT_LOGIT = -0.75


def _dataset(num_samples: int = 4, batch_size: int = 2, seed: int = 5, shuffle_buffer: int = 0):
    """As the trainer builds it: global seeds first, then the dataset."""
    keras.utils.set_random_seed(seed)
    return build_sam_dataset(
        num_samples=num_samples, image_size=IMAGE_SIZE, batch_size=batch_size,
        mask_size=MASK_SIZE, seed=seed, shuffle_buffer=shuffle_buffer)


class ConstantMaskStub(keras.Model):
    """Answers with one trainable constant logit; reads no prompt."""

    def __init__(self) -> None:
        super().__init__(name="constant_mask_stub")
        self.logit = self.add_weight(
            shape=(), initializer=keras.initializers.Constant(CONSTANT_LOGIT), name="logit")

    def call(self, inputs):
        gt = inputs[INPUT_GT_MASK]
        rows = keras.ops.max(gt, axis=(1, 2, 3))[:, None, None]
        return {
            LOW_RES_LOGITS: keras.ops.zeros_like(gt) + self.logit,
            IOU_SUPERVISION: keras.ops.concatenate([rows, rows], axis=-1) * 0.0,
        }


def _quietly(call, *args, **kwargs):
    """Ignore only Keras' epoch-end warning, so the logged VALUE is what is compared."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Your input ran out of data")
        return call(*args, **kwargs)


def _stub() -> ConstantMaskStub:
    model = ConstantMaskStub()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0),
                  loss={LOW_RES_LOGITS: SAMMaskLoss(), IOU_SUPERVISION: SAMIoULoss()},
                  jit_compile=False)
    return model


def _hand_mean(dataset) -> float:
    loss = SAMMaskLoss()
    values = []
    for _, targets in dataset:
        truth = targets[LOW_RES_LOGITS]
        values.append(float(loss(truth, keras.ops.full(truth.shape, CONSTANT_LOGIT))))
    return float(np.mean(values))


class TestTheLoggedLossIsTheMeanOverBatches:

    def test_the_dataset_states_its_batch_count(self) -> None:
        assert int(_dataset(num_samples=6, batch_size=2).cardinality()) == 3
        assert int(_dataset(num_samples=5, batch_size=2).cardinality()) == 3

    @pytest.mark.parametrize("num_samples", [2, 4])
    def test_evaluate_reports_the_mean_over_the_batches(self, num_samples: int) -> None:
        dataset = _dataset(num_samples=num_samples)
        expected = _hand_mean(dataset)
        logged = _quietly(_stub().evaluate, dataset, return_dict=True, verbose=0)
        assert logged[f"{LOW_RES_LOGITS}_loss"] == pytest.approx(expected, rel=1e-5)

    def test_fit_logs_the_mean_over_the_batches_every_epoch(self) -> None:
        dataset = _dataset(num_samples=2)
        expected = _hand_mean(dataset)
        history = _quietly(_stub().fit, dataset, epochs=3, verbose=0).history
        assert history[f"{LOW_RES_LOGITS}_loss"] == pytest.approx([expected] * 3, rel=1e-5)


def _points(dataset) -> Dict[bytes, Tuple[float, ...]]:
    points = {}
    for inputs, _ in dataset:
        for mask, point in zip(inputs[INPUT_GT_MASK].numpy(), inputs[INPUT_POINT_COORDS].numpy()):
            points[mask.tobytes()] = tuple(np.round(point.reshape(-1), 3).tolist())
    return points


def _passes(dataset, count: int) -> List[Dict[bytes, Tuple[float, ...]]]:
    return [_points(dataset) for _ in range(count)]


class TestThePointPromptIsRedrawnEveryPass:

    def test_two_passes_draw_different_points(self) -> None:
        first, second = _passes(_dataset(num_samples=4), 2)
        assert first.keys() == second.keys()
        assert any(first[sample] != second[sample] for sample in first)

    def test_a_shuffled_sample_gets_more_than_two_draws(self) -> None:
        passes = _passes(_dataset(num_samples=2, shuffle_buffer=2), 8)
        for sample in passes[0]:
            assert len({p[sample] for p in passes}) > 2, "the point stream restarts every pass"

    def test_the_draw_sequence_is_reproducible_from_the_seed(self) -> None:
        assert _passes(_dataset(seed=5), 3) == _passes(_dataset(seed=5), 3)
