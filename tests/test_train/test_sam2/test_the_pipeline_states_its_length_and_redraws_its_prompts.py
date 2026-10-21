"""Guards for two properties of ``build_sam2_video_dataset`` that a loss curve cannot show.

1. **The dataset states its length, so ``fit``/``evaluate`` report the mean over its
   batches.** The clip source is a ``tf.data.Dataset.from_generator`` (unknown
   cardinality). Keras 3.8 then learns the epoch end from an ``OutOfRange`` raised
   inside one extra ``test_function``/``train_function`` call, and in that call the
   loss trackers' data-independent ``count`` update still runs: every logged loss
   is ``n / (n + 1)`` of the true mean over ``n`` batches. MEASURED on the step-12.6
   memorisation run (plan-2026-10-14T062616-81938e81 D-258): one batch per epoch,
   logged mask loss 0.477, the same checkpoint scored 0.954 on the same batch.
2. **The frame-0 prompt is redrawn on every pass, reproducibly from the seed.** A
   stateful ``tf.random`` op in the ``map`` restarts from its seed on every new
   iterator once a global seed is set (the trainer always sets one), so every
   epoch drew the same sequence; with a shuffle only the assignment of draws to
   clips changed (MEASURED: two point configurations per clip over all of
   training, D-258). Without a global seed the same op draws irreproducibly.
"""

import warnings
from typing import Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.sam2_video_loss import SAM2GatedMaskLoss
from dl_techniques.losses.sam_mask_loss import SAMIoULoss
from dl_techniques.models.vision_language.sam.sam2.training_model import (
    INPUT_GT_MASKS,
    INPUT_POINT_COORDS,
    SAM2_IOU_SUPERVISION,
    SAM2_LOW_RES_LOGITS,
    SAM2_OBJECT_SCORE_LOGITS,
)
from train.sam2.data import build_sam2_video_dataset

IMAGE_SIZE = 64
MASK_SIZE = 16
NUM_FRAMES = 4
CONSTANT_LOGIT = -0.75


def _dataset(num_clips: int = 4, batch_size: int = 2, seed: int = 5, shuffle_buffer: int = 0):
    """The pipeline as the trainer builds it: global seeds set first (``main`` calls
    ``set_seeds`` before ``create_dataset``). Without a global seed a stateful op
    draws irreproducibly instead of repeating, so the seed is part of the condition.
    """
    keras.utils.set_random_seed(seed)
    return build_sam2_video_dataset(
        num_clips=num_clips,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        mask_size=MASK_SIZE,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )


class ConstantMaskStub(keras.Model):
    """Answers every frame with one trainable constant logit; reads no prompt."""

    def __init__(self) -> None:
        super().__init__(name="constant_mask_stub")
        self.logit = self.add_weight(
            shape=(), initializer=keras.initializers.Constant(CONSTANT_LOGIT), name="logit")

    def call(self, inputs: Dict[str, keras.KerasTensor]) -> Dict[str, keras.KerasTensor]:
        gt = inputs[INPUT_GT_MASKS]
        frames = keras.ops.max(gt, axis=(2, 3))[..., None]
        return {
            SAM2_LOW_RES_LOGITS: keras.ops.zeros_like(gt) + self.logit,
            SAM2_OBJECT_SCORE_LOGITS: keras.ops.zeros_like(frames) + self.logit,
            SAM2_IOU_SUPERVISION: keras.ops.concatenate([frames, frames], axis=-1) * 0.0,
        }


def _compiled_stub() -> ConstantMaskStub:
    model = ConstantMaskStub()
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.0),
        loss={
            SAM2_LOW_RES_LOGITS: SAM2GatedMaskLoss(),
            SAM2_OBJECT_SCORE_LOGITS: keras.losses.BinaryCrossentropy(from_logits=True),
            SAM2_IOU_SUPERVISION: SAMIoULoss(),
        },
        jit_compile=False,
    )
    return model


def _quietly(call, *args, **kwargs):
    """Run ``call`` with Keras' "input ran out of data" warning ignored.

    The warning is the epoch-end mechanism's own symptom; this suite's warning
    filter would turn it into an error before the logged VALUE is compared, and
    the value is what this guard is about.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Your input ran out of data")
        return call(*args, **kwargs)


def _hand_mean(dataset) -> float:
    """The mask loss of the constant answer, averaged over the dataset's batches."""
    loss = SAM2GatedMaskLoss()
    values = []
    for _, targets in dataset:
        truth = targets[SAM2_LOW_RES_LOGITS]
        values.append(float(loss(truth, keras.ops.full(truth.shape, CONSTANT_LOGIT))))
    return float(np.mean(values))


class TestTheLoggedLossIsTheMeanOverBatches:
    """``fit`` and ``evaluate`` report the true mean, not ``n / (n + 1)`` of it."""

    def test_the_dataset_states_its_batch_count(self) -> None:
        assert int(_dataset(num_clips=6, batch_size=2).cardinality()) == 3
        assert int(_dataset(num_clips=5, batch_size=2).cardinality()) == 2

    @pytest.mark.parametrize("num_clips", [2, 4])
    def test_evaluate_reports_the_mean_over_the_batches(self, num_clips: int) -> None:
        dataset = _dataset(num_clips=num_clips)
        expected = _hand_mean(dataset)
        logged = _quietly(_compiled_stub().evaluate, dataset, return_dict=True, verbose=0)
        assert logged[f"{SAM2_LOW_RES_LOGITS}_loss"] == pytest.approx(expected, rel=1e-5)

    def test_fit_logs_the_mean_over_the_batches_every_epoch(self) -> None:
        dataset = _dataset(num_clips=2)
        expected = _hand_mean(dataset)
        history = _quietly(_compiled_stub().fit, dataset, epochs=3, verbose=0).history
        assert history[f"{SAM2_LOW_RES_LOGITS}_loss"] == pytest.approx([expected] * 3, rel=1e-5)


def _points_by_clip(dataset) -> Dict[bytes, Tuple[float, ...]]:
    """One pass: each clip (keyed by its ground truth bytes) to its frame-0 point."""
    points = {}
    for inputs, _ in dataset:
        masks = inputs[INPUT_GT_MASKS].numpy()
        coords = inputs[INPUT_POINT_COORDS].numpy()
        for mask, point in zip(masks, coords):
            points[mask.tobytes()] = tuple(np.round(point.reshape(-1), 3).tolist())
    return points


def _passes(dataset, count: int) -> List[Dict[bytes, Tuple[float, ...]]]:
    return [_points_by_clip(dataset) for _ in range(count)]


def _inside_frame_zero(dataset) -> bool:
    scale = IMAGE_SIZE / MASK_SIZE
    for inputs, _ in dataset:
        masks = inputs[INPUT_GT_MASKS].numpy()
        for mask, point in zip(masks, inputs[INPUT_POINT_COORDS].numpy()):
            col = int(round((float(point[0, 0]) + 0.5) / scale - 0.5))
            row = int(round((float(point[0, 1]) + 0.5) / scale - 0.5))
            if mask[0, row, col] <= 0.0:
                return False
    return True


class TestTheFrameZeroPromptIsRedrawnEveryPass:
    """The trainer sees fresh clicks each epoch, and the sequence is the seed's."""

    def test_two_passes_draw_different_points(self) -> None:
        first, second = _passes(_dataset(num_clips=4), 2)
        assert first.keys() == second.keys()
        assert any(first[clip] != second[clip] for clip in first)

    def test_a_shuffled_clip_gets_more_than_the_two_draws_a_restarting_stream_gives(self) -> None:
        passes = _passes(_dataset(num_clips=2, shuffle_buffer=2), 8)
        for clip in passes[0]:
            assert len({p[clip] for p in passes}) > 2, "the point stream restarts every pass"

    def test_the_draw_sequence_is_reproducible_from_the_seed(self) -> None:
        assert _passes(_dataset(seed=5), 3) == _passes(_dataset(seed=5), 3)

    def test_every_redrawn_point_is_inside_frame_zeros_mask(self) -> None:
        dataset = _dataset(num_clips=4)
        assert all(_inside_frame_zero(dataset) for _ in range(4))
