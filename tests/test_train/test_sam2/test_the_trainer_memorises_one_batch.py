"""The trainer's own model, data and compile memorise one batch below the best constant.

D-122 item 9 of plan-2026-10-14T062616-81938e81: "trains" means beating a trivial
baseline or memorising one batch, not "the loss moved". Before D-260 the trainer's
SAM 2 fed ``[0, 255]`` pixels to Hiera and its 2-clip memorisation stayed at or
above the best constant logit (0.959) for 150 steps; its logged loss looked
memorised only because the generator dataset halved it (D-258).

Scored as the bar is scored: hand-computed mask loss of an INFERENCE-mode call on
arrays (no Keras metric bookkeeping), on fixed frame-0 clicks that are not the
pipeline's (numpy draws, one foreground cell of frame 0 per clip). MEASURED on the
RTX 4070 (scratchpad ``s128/m2_fixed_200``): 0.31 after 20 steps, 0.028 after 100,
against 0.959; with the standardization removed the same 20 steps stay above the
constant.
"""

import keras
import numpy as np

from dl_techniques.models.vision_language.sam.sam2.training_model import (
    INPUT_GT_MASKS,
    INPUT_IMAGE,
    INPUT_POINT_COORDS,
    INPUT_POINT_LABELS,
)
from train.common import set_seeds
from train.sam2.train_sam2 import (
    SAM2TrainingConfig,
    create_dataset,
    create_sam2_training_model,
)

STEPS = 20
DRAWS = 4


def _fixed_clicks(masks: np.ndarray, image_size: int, draw: int) -> np.ndarray:
    """One foreground cell of frame 0 per clip, mapped as the pipeline maps cells."""
    rng = np.random.default_rng(1280 + draw)
    batch, _, height, width = masks.shape
    points = []
    for clip in range(batch):
        rows, cols = np.nonzero(masks[clip, 0] > 0.5)
        pick = rng.integers(0, len(rows))
        points.append([[(cols[pick] + 0.5) * image_size / width - 0.5,
                        (rows[pick] + 0.5) * image_size / height - 0.5]])
    return np.asarray(points, "float32")


def test_the_trainers_model_memorises_one_batch_below_the_best_constant() -> None:
    config = SAM2TrainingConfig(num_clips_train=2, num_clips_val=2, batch_size=2, seed=42)
    set_seeds(config.seed)
    train_dataset, _ = create_dataset(config)
    model = create_sam2_training_model(config)
    model.fit(train_dataset.repeat(), steps_per_epoch=STEPS, epochs=1, verbose=0)

    inputs, _ = next(iter(train_dataset))
    images = inputs[INPUT_IMAGE].numpy()
    masks = inputs[INPUT_GT_MASKS].numpy()
    losses = []
    for draw in range(DRAWS):
        outputs = model({
            INPUT_IMAGE: images,
            INPUT_POINT_COORDS: _fixed_clicks(masks, config.image_size, draw),
            INPUT_POINT_LABELS: np.ones((masks.shape[0], 1), "int32"),
            INPUT_GT_MASKS: masks,
        }, training=False)
        losses.append(float(model.mask_loss(masks, outputs["low_res_logits"])))
    constant = min(
        float(model.mask_loss(masks, np.full(masks.shape, logit, "float32")))
        for logit in np.arange(-12.0, 12.0001, 0.25))
    assert max(losses) < 0.5 * constant, (
        f"after {STEPS} steps the mask loss over {DRAWS} fixed clicks is {losses}; "
        f"the best constant logit scores {constant:.4f}")
