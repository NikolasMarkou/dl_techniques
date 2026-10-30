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

Why ``STEPS`` is 150 and not 20. MEASURED 2026-10-30 on CPU (no CUDA device on the
box), ``max(losses)`` over the four fixed clicks against the bar ``0.5 * constant``
(constants 3.38-3.88, bars 1.69-1.94), ``SAM2TrainingConfig`` seeds 40-47, one
fresh model per seed, worst seed of the eight and the number of seeds under the
bar. Every column is one run per seed fitted in chunks; the single 150-step ``fit``
this test makes was measured on HEAD too (worst seed 47: 0.374 against a bar of 1.771)::

    steps                  20     40     60     100    150
    HEAD   worst max       3.15   2.11   1.28   0.68   0.43
    HEAD   seeds passing   2/8    5/8    8/8    8/8    8/8
    before 43cf54658
           worst max       3.70   2.89   2.37   1.72   1.14
    before seeds passing   1/8    4/8    7/8    8/8    8/8

At 20 steps the bar was passed by the luck of the seed, not by the model: only
seed 42 passed before commit 43cf54658 swapped the Hiera and memory-attention
FFNs for ``MLPBlock`` (which draws its initializers in a different order), and
42 stopped passing after it, although the forward output on fixed weights, the
kernel initializer (glorot uniform, same std and bounds), the bias (zeros) and
the activation are unchanged. 150 steps clears the bar on every measured seed on
either side of that commit (by 4.7x on HEAD's worst seed, 1.7x on the old tree's
worst), and costs about 15 s on top of the compile-dominated 110 s the 20-step
run took.
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

STEPS = 150
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
