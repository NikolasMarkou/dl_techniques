"""AccUNet's logits head memorises one batch under the trainer's loss.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 2.1 (D-144), the
D-122 training bar: stock ``fit`` with the segmentation combo loss on logits
(``LossConfig(from_logits=True)``, the trainer's compile) drives one fixed batch
well below its trivial baseline, for a binary map and for three classes.

The trivial baseline is the same loss on the constant logit 0 (probability 0.5
for one channel, uniform for three), measured in the test.

The batch is read back in training mode, as ``fit`` saw it. Measured
2026-10-17: after 300 full-batch steps the inference-mode output still scores
worse than the baseline (binary 0.887 against 0.448, pixel accuracy 0.89) while
the training-mode output scores 0.004 at accuracy 1.0. The BatchNormalization
moving statistics (momentum 0.99) have not converged to the one batch's
statistics; that is the normalisation's warm-up, not the head.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.segmentation_loss import LossConfig
from dl_techniques.losses.segmentation_wrapper_loss import SegmentationWrapperLoss
from dl_techniques.models.vision.accunet.model import (
    create_acc_unet_binary,
    create_acc_unet_multiclass,
)

SIZE = 32


def _batch(num_classes: int):
    rng = np.random.default_rng(0)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    x = rng.random((4, SIZE, SIZE, 3)).astype("float32") * 0.1
    labels = np.zeros((4, SIZE, SIZE), dtype=int)
    for i in range(4):
        cy, cx = rng.integers(8, SIZE - 8, size=2)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 36
        labels[i][disk] = 1
        x[i][disk] += 0.5
        if num_classes > 2:
            labels[i, :6, :] = 2
            x[i, :6, :, 1] += 0.5
    if num_classes == 1:
        return x, labels[..., None].astype("float32")
    return x, np.eye(num_classes, dtype="float32")[labels]


@pytest.mark.parametrize("num_classes", [1, 3])
def test_fit_on_logits_memorises_one_batch(num_classes):
    keras.utils.set_random_seed(0)
    x, y = _batch(num_classes)
    if num_classes == 1:
        model = create_acc_unet_binary(3, input_shape=(SIZE, SIZE), base_filters=8, mlfc_iterations=1)
    else:
        model = create_acc_unet_multiclass(
            3, num_classes, input_shape=(SIZE, SIZE), base_filters=8, mlfc_iterations=1)
    loss = SegmentationWrapperLoss("combo", LossConfig(num_classes=num_classes, from_logits=True))
    baseline = float(loss(y, np.zeros_like(y)))
    model.compile(optimizer=keras.optimizers.Adam(3e-3), loss=loss)
    history = model.fit(x, y, batch_size=4, epochs=150, shuffle=False, verbose=0)
    final = history.history["loss"][-1]
    assert final < 0.3 * baseline, (final, baseline)
    logits = keras.ops.convert_to_numpy(model(x, training=True))
    if num_classes == 1:
        predicted, truth = logits[..., 0] > 0.0, y[..., 0] > 0.5
    else:
        predicted, truth = logits.argmax(-1), y.argmax(-1)
    assert (predicted == truth).mean() > 0.95
