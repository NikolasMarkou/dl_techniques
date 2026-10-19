"""AccUNet's logits head memorises one batch under the trainer's loss.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 2.1 (D-144), the
D-122 training bar: stock ``fit`` with the segmentation combo loss on logits
(``LossConfig(from_logits=True)``, the trainer's compile) drives one fixed batch
well below its trivial baseline, for a binary map and for three classes.

The trivial baseline is the same loss on the constant logit 0 (probability 0.5
for one channel, uniform for three), measured in the test.

The batch is read back in training mode, as ``fit`` saw it, and in inference
mode (D-156). Until 2026-10-19 the inference-mode output collapsed: after 300
full-batch steps it scored worse than the baseline (binary 0.887 against 0.448,
pixel accuracy 0.89, every pixel background) while the training-mode output
scored 0.004 at accuracy 1.0. The cause was the BatchNormalization moving
statistics at Keras' momentum 0.99, not the reference's 0.9 (PyTorch 0.1), so
the inference arm requires every class to be predicted and the accuracy to
beat the majority class.
"""

import functools

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


@functools.lru_cache(maxsize=None)
def _fitted(num_classes: int):
    """Fit once per class count; both tests read the same fitted model."""
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
    return model, x, y, baseline, history.history["loss"][-1]


def _labels(logits, y, num_classes):
    if num_classes == 1:
        return (logits[..., 0] > 0.0).astype(int), (y[..., 0] > 0.5).astype(int)
    return logits.argmax(-1), y.argmax(-1)


@pytest.mark.parametrize("num_classes", [1, 3])
def test_fit_on_logits_memorises_one_batch(num_classes):
    model, x, y, baseline, final = _fitted(num_classes)
    assert final < 0.3 * baseline, (final, baseline)
    predicted, truth = _labels(
        keras.ops.convert_to_numpy(model(x, training=True)), y, num_classes)
    assert (predicted == truth).mean() > 0.95


@pytest.mark.parametrize("num_classes", [1, 3])
def test_inference_mode_does_not_collapse_to_one_class(num_classes):
    """D-156: the memorised batch, read in inference mode, keeps every class."""
    model, x, y, _, _ = _fitted(num_classes)
    predicted, truth = _labels(
        keras.ops.convert_to_numpy(model(x, training=False)), y, num_classes)
    majority = np.bincount(truth.ravel()).max() / truth.size
    assert set(np.unique(predicted)) == set(np.unique(truth)), (
        f"inference predicts classes {np.unique(predicted)}, truth has {np.unique(truth)}")
    accuracy = (predicted == truth).mean()
    assert accuracy > majority + 0.5 * (1.0 - majority), (accuracy, majority)


def test_every_batch_norm_uses_the_reference_statistics():
    """D-156: torch.nn.BatchNorm2d defaults, momentum 0.1 (Keras 0.9), eps 1e-5."""
    model = create_acc_unet_binary(3, input_shape=(SIZE, SIZE), base_filters=8, mlfc_iterations=1)
    bns = [layer for layer in model._flatten_layers()
           if isinstance(layer, keras.layers.BatchNormalization)]
    assert len(bns) > 50, len(bns)
    off = [(b.name, b.momentum, b.epsilon) for b in bns
           if (b.momentum, b.epsilon) != (0.9, 1e-5)]
    assert not off, off
