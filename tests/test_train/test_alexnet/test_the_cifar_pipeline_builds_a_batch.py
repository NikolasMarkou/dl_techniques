"""The AlexNet trainer's CIFAR pipeline yields a normalised, resized batch.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-098). Measured at 561e02165:
``python -m train.alexnet.train_alexnet --dataset cifar10`` died before its
first step with ``AttributeError: SymbolicTensor object has no attribute
'astype'``: ``create_cifar_dataset``'s ``prepare`` called the NumPy method
``images.astype`` inside a ``tf.data`` map, where it receives symbolic tensors.
The CIFAR path had never run; no test built it.
"""

import keras
import numpy as np

from train.alexnet import train_alexnet as module
from train.common import CIFAR10_MEAN, CIFAR10_STD


def _tiny_cifar():
    x = np.full((4, 32, 32, 3), 255, dtype=np.uint8)
    y = np.arange(4, dtype=np.uint8).reshape(4, 1)
    return (x, y), (x[:2], y[:2])


def test_the_cifar_pipeline_builds_a_batch(monkeypatch):
    monkeypatch.setattr(keras.datasets.cifar10, "load_data", _tiny_cifar)
    config = module.TrainingConfig(
        dataset="cifar10", image_size=64, batch_size=2, augment_data=False,
    )
    train_ds, val_ds, steps, val_steps = module.create_cifar_dataset(config)
    images, labels = next(iter(train_ds))
    assert images.shape == (2, 64, 64, 3)
    assert images.dtype == "float32"
    # A white pixel (255) maps to (1 - mean) / std after the cast.
    expected = (1.0 - np.asarray(CIFAR10_MEAN)) / np.asarray(CIFAR10_STD)
    # Divide, subtract, divide and a constant-image resize: a few float32
    # roundings against the float64 expectation.
    np.testing.assert_allclose(
        np.asarray(images)[0, 0, 0], expected,
        rtol=4 * np.finfo(np.float32).eps,
    )
    assert (steps, val_steps) == (2, 1)
    assert next(iter(val_ds))[0].shape == (2, 64, 64, 3)
