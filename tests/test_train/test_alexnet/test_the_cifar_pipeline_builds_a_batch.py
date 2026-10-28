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
from dl_techniques.datasets.vision.classification import CIFAR10_MEAN, CIFAR10_STD


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


def _tiny_white(*_args, **_kwargs):
    return _tiny_cifar()


def test_cifar100_is_normalised_with_cifar100_statistics(monkeypatch):
    # Step 13.5 (D-103): the CIFAR-100 path used the CIFAR-10 mean/std.
    from dl_techniques.datasets.vision.classification import CIFAR100_MEAN, CIFAR100_STD

    monkeypatch.setattr(keras.datasets.cifar100, "load_data", _tiny_white)
    config = module.TrainingConfig(
        dataset="cifar100", image_size=64, batch_size=2, augment_data=False,
    )
    train_ds, _val_ds, _steps, _val_steps = module.create_cifar_dataset(config)
    images, _labels = next(iter(train_ds))
    expected = (1.0 - np.asarray(CIFAR100_MEAN)) / np.asarray(CIFAR100_STD)
    np.testing.assert_allclose(
        np.asarray(images)[0, 0, 0], expected,
        rtol=4 * np.finfo(np.float32).eps,
    )


# The re-derivation of CIFAR100_MEAN/STD from the data lives with their one
# home's guard: tests/test_train/test_common_cifar100_statistics_have_one_home.py.
