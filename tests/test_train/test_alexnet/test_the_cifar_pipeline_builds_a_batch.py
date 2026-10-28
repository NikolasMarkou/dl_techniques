"""The AlexNet trainer's CIFAR pipeline yields a normalised, resized batch.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-098). Measured at 561e02165:
``python -m train.alexnet.train_alexnet --dataset cifar10`` died before its
first step with ``AttributeError: SymbolicTensor object has no attribute
'astype'``: the old ``create_cifar_dataset``'s ``prepare`` called the NumPy method
``images.astype`` inside a ``tf.data`` map, where it receives symbolic tensors.
The CIFAR path had never run; no test built it. It now runs through the shared loader.
"""

import numpy as np

import train.common.classification_data as classification_data
from dl_techniques.datasets.vision.classification import CIFAR10_MEAN, CIFAR10_STD
from train.alexnet import train_alexnet as module


def _white_cifar(name):
    """40 white train images and 10 white test images (selection takes 4 of the train)."""
    x = np.full((40, 32, 32, 3), 255, dtype=np.uint8)
    y = np.arange(40, dtype=np.int64) % 10
    return (x, y), (x[:10], y[:10])


def test_the_cifar_pipeline_builds_a_batch(monkeypatch):
    monkeypatch.setattr(classification_data, "_load_raw", _white_cifar)
    config = module.TrainingConfig(
        dataset="cifar10", image_size=64, batch_size=2, augment_data=False,
    )
    data = module.create_cifar_data(config)
    images, labels = next(iter(data.train_ds))
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
    assert (data.steps_per_epoch, data.n_select) == (18, 4)
    assert next(iter(data.select_ds))[0].shape == (4, 64, 64, 3)
    assert next(iter(data.test_ds))[0].shape == (10, 64, 64, 3)
    assert data.input_shape == (64, 64, 3)


def test_cifar100_is_normalised_with_cifar100_statistics(monkeypatch):
    # Step 13.5 (D-103): the CIFAR-100 path used the CIFAR-10 mean/std.
    from dl_techniques.datasets.vision.classification import CIFAR100_MEAN, CIFAR100_STD

    monkeypatch.setattr(classification_data, "_load_raw", _white_cifar)
    config = module.TrainingConfig(
        dataset="cifar100", image_size=64, batch_size=2, augment_data=False,
    )
    data = module.create_cifar_data(config)
    images, _labels = next(iter(data.train_ds))
    expected = (1.0 - np.asarray(CIFAR100_MEAN)) / np.asarray(CIFAR100_STD)
    np.testing.assert_allclose(
        np.asarray(images)[0, 0, 0], expected,
        rtol=4 * np.finfo(np.float32).eps,
    )


# The re-derivation of CIFAR100_MEAN/STD from the data lives with their one
# home's guard: tests/test_datasets/test_the_dataset_statistics_have_one_home.py.
