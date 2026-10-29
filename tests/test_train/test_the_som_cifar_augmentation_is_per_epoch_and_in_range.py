"""The CIFAR SoftSOM trainer augments on the [0, 1] image, anew every epoch.

Measured at HEAD: ``create_augmentation_layer`` used Keras ``RandomBrightness`` and
``RandomContrast``, whose default ``value_range`` is (0, 255), so on a [0, 1] image the
brightness delta of ``0.2 * strength`` scaled by 255 saturated it; and the layer was called once
over ``x_train``, so one frozen augmentation served all epochs. The trainer now maps the shared
``train.common.image_augment`` ops over a ``tf.data`` pipeline.
"""

import numpy as np
import pytest

from dl_techniques.datasets.vision.augment_hook import is_seeded
from train.som_nd_soft import train_cifar as module

N = 32


def _images() -> np.ndarray:
    return np.random.default_rng(0).random((N, 32, 32, 3)).astype(np.float32)


def _config(**kwargs):
    return module.CIFARSOMConfig(batch_size=8, seed=5, **kwargs)


def _epoch(dataset):
    batches = list(dataset.as_numpy_iterator())
    return (np.concatenate([a for a, _ in batches]), np.concatenate([c for _, c in batches]))


def test_the_hook_is_seeded_and_off_means_none():
    assert is_seeded(module.create_augmentation(_config()))
    assert module.create_augmentation(_config(use_data_augmentation=False)) is None


def test_the_target_is_the_clean_image_and_the_input_is_augmented():
    x = _images()
    augmented, clean = _epoch(module.make_train_dataset(x, _config()))
    assert augmented.shape == clean.shape == x.shape
    order = {tuple(row.ravel()[:6]): i for i, row in enumerate(x)}
    rows = [order[tuple(row.ravel()[:6])] for row in clean]
    assert sorted(rows) == list(range(N)), "every clean image once per epoch"
    np.testing.assert_array_equal(clean, x[rows])
    assert np.abs(augmented - clean).mean() > 0.05


def test_the_brightness_and_contrast_do_not_saturate_the_image():
    """Strength 0.5 means a brightness of 0.1 and a contrast of 0.1 on a [0, 1] image."""
    x = np.full((N, 32, 32, 3), 0.5, np.float32)
    augmented, _ = _epoch(module.make_train_dataset(
        x, _config(augmentation_strength=0.5)))
    interior = augmented[:, 8:24, 8:24]
    assert interior.min() > 0.3 and interior.max() < 0.7, (interior.min(), interior.max())


def test_a_second_epoch_draws_a_new_augmentation_and_a_rebuild_repeats_the_first():
    x = _images()
    dataset = module.make_train_dataset(x, _config())
    first, second = _epoch(dataset)[0], _epoch(dataset)[0]
    again = _epoch(module.make_train_dataset(x, _config()))[0]
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, second)


def test_without_augmentation_the_input_is_the_target():
    augmented, clean = _epoch(module.make_train_dataset(
        _images(), _config(use_data_augmentation=False)))
    np.testing.assert_array_equal(augmented, clean)


def test_the_remainder_is_dropped_like_steps_per_epoch_counts():
    x = _images()[:30]
    augmented, _ = _epoch(module.make_train_dataset(x, _config()))
    assert len(augmented) == 24
