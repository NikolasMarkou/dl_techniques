"""CliffordCLIP vision pretraining normalises with the contrastive stage's constants.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 14 (D-090).
Measured at d6995a932: ``_run_pretrain_vision`` built the CIFAR-100 pipelines
of ``train.cliffordnet.train_cliffordnet``, which normalise with CIFAR-100's
statistics (mean ``[0.5071, 0.4867, 0.4408]``), while the contrastive stage
(``train.common.image_text.augment_and_normalize``) feeds the same tower
CLIP's (mean ``[0.4815, 0.4578, 0.4082]``), so the pretrained tower met a
shifted input distribution when contrastive training began. One constant
pair now serves both: the same objects, not equal copies.
"""

import numpy as np
import pytest

import train.cliffordnet.train_cliffordnet as cifar_pipeline
import train.cliffordnet.train_clip as trainer
from train.common import image_text

BATCH = 2


def test_the_pretraining_uses_the_contrastive_objects():
    mean, std = trainer._PRETRAIN_NORMALIZATION
    assert mean is image_text.IMAGE_MEAN
    assert std is image_text.IMAGE_STD


def _constant_images(value):
    images = np.broadcast_to(np.asarray(value, dtype=np.float32),
                             (BATCH, 32, 32, 3)).copy()
    return images, np.zeros((BATCH,), dtype=np.int64)


def test_the_eval_pipeline_maps_the_contrastive_mean_to_zero():
    x, y = _constant_images(image_text.IMAGE_MEAN)
    _, val_ds = trainer._pretrain_vision_datasets(x, y, x, y, BATCH)
    images, _ = next(iter(val_ds))
    np.testing.assert_allclose(np.asarray(images), 0.0, atol=1e-6)


def test_the_eval_pipeline_scales_by_the_contrastive_std():
    one_std_up = np.asarray(image_text.IMAGE_MEAN) + np.asarray(image_text.IMAGE_STD)
    x, y = _constant_images(one_std_up)
    _, val_ds = trainer._pretrain_vision_datasets(x, y, x, y, BATCH)
    images, _ = next(iter(val_ds))
    np.testing.assert_allclose(np.asarray(images), 1.0, atol=1e-5)


def test_the_train_pipeline_maps_the_contrastive_mean_to_zero(monkeypatch):
    """With AutoAugment and erasing stubbed to identity, the crop's centre is the image.

    A pad-4 crop at any offset in ``[0, 8]`` keeps rows and columns 4..27 from
    the original image, so a constant image normalises to exactly 0 there.
    """
    monkeypatch.setattr(cifar_pipeline, "_autoaugment_cifar10", lambda image: image)
    monkeypatch.setattr(cifar_pipeline, "_random_erase", lambda image: image)
    x, y = _constant_images(image_text.IMAGE_MEAN)
    train_ds, _ = trainer._pretrain_vision_datasets(x, y, x, y, BATCH)
    images, _ = next(iter(train_ds))
    centre = np.asarray(images)[:, 4:28, 4:28, :]
    np.testing.assert_allclose(centre, 0.0, atol=1e-6)


@pytest.mark.parametrize("builder", ["build_train_dataset", "build_eval_dataset"])
def test_the_cifar_trainer_keeps_its_own_statistics(builder, monkeypatch):
    """The override is opt-in: ``train_cliffordnet`` itself still uses CIFAR's."""
    monkeypatch.setattr(cifar_pipeline, "_autoaugment_cifar10", lambda image: image)
    monkeypatch.setattr(cifar_pipeline, "_random_erase", lambda image: image)
    mean = cifar_pipeline._CIFAR_STATS["cifar100"]["mean"]
    x, y = _constant_images(mean)
    ds = getattr(cifar_pipeline, builder)(x, y, batch_size=BATCH, dataset_name="cifar100")
    images, _ = next(iter(ds))
    np.testing.assert_allclose(np.asarray(images)[:, 4:28, 4:28, :], 0.0, atol=1e-6)
