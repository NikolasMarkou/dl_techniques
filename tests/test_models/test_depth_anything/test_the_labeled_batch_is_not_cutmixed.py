"""The labeled batch trains un-mixed and un-jittered, as in the paper.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.6 (D-131). The
paper (arXiv 2401.10891, section 4.1): "We only apply horizontal flipping as
our data augmentation for labeled images"; the strong perturbations (colour
jitter, CutMix) are its recipe for the UNLABELED images (section 3.2). At
308a4dfbf both training paths passed the labeled batch and its depth target
through ``StrongAugmentation`` (CutMix with ``cutmix_prob``, jitter with
``color_jitter_strength``). The horizontal flip lives in the MegaDepth loader
(D-124).

With CutMix certain (``cutmix_prob=1``), a half-size box, strong jitter and
the donor permutation pinned away from the identity, the loss the labeled step
logs must equal the compiled loss of the clean batch.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.affine_invariant_loss import AffineInvariantLoss
from dl_techniques.models.vision.depth_anything import DepthAnything

IMAGE_SHAPE = (32, 32, 3)


@pytest.fixture
def reversed_donors(monkeypatch):
    monkeypatch.setattr(keras.random, "shuffle", lambda x, **kwargs: keras.ops.flip(x, axis=0))


def _model(**kwargs):
    keras.utils.set_random_seed(3)
    model = DepthAnything(
        encoder_kind="placeholder", encoder_type="vit_s", image_shape=IMAGE_SHAPE,
        cutmix_prob=1.0, color_jitter_strength=0.8, input_value_range=None,
        dpt_features=8, dpt_out_channels=[8, 8, 8, 8], **kwargs)
    model(np.zeros((1,) + IMAGE_SHAPE, "float32"))
    model.augmentation.cutmix_ratio_range = (0.5, 0.5)
    model.compile(optimizer=keras.optimizers.SGD(0.0), loss=AffineInvariantLoss())
    return model


def _batch():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (2,) + IMAGE_SHAPE).astype("float32")
    y = rng.uniform(0, 1, (2,) + IMAGE_SHAPE[:2] + (1,)).astype("float32")
    return x, y


def test_the_labeled_step_scores_the_clean_batch(reversed_donors):
    model = _model(use_feature_alignment=False)
    x, y = _batch()
    clean = float(keras.ops.mean(AffineInvariantLoss()(y, model(x, training=True))))
    logged = float(model._train_step_labeled(x, y)["loss"])
    assert logged == pytest.approx(clean, rel=1e-5, abs=0)


def test_only_the_unlabeled_batch_is_augmented():
    model = _model(use_feature_alignment=False, enable_semi_supervised=True)
    seen = []
    original = model._augment_with_targets

    def recorder(images, targets):
        seen.append(np.asarray(images))
        return original(images, targets)

    model._augment_with_targets = recorder
    x, y = _batch()
    model._train_step_labeled(x, y)
    assert seen == [], "the labeled step augmented its batch"
    x_unlab = np.flip(x, axis=1).copy()
    model._train_step_semi_supervised(x, x_unlab, y)
    assert len(seen) == 1 and np.array_equal(seen[0], x_unlab), (
        "the semi-supervised step must augment the unlabeled batch, and only it")
