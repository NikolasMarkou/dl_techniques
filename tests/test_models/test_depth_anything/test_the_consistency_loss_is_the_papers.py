"""The unlabeled consistency term is the paper's eq. 6-8, by value.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (e) (D-113). Measured at
e259d9977: ``DepthAnything._train_step_semi_supervised`` scored the student's
depth on the augmented unlabeled batch against the teacher's mixed pseudo-depth
with a plain L1, ``mean(|S - T|)``. Depth Anything V1 (arXiv 2401.10891,
section 3.2) computes the affine-invariant loss ``rho`` (eq. 1-3: median shift,
mean-absolute-deviation scale, then MAE) separately over the CutMix regions
``M`` and ``1 - M`` (eq. 6, 7) and averages the two by their pixel shares
(eq. 8). Without a mix, ``M`` covers the whole map and eq. 8 is ``rho`` itself.

The reference below is float64 NumPy written from the equations, independent of
``AffineInvariantLoss``. The model's term is read off ``train_step``'s logged
loss with the labeled weight 0 and the optimizer's learning rate 0.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything import DepthAnything

IMAGE_SHAPE = (32, 32, 3)
BATCH = 3


def _rho(pred, target, region):
    """Eq. 1-3 over the pixels where ``region`` is 1; float64."""
    pred, target = pred[region > 0], target[region > 0]
    if pred.size == 0:
        return 0.0

    def normalise(d):
        t = np.median(d)
        return (d - t) / max(np.mean(np.abs(d - t)), 1e-6)

    return float(np.mean(np.abs(normalise(pred) - normalise(target))))


def _reference(pred, pseudo, box):
    """Eq. 6-8 per image, then the batch mean."""
    own, donor = 1.0 - box[..., 0], box[..., 0]
    hw = own.size
    per_image = [
        own.sum() / hw * _rho(pred[b, ..., 0], pseudo[b, ..., 0], own)
        + donor.sum() / hw * _rho(pred[b, ..., 0], pseudo[b, ..., 0], donor)
        for b in range(pred.shape[0])
    ]
    return float(np.mean(per_image))


class _FixedMix:
    """A CutMix with a known box and donor order, no colour change."""

    def __init__(self, box, perm):
        self.box = keras.ops.convert_to_tensor(box, "float32")
        self.perm = keras.ops.convert_to_tensor(perm, "int32")

    def augment_with_mix(self, x, training=True):
        mix = (self.box, self.perm)
        return self.apply_mix_to_target(x, mix), mix

    @staticmethod
    def apply_mix_to_target(target, mix):
        box, perm = mix
        box = keras.ops.cast(box, target.dtype)
        donor = keras.ops.take(target, perm, axis=0)
        return target * (1.0 - box) + donor * box


def _model(augmentation):
    keras.utils.set_random_seed(7)
    model = DepthAnything(
        encoder_kind="placeholder", encoder_type="vit_s", image_shape=IMAGE_SHAPE,
        enable_semi_supervised=True, use_feature_alignment=False,
    )
    _ = model(keras.ops.zeros((1,) + IMAGE_SHAPE))
    model.augmentation = augmentation
    # The teacher reads the decoder in inference mode, so its BatchNorm moving
    # statistics, which the step's own training-mode forwards update before
    # the pseudo-label is drawn, would make the step's pseudo-label differ from
    # one taken here. Momentum 1 keeps them where they are.
    for layer in model._flatten_layers():
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.momentum = 1.0
    model.compile(
        optimizer=keras.optimizers.SGD(0.0),
        loss_weights={"labeled": 0.0, "unlabeled": 1.0},
    )
    return model


def _box(h0, h1, w0, w1):
    box = np.zeros(IMAGE_SHAPE[:2] + (1,), np.float32)
    box[h0:h1, w0:w1] = 1.0
    return box


@pytest.mark.parametrize("box, perm", [
    (None, None),                                   # no mix: rho on the whole map
    (_box(4, 20, 8, 30), np.array([2, 0, 1])),      # a box from another image
    (_box(0, 32, 0, 12), np.array([1, 2, 0])),      # a full-height strip
], ids=["no-mix", "box", "strip"])
def test_the_logged_unlabeled_term_equals_eq_6_to_8(box, perm):
    augmentation = None if box is None else _FixedMix(box, perm)
    model = _model(augmentation)
    rng = np.random.default_rng(0)
    x_lab = rng.standard_normal((BATCH,) + IMAGE_SHAPE).astype(np.float32)
    x_unlab = rng.standard_normal((BATCH,) + IMAGE_SHAPE).astype(np.float32)
    y_lab = rng.random((BATCH,) + IMAGE_SHAPE[:2] + (1,)).astype(np.float32)

    if box is None:
        x_student, region = x_unlab, np.zeros(IMAGE_SHAPE[:2] + (1,), np.float32)
        pseudo = np.asarray(model._pseudo_label_depth(x_unlab), np.float64)
    else:
        mix = (augmentation.box, augmentation.perm)
        x_student = np.asarray(augmentation.apply_mix_to_target(x_unlab, mix))
        region = box
        pseudo = np.asarray(augmentation.apply_mix_to_target(
            model._pseudo_label_depth(x_unlab), mix), np.float64)
    pred = np.asarray(model(x_student, training=True), np.float64)
    expected = _reference(pred, pseudo, region)

    logs = model.train_step(((x_lab, x_unlab), y_lab))
    got = float(logs["loss"])
    # Two float32 normalisations of maps whose MAD is O(1e-2..1e0) against a
    # float64 reference: agreement to 1e-4 relative is the float32 budget, and
    # the plain L1 the term replaced misses by orders of magnitude more.
    np.testing.assert_allclose(got, expected, rtol=1e-4)


def test_a_constant_shift_and_scale_of_the_student_costs_nothing():
    """Affine invariance per region: the property the plain L1 did not have."""
    from dl_techniques.models.vision.depth_anything.model import DepthAnything as M

    rng = np.random.default_rng(1)
    pseudo = rng.random((2, 8, 8, 1)).astype(np.float32)
    box = _box(2, 6, 0, 8)[:8, :8]
    region_mix = _FixedMix(box, np.array([1, 0]))
    mixed = region_mix.apply_mix_to_target(pseudo, (region_mix.box, region_mix.perm))
    # Per region, an affine copy of the target: scale 3, shift 5 outside the box,
    # scale 0.5, shift -2 inside it.
    pred = np.where(box > 0, 0.5 * np.asarray(mixed) - 2.0, 3.0 * np.asarray(mixed) + 5.0)
    value = M._unlabeled_consistency(
        keras.ops.convert_to_tensor(pred.astype(np.float32)), mixed, region_mix.box)
    assert float(value) == pytest.approx(0.0, abs=1e-5)
