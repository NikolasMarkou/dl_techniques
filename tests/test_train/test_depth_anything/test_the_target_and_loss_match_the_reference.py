"""The Depth Anything trainer's target and loss are the reference's, and the head can reach the target.

Guard for plan-2026-10-14T062616-81938e81 step 13.5 (D-108). Measured at
ffa4c2870: ``train.common.megadepth`` scaled each depth map to ``[-1, 1]``
(min-max depth), and the trainer compiled a masked L1 plus gradient matching on
the raw prediction. DepthAnything's head ends in a ReLU (D-031), so about half
of every target (all of it below 0) could not be predicted, and the Step 13
smoke val loss rose 1.59 to 1.81.

The reference (Depth Anything V1, arXiv 2401.10891, section 3.1, eq. 1-3)
trains relative depth on disparity ``1 / t`` normalized to ``[0, 1]`` per map,
with the affine-invariant mean absolute error: each map is shifted by its
median and divided by its mean absolute deviation before the L1.

Everything here runs the trainer's real code: the target from
``load_and_process_pair`` on a real HDF5/PNG pair written to ``tmp_path``, and
the loss the trainer's ``train_depth_anything`` hands to ``model.compile`` (data
and model stubbed).
"""

import h5py
import keras
import numpy as np
import pytest
from PIL import Image

from train.common import megadepth

SIZE = 16


def _pair(tmp_path, depth):
    rgb = (np.random.default_rng(0).random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
    rgb_path = tmp_path / "im.png"
    Image.fromarray(rgb).save(rgb_path)
    depth_path = tmp_path / "im.h5"
    with h5py.File(depth_path, "w") as handle:
        handle["depth"] = depth.astype(np.float32)
    return str(rgb_path), str(depth_path)


def _depth_map():
    rng = np.random.default_rng(1)
    depth = rng.uniform(2.0, 50.0, size=(SIZE, SIZE))
    depth[:3, :5] = 0.0  # invalid (SfM holes)
    return depth


def test_the_target_is_per_map_normalized_disparity_in_0_1(tmp_path):
    depth = _depth_map()
    # The map is exactly one patch, so the crop is the whole map.
    rgb, y_true = megadepth.load_and_process_pair(
        *_pair(tmp_path, depth), patch_size=SIZE, min_valid_ratio=0.1)
    target, mask = y_true[..., 0], y_true[..., 1]
    valid = depth > 0
    np.testing.assert_array_equal(mask, valid.astype(np.float32))
    disparity = 1.0 / depth[valid].astype(np.float64)
    expected = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    # float64 reference, float32 result: one rounding of a value in [0, 1].
    np.testing.assert_allclose(target[valid], expected, rtol=0, atol=2 ** -24)
    assert target[valid].min() == 0.0 and target[valid].max() == 1.0
    # The nearest valid pixel is the brightest one.
    assert target[valid][np.argmin(depth[valid])] == 1.0
    assert np.all(target[~valid] == 0.0)


def test_the_target_range_is_inside_the_heads_output_range(tmp_path):
    """Target in [0, 1]; a ReLU output covers [0, inf), so every target is reachable."""
    from dl_techniques.models.vision.depth_anything.model import create_depth_anything

    _rgb, y_true = megadepth.load_and_process_pair(
        *_pair(tmp_path, _depth_map()), patch_size=SIZE, min_valid_ratio=0.1)
    target = y_true[..., 0][y_true[..., 1] > 0]
    assert target.min() >= 0.0 and target.max() <= 1.0

    keras.utils.set_random_seed(0)
    model = create_depth_anything(
        encoder_type="vit_s", encoder_kind="placeholder",
        image_shape=(64, 64, 3), dpt_features=8, dpt_out_channels=[8, 8, 8, 8],
    )
    out = np.asarray(model(np.random.default_rng(2).standard_normal(
        (2, 64, 64, 3)).astype("float32"), training=False))
    # The output's lower bound is the target's lower bound.
    assert out.min() >= 0.0
    assert np.any(out > 0.0)


class _OneBatch:
    def __len__(self):
        return 1


class _CompileCapture:
    """Model stand-in that records what the trainer compiles, then stops the run."""

    class Stop(Exception):
        pass

    def __init__(self):
        self.compiled = None

    def summary(self, *args, **kwargs):
        return None

    def compile(self, **kwargs):
        self.compiled = kwargs
        raise self.Stop()


@pytest.fixture(scope="module")
def trainer_loss(tmp_path_factory):
    """The loss ``train_depth_anything`` passes to ``model.compile``."""
    from train.depth_anything import train_depth_anything as module

    tmp_path = tmp_path_factory.mktemp("da")
    paths = [str(tmp_path / f"p{i}") for i in range(20)]
    model = _CompileCapture()
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(module, "discover_megadepth_pairs", lambda *a, **k: (paths, paths))
        mp.setattr(module, "MegaDepthDataset", lambda *a, **k: _OneBatch())
        mp.setattr(module, "create_model", lambda config: model)
        config = module.DepthAnythingTrainingConfig(
            output_dir=str(tmp_path), experiment_name="run", epochs=1,
            steps_per_epoch=1)
        with pytest.raises(_CompileCapture.Stop):
            module.train_depth_anything(config)
    finally:
        mp.undo()
    return model.compiled["loss"]


def _batch():
    rng = np.random.default_rng(3)
    target = rng.uniform(0.0, 1.0, size=(2, 8, 8, 1)).astype("float32")
    mask = np.ones_like(target)
    mask[0, :2] = 0.0
    mask[1, :, :3] = 0.0
    prediction = np.maximum(
        rng.standard_normal((2, 8, 8, 1)), 0.0).astype("float32") + 0.1
    return np.concatenate([target, mask], axis=-1), prediction


def _reference(y_true, y_pred):
    """Eq. 1-3 in float64: per-image median/MAD alignment, masked MAE, batch mean."""
    losses = []
    for t, m, p in zip(y_true[..., 0], y_true[..., 1], y_pred[..., 0]):
        valid = m > 0

        def hat(d):
            d = d[valid].astype(np.float64)
            med = np.median(d)
            return (d - med) / np.mean(np.abs(d - med))

        losses.append(np.mean(np.abs(hat(p) - hat(t))))
    return float(np.mean(losses))


def test_the_loss_is_the_reference_affine_invariant_mae(trainer_loss):
    y_true, y_pred = _batch()
    expected = _reference(y_true, y_pred)
    got = float(trainer_loss(y_true, y_pred))
    np.testing.assert_allclose(got, expected, rtol=1e-5)


@pytest.mark.parametrize("scale, shift", [(0.25, 0.0), (3.0, 0.0), (1.0, 5.0), (7.0, -2.5)])
def test_the_loss_ignores_the_predictions_scale_and_shift(trainer_loss, scale, shift):
    y_true, y_pred = _batch()
    base = float(trainer_loss(y_true, y_pred))
    moved = float(trainer_loss(y_true, (scale * y_pred + shift).astype("float32")))
    np.testing.assert_allclose(moved, base, rtol=1e-5)


def test_an_affine_copy_of_the_target_has_zero_loss(trainer_loss):
    y_true, _ = _batch()
    prediction = (4.0 * y_true[..., :1] + 0.5).astype("float32")
    assert float(trainer_loss(y_true, prediction)) < 1e-5


def test_the_validation_crops_are_the_same_every_epoch(tmp_path):
    """D-109: a validation batch is the same on every call, so val_loss is comparable across epochs."""
    rng = np.random.default_rng(4)
    rgb_paths, depth_paths = [], []
    for i in range(3):
        sub = tmp_path / str(i)
        sub.mkdir()
        rgb = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(sub / "im.png")
        with h5py.File(sub / "im.h5", "w") as handle:
            handle["depth"] = rng.uniform(1.0, 9.0, size=(48, 48)).astype(np.float32)
        rgb_paths.append(str(sub / "im.png"))
        depth_paths.append(str(sub / "im.h5"))
    val = megadepth.MegaDepthDataset(
        rgb_paths, depth_paths, batch_size=3, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    first_rgb, first_y = val[0]
    for _ in range(3):
        rgb, y = val[0]
        np.testing.assert_array_equal(rgb, first_rgb)
        np.testing.assert_array_equal(y, first_y)
