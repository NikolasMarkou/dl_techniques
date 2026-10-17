"""A crop whose valid depth has no spread is never a MegaDepth sample.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.3 (D-125).
Part of MegaDepth stores ordinal depth: the valid pixels of the HDF5 map hold
one constant (2.0 or 5.0). Measured on the files the Step 1.4 gate used: 60 of
the first 94 validation crops (scene 0001) and 377 of 1956 crops sampled
across 168 scenes. ``normalized_disparity`` turns such a crop into an all-zero
target with the mask still on, and under the affine-invariant loss that sample
scores exactly 1.0 for any non-constant prediction with zero gradient: it
supervises nothing, puts a floor under ``val_loss`` and leaves the Pearson
correlation undefined (NaN). The loaders now treat it like a crop with too few
valid pixels: try another crop, and give up on the file (``None``) if none
has spread.
"""

import h5py
import numpy as np
from PIL import Image

from train.common import megadepth

SIZE = 16


def _write(directory, depth):
    directory.mkdir()
    rgb = (np.random.default_rng(0).random((48, 48, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(directory / "im.png")
    with h5py.File(directory / "im.h5", "w") as handle:
        handle["depth"] = depth.astype(np.float32)
    return str(directory / "im.png"), str(directory / "im.h5")


def _ordinal():
    depth = np.zeros((48, 48), dtype=np.float32)
    depth[8:, :] = 2.0  # MegaDepth's ordinal maps: one value on every valid pixel
    return depth


def _dense(seed):
    return np.random.default_rng(seed).uniform(1.0, 9.0, size=(48, 48))


def test_the_cropping_loader_refuses_an_ordinal_map(tmp_path):
    rgb_path, depth_path = _write(tmp_path / "ordinal", _ordinal())
    for seed in range(5):
        assert megadepth.load_and_process_pair(
            rgb_path, depth_path, SIZE, rng=np.random.default_rng(seed)) is None


def test_the_resizing_loader_refuses_an_ordinal_map(tmp_path):
    rgb_path, depth_path = _write(tmp_path / "ordinal", _ordinal())
    assert megadepth.load_and_resize_pair(rgb_path, depth_path, SIZE) is None


def test_a_dataset_with_ordinal_files_yields_only_targets_with_spread(tmp_path):
    pairs = [_write(tmp_path / "o0", _ordinal()), _write(tmp_path / "d0", _dense(1)),
             _write(tmp_path / "o1", _ordinal()), _write(tmp_path / "d1", _dense(2))]
    data = megadepth.MegaDepthDataset(
        [p[0] for p in pairs], [p[1] for p in pairs], batch_size=2, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    for index in range(len(data)):
        _rgb, y = data[index]
        assert len(y) == 2
        for target, mask in zip(y[..., 0], y[..., 1]):
            valid = mask > 0
            assert valid.any()
            assert target[valid].max() == 1.0 and target[valid].min() == 0.0


def test_a_dense_map_still_yields_its_crop(tmp_path):
    rgb_path, depth_path = _write(tmp_path / "dense", _dense(3))
    rgb, y = megadepth.load_and_process_pair(
        rgb_path, depth_path, SIZE, rng=np.random.default_rng(0))
    assert rgb.shape == (SIZE, SIZE, 3) and y.shape == (SIZE, SIZE, 2)
    assert y[..., 0].max() == 1.0
