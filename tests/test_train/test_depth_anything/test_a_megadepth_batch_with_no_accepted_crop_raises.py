"""A MegaDepth batch in which no crop qualifies raises; it is never zero-filled.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.4 (D-194). At
21b209dd0 ``MegaDepthDataset.__getitem__`` padded a batch with one all-zero
sample when none of its ``3 * batch_size`` files gave a crop that qualifies
(enough valid pixels, depth with spread). That sample is a mid-grey image with
an all-invalid mask, and it still took an optimizer step (the D-130 and D-188
pattern). Measured at the Depth Anything defaults (518 px crops,
``min_valid_ratio`` 0.1): 18.9% of training draws and 24.3% of validation
files are rejected, and no batch of the default split is all-rejected.
"""

import h5py
import numpy as np
import pytest
from PIL import Image

from train.common import megadepth

SIZE = 16


def _write(directory, depth):
    directory.mkdir()
    Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8)).save(directory / "im.png")
    with h5py.File(directory / "im.h5", "w") as handle:
        handle["depth"] = depth.astype(np.float32)
    return str(directory / "im.png"), str(directory / "im.h5")


def _ordinal():
    depth = np.zeros((32, 32), dtype=np.float32)
    depth[4:, :] = 5.0  # one constant on every valid pixel: no crop has spread
    return depth


@pytest.mark.parametrize("is_training", [False, True])
def test_a_batch_in_which_no_crop_qualifies_raises(tmp_path, is_training):
    pairs = [_write(tmp_path / f"ordinal_{i}", _ordinal()) for i in range(4)]
    data = megadepth.MegaDepthDataset(
        [p[0] for p in pairs], [p[1] for p in pairs], batch_size=2, patch_size=SIZE,
        augment=False, is_training=is_training, workers=0)
    with pytest.raises(RuntimeError, match="no MegaDepth crop qualified") as raised:
        data[0]
    assert "ordinal_" in str(raised.value)


def test_a_batch_with_one_rejected_file_is_still_full(tmp_path):
    """Control: a rejected file is skipped and the batch refilled, as before."""
    dense = np.random.default_rng(0).uniform(1.0, 9.0, (32, 32))
    pairs = [_write(tmp_path / "ordinal", _ordinal())] + [
        _write(tmp_path / f"dense_{i}", dense) for i in range(3)]
    data = megadepth.MegaDepthDataset(
        [p[0] for p in pairs], [p[1] for p in pairs], batch_size=2, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    rgb, y_true = data[0]
    assert rgb.shape == (2, SIZE, SIZE, 3)
    assert (y_true[..., 1].reshape(2, -1).max(axis=1) == 1.0).all()
