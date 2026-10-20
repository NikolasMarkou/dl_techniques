"""Each qualifying MegaDepth validation file is scored once per evaluation.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.4 (D-197). At
08bc99b6a a validation batch refilled a rejected file from the next window, and
the next batch began at its own window, so it read the refill again. Measured at
the Depth Anything defaults, 24.3% of the validation files are rejected: about a
quarter of the files were scored twice in ``val_loss``. A validation crop is
fixed per file, so the dataset now keeps the files that qualify and batches
those.

Every dense file here has its own uniform colour, so a sample's RGB names its
file; the ordinal maps (one depth value) never qualify (D-125).
"""

import h5py
import numpy as np
from PIL import Image

from train.common import megadepth

BATCH = 2
SIZE = 16


def _write(directory, colour, depth):
    directory.mkdir()
    Image.fromarray(np.full((32, 32, 3), colour, dtype=np.uint8)).save(directory / "im.png")
    with h5py.File(directory / "im.h5", "w") as handle:
        handle["depth"] = depth.astype(np.float32)
    return str(directory / "im.png"), str(directory / "im.h5")


def _files(tmp_path):
    """Files 0..9; 0, 3 and 7 are ordinal maps (rejected), the rest dense."""
    pairs = []
    for i in range(10):
        if i in (0, 3, 7):
            depth = np.zeros((32, 32), dtype=np.float32)
            depth[4:, :] = 5.0
        else:
            depth = np.random.default_rng(i).uniform(1.0, 9.0, (32, 32))
        pairs.append(_write(tmp_path / str(i), i * 15, depth))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _file_ids(rgb_batch):
    return [int(round((sample[0, 0, 0] + 1.0) * 127.5 / 15)) for sample in rgb_batch]


def test_each_qualifying_validation_file_is_scored_once(tmp_path):
    rgb_paths, depth_paths = _files(tmp_path)
    data = megadepth.MegaDepthDataset(
        rgb_paths, depth_paths, batch_size=BATCH, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    seen = [file_id for index in range(len(data)) for file_id in _file_ids(data[index][0])]
    assert seen == [1, 2, 4, 5, 6, 8]  # the 7 qualifying files, 3 full batches


def test_the_same_validation_batches_every_epoch(tmp_path):
    """Control: the validation set is fixed across epochs (D-109)."""
    rgb_paths, depth_paths = _files(tmp_path)
    data = megadepth.MegaDepthDataset(
        rgb_paths, depth_paths, batch_size=BATCH, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    first = [data[index][1] for index in range(len(data))]
    data.on_epoch_end()
    second = [data[index][1] for index in range(len(data))]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
