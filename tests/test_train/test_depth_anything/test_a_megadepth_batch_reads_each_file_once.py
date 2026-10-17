"""A MegaDepth batch reads its own window of files, each file once.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.3 (D-126). At
e7c5b583b ``MegaDepthDataset.__getitem__`` indexed ``idx * batch_size +
len(batch) + attempts``, and ``attempts`` also counted the loads that
succeeded, so the index advanced by 2 per sample: batch ``i`` read files
``i*B, i*B+2, ..., i*B+2(B-1)``. Consecutive batches overlapped by half, and
the files at odd offsets were never read. Measured on the Step 1.4 gate's
validation split: batch 1's first four crops were batch 0's last four.

Every file here has its own uniform colour, so a sample's RGB names its file.
"""

import h5py
import numpy as np
from PIL import Image

from train.common import megadepth

N_FILES = 16
BATCH = 4
SIZE = 16


def _files(tmp_path):
    rgb_paths, depth_paths = [], []
    for i in range(N_FILES):
        directory = tmp_path / str(i)
        directory.mkdir()
        Image.fromarray(np.full((32, 32, 3), i * 15, dtype=np.uint8)).save(directory / "im.png")
        with h5py.File(directory / "im.h5", "w") as handle:
            handle["depth"] = np.random.default_rng(i).uniform(1.0, 9.0, (32, 32)).astype(np.float32)
        rgb_paths.append(str(directory / "im.png"))
        depth_paths.append(str(directory / "im.h5"))
    return rgb_paths, depth_paths


def _file_ids(rgb_batch):
    return [int(round((sample[0, 0, 0] + 1.0) * 127.5 / 15)) for sample in rgb_batch]


def test_a_validation_batch_is_the_next_window_of_files(tmp_path):
    rgb_paths, depth_paths = _files(tmp_path)
    data = megadepth.MegaDepthDataset(
        rgb_paths, depth_paths, batch_size=BATCH, patch_size=SIZE,
        augment=False, is_training=False, workers=0)
    assert len(data) == N_FILES // BATCH
    for index in range(len(data)):
        rgb, _y = data[index]
        assert _file_ids(rgb) == list(range(index * BATCH, (index + 1) * BATCH))


def test_a_training_epoch_reads_every_file_exactly_once(tmp_path):
    rgb_paths, depth_paths = _files(tmp_path)
    data = megadepth.MegaDepthDataset(
        rgb_paths, depth_paths, batch_size=BATCH, patch_size=SIZE,
        augment=False, is_training=True, workers=0)
    seen = []
    for index in range(len(data)):
        rgb, _y = data[index]
        seen.extend(_file_ids(rgb))
    assert sorted(seen) == list(range(N_FILES))
