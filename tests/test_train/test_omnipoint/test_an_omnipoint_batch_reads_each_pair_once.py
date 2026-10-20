"""An OmniPoint batch reads its own window of pairs, each pair once per epoch.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.4 (D-193). At
d5e7964ae ``MegaDepthPinholeDataset.__getitem__`` and
``KittiDepthDataset.__getitem__`` indexed ``idx * batch_size + len(batch) +
attempts``. ``attempts`` also counted the loads that succeeded, so the index
advanced by 2 per sample: batch ``i`` read pairs ``i*B, i*B+2, ...``,
consecutive batches overlapped by half, and the pairs at odd offsets were
never read. This is the bug D-126 fixed in ``train.common.megadepth``.

Every pair here has its own uniform colour, so a sample's RGB names its pair.
"""

import h5py
import numpy as np
import pytest
from PIL import Image

from train.omnipoint.data import MegaDepthPinholeDataset
from train.omnipoint.kitti_depth import KittiDepthDataset

N_PAIRS = 16
BATCH = 4
SIZE = 8


def _megadepth_pairs(tmp_path):
    rgb_paths, depth_paths = [], []
    for i in range(N_PAIRS):
        Image.fromarray(np.full((12, 12, 3), i * 15, dtype=np.uint8)).save(tmp_path / f"{i}.png")
        with h5py.File(tmp_path / f"{i}.h5", "w") as handle:
            handle["depth"] = np.full((12, 12), 2.0 + i, dtype=np.float32)
        rgb_paths.append(str(tmp_path / f"{i}.png"))
        depth_paths.append(str(tmp_path / f"{i}.h5"))
    return rgb_paths, depth_paths


def _kitti_pairs(tmp_path):
    pairs = []
    for i in range(N_PAIRS):
        Image.fromarray(np.full((12, 12, 3), i * 15, dtype=np.uint8)).save(tmp_path / f"{i}_rgb.png")
        Image.fromarray(np.full((12, 12), 5000, dtype=np.uint16), mode="I;16").save(
            tmp_path / f"{i}_depth.png")
        pairs.append((str(tmp_path / f"{i}_rgb.png"), str(tmp_path / f"{i}_depth.png")))
    return pairs


def _pair_ids(rgb_batch):
    return [int(round((sample[0, 0, 0] + 1.0) * 127.5 / 15)) for sample in rgb_batch]


def _dataset(source, tmp_path, is_training):
    if source == "megadepth":
        rgb_paths, depth_paths = _megadepth_pairs(tmp_path)
        return MegaDepthPinholeDataset(rgb_paths, depth_paths, batch_size=BATCH,
                                       patch_size=SIZE, is_training=is_training, workers=1)
    return KittiDepthDataset(_kitti_pairs(tmp_path), batch_size=BATCH, patch_size=SIZE,
                             is_training=is_training, workers=1)


@pytest.mark.parametrize("source", ["megadepth", "kitti"])
def test_a_validation_batch_is_the_next_window_of_pairs(source, tmp_path):
    data = _dataset(source, tmp_path, is_training=False)
    assert len(data) == N_PAIRS // BATCH
    for index in range(len(data)):
        assert _pair_ids(data[index][0]) == list(range(index * BATCH, (index + 1) * BATCH))


@pytest.mark.parametrize("source", ["megadepth", "kitti"])
def test_a_training_epoch_reads_every_pair_exactly_once(source, tmp_path):
    data = _dataset(source, tmp_path, is_training=True)
    seen = []
    for index in range(len(data)):
        seen.extend(_pair_ids(data[index][0]))
    assert sorted(seen) == list(range(N_PAIRS))
