"""The MegaDepth training augmentation is a horizontal flip, nothing else.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.3 (D-124), the
root cause of W1 ("Depth Anything does not beat a constant prediction").
At e7c5b583b ``load_and_process_pair`` and ``load_and_resize_pair`` also
flipped up-down and rotated by a random multiple of 90 degrees. Measured on
the RTX 4070 (vit_s, 256 px, batch 8, stock fit, 1500 steps, the trainer's
optimizer recipe; scratchpad ``iter2/w1/w1_pipe.py``): with those transforms the
training loss stayed at 1.26 and the held-out Pearson at 0.05 (one scene) /
0.02 (19 held-out scenes); with a horizontal flip only, 0.82 and 0.23 / 0.39.
An up-down flip or a rotation removes where "down" is in the image, the
monocular cue a randomly initialised model learns first, and the model sat at
a nearly constant map. The reference (arXiv 2401.10891, section 4.1): "We only
apply horizontal flipping as our data augmentation for labeled images."

The axis the defect moves is orientation, so each augmented sample must equal
the un-augmented crop or its left-right mirror, over enough draws that HEAD's
three other orientations (each drawn with probability 3/4 per call) appear.
"""

import h5py
import numpy as np
import pytest
from PIL import Image

from train.common import megadepth

SIZE = 16
DRAWS = 40


def _pair(tmp_path):
    rng = np.random.default_rng(0)
    rgb = (rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "im.png")
    depth = rng.uniform(2.0, 50.0, size=(SIZE, SIZE)).astype(np.float32)
    with h5py.File(tmp_path / "im.h5", "w") as handle:
        handle["depth"] = depth
    return str(tmp_path / "im.png"), str(tmp_path / "im.h5")


def _orientation(sample, plain):
    """Which of identity / left-right mirror ``sample`` is, else ``None``."""
    rgb, y = sample
    base_rgb, base_y = plain
    if np.array_equal(rgb, base_rgb) and np.array_equal(y, base_y):
        return "identity"
    if np.array_equal(rgb, base_rgb[:, ::-1]) and np.array_equal(y, base_y[:, ::-1]):
        return "mirror"
    return None


def test_the_cropping_loader_only_mirrors_left_right(tmp_path):
    rgb_path, depth_path = _pair(tmp_path)
    plain = megadepth.load_and_process_pair(
        rgb_path, depth_path, SIZE, augment=False, rng=np.random.default_rng(0))
    seen = [
        _orientation(megadepth.load_and_process_pair(
            rgb_path, depth_path, SIZE, augment=True, rng=np.random.default_rng(seed)), plain)
        for seed in range(DRAWS)
    ]
    assert None not in seen, f"{seen.count(None)} of {DRAWS} samples were flipped up-down or rotated"
    assert {"identity", "mirror"} <= set(seen)


def test_the_resizing_loader_only_mirrors_left_right(tmp_path):
    rgb_path, depth_path = _pair(tmp_path)
    plain = megadepth.load_and_resize_pair(rgb_path, depth_path, SIZE, augment=False)
    seen = []
    state = np.random.get_state()
    try:
        for seed in range(DRAWS):
            np.random.seed(seed)
            seen.append(_orientation(megadepth.load_and_resize_pair(
                rgb_path, depth_path, SIZE, augment=True), plain))
    finally:
        np.random.set_state(state)
    assert None not in seen, f"{seen.count(None)} of {DRAWS} samples were flipped up-down or rotated"
    assert {"identity", "mirror"} <= set(seen)
