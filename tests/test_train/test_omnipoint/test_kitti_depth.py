"""Tests for `train.omnipoint.kitti_depth`.

Two kinds of check, deliberately both present:

- A REAL integration check against this machine's actual local KITTI depth mirror
  (`/media/arxwn/data0_4tb/datasets/KITTI/data/depth/`) -- proves the discovery/loading
  logic actually works on real data, not just a mock. Skipped (not silently passed) if
  the dataset root is absent on the running machine.
- A synthetic, temp-directory fixture -- fast, deterministic, CI-portable, and used for
  the missing/corrupt-path failure-mode test where a real bad path cannot be relied upon
  to exist.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from train.omnipoint.kitti_depth import (
    KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
    KittiDepthDataset,
    derive_pinhole_intrinsics,
    discover_kitti_depth_pairs,
    load_kitti_depth_pair,
)

KITTI_DEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"


# =====================================================================
# derive_pinhole_intrinsics
# =====================================================================


class TestDerivePinholeIntrinsics:
    def test_matches_hand_computed_formula(self):
        width, height, fov_deg = 1242, 375, 70.0
        fx, fy, cx, cy = derive_pinhole_intrinsics(width, height, fov_deg)

        expected_fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        assert fx == pytest.approx(expected_fx, rel=0, abs=1e-9)
        assert fy == pytest.approx(expected_fx, rel=0, abs=1e-9)
        assert cx == pytest.approx(width / 2.0, rel=0, abs=1e-9)
        assert cy == pytest.approx(height / 2.0, rel=0, abs=1e-9)

    def test_default_fov_constant_is_used(self):
        width, height = 800, 600
        fx_default, fy_default, cx_default, cy_default = derive_pinhole_intrinsics(
            width, height
        )
        fx_explicit, fy_explicit, cx_explicit, cy_explicit = derive_pinhole_intrinsics(
            width, height, horizontal_fov_deg=KITTI_DEFAULT_HORIZONTAL_FOV_DEG
        )
        assert fx_default == pytest.approx(fx_explicit, rel=0, abs=1e-12)
        assert fy_default == pytest.approx(fy_explicit, rel=0, abs=1e-12)
        assert cx_default == pytest.approx(cx_explicit, rel=0, abs=1e-12)
        assert cy_default == pytest.approx(cy_explicit, rel=0, abs=1e-12)

    def test_wider_fov_yields_smaller_focal_length(self):
        width, height = 1000, 500
        fx_narrow, *_ = derive_pinhole_intrinsics(width, height, horizontal_fov_deg=40.0)
        fx_wide, *_ = derive_pinhole_intrinsics(width, height, horizontal_fov_deg=100.0)
        assert fx_wide < fx_narrow


# =====================================================================
# discover_kitti_depth_pairs -- REAL local data
# =====================================================================


@pytest.mark.skipif(
    not Path(KITTI_DEPTH_ROOT).exists(),
    reason=f"Real KITTI depth mirror not present at {KITTI_DEPTH_ROOT}",
)
class TestDiscoverKittiDepthPairsReal:
    def test_finds_at_least_one_real_pair(self):
        pairs = discover_kitti_depth_pairs(KITTI_DEPTH_ROOT)
        assert len(pairs) > 0, (
            "discover_kitti_depth_pairs found ZERO pairs against the real KITTI "
            "mirror -- this is a blocker, not a vacuous pass: the on-disk matching "
            "convention this loader assumes may have changed."
        )
        # Report the concrete count for the executor's report (not asserted further --
        # this is informational, but the print aids `pytest -s` debugging).
        print(f"\nDiscovered {len(pairs)} real KITTI depth RGB+depth pairs.")

    def test_a_discovered_pair_has_shape_compatible_rgb_and_depth(self):
        pairs = discover_kitti_depth_pairs(KITTI_DEPTH_ROOT, max_files=1)
        assert len(pairs) == 1
        rgb_path, depth_path = pairs[0]

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path))

        assert rgb.ndim == 3 and rgb.shape[2] == 3
        assert depth.ndim == 2
        # RGB and depth GT must be the same (H, W) grid for a valid pair -- a mismatch
        # here would mean RGB and depth GT frames were mismatched across sequences.
        assert rgb.shape[:2] == depth.shape[:2], (
            f"Mismatched RGB {rgb.shape[:2]} vs depth {depth.shape[:2]} for pair "
            f"({rgb_path}, {depth_path})"
        )
        assert depth.dtype == np.uint16

    def test_pairs_are_unique_and_well_formed_paths(self):
        pairs = discover_kitti_depth_pairs(KITTI_DEPTH_ROOT, max_files=50)
        assert len(pairs) > 0
        assert len(set(pairs)) == len(pairs), "duplicate pairs discovered"
        for rgb_path, depth_path in pairs:
            assert Path(rgb_path).exists()
            assert Path(depth_path).exists()
            assert Path(rgb_path).suffix == ".png"
            assert Path(depth_path).suffix == ".png"


# =====================================================================
# Synthetic fixture -- fast, deterministic, CI-portable
# =====================================================================


def _make_synthetic_kitti_tree(tmp_path: Path, num_frames: int = 4) -> list:
    """Build a minimal on-disk KITTI depth tree matching the real convention.

    Returns the list of ``(rgb_path, depth_path)`` pairs this tree is expected to
    yield, independently constructed (not by calling the function under test) so the
    test has an independent oracle.
    """
    root = tmp_path / "kitti_depth_root"
    drive_name = "2011_09_26_drive_0001_sync"
    date = "2011_09_26"
    camera = "image_02"

    rgb_dir = root / "raw_image_values" / date / drive_name / camera / "data"
    depth_dir = root / "depth_values" / drive_name / "proj_depth" / "groundtruth" / camera
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    height, width = 40, 60
    expected_pairs = []
    rng = np.random.default_rng(0)
    for i in range(num_frames):
        frame_name = f"{i:010d}.png"
        rgb_arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr, mode="RGB").save(rgb_dir / frame_name)

        # Half the pixels valid (nonzero), half invalid (zero) -- so the valid_mask
        # test below has real signal to check.
        depth_arr = np.zeros((height, width), dtype=np.uint16)
        depth_arr[: height // 2, :] = 5000  # 5000 / 256.0 = ~19.5 m
        Image.fromarray(depth_arr, mode="I;16").save(depth_dir / frame_name)

        expected_pairs.append((str(rgb_dir / frame_name), str(depth_dir / frame_name)))

    return str(root), expected_pairs


class TestDiscoverKittiDepthPairsSynthetic:
    def test_discovers_exactly_the_synthetic_pairs(self, tmp_path):
        root, expected_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=4)
        pairs = discover_kitti_depth_pairs(root)
        assert sorted(pairs) == sorted(expected_pairs)


class TestLoadKittiDepthPair:
    def test_valid_mask_excludes_zero_depth_pixels(self, tmp_path):
        root, expected_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=1)
        rgb_path, depth_path = expected_pairs[0]

        result = load_kitti_depth_pair(rgb_path, depth_path, patch_size=32)
        assert result is not None
        rgb, depth, valid_mask, k = result

        assert rgb.shape == (32, 32, 3)
        assert depth.shape == (32, 32, 1)
        assert valid_mask.shape == (32, 32, 1)
        assert rgb.dtype == np.float32
        assert depth.dtype == np.float32
        assert valid_mask.dtype == np.float32
        assert len(k) == 4

        # Top half of the source was valid (depth=5000/256=~19.5m), bottom half was
        # depth=0 (invalid). Nearest-neighbor resize preserves this split.
        assert np.all(valid_mask[:14, :] == 1.0)
        assert np.all(valid_mask[18:, :] == 0.0)
        # Wherever invalid, depth must read exactly 0 -- never a manufactured value.
        assert np.all(depth[valid_mask == 0.0] == 0.0)
        # Wherever valid, depth must be strictly positive.
        assert np.all(depth[valid_mask == 1.0] > 0.0)

    def test_missing_file_returns_none_and_does_not_raise(self, tmp_path):
        root, expected_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=1)
        rgb_path, _ = expected_pairs[0]
        broken_depth_path = str(tmp_path / "does_not_exist.png")

        result = load_kitti_depth_pair(rgb_path, broken_depth_path, patch_size=32)
        assert result is None


# =====================================================================
# KittiDepthDataset
# =====================================================================


class TestKittiDepthDatasetSynthetic:
    def test_one_batch_shapes_and_dtypes(self, tmp_path):
        root, expected_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=8)
        ds = KittiDepthDataset(
            expected_pairs,
            batch_size=4,
            patch_size=32,
            is_training=False,
            workers=1,
        )
        assert len(ds) >= 1
        rgb, depth, valid_mask, k = ds[0]

        assert rgb.shape == (4, 32, 32, 3)
        assert depth.shape == (4, 32, 32, 1)
        assert valid_mask.shape == (4, 32, 32, 1)
        assert k.shape == (4, 4)
        assert rgb.dtype == np.float32
        assert depth.dtype == np.float32
        assert valid_mask.dtype == np.float32
        assert k.dtype == np.float32

        # valid_mask correctly excludes zero-depth pixels across the whole batch.
        assert np.all(depth[valid_mask == 0.0] == 0.0)
        assert np.all(depth[valid_mask == 1.0] > 0.0)
        assert set(np.unique(valid_mask)).issubset({0.0, 1.0})

    def test_broken_pair_in_list_is_skipped_not_raised(self, tmp_path):
        root, expected_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=4)
        broken_pair = (
            str(tmp_path / "missing_rgb.png"),
            str(tmp_path / "missing_depth.png"),
        )
        pairs_with_broken = [broken_pair] + expected_pairs

        ds = KittiDepthDataset(
            pairs_with_broken,
            batch_size=4,
            patch_size=32,
            is_training=False,
            workers=1,
        )
        # Must not raise despite the first pair being unloadable.
        rgb, depth, valid_mask, k = ds[0]
        assert rgb.shape == (4, 32, 32, 3)

    def test_empty_pairs_list_raises_value_error(self):
        with pytest.raises(ValueError):
            KittiDepthDataset([], batch_size=4, patch_size=32)


@pytest.mark.skipif(
    not Path(KITTI_DEPTH_ROOT).exists(),
    reason=f"Real KITTI depth mirror not present at {KITTI_DEPTH_ROOT}",
)
class TestKittiDepthDatasetReal:
    def test_one_batch_from_real_discovered_pairs(self):
        pairs = discover_kitti_depth_pairs(KITTI_DEPTH_ROOT, max_files=8)
        assert len(pairs) > 0
        ds = KittiDepthDataset(
            pairs,
            batch_size=min(4, len(pairs)),
            patch_size=64,
            is_training=False,
            workers=1,
        )
        rgb, depth, valid_mask, k = ds[0]
        bs = min(4, len(pairs))
        assert rgb.shape == (bs, 64, 64, 3)
        assert depth.shape == (bs, 64, 64, 1)
        assert valid_mask.shape == (bs, 64, 64, 1)
        assert k.shape == (bs, 4)
        assert np.isfinite(rgb).all()
        assert np.isfinite(depth).all()
        assert np.isfinite(k).all()
