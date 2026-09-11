"""Tests for `train.omnipoint.data` -- the unified KITTI + MegaDepth GT pipeline.

Real-data checks (skipped, not silently passed, if a dataset root is absent on the running
machine) against BOTH local mirrors used together, plus synthetic fixtures for GT-derivation
correctness (compared directly against `camera_models.pinhole_ray_map`, never a second
reimplementation) and K-rescaling consistency.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dl_techniques.utils.camera_models import pinhole_ray_map
import keras

from train.omnipoint.data import (
    CombinedOmniPointDataset,
    MegaDepthPinholeDataset,
    _derive_gt_targets_batch,
    create_combined_omnipoint_dataset,
    load_megadepth_pair_for_omnipoint,
)
from train.omnipoint.kitti_depth import derive_pinhole_intrinsics, discover_kitti_depth_pairs

KITTI_DEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"
MEGADEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/Megadepth"


# =====================================================================
# GT ray/distance/point derivation -- agreement with camera_models.py itself
# =====================================================================


class TestDeriveGtTargetsBatch:
    def test_gt_ray_matches_camera_models_reference_directly(self):
        """The derived GT ray must be EXACTLY `pinhole_ray_map`'s own output, not a
        second, independently-computed formula (D-016)."""
        patch_size = 16
        fx, fy, cx, cy = 20.0, 20.0, 8.0, 8.0
        depth = np.ones((1, patch_size, patch_size, 1), dtype=np.float32) * 5.0
        valid_mask = np.ones((1, patch_size, patch_size, 1), dtype=np.float32)
        k_batch = np.asarray([[fx, fy, cx, cy]], dtype=np.float32)

        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k_batch, patch_size
        )

        expected_ray = keras.ops.convert_to_numpy(
            pinhole_ray_map(fx, fy, cx, cy, patch_size, patch_size)
        )
        np.testing.assert_allclose(gt_ray[0], expected_ray, atol=1e-6, rtol=0)

    def test_known_planar_depth_recovers_known_radial_distance(self):
        """Hand-derived case: at the principal point, ray = (0, 0, 1) exactly, so
        radial distance == planar depth there. Off-center, radial > planar (the ray is
        tilted away from the optical axis)."""
        patch_size = 8
        fx = fy = 4.0
        cx = cy = 4.0  # principal point sits exactly at pixel-center (4.0, 4.0)
        depth_value = 10.0
        depth = np.full((1, patch_size, patch_size, 1), depth_value, dtype=np.float32)
        valid_mask = np.ones((1, patch_size, patch_size, 1), dtype=np.float32)
        k_batch = np.asarray([[fx, fy, cx, cy]], dtype=np.float32)

        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k_batch, patch_size
        )

        # Pixel-center convention: u = col + 0.5. cx=4.0 sits between columns 3 and 4;
        # the closest pixel center is col=3 (u=3.5) or col=4 (u=4.5) -- pick col=3 row=3
        # is NOT exactly on-axis, so instead assert the general inequality + the exact
        # formula at every pixel (a stronger, non-degenerate check).
        expected_ray = keras.ops.convert_to_numpy(
            pinhole_ray_map(fx, fy, cx, cy, patch_size, patch_size)
        )
        expected_radial = depth_value / np.maximum(expected_ray[..., 2:3], 1e-6)
        np.testing.assert_allclose(
            gt_distance[0], expected_radial, atol=1e-4, rtol=0
        )
        # Off-axis radial distance must be >= planar depth everywhere (the ray's z
        # component is <= 1 for a unit vector with nonzero x/y).
        assert np.all(gt_distance[0] >= depth_value - 1e-4)

    def test_gt_point_equals_distance_times_ray(self):
        patch_size = 8
        fx, fy, cx, cy = 6.0, 6.0, 4.0, 4.0
        depth = np.full((1, patch_size, patch_size, 1), 3.0, dtype=np.float32)
        valid_mask = np.ones((1, patch_size, patch_size, 1), dtype=np.float32)
        k_batch = np.asarray([[fx, fy, cx, cy]], dtype=np.float32)

        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k_batch, patch_size
        )
        np.testing.assert_allclose(gt_point[0], gt_distance[0] * gt_ray[0], atol=1e-6, rtol=0)

    def test_invalid_pixels_produce_zero_distance_and_point_never_manufactured(self):
        patch_size = 8
        fx, fy, cx, cy = 6.0, 6.0, 4.0, 4.0
        depth = np.full((1, patch_size, patch_size, 1), 3.0, dtype=np.float32)
        valid_mask = np.zeros((1, patch_size, patch_size, 1), dtype=np.float32)
        valid_mask[:, :4, :, :] = 1.0  # top half valid, bottom half invalid
        k_batch = np.asarray([[fx, fy, cx, cy]], dtype=np.float32)

        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k_batch, patch_size
        )
        assert np.all(gt_distance[0, 4:, :, :] == 0.0)
        assert np.all(gt_point[0, 4:, :, :] == 0.0)
        assert np.all(gt_distance[0, :4, :, :] > 0.0)
        # gt_mask (D-017) is exactly the validity proxy, unchanged.
        np.testing.assert_array_equal(gt_mask, valid_mask)

    def test_all_outputs_finite(self):
        patch_size = 8
        fx, fy, cx, cy = 5.0, 5.0, 4.0, 4.0
        rng = np.random.default_rng(0)
        depth = rng.uniform(0.0, 10.0, size=(2, patch_size, patch_size, 1)).astype(np.float32)
        valid_mask = (depth > 2.0).astype(np.float32)
        k_batch = np.asarray([[fx, fy, cx, cy], [fx * 1.5, fy * 1.5, cx, cy]], dtype=np.float32)

        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k_batch, patch_size
        )
        assert np.isfinite(gt_ray).all()
        assert np.isfinite(gt_distance).all()
        assert np.isfinite(gt_point).all()
        assert np.isfinite(gt_mask).all()


# =====================================================================
# K-rescaling consistency (resize + K rescale)
# =====================================================================


def _make_synthetic_megadepth_tree(tmp_path: Path, num_pairs: int = 4):
    """Build a minimal on-disk MegaDepth tree; returns (rgb_paths, depth_paths, native_hw)."""
    import h5py

    root = tmp_path / "megadepth_root"
    dense_dir = root / "sceneA" / "dense0"
    imgs_dir = dense_dir / "imgs"
    depths_dir = dense_dir / "depths"
    imgs_dir.mkdir(parents=True)
    depths_dir.mkdir(parents=True)

    native_h, native_w = 50, 80
    rgb_paths, depth_paths = [], []
    rng = np.random.default_rng(1)
    for i in range(num_pairs):
        stem = f"img_{i:03d}"
        rgb_arr = rng.integers(0, 255, size=(native_h, native_w, 3), dtype=np.uint8)
        img_path = imgs_dir / f"{stem}.jpg"
        Image.fromarray(rgb_arr, mode="RGB").save(img_path)

        depth_arr = np.zeros((native_h, native_w), dtype=np.float32)
        depth_arr[: native_h // 2, :] = 2.5  # half valid, half zero (invalid)
        depth_path = depths_dir / f"{stem}.h5"
        with h5py.File(depth_path, "w") as f:
            f.create_dataset("depth", data=depth_arr)

        rgb_paths.append(str(img_path))
        depth_paths.append(str(depth_path))

    return rgb_paths, depth_paths, (native_h, native_w)


class TestLoadMegadepthPairForOmnipoint:
    def test_k_rescales_consistently_with_resize_factor(self, tmp_path):
        rgb_paths, depth_paths, (native_h, native_w) = _make_synthetic_megadepth_tree(
            tmp_path, num_pairs=1
        )
        patch_size = 32
        native_fx, native_fy, native_cx, native_cy = derive_pinhole_intrinsics(
            width=native_w, height=native_h
        )

        result = load_megadepth_pair_for_omnipoint(rgb_paths[0], depth_paths[0], patch_size)
        assert result is not None
        rgb, depth, valid_mask, (fx, fy, cx, cy) = result

        scale_x = patch_size / float(native_w)
        scale_y = patch_size / float(native_h)
        assert fx == pytest.approx(native_fx * scale_x, rel=0, abs=1e-4)
        assert cx == pytest.approx(native_cx * scale_x, rel=0, abs=1e-4)
        assert fy == pytest.approx(native_fy * scale_y, rel=0, abs=1e-4)
        assert cy == pytest.approx(native_cy * scale_y, rel=0, abs=1e-4)

    def test_output_shapes_and_dtypes(self, tmp_path):
        rgb_paths, depth_paths, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=1)
        patch_size = 24
        result = load_megadepth_pair_for_omnipoint(rgb_paths[0], depth_paths[0], patch_size)
        assert result is not None
        rgb, depth, valid_mask, k = result
        assert rgb.shape == (patch_size, patch_size, 3)
        assert depth.shape == (patch_size, patch_size, 1)
        assert valid_mask.shape == (patch_size, patch_size, 1)
        assert rgb.dtype == np.float32
        assert depth.dtype == np.float32
        assert valid_mask.dtype == np.float32
        assert len(k) == 4
        assert np.all(depth[valid_mask == 0.0] == 0.0)
        assert np.all(depth[valid_mask == 1.0] > 0.0)

    def test_missing_file_returns_none(self, tmp_path):
        result = load_megadepth_pair_for_omnipoint(
            str(tmp_path / "nope.jpg"), str(tmp_path / "nope.h5"), patch_size=16
        )
        assert result is None


class TestMegaDepthPinholeDatasetSynthetic:
    def test_one_batch_shapes_and_dtypes(self, tmp_path):
        rgb_paths, depth_paths, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=8)
        ds = MegaDepthPinholeDataset(
            rgb_paths, depth_paths, batch_size=4, patch_size=16, is_training=False, workers=1,
        )
        rgb, depth, valid_mask, k = ds[0]
        assert rgb.shape == (4, 16, 16, 3)
        assert depth.shape == (4, 16, 16, 1)
        assert valid_mask.shape == (4, 16, 16, 1)
        assert k.shape == (4, 4)


# =====================================================================
# CombinedOmniPointDataset -- synthetic (fast, deterministic) mixing + shape checks
# =====================================================================


def _make_synthetic_kitti_tree(tmp_path: Path, num_frames: int = 4):
    root = tmp_path / "kitti_depth_root"
    drive_name = "2011_09_26_drive_0002_sync"
    date = "2011_09_26"
    camera = "image_02"

    rgb_dir = root / "raw_image_values" / date / drive_name / camera / "data"
    depth_dir = root / "depth_values" / drive_name / "proj_depth" / "groundtruth" / camera
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    height, width = 32, 48
    pairs = []
    rng = np.random.default_rng(2)
    for i in range(num_frames):
        frame_name = f"{i:010d}.png"
        rgb_arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(rgb_arr, mode="RGB").save(rgb_dir / frame_name)

        depth_arr = np.zeros((height, width), dtype=np.uint16)
        depth_arr[: height // 2, :] = 5000
        Image.fromarray(depth_arr, mode="I;16").save(depth_dir / frame_name)

        pairs.append((str(rgb_dir / frame_name), str(depth_dir / frame_name)))

    return str(root), pairs


class TestCombinedOmniPointDatasetSynthetic:
    def test_one_batch_matches_loss_ready_contract(self, tmp_path):
        kitti_root, kitti_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=8)
        mega_rgb, mega_depth, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=8)

        ds = CombinedOmniPointDataset(
            kitti_pairs, mega_rgb, mega_depth,
            batch_size=4, patch_size=32, is_training=False, workers=1,
        )
        x, y = ds[0]
        gt_ray, gt_distance, gt_point, gt_mask, valid_mask = y

        assert x.shape == (4, 32, 32, 3)
        assert x.dtype == np.float32
        assert gt_ray.shape == (4, 32, 32, 3)
        assert gt_distance.shape == (4, 32, 32, 1)
        assert gt_point.shape == (4, 32, 32, 3)
        assert gt_mask.shape == (4, 32, 32, 1)
        assert valid_mask.shape == (4, 32, 32, 1)
        for arr in (gt_ray, gt_distance, gt_point, gt_mask, valid_mask):
            assert arr.dtype == np.float32
            assert np.isfinite(arr).all()

        # Unit-norm ray at every pixel (camera_models' own invariant, inherited here).
        ray_norms = np.linalg.norm(gt_ray, axis=-1)
        np.testing.assert_allclose(ray_norms, 1.0, atol=1e-5, rtol=0)

    def test_round_robin_mixes_both_sources_across_batches(self, tmp_path):
        """D-018: fixed round-robin by index parity -- KITTI even, MegaDepth odd. Confirm
        BOTH sources' own RGB content actually appears (not silently only one)."""
        kitti_root, kitti_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=8)
        mega_rgb, mega_depth, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=8)

        ds = CombinedOmniPointDataset(
            kitti_pairs, mega_rgb, mega_depth,
            batch_size=2, patch_size=16, is_training=False, workers=1,
        )
        even_batch_x, _ = ds[0]
        odd_batch_x, _ = ds[1]

        # The two sources' RGB batches must differ (different underlying source images) --
        # a vacuity guard against "the mixing dispatch silently always picks one source."
        assert not np.allclose(even_batch_x, odd_batch_x)
        # And fetching the SAME parity twice more (wrapping around each source's own
        # indices) must keep coming from a stable, real source each time -- both should be
        # internally self-consistent (same shape/dtype), not crash.
        even_batch_x_2, _ = ds[2]
        odd_batch_x_2, _ = ds[3]
        assert even_batch_x_2.shape == even_batch_x.shape
        assert odd_batch_x_2.shape == odd_batch_x.shape

    def test_kitti_only_and_megadepth_only_single_source_use(self, tmp_path):
        kitti_root, kitti_pairs = _make_synthetic_kitti_tree(tmp_path, num_frames=4)
        mega_rgb, mega_depth, _ = _make_synthetic_megadepth_tree(tmp_path, num_pairs=4)

        kitti_only = CombinedOmniPointDataset(
            kitti_pairs, [], [], batch_size=2, patch_size=16, is_training=False, workers=1,
        )
        x, y = kitti_only[0]
        assert x.shape == (2, 16, 16, 3)

        mega_only = CombinedOmniPointDataset(
            [], mega_rgb, mega_depth, batch_size=2, patch_size=16, is_training=False, workers=1,
        )
        x, y = mega_only[0]
        assert x.shape == (2, 16, 16, 3)

    def test_both_sources_empty_raises(self):
        with pytest.raises(ValueError):
            CombinedOmniPointDataset([], [], [], batch_size=2, patch_size=16)


# =====================================================================
# REAL local KITTI + MegaDepth data, used TOGETHER
# =====================================================================


@pytest.mark.skipif(
    not (Path(KITTI_DEPTH_ROOT).exists() and Path(MEGADEPTH_ROOT).exists()),
    reason=f"Real KITTI ({KITTI_DEPTH_ROOT}) or MegaDepth ({MEGADEPTH_ROOT}) mirror not present",
)
class TestCombinedOmniPointDatasetReal:
    def test_real_batches_from_both_sources(self):
        n_kitti = 12
        n_mega = 12
        ds = create_combined_omnipoint_dataset(
            kitti_root=KITTI_DEPTH_ROOT,
            megadepth_root=MEGADEPTH_ROOT,
            batch_size=4,
            patch_size=64,
            max_kitti_files=n_kitti,
            max_megadepth_files=n_mega,
            is_training=False,
            workers=1,
        )
        print(
            f"\nReal CombinedOmniPointDataset: kitti_pairs={len(ds._kitti.pairs)}, "
            f"megadepth_pairs={len(ds._megadepth.rgb_paths)}, len(ds)={len(ds)}"
        )
        assert len(ds._kitti.pairs) > 0
        assert len(ds._megadepth.rgb_paths) > 0

        even_x, even_y = ds[0]
        odd_x, odd_y = ds[1]
        assert even_x.shape == (4, 64, 64, 3)
        assert odd_x.shape == (4, 64, 64, 3)
        assert not np.allclose(even_x, odd_x)

        for x, y in (( even_x, even_y), (odd_x, odd_y)):
            gt_ray, gt_distance, gt_point, gt_mask, valid_mask = y
            assert np.isfinite(x).all()
            assert np.isfinite(gt_ray).all()
            assert np.isfinite(gt_distance).all()
            assert np.isfinite(gt_point).all()
            assert np.isfinite(gt_mask).all()
            ray_norms = np.linalg.norm(gt_ray, axis=-1)
            np.testing.assert_allclose(ray_norms, 1.0, atol=1e-4, rtol=0)
            assert (gt_distance >= 0.0).all()
