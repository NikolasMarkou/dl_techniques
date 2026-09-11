"""Tests for `train.omnipoint.generate_synthetic_camera_eval_set`.

Two kinds of check:
- `--help` exits 0 (CLI contract).
- The core generation function, run against a SMALL number of REAL local KITTI pairs,
  writing to a pytest `tmp_path` (never the real `/media/arxwn/data0_4tb/datasets/
  omnipoint_synth/`, never repo-root `results/`). Skipped, not silently passed, if this
  machine's local KITTI mirror is absent.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

from train.omnipoint.generate_synthetic_camera_eval_set import (
    generate_synthetic_camera_eval_set,
    parse_arguments,
)
from train.omnipoint.kitti_depth import discover_kitti_depth_pairs

KITTI_DEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"

_kitti_available = len(discover_kitti_depth_pairs(KITTI_DEPTH_ROOT, max_files=1)) > 0

requires_real_kitti = pytest.mark.skipif(
    not _kitti_available,
    reason=f"Real KITTI depth mirror not found at {KITTI_DEPTH_ROOT} on this machine.",
)


# =====================================================================
# CLI contract
# =====================================================================


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "train.omnipoint.generate_synthetic_camera_eval_set",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    def test_parse_arguments_defaults(self):
        args = parse_arguments([])
        assert args.num_samples == 20
        assert args.patch_size == 224
        assert "omnipoint_synth" in args.output_dir


# =====================================================================
# Core generation function
# =====================================================================


@requires_real_kitti
class TestGenerateSyntheticCameraEvalSet:
    NUM_SAMPLES = 3
    PATCH_SIZE = 56  # small + fast; 56 % 14 == 0, matches Step 6's test convention

    def _generate(self, tmp_path):
        output_dir = tmp_path / "omnipoint_synth"
        manifest = generate_synthetic_camera_eval_set(
            kitti_root=KITTI_DEPTH_ROOT,
            output_dir=str(output_dir),
            num_samples=self.NUM_SAMPLES,
            patch_size=self.PATCH_SIZE,
        )
        return output_dir, manifest

    def test_manifest_is_valid_json_and_records_sources(self, tmp_path):
        output_dir, manifest = self._generate(tmp_path)

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            reloaded = json.load(f)

        assert reloaded["num_samples_written"] == manifest["num_samples_written"]
        assert manifest["num_samples_written"] > 0
        assert manifest["num_samples_written"] <= self.NUM_SAMPLES
        assert len(reloaded["samples"]) == reloaded["num_samples_written"]
        for sample in reloaded["samples"]:
            assert "source_rgb" in sample and "source_depth" in sample
            assert "fisheye" in sample and "equirectangular" in sample
        assert "SYNTHETIC" in reloaded["warning"]

    def test_output_files_created_with_correct_shapes(self, tmp_path):
        output_dir, manifest = self._generate(tmp_path)
        n = manifest["num_samples_written"]
        ps = self.PATCH_SIZE

        for camera_name in ("fisheye", "equirectangular"):
            camera_dir = output_dir / camera_name
            ray_map = np.load(camera_dir / "ray_map.npy")
            assert ray_map.shape == (ps, ps, 3)

            for i in range(n):
                depth = np.load(camera_dir / f"{i}_depth.npy")
                mask = np.load(camera_dir / f"{i}_valid_mask.npy")
                assert depth.shape == (ps, ps)
                assert mask.shape == (ps, ps)

                rgb_path = camera_dir / f"{i}_rgb.png"
                assert rgb_path.exists()
                from PIL import Image

                rgb = np.asarray(Image.open(rgb_path))
                assert rgb.shape == (ps, ps, 3)

    def test_reprojected_images_are_non_trivial_and_differ_from_source(self, tmp_path):
        output_dir, manifest = self._generate(tmp_path)
        n = manifest["num_samples_written"]
        assert n > 0

        from PIL import Image

        for i in range(n):
            fisheye_rgb = np.asarray(
                Image.open(output_dir / "fisheye" / f"{i}_rgb.png")
            ).astype(np.float32)
            equirect_rgb = np.asarray(
                Image.open(output_dir / "equirectangular" / f"{i}_rgb.png")
            ).astype(np.float32)

            # Non-trivial: not all-zero.
            assert fisheye_rgb.sum() > 0.0
            assert equirect_rgb.sum() > 0.0
            # Genuinely different resamples of the same source, not accidental copies.
            assert not np.allclose(fisheye_rgb, equirect_rgb)

    def test_ray_maps_are_unit_norm(self, tmp_path):
        output_dir, _ = self._generate(tmp_path)
        for camera_name in ("fisheye", "equirectangular"):
            ray_map = np.load(output_dir / camera_name / "ray_map.npy")
            norms = np.linalg.norm(ray_map, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-4, rtol=0.0)

    def test_raises_on_empty_kitti_root(self, tmp_path):
        empty_root = tmp_path / "no_such_kitti_root"
        with pytest.raises(ValueError):
            generate_synthetic_camera_eval_set(
                kitti_root=str(empty_root),
                output_dir=str(tmp_path / "out"),
                num_samples=3,
                patch_size=self.PATCH_SIZE,
            )
