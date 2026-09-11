"""Synthetic cross-camera VALIDATION-only dataset staging (plan Step 10).

.. warning::
    **This script generates SYNTHETIC reprojected data, never real fisheye/panorama
    captures.** Per plan decision D-001 (`plans/plan-2026-09-11T050223-1b47bcf6/decisions.md`),
    exhaustive local search found zero real fisheye/panorama/equirectangular imagery
    anywhere under `/media/arxwn/data0_4tb/datasets/` -- the same HARD constraint Step 6's
    in-process test (`tests/test_models/test_omnipoint/test_cross_camera_synthetic.py`) is
    built around. Every fisheye/equirectangular image this script writes is a real
    nearest-neighbor RESAMPLE of a genuine pinhole KITTI RGB+depth pair, reprojected
    through `camera_models.py`'s own fisheye/equirectangular inverse-projection formulas --
    it is a code-path-validation artifact, not a claim of real-world cross-camera
    generalization. A future reader of `omnipoint_synth/` MUST NOT mistake this directory
    for real evaluation data proving OmniPoint's paper-scale camera-agnostic claim; that
    claim is explicitly out of scope for this plan (see `models/vision/omnipoint/README.md`,
    `train/omnipoint/README.md`).

**Scope boundary (do not extend without a new plan step).** This is a STANDALONE, one-time
data-staging tool for future manual/eval use. It is NOT wired into `train_omnipoint.py`'s
training loop, its CLI, or any automated pipeline -- there is no import of this module
anywhere in `train_omnipoint.py`, and there must never be one added silently. Running this
script is a deliberate, separate, manual action; it does not run as a side effect of
training or of the automated test suite.

**Reprojection method** (all camera-geometry math imported from `camera_models.py`, none
re-derived here -- mirrors Step 6's own test-local reprojection helper in
`tests/test_models/test_omnipoint/test_cross_camera_synthetic.py`, kept local to that test
and to this script rather than promoted into `camera_models.py` itself; see the module-level
D-022 decision note below for why):

    1. Load a real KITTI pinhole RGB+depth pair (`kitti_depth.load_kitti_depth_pair`),
       which also derives that image's pinhole intrinsics `K` (D-006 approximation).
    2. Generate the pinhole camera's own per-pixel unit ray map (`pinhole_ray_map`). Ray
       DIRECTION does not depend on depth, so this ray map tells us, for every pinhole
       pixel, the exact 3D camera-space direction a co-located fisheye/equirectangular
       sensor would need to look along to see the same scene point.
    3. Convert KITTI's stored PLANAR depth to RADIAL (Euclidean) distance via
       `radial = planar / ray_z` (D-016's own formula, reused unchanged) -- radial distance
       is camera-model-invariant (the same physical point is the same distance from the
       camera regardless of which lens observes it), so this value is what gets carried
       over to the fisheye/equirectangular reprojections below unchanged.
    4. Feed the SAME pinhole ray directions into `fisheye_unproject_points` /
       `equirectangular_unproject_points` to get where each pinhole pixel's content lands
       on a same-resolution fisheye / equirectangular sensor.
    5. Nearest-neighbor-scatter the pinhole RGB, radial distance, and validity mask into
       that target pixel coordinate (last-write-wins on collisions; uncovered target
       pixels stay zero) -- exactly Step 6's `_scatter_reproject` logic, generalized here
       to also carry the depth/mask channels alongside RGB.

Output layout under ``output_dir`` (default
``/media/arxwn/data0_4tb/datasets/omnipoint_synth/``, created if absent -- this directory
itself, like every other dataset under ``data0_4tb``, is NOT tracked in git; only this
script is)::

    omnipoint_synth/
        fisheye/
            ray_map.npy          # (H, W, 3) unit ray map, shared by every sample
            {i}_rgb.png          # reprojected RGB, uint8
            {i}_depth.npy        # (H, W) float32 radial distance, 0.0 where uncovered
            {i}_valid_mask.npy   # (H, W) float32, 1.0 = covered AND real KITTI depth
        equirectangular/
            ray_map.npy
            {i}_rgb.png
            {i}_depth.npy
            {i}_valid_mask.npy
        manifest.json            # source KITTI pairs, camera parameters, generation params
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.camera_models import (
    equirectangular_ray_map,
    equirectangular_unproject_points,
    fisheye_ray_map,
    fisheye_unproject_points,
    pinhole_ray_map,
)
from dl_techniques.utils.logger import logger
from train.omnipoint.kitti_depth import (
    KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
    discover_kitti_depth_pairs,
    load_kitti_depth_pair,
)

# DECISION plan-2026-09-11T050223-1b47bcf6/D-022
# Do NOT promote Step 6's `_scatter_reproject`-style helper into `camera_models.py`. That
# file (Step 1, commit d922f3a00) is a pure ray-geometry module -- every function in it maps
# rays <-> pixel coordinates with no image/array data movement at all. A "scatter an RGB/
# depth image into reprojected pixel coordinates" helper is a DIFFERENT layer of concern
# (raster resampling, not ray geometry) with exactly ONE other call site in this whole plan
# (the Step 6 test). Two call sites across two files does not meet this repo's own
# earned-abstraction bar strongly enough to justify reopening a already-shipped, already-
# tested foundation file (Step 1 is the plan's highest-risk, most-depended-on module --
# every other step imports it) for a helper this script can trivially reimplement locally in
# ~15 lines. See decisions.md D-022.
KITTI_DEPTH_ROOT_DEFAULT = "/media/arxwn/data0_4tb/datasets/KITTI/data/depth"
OUTPUT_DIR_DEFAULT = "/media/arxwn/data0_4tb/datasets/omnipoint_synth/"

_EPSILON = 1e-8


def _scatter_reproject(
    rgb: np.ndarray,
    radial_distance: np.ndarray,
    valid_mask: np.ndarray,
    dst_uv: np.ndarray,
    dst_height: int,
    dst_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbor-scatter RGB + radial distance + validity into a target canvas.

    Mirrors `test_cross_camera_synthetic.py`'s own `_scatter_reproject`, generalized to
    also carry the depth/mask channels (that test only needed RGB).

    :param rgb: Source RGB image, ``(H, W, 3)``, values in ``[0, 1]``.
    :type rgb: np.ndarray
    :param radial_distance: Source per-pixel radial distance, ``(H, W)``.
    :type radial_distance: np.ndarray
    :param valid_mask: Source per-pixel validity, ``(H, W)``, ``1.0``/``0.0``.
    :type valid_mask: np.ndarray
    :param dst_uv: Target ``(u, v)`` pixel coordinates for every source pixel, ``(H, W, 2)``.
    :type dst_uv: np.ndarray
    :param dst_height: Target canvas height.
    :type dst_height: int
    :param dst_width: Target canvas width.
    :type dst_width: int
    :return: ``(rgb_canvas, depth_canvas, mask_canvas)`` -- ``(dst_height, dst_width, 3)``,
        ``(dst_height, dst_width)``, ``(dst_height, dst_width)``. Zero everywhere no source
        pixel landed. Collisions resolve last-write-wins (row-major source iteration order).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    rgb_canvas = np.zeros((dst_height, dst_width, 3), dtype=np.float32)
    depth_canvas = np.zeros((dst_height, dst_width), dtype=np.float32)
    mask_canvas = np.zeros((dst_height, dst_width), dtype=np.float32)

    u = np.round(dst_uv[..., 0]).astype(np.int64)
    v = np.round(dst_uv[..., 1]).astype(np.int64)
    valid = (u >= 0) & (u < dst_width) & (v >= 0) & (v < dst_height)

    rgb_canvas[v[valid], u[valid]] = rgb[valid]
    depth_canvas[v[valid], u[valid]] = radial_distance[valid]
    mask_canvas[v[valid], u[valid]] = valid_mask[valid]
    return rgb_canvas, depth_canvas, mask_canvas


def _planar_depth_to_radial(depth_planar: np.ndarray, pinhole_rays: np.ndarray) -> np.ndarray:
    """Convert KITTI's stored planar (z-axis) depth to radial (Euclidean) distance.

    Same formula as `train.omnipoint.data`'s `_derive_gt_targets_batch` (D-016):
    ``radial = planar / ray_z``, where ``ray_z`` is the pinhole unit ray's z-component.

    :param depth_planar: Planar depth, ``(H, W)``.
    :type depth_planar: np.ndarray
    :param pinhole_rays: Pinhole unit ray map, ``(H, W, 3)``.
    :type pinhole_rays: np.ndarray
    :return: Radial distance, ``(H, W)``.
    :rtype: np.ndarray
    """
    ray_z = np.maximum(pinhole_rays[..., 2], _EPSILON)
    return depth_planar / ray_z


def _write_sample(
    output_dir: Path,
    camera_name: str,
    index: int,
    rgb_canvas: np.ndarray,
    depth_canvas: np.ndarray,
    mask_canvas: np.ndarray,
) -> Dict[str, str]:
    """Write one reprojected sample's RGB/depth/mask files, return their relative paths."""
    from PIL import Image

    camera_dir = output_dir / camera_name
    camera_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = camera_dir / f"{index}_rgb.png"
    depth_path = camera_dir / f"{index}_depth.npy"
    mask_path = camera_dir / f"{index}_valid_mask.npy"

    rgb_uint8 = np.clip(rgb_canvas * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgb_uint8, mode="RGB").save(rgb_path)
    np.save(depth_path, depth_canvas.astype(np.float32))
    np.save(mask_path, mask_canvas.astype(np.float32))

    return {
        "rgb": str(rgb_path.relative_to(output_dir)),
        "depth": str(depth_path.relative_to(output_dir)),
        "valid_mask": str(mask_path.relative_to(output_dir)),
    }


def generate_synthetic_camera_eval_set(
    kitti_root: str,
    output_dir: str,
    num_samples: int = 20,
    patch_size: int = 224,
    fisheye_focal_ratio: float = 0.7,
    horizontal_fov_deg: float = KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
) -> Dict[str, Any]:
    """Generate a synthetic cross-camera VALIDATION-only eval set from real KITTI pairs.

    Reads up to ``num_samples`` real KITTI RGB+depth pairs, reprojects each through the
    fisheye and equirectangular camera models (see module docstring for the exact method),
    and writes the results plus a manifest under ``output_dir``.

    :param kitti_root: Path to the KITTI depth-benchmark root (forwarded to
        `discover_kitti_depth_pairs`).
    :type kitti_root: str
    :param output_dir: Directory to write the staged synthetic dataset into. Created if it
        does not exist.
    :type output_dir: str
    :param num_samples: Maximum number of KITTI pairs to reproject. Defaults to ``20``.
    :type num_samples: int
    :param patch_size: Output spatial size (height and width) for every image, forwarded to
        `load_kitti_depth_pair`. Defaults to ``224``.
    :type patch_size: int
    :param fisheye_focal_ratio: The synthetic fisheye camera's focal length ``f``, expressed
        as a fraction of ``patch_size`` (``f = fisheye_focal_ratio * patch_size``). Chosen
        so the pinhole's own ray cone maps to a fisheye sensor region of comparable size,
        matching Step 6's own test convention. Defaults to ``0.7``.
    :type fisheye_focal_ratio: float
    :param horizontal_fov_deg: Assumed horizontal FOV forwarded to
        `load_kitti_depth_pair`/`derive_pinhole_intrinsics` (D-006 approximation).
    :type horizontal_fov_deg: float
    :return: The manifest dict written to ``manifest.json`` (also returned for tests, so a
        caller does not need to re-read the file it just wrote).
    :rtype: Dict[str, Any]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = discover_kitti_depth_pairs(kitti_root, max_files=num_samples)
    if not pairs:
        raise ValueError(
            f"generate_synthetic_camera_eval_set: no KITTI pairs discovered under "
            f"{kitti_root!r} -- cannot generate a synthetic eval set from zero source pairs."
        )

    fisheye_f = fisheye_focal_ratio * patch_size
    cx = cy = patch_size / 2.0

    fisheye_rays = keras.ops.convert_to_numpy(
        fisheye_ray_map(fisheye_f, cx, cy, patch_size, patch_size)
    )
    equirect_rays = keras.ops.convert_to_numpy(
        equirectangular_ray_map(patch_size, patch_size)
    )
    (output_path / "fisheye").mkdir(parents=True, exist_ok=True)
    (output_path / "equirectangular").mkdir(parents=True, exist_ok=True)
    np.save(output_path / "fisheye" / "ray_map.npy", fisheye_rays)
    np.save(output_path / "equirectangular" / "ray_map.npy", equirect_rays)

    manifest: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warning": (
            "SYNTHETIC reprojected data for code-path validation ONLY -- not real "
            "fisheye/panorama captures. See module docstring of "
            "src/train/omnipoint/generate_synthetic_camera_eval_set.py."
        ),
        "kitti_root": str(kitti_root),
        "patch_size": patch_size,
        "horizontal_fov_deg": horizontal_fov_deg,
        "camera_params": {
            "fisheye": {"f": fisheye_f, "cx": cx, "cy": cy},
            "equirectangular": {"height": patch_size, "width": patch_size},
        },
        "samples": [],
    }

    written = 0
    for i, (rgb_path, depth_path) in enumerate(pairs):
        loaded = load_kitti_depth_pair(
            rgb_path, depth_path, patch_size, horizontal_fov_deg=horizontal_fov_deg
        )
        if loaded is None:
            logger.warning(
                f"generate_synthetic_camera_eval_set: skipping unusable KITTI pair "
                f"(rgb={rgb_path}, depth={depth_path})"
            )
            continue
        rgb_signed, depth_meters, valid_mask, k = loaded
        rgb01 = np.clip((rgb_signed + 1.0) / 2.0, 0.0, 1.0)
        fx, fy, kcx, kcy = k

        pinhole_rays = keras.ops.convert_to_numpy(
            pinhole_ray_map(fx, fy, kcx, kcy, patch_size, patch_size)
        )
        radial_distance = _planar_depth_to_radial(depth_meters[..., 0], pinhole_rays)

        fisheye_dst_uv = keras.ops.convert_to_numpy(
            fisheye_unproject_points(pinhole_rays, f=fisheye_f, cx=cx, cy=cy)
        )
        equirect_dst_uv = keras.ops.convert_to_numpy(
            equirectangular_unproject_points(pinhole_rays, height=patch_size, width=patch_size)
        )

        fisheye_rgb, fisheye_depth, fisheye_mask = _scatter_reproject(
            rgb01, radial_distance, valid_mask[..., 0], fisheye_dst_uv, patch_size, patch_size
        )
        equirect_rgb, equirect_depth, equirect_mask = _scatter_reproject(
            rgb01, radial_distance, valid_mask[..., 0], equirect_dst_uv, patch_size, patch_size
        )

        fisheye_files = _write_sample(
            output_path, "fisheye", written, fisheye_rgb, fisheye_depth, fisheye_mask
        )
        equirect_files = _write_sample(
            output_path, "equirectangular", written, equirect_rgb, equirect_depth, equirect_mask
        )

        manifest["samples"].append(
            {
                "index": written,
                "source_rgb": rgb_path,
                "source_depth": depth_path,
                "source_intrinsics": {"fx": fx, "fy": fy, "cx": kcx, "cy": kcy},
                "fisheye": fisheye_files,
                "equirectangular": equirect_files,
            }
        )
        written += 1

    manifest["num_samples_written"] = written
    manifest["num_source_pairs_considered"] = len(pairs)

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"generate_synthetic_camera_eval_set: wrote {written} synthetic fisheye + "
        f"{written} synthetic equirectangular samples to {output_path}"
    )
    return manifest


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. First statement of `main()`, so `--help` exits 0 for free."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage a SYNTHETIC cross-camera validation-only eval set: reprojects real "
            "KITTI pinhole RGB+depth pairs through the fisheye and equirectangular camera "
            "models in dl_techniques.utils.camera_models. Standalone tooling, NOT wired "
            "into train_omnipoint.py's training loop -- see this script's own module "
            "docstring for the scope boundary."
        )
    )
    parser.add_argument(
        "--kitti-root",
        type=str,
        default=KITTI_DEPTH_ROOT_DEFAULT,
        help="Path to the KITTI depth-benchmark root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory to write the staged synthetic dataset into (created if absent).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Maximum number of KITTI pairs to reproject.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Output spatial size (height and width) for every generated image.",
    )
    parser.add_argument(
        "--fisheye-focal-ratio",
        type=float,
        default=0.7,
        help="Synthetic fisheye focal length as a fraction of --patch-size.",
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
        help="Assumed horizontal FOV for the KITTI derived-intrinsics approximation (D-006).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    manifest = generate_synthetic_camera_eval_set(
        kitti_root=args.kitti_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        fisheye_focal_ratio=args.fisheye_focal_ratio,
        horizontal_fov_deg=args.horizontal_fov_deg,
    )
    print(
        f"Wrote {manifest['num_samples_written']} synthetic samples "
        f"(considered {manifest['num_source_pairs_considered']} source pairs) to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
