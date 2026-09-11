"""KITTI depth-benchmark pair discovery + derived-intrinsics pinhole camera model.

.. warning::
    **`derive_pinhole_intrinsics` is NOT a calibration.** Per plan decision D-006
    (`plans/plan-2026-09-11T050223-1b47bcf6/decisions.md`), this module's `K` is derived
    from a single assumed horizontal field-of-view constant and each image's own
    width/height -- it is not read from any KITTI calibration file. This mirror of the
    KITTI depth-completion/-prediction benchmark (`raw_image_values/` +
    `depth_values/`/`val/`) does not carry `calib_cam_to_cam.txt` at any level (see
    `findings/dataset-and-training-pipeline.md` #2, #7), and KITTI's separate `object/`
    subset's calibration files use a different rectification convention that must NOT be
    conflated with this one. Every GT ray/distance direction computed downstream from this
    `K` (in `data.py`, Step 8) inherits this approximation's systematic bias against the
    dataset's true, unrecoverable intrinsics. Documented here first because this is the
    first module to introduce the formula; `data.py`'s MegaDepth loader imports and reuses
    this SAME function rather than re-deriving it.

On-disk layout (verified directly against this machine's copy at
``/media/arxwn/data0_4tb/datasets/KITTI/data/depth/``, not merely assumed from the
benchmark's published description)::

    {root}/raw_image_values/{date}/{date}_drive_{NNNN}_sync/image_0{2,3}/data/{frame}.png
    {root}/depth_values/{date}_drive_{NNNN}_sync/proj_depth/groundtruth/image_0{2,3}/{frame}.png
    {root}/val/{date}_drive_{NNNN}_sync/proj_depth/groundtruth/image_0{2,3}/{frame}.png

The RGB root (``raw_image_values``) nests one extra ``{date}`` level (the first 10
characters of the drive name, e.g. ``2011_09_26``) that the two depth-GT roots
(``depth_values`` for the train split, ``val`` for the validation split) do not -- the
matching key between an RGB frame and its depth-GT frame is
``({date}_drive_{NNNN}_sync, image_0{2,3}, {frame}.png)``, with ``{date}`` recovered from
the drive name's own first 10 characters rather than stored separately. Depth PNGs are the
standard KITTI 16-bit encoding: ``depth_meters = uint16_value / 256.0``, with ``0`` meaning
"no LiDAR return" (invalid), never a real zero-distance measurement.
"""

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# KITTI's raw-sequence RGB cameras are the two colour cameras, `image_02` (left) and
# `image_03` (right); `image_00`/`image_01` are the grayscale pair and are not covered
# by `proj_depth/groundtruth/` in this benchmark subset.
KITTI_DEPTH_CAMERAS: Tuple[str, ...] = ("image_02", "image_03")

# The two depth-GT roots this loader walks: `depth_values/` is the benchmark's train
# split, `val/` is its validation split. Both share the same
# `{drive}/proj_depth/groundtruth/{camera}/{frame}.png` internal shape.
KITTI_DEPTH_SPLIT_DIRS: Tuple[str, ...] = ("depth_values", "val")

# Standard KITTI depth-PNG encoding: stored uint16 value / 256.0 = depth in meters;
# 0 = invalid (no LiDAR return), not a real zero-distance sample.
KITTI_DEPTH_PNG_SCALE = 256.0

# Assumed horizontal field-of-view (degrees) used by `derive_pinhole_intrinsics` in the
# absence of any real calibration file for this dataset (D-006). This single constant is
# the entire documented approximation.
KITTI_DEFAULT_HORIZONTAL_FOV_DEG = 70.0


# ---------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------


def discover_kitti_depth_pairs(
    root_dir: str,
    splits: Sequence[str] = KITTI_DEPTH_SPLIT_DIRS,
    cameras: Sequence[str] = KITTI_DEPTH_CAMERAS,
    max_files: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Discover matched (rgb_path, depth_path) pairs under a KITTI depth-benchmark root.

    Walks ``{root_dir}/depth_values/`` (train split) and ``{root_dir}/val/`` (validation
    split) for ground-truth depth PNGs under
    ``{drive}/proj_depth/groundtruth/{camera}/{frame}.png``, and matches each one to its
    RGB frame under
    ``{root_dir}/raw_image_values/{date}/{drive}/{camera}/data/{frame}.png``, where
    ``{date}`` is the drive name's own first 10 characters (``YYYY_MM_DD``).

    A depth-GT frame whose RGB counterpart is missing, or whose depth PNG fails to open
    for any reason, is skipped and logged -- discovery never raises mid-walk over a
    single bad pair (Failure Modes table: "Path missing/corrupt -- skip and log, not
    crash mid-epoch").

    :param root_dir: Path to the KITTI depth-benchmark root (containing
        ``raw_image_values/``, ``depth_values/``, ``val/``).
    :type root_dir: str
    :param splits: Depth-GT split subdirectory names to walk, relative to ``root_dir``.
        Defaults to :data:`KITTI_DEPTH_SPLIT_DIRS` (both train and val).
    :type splits: Sequence[str]
    :param cameras: Camera subdirectory names to match (``image_02``/``image_03``).
        Defaults to :data:`KITTI_DEPTH_CAMERAS`.
    :type cameras: Sequence[str]
    :param max_files: Maximum number of pairs to return. ``None`` for unlimited.
    :type max_files: Optional[int]
    :return: List of ``(rgb_path, depth_path)`` absolute-path tuples.
    :rtype: List[Tuple[str, str]]
    """
    root = Path(root_dir)
    raw_root = root / "raw_image_values"
    pairs: List[Tuple[str, str]] = []

    for split in splits:
        split_root = root / split
        if not split_root.exists():
            logger.warning(f"KITTI depth split root missing, skipping: {split_root}")
            continue

        for drive_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            drive_name = drive_dir.name  # e.g. "2011_09_26_drive_0009_sync"
            date = drive_name[:10]  # "2011_09_26"
            gt_root = drive_dir / "proj_depth" / "groundtruth"
            if not gt_root.exists():
                continue

            for camera in cameras:
                depth_cam_dir = gt_root / camera
                if not depth_cam_dir.exists():
                    continue
                rgb_cam_dir = raw_root / date / drive_name / camera / "data"
                if not rgb_cam_dir.exists():
                    logger.warning(
                        f"KITTI RGB directory missing for drive {drive_name}, "
                        f"camera {camera}, skipping: {rgb_cam_dir}"
                    )
                    continue

                for depth_path in sorted(depth_cam_dir.glob("*.png")):
                    try:
                        rgb_path = rgb_cam_dir / depth_path.name
                        if not rgb_path.exists():
                            raise FileNotFoundError(
                                f"no matching RGB frame for depth GT {depth_path}"
                            )
                        pairs.append((str(rgb_path), str(depth_path)))
                    except Exception as exc:  # noqa: BLE001 -- discovery must not crash
                        logger.warning(
                            f"Skipping KITTI depth pair (drive={drive_name}, "
                            f"camera={camera}, frame={depth_path.name}): {exc}"
                        )
                        continue

                    if max_files is not None and len(pairs) >= max_files:
                        logger.info(
                            f"Discovered {len(pairs)} KITTI depth RGB+depth pairs "
                            f"(capped at max_files={max_files})"
                        )
                        return pairs

    logger.info(f"Discovered {len(pairs)} KITTI depth RGB+depth pairs")
    return pairs


# ---------------------------------------------------------------------
# Derived (non-calibrated) pinhole intrinsics -- D-006
# ---------------------------------------------------------------------


def derive_pinhole_intrinsics(
    width: int,
    height: int,
    horizontal_fov_deg: float = KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
) -> Tuple[float, float, float, float]:
    """Derive a pinhole intrinsics tuple from image size + an assumed horizontal FOV.

    **This is not a calibration** (see module docstring, D-006): real KITTI intrinsics
    are unavailable to this pipeline, so ``K`` is synthesized from a single assumed
    horizontal field-of-view constant, applied identically to every sample:

    - ``fx = width / (2 * tan(horizontal_fov_deg / 2))``
    - ``fy = fx`` (square-pixel assumption)
    - ``cx = width / 2``
    - ``cy = height / 2``

    :param width: Image width in pixels.
    :type width: int
    :param height: Image height in pixels.
    :type height: int
    :param horizontal_fov_deg: Assumed horizontal field-of-view, in degrees. Defaults to
        :data:`KITTI_DEFAULT_HORIZONTAL_FOV_DEG` (70 degrees).
    :type horizontal_fov_deg: float
    :return: ``(fx, fy, cx, cy)`` -- the same 4-tuple shape
        ``camera_models.pinhole_ray_map``/``pinhole_unproject_points`` accept.
    :rtype: Tuple[float, float, float, float]
    """
    # DECISION plan-2026-09-11T050223-1b47bcf6/D-006
    # Do NOT attempt to read a real calibration file here (e.g. KITTI `object/`'s
    # `calib/*.txt`) -- that subset's `P0..P3` projection matrices use a different
    # rectification convention than this `depth/` subset's raw sequences and would
    # silently poison intrinsics if conflated (findings/dataset-and-training-pipeline.md
    # #2, #7 [HARD]: no `calib_cam_to_cam.txt` exists anywhere under `raw_image_values/`
    # on this machine). This FOV-derived `K` is a documented approximation, not a
    # calibration; see decisions.md D-006.
    fov_rad = math.radians(horizontal_fov_deg)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------
# Single-pair loading
# ---------------------------------------------------------------------


def load_kitti_depth_pair(
    rgb_path: str,
    depth_path: str,
    patch_size: int,
    horizontal_fov_deg: float = KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]]:
    """Load one KITTI RGB+depth pair, resize to ``patch_size``, derive scaled intrinsics.

    Reads the RGB frame's and depth GT's ACTUAL width/height from the files themselves
    (KITTI raw-sequence resolution is not perfectly fixed across dates), derives a
    pinhole ``K`` at that native resolution via :func:`derive_pinhole_intrinsics`, then
    resizes both images to ``(patch_size, patch_size)`` and rescales ``K`` to match --
    keeping ``K`` geometrically consistent with the pixel grid actually returned (RGB via
    bilinear resampling, depth via nearest-neighbour so invalid ``0`` pixels are never
    blended into a fabricated positive depth).

    :param rgb_path: Path to the RGB PNG frame.
    :type rgb_path: str
    :param depth_path: Path to the KITTI 16-bit depth-GT PNG
        (``depth_meters = uint16_value / 256.0``, ``0`` = invalid).
    :type depth_path: str
    :param patch_size: Output spatial size (both height and width).
    :type patch_size: int
    :param horizontal_fov_deg: Assumed horizontal FOV forwarded to
        :func:`derive_pinhole_intrinsics`.
    :type horizontal_fov_deg: float
    :return: ``(rgb, depth, valid_mask, K)`` where ``rgb`` is
        ``(patch_size, patch_size, 3)`` float32 in ``[-1, 1]``, ``depth`` is
        ``(patch_size, patch_size, 1)`` float32 metric depth, ``valid_mask`` is
        ``(patch_size, patch_size, 1)`` float32 (``1.0`` = real, finite, nonzero depth;
        ``0.0`` = invalid -- never manufactured from a zero-depth pixel), and ``K`` is
        the ``(fx, fy, cx, cy)`` tuple rescaled to the ``patch_size`` grid. Returns
        ``None`` if the pair cannot be loaded (missing/corrupt file) -- the caller is
        expected to skip and log, not crash.
    :rtype: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]]
    """
    from PIL import Image

    try:
        rgb_img = Image.open(rgb_path).convert("RGB")
        depth_img = Image.open(depth_path)

        width, height = rgb_img.size
        depth_raw = np.array(depth_img, dtype=np.float32)

        fx, fy, cx, cy = derive_pinhole_intrinsics(
            width=width, height=height, horizontal_fov_deg=horizontal_fov_deg
        )

        # Resize RGB (bilinear -- natural photometric interpolation) and rescale K.
        rgb_resized = rgb_img.resize((patch_size, patch_size), Image.BILINEAR)
        rgb = np.asarray(rgb_resized, dtype=np.float32) / 127.5 - 1.0

        # Resize depth with NEAREST -- bilinear would blend invalid (0) pixels into
        # fabricated small positive depths at every edge (Problem Statement edge case:
        # "GT ray+distance conversion must propagate an explicit invalid mask, not
        # manufacture a ray from depth=0").
        depth_pil = Image.fromarray(depth_raw)
        depth_resized = depth_pil.resize((patch_size, patch_size), Image.NEAREST)
        depth_meters = np.asarray(depth_resized, dtype=np.float32) / KITTI_DEPTH_PNG_SCALE

        scale_x = patch_size / float(width)
        scale_y = patch_size / float(height)
        fx *= scale_x
        cx *= scale_x
        fy *= scale_y
        cy *= scale_y

        valid_mask = np.isfinite(depth_meters) & (depth_meters > 0.0)

        return (
            rgb.astype(np.float32),
            depth_meters[..., np.newaxis].astype(np.float32),
            valid_mask[..., np.newaxis].astype(np.float32),
            (fx, fy, cx, cy),
        )
    except Exception as exc:  # noqa: BLE001 -- one bad sample must not crash an epoch
        logger.warning(f"Skipping KITTI pair (rgb={rgb_path}, depth={depth_path}): {exc}")
        return None


# ---------------------------------------------------------------------
# PyDataset
# ---------------------------------------------------------------------


class KittiDepthDataset(keras.utils.PyDataset):
    """Multiprocessing-capable dataset for KITTI depth-benchmark RGB+depth pairs.

    Mirrors :class:`train.common.megadepth.MegaDepthDataset`'s shape and conventions
    (a ``keras.utils.PyDataset`` subclass, ``workers=N`` multiprocessing, per-epoch
    shuffle when training). Produces per-batch ``(rgb, depth, valid_mask, K)`` where:

    - ``rgb``: ``(batch_size, patch_size, patch_size, 3)`` float32 in ``[-1, 1]``.
    - ``depth``: ``(batch_size, patch_size, patch_size, 1)`` float32 metric depth
      (meters); invalid pixels are ``0.0``.
    - ``valid_mask``: ``(batch_size, patch_size, patch_size, 1)`` float32, ``1.0`` where
      ``depth`` is a real finite positive measurement, ``0.0`` otherwise -- an explicit
      mask, never inferred by a downstream consumer from ``depth == 0`` alone.
    - ``K``: ``(batch_size, 4)`` float32, per-sample ``(fx, fy, cx, cy)`` derived from
      that sample's own on-disk resolution via :func:`derive_pinhole_intrinsics`
      (D-006 approximation -- see module docstring).

    :param pairs: List of ``(rgb_path, depth_path)`` tuples, e.g. from
        :func:`discover_kitti_depth_pairs`.
    :type pairs: List[Tuple[str, str]]
    :param batch_size: Number of samples per batch. Defaults to ``16``.
    :type batch_size: int
    :param patch_size: Output spatial size (height and width). Defaults to ``256``.
    :type patch_size: int
    :param horizontal_fov_deg: Assumed horizontal FOV forwarded to
        :func:`derive_pinhole_intrinsics`. Defaults to
        :data:`KITTI_DEFAULT_HORIZONTAL_FOV_DEG`.
    :type horizontal_fov_deg: float
    :param is_training: If ``True``, shuffles pair indices each epoch. Defaults to
        ``True``.
    :type is_training: bool
    :param workers: Number of multiprocessing workers. Defaults to ``8`` (megadepth
        precedent).
    :type workers: int
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
        patch_size: int = 256,
        horizontal_fov_deg: float = KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
        is_training: bool = True,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=True, **kwargs)
        if not pairs:
            raise ValueError("KittiDepthDataset: pairs is empty.")
        self.pairs = list(pairs)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.is_training = bool(is_training)
        self.n_pairs = len(self.pairs)
        self.indices = np.arange(self.n_pairs)
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, self.n_pairs // self.batch_size)

    def __getitem__(self, idx: int):
        batch_rgb, batch_depth, batch_mask, batch_k = [], [], [], []

        attempts = 0
        while len(batch_rgb) < self.batch_size and attempts < self.batch_size * 3:
            sample_idx = (idx * self.batch_size + len(batch_rgb) + attempts) % self.n_pairs
            i = self.indices[sample_idx]
            attempts += 1

            rgb_path, depth_path = self.pairs[i]
            result = load_kitti_depth_pair(
                rgb_path, depth_path, self.patch_size, self.horizontal_fov_deg
            )
            if result is None:
                continue
            rgb, depth, valid_mask, k = result
            batch_rgb.append(rgb)
            batch_depth.append(depth)
            batch_mask.append(valid_mask)
            batch_k.append(k)

        if not batch_rgb:
            # Every candidate in this batch slot failed to load -- pad with a single
            # all-invalid zero sample rather than raising mid-epoch (Failure Modes
            # table: skip and log, never crash).
            ps = self.patch_size
            batch_rgb = [np.zeros((ps, ps, 3), dtype=np.float32)]
            batch_depth = [np.zeros((ps, ps, 1), dtype=np.float32)]
            batch_mask = [np.zeros((ps, ps, 1), dtype=np.float32)]
            batch_k = [derive_pinhole_intrinsics(ps, ps, self.horizontal_fov_deg)]

        return (
            np.stack(batch_rgb),
            np.stack(batch_depth),
            np.stack(batch_mask),
            np.asarray(batch_k, dtype=np.float32),
        )

    def on_epoch_end(self) -> None:
        if self.is_training:
            np.random.shuffle(self.indices)
