"""Unified OmniPoint training data pipeline: KITTI depth + MegaDepth, one GT contract.

Both locally-available pinhole sources (KITTI depth-benchmark, MegaDepth) are adapted to
emit the SAME per-batch shape:

    x = rgb                                    -- (B, patch_size, patch_size, 3) float32, [-1, 1]
    y = (gt_ray, gt_distance, gt_point, gt_mask, valid_mask)

where every element of ``y`` matches
:class:`dl_techniques.losses.omnipoint_losses.OmniPointCombinedLoss`'s documented ``y_true``
5-tuple exactly, so a batch from :class:`CombinedOmniPointDataset` is directly usable with
``model.compile(loss=OmniPointCombinedLoss(...))`` once Step 9 (`train_omnipoint.py`) selects
the matching 4 slots (``pred_ray, pred_distance, pred_mask_logit, pred_scale``) out of
:class:`~dl_techniques.models.vision.omnipoint.model.OmniPoint`'s own 5-tuple ``call()``
output (``ray, distance, point, mask_logit, scale`` -- note the model's own ``point`` has no
matching consumer here; ``y_true``'s ``gt_point`` is compared against a *derived*
``pred_scale * pred_ray * pred_distance``, not the model's ``point`` output directly -- see
`decisions.md` D-016).

Both source loaders reuse :func:`train.omnipoint.kitti_depth.derive_pinhole_intrinsics` for
the SAME FOV-derived, non-calibrated pinhole ``K`` approximation established by D-006 (this
module does not re-derive that formula -- see `kitti_depth.py`'s module docstring for the
full caveat), and both reuse
:func:`dl_techniques.utils.camera_models.pinhole_ray_map` (Step 1's own reference
implementation) to generate the per-pixel GT ray map -- never a second, independently
re-derived ray formula (see D-016).

.. warning::
    **MegaDepth's stored depth is NOT calibrated metric depth** (D-019): it is MVS/SfM
    reconstruction depth at that reconstruction's own arbitrary, per-scene scale, not meters.
    This module treats it as if it were metric depth anyway -- the same spirit of documented
    approximation as D-006's derived ``K``, not a claim that MegaDepth-derived GT distances are
    physically accurate in meters. `OmniPointCombinedLoss`'s own per-sample scale alignment
    (`compute_optimal_scale`) partially absorbs a per-sample scale mismatch, but does not
    recover true metric units.

.. warning::
    **The mask-loss ground truth is a validity proxy, not a semantic sky mask** (D-017):
    neither KITTI nor MegaDepth carries any sky/semantic-validity label. `gt_mask` is set to
    the same `valid_mask` (real-vs-invalid depth) both loaders already compute -- the mask
    head is trained to predict "was this pixel measurable" as a stand-in for the paper's own
    semantic sky mask, since no semantic label exists locally.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import keras
import numpy as np

from dl_techniques.utils.camera_models import pinhole_ray_map
from dl_techniques.utils.logger import logger

from train.common.megadepth import discover_megadepth_pairs
from train.omnipoint.kitti_depth import (
    KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
    KittiDepthDataset,
    derive_pinhole_intrinsics,
    discover_kitti_depth_pairs,
)

# ---------------------------------------------------------------------

# Floor on the pinhole ray's z-component before it is used as a divisor when converting
# planar (z-axis) depth to radial (Euclidean) distance (see D-016) -- guards the same
# near-grazing-incidence degenerate case `camera_models._l2_normalize_rays` guards for
# ray normalization itself, applied here to the resulting scalar division instead.
_RAY_Z_EPSILON = 1e-6

# MegaDepth's own default FOV assumption reuses D-006's constant unchanged (see module
# docstring's first warning and `kitti_depth.py`'s own docstring) -- there is exactly ONE
# FOV-derived-K formula in this codebase, `derive_pinhole_intrinsics`, and exactly one
# default constant for it.
MEGADEPTH_DEFAULT_HORIZONTAL_FOV_DEG = KITTI_DEFAULT_HORIZONTAL_FOV_DEG


# ---------------------------------------------------------------------
# MegaDepth single-pair loading, adapted to OmniPoint's (rgb, depth, valid_mask, K) contract
# ---------------------------------------------------------------------


def load_megadepth_pair_for_omnipoint(
    rgb_path: str,
    depth_path: str,
    patch_size: int,
    horizontal_fov_deg: float = MEGADEPTH_DEFAULT_HORIZONTAL_FOV_DEG,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]]:
    """Load one MegaDepth RGB+HDF5-depth pair into the SAME contract as `load_kitti_depth_pair`.

    Deliberately mirrors
    :func:`train.omnipoint.kitti_depth.load_kitti_depth_pair`'s output shape/dtype/ordering
    exactly, so both sources can be consumed by one downstream GT-derivation function
    (:func:`_derive_gt_targets_batch`) with no source-specific branching past this point.

    Unlike `train.common.megadepth.load_and_process_pair`/`load_and_resize_pair`, this
    function does **not** normalize depth to a per-sample ``[-1, 1]`` range -- OmniPoint's
    ray/distance/point GT derivation needs a depth value on a single, sample-independent
    scale (even if that scale is only MegaDepth's own arbitrary reconstruction scale, see
    module docstring D-019); a per-crop-normalized depth would make `K`-based ray/distance
    back-projection geometrically meaningless.

    :param rgb_path: Path to the RGB image file (JPEG/PNG).
    :type rgb_path: str
    :param depth_path: Path to the MegaDepth HDF5 depth file (key ``"depth"``).
    :type depth_path: str
    :param patch_size: Output spatial size (both height and width).
    :type patch_size: int
    :param horizontal_fov_deg: Assumed horizontal FOV forwarded to
        :func:`train.omnipoint.kitti_depth.derive_pinhole_intrinsics`.
    :type horizontal_fov_deg: float
    :return: ``(rgb, depth, valid_mask, K)``, identical shapes/dtypes to
        `load_kitti_depth_pair`'s return contract. ``None`` if the pair cannot be loaded.
    :rtype: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]]
    """
    from PIL import Image

    try:
        with h5py.File(depth_path, "r") as f:
            depth_raw = f["depth"][:].astype(np.float32)
        depth_h, depth_w = depth_raw.shape

        rgb_img = Image.open(rgb_path).convert("RGB")
        if rgb_img.size != (depth_w, depth_h):
            # MegaDepth RGB/depth pairs are not always pixel-identical resolution --
            # resize RGB onto the depth map's own grid first (matches
            # `train.common.megadepth`'s own precedent), so `K` (derived from the depth
            # map's native size below) stays geometrically consistent with both.
            rgb_img = rgb_img.resize((depth_w, depth_h), Image.BILINEAR)

        fx, fy, cx, cy = derive_pinhole_intrinsics(
            width=depth_w, height=depth_h, horizontal_fov_deg=horizontal_fov_deg
        )

        rgb_resized = rgb_img.resize((patch_size, patch_size), Image.BILINEAR)
        rgb = np.asarray(rgb_resized, dtype=np.float32) / 127.5 - 1.0

        # NEAREST for depth -- same reasoning as `load_kitti_depth_pair`: bilinear would
        # blend invalid (<=0) pixels into fabricated small positive depths at every edge.
        depth_pil = Image.fromarray(depth_raw)
        depth_resized = depth_pil.resize((patch_size, patch_size), Image.NEAREST)
        depth_values = np.asarray(depth_resized, dtype=np.float32)

        scale_x = patch_size / float(depth_w)
        scale_y = patch_size / float(depth_h)
        fx *= scale_x
        cx *= scale_x
        fy *= scale_y
        cy *= scale_y

        valid_mask = np.isfinite(depth_values) & (depth_values > 0.0)

        return (
            rgb.astype(np.float32),
            depth_values[..., np.newaxis].astype(np.float32),
            valid_mask[..., np.newaxis].astype(np.float32),
            (fx, fy, cx, cy),
        )
    except Exception as exc:  # noqa: BLE001 -- one bad sample must not crash an epoch
        logger.warning(
            f"Skipping MegaDepth pair (rgb={rgb_path}, depth={depth_path}): {exc}"
        )
        return None


class MegaDepthPinholeDataset(keras.utils.PyDataset):
    """Single-process-friendly MegaDepth loader in OmniPoint's ``(rgb, depth, valid_mask, K)`` shape.

    Mirrors :class:`train.omnipoint.kitti_depth.KittiDepthDataset`'s internal batching loop
    (skip-and-log on a bad sample, zero-sample padding if an entire batch slot fails) so both
    can be driven identically by :class:`CombinedOmniPointDataset`.

    :param rgb_paths: List of RGB image file paths, e.g. from
        :func:`train.common.megadepth.discover_megadepth_pairs`.
    :type rgb_paths: List[str]
    :param depth_paths: List of HDF5 depth file paths, matched by index to `rgb_paths`.
    :type depth_paths: List[str]
    :param batch_size: Number of samples per batch. Defaults to ``16``.
    :type batch_size: int
    :param patch_size: Output spatial size (height and width). Defaults to ``256``.
    :type patch_size: int
    :param horizontal_fov_deg: Assumed horizontal FOV forwarded to
        :func:`load_megadepth_pair_for_omnipoint`. Defaults to
        :data:`MEGADEPTH_DEFAULT_HORIZONTAL_FOV_DEG`.
    :type horizontal_fov_deg: float
    :param is_training: If ``True``, shuffles pair indices each epoch. Defaults to ``True``.
    :type is_training: bool
    :param workers: Number of multiprocessing workers. Defaults to ``1`` (this class is meant
        to be driven synchronously as an internal helper of :class:`CombinedOmniPointDataset`,
        which owns its own worker parallelism).
    :type workers: int
    """

    def __init__(
        self,
        rgb_paths: List[str],
        depth_paths: List[str],
        batch_size: int = 16,
        patch_size: int = 256,
        horizontal_fov_deg: float = MEGADEPTH_DEFAULT_HORIZONTAL_FOV_DEG,
        is_training: bool = True,
        workers: int = 1,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=workers > 1, **kwargs)
        if not rgb_paths or not depth_paths:
            raise ValueError("MegaDepthPinholeDataset: rgb_paths/depth_paths is empty.")
        if len(rgb_paths) != len(depth_paths):
            raise ValueError(
                f"MegaDepthPinholeDataset: rgb_paths ({len(rgb_paths)}) and depth_paths "
                f"({len(depth_paths)}) must have matched lengths."
            )
        self.rgb_paths = list(rgb_paths)
        self.depth_paths = list(depth_paths)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.is_training = bool(is_training)
        self.n_pairs = len(self.rgb_paths)
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

            result = load_megadepth_pair_for_omnipoint(
                self.rgb_paths[i], self.depth_paths[i], self.patch_size, self.horizontal_fov_deg
            )
            if result is None:
                continue
            rgb, depth, valid_mask, k = result
            batch_rgb.append(rgb)
            batch_depth.append(depth)
            batch_mask.append(valid_mask)
            batch_k.append(k)

        if not batch_rgb:
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


# ---------------------------------------------------------------------
# Shared GT ray/distance/point/mask derivation -- the actual unification point
# ---------------------------------------------------------------------


# DECISION plan-2026-09-11T050223-1b47bcf6/D-016
# Do NOT derive `gt_distance` directly as the raw depth value, and do NOT re-derive a ray
# formula locally -- both are real, non-obvious traps this comment guards against:
#   1. `pinhole_ray_map` (Step 1) is the ONE reference ray-generation implementation; a
#      second, independently written formula here would be exactly the "possibly-divergent
#      reimplementation" this step's own test suite is required to rule out.
#   2. Both KITTI's and MegaDepth's stored depth is PLANAR (z-axis / optical-axis) depth,
#      not the RADIAL (Euclidean, ||P||) distance `OmniPoint`'s `distance` head predicts
#      (`P = distance * ray`, `ray` unit-norm). Feeding planar depth into `PointDistanceLoss`
#      as if it were radial distance silently biases every off-center pixel (the bias grows
#      with the pixel's distance from the principal point, i.e. worst at image corners) --
#      no shape/type symptom would reveal this. The correct conversion is
#      `radial = planar / ray_z` (`ray_z` = the unit ray's z-component, since the
#      *unnormalized* pinhole ray is `(x, y, 1)` with `z=1` exactly, so radial distance is
#      planar depth times the unnormalized ray's own norm, which equals `1 / ray_z` after
#      normalization). See decisions.md D-016 for the derivation and the y_true/y_pred
#      contract this feeds.
def _derive_gt_targets_batch(
    depth_batch: np.ndarray,
    valid_mask_batch: np.ndarray,
    k_batch: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive ``(gt_ray, gt_distance, gt_point, gt_mask)`` from a batch of planar-depth samples.

    Shared by both KITTI- and MegaDepth-sourced batches (the actual point of unifying the two
    loaders into one pipeline) -- reuses
    :func:`dl_techniques.utils.camera_models.pinhole_ray_map` (Step 1's own reference ray
    generator) per sample, since each sample's ``K`` differs (D-016: never re-derive the ray
    formula locally).

    :param depth_batch: Planar (z-axis) depth, shape ``(B, patch_size, patch_size, 1)``,
        ``0.0`` at invalid pixels (never a manufactured value).
    :type depth_batch: np.ndarray
    :param valid_mask_batch: Validity mask, shape ``(B, patch_size, patch_size, 1)``, ``1.0``
        valid / ``0.0`` invalid.
    :type valid_mask_batch: np.ndarray
    :param k_batch: Per-sample pinhole intrinsics, shape ``(B, 4)``, ``(fx, fy, cx, cy)``.
    :type k_batch: np.ndarray
    :param patch_size: Spatial size of every input map (height == width).
    :type patch_size: int
    :return: ``(gt_ray, gt_distance, gt_point, gt_mask)`` where ``gt_ray`` is
        ``(B, patch_size, patch_size, 3)`` unit vectors, ``gt_distance`` is
        ``(B, patch_size, patch_size, 1)`` RADIAL distance (``0.0`` at invalid pixels),
        ``gt_point`` is ``(B, patch_size, patch_size, 3)`` (``gt_distance * gt_ray``), and
        ``gt_mask`` is `valid_mask_batch` itself, unchanged (D-017: the validity mask doubles
        as the mask-loss ground truth -- see this module's docstring).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    batch_size = depth_batch.shape[0]
    gt_ray = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)
    gt_distance = np.zeros((batch_size, patch_size, patch_size, 1), dtype=np.float32)
    gt_point = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)

    for i in range(batch_size):
        fx, fy, cx, cy = (float(v) for v in k_batch[i])
        ray = pinhole_ray_map(fx, fy, cx, cy, patch_size, patch_size)
        ray_np = np.asarray(keras.ops.convert_to_numpy(ray), dtype=np.float32)

        ray_z = ray_np[..., 2:3]
        planar_depth = depth_batch[i]
        mask = valid_mask_batch[i]

        radial_distance = planar_depth / np.maximum(ray_z, _RAY_Z_EPSILON)
        radial_distance = radial_distance * mask  # never manufacture a distance at an
        # invalid (zero-depth) pixel -- mirrors the Problem Statement edge case already
        # honored by both loaders' own `valid_mask` construction.

        gt_ray[i] = ray_np
        gt_distance[i] = radial_distance
        gt_point[i] = radial_distance * ray_np

    # DECISION plan-2026-09-11T050223-1b47bcf6/D-017
    # Do NOT invent a separate mask-derivation heuristic here -- neither dataset carries a
    # semantic sky/validity label, so the mask-loss ground truth is deliberately the SAME
    # real-vs-invalid-depth `valid_mask` both loaders already compute. See decisions.md
    # D-017 for the trade-off this documents.
    gt_mask = valid_mask_batch.astype(np.float32)

    return gt_ray, gt_distance, gt_point, gt_mask


# ---------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------


class CombinedOmniPointDataset(keras.utils.PyDataset):
    """Round-robin-mixed KITTI + MegaDepth dataset, emitting `OmniPointCombinedLoss`-ready batches.

    Owns one :class:`~train.omnipoint.kitti_depth.KittiDepthDataset` and one
    :class:`MegaDepthPinholeDataset` internally (each constructed with ``workers=1`` -- this
    class is the one that owns worker parallelism, via its own `workers`/`use_multiprocessing`
    constructor arguments), and alternates between them per :meth:`__getitem__` call.

    DECISION plan-2026-09-11T050223-1b47bcf6/D-018
    Mixing strategy: **round-robin by even/odd batch index** (KITTI on even indices, MegaDepth
    on odd), a fixed 50/50 ratio -- not a configurable mixing ratio, and not a
    concatenate-and-shuffle-indices scheme. This is the simplest strategy that provably
    guarantees BOTH sources appear in every 2-batch window regardless of their relative sizes
    (92,554 KITTI pairs vs. a much smaller local MegaDepth mirror) -- a size-proportional
    concatenation would let the smaller source vanish for long within-epoch stretches, and a
    global shuffle-of-concatenated-indices does not guarantee balance batch-to-batch, only
    dataset-to-dataset over a full epoch. See decisions.md D-018 for the trade-off; a future
    caller wanting weighted mixing has a concrete, single-line seam to change
    (`__getitem__`'s `idx % 2` dispatch) rather than a design that would need reworking.

    Length is the SUM of both sources' own `__len__` (each source's own batches are all
    reachable across one epoch, modulo the two sources trading off which one repeats sooner
    when their sizes differ -- both already wrap around their own indices independently, same
    as `KittiDepthDataset`/`MegaDepthDataset` do standalone).

    :param kitti_pairs: List of ``(rgb_path, depth_path)`` KITTI tuples, e.g. from
        :func:`train.omnipoint.kitti_depth.discover_kitti_depth_pairs`. May be empty if
        `megadepth_rgb_paths` is non-empty (single-source use).
    :type kitti_pairs: List[Tuple[str, str]]
    :param megadepth_rgb_paths: List of MegaDepth RGB paths, e.g. from
        :func:`train.common.megadepth.discover_megadepth_pairs`. May be empty if `kitti_pairs`
        is non-empty.
    :type megadepth_rgb_paths: List[str]
    :param megadepth_depth_paths: List of MegaDepth HDF5 depth paths, matched by index to
        `megadepth_rgb_paths`.
    :type megadepth_depth_paths: List[str]
    :param batch_size: Number of samples per batch, shared by both sources. Defaults to ``16``.
    :type batch_size: int
    :param patch_size: Output spatial size (height and width), shared by both sources.
        Defaults to ``256``.
    :type patch_size: int
    :param kitti_horizontal_fov_deg: Forwarded to the internal `KittiDepthDataset`.
    :type kitti_horizontal_fov_deg: float
    :param megadepth_horizontal_fov_deg: Forwarded to the internal `MegaDepthPinholeDataset`.
    :type megadepth_horizontal_fov_deg: float
    :param is_training: If ``True``, both internal sources shuffle indices each epoch.
        Defaults to ``True``.
    :type is_training: bool
    :param workers: Number of multiprocessing workers for THIS combined dataset. Defaults to
        ``8`` (megadepth precedent).
    :type workers: int
    """

    def __init__(
        self,
        kitti_pairs: List[Tuple[str, str]],
        megadepth_rgb_paths: List[str],
        megadepth_depth_paths: List[str],
        batch_size: int = 16,
        patch_size: int = 256,
        kitti_horizontal_fov_deg: float = KITTI_DEFAULT_HORIZONTAL_FOV_DEG,
        megadepth_horizontal_fov_deg: float = MEGADEPTH_DEFAULT_HORIZONTAL_FOV_DEG,
        is_training: bool = True,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=True, **kwargs)

        has_kitti = bool(kitti_pairs)
        has_megadepth = bool(megadepth_rgb_paths) and bool(megadepth_depth_paths)
        if not has_kitti and not has_megadepth:
            raise ValueError(
                "CombinedOmniPointDataset: at least one of kitti_pairs or "
                "(megadepth_rgb_paths, megadepth_depth_paths) must be non-empty."
            )

        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.is_training = bool(is_training)

        self._kitti: Optional[KittiDepthDataset] = (
            KittiDepthDataset(
                kitti_pairs,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                horizontal_fov_deg=kitti_horizontal_fov_deg,
                is_training=self.is_training,
                workers=1,
            )
            if has_kitti
            else None
        )
        self._megadepth: Optional[MegaDepthPinholeDataset] = (
            MegaDepthPinholeDataset(
                megadepth_rgb_paths,
                megadepth_depth_paths,
                batch_size=self.batch_size,
                patch_size=self.patch_size,
                horizontal_fov_deg=megadepth_horizontal_fov_deg,
                is_training=self.is_training,
                workers=1,
            )
            if has_megadepth
            else None
        )

        logger.info(
            f"CombinedOmniPointDataset: kitti_pairs={len(kitti_pairs)}, "
            f"megadepth_pairs={len(megadepth_rgb_paths)}, batch_size={self.batch_size}, "
            f"patch_size={self.patch_size}"
        )

    def __len__(self) -> int:
        kitti_len = len(self._kitti) if self._kitti is not None else 0
        megadepth_len = len(self._megadepth) if self._megadepth is not None else 0
        return max(1, kitti_len + megadepth_len)

    def __getitem__(self, idx: int):
        if self._kitti is not None and self._megadepth is not None:
            # D-018: fixed round-robin, even -> KITTI, odd -> MegaDepth.
            if idx % 2 == 0:
                source_ds, source_idx = self._kitti, (idx // 2) % len(self._kitti)
            else:
                source_ds, source_idx = self._megadepth, (idx // 2) % len(self._megadepth)
        elif self._kitti is not None:
            source_ds, source_idx = self._kitti, idx % len(self._kitti)
        else:
            source_ds, source_idx = self._megadepth, idx % len(self._megadepth)

        rgb, depth, valid_mask, k = source_ds[source_idx]
        gt_ray, gt_distance, gt_point, gt_mask = _derive_gt_targets_batch(
            depth, valid_mask, k, self.patch_size
        )
        return rgb, (gt_ray, gt_distance, gt_point, gt_mask, valid_mask)

    def on_epoch_end(self) -> None:
        if self._kitti is not None:
            self._kitti.on_epoch_end()
        if self._megadepth is not None:
            self._megadepth.on_epoch_end()


# ---------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------


def create_combined_omnipoint_dataset(
    kitti_root: Optional[str],
    megadepth_root: Optional[str],
    batch_size: int = 16,
    patch_size: int = 256,
    max_kitti_files: Optional[int] = None,
    max_megadepth_files: Optional[int] = None,
    is_training: bool = True,
    workers: int = 8,
) -> CombinedOmniPointDataset:
    """Discover pairs under both dataset roots and build one `CombinedOmniPointDataset`.

    Either `kitti_root` or `megadepth_root` may be ``None`` (single-source use); at least one
    must be provided and yield at least one pair.

    :param kitti_root: Path to the KITTI depth-benchmark root, or ``None`` to skip KITTI.
    :type kitti_root: Optional[str]
    :param megadepth_root: Path to the MegaDepth dataset root, or ``None`` to skip MegaDepth.
    :type megadepth_root: Optional[str]
    :param batch_size: Number of samples per batch. Defaults to ``16``.
    :type batch_size: int
    :param patch_size: Output spatial size (height and width). Defaults to ``256``.
    :type patch_size: int
    :param max_kitti_files: Maximum number of KITTI pairs to discover. ``None`` for unlimited.
    :type max_kitti_files: Optional[int]
    :param max_megadepth_files: Maximum number of MegaDepth pairs to discover. ``None`` for
        unlimited.
    :type max_megadepth_files: Optional[int]
    :param is_training: If ``True``, both sources shuffle each epoch. Defaults to ``True``.
    :type is_training: bool
    :param workers: Number of multiprocessing workers for the combined dataset.
    :type workers: int
    :return: A constructed `CombinedOmniPointDataset`.
    :rtype: CombinedOmniPointDataset
    :raises ValueError: If both roots are ``None``, or if discovery finds zero pairs overall.
    """
    if kitti_root is None and megadepth_root is None:
        raise ValueError(
            "create_combined_omnipoint_dataset: at least one of kitti_root/megadepth_root "
            "must be provided."
        )

    kitti_pairs: List[Tuple[str, str]] = []
    if kitti_root is not None:
        kitti_pairs = discover_kitti_depth_pairs(kitti_root, max_files=max_kitti_files)

    megadepth_rgb_paths: List[str] = []
    megadepth_depth_paths: List[str] = []
    if megadepth_root is not None:
        megadepth_rgb_paths, megadepth_depth_paths = discover_megadepth_pairs(
            megadepth_root, max_files=max_megadepth_files
        )

    if not kitti_pairs and not megadepth_rgb_paths:
        raise ValueError(
            "create_combined_omnipoint_dataset: discovery found zero pairs from both "
            f"kitti_root={kitti_root!r} and megadepth_root={megadepth_root!r}."
        )

    return CombinedOmniPointDataset(
        kitti_pairs,
        megadepth_rgb_paths,
        megadepth_depth_paths,
        batch_size=batch_size,
        patch_size=patch_size,
        is_training=is_training,
        workers=workers,
    )
