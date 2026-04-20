"""MegaDepth dataset pipeline for monocular depth estimation.

Provides discovery, loading, and batching of MegaDepth RGB+depth pairs
for training depth estimation models.

MegaDepth directory structure::

    {root}/{scene}/dense{0,1}/imgs/*.jpg
    {root}/{scene}/dense{0,1}/depths/*.h5

Depth maps are HDF5 files with key ``"depth"`` containing float32
metric depth.  Invalid pixels have depth == 0.

The pipeline produces ``(rgb, y_true)`` pairs where:

- ``rgb``: float32 ``(patch_size, patch_size, 3)`` in ``[-1, +1]``
- ``y_true``: float32 ``(patch_size, patch_size, 2)`` —
  channel 0 = per-sample normalized depth ``[-1, +1]``,
  channel 1 = validity mask (1 = valid, 0 = invalid)

Usage::

    from train.common.megadepth import discover_megadepth_pairs, MegaDepthDataset

    rgb_paths, depth_paths = discover_megadepth_pairs("/path/to/Megadepth")
    ds = MegaDepthDataset(
        rgb_paths, depth_paths,
        batch_size=16, patch_size=256,
        is_training=True, workers=8,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import keras
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Discovery
# =====================================================================


def discover_megadepth_pairs(
    root: str,
    max_files: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Scan MegaDepth scenes for matched RGB+depth file pairs.

    Walks all ``{root}/{scene}/dense{N}/`` subdirectories, matching
    image files in ``imgs/`` to HDF5 depth files in ``depths/`` by
    filename stem.

    Args:
        root: Path to the MegaDepth dataset root directory.
        max_files: Maximum number of pairs to return.  ``None`` for
            unlimited.

    Returns:
        Tuple of ``(rgb_paths, depth_paths)`` — matched lists of
        absolute file paths.
    """
    root_path = Path(root)
    rgb_paths: List[str] = []
    depth_paths: List[str] = []

    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        for dense_dir in sorted(scene_dir.iterdir()):
            if not dense_dir.is_dir() or not dense_dir.name.startswith("dense"):
                continue

            imgs_dir = dense_dir / "imgs"
            depths_dir = dense_dir / "depths"
            if not imgs_dir.exists() or not depths_dir.exists():
                continue

            # Build stem → path maps
            img_stems = {}
            for fp in imgs_dir.iterdir():
                if fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img_stems[fp.stem] = str(fp)

            for fp in depths_dir.iterdir():
                if fp.suffix.lower() == ".h5":
                    stem = fp.stem
                    if stem in img_stems:
                        rgb_paths.append(img_stems[stem])
                        depth_paths.append(str(fp))

            if max_files and len(rgb_paths) >= max_files:
                rgb_paths = rgb_paths[:max_files]
                depth_paths = depth_paths[:max_files]
                return rgb_paths, depth_paths

    logger.info(f"Discovered {len(rgb_paths)} MegaDepth RGB+depth pairs")
    return rgb_paths, depth_paths


# =====================================================================
# Single-pair loading
# =====================================================================


def load_and_process_pair(
    rgb_path: str,
    depth_path: str,
    patch_size: int,
    min_valid_ratio: float = 0.1,
    augment: bool = False,
    max_crop_attempts: int = 10,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one RGB+depth pair, random crop, normalize, and augment.

    Loads an RGB image (JPEG/PNG) and its paired HDF5 depth map,
    resizes if needed to match spatial dimensions, extracts a random
    crop, normalizes depth per-sample to ``[-1, +1]``, and optionally
    applies geometric augmentations (horizontal/vertical flip + 90-degree
    rotations).

    Args:
        rgb_path: Path to the RGB image file.
        depth_path: Path to the HDF5 depth file (key ``"depth"``).
        patch_size: Spatial size of the output crop.
        min_valid_ratio: Minimum fraction of valid (non-zero) depth
            pixels required in a crop.  Crops below this threshold are
            rejected.  Defaults to ``0.1``.
        augment: Whether to apply random geometric augmentations.
        max_crop_attempts: Number of random crops to try before giving
            up.  Defaults to ``10``.

    Returns:
        Tuple of ``(rgb, y_true)`` where ``rgb`` has shape
        ``(patch_size, patch_size, 3)`` in ``[-1, 1]`` and ``y_true``
        has shape ``(patch_size, patch_size, 2)`` with
        ``[depth, mask]``.  Returns ``None`` if no valid crop was found.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load RGB
    rgb = plt.imread(rgb_path)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
    else:
        rgb = rgb.astype(np.float32) * 2.0 - 1.0

    # Load depth from HDF5
    with h5py.File(depth_path, "r") as f:
        depth = f["depth"][:].astype(np.float32)

    h, w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]

    # Resize RGB to match depth if needed
    if rgb_h != h or rgb_w != w:
        rgb_pil = Image.fromarray(
            ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )
        rgb_pil = rgb_pil.resize((w, h), Image.BILINEAR)
        rgb = np.array(rgb_pil, dtype=np.float32) / 127.5 - 1.0

    # Ensure minimum size for patching
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w) + 0.01
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_pil = Image.fromarray(
            ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)
        rgb = np.array(rgb_pil, dtype=np.float32) / 127.5 - 1.0
        depth_pil = Image.fromarray(depth)
        depth_pil = depth_pil.resize((new_w, new_h), Image.NEAREST)
        depth = np.array(depth_pil, dtype=np.float32)
        h, w = new_h, new_w

    # Random crop — try multiple times for enough valid pixels
    valid_mask = None
    for _ in range(max_crop_attempts):
        y = np.random.randint(0, h - patch_size + 1)
        x = np.random.randint(0, w - patch_size + 1)
        depth_patch = depth[y:y + patch_size, x:x + patch_size]
        valid_mask = (depth_patch > 0).astype(np.float32)
        if valid_mask.mean() >= min_valid_ratio:
            break
    else:
        if valid_mask is None or valid_mask.mean() < min_valid_ratio:
            return None

    rgb_patch = rgb[y:y + patch_size, x:x + patch_size, :3]
    depth_patch = depth_patch[..., np.newaxis]  # (ps, ps, 1)
    valid_mask = valid_mask[..., np.newaxis]     # (ps, ps, 1)

    # Normalize depth per-sample to [-1, +1] using valid pixel range
    valid_depths = depth_patch[valid_mask > 0]
    if len(valid_depths) > 0:
        d_min = valid_depths.min()
        d_max = valid_depths.max()
        d_range = d_max - d_min
        if d_range > 1e-6:
            depth_patch = np.where(
                valid_mask > 0,
                (depth_patch - d_min) / d_range * 2.0 - 1.0,
                0.0,
            )
        else:
            depth_patch = np.zeros_like(depth_patch)
    else:
        depth_patch = np.zeros_like(depth_patch)

    # Augmentation (numpy — runs in worker process)
    if augment:
        combined = np.concatenate(
            [rgb_patch, depth_patch, valid_mask], axis=-1
        )
        if np.random.random() > 0.5:
            combined = combined[:, ::-1, :]
        if np.random.random() > 0.5:
            combined = combined[::-1, :, :]
        k = np.random.randint(0, 4)
        combined = np.rot90(combined, k, axes=(0, 1))
        rgb_patch = combined[..., :3].copy()
        depth_patch = combined[..., 3:4].copy()
        valid_mask = combined[..., 4:5].copy()

    # y_true = concat([depth, mask], axis=-1)
    y_true = np.concatenate([depth_patch, valid_mask], axis=-1)

    return (
        rgb_patch.astype(np.float32),
        y_true.astype(np.float32),
    )


# =====================================================================
# PyDataset
# =====================================================================


class MegaDepthDataset(keras.utils.PyDataset):
    """Multiprocessing-capable dataset for MegaDepth RGB+depth pairs.

    Each worker process independently loads, crops, augments, and
    normalizes samples — bypassing the GIL for true parallel I/O.

    Produces ``(rgb_batch, y_true_batch)`` where:

    - ``rgb_batch``: ``(batch_size, patch_size, patch_size, 3)``
    - ``y_true_batch``: ``(batch_size, patch_size, patch_size, 2)``
      with ``[depth, mask]`` concatenated on the last axis.

    Args:
        rgb_paths: List of RGB image file paths.
        depth_paths: List of HDF5 depth file paths (matched by index).
        batch_size: Number of samples per batch.
        patch_size: Spatial size of cropped patches.
        min_valid_ratio: Minimum fraction of valid depth pixels per
            crop.  Defaults to ``0.1``.
        augment: Whether to apply random augmentations (flips + rot90).
            Defaults to ``True``.
        is_training: If ``True``, shuffles indices each epoch.
        workers: Number of multiprocessing workers.
        **kwargs: Passed to :class:`keras.utils.PyDataset`.
    """

    def __init__(
        self,
        rgb_paths: List[str],
        depth_paths: List[str],
        batch_size: int = 16,
        patch_size: int = 256,
        min_valid_ratio: float = 0.1,
        augment: bool = True,
        is_training: bool = True,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=True, **kwargs)
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.min_valid_ratio = min_valid_ratio
        self.augment = augment
        self.is_training = is_training
        self.n_pairs = len(rgb_paths)
        self.indices = np.arange(self.n_pairs)
        if is_training:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, self.n_pairs // self.batch_size)

    def __getitem__(self, idx: int):
        batch_rgb, batch_ytrue = [], []

        attempts = 0
        while len(batch_rgb) < self.batch_size and attempts < self.batch_size * 3:
            sample_idx = (
                idx * self.batch_size + len(batch_rgb) + attempts
            ) % self.n_pairs
            i = self.indices[sample_idx]
            attempts += 1

            result = load_and_process_pair(
                self.rgb_paths[i],
                self.depth_paths[i],
                self.patch_size,
                self.min_valid_ratio,
                augment=self.is_training and self.augment,
            )
            if result is None:
                continue
            rgb, y_true = result
            batch_rgb.append(rgb)
            batch_ytrue.append(y_true)

        # Pad if we couldn't fill the batch (rare)
        if not batch_rgb:
            ps = self.patch_size
            batch_rgb = [np.zeros((ps, ps, 3), dtype=np.float32)]
            batch_ytrue = [np.zeros((ps, ps, 2), dtype=np.float32)]

        return np.stack(batch_rgb), np.stack(batch_ytrue)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)
