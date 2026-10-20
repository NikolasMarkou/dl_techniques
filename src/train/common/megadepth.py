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
  channel 0 = per-map normalized **disparity** in ``[0, 1]`` (see
  :func:`normalized_disparity`; 1 is the nearest valid pixel, 0 the farthest,
  and invalid pixels are 0),
  channel 1 = validity mask (1 = valid, 0 = invalid)

The target is the Depth Anything V1 relative-depth target (arXiv 2401.10891,
section 3.1: "the depth value is first transformed into the disparity space by
d = 1/t and then normalized to 0~1 on each depth map"), which a non-negative
(ReLU) depth head can represent.

Usage::

    from train.common.megadepth import split_megadepth_pairs, MegaDepthDataset

    train_rgb, train_depth, val_rgb, val_depth = split_megadepth_pairs(
        "/path/to/Megadepth", train_split=0.9)
    ds = MegaDepthDataset(
        train_rgb, train_depth,
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

from train.common.datasets import IMAGENET_MEAN, IMAGENET_STD, fill_batch

logger = logging.getLogger(__name__)


# =====================================================================
# Discovery
# =====================================================================


def scene_of(path: str) -> str:
    """The MegaDepth scene a pair belongs to: ``{root}/{scene}/dense{N}/{imgs,depths}/{file}``.

    Args:
        path: An RGB or depth path returned by the discovery functions.

    Returns:
        The scene directory name.
    """
    return Path(path).parts[-4]


def discover_megadepth_scenes(
    root: str,
) -> List[Tuple[str, List[str], List[str]]]:
    """Every scene's matched RGB+depth pairs.

    Interface contract: returns ``[(scene, rgb_paths, depth_paths), ...]``,
    scenes sorted by name and each scene's pairs sorted by depth file path,
    with only the scenes that hold at least one pair. An image matches the
    HDF5 depth file of the same stem in the same ``dense{N}`` directory. Never
    raises for a missing ``imgs``/``depths`` directory (the scene is skipped).

    Args:
        root: Path to the MegaDepth dataset root directory.

    Returns:
        One ``(scene, rgb_paths, depth_paths)`` entry per non-empty scene.
    """
    scenes = []
    for scene_dir in sorted(Path(root).iterdir()):
        if not scene_dir.is_dir():
            continue
        pairs = []
        for dense_dir in sorted(scene_dir.iterdir()):
            if not dense_dir.is_dir() or not dense_dir.name.startswith("dense"):
                continue
            imgs_dir = dense_dir / "imgs"
            depths_dir = dense_dir / "depths"
            if not imgs_dir.exists() or not depths_dir.exists():
                continue
            img_stems = {
                fp.stem: str(fp) for fp in imgs_dir.iterdir()
                if fp.suffix.lower() in (".jpg", ".jpeg", ".png")}
            for fp in sorted(depths_dir.iterdir()):
                if fp.suffix.lower() == ".h5" and fp.stem in img_stems:
                    pairs.append((img_stems[fp.stem], str(fp)))
        if pairs:
            scenes.append((scene_dir.name, [p[0] for p in pairs], [p[1] for p in pairs]))
    return scenes


def _take_across_scenes(
    scenes: List[Tuple[str, List[str], List[str]]],
    max_files: Optional[int],
) -> Tuple[List[str], List[str]]:
    """Up to ``max_files`` pairs, spread as evenly over the scenes as their sizes allow.

    Every scene gets the same quota, a scene smaller than the quota gives up
    the rest to the others (a water-fill), and a scene's share is taken evenly
    spaced through its sorted pairs. ``None`` takes every pair. The result
    stays grouped by scene, in scene order.
    """
    if not max_files:
        quota = {name: len(rgb) for name, rgb, _ in scenes}
    else:
        quota = {name: 0 for name, _, _ in scenes}
        sizes = {name: len(rgb) for name, rgb, _ in scenes}
        remaining = int(max_files)
        open_scenes = [name for name, _, _ in scenes]
        while remaining > 0 and open_scenes:
            share = max(1, remaining // len(open_scenes))
            for name in list(open_scenes):
                give = min(share, sizes[name] - quota[name], remaining)
                quota[name] += give
                remaining -= give
                if quota[name] == sizes[name]:
                    open_scenes.remove(name)
                if remaining == 0:
                    break
    rgb_out, depth_out = [], []
    for name, rgb, depth in scenes:
        k = quota[name]
        for index in (np.arange(k) * len(rgb)) // max(k, 1):
            rgb_out.append(rgb[index])
            depth_out.append(depth[index])
    return rgb_out, depth_out


def discover_megadepth_pairs(
    root: str,
    max_files: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Scan MegaDepth scenes for matched RGB+depth file pairs.

    Walks all ``{root}/{scene}/dense{N}/`` subdirectories, matching
    image files in ``imgs/`` to HDF5 depth files in ``depths/`` by
    filename stem (:func:`discover_megadepth_scenes`).

    Args:
        root: Path to the MegaDepth dataset root directory.
        max_files: Maximum number of pairs to return, spread evenly across
            the scenes (:func:`_take_across_scenes`).  ``None`` for
            unlimited.

    Returns:
        Tuple of ``(rgb_paths, depth_paths)`` — matched lists of
        absolute file paths, grouped by scene in scene order.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-129
    # Do NOT go back to the first max_files pairs in scene order: MegaDepth's
    # scenes hold 19 to 3970 pairs, so the trainer's default 11,000 covered 5
    # of 195 scenes.
    rgb_paths, depth_paths = _take_across_scenes(
        discover_megadepth_scenes(root), max_files)
    logger.info(f"Discovered {len(rgb_paths)} MegaDepth RGB+depth pairs")
    return rgb_paths, depth_paths


def split_megadepth_pairs(
    root: str,
    train_split: float = 0.9,
    max_train_files: Optional[int] = None,
    max_val_files: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """A scene-disjoint train/validation split of MegaDepth.

    Interface contract: returns ``(train_rgb, train_depth, val_rgb,
    val_depth)``. The scenes (sorted by name) are dealt so that
    ``1 - train_split`` of them, spread evenly over the sorted list, are
    validation scenes and the rest training scenes; no scene contributes to
    both. Each side is then capped at its ``max_*_files``, taken evenly
    across its scenes (``None`` keeps every pair). Deterministic: the same
    root and arguments give the same lists. Raises ``ValueError`` when
    ``train_split`` is not in ``(0, 1)`` or a side would get no scene.

    Args:
        root: Path to the MegaDepth dataset root directory.
        train_split: Fraction of the scenes used for training.
        max_train_files: Cap on the training pairs, or ``None``.
        max_val_files: Cap on the validation pairs, or ``None``.

    Returns:
        The four path lists.

    Raises:
        ValueError: If ``train_split`` is outside ``(0, 1)`` or either side
            has no scene.
    """
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    scenes = discover_megadepth_scenes(root)
    n_val = int(round(len(scenes) * (1.0 - train_split)))
    val_index = {(2 * j + 1) * len(scenes) // (2 * n_val) for j in range(n_val)}
    train_scenes = [s for i, s in enumerate(scenes) if i not in val_index]
    val_scenes = [s for i, s in enumerate(scenes) if i in val_index]
    if not train_scenes or not val_scenes:
        raise ValueError(
            f"train_split={train_split} over {len(scenes)} scenes leaves "
            f"{len(train_scenes)} training and {len(val_scenes)} validation scenes")
    train_rgb, train_depth = _take_across_scenes(train_scenes, max_train_files)
    val_rgb, val_depth = _take_across_scenes(val_scenes, max_val_files)
    logger.info(
        f"MegaDepth split: {len(train_rgb)} training pairs from "
        f"{len(train_scenes)} scenes, {len(val_rgb)} validation pairs from "
        f"{len(val_scenes)} scenes")
    return train_rgb, train_depth, val_rgb, val_depth


# =====================================================================
# Target
# =====================================================================


def normalized_disparity(
        depth: np.ndarray, valid_mask: np.ndarray,
) -> np.ndarray:
    """Per-map min-max normalized disparity ``1 / depth`` over the valid pixels.

    Interface contract: ``depth`` and ``valid_mask`` have the same shape;
    ``valid_mask`` is 1 where ``depth`` is a valid (positive) value. Returns a
    float32 array of that shape in ``[0, 1]``: the nearest valid pixel is 1, the
    farthest 0, and invalid pixels are 0. A map with no valid pixel, or whose
    valid disparities span less than ``1e-6`` of their maximum, returns zeros.
    Never raises.

    :param depth: Depth map (any positive scale; MegaDepth's is SfM depth).
    :param valid_mask: Validity mask, 1 = valid.
    :return: The normalized disparity, float32.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-108
    # Do NOT go back to min-max DEPTH scaled to [-1, 1]: DepthAnything's head
    # ends in a ReLU (D-031), so the negative half of that target could never be
    # predicted, and the reference trains on per-map normalized disparity.
    valid = valid_mask > 0
    disparity = np.zeros(depth.shape, dtype=np.float64)
    disparity[valid] = 1.0 / depth[valid].astype(np.float64)
    if not np.any(valid):
        return disparity.astype(np.float32)
    lo = disparity[valid].min()
    hi = disparity[valid].max()
    if hi - lo <= 1e-6 * hi:
        return np.zeros(depth.shape, dtype=np.float32)
    disparity[valid] = (disparity[valid] - lo) / (hi - lo)
    return disparity.astype(np.float32)


def standardize_imagenet(rgb: np.ndarray) -> np.ndarray:
    """This module's ``[-1, +1]`` RGB -> ImageNet-standardized RGB.

    Interface contract (callers: :class:`MegaDepthDataset` and
    :class:`UnlabeledImageDataset` with ``imagenet_standardize=True``, and the
    Depth Anything trainer's monitor samples): returns float32
    ``((rgb + 1) / 2 - IMAGENET_MEAN) / IMAGENET_STD`` per channel, the last
    axis being R, G, B; the input convention the official DINOv2 / Depth
    Anything encoders were trained with (the reference's ``NormalizeImage``).
    Pure; any shape whose last axis is 3.

    :param rgb: RGB in ``[-1, +1]``.
    :return: The standardized RGB, same shape.
    """
    mean = np.asarray(IMAGENET_MEAN, np.float32)
    std = np.asarray(IMAGENET_STD, np.float32)
    return (((np.asarray(rgb, np.float32) + 1.0) * 0.5 - mean) / std).astype(np.float32)


def _has_spread(disparity: np.ndarray) -> bool:
    """Whether a :func:`normalized_disparity` target can supervise anything.

    Interface contract (2 callers, the two pair loaders): ``disparity`` is the
    output of :func:`normalized_disparity`, which is all zeros exactly when
    the valid pixels have no spread or there are none. Such a target scores
    1.0 under the affine-invariant loss for any non-constant prediction, with
    zero gradient. Never raises.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-125
    # Do NOT accept a crop whose valid depth is one constant: part of
    # MegaDepth stores ordinal depth (2.0 or 5.0 on every valid pixel), whose
    # all-zero target supervises nothing, floors val_loss and makes the
    # Pearson correlation undefined. Skip it like a crop with too few valid pixels.
    return bool(disparity.max() > 0)


def _mirror_left_right(rgb, disparity, valid_mask, draw):
    """The labeled-image augmentation: a horizontal flip with probability 0.5.

    Interface contract (2 callers, the two pair loaders): ``rgb``,
    ``disparity`` and ``valid_mask`` are ``(H, W, C)`` arrays of one sample;
    ``draw()`` returns a float in ``[0, 1)`` and is called exactly once.
    Returns the three arrays, all mirrored left-right or all unchanged.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-124
    # Do NOT add up-down flips or 90-degree rotations back: they remove where
    # "down" is in the image, and from random weights Depth Anything then sat
    # at a nearly constant map (training loss 1.26, held-out Pearson 0.05 vs
    # 0.82 and 0.23 with this flip only). The paper (arXiv 2401.10891, section
    # 4.1): "We only apply horizontal flipping ... for labeled images."
    if draw() > 0.5:
        return (rgb[:, ::-1].copy(), disparity[:, ::-1].copy(),
                valid_mask[:, ::-1].copy())
    return rgb, disparity, valid_mask


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
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one RGB+depth pair, random crop, normalize, and augment.

    Loads an RGB image (JPEG/PNG) and its paired HDF5 depth map,
    resizes if needed to match spatial dimensions, extracts a random
    crop, converts depth to per-map normalized disparity in ``[0, 1]``
    (:func:`normalized_disparity`), and optionally mirrors it left-right
    (the reference's only labeled-image augmentation). A crop qualifies when
    at least ``min_valid_ratio`` of it is valid and its valid depth has
    spread (MegaDepth's ordinal maps, one constant value, never do).

    Args:
        rgb_path: Path to the RGB image file.
        depth_path: Path to the HDF5 depth file (key ``"depth"``).
        patch_size: Spatial size of the output crop.
        min_valid_ratio: Minimum fraction of valid (non-zero) depth
            pixels required in a crop.  Crops below this threshold, or whose
            valid depth is one constant, are rejected.  Defaults to ``0.1``.
        augment: Whether to mirror the sample left-right with probability 0.5.
        max_crop_attempts: Number of random crops to try before giving
            up.  Defaults to ``10``.
        rng: Generator for the crop position and the augmentation.
            ``None`` (the default) draws from the global ``np.random``
            state, as training does; a seeded generator gives the same crop
            on every call, which is how :class:`MegaDepthDataset` keeps its
            validation set fixed across epochs.

    Returns:
        Tuple of ``(rgb, y_true)`` where ``rgb`` has shape
        ``(patch_size, patch_size, 3)`` in ``[-1, 1]`` and ``y_true``
        has shape ``(patch_size, patch_size, 2)`` with
        ``[disparity, mask]``.  Returns ``None`` if no crop qualified.
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

    randint = np.random.randint if rng is None else rng.integers
    uniform = np.random.random if rng is None else rng.random

    # Random crop — try multiple times for enough valid pixels with spread
    for _ in range(max_crop_attempts):
        y = randint(0, h - patch_size + 1)
        x = randint(0, w - patch_size + 1)
        valid_mask = (depth[y:y + patch_size, x:x + patch_size] > 0).astype(np.float32)
        if valid_mask.mean() < min_valid_ratio:
            continue
        valid_mask = valid_mask[..., np.newaxis]  # (ps, ps, 1)
        depth_patch = normalized_disparity(
            depth[y:y + patch_size, x:x + patch_size, np.newaxis], valid_mask)
        if _has_spread(depth_patch):
            break
    else:
        return None

    rgb_patch = rgb[y:y + patch_size, x:x + patch_size, :3]

    # Augmentation (numpy — runs in worker process)
    if augment:
        rgb_patch, depth_patch, valid_mask = _mirror_left_right(
            rgb_patch, depth_patch, valid_mask, uniform)

    # y_true = concat([depth, mask], axis=-1)
    y_true = np.concatenate([depth_patch, valid_mask], axis=-1)

    return (
        rgb_patch.astype(np.float32),
        y_true.astype(np.float32),
    )


def load_and_resize_pair(
    rgb_path: str,
    depth_path: str,
    target_size: int,
    min_valid_ratio: float = 0.1,
    augment: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one RGB+depth pair, resize the full image, normalize, augment.

    Unlike :func:`load_and_process_pair` which extracts random crops,
    this resizes the **entire** image to ``(target_size, target_size)``.
    This gives the model global scene context at the cost of resolution.
    The target channel is :func:`normalized_disparity`, as there.

    Args:
        rgb_path: Path to the RGB image file.
        depth_path: Path to the HDF5 depth file (key ``"depth"``).
        target_size: Output spatial size (both height and width).
        min_valid_ratio: Minimum fraction of valid depth pixels.
        augment: Whether to mirror the sample left-right with probability 0.5.

    Returns:
        Tuple of ``(rgb, y_true)``, or ``None`` if below the valid ratio or
        if the valid depth is one constant (an ordinal map).
    """
    from PIL import Image

    # Load RGB
    import matplotlib.pyplot as plt
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

    # Build validity mask before resizing
    valid_mask = (depth > 0).astype(np.float32)

    # Check valid ratio on original image
    if valid_mask.mean() < min_valid_ratio:
        return None

    # Resize everything to target_size
    rgb_pil = Image.fromarray(
        ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    )
    rgb_pil = rgb_pil.resize(
        (target_size, target_size), Image.BILINEAR,
    )
    rgb = np.array(rgb_pil, dtype=np.float32) / 127.5 - 1.0

    # NEAREST, not BILINEAR: a bilinear resize blends the invalid zeros into
    # their valid neighbours, and 1 / (a blended small depth) is a huge
    # disparity that would set the per-map maximum.
    depth_pil = Image.fromarray(depth)
    depth_pil = depth_pil.resize(
        (target_size, target_size), Image.NEAREST,
    )
    depth = np.array(depth_pil, dtype=np.float32)

    mask_pil = Image.fromarray(valid_mask)
    mask_pil = mask_pil.resize(
        (target_size, target_size), Image.NEAREST,
    )
    valid_mask = np.array(mask_pil, dtype=np.float32)

    rgb = rgb[..., :3]
    depth = depth[..., np.newaxis]
    valid_mask = valid_mask[..., np.newaxis]

    depth = normalized_disparity(depth, valid_mask * (depth > 0))
    if not _has_spread(depth):
        return None

    # Augmentation
    if augment:
        rgb, depth, valid_mask = _mirror_left_right(
            rgb, depth, valid_mask, np.random.random)

    y_true = np.concatenate([depth, valid_mask], axis=-1)
    return (
        rgb.astype(np.float32),
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
      with ``[disparity, mask]`` concatenated on the last axis
      (disparity in ``[0, 1]``, :func:`normalized_disparity`).

    Args:
        rgb_paths: List of RGB image file paths.
        depth_paths: List of HDF5 depth file paths (matched by index).
        batch_size: Number of samples per batch.
        patch_size: Spatial size of cropped patches.
        min_valid_ratio: Minimum fraction of valid depth pixels per
            crop.  Defaults to ``0.1``.
        augment: Whether to mirror samples left-right at random (the only
            labeled-image augmentation of the reference). Defaults to ``True``.
        is_training: If ``True``, shuffles indices each epoch and crops at
            random; if ``False``, each file's crop is fixed for the run
            (seeded by its index), and the constructor loads every file once
            to keep only those whose crop qualifies, so each validation batch
            holds the next ``batch_size`` of them and scores each once.
        workers: Number of multiprocessing workers.
        imagenet_standardize: If ``True``, the RGB batch is
            :func:`standardize_imagenet`'d (the official DINOv2 input);
            otherwise it stays in ``[-1, +1]``. Defaults to ``False``.
        **kwargs: Passed to :class:`keras.utils.PyDataset`.

    Raises:
        RuntimeError: If ``is_training`` is ``False`` and no file's crop
            qualifies, or (from :meth:`__getitem__`) if no crop qualifies in
            a training batch's ``3 * batch_size`` attempts.
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
        use_resize: bool = False,
        imagenet_standardize: bool = False,
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
        self.use_resize = use_resize
        self.imagenet_standardize = bool(imagenet_standardize)
        self.n_pairs = len(rgb_paths)
        self.indices = np.arange(self.n_pairs)
        if is_training:
            np.random.shuffle(self.indices)
        else:
            # DECISION plan-2026-10-14T062616-81938e81/D-197
            # Do NOT let a validation batch refill a rejected file from the
            # next window: 24.3% of the default validation files are rejected,
            # so about a quarter of the files were scored twice in val_loss and
            # one in four not at all. A validation file's crop is fixed, so
            # whether it qualifies is fixed too: keep the files that qualify.
            self.indices = self._qualifying_files(max(1, int(workers)))

    def _qualifying_files(self, threads: int) -> np.ndarray:
        """The indices of the files whose fixed crop qualifies, in order.

        Loads every file once (``threads`` at a time). Raises ``RuntimeError``
        naming the files when none qualifies.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads) as pool:
            accepted = list(pool.map(
                lambda i: self._load_one(i) is not None, range(self.n_pairs)))
        kept = np.flatnonzero(accepted)
        logger.info(
            f"MegaDepthDataset: {len(kept)} of {self.n_pairs} validation files "
            f"give a qualifying crop")
        if not len(kept):
            raise RuntimeError(
                f"MegaDepthDataset: no MegaDepth crop qualified (enough valid "
                f"pixels, depth with spread) in any of the {self.n_pairs} "
                f"validation files; tried {list(self.rgb_paths)}")
        return kept

    def __len__(self) -> int:
        return max(1, len(self.indices) // self.batch_size)

    def __getitem__(self, idx: int):
        # DECISION plan-2026-10-14T062616-81938e81/D-126
        # Do NOT add len(batch_rgb) to this offset: `attempt` already
        # counts every load, kept or skipped, so the sum advanced by 2
        # per sample. Consecutive batches overlapped by half and the files
        # at odd offsets were never read.
        # DECISION plan-2026-10-14T062616-81938e81/D-194
        # Do NOT pad a batch in which no crop qualified with a zero sample:
        # a mid-grey image with an all-invalid mask still took an optimizer
        # step. Measured at the trainer defaults, 18.9% of training draws are
        # rejected and no default batch is all-rejected, so fill_batch raises,
        # naming the files, and draws no further samples.
        samples = fill_batch(
            "MegaDepthDataset", idx, self.batch_size, self.indices,
            self._load_one, lambda i: self.rgb_paths[i],
            "MegaDepth crop qualified (enough valid pixels, depth with spread)",
        )
        rgb = np.stack([sample[0] for sample in samples])
        if self.imagenet_standardize:
            rgb = standardize_imagenet(rgb)
        return rgb, np.stack([sample[1] for sample in samples])

    def _load_one(self, i: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """File ``i``'s ``(rgb, y_true)`` sample, or ``None`` if no crop qualified."""
        if self.use_resize:
            return load_and_resize_pair(
                self.rgb_paths[i],
                self.depth_paths[i],
                self.patch_size,
                self.min_valid_ratio,
                augment=self.is_training and self.augment,
            )
        # DECISION plan-2026-10-14T062616-81938e81/D-109
        # Do NOT let a validation crop draw from the global RNG: each
        # epoch would score different crops of the same files, and the
        # val_loss trend would be crop noise. Seeding by file index
        # fixes each file's crop for the whole run.
        return load_and_process_pair(
            self.rgb_paths[i],
            self.depth_paths[i],
            self.patch_size,
            self.min_valid_ratio,
            augment=self.is_training and self.augment,
            rng=None if self.is_training else np.random.default_rng(int(i)),
        )

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


# =====================================================================
# Unlabeled image dataset + labeled/unlabeled pairing (semi-supervised)
# =====================================================================


def _discover_image_files(glob_pattern: str) -> List[str]:
    """Resolve a glob pattern to a sorted list of image file paths."""
    import glob as _glob
    files = sorted(_glob.glob(glob_pattern, recursive=True))
    return files


class UnlabeledImageDataset(keras.utils.PyDataset):
    """Minimal RGB-only PyDataset for unlabeled depth-anything pretraining.

    Each ``__getitem__(idx)`` returns a numpy array
    ``(batch_size, patch_size, patch_size, 3)`` of float32 RGB normalized to
    ``[-1, +1]`` (matching :class:`MegaDepthDataset`'s RGB convention).

    This dataset is intentionally simple: it does **not** load depth maps.
    Use it as the unlabeled stream paired with a labeled
    :class:`MegaDepthDataset` via :func:`pair_labeled_unlabeled`.

    Args:
        rgb_paths: List of RGB image file paths.
        batch_size: Number of samples per batch.
        patch_size: Spatial size of resized images.
        is_training: If ``True``, shuffles indices each epoch.
        workers: Number of multiprocessing workers (default 4).
        imagenet_standardize: If ``True``, the batch is
            :func:`standardize_imagenet`'d, as :class:`MegaDepthDataset`'s.
        **kwargs: Passed to :class:`keras.utils.PyDataset`.
    """

    def __init__(
        self,
        rgb_paths: List[str],
        batch_size: int = 16,
        patch_size: int = 256,
        is_training: bool = True,
        workers: int = 4,
        imagenet_standardize: bool = False,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=True, **kwargs)
        if not rgb_paths:
            raise ValueError("UnlabeledImageDataset: rgb_paths is empty.")
        self.rgb_paths = list(rgb_paths)
        self.batch_size = int(batch_size)
        self.patch_size = int(patch_size)
        self.is_training = bool(is_training)
        self.imagenet_standardize = bool(imagenet_standardize)
        self.n = len(self.rgb_paths)
        self.indices = np.arange(self.n)
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, self.n // self.batch_size)

    def _load_one(self, path: str) -> np.ndarray:
        """Load + resize one image to (patch_size, patch_size, 3) in [-1, +1]."""
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(
            (self.patch_size, self.patch_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # → [-1, +1]
        return arr

    def __getitem__(self, idx: int) -> np.ndarray:
        batch = []
        for k in range(self.batch_size):
            sample_idx = (idx * self.batch_size + k) % self.n
            i = self.indices[sample_idx]
            # DECISION plan-2026-10-14T062616-81938e81/D-130
            # Do NOT pad an unreadable image with zeros again: a missing or
            # corrupt file became a uniform grey input for the student and the
            # teacher alike, with nothing in the log. The run stops instead.
            try:
                batch.append(self._load_one(self.rgb_paths[i]))
            except OSError as exc:
                raise OSError(
                    f"UnlabeledImageDataset: cannot read {self.rgb_paths[i]}: {exc}"
                ) from exc
        batch = np.stack(batch).astype(np.float32)
        return standardize_imagenet(batch) if self.imagenet_standardize else batch

    def on_epoch_end(self) -> None:
        if self.is_training:
            np.random.shuffle(self.indices)


def pair_labeled_unlabeled(
    labeled_ds: "keras.utils.PyDataset",
    unlabeled_ds: "keras.utils.PyDataset",
    patch_size: int,
    batch_size: int,
):
    """Yield ``((x_lab, x_unlab), y_lab)`` batches from two PyDatasets.

    Wraps both PyDatasets via :func:`tf.data.Dataset.from_generator` and
    zips them into the structure consumed by
    :meth:`DepthAnything._train_step_semi_supervised`.

    The unlabeled stream cycles independently of the labeled stream
    (`.repeat()`) so the pairing length is bounded by the labeled dataset.

    Args:
        labeled_ds: A ``keras.utils.PyDataset`` yielding
            ``(rgb_batch, y_true_batch)``.
        unlabeled_ds: An :class:`UnlabeledImageDataset` (or any PyDataset
            yielding ``rgb_batch`` of matching dtype/shape).
        patch_size: Spatial patch size (must match both datasets).
        batch_size: Per-batch size (must match both datasets).

    Returns:
        A ``tf.data.Dataset`` yielding ``((x_lab, x_unlab), y_lab)``.
    """
    import tensorflow as tf

    rgb_spec = tf.TensorSpec(
        shape=(batch_size, patch_size, patch_size, 3), dtype=tf.float32
    )
    y_spec = tf.TensorSpec(
        shape=(batch_size, patch_size, patch_size, 2), dtype=tf.float32
    )

    def _lab_gen():
        for i in range(len(labeled_ds)):
            rgb, y = labeled_ds[i]
            yield rgb.astype(np.float32), y.astype(np.float32)

    def _unlab_gen():
        # Loop forever — outer zip will stop at labeled length.
        while True:
            for i in range(len(unlabeled_ds)):
                yield unlabeled_ds[i].astype(np.float32)

    lab_tf = tf.data.Dataset.from_generator(
        _lab_gen, output_signature=(rgb_spec, y_spec)
    )
    unlab_tf = tf.data.Dataset.from_generator(
        _unlab_gen, output_signature=rgb_spec
    )

    paired = tf.data.Dataset.zip((lab_tf, unlab_tf)).map(
        lambda lab, unlab: ((lab[0], unlab), lab[1]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return paired
