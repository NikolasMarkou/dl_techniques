"""Fixtures shared by the segmentation data and transform tests: pairs whose colour IS their class.

``picture(mask)`` paints a mask with ``PALETTE`` (void grey), so ``class_of(image)`` reads the class
back from the pixels; ``blocks`` makes 8x8-block masks (a JPEG keeps their colours);
``write_pair`` / ``make_folder`` / ``listing_of`` write PNG or JPEG pairs under a ``tmp_path``;
``write_csv`` writes a pair CSV; ``load`` builds a :class:`~train.common.segmentation_data.SegmentationData`
from a listing with small defaults.
"""

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image

import train.common.segmentation_data as sd

# Colours of the classes in the images: the class of a pixel can be read back from its colour.
PALETTE = np.asarray([[200, 40, 40], [40, 200, 40], [40, 40, 200], [200, 200, 40]], np.uint8)


def picture(mask: np.ndarray) -> np.ndarray:
    """An RGB image whose colour is the class: ``PALETTE[mask]`` (void gets grey)."""
    colours = np.vstack([PALETTE, np.asarray([[128, 128, 128]], np.uint8)])
    return colours[np.where(mask < len(PALETTE), mask, len(PALETTE))]


def class_of(image_u8: np.ndarray) -> np.ndarray:
    """The class a decoded pixel names: the nearest palette colour (JPEG noise is small)."""
    colours = np.vstack([PALETTE, [[128, 128, 128]]]).astype(np.int32)
    distance = ((image_u8.astype(np.int32)[..., None, :] - colours) ** 2).sum(-1)
    return distance.argmin(-1)


def blocks(height: int, width: int, classes: int = 3, seed: int = 0) -> np.ndarray:
    """A mask of 8x8 blocks with random classes (so a JPEG keeps the colours)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, classes, size=((height + 7) // 8, (width + 7) // 8))
    return np.kron(small, np.ones((8, 8), np.int64))[:height, :width].astype(np.uint8)


def write_pair(image_dir: Path, mask_dir: Path, stem: str, mask: np.ndarray, image_suffix: str = ".png") -> Tuple[Path, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_path, mask_path = image_dir / f"{stem}{image_suffix}", mask_dir / f"{stem}.png"
    Image.fromarray(picture(mask)).save(image_path, quality=100) if image_suffix != ".png" else Image.fromarray(picture(mask)).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


def make_folder(root: Path, count: int = 12, size: Tuple[int, int] = (24, 32), classes: int = 3,
                prefix: str = "", image_suffix: str = ".png") -> Tuple[Path, Path]:
    images, masks = root / f"{prefix}images", root / f"{prefix}masks"
    for index in range(count):
        write_pair(images, masks, f"p{index:03d}", blocks(*size, classes=classes, seed=index), image_suffix)
    return images, masks


def listing_of(root: Path, **kwargs) -> sd.PairListing:
    images, masks = make_folder(root, **kwargs)
    return sd.scan_segmentation_folder(str(images), str(masks))


def write_csv(path: Path, rows: Sequence[Sequence[str]], header: Sequence[str] = ("image", "mask")) -> Path:
    path.write_text("\n".join([",".join(header)] + [",".join(r) for r in rows]) + "\n", encoding="utf-8")
    return path


def load(listing, **overrides) -> sd.SegmentationData:
    options = dict(num_classes=3, class_names=["a", "b", "c"], ignore_index=None, batch_size=2, seed=0,
                   size=(16, 24), validation_split=0.25, eval_samples=None, max_samples=None)
    options.update(overrides)
    return sd.load_pair_data(listing, **options)
