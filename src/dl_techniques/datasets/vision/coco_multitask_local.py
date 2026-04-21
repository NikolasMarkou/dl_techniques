"""
Local COCO 2017 multi-task loader — image classification + semantic segmentation.

Reads directly from a local COCO directory (``train2017/``, ``val2017/``,
``annotations/instances_*.json``) via ``pycocotools``.  Produces dict labels
that match a :class:`CliffordNetUNet`-style multi-head forward pass:

    (image, {"classification": (B, 80) float32 multi-hot,
             "segmentation":   (B, H, W)  int32  class indices in [0..80]})

Class indexing
--------------
- Classification: 80 contiguous indices ``0..79`` (one per COCO "thing" category).
- Segmentation:   81 contiguous indices ``0..80`` where ``0 = background`` and
  ``k+1`` is the classification index ``k``.  Overlapping instances at the
  same pixel use the last-encountered class (paint order).

This loader is deliberately independent of ``COCODatasetBuilder`` (which uses
tensorflow_datasets) so it can read from the user's local copy without going
through tfds.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import keras

from dl_techniques.utils.logger import logger

# pycocotools is required for instance annotation parsing & mask decoding.
# We import at module import time to surface the dependency early.
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as cocomask
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pycocotools is required for coco_multitask_local. "
        "Install via `pip install pycocotools`."
    ) from e


COCO_DEFAULT_ROOT: str = "/media/arxwn/data0_4tb/datasets/coco_2017"
NUM_COCO_CLASSES: int = 80  # 80 "thing" categories
SEG_BACKGROUND: int = 0


@dataclass
class COCOMultiTaskConfig:
    """Configuration for :class:`COCO2017MultiTaskLoader`."""

    coco_root: str = COCO_DEFAULT_ROOT
    split: str = "train2017"  # "train2017" or "val2017"
    image_size: int = 384
    batch_size: int = 8

    # Data subset & shuffling
    max_images: Optional[int] = None  # for quick smoke runs
    shuffle: bool = True
    seed: int = 0

    # Augmentation (training only)
    augment: bool = True
    hflip_prob: float = 0.5

    # Multiprocessing
    workers: int = 4
    use_multiprocessing: bool = True

    # Normalisation — simple [0,1] scaling.  Callers wanting ImageNet mean/std
    # can subtract downstream.
    pixel_scale: float = 1.0 / 255.0


class COCO2017MultiTaskLoader(keras.utils.PyDataset):
    """Keras :class:`PyDataset` yielding (image, {classification, segmentation}).

    :param config: Loader configuration.
    """

    def __init__(self, config: COCOMultiTaskConfig, **kwargs: Any) -> None:
        super().__init__(
            workers=config.workers,
            use_multiprocessing=config.use_multiprocessing,
            **kwargs,
        )
        self.config = config
        self._rng = random.Random(config.seed)

        ann_file = os.path.join(
            config.coco_root, "annotations", f"instances_{config.split}.json"
        )
        images_dir = os.path.join(config.coco_root, config.split)
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotations not found: {ann_file}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        logger.info(f"Loading COCO annotations from {ann_file}")
        self.coco = COCO(ann_file)
        self.images_dir = images_dir

        # Map COCO category id → contiguous classification index [0..79].
        cat_ids = sorted(self.coco.getCatIds())
        if len(cat_ids) != NUM_COCO_CLASSES:
            logger.warning(
                f"Expected {NUM_COCO_CLASSES} COCO categories, got {len(cat_ids)}"
            )
        self.cat_id_to_idx: Dict[int, int] = {cid: i for i, cid in enumerate(cat_ids)}
        self.idx_to_cat_id: Dict[int, int] = {i: cid for cid, i in self.cat_id_to_idx.items()}

        # Build ordered image id list; filter images that have no annotations?
        # We keep them — classification target is zeros, segmentation is all
        # background — so the model still learns from negative examples.
        self.image_ids: List[int] = list(self.coco.getImgIds())
        if config.max_images is not None:
            self.image_ids = self.image_ids[: config.max_images]
        if config.shuffle:
            self._rng.shuffle(self.image_ids)

        logger.info(
            f"COCO2017MultiTaskLoader: split={config.split}, "
            f"#images={len(self.image_ids)}, image_size={config.image_size}, "
            f"batch_size={config.batch_size}"
        )

    def __len__(self) -> int:
        n = len(self.image_ids)
        return (n + self.config.batch_size - 1) // self.config.batch_size

    # ------------------------------------------------------------------
    # Per-sample helpers
    # ------------------------------------------------------------------

    def _build_mask(self, image_id: int, out_hw: Tuple[int, int]) -> np.ndarray:
        """Rasterize all instances of *image_id* into a (H, W) int32 class map.

        Class indices are ``0`` (background) or ``k+1`` for classification
        index ``k``.  Overlapping instances: last-painted wins.
        """
        info = self.coco.loadImgs(image_id)[0]
        h, w = info["height"], info["width"]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Rasterise at the original image resolution, then resize at the end.
        mask_full = np.zeros((h, w), dtype=np.int32)
        for ann in anns:
            cat_id = ann["category_id"]
            cls_idx = self.cat_id_to_idx.get(cat_id)
            if cls_idx is None:
                continue
            seg_val = cls_idx + 1  # 0 reserved for background
            # pycocotools returns a binary mask shape (h, w)
            m = self.coco.annToMask(ann)
            if m.shape != (h, w):
                continue
            mask_full[m > 0] = seg_val

        # Nearest-neighbor resize to out_hw
        tgt_h, tgt_w = out_hw
        if (tgt_h, tgt_w) != (h, w):
            # Use PIL for fast nearest resize on int arrays via mode 'I'
            mask_pil = Image.fromarray(mask_full.astype(np.int32), mode="I")
            mask_pil = mask_pil.resize((tgt_w, tgt_h), resample=Image.NEAREST)
            mask_full = np.asarray(mask_pil, dtype=np.int32)

        return mask_full

    def _build_classification(self, image_id: int) -> np.ndarray:
        """80-dim multi-hot vector — 1 wherever ≥ 1 instance of that class."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        y = np.zeros((NUM_COCO_CLASSES,), dtype=np.float32)
        for ann in anns:
            cls_idx = self.cat_id_to_idx.get(ann["category_id"])
            if cls_idx is not None:
                y[cls_idx] = 1.0
        return y

    def _load_image(self, image_id: int, size: int) -> np.ndarray:
        info = self.coco.loadImgs(image_id)[0]
        path = os.path.join(self.images_dir, info["file_name"])
        img = Image.open(path).convert("RGB").resize((size, size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) * self.config.pixel_scale
        return arr

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        size = self.config.image_size
        bs = self.config.batch_size
        start = idx * bs
        end = min(start + bs, len(self.image_ids))
        batch_ids = self.image_ids[start:end]

        imgs = np.empty((len(batch_ids), size, size, 3), dtype=np.float32)
        cls = np.empty((len(batch_ids), NUM_COCO_CLASSES), dtype=np.float32)
        seg = np.empty((len(batch_ids), size, size), dtype=np.int32)

        for i, image_id in enumerate(batch_ids):
            img = self._load_image(image_id, size)
            mask = self._build_mask(image_id, (size, size))

            if self.config.augment and self._rng.random() < self.config.hflip_prob:
                img = img[:, ::-1, :]
                mask = mask[:, ::-1]

            imgs[i] = img
            cls[i] = self._build_classification(image_id)
            seg[i] = mask

        labels: Dict[str, np.ndarray] = {
            "classification": cls,
            "segmentation": seg,
        }
        return imgs, labels

    def on_epoch_end(self) -> None:
        if self.config.shuffle:
            self._rng.shuffle(self.image_ids)

    # ------------------------------------------------------------------
    # Convenience: probe a single batch for shape / distribution inspection
    # ------------------------------------------------------------------

    def probe(self) -> Dict[str, Any]:
        """Load one batch and return summary statistics (shape, class histogram)."""
        x, y = self[0]
        seg_hist = np.bincount(y["segmentation"].ravel(), minlength=NUM_COCO_CLASSES + 1)
        cls_freq = y["classification"].sum(axis=0)
        return {
            "image_shape": tuple(x.shape),
            "classification_shape": tuple(y["classification"].shape),
            "segmentation_shape": tuple(y["segmentation"].shape),
            "image_range": (float(x.min()), float(x.max())),
            "classification_positive_fraction": float(
                y["classification"].mean()
            ),
            "segmentation_class_counts": seg_hist.tolist(),
            "classification_per_image_avg_labels": float(
                y["classification"].sum(axis=1).mean()
            ),
            "segmentation_unique_classes_sample0": int(
                len(np.unique(y["segmentation"][0]))
            ),
            "classification_frequency_top10": sorted(
                enumerate(cls_freq.tolist()), key=lambda kv: -kv[1]
            )[:10],
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_coco_multitask_datasets(
    coco_root: str = COCO_DEFAULT_ROOT,
    image_size: int = 384,
    batch_size: int = 8,
    max_train_images: Optional[int] = None,
    max_val_images: Optional[int] = None,
    workers: int = 4,
    augment_train: bool = True,
    seed: int = 0,
) -> Tuple[COCO2017MultiTaskLoader, COCO2017MultiTaskLoader]:
    """Create matched train / val :class:`COCO2017MultiTaskLoader` instances."""
    train_cfg = COCOMultiTaskConfig(
        coco_root=coco_root,
        split="train2017",
        image_size=image_size,
        batch_size=batch_size,
        max_images=max_train_images,
        shuffle=True,
        augment=augment_train,
        workers=workers,
        seed=seed,
    )
    val_cfg = COCOMultiTaskConfig(
        coco_root=coco_root,
        split="val2017",
        image_size=image_size,
        batch_size=batch_size,
        max_images=max_val_images,
        shuffle=False,
        augment=False,
        workers=max(1, workers // 2),
        seed=seed + 1,
    )
    return (
        COCO2017MultiTaskLoader(train_cfg),
        COCO2017MultiTaskLoader(val_cfg),
    )
