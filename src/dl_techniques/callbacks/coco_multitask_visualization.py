"""COCO multi-task visualization callback.

Saves a grid of (RGB | GT semantic mask | predicted semantic mask) for a
fixed set of validation samples at regular training epochs, plus a text
summary of the classification head's top-K predictions vs ground-truth
labels.

Model-agnostic w.r.t. head naming: expects ``model(x, training=False)``
to return a ``dict`` containing at least:
- ``"segmentation"``: ``(B, H, W, num_classes)`` logits
- ``"classification"``: ``(B, num_classes)`` logits

Aux outputs (``segmentation_aux_k``, etc.) are ignored — only the primary
spatial output is visualized.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger


# COCO 80 category names (classification index 0..79; seg index 0=bg, k+1=class k).
COCO_CLASS_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
    "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear", "hair_drier",
    "toothbrush",
]


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    return plt, cm


def _seg_to_rgb(mask: np.ndarray, cmap) -> np.ndarray:
    """Map int class indices (H, W) in [0, 80] to an RGB image via a colormap.

    Class 0 (background) is rendered black; classes 1..80 use the colormap.
    """
    out = np.zeros(mask.shape + (3,), dtype=np.float32)
    # Normalize class indices to [0, 1] by (k-1)/79 for k>=1; bg stays 0.
    nonbg = mask > 0
    if nonbg.any():
        norm = (mask - 1).clip(min=0, max=79) / 79.0
        rgba = cmap(norm)  # (H, W, 4)
        rgb = rgba[..., :3]
        out[nonbg] = rgb[nonbg]
    return out


class COCOMultiTaskPredictionGridCallback(keras.callbacks.Callback):
    """Save RGB | GT seg | predicted seg comparison grid every *frequency* epochs.

    Also writes a small text file per save with classification top-K for each
    sample vs ground-truth labels.
    """

    def __init__(
        self,
        val_rgb: np.ndarray,             # (N, H, W, 3) in [0, 1]
        val_seg: np.ndarray,             # (N, H, W) int32 in [0, num_seg_classes-1]
        val_cls: np.ndarray,             # (N, num_cls_classes) multi-hot float32
        output_dir: str,
        frequency: int = 5,
        max_samples: int = 6,
        top_k: int = 5,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.val_rgb = np.asarray(val_rgb[:max_samples])
        self.val_seg = np.asarray(val_seg[:max_samples])
        self.val_cls = np.asarray(val_cls[:max_samples])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.max_samples = self.val_rgb.shape[0]
        self.top_k = top_k
        self.class_names = class_names or COCO_CLASS_NAMES
        logger.info(
            f"COCOMultiTaskPredictionGridCallback: {self.max_samples} samples, "
            f"every {frequency} epochs → {self.output_dir}"
        )

    def _save_grid(self, epoch: int, pred_seg: np.ndarray, pred_cls: np.ndarray) -> None:
        plt, cm = _import_mpl()
        cmap = cm.get_cmap("tab20", 80)

        n = self.max_samples
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n), squeeze=False)
        for i in range(n):
            # Col 1: RGB
            axes[i][0].imshow(np.clip(self.val_rgb[i], 0.0, 1.0))
            axes[i][0].set_title("RGB" if i == 0 else "")
            axes[i][0].axis("off")

            # Col 2: GT seg
            axes[i][1].imshow(_seg_to_rgb(self.val_seg[i].astype(np.int32), cmap))
            axes[i][1].set_title("GT seg" if i == 0 else "")
            axes[i][1].axis("off")

            # Col 3: Pred seg (argmax)
            pred = pred_seg[i].argmax(axis=-1).astype(np.int32)
            axes[i][2].imshow(_seg_to_rgb(pred, cmap))
            axes[i][2].set_title("Pred seg" if i == 0 else "")
            axes[i][2].axis("off")

        fig.suptitle(f"COCO multi-task epoch {epoch}", fontsize=12)
        fig.tight_layout()
        path = self.output_dir / f"epoch_{epoch:03d}_seg.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        logger.info(f"Saved seg prediction grid: {path}")

    def _save_cls_summary(self, epoch: int, pred_cls: np.ndarray) -> None:
        # pred_cls: (N, 80) logits
        # Convert to probabilities via sigmoid (multi-label) for ranking only.
        probs = 1.0 / (1.0 + np.exp(-pred_cls))
        lines: List[str] = [f"Classification top-{self.top_k} (epoch {epoch})\n"]
        for i in range(self.max_samples):
            gt_idx = np.flatnonzero(self.val_cls[i] > 0.5).tolist()
            gt_names = [self.class_names[j] for j in gt_idx]
            topk = np.argsort(-probs[i])[: self.top_k]
            topk_pairs = [f"{self.class_names[j]}({probs[i][j]:.2f})" for j in topk]
            lines.append(
                f"[{i}] GT={gt_names or ['<none>']}  TopK={topk_pairs}"
            )
        path = self.output_dir / f"epoch_{epoch:03d}_cls.txt"
        path.write_text("\n".join(lines))
        logger.info(f"Saved cls summary: {path}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.frequency != 0:
            return
        try:
            out = self.model(self.val_rgb, training=False)
            if not isinstance(out, dict):
                logger.warning(
                    f"Expected dict output; got {type(out)}. Skipping visualization."
                )
                return
            seg = out.get("segmentation")
            cls = out.get("classification")
            if seg is None or cls is None:
                logger.warning(
                    "Model output missing 'segmentation' or 'classification'; "
                    "skipping visualization."
                )
                return
            pred_seg = np.asarray(keras.ops.convert_to_numpy(seg))
            pred_cls = np.asarray(keras.ops.convert_to_numpy(cls))

            self._save_grid(epoch + 1, pred_seg, pred_cls)
            self._save_cls_summary(epoch + 1, pred_cls)
            del out, pred_seg, pred_cls
        finally:
            gc.collect()
