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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        # Optional detection visualization — if provided, also draws
        # RGB + GT boxes + predicted boxes grids per epoch.
        val_boxes: Optional[np.ndarray] = None,  # (N, max_boxes, 5) xyxy-norm
        detection_reg_max: Optional[int] = None,
        detection_num_classes: Optional[int] = None,
        detection_strides: Optional[Tuple[int, ...]] = None,
        detection_score_threshold: float = 0.25,
        detection_iou_threshold: float = 0.45,
        detection_max_draw: int = 30,
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

        # Detection setup (optional)
        self.val_boxes = (
            np.asarray(val_boxes[:max_samples]) if val_boxes is not None else None
        )
        self.detection_reg_max = detection_reg_max
        self.detection_num_classes = detection_num_classes
        self.detection_strides = (
            tuple(int(s) for s in detection_strides)
            if detection_strides is not None
            else None
        )
        self.detection_score_threshold = float(detection_score_threshold)
        self.detection_iou_threshold = float(detection_iou_threshold)
        self.detection_max_draw = int(detection_max_draw)

        # Pre-compute anchors once when detection enabled.
        if self._detection_enabled():
            from dl_techniques.utils.yolo_decode import make_anchors_np
            H, W = self.val_rgb.shape[1], self.val_rgb.shape[2]
            self._det_anchors, self._det_stride_tensor = make_anchors_np(
                (H, W), self.detection_strides,
            )
        else:
            self._det_anchors = None
            self._det_stride_tensor = None

        logger.info(
            f"COCOMultiTaskPredictionGridCallback: {self.max_samples} samples, "
            f"every {frequency} epochs → {self.output_dir}"
            + (" (with detection boxes)" if self._detection_enabled() else "")
        )

    def _detection_enabled(self) -> bool:
        return (
            self.val_boxes is not None
            and self.detection_reg_max is not None
            and self.detection_num_classes is not None
            and self.detection_strides is not None
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

    def _save_det_grid(self, epoch: int, pred_det: np.ndarray) -> None:
        """RGB with GT boxes (green) + predicted boxes (red, argmax class+score)."""
        plt, _ = _import_mpl()
        import matplotlib.patches as mpatches
        from dl_techniques.utils.yolo_decode import (
            decode_predictions, nms_per_class,
        )

        H, W = self.val_rgb.shape[1], self.val_rgb.shape[2]
        per_image = decode_predictions(
            pred_det,
            self._det_anchors,
            self._det_stride_tensor,
            reg_max=int(self.detection_reg_max),  # type: ignore[arg-type]
            num_classes=int(self.detection_num_classes),  # type: ignore[arg-type]
            score_threshold=self.detection_score_threshold,
        )

        n = self.max_samples
        fig, axes = plt.subplots(n, 1, figsize=(8, 5 * n), squeeze=False)
        for i in range(n):
            ax = axes[i][0]
            ax.imshow(np.clip(self.val_rgb[i], 0.0, 1.0))

            # GT boxes (green)
            if self.val_boxes is not None:
                valid = self.val_boxes[i, :, 0] >= 0
                for row in self.val_boxes[i][valid]:
                    cls_idx = int(row[0])
                    x1, y1, x2, y2 = row[1] * W, row[2] * H, row[3] * W, row[4] * H
                    ax.add_patch(mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="#00ff00", facecolor="none",
                    ))
                    ax.text(
                        x1, max(0, y1 - 3),
                        self.class_names[cls_idx] if cls_idx < len(self.class_names) else str(cls_idx),
                        color="#00ff00", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"),
                    )

            # Predicted boxes (NMS'd, red)
            boxes_px, scores, classes = per_image[i]
            keep_b, keep_s, keep_c = nms_per_class(
                boxes_px, scores, classes,
                iou_threshold=self.detection_iou_threshold,
                max_per_class=self.detection_max_draw,
                max_total=self.detection_max_draw,
            )
            for j in range(keep_b.shape[0]):
                x1, y1, x2, y2 = keep_b[j]
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="#ff3030", facecolor="none",
                    linestyle="--",
                ))
                ax.text(
                    x2 - 40, min(H - 2, y2 - 2),
                    f"{self.class_names[keep_c[j]] if keep_c[j] < len(self.class_names) else keep_c[j]}:{keep_s[j]:.2f}",
                    color="#ff3030", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5, edgecolor="none"),
                )

            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_title(f"Sample {i} — GT (solid green) vs Pred (dashed red)")
            ax.axis("off")

        fig.suptitle(f"COCO detection epoch {epoch}", fontsize=12)
        fig.tight_layout()
        path = self.output_dir / f"epoch_{epoch:03d}_det.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        logger.info(f"Saved det prediction grid: {path}")

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

            if self._detection_enabled():
                det = out.get("detection")
                if det is not None:
                    pred_det = np.asarray(keras.ops.convert_to_numpy(det))
                    try:
                        self._save_det_grid(epoch + 1, pred_det)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Detection viz failed: {e}")
                    del pred_det
            del out, pred_seg, pred_cls
        finally:
            gc.collect()
