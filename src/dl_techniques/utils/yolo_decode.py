"""YOLOv12 prediction decoder — standalone utilities for inference / mAP eval.

Reuses the existing YOLO infrastructure in this repo:

- :class:`dl_techniques.layers.anchor_generator.AnchorGenerator` — pixel-coord
  anchor point generation as a Keras layer.
- ``_make_anchors`` helpers in :class:`YOLOv12ObjectDetectionLoss` and
  :class:`CliffordDetectionLoss` — the same logic in *feature-map* coords.

This module contributes what's still missing:

1. :func:`decode_dfl_logits` — expectation over softmax of the DFL regression
   logits (lifted from ``yolo12_multitask_loss.py:380-402`` into a reusable
   numpy-eager form).
2. :func:`decode_predictions` — combines DFL + anchors + strides → absolute
   xyxy pixel boxes + per-class scores for each image in a batch.
3. :func:`nms_per_class` — eager numpy per-class greedy NMS suitable for
   val-end callbacks (D-009).  No graph dependency.

All functions operate on numpy arrays (the callback path materializes
predictions to numpy after the forward pass).  For graph-mode NMS inside
the training step, use ``dl_techniques.utils.bounding_box.bbox_nms`` instead.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Anchors (numpy, pixel coordinates — matches AnchorGenerator's convention)
# ---------------------------------------------------------------------------


def make_anchors_np(
    input_shape: Tuple[int, int],
    strides: Sequence[int],
    grid_cell_offset: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate anchor centers + per-anchor stride tensor in **pixel** coords.

    Mirrors :class:`AnchorGenerator._make_anchors` but as a standalone numpy
    function for callback use.  For each stride ``s``, creates an
    ``(H/s × W/s)`` grid of centers at ``((j + 0.5) * s, (i + 0.5) * s)``.

    :param input_shape: ``(H, W)`` — training image size.
    :param strides: Feature-map strides, one per detection scale.
    :param grid_cell_offset: Offset to place anchors at cell centers. Default 0.5.
    :returns: ``(anchors, strides_tensor)`` — shapes ``(N, 2)`` and ``(N, 1)``.
    """
    H, W = input_shape
    anchor_points: List[np.ndarray] = []
    stride_tensor: List[np.ndarray] = []

    for s in strides:
        h, w = H // int(s), W // int(s)
        if h <= 0 or w <= 0:
            raise ValueError(
                f"stride {s} too large for input_shape {(H, W)}: feature map {(h, w)}"
            )
        x_coords = (np.arange(w, dtype=np.float32) + grid_cell_offset) * s
        y_coords = (np.arange(h, dtype=np.float32) + grid_cell_offset) * s
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
        xy = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)
        anchor_points.append(xy)
        stride_tensor.append(
            np.full((h * w, 1), float(s), dtype=np.float32)
        )

    return (
        np.concatenate(anchor_points, axis=0),
        np.concatenate(stride_tensor, axis=0),
    )


# ---------------------------------------------------------------------------
# DFL → per-edge distances → xyxy boxes
# ---------------------------------------------------------------------------


def _softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_dfl_logits(
    reg_logits: np.ndarray, reg_max: int
) -> np.ndarray:
    """Convert DFL regression logits to per-edge distances (in stride units).

    Each of the 4 box edges (left, top, right, bottom) is predicted as a
    categorical distribution over ``reg_max`` bins indexed 0..reg_max-1.
    The decoded distance is the expectation over softmax(logits).

    :param reg_logits: ``(..., 4*reg_max)`` raw logits.
    :param reg_max: Number of DFL bins per edge.
    :returns: ``(..., 4)`` edge distances in stride units (pre-stride-scaling).
    """
    shape = reg_logits.shape
    # Reshape last dim → (4, reg_max)
    reshaped = reg_logits.reshape(*shape[:-1], 4, reg_max)
    probs = _softmax_np(reshaped, axis=-1)
    bins = np.arange(reg_max, dtype=np.float32)
    return np.sum(probs * bins, axis=-1)


def dist_to_xyxy(
    distances: np.ndarray, anchor_centers: np.ndarray
) -> np.ndarray:
    """Convert (left, top, right, bottom) distances + anchor centers to xyxy.

    :param distances: ``(..., 4)`` — l, t, r, b in pixel units.
    :param anchor_centers: ``(N, 2)`` — anchor (x, y) centers in pixels.
        Broadcastable over leading dims of ``distances``.
    :returns: ``(..., 4)`` xyxy boxes.
    """
    x1 = anchor_centers[..., 0] - distances[..., 0]
    y1 = anchor_centers[..., 1] - distances[..., 1]
    x2 = anchor_centers[..., 0] + distances[..., 2]
    y2 = anchor_centers[..., 1] + distances[..., 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


# ---------------------------------------------------------------------------
# Per-class decode: raw y_pred → (boxes, scores, classes) in pixel coords
# ---------------------------------------------------------------------------


def decode_predictions(
    y_pred: np.ndarray,
    anchors: np.ndarray,
    strides: np.ndarray,
    reg_max: int,
    num_classes: int,
    score_threshold: float = 0.01,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode a batch of raw YOLOv12 predictions to per-image detections.

    :param y_pred: ``(B, A, 4*reg_max + num_classes)`` raw logits.
    :param anchors: ``(A, 2)`` anchor centers in pixel coords.
    :param strides: ``(A, 1)`` per-anchor stride.
    :param reg_max: DFL bin count.
    :param num_classes: Number of classes.
    :param score_threshold: Filter candidates with max class prob below this
        threshold.  Default 0.01.  Applied **before** NMS.
    :returns: List of ``(boxes, scores, classes)`` tuples per image — arrays of
        shape ``(M, 4)``, ``(M,)``, ``(M,)`` respectively; ``M`` varies per image.
        Each class is expanded separately so the same box can appear under
        multiple class rows if multi-label-style soft scores are above threshold.
    """
    B, A, D = y_pred.shape
    expected_d = 4 * reg_max + num_classes
    if D != expected_d:
        raise ValueError(
            f"y_pred last dim {D} != expected 4*reg_max + num_classes = {expected_d}"
        )

    reg_logits = y_pred[..., : 4 * reg_max]
    cls_logits = y_pred[..., 4 * reg_max:]
    # sigmoid — multi-label style, matches the training loss convention.
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))

    # DFL → distances in stride units, then scale to pixels by per-anchor stride.
    dist_units = decode_dfl_logits(reg_logits, reg_max)  # (B, A, 4)
    dist_px = dist_units * strides  # broadcast (A, 1) → (B, A, 4)
    boxes_px = dist_to_xyxy(dist_px, anchors)  # (B, A, 4)

    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for b in range(B):
        # For each anchor, take the best class (single-label emission, standard
        # COCOeval convention — one detection per anchor per image).
        scores_all = cls_probs[b]  # (A, K)
        cls_idx = np.argmax(scores_all, axis=-1)  # (A,)
        best_scores = scores_all[np.arange(A), cls_idx]  # (A,)
        keep = best_scores >= score_threshold
        out.append((
            boxes_px[b][keep].astype(np.float32),
            best_scores[keep].astype(np.float32),
            cls_idx[keep].astype(np.int32),
        ))
    return out


# ---------------------------------------------------------------------------
# Per-class NMS (eager numpy — for val-end callback; D-009)
# ---------------------------------------------------------------------------


def _pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ``(M, 4)`` xyxy boxes ``a`` and ``(N, 4)`` xyxy boxes ``b`` → ``(M, N)``."""
    x11, y11, x12, y12 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    x21, y21, x22, y22 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xa = np.maximum(x11, x21)
    ya = np.maximum(y11, y21)
    xb = np.minimum(x12, x22)
    yb = np.minimum(y12, y22)
    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area_a = np.clip(x12 - x11, 0, None) * np.clip(y12 - y11, 0, None)
    area_b = np.clip(x22 - x21, 0, None) * np.clip(y22 - y21, 0, None)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.45,
    max_per_class: int = 100,
    max_total: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class greedy NMS.

    Each class is NMS-ed independently, then the top ``max_total`` detections
    across all classes are kept.  Matches the convention expected by
    ``pycocotools.cocoeval.COCOeval(bbox)``.

    :param boxes: ``(M, 4)`` xyxy.
    :param scores: ``(M,)`` confidence.
    :param classes: ``(M,)`` int class ids.
    :param iou_threshold: IoU threshold above which boxes are suppressed.
    :param max_per_class: Max kept per class before the global top-k cut.
    :param max_total: Max total detections after merging classes.
    :returns: ``(kept_boxes, kept_scores, kept_classes)``.
    """
    if boxes.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_classes: List[np.ndarray] = []

    for c in np.unique(classes):
        mask = classes == c
        cb = boxes[mask]
        cs = scores[mask]
        order = np.argsort(-cs)  # desc, indices into cb/cs
        remaining = order.tolist()
        selected: List[int] = []

        while remaining and len(selected) < max_per_class:
            i = remaining[0]
            selected.append(i)
            remaining = remaining[1:]
            if not remaining:
                break
            cur_box = cb[i:i + 1]
            other_boxes = cb[remaining]
            ious = _pairwise_iou(cur_box, other_boxes)[0]
            remaining = [
                j for j, iou in zip(remaining, ious) if iou <= iou_threshold
            ]

        if selected:
            all_boxes.append(cb[selected])
            all_scores.append(cs[selected])
            all_classes.append(np.full(len(selected), int(c), dtype=np.int32))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    merged_boxes = np.concatenate(all_boxes, axis=0)
    merged_scores = np.concatenate(all_scores, axis=0)
    merged_classes = np.concatenate(all_classes, axis=0)

    # Global top-k cut
    if merged_scores.size > max_total:
        top = np.argsort(-merged_scores)[:max_total]
        merged_boxes = merged_boxes[top]
        merged_scores = merged_scores[top]
        merged_classes = merged_classes[top]

    return (
        merged_boxes.astype(np.float32),
        merged_scores.astype(np.float32),
        merged_classes.astype(np.int32),
    )
