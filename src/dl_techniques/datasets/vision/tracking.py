"""Tracking pair pipeline: COCO-sourced and synthetic (exemplar, search) pairs.

This module turns "an image plus a target box" into the training examples
the Siamese trackers consume: a small exemplar crop and a larger search crop,
both centered on the target with the SiamFC context rule, plus the ground
truth box expressed in search-crop pixels. Two image sources feed the same
geometry:

- :func:`coco_tracking_generator` -- real pairs from COCO 2017 via TFDS
  (``coco/2017`` raw examples: ``image`` plus ``objects/bbox`` in
  ``[ymin, xmin, ymax, xmax]`` normalized coordinates, the same contract
  ``coco.py`` reads);
- :func:`synthetic_tracking_generator` -- seeded noise backgrounds with a
  random colored-rectangle target (plus unannotated distractor rectangles),
  for tests, smoke runs and offline development.

Tracker-specific targets are built on top of the pairs, not inside the
sources: :func:`build_siamfc_example` packs the radius-based logistic label
(:func:`create_siamfc_label` in ``losses/siamese_tracking_loss.py``) and
:func:`build_rpn_example` matches anchors (:func:`match_rpn_targets`) and
packs the cls/reg targets the ``DaSiamRPN`` losses consume. Anchors are
passed in as an array so this module never imports ``models/``.

Only NumPy and SciPy run at module scope (the ``dtsprompt.py`` /
``synthetic_warp.py`` rule); TensorFlow and TFDS are imported lazily inside
the two functions that need them.
"""

import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from scipy.ndimage import zoom as _scipy_zoom
except ImportError:  # pragma: no cover
    _scipy_zoom = None

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

#: Context rule from the SiamFC paper: exemplar side ``sqrt((w + c*(w+h)) *
#: (h + c*(w+h)))`` with ``c = 0.5``, i.e. ``p = (w+h)/4`` padding per side.
CONTEXT_AMOUNT = 0.5

#: Boxes with a side below this (original-image pixels) are skipped: the
#: context square would be dominated by padding.
MIN_BOX_SIDE_PX = 8.0


# ---------------------------------------------------------------------
# pure geometry (NumPy)
# ---------------------------------------------------------------------


def exemplar_side_for_box(width: float, height: float, context: float = CONTEXT_AMOUNT) -> float:
    """Context-square side for a target box.

    :param width: Box width in pixels.
    :type width: float
    :param height: Box height in pixels.
    :type height: float
    :param context: Context amount, 0.5 reproduces the paper.
    :type context: float
    :return: Square side in pixels.
    :rtype: float
    :raises ValueError: If the box has non-positive extent.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"box must have positive extent, got ({width}, {height})")
    if context < 0:
        raise ValueError(f"context must be non-negative, got {context}")
    return float(np.sqrt((width + context * (width + height)) * (height + context * (width + height))))


def mean_pad_crop(image: np.ndarray, center_x: float, center_y: float, side: int) -> np.ndarray:
    """Square crop centered at a (possibly out-of-frame) point, mean-padded.

    Out-of-frame pixels are filled with the per-channel image mean (the
    paper's padding convention), never zeros.

    :param image: Array ``(H, W, C)`` float.
    :type image: numpy.ndarray
    :param center_x: Crop center x in pixels.
    :type center_x: float
    :param center_y: Crop center y in pixels.
    :type center_y: float
    :param side: Square side in pixels, positive.
    :type side: int
    :return: Array ``(side, side, C)``.
    :rtype: numpy.ndarray
    """
    if side <= 0:
        raise ValueError(f"side must be positive, got {side}")
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    frame = image.reshape(height, width, channels)
    mean_color = frame.mean(axis=(0, 1), keepdims=True)
    canvas = np.repeat(mean_color, side * side, axis=0).reshape(side, side, channels)
    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    src_x1, src_y1 = max(x1, 0), max(y1, 0)
    src_x2, src_y2 = min(x1 + side, width), min(y1 + side, height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[src_y1 - y1 : src_y2 - y1, src_x1 - x1 : src_x2 - x1, :] = frame[
            src_y1:src_y2, src_x1:src_x2, :
        ]
    return canvas


def resize_crop(crop: np.ndarray, out_size: int) -> np.ndarray:
    """Bilinear resize of a square crop via SciPy (no cv2/skimage dependency).

    :param crop: Array ``(S, S, C)``.
    :type crop: numpy.ndarray
    :param out_size: Output square extent.
    :type out_size: int
    :return: Array ``(out_size, out_size, C)`` float32.
    :rtype: numpy.ndarray
    """
    if _scipy_zoom is None:  # pragma: no cover
        raise ImportError("scipy is required for resize_crop")
    side = crop.shape[0]
    if side == out_size:
        return np.asarray(crop, dtype=np.float32)
    factor = out_size / float(side)
    return _scipy_zoom(crop, (factor, factor, 1.0), order=1).astype(np.float32)


def crop_pair(
    image: np.ndarray,
    box_cxcywh: np.ndarray,
    exemplar_size: int,
    search_size: int,
    context: float = CONTEXT_AMOUNT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop an exemplar/search pair centered on a target box.

    The exemplar covers the context square resized to ``exemplar_size``; the
    search covers the same physical square scaled by
    ``search_size / exemplar_size`` (so the target scale matches across the
    pair), resized to ``search_size``.

    :param image: Array ``(H, W, 3)`` float in [0, 1].
    :type image: numpy.ndarray
    :param box_cxcywh: Target ``(cx, cy, w, h)`` in original pixels.
    :type box_cxcywh: numpy.ndarray
    :param exemplar_size: Exemplar output extent.
    :type exemplar_size: int
    :param search_size: Search output extent.
    :type search_size: int
    :param context: Context amount.
    :type context: float
    :return: ``(z, x, gt_search)`` with ``z`` ``(E, E, 3)``, ``x``
        ``(S, S, 3)``, ``gt_search`` ``(cx, cy, w, h)`` in search pixels in
        the anchor frame (search-crop center at the origin, so a centered
        target decodes to ``(0, 0, w_s, h_s)``).
    :rtype: tuple
    """
    cx, cy, w, h = (float(v) for v in box_cxcywh)
    side_z = exemplar_side_for_box(w, h, context)
    side_x = side_z * search_size / float(exemplar_size)
    z = resize_crop(mean_pad_crop(image, cx, cy, int(round(side_z))), exemplar_size)
    x = resize_crop(mean_pad_crop(image, cx, cy, int(round(side_x))), search_size)
    scale = search_size / side_x
    gt_search = np.array([0.0, 0.0, w * scale, h * scale], dtype=np.float32)
    return z.astype(np.float32), x.astype(np.float32), gt_search


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(cx, cy, w, h)`` box sets.

    :param boxes_a: Array ``(N, 4)``.
    :type boxes_a: numpy.ndarray
    :param boxes_b: Array ``(M, 4)``.
    :type boxes_b: numpy.ndarray
    :return: Array ``(N, M)``.
    :rtype: numpy.ndarray
    """
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    a_x1, a_y1 = a[:, 0] - a[:, 2] / 2.0, a[:, 1] - a[:, 3] / 2.0
    a_x2, a_y2 = a[:, 0] + a[:, 2] / 2.0, a[:, 1] + a[:, 3] / 2.0
    b_x1, b_y1 = b[:, 0] - b[:, 2] / 2.0, b[:, 1] - b[:, 3] / 2.0
    b_x2, b_y2 = b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0
    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])
    inter = np.maximum(inter_x2 - inter_x1, 0.0) * np.maximum(inter_y2 - inter_y1, 0.0)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def match_rpn_targets(
    anchors: np.ndarray,
    gt_box: np.ndarray,
    pos_iou: float = 0.6,
    neg_iou: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match anchors against one ground-truth box (SiamRPN convention).

    Both inputs live in the anchor frame (search-crop center at the origin):
    ``anchors`` from :func:`generate_dasiamrpn_anchors` and ``gt_box`` as
    returned by :func:`crop_pair`.

    Anchors with ``IoU >= pos_iou`` are positive, ``IoU <= neg_iou`` negative,
    the band between is ignored. Regression deltas use the reference
    parameterization ``dx = (gx-ax)/aw, dy = (gy-ah)/ah, dw = log(gw/aw),
    dh = log(gh/ah)``, which :func:`decode_dasiamrpn_boxes` inverts exactly.

    :param anchors: Array ``(N, 4)`` in ``(cx, cy, w, h)`` order.
    :type anchors: numpy.ndarray
    :param gt_box: Array ``(4,)`` in ``(cx, cy, w, h)`` order.
    :type gt_box: numpy.ndarray
    :param pos_iou: Positive IoU threshold.
    :type pos_iou: float
    :param neg_iou: Negative IoU threshold.
    :type neg_iou: float
    :return: ``(cls_labels, cls_weights, reg_deltas, reg_weights)`` with
        labels in ``{-1, 0, 1}`` (``-1`` ignored), deltas ``(N, 4)``.
    :rtype: tuple
    :raises ValueError: If the thresholds are unordered.
    """
    if not 0.0 <= neg_iou < pos_iou <= 1.0:
        raise ValueError(f"need 0 <= neg ({neg_iou}) < pos ({pos_iou}) <= 1")
    anchors = np.asarray(anchors, dtype=np.float64)
    gt = np.asarray(gt_box, dtype=np.float64).reshape(4)
    ious = box_iou(anchors, gt.reshape(1, 4))[:, 0]
    cls_labels = np.full(anchors.shape[0], -1, dtype=np.int32)
    cls_labels[ious >= pos_iou] = 1
    cls_labels[ious <= neg_iou] = 0
    cls_weights = (cls_labels >= 0).astype(np.float32)
    reg_weights = (cls_labels == 1).astype(np.float32)
    deltas = np.zeros((anchors.shape[0], 4), dtype=np.float32)
    deltas[:, 0] = (gt[0] - anchors[:, 0]) / np.maximum(anchors[:, 2], 1e-6)
    deltas[:, 1] = (gt[1] - anchors[:, 1]) / np.maximum(anchors[:, 3], 1e-6)
    deltas[:, 2] = np.log(np.maximum(gt[2], 1e-6) / np.maximum(anchors[:, 2], 1e-6))
    deltas[:, 3] = np.log(np.maximum(gt[3], 1e-6) / np.maximum(anchors[:, 3], 1e-6))
    return cls_labels, cls_weights, deltas.astype(np.float32), reg_weights


def flat_to_grid(flat: np.ndarray, anchor_num: int, score_size: int) -> np.ndarray:
    """Reshape anchor-major ``(A*S*S, ...)`` targets to ``(S, S, A, ...)``.

    Inverts the reference anchor layout (row ``a*S*S + i*S + j``), aligning
    flat matching outputs with the model output layout ``(B, S, S, A, ...)``.

    :param flat: Array ``(A*S*S, ...)``.
    :type flat: numpy.ndarray
    :param anchor_num: Anchors per position.
    :type anchor_num: int
    :param score_size: Score-grid extent.
    :type score_size: int
    :return: Array ``(S, S, A, ...)``.
    :rtype: numpy.ndarray
    :raises ValueError: If the leading extent does not factor.
    """
    flat = np.asarray(flat)
    if flat.shape[0] != anchor_num * score_size * score_size:
        raise ValueError(
            f"leading extent {flat.shape[0]} != A*S*S "
            f"({anchor_num}*{score_size}*{score_size})"
        )
    reshaped = flat.reshape(anchor_num, score_size, score_size, *flat.shape[1:])
    axes = (1, 2, 0) + tuple(range(3, reshaped.ndim))
    return reshaped.transpose(axes)


# ---------------------------------------------------------------------
# photometric augmentation (NumPy, training-time)
# ---------------------------------------------------------------------


def photometric_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_delta: float = 0.125,
    contrast_range: Tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Brightness/contrast jitter on a [0, 1] float crop.

    :param image: Array ``(H, W, 3)`` in [0, 1].
    :type image: numpy.ndarray
    :param rng: Seeded generator.
    :type rng: numpy.random.Generator
    :param brightness_delta: Uniform brightness shift bound.
    :type brightness_delta: float
    :param contrast_range: Contrast multiplier range.
    :type contrast_range: tuple
    :return: Jittered array clipped to [0, 1].
    :rtype: numpy.ndarray
    """
    out = image.astype(np.float32)
    if brightness_delta > 0:
        out = out + rng.uniform(-brightness_delta, brightness_delta)
    lo, hi = contrast_range
    if hi > lo:
        out = (out - 0.5) * rng.uniform(lo, hi) + 0.5
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# image sources (Python iterators yielding (image, box) pairs)
# ---------------------------------------------------------------------


def synthetic_tracking_generator(
    num_samples: int,
    image_size: int = 512,
    min_box_side: int = 48,
    max_box_side: int = 160,
    seed: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Seeded synthetic tracking pairs: noise background, rectangle target.

    Each sample draws a flat-noise background, one solid-color target
    rectangle and two unannotated distractor rectangles (teaching the
    correlation to discriminate, cheaply). Yields ``(image, box)`` with the
    image ``(image_size, image_size, 3)`` in [0, 1] and the box
    ``(cx, cy, w, h)`` in absolute pixels.

    :param num_samples: Samples to yield.
    :type num_samples: int
    :param image_size: Square image extent.
    :type image_size: int
    :param min_box_side: Minimum target side.
    :type min_box_side: int
    :param max_box_side: Maximum target side.
    :type max_box_side: int
    :param seed: Seed.
    :type seed: int
    :return: Iterator of ``(image, box)`` tuples.
    :rtype: iterator
    """
    rng = np.random.default_rng(seed)
    for _ in range(num_samples):
        image = (rng.random((image_size, image_size, 3), dtype=np.float32) * 0.4).astype(
            np.float32
        )
        boxes = []
        for is_target in (True, False, False):
            w = float(rng.integers(min_box_side, max_box_side + 1))
            h = float(rng.integers(min_box_side, max_box_side + 1))
            cx = float(rng.uniform(w / 2.0, image_size - w / 2.0))
            cy = float(rng.uniform(h / 2.0, image_size - h / 2.0))
            color = rng.random(3, dtype=np.float32) * 0.6 + 0.4
            x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
            image[y1:y2, x1:x2, :] = color
            if is_target:
                boxes.append(np.array([cx, cy, w, h], dtype=np.float32))
        yield image, boxes[0]


def coco_tracking_generator(
    data_dir: Optional[str] = None,
    split: str = "train",
    min_box_side_px: float = MIN_BOX_SIDE_PX,
    seed: int = 0,
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Real tracking pairs from COCO 2017 via TFDS.

    Iterates raw ``coco/2017`` examples (the same source ``coco.py``
    reads), keeps images with at least one box whose sides clear
    ``min_box_side_px``, and yields ``(image, box)`` with the image in
    [0, 1] float32 and one uniformly sampled box as ``(cx, cy, w, h)`` in
    absolute pixels. Requires the TFDS COCO download (or ``data_dir``
    pointing at a prepared copy); raises at iteration time otherwise.

    :param data_dir: TFDS data directory (None for the default cache).
    :type data_dir: str or None
    :param split: TFDS split (``"train"`` or ``"validation"``).
    :type split: str
    :param min_box_side_px: Minimum box side in pixels.
    :type min_box_side_px: float
    :param seed: Seed for the per-image box choice.
    :type seed: int
    :param max_samples: Stop after this many yields (None for unbounded).
    :type max_samples: int or None
    :return: Iterator of ``(image, box)`` tuples.
    :rtype: iterator
    """
    import tensorflow_datasets as tfds

    rng = np.random.default_rng(seed)
    kwargs: Dict[str, Any] = {"split": split, "as_supervised": False}
    if data_dir:
        kwargs["data_dir"] = data_dir
    ds = tfds.load("coco/2017", **kwargs)
    yielded = 0
    for example in ds:
        image = example["image"].numpy().astype(np.float32) / 255.0
        height, width = image.shape[:2]
        bboxes = example["objects"]["bbox"].numpy()  # ymin,xmin,ymax,xmax normalized
        keep: List[np.ndarray] = []
        for ymin, xmin, ymax, xmax in bboxes:
            w = (xmax - xmin) * width
            h = (ymax - ymin) * height
            if w >= min_box_side_px and h >= min_box_side_px:
                keep.append(
                    np.array(
                        [
                            (xmin + xmax) / 2.0 * width,
                            (ymin + ymax) / 2.0 * height,
                            w,
                            h,
                        ],
                        dtype=np.float32,
                    )
                )
        if not keep:
            continue
        yield image, keep[int(rng.integers(len(keep)))]
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            return


# ---------------------------------------------------------------------
# tracker-specific example builders (NumPy, composed over the sources)
# ---------------------------------------------------------------------


def build_siamfc_example(
    image: np.ndarray,
    box_cxcywh: np.ndarray,
    score_size: int,
    exemplar_size: int = 127,
    search_size: int = 255,
    pos_radius_px: float = 25.0,
    neg_radius_px: float = 50.0,
    total_stride: int = 8,
    context: float = CONTEXT_AMOUNT,
    brightness_delta: float = 0.125,
    rng: Optional[np.random.Generator] = None,
    augment: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Build one SiamFC training example from an image plus a target box.

    Crops the centered pair, optionally jitters photometry (geometry stays
    centered so the centered label stays valid), and packs the radius-based
    label :class:`SiamFCLogisticLoss` consumes.

    :param image: Array ``(H, W, 3)`` in [0, 1].
    :type image: numpy.ndarray
    :param box_cxcywh: Target ``(cx, cy, w, h)`` in original pixels.
    :type box_cxcywh: numpy.ndarray
    :param score_size: Score-map extent for the label.
    :type score_size: int
    :param exemplar_size: Exemplar output extent.
    :type exemplar_size: int
    :param search_size: Search output extent.
    :type search_size: int
    :param pos_radius_px: Positive radius in search pixels.
    :type pos_radius_px: float
    :param neg_radius_px: Negative radius in search pixels.
    :type neg_radius_px: float
    :param total_stride: Network total stride.
    :type total_stride: int
    :param context: Context amount.
    :type context: float
    :param brightness_delta: Photometric brightness bound forwarded to jitter.
    :type brightness_delta: float
    :param rng: Seeded generator (required when ``augment`` is True).
    :type rng: numpy.random.Generator or None
    :param augment: Whether to apply photometric jitter.
    :type augment: bool
    :return: ``((z, x), packed_label)`` with label ``(S, S, 2)``.
    :rtype: tuple
    """
    from dl_techniques.losses.siamese_tracking_loss import create_siamfc_label

    z, x, _ = crop_pair(image, box_cxcywh, exemplar_size, search_size, context)
    if augment:
        if rng is None:
            raise ValueError("rng is required when augment is True")
        z = photometric_jitter(z, rng, brightness_delta=brightness_delta)
        x = photometric_jitter(x, rng, brightness_delta=brightness_delta)
    label = create_siamfc_label(score_size, pos_radius_px, neg_radius_px, total_stride)
    return (z, x), label


def build_rpn_example(
    image: np.ndarray,
    box_cxcywh: np.ndarray,
    anchors: np.ndarray,
    anchor_num: int,
    score_size: int,
    exemplar_size: int = 127,
    search_size: int = 271,
    pos_iou: float = 0.6,
    neg_iou: float = 0.3,
    context: float = CONTEXT_AMOUNT,
    brightness_delta: float = 0.125,
    rng: Optional[np.random.Generator] = None,
    augment: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    """Build one DaSiamRPN training example from an image plus a target box.

    Crops the centered pair, matches ``anchors`` (anchor-major
    ``(A*S*S, 4)``) against the centered ground-truth box, and packs the
    grid-aligned cls/reg targets the ``DaSiamRPN`` losses consume. The
    ground-truth box sits at the search-crop center; the anchor grid center
    is half a stride off that (``-stride/2`` for odd score sizes), a
    sub-pixel offset IoU matching absorbs without effect.

    :param image: Array ``(H, W, 3)`` in [0, 1].
    :type image: numpy.ndarray
    :param box_cxcywh: Target ``(cx, cy, w, h)`` in original pixels.
    :type box_cxcywh: numpy.ndarray
    :param anchors: Array ``(A*S*S, 4)`` in ``(cx, cy, w, h)`` order.
    :type anchors: numpy.ndarray
    :param anchor_num: Anchors per position.
    :type anchor_num: int
    :param score_size: Score-grid extent.
    :type score_size: int
    :param exemplar_size: Exemplar output extent.
    :type exemplar_size: int
    :param search_size: Search output extent.
    :type search_size: int
    :param pos_iou: Positive IoU threshold.
    :type pos_iou: float
    :param neg_iou: Negative IoU threshold.
    :type neg_iou: float
    :param context: Context amount.
    :type context: float
    :param brightness_delta: Photometric brightness bound forwarded to jitter.
    :type brightness_delta: float
    :param rng: Seeded generator (required when ``augment`` is True).
    :type rng: numpy.random.Generator or None
    :param augment: Whether to apply photometric jitter.
    :type augment: bool
    :return: ``((z, x), {"cls": (S, S, A, 2), "reg": (S, S, A, 5)})``.
    :rtype: tuple
    """
    z, x, gt_search = crop_pair(image, box_cxcywh, exemplar_size, search_size, context)
    if augment:
        if rng is None:
            raise ValueError("rng is required when augment is True")
        z = photometric_jitter(z, rng, brightness_delta=brightness_delta)
        x = photometric_jitter(x, rng, brightness_delta=brightness_delta)
    cls_labels, cls_weights, reg_deltas, reg_weights = match_rpn_targets(
        anchors, gt_search, pos_iou, neg_iou
    )
    cls_grid = flat_to_grid(
        np.stack(
            [cls_labels.astype(np.float32), cls_weights], axis=-1
        ).reshape(-1, 2),
        anchor_num,
        score_size,
    )
    reg_grid = flat_to_grid(
        np.concatenate(
            [reg_deltas, reg_weights[:, None]], axis=-1
        ).reshape(-1, 5),
        anchor_num,
        score_size,
    )
    return (z, x), {"cls": cls_grid.astype(np.float32), "reg": reg_grid.astype(np.float32)}


# ---------------------------------------------------------------------
# MambaLCT clip builder (NumPy, composed over the sources above)
# ---------------------------------------------------------------------


def build_mambalct_clip_example(
    image: np.ndarray,
    box_cxcywh: np.ndarray,
    clip_length: int = 2,
    template_size: int = 128,
    search_size: int = 256,
    max_shift_ratio: float = 0.1,
    context: float = CONTEXT_AMOUNT,
    brightness_delta: float = 0.125,
    rng: Optional[np.random.Generator] = None,
    augment: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    """Build one MambaLCT training clip from an image plus a target box.

    The template is a centered crop of the target; each of the
    ``clip_length`` search frames is cropped around a jittered center (a
    still-image pseudo-motion standing in for real video when the source is
    COCO or synthetic — see the trainer README). Boxes are reported per
    frame in normalized search-crop ``(cx, cy, w, h)`` coordinates and the
    score is 1.0 when the jittered box center stays inside the crop.

    :param image: Array ``(H, W, 3)`` in [0, 1].
    :type image: numpy.ndarray
    :param box_cxcywh: Target ``(cx, cy, w, h)`` in original pixels.
    :type box_cxcywh: numpy.ndarray
    :param clip_length: Search frames in the clip, positive.
    :type clip_length: int
    :param template_size: Template output extent.
    :type template_size: int
    :param search_size: Search output extent.
    :type search_size: int
    :param max_shift_ratio: Uniform center-jitter bound as a fraction of the
        search-crop physical side.
    :type max_shift_ratio: float
    :param context: Context amount forwarded to :func:`crop_pair`.
    :type context: float
    :param brightness_delta: Photometric brightness bound forwarded to jitter.
    :type brightness_delta: float
    :param rng: Seeded generator (required when ``augment`` is True).
    :type rng: numpy.random.Generator or None
    :param augment: Whether to apply center jitter and photometric jitter.
    :type augment: bool
    :return: ``((template, search_clip), {"scores": (T, 1), "boxes": (T, 4)})``
        with template ``(E, E, 3)`` and search clip ``(T, S, S, 3)``.
    :rtype: tuple
    :raises ValueError: If ``clip_length`` is not positive or ``rng`` is
        missing when ``augment`` is True.
    """
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}")
    if max_shift_ratio < 0:
        raise ValueError(
            f"max_shift_ratio must be non-negative, got {max_shift_ratio}"
        )
    if augment and rng is None:
        raise ValueError("rng is required when augment is True")

    cx, cy, w, h = (float(v) for v in box_cxcywh)
    side_z = exemplar_side_for_box(w, h, context)
    side_x = side_z * search_size / float(template_size)
    template = resize_crop(
        mean_pad_crop(image, cx, cy, int(round(side_z))), template_size
    )
    if augment:
        assert rng is not None
        template = photometric_jitter(
            template, rng, brightness_delta=brightness_delta
        )

    frames = []
    boxes = []
    scores = []
    for _ in range(clip_length):
        if augment:
            assert rng is not None
            shift = rng.uniform(-max_shift_ratio, max_shift_ratio, size=2)
            crop_cx = cx + shift[0] * side_x
            crop_cy = cy + shift[1] * side_x
        else:
            crop_cx, crop_cy = cx, cy
        frame = resize_crop(
            mean_pad_crop(image, crop_cx, crop_cy, int(round(side_x))),
            search_size,
        )
        if augment:
            assert rng is not None
            frame = photometric_jitter(
                frame, rng, brightness_delta=brightness_delta
            )
        frames.append(frame.astype(np.float32))
        boxes.append(
            np.array(
                [
                    (cx - crop_cx) / side_x + 0.5,
                    (cy - crop_cy) / side_x + 0.5,
                    w / side_x,
                    h / side_x,
                ],
                dtype=np.float32,
            )
        )
        visible = (
            0.0 <= boxes[-1][0] <= 1.0 and 0.0 <= boxes[-1][1] <= 1.0
        )
        scores.append([1.0 if visible else 0.0])
    search_clip = np.stack(frames, axis=0)
    return (template.astype(np.float32), search_clip), {
        "scores": np.array(scores, dtype=np.float32),
        "boxes": np.stack(boxes, axis=0),
    }
