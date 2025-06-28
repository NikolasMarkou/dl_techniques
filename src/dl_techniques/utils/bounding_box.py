"""
Object Detection IoU Utility Module

This module provides utilities for calculating various Intersection over Union (IoU)
metrics commonly used in object detection tasks.
"""

import keras
from keras import ops
from typing import Union, Tuple

# ---------------------------------------------------------------------

def bbox_iou(
        box1: Union[keras.KerasTensor, "tf.Tensor"],
        box2: Union[keras.KerasTensor, "tf.Tensor"],
        xywh: bool = False,
        GIoU: bool = False,
        DIoU: bool = False,
        CIoU: bool = False,
        eps: float = 1e-7
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate IoU, GIoU, DIoU, or CIoU for batches of bounding boxes.

    This function computes various Intersection over Union metrics between two sets
    of bounding boxes. It supports standard IoU as well as advanced variants:
    - GIoU (Generalized IoU): Addresses the issue when boxes don't overlap
    - DIoU (Distance IoU): Considers the distance between box centers
    - CIoU (Complete IoU): Adds aspect ratio consistency penalty

    Args:
        box1: First set of bounding boxes with shape (..., 4).
              Format depends on `xywh` parameter.
        box2: Second set of bounding boxes with shape (..., 4).
              Format depends on `xywh` parameter.
        xywh: If True, boxes are in (center_x, center_y, width, height) format.
              If False, boxes are in (x1, y1, x2, y2) format where (x1,y1) is
              top-left corner and (x2,y2) is bottom-right corner.
        GIoU: If True, calculate Generalized IoU instead of standard IoU.
        DIoU: If True, calculate Distance IoU instead of standard IoU.
        CIoU: If True, calculate Complete IoU instead of standard IoU.
        eps: Small epsilon value to prevent division by zero and ensure
             numerical stability. Default: 1e-7.

    Returns:
        IoU values with shape matching the broadcast shape of input boxes
        but with the last dimension removed. Values are in range [-1, 1]
        for GIoU and [0, 1] for other variants.

    Raises:
        ValueError: If multiple IoU variants are selected simultaneously.

    Example:
        >>> # Standard IoU calculation
        >>> box1 = keras.random.uniform((2, 4))  # 2 boxes in xyxy format
        >>> box2 = keras.random.uniform((2, 4))
        >>> iou_scores = bbox_iou(box1, box2)
        >>> print(iou_scores.shape)  # (2,)

        >>> # CIoU with center format
        >>> box1_xywh = keras.random.uniform((3, 4))  # 3 boxes in xywh format
        >>> box2_xywh = keras.random.uniform((3, 4))
        >>> ciou_scores = bbox_iou(box1_xywh, box2_xywh, xywh=True, CIoU=True)
        >>> print(ciou_scores.shape)  # (3,)

    Note:
        Only one of GIoU, DIoU, or CIoU should be True at a time.
        If all are False, standard IoU is calculated.
    """
    # Validate that only one IoU variant is selected
    iou_variants = [GIoU, DIoU, CIoU]
    if sum(iou_variants) > 1:
        raise ValueError("Only one of GIoU, DIoU, or CIoU can be True at a time")

    # Convert boxes to corner format and extract coordinates
    corner_coords = _convert_to_corner_format(box1, box2, xywh)
    b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2 = corner_coords

    # Calculate box dimensions
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Calculate intersection area
    inter_area = _calculate_intersection_area(
        b1_x1, b1_y1, b1_x2, b1_y2,
        b2_x1, b2_y1, b2_x2, b2_y2
    )

    # Calculate union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area + eps

    # Calculate basic IoU
    iou = ops.squeeze(inter_area / union_area, axis=-1)

    # Apply advanced IoU variants if requested
    if CIoU:
        return _calculate_ciou(
            iou, b1_x1, b1_y1, b1_x2, b1_y2,
            b2_x1, b2_y1, b2_x2, b2_y2, w1, h1, w2, h2, eps
        )
    elif DIoU:
        return _calculate_diou(
            iou, b1_x1, b1_y1, b1_x2, b1_y2,
            b2_x1, b2_y1, b2_x2, b2_y2, eps
        )
    elif GIoU:
        return _calculate_giou(
            iou, b1_x1, b1_y1, b1_x2, b1_y2,
            b2_x1, b2_y1, b2_x2, b2_y2, union_area, eps
        )

    return iou

# ---------------------------------------------------------------------

def _convert_to_corner_format(
        box1: Union[keras.KerasTensor, "tf.Tensor"],
        box2: Union[keras.KerasTensor, "tf.Tensor"],
        xywh: bool
) -> Tuple[Union[keras.KerasTensor, "tf.Tensor"], ...]:
    """
    Convert bounding boxes to corner format (x1, y1, x2, y2).

    Args:
        box1: First set of bounding boxes.
        box2: Second set of bounding boxes.
        xywh: If True, input boxes are in center format.

    Returns:
        Tuple of 8 tensors: (b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2)
    """
    if xywh:
        # Convert from center format (cx, cy, w, h) to corner format
        b1_x, b1_y, b1_w, b1_h = (
            box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
        )
        b2_x, b2_y, b2_w, b2_h = (
            box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]
        )

        # Calculate corner coordinates
        b1_x1, b1_y1 = b1_x - b1_w / 2, b1_y - b1_h / 2
        b1_x2, b1_y2 = b1_x + b1_w / 2, b1_y + b1_h / 2
        b2_x1, b2_y1 = b2_x - b2_w / 2, b2_y - b2_h / 2
        b2_x2, b2_y2 = b2_x + b2_w / 2, b2_y + b2_h / 2
    else:
        # Boxes are already in corner format
        b1_x1, b1_y1 = box1[..., 0:1], box1[..., 1:2]
        b1_x2, b1_y2 = box1[..., 2:3], box1[..., 3:4]
        b2_x1, b2_y1 = box2[..., 0:1], box2[..., 1:2]
        b2_x2, b2_y2 = box2[..., 2:3], box2[..., 3:4]

    return b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2

# ---------------------------------------------------------------------

def _calculate_intersection_area(
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"]
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate the intersection area between two sets of bounding boxes.

    Args:
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.

    Returns:
        Intersection area tensor.
    """
    # Find intersection rectangle coordinates
    inter_x1 = ops.maximum(b1_x1, b2_x1)
    inter_y1 = ops.maximum(b1_y1, b2_y1)
    inter_x2 = ops.minimum(b1_x2, b2_x2)
    inter_y2 = ops.minimum(b1_y2, b2_y2)

    # Calculate intersection dimensions (clamp to 0 if no overlap)
    inter_w = ops.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = ops.maximum(inter_y2 - inter_y1, 0.0)

    return inter_w * inter_h

# ---------------------------------------------------------------------

def _calculate_enclosing_box_diagonal(
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"],
        eps: float = 1e-7
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate the diagonal distance of the smallest enclosing box.

    Args:
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.
        eps: Small epsilon for numerical stability.

    Returns:
        Squared diagonal distance of enclosing box.
    """
    # Find enclosing box coordinates
    c_x1 = ops.minimum(b1_x1, b2_x1)
    c_y1 = ops.minimum(b1_y1, b2_y1)
    c_x2 = ops.maximum(b1_x2, b2_x2)
    c_y2 = ops.maximum(b1_y2, b2_y2)

    # Calculate squared diagonal distance
    c2 = ops.squeeze((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps, axis=-1)
    return c2

# ---------------------------------------------------------------------

def _calculate_center_distance(
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"]
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate the squared distance between box centers.

    Args:
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.

    Returns:
        Squared distance between box centers.
    """
    # Calculate box centers
    b1_center_x = (b1_x1 + b1_x2) / 2
    b1_center_y = (b1_y1 + b1_y2) / 2
    b2_center_x = (b2_x1 + b2_x2) / 2
    b2_center_y = (b2_y1 + b2_y2) / 2

    # Calculate squared center distance
    rho2 = ops.squeeze(
        (b2_center_x - b1_center_x) ** 2 + (b2_center_y - b1_center_y) ** 2,
        axis=-1
    )
    return rho2

# ---------------------------------------------------------------------

def _calculate_giou(
        iou: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"],
        union_area: Union[keras.KerasTensor, "tf.Tensor"],
        eps: float = 1e-7
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate Generalized IoU (GIoU).

    GIoU addresses the issue when boxes don't overlap by considering
    the area of the smallest enclosing box.

    Args:
        iou: Standard IoU values.
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.
        union_area: Union area of the boxes.
        eps: Small epsilon for numerical stability.

    Returns:
        GIoU values.
    """
    # Calculate enclosing box area
    c_x1 = ops.minimum(b1_x1, b2_x1)
    c_y1 = ops.minimum(b1_y1, b2_y1)
    c_x2 = ops.maximum(b1_x2, b2_x2)
    c_y2 = ops.maximum(b1_y2, b2_y2)

    c_area = ops.squeeze((c_x2 - c_x1) * (c_y2 - c_y1) + eps, axis=-1)
    union_area_squeezed = ops.squeeze(union_area, axis=-1)

    # GIoU = IoU - (C - U) / C
    # where C is enclosing area and U is union area
    return iou - (c_area - union_area_squeezed) / c_area

# ---------------------------------------------------------------------

def _calculate_diou(
        iou: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"],
        eps: float = 1e-7
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate Distance IoU (DIoU).

    DIoU considers the distance between box centers relative to the
    diagonal of the smallest enclosing box.

    Args:
        iou: Standard IoU values.
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.
        eps: Small epsilon for numerical stability.

    Returns:
        DIoU values.
    """
    # Calculate diagonal distance of enclosing box
    c2 = _calculate_enclosing_box_diagonal(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps
    )

    # Calculate center distance
    rho2 = _calculate_center_distance(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )

    # DIoU = IoU - ρ²/c²
    # where ρ² is squared center distance and c² is squared diagonal distance
    return iou - rho2 / c2

# ---------------------------------------------------------------------

def _calculate_ciou(
        iou: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b1_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b1_y2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y1: Union[keras.KerasTensor, "tf.Tensor"],
        b2_x2: Union[keras.KerasTensor, "tf.Tensor"],
        b2_y2: Union[keras.KerasTensor, "tf.Tensor"],
        w1: Union[keras.KerasTensor, "tf.Tensor"],
        h1: Union[keras.KerasTensor, "tf.Tensor"],
        w2: Union[keras.KerasTensor, "tf.Tensor"],
        h2: Union[keras.KerasTensor, "tf.Tensor"],
        eps: float = 1e-7
) -> Union[keras.KerasTensor, "tf.Tensor"]:
    """
    Calculate Complete IoU (CIoU).

    CIoU extends DIoU by adding an aspect ratio consistency penalty term.

    Args:
        iou: Standard IoU values.
        b1_x1, b1_y1, b1_x2, b1_y2: Corner coordinates of first box set.
        b2_x1, b2_y1, b2_x2, b2_y2: Corner coordinates of second box set.
        w1, h1: Width and height of first box set.
        w2, h2: Width and height of second box set.
        eps: Small epsilon for numerical stability.

    Returns:
        CIoU values.
    """
    # Calculate DIoU components
    c2 = _calculate_enclosing_box_diagonal(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2, eps
    )
    rho2 = _calculate_center_distance(
        b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2
    )

    # Calculate aspect ratio penalty
    pi_squared = ops.convert_to_tensor(9.869604401089358, dtype=iou.dtype)  # π²

    # Squeeze dimensions for consistency
    w1_sq = ops.squeeze(w1, axis=-1)
    h1_sq = ops.squeeze(h1, axis=-1)
    w2_sq = ops.squeeze(w2, axis=-1)
    h2_sq = ops.squeeze(h2, axis=-1)

    # Calculate aspect ratio consistency term
    v = (4.0 / pi_squared) * ops.power(
        ops.arctan(w2_sq / (h2_sq + eps)) - ops.arctan(w1_sq / (h1_sq + eps)), 2
    )

    # Calculate alpha (trade-off parameter)
    alpha = v / (v - iou + (1.0 + eps))
    alpha = ops.stop_gradient(alpha)  # Don't backpropagate through alpha

    # CIoU = IoU - ρ²/c² - αv
    return iou - (rho2 / c2 + v * alpha)

# ---------------------------------------------------------------------

def bbox_nms(
        boxes: Union[keras.KerasTensor, "tf.Tensor"],
        scores: Union[keras.KerasTensor, "tf.Tensor"],
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        max_outputs: int = 100,
        xywh: bool = False
) -> Tuple[Union[keras.KerasTensor, "tf.Tensor"], ...]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    This function filters out redundant bounding boxes by suppressing
    boxes with high IoU overlap with higher-scoring boxes.

    Args:
        boxes: Bounding boxes tensor with shape (N, 4).
        scores: Confidence scores for each box with shape (N,).
        iou_threshold: IoU threshold for suppression. Boxes with IoU
                      greater than this threshold are suppressed.
        score_threshold: Minimum score threshold. Boxes with scores
                        below this threshold are filtered out.
        max_outputs: Maximum number of boxes to keep after NMS.
        xywh: If True, boxes are in (center_x, center_y, width, height) format.
              If False, boxes are in (x1, y1, x2, y2) format.

    Returns:
        Tuple containing:
        - selected_boxes: Boxes that survived NMS, shape (M, 4) where M <= max_outputs
        - selected_scores: Corresponding scores, shape (M,)
        - selected_indices: Original indices of selected boxes, shape (M,)

    Note:
        This is a simplified NMS implementation. For production use,
        consider using optimized backend-specific implementations.
    """
    # Filter by score threshold
    valid_mask = scores >= score_threshold
    valid_boxes = boxes[valid_mask]
    valid_scores = scores[valid_mask]
    valid_indices = ops.arange(ops.shape(boxes)[0])[valid_mask]

    # Sort by scores in descending order
    sorted_indices = ops.argsort(-valid_scores, axis=0)
    sorted_boxes = ops.take(valid_boxes, sorted_indices, axis=0)
    sorted_scores = ops.take(valid_scores, sorted_indices, axis=0)
    sorted_original_indices = ops.take(valid_indices, sorted_indices, axis=0)

    # Initialize selection mask
    num_boxes = ops.shape(sorted_boxes)[0]
    keep_mask = ops.ones((num_boxes,), dtype="bool")

    # Simplified NMS loop (this is a conceptual implementation)
    # In practice, you'd want to use optimized backend implementations
    selected_boxes = []
    selected_scores = []
    selected_indices = []

    for i in range(min(max_outputs, num_boxes)):
        if not keep_mask[i]:
            continue

        current_box = sorted_boxes[i:i + 1]
        current_score = sorted_scores[i]
        current_idx = sorted_original_indices[i]

        selected_boxes.append(current_box)
        selected_scores.append(current_score)
        selected_indices.append(current_idx)

        # Calculate IoU with remaining boxes
        if i < num_boxes - 1:
            remaining_boxes = sorted_boxes[i + 1:]
            ious = bbox_iou(current_box, remaining_boxes, xywh=xywh)

            # Update keep mask
            suppress_mask = ious > iou_threshold
            keep_mask = ops.concatenate([
                keep_mask[:i + 1],
                keep_mask[i + 1:] & ~suppress_mask
            ])

    if selected_boxes:
        return (
            ops.concatenate(selected_boxes, axis=0),
            ops.stack(selected_scores),
            ops.stack(selected_indices)
        )
    else:
        # Return empty tensors with correct shapes
        return (
            ops.zeros((0, 4), dtype=boxes.dtype),
            ops.zeros((0,), dtype=scores.dtype),
            ops.zeros((0,), dtype="int32")
        )

# ---------------------------------------------------------------------
