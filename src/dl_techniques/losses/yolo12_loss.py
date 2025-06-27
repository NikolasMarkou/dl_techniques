"""
YOLOv12 Loss Function Implementation - Fixed Version.

This module provides a comprehensive loss function for YOLOv12 object detection,
including Task-Aligned Assigner, Distribution Focal Loss (DFL), and Complete
Intersection over Union (CIoU) loss.

The implementation follows Keras 3 best practices and integrates seamlessly
with the YOLOv12 model architecture.

Components:
    - Task-Aligned Assigner: Assigns ground truth boxes to predictions
    - CIoU Loss: Complete Intersection over Union for bbox regression
    - DFL Loss: Distribution Focal Loss for bbox regression
    - BCE Loss: Binary Cross Entropy for classification
    - Anchor generation and bbox decoding utilities

References:
    - YOLOv12: Real-Time Object Detection with Enhanced Architecture
    - Task-Aligned One-stage Object Detection (TOOD)

File: src/dl_techniques/losses/yolo12_loss.py
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from dl_techniques.utils.logger import logger


def bbox_iou(
    box1: keras.KerasTensor,
    box2: keras.KerasTensor,
    xywh: bool = False,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7
) -> keras.KerasTensor:
    """Calculate IoU, GIoU, DIoU, or CIoU for batches of bounding boxes.

    Args:
        box1: First set of boxes, shape (..., 4).
        box2: Second set of boxes, shape (..., 4).
        xywh: If True, boxes are in (cx, cy, w, h) format, else (x1, y1, x2, y2).
        GIoU: If True, calculate Generalized IoU.
        DIoU: If True, calculate Distance IoU.
        CIoU: If True, calculate Complete IoU.
        eps: Small value to avoid division by zero.

    Returns:
        IoU values, shape matching the broadcast shape of input boxes without the last dimension.
    """
    # Get the box coordinates
    if xywh:
        # Convert from center format to corner format
        b1_x, b1_y, b1_w, b1_h = box1[..., 0:1], box1[..., 1:2], box1[..., 2:3], box1[..., 3:4]
        b2_x, b2_y, b2_w, b2_h = box2[..., 0:1], box2[..., 1:2], box2[..., 2:3], box2[..., 3:4]

        b1_x1, b1_y1 = b1_x - b1_w / 2, b1_y - b1_h / 2
        b1_x2, b1_y2 = b1_x + b1_w / 2, b1_y + b1_h / 2
        b2_x1, b2_y1 = b2_x - b2_w / 2, b2_y - b2_h / 2
        b2_x2, b2_y2 = b2_x + b2_w / 2, b2_y + b2_h / 2

        w1, h1 = b1_w, b1_h
        w2, h2 = b2_w, b2_h
    else:
        # Boxes are already in corner format
        b1_x1, b1_y1 = box1[..., 0:1], box1[..., 1:2]
        b1_x2, b1_y2 = box1[..., 2:3], box1[..., 3:4]
        b2_x1, b2_y1 = box2[..., 0:1], box2[..., 1:2]
        b2_x2, b2_y2 = box2[..., 2:3], box2[..., 3:4]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Calculate intersection
    inter_x1 = ops.maximum(b1_x1, b2_x1)
    inter_y1 = ops.maximum(b1_y1, b2_y1)
    inter_x2 = ops.minimum(b1_x2, b2_x2)
    inter_y2 = ops.minimum(b1_y2, b2_y2)

    inter_w = ops.maximum(inter_x2 - inter_x1, 0)
    inter_h = ops.maximum(inter_y2 - inter_y1, 0)
    inter = inter_w * inter_h

    # Calculate union areas
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter + eps

    # Calculate IoU and squeeze the last dimension (which should be 1)
    iou = ops.squeeze(inter / union, axis=-1)

    if CIoU or DIoU or GIoU:
        # Calculate enclosing box
        c_x1 = ops.minimum(b1_x1, b2_x1)
        c_y1 = ops.minimum(b1_y1, b2_y1)
        c_x2 = ops.maximum(b1_x2, b2_x2)
        c_y2 = ops.maximum(b1_y2, b2_y2)

        if CIoU or DIoU:
            # Calculate diagonal distance of enclosing box
            c2 = ops.squeeze((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps, axis=-1)

            # Calculate center distance
            rho2 = ops.squeeze(((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                               (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4, axis=-1)

            if CIoU:
                # Calculate aspect ratio penalty
                pi_squared = ops.convert_to_tensor(np.pi ** 2, dtype=iou.dtype)

                w1_sq = ops.squeeze(w1, axis=-1)
                h1_sq = ops.squeeze(h1, axis=-1)
                w2_sq = ops.squeeze(w2, axis=-1)
                h2_sq = ops.squeeze(h2, axis=-1)

                v = (4 / pi_squared) * ops.power(
                    ops.arctan(w2_sq / (h2_sq + eps)) - ops.arctan(w1_sq / (h1_sq + eps)), 2
                )
                alpha = v / (v - iou + (1 + eps))
                alpha = ops.stop_gradient(alpha)  # Don't backpropagate through alpha
                return iou - (rho2 / c2 + v * alpha)
            else:  # DIoU
                return iou - rho2 / c2
        else:  # GIoU
            c_area = ops.squeeze((c_x2 - c_x1) * (c_y2 - c_y1) + eps, axis=-1)
            union_sq = ops.squeeze(union, axis=-1)
            return iou - (c_area - union_sq) / c_area

    return iou


@keras.saving.register_keras_serializable()
class YOLOv12Loss(keras.losses.Loss):
    """YOLOv12 Loss Function with Task-Aligned Assigner.

    This loss function combines multiple components:
    - Task-Aligned Assigner for optimal ground truth assignment
    - CIoU loss for bounding box regression
    - Distribution Focal Loss (DFL) for refined localization
    - Binary Cross Entropy for classification

    Args:
        num_classes: Number of object classes.
        input_shape: Model input shape (height, width) for anchor generation.
        reg_max: Maximum value for DFL regression.
        box_weight: Weight for bounding box loss.
        cls_weight: Weight for classification loss.
        dfl_weight: Weight for DFL loss.
        assigner_topk: Top-k candidates for task-aligned assignment.
        assigner_alpha: Alpha parameter for alignment metric.
        assigner_beta: Beta parameter for alignment metric.
        name: Loss function name.
    """

    def __init__(
        self,
        num_classes: int = 80,
        input_shape: Tuple[int, int] = (640, 640),
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        assigner_topk: int = 10,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        name: str = "yolov12_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.input_shape = input_shape
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner_topk = assigner_topk
        self.assigner_alpha = assigner_alpha
        self.assigner_beta = assigner_beta

        # Initialize BCE loss
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

        # Generate anchors and strides
        self.anchors, self.strides = self._make_anchors()

        logger.info(f"YOLOv12Loss initialized with {num_classes} classes")

    def _make_anchors(self, grid_cell_offset: float = 0.5) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generate anchor points and strides for all feature map levels.

        Args:
            grid_cell_offset: Offset for grid cell centers.

        Returns:
            Tuple of (anchor_points, stride_tensor).
        """
        H, W = self.input_shape
        strides_config = [8, 16, 32]  # P3, P4, P5 strides

        anchor_points = []
        stride_tensor = []

        for stride in strides_config:
            h, w = H // stride, W // stride

            # Create grid coordinates
            x_coords = ops.arange(w, dtype="float32") + grid_cell_offset
            y_coords = ops.arange(h, dtype="float32") + grid_cell_offset

            # Create meshgrid using broadcasting
            x_grid = ops.expand_dims(x_coords, 0)  # (1, w)
            y_grid = ops.expand_dims(y_coords, 1)  # (h, 1)

            x_grid = ops.tile(x_grid, [h, 1])  # (h, w)
            y_grid = ops.tile(y_grid, [1, w])  # (h, w)

            # Stack and reshape
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)  # (h, w, 2)
            xy_grid = ops.reshape(xy_grid, (-1, 2))  # (h*w, 2)

            anchor_points.append(xy_grid)
            stride_tensor.append(ops.full((h * w, 1), stride, dtype="float32"))

        anchors = ops.concatenate(anchor_points, axis=0)
        strides = ops.concatenate(stride_tensor, axis=0)

        return anchors, strides

    def _task_aligned_assigner(
        self,
        pred_scores: keras.KerasTensor,
        pred_bboxes: keras.KerasTensor,
        anchors: keras.KerasTensor,
        gt_labels: keras.KerasTensor,
        gt_bboxes: keras.KerasTensor,
        mask_gt: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Simplified task-aligned assignment of ground truth boxes to anchors.

        Args:
            pred_scores: Predicted class scores, shape (batch, num_anchors, num_classes).
            pred_bboxes: Predicted bounding boxes, shape (batch, num_anchors, 4).
            anchors: Anchor points, shape (num_anchors, 2).
            gt_labels: Ground truth labels, shape (batch, max_gt, 1).
            gt_bboxes: Ground truth boxes, shape (batch, max_gt, 4).
            mask_gt: Valid ground truth mask, shape (batch, max_gt, 1).

        Returns:
            Tuple of (target_gt_idx, fg_mask).
        """
        batch_size = ops.shape(pred_scores)[0]
        num_anchors = ops.shape(pred_scores)[1]
        max_gt = ops.shape(gt_labels)[1]

        # Reshape inputs for easier broadcasting
        anchors_exp = ops.reshape(anchors, (1, 1, num_anchors, 2))
        gt_bboxes_exp = ops.expand_dims(gt_bboxes, 2)
        pred_bboxes_exp = ops.expand_dims(pred_bboxes, 1)

        # Check if anchors are inside GT boxes
        gt_x1y1, gt_x2y2 = ops.split(gt_bboxes_exp, 2, axis=-1)

        # Compare anchors with GT boxes
        inside_x1y1 = anchors_exp >= gt_x1y1
        inside_x2y2 = anchors_exp <= gt_x2y2

        # Check both conditions and reduce over coordinate dimension
        inside_gt = ops.logical_and(
            ops.all(inside_x1y1, axis=-1),
            ops.all(inside_x2y2, axis=-1)
        )

        # Calculate IoU between predicted boxes and GT boxes
        ious = bbox_iou(pred_bboxes_exp, gt_bboxes_exp, xywh=False, CIoU=True)

        # Get classification scores for GT classes
        gt_labels_int = ops.cast(ops.squeeze(gt_labels, axis=-1), "int32")
        gt_labels_one_hot = ops.one_hot(gt_labels_int, self.num_classes)

        # Expand dimensions for broadcasting
        gt_labels_exp = ops.expand_dims(gt_labels_one_hot, 2)
        pred_scores_exp = ops.expand_dims(pred_scores, 1)
        pred_scores_sigmoid = ops.nn.sigmoid(pred_scores_exp)

        # Get scores for corresponding GT classes
        cls_scores_expanded = gt_labels_exp * pred_scores_sigmoid
        cls_scores = ops.sum(cls_scores_expanded, axis=-1)

        # Calculate alignment metric
        align_metric = ops.power(cls_scores, self.assigner_alpha) * ops.power(ious, self.assigner_beta)

        # Apply inside GT and valid GT constraints
        mask_gt_2d = ops.squeeze(mask_gt, axis=-1)
        mask_gt_3d = ops.expand_dims(mask_gt_2d, -1)
        mask_gt_broadcast = ops.cast(mask_gt_3d, "bool")

        # Combine inside GT and valid GT masks
        inside_and_valid = ops.logical_and(inside_gt, mask_gt_broadcast)

        # Zero out alignment metric for invalid assignments
        align_metric = ops.where(inside_and_valid, align_metric, ops.zeros_like(align_metric))

        # For each anchor, find the GT with the highest alignment metric
        target_gt_idx = ops.argmax(align_metric, axis=1)

        # Create foreground mask (anchors assigned to any GT)
        max_align_metric = ops.max(align_metric, axis=1)
        fg_mask = max_align_metric > 0

        return target_gt_idx, fg_mask

    def _get_targets(
        self,
        gt_labels: keras.KerasTensor,
        gt_bboxes: keras.KerasTensor,
        target_gt_idx: keras.KerasTensor,
        fg_mask: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Get target labels, boxes, and scores for loss calculation.

        Args:
            gt_labels: Ground truth labels.
            gt_bboxes: Ground truth boxes.
            target_gt_idx: Indices of matched ground truths.
            fg_mask: Foreground mask.

        Returns:
            Tuple of (target_labels, target_bboxes, target_scores).
        """
        batch_size = ops.shape(target_gt_idx)[0]
        num_anchors = ops.shape(target_gt_idx)[1]
        max_gt = ops.shape(gt_labels)[1]

        # Flatten gt_labels and gt_bboxes for easier indexing
        flat_gt_labels = ops.reshape(gt_labels, [batch_size * max_gt, 1])
        flat_gt_bboxes = ops.reshape(gt_bboxes, [batch_size * max_gt, 4])

        # Create flat indices for gathering
        batch_indices = ops.expand_dims(ops.arange(batch_size), 1)
        batch_indices = ops.tile(batch_indices, [1, num_anchors])

        # Calculate flat indices
        flat_indices = batch_indices * max_gt + target_gt_idx
        flat_indices = ops.reshape(flat_indices, [-1])

        # Gather targets
        gathered_labels = ops.take(flat_gt_labels, flat_indices, axis=0)
        gathered_bboxes = ops.take(flat_gt_bboxes, flat_indices, axis=0)

        # Reshape back
        target_labels = ops.reshape(gathered_labels, [batch_size, num_anchors, 1])
        target_bboxes = ops.reshape(gathered_bboxes, [batch_size, num_anchors, 4])

        # Convert labels to one-hot scores
        target_labels_int = ops.cast(ops.squeeze(target_labels, -1), "int32")
        target_scores = ops.one_hot(target_labels_int, self.num_classes)

        # Apply foreground mask
        fg_mask_expanded = ops.expand_dims(ops.cast(fg_mask, "float32"), -1)
        target_scores = target_scores * fg_mask_expanded

        return target_labels_int, target_bboxes, target_scores

    def _bbox_to_dist(
        self,
        bboxes: keras.KerasTensor,
        anchors: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convert bounding boxes to distance format (ltrb).

        Args:
            bboxes: Bounding boxes in xyxy format.
            anchors: Anchor points.

        Returns:
            Distance format boxes (left, top, right, bottom).
        """
        x1y1, x2y2 = ops.split(bboxes, 2, axis=-1)
        anchors_expanded = ops.expand_dims(anchors, 0)

        lt = anchors_expanded - x1y1  # left, top
        rb = x2y2 - anchors_expanded  # right, bottom

        return ops.concatenate([lt, rb], axis=-1)

    def _dist_to_bbox(
        self,
        distance: keras.KerasTensor,
        anchors: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convert distance predictions to bounding boxes.

        Args:
            distance: Distance predictions in ltrb format.
            anchors: Anchor points.

        Returns:
            Bounding boxes in xyxy format.
        """
        lt, rb = ops.split(distance, 2, axis=-1)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        return ops.concatenate([x1y1, x2y2], axis=-1)

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Calculate YOLOv12 loss.

        Args:
            y_true: Ground truth labels and boxes, shape (batch, max_boxes, 5).
                    Format: (class_id, x1, y1, x2, y2) in absolute coordinates.
            y_pred: Model predictions, shape (batch, num_anchors, 4*reg_max + num_classes).

        Returns:
            Total loss value.
        """
        # Split predictions
        pred_dist, pred_scores = ops.split(
            y_pred, [4 * self.reg_max], axis=-1
        )

        # Reshape distance predictions
        batch_size = ops.shape(y_pred)[0]
        num_anchors = ops.shape(self.anchors)[0]
        pred_dist = ops.reshape(pred_dist, [batch_size, num_anchors, 4, self.reg_max])

        # Decode predicted boxes
        pred_dist_softmax = ops.nn.softmax(pred_dist, axis=-1)
        pred_dist_mean = ops.sum(
            pred_dist_softmax * ops.arange(self.reg_max, dtype="float32"),
            axis=-1
        )
        pred_bboxes = self._dist_to_bbox(pred_dist_mean, self.anchors * self.strides)

        # Extract ground truth components
        gt_labels = y_true[..., :1]  # (batch, max_boxes, 1)
        gt_bboxes = y_true[..., 1:]  # (batch, max_boxes, 4)

        # Create valid GT mask (non-zero coordinates)
        mask_gt = ops.sum(gt_bboxes, axis=-1, keepdims=True) > 0

        # Check if we have any valid ground truths
        total_valid_gt = ops.sum(ops.cast(mask_gt, "float32"))

        # Initialize losses
        loss_cls = ops.convert_to_tensor(0.0, dtype=pred_scores.dtype)
        loss_box = ops.convert_to_tensor(0.0, dtype=pred_scores.dtype)
        loss_dfl = ops.convert_to_tensor(0.0, dtype=pred_scores.dtype)

        # Only compute losses if we have valid ground truths
        def compute_losses():
            # Task-aligned assignment
            target_gt_idx, fg_mask = self._task_aligned_assigner(
                ops.nn.sigmoid(pred_scores),
                pred_bboxes,
                self.anchors * self.strides,
                gt_labels,
                gt_bboxes,
                mask_gt
            )

            # Get targets
            target_labels, target_bboxes, target_scores = self._get_targets(
                gt_labels, gt_bboxes, target_gt_idx, fg_mask
            )

            # Calculate normalization factor
            target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)

            # Classification loss
            cls_loss = self.bce(target_scores, pred_scores)
            cls_loss = ops.sum(cls_loss) / target_scores_sum

            # Box and DFL losses for foreground samples
            num_fg = ops.sum(ops.cast(fg_mask, "float32"))

            def compute_box_losses():
                # Use boolean masking to select positive samples
                flat_fg_mask = ops.reshape(fg_mask, [-1])
                flat_pred_bboxes = ops.reshape(pred_bboxes, [-1, 4])
                flat_target_bboxes = ops.reshape(target_bboxes, [-1, 4])
                flat_target_scores = ops.reshape(target_scores, [-1, self.num_classes])
                flat_pred_dist = ops.reshape(pred_dist, [-1, 4, self.reg_max])

                # Select positive samples
                pred_bboxes_pos = flat_pred_bboxes[flat_fg_mask]
                target_bboxes_pos = flat_target_bboxes[flat_fg_mask]
                target_scores_pos = flat_target_scores[flat_fg_mask]
                pred_dist_pos = flat_pred_dist[flat_fg_mask]

                # Get corresponding anchor indices
                batch_size = ops.shape(pred_bboxes)[0]
                num_anchors = ops.shape(pred_bboxes)[1]
                anchor_indices = ops.tile(ops.arange(num_anchors), [batch_size])
                anchor_indices_pos = anchor_indices[flat_fg_mask]
                anchors_pos = ops.take(self.anchors * self.strides, anchor_indices_pos, axis=0)

                # Calculate IoU loss
                iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
                weight = ops.sum(target_scores_pos, axis=-1)
                box_loss = ops.sum((1.0 - iou) * weight) / target_scores_sum

                # Calculate DFL loss
                target_ltrb = self._bbox_to_dist(
                    ops.expand_dims(target_bboxes_pos, 0),
                    ops.expand_dims(anchors_pos, 0)
                )
                target_ltrb = ops.squeeze(target_ltrb, 0)
                target_ltrb = ops.clip(target_ltrb, 0, self.reg_max - 1.01)

                target_ltrb_long = ops.cast(target_ltrb, "int32")
                target_ltrb_long = ops.clip(target_ltrb_long, 0, self.reg_max - 2)

                pred_dist_flat = ops.reshape(pred_dist_pos, [-1, self.reg_max])
                target_flat = ops.reshape(target_ltrb_long, [-1])

                dfl_loss_val = keras.losses.sparse_categorical_crossentropy(
                    target_flat, pred_dist_flat, from_logits=True
                )
                dfl_loss_val = ops.sum(dfl_loss_val) / target_scores_sum

                return box_loss, dfl_loss_val

            def no_positive_samples():
                return (
                    ops.convert_to_tensor(0.0, dtype=pred_scores.dtype),
                    ops.convert_to_tensor(0.0, dtype=pred_scores.dtype)
                )

            # Compute box losses only if we have positive samples
            box_loss, dfl_loss = ops.cond(
                num_fg > 0,
                compute_box_losses,
                no_positive_samples
            )

            return cls_loss, box_loss, dfl_loss

        def no_valid_gt():
            return (
                ops.convert_to_tensor(0.01, dtype=pred_scores.dtype),  # Small classification loss
                ops.convert_to_tensor(0.0, dtype=pred_scores.dtype),
                ops.convert_to_tensor(0.0, dtype=pred_scores.dtype)
            )

        # Compute losses only if we have valid ground truths
        loss_cls, loss_box, loss_dfl = ops.cond(
            total_valid_gt > 0,
            compute_losses,
            no_valid_gt
        )

        # Calculate total loss
        total_loss = (
            self.box_weight * loss_box +
            self.cls_weight * loss_cls +
            self.dfl_weight * loss_dfl
        )

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "reg_max": self.reg_max,
            "box_weight": self.box_weight,
            "cls_weight": self.cls_weight,
            "dfl_weight": self.dfl_weight,
            "assigner_topk": self.assigner_topk,
            "assigner_alpha": self.assigner_alpha,
            "assigner_beta": self.assigner_beta,
        })
        return config


def create_yolov12_loss(
    num_classes: int = 80,
    input_shape: Tuple[int, int] = (640, 640),
    **kwargs
) -> YOLOv12Loss:
    """Create YOLOv12 loss function with specified configuration.

    Args:
        num_classes: Number of object classes.
        input_shape: Model input shape.
        **kwargs: Additional arguments for YOLOv12Loss.

    Returns:
        YOLOv12Loss instance.
    """
    loss_fn = YOLOv12Loss(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    logger.info(f"YOLOv12 loss function created for {num_classes} classes")
    return loss_fn