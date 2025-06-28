"""
YOLOv12 Multi-Task Loss Function using Native Keras Components.

This module provides comprehensive loss functions for training YOLOv12 on multiple tasks
with proper loss balancing. It leverages native Keras losses where possible and integrates
seamlessly with the Named Outputs (Functional API) multi-task model.

Components:
    - Detection Loss: Custom YOLOv12Loss (specialized for object detection)
    - Segmentation Loss: Combined native Dice and BinaryFocalCrossentropy
    - Classification Loss: Native BinaryCrossentropy with configuration options
    - Multi-task loss coordination with TaskType enum support
"""

import keras
from keras import ops
from typing import Dict, Any, List, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.bounding_box import bbox_iou
from dl_techniques.utils.vision_task_types import (
    TaskType, TaskConfiguration, parse_task_list)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class YOLOv12ObjectDetectionLoss(keras.losses.Loss):
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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DiceFocalSegmentationLoss(keras.losses.Loss):
    """
    Combined Dice and Binary Focal Cross-Entropy Loss for segmentation using native Keras components.

    This loss combines the native Keras Dice loss with BinaryFocalCrossentropy to handle
    class imbalance common in segmentation tasks like crack detection.
    """

    def __init__(
        self,
        # Dice loss parameters
        dice_smooth: float = 1e-6,
        dice_axis: Union[int, List[int]] = [1, 2],
        # Focal loss parameters
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_from_logits: bool = False,
        # Loss weighting
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        # Base parameters
        reduction: str = "sum_over_batch_size",
        name: str = "dice_focal_segmentation_loss",
        **kwargs
    ):
        """
        Initialize combined Dice-Focal segmentation loss.

        Args:
            dice_smooth: Smoothing constant for Dice loss.
            dice_axis: Axis or axes to compute Dice over (spatial dimensions).
            focal_alpha: Weighting factor for rare class in focal loss.
            focal_gamma: Focusing parameter for focal loss.
            focal_from_logits: Whether focal loss input is from logits.
            dice_weight: Weight for dice loss component.
            focal_weight: Weight for focal loss component.
            reduction: Type of reduction to apply.
            name: Loss function name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)

        # Store parameters
        self.dice_smooth = dice_smooth
        self.dice_axis = dice_axis if isinstance(dice_axis, list) else [dice_axis]
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_from_logits = focal_from_logits
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Initialize native Keras losses
        self.dice_loss = keras.losses.Dice(
            reduction="none",  # We'll handle reduction ourselves
            axis=self.dice_axis,
            name="dice_component"
        )

        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            from_logits=self.focal_from_logits,
            reduction="none",  # We'll handle reduction ourselves
            name="focal_component"
        )

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Calculate combined Dice and Focal loss.

        Args:
            y_true: Ground truth masks (batch_size, height, width, 1).
            y_pred: Predicted masks (batch_size, height, width, 1).

        Returns:
            Combined loss value.
        """
        # Calculate individual loss components
        dice_loss_val = self.dice_loss(y_true, y_pred)
        focal_loss_val = self.focal_loss(y_true, y_pred)

        # Combine losses with weights
        combined_loss = (
            self.dice_weight * dice_loss_val +
            self.focal_weight * focal_loss_val
        )

        return combined_loss

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "dice_smooth": self.dice_smooth,
            "dice_axis": self.dice_axis,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "focal_from_logits": self.focal_from_logits,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
        })
        return config


@keras.saving.register_keras_serializable()
class YOLOv12MultiTaskLoss(keras.losses.Loss):
    """
    Multi-task loss function for YOLOv12 using native Keras components where possible.

    This loss function works with the Named Outputs (Functional API) multi-task model
    and supports TaskType enum-based configuration. It automatically handles different
    task combinations and provides proper loss balancing.
    """

    def __init__(
        self,
        # Task configuration
        task_config: Union[
            TaskConfiguration,
            List[TaskType],
            List[str],
            TaskType,
            str
        ] = TaskType.DETECTION,

        # Task loss weights
        detection_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        classification_weight: float = 1.0,

        # Detection loss parameters (YOLOv12Loss)
        num_classes: int = 1,
        input_shape: tuple = (640, 640),
        reg_max: int = 16,
        detection_box_weight: float = 7.5,
        detection_cls_weight: float = 0.5,
        detection_dfl_weight: float = 1.5,

        # Segmentation loss parameters (DiceFocalSegmentationLoss)
        seg_dice_weight: float = 0.5,
        seg_focal_weight: float = 0.5,
        seg_focal_alpha: float = 0.25,
        seg_focal_gamma: float = 2.0,
        seg_from_logits: bool = False,

        # Classification loss parameters (BinaryCrossentropy)
        cls_from_logits: bool = False,
        cls_label_smoothing: float = 0.0,

        # Uncertainty weighting (learnable task balancing)
        use_uncertainty_weighting: bool = False,
        uncertainty_regularization: float = 1.0,

        # Base parameters
        reduction: str = "sum_over_batch_size",
        name: str = "yolov12_multitask_loss",
        **kwargs
    ):
        """
        Initialize YOLOv12 multitask loss function.

        Args:
            task_config: Task configuration using TaskType enums or strings.
            detection_weight: Weight for detection loss.
            segmentation_weight: Weight for segmentation loss.
            classification_weight: Weight for classification loss.
            num_classes: Number of detection classes.
            input_shape: Input image shape for detection loss.
            reg_max: Maximum value for DFL regression.
            detection_box_weight: Weight for bbox loss in detection.
            detection_cls_weight: Weight for classification loss in detection.
            detection_dfl_weight: Weight for DFL loss in detection.
            seg_dice_weight: Weight for dice component in segmentation.
            seg_focal_weight: Weight for focal component in segmentation.
            seg_focal_alpha: Alpha parameter for segmentation focal loss.
            seg_focal_gamma: Gamma parameter for segmentation focal loss.
            seg_from_logits: Whether segmentation predictions are logits.
            cls_from_logits: Whether classification predictions are logits.
            cls_label_smoothing: Label smoothing for classification.
            use_uncertainty_weighting: Whether to use learnable uncertainty weighting.
            uncertainty_regularization: Regularization strength for uncertainty.
            reduction: Type of reduction to apply.
            name: Loss function name.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)

        # Parse and store task configuration
        self.task_config = parse_task_list(task_config)

        # Store all parameters for serialization
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.reg_max = reg_max
        self.detection_box_weight = detection_box_weight
        self.detection_cls_weight = detection_cls_weight
        self.detection_dfl_weight = detection_dfl_weight
        self.seg_dice_weight = seg_dice_weight
        self.seg_focal_weight = seg_focal_weight
        self.seg_focal_alpha = seg_focal_alpha
        self.seg_focal_gamma = seg_focal_gamma
        self.seg_from_logits = seg_from_logits
        self.cls_from_logits = cls_from_logits
        self.cls_label_smoothing = cls_label_smoothing
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty_regularization = uncertainty_regularization

        # Initialize task-specific loss functions
        self.detection_loss = None
        self.segmentation_loss = None
        self.classification_loss = None

        self._build_loss_functions()

        # Initialize uncertainty weights if requested
        if self.use_uncertainty_weighting:
            self._build_uncertainty_weights()

        enabled_tasks = self.task_config.get_task_names()
        logger.info(
            f"YOLOv12MultiTaskLoss initialized with tasks: {enabled_tasks}. "
            f"Uncertainty weighting: {use_uncertainty_weighting}"
        )

    def _build_loss_functions(self) -> None:
        """Build task-specific loss functions based on enabled tasks."""

        if self.task_config.has_detection():
            self.detection_loss = YOLOv12ObjectDetectionLoss(
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                reg_max=self.reg_max,
                box_weight=self.detection_box_weight,
                cls_weight=self.detection_cls_weight,
                dfl_weight=self.detection_dfl_weight,
                reduction="none",  # We handle reduction at the multi-task level
                name="detection_loss"
            )

        if self.task_config.has_segmentation():
            self.segmentation_loss = DiceFocalSegmentationLoss(
                dice_weight=self.seg_dice_weight,
                focal_weight=self.seg_focal_weight,
                focal_alpha=self.seg_focal_alpha,
                focal_gamma=self.seg_focal_gamma,
                focal_from_logits=self.seg_from_logits,
                reduction="none",  # We handle reduction at the multi-task level
                name="segmentation_loss"
            )

        if self.task_config.has_classification():
            self.classification_loss = keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=False,
                alpha=0.25,
                gamma=2.0,
                from_logits=self.cls_from_logits,
                label_smoothing=self.cls_label_smoothing,
                reduction="none",  # We handle reduction at the multi-task level
                name="classification_loss"
            )

    def _build_uncertainty_weights(self) -> None:
        """Build learnable uncertainty weighting parameters."""
        if self.task_config.has_detection():
            self.detection_log_var = self.add_weight(
                name="detection_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )

        if self.task_config.has_segmentation():
            self.segmentation_log_var = self.add_weight(
                name="segmentation_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )

        if self.task_config.has_classification():
            self.classification_log_var = self.add_weight(
                name="classification_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )

    def call(
        self,
        y_true: Union[Dict[str, keras.KerasTensor], keras.KerasTensor],
        y_pred: Union[Dict[str, keras.KerasTensor], keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Calculate multi-task loss.

        Args:
            y_true: Ground truth labels. For multi-task: dictionary with task keys.
                   For single-task: direct tensor.
            y_pred: Model predictions. For multi-task: dictionary with task keys.
                   For single-task: direct tensor.

        Returns:
            Total multi-task loss.
        """
        # Handle single-task case (direct tensors)
        if not isinstance(y_true, dict) or not isinstance(y_pred, dict):
            return self._compute_single_task_loss(y_true, y_pred)

        # Multi-task case (dictionaries)
        return self._compute_multi_task_loss(y_true, y_pred)

    def _compute_single_task_loss(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute loss for single-task model."""
        enabled_tasks = self.task_config.get_enabled_tasks()

        if len(enabled_tasks) != 1:
            raise ValueError(
                f"Single task loss expects exactly 1 enabled task, got {len(enabled_tasks)}"
            )

        task = enabled_tasks[0]

        if task == TaskType.DETECTION:
            return self.detection_loss(y_true, y_pred)
        elif task == TaskType.SEGMENTATION:
            return self.segmentation_loss(y_true, y_pred)
        elif task == TaskType.CLASSIFICATION:
            return self.classification_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown task type: {task}")

    def _compute_multi_task_loss(
        self,
        y_true: Dict[str, keras.KerasTensor],
        y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Compute loss for multi-task model."""
        total_loss = ops.convert_to_tensor(0.0, dtype=self.dtype)
        individual_losses = {}

        # Detection loss
        if (self.task_config.has_detection() and
            TaskType.DETECTION.value in y_true and
            TaskType.DETECTION.value in y_pred):

            det_loss = self.detection_loss(
                y_true[TaskType.DETECTION.value],
                y_pred[TaskType.DETECTION.value]
            )
            individual_losses['detection'] = det_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.detection_log_var)
                det_loss_weighted = precision * det_loss + self.detection_log_var
            else:
                det_loss_weighted = self.detection_weight * det_loss

            total_loss = total_loss + det_loss_weighted

        # Segmentation loss
        if (self.task_config.has_segmentation() and
            TaskType.SEGMENTATION.value in y_true and
            TaskType.SEGMENTATION.value in y_pred):

            seg_loss = self.segmentation_loss(
                y_true[TaskType.SEGMENTATION.value],
                y_pred[TaskType.SEGMENTATION.value]
            )
            individual_losses['segmentation'] = seg_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.segmentation_log_var)
                seg_loss_weighted = precision * seg_loss + self.segmentation_log_var
            else:
                seg_loss_weighted = self.segmentation_weight * seg_loss

            total_loss = total_loss + seg_loss_weighted

        # Classification loss
        if (self.task_config.has_classification() and
            TaskType.CLASSIFICATION.value in y_true and
            TaskType.CLASSIFICATION.value in y_pred):

            cls_loss = self.classification_loss(
                y_true[TaskType.CLASSIFICATION.value],
                y_pred[TaskType.CLASSIFICATION.value]
            )
            individual_losses['classification'] = cls_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.classification_log_var)
                cls_loss_weighted = precision * cls_loss + self.classification_log_var
            else:
                cls_loss_weighted = self.classification_weight * cls_loss

            total_loss = total_loss + cls_loss_weighted

        # Add uncertainty regularization
        if self.use_uncertainty_weighting:
            uncertainty_reg = self.uncertainty_regularization * self._get_uncertainty_regularization()
            total_loss = total_loss + uncertainty_reg

        # Store individual losses for monitoring
        self.individual_losses = individual_losses

        return total_loss

    def _get_uncertainty_regularization(self) -> keras.KerasTensor:
        """Calculate uncertainty regularization term."""
        reg_term = ops.convert_to_tensor(0.0, dtype=self.dtype)

        if self.task_config.has_detection() and hasattr(self, 'detection_log_var'):
            reg_term = reg_term + self.detection_log_var

        if self.task_config.has_segmentation() and hasattr(self, 'segmentation_log_var'):
            reg_term = reg_term + self.segmentation_log_var

        if self.task_config.has_classification() and hasattr(self, 'classification_log_var'):
            reg_term = reg_term + self.classification_log_var

        return reg_term

    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights for monitoring.

        Returns:
            Dictionary mapping task names to their current weights.
        """
        weights = {}

        if self.task_config.has_detection():
            if self.use_uncertainty_weighting and hasattr(self, 'detection_log_var'):
                weights['detection'] = float(ops.exp(-self.detection_log_var))
            else:
                weights['detection'] = self.detection_weight

        if self.task_config.has_segmentation():
            if self.use_uncertainty_weighting and hasattr(self, 'segmentation_log_var'):
                weights['segmentation'] = float(ops.exp(-self.segmentation_log_var))
            else:
                weights['segmentation'] = self.segmentation_weight

        if self.task_config.has_classification():
            if self.use_uncertainty_weighting and hasattr(self, 'classification_log_var'):
                weights['classification'] = float(ops.exp(-self.classification_log_var))
            else:
                weights['classification'] = self.classification_weight

        return weights

    def get_individual_losses(self) -> Dict[str, float]:
        """
        Get individual task losses from the last forward pass.

        Returns:
            Dictionary mapping task names to their loss values.
        """
        if not hasattr(self, 'individual_losses'):
            return {}

        return {
            name: float(loss) if hasattr(loss, 'numpy') else float(loss)
            for name, loss in self.individual_losses.items()
        }

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization."""
        config = super().get_config()
        config.update({
            # Serialize task config as task names list for simplicity
            "task_config": self.task_config.get_task_names(),
            "detection_weight": self.detection_weight,
            "segmentation_weight": self.segmentation_weight,
            "classification_weight": self.classification_weight,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "reg_max": self.reg_max,
            "detection_box_weight": self.detection_box_weight,
            "detection_cls_weight": self.detection_cls_weight,
            "detection_dfl_weight": self.detection_dfl_weight,
            "seg_dice_weight": self.seg_dice_weight,
            "seg_focal_weight": self.seg_focal_weight,
            "seg_focal_alpha": self.seg_focal_alpha,
            "seg_focal_gamma": self.seg_focal_gamma,
            "seg_from_logits": self.seg_from_logits,
            "cls_from_logits": self.cls_from_logits,
            "cls_label_smoothing": self.cls_label_smoothing,
            "use_uncertainty_weighting": self.use_uncertainty_weighting,
            "uncertainty_regularization": self.uncertainty_regularization,
        })
        return config

# ---------------------------------------------------------------------
# Factory functions for common use cases
# ---------------------------------------------------------------------

def create_yolov12_multitask_loss(
    tasks: Union[
        List[TaskType],
        List[str],
        TaskConfiguration,
        TaskType,
        str
    ] = TaskType.DETECTION,
    num_classes: int = 1,
    input_shape: tuple = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Create YOLOv12 multi-task loss function with specified configuration.

    Args:
        tasks: Tasks to enable - TaskType enums, strings, or TaskConfiguration.
        num_classes: Number of classes for detection/classification.
        input_shape: Input image shape.
        **kwargs: Additional arguments for YOLOv12MultiTaskLoss.

    Returns:
        YOLOv12MultiTaskLoss instance.

    Example:
        >>> loss = create_yolov12_multitask_loss(
        ...     tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        ...     num_classes=1,
        ...     input_shape=(256, 256)
        ... )
    """
    loss_fn = YOLOv12MultiTaskLoss(
        task_config=tasks,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    task_config = parse_task_list(tasks)
    task_names = task_config.get_task_names()
    logger.info(f"YOLOv12 multi-task loss created for tasks: {task_names}")
    return loss_fn


def create_detection_only_loss(
    num_classes: int = 1,
    input_shape: tuple = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """Create loss function for detection-only models."""
    return create_yolov12_multitask_loss(
        tasks=TaskType.DETECTION,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )


def create_segmentation_only_loss(
    input_shape: tuple = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """Create loss function for segmentation-only models."""
    return create_yolov12_multitask_loss(
        tasks=TaskType.SEGMENTATION,
        num_classes=1,  # Binary segmentation
        input_shape=input_shape,
        **kwargs
    )


def create_detection_segmentation_loss(
    num_classes: int = 1,
    input_shape: tuple = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """Create loss function for detection + segmentation models."""
    return create_yolov12_multitask_loss(
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

# ---------------------------------------------------------------------
# Utility functions for loss monitoring and callbacks
# ---------------------------------------------------------------------

class MultiTaskLossCallback(keras.callbacks.Callback):
    """
    Callback for monitoring individual task losses during training.

    This callback extracts and logs individual task losses from the
    multi-task loss function for better monitoring and debugging.
    """

    def __init__(self, loss_fn: YOLOv12MultiTaskLoss, log_freq: int = 1):
        """
        Initialize callback.

        Args:
            loss_fn: The multi-task loss function instance.
            log_freq: Frequency of logging (every N epochs).
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.log_freq = log_freq

    def on_epoch_end(self, epoch, logs=None):
        """Log individual task losses and weights."""
        if (epoch + 1) % self.log_freq == 0:
            # Get current task weights
            weights = self.loss_fn.get_task_weights()

            # Log weights
            for task, weight in weights.items():
                logs[f'{task}_weight'] = weight

            # Get individual losses if available
            individual_losses = self.loss_fn.get_individual_losses()
            for task, loss_val in individual_losses.items():
                logs[f'{task}_loss'] = loss_val

# ---------------------------------------------------------------------
