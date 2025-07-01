"""
Defines a custom Keras loss function for the YOLOv12 multi-task model.

This module provides a flexible and powerful multi-task loss system, centered
around the `YOLOv12MultiTaskLoss` class. It is specifically designed to be used as a
single loss function for a Keras model with multiple named outputs.

The core design philosophy is that the single `YOLOv12MultiTaskLoss` instance is
passed as the main loss function during `model.compile()`. The loss function then
automatically infers which task-specific sub-loss to apply based on the tensor
shapes of `y_true` and `y_pred` that Keras provides for each model output.

This module also supports learnable uncertainty weighting to automatically
balance the contribution of each task's loss during training.

FIXED VERSION: Now supports configurable segmentation classes for COCO pretraining
(80 classes) vs crack detection fine-tuning (1 class).

Key Components:
    - YOLOv12MultiTaskLoss: The main, user-facing loss class that orchestrates all
      sub-losses and handles dynamic or static weighting. It is the single entry
      point for the training script.

    - YOLOv12ObjectDetectionLoss, DiceFocalSegmentationLoss, ClassificationFocalLoss:
      Internal, task-specific loss classes used by the main orchestrator.

    - Factory Function: A convenience function `create_yolov12_multitask_loss` to
      easily instantiate the main loss function with the correct configuration.
"""

import keras
from keras import ops
from typing import Dict, Any, List, Union, Tuple, Optional

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.bounding_box import bbox_iou
from dl_techniques.utils.vision_task_types import (
    TaskType, TaskConfiguration, parse_task_list
)


# ---------------------------------------------------------------------
# Internal, Task-Specific Loss Implementations
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="yolov12_losses")
class YOLOv12ObjectDetectionLoss(keras.losses.Loss):
    """
    Internal loss for YOLOv12 object detection.

    This class is not intended to be used directly but is called by
    `YOLOv12MultiTaskLoss`. It combines a Task-Aligned Assigner, CIoU loss,
    Distribution Focal Loss (DFL), and Binary Cross-Entropy.

    Args:
        num_classes: Number of object detection classes.
        input_shape: Input image dimensions as (height, width).
        reg_max: Maximum regression range for distribution focal loss.
        box_weight: Weight for bounding box regression loss.
        cls_weight: Weight for classification loss.
        dfl_weight: Weight for distribution focal loss.
        assigner_alpha: Alpha parameter for task-aligned assigner.
        assigner_beta: Beta parameter for task-aligned assigner.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_shape: Tuple[int, int] = (256, 256),
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        assigner_alpha: float = 0.5,
        assigner_beta: float = 6.0,
        name: str = "yolov12_detection_loss_internal",
        **kwargs
    ):
        super().__init__(name=name, reduction="none", **kwargs)

        # Store configuration parameters
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner_alpha = assigner_alpha
        self.assigner_beta = assigner_beta

        # Initialize binary focal cross-entropy loss for classification
        # focal cross entropy handles class imbalance better
        self.cfc_loss_fn = keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,
            gamma=2.0,
            from_logits=True,
            reduction="none"
        )

        # Generate anchor points and strides for all feature map levels
        self.anchors, self.strides = self._make_anchors()

    def _make_anchors(
        self,
        grid_cell_offset: float = 0.5
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Generates anchor points and strides for all feature map levels.

        Args:
            grid_cell_offset: Offset for anchor point generation.

        Returns:
            Tuple of anchor points and corresponding stride tensors.
        """
        H, W = self.input_shape
        strides_config = [8, 16, 32]  # Multi-scale feature map strides
        anchor_points, stride_tensor = [], []

        for stride in strides_config:
            # Calculate feature map dimensions for current stride
            h, w = H // stride, W // stride

            # Generate coordinate grids
            x_coords = ops.arange(w, dtype="float32") + grid_cell_offset
            y_coords = ops.arange(h, dtype="float32") + grid_cell_offset
            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")

            # Stack and reshape coordinates
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))
            anchor_points.append(xy_grid)

            # Create stride tensor for current level
            stride_tensor.append(
                ops.full((h * w, 1), stride, dtype="float32")
            )

        return (
            ops.concatenate(anchor_points, 0),
            ops.concatenate(stride_tensor, 0)
        )

    def _task_aligned_assigner(
            self,
            pred_scores: keras.KerasTensor,
            pred_bboxes: keras.KerasTensor,
            anchors: keras.KerasTensor,
            gt_labels: keras.KerasTensor,
            gt_bboxes: keras.KerasTensor,
            mask_gt: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Performs task-aligned assignment of ground truth boxes to anchor points.

        This assigner determines which anchors should be responsible for detecting which
        ground truth objects. It addresses the issue of early-stage training bias towards
        small boxes by defining a set of "candidate" anchors before calculating the final
        assignment metric.

        An anchor is considered a candidate for a ground truth box if:
        1.  Its center point falls inside the ground truth box (`is_in_gt`).
            OR
        2.  It is one of the top 10 anchors with the highest IoU with that ground truth box.

        This ensures that for large objects, anchors across the entire object are
        considered for assignment, encouraging the model to learn to predict large boxes.

        Args:
            pred_scores: Predicted classification scores. Shape: (batch_size, num_anchors, num_classes).
            pred_bboxes: Predicted bounding boxes in pixel coordinates. Shape: (batch_size, num_anchors, 4).
            anchors: Anchor center points in pixel coordinates. Shape: (num_anchors, 2).
            gt_labels: Ground truth labels. Shape: (batch_size, num_gt, 1).
            gt_bboxes: Ground truth bounding boxes in pixel coordinates. Shape: (batch_size, num_gt, 4).
            mask_gt: A mask indicating valid (non-padded) ground truth boxes. Shape: (batch_size, num_gt, 1).

        Returns:
            A tuple containing:
            - target_gt_idx: The index of the GT box assigned to each anchor. Shape: (batch_size, num_anchors).
            - fg_mask: A boolean mask indicating which anchors are foreground (positive). Shape: (batch_size, num_anchors).
        """
        num_anchors = ops.shape(pred_scores)[1]

        # Prepare tensors for vectorized operations by adding new dimensions.
        # This allows for batch-wise comparison between all anchors and all GT boxes.
        anchors_exp = ops.reshape(anchors, (1, 1, num_anchors, 2))
        gt_bboxes_exp = ops.expand_dims(gt_bboxes, 2)
        pred_bboxes_exp = ops.expand_dims(pred_bboxes, 1)
        pred_scores_exp = ops.expand_dims(pred_scores, 1)

        # --- Candidate Selection ---
        # This is the core logic to counteract the small-box bias. We create a mask
        # of potential "good" anchors before calculating the final assignment metric.

        # 1. Select candidates based on spatial location (`is_in_gt`).
        # An anchor is a candidate if its center point is physically inside the GT box.
        # This is crucial for large objects, as it ensures anchors across the entire
        # object are considered, even if their initial predicted box is small and has poor IoU.
        gt_x1y1, gt_x2y2 = ops.split(gt_bboxes_exp, 2, axis=-1)
        is_in_gt = (
                ops.all(anchors_exp >= gt_x1y1, -1) &
                ops.all(anchors_exp <= gt_x2y2, -1)
        )  # Shape: (batch_size, num_gt, num_anchors)

        # 2. Select candidates based on IoU (`topk`).
        # This focuses the assignment on the `topk` anchors that best match the GT box's shape,
        # providing stability by limiting the number of candidates per GT.
        topk = 10
        # Calculate a simple IoU (not CIoU, for speed) between all predictions and all GTs.
        iou_candidates = bbox_iou(pred_bboxes_exp, gt_bboxes_exp, xywh=False,
                                  CIoU=False)  # Shape: (batch_size, num_gt, num_anchors)

        # --- THE FIX FOR THE TypeError ---
        # Find the top `k` IoU values and their indices along the last axis (num_anchors).
        # The `axis` argument is removed because it is not supported by `keras.ops.top_k`.
        # The function operates on the last axis by default, which is correct for our tensor shape.
        _, topk_idx = ops.top_k(iou_candidates, k=min(topk, num_anchors))

        # Create a one-hot mask from the top-k indices to identify which anchors were selected.
        is_in_topk = ops.one_hot(topk_idx, num_anchors, dtype=iou_candidates.dtype)
        is_in_topk = ops.sum(is_in_topk, axis=2)  # Shape: (batch_size, num_gt, num_anchors)

        # Combine the two candidate masks. An anchor is a candidate if it's in the GT box OR in the top-k IoU.
        # This provides a robust set of potential assignments for the next step.
        candidate_mask = ops.where(is_in_gt, ops.cast(1.0, iou_candidates.dtype), is_in_topk) > 0

        # --- Alignment Metric Calculation ---
        # Now we calculate the final metric, but only for the candidates we just selected.

        # Calculate the final IoU (using CIoU for a more accurate regression signal).
        # We then use the candidate mask to zero out the IoU for all non-candidate pairs.
        ious = bbox_iou(pred_bboxes_exp, gt_bboxes_exp, xywh=False, CIoU=True)
        ious = ops.where(candidate_mask, ious, 0.0)

        # Prepare one-hot labels for classification score calculation.
        gt_labels_int = ops.cast(ops.squeeze(gt_labels, -1), "int32")
        gt_labels_one_hot = ops.one_hot(gt_labels_int, self.num_classes)
        gt_labels_one_hot = ops.expand_dims(gt_labels_one_hot, 2)

        # Calculate classification alignment scores.
        # Zero out scores for all non-candidate pairs.
        cls_scores = ops.sum(gt_labels_one_hot * ops.nn.sigmoid(pred_scores_exp), -1)
        cls_scores = ops.where(candidate_mask, cls_scores, 0.0)

        # --- START FINAL, CORRECT FIX ---
        # The original alignment metric `cls**alpha * iou**beta` is a product,
        # which is numerically unstable. If the classification score for the
        # correct class is zero, the entire metric becomes zero, preventing
        # the anchor from being assigned, regardless of its IoU. This leads
        # to the mode collapse on the "person" class.
        #
        # A more robust and stable approach, used in many modern detectors,
        # is to use a WEIGHTED SUM of the two metrics. This ensures that an
        # anchor with excellent IoU can still be a top candidate even if its
        # initial classification is wrong.

        # We keep the alpha and beta hyperparameters but use them as weights.
        align_metric = self.assigner_alpha * cls_scores + self.assigner_beta * ious
        # --- END FINAL, CORRECT FIX ---

        # Apply the mask for valid (non-padded) ground truth boxes to ignore padded GTs.
        mask_gt_broadcast = ops.cast(ops.expand_dims(ops.squeeze(mask_gt, -1), -1), "bool")
        align_metric = ops.where(mask_gt_broadcast, align_metric, 0.0)

        # We also need to ensure that we only consider candidates for the final assignment.
        align_metric = ops.where(candidate_mask, align_metric, 0.0)

        # --- Final Assignment ---
        # For each anchor, find the GT box that gives it the highest alignment score.
        target_gt_idx = ops.argmax(align_metric, axis=1)

        # An anchor is considered foreground (positive) if its best score is greater than 0.
        fg_mask = ops.max(align_metric, axis=1) > 0

        # If an anchor is background, assign it to a dummy GT index (0) to prevent errors.
        target_gt_idx = ops.where(fg_mask, target_gt_idx, 0)

        return target_gt_idx, fg_mask

    def _get_targets(
        self,
        gt_labels: keras.KerasTensor,
        gt_bboxes: keras.KerasTensor,
        target_gt_idx: keras.KerasTensor,
        fg_mask: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Gathers targets for assigned anchors.

        Args:
            gt_labels: Ground truth labels.
            gt_bboxes: Ground truth bounding boxes.
            target_gt_idx: Indices of assigned ground truth targets.
            fg_mask: Foreground mask indicating positive anchors.

        Returns:
            Tuple of target bounding boxes and target scores.
        """
        # --- START CORRECTION: REVERTING TO THE ORIGINAL, WORKING `ops.take` LOGIC ---
        # The recommendation to use `ops.gather_nd` was incorrect as the function
        # does not exist in the public `keras.ops` API. The original implementation
        # using manual index flattening is the correct and functional approach.

        batch_size = ops.shape(target_gt_idx)[0]
        num_anchors = ops.shape(target_gt_idx)[1]
        max_gt = ops.shape(gt_labels)[1]

        # Create batch indices for gathering.
        batch_indices = ops.tile(
            ops.expand_dims(ops.arange(batch_size, dtype=target_gt_idx.dtype), 1),
            [1, num_anchors]
        )

        # Flatten indices for the `ops.take` operation.
        # This correctly maps each anchor to its assigned ground truth within the flattened tensor.
        flat_indices = ops.reshape(
            batch_indices * max_gt + target_gt_idx,
            [-1]
        )

        # Gather target labels by taking from the flattened ground truth tensor.
        target_labels = ops.reshape(
            ops.take(ops.reshape(gt_labels, [-1, 1]), flat_indices, axis=0),
            [batch_size, num_anchors, 1]
        )

        # Gather target bounding boxes.
        target_bboxes = ops.reshape(
            ops.take(ops.reshape(gt_bboxes, [-1, 4]), flat_indices, axis=0),
            [batch_size, num_anchors, 4]
        )

        # Convert labels to one-hot scores with foreground mask.
        target_labels_int = ops.cast(ops.squeeze(target_labels, -1), "int32")
        target_scores = (
            ops.one_hot(target_labels_int, self.num_classes) *
            ops.expand_dims(ops.cast(fg_mask, "float32"), -1)
        )

        return target_bboxes, target_scores
        # --- END CORRECTION ---

    def _dist_to_bbox(
        self,
        distance: keras.KerasTensor,
        anchors: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Converts distance predictions to bounding boxes.

        Args:
            distance: Predicted distances (left, top, right, bottom).
            anchors: Anchor points.

        Returns:
            Bounding boxes in (x1, y1, x2, y2) format.
        """
        lt, rb = ops.split(distance, 2, axis=-1)
        return ops.concatenate([anchors - lt, anchors + rb], -1)

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the YOLOv12 detection loss.

        Args:
            y_true: Ground truth tensor containing labels and bounding boxes.
            y_pred: Predicted tensor containing distance and classification scores.

        Returns:
            Computed loss value.
        """
        # Split predictions into distance and classification components
        pred_dist, pred_scores = ops.split(
            y_pred,
            [4 * self.reg_max],
            -1
        )

        # Convert distance predictions to bounding boxes
        pred_dist_reshaped = ops.reshape(
            pred_dist,
            [ops.shape(y_pred)[0], -1, 4, self.reg_max]
        )
        pred_dist_mean = ops.sum(
            ops.nn.softmax(pred_dist_reshaped, -1) *
            ops.arange(self.reg_max, dtype="float32"),
            -1
        )

        scaled_pred_dist = pred_dist_mean * self.strides

        pred_bboxes = self._dist_to_bbox(
            scaled_pred_dist,
            self.anchors * self.strides
        )

        # Extract ground truth labels and bounding boxes
        gt_labels, gt_bboxes_normalized = y_true[..., :1], y_true[..., 1:]

        patch_h, patch_w = self.input_shape
        scale_tensor = ops.cast([patch_w, patch_h, patch_w, patch_h], dtype=gt_bboxes_normalized.dtype)
        gt_bboxes = gt_bboxes_normalized * scale_tensor

        mask_gt = ops.sum(gt_bboxes, -1, keepdims=True) > 0

        def compute_losses():
            """
            Compute all loss components for a batch of images with ground truth.
            This function is called only when at least one ground truth box exists in the batch.
            It performs the following steps:
            1.  Assigns ground truth boxes to the most suitable anchor points.
            2.  Calculates classification loss for all assigned anchors.
            3.  Calculates bounding box (IoU) and Distribution Focal Loss (DFL) only for
                the positive anchor assignments (foreground).

            Returns:
                A tuple of three scalar tensors: (loss_cls, loss_box, loss_dfl).
            """
            # Step 1: Assign ground truth boxes to anchor points using Task-Aligned Assigner.
            # This determines which anchors are responsible for which ground truth objects (fg_mask)
            # and gets the index of the ground truth box assigned to each anchor (target_gt_idx).
            target_gt_idx, fg_mask = self._task_aligned_assigner(
                pred_scores, pred_bboxes, self.anchors * self.strides,
                gt_labels, gt_bboxes, mask_gt
            )

            # Step 2: Gather the target labels and bounding boxes for each anchor based on the assignment.
            # For background anchors, the target scores will be all zeros.
            target_bboxes, target_scores = self._get_targets(
                gt_labels, gt_bboxes, target_gt_idx, fg_mask
            )

            # Step 3: Calculate a normalization factor. This is the total number of positive
            # assignments across the batch, used to average the losses.
            target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)

            # Step 4: Calculate the Classification Loss (BCE Focal Loss).
            # This is calculated for all anchors (both foreground and background).
            # The background anchors are penalized for any confident predictions, while
            # foreground anchors are rewarded for correctly classifying their target.
            loss_cls = (
                    ops.sum(self.cfc_loss_fn(target_scores, pred_scores)) /
                    target_scores_sum
            )

            def compute_box_and_dfl_only():
                """
                A nested function to compute losses that only apply to foreground anchors.
                This is wrapped in an ops.cond to avoid running on batches with no positive assignments.

                Returns:
                    A tuple of two scalar tensors: (loss_box, loss_dfl).
                """
                # Get a boolean mask for only the positive (foreground) anchors.
                flat_fg_mask = ops.reshape(fg_mask, [-1])

                # Gather the predictions and targets for only the positive anchors.
                pred_bboxes_pos = ops.reshape(pred_bboxes, [-1, 4])[flat_fg_mask]
                target_bboxes_pos = ops.reshape(target_bboxes, [-1, 4])[flat_fg_mask]
                target_scores_pos = ops.reshape(target_scores, [-1, self.num_classes])[flat_fg_mask]

                # --- Bounding Box Regression Loss (CIoU Loss) ---
                iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
                loss_box = (
                        ops.sum((1.0 - iou) * ops.sum(target_scores_pos, -1)) /
                        target_scores_sum
                )

                # --- START DFL CORRECTION: FIXING TENSOR SHAPES ---
                # The previous implementation failed because the tensor shapes were incorrect
                # for sparse_categorical_crossentropy. We must flatten the LTRB dimension
                # into the batch dimension to properly compute the loss.

                # Get the coordinates and strides of the positive anchors.
                anchor_indices = ops.tile(ops.arange(ops.shape(pred_bboxes)[1]), [ops.shape(pred_bboxes)[0]])
                anchors_pos = ops.take(self.anchors * self.strides, anchor_indices[flat_fg_mask], 0)
                strides_pos = ops.take(ops.squeeze(self.strides), anchor_indices[flat_fg_mask], 0)
                strides_pos = ops.expand_dims(strides_pos, -1)

                # Calculate scaled ground truth distances. Shape: (num_pos, 4)
                target_ltrb_pixels = ops.concatenate([
                    anchors_pos - target_bboxes_pos[..., :2],
                    target_bboxes_pos[..., 2:] - anchors_pos
                ], -1)
                target_ltrb = target_ltrb_pixels / strides_pos
                target_ltrb = ops.clip(target_ltrb, 0, self.reg_max - 1.01)

                # Get integer bins and interpolation weights. Shape of all: (num_pos, 4)
                target_ltrb_left = ops.floor(target_ltrb)
                target_ltrb_right = target_ltrb_left + 1
                weight_left = target_ltrb_right - target_ltrb
                weight_right = target_ltrb - target_ltrb_left

                # Gather DFL predictions. Shape: (num_pos, 4, reg_max)
                pred_dist_pos = ops.reshape(pred_dist_reshaped, [-1, 4, self.reg_max])[flat_fg_mask]

                # Reshape all tensors to flatten the LTRB dimension into the batch dimension.
                # This creates the (batch_size, num_classes) structure required by the loss function.
                target_left_flat = ops.reshape(target_ltrb_left, [-1])  # Shape: (num_pos * 4,)
                target_right_flat = ops.reshape(target_ltrb_right, [-1])  # Shape: (num_pos * 4,)
                weight_left_flat = ops.reshape(weight_left, [-1])  # Shape: (num_pos * 4,)
                weight_right_flat = ops.reshape(weight_right, [-1])  # Shape: (num_pos * 4,)
                pred_dist_flat = ops.reshape(pred_dist_pos, [-1, self.reg_max])  # Shape: (num_pos * 4, reg_max)

                # Compute the cross-entropy loss for both the left and right bins with the correct shapes.
                loss_dfl_left = keras.losses.sparse_categorical_crossentropy(
                    ops.cast(target_left_flat, "int32"),
                    pred_dist_flat,
                    from_logits=True
                )
                loss_dfl_right = keras.losses.sparse_categorical_crossentropy(
                    ops.cast(target_right_flat, "int32"),
                    pred_dist_flat,
                    from_logits=True
                )

                # Combine the weighted losses and normalize.
                loss_dfl = (
                                   ops.sum(loss_dfl_left * weight_left_flat) +
                                   ops.sum(loss_dfl_right * weight_right_flat)
                           ) / target_scores_sum
                # --- END DFL CORRECTION ---

                return loss_box, loss_dfl

            # Step 5: Conditionally compute the box and DFL losses.
            # These are only computed if there is at least one positive anchor in the batch
            # to avoid errors with empty tensors.
            loss_box, loss_dfl = ops.cond(
                ops.sum(ops.cast(fg_mask, "float32")) > 0,
                compute_box_and_dfl_only,  # If true, call the function that returns 2 values
                lambda: (ops.convert_to_tensor(0.0), ops.convert_to_tensor(0.0))  # If false, return 2 zero tensors
            )

            # Step 6: Return all three loss components.
            return loss_cls, loss_box, loss_dfl

        # Compute losses only if ground truth exists
        loss_cls, loss_box, loss_dfl = ops.cond(
            ops.sum(ops.cast(mask_gt, "float32")) > 0,
            compute_losses,
            lambda: (0.0, 0.0, 0.0)
        )

        # Combine weighted losses
        total_loss = (
            self.box_weight * loss_box +
            self.cls_weight * loss_cls +
            self.dfl_weight * loss_dfl
        )

        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable config of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "reg_max": self.reg_max,
            "box_weight": self.box_weight,
            "cls_weight": self.cls_weight,
            "dfl_weight": self.dfl_weight,
            "assigner_alpha": self.assigner_alpha,
            "assigner_beta": self.assigner_beta
        })
        return config


@keras.saving.register_keras_serializable(package="yolov12_losses")
class DiceFocalSegmentationLoss(keras.losses.Loss):
    """
    Internal combined Dice and Focal Loss for segmentation.

    This loss combines the benefits of Dice loss (good for handling class imbalance)
    and Focal loss (focuses on hard examples) for semantic segmentation tasks.

    FIXED VERSION: Now supports both binary (1 class) and multi-class segmentation
    with proper shape handling and gradient flow.

    Args:
        num_classes: Number of segmentation classes (1 for binary, >1 for multi-class).
        dice_weight: Weight for the Dice loss component.
        focal_weight: Weight for the Focal loss component.
        focal_alpha: Alpha parameter for Focal loss.
        focal_gamma: Gamma parameter for Focal loss.
        from_logits: Whether predictions are logits or probabilities.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        num_classes: int = 1,  # NEW: Number of segmentation classes
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        from_logits: bool = True,
        name: str = "dice_focal_segmentation_loss_internal",
        **kwargs
    ):
        super().__init__(name=name, reduction="none", **kwargs)

        # Store configuration parameters
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.from_logits = from_logits

        # We'll implement the losses directly in the call method for better control

    def _compute_dice_loss(self, y_true, y_pred):
        """Compute Dice loss with consistent output shape."""
        # Apply sigmoid/softmax if from_logits
        if self.from_logits:
            if self.num_classes == 1:
                y_pred_prob = ops.sigmoid(y_pred)
            else:
                y_pred_prob = ops.softmax(y_pred, axis=-1)
        else:
            y_pred_prob = y_pred

        # Compute Dice loss
        if self.num_classes == 1:
            # Binary Dice loss
            intersection = ops.sum(y_true * y_pred_prob, axis=[1, 2])
            union = ops.sum(y_true, axis=[1, 2]) + ops.sum(y_pred_prob, axis=[1, 2])
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_loss = 1.0 - dice  # Shape: [batch_size]
        else:
            # Multi-class Dice loss - compute per class and average
            intersection = ops.sum(y_true * y_pred_prob, axis=[1, 2])  # [batch_size, num_classes]
            union = ops.sum(y_true, axis=[1, 2]) + ops.sum(y_pred_prob, axis=[1, 2])  # [batch_size, num_classes]
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)  # [batch_size, num_classes]
            dice_loss = ops.mean(1.0 - dice, axis=-1)  # Average over classes -> [batch_size]

        return dice_loss

    def _compute_focal_loss(self, y_true, y_pred):
        """Compute focal loss with consistent output shape."""
        if self.num_classes == 1:
            # Binary focal loss
            if self.from_logits:
                # Manually compute binary focal loss from logits
                p = ops.sigmoid(y_pred)
            else:
                p = y_pred

            # Clip to prevent log(0)
            p = ops.clip(p, 1e-7, 1.0 - 1e-7)

            # Compute focal weight
            alpha_t = y_true * self.focal_alpha + (1 - y_true) * (1 - self.focal_alpha)
            p_t = y_true * p + (1 - y_true) * (1 - p)
            focal_weight = alpha_t * ops.power(1 - p_t, self.focal_gamma)

            # Compute focal loss
            focal_loss = -focal_weight * (
                y_true * ops.log(p) + (1 - y_true) * ops.log(1 - p)
            )

            # Reduce spatial dimensions: [batch_size, height, width, 1] -> [batch_size]
            focal_loss = ops.mean(focal_loss, axis=[1, 2, 3])

        else:
            # Multi-class focal loss
            if self.from_logits:
                # Manually compute categorical focal loss from logits
                y_pred_softmax = ops.softmax(y_pred, axis=-1)
            else:
                y_pred_softmax = y_pred

            # Clip to prevent log(0)
            y_pred_softmax = ops.clip(y_pred_softmax, 1e-7, 1.0 - 1e-7)

            # Compute cross entropy
            ce_loss = -ops.sum(y_true * ops.log(y_pred_softmax), axis=-1)  # [batch_size, height, width]

            # Compute p_t for focal weighting
            p_t = ops.sum(y_true * y_pred_softmax, axis=-1)  # [batch_size, height, width]

            # Apply focal weighting
            focal_weight = ops.power(1 - p_t, self.focal_gamma)
            focal_loss = focal_weight * ce_loss

            # Reduce spatial dimensions: [batch_size, height, width] -> [batch_size]
            focal_loss = ops.mean(focal_loss, axis=[1, 2])

        return focal_loss

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the combined Dice and Focal loss.

        Args:
            y_true: Ground truth segmentation masks.
            y_pred: Predicted segmentation masks.

        Returns:
            Combined loss value with shape [batch_size].
        """
        # Handle shape compatibility for different number of classes
        if self.num_classes == 1:
            # Binary segmentation - ensure both tensors have same shape
            if y_pred.shape[-1] == 1 and len(y_true.shape) == 4 and y_true.shape[-1] == 1:
                # Both are single channel - proceed normally
                pass
            elif y_pred.shape[-1] == 1 and len(y_true.shape) == 3:
                # y_true has no channel dimension, add it
                y_true = ops.expand_dims(y_true, -1)
            else:
                logger.warning(f"Shape mismatch in binary segmentation: y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}")

        else:
            # Multi-class segmentation
            if len(y_true.shape) == 3:
                # Convert sparse labels to one-hot
                y_true = ops.one_hot(ops.cast(y_true, "int32"), self.num_classes)
            elif y_true.shape[-1] != self.num_classes:
                logger.warning(f"Multi-class segmentation shape mismatch: y_true.shape={y_true.shape}, expected classes={self.num_classes}")

        # Compute individual loss components
        dice_loss = self._compute_dice_loss(y_true, y_pred)  # [batch_size]
        focal_loss = self._compute_focal_loss(y_true, y_pred)  # [batch_size]

        # Both losses now have shape [batch_size], so we can combine them
        combined_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return combined_loss

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable config of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "from_logits": self.from_logits
        })
        return config


@keras.saving.register_keras_serializable(package="yolov12_losses")
class ClassificationFocalLoss(keras.losses.Loss):
    """
    Internal Focal Loss for image-level classification.

    Focal loss addresses class imbalance by down-weighting easy examples and
    focusing training on hard negatives.

    Args:
        alpha: Weighting factor for rare class (typically between 0 and 1).
        gamma: Focusing parameter (typically 2.0).
        from_logits: Whether predictions are logits or probabilities.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        name: str = "classification_focal_loss_internal",
        **kwargs
    ):
        super().__init__(name=name, reduction="none", **kwargs)

        # Initialize focal loss
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            alpha=alpha,
            gamma=gamma,
            from_logits=from_logits,
            apply_class_balancing=True,
            reduction="none"
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the focal loss for classification.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted classification scores.

        Returns:
            Focal loss value.
        """
        # Ensure y_true has the same shape as y_pred if it's a scalar target
        if len(y_true.shape) == 1:
            y_true = ops.expand_dims(y_true, -1)

        return self.focal_loss(y_true, y_pred)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable config of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "alpha": self.focal_loss.alpha,
            "gamma": self.focal_loss.gamma,
            "from_logits": self.focal_loss.from_logits
        })
        return config


# ---------------------------------------------------------------------
# Main Multi-Task Loss Orchestrator
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="yolov12_losses")
class YOLOv12MultiTaskLoss(keras.losses.Loss):
    """
    A single, "smart" loss function for YOLOv12 multi-task models.

    This class is designed to be the sole loss function passed to `model.compile`.
    It internally manages and routes to task-specific losses based on tensor shapes,
    and handles learnable uncertainty weighting.

    The loss function automatically infers which task-specific sub-loss to apply
    based on the tensor shapes of y_true and y_pred that Keras provides for each
    model output:
        - 3D tensors: Object detection (batch, anchors, features)
        - 4D tensors: Segmentation (batch, height, width, classes)
        - 2D tensors: Classification (batch, classes)

    FIXED VERSION: Now supports configurable detection and segmentation class counts.

    Args:
        tasks: Task configuration specifying which tasks to enable.
        num_detection_classes: Number of classes for detection task.
        num_segmentation_classes: Number of classes for segmentation task.
        num_classification_classes: Number of classes for classification task.
        num_classes: Fallback for backward compatibility.
        input_shape: Input image dimensions as (height, width).
        use_uncertainty_weighting: Whether to use learnable uncertainty weighting.
        detection_weight: Static weight for detection loss (ignored if uncertainty weighting is used).
        segmentation_weight: Static weight for segmentation loss (ignored if uncertainty weighting is used).
        classification_weight: Static weight for classification loss (ignored if uncertainty weighting is used).
        reg_max: Maximum regression range for distribution focal loss.
        name: Name of the loss function.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        tasks: Union[TaskConfiguration, List[str], List[TaskType]],
        # Separate class counts for each task
        num_detection_classes: Optional[int] = None,
        num_segmentation_classes: Optional[int] = None,
        num_classification_classes: Optional[int] = None,
        # Backward compatibility
        num_classes: int = 80,
        input_shape: Tuple[int, int] = (640, 640),
        use_uncertainty_weighting: bool = False,
        detection_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        classification_weight: float = 1.0,
        reg_max: int = 16,
        name: str = "yolov12_multitask_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Parse and store task configuration
        self.task_config = parse_task_list(tasks)

        # Configure class counts
        self.num_detection_classes = num_detection_classes if num_detection_classes is not None else num_classes
        self.num_segmentation_classes = num_segmentation_classes if num_segmentation_classes is not None else num_classes
        self.num_classification_classes = num_classification_classes if num_classification_classes is not None else num_classes

        # Store other configuration
        self.num_classes = num_classes  # Backward compatibility
        self.input_shape = input_shape
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.reg_max = reg_max

        # Initialize internal loss storage
        self._internal_losses = {}

        # Build task-specific loss functions
        self._build_loss_functions()

        # Build uncertainty weights if enabled
        if self.use_uncertainty_weighting:
            self._build_uncertainty_weights()

        # Log initialization
        logger.info(
            f"YOLOv12MultiTaskLoss initialized for tasks: "
            f"{self.task_config.get_task_names()}. "
            f"Uncertainty weighting: {self.use_uncertainty_weighting}"
        )
        logger.info(f"Class counts - Detection: {self.num_detection_classes}, "
                   f"Segmentation: {self.num_segmentation_classes}, "
                   f"Classification: {self.num_classification_classes}")

    def _build_loss_functions(self) -> None:
        """Instantiate internal, task-specific loss functions."""
        if self.task_config.has_detection():
            self._internal_losses[TaskType.DETECTION.value] = YOLOv12ObjectDetectionLoss(
                num_classes=self.num_detection_classes,
                input_shape=self.input_shape,
                reg_max=self.reg_max
            )

        if self.task_config.has_segmentation():
            self._internal_losses[TaskType.SEGMENTATION.value] = DiceFocalSegmentationLoss(
                num_classes=self.num_segmentation_classes,  # Use segmentation-specific class count
                from_logits=True
            )

        if self.task_config.has_classification():
            self._internal_losses[TaskType.CLASSIFICATION.value] = ClassificationFocalLoss(
                from_logits=True
            )

    def _build_uncertainty_weights(self) -> None:
        """Create learnable log-variance weights for each enabled task."""
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

    def _infer_task_from_shapes(
        self,
        y_pred: keras.KerasTensor
    ) -> Optional[str]:
        """
        Infers the current task by inspecting the rank of the prediction tensor.

        This method relies on the architectural assumption that each task output has a
        unique tensor rank:
            - Rank 3 (batch, anchors, features): Object Detection
            - Rank 4 (batch, height, width, classes): Segmentation
            - Rank 2 (batch, classes): Classification

        If future tasks are added that collide in rank (e.g., another 4D tensor task),
        this inference mechanism will need to be replaced with a more explicit one.

        Args:
            y_pred: Prediction tensor from a model output.

        Returns:
            The corresponding task name string or None if the rank is unrecognized.
        """
        pred_shape = y_pred.shape

        if len(pred_shape) == 3:
            return TaskType.DETECTION.value
        elif len(pred_shape) == 4:
            return TaskType.SEGMENTATION.value
        elif len(pred_shape) == 2:
            return TaskType.CLASSIFICATION.value

        return None

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Calculates the loss for a single task output.

        This method is called by Keras for each model output. It automatically
        routes to the appropriate task-specific loss based on tensor shapes.

        Args:
            y_true: Ground truth tensor.
            y_pred: Prediction tensor.

        Returns:
            Computed loss value for the current task.
        """
        # Infer task from prediction tensor shape
        task_name = self._infer_task_from_shapes(y_pred)

        # Return zero loss if task cannot be inferred or is not supported
        if task_name is None or task_name not in self._internal_losses:
            return ops.convert_to_tensor(0.0, dtype=y_pred.dtype)

        # Compute raw loss using task-specific loss function
        raw_loss = self._internal_losses[task_name](y_true, y_pred)

        # Apply weighting (either learnable uncertainty or static)
        if self.use_uncertainty_weighting:
            if task_name == TaskType.DETECTION.value:
                return (
                    ops.exp(-self.detection_log_var) * raw_loss +
                    self.detection_log_var
                )
            elif task_name == TaskType.SEGMENTATION.value:
                return (
                    ops.exp(-self.segmentation_log_var) * raw_loss +
                    self.segmentation_log_var
                )
            else:  # Classification
                return (
                    ops.exp(-self.classification_log_var) * raw_loss +
                    self.classification_log_var
                )
        else:
            # Apply static weights
            if task_name == TaskType.DETECTION.value:
                return self.detection_weight * raw_loss
            elif task_name == TaskType.SEGMENTATION.value:
                return self.segmentation_weight * raw_loss
            else:  # Classification
                return self.classification_weight * raw_loss

    def get_task_weights(self) -> Dict[str, float]:
        """
        Returns the current weights for each task, for callback logging.

        Returns:
            Dictionary mapping task names to their current weights.
        """
        weights = {}

        if self.use_uncertainty_weighting:
            # Extract weights from learnable uncertainty parameters
            if self.task_config.has_detection():
                weights[TaskType.DETECTION.value] = float(
                    ops.exp(-self.detection_log_var)
                )
            if self.task_config.has_segmentation():
                weights[TaskType.SEGMENTATION.value] = float(
                    ops.exp(-self.segmentation_log_var)
                )
            if self.task_config.has_classification():
                weights[TaskType.CLASSIFICATION.value] = float(
                    ops.exp(-self.classification_log_var)
                )
        else:
            # Return static weights
            if self.task_config.has_detection():
                weights[TaskType.DETECTION.value] = self.detection_weight
            if self.task_config.has_segmentation():
                weights[TaskType.SEGMENTATION.value] = self.segmentation_weight
            if self.task_config.has_classification():
                weights[TaskType.CLASSIFICATION.value] = self.classification_weight

        return weights

    def get_individual_losses(self) -> Dict[str, float]:
        """
        Provides API compatibility for the callback.

        Returns:
            Empty dictionary (individual losses are computed during training).
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable config of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "tasks": self.task_config.get_task_names(),
            # Separate class counts
            "num_detection_classes": self.num_detection_classes,
            "num_segmentation_classes": self.num_segmentation_classes,
            "num_classification_classes": self.num_classification_classes,
            # Backward compatibility
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "use_uncertainty_weighting": self.use_uncertainty_weighting,
            "detection_weight": self.detection_weight,
            "segmentation_weight": self.segmentation_weight,
            "classification_weight": self.classification_weight,
            "reg_max": self.reg_max,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a loss instance from its config.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            YOLOv12MultiTaskLoss instance.
        """
        return cls(**config)


# ---------------------------------------------------------------------
# Public-Facing Factory Function
# ---------------------------------------------------------------------

def create_yolov12_multitask_loss(
    tasks: Union[List[TaskType], List[str], TaskConfiguration],
    num_detection_classes: Optional[int] = None,
    num_segmentation_classes: Optional[int] = None,
    num_classification_classes: Optional[int] = None,
    num_classes: int = 80,  # Backward compatibility
    input_shape: Tuple[int, int] = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Factory function to create the YOLOv12MultiTaskLoss instance.

    This is the intended entry point for creating the loss function in the
    training script. It provides a clean interface for instantiating the
    multi-task loss with the correct configuration.

    Args:
        tasks: Task configuration specifying which tasks to enable.
        num_detection_classes: Number of classes for detection task.
        num_segmentation_classes: Number of classes for segmentation task.
        num_classification_classes: Number of classes for classification task.
        num_classes: Fallback class count for backward compatibility.
        input_shape: Input image dimensions as (height, width).
        **kwargs: Additional keyword arguments passed to YOLOv12MultiTaskLoss.

    Returns:
        Configured YOLOv12MultiTaskLoss instance ready for use in model.compile().

    Example:
        >>> # COCO pretraining
        >>> loss_fn = create_yolov12_multitask_loss(
        ...     tasks=['detection', 'segmentation'],
        ...     num_detection_classes=80,
        ...     num_segmentation_classes=80,
        ...     input_shape=(640, 640),
        ...     use_uncertainty_weighting=True
        ... )
        >>>
        >>> # Crack detection fine-tuning
        >>> loss_fn = create_yolov12_multitask_loss(
        ...     tasks=['detection', 'segmentation'],
        ...     num_detection_classes=1,
        ...     num_segmentation_classes=1,
        ...     input_shape=(640, 640)
        ... )
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    return YOLOv12MultiTaskLoss(
        tasks=tasks,
        num_detection_classes=num_detection_classes,
        num_segmentation_classes=num_segmentation_classes,
        num_classification_classes=num_classification_classes,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )


def create_yolov12_coco_loss(
    input_shape: Tuple[int, int] = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Create loss function specifically for COCO pretraining.

    Args:
        input_shape: Input image dimensions.
        **kwargs: Additional arguments.

    Returns:
        Loss function configured for COCO pretraining.
    """
    return create_yolov12_multitask_loss(
        tasks=['detection', 'segmentation'],
        num_detection_classes=80,
        num_segmentation_classes=80,
        input_shape=input_shape,
        **kwargs
    )


def create_yolov12_crack_loss(
    input_shape: Tuple[int, int] = (640, 640),
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Create loss function specifically for crack detection fine-tuning.

    Args:
        input_shape: Input image dimensions.
        **kwargs: Additional arguments.

    Returns:
        Loss function configured for crack detection.
    """
    return create_yolov12_multitask_loss(
        tasks=['detection', 'segmentation'],
        num_detection_classes=1,
        num_segmentation_classes=1,
        input_shape=input_shape,
        **kwargs
    )