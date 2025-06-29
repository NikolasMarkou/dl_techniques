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

Key Components:
    - YOLOv12MultiTaskLoss: The main, user-facing loss class that orchestrates all
      sub-losses and handles dynamic or static weighting. It is the single entry
      point for the training script.

    - YOLOv12ObjectDetectionLoss, DiceFocalSegmentationLoss, ClassificationFocalLoss:
      Internal, task-specific loss classes used by the main orchestrator.

    - Factory Function: A convenience function `create_yolov12_multitask_loss` to
      easily instantiate the main loss function with the correct configuration.

Usage in `train.py`::

    # 1. Create the model and a single loss instance using the factory.
    model, loss_fn = create_model_and_loss(
        task_config=task_config,
        patch_size=args.patch_size,
        scale=args.scale,
        use_uncertainty_weighting=args.uncertainty_weighting
    )

    # 2. Compile the model, passing the single loss instance.
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,  # A single loss object, not a dictionary.
        run_eagerly=args.run_eagerly
    )

    # 3. Create callbacks that can interact with the loss function.
    callbacks = create_callbacks(
        loss_fn=loss_fn,
        # ...
    )

    # 4. Fit the model. Keras will call the loss_fn for each named output,
    #    and the loss_fn will route to the correct sub-loss internally.
    model.fit(...)
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

        # Initialize binary cross-entropy loss for classification
        self.bce = keras.losses.BinaryCrossentropy(
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
        Vectorized Task-Aligned Assigner for optimal anchor-target matching.

        Args:
            pred_scores: Predicted classification scores.
            pred_bboxes: Predicted bounding boxes.
            anchors: Anchor points.
            gt_labels: Ground truth labels.
            gt_bboxes: Ground truth bounding boxes.
            mask_gt: Mask for valid ground truth boxes.

        Returns:
            Tuple of target ground truth indices and foreground mask.
        """
        num_anchors = ops.shape(pred_scores)[1]

        # Expand dimensions for broadcasting
        anchors_exp = ops.reshape(anchors, (1, 1, num_anchors, 2))
        gt_bboxes_exp = ops.expand_dims(gt_bboxes, 2)

        # Check if anchors are inside ground truth boxes
        gt_x1y1, gt_x2y2 = ops.split(gt_bboxes_exp, 2, axis=-1)
        is_in_gt = (
            ops.all(anchors_exp >= gt_x1y1, -1) &
            ops.all(anchors_exp <= gt_x2y2, -1)
        )

        # Calculate IoU between predictions and ground truth
        ious = bbox_iou(
            ops.expand_dims(pred_bboxes, 1),
            gt_bboxes_exp,
            xywh=False,
            CIoU=True
        )

        # Convert labels to one-hot encoding
        gt_labels_int = ops.cast(ops.squeeze(gt_labels, -1), "int32")
        gt_labels_one_hot = ops.one_hot(gt_labels_int, self.num_classes)

        # Calculate classification scores for alignment metric
        cls_scores = ops.sum(
            ops.expand_dims(gt_labels_one_hot, 2) *
            ops.nn.sigmoid(ops.expand_dims(pred_scores, 1)),
            -1
        )

        # Compute task-aligned assignment metric
        align_metric = (
            ops.power(cls_scores, self.assigner_alpha) *
            ops.power(ious, self.assigner_beta)
        )

        # Apply ground truth mask
        mask_gt_broadcast = ops.cast(
            ops.expand_dims(ops.squeeze(mask_gt, -1), -1),
            "bool"
        )
        align_metric = ops.where(
            is_in_gt & mask_gt_broadcast,
            align_metric,
            0.0
        )

        # Find best matching ground truth for each anchor
        target_gt_idx = ops.argmax(align_metric, axis=1)
        fg_mask = ops.max(align_metric, axis=1) > 0

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
        batch_size = ops.shape(target_gt_idx)[0]
        num_anchors = ops.shape(target_gt_idx)[1]
        max_gt = ops.shape(gt_labels)[1]

        # Create batch indices for gathering
        batch_indices = ops.tile(
            ops.expand_dims(ops.arange(batch_size), 1),
            [1, num_anchors]
        )

        # Flatten indices for gathering operation
        flat_indices = ops.reshape(
            batch_indices * max_gt + target_gt_idx,
            [-1]
        )

        # Gather target labels
        target_labels = ops.reshape(
            ops.take(ops.reshape(gt_labels, [-1, 1]), flat_indices, 0),
            [batch_size, num_anchors, 1]
        )

        # Gather target bounding boxes
        target_bboxes = ops.reshape(
            ops.take(ops.reshape(gt_bboxes, [-1, 4]), flat_indices, 0),
            [batch_size, num_anchors, 4]
        )

        # Convert labels to one-hot scores with foreground mask
        target_labels_int = ops.cast(ops.squeeze(target_labels, -1), "int32")
        target_scores = (
            ops.one_hot(target_labels_int, self.num_classes) *
            ops.expand_dims(ops.cast(fg_mask, "float32"), -1)
        )

        return target_bboxes, target_scores

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
        pred_bboxes = self._dist_to_bbox(
            pred_dist_mean,
            self.anchors * self.strides
        )

        # Extract ground truth labels and bounding boxes
        gt_labels, gt_bboxes = y_true[..., :1], y_true[..., 1:]
        mask_gt = ops.sum(gt_bboxes, -1, keepdims=True) > 0

        def compute_losses():
            """Compute all loss components when ground truth exists."""
            # Perform task-aligned assignment
            target_gt_idx, fg_mask = self._task_aligned_assigner(
                pred_scores, pred_bboxes, self.anchors * self.strides,
                gt_labels, gt_bboxes, mask_gt
            )

            # Get targets for assigned anchors
            target_bboxes, target_scores = self._get_targets(
                gt_labels, gt_bboxes, target_gt_idx, fg_mask
            )

            # Calculate normalization factor
            target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)

            # Classification loss
            loss_cls = (
                ops.sum(self.bce(target_scores, pred_scores)) /
                target_scores_sum
            )

            def compute_box_dfl_losses():
                """Compute box regression and DFL losses for positive anchors."""
                # Get positive anchor indices
                flat_fg_mask = ops.reshape(fg_mask, [-1])

                # Extract positive predictions and targets
                pred_bboxes_pos = ops.reshape(pred_bboxes, [-1, 4])[flat_fg_mask]
                target_bboxes_pos = ops.reshape(target_bboxes, [-1, 4])[flat_fg_mask]
                target_scores_pos = ops.reshape(target_scores, [-1, self.num_classes])[flat_fg_mask]

                # Box regression loss (CIoU)
                iou = bbox_iou(
                    pred_bboxes_pos,
                    target_bboxes_pos,
                    xywh=False,
                    CIoU=True
                )
                loss_box = (
                    ops.sum((1.0 - iou) * ops.sum(target_scores_pos, -1)) /
                    target_scores_sum
                )

                # Distribution Focal Loss (DFL) computation
                anchor_indices = ops.tile(
                    ops.arange(ops.shape(pred_bboxes)[1]),
                    [ops.shape(pred_bboxes)[0]]
                )
                anchors_pos = ops.take(
                    self.anchors * self.strides,
                    anchor_indices[flat_fg_mask],
                    0
                )

                # Convert to left-top-right-bottom format and clip
                target_ltrb = ops.concatenate([
                    anchors_pos - target_bboxes_pos[..., :2],
                    target_bboxes_pos[..., 2:] - anchors_pos
                ], -1)
                target_ltrb = ops.clip(target_ltrb, 0, self.reg_max - 1.01)

                # Prepare tensors for DFL loss
                pred_dist_pos = ops.reshape(
                    pred_dist_reshaped,
                    [-1, 4, self.reg_max]
                )[flat_fg_mask]
                target_ltrb_flat = ops.reshape(target_ltrb, [-1])
                pred_dist_flat = ops.reshape(pred_dist_pos, [-1, self.reg_max])

                # Compute DFL loss
                loss_dfl = (
                    ops.sum(keras.losses.sparse_categorical_crossentropy(
                        ops.cast(target_ltrb_flat, "int32"),
                        pred_dist_flat,
                        from_logits=True
                    )) / target_scores_sum
                )

                return loss_box, loss_dfl

            # Compute box and DFL losses only if there are positive anchors
            loss_box, loss_dfl = ops.cond(
                ops.sum(ops.cast(fg_mask, "float32")) > 0,
                compute_box_dfl_losses,
                lambda: (0.0, 0.0)
            )

            return loss_cls, loss_box, loss_dfl

        # Compute losses only if ground truth exists
        loss_cls, loss_box, loss_dfl = ops.cond(
            ops.sum(ops.cast(mask_gt, "float32")) > 0,
            compute_losses,
            lambda: (0.0, 0.0, 0.0)
        )

        # Combine weighted losses and normalize by ground truth count
        num_gts = ops.sum(ops.cast(mask_gt, "float32"))
        total_loss = (
            self.box_weight * loss_box +
            self.cls_weight * loss_cls +
            self.dfl_weight * loss_dfl
        )

        return total_loss / (num_gts + 1e-8)

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

    Args:
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
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Initialize component losses
        self.dice_loss = keras.losses.Dice(axis=[1, 2], reduction="none")
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(
            alpha=focal_alpha,
            gamma=focal_gamma,
            from_logits=from_logits,
            reduction="none"
        )

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
            Combined loss value.
        """
        # Compute individual loss components
        dice_loss = self.dice_loss(y_true, y_pred)
        focal_loss = ops.mean(self.focal_loss(y_true, y_pred), axis=[1, 2])

        # Return weighted combination
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable config of the loss.

        Returns:
            Dictionary containing the loss configuration.
        """
        config = super().get_config()
        config.update({
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "focal_alpha": self.focal_loss.alpha,
            "focal_gamma": self.focal_loss.gamma,
            "from_logits": self.focal_loss.from_logits
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

    Args:
        tasks: Task configuration specifying which tasks to enable.
        num_classes: Number of classes for detection/classification tasks.
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
        num_classes: int,
        input_shape: Tuple[int, int],
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
        self.num_classes = num_classes
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

    def _build_loss_functions(self) -> None:
        """Instantiate internal, task-specific loss functions."""
        if self.task_config.has_detection():
            self._internal_losses[TaskType.DETECTION.value] = YOLOv12ObjectDetectionLoss(
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                reg_max=self.reg_max
            )

        if self.task_config.has_segmentation():
            self._internal_losses[TaskType.SEGMENTATION.value] = DiceFocalSegmentationLoss(
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
        Infers the current task by inspecting the shape of the prediction tensor.

        Args:
            y_pred: Prediction tensor from model output.

        Returns:
            Task name string or None if task cannot be inferred.
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
    num_classes: int,
    input_shape: Tuple[int, int],
    **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Factory function to create the YOLOv12MultiTaskLoss instance.

    This is the intended entry point for creating the loss function in the
    training script. It provides a clean interface for instantiating the
    multi-task loss with the correct configuration.

    Args:
        tasks: Task configuration specifying which tasks to enable.
        num_classes: Number of classes for detection/classification tasks.
        input_shape: Input image dimensions as (height, width).
        **kwargs: Additional keyword arguments passed to YOLOv12MultiTaskLoss.

    Returns:
        Configured YOLOv12MultiTaskLoss instance ready for use in model.compile().

    Example:
        >>> loss_fn = create_yolov12_multitask_loss(
        ...     tasks=['detection', 'segmentation'],
        ...     num_classes=80,
        ...     input_shape=(640, 640),
        ...     use_uncertainty_weighting=True
        ... )
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """
    return YOLOv12MultiTaskLoss(
        tasks=tasks,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )