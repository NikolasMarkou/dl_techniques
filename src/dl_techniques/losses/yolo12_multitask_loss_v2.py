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

Usage in `train.py`:
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
    TaskType, TaskConfiguration, parse_task_list)


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
        num_classes: Number of object classes.
        input_shape: Model input shape (height, width) for anchor generation.
        reg_max: Maximum value for DFL regression.
        box_weight: Weight for bounding box loss component.
        cls_weight: Weight for classification loss component.
        dfl_weight: Weight for DFL loss component.
        name: Loss function name.
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
        # We always use 'none' reduction internally, as the outer loss handles it.
        super().__init__(name=name, reduction="none", **kwargs)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner_alpha = assigner_alpha
        self.assigner_beta = assigner_beta

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.anchors, self.strides = self._make_anchors()

    def _make_anchors(self, grid_cell_offset: float = 0.5) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Generates anchor points and strides for all feature map levels."""
        H, W = self.input_shape
        strides_config = [8, 16, 32]  # Strides for P3, P4, P5
        anchor_points, stride_tensor = [], []

        for stride in strides_config:
            h, w = H // stride, W // stride
            x_coords = ops.arange(w, dtype="float32") + grid_cell_offset
            y_coords = ops.arange(h, dtype="float32") + grid_cell_offset
            y_grid, x_grid = ops.meshgrid(y_coords, x_coords, indexing="ij")
            xy_grid = ops.stack([x_grid, y_grid], axis=-1)
            xy_grid = ops.reshape(xy_grid, (-1, 2))
            anchor_points.append(xy_grid)
            stride_tensor.append(ops.full((h * w, 1), stride, dtype="float32"))

        return ops.concatenate(anchor_points, 0), ops.concatenate(stride_tensor, 0)

    def _task_aligned_assigner(self, pred_scores, pred_bboxes, anchors, gt_labels, gt_bboxes, mask_gt):
        """Vectorized Task-Aligned Assigner."""
        num_anchors = ops.shape(pred_scores)[1]
        anchors_exp = ops.reshape(anchors, (1, 1, num_anchors, 2))
        gt_bboxes_exp = ops.expand_dims(gt_bboxes, 2)
        gt_x1y1, gt_x2y2 = ops.split(gt_bboxes_exp, 2, axis=-1)

        # Check if anchor centers are inside GT boxes
        is_in_gt = ops.all(anchors_exp >= gt_x1y1, axis=-1) & ops.all(anchors_exp <= gt_x2y2, axis=-1)
        ious = bbox_iou(ops.expand_dims(pred_bboxes, 1), gt_bboxes_exp, xywh=False, CIoU=True)

        gt_labels_int = ops.cast(ops.squeeze(gt_labels, -1), "int32")
        gt_labels_one_hot = ops.one_hot(gt_labels_int, self.num_classes)
        cls_scores = ops.sum(ops.expand_dims(gt_labels_one_hot, 2) * ops.nn.sigmoid(ops.expand_dims(pred_scores, 1)), axis=-1)

        align_metric = ops.power(cls_scores, self.assigner_alpha) * ops.power(ious, self.assigner_beta)
        mask_gt_broadcast = ops.cast(ops.expand_dims(ops.squeeze(mask_gt, -1), -1), "bool")
        align_metric = ops.where(is_in_gt & mask_gt_broadcast, align_metric, 0.)

        target_gt_idx = ops.argmax(align_metric, axis=1)
        fg_mask = ops.max(align_metric, axis=1) > 0
        return target_gt_idx, fg_mask

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Gathers targets for assigned anchors."""
        batch_size, num_anchors, max_gt = ops.shape(target_gt_idx)[0], ops.shape(target_gt_idx)[1], ops.shape(gt_labels)[1]
        batch_indices = ops.tile(ops.expand_dims(ops.arange(batch_size), 1), [1, num_anchors])
        flat_indices = ops.reshape(batch_indices * max_gt + target_gt_idx, [-1])

        target_labels = ops.reshape(ops.take(ops.reshape(gt_labels, [-1, 1]), flat_indices, 0), [batch_size, num_anchors, 1])
        target_bboxes = ops.reshape(ops.take(ops.reshape(gt_bboxes, [-1, 4]), flat_indices, 0), [batch_size, num_anchors, 4])

        target_labels_int = ops.cast(ops.squeeze(target_labels, -1), "int32")
        target_scores = ops.one_hot(target_labels_int, self.num_classes) * ops.expand_dims(ops.cast(fg_mask, "float32"), -1)
        return target_bboxes, target_scores

    def _dist_to_bbox(self, distance, anchors):
        """Converts distance predictions to bounding boxes."""
        lt, rb = ops.split(distance, 2, axis=-1)
        return ops.concatenate([anchors - lt, anchors + rb], axis=-1)

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        # Decode predictions
        pred_dist, pred_scores = ops.split(y_pred, [4 * self.reg_max], axis=-1)
        pred_dist_reshaped = ops.reshape(pred_dist, [ops.shape(y_pred)[0], -1, 4, self.reg_max])
        pred_dist_mean = ops.sum(ops.nn.softmax(pred_dist_reshaped, axis=-1) * ops.arange(self.reg_max, dtype="float32"), axis=-1)
        pred_bboxes = self._dist_to_bbox(pred_dist_mean, self.anchors * self.strides)

        # Prepare ground truth
        gt_labels, gt_bboxes = y_true[..., :1], y_true[..., 1:]
        mask_gt = ops.sum(gt_bboxes, axis=-1, keepdims=True) > 0

        def compute_losses():
            target_gt_idx, fg_mask = self._task_aligned_assigner(pred_scores, pred_bboxes, self.anchors * self.strides, gt_labels, gt_bboxes, mask_gt)
            target_bboxes, target_scores = self._get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

            # Classification loss
            target_scores_sum = ops.maximum(ops.sum(target_scores), 1.0)
            loss_cls = ops.sum(self.bce(target_scores, pred_scores)) / target_scores_sum

            # Bbox and DFL losses for foreground samples
            def compute_box_dfl_losses():
                flat_fg_mask = ops.reshape(fg_mask, [-1])
                pred_bboxes_pos = ops.reshape(pred_bboxes, [-1, 4])[flat_fg_mask]
                target_bboxes_pos = ops.reshape(target_bboxes, [-1, 4])[flat_fg_mask]
                target_scores_pos = ops.reshape(target_scores, [-1, self.num_classes])[flat_fg_mask]

                # Bbox loss (CIoU)
                iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
                loss_box = ops.sum((1.0 - iou) * ops.sum(target_scores_pos, -1)) / target_scores_sum

                # DFL loss
                anchor_indices = ops.tile(ops.arange(ops.shape(pred_bboxes)[1]), [ops.shape(pred_bboxes)[0]])
                anchors_pos = ops.take(self.anchors * self.strides, anchor_indices[flat_fg_mask], 0)
                target_ltrb = ops.concatenate([anchors_pos - target_bboxes_pos[..., :2], target_bboxes_pos[..., 2:] - anchors_pos], axis=-1)
                target_ltrb = ops.clip(target_ltrb, 0, self.reg_max - 1.01)
                pred_dist_pos = ops.reshape(pred_dist_reshaped, [-1, 4, self.reg_max])[flat_fg_mask]
                loss_dfl = ops.sum(keras.losses.sparse_categorical_crossentropy(ops.cast(target_ltrb, "int32"), ops.reshape(pred_dist_pos, [-1, self.reg_max]), from_logits=True)) / target_scores_sum
                return loss_box, loss_dfl

            loss_box, loss_dfl = ops.cond(ops.sum(ops.cast(fg_mask, "float32")) > 0, compute_box_dfl_losses, lambda: (0., 0.))
            return loss_cls, loss_box, loss_dfl

        # Return zero loss if no ground truths are present in the batch, avoids NaNs
        loss_cls, loss_box, loss_dfl = ops.cond(ops.sum(ops.cast(mask_gt, "float32")) > 0, compute_losses, lambda: (0., 0., 0.))

        # Return the weighted sum of the loss components
        total_loss = self.box_weight * loss_box + self.cls_weight * loss_cls + self.dfl_weight * loss_dfl
        return total_loss

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the loss."""
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "input_shape": self.input_shape, "reg_max": self.reg_max, "box_weight": self.box_weight, "cls_weight": self.cls_weight, "dfl_weight": self.dfl_weight, "assigner_alpha": self.assigner_alpha, "assigner_beta": self.assigner_beta})
        return config


@keras.saving.register_keras_serializable(package="yolov12_losses")
class DiceFocalSegmentationLoss(keras.losses.Loss):
    """Internal combined Dice and Focal Loss for segmentation."""
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5, focal_alpha: float = 0.25, focal_gamma: float = 2.0, from_logits: bool = True, name: str = "dice_focal_segmentation_loss_internal", **kwargs):
        super().__init__(name=name, reduction="none", **kwargs)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = keras.losses.Dice(axis=[1, 2], reduction="none")
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(alpha=focal_alpha, gamma=focal_gamma, from_logits=from_logits, reduction="none")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        dice_loss = self.dice_loss(y_true, y_pred)
        focal_loss = ops.mean(self.focal_loss(y_true, y_pred), axis=[1, 2])
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"dice_weight": self.dice_weight, "focal_weight": self.focal_weight, "focal_alpha": self.focal_loss.alpha, "focal_gamma": self.focal_loss.gamma, "from_logits": self.focal_loss.from_logits})
        return config


@keras.saving.register_keras_serializable(package="yolov12_losses")
class ClassificationFocalLoss(keras.losses.Loss):
    """Internal Focal Loss for image-level classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, from_logits: bool = True, name: str = "classification_focal_loss_internal", **kwargs):
        super().__init__(name=name, reduction="none", **kwargs)
        self.focal_loss = keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma, from_logits=from_logits, reduction="none")

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        return self.focal_loss(y_true, y_pred)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"alpha": self.focal_loss.alpha, "gamma": self.focal_loss.gamma, "from_logits": self.focal_loss.from_logits})
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
    """

    def __init__(
        self,
        # Task configuration
        task_config: Union[TaskConfiguration, List[str], List[TaskType]],
        num_classes: int,
        input_shape: Tuple[int, int],
        # Loss weighting
        use_uncertainty_weighting: bool = False,
        detection_weight: float = 1.0,
        segmentation_weight: float = 1.0,
        classification_weight: float = 1.0,
        # Sub-loss parameters
        reg_max: int = 16,
        # Base parameters
        name: str = "yolov12_multitask_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        # Store all configuration parameters for serialization
        self.task_config = parse_task_list(task_config)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight
        self.reg_max = reg_max
        self._internal_losses = {}

        # Build internal loss functions based on the task configuration
        self._build_loss_functions()

        # Build learnable uncertainty weights if enabled
        if self.use_uncertainty_weighting:
            self._build_uncertainty_weights()

        enabled_tasks = self.task_config.get_task_names()
        logger.info(f"YOLOv12MultiTaskLoss initialized for tasks: {enabled_tasks}. Uncertainty weighting: {self.use_uncertainty_weighting}")

    def _build_loss_functions(self) -> None:
        """Instantiate internal, task-specific loss functions."""
        if self.task_config.has_detection():
            self._internal_losses[TaskType.DETECTION.value] = YOLOv12ObjectDetectionLoss(
                num_classes=self.num_classes,
                input_shape=self.input_shape,
                reg_max=self.reg_max
            )
        if self.task_config.has_segmentation():
            self._internal_losses[TaskType.SEGMENTATION.value] = DiceFocalSegmentationLoss(from_logits=True)
        if self.task_config.has_classification():
            self._internal_losses[TaskType.CLASSIFICATION.value] = ClassificationFocalLoss(from_logits=True)

    def _build_uncertainty_weights(self) -> None:
        """Create learnable log-variance weights for each enabled task."""
        if self.task_config.has_detection():
            self.detection_log_var = self.add_weight(name="detection_log_var", shape=(), initializer="zeros", trainable=True)
        if self.task_config.has_segmentation():
            self.segmentation_log_var = self.add_weight(name="segmentation_log_var", shape=(), initializer="zeros", trainable=True)
        if self.task_config.has_classification():
            self.classification_log_var = self.add_weight(name="classification_log_var", shape=(), initializer="zeros", trainable=True)

    def _infer_task_from_shapes(self, y_pred: keras.KerasTensor) -> Optional[str]:
        """
        Infers the current task by inspecting the shape of the prediction tensor.

        This is the core of the internal routing mechanism. It relies on the assumption
        that each task head produces a uniquely shaped output tensor.
        """
        pred_shape = y_pred.shape
        # Detection: (batch, num_anchors, num_classes + 4 * reg_max) -> 3 dims
        if len(pred_shape) == 3:
            return TaskType.DETECTION.value
        # Segmentation: (batch, height, width, 1) -> 4 dims
        elif len(pred_shape) == 4:
            return TaskType.SEGMENTATION.value
        # Classification: (batch, num_classes) -> 2 dims
        elif len(pred_shape) == 2:
            return TaskType.CLASSIFICATION.value
        return None

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Calculates the loss for a single task output.

        Keras calls this method separately for each named output of the model.
        """
        # 1. Infer which task this call is for
        task_name = self._infer_task_from_shapes(y_pred)
        if task_name is None or task_name not in self._internal_losses:
            logger.warning(f"Could not infer task or find loss for pred_shape: {y_pred.shape}. Returning 0 loss.")
            return ops.convert_to_tensor(0.0, dtype=y_pred.dtype)

        # 2. Compute the raw loss using the appropriate internal loss function
        raw_loss = self._internal_losses[task_name](y_true, y_pred)

        # 3. Apply weighting (either learnable uncertainty or fixed static weight)
        if self.use_uncertainty_weighting:
            if task_name == TaskType.DETECTION.value:
                precision = ops.exp(-self.detection_log_var)
                weighted_loss = precision * raw_loss + self.detection_log_var
            elif task_name == TaskType.SEGMENTATION.value:
                precision = ops.exp(-self.segmentation_log_var)
                weighted_loss = precision * raw_loss + self.segmentation_log_var
            else:  # Classification
                precision = ops.exp(-self.classification_log_var)
                weighted_loss = precision * raw_loss + self.classification_log_var
        else:
            if task_name == TaskType.DETECTION.value:
                weighted_loss = self.detection_weight * raw_loss
            elif task_name == TaskType.SEGMENTATION.value:
                weighted_loss = self.segmentation_weight * raw_loss
            else: # Classification
                weighted_loss = self.classification_weight * raw_loss

        return weighted_loss

    def get_task_weights(self) -> Dict[str, float]:
        """
        Returns the current weights for each task.

        Called by the callback to log weights during training.
        """
        weights = {}
        if self.use_uncertainty_weighting:
            if self.task_config.has_detection():
                weights[TaskType.DETECTION.value] = float(ops.exp(-self.detection_log_var))
            if self.task_config.has_segmentation():
                weights[TaskType.SEGMENTATION.value] = float(ops.exp(-self.segmentation_log_var))
            if self.task_config.has_classification():
                weights[TaskType.CLASSIFICATION.value] = float(ops.exp(-self.classification_log_var))
        else:
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

        Note: In a named-output model, Keras logs individual losses automatically
        (e.g., 'detection_loss'). This method is not strictly necessary for logging
        but is kept for compatibility with the provided `EnhancedMultiTaskCallback`.
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the loss."""
        config = super().get_config()
        config.update({
            "task_config": self.task_config.get_task_names(),
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "use_uncertainty_weighting": self.use_uncertainty_weighting,
            "detection_weight": self.detection_weight,
            "segmentation_weight": self.segmentation_weight,
            "classification_weight": self.classification_weight,
            "reg_max": self.reg_max,
        })
        return config

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
    training script.

    Args:
        tasks: The tasks to enable, e.g., ['detection', 'segmentation'].
        num_classes: Number of object classes.
        input_shape: Model input shape (height, width).
        **kwargs: Additional arguments for YOLOv12MultiTaskLoss, such as
                  `use_uncertainty_weighting` or static weights.

    Returns:
        An instance of YOLOv12MultiTaskLoss configured for the specified tasks.
    """
    task_config = parse_task_list(tasks)
    return YOLOv12MultiTaskLoss(
        task_config=task_config,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )