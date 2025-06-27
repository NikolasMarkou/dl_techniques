"""
YOLOv12 Multi-Task Loss Function for simultaneous object detection,
segmentation, and classification.

This module provides comprehensive loss functions for training YOLOv12
on multiple tasks with proper loss balancing and weighting strategies.

Components:
    - Detection Loss: Task-aligned assignment with CIoU and DFL
    - Segmentation Loss: Combined Dice and Focal Loss
    - Classification Loss: Binary Cross-Entropy with label smoothing
    - Multi-task loss balancing with uncertainty weighting

File: src/dl_techniques/losses/yolo12_multitask_loss.py
"""

import keras
from keras import ops
from typing import Dict, Any, List

from dl_techniques.utils.logger import logger
from .yolo12_loss import YOLOv12Loss, bbox_iou


@keras.saving.register_keras_serializable()
class DiceFocalLoss(keras.losses.Loss):
    """
    Combined Dice and Focal Loss for segmentation tasks.

    Handles class imbalance common in crack segmentation where
    crack pixels are much fewer than background pixels.
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            dice_weight: float = 0.5,
            focal_weight: float = 0.5,
            smooth: float = 1e-6,
            name: str = "dice_focal_loss",
            **kwargs
    ):
        """
        Initialize Dice-Focal Loss.

        Args:
            alpha: Weighting factor for rare class in focal loss.
            gamma: Focusing parameter for focal loss.
            dice_weight: Weight for dice loss component.
            focal_weight: Weight for focal loss component.
            smooth: Smoothing factor for dice loss.
            name: Loss function name.
        """
        super().__init__(name=name, **kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth

    def call(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Calculate combined Dice and Focal loss.

        Args:
            y_true: Ground truth masks (batch_size, height, width, 1).
            y_pred: Predicted masks (batch_size, height, width, 1).

        Returns:
            Combined loss value.
        """
        # Ensure predictions are in [0, 1] range
        y_pred = ops.clip(y_pred, self.smooth, 1.0 - self.smooth)

        # Calculate Dice Loss
        dice_loss = self._dice_loss(y_true, y_pred)

        # Calculate Focal Loss
        focal_loss = self._focal_loss(y_true, y_pred)

        # Combine losses
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return total_loss

    def _dice_loss(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Calculate Dice loss for segmentation."""
        # Flatten tensors
        y_true_flat = ops.reshape(y_true, (ops.shape(y_true)[0], -1))
        y_pred_flat = ops.reshape(y_pred, (ops.shape(y_pred)[0], -1))

        # Calculate intersection and union
        intersection = ops.sum(y_true_flat * y_pred_flat, axis=1)
        union = ops.sum(y_true_flat, axis=1) + ops.sum(y_pred_flat, axis=1)

        # Calculate Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice coefficient)
        return ops.mean(1.0 - dice_coeff)

    def _focal_loss(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Calculate Focal loss for class imbalance."""
        # Calculate binary cross entropy
        bce = -y_true * ops.log(y_pred) - (1 - y_true) * ops.log(1 - y_pred)

        # Calculate focal weight
        pt = ops.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = self.alpha * ops.power(1 - pt, self.gamma)

        # Apply focal weight
        focal_loss = focal_weight * bce

        return ops.mean(focal_loss)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
            "smooth": self.smooth,
        })
        return config


@keras.saving.register_keras_serializable()
class YOLOv12MultiTaskLoss(keras.losses.Loss):
    """
    Multi-task loss function for YOLOv12 combining detection, segmentation,
    and classification losses with adaptive weighting.
    """

    def __init__(
            self,
            # Task weights
            detection_weight: float = 1.0,
            segmentation_weight: float = 1.0,
            classification_weight: float = 1.0,

            # Detection loss parameters
            num_classes: int = 1,
            patch_size: int = 256,
            reg_max: int = 16,
            detection_box_weight: float = 7.5,
            detection_cls_weight: float = 0.5,
            detection_dfl_weight: float = 1.5,

            # Segmentation loss parameters
            seg_alpha: float = 0.25,
            seg_gamma: float = 2.0,
            seg_dice_weight: float = 0.5,
            seg_focal_weight: float = 0.5,

            # Classification loss parameters
            cls_label_smoothing: float = 0.1,
            cls_pos_weight: float = 2.0,

            # Adaptive weighting
            use_uncertainty_weighting: bool = False,
            uncertainty_regularization: float = 1.0,

            name: str = "yolov12_multitask_loss",
            **kwargs
    ):
        """
        Initialize multi-task loss function.

        Args:
            detection_weight: Weight for detection loss.
            segmentation_weight: Weight for segmentation loss.
            classification_weight: Weight for classification loss.
            num_classes: Number of detection classes.
            patch_size: Size of input patches.
            reg_max: Maximum value for DFL regression.
            detection_box_weight: Weight for bbox loss in detection.
            detection_cls_weight: Weight for classification loss in detection.
            detection_dfl_weight: Weight for DFL loss in detection.
            seg_alpha: Alpha parameter for segmentation focal loss.
            seg_gamma: Gamma parameter for segmentation focal loss.
            seg_dice_weight: Weight for dice component in segmentation loss.
            seg_focal_weight: Weight for focal component in segmentation loss.
            cls_label_smoothing: Label smoothing for classification loss.
            cls_pos_weight: Positive class weight for classification.
            use_uncertainty_weighting: Whether to use uncertainty-based weighting.
            uncertainty_regularization: Regularization strength for uncertainty.
            name: Loss function name.
        """
        super().__init__(name=name, **kwargs)

        # Store task weights
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.classification_weight = classification_weight

        # Store parameters for serialization
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.reg_max = reg_max
        self.detection_box_weight = detection_box_weight
        self.detection_cls_weight = detection_cls_weight
        self.detection_dfl_weight = detection_dfl_weight
        self.seg_alpha = seg_alpha
        self.seg_gamma = seg_gamma
        self.seg_dice_weight = seg_dice_weight
        self.seg_focal_weight = seg_focal_weight
        self.cls_label_smoothing = cls_label_smoothing
        self.cls_pos_weight = cls_pos_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.uncertainty_regularization = uncertainty_regularization

        # Initialize component losses
        self.detection_loss = YOLOv12Loss(
            num_classes=num_classes,
            input_shape=(patch_size, patch_size),
            reg_max=reg_max,
            box_weight=detection_box_weight,
            cls_weight=detection_cls_weight,
            dfl_weight=detection_dfl_weight,
            name="detection_loss"
        )

        self.segmentation_loss = DiceFocalLoss(
            alpha=seg_alpha,
            gamma=seg_gamma,
            dice_weight=seg_dice_weight,
            focal_weight=seg_focal_weight,
            name="segmentation_loss"
        )

        self.classification_loss = keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=cls_label_smoothing,
            name="classification_loss"
        )

        # Uncertainty weights (learned parameters)
        if self.use_uncertainty_weighting:
            self.detection_log_var = self.add_weight(
                name="detection_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )
            self.segmentation_log_var = self.add_weight(
                name="segmentation_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )
            self.classification_log_var = self.add_weight(
                name="classification_log_var",
                shape=(),
                initializer="zeros",
                trainable=True
            )

        logger.info("YOLOv12MultiTaskLoss initialized with uncertainty weighting: "
                    f"{use_uncertainty_weighting}")

    def call(
            self,
            y_true: Dict[str, keras.KerasTensor],
            y_pred: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Calculate multi-task loss.

        Args:
            y_true: Dictionary of ground truth labels:
                - 'detection': (batch_size, max_boxes, 5)
                - 'segmentation': (batch_size, height, width, 1)
                - 'classification': (batch_size,) or (batch_size, 1)
            y_pred: Dictionary of model predictions:
                - 'detection': (batch_size, num_anchors, 4*reg_max + num_classes)
                - 'segmentation': (batch_size, height, width, 1)
                - 'classification': (batch_size, 1)

        Returns:
            Total multi-task loss.
        """
        total_loss = 0.0
        individual_losses = {}

        # Detection loss
        if 'detection' in y_true and 'detection' in y_pred:
            det_loss = self.detection_loss(y_true['detection'], y_pred['detection'])
            individual_losses['detection'] = det_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.detection_log_var)
                det_loss_weighted = precision * det_loss + self.detection_log_var
            else:
                det_loss_weighted = self.detection_weight * det_loss

            total_loss += det_loss_weighted

        # Segmentation loss
        if 'segmentation' in y_true and 'segmentation' in y_pred:
            seg_loss = self.segmentation_loss(y_true['segmentation'], y_pred['segmentation'])
            individual_losses['segmentation'] = seg_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.segmentation_log_var)
                seg_loss_weighted = precision * seg_loss + self.segmentation_log_var
            else:
                seg_loss_weighted = self.segmentation_weight * seg_loss

            total_loss += seg_loss_weighted

        # Classification loss
        if 'classification' in y_true and 'classification' in y_pred:
            # Handle different shapes for classification labels
            y_true_cls = y_true['classification']
            y_pred_cls = y_pred['classification']

            # Ensure both have the same shape
            if len(ops.shape(y_true_cls)) != len(ops.shape(y_pred_cls)):
                if len(ops.shape(y_true_cls)) == 1:
                    y_true_cls = ops.expand_dims(y_true_cls, -1)
                if len(ops.shape(y_pred_cls)) == 1:
                    y_pred_cls = ops.expand_dims(y_pred_cls, -1)

            # Apply positive class weighting manually
            cls_loss = self._weighted_binary_crossentropy(y_true_cls, y_pred_cls)
            individual_losses['classification'] = cls_loss

            if self.use_uncertainty_weighting:
                precision = ops.exp(-self.classification_log_var)
                cls_loss_weighted = precision * cls_loss + self.classification_log_var
            else:
                cls_loss_weighted = self.classification_weight * cls_loss

            total_loss += cls_loss_weighted

        # Add uncertainty regularization
        if self.use_uncertainty_weighting:
            uncertainty_reg = self.uncertainty_regularization * (
                    self.detection_log_var + self.segmentation_log_var + self.classification_log_var
            )
            total_loss += uncertainty_reg

        # Store individual losses for monitoring (if needed)
        if hasattr(self, 'individual_losses'):
            self.individual_losses = individual_losses

        return total_loss

    def _weighted_binary_crossentropy(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Calculate weighted binary cross-entropy for imbalanced classification."""
        # Ensure predictions are in valid range
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)

        # Calculate binary cross entropy
        bce = -y_true * ops.log(y_pred) - (1 - y_true) * ops.log(1 - y_pred)

        # Apply positive class weighting
        weights = y_true * self.cls_pos_weight + (1 - y_true) * 1.0
        weighted_bce = weights * bce

        # Apply label smoothing
        if self.cls_label_smoothing > 0:
            y_true_smooth = y_true * (1 - self.cls_label_smoothing) + 0.5 * self.cls_label_smoothing
            bce_smooth = -y_true_smooth * ops.log(y_pred) - (1 - y_true_smooth) * ops.log(1 - y_pred)
            weighted_bce = weights * bce_smooth

        return ops.mean(weighted_bce)

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (useful for monitoring)."""
        if self.use_uncertainty_weighting:
            return {
                'detection': float(ops.exp(-self.detection_log_var)),
                'segmentation': float(ops.exp(-self.segmentation_log_var)),
                'classification': float(ops.exp(-self.classification_log_var))
            }
        else:
            return {
                'detection': self.detection_weight,
                'segmentation': self.segmentation_weight,
                'classification': self.classification_weight
            }

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "detection_weight": self.detection_weight,
            "segmentation_weight": self.segmentation_weight,
            "classification_weight": self.classification_weight,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "reg_max": self.reg_max,
            "detection_box_weight": self.detection_box_weight,
            "detection_cls_weight": self.detection_cls_weight,
            "detection_dfl_weight": self.detection_dfl_weight,
            "seg_alpha": self.seg_alpha,
            "seg_gamma": self.seg_gamma,
            "seg_dice_weight": self.seg_dice_weight,
            "seg_focal_weight": self.seg_focal_weight,
            "cls_label_smoothing": self.cls_label_smoothing,
            "cls_pos_weight": self.cls_pos_weight,
            "use_uncertainty_weighting": self.use_uncertainty_weighting,
            "uncertainty_regularization": self.uncertainty_regularization,
        })
        return config


def create_multitask_loss(
        tasks: List[str] = ["detection", "segmentation", "classification"],
        patch_size: int = 256,
        **kwargs
) -> YOLOv12MultiTaskLoss:
    """
    Create multi-task loss function with specified configuration.

    Args:
        tasks: List of tasks to include in loss calculation.
        patch_size: Size of input patches.
        **kwargs: Additional arguments for YOLOv12MultiTaskLoss.

    Returns:
        YOLOv12MultiTaskLoss instance.
    """
    # Adjust task weights based on enabled tasks
    task_weights = {}
    if "detection" not in tasks:
        task_weights["detection_weight"] = 0.0
    if "segmentation" not in tasks:
        task_weights["segmentation_weight"] = 0.0
    if "classification" not in tasks:
        task_weights["classification_weight"] = 0.0

    # Merge with user-provided kwargs
    kwargs.update(task_weights)

    loss_fn = YOLOv12MultiTaskLoss(
        patch_size=patch_size,
        **kwargs
    )

    logger.info(f"Multi-task loss function created for tasks: {tasks}")
    return loss_fn


# Utility functions for loss monitoring
def extract_individual_losses(multitask_loss: YOLOv12MultiTaskLoss) -> Dict[str, float]:
    """Extract individual task losses for monitoring."""
    if hasattr(multitask_loss, 'individual_losses'):
        return {
            name: float(loss.numpy()) if hasattr(loss, 'numpy') else float(loss)
            for name, loss in multitask_loss.individual_losses.items()
        }
    return {}


# Example usage and testing
if __name__ == "__main__":
    # Test Dice-Focal Loss
    dice_focal = DiceFocalLoss()

    # Create dummy segmentation data
    y_true_seg = ops.zeros((2, 256, 256, 1))
    y_true_seg = ops.scatter_update(y_true_seg, [[0, 100], [0, 150]], [1.0, 1.0])
    y_pred_seg = ops.random.uniform((2, 256, 256, 1))

    seg_loss = dice_focal(y_true_seg, y_pred_seg)
    print(f"Segmentation loss: {seg_loss}")

    # Test Multi-task Loss
    multitask_loss = create_multitask_loss(
        tasks=["detection", "segmentation", "classification"],
        patch_size=256
    )

    # Create dummy multi-task data
    y_true_multitask = {
        'detection': ops.zeros((2, 10, 5)),  # (batch, max_boxes, 5)
        'segmentation': y_true_seg,
        'classification': ops.convert_to_tensor([[1.0], [0.0]])
    }

    y_pred_multitask = {
        'detection': ops.random.normal((2, 1000, 17)),  # Dummy detection predictions
        'segmentation': y_pred_seg,
        'classification': ops.convert_to_tensor([[0.8], [0.2]])
    }

    total_loss = multitask_loss(y_true_multitask, y_pred_multitask)
    print(f"Multi-task loss: {total_loss}")

    # Check task weights
    weights = multitask_loss.get_task_weights()
    print(f"Task weights: {weights}")