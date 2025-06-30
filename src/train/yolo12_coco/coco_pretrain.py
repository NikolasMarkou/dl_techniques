"""
COCO Pre-training Script for YOLOv12 Feature Extractor

This script implements Phase 1 of the two-phase training approach:
pre-training the YOLOv12 multi-task model on COCO dataset to learn
powerful, general-purpose visual features.

The script will:
1. Load and preprocess the COCO 2017 dataset
2. Create a YOLOv12 multi-task model (detection + segmentation)
3. Train the model on COCO for a substantial number of epochs
4. Save the trained feature extractor weights for later fine-tuning

FIXED VERSION: Now supports configurable detection and segmentation classes.
For COCO pretraining: 80 detection classes, 80 segmentation classes.
For crack detection fine-tuning: 1 detection class, 1 segmentation class.

Fixed Issues:
- Visualizations now respect the --no-visualizations argument
- Improved validation data handling for real vs dummy datasets
- Better steps per epoch calculation
- Memory management options to prevent OOM errors

Memory Management:
- Use --shuffle-buffer 50 if you get "Killed" errors
- Use --limit-train-samples 5000 for testing/development
- Use --batch-size 4 or lower if you run out of GPU memory
- Use --img-size 416 instead of 640 to reduce memory usage

Usage:
    # COCO pretraining (default)
    python coco_pretrain.py --scale s --epochs 50 --batch-size 16 \
                           --img-size 640 --cache-dir /path/to/cache

    # Memory-constrained training
    python coco_pretrain.py --scale s --epochs 10 --batch-size 4 \
                           --img-size 416 --shuffle-buffer 50 \
                           --limit-train-samples 5000

    # Custom class configuration
    python coco_pretrain.py --scale s --detection-classes 80 \
                           --segmentation-classes 80 --epochs 50

Requirements:
    - tensorflow-datasets: pip install tensorflow-datasets
    - Sufficient disk space for COCO dataset (~37GB)
    - GPU with adequate memory for chosen batch size and image size

File: coco_pretrain.py
"""

import os
import argparse
import json
import keras
import tensorflow as tf
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
import matplotlib
from matplotlib import patches

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from dl_techniques.utils.logger import logger
from dl_techniques.utils.vision_task_types import TaskType
from dl_techniques.utils.datasets.coco import COCODatasetBuilder
from dl_techniques.models.yolo12_multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_yolov12_multitask_loss

def setup_gpu():
    """Configure GPU settings for optimal training."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")


def create_coco_model_and_loss(
        scale: str,
        img_size: int,
        detection_classes: int = 80,
        segmentation_classes: int = 80,
        use_uncertainty_weighting: bool = False
) -> Tuple[keras.Model, keras.losses.Loss]:
    """
    Create YOLOv12 multi-task model and loss for COCO pre-training.

    FIXED VERSION: Now supports separate class counts for detection and segmentation.

    Args:
        scale: Model scale ('n', 's', 'm', 'l', 'x').
        img_size: Input image size.
        detection_classes: Number of detection classes.
        segmentation_classes: Number of segmentation classes.
        use_uncertainty_weighting: Whether to use uncertainty-based loss weighting.

    Returns:
        Tuple of (model, loss_function).
    """
    logger.info(f"Creating YOLOv12-{scale} model for COCO pre-training...")
    logger.info(f"  Detection classes: {detection_classes}")
    logger.info(f"  Segmentation classes: {segmentation_classes}")

    # Create model for detection and segmentation tasks
    model = create_yolov12_multitask(
        num_detection_classes=detection_classes,
        num_segmentation_classes=segmentation_classes,
        input_shape=(img_size, img_size, 3),
        scale=scale,
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION]
    )

    # Create loss function
    loss_fn = create_yolov12_multitask_loss(
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        num_detection_classes=detection_classes,
        num_segmentation_classes=segmentation_classes,
        input_shape=(img_size, img_size),
        use_uncertainty_weighting=use_uncertainty_weighting,
        # COCO-specific loss weights (can be tuned)
        detection_weight=1.0,
        segmentation_weight=0.5  # Lower weight for segmentation
    )

    return model, loss_fn


def create_coco_callbacks(
        results_dir: str,
        validation_dataset=None,
        img_size: int = 640,
        monitor: str = 'val_loss',
        patience: int = 10,
        save_best_only: bool = True,
        enable_visualizations: bool = True,
        visualization_freq: int = 5
) -> list:
    """
    Create callbacks for COCO pre-training.

    Args:
        results_dir: Directory to save results.
        validation_dataset: Validation dataset for visualization callback.
        img_size: Input image size for visualization.
        monitor: Metric to monitor for early stopping and checkpointing.
        patience: Early stopping patience.
        save_best_only: Whether to save only the best model.
        enable_visualizations: Whether to enable per-epoch visualizations.
        visualization_freq: Frequency of visualization (every N epochs).

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        # Early stopping to prevent overfitting
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode='min',
            verbose=1,
            min_delta=1e-4
        ),

        # Model checkpointing
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_coco_model.keras'),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),

        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(patience // 2, 3),
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        ),

        # CSV logging
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'coco_training_log.csv'),
            append=False
        ),

        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        ),

        # Progress logging callback
        ProgressLoggingCallback()
    ]

    # Add visualization callback if enabled and validation dataset is provided
    if enable_visualizations and validation_dataset is not None:
        viz_callback = COCOVisualizationCallback(
            validation_dataset=validation_dataset,
            results_dir=results_dir,
            img_size=img_size,
            num_samples=4,
            visualization_freq=visualization_freq,
            confidence_threshold=0.3,
            reg_max=16
        )
        callbacks.append(viz_callback)
        logger.info(f"✅ COCO visualization callback added (every {visualization_freq} epochs)")
    elif enable_visualizations:
        logger.warning("⚠️ Visualizations enabled but no validation dataset provided")
    else:
        logger.info("📊 COCO visualizations disabled")

    return callbacks


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Custom callback for enhanced progress logging during COCO pre-training."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Enhanced logging every epoch
        loss_str = f"Loss: {logs.get('loss', 0):.4f}"

        if 'val_loss' in logs:
            loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"

        # Add task-specific losses if available
        task_losses = []
        for task in ['detection', 'segmentation']:
            if f'{task}_loss' in logs:
                task_losses.append(f"{task}: {logs[f'{task}_loss']:.4f}")

        if task_losses:
            loss_str += f" | {', '.join(task_losses)}"

        # Add learning rate if available
        if 'lr' in logs:
            loss_str += f" | LR: {logs['lr']:.2e}"

        logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")


class COCOVisualizationCallback(keras.callbacks.Callback):
    """
    Callback for generating per-epoch COCO detection and segmentation visualizations.

    This callback creates visualization images showing model predictions vs ground truth
    for both detection and segmentation tasks on COCO dataset samples.
    """

    # COCO class names for visualization
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(
            self,
            validation_dataset,
            results_dir: str,
            img_size: int = 640,
            num_samples: int = 4,
            visualization_freq: int = 5,
            confidence_threshold: float = 0.3,
            reg_max: int = 16
    ):
        """
        Initialize the COCO visualization callback.

        Args:
            validation_dataset: TensorFlow validation dataset.
            results_dir: Directory to save visualizations.
            img_size: Input image size (should match training).
            num_samples: Number of samples to visualize per epoch.
            visualization_freq: Frequency of visualization (every N epochs).
            confidence_threshold: Confidence threshold for detections.
            reg_max: DFL regression maximum value.
        """
        super().__init__()
        self.validation_dataset = validation_dataset
        self.results_dir = results_dir
        self.img_size = img_size
        self.num_samples = num_samples
        self.visualization_freq = visualization_freq
        self.confidence_threshold = confidence_threshold
        self.reg_max = reg_max

        # Create visualization directory
        self.viz_dir = os.path.join(results_dir, 'coco_epoch_visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

        # Pre-sample validation data for consistent visualization
        self._prepare_visualization_samples()

        logger.info(f"COCO Visualization Callback initialized - saving to {self.viz_dir}")

    def _prepare_visualization_samples(self):
        """Prepare a fixed set of samples for consistent visualization across epochs."""
        try:
            # Take a batch from validation dataset
            sample_batch = next(iter(self.validation_dataset.take(1)))

            if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                self.sample_images, self.sample_targets = sample_batch

                # Limit to num_samples
                if tf.shape(self.sample_images)[0] > self.num_samples:
                    self.sample_images = self.sample_images[:self.num_samples]
                    if isinstance(self.sample_targets, dict):
                        self.sample_targets = {
                            k: v[:self.num_samples] for k, v in self.sample_targets.items()
                        }
                    else:
                        self.sample_targets = self.sample_targets[:self.num_samples]

                logger.info(f"Prepared {tf.shape(self.sample_images)[0]} COCO samples for visualization")
            else:
                logger.error("Unexpected COCO sample batch format for visualization")
                self.sample_images = None
                self.sample_targets = None

        except Exception as e:
            logger.error(f"Failed to prepare COCO visualization samples: {e}")
            self.sample_images = None
            self.sample_targets = None

    def _get_anchor_grid(self):
        """Generate anchor grid for YOLOv12 detection decoding."""
        if hasattr(self, '_anchor_grid_cache'):
            return self._anchor_grid_cache

        strides_config = [8, 16, 32]  # Match YOLOv12 architecture
        anchor_points = []

        for stride in strides_config:
            h, w = self.img_size // stride, self.img_size // stride
            # Generate anchor centers (offset by 0.5 to center in grid cell)
            x_coords = (np.arange(w) + 0.5) * stride
            y_coords = (np.arange(h) + 0.5) * stride
            y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
            xy_grid = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)
            anchor_points.append(xy_grid)

        # Cache the result
        self._anchor_grid_cache = np.concatenate(anchor_points, 0)
        return self._anchor_grid_cache

    def _decode_coco_detections(
            self,
            detection_pred: np.ndarray,
            max_detections: int = 10
    ) -> List[Dict[str, any]]:
        """
        Decode YOLOv12 detection predictions for COCO dataset.

        Args:
            detection_pred: Raw detection predictions.
            max_detections: Maximum number of detections to return.

        Returns:
            List of decoded detections.
        """
        try:
            # detection_pred shape: (1, num_anchors, 4*reg_max + num_classes)
            batch_size, num_anchors, features = detection_pred.shape
            num_classes = features - 4 * self.reg_max

            # Split into distance and classification
            dist_pred = detection_pred[0, :, :4 * self.reg_max]  # (num_anchors, 4*reg_max)
            cls_pred = detection_pred[0, :, 4 * self.reg_max:]  # (num_anchors, num_classes)

            # Apply sigmoid to classification scores
            cls_scores = 1 / (1 + np.exp(-cls_pred))

            # Find top detections across all classes
            max_scores = np.max(cls_scores, axis=1)
            best_classes = np.argmax(cls_scores, axis=1)

            # Get top detections
            top_indices = np.argsort(max_scores)[-max_detections:][::-1]
            valid_indices = top_indices[max_scores[top_indices] > self.confidence_threshold]

            if len(valid_indices) == 0:
                return []

            # Decode distance predictions for valid anchors
            valid_dist = dist_pred[valid_indices]  # (num_valid, 4*reg_max)
            valid_scores = cls_scores[valid_indices]  # (num_valid, num_classes)
            valid_classes = best_classes[valid_indices]

            # Reshape and apply softmax to distance predictions
            valid_dist = valid_dist.reshape(-1, 4, self.reg_max)
            softmax_dist = np.exp(valid_dist) / (np.sum(np.exp(valid_dist), axis=-1, keepdims=True) + 1e-8)

            # Compute weighted distances
            range_vals = np.arange(self.reg_max)
            decoded_dist = np.sum(softmax_dist * range_vals, axis=-1)  # (num_valid, 4)

            # Get anchor coordinates
            anchor_grid = self._get_anchor_grid()
            valid_anchor_coords = anchor_grid[valid_indices]

            # Convert distances to bounding boxes
            detections = []
            for i, (anchor_coords, distances, scores, class_id) in enumerate(
                    zip(valid_anchor_coords, decoded_dist, valid_scores, valid_classes)
            ):
                cx, cy = anchor_coords
                l, t, r, b = distances

                # Convert to x1, y1, x2, y2
                x1 = max(0, cx - l)
                y1 = max(0, cy - t)
                x2 = min(self.img_size, cx + r)
                y2 = min(self.img_size, cy + b)

                # Get confidence for this class
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': self.COCO_CLASSES[class_id] if class_id < len(
                            self.COCO_CLASSES) else f'class_{class_id}'
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Failed to decode COCO detections: {e}")
            return []

    def on_epoch_end(self, epoch, logs=None):
        """Generate COCO visualizations at the end of each epoch."""
        if (epoch + 1) % self.visualization_freq != 0:
            return

        if self.sample_images is None:
            logger.warning("No COCO samples available for visualization")
            return

        try:
            # Get predictions for the sample batch
            predictions = self.model(self.sample_images, training=False)

            # Create epoch-specific directory
            epoch_dir = os.path.join(self.viz_dir, f'epoch_{epoch + 1:03d}')
            os.makedirs(epoch_dir, exist_ok=True)

            # Generate visualizations
            self._visualize_coco_detections(predictions, epoch_dir, epoch + 1)
            self._visualize_coco_segmentation(predictions, epoch_dir, epoch + 1)

            logger.info(f"Generated COCO visualizations for epoch {epoch + 1}")

        except Exception as e:
            logger.error(f"Failed to generate COCO visualizations for epoch {epoch + 1}: {e}")

    def _visualize_coco_detections(self, predictions, save_dir: str, epoch: int):
        """Generate COCO detection visualization with bounding boxes."""
        try:
            # Extract detection predictions
            if isinstance(predictions, dict):
                detection_pred = predictions.get('detection')
            else:
                detection_pred = predictions

            if detection_pred is None:
                logger.warning("No detection predictions found")
                return

            # Get ground truth detection targets
            gt_boxes = None
            if isinstance(self.sample_targets, dict):
                gt_boxes = self.sample_targets.get('detection')

            # Convert to numpy
            images_np = self.sample_images.numpy()
            pred_np = detection_pred.numpy()
            gt_np = gt_boxes.numpy() if gt_boxes is not None else None

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()

            for i in range(min(4, len(images_np))):
                if i >= len(axes):
                    break

                # Display image
                img_display = images_np[i]
                axes[i].imshow(img_display)
                axes[i].set_title(f'COCO Sample {i + 1} - Epoch {epoch}', fontsize=12)
                axes[i].axis('off')

                # Draw ground truth boxes
                if gt_np is not None and i < len(gt_np):
                    for gt_box in gt_np[i]:
                        if gt_box.sum() == 0:  # Skip padding/empty boxes
                            continue

                        # Ground truth format: [class_id, x1, y1, x2, y2] (normalized)
                        if len(gt_box) >= 5 and gt_box[0] >= 0:  # Valid class
                            class_id, x1, y1, x2, y2 = gt_box[:5]

                            # Convert from normalized to pixel coordinates
                            x1_px = x1 * self.img_size
                            y1_px = y1 * self.img_size
                            x2_px = x2 * self.img_size
                            y2_px = y2 * self.img_size

                            # Draw ground truth rectangle
                            rect = patches.Rectangle(
                                (x1_px, y1_px), x2_px - x1_px, y2_px - y1_px,
                                linewidth=2, edgecolor='green', facecolor='none',
                                linestyle='-', alpha=0.8
                            )
                            axes[i].add_patch(rect)

                            # Add class label
                            class_name = self.COCO_CLASSES[int(class_id)] if int(class_id) < len(
                                self.COCO_CLASSES) else f'class_{int(class_id)}'
                            axes[i].text(
                                x1_px, y1_px - 5,
                                f'GT: {class_name}',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.8),
                                color='white', fontsize=8, fontweight='bold'
                            )

                # Decode and draw predictions
                detections = self._decode_coco_detections(pred_np[i:i + 1])

                for det in detections[:5]:  # Show top 5 detections
                    x1, y1, x2, y2 = det['bbox']

                    # Draw predicted bounding box
                    pred_rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', alpha=0.9
                    )
                    axes[i].add_patch(pred_rect)

                    # Add prediction label
                    axes[i].text(
                        x1, max(0, y1 - 15),
                        f"{det['class_name']}: {det['confidence']:.2f}",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8),
                        color='white', fontsize=8, fontweight='bold'
                    )

            # Add legend
            legend_elements = [
                patches.Patch(color='green', label='Ground Truth'),
                patches.Patch(color='red', label='Predictions')
            ]
            fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

            plt.suptitle(f'COCO Detection Results - Epoch {epoch}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'coco_detection_results.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to visualize COCO detections: {e}")

    def _visualize_coco_segmentation(self, predictions, save_dir: str, epoch: int):
        """Generate COCO segmentation visualization."""
        try:
            # Extract segmentation predictions
            if isinstance(predictions, dict):
                seg_pred = predictions.get('segmentation')
            else:
                return

            if seg_pred is None:
                logger.warning("No segmentation predictions found")
                return

            # Get ground truth segmentation if available
            seg_gt = None
            if isinstance(self.sample_targets, dict):
                seg_gt = self.sample_targets.get('segmentation')

            # Convert to numpy
            images_np = self.sample_images.numpy()
            pred_np = tf.nn.sigmoid(seg_pred).numpy()  # Apply sigmoid
            gt_np = seg_gt.numpy() if seg_gt is not None else None

            # Create figure
            n_cols = 3 if gt_np is not None else 2
            fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))

            for i in range(min(2, len(images_np))):
                # Original image
                axes[i, 0].imshow(images_np[i])
                axes[i, 0].set_title(f'COCO Image {i + 1}', fontsize=12)
                axes[i, 0].axis('off')

                # Predicted mask
                if pred_np.shape[-1] == 1:
                    # Binary segmentation
                    pred_mask = pred_np[i, ..., 0]
                else:
                    # Multi-class segmentation - show argmax
                    pred_mask = np.argmax(pred_np[i], axis=-1)

                im_pred = axes[i, 1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1 if pred_np.shape[-1] == 1 else 79)
                axes[i, 1].set_title(f'Predicted Mask {i + 1}', fontsize=12)
                axes[i, 1].axis('off')
                plt.colorbar(im_pred, ax=axes[i, 1], shrink=0.8)

                # Ground truth mask (if available)
                if gt_np is not None and i < len(gt_np):
                    if gt_np.shape[-1] == 1:
                        # Binary ground truth
                        gt_mask = gt_np[i, ..., 0]
                    else:
                        # Multi-class ground truth - show argmax
                        gt_mask = np.argmax(gt_np[i], axis=-1)

                    im_gt = axes[i, 2].imshow(gt_mask, cmap='hot', vmin=0, vmax=1 if gt_np.shape[-1] == 1 else 79)
                    axes[i, 2].set_title(f'Ground Truth {i + 1}', fontsize=12)
                    axes[i, 2].axis('off')
                    plt.colorbar(im_gt, ax=axes[i, 2], shrink=0.8)

                    # Calculate and display IoU for binary masks
                    if pred_np.shape[-1] == 1 and gt_np.shape[-1] == 1:
                        pred_binary = (pred_mask > 0.5).astype(float)
                        gt_binary = (gt_mask > 0.5).astype(float)
                        intersection = np.sum(pred_binary * gt_binary)
                        union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
                        iou = intersection / (union + 1e-8)

                        axes[i, 1].text(
                            5, pred_mask.shape[0] - 20, f'IoU: {iou:.3f}',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8),
                            fontsize=10, fontweight='bold'
                        )

            plt.suptitle(f'COCO Segmentation Results - Epoch {epoch}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'coco_segmentation_results.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to visualize COCO segmentation: {e}")


def save_feature_extractor_weights(
        model: keras.Model,
        save_path: str,
        scale: str,
        img_size: int,
        detection_classes: int = 80,
        segmentation_classes: int = 80
) -> None:
    """
    Extract and save the feature extractor weights for fine-tuning.

    Args:
        model: Trained YOLOv12 model.
        save_path: Path to save the weights.
        scale: Model scale.
        img_size: Input image size.
        detection_classes: Number of detection classes.
        segmentation_classes: Number of segmentation classes.
    """
    try:
        # Get the shared feature extractor layer
        feature_extractor = model.get_layer('shared_feature_extractor')

        # Create the weights save path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the weights
        feature_extractor.save_weights(save_path)

        logger.info(f"✅ Feature extractor weights saved to: {save_path}")

        # Also save metadata about the pre-trained model
        metadata = {
            'model_scale': scale,
            'input_size': img_size,
            'detection_classes': detection_classes,
            'segmentation_classes': segmentation_classes,
            'tasks': ['detection', 'segmentation'],
            'pretrained_on': 'COCO 2017',
            'save_timestamp': datetime.now().isoformat(),
            'weights_path': save_path
        }

        metadata_path = save_path.replace('.weights.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Metadata saved to: {metadata_path}")

    except Exception as e:
        logger.error(f"❌ Failed to save feature extractor weights: {e}")
        raise


def calculate_steps_per_epoch(
        dataset: tf.data.Dataset,
        batch_size: int,
        default_size: int = 118000,
        min_steps: int = 100,
        max_steps: int = 1000
) -> int:
    """
    Calculate appropriate steps per epoch based on actual dataset size.

    FIXED: Improved error handling for empty datasets.

    Args:
        dataset: The training dataset.
        batch_size: Batch size.
        default_size: Default dataset size (e.g., COCO train size).
        min_steps: Minimum steps per epoch.
        max_steps: Maximum steps per epoch for dummy datasets.

    Returns:
        Steps per epoch.
    """
    try:
        # Try to get the actual size of the dataset
        # For dummy datasets, we'll estimate based on the first few batches
        sample_count = 0
        max_sample_batches = min_steps  # Limit sampling to avoid hanging

        try:
            for batch in dataset.take(max_sample_batches):
                sample_count += 1

            logger.info(f"Sampled {sample_count} batches from dataset")

            if sample_count == 0:
                logger.warning("Dataset appears to be empty! Using minimum steps.")
                return min_steps
            elif sample_count < min_steps:
                # Dataset is smaller than expected, likely a dummy dataset
                steps = min(sample_count * 10, max_steps)  # Scale up but cap
                logger.info(f"Detected small dataset, using {steps} steps per epoch")
            else:
                # Larger dataset, use default calculation
                steps = max(default_size // batch_size, min_steps)
                logger.info(f"Using default steps calculation: {steps} steps per epoch")

        except tf.errors.OutOfRangeError:
            logger.warning("Dataset exhausted during sampling, using minimum steps")
            return min_steps

    except Exception as e:
        logger.warning(f"Could not determine dataset size: {e}")
        # Fallback to a reasonable default for testing
        steps = min_steps
        logger.info(f"Using fallback steps per epoch: {steps}")

    logger.info(f"Calculated steps per epoch: {steps}")
    return steps


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    try:
        import tensorflow_datasets as tfds
        logger.info(f"✅ tensorflow-datasets available (version: {tfds.__version__})")
    except ImportError:
        missing_deps.append("tensorflow-datasets")
        logger.error("❌ tensorflow-datasets not found")
        logger.error("Install with: pip install tensorflow-datasets")

    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.error("Please install missing dependencies and try again")
        return False

    return True


def main():
    """Main COCO pre-training function."""
    parser = argparse.ArgumentParser(
        description='Pre-train YOLOv12 Feature Extractor on COCO Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--scale', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv12 model scale')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')

    # NEW: Separate class configuration
    parser.add_argument('--detection-classes', type=int, default=80,
                        help='Number of detection classes (COCO: 80, crack: 1)')
    parser.add_argument('--segmentation-classes', type=int, default=80,
                        help='Number of segmentation classes (COCO: 80, crack: 1)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')

    # Dataset arguments
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache COCO dataset')
    parser.add_argument('--max-boxes', type=int, default=100,
                        help='Maximum boxes per image')
    parser.add_argument('--shuffle-buffer', type=int, default=100,
                        help='Shuffle buffer size (reduce if out of memory)')
    parser.add_argument('--limit-train-samples', type=int, default=None,
                        help='Limit training samples for memory efficiency')

    # Loss arguments
    parser.add_argument('--uncertainty-weighting', action='store_true',
                        help='Use uncertainty-based adaptive task weighting')

    # Control arguments
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save-dir', type=str, default='coco_pretrain_results',
                        help='Directory to save results')
    parser.add_argument('--weights-name', type=str, default=None,
                        help='Custom name for saved weights file')
    parser.add_argument('--run-eagerly', action='store_true',
                        help='Run model in eager mode')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Visualization arguments
    parser.add_argument('--no-visualizations', action='store_true',
                        help='Disable per-epoch visualizations')
    parser.add_argument('--visualization-freq', type=int, default=5,
                        help='Frequency of visualization (every N epochs)')
    parser.add_argument('--viz-samples', type=int, default=4,
                        help='Number of samples to visualize per epoch')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                        help='Confidence threshold for detection visualizations')

    args = parser.parse_args()

    # Check dependencies early
    logger.info("🔍 Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Missing required dependencies. Please install them and try again.")
        return

    # Setup
    setup_gpu()

    # Set random seeds
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.save_dir}/yolov12_{args.scale}_coco_pretrain_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info("🚀 Starting COCO Pre-training for YOLOv12 Feature Extractor")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Results directory: {results_dir}")

    # Log visualization settings respecting the argument
    if not args.no_visualizations:
        logger.info(f"📊 Per-epoch visualizations enabled (every {args.visualization_freq} epochs)")
        logger.info(f"📊 Visualizations will be saved to: {os.path.join(results_dir, 'coco_epoch_visualizations')}")
    else:
        logger.info("📊 Per-epoch visualizations disabled")

    # Memory management advice
    if args.limit_train_samples:
        logger.info(f"💾 Training limited to {args.limit_train_samples} samples for memory efficiency")
    if args.shuffle_buffer < 1000:
        logger.info(f"💾 Using reduced shuffle buffer ({args.shuffle_buffer}) for memory efficiency")
        logger.info("💾 Consider using --shuffle-buffer 1000 if you have more RAM available")

    # Create COCO dataset
    logger.info("📁 Creating COCO dataset...")
    dataset_builder = COCODatasetBuilder(
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_boxes_per_image=args.max_boxes,
        cache_dir=args.cache_dir,
        use_detection=True,
        use_segmentation=True,
        segmentation_classes=args.segmentation_classes,  # Use configurable segmentation classes
        augment_data=True,
        shuffle_buffer_size=args.shuffle_buffer,  # Pass shuffle buffer size
        limit_train_samples=args.limit_train_samples  # Pass sample limit
    )

    train_ds, val_ds = dataset_builder.create_datasets()
    dataset_info = dataset_builder.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")

    # Create model and loss
    logger.info("🏗️ Creating model and loss function...")
    model, loss_fn = create_coco_model_and_loss(
        scale=args.scale,
        img_size=args.img_size,
        detection_classes=args.detection_classes,
        segmentation_classes=args.segmentation_classes,
        use_uncertainty_weighting=args.uncertainty_weighting
    )

    # Build model
    sample_input = tf.zeros((1, args.img_size, args.img_size, 3))
    _ = model(sample_input, training=False)

    logger.info("Model summary:")
    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    # Create optimizer
    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            clipnorm=1.0
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=args.learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=1.0
        )

    logger.info(f"Using optimizer: {type(optimizer).__name__}")

    # Compile model
    logger.info("⚙️ Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        run_eagerly=args.run_eagerly
    )

    # FIXED: Determine validation usage and visualization settings
    dataset_info = dataset_builder.get_dataset_info()
    use_validation = not dataset_info.get('using_dummy_data', False)

    if use_validation:
        monitor = 'val_loss'
        logger.info("📊 Using validation-based monitoring")
    else:
        monitor = 'loss'
        logger.info("📊 Using training loss monitoring (dummy data mode)")

    # FIXED: Respect the --no-visualizations argument
    enable_visualizations = not args.no_visualizations and use_validation
    if args.no_visualizations:
        logger.info("📊 Visualizations disabled by --no-visualizations flag")
    elif not use_validation:
        logger.info("📊 Visualizations disabled for dummy data mode")

    callbacks = create_coco_callbacks(
        results_dir=results_dir,
        validation_dataset=val_ds if use_validation else None,
        img_size=args.img_size,
        monitor=monitor,
        patience=args.patience,
        enable_visualizations=enable_visualizations,  # FIXED: Use computed value
        visualization_freq=args.visualization_freq
    )

    # Calculate steps (estimate for COCO train: ~118k images, but adapt to actual dataset)
    steps_per_epoch = calculate_steps_per_epoch(
        dataset=train_ds,
        batch_size=args.batch_size,
        default_size=118000,  # COCO training set size
        min_steps=100,        # Minimum for testing
        max_steps=500         # Maximum for dummy datasets
    )

    # Train model
    logger.info("🏋️ Starting COCO pre-training...")
    try:
        # Determine validation data and steps
        validation_data = val_ds if use_validation else None
        validation_steps = None if not use_validation else 50  # Limit validation steps for efficiency

        history = model.fit(
            train_ds,
            validation_data=validation_data,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("✅ COCO pre-training completed successfully!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

    # Save feature extractor weights
    logger.info("💾 Saving feature extractor weights...")

    if args.weights_name:
        weights_filename = f"{args.weights_name}.weights.h5"
    else:
        weights_filename = f"yolov12_{args.scale}_coco_pretrained_feature_extractor.weights.h5"

    weights_path = os.path.join(results_dir, weights_filename)

    save_feature_extractor_weights(
        model=model,
        save_path=weights_path,
        scale=args.scale,
        img_size=args.img_size,
        detection_classes=args.detection_classes,
        segmentation_classes=args.segmentation_classes
    )

    # Save training configuration
    config = {
        'training_args': vars(args),
        'dataset_info': dataset_info,
        'model_info': {
            'scale': args.scale,
            'input_size': args.img_size,
            'total_parameters': model.count_params(),
            'detection_classes': args.detection_classes,
            'segmentation_classes': args.segmentation_classes
        },
        'training_results': {
            'epochs_completed': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history.get('val_loss', [0])[-1]),
            'best_validation_loss': float(min(history.history.get('val_loss', [float('inf')])))
        },
        'saved_weights': {
            'weights_path': weights_path,
            'relative_path': weights_filename
        }
    }

    config_path = os.path.join(results_dir, 'coco_pretrain_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Configuration saved to: {config_path}")

    # Create usage instructions
    instructions = f"""
    🎉 COCO Pre-training Completed Successfully!
    
    The feature extractor has been trained on COCO and saved to:
    {weights_path}
    
    Configuration:
    - Detection classes: {args.detection_classes}
    - Segmentation classes: {args.segmentation_classes}
    - Model scale: {args.scale}
    - Input size: {args.img_size}
    
    To use these pre-trained weights in your fine-tuning:
    
    1. In your fine-tuning script, load the weights:
       ```python
       # After creating your fine-tuning model with matching architecture
       feature_extractor = model.get_layer('shared_feature_extractor')
       feature_extractor.load_weights('{weights_path}')
       ```
    
    2. For Phase 2a (freeze feature extractor, train heads only):
       ```python
       feature_extractor.trainable = False
       model.compile(optimizer=optimizer, loss=loss_fn)
       # Train for ~20-30 epochs
       ```
    
    3. For Phase 2b (end-to-end fine-tuning):
       ```python
       feature_extractor.trainable = True
       # Use very low learning rate (e.g., 1e-6)
       model.compile(optimizer=low_lr_optimizer, loss=loss_fn)
       # Train for ~10-15 epochs
       ```
    
    Next steps:
    - For crack detection, create a model with 1 detection class and 1 segmentation class
    - Load these pre-trained weights into your crack detection model
    - Use the two-phase fine-tuning approach for best results
    
    Happy fine-tuning! 🚀
    """

    instructions_path = os.path.join(results_dir, 'usage_instructions.txt')
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(instructions)
    logger.info(f"📋 Usage instructions saved to: {instructions_path}")

if __name__ == '__main__':
    main()