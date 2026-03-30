"""COCO pre-training script for YOLOv12 feature extractor (detection + segmentation)."""

import os
import json
import keras
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Tuple, List

import matplotlib
from matplotlib import patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.layers.vision_heads.task_types import TaskType
from dl_techniques.models.yolo12.multitask import create_yolov12_multitask
from dl_techniques.losses.yolo12_multitask_loss import create_yolov12_multitask_loss
from dl_techniques.datasets.vision.coco import COCODatasetBuilder, COCO_CLASSES as COCO_CLASSES_ORIGINAL


@keras.saving.register_keras_serializable()
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule combining linear warmup and cosine decay."""

    def __init__(self, initial_learning_rate: float, decay_steps: int,
                 warmup_steps: int, alpha: float = 0.0, name: str = None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.name = name
        self.cosine_decay = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=self.decay_steps, alpha=self.alpha
        )

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupCosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            warmup_steps_tf = tf.cast(self.warmup_steps, dtype)
            global_step_tf = tf.cast(step, dtype)
            warmup_percent_done = global_step_tf / warmup_steps_tf
            warmup_learning_rate = initial_learning_rate * warmup_percent_done
            is_warmup = global_step_tf < warmup_steps_tf
            cosine_step = global_step_tf - warmup_steps_tf
            return tf.cond(
                is_warmup,
                lambda: warmup_learning_rate,
                lambda: self.cosine_decay(cosine_step)
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
            "name": self.name,
        }


def create_coco_model_and_loss(
        scale: str, img_size: int,
        detection_classes: int = 80, segmentation_classes: int = 80,
        use_uncertainty_weighting: bool = False
) -> Tuple[keras.Model, keras.losses.Loss]:
    """Create YOLOv12 multi-task model and loss for COCO pre-training."""
    logger.info(f"Creating YOLOv12-{scale} for COCO (det={detection_classes}, seg={segmentation_classes})")

    model = create_yolov12_multitask(
        num_detection_classes=detection_classes,
        num_segmentation_classes=segmentation_classes,
        input_shape=(img_size, img_size, 3),
        scale=scale,
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION]
    )

    loss_fn = create_yolov12_multitask_loss(
        tasks=[TaskType.DETECTION, TaskType.SEGMENTATION],
        num_detection_classes=detection_classes,
        num_segmentation_classes=segmentation_classes,
        input_shape=(img_size, img_size),
        use_uncertainty_weighting=use_uncertainty_weighting,
        detection_weight=1.0, segmentation_weight=0.1
    )

    return model, loss_fn


def create_coco_callbacks(
        results_dir: str, validation_dataset=None, img_size: int = 640,
        monitor: str = 'val_loss', patience: int = 10,
        save_best_only: bool = True, enable_visualizations: bool = True,
        visualization_freq: int = 5
) -> list:
    """Create callbacks for COCO pre-training."""
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True,
            mode='min', verbose=1, min_delta=1e-4
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_coco_model.keras'),
            monitor=monitor, save_best_only=save_best_only,
            save_weights_only=False, mode='min', verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'coco_training_log.csv'), append=False
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=0
        ),
        ProgressLoggingCallback()
    ]

    if enable_visualizations and validation_dataset is not None:
        viz_callback = COCOVisualizationCallback(
            validation_dataset=validation_dataset, results_dir=results_dir,
            img_size=img_size, num_samples=4, visualization_freq=visualization_freq,
            confidence_threshold=0.3, reg_max=16
        )
        callbacks.append(viz_callback)
        logger.info(f"COCO visualization callback added (every {visualization_freq} epochs)")
    elif enable_visualizations:
        logger.warning("Visualizations enabled but no validation dataset provided")

    return callbacks


class ProgressLoggingCallback(keras.callbacks.Callback):
    """Progress logging during COCO pre-training."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        loss_str = f"Loss: {logs.get('loss', 0):.4f}"
        if 'val_loss' in logs:
            loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"
        task_losses = []
        for task in ['detection', 'segmentation']:
            if f'{task}_loss' in logs:
                task_losses.append(f"{task}: {logs[f'{task}_loss']:.4f}")
        if task_losses:
            loss_str += f" | {', '.join(task_losses)}"
        if 'lr' in logs:
            loss_str += f" | LR: {logs['lr']:.2e}"
        logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")


class COCOVisualizationCallback(keras.callbacks.Callback):
    """Callback for per-epoch COCO detection and segmentation visualizations."""

    COCO_CLASSES = COCO_CLASSES_ORIGINAL

    def __init__(self, validation_dataset, results_dir: str, img_size: int = 640,
                 num_samples: int = 4, visualization_freq: int = 5,
                 confidence_threshold: float = 0.3, reg_max: int = 16):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.results_dir = results_dir
        self.img_size = img_size
        self.num_samples = num_samples
        self.visualization_freq = visualization_freq
        self.confidence_threshold = confidence_threshold
        self.reg_max = reg_max

        self.viz_dir = os.path.join(results_dir, 'coco_epoch_visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        self._prepare_visualization_samples()
        logger.info(f"COCO Visualization Callback initialized - saving to {self.viz_dir}")

    def _prepare_visualization_samples(self):
        """Prepare a fixed set of samples for consistent visualization across epochs."""
        try:
            sample_batch = next(iter(self.validation_dataset.take(1)))
            if isinstance(sample_batch, tuple) and len(sample_batch) == 2:
                self.sample_images, self.sample_targets = sample_batch
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
                logger.error("Unexpected COCO sample batch format")
                self.sample_images = None
                self.sample_targets = None
        except Exception as e:
            logger.error(f"Failed to prepare COCO visualization samples: {e}")
            self.sample_images = None
            self.sample_targets = None

    def _get_anchor_grid_and_strides(self):
        """Generate anchor grid and corresponding strides for YOLOv12 decoding."""
        if hasattr(self, '_anchor_cache'):
            return self._anchor_cache

        strides_config = [8, 16, 32]
        anchor_points, stride_tensor = [], []

        for stride in strides_config:
            h, w = self.img_size // stride, self.img_size // stride
            x_coords = (np.arange(w) + 0.5)
            y_coords = (np.arange(h) + 0.5)
            y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
            xy_grid = np.stack([x_grid, y_grid], axis=-1).reshape(-1, 2)
            anchor_points.append(xy_grid * stride)
            level_strides = np.full((h * w, 1), stride, dtype=np.float32)
            stride_tensor.append(level_strides)

        self._anchor_cache = (
            np.concatenate(anchor_points, 0),
            np.concatenate(stride_tensor, 0)
        )
        return self._anchor_cache

    def _decode_coco_detections(self, detection_pred: np.ndarray,
                                max_detections: int = 100) -> List[Dict[str, any]]:
        """Decode YOLOv12 detection predictions for COCO dataset."""
        try:
            dist_features = 4 * self.reg_max
            batch_size, num_anchors, total_features = detection_pred.shape
            num_classes = total_features - dist_features

            dist_pred = detection_pred[0, :, :dist_features]
            cls_pred = detection_pred[0, :, dist_features:]
            cls_scores = 1 / (1 + np.exp(-cls_pred))

            max_scores = np.max(cls_scores, axis=1)
            best_classes = np.argmax(cls_scores, axis=1)

            valid_indices = np.where(max_scores > self.confidence_threshold)[0]
            if len(valid_indices) == 0:
                return []

            if len(valid_indices) > max_detections:
                top_indices = np.argsort(max_scores[valid_indices])[-max_detections:]
                valid_indices = valid_indices[top_indices]

            valid_dist_logits = dist_pred[valid_indices]
            valid_scores = cls_scores[valid_indices]
            valid_classes = best_classes[valid_indices]

            valid_dist_logits = valid_dist_logits.reshape(-1, 4, self.reg_max)

            # Numerically stable softmax
            max_logits = np.max(valid_dist_logits, axis=-1, keepdims=True)
            stable_logits = valid_dist_logits - max_logits
            exp_logits = np.exp(stable_logits)
            softmax_dist = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            range_vals = np.arange(self.reg_max)
            decoded_dist = np.sum(softmax_dist * range_vals, axis=-1)

            anchor_grid, stride_tensor = self._get_anchor_grid_and_strides()
            valid_anchor_coords = anchor_grid[valid_indices]
            valid_strides = stride_tensor[valid_indices]

            pixel_dist = decoded_dist * valid_strides

            detections = []
            for i in range(len(valid_indices)):
                anchor_coords = valid_anchor_coords[i]
                distances = pixel_dist[i]
                scores = valid_scores[i]
                class_id = valid_classes[i]

                cx, cy = anchor_coords
                l, t, r, b = distances
                x1 = max(0, cx - l)
                y1 = max(0, cy - t)
                x2 = min(self.img_size, cx + r)
                y2 = min(self.img_size, cy + b)

                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(
                        self.COCO_CLASSES) and class_id < num_classes else f'class_{class_id}'
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': class_name
                    })

            return detections
        except Exception as e:
            logger.error(f"Failed to decode COCO detections: {e}")
            return []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.visualization_freq != 0:
            return
        if self.sample_images is None:
            return

        try:
            predictions = self.model(self.sample_images, training=False)
            epoch_dir = os.path.join(self.viz_dir, f'epoch_{epoch + 1:03d}')
            os.makedirs(epoch_dir, exist_ok=True)
            self._visualize_coco_detections(predictions, epoch_dir, epoch + 1)
            self._visualize_coco_segmentation(predictions, epoch_dir, epoch + 1)
            logger.info(f"Generated COCO visualizations for epoch {epoch + 1}")
        except Exception as e:
            logger.error(f"Failed to generate COCO visualizations for epoch {epoch + 1}: {e}")

    def _visualize_coco_detections(self, predictions, save_dir: str, epoch: int):
        """Generate COCO detection visualization with bounding boxes."""
        try:
            if isinstance(predictions, dict):
                detection_pred = predictions.get('detection')
            else:
                detection_pred = predictions
            if detection_pred is None:
                return

            gt_boxes = None
            if isinstance(self.sample_targets, dict):
                gt_boxes = self.sample_targets.get('detection')

            images_np = self.sample_images.numpy()
            pred_np = detection_pred.numpy()
            gt_np = gt_boxes.numpy() if gt_boxes is not None else None

            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()

            for i in range(min(4, len(images_np))):
                if i >= len(axes):
                    break
                axes[i].imshow(images_np[i])
                axes[i].set_title(f'COCO Sample {i + 1} - Epoch {epoch}', fontsize=12)
                axes[i].axis('off')

                if gt_np is not None and i < len(gt_np):
                    for gt_box in gt_np[i]:
                        if gt_box.sum() == 0:
                            continue
                        if len(gt_box) >= 5 and gt_box[0] >= 0:
                            class_id, x1, y1, x2, y2 = gt_box[:5]
                            x1_px = x1 * self.img_size
                            y1_px = y1 * self.img_size
                            x2_px = x2 * self.img_size
                            y2_px = y2 * self.img_size
                            rect = patches.Rectangle(
                                (x1_px, y1_px), x2_px - x1_px, y2_px - y1_px,
                                linewidth=2, edgecolor='green', facecolor='none',
                                linestyle='-', alpha=0.8
                            )
                            axes[i].add_patch(rect)
                            class_name = self.COCO_CLASSES[int(class_id)] if int(class_id) < len(
                                self.COCO_CLASSES) else f'class_{int(class_id)}'
                            axes[i].text(
                                x1_px, y1_px - 5, f'GT: {class_name}',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="green", alpha=0.8),
                                color='white', fontsize=8, fontweight='bold'
                            )

                detections = self._decode_coco_detections(pred_np[i:i + 1])
                for det in detections[:5]:
                    x1, y1, x2, y2 = det['bbox']
                    pred_rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', alpha=0.9
                    )
                    axes[i].add_patch(pred_rect)
                    axes[i].text(
                        x1, max(0, y1 - 15),
                        f"{det['class_name']}: {det['confidence']:.2f}",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8),
                        color='white', fontsize=8, fontweight='bold'
                    )

            legend_elements = [
                patches.Patch(color='green', label='Ground Truth'),
                patches.Patch(color='red', label='Predictions')
            ]
            fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
            plt.suptitle(f'COCO Detection Results - Epoch {epoch}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'coco_detection_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to visualize COCO detections: {e}")

    def _visualize_coco_segmentation(self, predictions, save_dir: str, epoch: int):
        """Generate COCO segmentation visualization."""
        try:
            if isinstance(predictions, dict):
                seg_pred = predictions.get('segmentation')
            else:
                return
            if seg_pred is None:
                return

            seg_gt = None
            if isinstance(self.sample_targets, dict):
                seg_gt = self.sample_targets.get('segmentation')

            images_np = self.sample_images.numpy()
            pred_np = tf.nn.sigmoid(seg_pred).numpy()
            gt_np = seg_gt.numpy() if seg_gt is not None else None

            n_cols = 3 if gt_np is not None else 2
            fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))

            for i in range(min(2, len(images_np))):
                axes[i, 0].imshow(images_np[i])
                axes[i, 0].set_title(f'COCO Image {i + 1}', fontsize=12)
                axes[i, 0].axis('off')

                if pred_np.shape[-1] == 1:
                    pred_mask = pred_np[i, ..., 0]
                else:
                    pred_mask = np.argmax(pred_np[i], axis=-1)

                im_pred = axes[i, 1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1 if pred_np.shape[-1] == 1 else 79)
                axes[i, 1].set_title(f'Predicted Mask {i + 1}', fontsize=12)
                axes[i, 1].axis('off')
                plt.colorbar(im_pred, ax=axes[i, 1], shrink=0.8)

                if gt_np is not None and i < len(gt_np):
                    if gt_np.shape[-1] == 1:
                        gt_mask = gt_np[i, ..., 0]
                    else:
                        gt_mask = np.argmax(gt_np[i], axis=-1)

                    im_gt = axes[i, 2].imshow(gt_mask, cmap='hot', vmin=0, vmax=1 if gt_np.shape[-1] == 1 else 79)
                    axes[i, 2].set_title(f'Ground Truth {i + 1}', fontsize=12)
                    axes[i, 2].axis('off')
                    plt.colorbar(im_gt, ax=axes[i, 2], shrink=0.8)

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
            plt.savefig(os.path.join(save_dir, 'coco_segmentation_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to visualize COCO segmentation: {e}")


def save_feature_extractor_weights(
        model: keras.Model, save_path: str, scale: str, img_size: int,
        detection_classes: int = 80, segmentation_classes: int = 80
) -> None:
    """Extract and save the feature extractor weights for fine-tuning."""
    try:
        feature_extractor = model.get_layer('shared_feature_extractor')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        feature_extractor.save_weights(save_path)
        logger.info(f"Feature extractor weights saved to: {save_path}")

        metadata = {
            'model_scale': scale, 'input_size': img_size,
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
        logger.info(f"Metadata saved to: {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to save feature extractor weights: {e}")
        raise


def calculate_steps_per_epoch(
        dataset: tf.data.Dataset, batch_size: int,
        default_size: int = 118000, min_steps: int = 100, max_steps: int = 1000
) -> int:
    """Calculate appropriate steps per epoch based on actual dataset size."""
    try:
        sample_count = 0
        max_sample_batches = min_steps
        try:
            for batch in dataset.take(max_sample_batches):
                sample_count += 1

            if sample_count == 0:
                logger.warning("Dataset appears empty, using minimum steps.")
                return min_steps
            elif sample_count < min_steps:
                steps = min(sample_count * 10, max_steps)
                logger.info(f"Small dataset detected, using {steps} steps per epoch")
            else:
                steps = max(default_size // batch_size, min_steps)
        except tf.errors.OutOfRangeError:
            logger.warning("Dataset exhausted during sampling, using minimum steps")
            return min_steps
    except Exception as e:
        logger.warning(f"Could not determine dataset size: {e}")
        steps = min_steps

    logger.info(f"Steps per epoch: {steps}")
    return steps


def check_dependencies():
    """Check if all required dependencies are available."""
    try:
        import tensorflow_datasets as tfds
        logger.info(f"tensorflow-datasets available (version: {tfds.__version__})")
        return True
    except ImportError:
        logger.error("tensorflow-datasets not found. Install with: pip install tensorflow-datasets")
        return False


def main():
    """Main COCO pre-training function."""
    parser = argparse.ArgumentParser(
        description='Pre-train YOLOv12 Feature Extractor on COCO Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--detection-classes', type=int, default=80)
    parser.add_argument('--segmentation-classes', type=int, default=80)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])

    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--max-boxes', type=int, default=100)
    parser.add_argument('--shuffle-buffer', type=int, default=100)
    parser.add_argument('--limit-train-samples', type=int, default=None)

    parser.add_argument('--uncertainty-weighting', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='coco_pretrain_results')
    parser.add_argument('--weights-name', type=str, default=None)
    parser.add_argument('--run-eagerly', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)

    parser.add_argument('--no-visualizations', action='store_true')
    parser.add_argument('--visualization-freq', type=int, default=1)
    parser.add_argument('--viz-samples', type=int, default=4)
    parser.add_argument('--confidence-threshold', type=float, default=0.3)

    args = parser.parse_args()

    if not check_dependencies():
        return

    setup_gpu()

    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.save_dir}/yolov12_{args.scale}_coco_pretrain_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Starting COCO Pre-training for YOLOv12-{args.scale}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Results directory: {results_dir}")

    if args.limit_train_samples:
        logger.info(f"Training limited to {args.limit_train_samples} samples")

    dataset_builder = COCODatasetBuilder(
        img_size=args.img_size, batch_size=args.batch_size,
        max_boxes_per_image=args.max_boxes, cache_dir=args.cache_dir,
        use_detection=True, use_segmentation=True,
        segmentation_classes=args.segmentation_classes,
        augment_data=True, shuffle_buffer_size=args.shuffle_buffer,
        limit_train_samples=args.limit_train_samples
    )

    train_ds, val_ds = dataset_builder.create_datasets()
    dataset_info = dataset_builder.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")

    model, loss_fn = create_coco_model_and_loss(
        scale=args.scale, img_size=args.img_size,
        detection_classes=args.detection_classes,
        segmentation_classes=args.segmentation_classes,
        use_uncertainty_weighting=args.uncertainty_weighting
    )

    if args.limit_train_samples:
        steps_per_epoch = args.limit_train_samples // args.batch_size
    else:
        steps_per_epoch = calculate_steps_per_epoch(
            dataset=train_ds, batch_size=args.batch_size,
            default_size=118000, min_steps=100, max_steps=500
        )

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=total_steps - warmup_steps,
        warmup_steps=warmup_steps, alpha=0.1
    )

    sample_input = tf.zeros((1, args.img_size, args.img_size, 3))
    _ = model(sample_input, training=False)

    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=args.weight_decay, clipnorm=1.0
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True, clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    logger.info(f"Using optimizer: {type(optimizer).__name__}")
    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=args.run_eagerly, jit_compile=True)

    dataset_info = dataset_builder.get_dataset_info()
    use_validation = not dataset_info.get('using_dummy_data', False)
    monitor = 'val_loss' if use_validation else 'loss'

    enable_visualizations = not args.no_visualizations and use_validation
    callbacks = create_coco_callbacks(
        results_dir=results_dir,
        validation_dataset=val_ds if use_validation else None,
        img_size=args.img_size, monitor=monitor, patience=args.patience,
        enable_visualizations=enable_visualizations,
        visualization_freq=args.visualization_freq
    )

    logger.info("Starting COCO pre-training...")
    try:
        validation_data = val_ds if use_validation else None
        validation_steps = None if not use_validation else 50

        history = model.fit(
            train_ds, validation_data=validation_data,
            epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps, callbacks=callbacks, verbose=1
        )
        logger.info("COCO pre-training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save feature extractor weights
    if args.weights_name:
        weights_filename = f"{args.weights_name}.weights.h5"
    else:
        weights_filename = f"yolov12_{args.scale}_coco_pretrained_feature_extractor.weights.h5"

    weights_path = os.path.join(results_dir, weights_filename)
    save_feature_extractor_weights(
        model=model, save_path=weights_path, scale=args.scale,
        img_size=args.img_size, detection_classes=args.detection_classes,
        segmentation_classes=args.segmentation_classes
    )

    config = {
        'training_args': vars(args),
        'dataset_info': dataset_info,
        'model_info': {
            'scale': args.scale, 'input_size': args.img_size,
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
    logger.info(f"Configuration saved to: {config_path}")

    instructions = f"""COCO Pre-training Completed!

Feature extractor saved to: {weights_path}

Configuration:
- Detection classes: {args.detection_classes}
- Segmentation classes: {args.segmentation_classes}
- Model scale: {args.scale}
- Input size: {args.img_size}

To use pre-trained weights for fine-tuning:
  feature_extractor = model.get_layer('shared_feature_extractor')
  feature_extractor.load_weights('{weights_path}')

Phase 2a (freeze feature extractor, train heads):
  feature_extractor.trainable = False
  # Train for ~20-30 epochs

Phase 2b (end-to-end fine-tuning with low LR):
  feature_extractor.trainable = True
  # Train for ~10-15 epochs with LR ~1e-6
"""

    instructions_path = os.path.join(results_dir, 'usage_instructions.txt')
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    logger.info(f"Usage instructions saved to: {instructions_path}")


if __name__ == '__main__':
    main()
