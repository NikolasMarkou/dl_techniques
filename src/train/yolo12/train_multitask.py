"""Training script for YOLOv12 multi-task model (detection + segmentation + classification)."""

import os
import sys
import keras
import json
import argparse
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.sut import OptimizedSUTDataset
from dl_techniques.models.yolo12.multitask import create_yolov12_multitask
from dl_techniques.layers.vision_heads.task_types import (
    TaskType,
    TaskConfiguration,
    parse_task_list
)
from dl_techniques.losses.yolo12_multitask_loss import (
    create_yolov12_multitask_loss,
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def parse_tasks(task_strings: List[str]) -> TaskConfiguration:
    """Parse task strings into TaskConfiguration."""
    try:
        task_config = parse_task_list(task_strings)
        logger.info(f"Parsed tasks: {task_config.get_task_names()}")
        return task_config
    except ValueError as e:
        logger.error(f"Invalid task configuration: {e}")
        logger.info(f"Valid tasks are: {[t.value for t in TaskType.all_tasks()]}")
        raise


class PerEpochVisualizationCallback(keras.callbacks.Callback):
    """Callback for generating per-epoch visualizations for all tasks."""

    def __init__(
        self,
        validation_dataset,
        task_config: TaskConfiguration,
        results_dir: str,
        num_samples: int = 8,
        visualization_freq: int = 1,
        patch_size: int = 256
    ):
        super().__init__()
        self.validation_dataset = validation_dataset
        self.task_config = task_config
        self.results_dir = results_dir
        self.num_samples = num_samples
        self.visualization_freq = visualization_freq
        self.patch_size = patch_size

        self.viz_dir = os.path.join(results_dir, 'epoch_visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        self._prepare_visualization_samples()
        logger.info(f"PerEpochVisualizationCallback initialized for tasks: {task_config.get_task_names()}")

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
                logger.info(f"Prepared {tf.shape(self.sample_images)[0]} samples for visualization")
            else:
                logger.error("Unexpected sample batch format for visualization")
                self.sample_images = None
                self.sample_targets = None
        except Exception as e:
            logger.error(f"Failed to prepare visualization samples: {e}")
            self.sample_images = None
            self.sample_targets = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.visualization_freq != 0:
            return
        if self.sample_images is None:
            return

        try:
            predictions = self.model(self.sample_images, training=False)
            epoch_dir = os.path.join(self.viz_dir, f'epoch_{epoch+1:03d}')
            os.makedirs(epoch_dir, exist_ok=True)

            if self.task_config.has_detection():
                self._visualize_detection(predictions, epoch_dir, epoch + 1)
            if self.task_config.has_segmentation():
                self._visualize_segmentation(predictions, epoch_dir, epoch + 1)
            if self.task_config.has_classification():
                self._visualize_classification(predictions, epoch_dir, epoch + 1)

            logger.info(f"Generated visualizations for epoch {epoch + 1}")
        except Exception as e:
            logger.error(f"Failed to generate visualizations for epoch {epoch + 1}: {e}")

    def _visualize_detection(self, predictions, save_dir: str, epoch: int):
        """Generate detection visualization with bounding boxes."""
        try:
            if isinstance(predictions, dict):
                detection_pred = predictions.get('detection')
            else:
                detection_pred = predictions
            if detection_pred is None:
                return

            gt_boxes_normalized = None
            if isinstance(self.sample_targets, dict):
                gt_boxes_normalized = self.sample_targets.get('detection')

            images_np = self.sample_images.numpy()
            pred_np = detection_pred.numpy()
            gt_np = gt_boxes_normalized.numpy() if gt_boxes_normalized is not None else None

            reg_max = 16
            pred_dist, pred_scores = np.split(pred_np, [4 * reg_max], axis=-1)
            pred_dist = pred_dist.reshape((pred_np.shape[0], -1, 4, reg_max))
            softmax_dist = np.exp(pred_dist) / (np.sum(np.exp(pred_dist), axis=-1, keepdims=True) + 1e-8)
            decoded_dist = np.sum(softmax_dist * np.arange(reg_max), axis=-1)
            pred_scores_sigmoid = 1 / (1 + np.exp(-pred_scores))
            best_pred_indices = np.argmax(pred_scores_sigmoid.reshape(pred_np.shape[0], -1), axis=1)

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            for i, img in enumerate(images_np[:8]):
                if i >= len(axes):
                    break
                img_display = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                axes[i].imshow(img_display)
                axes[i].set_title(f'Sample {i+1}', fontsize=10)
                axes[i].axis('off')

                if gt_np is not None and i < len(gt_np):
                    for gt_box in gt_np[i]:
                        if gt_box.sum() == 0:
                            continue
                        if len(gt_box) >= 5:
                            _, x1, y1, x2, y2 = gt_box[:5]
                        else:
                            continue
                        x1_px, y1_px = x1 * self.patch_size, y1 * self.patch_size
                        x2_px, y2_px = x2 * self.patch_size, y2 * self.patch_size
                        rect = patches.Rectangle(
                            (x1_px, y1_px), x2_px - x1_px, y2_px - y1_px,
                            linewidth=2, edgecolor='green', facecolor='none',
                            linestyle='-', alpha=0.8
                        )
                        axes[i].add_patch(rect)

                max_confidence = np.max(pred_scores_sigmoid[i])
                color = 'red' if max_confidence > 0.5 else 'orange'
                axes[i].text(5, 20, f'Max Conf: {max_confidence:.3f}',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                           color='white', fontsize=9, fontweight='bold')

            legend_elements = []
            if gt_np is not None:
                legend_elements.append(patches.Patch(color='green', label='Ground Truth'))
            legend_elements.append(patches.Patch(color='red', label='High Confidence (>0.5)'))
            legend_elements.append(patches.Patch(color='orange', label='Low Confidence (<=0.5)'))

            if legend_elements:
                fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

            plt.suptitle(f'Detection Results - Epoch {epoch}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(save_dir, 'detection_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            import traceback
            logger.error(f"Failed to visualize detection: {e}\n{traceback.format_exc()}")

    def _visualize_segmentation(self, predictions, save_dir: str, epoch: int):
        """Generate segmentation visualization with masks."""
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
            fig, axes = plt.subplots(4, n_cols, figsize=(15, 16))

            for i in range(min(4, len(images_np))):
                img_display = ((images_np[i] - images_np[i].min()) /
                              (images_np[i].max() - images_np[i].min() + 1e-8) * 255).astype(np.uint8)
                axes[i, 0].imshow(img_display)
                axes[i, 0].set_title(f'Input {i+1}', fontsize=12)
                axes[i, 0].axis('off')

                pred_mask = pred_np[i, ..., 0] if len(pred_np.shape) == 4 else pred_np[i]
                im_pred = axes[i, 1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
                axes[i, 1].set_title(f'Predicted Mask {i+1}', fontsize=12)
                axes[i, 1].axis('off')
                plt.colorbar(im_pred, ax=axes[i, 1], shrink=0.8)

                if gt_np is not None and i < len(gt_np):
                    gt_mask = gt_np[i, ..., 0] if len(gt_np.shape) == 4 else gt_np[i]
                    im_gt = axes[i, 2].imshow(gt_mask, cmap='hot', vmin=0, vmax=1)
                    axes[i, 2].set_title(f'Ground Truth {i+1}', fontsize=12)
                    axes[i, 2].axis('off')
                    plt.colorbar(im_gt, ax=axes[i, 2], shrink=0.8)

                    pred_binary = (pred_mask > 0.5).astype(float)
                    gt_binary = (gt_mask > 0.5).astype(float)
                    intersection = np.sum(pred_binary * gt_binary)
                    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
                    iou = intersection / (union + 1e-8)
                    axes[i, 1].text(5, pred_mask.shape[0] - 20, f'IoU: {iou:.3f}',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8),
                                   fontsize=10, fontweight='bold')

            plt.suptitle(f'Segmentation Results - Epoch {epoch}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(save_dir, 'segmentation_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            import traceback
            logger.error(f"Failed to visualize segmentation: {e}\n{traceback.format_exc()}")

    def _visualize_classification(self, predictions, save_dir: str, epoch: int):
        """Generate classification visualization with confidence scores."""
        try:
            if isinstance(predictions, dict):
                cls_pred = predictions.get('classification')
            else:
                return
            if cls_pred is None:
                return

            cls_gt = None
            if isinstance(self.sample_targets, dict):
                cls_gt = self.sample_targets.get('classification')

            images_np = self.sample_images.numpy()
            pred_probs = tf.nn.sigmoid(cls_pred).numpy()
            gt_labels = cls_gt.numpy() if cls_gt is not None else None

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            correct_predictions = 0
            total_predictions = 0

            for i, (img, pred_prob) in enumerate(zip(images_np[:8], pred_probs[:8])):
                if i >= len(axes):
                    break
                img_display = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
                axes[i].imshow(img_display)

                prob = pred_prob[0] if len(pred_prob.shape) > 0 and pred_prob.shape[0] > 0 else float(pred_prob)
                pred_class = "Crack" if prob > 0.5 else "No Crack"
                confidence = prob if prob > 0.5 else 1 - prob
                pred_color = 'red' if pred_class == "Crack" else 'green'
                title = f'Sample {i+1}\nPred: {pred_class} ({confidence:.3f})'

                if gt_labels is not None and i < len(gt_labels):
                    gt_prob = gt_labels[i][0] if len(gt_labels[i]) > 0 else gt_labels[i]
                    gt_class = "Crack" if gt_prob > 0.5 else "No Crack"
                    title += f'\nGT: {gt_class}'
                    pred_correct = (prob > 0.5) == (gt_prob > 0.5)
                    if pred_correct:
                        correct_predictions += 1
                        border_color = 'lime'
                    else:
                        border_color = 'red'
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3)
                    total_predictions += 1

                axes[i].set_title(title, fontsize=9, color=pred_color, fontweight='bold')
                axes[i].axis('off')

                bar_width = confidence * (img.shape[1] - 20)
                bar_x, bar_y = 10, img.shape[0] - 25
                axes[i].add_patch(patches.Rectangle(
                    (bar_x, bar_y), bar_width, 15,
                    facecolor=pred_color, alpha=0.7, edgecolor='black'
                ))
                axes[i].text(bar_x + bar_width/2, bar_y + 7.5, f'{confidence:.2f}',
                           ha='center', va='center', fontsize=8, fontweight='bold', color='white')

            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                plt.figtext(0.5, 0.02, f'Batch Accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})',
                           ha='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

            legend_elements = [
                patches.Patch(color='red', label='Crack Prediction'),
                patches.Patch(color='green', label='No Crack Prediction'),
            ]
            if gt_labels is not None:
                legend_elements.extend([
                    patches.Patch(color='lime', label='Correct Prediction'),
                    patches.Patch(color='red', label='Incorrect Prediction', linestyle='--')
                ])
            fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

            plt.suptitle(f'Classification Results - Epoch {epoch}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(save_dir, 'classification_results.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            import traceback
            logger.error(f"Failed to visualize classification: {e}\n{traceback.format_exc()}")


class EnhancedMultiTaskCallback(keras.callbacks.Callback):
    """Callback for multitask training monitoring with Named Outputs support."""

    def __init__(self, loss_fn=None, validation_dataset=None, validation_steps=None,
                 log_dir=None, log_freq: int = 1):
        super().__init__()
        self.loss_fn = loss_fn
        self.validation_dataset = validation_dataset
        self.validation_steps = validation_steps
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.epoch_losses = []
        self.task_weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if self.loss_fn and hasattr(self.loss_fn, 'get_task_weights'):
            try:
                task_weights = self.loss_fn.get_task_weights()
                for task, weight in task_weights.items():
                    logs[f'{task}_weight'] = float(weight)
                self.task_weight_history.append(task_weights.copy())
            except Exception as e:
                logger.warning(f"Failed to get task weights: {e}")

        if self.loss_fn and hasattr(self.loss_fn, 'get_individual_losses'):
            try:
                individual_losses = self.loss_fn.get_individual_losses()
                for task, loss_val in individual_losses.items():
                    logs[f'{task}_loss'] = float(loss_val)
            except Exception as e:
                logger.warning(f"Failed to get individual losses: {e}")

        self.epoch_losses.append(logs.copy())

        if hasattr(self.model.optimizer, 'learning_rate'):
            try:
                lr = float(self.model.optimizer.learning_rate)
                logs['learning_rate'] = lr
            except Exception:
                pass

        if (epoch + 1) % self.log_freq == 0:
            loss_str = f"Loss: {logs.get('loss', 0):.4f}"
            if 'val_loss' in logs:
                loss_str += f", Val Loss: {logs.get('val_loss', 0):.4f}"
            task_losses = []
            for task in ['detection', 'segmentation', 'classification']:
                if f'{task}_loss' in logs:
                    task_losses.append(f"{task}: {logs[f'{task}_loss']:.4f}")
            if task_losses:
                loss_str += f" | {', '.join(task_losses)}"
            logger.info(f"Epoch {epoch + 1}/{self.params.get('epochs', '?')} - {loss_str}")

    def get_task_weight_history(self) -> List[Dict[str, float]]:
        return self.task_weight_history


def create_dataset_splits(
    data_dir: str,
    patch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[OptimizedSUTDataset, OptimizedSUTDataset, OptimizedSUTDataset]:
    """Create train/validation/test dataset splits."""
    full_dataset = OptimizedSUTDataset(
        data_dir=data_dir, patch_size=patch_size,
        patches_per_image=1, include_segmentation=True
    )

    np.random.seed(random_seed)
    annotations = full_dataset.annotations.copy()
    np.random.shuffle(annotations)

    n_total = len(annotations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_annotations = annotations[:n_train]
    val_annotations = annotations[n_train:n_train + n_val]
    test_annotations = annotations[n_train + n_val:]

    logger.info(f"Dataset splits - Train: {len(train_annotations)}, "
               f"Val: {len(val_annotations)}, Test: {len(test_annotations)}")

    train_dataset = OptimizedSUTDataset(
        data_dir=data_dir, patch_size=patch_size,
        patches_per_image=16, positive_ratio=0.7, include_segmentation=True
    )
    train_dataset.annotations = train_annotations

    val_dataset = OptimizedSUTDataset(
        data_dir=data_dir, patch_size=patch_size,
        patches_per_image=8, positive_ratio=0.5, include_segmentation=True
    )
    val_dataset.annotations = val_annotations

    test_dataset = OptimizedSUTDataset(
        data_dir=data_dir, patch_size=patch_size,
        patches_per_image=8, positive_ratio=0.5, include_segmentation=True
    )
    test_dataset.annotations = test_annotations

    return train_dataset, val_dataset, test_dataset


def create_model_and_loss(
    task_config: TaskConfiguration,
    patch_size: int,
    scale: str,
    num_classes: int = 1,
    use_uncertainty_weighting: bool = False,
    **loss_kwargs
) -> Tuple[keras.Model, keras.losses.Loss]:
    """Create multitask model and loss function with TaskType enum support."""
    logger.info(f"Creating model with tasks: {task_config.get_task_names()}")

    model = create_yolov12_multitask(
        num_classes=num_classes,
        input_shape=(patch_size, patch_size, 3),
        scale=scale,
        tasks=task_config
    )

    loss_fn = create_yolov12_multitask_loss(
        tasks=task_config,
        num_classes=num_classes,
        input_shape=(patch_size, patch_size),
        use_uncertainty_weighting=use_uncertainty_weighting,
        **loss_kwargs
    )

    return model, loss_fn


def test_model_compilation(model: keras.Model, train_dataset,
                          task_config: TaskConfiguration,
                          run_eagerly: bool = False) -> bool:
    """Test model compilation with sample data to ensure compatibility."""
    logger.info("Testing model compilation...")
    try:
        sample_batch = next(iter(train_dataset))
        sample_x, sample_y = sample_batch

        logger.info(f"Sample input shape: {sample_x.shape}")
        if isinstance(sample_y, dict):
            target_info = {k: str(v.shape) for k, v in sample_y.items()}
            logger.info(f"Sample targets (dict): {target_info}")
        else:
            logger.info(f"Sample targets: {sample_y.shape}")

        predictions = model(sample_x, training=False)
        if isinstance(predictions, dict):
            pred_info = {k: str(v.shape) for k, v in predictions.items()}
            logger.info(f"Model outputs (dict): {pred_info}")
        else:
            logger.info(f"Model output (tensor): {predictions.shape}")

        loss_value = model.evaluate(sample_x, sample_y, steps=1, verbose=0)
        logger.info(f"Model compilation test successful")
        return True
    except Exception as e:
        logger.error(f"Model compilation test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if not run_eagerly:
            logger.info("Suggestion: Try running with --run-eagerly flag")
        return False


def create_callbacks(
    results_dir: str,
    loss_fn: keras.losses.Loss,
    task_config: TaskConfiguration,
    val_dataset,
    validation_steps: Optional[int],
    patch_size: int = 256,
    monitor: str = 'val_loss',
    patience: int = 20,
    enable_visualizations: bool = True,
    visualization_freq: int = 1
) -> List[keras.callbacks.Callback]:
    """Create training callbacks with multi-task support and visualizations."""
    callbacks = [
        EnhancedMultiTaskCallback(
            loss_fn=loss_fn, validation_dataset=val_dataset,
            validation_steps=validation_steps,
            log_dir=os.path.join(results_dir, 'logs'), log_freq=1
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True,
            mode='min', verbose=1, min_delta=1e-4
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(results_dir, 'best_model.keras'),
            monitor=monitor, save_best_only=True, save_weights_only=False,
            mode='min', verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=max(patience // 3, 5),
            min_lr=1e-7, verbose=1, cooldown=2
        ),
        keras.callbacks.CSVLogger(
            filename=os.path.join(results_dir, 'training_log.csv'),
            append=False, separator=','
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(results_dir, 'tensorboard'),
            histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=0
        ),
    ]

    if enable_visualizations:
        viz_callback = PerEpochVisualizationCallback(
            validation_dataset=val_dataset, task_config=task_config,
            results_dir=results_dir, num_samples=8,
            visualization_freq=visualization_freq, patch_size=patch_size
        )
        callbacks.append(viz_callback)
        logger.info(f"Added per-epoch visualization callback (freq: {visualization_freq})")

    if hasattr(loss_fn, 'use_uncertainty_weighting') and loss_fn.use_uncertainty_weighting:
        logger.info("Added uncertainty weighting monitoring")

    return callbacks


def train_model(args: argparse.Namespace) -> None:
    """Main training function for YOLOv12 multi-task with Named Outputs."""
    logger.info("Starting YOLOv12 Multi-Task training")
    logger.info(f"Arguments: {vars(args)}")

    setup_gpu()
    task_config = parse_tasks(args.tasks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks_str = "_".join(task_config.get_task_names())
    results_dir = f"results/yolov12_multitask_{args.scale}_{tasks_str}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        data_dir=args.data_dir, patch_size=args.patch_size, random_seed=args.random_seed
    )

    train_info = train_dataset.get_dataset_info()
    val_info = val_dataset.get_dataset_info()
    test_info = test_dataset.get_dataset_info()
    logger.info(f"Train: {train_info}, Val: {val_info}, Test: {test_info}")

    train_tf_dataset = train_dataset.create_tf_dataset(
        batch_size=args.batch_size, shuffle=True, repeat=True
    )
    val_tf_dataset = val_dataset.create_tf_dataset(
        batch_size=args.batch_size, shuffle=False, repeat=False
    )

    steps_per_epoch = max(train_info['total_patches_per_epoch'] // args.batch_size, 1)
    validation_steps = max(val_info['total_patches_per_epoch'] // args.batch_size, 1)
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

    model, loss_fn = create_model_and_loss(
        task_config=task_config, patch_size=args.patch_size, scale=args.scale,
        num_classes=1, use_uncertainty_weighting=args.uncertainty_weighting,
        detection_weight=args.detection_weight,
        segmentation_weight=args.segmentation_weight,
        classification_weight=args.classification_weight
    )

    sample_input = tf.zeros((1, args.patch_size, args.patch_size, 3))
    _ = model(sample_input, training=False)

    model.summary(print_fn=logger.info)
    logger.info(f"Total parameters: {model.count_params():,}")

    if args.optimizer.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, clipnorm=1.0
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=args.learning_rate, momentum=0.9, nesterov=True, clipnorm=1.0
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    logger.info(f"Using optimizer: {type(optimizer).__name__}")

    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=args.run_eagerly)

    compilation_success = test_model_compilation(
        model=model, train_dataset=train_tf_dataset,
        task_config=task_config, run_eagerly=args.run_eagerly
    )

    if not compilation_success and not args.run_eagerly:
        logger.warning("Compilation test failed, switching to eager execution...")
        model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=True)

    callbacks = create_callbacks(
        results_dir=results_dir, loss_fn=loss_fn, task_config=task_config,
        val_dataset=val_tf_dataset, validation_steps=None, patch_size=args.patch_size,
        monitor='val_loss', patience=args.patience,
        enable_visualizations=args.enable_visualizations,
        visualization_freq=args.visualization_freq
    )

    logger.info("Starting training...")
    try:
        history = model.fit(
            train_tf_dataset, validation_data=val_tf_dataset,
            epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            validation_steps=None, callbacks=callbacks, verbose=1
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if 'history' in locals():
            save_training_results(
                model=model, history=history, results_dir=results_dir,
                args=args, task_config=task_config, train_info=train_info,
                val_info=val_info, test_info=test_info, callbacks=callbacks
            )
        raise

    test_results = None
    if args.evaluate and test_dataset.annotations:
        logger.info("Evaluating on test set...")
        try:
            test_tf_dataset = test_dataset.create_tf_dataset(
                batch_size=args.batch_size, shuffle=False, repeat=False
            )
            test_steps = max(test_info['total_patches_per_epoch'] // args.batch_size, 1)
            test_loss = model.evaluate(test_tf_dataset, steps=test_steps, verbose=1)
            test_results = {'test_loss': float(test_loss)}
            logger.info(f"Test Loss: {test_loss:.6f}")
        except Exception as e:
            logger.error(f"Test evaluation failed: {e}")

    save_training_results(
        model=model, history=history, results_dir=results_dir, args=args,
        task_config=task_config, train_info=train_info, val_info=val_info,
        test_info=test_info, test_results=test_results, callbacks=callbacks
    )
    logger.info("Training completed successfully!")


def save_training_results(
    model: keras.Model,
    history: keras.callbacks.History,
    results_dir: str,
    args: argparse.Namespace,
    task_config: TaskConfiguration,
    train_info: Dict,
    val_info: Dict,
    test_info: Dict,
    test_results: Optional[Dict] = None,
    callbacks: Optional[List] = None
):
    """Save comprehensive training results."""
    viz_dir = os.path.join(results_dir, 'visualizations')
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    plot_enhanced_training_history(history, viz_dir, task_config)

    try:
        keras.utils.plot_model(
            model, to_file=os.path.join(viz_dir, 'model_architecture.png'),
            show_shapes=True, show_layer_names=True, expand_nested=False,
            dpi=150, rankdir='TB'
        )
    except Exception as e:
        logger.warning(f"Failed to save model architecture plot: {e}")

    config = {
        'training_args': vars(args),
        'task_configuration': {
            'enabled_tasks': task_config.get_task_names(),
            'is_single_task': task_config.is_single_task(),
            'is_multi_task': task_config.is_multi_task()
        },
        'model_config': {
            'scale': args.scale, 'patch_size': args.patch_size,
            'total_parameters': model.count_params(),
            'model_type': 'Named Outputs (Functional API)'
        },
        'dataset_info': {'train': train_info, 'validation': val_info, 'test': test_info},
        'training_results': {
            'epochs_completed': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            'best_validation_loss': float(min(history.history.get('val_loss', [float('inf')]))) if 'val_loss' in history.history else None,
        },
        'loss_configuration': {
            'type': 'YOLOv12MultiTaskLoss',
            'uses_uncertainty_weighting': args.uncertainty_weighting,
            'task_weights': {
                'detection': args.detection_weight,
                'segmentation': args.segmentation_weight,
                'classification': args.classification_weight
            }
        },
        'visualization_config': {
            'enabled': args.enable_visualizations,
            'frequency': args.visualization_freq,
            'per_epoch_visualizations_path': 'epoch_visualizations/'
        }
    }

    if test_results:
        config['test_results'] = test_results

    if callbacks:
        for callback in callbacks:
            if isinstance(callback, EnhancedMultiTaskCallback):
                weight_history = callback.get_task_weight_history()
                if weight_history:
                    config['task_weight_evolution'] = weight_history
                break

    with open(os.path.join(results_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    try:
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        saved_model_path = os.path.join(model_dir, 'saved_model')
        model.export(saved_model_path)
        logger.info(f"SavedModel exported to: {saved_model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    create_training_summary(results_dir, config, history, task_config)
    logger.info(f"All results saved to: {results_dir}")


def plot_enhanced_training_history(
    history: keras.callbacks.History,
    save_dir: str,
    task_config: TaskConfiguration
):
    """Plot comprehensive training history."""
    history_dict = history.history
    epochs = range(1, len(history_dict['loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    task_colors = {'detection': 'green', 'segmentation': 'orange', 'classification': 'purple'}

    # Total loss
    axes[0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        axes[0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Task-specific losses
    tasks_plotted = False
    for task in task_config.get_task_names():
        task_loss_key = f'{task}_loss'
        if task_loss_key in history_dict:
            color = task_colors.get(task, 'gray')
            axes[1].plot(epochs, history_dict[task_loss_key],
                        color=color, label=f'{task.title()} Loss', linewidth=2)
            tasks_plotted = True
    if tasks_plotted:
        axes[1].set_title('Task-Specific Losses', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Task-specific losses\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes)

    # Learning rate
    if 'learning_rate' in history_dict:
        axes[2].plot(epochs, history_dict['learning_rate'], 'orange', linewidth=2)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Learning rate\nnot tracked',
                     ha='center', va='center', transform=axes[2].transAxes)

    # Task weights
    weights_plotted = False
    for task in task_config.get_task_names():
        weight_key = f'{task}_weight'
        if weight_key in history_dict:
            color = task_colors.get(task, 'gray')
            axes[3].plot(epochs, history_dict[weight_key],
                        color=color, label=f'{task.title()} Weight', linewidth=2)
            weights_plotted = True
    if weights_plotted:
        axes[3].set_title('Task Weights (Uncertainty)', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Weight')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'Fixed task weighting\n(No uncertainty)',
                     ha='center', va='center', transform=axes[3].transAxes)

    # Loss smoothing
    if len(epochs) > 5:
        window = max(3, len(epochs) // 10)
        loss_ma = pd.Series(history_dict['loss']).rolling(window=window, center=True).mean()
        axes[4].plot(epochs, history_dict['loss'], alpha=0.3, color='blue', label='Raw Training Loss')
        axes[4].plot(epochs, loss_ma, linewidth=2, color='blue', label=f'Smoothed (MA-{window})')
        if 'val_loss' in history_dict:
            val_loss_ma = pd.Series(history_dict['val_loss']).rolling(window=window, center=True).mean()
            axes[4].plot(epochs, history_dict['val_loss'], alpha=0.3, color='red', label='Raw Validation Loss')
            axes[4].plot(epochs, val_loss_ma, linewidth=2, color='red', label=f'Smoothed Val (MA-{window})')
        axes[4].set_title('Training Stability', fontsize=14, fontweight='bold')
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('Loss')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

    # Final loss comparison
    final_losses = []
    for task in task_config.get_task_names():
        task_loss_key = f'{task}_loss'
        if task_loss_key in history_dict:
            final_losses.append((task.title(), history_dict[task_loss_key][-1]))
    if final_losses:
        tasks, losses = zip(*final_losses)
        colors = [task_colors.get(task.lower(), 'gray') for task in tasks]
        bars = axes[5].bar(tasks, losses, color=colors)
        axes[5].set_title('Final Task Losses', fontsize=14, fontweight='bold')
        axes[5].set_ylabel('Loss Value')
        axes[5].tick_params(axis='x', rotation=45)
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[5].text(bar.get_x() + bar.get_width()/2., height,
                        f'{loss:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_training_summary(
    results_dir: str, config: Dict,
    history: keras.callbacks.History, task_config: TaskConfiguration
):
    """Create detailed training summary."""
    with open(os.path.join(results_dir, 'training_summary.txt'), 'w') as f:
        f.write("YOLOv12 Multi-Task Training Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Task Configuration:\n")
        f.write(f"  Enabled Tasks: {', '.join(task_config.get_task_names())}\n")
        f.write(f"  Task Type: {'Multi-task' if task_config.is_multi_task() else 'Single-task'}\n")
        f.write(f"  Model Architecture: Named Outputs (Functional API)\n\n")

        f.write("Model Details:\n")
        f.write(f"  Scale: {config['model_config']['scale']}\n")
        f.write(f"  Patch Size: {config['model_config']['patch_size']}x{config['model_config']['patch_size']}\n")
        f.write(f"  Total Parameters: {config['model_config']['total_parameters']:,}\n")

        f.write("Training Configuration:\n")
        f.write(f"  Batch Size: {config['training_args']['batch_size']}\n")
        f.write(f"  Learning Rate: {config['training_args']['learning_rate']}\n")
        f.write(f"  Optimizer: {config['training_args']['optimizer']}\n")
        f.write(f"  Uncertainty Weighting: {config['loss_configuration']['uses_uncertainty_weighting']}\n\n")

        if 'visualization_config' in config:
            f.write("Visualization Configuration:\n")
            f.write(f"  Per-Epoch Visualizations: {config['visualization_config']['enabled']}\n")
            f.write(f"  Visualization Frequency: Every {config['visualization_config']['frequency']} epoch(s)\n")
            f.write(f"  Visualizations Path: {config['visualization_config']['per_epoch_visualizations_path']}\n\n")

        f.write("Training Results:\n")
        f.write(f"  Epochs Completed: {config['training_results']['epochs_completed']}\n")
        f.write(f"  Final Training Loss: {config['training_results']['final_training_loss']:.6f}\n")
        if config['training_results']['final_validation_loss']:
            f.write(f"  Final Validation Loss: {config['training_results']['final_validation_loss']:.6f}\n")
            f.write(f"  Best Validation Loss: {config['training_results']['best_validation_loss']:.6f}\n")

        if 'test_results' in config:
            f.write(f"\nTest Results:\n")
            for metric, value in config['test_results'].items():
                f.write(f"  {metric.replace('_', ' ').title()}: {value:.6f}\n")

        if 'task_weight_evolution' in config and config['task_weight_evolution']:
            f.write(f"\nTask Weight Evolution (Final):\n")
            final_weights = config['task_weight_evolution'][-1]
            for task, weight in final_weights.items():
                f.write(f"  {task.title()}: {weight:.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv12 Multi-Task Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--tasks', nargs='+',
                       choices=[task.value for task in TaskType.all_tasks()],
                       default=['detection', 'segmentation', 'classification'],
                       help='Tasks to enable for training')

    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--patch-size', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--patience', type=int, default=20)

    parser.add_argument('--uncertainty-weighting', action='store_true')
    parser.add_argument('--detection-weight', type=float, default=1.0)
    parser.add_argument('--segmentation-weight', type=float, default=1.0)
    parser.add_argument('--classification-weight', type=float, default=1.0)

    parser.add_argument('--enable-visualizations', action='store_true', default=True)
    parser.add_argument('--disable-visualizations', action='store_true')
    parser.add_argument('--visualization-freq', type=int, default=1)

    parser.add_argument('--no-evaluate', action='store_true')
    parser.add_argument('--run-eagerly', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)

    args = parser.parse_args()
    args.evaluate = not args.no_evaluate
    if args.disable_visualizations:
        args.enable_visualizations = False

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")
    if not args.tasks:
        raise ValueError("At least one task must be specified")

    logger.info(f"YOLOv12 Multi-Task Training - Tasks: {args.tasks}, Scale: {args.scale}")

    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    try:
        train_model(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
