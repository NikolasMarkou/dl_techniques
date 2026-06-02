"""
ResNet ImageNet Classification Training with Deep Supervision

Training pipeline for ResNet on ImageNet with optional deep supervision,
adaptive loss weight scheduling, and comprehensive monitoring.

Supports: ResNet-18/34/50/101/152, single and multi-output training.

Usage:
    python train_resnet.py --variant resnet50 --enable-deep-supervision --epochs 100
    python train_resnet.py --variant resnet18 --no-deep-supervision --epochs 90
"""

import os
import gc
import json
import time
import keras
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from train.common import setup_gpu, create_callbacks as create_common_callbacks, save_config_json, convert_keras_history_to_training_history, make_imagenet_filesystem_dataset, EpochMetricsPlotCallback
from dl_techniques.metrics.primary_output_metrics import (
    PrimaryOutputAccuracy, PrimaryOutputTopKAccuracy,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
    deep_supervision_schedule_builder
)
from dl_techniques.models.resnet.model import (
    create_resnet,
    get_model_output_info,
    create_inference_model_from_training_model
)
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    ClassificationResults,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for ResNet ImageNet training."""

    # Data
    train_data_dir: str = "/path/to/imagenet/train"
    val_data_dir: str = "/path/to/imagenet/val"
    image_size: int = 224
    batch_size: int = 256

    # Model
    model_variant: str = 'resnet50'
    num_classes: int = 1000
    pretrained: bool = False

    # Deep Supervision
    enable_deep_supervision: bool = False
    deep_supervision_schedule_type: str = 'step_wise'
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # Training
    epochs: int = 100
    learning_rate: float = 0.1
    optimizer_type: str = 'sgd'
    lr_schedule_type: str = 'cosine_decay'
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clipping: float = 1.0
    momentum: float = 0.9

    # Augmentation
    augment_data: bool = True
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = None

    # Data Pipeline
    cache_dataset: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Output
    output_dir: str = 'results'
    experiment_name: Optional[str] = None
    save_training_samples: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ds_suffix = '_ds' if self.enable_deep_supervision else ''
            self.experiment_name = f"resnet_{self.model_variant}{ds_suffix}_{timestamp}"

        if self.image_size <= 0:
            raise ValueError("Invalid image_size: must be positive")
        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")
        if not Path(self.train_data_dir).exists():
            raise ValueError(f"Training directory does not exist: {self.train_data_dir}")
        if not Path(self.val_data_dir).exists():
            raise ValueError(f"Validation directory does not exist: {self.val_data_dir}")


# =============================================================================
# IMAGENET DATA PIPELINE
# =============================================================================

# The ImageNet class-subdir tf.data pipeline (decode/resize/augment/normalize)
# now lives in train.common.make_imagenet_filesystem_dataset. ResNet opts into
# the 4 colour augmentations via augment_color=True; the IMAGENET_MEAN/STD
# normalization and the unconditional clip_by_value(0,255) are applied inside
# the helper. See plan_2026-06-02_35651564/D-001.


# =============================================================================
# CUSTOM METRICS
# =============================================================================


# PrimaryOutputAccuracy and PrimaryOutputTopKAccuracy imported from dl_techniques.metrics


# =============================================================================
# CALLBACKS
# =============================================================================

class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """Dynamic weight scheduler for deep supervision training."""

    def __init__(self, config: TrainingConfig, num_outputs: int) -> None:
        super().__init__()
        self.config = config
        self.num_outputs = num_outputs
        self.total_epochs = config.epochs
        ds_config = {
            'type': config.deep_supervision_schedule_type,
            'config': config.deep_supervision_schedule_config
        }
        self.scheduler = deep_supervision_schedule_builder(ds_config, self.num_outputs, invert_order=True)

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        new_weights = self.scheduler(progress)
        self.model.loss_weights = new_weights
        weights_str = ", ".join(f"{w:.4f}" for w in new_weights)
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} - DS weights: [{weights_str}]")


def create_callbacks(config: TrainingConfig, num_outputs: int) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks: common (checkpoint, early stop, CSV, analyzer) + domain-specific."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name or config.model_variant,
        results_dir_prefix="resnet",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )

    if config.enable_deep_supervision and num_outputs > 1:
        callbacks.append(DeepSupervisionWeightScheduler(config, num_outputs))
    viz_dir = Path(config.output_dir) / config.experiment_name / "training_metrics"
    callbacks.append(EpochMetricsPlotCallback(
        str(viz_dir),
        ["accuracy", "top5_accuracy", "primary_accuracy", "primary_top5_accuracy"],
        every_n=5,
    ))

    return callbacks, results_dir


# =============================================================================
# VISUALIZATION
# =============================================================================

def setup_visualization_manager(experiment_name: str, results_dir: str) -> VisualizationManager:
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        color_scheme=ColorScheme(
            primary="#2E86AB", secondary="#A23B72",
            success="#06D6A0", warning="#FFD166"
        ),
        title_fontsize=14, label_fontsize=12,
        save_format="png", dpi=300, fig_size=(12, 8)
    )

    viz_manager = VisualizationManager(
        experiment_name=experiment_name, output_dir=viz_dir, config=config
    )
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)
    return viz_manager


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_resnet_imagenet(config: TrainingConfig, gpu_id: Optional[int] = None) -> keras.Model:
    """Orchestrate the complete ResNet ImageNet training pipeline."""
    setup_gpu(gpu_id)

    logger.info(f"Experiment: {config.experiment_name}, Variant: {config.model_variant}")
    logger.info(f"Deep Supervision: {'ENABLED (' + config.deep_supervision_schedule_type + ')' if config.enable_deep_supervision else 'DISABLED'}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config_json(config, str(output_dir), "config.json")

    # Datasets
    logger.info("Creating ImageNet datasets...")
    train_dataset = make_imagenet_filesystem_dataset(
        config.train_data_dir,
        config.image_size,
        config.batch_size,
        is_training=True,
        augment=config.augment_data,
        augment_color=True,
        num_parallel_calls=config.num_parallel_calls,
        cache_val=config.cache_dataset,
        prefetch_buffer=config.prefetch_buffer,
    )
    val_dataset = make_imagenet_filesystem_dataset(
        config.val_data_dir,
        config.image_size,
        config.batch_size,
        is_training=False,
        augment=config.augment_data,
        augment_color=True,
        num_parallel_calls=config.num_parallel_calls,
        cache_val=config.cache_dataset,
        prefetch_buffer=config.prefetch_buffer,
    )

    train_dir = Path(config.train_data_dir)
    num_train_images = sum(
        len(list((train_dir / cd).glob("*.JPEG")))
        for cd in train_dir.iterdir() if cd.is_dir()
    )
    steps_per_epoch = num_train_images // config.batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # Model
    input_shape = (config.image_size, config.image_size, 3)
    model = create_resnet(
        variant=config.model_variant, num_classes=config.num_classes,
        input_shape=input_shape, pretrained=config.pretrained,
        enable_deep_supervision=config.enable_deep_supervision,
        kernel_regularizer=keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None
    )
    model.summary()

    model_info = get_model_output_info(model)
    has_multiple_outputs = model_info['has_deep_supervision']
    num_outputs = model_info['num_outputs']
    logger.info(f"Model: {num_outputs} output(s)")

    # Multi-output dataset adaptation
    if has_multiple_outputs:
        def create_multiscale_labels(image, label):
            return image, tuple([label for _ in range(num_outputs)])
        train_dataset = train_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(create_multiscale_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimization
    lr_schedule = learning_rate_schedule_builder({
        'type': config.lr_schedule_type, 'learning_rate': config.learning_rate,
        'decay_steps': steps_per_epoch * config.epochs,
        'warmup_steps': steps_per_epoch * config.warmup_epochs, 'alpha': 0.01
    })

    opt_config = {'type': config.optimizer_type, 'gradient_clipping_by_norm': config.gradient_clipping}
    if config.optimizer_type.lower() == 'sgd':
        opt_config['momentum'] = config.momentum

    optimizer = optimizer_builder(opt_config, lr_schedule)

    # Loss and metrics
    if has_multiple_outputs:
        loss_fns = [keras.losses.SparseCategoricalCrossentropy(from_logits=True)] * num_outputs
        initial_weights = [1.0 / num_outputs] * num_outputs
        primary_output_name = model.output[0].name.split('/')[0]
        metrics = {primary_output_name: [PrimaryOutputAccuracy(), PrimaryOutputTopKAccuracy(k=5, name='primary_top5_accuracy')]}
    else:
        loss_fns = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        initial_weights = None
        metrics = ['accuracy', 'top_5_accuracy']

    model.compile(optimizer=optimizer, loss=loss_fns, loss_weights=initial_weights, metrics=metrics)

    # Train
    callbacks, _ = create_callbacks(config, num_outputs)

    val_steps = config.validation_steps
    if val_steps is None:
        val_dir = Path(config.val_data_dir)
        num_val_images = sum(
            len(list((val_dir / cd).glob("*.JPEG")))
            for cd in val_dir.iterdir() if cd.is_dir()
        )
        val_steps = num_val_images // config.batch_size

    start_time = time.time()
    history = model.fit(
        train_dataset, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset, validation_steps=val_steps,
        callbacks=callbacks, verbose=1
    )
    logger.info(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")

    # Save inference model
    if config.enable_deep_supervision and has_multiple_outputs:
        inference_model = create_inference_model_from_training_model(model)
        inference_model.save(output_dir / "inference_model.keras")
        logger.info("Inference model saved")

    # Save history
    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    gc.collect()
    return model


# =============================================================================
# CLI
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train ResNet on ImageNet with Deep Supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--train-data-dir', type=str, required=True)
    parser.add_argument('--val-data-dir', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=224)

    parser.add_argument('--variant', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--pretrained', action='store_true', default=False)

    parser.add_argument('--enable-deep-supervision', action='store_true', default=False)
    parser.add_argument('--no-deep-supervision', dest='enable_deep_supervision', action='store_false')
    parser.add_argument('--deep-supervision-schedule', type=str, default='step_wise',
                        choices=[
                            'constant_equal', 'constant_low_to_high', 'constant_high_to_low',
                            'linear_low_to_high', 'non_linear_low_to_high', 'custom_sigmoid_low_to_high',
                            'scale_by_scale_low_to_high', 'cosine_annealing', 'curriculum', 'step_wise'
                        ])

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr-schedule', type=str, default='cosine_decay',
                        choices=['cosine_decay', 'exponential_decay', 'constant'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--no-augmentation', dest='augment_data', action='store_false')
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--monitor-every', type=int, default=5)
    parser.add_argument('--early-stopping-patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index')

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    config = TrainingConfig(
        train_data_dir=args.train_data_dir, val_data_dir=args.val_data_dir,
        image_size=args.image_size, batch_size=args.batch_size,
        model_variant=args.variant, num_classes=args.num_classes,
        pretrained=args.pretrained,
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.deep_supervision_schedule,
        epochs=args.epochs, learning_rate=args.learning_rate,
        optimizer_type=args.optimizer, lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        augment_data=args.augment_data, label_smoothing=args.label_smoothing,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir, experiment_name=args.experiment_name
    )

    logger.info(f"Config: {config.model_variant}, DS={'on' if config.enable_deep_supervision else 'off'}, "
                f"{config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}")

    try:
        model = train_resnet_imagenet(config, gpu_id=args.gpu)
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
