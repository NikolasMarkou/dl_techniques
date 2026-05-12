"""
Vision Transformer Classification Training (resnet Pattern-4 style)

Training pipeline for ViT on CIFAR-10/CIFAR-100/ImageNet. Mirrors the
structure of ``train_resnet.py`` without the deep-supervision path
(DS is deferred to a follow-up plan, see plan_2026-05-12_f2d29729).

Supports variants: vit_pico / vit_tiny / vit_small / vit_base / vit_large /
vit_huge.

Usage:
    python -m train.vit.train_vit --dataset cifar10 --variant vit_pico \\
        --epochs 50 --batch-size 128 --learning-rate 3e-4 \\
        --optimizer adamw --weight-decay 0.05 --warmup-epochs 5 \\
        --output-dir results/vit_cifar10 --experiment-name iter1_baseline --gpu 0
"""

import os
import gc
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vit import ViT, create_vit
from dl_techniques.visualization import (
    VisualizationManager,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for ViT classification training.

    Mirrors :class:`train.resnet.train_resnet.TrainingConfig` with ViT-specific
    additions (patch_size, dropout_rate, attention_dropout_rate). Deep-supervision
    fields are intentionally absent -- DS for ViT is deferred to a follow-up plan.
    """

    # Data
    dataset: str = "cifar10"  # cifar10 | cifar100 | imagenet
    train_data_dir: Optional[str] = None  # only used for imagenet
    val_data_dir: Optional[str] = None
    image_size: int = 32  # 32 for CIFAR, 224 for ImageNet
    batch_size: int = 128

    # Model
    model_variant: str = "vit_pico"
    num_classes: int = 10
    pretrained: Union[bool, str] = False
    patch_size: int = 4  # 4 for CIFAR/32x32, 16 for ImageNet/224x224
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # used only when optimizer_type='sgd'

    # Augmentation
    augment_data: bool = True
    label_smoothing: float = 0.1

    # Monitoring
    monitor_every_n_epochs: int = 5
    save_best_only: bool = True
    early_stopping_patience: int = 15
    validation_steps: Optional[int] = None

    # Data pipeline
    cache_dataset: bool = False
    prefetch_buffer: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    save_training_samples: bool = True
    save_model_checkpoints: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"vit_{self.dataset}_{self.model_variant}_{timestamp}"

        if self.image_size <= 0:
            raise ValueError("Invalid image_size: must be positive")
        if self.num_classes <= 0:
            raise ValueError("Invalid num_classes: must be positive")
        if self.dataset == "imagenet":
            if not self.train_data_dir or not Path(self.train_data_dir).exists():
                raise ValueError(
                    f"--train-data-dir is required for --dataset imagenet "
                    f"(got: {self.train_data_dir!r})"
                )
            if not self.val_data_dir or not Path(self.val_data_dir).exists():
                raise ValueError(
                    f"--val-data-dir is required for --dataset imagenet "
                    f"(got: {self.val_data_dir!r})"
                )


# =============================================================================
# CIFAR DATA PIPELINE
# =============================================================================

def get_cifar_preprocessing() -> Tuple[List[float], List[float]]:
    """Return CIFAR mean/std for normalization."""
    return [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]


def _cifar_augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Standard CIFAR augmentation: random flip + pad-and-crop + small color jitter."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(
        tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"),
        size=tf.shape(image),
    )
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return tf.clip_by_value(image, 0.0, 1.0), label


def create_cifar_dataset(
        config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """Build CIFAR-10 or CIFAR-100 in-memory datasets.

    Returns:
        train_ds, val_ds, steps_per_epoch, val_steps
    """
    ds_name = config.dataset.lower()
    if ds_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    elif ds_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    else:
        raise ValueError(f"create_cifar_dataset called with dataset={ds_name!r}")

    y_train = y_train.flatten()
    y_test = y_test.flatten()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Normalize
    mean, std = get_cifar_preprocessing()
    x_train = (x_train - np.array(mean)) / np.array(std)
    x_test = (x_test - np.array(mean)) / np.array(std)

    logger.info(f"{ds_name.upper()}: {x_train.shape[0]} train, {x_test.shape[0]} test")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000)
    if config.augment_data:
        train_ds = train_ds.map(_cifar_augment, num_parallel_calls=config.num_parallel_calls)
    train_ds = (
        train_ds.repeat()
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(config.prefetch_buffer)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(config.batch_size)
        .prefetch(config.prefetch_buffer)
    )

    steps_per_epoch = len(x_train) // config.batch_size
    val_steps = max(1, len(x_test) // config.batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


# =============================================================================
# IMAGENET DATA PIPELINE
# =============================================================================

def get_imagenet_preprocessing() -> Tuple[List[float], List[float]]:
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _preprocess_imagenet(
        image: tf.Tensor, label: tf.Tensor, config: TrainingConfig, is_training: bool
) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    if is_training and config.augment_data:
        image = tf.image.resize(image, [config.image_size + 32, config.image_size + 32])
        image = tf.image.random_crop(image, [config.image_size, config.image_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize(image, [int(config.image_size * 1.15), int(config.image_size * 1.15)])
        image = tf.image.resize_with_crop_or_pad(image, config.image_size, config.image_size)
    image = tf.clip_by_value(image, 0.0, 255.0) / 255.0
    mean, std = get_imagenet_preprocessing()
    image = (image - mean) / std
    return image, label


def create_imagenet_dataset(
        data_dir: str, config: TrainingConfig, is_training: bool = True
) -> tf.data.Dataset:
    """Build an ImageNet-style tf.data pipeline from a class-subdir layout."""
    data_dir = Path(data_dir)
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"Found {len(class_names)} classes in {data_dir}")

    image_paths: List[str] = []
    labels: List[int] = []
    for class_name in class_names:
        class_idx = class_to_idx[class_name]
        for img_file in (data_dir / class_name).glob("*.JPEG"):
            image_paths.append(str(img_file))
            labels.append(class_idx)
    logger.info(f"Found {len(image_paths)} images")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True).repeat()

    def _load(path, label):
        image = tf.io.read_file(path)
        return _preprocess_imagenet(image, label, config, is_training)

    dataset = dataset.map(_load, num_parallel_calls=config.num_parallel_calls)
    if config.cache_dataset and not is_training:
        dataset = dataset.cache()
    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(config.prefetch_buffer)
    return dataset


# =============================================================================
# CALLBACKS
# =============================================================================

class MetricsVisualizationCallback(keras.callbacks.Callback):
    """Per-epoch metrics plotting (loss, accuracy, top-5, lr). Mirrors resnet trainer."""

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.config = config
        self.visualization_dir = (
            Path(config.output_dir) / config.experiment_name / "training_metrics"
        )
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics: Dict[str, List[float]] = {
            "loss": [], "accuracy": [], "top5_accuracy": [],
        }
        self.val_metrics: Dict[str, List[float]] = {
            "val_loss": [], "val_accuracy": [], "val_top5_accuracy": [],
        }

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            try:
                converted = float(metric_value)
                if metric_name in self.train_metrics:
                    self.train_metrics[metric_name].append(converted)
                elif metric_name in self.val_metrics:
                    self.val_metrics[metric_name].append(converted)
            except (ValueError, TypeError):
                pass

        if (epoch + 1) % self.config.monitor_every_n_epochs == 0 or epoch == 0:
            self._create_plots(epoch + 1)

    def _create_plots(self, epoch: int) -> None:
        try:
            if not self.train_metrics.get("loss"):
                return
            num_epochs = len(self.train_metrics["loss"])
            epochs_range = range(1, num_epochs + 1)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"ViT Training Metrics - Epoch {epoch}", fontsize=16)

            axes[0, 0].plot(epochs_range, self.train_metrics["loss"], "b-", label="Train", linewidth=2)
            if self.val_metrics.get("val_loss"):
                axes[0, 0].plot(epochs_range, self.val_metrics["val_loss"], "r-", label="Val", linewidth=2)
            axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

            if self.train_metrics.get("accuracy"):
                axes[0, 1].plot(epochs_range, self.train_metrics["accuracy"], "b-", label="Train", linewidth=2)
            if self.val_metrics.get("val_accuracy"):
                axes[0, 1].plot(epochs_range, self.val_metrics["val_accuracy"], "r-", label="Val", linewidth=2)
            axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

            if self.train_metrics.get("top5_accuracy"):
                axes[1, 0].plot(epochs_range, self.train_metrics["top5_accuracy"], "b-", label="Train", linewidth=2)
            if self.val_metrics.get("val_top5_accuracy"):
                axes[1, 0].plot(epochs_range, self.val_metrics["val_top5_accuracy"], "r-", label="Val", linewidth=2)
            axes[1, 0].set_title("Top-5 Accuracy"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].axis("off")
            plt.tight_layout()
            plt.savefig(self.visualization_dir / f"epoch_{epoch:03d}_metrics.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


def create_callbacks(config: TrainingConfig) -> Tuple[List[keras.callbacks.Callback], str]:
    """Common callbacks (early-stop, ckpt, CSV, analyzer) + ViT-specific metrics viz."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name or config.model_variant,
        results_dir_prefix="vit",
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    callbacks.append(MetricsVisualizationCallback(config))
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
            success="#06D6A0", warning="#FFD166",
        ),
        title_fontsize=14, label_fontsize=12,
        save_format="png", dpi=300, fig_size=(12, 8),
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


def convert_keras_history_to_training_history(
        keras_history: keras.callbacks.History,
) -> TrainingHistory:
    history_dict = keras_history.history
    epochs = list(range(len(history_dict["loss"])))
    train_metrics: Dict[str, List[float]] = {}
    val_metrics: Dict[str, List[float]] = {}
    for key, values in history_dict.items():
        if key.startswith("val_") and key != "val_loss":
            val_metrics[key.replace("val_", "")] = values
        elif not key.startswith("val_") and key != "loss":
            train_metrics[key] = values
    return TrainingHistory(
        epochs=epochs,
        train_loss=history_dict["loss"],
        val_loss=history_dict.get("val_loss", []),
        train_metrics=train_metrics, val_metrics=val_metrics,
    )


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_vit(config: TrainingConfig, gpu_id: Optional[int] = None) -> keras.Model:
    """Orchestrate the ViT classification training pipeline."""
    setup_gpu(gpu_id)

    logger.info(f"Experiment: {config.experiment_name}, Variant: {config.model_variant}")
    logger.info(f"Dataset: {config.dataset}, Image size: {config.image_size}, "
                f"Patch size: {config.patch_size}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # ---- Datasets ----
    if config.dataset in ("cifar10", "cifar100"):
        train_ds, val_ds, steps_per_epoch, val_steps = create_cifar_dataset(config)
        input_shape = (config.image_size, config.image_size, 3)
    elif config.dataset == "imagenet":
        train_ds = create_imagenet_dataset(config.train_data_dir, config, is_training=True)
        val_ds = create_imagenet_dataset(config.val_data_dir, config, is_training=False)
        train_dir = Path(config.train_data_dir)
        num_train_images = sum(
            len(list((train_dir / cd).glob("*.JPEG")))
            for cd in train_dir.iterdir() if cd.is_dir()
        )
        steps_per_epoch = num_train_images // config.batch_size
        val_dir = Path(config.val_data_dir)
        num_val_images = sum(
            len(list((val_dir / cd).glob("*.JPEG")))
            for cd in val_dir.iterdir() if cd.is_dir()
        )
        val_steps = max(1, num_val_images // config.batch_size)
        input_shape = (config.image_size, config.image_size, 3)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    logger.info(f"Steps per epoch: {steps_per_epoch}, Val steps: {val_steps}")

    # ---- Model ----
    # LESSONS L72 (Double Weight Decay guard): when optimizer is AdamW, pass
    # weight_decay to optimizer_builder ONLY and leave kernel_regularizer=None.
    # When optimizer is SGD, use kernel_regularizer=L2(weight_decay) instead.
    use_adamw = config.optimizer_type.lower() == "adamw"
    kernel_reg = (
        None
        if use_adamw
        else (keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None)
    )

    model = create_vit(
        variant=config.model_variant,
        num_classes=config.num_classes,
        input_shape=input_shape,
        patch_size=config.patch_size,
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        kernel_regularizer=kernel_reg,
    )
    # Probe build so summary + count_params work.
    model.build((None,) + input_shape)
    model.summary()

    # ---- Optimization ----
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })

    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if use_adamw:
        opt_config["weight_decay"] = config.weight_decay
    elif config.optimizer_type.lower() == "sgd":
        opt_config["momentum"] = config.momentum

    optimizer = optimizer_builder(opt_config, lr_schedule)

    # ---- Loss + metrics ----
    # Keras' SparseCategoricalCrossentropy doesn't support label_smoothing
    # (that's a CategoricalCrossentropy feature). Keep config.label_smoothing
    # for potential future one-hot path -- currently informational only.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if config.label_smoothing > 0.0:
        logger.warning(
            f"label_smoothing={config.label_smoothing} requested but "
            "SparseCategoricalCrossentropy doesn't support it; ignoring."
        )

    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    # ---- Train ----
    callbacks, _ = create_callbacks(config)

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_h = (time.time() - start_time) / 3600.0
    logger.info(f"Training completed in {elapsed_h:.2f} hours")

    # ---- Save history ----
    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(output_dir / "training_history.json", "w") as f:
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
        description="Train ViT on CIFAR-10 / CIFAR-100 / ImageNet (resnet Pattern-4 style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--train-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Required only when --dataset imagenet")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Auto: 32 for CIFAR, 224 for ImageNet")

    # Model
    parser.add_argument("--variant", type=str, default="vit_pico",
                        choices=["vit_pico", "vit_tiny", "vit_small",
                                 "vit_base", "vit_large", "vit_huge"])
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Auto: 10 for cifar10, 100 for cifar100, 1000 for imagenet")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Auto: 4 for CIFAR, 16 for ImageNet")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to local .keras weights file (boolean True is not supported)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Augmentation
    parser.add_argument("--no-augmentation", dest="augment_data", action="store_false")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    # Auto-derive image_size / num_classes / patch_size from dataset.
    ds = args.dataset.lower()
    image_size = args.image_size if args.image_size is not None else (
        32 if ds in ("cifar10", "cifar100") else 224
    )
    num_classes = args.num_classes if args.num_classes is not None else (
        10 if ds == "cifar10" else (100 if ds == "cifar100" else 1000)
    )
    patch_size = args.patch_size if args.patch_size is not None else (
        4 if ds in ("cifar10", "cifar100") else 16
    )

    pretrained_arg: Union[bool, str] = args.pretrained if args.pretrained else False

    config = TrainingConfig(
        dataset=ds,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        model_variant=args.variant,
        num_classes=num_classes,
        pretrained=pretrained_arg,
        patch_size=patch_size,
        dropout_rate=args.dropout,
        attention_dropout_rate=args.attention_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        augment_data=args.augment_data,
        label_smoothing=args.label_smoothing,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )

    logger.info(
        f"Config: variant={config.model_variant}, dataset={config.dataset}, "
        f"{config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}"
    )

    try:
        train_vit(config, gpu_id=args.gpu)
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
