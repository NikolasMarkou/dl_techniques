"""Depth Anything monocular depth estimation training script.

Pattern-5 trainer for ``dl_techniques.models.depth_anything.DepthAnything`` on
MegaDepth.  Mirrors ``train.cliffordnet.train_depth_estimation`` 1:1 in
structure, swapping in ``create_depth_anything`` as the model factory and
using a locally-defined ``DepthEstimationLoss`` (masked L1 + multi-scale
gradient matching) that is compatible with the model's ``sigmoid``-clamped
output.

> **Important — read** ``src/dl_techniques/models/depth_anything/README.md``
> **before running this script.** The DepthAnything implementation has known
> gaps (placeholder encoder, unimplemented semi-supervised pipeline, sigmoid
> output that conflicts with AffineInvariantLoss).  This script is a thin
> Pattern-5 scaffold; the model itself still needs work for production-grade
> depth quality.  See "Known Issues" in the model README for the full list.

Usage::

    # Smoke run (does not actually train to convergence — model is a placeholder):
    MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \\
        --encoder-type vit_s \\
        --epochs 2 --batch-size 4 --patch-size 256 \\
        --max-train-files 100 --max-val-files 20 \\
        --gpu 0

    # With pretrained encoder transfer:
    MPLBACKEND=Agg .venv/bin/python -m train.depth_anything.train_depth_anything \\
        --encoder-type vit_l --epochs 100 --batch-size 16 --patch-size 384 \\
        --init-from results/depth_anything_pretrain_*/model_inference.keras \\
        --seed 42 --gpu 0
"""

import gc
import json
import random
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
)
from train.common.megadepth import (
    discover_megadepth_pairs,
    load_and_process_pair as _load_and_process_pair,
    MegaDepthDataset,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.depth_anything.model import (
    DepthAnything,
    create_depth_anything,
)
from dl_techniques.metrics.depth_metrics import (
    AbsRelMetric,
    DeltaThresholdMetric,
    RMSELogMetric,
    RMSEMetric,
    SqRelMetric,
)
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.callbacks.depth_visualization import (
    DepthPredictionGridCallback,
    DepthMetricsCurveCallback,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class DepthAnythingTrainingConfig:
    """Configuration for DepthAnything depth estimation training on MegaDepth."""

    # Data
    megadepth_root: str = "/media/arxwn/data0_4tb/datasets/Megadepth"
    train_split: float = 0.9
    patch_size: int = 384
    min_valid_ratio: float = 0.1

    # Memory
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 1000
    dataset_shuffle_buffer: int = 5000

    # Model — DepthAnything-specific
    encoder_type: str = "vit_l"            # one of {'vit_s','vit_b','vit_l'}
    encoder_kind: str = "real"             # 'real' (in-tree ViT) or 'placeholder' (Conv-BN-ReLU)
    decoder_dims: Tuple[int, ...] = (256, 128, 64, 32)
    output_channels: int = 1
    use_feature_alignment: bool = False    # see Known Issue #2 in model README
    enable_semi_supervised: bool = False   # adds FAL term on unlabeled batches inside train_step
    cutmix_prob: float = 0.5
    color_jitter_strength: float = 0.2

    # Training
    batch_size: int = 16
    epochs: int = 100
    augment_data: bool = True
    steps_per_epoch: Optional[int] = None

    # Optimization
    learning_rate: float = 5e-6
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Monitoring
    monitor_every_n_epochs: int = 5
    early_stopping_patience: int = 20
    validation_steps: Optional[int] = 200

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Pretrained init (e.g. backbone from a prior depth_anything run, or any compatible .keras checkpoint)
    init_from: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"depth_anything_{self.encoder_type}_{timestamp}"
            )
        if not 0.0 < self.min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be in (0, 1]")


# ---------------------------------------------------------------------
# DEPTH ESTIMATION LOSS (masked L1 + multi-scale gradient matching)
# ---------------------------------------------------------------------


class DepthEstimationLoss(keras.losses.Loss):
    """Masked L1 + multi-scale gradient matching loss for depth estimation.

    ``L = L1_masked + gradient_weight * L_grad``

    Identical to ``train.cliffordnet.train_depth_estimation.DepthEstimationLoss``.
    Compatible with any output range (so it works with DepthAnything's default
    ``sigmoid``-clamped output — see Known Issue #6 in the model README).
    """

    def __init__(
        self,
        gradient_weight: float = 0.5,
        n_scales: int = 4,
        name: str = "depth_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gradient_weight = gradient_weight
        self.n_scales = n_scales

    @staticmethod
    def _gradient_xy(d):
        dx = d[:, :, 1:, :] - d[:, :, :-1, :]
        dy = d[:, 1:, :, :] - d[:, :-1, :, :]
        return dx, dy

    @staticmethod
    def _downsample_2x(d):
        return keras.ops.average_pool(d, pool_size=2, strides=2, padding="valid")

    def call(self, y_true_and_mask, y_pred):
        depth_true = y_true_and_mask[..., :1]
        mask = y_true_and_mask[..., 1:]

        l1_error = keras.ops.abs(y_pred - depth_true) * mask
        valid_count = keras.ops.maximum(
            keras.ops.sum(mask), keras.ops.cast(1.0, mask.dtype)
        )
        l_l1 = keras.ops.sum(l1_error) / valid_count

        l_grad = keras.ops.cast(0.0, y_pred.dtype)
        d_pred = y_pred
        d_true = depth_true
        m = mask

        for _ in range(self.n_scales):
            dx_pred, dy_pred = self._gradient_xy(d_pred)
            dx_true, dy_true = self._gradient_xy(d_true)

            mx = keras.ops.minimum(m[:, :, 1:, :], m[:, :, :-1, :])
            my = keras.ops.minimum(m[:, 1:, :, :], m[:, :-1, :, :])

            mx_count = keras.ops.maximum(
                keras.ops.sum(mx), keras.ops.cast(1.0, mx.dtype)
            )
            my_count = keras.ops.maximum(
                keras.ops.sum(my), keras.ops.cast(1.0, my.dtype)
            )

            l_grad = l_grad + (
                keras.ops.sum(keras.ops.abs(dx_pred - dx_true) * mx) / mx_count
                + keras.ops.sum(keras.ops.abs(dy_pred - dy_true) * my) / my_count
            )

            d_pred = self._downsample_2x(d_pred)
            d_true = self._downsample_2x(d_true)
            m = self._downsample_2x(m)
            m = keras.ops.cast(m > 0.5, m.dtype)

        return l_l1 + self.gradient_weight * l_grad

    def get_config(self):
        config = super().get_config()
        config.update({
            "gradient_weight": self.gradient_weight,
            "n_scales": self.n_scales,
        })
        return config


# ---------------------------------------------------------------------
# DEPTH-ONLY METRIC WRAPPERS
# ---------------------------------------------------------------------
#
# MegaDepthDataset yields ``y_true = concat([depth, mask], axis=-1)`` (last-axis
# size 2) but ``dl_techniques.metrics.depth_metrics`` expect ``y_true`` to be a
# single-channel depth tensor.  Wrap each metric so it slices off the mask
# channel before calling the underlying ``update_state``.


class _DepthOnlyMetricWrapper(keras.metrics.Metric):
    """Slices ``y_true[..., :1]`` and forwards to a wrapped depth metric."""

    def __init__(self, inner: keras.metrics.Metric, name: Optional[str] = None, **kwargs):
        super().__init__(name=name or inner.name, **kwargs)
        self._inner = inner

    def update_state(self, y_true, y_pred, sample_weight=None):
        depth_true = y_true[..., :1]
        mask = y_true[..., 1:]
        # Mask invalid pixels into a sample_weight tensor for the inner metric.
        if sample_weight is None:
            sample_weight = mask
        else:
            sample_weight = sample_weight * mask
        return self._inner.update_state(depth_true, y_pred, sample_weight)

    def result(self):
        return self._inner.result()

    def reset_state(self):
        self._inner.reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({"inner": keras.metrics.serialize(self._inner)})
        return config


# ---------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------


def _load_validation_samples(
    val_rgb_paths: List[str],
    val_depth_paths: List[str],
    patch_size: int,
    min_valid_ratio: float,
    max_samples: int = 8,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a fixed set of validation samples for visualization."""
    rgb_list, depth_list, mask_list = [], [], []
    for rp, dp in zip(val_rgb_paths[:max_samples], val_depth_paths[:max_samples]):
        result = _load_and_process_pair(
            rp, dp, patch_size, min_valid_ratio, augment=False,
        )
        if result is None:
            continue
        rgb, y_true = result
        rgb_list.append(rgb)
        depth_list.append(y_true[..., :1])
        mask_list.append(y_true[..., 1:])

    if not rgb_list:
        return None
    return np.stack(rgb_list), np.stack(depth_list), np.stack(mask_list)


def create_callbacks(
    config: DepthAnythingTrainingConfig,
    val_rgb_paths: List[str],
    val_depth_paths: List[str],
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks: common utilities + depth-specific."""
    common_callbacks, results_dir = create_common_callbacks(
        model_name=config.encoder_type,
        results_dir_prefix="depth_anything",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    config.output_dir = str(Path(results_dir).parent)
    config.experiment_name = Path(results_dir).name

    viz_dir = str(Path(results_dir) / "visualization_plots")

    common_callbacks.append(DepthMetricsCurveCallback(
        output_dir=viz_dir,
        train_metrics=["loss", "abs_rel", "delta_1.25"],
        frequency=5,
    ))

    val_data = _load_validation_samples(
        val_rgb_paths, val_depth_paths,
        config.patch_size, config.min_valid_ratio,
    )
    if val_data is not None:
        val_rgb, val_depth, val_mask = val_data
        common_callbacks.append(DepthPredictionGridCallback(
            val_rgb=tf.constant(val_rgb),
            val_depth=tf.constant(val_depth),
            val_mask=tf.constant(val_mask),
            output_dir=viz_dir,
            frequency=config.monitor_every_n_epochs,
            title="Depth Anything Estimation",
        ))
        logger.info(
            f"Depth monitor: loaded {val_rgb.shape[0]} validation "
            f"patches, shape {val_rgb.shape}"
        )
    else:
        logger.warning("Depth monitor: failed to load validation samples")

    return common_callbacks, results_dir


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------


def create_model(config: DepthAnythingTrainingConfig) -> DepthAnything:
    """Create a DepthAnything model for monocular depth estimation."""
    return create_depth_anything(
        encoder_type=config.encoder_type,
        image_shape=(config.patch_size, config.patch_size, 3),
        decoder_dims=list(config.decoder_dims),
        output_channels=config.output_channels,
        use_feature_alignment=config.use_feature_alignment,
        cutmix_prob=config.cutmix_prob,
        color_jitter_strength=config.color_jitter_strength,
        encoder_kind=config.encoder_kind,
        enable_semi_supervised=config.enable_semi_supervised,
    )


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy, TensorFlow, and Keras for reproducible init."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    logger.info(f"Seeded random (python, numpy, tf, keras) = {seed}")


def train_depth_anything(config: DepthAnythingTrainingConfig) -> DepthAnything:
    """Train DepthAnything for monocular depth estimation on MegaDepth."""
    _set_seeds(config.seed)
    logger.info(
        f"Starting DepthAnything depth estimation training: "
        f"{config.experiment_name}"
    )

    rgb_paths, depth_paths = discover_megadepth_pairs(
        config.megadepth_root,
        max_files=(config.max_train_files or 0) + (config.max_val_files or 0)
        if config.max_train_files and config.max_val_files
        else None,
    )
    if len(rgb_paths) < 10:
        raise ValueError(
            f"Only found {len(rgb_paths)} MegaDepth pairs — "
            f"need at least 10. Check megadepth_root: {config.megadepth_root}"
        )

    n = len(rgb_paths)
    split_idx = int(n * config.train_split)
    train_rgb = rgb_paths[:split_idx]
    train_depth = depth_paths[:split_idx]
    val_rgb = rgb_paths[split_idx:]
    val_depth = depth_paths[split_idx:]

    if config.max_train_files and len(train_rgb) > config.max_train_files:
        train_rgb = train_rgb[:config.max_train_files]
        train_depth = train_depth[:config.max_train_files]
    if config.max_val_files and len(val_rgb) > config.max_val_files:
        val_rgb = val_rgb[:config.max_val_files]
        val_depth = val_depth[:config.max_val_files]

    logger.info(
        f"Dataset: {len(train_rgb)} training, "
        f"{len(val_rgb)} validation pairs"
    )

    import os
    num_workers = max(1, min(os.cpu_count() or 4, 10) - 2)
    logger.info(f"Using {num_workers} data-loading worker processes")

    train_ds = MegaDepthDataset(
        train_rgb, train_depth,
        batch_size=config.batch_size,
        patch_size=config.patch_size,
        min_valid_ratio=config.min_valid_ratio,
        augment=config.augment_data,
        is_training=True, workers=num_workers,
        use_resize=False,
    )
    val_ds = MegaDepthDataset(
        val_rgb, val_depth,
        batch_size=config.batch_size,
        patch_size=config.patch_size,
        min_valid_ratio=config.min_valid_ratio,
        augment=False,
        is_training=False, workers=max(1, num_workers // 2),
        use_resize=False,
    )

    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = len(train_ds)
    logger.info(f"Using {steps_per_epoch} steps per epoch")

    model = create_model(config)
    model.summary(print_fn=logger.info)

    if config.init_from is not None:
        logger.info(f"Initializing from pretrained checkpoint: {config.init_from}")
        report = load_weights_from_checkpoint(
            target=model,
            ckpt_path=config.init_from,
            skip_prefixes=("dpt_decoder",),
        )
        if report.num_loaded == 0:
            raise RuntimeError(
                f"Pretrained init loaded 0 layers from {config.init_from} — "
                f"likely a checkpoint / architecture mismatch."
            )

    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
            "weight_decay": config.weight_decay,
        },
        lr_schedule,
    )

    metrics = [
        _DepthOnlyMetricWrapper(AbsRelMetric(), name="abs_rel"),
        _DepthOnlyMetricWrapper(SqRelMetric(),  name="sq_rel"),
        _DepthOnlyMetricWrapper(RMSEMetric(),   name="rmse"),
        _DepthOnlyMetricWrapper(RMSELogMetric(), name="rmse_log"),
        _DepthOnlyMetricWrapper(
            DeltaThresholdMetric(threshold=1.25),
            name="delta_1.25",
        ),
    ]

    model.compile(
        optimizer=optimizer,
        loss=DepthEstimationLoss(gradient_weight=0.5, n_scales=4),
        metrics=metrics,
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    callbacks, results_dir = create_callbacks(config, val_rgb, val_depth)
    output_dir = Path(results_dir)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    start_time = time.time()

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

    try:
        history_dict = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    try:
        trained_epochs = len(history.history.get("loss", []))
        best_loss = min(history.history.get("val_loss", [float("nan")]))
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Depth Anything — Monocular Depth Estimation Training Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Encoder type      : {config.encoder_type}\n")
            f.write(f"Parameters        : {model.count_params():,}\n")
            f.write(f"Epochs trained    : {trained_epochs}\n")
            f.write(f"Best val_loss     : {best_loss:.6f}\n")
            f.write(f"Batch size        : {config.batch_size}\n")
            f.write(f"Patch size        : {config.patch_size}\n")
            f.write(f"Learning rate     : {config.learning_rate}\n")
            f.write(f"Train pairs       : {len(train_rgb)}\n")
            f.write(f"Val pairs         : {len(val_rgb)}\n")
            f.write(f"Duration          : {elapsed:.1f}s\n")
        logger.info(f"Summary written to: {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write training summary: {e}")

    try:
        inference_path = output_dir / "model_inference.keras"
        model.save(str(inference_path))
        logger.info(f"Inference model saved to: {inference_path}")
    except Exception as e:
        logger.warning(f"Failed to save inference model: {e}")

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Depth Anything for monocular depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--megadepth-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/Megadepth",
        help="Path to MegaDepth dataset root",
    )
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--max-train-files", type=int, default=10000)
    parser.add_argument("--max-val-files", type=int, default=1000)
    parser.add_argument("--min-valid-ratio", type=float, default=0.1)

    # Model
    parser.add_argument(
        "--encoder-type", choices=["vit_s", "vit_b", "vit_l"],
        default="vit_l",
        help="DepthAnything encoder variant",
    )
    parser.add_argument(
        "--encoder-kind", choices=["real", "placeholder"], default="real",
        help=(
            "'real' uses an in-tree dl_techniques.models.vit.ViT backbone; "
            "'placeholder' uses the legacy Conv-BN-ReLU stack (compat only)."
        ),
    )
    parser.add_argument(
        "--enable-semi-supervised", action="store_true",
        help=(
            "Activate the semi-supervised train_step path (FAL on unlabeled). "
            "Note: requires data of shape ((x_lab, x_unlab), y_lab); the current "
            "MegaDepthDataset only yields labeled batches — see train README."
        ),
    )
    parser.add_argument(
        "--use-feature-alignment", action="store_true",
        help="Enable feature-alignment loss (see Known Issue #2 in model README)",
    )
    parser.add_argument(
        "--cutmix-prob", type=float, default=0.5,
        help="Probability of applying CutMix during training",
    )
    parser.add_argument(
        "--color-jitter-strength", type=float, default=0.2,
        help="Strength of color jitter during training",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--lr-schedule", type=str, default="cosine_decay",
        help="Learning rate schedule type (forwarded to learning_rate_schedule_builder)",
    )
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=200)

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Pretrained init + reproducibility
    parser.add_argument(
        "--init-from", type=str, default=None,
        help=(
            "Path to a .keras checkpoint whose non-decoder weights will be "
            "loaded into the freshly-built depth model before training."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Python/NumPy/TF/Keras init.",
    )

    # GPU
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    setup_gpu(gpu_id=args.gpu)

    config = DepthAnythingTrainingConfig(
        megadepth_root=args.megadepth_root,
        train_split=args.train_split,
        patch_size=args.patch_size,
        min_valid_ratio=args.min_valid_ratio,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        encoder_type=args.encoder_type,
        encoder_kind=args.encoder_kind,
        enable_semi_supervised=args.enable_semi_supervised,
        use_feature_alignment=args.use_feature_alignment,
        cutmix_prob=args.cutmix_prob,
        color_jitter_strength=args.color_jitter_strength,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        warmup_epochs=args.warmup_epochs,
        lr_schedule_type=args.lr_schedule,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        init_from=args.init_from,
        seed=args.seed,
    )

    logger.info(
        f"Config: model=DepthAnything-{config.encoder_type}, "
        f"epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patch={config.patch_size}"
    )

    train_depth_anything(config)


if __name__ == "__main__":
    main()
