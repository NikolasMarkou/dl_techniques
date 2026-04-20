"""CliffordNet monocular depth estimation training script.

Trains a CliffordNet U-Net depth estimator for monocular depth
estimation on MegaDepth paired RGB+depth data.  The model predicts
depth purely from a single RGB image:

    model(rgb) → depth

RGB is the direct input to a CliffordNet U-Net encoder-decoder backbone
(with bias) with a 1×1 Conv depth projection head.  No conditioning —
the model learns to predict depth purely from the image.  Depth maps are
loaded from HDF5 files, normalized per-sample to [-1, +1], and invalid
pixels (depth == 0) are masked in the loss.

Usage::

    # Train small model on MegaDepth
    python -m train.cliffordnet.train_depth_estimation \\
        --model-variant small \\
        --epochs 100 \\
        --gpu 0

    # Quick sanity check
    python -m train.cliffordnet.train_depth_estimation \\
        --model-variant tiny \\
        --batch-size 8 \\
        --epochs 2 \\
        --gpu 0
"""

import gc
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

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
    deep_supervision_schedule_builder,
)
from dl_techniques.models.cliffordnet.depth import CliffordNetDepthEstimator
from dl_techniques.metrics.depth_metrics import AbsRelMetric, DeltaThresholdMetric
from dl_techniques.callbacks.depth_visualization import (
    DepthPredictionGridCallback,
    DepthMetricsCurveCallback,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class DepthTrainingConfig:
    """Configuration for CliffordNet depth estimation training."""

    # Data
    megadepth_root: str = "/media/arxwn/data0_4tb/datasets/Megadepth"
    train_split: float = 0.9
    patch_size: int = 256
    min_valid_ratio: float = 0.1

    # Memory
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 1000
    dataset_shuffle_buffer: int = 5000

    # Model (CliffordNet variants: tiny, small, base, large, xlarge)
    model_variant: str = "base"

    # Training
    batch_size: int = 16
    epochs: int = 100
    patches_per_image: int = 4
    augment_data: bool = True
    steps_per_epoch: Optional[int] = None

    # Two-phase training: phase 1 = resize full image, phase 2 = random patches
    two_phase: bool = False
    phase1_epochs: int = 50
    phase2_epochs: int = 50

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Deep supervision
    enable_deep_supervision: bool = False
    deep_supervision_schedule_type: str = "linear_low_to_high"
    deep_supervision_schedule_config: Dict[str, Any] = field(
        default_factory=dict,
    )

    # Monitoring
    monitor_every_n_epochs: int = 5
    early_stopping_patience: int = 20
    validation_steps: Optional[int] = 200

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.two_phase:
            self.epochs = self.phase1_epochs + self.phase2_epochs
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"cliffordnet_depth_{self.model_variant}_{timestamp}"
            )
        if not 0.0 < self.min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be in (0, 1]")


# ---------------------------------------------------------------------
# DEPTH ESTIMATION LOSS (SSI + multi-scale gradient matching)
# ---------------------------------------------------------------------


class DepthEstimationLoss(keras.losses.Loss):
    """Masked L1 + multi-scale gradient matching loss for depth estimation.

    ``L = L1_masked + gradient_weight * L_grad``

    Since depth is already per-sample normalized to [-1, +1] in the data
    pipeline, we use plain masked L1 (not SSI) for the primary term.
    The gradient matching term compares finite-difference depth gradients
    at multiple scales to preserve edges (MiDaS recipe).  Both terms
    respect the validity mask.
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
        """Compute finite-difference depth gradients in x and y."""
        dx = d[:, :, 1:, :] - d[:, :, :-1, :]
        dy = d[:, 1:, :, :] - d[:, :-1, :, :]
        return dx, dy

    @staticmethod
    def _downsample_2x(d):
        """Average-pool 2× downsample."""
        return keras.ops.average_pool(d, pool_size=2, strides=2, padding="valid")

    def call(self, y_true_and_mask, y_pred):
        depth_true = y_true_and_mask[..., :1]
        mask = y_true_and_mask[..., 1:]

        # --- Masked L1 loss ---
        l1_error = keras.ops.abs(y_pred - depth_true) * mask
        valid_count = keras.ops.maximum(
            keras.ops.sum(mask), keras.ops.cast(1.0, mask.dtype)
        )
        l_l1 = keras.ops.sum(l1_error) / valid_count

        # --- Multi-scale gradient matching loss (masked) ---
        l_grad = keras.ops.cast(0.0, y_pred.dtype)
        d_pred = y_pred
        d_true = depth_true
        m = mask

        for _ in range(self.n_scales):
            dx_pred, dy_pred = self._gradient_xy(d_pred)
            dx_true, dy_true = self._gradient_xy(d_true)

            # Erode mask: both neighbors must be valid for a gradient
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



class DeepSupervisionWeightScheduler(keras.callbacks.Callback):
    """Dynamically adjusts deep supervision loss weights during training.

    Uses ``deep_supervision_schedule_builder`` to compute per-output
    weights as a function of training progress (0 → 1).
    """

    def __init__(
        self,
        config: DepthTrainingConfig,
        num_outputs: int,
    ) -> None:
        super().__init__()
        self.total_epochs = config.epochs
        self.num_outputs = num_outputs

        ds_config = {
            "type": config.deep_supervision_schedule_type,
            "config": config.deep_supervision_schedule_config,
        }
        self.scheduler = deep_supervision_schedule_builder(
            ds_config, self.num_outputs, invert_order=False,
        )
        logger.info(
            f"DS weight scheduler ({config.deep_supervision_schedule_type}) "
            f"for {num_outputs} outputs"
        )

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        new_weights = self.scheduler(progress)
        self.model.loss_weights = new_weights
        weights_str = ", ".join(f"{w:.4f}" for w in new_weights)
        logger.info(
            f"Epoch {epoch + 1}/{self.total_epochs} — "
            f"DS weights: [{weights_str}]"
        )


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
    """Load a fixed set of validation samples for visualization.

    Returns ``(rgb, depth, mask)`` numpy arrays or ``None`` on failure.
    """
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
    config: DepthTrainingConfig,
    val_rgb_paths: List[str],
    val_depth_paths: List[str],
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Create training callbacks: common utilities + depth-specific."""
    common_callbacks, results_dir = create_common_callbacks(
        model_name=config.model_variant,
        results_dir_prefix="cliffordnet_depth",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )

    # Redirect config paths to the common results directory
    config.output_dir = str(Path(results_dir).parent)
    config.experiment_name = Path(results_dir).name

    viz_dir = str(Path(results_dir) / "visualization_plots")

    # Metric curve plots
    common_callbacks.append(DepthMetricsCurveCallback(
        output_dir=viz_dir,
        train_metrics=["loss", "abs_rel", "delta_1.25"],
        frequency=5,
    ))

    # Depth prediction grids
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
            title="CliffordNet Depth Estimation",
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


def create_model(config: DepthTrainingConfig) -> keras.Model:
    """Create a CliffordNet U-Net for monocular depth estimation.

    Uses CliffordNet encoder-decoder with standard (with-bias)
    CliffordNet blocks, RGB input, and 1-channel depth output.
    """
    return CliffordNetDepthEstimator.from_variant(
        variant=config.model_variant,
        in_channels=3,
        out_channels=1,
        enable_deep_supervision=config.enable_deep_supervision,
    )




# ---------------------------------------------------------------------
# MULTI-SCALE DATASET WRAPPER
# ---------------------------------------------------------------------


class _MultiScaleDataset(keras.utils.PyDataset):
    """Wraps a MegaDepthDataset to produce multi-scale ``y_true`` labels.

    For each output dimension, resizes both the depth channel (bilinear)
    and the mask channel (nearest-neighbor) and re-concatenates them.
    """

    def __init__(
        self,
        base_dataset: "MegaDepthDataset",
        output_dims: List[Tuple[Optional[int], Optional[int]]],
    ) -> None:
        # PyDataset requires workers/use_multiprocessing — delegate to base
        super().__init__(
            workers=base_dataset.workers,
            use_multiprocessing=base_dataset.use_multiprocessing,
        )
        self.base = base_dataset
        self.output_dims = output_dims

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        rgb, y_true = self.base[idx]
        # y_true: (B, H, W, 2) — channel 0=depth, channel 1=mask
        labels = []
        for dim in self.output_dims:
            if dim[0] is None or dim == (y_true.shape[1], y_true.shape[2]):
                labels.append(y_true)
            else:
                h, w = dim
                # Use scipy for resizing — runs in worker processes
                # where CUDA context is not available
                from scipy.ndimage import zoom
                scale_h = h / y_true.shape[1]
                scale_w = w / y_true.shape[2]
                # zoom each sample: (B, H, W, 2) → (B, h, w, 2)
                resized = zoom(
                    y_true, (1, scale_h, scale_w, 1), order=1,
                )
                # Re-binarize mask after interpolation
                resized[..., 1:] = (resized[..., 1:] > 0.5).astype(
                    resized.dtype,
                )
                labels.append(resized)
        return rgb, tuple(labels)

    def on_epoch_end(self):
        self.base.on_epoch_end()


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_depth_estimation(config: DepthTrainingConfig) -> keras.Model:
    """Train CliffordNet for monocular depth estimation on MegaDepth."""
    logger.info(
        f"Starting CliffordNet depth estimation training: "
        f"{config.experiment_name}"
    )

    # Discover paired data
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

    # Train/val split
    n = len(rgb_paths)
    split_idx = int(n * config.train_split)
    train_rgb = rgb_paths[:split_idx]
    train_depth = depth_paths[:split_idx]
    val_rgb = rgb_paths[split_idx:]
    val_depth = depth_paths[split_idx:]

    # Apply file limits
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

    # Determine worker count: leave 2 cores free for GPU/TF
    import os
    num_workers = max(1, min(os.cpu_count() or 4, 10) - 2)
    logger.info(f"Using {num_workers} data-loading worker processes")

    # Create datasets (multiprocessing PyDataset)
    def _make_datasets(use_resize: bool = False):
        """Create train/val datasets with optional resize mode."""
        _train = MegaDepthDataset(
            train_rgb, train_depth,
            batch_size=config.batch_size,
            patch_size=config.patch_size,
            min_valid_ratio=config.min_valid_ratio,
            augment=config.augment_data,
            is_training=True, workers=num_workers,
            use_resize=use_resize,
        )
        _val = MegaDepthDataset(
            val_rgb, val_depth,
            batch_size=config.batch_size,
            patch_size=config.patch_size,
            min_valid_ratio=config.min_valid_ratio,
            augment=False,
            is_training=False, workers=max(1, num_workers // 2),
            use_resize=use_resize,
        )
        return _train, _val

    train_ds, val_ds = _make_datasets(use_resize=False)

    # Steps per epoch — len(train_ds) is n_pairs // batch_size
    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = len(train_ds)
    logger.info(f"Using {steps_per_epoch} steps per epoch")

    # Create model — RGB input, 1-channel depth output
    model = create_model(config)
    ps = config.patch_size
    model.build((None, ps, ps, 3))
    model.summary()

    # Optimizer with LR schedule
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

    # Detect multi-output (deep supervision) via probe forward pass
    has_multiple_outputs = config.enable_deep_supervision
    if has_multiple_outputs:
        probe = tf.zeros((1, ps, ps, 3))
        probe_out = model(probe, training=False)
        if isinstance(probe_out, (list, tuple)):
            num_outputs = len(probe_out)
            output_dims = [
                (o.shape[1], o.shape[2]) for o in probe_out
            ]
        else:
            has_multiple_outputs = False
            num_outputs = 1
            output_dims = None
        del probe, probe_out
    else:
        num_outputs = 1
        output_dims = None

    logger.info(
        f"Model has {num_outputs} output(s)"
        + (" (deep supervision)" if has_multiple_outputs else "")
    )

    # Wrap datasets for multi-scale labels if needed
    if has_multiple_outputs:
        logger.info(f"Multi-scale output dims: {output_dims}")
        train_ds = _MultiScaleDataset(train_ds, output_dims)
        val_ds = _MultiScaleDataset(val_ds, output_dims)

    # Compile with SSI + gradient matching loss (MiDaS/Depth Anything V2 recipe)
    if has_multiple_outputs:
        loss_fns = [
            DepthEstimationLoss(gradient_weight=0.5, n_scales=4)
            for _ in range(num_outputs)
        ]
        initial_weights = [1.0 / num_outputs] * num_outputs
        # Metrics only on primary (full-res) output, empty for auxiliaries
        metrics = [
            [AbsRelMetric(name="abs_rel"),
             DeltaThresholdMetric(threshold=1.25, name="delta_1.25")],
        ] + [[] for _ in range(num_outputs - 1)]
    else:
        loss_fns = DepthEstimationLoss(gradient_weight=0.5, n_scales=4)
        initial_weights = None
        metrics = [
            AbsRelMetric(name="abs_rel"),
            DeltaThresholdMetric(threshold=1.25, name="delta_1.25"),
        ]

    model.compile(
        optimizer=optimizer,
        loss=loss_fns,
        loss_weights=initial_weights,
        metrics=metrics,
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Callbacks
    callbacks, results_dir = create_callbacks(config, val_rgb, val_depth)
    if has_multiple_outputs:
        callbacks.append(
            DeepSupervisionWeightScheduler(config, num_outputs)
        )
    output_dir = Path(results_dir)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Train
    start_time = time.time()

    if config.two_phase:
        # Phase 1: full image resized to patch_size (global context)
        logger.info(
            f"=== PHASE 1: Full-image resize training "
            f"({config.phase1_epochs} epochs) ==="
        )
        phase1_train, phase1_val = _make_datasets(use_resize=True)
        if has_multiple_outputs:
            phase1_train = _MultiScaleDataset(phase1_train, output_dims)
            phase1_val = _MultiScaleDataset(phase1_val, output_dims)

        history = model.fit(
            phase1_train,
            epochs=config.phase1_epochs,
            validation_data=phase1_val,
            callbacks=callbacks,
            verbose=1,
        )

        # Phase 2: random patch training (local detail)
        logger.info(
            f"=== PHASE 2: Random patch training "
            f"({config.phase2_epochs} epochs) ==="
        )

        # Reset EarlyStopping state — phase 2 is a different data
        # distribution so the phase 1 best val_loss is not comparable.
        for cb in callbacks:
            if isinstance(cb, keras.callbacks.EarlyStopping):
                cb.best = float("inf")
                cb.wait = 0
                cb.stopped_epoch = 0
                logger.info("Reset EarlyStopping for phase 2")
                break

        phase2_train, phase2_val = _make_datasets(use_resize=False)
        if has_multiple_outputs:
            phase2_train = _MultiScaleDataset(phase2_train, output_dims)
            phase2_val = _MultiScaleDataset(phase2_val, output_dims)

        history = model.fit(
            phase2_train,
            initial_epoch=config.phase1_epochs,
            epochs=config.phase1_epochs + config.phase2_epochs,
            validation_data=phase2_val,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            train_ds,
            epochs=config.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.2f} seconds")

    # Save training history
    try:
        history_dict = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    # Training summary
    try:
        trained_epochs = len(history.history.get("loss", []))
        best_loss = min(history.history.get("val_loss", [float("nan")]))
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Monocular Depth Estimation Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model variant     : {config.model_variant}\n")
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

    # Save inference model (already supports flexible spatial dims via Input(None,None,3))
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
        description="Train CliffordNet Monocular Depth Estimation",
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
        "--model-variant",
        choices=list(CliffordNetDepthEstimator.MODEL_VARIANTS.keys()),
        default="base",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--monitor-every", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)

    # Deep supervision
    parser.add_argument(
        "--enable-deep-supervision", action="store_true",
        help="Enable deep supervision with auxiliary decoder outputs",
    )
    parser.add_argument(
        "--ds-schedule", type=str, default="linear_low_to_high",
        help="Deep supervision weight schedule type",
    )

    # Two-phase training
    parser.add_argument(
        "--two-phase", action="store_true",
        help="Phase 1: full-image resize (global), Phase 2: patches (local)",
    )
    parser.add_argument(
        "--phase1-epochs", type=int, default=50,
        help="Number of epochs for phase 1 (resize)",
    )
    parser.add_argument(
        "--phase2-epochs", type=int, default=50,
        help="Number of epochs for phase 2 (patches)",
    )

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)

    # GPU
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    setup_gpu(gpu_id=args.gpu)

    config = DepthTrainingConfig(
        megadepth_root=args.megadepth_root,
        train_split=args.train_split,
        patch_size=args.patch_size,
        min_valid_ratio=args.min_valid_ratio,
        max_train_files=args.max_train_files,
        max_val_files=args.max_val_files,
        model_variant=args.model_variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patches_per_image=args.patches_per_image,
        monitor_every_n_epochs=args.monitor_every,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        steps_per_epoch=args.max_steps_per_epoch,
        enable_deep_supervision=args.enable_deep_supervision,
        deep_supervision_schedule_type=args.ds_schedule,
        two_phase=args.two_phase,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
    )

    logger.info(
        f"Config: model=CliffordNet-{config.model_variant}, "
        f"epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patch={config.patch_size}"
    )

    train_depth_estimation(config)


if __name__ == "__main__":
    main()
