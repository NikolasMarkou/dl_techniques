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
import h5py
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
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

    # Optimization
    learning_rate: float = 1e-3
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

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = (
                f"cliffordnet_depth_{self.model_variant}_{timestamp}"
            )
        if not 0.0 < self.min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio must be in (0, 1]")


# ---------------------------------------------------------------------
# MEGADEPTH DATA PIPELINE
# ---------------------------------------------------------------------


def discover_megadepth_pairs(
    root: str,
    max_files: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Scan MegaDepth scenes for matched RGB+depth file pairs.

    MegaDepth structure::

        {root}/{scene}/dense{0,1}/imgs/*.jpg
        {root}/{scene}/dense{0,1}/depths/*.{h5,npy}

    Returns matched lists ``(rgb_paths, depth_paths)`` paired by stem.
    """
    root_path = Path(root)
    rgb_paths = []
    depth_paths = []

    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        for dense_dir in sorted(scene_dir.iterdir()):
            if not dense_dir.is_dir() or not dense_dir.name.startswith("dense"):
                continue

            imgs_dir = dense_dir / "imgs"
            depths_dir = dense_dir / "depths"
            if not imgs_dir.exists() or not depths_dir.exists():
                continue

            # Build stem → path maps
            img_stems = {}
            for fp in imgs_dir.iterdir():
                if fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img_stems[fp.stem] = str(fp)

            for fp in depths_dir.iterdir():
                if fp.suffix.lower() == ".h5":
                    stem = fp.stem
                    if stem in img_stems:
                        rgb_paths.append(img_stems[stem])
                        depth_paths.append(str(fp))

            if max_files and len(rgb_paths) >= max_files:
                rgb_paths = rgb_paths[:max_files]
                depth_paths = depth_paths[:max_files]
                return rgb_paths, depth_paths

    logger.info(f"Discovered {len(rgb_paths)} MegaDepth RGB+depth pairs")
    return rgb_paths, depth_paths


def _load_and_process_pair(
    rgb_path: str,
    depth_path: str,
    patch_size: int,
    min_valid_ratio: float,
    augment: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one RGB+depth pair, crop, normalize, augment.

    Returns ``(rgb, y_true)`` where ``y_true = concat([depth, mask],
    axis=-1)``.  Returns ``None`` when the patch has too few valid
    pixels after all crop attempts.
    """
    from PIL import Image

    # Load RGB
    rgb = plt.imread(rgb_path)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 127.5 - 1.0
    else:
        rgb = rgb.astype(np.float32) * 2.0 - 1.0

    # Load depth from HDF5
    with h5py.File(depth_path, "r") as f:
        depth = f["depth"][:].astype(np.float32)

    h, w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]

    # Resize RGB to match depth if needed
    if rgb_h != h or rgb_w != w:
        rgb_pil = Image.fromarray(
            ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )
        rgb_pil = rgb_pil.resize((w, h), Image.BILINEAR)
        rgb = np.array(rgb_pil, dtype=np.float32) / 127.5 - 1.0

    # Ensure minimum size for patching
    if h < patch_size or w < patch_size:
        scale = max(patch_size / h, patch_size / w) + 0.01
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_pil = Image.fromarray(
            ((rgb + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)
        rgb = np.array(rgb_pil, dtype=np.float32) / 127.5 - 1.0
        depth_pil = Image.fromarray(depth)
        depth_pil = depth_pil.resize((new_w, new_h), Image.NEAREST)
        depth = np.array(depth_pil, dtype=np.float32)
        h, w = new_h, new_w

    # Random crop — try up to 10 times for enough valid pixels
    for _ in range(10):
        y = np.random.randint(0, h - patch_size + 1)
        x = np.random.randint(0, w - patch_size + 1)
        depth_patch = depth[y:y + patch_size, x:x + patch_size]
        valid_mask = (depth_patch > 0).astype(np.float32)
        if valid_mask.mean() >= min_valid_ratio:
            break
    else:
        if valid_mask.mean() < min_valid_ratio:
            return None

    rgb_patch = rgb[y:y + patch_size, x:x + patch_size, :3]
    depth_patch = depth_patch[..., np.newaxis]  # (ps, ps, 1)
    valid_mask = valid_mask[..., np.newaxis]  # (ps, ps, 1)

    # Normalize depth per-sample to [-1, +1] using valid pixel range
    valid_depths = depth_patch[valid_mask > 0]
    if len(valid_depths) > 0:
        d_min = valid_depths.min()
        d_max = valid_depths.max()
        d_range = d_max - d_min
        if d_range > 1e-6:
            depth_patch = np.where(
                valid_mask > 0,
                (depth_patch - d_min) / d_range * 2.0 - 1.0,
                0.0,
            )
        else:
            depth_patch = np.zeros_like(depth_patch)
    else:
        depth_patch = np.zeros_like(depth_patch)

    # Augmentation (numpy — runs in worker process)
    if augment:
        combined = np.concatenate(
            [rgb_patch, depth_patch, valid_mask], axis=-1
        )
        if np.random.random() > 0.5:
            combined = combined[:, ::-1, :]
        if np.random.random() > 0.5:
            combined = combined[::-1, :, :]
        k = np.random.randint(0, 4)
        combined = np.rot90(combined, k, axes=(0, 1))
        rgb_patch = combined[..., :3].copy()
        depth_patch = combined[..., 3:4].copy()
        valid_mask = combined[..., 4:5].copy()

    # y_true = concat([depth, mask], axis=-1)
    y_true = np.concatenate([depth_patch, valid_mask], axis=-1)

    return (
        rgb_patch.astype(np.float32),
        y_true.astype(np.float32),
    )


class MegaDepthDataset(keras.utils.PyDataset):
    """Multiprocessing-capable dataset for MegaDepth RGB→depth pairs.

    Each worker process independently loads, crops, augments, and
    normalizes samples — bypassing the GIL for true parallel I/O.
    """

    def __init__(
        self,
        rgb_paths: List[str],
        depth_paths: List[str],
        config: DepthTrainingConfig,
        is_training: bool = True,
        workers: int = 8,
        **kwargs,
    ):
        super().__init__(workers=workers, use_multiprocessing=True, **kwargs)
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        self.config = config
        self.is_training = is_training
        self.n_pairs = len(rgb_paths)
        self.indices = np.arange(self.n_pairs)
        if is_training:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return max(1, self.n_pairs // self.config.batch_size)

    def __getitem__(self, idx: int):
        batch_rgb, batch_ytrue = [], []

        attempts = 0
        while len(batch_rgb) < self.config.batch_size and attempts < self.config.batch_size * 3:
            sample_idx = (
                idx * self.config.batch_size + len(batch_rgb) + attempts
            ) % self.n_pairs
            i = self.indices[sample_idx]
            attempts += 1

            result = _load_and_process_pair(
                self.rgb_paths[i],
                self.depth_paths[i],
                self.config.patch_size,
                self.config.min_valid_ratio,
                augment=self.is_training and self.config.augment_data,
            )
            if result is None:
                continue
            rgb, y_true = result
            batch_rgb.append(rgb)
            batch_ytrue.append(y_true)

        # Pad if we couldn't fill the batch (rare)
        if not batch_rgb:
            ps = self.config.patch_size
            batch_rgb = [np.zeros((ps, ps, 3), dtype=np.float32)]
            batch_ytrue = [np.zeros((ps, ps, 2), dtype=np.float32)]

        return np.stack(batch_rgb), np.stack(batch_ytrue)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


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
        model_name=config.experiment_name,
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
    )




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
    train_ds = MegaDepthDataset(
        train_rgb, train_depth, config,
        is_training=True, workers=num_workers,
    )
    val_ds = MegaDepthDataset(
        val_rgb, val_depth, config,
        is_training=False, workers=max(1, num_workers // 2),
    )

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
        },
        lr_schedule,
    )

    # Compile with SSI + gradient matching loss (MiDaS/Depth Anything V2 recipe)
    model.compile(
        optimizer=optimizer,
        loss=DepthEstimationLoss(gradient_weight=0.5, n_scales=4),
        metrics=[
            AbsRelMetric(name="abs_rel"),
            DeltaThresholdMetric(threshold=1.25, name="delta_1.25"),
        ],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Callbacks
    callbacks, results_dir = create_callbacks(config, val_rgb, val_depth)
    output_dir = Path(results_dir)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Train
    start_time = time.time()

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
    )

    logger.info(
        f"Config: model=CliffordNet-{config.model_variant}, "
        f"epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patch={config.patch_size}"
    )

    train_depth_estimation(config)


if __name__ == "__main__":
    main()
