"""CliffordNet monocular depth estimation training script.

Trains a bias-free CliffordNet conditional denoiser for monocular depth
estimation on MegaDepth paired RGB+depth data.  The model predicts
depth purely from a single RGB image:

    model([grayscale(rgb), rgb]) → depth

The primary input is the grayscale luminance of the RGB image (bias-free
networks cannot produce non-zero output from zero input).  The model
learns ``depth = grayscale + residual`` via RGB conditioning.  Depth
maps are loaded from HDF5 files,
normalized per-sample to [-1, +1], and invalid pixels (depth == 0) are
masked in the loss.

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
    generate_training_curves,
)
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.cliffordnet.conditional_denoiser import (
    CliffordNetConditionalDenoiser,
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
    patch_size: int = 128
    min_valid_ratio: float = 0.1

    # Memory
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 1000
    dataset_shuffle_buffer: int = 5000

    # Model
    model_variant: str = "small"
    stochastic_depth_rate: float = 0.1
    use_geometric_downsample: bool = True
    use_geometric_upsample: bool = False
    upsample_interpolation: str = "nearest"

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
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load one RGB+depth pair, crop, normalize, augment.

    Returns ``(zeros_input, rgb, y_true)`` ready for the model, where
    ``y_true = concat([depth, mask], axis=-1)``.  The zeros input forces
    the model to predict depth purely from RGB conditioning (monocular
    depth estimation).  Returns ``None`` when the patch has too few
    valid pixels after all crop attempts.
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

    # Monocular depth: pass grayscale(RGB) as primary input.
    # Bias-free networks produce zero output from zero input (no bias
    # terms to create activation), so we need a non-zero signal.
    # Grayscale provides initial activation; RGB conditioning adds
    # color detail.  Model learns residual: depth - grayscale(RGB).
    gray = (
        0.2989 * rgb_patch[..., 0:1]
        + 0.5870 * rgb_patch[..., 1:2]
        + 0.1140 * rgb_patch[..., 2:3]
    )  # (ps, ps, 1), luminance in [-1, +1]
    gray_input = gray.copy()

    # y_true = concat([depth, mask], axis=-1)
    y_true = np.concatenate([depth_patch, valid_mask], axis=-1)

    return (
        gray_input.astype(np.float32),
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
        batch_noisy, batch_rgb, batch_ytrue = [], [], []

        attempts = 0
        while len(batch_noisy) < self.config.batch_size and attempts < self.config.batch_size * 3:
            # Wrap around for infinite-style iteration
            sample_idx = (
                idx * self.config.batch_size + len(batch_noisy) + attempts
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
            zeros_input, rgb, y_true = result
            batch_noisy.append(zeros_input)
            batch_rgb.append(rgb)
            batch_ytrue.append(y_true)

        # Pad if we couldn't fill the batch (rare)
        if not batch_noisy:
            ps = self.config.patch_size
            batch_noisy = [np.zeros((ps, ps, 1), dtype=np.float32)]
            batch_rgb = [np.zeros((ps, ps, 3), dtype=np.float32)]
            batch_ytrue = [np.zeros((ps, ps, 2), dtype=np.float32)]

        x = (np.stack(batch_noisy), np.stack(batch_rgb))
        y = np.stack(batch_ytrue)
        return x, y

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


# ---------------------------------------------------------------------
# MASKED LOSS
# ---------------------------------------------------------------------


class MaskedMSELoss(keras.losses.Loss):
    """MSE loss that ignores invalid depth pixels via a validity mask."""

    def __init__(self, name: str = "masked_mse", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true_and_mask, y_pred):
        # y_true_and_mask is (depth, mask) concatenated along last axis
        depth_true = y_true_and_mask[..., :1]
        mask = y_true_and_mask[..., 1:]

        sq_error = keras.ops.square(y_pred - depth_true) * mask
        # Mean over valid pixels only
        valid_count = keras.ops.maximum(
            keras.ops.sum(mask), keras.ops.cast(1.0, mask.dtype)
        )
        return keras.ops.sum(sq_error) / valid_count


# ---------------------------------------------------------------------
# MONITORING CALLBACKS
# ---------------------------------------------------------------------


class DepthMetricsVisualizationCallback(keras.callbacks.Callback):
    """Visualizes training/validation loss curves during training."""

    def __init__(self, config: DepthTrainingConfig):
        super().__init__()
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.visualization_dir = self.output_dir / "visualization_plots"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics = {"loss": [], "mae": []}
        self.val_metrics = {"val_loss": [], "val_mae": []}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            logs = {}
        for key in self.train_metrics:
            if key in logs:
                self.train_metrics[key].append(logs[key])
        for key in self.val_metrics:
            if key in logs:
                self.val_metrics[key].append(logs[key])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            self._create_metrics_plots(epoch + 1)

    def _create_metrics_plots(self, epoch: int):
        try:
            history_dict = {**self.train_metrics, **self.val_metrics}
            generate_training_curves(
                history=history_dict,
                results_dir=str(self.visualization_dir),
                filename=f"epoch_{epoch:03d}_metrics",
            )
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to create metrics plots: {e}")


class DepthVisualizationCallback(keras.callbacks.Callback):
    """Saves RGB → predicted depth vs GT depth comparison grids."""

    def __init__(
        self,
        config: DepthTrainingConfig,
        val_rgb_paths: List[str],
        val_depth_paths: List[str],
    ):
        super().__init__()
        self.config = config
        self.monitor_freq = config.monitor_every_n_epochs
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.results_dir = self.output_dir / "visualization_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.val_rgb_paths = val_rgb_paths[:8]
        self.val_depth_paths = val_depth_paths[:8]
        self.test_data = None

    def on_train_begin(self, logs=None):
        """Load a small set of validation samples for monitoring."""
        try:
            rgb_list, depth_list, mask_list = [], [], []
            for rp, dp in zip(self.val_rgb_paths, self.val_depth_paths):
                result = _load_and_process_pair(
                    rp, dp,
                    self.config.patch_size,
                    self.config.min_valid_ratio,
                    augment=False,
                )
                if result is None:
                    continue
                _, rgb, y_true = result
                rgb_list.append(rgb)
                depth_list.append(y_true[..., :1])
                mask_list.append(y_true[..., 1:])

            if rgb_list:
                self.test_data = {
                    "rgb": tf.constant(np.stack(rgb_list)),
                    "depth": tf.constant(np.stack(depth_list)),
                    "mask": tf.constant(np.stack(mask_list)),
                }
                logger.info(
                    f"Depth monitor: loaded {len(rgb_list)} validation "
                    f"patches, shape {self.test_data['rgb'].shape}"
                )
        except Exception as e:
            logger.warning(f"Depth monitor: failed to load test data: {e}")
            self.test_data = None

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.monitor_freq != 0:
            return
        if self.test_data is None:
            return

        try:
            rgb = self.test_data["rgb"]
            gt_depth = self.test_data["depth"]
            mask = self.test_data["mask"]

            # Predict depth from grayscale(RGB) + RGB (inference mode)
            gray_input = (
                0.2989 * rgb[..., 0:1]
                + 0.5870 * rgb[..., 1:2]
                + 0.1140 * rgb[..., 2:3]
            )
            pred_depth = self.model((gray_input, rgb), training=False)

            # Compute masked metrics
            valid_pixels = tf.reduce_sum(mask)
            if valid_pixels > 0:
                masked_mse = tf.reduce_sum(
                    tf.square(pred_depth - gt_depth) * mask
                ) / valid_pixels
                logger.info(
                    f"Epoch {epoch + 1} depth monitor — "
                    f"masked MSE: {masked_mse.numpy():.6f}"
                )

            self._save_grid(epoch + 1, rgb, gt_depth, pred_depth, mask)

            del pred_depth
            gc.collect()
        except Exception as e:
            logger.warning(
                f"Depth monitor error at epoch {epoch + 1}: {e}"
            )

    def _save_grid(
        self,
        epoch: int,
        rgb: tf.Tensor,
        gt_depth: tf.Tensor,
        pred_depth: tf.Tensor,
        mask: tf.Tensor,
    ):
        """Save RGB | GT depth | predicted depth comparison grid."""
        try:
            n = min(8, rgb.shape[0])
            fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
            fig.suptitle(
                f"CliffordNet Depth Estimation — Epoch {epoch}",
                fontsize=16, y=0.98,
            )

            for i in range(n):
                # RGB: [-1, 1] → [0, 1]
                rgb_img = np.clip((rgb[i].numpy() + 1.0) / 2.0, 0, 1)

                # GT depth: [-1, 1] → [0, 1], mask invalid regions
                gt = gt_depth[i].numpy().squeeze(-1)
                gt_vis = np.clip((gt + 1.0) / 2.0, 0, 1)
                m = mask[i].numpy().squeeze(-1)
                gt_vis = np.where(m > 0, gt_vis, 0.5)  # gray for invalid

                # Predicted depth
                pred = pred_depth[i].numpy().squeeze(-1)
                pred_vis = np.clip((pred + 1.0) / 2.0, 0, 1)

                labels = ["RGB", "GT Depth", "Predicted"]
                images = [rgb_img, gt_vis, pred_vis]
                cmaps = [None, "viridis", "viridis"]

                for row, (img, cmap) in enumerate(zip(images, cmaps)):
                    axes[row, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    if i == 0:
                        axes[row, i].set_ylabel(
                            labels[row], fontsize=12, rotation=0,
                            ha="right", va="center",
                        )
                    axes[row, i].axis("off")

            plt.tight_layout()
            plt.subplots_adjust(top=0.92, left=0.10)
            plt.savefig(
                self.results_dir / f"epoch_{epoch:03d}_depth.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
            plt.clf()
            gc.collect()
        except Exception as e:
            logger.warning(f"Failed to save depth grid: {e}")


# ---------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------


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

    common_callbacks.append(DepthMetricsVisualizationCallback(config))
    common_callbacks.append(
        DepthVisualizationCallback(config, val_rgb_paths, val_depth_paths)
    )

    return common_callbacks, results_dir


# ---------------------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------------------


def create_model(config: DepthTrainingConfig) -> CliffordNetConditionalDenoiser:
    """Create a CliffordNet conditional denoiser configured for depth."""
    return CliffordNetConditionalDenoiser.from_variant(
        variant=config.model_variant,
        in_channels=1,
        enable_dense_conditioning=True,
        dense_cond_channels=3,
        enable_discrete_conditioning=False,
        num_classes=0,
        stochastic_depth_rate=config.stochastic_depth_rate,
        use_geometric_downsample=config.use_geometric_downsample,
        use_geometric_upsample=config.use_geometric_upsample,
        upsample_interpolation=config.upsample_interpolation,
    )


# ---------------------------------------------------------------------
# BIAS-FREE VERIFICATION
# ---------------------------------------------------------------------


def _verify_bias_free(model: keras.Model) -> None:
    """Log bias-free compliance check for the model."""
    bias_layers = []
    for layer in model._flatten_layers():
        if hasattr(layer, "use_bias") and layer.use_bias:
            bias_layers.append(layer.name)
        if isinstance(layer, keras.layers.BatchNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{layer.name} (BN center=True)")
        if isinstance(layer, keras.layers.LayerNormalization):
            if hasattr(layer, "center") and layer.center:
                bias_layers.append(f"{layer.name} (LN center=True)")

    if bias_layers:
        logger.warning(
            f"Bias-free check: {len(bias_layers)} layer(s) have "
            f"bias/centering: {bias_layers[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED — all layers are bias-free")


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

    # Create model
    model = create_model(config)
    ps = config.patch_size
    model.build([
        (None, ps, ps, 1),   # noisy_depth
        (None, ps, ps, 3),   # rgb conditioning
    ])
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

    # Compile with masked MSE loss
    model.compile(
        optimizer=optimizer,
        loss=MaskedMSELoss(),
        metrics=["mae"],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Verify bias-free compliance
    _verify_bias_free(model)

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
            f.write("CliffordNet Depth Estimation Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model variant     : {config.model_variant}\n")
            f.write(f"Parameters        : {model.count_params():,}\n")
            f.write(f"Epochs trained    : {trained_epochs}\n")
            f.write(f"Best val_loss     : {best_loss:.6f}\n")
            f.write(f"Noise range       : [{config.noise_sigma_min}, {config.noise_sigma_max}]\n")
            f.write(f"Batch size        : {config.batch_size}\n")
            f.write(f"Patch size        : {config.patch_size}\n")
            f.write(f"Learning rate     : {config.learning_rate}\n")
            f.write(f"Train pairs       : {len(train_rgb)}\n")
            f.write(f"Val pairs         : {len(val_rgb)}\n")
            f.write(f"Duration          : {elapsed:.1f}s\n")
        logger.info(f"Summary written to: {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write training summary: {e}")

    # Save inference model with flexible spatial dims
    try:
        inference_model = create_model(config)
        inference_model.build([
            (None, None, None, 1),
            (None, None, None, 3),
        ])
        inference_model.set_weights(model.get_weights())
        inference_path = output_dir / "model_inference.keras"
        inference_model.save(str(inference_path))
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
        choices=["tiny", "small", "base"],
        default="small",
    )
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.1)
    parser.add_argument(
        "--downsample-mode",
        choices=["clifford", "conv"],
        default="clifford",
    )
    parser.add_argument(
        "--upsample-mode",
        choices=["clifford", "bilinear", "nearest"],
        default="nearest",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=128)
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
        stochastic_depth_rate=args.stochastic_depth_rate,
        use_geometric_downsample=(args.downsample_mode == "clifford"),
        use_geometric_upsample=(args.upsample_mode == "clifford"),
        upsample_interpolation=(
            "nearest" if args.upsample_mode == "nearest" else "bilinear"
        ),
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
        f"Config: model={config.model_variant}, "
        f"epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patch={config.patch_size}, "
        f"downsample={'clifford' if config.use_geometric_downsample else 'conv'}"
    )

    train_depth_estimation(config)


if __name__ == "__main__":
    main()
