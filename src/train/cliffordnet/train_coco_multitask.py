"""CliffordNetUNet COCO 2017 multi-task training script.

Trains a :class:`CliffordNetUNet` on COCO 2017 with two simultaneous heads:

- **classification** — multi-label (80 classes) from the bottleneck features.
- **segmentation**   — semantic (81 classes, 0 = background) from decoder level 0
  with always-on deep supervision (aux heads at every deeper decoder level).

The goal is to produce a pretrained backbone that can be reused for depth
estimation, detection, or instance segmentation by swapping head configs.

Usage::

    # Smoke run (tiny variant, subset, 1 epoch)
    MPLBACKEND=Agg python -m train.cliffordnet.train_coco_multitask \\
        --variant tiny --epochs 1 --max-train-images 200 --max-val-images 50 \\
        --image-size 128 --batch-size 4 --gpu 0

    # Full training
    MPLBACKEND=Agg python -m train.cliffordnet.train_coco_multitask \\
        --variant base --epochs 60 --image-size 384 --batch-size 8 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import keras
import numpy as np
import tensorflow as tf

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
)
from dl_techniques.datasets.vision.coco_multitask_local import (
    COCO2017MultiTaskLoader,
    COCOMultiTaskConfig,
    COCO_DEFAULT_ROOT,
    NUM_COCO_CLASSES,
)
from dl_techniques.callbacks.coco_multitask_visualization import (
    COCOMultiTaskPredictionGridCallback,
)
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from dl_techniques.metrics.brier_score import BrierScore
from dl_techniques.models.cliffordnet.unet import CliffordNetUNet
from dl_techniques.optimization import (
    deep_supervision_schedule_builder,
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger


NUM_SEG_CLASSES: int = NUM_COCO_CLASSES + 1  # +1 background


class _LogitF1Score(keras.metrics.F1Score):
    """F1Score that applies sigmoid to its logit inputs before thresholding.

    Keras 3's F1Score requires ``threshold`` in ``(0, 1]`` — i.e. it expects
    probability-space inputs.  Our classification head outputs raw logits
    (consistent with ``BinaryCrossentropy(from_logits=True)``), so we sigmoid
    internally and keep a natural ``threshold=0.5``.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            y_true, keras.ops.sigmoid(y_pred), sample_weight
        )


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class COCOMultiTaskTrainingConfig:
    """Configuration for CliffordNetUNet multi-task COCO training."""

    # Data
    coco_root: str = COCO_DEFAULT_ROOT
    image_size: int = 384
    max_train_images: Optional[int] = None
    max_val_images: Optional[int] = None

    # Model
    model_variant: str = "tiny"
    classification_hidden_dim: Optional[int] = None  # extra Dense in cls head
    segmentation_hidden_dim: Optional[int] = 64  # Conv3x3 before 1x1 out

    # Training
    batch_size: int = 8
    epochs: int = 60
    augment_data: bool = True

    # Loss weighting (per head; deep supervision aux weights decay geometrically
    # and are managed by the DS scheduler callback)
    classification_loss_weight: float = 1.0
    segmentation_loss_weight: float = 1.0

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 3
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0

    # Deep supervision (for segmentation only)
    deep_supervision_schedule_type: str = "linear_low_to_high"
    deep_supervision_schedule_config: Dict[str, Any] = field(default_factory=dict)

    # Monitoring
    early_stopping_patience: int = 20
    visualization_frequency: int = 1  # save prediction grid every N epochs
    visualization_samples: int = 6
    include_tensorboard: bool = True

    # I/O
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    workers: int = 4
    use_multiprocessing: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"cliffordnet_coco_multitask_{self.model_variant}_{ts}"


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def create_model(config: COCOMultiTaskTrainingConfig) -> keras.Model:
    """Create a CliffordNetUNet with classification + segmentation heads."""
    head_configs = {
        "classification": {
            "type": "classification",
            "tap": "bottleneck",
            "num_classes": NUM_COCO_CLASSES,
            "hidden_dim": config.classification_hidden_dim,
        },
        "segmentation": {
            "type": "segmentation",
            "tap": 0,
            "num_classes": NUM_SEG_CLASSES,
            "out_channels": NUM_SEG_CLASSES,
            "deep_supervision": True,
            "hidden_dim": config.segmentation_hidden_dim,
        },
    }
    logger.info(
        f"Creating CliffordNetUNet ({config.model_variant}) with heads: "
        f"{list(head_configs.keys())}"
    )
    return CliffordNetUNet.from_variant(
        config.model_variant,
        in_channels=3,
        head_configs=head_configs,
    )


# ---------------------------------------------------------------------
# SEGMENTATION DEEP-SUPERVISION: multi-scale label wrapper
# ---------------------------------------------------------------------


class _MultiScaleSegDataset(keras.utils.PyDataset):
    """Wrap :class:`COCO2017MultiTaskLoader` to emit aux segmentation labels.

    Generates ``segmentation_aux_k`` labels at ``H/2^k, W/2^k`` via
    nearest-neighbor downsampling of the primary segmentation map.
    """

    def __init__(
        self,
        base: COCO2017MultiTaskLoader,
        seg_aux_sizes: List[int],  # ordered [H_aux_1, H_aux_2, ...] (square)
    ) -> None:
        super().__init__(
            workers=base.config.workers,
            use_multiprocessing=base.config.use_multiprocessing,
        )
        self.base = base
        self.seg_aux_sizes = list(seg_aux_sizes)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        import numpy as np
        from PIL import Image

        x, labels = self.base[idx]
        cls = labels["classification"]
        seg = labels["segmentation"]  # (B, H, W) int32
        out: Dict[str, Any] = {
            "classification": cls,
            "segmentation": seg,
        }
        for k, size in enumerate(self.seg_aux_sizes, start=1):
            # Per-sample nearest-neighbor resize
            b = seg.shape[0]
            small = np.empty((b, size, size), dtype=np.int32)
            for i in range(b):
                pil = Image.fromarray(seg[i].astype(np.int32), mode="I")
                pil = pil.resize((size, size), resample=Image.NEAREST)
                small[i] = np.asarray(pil, dtype=np.int32)
            out[f"segmentation_aux_{k}"] = small
        return x, out

    def on_epoch_end(self) -> None:
        self.base.on_epoch_end()


# ---------------------------------------------------------------------
# DS WEIGHT SCHEDULER (segmentation outputs only)
# ---------------------------------------------------------------------


class _SegDeepSupervisionScheduler(keras.callbacks.Callback):
    """Adjusts loss_weights for segmentation outputs over training progress.

    Leaves the ``classification`` weight fixed at ``cls_weight`` and
    re-distributes the segmentation budget across ``[segmentation, seg_aux_1, ...]``
    via :func:`deep_supervision_schedule_builder`.
    """

    def __init__(
        self,
        total_epochs: int,
        seg_output_names: List[str],
        cls_weight: float,
        seg_total_weight: float,
        schedule_type: str,
        schedule_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.seg_output_names = list(seg_output_names)
        self.cls_weight = cls_weight
        self.seg_total_weight = seg_total_weight
        ds_config = {"type": schedule_type, "config": schedule_config}
        self.scheduler = deep_supervision_schedule_builder(
            ds_config, len(seg_output_names), invert_order=False,
        )
        logger.info(
            f"SegDS scheduler ({schedule_type}) "
            f"for seg outputs {seg_output_names}"
        )

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        progress = min(1.0, epoch / max(1, self.total_epochs - 1))
        seg_weights = self.scheduler(progress)
        weights: Dict[str, float] = {"classification": float(self.cls_weight)}
        for name, w in zip(self.seg_output_names, seg_weights):
            weights[name] = float(w) * float(self.seg_total_weight)
        self.model.loss_weights = weights
        wstr = ", ".join(f"{k}={v:.4f}" for k, v in weights.items())
        logger.info(f"Epoch {epoch + 1}/{self.total_epochs} — weights: {wstr}")


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_coco_multitask(config: COCOMultiTaskTrainingConfig) -> keras.Model:
    logger.info(f"Starting CliffordNet COCO multi-task training: {config.experiment_name}")

    # Build datasets
    train_cfg = COCOMultiTaskConfig(
        coco_root=config.coco_root,
        split="train2017",
        image_size=config.image_size,
        batch_size=config.batch_size,
        max_images=config.max_train_images,
        shuffle=True,
        augment=config.augment_data,
        workers=config.workers,
        use_multiprocessing=config.use_multiprocessing,
        seed=0,
    )
    val_cfg = COCOMultiTaskConfig(
        coco_root=config.coco_root,
        split="val2017",
        image_size=config.image_size,
        batch_size=config.batch_size,
        max_images=config.max_val_images,
        shuffle=False,
        augment=False,
        workers=max(1, config.workers // 2),
        use_multiprocessing=config.use_multiprocessing,
        seed=1,
    )
    train_base = COCO2017MultiTaskLoader(train_cfg)
    val_base = COCO2017MultiTaskLoader(val_cfg)

    steps_per_epoch = len(train_base)
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # Build model + probe to discover output structure
    model = create_model(config)
    model.build((None, config.image_size, config.image_size, 3))
    model.summary()

    probe = tf.zeros((1, config.image_size, config.image_size, 3))
    probe_out = model(probe, training=False)
    if not isinstance(probe_out, dict):
        raise RuntimeError(f"Expected dict output, got {type(probe_out)}")
    seg_primary = "segmentation"
    seg_aux_names = sorted(
        (k for k in probe_out if k.startswith("segmentation_aux_")),
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    seg_output_names = [seg_primary] + seg_aux_names
    seg_aux_spatial_sizes = [probe_out[n].shape[1] for n in seg_aux_names]
    del probe, probe_out

    logger.info(f"Segmentation outputs: {seg_output_names}")
    logger.info(f"Aux spatial sizes: {seg_aux_spatial_sizes}")

    # Wrap datasets to emit aux segmentation labels
    train_ds = _MultiScaleSegDataset(train_base, seg_aux_spatial_sizes)
    val_ds = _MultiScaleSegDataset(val_base, seg_aux_spatial_sizes)

    # Optimizer + LR
    lr_schedule = learning_rate_schedule_builder(
        {
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": steps_per_epoch * config.epochs,
            "warmup_steps": steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
            "weight_decay": config.weight_decay,
        },
        lr_schedule,
    )

    # Losses: BCE for classification (multi-label, from logits); SCCE for segmentation
    losses: Dict[str, Any] = {
        "classification": keras.losses.BinaryCrossentropy(from_logits=True),
    }
    for name in seg_output_names:
        losses[name] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Initial loss_weights (will be overwritten by scheduler on epoch_begin)
    n_seg = len(seg_output_names)
    initial_weights: Dict[str, float] = {
        "classification": config.classification_loss_weight,
    }
    for i, name in enumerate(seg_output_names):
        # Primary gets most weight; aux decreasing geometrically
        w = (0.5 ** i) if i > 0 else 1.0
        initial_weights[name] = w * (config.segmentation_loss_weight / sum(
            (0.5 ** j) if j > 0 else 1.0 for j in range(n_seg)
        ))

    # Metrics — primary outputs only.
    # Note on classification: BinaryAccuracy(threshold=0) is misleading on
    # multi-label COCO — the "always negative" baseline is ~94% (4.5 / 80
    # average positives per image).  Use macro-F1 + multi-label AUC, which
    # start near 0 and grow as the model learns to predict positives.
    # Precision/Recall skipped because Keras 3 requires sigmoid-space thresholds
    # in (0, 1]; we compute them post-hoc from AUC / F1 if needed.
    metrics: Dict[str, Any] = {
        "classification": [
            _LogitF1Score(
                average="macro", threshold=0.5, name="macro_f1",
            ),
            keras.metrics.AUC(
                multi_label=True, from_logits=True, name="auc",
            ),
            BrierScore(
                from_logits=True, name="brier",
            ),
        ],
        "segmentation": [
            keras.metrics.SparseCategoricalAccuracy(name="pix_acc"),
        ],
    }
    for name in seg_aux_names:
        metrics[name] = []

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=initial_weights,
        metrics=metrics,
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    # Callbacks
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="cliffordnet_coco_multitask",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_analyzer=False,  # large — skip for multi-task
        include_tensorboard=config.include_tensorboard,
    )
    callbacks.append(
        _SegDeepSupervisionScheduler(
            total_epochs=config.epochs,
            seg_output_names=seg_output_names,
            cls_weight=config.classification_loss_weight,
            seg_total_weight=config.segmentation_loss_weight,
            schedule_type=config.deep_supervision_schedule_type,
            schedule_config=config.deep_supervision_schedule_config,
        )
    )

    # Prediction-grid visualization: load a fixed set of val samples once.
    try:
        vis_batches = []
        needed = config.visualization_samples
        for bi in range(len(val_base)):
            xb, yb = val_base[bi]
            vis_batches.append((xb, yb["classification"], yb["segmentation"]))
            if sum(x[0].shape[0] for x in vis_batches) >= needed:
                break
        if vis_batches:
            vis_rgb = np.concatenate([b[0] for b in vis_batches], axis=0)[:needed]
            vis_cls = np.concatenate([b[1] for b in vis_batches], axis=0)[:needed]
            vis_seg = np.concatenate([b[2] for b in vis_batches], axis=0)[:needed]
            vis_dir = Path(results_dir) / "visualization_plots"
            callbacks.append(
                COCOMultiTaskPredictionGridCallback(
                    val_rgb=vis_rgb,
                    val_seg=vis_seg,
                    val_cls=vis_cls,
                    output_dir=str(vis_dir),
                    frequency=config.visualization_frequency,
                    max_samples=needed,
                )
            )
            logger.info(
                f"Visualization callback enabled: "
                f"{vis_rgb.shape[0]} samples every {config.visualization_frequency} epochs"
            )
        else:
            logger.warning("No val samples for visualization callback; skipping.")
    except Exception as e:
        logger.warning(f"Failed to set up visualization callback: {e}")

    # Per-epoch training curves PNGs (loss / classification / segmentation).
    curves_dir = Path(results_dir) / "training_curves"
    callbacks.append(
        TrainingCurvesCallback(
            output_dir=str(curves_dir),
            frequency=config.visualization_frequency,
        )
    )
    logger.info(f"Training curves callback enabled → {curves_dir}")

    output_dir = Path(results_dir)
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {k: v for k, v in config.__dict__.items()},
            f, indent=2, default=str,
        )

    start = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start
    logger.info(f"Training completed in {elapsed / 60:.1f} min")

    final_path = output_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    return model


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CliffordNetUNet on COCO multi-task")
    parser.add_argument(
        "--variant",
        choices=list(CliffordNetUNet.MODEL_VARIANTS.keys()),
        default="tiny",
    )
    parser.add_argument("--coco-root", type=str, default=COCO_DEFAULT_ROOT)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-multiprocessing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--viz-frequency", type=int, default=1,
        help="Save RGB|GT|pred seg grid every N epochs (default: 1 = every epoch).",
    )
    parser.add_argument(
        "--viz-samples", type=int, default=6,
        help="Number of fixed val samples shown in the grid.",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true",
        help="Disable TensorBoard logging.",
    )
    args = parser.parse_args()

    setup_gpu(args.gpu)

    config = COCOMultiTaskTrainingConfig(
        coco_root=args.coco_root,
        image_size=args.image_size,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        model_variant=args.variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        augment_data=not args.no_augment,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        workers=args.workers,
        use_multiprocessing=not args.no_multiprocessing,
        visualization_frequency=args.viz_frequency,
        visualization_samples=args.viz_samples,
        include_tensorboard=not args.no_tensorboard,
    )
    train_coco_multitask(config)


if __name__ == "__main__":
    main()
