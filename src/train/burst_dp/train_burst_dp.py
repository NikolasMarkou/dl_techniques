"""Training script for BurstDP on COCO 2017 with synthetic auxiliary views.

Usage::

    MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \\
        --preset burst_dp_small \\
        --image-size 256 \\
        --batch-size 4 \\
        --epochs 40 \\
        --n-max 5 --n-min 1 \\
        --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \\
        --out-dir src/results/burst_dp/run01 \\
        --gpu 0

The training pipeline:
    - shared ViT encoder
    - ref-conditioned set-fusion stack
    - 2 heads (recon, segmentation)

Losses:
    - reconstruction : Charbonnier loss vs. clean reference
    - segmentation   : sparse categorical cross-entropy

Periodic visualization: every ``--viz-every-steps`` optimizer steps and / or
every ``--viz-every-epochs`` epochs the trainer saves a recon+segmentation
comparison grid under ``out_dir/viz/`` for a fixed val batch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.coco_multitask_local import COCO_DEFAULT_ROOT
from dl_techniques.callbacks.burst_dp_visualization import BurstDPVisualizationCallback
from dl_techniques.datasets.vision.coco_burst_dp import (
    build_coco_burst_dp_datasets,
    default_anchor_spec,
    default_aux_spec,
)
from dl_techniques.datasets.vision.image_folder_burst_dp import (
    build_div2k_burst_dp_datasets,
    build_vggface2_burst_dp_datasets,
)
# COCOBurstDPConfig + COCO2017BurstDPLoader were referenced by the now-removed
# tf.data wrapper (see D-001 below). Keras 3 model.fit consumes the PyDataset
# directly, so neither name is needed in this module anymore.
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.models.burst_dp import (
    DEFAULT_NUM_SEG_CLASSES,
    BurstDP,
    create_burst_dp,
)
from dl_techniques.utils.logger import logger

# train.common helpers — local imports are kept conditional so the unit test
# environment does not need the full training stack on PYTHONPATH.
try:
    from train.common.gpu import setup_gpu
except Exception:  # pragma: no cover — fallback for editable installs
    setup_gpu = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def charbonnier_loss(epsilon: float = 1e-3):
    """Robust L1-ish loss: sqrt((y_true - y_pred)^2 + eps^2)."""
    def _loss(y_true, y_pred):
        diff = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(diff * diff + epsilon * epsilon))
    _loss.__name__ = "charbonnier_loss"
    return _loss


def sparse_seg_ce_loss():
    """Sparse categorical cross-entropy from logits."""
    cce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def _loss(y_true, y_pred):
        # y_true: (B, H, W) int; y_pred: (B, H, W, C) logits.
        return cce(y_true, y_pred)
    _loss.__name__ = "sparse_seg_ce"
    return _loss


# DECISION plan_2026-05-19_39a6a454/D-001
# Earlier versions of this script wrapped `COCO2017BurstDPLoader` (a
# `keras.utils.PyDataset` with `workers=4, use_multiprocessing=True`) in
# `tf.data.Dataset.from_generator(...)` for `model.fit`. That wrap actively
# defeats the PyDataset's parallel worker pool (the generator runs single-
# threaded in the main process). Keras 3 supports passing a `PyDataset`
# directly to `model.fit(...)` and iterates with the configured workers.
# Reuse-review proposed "switch to indefinite generator + steps_per_epoch"
# (the video_jepa pattern). That is the right shape for *generator-based*
# loaders without a PyDataset abstraction; for us, the cleaner fix is to
# drop the wrap entirely. See plans/plan_2026-05-19_39a6a454/findings/
# 03-train-script-current-state.md.


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BurstDP on COCO 2017.")
    p.add_argument("--preset", type=str, default="burst_dp_small",
                   choices=["burst_dp_pico", "burst_dp_tiny", "burst_dp_small", "burst_dp_base"])
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--n-min", type=int, default=1)

    p.add_argument("--dataset", type=str, default="coco",
                   choices=["coco", "div2k", "vggface2"],
                   help="Training dataset. 'div2k' and 'vggface2' are fidelity-only "
                        "(zero seg loss + seg metrics dropped).")
    p.add_argument("--coco-root", type=str, default=COCO_DEFAULT_ROOT)
    p.add_argument("--div2k-root", type=str,
                   default="/media/arxwn/data0_4tb/datasets/div2k")
    p.add_argument("--vggface2-root", type=str,
                   default="/media/arxwn/data0_4tb/datasets/VGG-Face2/data")
    p.add_argument("--max-train-images", type=int, default=None)
    p.add_argument("--max-val-images", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=1)

    p.add_argument("--loss-recon", type=float, default=1.0)
    p.add_argument("--loss-seg", type=float, default=1.0)

    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable fp16 mixed precision (recommended on RTX 4090).")
    p.add_argument("--workers", type=int, default=4)

    # Periodic visualization. Either trigger > 0 enables the callback; both
    # 0 means no visualization.
    p.add_argument("--viz-every-steps", type=int, default=500,
                   help="Save a recon+seg grid every N optimizer steps. 0 disables.")
    p.add_argument("--viz-every-epochs", type=int, default=1,
                   help="Save a recon+seg grid every M epochs. 0 disables.")
    p.add_argument("--viz-num-samples", type=int, default=4,
                   help="Rows in the visualization grid (capped to val batch size).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="src/results/burst_dp/run")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    args = build_argparser().parse_args(argv)

    # Reproducibility
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # GPU
    if setup_gpu is not None:
        setup_gpu(gpu_id=args.gpu)

    # Mixed precision
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (fp16) enabled.")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Data ---
    if args.dataset == "coco":
        train_ds, val_ds = build_coco_burst_dp_datasets(
            coco_root=args.coco_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            seed=args.seed,
        )
        fidelity_only = False
    elif args.dataset == "div2k":
        train_ds, val_ds = build_div2k_burst_dp_datasets(
            div2k_root=args.div2k_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            seed=args.seed,
        )
        fidelity_only = True
    elif args.dataset == "vggface2":
        train_ds, val_ds = build_vggface2_burst_dp_datasets(
            vggface2_root=args.vggface2_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            seed=args.seed,
        )
        fidelity_only = True
    else:  # pragma: no cover — argparse choices guard this
        raise ValueError(f"Unknown --dataset: {args.dataset!r}")

    logger.info(f"Train probe: {train_ds.probe()}")

    # --- Model ---
    model = create_burst_dp(
        preset=args.preset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        n_max=args.n_max,
    )

    # --- Loss + optimizer ---
    losses: Dict[str, Any] = {
        "recon": charbonnier_loss(),
        "segmentation": sparse_seg_ce_loss(),
    }
    loss_weights: Dict[str, float] = {
        "recon": args.loss_recon,
        "segmentation": args.loss_seg,
    }

    steps_per_epoch = max(1, len(train_ds))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    cosine_steps = max(1, total_steps - warmup_steps)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=cosine_steps,
        warmup_target=args.lr,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
    )

    # Per-head metrics (see VISION_BENCHMARKS.md / METRICS.md):
    #   recon        -> PSNR + SSIM        (fidelity, METRICS.md §13)
    #   segmentation -> pixel-acc + mIoU   (METRICS.md §3; mIoU is the
    #                                       primary semseg metric).
    # MeanIoU(sparse_y_true=True, sparse_y_pred=False) takes argmax over
    # the logits' last axis internally, so passing the raw seg logits is
    # correct.
    metrics: Dict[str, Any] = {
        "recon": [PsnrMetric(max_val=1.0), SsimMetric(max_val=1.0)],
        "segmentation": [
            keras.metrics.SparseCategoricalAccuracy(name="pixel_acc"),
            keras.metrics.MeanIoU(
                num_classes=DEFAULT_NUM_SEG_CLASSES,
                sparse_y_true=True,
                sparse_y_pred=False,
                name="miou",
            ),
        ],
    }

    # DECISION plan_2026-05-19_64f2a17b/D-001
    # Fidelity-only datasets (DIV2K, VGG-Face2) lack segmentation labels.
    # The :class:`BurstDP` model is hardcoded dual-head; we keep the seg
    # head in the graph but zero its loss weight and drop its metrics.
    # Trade-off: ~5-15% wasted compute on the seg head at the cost of
    # zero blast radius on `BurstDP`, `BurstDPConfig`, and existing
    # checkpoints. Pivot fallback is Option B (add an `enable_segmentation`
    # config flag) — see plans/plan_2026-05-19_64f2a17b/decisions.md D-001.
    if fidelity_only:
        if args.loss_seg != 0.0:
            logger.warning(
                f"--dataset {args.dataset} is fidelity-only: overriding "
                f"--loss-seg {args.loss_seg} → 0.0 and dropping seg metrics."
            )
        loss_weights["segmentation"] = 0.0
        metrics.pop("segmentation", None)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # --- Callbacks ---
    # `TerminateOnNaN` MUST be first: the masked-softmax path in the fusion
    # block can produce NaN if the any-valid gate breaks, and we want training
    # to halt before any subsequent checkpoint commits a corrupted model.
    ckpt_path = str(out_dir / "burst_dp_best.keras")
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]
    if args.viz_every_steps > 0 or args.viz_every_epochs > 0:
        callbacks.append(
            BurstDPVisualizationCallback(
                val_dataset=val_ds,
                output_dir=str(out_dir),
                every_steps=args.viz_every_steps,
                every_epochs=args.viz_every_epochs,
                num_samples=args.viz_num_samples,
                seed=args.seed,
            )
        )
    callbacks.extend([
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tb")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ])

    # Persist run config
    run_cfg = {
        "args": vars(args),
        "model_config": model.config.to_dict(),
        "anchor_spec": asdict(default_anchor_spec()),
        "aux_spec": asdict(default_aux_spec()),
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "cosine_steps": cosine_steps,
        "fidelity_only": fidelity_only,
        "loss_weights": loss_weights,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # --- Fit ---
    # PyDataset passes directly to model.fit; Keras 3 iterates with the
    # configured `workers` + `use_multiprocessing` from the loader's
    # __init__ (see D-001 anchor at top of file).
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model.
    final_path = str(out_dir / "burst_dp_final.keras")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
