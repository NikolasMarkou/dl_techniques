"""E1 — ViT-pico × CIFAR-10 trainer.

Vision classification benchmark using ViT-pico (192-dim, 6 layers, 3 heads).
The `--norm-type` CLI flag swaps the normalization layer across the entire
ViT stack (every TransformerLayer attention/FFN norm + the final norm before
the head).

**Mode constraint**: per plan_2026-05-14_3764496e D-003, ViT's factory does
not forward `normalization_kwargs`, so this experiment runs in **OOB mode
only**. ZeroCenteredRMSNorm and RMSNorm both get d-many params from their
default `use_scale=True`. The 1-vs-d-param confound is documented; the
PARAM_MATCHED contrast is preserved on E3/E4/E5.

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e1_vit_cifar10 \\
        --norm-type rms_norm --seed 0 --epochs 50 --out-dir results/e1
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vit.model import create_vit
from dl_techniques.utils.logger import logger

from train.rms_variants_train.callbacks import (
    GradientNormCallback,
    NormInternalStatsCallback,
    NormLayerActivationCallback,
    WeightNormTrajectoryCallback,
)
from train.rms_variants_train.config import ExperimentConfig, NORM_VARIANTS
from train.rms_variants_train.seed_utils import set_seeds


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def _load_cifar10(seed: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 as float32 in [0, 1], one-hot labels."""
    (x_tr, y_tr), (x_val, y_val) = keras.datasets.cifar10.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0
    y_tr = y_tr.flatten().astype(np.int64)
    y_val = y_val.flatten().astype(np.int64)
    return (x_tr, y_tr), (x_val, y_val)


def _to_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------


# Regime axis (Phase 3). Each entry maps to a 4-tuple
# ``(lr, batch, mp, depth_override)`` where ``None`` means "do not override".
# ``mp`` is a bool; ``depth_override`` is ignored on E1 (depth fixed by variant).
_REGIME_MAP: Dict[str, Tuple[Optional[float], Optional[int], Optional[bool], Optional[int]]] = {
    "default": (None, None, None, None),
    "lr_low":  (1e-4, None, None, None),
    "lr_high": (1e-3, None, None, None),
    "mp_fp16": (None, None, True, None),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 ViT-pico × CIFAR-10")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob"],
                   help="ViT factory does not plumb norm_kwargs (D-003).")
    p.add_argument("--regime", type=str, default="default",
                   choices=list(_REGIME_MAP.keys()),
                   help="Phase 3 regime sub-experiment selector.")
    p.add_argument("--variant", type=str, default="vit_pico")
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------


def run(cfg: ExperimentConfig, *, variant: str, patch_size: int, dropout_rate: float,
        warmup_epochs: int) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    (x_tr, y_tr), (x_val, y_val) = _load_cifar10(cfg.seed)
    logger.info(f"[e1] CIFAR-10 train={x_tr.shape}, val={x_val.shape}")

    train_ds = _to_tf_dataset(x_tr, y_tr, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_ds = _to_tf_dataset(x_val, y_val, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed)

    model = create_vit(
        variant=variant,
        num_classes=10,
        input_shape=(32, 32, 3),
        patch_size=patch_size,
        include_top=True,
        dropout_rate=dropout_rate,
        attention_dropout_rate=0.0,
        pos_dropout_rate=0.0,
        normalization_type=cfg.norm_type,
        normalization_position="post",
        ffn_type="mlp",
    )

    # AdamW with cosine LR schedule
    steps_per_epoch = max(1, len(x_tr) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.learning_rate,
        decay_steps=max(1, total_steps - warmup_steps),
        warmup_target=cfg.learning_rate,
        warmup_steps=warmup_steps,
        alpha=1e-2,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=cfg.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    logger.info(f"[e1] model: variant={variant}, norm={cfg.norm_type}, params={n_params}")

    # Calibration batch (use first val batch deterministically)
    cal_x = tf.convert_to_tensor(x_val[:32])
    cal_y = tf.convert_to_tensor(y_val[:32])

    callbacks_ = [
        GradientNormCallback(
            calibration_data=(cal_x, cal_y), out_dir=cfg.out_dir,
            loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ),
        WeightNormTrajectoryCallback(out_dir=cfg.out_dir),
        NormLayerActivationCallback(calibration_data=cal_x, out_dir=cfg.out_dir),
        NormInternalStatsCallback(out_dir=cfg.out_dir),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg.epochs, verbose=2, callbacks=callbacks_,
    )
    wall_s = time.time() - t0

    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    best_val_acc = float(max(history.history["val_accuracy"]))

    results_csv = os.path.join(cfg.out_dir, "results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "experiment", "norm_type", "mode", "seed", "epochs",
                "trainable_params",
                "final_loss", "final_val_loss",
                "final_acc", "final_val_acc", "best_val_acc",
                "wall_s",
            ])
        w.writerow([
            "e1", cfg.norm_type, cfg.mode, cfg.seed, cfg.epochs,
            n_params,
            final_loss, final_val_loss,
            final_acc, final_val_acc, best_val_acc,
            wall_s,
        ])

    logger.info(
        f"[e1] DONE: val_acc={final_val_acc:.4f} (best={best_val_acc:.4f}), "
        f"wall_s={wall_s:.1f}"
    )
    return {"val_acc": final_val_acc, "best_val_acc": best_val_acc, "wall_s": wall_s}


def main() -> None:
    args = _parse_args()
    # Apply regime override (Phase 3 sub-experiment axis).
    lr_o, bs_o, mp_o, _ = _REGIME_MAP[args.regime]
    if lr_o is not None:
        args.learning_rate = lr_o
    if bs_o is not None:
        args.batch_size = bs_o
    if mp_o is True:
        keras.mixed_precision.set_global_policy("mixed_float16")
    cfg = ExperimentConfig(
        experiment_name="e1",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        model_variant=args.variant,
        dataset="cifar10",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mixed_precision=bool(mp_o),
        out_dir=args.out_dir,
        extras={"regime": args.regime},
    )
    run(
        cfg,
        variant=args.variant,
        patch_size=args.patch_size,
        dropout_rate=args.dropout_rate,
        warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
