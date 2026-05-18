"""E2 — ResNet-18 × CIFAR-100 trainer.

Image classification on a deeper conv stack. The `--norm-type` flag swaps
the BatchNormalization layers (used in every conv block + the stem) for one
of the four RMSNorm-family variants via ResNet's `normalization_type` kwarg.

**Mode constraint** (superseded by plan_2026-05-18_6776f8ba D-003):
ResNet's factory NOW forwards `normalization_kwargs` (Step 4 of the
plan_2026-05-18_6776f8ba refinement), so `--mode param_matched` is
supported here. In `param_matched` mode the trainer threads
`normalization_kwargs={'use_scale': False}` to the stem norm AND to
every BasicBlock/BottleneckBlock internal norm (bn1/bn2/bn3 +
shortcut_bn), eliminating their per-channel scale parameters.

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e2_resnet_cifar100 \\
        --norm-type rms_norm --seed 0 --epochs 80 --out-dir results/e2
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

from dl_techniques.models.resnet.model import create_resnet
from dl_techniques.utils.logger import logger

from train.rms_variants_train.callbacks import (
    DistributionShiftProbe,
    GradientNormCallback,
    NormInternalStatsCallback,
    NormLayerActivationCallback,
    WeightNormTrajectoryCallback,
)
from train.rms_variants_train.config import ExperimentConfig, NORM_VARIANTS
from train.rms_variants_train.seed_utils import set_seeds


def _load_cifar100(seed: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_tr, y_tr), (x_val, y_val) = keras.datasets.cifar100.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0
    y_tr = y_tr.flatten().astype(np.int64)
    y_val = y_val.flatten().astype(np.int64)
    return (x_tr, y_tr), (x_val, y_val)


def _to_tf_dataset(x, y, *, batch_size, shuffle, seed) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Regime axis (Phase 3). E2 only accepts the ``default`` choice — it is NOT
# in the Phase 3 regime sweep (kept for CLI uniformity across all 5 trainers).
_REGIME_MAP: Dict[str, Tuple[Optional[float], Optional[int], Optional[bool], Optional[int]]] = {
    "default": (None, None, None, None),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2 ResNet-18 × CIFAR-100")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob", "param_matched"],
                   help="oob: default ResNet. param_matched: thread "
                        "use_scale=False through ResNet into stem + every "
                        "BasicBlock/BottleneckBlock norm "
                        "(plan_2026-05-18_6776f8ba D-003).")
    p.add_argument("--regime", type=str, default="default",
                   choices=list(_REGIME_MAP.keys()),
                   help="Phase 3 stub — E2 only supports 'default'.")
    p.add_argument("--variant", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def run(cfg: ExperimentConfig, *, variant: str, warmup_epochs: int) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    (x_tr, y_tr), (x_val, y_val) = _load_cifar100(cfg.seed)
    logger.info(f"[e2] CIFAR-100 train={x_tr.shape}, val={x_val.shape}")

    train_ds = _to_tf_dataset(x_tr, y_tr, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_ds = _to_tf_dataset(x_val, y_val, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed)

    # DECISION plan_2026-05-18_6776f8ba/D-003 (Step 5)
    # `cfg.norm_kwargs()` resolves to `{'use_scale': False, ...}` only
    # for RMSNorm-family variants in param_matched mode.
    model = create_resnet(
        variant=variant,
        num_classes=100,
        input_shape=(32, 32, 3),
        normalization_type=cfg.norm_type,
        normalization_kwargs=dict(cfg.norm_kwargs()),
    )

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
    logger.info(f"[e2] model: variant={variant}, norm={cfg.norm_type}, params={n_params}")

    cal_x = tf.convert_to_tensor(x_val[:32])
    cal_y = tf.convert_to_tensor(y_val[:32])

    # CIFAR-100 train pipeline normalizes images by /255.0 — apply the same
    # to the CIFAR-100-C tensors the probe will pull from TFDS so
    # model.predict sees in-distribution magnitudes (mirrors E1 wiring).
    def _cifar100c_preprocess(x: np.ndarray, y: np.ndarray):
        return x.astype(np.float32) / 255.0, y

    callbacks_ = [
        GradientNormCallback(
            calibration_data=(cal_x, cal_y), out_dir=cfg.out_dir,
            loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ),
        WeightNormTrajectoryCallback(out_dir=cfg.out_dir),
        NormLayerActivationCallback(calibration_data=cal_x, out_dir=cfg.out_dir),
        NormInternalStatsCallback(out_dir=cfg.out_dir),
        # CIFAR-100-C distribution-shift probe
        # (plan_2026-05-18_6776f8ba step 7). Pre-condition check at PLAN
        # time on 2026-05-18 showed `cifar100_corrupted` is NOT registered
        # in the installed `tensorflow_datasets` namespace (Falsification
        # Scenario 4 in plan.md fires). Wiring the probe anyway because
        # the existing soft-fail branch (plan_e1f12eab D-002, callbacks.py
        # line 874-879) will emit a populated `dist_shift.csv` with
        # `reason='dataset_missing:DatasetNotFoundError'` rows per
        # corruption — informational signal preserved, cell never poisoned.
        # If the user later installs a TFDS build with CIFAR-100-C, this
        # wire activates automatically without further code changes.
        DistributionShiftProbe(
            out_dir=cfg.out_dir,
            dataset_name_template="cifar100_corrupted/{corruption}_{severity}",
            severity=3,
            max_samples_per_corruption=500,
            preprocess_fn=_cifar100c_preprocess,
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg.epochs, verbose=2, callbacks=callbacks_,
    )
    wall_s = time.time() - t0

    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    best_val_acc = float(max(history.history["val_accuracy"]))
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    # Classification generalization gap (plan_e1f12eab Step 3; NaN-tolerant per EC1).
    try:
        generalization_gap = final_acc - final_val_acc
    except (TypeError, ValueError):
        generalization_gap = float("nan")

    # Per-epoch history.csv — consumed by report.py post-hoc derivations.
    hist_csv = os.path.join(cfg.out_dir, "history.csv")
    _hist = history.history
    _hist_keys = list(_hist.keys())
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + _hist_keys)
        for i in range(len(_hist[_hist_keys[0]])):
            w.writerow([i] + [_hist[k][i] for k in _hist_keys])

    results_csv = os.path.join(cfg.out_dir, "results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "experiment", "norm_type", "mode", "seed", "regime", "epochs",
                "trainable_params",
                "final_loss", "final_val_loss",
                "final_acc", "final_val_acc", "best_val_acc",
                "generalization_gap",
                "wall_s",
            ])
        w.writerow([
            "e2", cfg.norm_type, cfg.mode, cfg.seed,
            cfg.extras.get("regime", "default"), cfg.epochs,
            n_params,
            final_loss, final_val_loss,
            final_acc, final_val_acc, best_val_acc,
            generalization_gap,
            wall_s,
        ])

    logger.info(
        f"[e2] DONE: val_acc={final_val_acc:.4f} (best={best_val_acc:.4f}), "
        f"wall_s={wall_s:.1f}"
    )
    return {"val_acc": final_val_acc, "best_val_acc": best_val_acc, "wall_s": wall_s}


def main() -> None:
    args = _parse_args()
    # Apply regime override (Phase 3). E2 only supports 'default' — no-op.
    _ = _REGIME_MAP[args.regime]
    cfg = ExperimentConfig(
        experiment_name="e2",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        model_variant=args.variant,
        dataset="cifar100",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        out_dir=args.out_dir,
        extras={"regime": args.regime},
    )
    run(cfg, variant=args.variant, warmup_epochs=args.warmup_epochs)


if __name__ == "__main__":
    main()
