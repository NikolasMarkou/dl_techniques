"""E5 — Norm-layer microbenchmark.

A controlled synthetic Gaussian regression task used as the LAYER-LEVEL
SANITY check: K=16 stack of ``Dense(d) -> Norm -> GELU`` (no residuals)
trained on the polynomial target

    y = sin(x @ a) + 0.1 * (x @ b) ** 2

with the input deliberately biased by a constant offset, so that:
- ``RMSNorm`` baseline shows mean-shift in the residual stream,
- ``ZeroCenteredRMSNorm`` removes it,
- ``BandRMS`` shows tight per-sample RMS,
- ``ZeroCenteredBandRMSNorm`` shows both.

The expected qualitative differences in the per-epoch probe CSVs are the
**STOP-IF gate** from plan.md Pre-Mortem scenario B — if E5 doesn't
discriminate the variants here, the callbacks are broken and we must fix
them before scaling to E1-E4.

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e5_norm_layer_microbench \\
        --norm-type rms_norm --seed 0 --epochs 5 --out-dir /tmp/e5

Output:
    {out_dir}/results.csv         single-row summary (final loss + run-id)
    {out_dir}/grad_norm.csv       per-epoch gradient norm
    {out_dir}/weight_norm.csv     per-epoch weight L2 per norm layer
    {out_dir}/activation_stats.csv per-epoch activation stats per norm layer
    {out_dir}/norm_internal.csv   per-epoch internal stats per norm layer
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

from dl_techniques.layers.norms.factory import create_normalization_layer
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
# Data: biased Gaussian regression
# ---------------------------------------------------------------------


def _make_synthetic_dataset(
    n_train: int = 4096,
    n_val: int = 512,
    d: int = 64,
    seed: int = 0,
    bias: float = 3.0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d,)).astype(np.float32)
    b = rng.normal(size=(d,)).astype(np.float32)

    def _gen(n: int) -> Tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d)).astype(np.float32) + bias
        # Target: sin + small quadratic, scaled to roughly unit variance.
        y = (np.sin(x @ a) + 0.1 * (x @ b) ** 2).astype(np.float32)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return x, y.reshape(n, 1)

    return _gen(n_train), _gen(n_val)


# ---------------------------------------------------------------------
# Model: K=16 stack of Dense+Norm+GELU
# ---------------------------------------------------------------------


def _build_model(d: int, depth: int, norm_type: str, norm_kwargs: dict) -> keras.Model:
    inputs = keras.Input(shape=(d,), name="x")
    h = inputs
    for k in range(depth):
        h = keras.layers.Dense(d, name=f"dense_{k}")(h)
        h = create_normalization_layer(norm_type, name=f"norm_{k}", **norm_kwargs)(h)
        h = keras.layers.Activation("gelu", name=f"act_{k}")(h)
    outputs = keras.layers.Dense(1, name="head")(h)
    return keras.Model(inputs, outputs, name=f"e5_microbench_{norm_type}")


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------


# Regime axis (Phase 3). Maps to ``(lr, batch, mp, depth_override)``.
_REGIME_MAP: Dict[str, Tuple[Optional[float], Optional[int], Optional[bool], Optional[int]]] = {
    "default": (None, None, None, None),
    "bs_32":   (None, 32, None, None),
    "bs_256":  (None, 256, None, None),
    "lr_low":  (1e-4, None, None, None),
    "lr_high": (1e-3, None, None, None),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5 norm-layer microbench")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob", "param_matched"])
    p.add_argument("--regime", type=str, default="default",
                   choices=list(_REGIME_MAP.keys()),
                   help="Phase 3 regime sub-experiment selector (LR/BS axis).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-band-width", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--bias", type=float, default=3.0, help="Input mean offset")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def run(cfg: ExperimentConfig, *, depth: int, hidden_dim: int, bias: float) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    (x_tr, y_tr), (x_val, y_val) = _make_synthetic_dataset(
        d=hidden_dim, seed=cfg.seed, bias=bias
    )
    logger.info(
        f"[e5] dataset: train={x_tr.shape}, val={x_val.shape}, "
        f"train_mean={x_tr.mean():.3f}, train_std={x_tr.std():.3f}"
    )

    model = _build_model(
        d=hidden_dim, depth=depth, norm_type=cfg.norm_type, norm_kwargs=cfg.norm_kwargs()
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    logger.info(
        f"[e5] model: depth={depth}, d={hidden_dim}, norm={cfg.norm_type}, "
        f"mode={cfg.mode}, trainable_params={n_params}"
    )

    # Calibration batch for activation + gradient probes
    cal_x = tf.convert_to_tensor(x_val[:32])
    cal_y = tf.convert_to_tensor(y_val[:32])

    callbacks_ = [
        GradientNormCallback(
            calibration_data=(cal_x, cal_y), out_dir=cfg.out_dir,
            loss_fn=keras.losses.MeanSquaredError(),
        ),
        WeightNormTrajectoryCallback(out_dir=cfg.out_dir),
        NormLayerActivationCallback(calibration_data=cal_x, out_dir=cfg.out_dir),
        NormInternalStatsCallback(out_dir=cfg.out_dir),
    ]

    t0 = time.time()
    history = model.fit(
        x_tr,
        y_tr,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=2,
        callbacks=callbacks_,
    )
    wall_s = time.time() - t0

    # Final-epoch summary row
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    final_mae = float(history.history.get("mae", [float("nan")])[-1])
    final_val_mae = float(history.history.get("val_mae", [float("nan")])[-1])
    # Regression generalization gap: val_loss - train_loss
    # (positive = worse generalization). plan_e1f12eab Step 3 / EC1.
    try:
        generalization_gap = final_val_loss - final_loss
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
            w.writerow(
                [
                    "experiment",
                    "norm_type",
                    "mode",
                    "seed",
                    "regime",
                    "epochs",
                    "trainable_params",
                    "final_loss",
                    "final_val_loss",
                    "final_mae",
                    "final_val_mae",
                    "generalization_gap",
                    "wall_s",
                ]
            )
        w.writerow(
            [
                "e5",
                cfg.norm_type,
                cfg.mode,
                cfg.seed,
                cfg.extras.get("regime", "default"),
                cfg.epochs,
                n_params,
                final_loss,
                final_val_loss,
                final_mae,
                final_val_mae,
                generalization_gap,
                wall_s,
            ]
        )
    logger.info(
        f"[e5] DONE: val_loss={final_val_loss:.4f}, val_mae={final_val_mae:.4f}, "
        f"wall_s={wall_s:.1f}"
    )
    return {
        "final_val_loss": final_val_loss,
        "final_val_mae": final_val_mae,
        "wall_s": wall_s,
    }


def main() -> None:
    args = _parse_args()
    # Apply regime override (Phase 3 LR/BS axis).
    lr_o, bs_o, _, _ = _REGIME_MAP[args.regime]
    if lr_o is not None:
        args.learning_rate = lr_o
    if bs_o is not None:
        args.batch_size = bs_o
    cfg = ExperimentConfig(
        experiment_name="e5",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_band_width=args.max_band_width,
        epsilon=args.epsilon,
        out_dir=args.out_dir,
        extras={"regime": args.regime},
    )
    run(cfg, depth=args.depth, hidden_dim=args.hidden_dim, bias=args.bias)


if __name__ == "__main__":
    main()
