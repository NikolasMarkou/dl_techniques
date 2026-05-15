"""E4 — Deep-residual regression under mixed-fp16 (adversarial regime).

A 24-block residual stack ``[Dense(d) -> Norm -> GELU + residual]`` trained on
a synthetic polynomial regression task under **mixed_float16 + batch=16** —
deliberately adversarial: deep residual streams + low-precision + small batch
maximally stress (a) γ growth (mean shift accumulates over 24 layers),
(b) DC drift in fp16, and (c) gradient noise. This is where the
zero-centered and band variants should show maximal lift, if any.

Supports both OOB and PARAM_MATCHED modes.

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e4_deep_residual_reg \\
        --norm-type rms_norm --seed 0 --epochs 60 --out-dir results/e4
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Tuple

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
# Data
# ---------------------------------------------------------------------


def _make_dataset(
    n_train: int, n_val: int, d: int, seed: int, bias: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d,)).astype(np.float32)
    b = rng.normal(size=(d, d)).astype(np.float32) * 0.1

    def _gen(n: int) -> Tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, d)).astype(np.float32) + bias
        # Target combines a sine + quadratic + bilinear interaction.
        y = (
            np.sin(x @ a)
            + 0.1 * np.sum(x * (x @ b), axis=-1)
        ).astype(np.float32)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return x, y.reshape(n, 1)

    return _gen(n_train), _gen(n_val)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class _ResBlock(keras.layers.Layer):
    """Dense(d) -> Norm -> GELU + residual, dim-preserving."""

    def __init__(
        self,
        d: int,
        norm_type: str,
        norm_kwargs: dict,
        block_idx: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d = d
        self.norm_type = norm_type
        self.norm_kwargs = dict(norm_kwargs)
        self.block_idx = block_idx
        self.dense = keras.layers.Dense(d, name=f"dense_{block_idx}")
        self.norm = create_normalization_layer(
            norm_type, name=f"norm_{block_idx}", **norm_kwargs
        )
        self.act = keras.layers.Activation("gelu", name=f"act_{block_idx}")

    def call(self, inputs, training=None):
        h = self.dense(inputs)
        h = self.norm(h, training=training)
        h = self.act(h)
        return inputs + h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d": self.d, "norm_type": self.norm_type,
                    "norm_kwargs": self.norm_kwargs, "block_idx": self.block_idx})
        return cfg


def _build_model(
    *, d: int, depth: int, norm_type: str, norm_kwargs: dict
) -> keras.Model:
    inputs = keras.Input(shape=(d,), name="x", dtype="float32")
    h = keras.layers.Dense(d, name="proj_in")(inputs)
    for k in range(depth):
        h = _ResBlock(
            d=d, norm_type=norm_type, norm_kwargs=norm_kwargs, block_idx=k,
            name=f"resblock_{k}",
        )(h)
    h = create_normalization_layer(norm_type, name="final_norm", **norm_kwargs)(h)
    # Force fp32 head for numerical safety under mixed_float16.
    outputs = keras.layers.Dense(1, name="head", dtype="float32")(h)
    return keras.Model(inputs, outputs, name=f"e4_deepres_{norm_type}")


# ---------------------------------------------------------------------
# Args / run
# ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E4 DeepResidual+fp16 regression")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob", "param_matched"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16, help="Small batch is adversarial")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--bias", type=float, default=2.0)
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-val", type=int, default=1024)
    p.add_argument("--mixed-precision", action="store_true", default=True,
                   help="On by default — this IS the adversarial regime.")
    p.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    p.add_argument("--max-band-width", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def run(cfg: ExperimentConfig, *, depth: int, hidden_dim: int, bias: float,
        n_train: int, n_val: int, mixed_precision: bool) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    original_policy = keras.mixed_precision.global_policy()
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("[e4] mixed_float16 policy active")

    try:
        (x_tr, y_tr), (x_val, y_val) = _make_dataset(
            n_train=n_train, n_val=n_val, d=hidden_dim, seed=cfg.seed, bias=bias,
        )
        logger.info(
            f"[e4] data: train={x_tr.shape}, val={x_val.shape}, "
            f"train_mean={x_tr.mean():.3f}"
        )

        norm_kwargs = cfg.norm_kwargs()
        model = _build_model(
            d=hidden_dim, depth=depth,
            norm_type=cfg.norm_type, norm_kwargs=norm_kwargs,
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        if mixed_precision:
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
        logger.info(
            f"[e4] model: depth={depth}, d={hidden_dim}, norm={cfg.norm_type}, "
            f"mode={cfg.mode}, params={n_params}"
        )

        cal_x = tf.convert_to_tensor(x_val[:64])
        cal_y = tf.convert_to_tensor(y_val[:64])
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
            x_tr, y_tr,
            validation_data=(x_val, y_val),
            epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=2,
            callbacks=callbacks_,
        )
        wall_s = time.time() - t0

        final_loss = float(history.history["loss"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        final_mae = float(history.history.get("mae", [float("nan")])[-1])
        final_val_mae = float(history.history.get("val_mae", [float("nan")])[-1])

        results_csv = os.path.join(cfg.out_dir, "results.csv")
        write_header = not os.path.exists(results_csv)
        with open(results_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "experiment", "norm_type", "mode", "seed", "epochs",
                    "trainable_params",
                    "final_loss", "final_val_loss",
                    "final_mae", "final_val_mae",
                    "mixed_precision", "wall_s",
                ])
            w.writerow([
                "e4", cfg.norm_type, cfg.mode, cfg.seed, cfg.epochs,
                n_params,
                final_loss, final_val_loss,
                final_mae, final_val_mae,
                int(mixed_precision), wall_s,
            ])
        logger.info(
            f"[e4] DONE: val_loss={final_val_loss:.4f}, val_mae={final_val_mae:.4f}, "
            f"wall_s={wall_s:.1f}"
        )
        return {
            "final_val_loss": final_val_loss,
            "final_val_mae": final_val_mae,
            "wall_s": wall_s,
        }
    finally:
        keras.mixed_precision.set_global_policy(original_policy)


def main() -> None:
    args = _parse_args()
    cfg = ExperimentConfig(
        experiment_name="e4",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        dataset="synthetic_deepres",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        max_band_width=args.max_band_width,
        out_dir=args.out_dir,
    )
    run(
        cfg,
        depth=args.depth, hidden_dim=args.hidden_dim, bias=args.bias,
        n_train=args.n_train, n_val=args.n_val,
        mixed_precision=args.mixed_precision,
    )


if __name__ == "__main__":
    main()
