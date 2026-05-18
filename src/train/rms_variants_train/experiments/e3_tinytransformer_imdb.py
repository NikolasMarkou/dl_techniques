"""E3 — TinyTransformer × IMDb binary sentiment classification.

A small 4-layer transformer encoder (d=128, 4 heads) trained on the IMDb
binary sentiment dataset (Keras built-in, vocab=20000, seq_len=128). The
encoder is built directly from ``dl_techniques.layers.transformers.
transformer.TransformerLayer`` so we can plumb full norm kwargs via
``attention_norm_args`` and ``ffn_norm_args``.

This experiment **supports both OOB and PARAM_MATCHED modes** (E3 + E4 + E5
form the param-matched contrast).

Run:
    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e3_tinytransformer_imdb \\
        --norm-type zero_centered_rms_norm --seed 0 --mode param_matched \\
        --epochs 20 --out-dir results/e3
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
from dl_techniques.layers.transformers.transformer import TransformerLayer
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


def _load_imdb(
    vocab_size: int, max_len: int, seed: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_tr, y_tr), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=vocab_size, seed=seed,
    )
    # Pad/truncate to max_len. Token 0 is reserved as pad.
    x_tr = keras.preprocessing.sequence.pad_sequences(
        x_tr, maxlen=max_len, padding="post", truncating="post"
    ).astype(np.int32)
    x_val = keras.preprocessing.sequence.pad_sequences(
        x_val, maxlen=max_len, padding="post", truncating="post"
    ).astype(np.int32)
    return (x_tr, y_tr.astype(np.int32)), (x_val, y_val.astype(np.int32))


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def _build_model(
    *,
    vocab_size: int,
    max_len: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    intermediate_size: int,
    norm_type: str,
    norm_kwargs: dict,
    dropout_rate: float,
) -> keras.Model:
    inputs = keras.Input(shape=(max_len,), dtype="int32", name="tokens")
    x = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=d_model, mask_zero=True, name="embedding"
    )(inputs)
    # Sinusoidal-style learned position embedding (simple).
    pos = keras.layers.Embedding(
        input_dim=max_len, output_dim=d_model, name="pos_embedding"
    )(keras.ops.arange(max_len))
    x = x + pos
    x = keras.layers.Dropout(dropout_rate, name="emb_drop")(x)
    for i in range(num_layers):
        x = TransformerLayer(
            hidden_size=d_model,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            attention_type="multi_head",
            normalization_type=norm_type,
            normalization_position="pre",
            attention_norm_args=norm_kwargs,
            ffn_norm_args=norm_kwargs,
            ffn_type="mlp",
            dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            name=f"transformer_{i}",
        )(x)
    x = create_normalization_layer(norm_type, name="final_norm", **norm_kwargs)(x)
    # Mean-pool over time (ignoring zero-pad positions by simple mean).
    x = keras.layers.GlobalAveragePooling1D(name="pool")(x)
    outputs = keras.layers.Dense(1, activation=None, name="head")(x)
    return keras.Model(inputs, outputs, name=f"e3_tinytransformer_{norm_type}")


# ---------------------------------------------------------------------
# Args / run
# ---------------------------------------------------------------------


# Regime axis (Phase 3). Maps to ``(lr, batch, mp, depth_override)``.
_REGIME_MAP: Dict[str, Tuple[Optional[float], Optional[int], Optional[bool], Optional[int]]] = {
    "default": (None, None, None, None),
    "mp_fp16": (None, None, True, None),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E3 TinyTransformer × IMDb")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob", "param_matched"])
    p.add_argument("--regime", type=str, default="default",
                   choices=list(_REGIME_MAP.keys()),
                   help="Phase 3 regime sub-experiment selector.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument("--max-band-width", type=float, default=0.1)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def run(cfg: ExperimentConfig, *, vocab_size: int, max_len: int, d_model: int,
        num_heads: int, num_layers: int, dropout_rate: float, warmup_epochs: int) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    (x_tr, y_tr), (x_val, y_val) = _load_imdb(vocab_size, max_len, cfg.seed)
    logger.info(f"[e3] IMDb train={x_tr.shape}, val={x_val.shape}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(buffer_size=10000, seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    )

    norm_kwargs = cfg.norm_kwargs()
    model = _build_model(
        vocab_size=vocab_size, max_len=max_len, d_model=d_model,
        num_heads=num_heads, num_layers=num_layers,
        intermediate_size=4 * d_model,
        norm_type=cfg.norm_type, norm_kwargs=norm_kwargs,
        dropout_rate=dropout_rate,
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
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)],
    )
    n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    logger.info(f"[e3] model: norm={cfg.norm_type}, mode={cfg.mode}, params={n_params}")

    cal_x = tf.convert_to_tensor(x_val[:32])
    cal_y = tf.convert_to_tensor(y_val[:32])
    callbacks_ = [
        GradientNormCallback(
            calibration_data=(cal_x, cal_y), out_dir=cfg.out_dir,
            loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
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

    final_acc = float(history.history["accuracy"][-1])
    final_val_acc = float(history.history["val_accuracy"][-1])
    best_val_acc = float(max(history.history["val_accuracy"]))
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

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
                "wall_s",
            ])
        w.writerow([
            "e3", cfg.norm_type, cfg.mode, cfg.seed,
            cfg.extras.get("regime", "default"), cfg.epochs,
            n_params,
            final_loss, final_val_loss,
            final_acc, final_val_acc, best_val_acc,
            wall_s,
        ])
    logger.info(
        f"[e3] DONE: val_acc={final_val_acc:.4f} (best={best_val_acc:.4f}), "
        f"wall_s={wall_s:.1f}"
    )
    return {"val_acc": final_val_acc, "best_val_acc": best_val_acc, "wall_s": wall_s}


def main() -> None:
    import keras as _keras  # local — avoid TF policy side-effects at import time
    args = _parse_args()
    # Apply regime override (Phase 3).
    lr_o, bs_o, mp_o, _ = _REGIME_MAP[args.regime]
    if lr_o is not None:
        args.learning_rate = lr_o
    if bs_o is not None:
        args.batch_size = bs_o
    if mp_o is True:
        _keras.mixed_precision.set_global_policy("mixed_float16")
    cfg = ExperimentConfig(
        experiment_name="e3",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        dataset="imdb",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mixed_precision=bool(mp_o),
        max_band_width=args.max_band_width,
        out_dir=args.out_dir,
        extras={"regime": args.regime},
    )
    run(
        cfg,
        vocab_size=args.vocab_size, max_len=args.max_len,
        d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate, warmup_epochs=args.warmup_epochs,
    )


if __name__ == "__main__":
    main()
