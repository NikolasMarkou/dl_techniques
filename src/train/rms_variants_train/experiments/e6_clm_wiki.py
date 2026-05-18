"""E6 — 4-layer causal-LM × Wikipedia 10k (RMS-norm variants).

Pattern-3 NLP trainer (see ``src/train/CLAUDE.md``). Mirrors the
``gpt2/pretrain.py`` shape but builds a *very* small 4-layer / d=192 / 4-head
causal transformer directly from
``dl_techniques.layers.transformers.transformer.TransformerLayer`` so that the
norm variant under test is plumbed via ``attention_norm_args`` /
``ffn_norm_args`` exactly as in E3.

Headline metric: ``final_val_perplexity`` (taken from the keras ``Perplexity``
metric assembled by ``build_clm_metrics``).

Data path
---------
* Default: Wikipedia 10k articles via
  ``dl_techniques.datasets.nlp.load_wikipedia_train_val`` +
  ``train.common.nlp.preprocess_clm_packed_dataset`` (tiktoken cl100k_base,
  EOT = ``DEFAULT_SEP_TOKEN_ID``, packed concat-and-chunk pipeline).
* CPU smoke fast-path (``--synthetic-smoke``): generates a tiny synthetic
  packed corpus in-process — used by ``test_e6_clm.py`` to verify build +
  1-step fit + non-NaN loss without any HF download. **Never used in real
  sweeps.**

Run (full)::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e6_clm_wiki \\
        --norm-type rms_norm --seed 0 --mode oob \\
        --epochs 4 --out-dir results/e6

Run (CPU 1-step smoke)::

    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m \\
        train.rms_variants_train.experiments.e6_clm_wiki \\
        --norm-type rms_norm --seed 0 --mode oob \\
        --epochs 1 --steps-per-epoch 1 --synthetic-smoke --out-dir /tmp/e6smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Dict, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

# DECISION plan_2026-05-18_6776f8ba/D-004: norm choice in a causal-LM stack
# matters most where the residual stream is read or written — block-input
# (pre-attn norm), block-output (pre-FFN norm), and final-pre-logits norm.
# We therefore plumb the *same* `(norm_type, norm_kwargs)` pair into all three
# positions via the existing `TransformerLayer(attention_norm_args=...,
# ffn_norm_args=...)` interface PLUS one explicit `create_normalization_layer`
# call at the head (final-pre-logits). Any divergence in placement here breaks
# the contract that "norm choice is the single experimental variable."
# Reverting this site is the smallest revert that restores the no-norm-choice
# baseline; do not refactor the three call-sites into a single helper without
# preserving the three-position invariant in plan.md F2.
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.losses import MaskedCausalLMLoss
from dl_techniques.utils.logger import logger

from train.common.nlp import (
    DEFAULT_SEP_TOKEN_ID,
    build_clm_metrics,
    create_warmup_lr_schedule,
    estimate_clm_steps_per_epoch,
    prepare_dict_keyed_compile,
    preprocess_clm_packed_dataset,
)
from train.rms_variants_train.config import ExperimentConfig, NORM_VARIANTS
from train.rms_variants_train.seed_utils import set_seeds


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# cl100k_base vocab size (tiktoken). Used as the dense head's `units`.
_VOCAB_SIZE = 100277
# EOT token appended between articles in the packed pipeline. We reuse the
# campaign-canonical SEP id (100265) — packed CLM treats this as just another
# token, no special masking applied (LESSONS — pad_token_id/eot bookkeeping
# is the canonical silent CLM bug).
_EOT_TOKEN_ID = DEFAULT_SEP_TOKEN_ID

# Default Wikipedia subset for the rms-variants campaign (10k articles is
# small enough to fit a 4-epoch run inside a ~1h GPU budget per cell, large
# enough to produce a stable val PPL signal at this model scale).
WIKI_ARTICLES_DEFAULT = 10_000


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="rms_variants_train")
class TinyCLM(keras.Model):
    """4-layer / d=192 / 4-head causal transformer with dict-keyed output.

    The model returns ``{"logits": <B, T, V>}`` so the trainer can use the
    standard CLM compile shape (``MaskedCausalLMLoss`` under ``loss={"logits":
    ...}``, metrics under ``metrics={"logits": build_clm_metrics(...)}``).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        intermediate_size: int,
        norm_type: str,
        norm_kwargs: dict,
        dropout_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.intermediate_size = int(intermediate_size)
        self.norm_type = norm_type
        self.norm_kwargs = dict(norm_kwargs)
        self.dropout_rate = float(dropout_rate)

        self.tok_emb = keras.layers.Embedding(
            self.vocab_size, self.d_model, name="tok_emb"
        )
        self.pos_emb = keras.layers.Embedding(
            self.max_seq_len, self.d_model, name="pos_emb"
        )
        self.emb_drop = keras.layers.Dropout(self.dropout_rate, name="emb_drop")
        # D-004 norm plumbing: per-block attn + ffn norms (Pattern-3 path).
        self.blocks = [
            TransformerLayer(
                hidden_size=self.d_model,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head",
                normalization_type=self.norm_type,
                normalization_position="pre",
                attention_norm_args=self.norm_kwargs,
                ffn_norm_args=self.norm_kwargs,
                ffn_type="mlp",
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.dropout_rate,
                name=f"transformer_{i}",
            )
            for i in range(self.num_layers)
        ]
        # D-004 norm plumbing: final pre-logits norm (3rd norm position).
        self.final_norm = create_normalization_layer(
            self.norm_type, name="final_norm", **self.norm_kwargs
        )
        self.head = keras.layers.Dense(
            self.vocab_size, use_bias=False, name="head_logits"
        )
        # Re-usable causal artifacts (pre-built per max_seq_len). The
        # attention mask uses float dtype (`_apply_attention_mask` multiplies
        # `(1 - mask) * -inf` — casting from bool inside a tf.function trips
        # `pred must not be a Python bool` on TF 2.18). Shape `(1, T, T)` is
        # broadcastable across the batch dim; MHCA reads it as `(batch,
        # q_seq, kv_seq)` and expands to `(batch, 1, q, k)` internally.
        self._positions = keras.ops.arange(self.max_seq_len)
        _mask = np.tril(np.ones((self.max_seq_len, self.max_seq_len), dtype=np.float32))
        self._causal_mask = keras.ops.convert_to_tensor(_mask[None, :, :])  # (1, T, T)

    def call(self, inputs, training=None):
        # inputs shape: (B, T) int32.
        x = self.tok_emb(inputs)
        T = keras.ops.shape(inputs)[1]
        pos = self.pos_emb(self._positions[:T])
        x = x + pos
        x = self.emb_drop(x, training=training)
        # Slice the pre-built causal mask to actual seq len.
        attn_mask = self._causal_mask[:, :T, :T]
        for blk in self.blocks:
            x = blk(x, attention_mask=attn_mask, training=training)
        x = self.final_norm(x)
        logits = self.head(x)
        return {"logits": logits}

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                vocab_size=self.vocab_size,
                max_seq_len=self.max_seq_len,
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                intermediate_size=self.intermediate_size,
                norm_type=self.norm_type,
                norm_kwargs=self.norm_kwargs,
                dropout_rate=self.dropout_rate,
            )
        )
        return cfg


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def _synthetic_packed_dataset(
    *, max_seq_len: int, batch_size: int, num_batches: int, seed: int,
) -> tf.data.Dataset:
    """In-memory synthetic packed-CLM dataset for the CPU smoke path.

    Emits ``num_batches`` deterministic ``(input_ids, labels)`` batches with
    ``input_ids[:, 1:] == labels[:, :-1]`` style shift, drawn from a tiny
    vocab slice (≤ 1024 token ids) to keep the head Dense gradient small on
    CPU. Real CLM training NEVER takes this path — guard via ``--synthetic-smoke``.
    """
    rng = np.random.default_rng(seed)
    inp_len = max_seq_len - 1
    xs = rng.integers(0, 1024, size=(num_batches * batch_size, inp_len), dtype=np.int32)
    ys = rng.integers(0, 1024, size=(num_batches * batch_size, inp_len), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_wiki_packed(
    *, max_train_articles: int, max_val_articles: int,
    max_seq_len: int, batch_size: int, seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Real path — Wikipedia 10k subset via load_wikipedia_train_val."""
    # Local import — avoid HF datasets import cost on CPU smoke path.
    from dl_techniques.datasets.nlp import load_wikipedia_train_val
    train_raw, val_raw, n_train, _n_val = load_wikipedia_train_val(
        max_train_samples=max_train_articles,
        max_val_samples=max_val_articles,
        seed=seed,
        return_counts=True,
        num_shards=1,
    )
    train_ds = preprocess_clm_packed_dataset(
        train_raw, encoding_name="cl100k_base",
        chunk_length=max_seq_len, batch_size=batch_size,
        eot_token_id=_EOT_TOKEN_ID, repeat=True,
    )
    val_ds = preprocess_clm_packed_dataset(
        val_raw, encoding_name="cl100k_base",
        chunk_length=max_seq_len, batch_size=batch_size,
        eot_token_id=_EOT_TOKEN_ID, repeat=False,
    )
    return train_ds, val_ds, n_train


# ---------------------------------------------------------------------
# Regime axis
# ---------------------------------------------------------------------


# `(lr, batch, mp, depth_override, wd_override)` — 5-tuple per plan_2026-05-18
# step 6. E6 supports default/mp_fp16/lr_extreme/wd_zero (mirrors E3's set,
# minus bs_4 because tiny CLM batches < 8 destabilise gradient estimates).
_REGIME_MAP: Dict[str, Tuple[
    Optional[float], Optional[int], Optional[bool], Optional[int], Optional[float]
]] = {
    "default":    (None, None, None, None, None),
    "mp_fp16":    (None, None, True, None, None),
    "lr_extreme": (3e-3, None, None, None, None),
    "wd_zero":    (None, None, None, None, 0.0),
}


# ---------------------------------------------------------------------
# Args / run
# ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6 4L causal-LM × Wikipedia 10k")
    p.add_argument("--norm-type", type=str, default="rms_norm", choices=list(NORM_VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="oob", choices=["oob", "param_matched"])
    p.add_argument("--regime", type=str, default="default",
                   choices=list(_REGIME_MAP.keys()),
                   help="Phase 3 regime sub-experiment selector.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=768)
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument("--max-band-width", type=float, default=0.1)
    p.add_argument("--max-train-articles", type=int, default=WIKI_ARTICLES_DEFAULT)
    p.add_argument("--max-val-articles", type=int, default=500)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Override the chunk-aware steps_per_epoch estimate.")
    p.add_argument("--synthetic-smoke", action="store_true",
                   help="CPU fast-path: skip HF Wikipedia load and use a tiny "
                        "synthetic packed corpus. Smoke-test only — never use "
                        "in real sweeps.")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def run(cfg: ExperimentConfig, *,
        max_seq_len: int, d_model: int, num_heads: int, num_layers: int,
        intermediate_size: int, dropout_rate: float, warmup_ratio: float,
        max_train_articles: int, max_val_articles: int,
        steps_per_epoch_override: Optional[int],
        synthetic_smoke: bool) -> dict:
    set_seeds(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---- Data ---------------------------------------------------------
    if synthetic_smoke:
        logger.info("[e6] SYNTHETIC SMOKE path active — no HF Wikipedia load.")
        train_ds_raw = _synthetic_packed_dataset(
            max_seq_len=max_seq_len, batch_size=cfg.batch_size,
            num_batches=4, seed=cfg.seed,
        )
        val_ds_raw = _synthetic_packed_dataset(
            max_seq_len=max_seq_len, batch_size=cfg.batch_size,
            num_batches=2, seed=cfg.seed + 1,
        )
        n_train_articles = 16  # synthetic — just a heuristic for steps calc
    else:
        train_ds_raw, val_ds_raw, n_train_articles = _load_wiki_packed(
            max_train_articles=max_train_articles,
            max_val_articles=max_val_articles,
            max_seq_len=max_seq_len, batch_size=cfg.batch_size, seed=cfg.seed,
        )

    # Wrap labels for dict-output model: (x, y) → (x, {"logits": y}).
    def _wrap(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(
            lambda x, y: (x, {"logits": y}),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = _wrap(train_ds_raw)
    val_ds = _wrap(val_ds_raw)

    steps_per_epoch = estimate_clm_steps_per_epoch(
        num_articles=n_train_articles,
        max_seq_length=max_seq_len,
        batch_size=cfg.batch_size,
        override=steps_per_epoch_override,
    )
    if synthetic_smoke and steps_per_epoch_override is None:
        # Match the synthetic generator's 4 batches.
        steps_per_epoch = 4
    logger.info(
        f"[e6] data: synthetic={synthetic_smoke}, n_train_articles={n_train_articles}, "
        f"steps_per_epoch={steps_per_epoch}"
    )

    # ---- Model --------------------------------------------------------
    norm_kwargs = cfg.norm_kwargs()
    model = TinyCLM(
        vocab_size=_VOCAB_SIZE, max_seq_len=max_seq_len,
        d_model=d_model, num_heads=num_heads, num_layers=num_layers,
        intermediate_size=intermediate_size,
        norm_type=cfg.norm_type, norm_kwargs=norm_kwargs,
        dropout_rate=dropout_rate,
        name=f"tiny_clm_{cfg.norm_type}",
    )
    # Build via dummy forward pass.
    dummy = np.zeros((1, max_seq_len - 1), dtype=np.int32)
    _ = model(dummy, training=False)
    n_params = int(model.count_params())
    logger.info(f"[e6] model: norm={cfg.norm_type}, mode={cfg.mode}, params={n_params}")

    # ---- Compile ------------------------------------------------------
    lr_schedule = create_warmup_lr_schedule(
        cfg.learning_rate, cfg.epochs, steps_per_epoch, warmup_ratio,
    )
    prepare_dict_keyed_compile(model)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.weight_decay,
            clipnorm=1.0,
        ),
        loss={"logits": MaskedCausalLMLoss()},
        metrics={"logits": build_clm_metrics("cl100k_base")},
    )

    # ---- Fit ----------------------------------------------------------
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            keras.callbacks.TerminateOnNaN(),
        ],
        verbose=2,
    )
    wall_s = time.time() - t0

    # ---- Headline metrics --------------------------------------------
    # MaskedCausalLMLoss is built with from_logits=True; PPL = exp(loss).
    final_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history.get("val_loss", [float("nan")])[-1])
    try:
        final_val_perplexity = float(math.exp(final_val_loss))
    except (OverflowError, ValueError):
        final_val_perplexity = float("nan")
    # Generalization gap on CLM = val_loss - train_loss (NaN-tolerant).
    try:
        generalization_gap = final_val_loss - final_loss
    except (TypeError, ValueError):
        generalization_gap = float("nan")

    # ---- Per-epoch history.csv ---------------------------------------
    hist_csv = os.path.join(cfg.out_dir, "history.csv")
    _hist = history.history
    _hist_keys = list(_hist.keys())
    with open(hist_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + _hist_keys)
        for i in range(len(_hist[_hist_keys[0]])):
            w.writerow([i] + [_hist[k][i] for k in _hist_keys])

    # ---- results.csv (append-or-create with header) ------------------
    results_csv = os.path.join(cfg.out_dir, "results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "experiment", "norm_type", "mode", "seed", "regime", "epochs",
                "trainable_params",
                "final_loss", "final_val_loss", "final_val_perplexity",
                "generalization_gap",
                "wall_s",
            ])
        w.writerow([
            "e6", cfg.norm_type, cfg.mode, cfg.seed,
            cfg.extras.get("regime", "default"), cfg.epochs,
            n_params,
            final_loss, final_val_loss, final_val_perplexity,
            generalization_gap,
            wall_s,
        ])
    logger.info(
        f"[e6] DONE: val_loss={final_val_loss:.4f}, "
        f"val_ppl={final_val_perplexity:.2f}, wall_s={wall_s:.1f}"
    )
    return {
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_perplexity,
        "wall_s": wall_s,
    }


def main() -> None:
    import keras as _keras
    args = _parse_args()
    lr_o, bs_o, mp_o, _, wd_o = _REGIME_MAP[args.regime]
    if lr_o is not None:
        args.learning_rate = lr_o
    if bs_o is not None:
        args.batch_size = bs_o
    if mp_o is True:
        _keras.mixed_precision.set_global_policy("mixed_float16")
    if wd_o is not None:
        args.weight_decay = wd_o
    cfg = ExperimentConfig(
        experiment_name="e6",
        norm_type=args.norm_type,
        seed=args.seed,
        mode=args.mode,
        dataset="wikipedia_10k",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=max(1, int(args.warmup_ratio * args.epochs)),
        mixed_precision=bool(mp_o),
        max_band_width=args.max_band_width,
        out_dir=args.out_dir,
        extras={"regime": args.regime},
    )
    run(
        cfg,
        max_seq_len=args.max_seq_length,
        d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        dropout_rate=args.dropout_rate,
        warmup_ratio=args.warmup_ratio,
        max_train_articles=args.max_train_articles,
        max_val_articles=args.max_val_articles,
        steps_per_epoch_override=args.steps_per_epoch,
        synthetic_smoke=args.synthetic_smoke,
    )


if __name__ == "__main__":
    main()
