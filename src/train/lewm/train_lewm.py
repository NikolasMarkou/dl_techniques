"""
LeWM smoke-test training script.

Usage:

.. code-block:: bash

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lewm.train_lewm \\
        --synthetic --batch-size 2 --epochs 1 --steps-per-epoch 2

This trains a LeWM model on synthetic random data. Loss is added via
``self.add_loss`` inside ``LeWM.call``, so we compile with ``loss=None``.
``jit_compile=False`` avoids XLA tracing issues with the dynamic rollout
loop / add_loss.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Force non-interactive matplotlib before any import that might pull it in.
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import tensorflow as tf

from train.common import setup_gpu
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM
from dl_techniques.datasets.pusht_hdf5 import (
    synthetic_lewm_dataset,
    PushTHDF5Dataset,
)
from dl_techniques.utils.logger import logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _build_model(args: argparse.Namespace) -> LeWM:
    cfg = LeWMConfig(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_scale=args.encoder_scale,
        embed_dim=args.embed_dim,
        history_size=args.history_size,
        num_preds=args.num_preds,
        num_frames=args.history_size + args.num_preds,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        projector_hidden_dim=args.embed_dim,
        action_dim=args.action_dim,
        smoothed_dim=args.smoothed_dim,
        mlp_scale=args.mlp_scale,
        sigreg_weight=args.sigreg_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
    )
    return LeWM(config=cfg)


def _build_dataset(args: argparse.Namespace) -> tf.data.Dataset:
    if args.synthetic:
        return synthetic_lewm_dataset(
            num_episodes=max(args.steps_per_epoch * args.epochs * args.batch_size, 8),
            img_size=args.img_size,
            action_dim=args.action_dim,
            batch_size=args.batch_size,
            history_size=args.history_size,
            num_preds=args.num_preds,
            seed=args.seed,
        )
    else:
        if not args.hdf5_path:
            raise ValueError("Provide --hdf5-path or use --synthetic.")
        return PushTHDF5Dataset(
            h5_path=args.hdf5_path,
            img_size=args.img_size,
            action_dim=args.action_dim,
            history_size=args.history_size,
            num_preds=args.num_preds,
            frameskip=args.frameskip,
            batch_size=args.batch_size,
        ).as_tf_dataset()


def _build_callbacks(results_dir: Path) -> list:
    """Use a minimal, local callback list to avoid the EpochAnalyzerCallback
    which doesn't understand our dict inputs / add_loss-only training."""
    results_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(results_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / "last.keras"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
    ]


def _results_dir(prefix: str = "lewm") -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("results") / f"{prefix}_{ts}"
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeWM smoke-test trainer")
    # Data source
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic random data (default).")
    src.add_argument("--hdf5-path", type=str, default=None,
                     help="Path to PushT-style HDF5 replay file.")

    # Basic training args
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (sets CUDA_VISIBLE_DEVICES). None for CPU/default.")

    # Model config (defaults target tiny smoke-test; full LeWM defaults live
    # in LeWMConfig — use --encoder-scale tiny --img-size 224 --patch-size 14
    # for the full-spec build).
    p.add_argument("--img-size", type=int, default=56,
                   help="Square image edge (default 56 for CPU smoke; 224 for full).")
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--encoder-scale", type=str, default="tiny")
    p.add_argument("--embed-dim", type=int, default=192)

    p.add_argument("--history-size", type=int, default=2)
    p.add_argument("--num-preds", type=int, default=1)

    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-head", type=int, default=48)
    p.add_argument("--mlp-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--mlp-scale", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=1)

    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=64,
                   help="Number of random projections in SIGReg (default 64 "
                        "for smoke; 1024 for full LeWM).")

    args = p.parse_args()
    # Default to synthetic if neither flag is set.
    if not args.synthetic and not args.hdf5_path:
        args.synthetic = True
    return args


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    setup_gpu(args.gpu)

    logger.info(f"LeWM smoke-test training — args: {vars(args)}")

    model = _build_model(args)
    dataset = _build_dataset(args)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        loss=None,  # training loss comes from self.add_loss inside LeWM.call
        jit_compile=False,
    )

    results_dir = _results_dir(prefix="lewm")
    logger.info(f"Results dir: {results_dir}")
    callbacks = _build_callbacks(results_dir)

    history = model.fit(
        dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=2,
    )
    logger.info(f"Final loss history: {history.history}")

    # Save final model explicitly — round-trips via get_config/from_config.
    final_path = results_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
