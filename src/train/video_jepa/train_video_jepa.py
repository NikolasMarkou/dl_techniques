"""Video-JEPA-Clifford smoke-test training script.

Usage:

.. code-block:: bash

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
        -m train.video_jepa.train_video_jepa \
        --epochs 2 --batch-size 2 --T 4 --img-size 64 \
        --output-dir results/video_jepa_smoke_iter2 --seed 0

Loss is added via ``self.add_loss`` inside :meth:`VideoJEPA.call` so we
compile with ``loss=None``. ``jit_compile=False`` avoids XLA tracing
issues with the add_loss / reshape-heavy forward.

Iter-2 logging (D-012): the model exposes per-loss ``keras.metrics.Mean``
trackers (``next_frame_loss``, ``mask_loss``, ``sigreg_loss``) via its
``metrics`` property, so the :class:`CSVLogger` automatically writes each
component as a named column alongside the aggregated ``loss``. With
``mask_prediction_enabled=False`` the ``mask_loss`` column stays at 0.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
from pathlib import Path

import numpy as np

# Force non-interactive matplotlib before any import that might pull it in.
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import tensorflow as tf

from train.common import setup_gpu
from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.model import VideoJEPA
from dl_techniques.datasets.synthetic_drone_video import (
    synthetic_drone_video_dataset,
)
from dl_techniques.utils.logger import logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _build_config(args: argparse.Namespace) -> VideoJEPAConfig:
    return VideoJEPAConfig(
        img_size=args.img_size,
        img_channels=args.img_channels,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_frames=args.T,
        history_size_k=args.T,
        encoder_clifford_depth=args.encoder_clifford_depth,
        encoder_shifts=tuple(args.encoder_shifts),
        predictor_depth=args.predictor_depth,
        predictor_num_heads=args.predictor_num_heads,
        predictor_dim_head=args.predictor_dim_head,
        predictor_mlp_dim=args.predictor_mlp_dim,
        predictor_shifts=tuple(args.predictor_shifts),
        cond_dim=args.embed_dim,  # AdaLN broadcast invariant
        telemetry_dim=args.telemetry_dim,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_weight=args.sigreg_weight,
        dropout=args.dropout,
        # --- iter-2: V-JEPA tube-masked latent prediction ---
        mask_prediction_enabled=args.mask_prediction_enabled,
        mask_ratio=args.mask_ratio,
        lambda_next_frame=args.lambda_next_frame,
        lambda_mask=args.lambda_mask,
    )


def _build_callbacks(output_dir: Path) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "last.keras"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video-JEPA-Clifford smoke trainer")
    # Basic training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Results directory. Auto-timestamp if omitted.")

    # Window + image
    p.add_argument("--T", type=int, default=4,
                   help="Frames per clip (= num_frames = history_size_k).")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--img-channels", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=64)

    # Encoder
    p.add_argument("--encoder-clifford-depth", type=int, default=2)
    p.add_argument("--encoder-shifts", type=int, nargs="+", default=[1, 2])

    # Predictor
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--predictor-num-heads", type=int, default=4)
    p.add_argument("--predictor-dim-head", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=128)
    p.add_argument("--predictor-shifts", type=int, nargs="+", default=[1, 2])

    # Telemetry
    p.add_argument("--telemetry-dim", type=int, default=7)

    # SIGReg
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=64,
                   help="Default 64 for smoke; 1024 for full.")
    p.add_argument("--sigreg-weight", type=float, default=0.09)

    # Dropout
    p.add_argument("--dropout", type=float, default=0.0)

    # Iter-2: V-JEPA tube-masked latent prediction (D-008..D-012)
    p.add_argument(
        "--mask-prediction-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, add V-JEPA-style masked-latent prediction loss "
             "(alongside next-frame). Use --no-mask-prediction-enabled "
             "to fall back to iter-1 two-loss training.",
    )
    p.add_argument("--mask-ratio", type=float, default=0.6,
                   help="Fraction of spatial patch positions masked.")
    p.add_argument("--lambda-next-frame", type=float, default=1.0,
                   help="Weight on the next-frame prediction loss.")
    p.add_argument("--lambda-mask", type=float, default=1.0,
                   help="Weight on the mask-prediction loss.")

    args = p.parse_args()
    return args


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"video_jepa_{ts}"


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    setup_gpu(args.gpu)

    logger.info(f"Video-JEPA smoke training — args: {vars(args)}")

    cfg = _build_config(args)
    logger.info(f"Config: {cfg.to_dict()}")

    model = VideoJEPA(config=cfg)

    dataset = synthetic_drone_video_dataset(
        batch_size=args.batch_size,
        num_batches=args.steps_per_epoch,
        T=args.T,
        img_size=args.img_size,
        img_channels=args.img_channels,
        telemetry_dim=args.telemetry_dim,
        seed=args.seed,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        loss=None,  # training loss comes from self.add_loss
        jit_compile=False,
    )

    output_dir = _resolve_output_dir(args)
    logger.info(f"Output dir: {output_dir}")
    callbacks = _build_callbacks(output_dir)

    history = model.fit(
        dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
        verbose=2,
    )
    logger.info(f"Final loss history: {history.history}")

    # Save final model explicitly.
    final_path = output_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
