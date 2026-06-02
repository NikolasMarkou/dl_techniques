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
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Force non-interactive matplotlib before any import that might pull it in.
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import tensorflow as tf

from train.common import setup_gpu, create_base_argument_parser, set_seeds
from dl_techniques.models.lewm.config import LeWMConfig
from dl_techniques.models.lewm.model import LeWM
from dl_techniques.datasets.pusht_hdf5 import (
    synthetic_lewm_dataset,
    PushTHDF5Dataset,
)
from dl_techniques.utils.logger import logger


def _set_seed(seed: int) -> None:
    set_seeds(seed)


def _build_model(args: argparse.Namespace) -> LeWM:
    # Fail-fast validation: img/patch divisibility and embed_dim vs encoder
    # scale. Without this a wrong combo crashes deep inside ViT/projector with
    # an opaque shape error far from the CLI.
    from dl_techniques.models.vit.model import ViT
    if args.img_size % args.patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) must be divisible by patch_size "
            f"({args.patch_size})."
        )
    if args.encoder_scale not in ViT.SCALE_CONFIGS:
        raise ValueError(
            f"encoder_scale={args.encoder_scale!r} not in ViT.SCALE_CONFIGS "
            f"({list(ViT.SCALE_CONFIGS.keys())})."
        )
    expected_embed = ViT.SCALE_CONFIGS[args.encoder_scale][0]
    if args.embed_dim != expected_embed:
        raise ValueError(
            f"embed_dim ({args.embed_dim}) must equal the ViT encoder output "
            f"dim for scale={args.encoder_scale!r} ({expected_embed}). The "
            f"projector is identity-shaped; a mismatch crashes at first matmul."
        )

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


# Smoke preset overrides for fast CPU iteration. Mirror with care: these are
# applied AFTER argparse so they only override defaults the user didn't set
# (handled by tracking which flags were explicitly passed).
_SMOKE_OVERRIDES: Dict[str, Any] = {
    "img_size": 56,
    "patch_size": 14,
    "encoder_scale": "tiny",
    "embed_dim": 192,
    "history_size": 2,
    "num_preds": 1,
    "depth": 2,
    "heads": 4,
    "dim_head": 48,
    "mlp_dim": 256,
    "sigreg_num_proj": 64,
    "batch_size": 2,
    "epochs": 1,
    "steps_per_epoch": 2,
}


def parse_args() -> argparse.Namespace:
    # Adopt the project's base argument parser for shared flags
    # (--epochs, --batch-size, --learning-rate, --weight-decay, --gpu, ...).
    # --dataset / --image-size / --lr-schedule / --patience / --show-plots are
    # inherited but unused by this script; that drift is acceptable per
    # train/CLAUDE.md guidance to prefer the base parser for consistency.
    p = create_base_argument_parser(
        description="LeWM trainer (upstream defaults; --smoke for fast CPU iteration)",
        default_dataset="cifar10",  # ignored
    )
    # Override base defaults to upstream LeWM values.
    p.set_defaults(batch_size=16, epochs=50, learning_rate=5e-5, weight_decay=1e-3)

    # Smoke preset.
    p.add_argument("--smoke", action="store_true",
                   help="Tiny preset for fast CPU iteration. Overrides defaults: "
                        "img=56 patch=14 depth=2 heads=4 dim_head=48 mlp_dim=256 "
                        "history=2 sigreg_num_proj=64 batch=2 epochs=1 steps=2. "
                        "User-provided flags still win.")

    # Data source (mutually exclusive).
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic random data (default fallback).")
    src.add_argument("--hdf5-path", type=str, default=None,
                     help="Path to PushT-style HDF5 replay file.")

    # LeWM-specific flags not in the base parser.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps-per-epoch", type=int, default=200)

    # Model config — full LeWM upstream defaults.
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=14)
    p.add_argument("--encoder-scale", type=str, default="tiny")
    p.add_argument("--embed-dim", type=int, default=192)

    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)

    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--mlp-dim", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--smoothed-dim", type=int, default=10)
    p.add_argument("--mlp-scale", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=1)

    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)

    # Track which flags the user explicitly set so --smoke only overrides
    # unmodified defaults.
    explicit = _explicitly_set_flags(p)
    args = p.parse_args()

    if args.smoke:
        for key, value in _SMOKE_OVERRIDES.items():
            if key not in explicit:
                setattr(args, key, value)

    # Default to synthetic if neither flag is set.
    if not args.synthetic and not args.hdf5_path:
        args.synthetic = True
    return args


def _explicitly_set_flags(parser: argparse.ArgumentParser) -> set:
    """Inspect sys.argv against parser actions to record which dest names the
    user passed explicitly. Used so --smoke does not silently overwrite an
    intentional --depth 12."""
    # Build dest <- option string map.
    dest_by_opt: Dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            dest_by_opt[opt] = action.dest
    explicit: set = set()
    for token in sys.argv[1:]:
        key = token.split("=", 1)[0]
        if key in dest_by_opt:
            explicit.add(dest_by_opt[key])
    return explicit


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

    # Verify the saved model actually reloads and reproduces a forward pass.
    # FATAL on failure so CI catches serialization regressions.
    try:
        sample = next(iter(dataset))[0]
        y_orig = keras.ops.convert_to_numpy(model(sample, training=False))
        reloaded = keras.models.load_model(str(final_path))
        y_reload = keras.ops.convert_to_numpy(reloaded(sample, training=False))
        max_diff = float(np.max(np.abs(y_orig - y_reload)))
        if max_diff < 1e-4:
            logger.info(f"Reload check PASSED (max|delta|={max_diff:.2e}).")
        else:
            logger.error(
                f"Reload check FAILED: max|delta|={max_diff:.2e} >= 1e-4."
            )
            raise RuntimeError(
                f"Reload check FAILED: max|delta|={max_diff:.2e} >= 1e-4."
            )
    except Exception as e:  # noqa: BLE001 - surface any reload failure loudly
        logger.error(f"Reload check FAILED with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
