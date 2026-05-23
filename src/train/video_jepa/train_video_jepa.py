"""Video-JEPA-Clifford training script.

Defaults track the BDD100K **full-spec** sanity build. Use ``--smoke`` for
fast CPU/iteration runs (synthetic dataset, tiny dims). User-provided
flags always win over ``--smoke`` overrides.

Usage:

.. code-block:: bash

    # Smoke (synthetic, seconds on CPU):
    MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \\
        --smoke

    # Full BDD100K sanity (GPU 0):
    MPLBACKEND=Agg .venv/bin/python -m train.video_jepa.train_video_jepa \\
        --dataset bdd100k --gpu 0 \\
        --videos-root /media/arxwn/data0_4tb/datasets/bdd_data/train/videos \\
        --output-dir results/video_jepa_sanity_bdd100k

Loss is added via ``self.add_loss`` inside :meth:`VideoJEPA.call` so we
compile with ``loss=None``. ``jit_compile=False`` avoids XLA tracing
issues with the add_loss / reshape-heavy forward.

After training, the script saves ``final_model.keras`` and verifies the
saved model reloads and reproduces a forward pass to ``max|delta|<1e-4``.
A failure exits 1 so CI catches serialization regressions.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Force non-interactive matplotlib before any import that might pull it in.
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import tensorflow as tf

from train.common import setup_gpu, create_base_argument_parser
from dl_techniques.models.video_jepa.config import VideoJEPAConfig
from dl_techniques.models.video_jepa.model import VideoJEPA
from dl_techniques.datasets.synthetic_drone_video import (
    synthetic_drone_video_dataset,
)
from dl_techniques.datasets.bdd100k_video import bdd100k_video_dataset
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from dl_techniques.callbacks.jepa_visualization import (
    LatentMaskOverlayCallback,
    PatchPredictionErrorCallback,
)
from dl_techniques.utils.logger import logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def _validate_args(args: argparse.Namespace) -> None:
    """Fail-fast CLI validation. Without these, mismatches crash deep inside
    encoder/predictor build with opaque shape errors far from the CLI.
    """
    if args.img_size % args.patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) must be divisible by patch_size "
            f"({args.patch_size})."
        )
    if args.embed_dim <= 0 or args.embed_dim % 2 != 0:
        raise ValueError(
            f"embed_dim must be positive and even (sine2D PE wants D//2); "
            f"got {args.embed_dim}."
        )
    expected = args.predictor_num_heads * args.predictor_dim_head
    if args.embed_dim != expected:
        raise ValueError(
            f"embed_dim ({args.embed_dim}) must equal "
            f"predictor_num_heads * predictor_dim_head "
            f"({args.predictor_num_heads} * {args.predictor_dim_head} = "
            f"{expected}). The temporal MHA wiring assumes this product; "
            f"a mismatch crashes deep inside the predictor."
        )
    if args.batch_size < 2:
        raise ValueError(
            f"batch_size must be >= 2 (CliffordNetBlock BatchNorm stability); "
            f"got {args.batch_size}."
        )


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
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_weight=args.sigreg_weight,
        dropout=args.dropout,
        mask_prediction_enabled=args.mask_prediction_enabled,
        mask_ratio=args.mask_ratio,
        lambda_next_frame=args.lambda_next_frame,
        lambda_mask=args.lambda_mask,
    )


def _extract_pixels(batch) -> np.ndarray:
    """Pull the pixel tensor out of a training-dataset batch.

    Both producers yield ``({"pixels": (B, T, H, W, C)}, 0.0)``.
    """
    inputs = batch[0] if isinstance(batch, tuple) else batch
    if isinstance(inputs, dict):
        pixels = inputs.get("pixels")
    else:
        pixels = inputs
    if pixels is None:
        raise ValueError(
            f"Could not extract 'pixels' from dataset batch of type {type(batch)}."
        )
    return np.asarray(pixels)


def _build_callbacks(
    output_dir: Path,
    eval_pixels: np.ndarray,
    visualization_frequency: int,
    early_stopping_patience: Optional[int] = None,
    has_validation: bool = False,
) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    jepa_viz_dir = output_dir / "jepa_viz"
    curves_dir = output_dir / "training_curves"
    monitor = "val_loss" if has_validation else "loss"
    cbs = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "last.keras"),
            save_best_only=False,
            save_weights_only=False,
            verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        TrainingCurvesCallback(
            output_dir=str(curves_dir),
            frequency=visualization_frequency,
        ),
        LatentMaskOverlayCallback(
            eval_pixels=eval_pixels,
            output_dir=str(jepa_viz_dir),
            frequency=visualization_frequency,
        ),
        PatchPredictionErrorCallback(
            eval_pixels=eval_pixels,
            output_dir=str(jepa_viz_dir),
            frequency=visualization_frequency,
        ),
    ]
    if early_stopping_patience is not None and early_stopping_patience > 0:
        cbs.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    return cbs


# Smoke preset — fast CPU/iteration. Applied AFTER argparse so it only
# overrides defaults the user did not explicitly set (mirrors LeWM).
_SMOKE_OVERRIDES: Dict[str, Any] = {
    "dataset": "synthetic",
    "T": 4,
    "img_size": 64,
    "patch_size": 8,
    "embed_dim": 64,
    "predictor_num_heads": 4,
    "predictor_dim_head": 16,
    "predictor_mlp_dim": 128,
    "encoder_clifford_depth": 2,
    "predictor_depth": 2,
    "sigreg_num_proj": 64,
    "batch_size": 2,
    "epochs": 2,
    "steps_per_epoch": 4,
}


def parse_args() -> argparse.Namespace:
    # Adopt the project's base argument parser for shared flags
    # (--epochs, --batch-size, --learning-rate, --weight-decay, --gpu,
    #  --dataset, --image-size, --lr-schedule, --patience, --show-plots).
    # --image-size / --lr-schedule / --patience / --show-plots are inherited
    # but unused by this script; --dataset is repurposed to {synthetic, bdd100k}.
    p = create_base_argument_parser(
        description="Video-JEPA-Clifford trainer (BDD full-spec defaults; "
                    "--smoke for fast iteration)",
        default_dataset="bdd100k",
    )
    # Override base defaults to upstream BDD full-spec values.
    p.set_defaults(
        batch_size=4,
        epochs=100,
        learning_rate=3e-4,
        weight_decay=1e-4,
    )
    # Repurpose --dataset choices for video_jepa.
    for action in p._actions:
        if action.dest == "dataset":
            action.choices = ["synthetic", "bdd100k"]
            action.help = "Which dataset to train on. 'bdd100k' requires --videos-root."

    # --- Smoke preset ---
    p.add_argument("--smoke", action="store_true",
                   help="Tiny preset for fast CPU iteration: synthetic, "
                        "T=4, img=64, patch=8, embed=64, depth=2, batch=2, "
                        "epochs=2, steps=4, sigreg_num_proj=64. "
                        "User-provided flags still win.")

    # --- Training extras not in the base parser ---
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Results directory. Auto-timestamp if omitted.")

    # --- Window + image (full-spec defaults: T=8 / 112² / patch=8 / D=128) ---
    p.add_argument("--T", type=int, default=8,
                   help="Frames per clip (= num_frames = history_size_k).")
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--img-channels", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=128)

    # --- Encoder ---
    p.add_argument("--encoder-clifford-depth", type=int, default=2)
    p.add_argument("--encoder-shifts", type=int, nargs="+", default=[1, 2])

    # --- Predictor ---
    p.add_argument("--predictor-depth", type=int, default=2)
    p.add_argument("--predictor-num-heads", type=int, default=8)
    p.add_argument("--predictor-dim-head", type=int, default=16)
    p.add_argument("--predictor-mlp-dim", type=int, default=256)
    p.add_argument("--predictor-shifts", type=int, nargs="+", default=[1, 2])

    # --- BDD100K data source ---
    p.add_argument("--videos-root", type=str,
                   default="/media/arxwn/data0_4tb/datasets/bdd_data/train/videos",
                   help="Root directory for BDD100K .mov files (flat layout).")

    # --- SIGReg (full-spec default: 1024 projections) ---
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=1024,
                   help="Default 1024 for full training; --smoke uses 64.")
    p.add_argument("--sigreg-weight", type=float, default=0.09)

    # --- Dropout ---
    p.add_argument("--dropout", type=float, default=0.0)

    # --- V-JEPA tube-masked latent prediction (D-008..D-012) ---
    p.add_argument(
        "--mask-prediction-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, add V-JEPA-style masked-latent prediction loss "
             "(alongside next-frame). Use --no-mask-prediction-enabled "
             "to fall back to iter-1 two-loss training.",
    )
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--lambda-next-frame", type=float, default=1.0)
    p.add_argument("--lambda-mask", type=float, default=1.0)

    # --- Validation / EarlyStopping / Viz ---
    p.add_argument("--val-steps", type=int, default=0,
                   help="Validation batches per epoch. 0 disables validation. "
                        "BDD100K only.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of BDD100K files held out for val.")
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="Patience (epochs) for EarlyStopping. 0 disables.")
    p.add_argument("--visualization-frequency", type=int, default=1,
                   help="Write visualization PNGs every N epochs.")

    explicit = _explicitly_set_flags(p)
    args = p.parse_args()

    if args.smoke:
        for key, value in _SMOKE_OVERRIDES.items():
            if key not in explicit:
                setattr(args, key, value)

    return args


def _explicitly_set_flags(parser: argparse.ArgumentParser) -> set:
    """Inspect sys.argv against parser actions to record which dest names
    the user passed explicitly. Used so --smoke does not silently overwrite
    an intentional --depth 12."""
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


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"video_jepa_{ts}"


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_seed(args.seed)
    setup_gpu(args.gpu)

    logger.info(f"Video-JEPA training — args: {vars(args)}")

    cfg = _build_config(args)
    logger.info(f"Config: {cfg.to_dict()}")

    model = VideoJEPA(config=cfg)

    val_dataset = None
    if args.dataset == "bdd100k":
        logger.info(
            f"Using BDD100K loader from {args.videos_root} "
            f"(steps_per_epoch={args.steps_per_epoch}, "
            f"val_steps={args.val_steps})."
        )
        split_train = "train" if args.val_steps > 0 else "all"
        dataset = bdd100k_video_dataset(
            videos_root=args.videos_root,
            batch_size=args.batch_size,
            T=args.T,
            img_size=args.img_size,
            seed=args.seed,
            split=split_train,
            val_fraction=args.val_fraction,
        )
        if args.val_steps > 0:
            val_dataset = bdd100k_video_dataset(
                videos_root=args.videos_root,
                batch_size=args.batch_size,
                T=args.T,
                img_size=args.img_size,
                seed=args.seed,
                split="val",
                val_fraction=args.val_fraction,
            )
    else:
        logger.info("Using synthetic drone-video dataset (smoke).")
        dataset = synthetic_drone_video_dataset(
            batch_size=args.batch_size,
            num_batches=args.steps_per_epoch,
            T=args.T,
            img_size=args.img_size,
            img_channels=args.img_channels,
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

    first_batch = next(iter(dataset))
    eval_pixels = _extract_pixels(first_batch)
    logger.info(
        f"Cached visualization eval batch: shape={eval_pixels.shape} "
        f"(freq={args.visualization_frequency})"
    )

    callbacks = _build_callbacks(
        output_dir,
        eval_pixels=eval_pixels,
        visualization_frequency=args.visualization_frequency,
        early_stopping_patience=args.early_stopping_patience,
        has_validation=val_dataset is not None,
    )

    history = model.fit(
        dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=args.val_steps if val_dataset is not None else None,
        callbacks=callbacks,
        verbose=2,
    )
    logger.info(f"Final loss history: {history.history}")

    # Save final model explicitly.
    final_path = output_dir / "final_model.keras"
    model.save(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Verify the saved model actually reloads and reproduces a forward pass.
    # FATAL on failure so CI catches serialization regressions (mirrors LeWM).
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
