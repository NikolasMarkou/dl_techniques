"""LeVJEPA training script.

Wires :class:`~dl_techniques.models.vision.levjepa.training.LeVJEPATrainingModel`
to this repo's existing ``tf.data`` video loaders (per plan.md D-002, mirroring
``src/train/video_jepa/train_video_jepa.py``'s ``--dataset`` scaffolding) and
the new video multi-crop transform
(:func:`~dl_techniques.datasets.vision.multi_crop_video.make_multi_crop_video_map_fn`).

Loss is added via ``self.add_loss`` inside
``LeVJEPATrainingModel.call`` (no custom ``train_step`` anywhere in this
package), so this script compiles with ``loss=None``.

Usage:

.. code-block:: bash

    # Smoke (synthetic drone video, seconds on CPU):
    MPLBACKEND=Agg .venv/bin/python -m train.levjepa.train_levjepa --smoke

    # BDD100K:
    MPLBACKEND=Agg .venv/bin/python -m train.levjepa.train_levjepa \\
        --dataset bdd100k --videos-root /path/to/bdd100k/videos --gpu 0
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Force non-interactive matplotlib before any import that might pull it in.
os.environ.setdefault("MPLBACKEND", "Agg")

import keras
import tensorflow as tf

from train.common import setup_gpu, set_seeds
from train.common.args import explicitly_set_flags
from dl_techniques.optimization import optimizer_builder, WarmupSchedule
from dl_techniques.models.vision.levjepa.model import create_levjepa, SCALE_CONFIGS
from dl_techniques.models.vision.levjepa.training import LeVJEPATrainingModel
from dl_techniques.callbacks.ema_shadow_callback import EMAShadowCallback
from dl_techniques.datasets.vision.multi_crop_video import make_multi_crop_video_map_fn
from dl_techniques.datasets.synthetic_drone_video import synthetic_drone_video_dataset
from dl_techniques.datasets.bdd100k_video import bdd100k_video_dataset
from train.levjepa.schedule import FlatSchedule
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@dataclasses.dataclass
class TrainingConfig:
    """Resolved LeVJEPA training configuration."""

    dataset: str = "synthetic_drone"
    videos_root: str = "/media/arxwn/data0_4tb/datasets/bdd_data/train/videos"
    num_frames: int = 8
    img_size: int = 64
    img_channels: int = 3
    variant: str = "vit_tiny"
    tubelet_size: int = 2
    batch_size: int = 4
    epochs: int = 2
    steps_per_epoch: int = 8
    sigreg_weight: float = 0.02
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    local_crops_number: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.04
    warmup_steps: int = 10
    ema_decay: float = 0.999
    ema_update_every: int = 1
    seed: int = 0
    gpu: Optional[int] = None
    output_dir: Optional[str] = None
    # DECISION plan-2026-09-03T113223-2a714a91/D-023
    # These three defaults are deliberately the encoder's RISKIEST config,
    # not its degenerate ("full", no RoPE, no drop) defaults. CRITICAL-2 of
    # the iter-1 adversarial review found the trainer never wired
    # attn_mode/use_rope/token_dropout_rate at all, so the block-causal mask
    # (Step 3), VideoRoPE3D (Step 2) and random_token_drop (Step 3) -- the
    # plan's own defining, paper-differentiating mechanisms -- were
    # unreachable from every real run, including the smoke run backing
    # Success Criterion 10. Do NOT quietly revert these to
    # attn_mode="full"/use_rope=False/token_dropout_rate=0.0 "to match the
    # encoder's own constructor defaults" -- that would silently reopen the
    # exact gap this decision closes. See decisions.md D-023.
    attn_mode: str = "block_causal"
    use_rope: bool = True
    token_dropout_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.dataset not in ("synthetic_drone", "bdd100k"):
            raise ValueError(
                f"dataset must be one of 'synthetic_drone', 'bdd100k', got "
                f"{self.dataset!r}"
            )
        if self.attn_mode not in ("full", "block_causal"):
            raise ValueError(
                f"attn_mode must be one of 'full', 'block_causal', got "
                f"{self.attn_mode!r}"
            )
        if not (0.0 <= self.token_dropout_rate < 1.0):
            raise ValueError(
                f"token_dropout_rate must be in [0, 1), got {self.token_dropout_rate}"
            )
        if self.variant not in SCALE_CONFIGS:
            raise ValueError(
                f"variant must be one of {list(SCALE_CONFIGS.keys())}, got "
                f"{self.variant!r}"
            )
        if self.batch_size < 2:
            raise ValueError(f"batch_size must be >= 2, got {self.batch_size}")
        if self.num_frames <= 1:
            raise ValueError(
                f"num_frames must be > 1 (LeVJEPATrainingModel requires a "
                f"video-mode encoder), got {self.num_frames}"
            )
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames ({self.num_frames}) must be divisible by "
                f"tubelet_size ({self.tubelet_size})"
            )
        if self.local_crops_number < 1:
            raise ValueError(
                f"local_crops_number must be >= 1 (a multiview forward pass "
                f"needs at least one local view), got {self.local_crops_number}"
            )
        if self.sigreg_weight < 0.0:
            raise ValueError(f"sigreg_weight must be >= 0, got {self.sigreg_weight}")
        if self.epochs <= 0 or self.steps_per_epoch <= 0:
            raise ValueError(
                f"epochs ({self.epochs}) and steps_per_epoch "
                f"({self.steps_per_epoch}) must both be positive"
            )
        patch_size = SCALE_CONFIGS[self.variant][4]
        if self.img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({self.img_size}) must be divisible by variant "
                f"'{self.variant}''s patch_size ({patch_size})"
            )


# Smoke preset — fast CPU/iteration. Applied AFTER argparse so it only
# overrides flags the user did not explicitly type (mirrors train_video_jepa.py).
_SMOKE_OVERRIDES: Dict[str, Any] = {
    "dataset": "synthetic_drone",
    "num_frames": 4,
    "img_size": 32,
    "variant": "vit_tiny",
    "tubelet_size": 2,
    "batch_size": 2,
    "epochs": 1,
    "steps_per_epoch": 2,
    "sigreg_num_proj": 32,
    "local_crops_number": 1,
    "warmup_steps": 1,
    # Pinned explicitly (not merely inherited from the parser defaults) so
    # a future change to those defaults cannot silently regress the smoke
    # run back to the degenerate config CRITICAL-2 found. See D-023.
    "attn_mode": "block_causal",
    "use_rope": True,
    "token_dropout_rate": 0.5,
}


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. The FIRST statement of this function parses argv,
    so ``--help`` exits 0 with a ``usage:`` line and allocates nothing.
    """
    parser = argparse.ArgumentParser(
        description="LeVJEPA trainer (video multiview SSL, SIGReg regularized)."
    )
    parser.add_argument(
        "--dataset", choices=["synthetic_drone", "bdd100k"], default="synthetic_drone",
        help="Which tf.data video loader to use.",
    )
    parser.add_argument(
        "--videos-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/bdd_data/train/videos",
        help="Root directory for BDD100K .mov files (flat layout). Required "
             "for --dataset bdd100k.",
    )
    parser.add_argument("--num-frames", type=int, default=8,
                         help="Frames per clip. Must be > 1 (video mode).")
    parser.add_argument("--img-size", type=int, default=64,
                         help="Square source clip edge length.")
    parser.add_argument("--img-channels", type=int, default=3)
    parser.add_argument("--variant", type=str, default="vit_tiny",
                         choices=list(SCALE_CONFIGS.keys()))
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", "--steps-per-epoch", dest="steps_per_epoch",
                         type=int, default=8)
    parser.add_argument("--sigreg-weight", type=float, default=0.02)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--local-crops-number", type=int, default=2)
    parser.add_argument(
        "--attn-mode", type=str, choices=["full", "block_causal"],
        default="block_causal",
        help="Encoder attention mode. 'block_causal' (default) is the "
             "paper's headline recipe and the config this trainer actually "
             "exercises; 'full' is the degenerate unmasked mode.",
    )
    parser.add_argument(
        "--use-rope", action=argparse.BooleanOptionalAction, default=True,
        help="Use VideoRoPE3D rotary position embedding instead of the "
             "frozen 3D sincos table (default: on). Pass --no-use-rope to "
             "use sincos instead.",
    )
    parser.add_argument(
        "--token-dropout-rate", type=float, default=0.5,
        help="Train-time patch-token drop fraction, in [0, 1). Default 0.5 "
             "is deliberately modest relative to the paper's 0.95 -- see "
             "D-023 for why a smoke-scale token grid needs a gentler rate.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-update-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Results directory. Auto-timestamp if omitted.")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny preset for fast CPU iteration: synthetic_drone, T=4, "
             "img=32, vit_tiny, batch=2, epochs=1, steps=2, "
             "sigreg_num_proj=32, local_crops_number=1, warmup_steps=1. "
             "User-provided flags still win.",
    )

    explicit = explicitly_set_flags(parser, argv)
    args = parser.parse_args(argv)

    if args.smoke:
        for key, value in _SMOKE_OVERRIDES.items():
            if key not in explicit:
                setattr(args, key, value)

    return args


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in fields and k != "smoke"}
    return TrainingConfig(**kwargs)


def _resolve_output_dir(config: TrainingConfig) -> Path:
    if config.output_dir:
        return Path(config.output_dir)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"levjepa_{ts}"


def _build_dataset(config: TrainingConfig) -> "tf.data.Dataset":
    """Raw clip loader (D-002) -> unbatch -> video multi-crop -> rebatch."""
    if config.dataset == "bdd100k":
        logger.info(f"Using BDD100K loader from {config.videos_root}.")
        raw_dataset = bdd100k_video_dataset(
            videos_root=config.videos_root,
            batch_size=config.batch_size,
            T=config.num_frames,
            img_size=config.img_size,
            seed=config.seed,
        )
    else:
        logger.info("Using synthetic drone-video dataset.")
        raw_dataset = synthetic_drone_video_dataset(
            batch_size=config.batch_size,
            num_batches=config.steps_per_epoch,
            T=config.num_frames,
            img_size=config.img_size,
            img_channels=config.img_channels,
            seed=config.seed,
        )

    transform = make_multi_crop_video_map_fn(
        crop_size=config.img_size,
        num_frames=config.num_frames,
        local_crops_number=config.local_crops_number,
    )
    return (
        raw_dataset.unbatch()
        .map(transform, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


def _build_model(config: TrainingConfig) -> LeVJEPATrainingModel:
    """Construct the encoder + training-model, forwarding the config's
    attn_mode/use_rope/token_dropout_rate through to :func:`create_levjepa`.

    Split out from :func:`main` so a test can build the model (and inspect
    ``model.encoder.attn_mode``/``use_rope``/``token_dropout_rate``) without
    also running a full ``model.fit()`` — see
    ``tests/test_train/test_levjepa/test_train_levjepa.py::TestSmokeConfigWiring``.
    """
    encoder = create_levjepa(
        variant=config.variant,
        input_shape=(config.img_size, config.img_size, config.img_channels),
        num_frames=config.num_frames,
        tubelet_size=config.tubelet_size,
        attn_mode=config.attn_mode,
        use_rope=config.use_rope,
        token_dropout_rate=config.token_dropout_rate,
    )
    return LeVJEPATrainingModel(
        encoder=encoder,
        sigreg_knots=config.sigreg_knots,
        sigreg_num_proj=config.sigreg_num_proj,
        sigreg_weight=config.sigreg_weight,
    )


def main(argv: Optional[Sequence[str]] = None):
    args = parse_arguments(argv)
    config = _build_config(args)
    set_seeds(config.seed)
    setup_gpu(config.gpu)

    logger.info(f"LeVJEPA training — config: {dataclasses.asdict(config)}")

    model = _build_model(config)

    schedule = WarmupSchedule(
        warmup_steps=config.warmup_steps,
        primary_schedule=FlatSchedule(learning_rate=config.lr),
    )
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "weight_decay": config.weight_decay,
            "exclude_from_weight_decay": ["bias", "gamma", "beta"],
        },
        schedule,
    )
    model.compile(optimizer=optimizer, loss=None, jit_compile=False)

    dataset = _build_dataset(config)

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
        EMAShadowCallback(decay=config.ema_decay, update_every=config.ema_update_every),
    ]

    history = model.fit(
        dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=2,
    )
    final_summary = {k: float(v[-1]) for k, v in history.history.items()}
    logger.info(f"Final-epoch metrics: {final_summary}")
    return history


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
