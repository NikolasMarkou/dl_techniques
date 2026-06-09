"""Training script for BurstDP on COCO 2017 with synthetic auxiliary views.

Usage::

    MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \\
        --preset burst_dp_small \\
        --image-size 256 \\
        --batch-size 4 \\
        --epochs 40 \\
        --n-max 5 --n-min 1 \\
        --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \\
        --out-dir src/results/burst_dp/run01 \\
        --gpu 0

The training pipeline:
    - shared ViT encoder
    - ref-conditioned set-fusion stack
    - 2 heads (recon, segmentation)

Losses:
    - reconstruction : Charbonnier loss vs. clean reference
    - segmentation   : sparse categorical cross-entropy

Periodic visualization: every ``--viz-every-steps`` optimizer steps and / or
every ``--viz-every-epochs`` epochs the trainer saves a recon+segmentation
comparison grid under ``out_dir/viz/`` for a fixed val batch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.coco_multitask_local import COCO_DEFAULT_ROOT
from dl_techniques.callbacks.burst_dp_visualization import BurstDPVisualizationCallback
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from dl_techniques.datasets.vision.coco_burst_dp import (
    DistortionSpec,
    build_coco_burst_dp_datasets,
    default_anchor_spec,
    default_aux_spec,
)
from dl_techniques.datasets.vision.image_folder_burst_dp import (
    build_div2k_burst_dp_datasets,
    build_vggface2_burst_dp_datasets,
)
# COCOBurstDPConfig + COCO2017BurstDPLoader were referenced by the now-removed
# tf.data wrapper (see D-001 below). Keras 3 model.fit consumes the PyDataset
# directly, so neither name is needed in this module anymore.
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.models.burst_dp import (
    DEFAULT_NUM_SEG_CLASSES,
    BurstDP,
    create_burst_dp,
)
from dl_techniques.utils.logger import logger

# train.common helpers — local imports are kept conditional so the unit test
# environment does not need the full training stack on PYTHONPATH.
try:
    from train.common.gpu import setup_gpu
    from train.common import set_seeds
except Exception:  # pragma: no cover — fallback for editable installs
    setup_gpu = None  # type: ignore[assignment]
    set_seeds = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def charbonnier_loss(epsilon: float = 1e-3):
    """Robust L1-ish loss: sqrt((y_true - y_pred)^2 + eps^2)."""
    def _loss(y_true, y_pred):
        diff = y_true - y_pred
        return tf.reduce_mean(tf.sqrt(diff * diff + epsilon * epsilon))
    _loss.__name__ = "charbonnier_loss"
    return _loss


def sparse_seg_ce_loss():
    """Sparse categorical cross-entropy from logits."""
    cce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def _loss(y_true, y_pred):
        # y_true: (B, H, W) int; y_pred: (B, H, W, C) logits.
        return cce(y_true, y_pred)
    _loss.__name__ = "sparse_seg_ce"
    return _loss


# DECISION plan_2026-05-19_39a6a454/D-001
# Earlier versions of this script wrapped `COCO2017BurstDPLoader` (a
# `keras.utils.PyDataset` with `workers=4, use_multiprocessing=True`) in
# `tf.data.Dataset.from_generator(...)` for `model.fit`. That wrap actively
# defeats the PyDataset's parallel worker pool (the generator runs single-
# threaded in the main process). Keras 3 supports passing a `PyDataset`
# directly to `model.fit(...)` and iterates with the configured workers.
# Reuse-review proposed "switch to indefinite generator + steps_per_epoch"
# (the video_jepa pattern). That is the right shape for *generator-based*
# loaders without a PyDataset abstraction; for us, the cleaner fix is to
# drop the wrap entirely. See plans/plan_2026-05-19_39a6a454/findings/
# 03-train-script-current-state.md.


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train BurstDP on COCO 2017.")
    p.add_argument("--preset", type=str, default="burst_dp_small",
                   choices=["burst_dp_pico", "burst_dp_tiny", "burst_dp_small", "burst_dp_base"])
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--n-min", type=int, default=1)

    p.add_argument("--dataset", type=str, default="coco",
                   choices=["coco", "div2k", "vggface2"],
                   help="Training dataset. 'div2k' and 'vggface2' are fidelity-only "
                        "(zero seg loss + seg metrics dropped).")
    p.add_argument("--coco-root", type=str, default=COCO_DEFAULT_ROOT)
    p.add_argument("--div2k-root", type=str,
                   default="/media/arxwn/data0_4tb/datasets/div2k")
    p.add_argument("--vggface2-root", type=str,
                   default="/media/arxwn/data0_4tb/datasets/VGG-Face2/data")
    p.add_argument("--max-train-images", type=int, default=None)
    p.add_argument("--max-val-images", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=1)

    p.add_argument("--loss-recon", type=float, default=1.0)
    p.add_argument("--loss-seg", type=float, default=1.0)

    p.add_argument("--mixed-precision", action="store_true",
                   help="Enable fp16 mixed precision (recommended on RTX 4090).")
    p.add_argument("--workers", type=int, default=4)

    # Periodic visualization. Either trigger > 0 enables the callback; both
    # 0 means no visualization.
    p.add_argument("--viz-every-steps", type=int, default=500,
                   help="Save a recon+seg grid every N optimizer steps. 0 disables.")
    p.add_argument("--viz-every-epochs", type=int, default=1,
                   help="Save a recon+seg grid every M epochs. 0 disables.")
    p.add_argument("--viz-num-samples", type=int, default=4,
                   help="Rows in the visualization grid (capped to val batch size).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="src/results/burst_dp/run")

    # ---- Aux DistortionSpec overrides (Axis A, aux-only knob-up). ----
    # Each flag defaults to None ⇒ "do not override" (inherit default_aux_spec).
    # Anchor spec is intentionally NOT exposed — it defines the test-time input
    # distribution and widening it would change the deployed-model contract.
    # See D-001 anchor in main() below for the rationale.
    aux = p.add_argument_group(
        "aux distortion overrides",
        description="Per-field overrides of the aux DistortionSpec. All "
                    "default to None (no override). Anchor spec is fixed by "
                    "design — see DECISION D-001.",
    )
    # Photometric noise
    aux.add_argument("--aux-noise-sigma-min", type=float, default=None)
    aux.add_argument("--aux-noise-sigma-max", type=float, default=None)
    # Brightness / contrast
    aux.add_argument("--aux-brightness-jitter", type=float, default=None)
    aux.add_argument("--aux-contrast-jitter", type=float, default=None)
    # Gaussian blur
    aux.add_argument("--aux-blur-sigma-min", type=float, default=None)
    aux.add_argument("--aux-blur-sigma-max", type=float, default=None)
    # Motion blur
    aux.add_argument("--aux-motion-blur-prob", type=float, default=None)
    aux.add_argument("--aux-motion-blur-length-min", type=int, default=None)
    aux.add_argument("--aux-motion-blur-length-max", type=int, default=None)
    # Occlusion
    aux.add_argument("--aux-occlusion-prob", type=float, default=None)
    aux.add_argument("--aux-occlusion-max-frac", type=float, default=None)
    aux.add_argument("--aux-occlusion-num-boxes-min", type=int, default=None)
    aux.add_argument("--aux-occlusion-num-boxes-max", type=int, default=None)
    # Affine
    aux.add_argument("--aux-affine-angle-min", type=float, default=None)
    aux.add_argument("--aux-affine-angle-max", type=float, default=None)
    aux.add_argument("--aux-affine-translate-frac-min", type=float, default=None)
    aux.add_argument("--aux-affine-translate-frac-max", type=float, default=None)
    aux.add_argument("--aux-affine-scale-min", type=float, default=None)
    aux.add_argument("--aux-affine-scale-max", type=float, default=None)
    aux.add_argument("--aux-allow-affine", type=str, default=None,
                     choices=["true", "false"],
                     help="Override allow_affine (true/false). Default None = "
                          "keep default_aux_spec().allow_affine.")
    return p


# ---------------------------------------------------------------------------
# Aux DistortionSpec construction from CLI
# ---------------------------------------------------------------------------


_AUX_FLAG_NAMES = (
    "aux_noise_sigma_min", "aux_noise_sigma_max",
    "aux_brightness_jitter", "aux_contrast_jitter",
    "aux_blur_sigma_min", "aux_blur_sigma_max",
    "aux_motion_blur_prob", "aux_motion_blur_length_min", "aux_motion_blur_length_max",
    "aux_occlusion_prob", "aux_occlusion_max_frac",
    "aux_occlusion_num_boxes_min", "aux_occlusion_num_boxes_max",
    "aux_affine_angle_min", "aux_affine_angle_max",
    "aux_affine_translate_frac_min", "aux_affine_translate_frac_max",
    "aux_affine_scale_min", "aux_affine_scale_max",
    "aux_allow_affine",
)


def _resolve_range(
    cli_min: Optional[float],
    cli_max: Optional[float],
    default: tuple,
    *,
    name: str,
    int_cast: bool = False,
) -> tuple:
    """Merge a (min, max) range. Each half independently inherits the default."""
    lo = default[0] if cli_min is None else cli_min
    hi = default[1] if cli_max is None else cli_max
    if int_cast:
        lo = int(lo)
        hi = int(hi)
    if lo > hi:
        raise ValueError(
            f"--{name}-min ({lo}) must be <= --{name}-max ({hi})."
        )
    return (lo, hi)


def _check_prob(val: Optional[float], name: str) -> None:
    if val is None:
        return
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"--{name} must be in [0, 1]; got {val}.")


def _build_aux_spec_from_args(args: argparse.Namespace) -> Optional[DistortionSpec]:
    """Build an aux ``DistortionSpec`` from CLI overrides.

    Returns ``None`` when no ``--aux-*`` flag was supplied (preserving the
    legacy default-spec path). Otherwise starts from ``default_aux_spec()``
    and overrides only the user-supplied fields.

    Raises
    ------
    ValueError
        On invalid ranges (min > max), out-of-domain probabilities, or
        non-positive motion-blur lengths.
    """
    # "Any flag set?" detection.
    if all(getattr(args, n, None) is None for n in _AUX_FLAG_NAMES):
        return None

    base = default_aux_spec()

    # Probabilities
    _check_prob(args.aux_motion_blur_prob, "aux-motion-blur-prob")
    _check_prob(args.aux_occlusion_prob, "aux-occlusion-prob")

    # Noise sigma must be >= 0
    if args.aux_noise_sigma_min is not None and args.aux_noise_sigma_min < 0:
        raise ValueError(f"--aux-noise-sigma-min must be >= 0; got {args.aux_noise_sigma_min}.")
    if args.aux_noise_sigma_max is not None and args.aux_noise_sigma_max < 0:
        raise ValueError(f"--aux-noise-sigma-max must be >= 0; got {args.aux_noise_sigma_max}.")

    noise_range = _resolve_range(
        args.aux_noise_sigma_min, args.aux_noise_sigma_max,
        base.noise_sigma_range, name="aux-noise-sigma",
    )
    blur_range = _resolve_range(
        args.aux_blur_sigma_min, args.aux_blur_sigma_max,
        base.blur_sigma_range, name="aux-blur-sigma",
    )
    motion_len_range = _resolve_range(
        args.aux_motion_blur_length_min, args.aux_motion_blur_length_max,
        base.motion_blur_length_range, name="aux-motion-blur-length",
        int_cast=True,
    )
    if motion_len_range[0] < 1:
        raise ValueError(
            f"--aux-motion-blur-length-min must be >= 1; got {motion_len_range[0]}."
        )
    occ_boxes_range = _resolve_range(
        args.aux_occlusion_num_boxes_min, args.aux_occlusion_num_boxes_max,
        base.occlusion_num_boxes_range, name="aux-occlusion-num-boxes",
        int_cast=True,
    )
    if occ_boxes_range[0] < 1:
        raise ValueError(
            f"--aux-occlusion-num-boxes-min must be >= 1; got {occ_boxes_range[0]}."
        )
    affine_angle_range = _resolve_range(
        args.aux_affine_angle_min, args.aux_affine_angle_max,
        base.affine_angle_range_deg, name="aux-affine-angle",
    )
    affine_translate_range = _resolve_range(
        args.aux_affine_translate_frac_min, args.aux_affine_translate_frac_max,
        base.affine_translate_frac_range, name="aux-affine-translate-frac",
    )
    affine_scale_range = _resolve_range(
        args.aux_affine_scale_min, args.aux_affine_scale_max,
        base.affine_scale_range, name="aux-affine-scale",
    )

    # allow_affine override (string → bool)
    if args.aux_allow_affine is None:
        allow_affine = base.allow_affine
    else:
        allow_affine = (args.aux_allow_affine.lower() == "true")

    # Brightness / contrast / occlusion-max-frac
    brightness = base.brightness_jitter if args.aux_brightness_jitter is None else args.aux_brightness_jitter
    contrast = base.contrast_jitter if args.aux_contrast_jitter is None else args.aux_contrast_jitter
    motion_prob = base.motion_blur_prob if args.aux_motion_blur_prob is None else args.aux_motion_blur_prob
    occ_prob = base.occlusion_prob if args.aux_occlusion_prob is None else args.aux_occlusion_prob
    occ_max_frac = base.occlusion_max_frac if args.aux_occlusion_max_frac is None else args.aux_occlusion_max_frac

    if brightness < 0:
        raise ValueError(f"--aux-brightness-jitter must be >= 0; got {brightness}.")
    if contrast < 0:
        raise ValueError(f"--aux-contrast-jitter must be >= 0; got {contrast}.")
    if not (0.0 <= occ_max_frac <= 1.0):
        raise ValueError(f"--aux-occlusion-max-frac must be in [0, 1]; got {occ_max_frac}.")

    spec = DistortionSpec(
        noise_sigma_range=noise_range,
        brightness_jitter=brightness,
        contrast_jitter=contrast,
        blur_sigma_range=blur_range,
        motion_blur_prob=motion_prob,
        motion_blur_length_range=motion_len_range,
        occlusion_prob=occ_prob,
        occlusion_max_frac=occ_max_frac,
        occlusion_num_boxes_range=occ_boxes_range,
        affine_angle_range_deg=affine_angle_range,
        affine_translate_frac_range=affine_translate_range,
        affine_scale_range=affine_scale_range,
        allow_affine=allow_affine,
    )

    # Warn if affine ranges supplied but allow_affine=False.
    if not allow_affine:
        affine_flags_set = any(
            getattr(args, n) is not None for n in (
                "aux_affine_angle_min", "aux_affine_angle_max",
                "aux_affine_translate_frac_min", "aux_affine_translate_frac_max",
                "aux_affine_scale_min", "aux_affine_scale_max",
            )
        )
        if affine_flags_set:
            logger.warning(
                "Aux affine range flags supplied but --aux-allow-affine=false; "
                "affine ranges will be inert at sample time."
            )

    return spec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> None:
    args = build_argparser().parse_args(argv)

    # Reproducibility
    set_seeds(args.seed)

    # GPU
    if setup_gpu is not None:
        setup_gpu(gpu_id=args.gpu)

    # Mixed precision
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (fp16) enabled.")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # --- Aux DistortionSpec (CLI override path) ---
    # DECISION plan_2026-05-19_b225c8df/D-001
    # Aux-only knob surface; anchor remains at `default_anchor_spec()` because
    # anchor defines the test-time input distribution (Finding 04, HARD
    # constraint). Widening anchor here would change the deployed-model
    # contract, not just training difficulty. See
    # plans/plan_2026-05-19_b225c8df/decisions.md D-001.
    aux_spec_override = _build_aux_spec_from_args(args)
    if aux_spec_override is not None:
        logger.info(
            f"Aux DistortionSpec overridden by CLI: {asdict(aux_spec_override)}"
        )

    # --- Data ---
    if args.dataset == "coco":
        train_ds, val_ds = build_coco_burst_dp_datasets(
            coco_root=args.coco_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            aux_spec=aux_spec_override,
            seed=args.seed,
        )
        fidelity_only = False
    elif args.dataset == "div2k":
        train_ds, val_ds = build_div2k_burst_dp_datasets(
            div2k_root=args.div2k_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            aux_spec=aux_spec_override,
            seed=args.seed,
        )
        fidelity_only = True
    elif args.dataset == "vggface2":
        train_ds, val_ds = build_vggface2_burst_dp_datasets(
            vggface2_root=args.vggface2_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            n_max=args.n_max,
            n_min=args.n_min,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            workers=args.workers,
            aux_spec=aux_spec_override,
            seed=args.seed,
        )
        fidelity_only = True
    else:  # pragma: no cover — argparse choices guard this
        raise ValueError(f"Unknown --dataset: {args.dataset!r}")

    logger.info(f"Train probe: {train_ds.probe()}")

    # --- Model ---
    model = create_burst_dp(
        preset=args.preset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        n_max=args.n_max,
    )

    # --- Loss + optimizer ---
    losses: Dict[str, Any] = {
        "recon": charbonnier_loss(),
        "segmentation": sparse_seg_ce_loss(),
    }
    loss_weights: Dict[str, float] = {
        "recon": args.loss_recon,
        "segmentation": args.loss_seg,
    }

    steps_per_epoch = max(1, len(train_ds))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    cosine_steps = max(1, total_steps - warmup_steps)

    # DECISION plan_2026-05-20_b8f8df89/D-001
    # `initial_learning_rate` is the START of the warmup ramp, not the post-
    # warmup LR. It MUST be 0.0 (or near-zero) for warmup to actually ramp:
    # setting it equal to `warmup_target` makes the "warmup" a flat plateau
    # with no ramp-up — the cold-start protection a from-scratch ViT needs.
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=cosine_steps,
        warmup_target=args.lr,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
    )

    # Per-head metrics (see research/benchmarks/VISION_BENCHMARKS.md / research/benchmarks/METRICS.md):
    #   recon        -> PSNR + SSIM        (fidelity, research/benchmarks/METRICS.md §13)
    #   segmentation -> pixel-acc + mIoU   (research/benchmarks/METRICS.md §3; mIoU is the
    #                                       primary semseg metric).
    # MeanIoU(sparse_y_true=True, sparse_y_pred=False) takes argmax over
    # the logits' last axis internally, so passing the raw seg logits is
    # correct.
    metrics: Dict[str, Any] = {
        "recon": [PsnrMetric(max_val=1.0), SsimMetric(max_val=1.0)],
        "segmentation": [
            keras.metrics.SparseCategoricalAccuracy(name="pixel_acc"),
            keras.metrics.MeanIoU(
                num_classes=DEFAULT_NUM_SEG_CLASSES,
                sparse_y_true=True,
                sparse_y_pred=False,
                name="miou",
            ),
        ],
    }

    # DECISION plan_2026-05-19_64f2a17b/D-001
    # Fidelity-only datasets (DIV2K, VGG-Face2) lack segmentation labels.
    # The :class:`BurstDP` model is hardcoded dual-head; we keep the seg
    # head in the graph but zero its loss weight and drop its metrics.
    # Trade-off: ~5-15% wasted compute on the seg head at the cost of
    # zero blast radius on `BurstDP`, `BurstDPConfig`, and existing
    # checkpoints. Pivot fallback is Option B (add an `enable_segmentation`
    # config flag) — see plans/plan_2026-05-19_64f2a17b/decisions.md D-001.
    if fidelity_only:
        if args.loss_seg != 0.0:
            logger.warning(
                f"--dataset {args.dataset} is fidelity-only: overriding "
                f"--loss-seg {args.loss_seg} → 0.0 and dropping seg metrics."
            )
        loss_weights["segmentation"] = 0.0
        metrics.pop("segmentation", None)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    # --- Callbacks ---
    # `TerminateOnNaN` MUST be first: the masked-softmax path in the fusion
    # block can produce NaN if the any-valid gate breaks, and we want training
    # to halt before any subsequent checkpoint commits a corrupted model.
    ckpt_path = str(out_dir / "burst_dp_best.keras")
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]
    if args.viz_every_steps > 0 or args.viz_every_epochs > 0:
        callbacks.append(
            BurstDPVisualizationCallback(
                val_dataset=val_ds,
                output_dir=str(out_dir),
                every_steps=args.viz_every_steps,
                every_epochs=args.viz_every_epochs,
                num_samples=args.viz_num_samples,
                seed=args.seed,
            )
        )
    # Per-epoch loss/metric curve PNGs grouped into loss / segmentation / other
    # (train + val on each axis). Same callback used by train_coco_multitask.
    callbacks.append(
        TrainingCurvesCallback(
            output_dir=str(out_dir / "training_curves"),
            frequency=max(1, args.viz_every_epochs),
        )
    )
    callbacks.extend([
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tb")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ])

    # Persist run config
    run_cfg = {
        "args": vars(args),
        "model_config": model.config.to_dict(),
        "anchor_spec": asdict(train_ds.cfg.anchor_spec),
        "aux_spec": asdict(train_ds.cfg.aux_spec),
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "cosine_steps": cosine_steps,
        "fidelity_only": fidelity_only,
        "loss_weights": loss_weights,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # --- Fit ---
    # PyDataset passes directly to model.fit; Keras 3 iterates with the
    # configured `workers` + `use_multiprocessing` from the loader's
    # __init__ (see D-001 anchor at top of file).
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model.
    final_path = str(out_dir / "burst_dp_final.keras")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
