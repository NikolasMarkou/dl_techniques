"""Bias-free, strictly-homogeneous Clifford U-Net denoiser trainer with a frozen
Gabor stem and a noise-sigma curriculum.

Trains ``create_cliffordunet_denoiser`` (bias-free, degree-1-homogeneous Clifford
geometric-block U-Net) with an optional NON-LEARNABLE Gabor depthwise stem on
DIV2K + COCO. Patches of 256x256 are sampled with geometric augmentation
(flips + rot90), normalized to ``[-0.5, +0.5]`` (required for bias-free /
scaling-invariant denoising), and corrupted with additive Gaussian noise whose
per-image sigma is drawn from ``[sigma_min, sigma_max]``. The upper bound
``sigma_max`` is a live ``tf.Variable`` widened every epoch by
``NoiseSigmaCurriculumCallback`` (curriculum: start with low noise, progressively
widen the spread).

This is the Clifford sibling of ``train_convunext_denoiser.py``. The full data
pipeline, noise curriculum, callbacks, visualization, Miyasawa machinery,
self-iterate, WW-PGD and smoke mode are reused verbatim; only the model built by
``build_model`` (and the ConvNeXt-only config/CLI surface) differ.

Design notes (plan_2026-07-01_6dc255c1):
- The noise function reads a captured ``tf.Variable`` (validated: a Variable read
  inside ``tf.data.map`` reflects per-epoch ``.assign``). This is a DELIBERATE
  Variable-backed variant of ``add_noise_to_patch``, not a 4th blind copy.
- The Clifford block is pinned to its degree-1-homogeneous configuration by the
  factory (``bias_free_batch_norm`` detail stream (degree-1) times a
  ``zero_centered_rms_norm`` + ``ctx_mode="abs"`` context stream (degree-0),
  LeakyReLU everywhere, sigmoid gate removed). The bilinear geometric product of a
  degree-1 and a degree-0 stream is degree-1, so ``f(alpha*x) = alpha*f(x)`` holds
  in the ``[-0.5, 0.5]`` operating regime (D-004/D-005/D-006 of the plan).
- Weighted COCO+DIV2K sourcing via ``select_weighted_image_paths`` prevents COCO's
  ~118K images from drowning DIV2K's ~800.

Reference PSNR baselines / SOTA (additive white Gaussian noise denoising)
------------------------------------------------------------------------
Published average PSNR (dB) on the standard AWGN denoising benchmarks. These are
REFERENCE TARGETS for interpreting this trainer's val-PSNR, NOT a like-for-like
leaderboard (see caveats below).

Noise-scale note: benchmark sigma is on the [0, 255] pixel scale. This trainer
adds noise in the [-0.5, +0.5] normalized space, so::

    sigma_255  =  sigma_here * 255.0

i.e. this run's curriculum ``sigma_max 0.025 -> 0.25`` corresponds to
``sigma_255 ~= 6.4 -> 63.75`` — it spans (and exceeds) the classic 15/25/50
benchmark regimes as a single *blind* model. PSNR is scale-invariant, so dB
computed here with ``max_val=1.0`` on [-0.5,+0.5] images is directly comparable to
the published ``max_val=255`` numbers.

Grayscale (1-ch), Set12 / BSD68:
    sigma=15:  DnCNN 32.86/31.73 | FFDNet 32.75/31.63 | DRUNet 33.25/31.91 |
               SwinIR 33.36/31.97 | Restormer 33.42/31.96
    sigma=25:  DnCNN 30.44/29.23 | FFDNet 30.43/29.19 | DRUNet 30.94/29.48 |
               SwinIR 31.01/29.50 | Restormer 31.08/29.52
    sigma=50:  DnCNN 27.18/26.23 | FFDNet 27.32/26.29 | DRUNet 27.90/26.59 |
               SwinIR 27.91/26.58 | Restormer 28.00/26.62

Color (3-ch) average PSNR (dB), Table 2 of Zhang et al. SCUNet (arXiv:2203.13278):
    | dataset  | s | DnCNN | FFDNet | DRUNet | SwinIR | Restormer | SCUNet |
    |----------|---|-------|--------|--------|--------|-----------|--------|
    | CBSD68   |15 | 33.90 | 33.87  | 34.30  | 34.42  | 34.40     | 34.40  |
    | CBSD68   |25 | 31.24 | 31.21  | 31.69  | 31.78  | 31.79     | 31.79  |
    | CBSD68   |50 | 27.95 | 27.96  | 28.51  | 28.56  | 28.60     | 28.61  |
    | Kodak24  |15 | 34.60 | 34.63  | 35.31  | 35.34  | 35.47     | 35.34  |
    | Kodak24  |25 | 32.14 | 32.13  | 32.89  | 32.89  | 33.04     | 32.92  |
    | Kodak24  |50 | 28.95 | 28.98  | 29.86  | 29.79  | 30.01     | 29.87  |
    | McMaster |15 | 33.45 | 34.66  | 35.40  | 35.61  | 35.61     | 35.60  |
    | McMaster |25 | 31.52 | 32.35  | 33.14  | 33.20  | 33.34     | 33.34  |
    | McMaster |50 | 28.62 | 29.18  | 30.08  | 30.22  | 30.30     | 30.29  |
    | Urban100 |15 | 32.98 | 33.83  | 34.81  | 35.13  | 35.13     | 35.18  |
    | Urban100 |25 | 30.81 | 31.40  | 32.60  | 32.90  | 32.96     | 33.03  |
    | Urban100 |50 | 27.59 | 28.05  | 29.61  | 29.82  | 30.02     | 30.14  |

Bias-free reference (most architecturally comparable to this model):
    Mohan et al., "Robust and Interpretable Blind Image Denoising via Bias-Free
    CNNs", ICLR 2020 (arXiv:1906.05478). BF-CNN MATCHES its biased DnCNN-style
    counterpart WITHIN the training noise range (e.g. ~29.2 dB on BSD68 s=25) but
    generalizes far better OUTSIDE it — a biased CNN collapses on unseen noise
    levels while the bias-free model degrades gracefully. That cross-noise
    robustness (not peak in-range PSNR) is the property this bias-free + noise-
    curriculum trainer targets.

Caveats (read before comparing):
  1. Different test set: numbers above are on Set12/BSD68/CBSD68/Kodak24/McMaster/
     Urban100. This trainer reports val-PSNR on DIV2K-validation patches — easier,
     cleaner content than Urban100, so absolute dB is NOT directly comparable.
  2. Blind wide-range vs fixed-sigma specialists: most SOTA rows are sigma-specific
     (or narrow blind). A single model trained over sigma_255 ~6..64 with a
     curriculum is solving a harder, broader task; expect lower peak PSNR at any
     single sigma than a specialist tuned for it.
  3. Capacity/training budget differ (these are trained to convergence on large
     corpora). Treat the table as orientation, not a target to "beat".

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_cliffordunet_denoiser \\
        --variant base --epochs 100 --batch-size 16 --patch-size 256 --gpu 1

    # Quick mechanism check (tiny, 2 epochs):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_cliffordunet_denoiser --smoke

Mixed precision (``--mixed-precision``, OFF by default): correct and serializes, but
the decoder's bilinear-upsample gradient (``ResizeBilinearGrad``) emits fp32, which
XLA's strict dtype checker rejects, so the mixed-precision path must run with
``jit_compile=False`` -- and for this conv-heavy U-Net, losing XLA fusion typically
outweighs the fp16 tensor-core gain. Kept as an option for higher-res / other GPUs /
a future XLA-clean upsample.
"""

import keras
import argparse
from dataclasses import dataclass
from typing import List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.bias_free_denoisers.bfcliffordunet import (
    create_cliffordunet_denoiser,
    CLIFFORDUNET_CONFIGS,
)

# Shared bfunet trainer substrate. The data / curriculum / callback / dashboard /
# self-iterate / train-orchestration skeleton lives ONCE in train.bfunet.common; this
# trainer imports it and mirrors the ConvUNeXt sibling's re-export shape (so a future
# clifford sibling test can import these names from this module path).
# `import ... as common` is used for common.train().
from train.bfunet.common import (
    BFUnetTrainingConfig,
    decode_full_image, random_crop_patch, load_and_preprocess_image,
    collect_training_paths, create_dataset, make_curriculum_noise_fn,
    build_self_iterate_pool, create_self_iterate_dataset,
    _denorm, render_training_dashboard, _mean_psnr, denoise_k_passes,
    multi_pass_psnr, build_fixed_val_batch, build_dashboard_from_dir,
    _read_current_lr, DenoisingVisualizationCallback, LRLoggerCallback,
    add_common_arguments, reject_self_iterate_with_nonadditive,
    _homogeneity_probe,
)
from train.bfunet import common as common


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig(BFUnetTrainingConfig):
    """Configuration for the bias-free homogeneous Clifford U-Net denoiser trainer.

    Subclasses the shared ``BFUnetTrainingConfig`` (Data / Memory / Noise-curriculum /
    shared U-Net topology / Training / Optimization / Self-iterate / WW-PGD / init_from /
    Analysis / Output) and adds ONLY the Clifford-specific fields + model validation.
    """

    # Overridable experiment-name prefix (plain class attr, NOT a dataclass field).
    experiment_prefix = "cliffordunet_denoiser_"

    # Clifford geometric-product base channel-shift offsets (ints >= 1). None -> use the
    # variant default from CLIFFORDUNET_CONFIGS. Sized per U-Net level so every kept shift
    # satisfies s < channels at that level (the factory clamps + asserts). Override to
    # experiment with different geometric-product reach.
    shifts: Optional[List[int]] = None
    # Clifford algebra components for the local interaction: "inner" | "wedge" | "full".
    # Matches the factory default ("full").
    cli_mode: str = "full"
    # Context mode. HOMOGENEITY-CRITICAL: "abs" keeps the context (z_ctx) stream degree-0
    # so the bilinear geometric product (degree-1 z_det (X) degree-0 z_ctx) is degree-1 and
    # the block is degree-1 homogeneous. "diff" subtracts the degree-1 z_det from z_ctx,
    # remixing the degrees and making the product degree-2 (f(ax) ~ a^2 f(x)) — it BREAKS
    # strict Miyasawa homogeneity (empirically confirmed, bfcliffordunet D-004/D-005). Keep
    # "abs" unless you explicitly do not need homogeneity.
    ctx_mode: str = "abs"
    # LayerScale gamma init for the Clifford GatedGeometricResidual. Matches the factory
    # default (1e-5); larger values unmask in-block behavior earlier in training.
    layer_scale_init: float = 1e-5
    # filter_multiplier (per-encoder-level channel-growth multiplier) is inherited from
    # BFUnetTrainingConfig (float, default 2.0) and surfaced via the shared
    # add_common_arguments --filter-multiplier CLI; build_model passes it to the factory.

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in CLIFFORDUNET_CONFIGS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: {list(CLIFFORDUNET_CONFIGS)}"
            )
        if self.cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"cli_mode must be 'inner', 'wedge', or 'full', got {self.cli_mode!r}"
            )
        if self.ctx_mode not in ("diff", "abs"):
            raise ValueError(
                f"ctx_mode must be 'diff' or 'abs', got {self.ctx_mode!r}"
            )


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def build_model(config: TrainingConfig) -> keras.Model:
    """Build the bias-free homogeneous Clifford U-Net denoiser from the variant config."""
    cfg = CLIFFORDUNET_CONFIGS[config.variant].copy()
    cfg.pop("description", None)
    if config.initial_filters is not None:
        cfg["initial_filters"] = config.initial_filters  # override variant level-0 width
    if config.shifts is not None:
        cfg["shifts"] = list(config.shifts)  # override variant geometric-product shifts
    if config.depth is not None:
        cfg["depth"] = config.depth  # override variant number of U-Net levels
    if config.blocks_per_level is not None:
        cfg["blocks_per_level"] = config.blocks_per_level  # override variant blocks/level
    # No-projection Gabor stem requires an exact channel match; fail early with a clear
    # message before the factory builds (the factory also validates as a backstop).
    if config.use_gabor_stem and not config.gabor_stem_projection:
        gabor_out = config.channels * config.gabor_filters
        if gabor_out != cfg["initial_filters"]:
            raise ValueError(
                f"--no-gabor-projection requires channels({config.channels}) * "
                f"gabor_filters({config.gabor_filters}) = {gabor_out} to equal "
                f"initial_filters({cfg['initial_filters']}). Pass --initial-filters "
                f"{gabor_out} (or adjust --gabor-filters)."
            )
    # Resolve the final-projection group count: -1 means one group per output channel
    # (groups == channels), so each output channel reads a disjoint feature group.
    final_projection_groups = (
        config.channels if config.final_projection_groups == -1
        else config.final_projection_groups
    )
    if final_projection_groups > 1 and (
        cfg["initial_filters"] % final_projection_groups != 0
        or config.channels % final_projection_groups != 0
    ):
        raise ValueError(
            f"--final-projection-groups resolved to {final_projection_groups}, which must "
            f"divide BOTH initial_filters({cfg['initial_filters']}) and "
            f"channels({config.channels})."
        )
    input_shape = (config.patch_size, config.patch_size, config.channels)
    # NOTE: the Clifford block's activation/normalization/gate are PINNED to the
    # degree-1-homogeneous configuration inside create_cliffordunet_denoiser
    # (bias_free_batch_norm z_det (X) zero_centered_rms_norm z_ctx, LeakyReLU
    # everywhere, gate removed); the trainer does NOT pass an activation. See
    # bfcliffordunet _homogeneous_block_kwargs (D-005).
    return create_cliffordunet_denoiser(
        input_shape=input_shape,
        filter_multiplier=config.filter_multiplier,
        cli_mode=config.cli_mode,
        ctx_mode=config.ctx_mode,
        layer_scale_init=config.layer_scale_init,
        use_gabor_stem=config.use_gabor_stem,
        gabor_filters=config.gabor_filters,
        gabor_kernel_size=config.gabor_kernel_size,
        gabor_stem_projection=config.gabor_stem_projection,
        use_laplacian_pyramid=config.use_laplacian_pyramid,
        high_freq_blocks=config.high_freq_blocks,
        zero_pad_channels=config.zero_pad_channels,
        final_projection_groups=final_projection_groups,
        downsample_pool_type=config.downsample_pool_type,
        enable_deep_supervision=config.enable_deep_supervision,
        expose_bottleneck=config.expose_bottleneck,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name=f"cliffordunet_denoiser_{config.variant}",
        **cfg,
    )


def verify_bias_free(model: keras.Model) -> None:
    """Log a bias-free compliance check (informational)."""
    # The Clifford block normalizes with bias_free_batch_norm (z_det) and
    # zero_centered_rms_norm (z_ctx) and never instantiates keras LayerNormalization
    # or GlobalResponseNormalization, so a static offender scan only needs the
    # use_bias check; degree-1 homogeneity breaks (norm/activation/context) are caught
    # by the numeric black-box probe below, not by layer-type inspection.
    offenders = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            offenders.append(layer.name)
    if offenders:
        logger.warning(
            f"Bias-free check: {len(offenders)} layer(s) carry bias/centering: "
            f"{offenders[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")

    # Numeric black-box degree-1 homogeneity probe (shared; informational, NEVER raises).
    # Detects f(a*x) != a*f(x) breaks the static offender scan above cannot see. Lives in
    # common.py (D-003, D-005). NOTE: a PASS on an UNTRAINED model does NOT prove
    # homogeneity — LayerScale gamma=1e-5 makes each residual branch near-identity at init,
    # masking in-block norm/activation degree-0 breaks until training grows gamma (D-005).
    _homogeneity_probe(model)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> keras.Model:
    """Train the bias-free homogeneous Clifford U-Net denoiser with the noise curriculum.

    Thin delegator: all orchestration lives in ``common.train``; this trainer injects
    its own ``build_model`` + ``verify_bias_free`` plus the CliffordUNet label,
    results-dir prefix, and bottleneck-name prefix (the bottleneck-health monitor brands
    its artifacts "cliffordunet_bottleneck").
    """
    return common.train(
        config,
        build_model,
        verify_bias_free,
        model_label="CliffordUNet",
        results_dir_prefix="cliffordunet_denoiser",
        bottleneck_name_prefix="cliffordunet_bottleneck",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Define and parse the trainer CLI surface.

    The shared flags (noise curriculum, self-iterate, ww-pgd, visualization,
    smoke/dashboard run modes, data/optimization knobs) come from
    ``add_common_arguments``; this trainer adds only ``--variant`` and the Clifford
    knobs. Returns the parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Train bias-free homogeneous CliffordUNet denoiser (Gabor stem + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=list(CLIFFORDUNET_CONFIGS), default="base")
    parser.add_argument("--shifts", type=int, nargs="+", default=None,
                        help="Override the variant's Clifford geometric-product base shift "
                             "offsets (ints >= 1, e.g. --shifts 1 2 3). Sized per U-Net level "
                             "so every kept shift satisfies s < channels. Default: variant value.")
    parser.add_argument("--cli-mode", type=str, choices=["inner", "wedge", "full"],
                        default="full",
                        help="Clifford algebra components for the local interaction. "
                             "Default 'full'.")
    parser.add_argument("--ctx-mode", type=str, choices=["diff", "abs"], default="abs",
                        help="Clifford context mode. HOMOGENEITY-CRITICAL: 'abs' (default) "
                             "keeps the context stream degree-0 so the block is degree-1 "
                             "homogeneous (strict Miyasawa). 'diff' subtracts z_det and makes "
                             "the geometric product degree-2, breaking homogeneity (D-004/D-005).")
    parser.add_argument("--layer-scale-init", type=float, default=1e-5,
                        help="Initial LayerScale gamma for the Clifford GatedGeometricResidual. "
                             "Default 1e-5 (matches the factory).")
    args = parser.parse_args()
    reject_self_iterate_with_nonadditive(parser, args)
    return args


def main():
    """Parse args, handle the ``--dashboard`` early-exit and ``--smoke`` config,
    build the ``TrainingConfig``, and run ``train()``."""
    args = parse_arguments()

    # Standalone dashboard rebuild (no training, no GPU needed): regenerate the
    # combined per-epoch dashboard from an experiment dir's CSV + config.
    if args.dashboard:
        build_dashboard_from_dir(args.dashboard)
        return

    setup_gpu(gpu_id=args.gpu)

    if args.smoke:
        # Mechanism check: tiny, fast, constant LR (avoid cosine collapse at 2 epochs).
        config = TrainingConfig(
            variant="tiny",
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            zero_pad_channels=args.zero_pad_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_deep_supervision=args.deep_supervision,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            cli_mode=args.cli_mode,
            ctx_mode=args.ctx_mode,
            layer_scale_init=args.layer_scale_init,
            filter_multiplier=args.filter_multiplier,
            shifts=args.shifts,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
            gabor_filters=8,
            epochs=2,
            curriculum_epochs=2,
            batch_size=2,
            patch_size=64,
            channels=3,
            patches_per_image=2,
            max_train_files=8,
            max_val_files=8,  # >= viz_samples so the smoke grid also shows 8 columns
            steps_per_epoch=3,
            validation_steps=2,
            warmup_epochs=0,
            # Mechanism check only; cosine_decay is the only supported schedule
            # (builder has no 'constant'). 2-epoch PSNR quality is NOT asserted.
            lr_schedule_type="cosine_decay",
            learning_rate=1e-3,
            weight_decay=args.weight_decay,
            sigma_max_start=0.025,
            sigma_max_end=0.25,
            curriculum_schedule="linear",
            noise_type=(
                "composite" if args.composite_noise
                else "multiplicative" if args.multiplicative_noise
                else "additive"
            ),
            composite_additive_ratio=args.composite_additive_ratio,
            # Self-iterate: SMALL pool so a smoke run exercises >=1 regeneration cheaply.
            # Cap at 32 (>= smoke batch_size 2) and force regen_freq=1 so the single
            # epoch boundary triggers a regeneration; mix_ratio is honored from args.
            self_iterate=args.self_iterate,
            self_iterate_pool_size=min(args.self_iterate_pool_size, 32),
            self_iterate_regen_freq=1,
            self_iterate_mix_ratio=args.self_iterate_mix_ratio,
            ww_pgd=args.ww_pgd,
            ww_pgd_log_alpha=args.ww_pgd_log_alpha,
            init_from=args.init_from,
            viz_freq=1,
            viz_samples=args.viz_samples,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "cliffordunet_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            zero_pad_channels=args.zero_pad_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            cli_mode=args.cli_mode,
            ctx_mode=args.ctx_mode,
            layer_scale_init=args.layer_scale_init,
            filter_multiplier=args.filter_multiplier,
            shifts=args.shifts,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
            gabor_filters=args.gabor_filters,
            gabor_stem_projection=not args.no_gabor_projection,
            initial_filters=args.initial_filters,
            final_projection_groups=args.final_projection_groups,
            enable_deep_supervision=args.deep_supervision,
            epochs=args.epochs,
            curriculum_epochs=args.curriculum_epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            channels=args.channels,
            patches_per_image=args.patches_per_image,
            mixed_precision=args.mixed_precision,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,  # None -> 10% of epochs
            sigma_max_start=args.sigma_max_start,
            sigma_max_end=args.sigma_max_end,
            curriculum_schedule=args.curriculum_schedule,
            noise_type=(
                "composite" if args.composite_noise
                else "multiplicative" if args.multiplicative_noise
                else "additive"
            ),
            composite_additive_ratio=args.composite_additive_ratio,
            self_iterate=args.self_iterate,
            self_iterate_pool_size=args.self_iterate_pool_size,
            self_iterate_regen_freq=args.self_iterate_regen_freq,
            self_iterate_mix_ratio=args.self_iterate_mix_ratio,
            ww_pgd=args.ww_pgd,
            ww_pgd_log_alpha=args.ww_pgd_log_alpha,
            init_from=args.init_from,
            max_train_files=args.max_train_files or 10000,
            max_val_files=args.max_val_files or 500,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps if args.validation_steps is not None else 100,
            viz_freq=args.viz_freq,
            viz_samples=args.viz_samples,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

    # --ww-pgd-log-alpha implies --ww-pgd: enabling the alpha trajectory log turns
    # on the projection it instruments (so the flag is usable standalone).
    if config.ww_pgd_log_alpha:
        config.ww_pgd = True

    logger.info(
        f"Config: variant={config.variant} (cli={config.cli_mode}, ctx={config.ctx_mode}), "
        f"gabor_stem={config.use_gabor_stem}, epochs={config.epochs}, "
        f"patch={config.patch_size}x{config.channels}, "
        f"sigma_max {config.sigma_max_start}->{config.sigma_max_end} "
        f"({config.curriculum_schedule})"
    )

    try:
        train(config)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
