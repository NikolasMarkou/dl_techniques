"""Train a bias-free plain U-Net denoiser (the baseline sibling of the ConvUNeXt trainer).

This trains ``create_bfunet_denoiser`` — a classic bias-free U-Net (plain
``BiasFreeConv2D`` / ``BiasFreeResidualBlock`` blocks, no ConvNeXt inverted
bottleneck) — through the SAME shared substrate (``train.bfunet.common``) as the
ConvUNeXt and Clifford trainers: identical data pipeline, noise curriculum, callbacks,
dashboard, self-iterate, and eval. It is the apples-to-apples baseline: same training
machinery, same optional infrastructure features (Gabor stem, Laplacian pyramid,
zero-pad channel matching, pooling-type, expose-bottleneck, block-normalization choice,
grouped final projection, dropout) — only the block internals differ.

Bias-free / degree-1 homogeneity is preserved: ``final_activation='linear'`` and every
conv/norm is bias-free (``use_bias=False``, ``center=False``), so the residual is an
interpretable scaled score (Miyasawa/Tweedie) and one model generalizes across noise
levels it never saw.

Usage::

    # Base U-Net denoiser, full run
    MPLBACKEND=Agg python -m train.bfunet.train_unet_denoiser \
        --variant base --epochs 100 --batch-size 4 --gpu 1

    # With the frozen Gabor stem + Laplacian pyramid (ConvUNeXt-parity features)
    MPLBACKEND=Agg python -m train.bfunet.train_unet_denoiser \
        --variant base --laplacian-pyramid --block-normalization batchnorm --gpu 1

    # Fast end-to-end mechanism check
    MPLBACKEND=Agg python -m train.bfunet.train_unet_denoiser --smoke
"""

import keras
import argparse
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
    BFUNET_CONFIGS,
)

# Shared bfunet trainer substrate. The data / curriculum / callback / dashboard /
# self-iterate / train-orchestration skeleton lives ONCE in train.bfunet.common; this
# trainer imports it and RE-EXPORTS the frozen-API names so
# `from train.bfunet.train_unet_denoiser import <name>` keeps resolving (mirrors the
# ConvUNeXt path's test contract). `import ... as common` is used for common.train().
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
    """Configuration for the bias-free plain U-Net denoiser trainer.

    Subclasses the shared ``BFUnetTrainingConfig`` (Data / Memory / Noise-curriculum /
    shared U-Net topology / Training / Optimization / Self-iterate / WW-PGD / init_from /
    Analysis / Output) and adds ONLY the plain-U-Net-specific fields + model validation.
    """

    # Overridable experiment-name prefix (plain class attr, NOT a dataclass field).
    experiment_prefix = "unet_denoiser_"

    # Plain-U-Net-specific model fields.
    use_residual_blocks: bool = True    # BiasFreeResidualBlock vs plain BiasFreeConv2D
    conv_kernel_size: int = 3           # kernel for all non-stem conv blocks
    initial_kernel_size: int = 5        # kernel for the level-0 first conv (the "stem" conv)
    # Standard dropout inside the conv blocks (after activation). 0.0 = OFF / byte-identical.
    dropout_rate: float = 0.0
    # NOTE: block_activation / block_activation_alpha / block_normalization are inherited
    # from BFUnetTrainingConfig (defaults leaky_relu / 0.1 / batchnorm); do not redeclare.

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in BFUNET_CONFIGS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: {list(BFUNET_CONFIGS)}"
            )
        # The plain U-Net requires depth >= 3 (stricter than the shared >= 2 guard).
        if self.depth is not None and self.depth < 3:
            raise ValueError(f"U-Net baseline requires depth >= 3, got {self.depth}")
        # block_normalization membership is validated by BFUnetTrainingConfig.__post_init__.


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def build_model(config: TrainingConfig) -> keras.Model:
    """Build the bias-free plain U-Net denoiser from the variant config."""
    cfg = BFUNET_CONFIGS[config.variant].copy()
    cfg.pop("description", None)
    if config.initial_filters is not None:
        cfg["initial_filters"] = config.initial_filters  # override variant level-0 width
    if config.depth is not None:
        cfg["depth"] = config.depth  # override variant U-Net level count
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
    # Resolve the final-projection group count: -1 means one group per output channel.
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
    # The plain U-Net's filter_multiplier is an int; the shared config carries a float
    # (default 2.0). Require a whole number and cast (silent truncation would be a bug).
    if config.filter_multiplier != round(config.filter_multiplier):
        raise ValueError(
            f"filter_multiplier must be a whole number for the U-Net baseline, "
            f"got {config.filter_multiplier}"
        )
    filter_multiplier = int(round(config.filter_multiplier))
    input_shape = (config.patch_size, config.patch_size, config.channels)
    # The bare "leaky_relu" string resolves to slope 0.2 in Keras; construct an explicit
    # LeakyReLU(negative_slope=alpha) so the 0.1 default is honored. The instance
    # round-trips through .keras via BiasFreeConv2D/BiasFreeResidualBlock get_config/from_config.
    if config.block_activation == "leaky_relu":
        activation = keras.layers.LeakyReLU(negative_slope=config.block_activation_alpha)
    else:
        activation = config.block_activation
    # DECISION plan_2026-07-05_2199bb8e/D-007: map trainer 'batchnorm' -> real homogeneous
    # BiasFreeBatchNorm (plain U-Net). Stock keras BatchNormalization(center=False) subtracts
    # moving_mean and is NOT degree-1 homogeneous; the additive 'bias_free_batchnorm' option
    # (D-006) is variance-only so f(ax)=a*f(x) holds at inference. 'layernorm' passes through
    # unchanged. Do NOT revert to config.block_normalization directly here.
    norm = 'bias_free_batchnorm' if config.block_normalization == 'batchnorm' else config.block_normalization
    return create_bfunet_denoiser(
        input_shape=input_shape,
        filter_multiplier=filter_multiplier,
        kernel_size=config.conv_kernel_size,
        initial_kernel_size=config.initial_kernel_size,
        activation=activation,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        use_residual_blocks=config.use_residual_blocks,
        enable_deep_supervision=config.enable_deep_supervision,
        # ConvUNeXt-parity infrastructure features (all shared BFUnetTrainingConfig fields).
        use_gabor_stem=config.use_gabor_stem,
        gabor_filters=config.gabor_filters,
        gabor_kernel_size=config.gabor_kernel_size,
        gabor_stem_projection=config.gabor_stem_projection,
        use_laplacian_pyramid=config.use_laplacian_pyramid,
        high_freq_blocks=config.high_freq_blocks,
        zero_pad_channels=config.zero_pad_channels,
        downsample_pool_type=config.downsample_pool_type,
        expose_bottleneck=config.expose_bottleneck,
        block_normalization=norm,
        final_projection_groups=final_projection_groups,
        dropout_rate=config.dropout_rate,
        model_name=f"unet_denoiser_{config.variant}",
        **cfg,
    )


def verify_bias_free(model: keras.Model) -> None:
    """Log a bias-free compliance check (informational)."""
    offenders = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            offenders.append(layer.name)
        if isinstance(layer, keras.layers.BatchNormalization) and getattr(
            layer, "center", False
        ):
            offenders.append(f"{layer.name} (BN center=True)")
        if isinstance(layer, keras.layers.LayerNormalization) and getattr(
            layer, "center", False
        ):
            offenders.append(f"{layer.name} (LN center=True)")
    if offenders:
        logger.warning(
            f"Bias-free check: {len(offenders)} layer(s) carry an additive term: "
            f"{offenders}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")
    _homogeneity_probe(model)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> keras.Model:
    """Train the bias-free plain U-Net denoiser with the noise curriculum.

    Thin delegator: all orchestration lives in ``common.train``; this trainer injects its
    own ``build_model`` + ``verify_bias_free`` plus the U-Net label and results-dir prefix.
    Uses the DEFAULT bottleneck name, so no ``bottleneck_name_prefix`` is passed.
    """
    return common.train(
        config,
        build_model,
        verify_bias_free,
        model_label="UNet",
        results_dir_prefix="unet_denoiser",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bias-free plain U-Net denoiser (baseline; Gabor stem + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=list(BFUNET_CONFIGS), default="base")
    parser.add_argument("--no-residual-blocks", action="store_true",
                        help="Use plain BiasFreeConv2D blocks instead of residual blocks.")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Kernel size for all non-stem conv blocks (default 3).")
    parser.add_argument("--initial-kernel-size", type=int, default=5,
                        help="Kernel size for the level-0 first conv (default 5).")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate inside the conv blocks (after activation). "
                             "Default 0.0 = OFF, byte-identical.")
    args = parser.parse_args()
    reject_self_iterate_with_nonadditive(parser, args)
    return args


def main():
    args = parse_arguments()

    # Standalone dashboard rebuild (no training, no GPU needed).
    if args.dashboard:
        build_dashboard_from_dir(args.dashboard)
        return

    setup_gpu(gpu_id=args.gpu)

    if args.smoke:
        # Mechanism check: tiny, fast, constant-ish LR (avoid cosine collapse at 2 epochs).
        config = TrainingConfig(
            variant="tiny",
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            high_freq_blocks=args.high_freq_blocks,
            zero_pad_channels=args.zero_pad_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_deep_supervision=args.deep_supervision,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
            use_residual_blocks=not args.no_residual_blocks,
            conv_kernel_size=args.kernel_size,
            initial_kernel_size=args.initial_kernel_size,
            gabor_filters=8,
            epochs=2,
            curriculum_epochs=2,
            batch_size=2,
            patch_size=64,
            channels=3,
            patches_per_image=2,
            max_train_files=8,
            max_val_files=8,
            steps_per_epoch=3,
            validation_steps=2,
            warmup_epochs=0,
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
            self_iterate=args.self_iterate,
            self_iterate_pool_size=min(args.self_iterate_pool_size, 32),
            self_iterate_regen_freq=1,
            self_iterate_mix_ratio=args.self_iterate_mix_ratio,
            ww_pgd=args.ww_pgd,
            ww_pgd_log_alpha=args.ww_pgd_log_alpha,
            init_from=args.init_from,
            block_activation=args.block_activation,
            block_activation_alpha=args.block_activation_alpha,
            dropout_rate=args.dropout,
            block_normalization=args.block_normalization,
            viz_freq=1,
            viz_samples=args.viz_samples,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "unet_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            high_freq_blocks=args.high_freq_blocks,
            zero_pad_channels=args.zero_pad_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            gabor_filters=args.gabor_filters,
            gabor_stem_projection=not args.no_gabor_projection,
            initial_filters=args.initial_filters,
            filter_multiplier=args.filter_multiplier,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
            final_projection_groups=args.final_projection_groups,
            use_residual_blocks=not args.no_residual_blocks,
            conv_kernel_size=args.kernel_size,
            initial_kernel_size=args.initial_kernel_size,
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
            warmup_epochs=args.warmup_epochs,
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
            block_activation=args.block_activation,
            block_activation_alpha=args.block_activation_alpha,
            dropout_rate=args.dropout,
            block_normalization=args.block_normalization,
            max_train_files=args.max_train_files or 10000,
            max_val_files=args.max_val_files or 500,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps if args.validation_steps is not None else 100,
            viz_freq=args.viz_freq,
            viz_samples=args.viz_samples,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

    if config.ww_pgd_log_alpha:
        config.ww_pgd = True

    logger.info(
        f"Config: variant={config.variant}, residual={config.use_residual_blocks}, "
        f"gabor_stem={config.use_gabor_stem}, norm={config.block_normalization}, "
        f"epochs={config.epochs}, patch={config.patch_size}x{config.channels}, "
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
