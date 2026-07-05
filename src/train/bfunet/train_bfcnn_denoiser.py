"""Train a bias-free flat CNN (BFCNN) denoiser — the flat-ResNet sibling of the U-Net trainer.

This trains ``create_bfcnn_denoiser`` / ``create_bfcnn_variant`` — a bias-free flat
ResNet denoiser (an initial ``BiasFreeConv2D`` stem, a stack of
``BiasFreeResidualBlock`` blocks, a 1x1 bias-free output conv; NO U-Net skips /
encoder-decoder) — through the SAME shared substrate (``train.bfunet.common``) as the
plain U-Net, ConvUNeXt, and Clifford trainers: identical data pipeline, noise
curriculum, callbacks, dashboard, self-iterate, WW-PGD, and eval.

BFCNN is a flat CNN, so it consumes NONE of the shared U-Net topology knobs (Gabor
stem, Laplacian pyramid, depth, filter_multiplier, blocks_per_level, pooling type,
expose-bottleneck, deep-supervision): ``build_model`` passes only the closed BFCNN
kwarg set and the subclass defaults the U-Net-only fields so nothing forwards into the
factory (which would ``TypeError``).

Bias-free / degree-1 homogeneity is preserved: ``final_activation='linear'`` and every
conv is bias-free (``BiasFreeConv2D`` / ``BiasFreeResidualBlock``), so the residual is
an interpretable scaled score (Miyasawa/Tweedie) and one model generalizes across noise
levels it never saw.

Usage::

    # Base BFCNN denoiser, full run
    MPLBACKEND=Agg python -m train.bfunet.train_bfcnn_denoiser \
        --variant base --epochs 100 --batch-size 16 --gpu 1

    # Custom flat ResNet (explicit block/filter geometry)
    MPLBACKEND=Agg python -m train.bfunet.train_bfcnn_denoiser \
        --variant custom --num-blocks 12 --filters 64 --gpu 1

    # Fast end-to-end mechanism check
    MPLBACKEND=Agg python -m train.bfunet.train_bfcnn_denoiser --smoke
"""

import keras
import argparse
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.bias_free_denoisers.bfcnn import (
    create_bfcnn_denoiser,
    create_bfcnn_variant,
    BFCNN_CONFIGS,
)

# Shared bfunet trainer substrate. The data / curriculum / callback / dashboard /
# self-iterate / train-orchestration skeleton lives ONCE in train.bfunet.common; this
# trainer imports it and RE-EXPORTS the frozen-API names so
# `from train.bfunet.train_bfcnn_denoiser import <name>` keeps resolving (mirrors the
# other trainers' test contract). `import ... as common` is used for common.train().
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
class BFCNNTrainingConfig(BFUnetTrainingConfig):
    """Configuration for the bias-free flat-CNN (BFCNN) denoiser trainer.

    Subclasses the shared ``BFUnetTrainingConfig`` (Data / Memory / Noise-curriculum /
    shared topology / Training / Optimization / Self-iterate / WW-PGD / init_from /
    Analysis / Output) and adds ONLY the BFCNN-specific fields + validation. The U-Net
    topology knobs the base carries are irrelevant to a flat CNN: ``use_gabor_stem`` is
    defaulted OFF and ``depth``/``blocks_per_level``/``initial_filters`` stay ``None`` so
    ``build_model`` never forwards them into the BFCNN factory (which rejects them).
    """

    # Overridable experiment-name prefix (plain class attr, NOT a dataclass field).
    experiment_prefix = "bfcnn_denoiser_"

    # BFCNN default is the flat "tiny" ResNet (overrides the base "base" default).
    variant: str = "tiny"

    # DECISION plan_2026-07-04_6e78d66d/D-004: carry bfcnn's full 6 train + 2 val dataset
    # breadth as subclass defaults so the merge does NOT silently narrow sourcing to the
    # base's COCO+DIV2K (2+1). Do NOT drop these back to the base default_factory.
    train_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/Megadepth",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
            "/media/arxwn/data0_4tb/datasets/WFLW/images",
            "/media/arxwn/data0_4tb/datasets/bdd_data/train",
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
            "/media/arxwn/data0_4tb/datasets/VGG-Face2/data/train",
        ]
    )
    val_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/div2k/validation",
            "/media/arxwn/data0_4tb/datasets/COCO/val2017",
        ]
    )

    # BFCNN is a flat CNN — no Gabor stem. Default OFF so no U-Net stem field is read.
    use_gabor_stem: bool = False

    # BFCNN-specific model fields (only used when variant == "custom").
    num_blocks: int = 8
    filters: int = 64
    initial_kernel_size: int = 5    # kernel for the stem conv
    kernel_size: int = 3            # kernel for the residual blocks
    # NOTE: block activation/normalization are inherited from BFUnetTrainingConfig
    # (block_activation='leaky_relu', block_activation_alpha=0.1, block_normalization=
    # 'batchnorm' -> mapped to the real BiasFreeBatchNorm in build_model, D-006). BFCNN
    # no longer carries a local `activation` field / --activation flag (retired, D-004).

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in BFCNN_CONFIGS and self.variant != "custom":
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: "
                f"{list(BFCNN_CONFIGS) + ['custom']}"
            )


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def build_model(config: BFCNNTrainingConfig) -> keras.Model:
    """Build the bias-free flat-CNN (BFCNN) denoiser from the config.

    Named variants dispatch to ``create_bfcnn_variant``; ``variant=='custom'`` dispatches
    to ``create_bfcnn_denoiser`` with the explicit block/filter geometry. BFCNN accepts
    NONE of the shared U-Net-parity kwargs (Gabor / Laplacian / depth / pooling / …), so
    build_model passes only the closed BFCNN kwarg set.
    """
    input_shape = (config.patch_size, config.patch_size, config.channels)
    # DECISION plan_2026-07-05_2199bb8e/D-006: map trainer 'batchnorm' -> real homogeneous
    # BiasFreeBatchNorm (variance-only, no moving_mean/beta -> degree-1 homogeneous at
    # inference), NOT stock keras.BatchNormalization(center=False) which subtracts
    # moving_mean and breaks f(ax)=a*f(x). 'layernorm' passes through unchanged. Do NOT
    # revert this to bare 'batchnorm' -> the residual blocks would silently rebuild stock
    # BN and the model would stop being homogeneous. See decisions.md D-006.
    norm = (
        "bias_free_batchnorm"
        if config.block_normalization == "batchnorm"
        else config.block_normalization
    )
    # Exact slope 0.1 requires a LeakyReLU INSTANCE — the bare string "leaky_relu" resolves
    # to Keras slope 0.2. Shared across stem + N blocks (safe: BiasFreeConv2D wraps it in a
    # per-block Activation and dict-serializes it, A7).
    act = (
        keras.layers.LeakyReLU(negative_slope=config.block_activation_alpha)
        if config.block_activation == "leaky_relu"
        else config.block_activation
    )
    if config.variant in BFCNN_CONFIGS:
        return create_bfcnn_variant(
            config.variant,
            input_shape=input_shape,
            activation=act,
            normalization_type=norm,
        )
    # variant == "custom" (validated in __post_init__)
    return create_bfcnn_denoiser(
        input_shape=input_shape,
        num_blocks=config.num_blocks,
        filters=config.filters,
        initial_kernel_size=config.initial_kernel_size,
        kernel_size=config.kernel_size,
        activation=act,
        normalization_type=norm,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name="bfcnn_denoiser_custom",
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


def train(config: BFCNNTrainingConfig) -> keras.Model:
    """Train the bias-free BFCNN denoiser with the noise curriculum.

    Thin delegator: all orchestration lives in ``common.train``; this trainer injects its
    own ``build_model`` + ``verify_bias_free`` plus the BFCNN label and results-dir prefix.
    Uses the DEFAULT bottleneck name, so no ``bottleneck_name_prefix`` is passed.
    """
    # DECISION plan_2026-07-04_6e78d66d/D-002: normalization ([-0.5,+0.5]) and the
    # PsnrMetric(max_val=1.0) compile are owned by common.train (common.py:1517). Do NOT
    # re-introduce bfcnn's old [-1,+1] range or PsnrMetric(max_val=2.0) here — the trainer
    # inherits the single shared convention and sets no per-model range/metric knob.
    return common.train(
        config,
        build_model,
        verify_bias_free,
        model_label="BFCNN",
        results_dir_prefix="bfcnn",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bias-free flat-CNN (BFCNN) denoiser (flat ResNet + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=list(BFCNN_CONFIGS) + ["custom"],
                        default="tiny",
                        help="BFCNN size preset; 'custom' uses --num-blocks/--filters/etc.")
    parser.add_argument("--num-blocks", type=int, default=8,
                        help="Number of residual blocks (variant='custom' only).")
    parser.add_argument("--filters", type=int, default=64,
                        help="Residual-block filter width (variant='custom' only).")
    parser.add_argument("--initial-kernel-size", type=int, default=5,
                        help="Kernel size for the stem conv (variant='custom' only).")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Kernel size for the residual blocks (variant='custom' only).")
    # NOTE: activation is set via the shared --block-activation / --block-activation-alpha
    # flags (add_common_arguments); the retired local --activation flag is gone (D-004).
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
        config = BFCNNTrainingConfig(
            variant=args.variant,
            num_blocks=args.num_blocks,
            filters=args.filters,
            initial_kernel_size=args.initial_kernel_size,
            kernel_size=args.kernel_size,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
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
            viz_freq=1,
            viz_samples=args.viz_samples,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "bfcnn_denoiser_smoke",
        )
    else:
        config = BFCNNTrainingConfig(
            variant=args.variant,
            num_blocks=args.num_blocks,
            filters=args.filters,
            initial_kernel_size=args.initial_kernel_size,
            kernel_size=args.kernel_size,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
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
        f"Config: variant={config.variant}, "
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
