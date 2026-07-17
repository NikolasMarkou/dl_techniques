"""Bias-free ConvNeXt (ConvUNext) denoiser trainer with a frozen Gabor stem and
a noise-sigma curriculum.

Trains ``create_convunext_denoiser`` (bias-free ConvNeXt U-Net) with an optional
NON-LEARNABLE Gabor depthwise stem on DIV2K + COCO. Patches of 256x256 are sampled
with geometric augmentation (flips + rot90), normalized to ``[0, 1]`` (``image / 255``;
the strictly-positive domain is what forces the bias-free net's filters to sum to one --
see ``research/2026_bfunet_unit_domain_migration.md``), and corrupted with additive Gaussian
noise whose per-image sigma is drawn from ``[sigma_min, sigma_max]``. The upper
bound ``sigma_max`` is a live ``tf.Variable`` widened every epoch by
``NoiseSigmaCurriculumCallback`` (curriculum: start with low noise, progressively
widen the spread).

Design notes (plan_2026-06-19_ed071c02):
- The noise function reads a captured ``tf.Variable`` (validated: a Variable read
  inside ``tf.data.map`` reflects per-epoch ``.assign``). This is a DELIBERATE
  Variable-backed variant of ``add_noise_to_patch`` (D-003), not a 4th blind copy.
- ConvNeXt V1 blocks are the default (strict bias-freedom: LayerNorm center=False).
  ConvNeXt V2 is opt-in but its GRN beta is trainable, so V2 is not strictly
  Mohan-compliant.
- Weighted COCO+DIV2K sourcing via ``select_weighted_image_paths`` prevents COCO's
  ~118K images from drowning DIV2K's ~800 (D-002 of plan_2026-06-18_1cca4fc1).

Reference PSNR baselines / SOTA (additive white Gaussian noise denoising)
------------------------------------------------------------------------
Published average PSNR (dB) on the standard AWGN denoising benchmarks. These are
REFERENCE TARGETS for interpreting this trainer's val-PSNR, NOT a like-for-like
leaderboard (see caveats below).

Noise-scale note: benchmark sigma is on the [0, 255] pixel scale. This trainer
adds noise in the [0, 1] normalized space, so::

    sigma_255  =  sigma_here * 255.0

i.e. this run's curriculum ``sigma_max 0.025 -> 0.25`` corresponds to
``sigma_255 ~= 6.4 -> 63.75`` — it spans (and exceeds) the classic 15/25/50
benchmark regimes as a single *blind* model. PSNR is scale-invariant, so dB
computed here with ``max_val=1.0`` on [0,1] images is directly comparable to
the published ``max_val=255`` numbers. NOTE: the [-0.5,+0.5] -> [0,1] migration was a
pure DC shift; peak-to-peak width is 1.0 in BOTH domains, so every sigma above and
``max_val=1.0`` are UNCHANGED and still exactly correct. Do not rescale them.

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

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser \\
        --variant base --epochs 100 --batch-size 16 --patch-size 256 --gpu 1

    # Quick mechanism check (tiny, 2 epochs):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser --smoke

Mixed precision (``--mixed-precision``, OFF by default): correct and serializes, but
benchmarked SLOWER than the fp32 default on base@256/batch-4 (RTX 4090): ~22 vs 36 img/s.
The decoder's bilinear-upsample gradient (``ResizeBilinearGrad``) emits fp32, which XLA's
strict dtype checker rejects, so the mixed-precision path must run with ``jit_compile=False``
-- and for this conv-heavy U-Net, losing XLA fusion outweighs the fp16 tensor-core gain.
Kept as an option for higher-res / other GPUs / a future XLA-clean upsample.
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
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization
from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    CONVUNEXT_CONFIGS,
)

# Shared bfunet trainer substrate. The data / curriculum / callback / dashboard /
# self-iterate / train-orchestration skeleton lives ONCE in train.bfunet.common; this
# trainer imports it and RE-EXPORTS the frozen-API names so
# `from train.bfunet.train_convunext_denoiser import <name>` keeps resolving (3 test
# files depend on this exact path). `import ... as common` is used for common.train().
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
    """Configuration for the bias-free ConvNeXt denoiser trainer.

    Subclasses the shared ``BFUnetTrainingConfig`` (Data / Memory / Noise-curriculum /
    shared U-Net topology / Training / Optimization / Self-iterate / WW-PGD / init_from /
    Analysis / Output) and adds ONLY the ConvNeXt-specific fields + model validation.
    """

    # Overridable experiment-name prefix (plain class attr, NOT a dataclass field).
    experiment_prefix = "convunext_denoiser_"

    # ConvNeXt-specific model fields.
    convnext_version: str = "v1"    # v1 = strict bias-free; v2 opt-in (GRN beta trains)
    # Grow output channels at decoder level 0: append `output_channels` zero-initialized channels before the level-0 ConvNeXt blocks (widened to absorb them), then keep ONLY those as the output, dropping the learned 1x1 final projection. Bias-free, default OFF. See bfconvunext create_convunext_denoiser.
    extra_zero_output_channels: bool = False
    # Opt-in depthwise-kernel init/regularization pass-through (plan_2026-06-20_353a3a76).
    # Both default None -> OFF, byte-identical to the pre-feature model (the blocks fall
    # back to their hardcoded TruncatedNormal(0,0.02) init + deepcopy(kernel_regularizer)).
    # depthwise_initializer: a keras init name; the alias 'orthonormal' maps to
    # Orthogonal(gain=1.0) via _resolve_depthwise_initializer (D-002). depthwise_l2: L2
    # weight on the depthwise kernels; None=off, else wired to keras.regularizers.L2.
    depthwise_initializer: Optional[str] = None
    depthwise_l2: Optional[float] = None
    # block_activation / block_activation_alpha / block_normalization are INHERITED from
    # BFUnetTrainingConfig (defaults leaky_relu / 0.1 / batchnorm), along with the
    # block_normalization membership validation. build_model consumes them unchanged.
    # Standard MLP dropout inside the ConvNeXt inverted-bottleneck blocks (NOT stochastic
    # depth). 0.0 = OFF / byte-identical to all existing checkpoints. Wired to
    # create_convunext_denoiser(dropout_rate=...).
    dropout_rate: float = 0.0
    # Optional bias-free linear-attention blocks inserted at the bottleneck BEFORE the
    # ConvNeXt block loop (create_convunext_denoiser). Both default OFF / byte-identical:
    # bottleneck_attention_blocks=0 adds zero graph nodes. When blocks>0, N residual
    # LinearAttention blocks run at the bottleneck (Miyasawa-safe 'linear' type, hardcoded);
    # bottleneck_filters must be divisible by bottleneck_attention_heads (factory asserts).
    bottleneck_attention_blocks: int = 0
    bottleneck_attention_heads: int = 8

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in CONVUNEXT_CONFIGS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: {list(CONVUNEXT_CONFIGS)}"
            )
        if self.convnext_version not in ("v1", "v2"):
            raise ValueError("convnext_version must be 'v1' or 'v2'")
        if self.bottleneck_attention_blocks < 0:
            raise ValueError(
                f"bottleneck_attention_blocks must be >= 0, got "
                f"{self.bottleneck_attention_blocks}"
            )
        if self.bottleneck_attention_heads < 1:
            raise ValueError(
                f"bottleneck_attention_heads must be >= 1, got "
                f"{self.bottleneck_attention_heads}"
            )


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def _resolve_depthwise_initializer(name: Optional[str]):
    """Resolve the trainer's --depthwise-initializer string to an initializer.

    'orthonormal' -> keras.initializers.Orthogonal(gain=1.0) (the verified
    orthonormal path for a depthwise (K,K,C,1) kernel; unit-norm). None -> None
    (OFF, byte-identical). Any other string passes through unchanged for
    keras.initializers.get() resolution inside the block.
    """
    if name is None:
        return None
    if name == "orthonormal":
        return keras.initializers.Orthogonal(gain=1.0)
    return name


def build_model(config: TrainingConfig) -> keras.Model:
    """Build the bias-free ConvNeXt denoiser from the variant config."""
    cfg = CONVUNEXT_CONFIGS[config.variant].copy()
    cfg.pop("description", None)
    cfg["convnext_version"] = config.convnext_version  # override variant default
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
    # DECISION plan_2026-06-21_eb7fd829/D-003: the bare "leaky_relu" string resolves to
    # slope 0.2 in Keras, so DO NOT pass the string when alpha 0.1 is wanted. Construct an
    # explicit keras.layers.LeakyReLU(negative_slope=alpha) instance to honor the 0.1
    # default. The instance round-trips through .keras via ConvNext*Block.get_config layer
    # serialization (D-001). Any other block_activation value is a plain string passed
    # straight to the factory. See decisions.md D-003.
    if config.block_activation == "leaky_relu":
        block_activation = keras.layers.LeakyReLU(
            negative_slope=config.block_activation_alpha
        )
    else:
        block_activation = config.block_activation
    return create_convunext_denoiser(
        input_shape=input_shape,
        use_gabor_stem=config.use_gabor_stem,
        gabor_filters=config.gabor_filters,
        gabor_kernel_size=config.gabor_kernel_size,
        gabor_activation=config.gabor_activation,
        gabor_stem_projection=config.gabor_stem_projection,
        use_laplacian_pyramid=config.use_laplacian_pyramid,
        high_freq_blocks=config.high_freq_blocks,
        bottleneck_attention_blocks=config.bottleneck_attention_blocks,
        bottleneck_attention_heads=config.bottleneck_attention_heads,
        zero_pad_channels=config.zero_pad_channels,
        extra_zero_output_channels=config.extra_zero_output_channels,
        final_projection_groups=final_projection_groups,
        downsample_pool_type=config.downsample_pool_type,
        enable_deep_supervision=config.enable_deep_supervision,
        expose_bottleneck=config.expose_bottleneck,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name=f"convunext_denoiser_{config.variant}",
        block_activation=block_activation,
        # The SAME constructed activation drives block + stem + deep-supervision so the
        # trainer path has no GELU anywhere. Sharing one stateless LeakyReLU instance is
        # safe: blocks/stem wrap it in their own Activation sublayers and the supervision
        # helper CLONES it (see _make_supervision_activation, D-006).
        stem_activation=block_activation,
        supervision_activation=block_activation,
        depthwise_initializer=_resolve_depthwise_initializer(config.depthwise_initializer),
        depthwise_regularizer=(
            keras.regularizers.L2(config.depthwise_l2)
            if config.depthwise_l2 is not None else None
        ),
        dropout_rate=config.dropout_rate,
        block_normalization=config.block_normalization,
        filter_multiplier=config.filter_multiplier,
        **cfg,
    )


def verify_bias_free(model: keras.Model) -> None:
    """Log a bias-free compliance check (informational)."""
    offenders = []
    for layer in model._flatten_layers():
        if getattr(layer, "use_bias", False):
            offenders.append(layer.name)
        if isinstance(layer, keras.layers.LayerNormalization) and getattr(
            layer, "center", False
        ):
            offenders.append(f"{layer.name} (LN center=True)")
        if isinstance(layer, GlobalResponseNormalization) and getattr(layer, "beta", None) is not None and layer.beta.trainable:
            offenders.append(f"{layer.name} (GRN beta - bias-like offset)")
    if offenders:
        logger.warning(
            f"Bias-free check: {len(offenders)} layer(s) carry bias/centering "
            f"(expected for ConvNeXt V2 GRN beta): {offenders[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")

    # Numeric black-box degree-1 homogeneity probe (shared; informational, NEVER raises).
    # Detects f(a*x) != a*f(x) breaks the static offender scan above cannot see (GRN
    # stem, GELU, non-homogeneous norm blocks). Lives in common.py (D-003, D-005).
    _homogeneity_probe(model)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> keras.Model:
    """Train the bias-free ConvUNeXt denoiser with the noise curriculum.

    Thin delegator: all orchestration lives in ``common.train``; this trainer injects
    its own ``build_model`` + ``verify_bias_free`` plus the ConvUNeXt label and
    results-dir prefix. ConvUNeXt uses the DEFAULT bottleneck name, so no
    ``bottleneck_name_prefix`` is passed.
    """
    return common.train(
        config,
        build_model,
        verify_bias_free,
        model_label="ConvUNeXt",
        results_dir_prefix="convunext_denoiser",
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bias-free ConvUNeXt denoiser (Gabor stem + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--variant", choices=list(CONVUNEXT_CONFIGS), default="base")
    parser.add_argument("--convnext-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--extra-zero-output-channels", action="store_true", help="Grow output channels at decoder level 0: append output_channels zero-initialized channels before the level-0 ConvNeXt blocks (widened), then keep ONLY those as the output instead of the learned 1x1 projection. Bias-free; default OFF.")
    parser.add_argument(
        "--depthwise-initializer", type=str, default=None,
        help="Opt-in initializer for the ConvNeXt depthwise kernels. The alias "
             "'orthonormal' maps to keras Orthogonal(gain=1.0) (unit-norm on the "
             "(K,K,C,1) depthwise shape); any other value passes through to "
             "keras.initializers.get(). Default None = byte-identical OFF "
             "(blocks keep their hardcoded TruncatedNormal(0,0.02)).",
    )
    parser.add_argument(
        "--depthwise-l2", type=float, default=None,
        help="Opt-in L2 weight on the ConvNeXt depthwise kernels (wired to "
             "keras.regularizers.L2). Default None = off (blocks keep their "
             "deepcopy(kernel_regularizer) default).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Opt-in MLP dropout rate inside the ConvNeXt inverted-bottleneck blocks "
             "(wired to create_convunext_denoiser dropout_rate). Default 0.0 = OFF, "
             "byte-identical to existing checkpoints. Typical: 0.1-0.3.",
    )
    parser.add_argument(
        "--bottleneck-attention-blocks", type=int, default=0,
        help="N bias-free linear-attention blocks inserted at the bottleneck before the "
             "ConvNeXt blocks (default 0 = OFF).",
    )
    parser.add_argument(
        "--bottleneck-attention-heads", type=int, default=8,
        help="num_heads for the bottleneck linear-attention blocks (bottleneck_filters "
             "must be divisible by this when blocks>0).",
    )
    args = parser.parse_args()
    reject_self_iterate_with_nonadditive(parser, args)
    return args


def main():
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
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            clip_noise=not args.no_clip,
            high_freq_blocks=args.high_freq_blocks,
            bottleneck_attention_blocks=args.bottleneck_attention_blocks,
            bottleneck_attention_heads=args.bottleneck_attention_heads,
            zero_pad_channels=args.zero_pad_channels,
            extra_zero_output_channels=args.extra_zero_output_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_deep_supervision=args.deep_supervision,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
            gabor_filters=8,
            gabor_kernel_size=args.gabor_kernel_size,
            gabor_activation=args.gabor_activation,
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
            depthwise_initializer=args.depthwise_initializer,
            depthwise_l2=args.depthwise_l2,
            block_activation=args.block_activation,
            block_activation_alpha=args.block_activation_alpha,
            dropout_rate=args.dropout,
            block_normalization=args.block_normalization,
            viz_freq=1,
            viz_samples=args.viz_samples,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "convunext_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            use_laplacian_pyramid=args.laplacian_pyramid,
            clip_noise=not args.no_clip,
            high_freq_blocks=args.high_freq_blocks,
            bottleneck_attention_blocks=args.bottleneck_attention_blocks,
            bottleneck_attention_heads=args.bottleneck_attention_heads,
            zero_pad_channels=args.zero_pad_channels,
            extra_zero_output_channels=args.extra_zero_output_channels,
            downsample_pool_type=("average" if args.mean_pooling else "max"),
            expose_bottleneck=args.expose_bottleneck,
            enable_analyzer=args.analyzer,
            analyzer_freq=args.analyzer_freq,
            gabor_filters=args.gabor_filters,
            gabor_kernel_size=args.gabor_kernel_size,
            gabor_activation=args.gabor_activation,
            gabor_stem_projection=not args.no_gabor_projection,
            initial_filters=args.initial_filters,
            filter_multiplier=args.filter_multiplier,
            depth=args.depth,
            blocks_per_level=args.blocks_per_level,
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
            depthwise_initializer=args.depthwise_initializer,
            depthwise_l2=args.depthwise_l2,
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

    # --ww-pgd-log-alpha implies --ww-pgd: enabling the alpha trajectory log turns
    # on the projection it instruments (so the flag is usable standalone).
    if config.ww_pgd_log_alpha:
        config.ww_pgd = True

    logger.info(
        f"Config: variant={config.variant} ({config.convnext_version}), "
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
