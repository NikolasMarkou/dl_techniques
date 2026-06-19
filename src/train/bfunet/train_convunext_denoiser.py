"""Bias-free ConvNeXt (ConvUNext) denoiser trainer with a frozen Gabor stem and
a noise-sigma curriculum.

Trains ``create_convunext_denoiser`` (bias-free ConvNeXt U-Net) with an optional
NON-LEARNABLE Gabor depthwise stem on DIV2K + COCO. Patches of 256x256 are sampled
with geometric augmentation (flips + rot90), normalized to ``[-1, +1]`` (required
for bias-free / scaling-invariant denoising), and corrupted with additive Gaussian
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

Usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser \\
        --variant base --epochs 100 --batch-size 16 --patch-size 256 --gpu 1

    # Quick mechanism check (tiny, 2 epochs):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.train_convunext_denoiser --smoke
"""

import gc
import json
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    save_config_json,
    collect_image_paths,
    augment_patch,
)
from train.superpoint.homographic_adaptation import select_weighted_image_paths
from dl_techniques.metrics.psnr_metric import PsnrMetric
from dl_techniques.metrics.ssim_metric import SsimMetric
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
    CONVUNEXT_CONFIGS,
)
from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the bias-free ConvNeXt denoiser trainer."""

    # Data
    train_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/COCO/train2017",
            "/media/arxwn/data0_4tb/datasets/div2k/train",
        ]
    )
    val_image_dirs: List[str] = field(
        default_factory=lambda: [
            "/media/arxwn/data0_4tb/datasets/div2k/validation",
        ]
    )
    dataset_weights: Optional[List[float]] = None  # None -> equal weight per dir
    patch_size: int = 256
    channels: int = 3
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"]
    )

    # Memory / sourcing
    max_train_files: Optional[int] = 10000
    max_val_files: Optional[int] = 500
    parallel_reads: int = 8
    dataset_shuffle_buffer: int = 1024
    seed: int = 42

    # Noise curriculum
    noise_sigma_min: float = 0.0
    sigma_max_start: float = 0.05   # narrow range at epoch 0 (low noise)
    sigma_max_end: float = 0.5      # wide range at the final curriculum epoch
    curriculum_epochs: Optional[int] = None  # None -> use `epochs`
    curriculum_schedule: str = "linear"      # linear | cosine | exp

    # Model (bias-free ConvNeXt U-Net)
    variant: str = "base"           # tiny | small | base | large | xlarge
    convnext_version: str = "v1"    # v1 = strict bias-free; v2 opt-in (GRN beta trains)
    use_gabor_stem: bool = True
    gabor_filters: int = 32
    gabor_kernel_size: int = 7
    enable_deep_supervision: bool = False

    # Training
    batch_size: int = 16
    epochs: int = 100
    patches_per_image: int = 8
    augment_data: bool = True
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = 100

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 15

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"convunext_denoiser_{self.variant}_{timestamp}"
        if self.curriculum_epochs is None:
            self.curriculum_epochs = self.epochs
        if self.patch_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid patch size or channel configuration")
        if self.noise_sigma_min < 0:
            raise ValueError("noise_sigma_min must be >= 0")
        if self.sigma_max_end <= self.noise_sigma_min:
            raise ValueError("sigma_max_end must exceed noise_sigma_min")
        if self.variant not in CONVUNEXT_CONFIGS:
            raise ValueError(
                f"Unknown variant {self.variant!r}; choices: {list(CONVUNEXT_CONFIGS)}"
            )
        if self.convnext_version not in ("v1", "v2"):
            raise ValueError("convnext_version must be 'v1' or 'v2'")
        if not self.train_image_dirs or not self.val_image_dirs:
            raise ValueError("train/val image dirs must be non-empty")


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------


def load_and_preprocess_image(
    image_path: tf.Tensor, config: TrainingConfig
) -> tf.Tensor:
    """Decode an image, normalize to [-1, +1], and crop a random patch."""
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image_string, channels=config.channels, expand_animations=False
    )
    image.set_shape([None, None, config.channels])
    image = tf.cast(image, tf.float32)

    # Normalize to [-1, +1] (critical for bias-free architecture).
    image = (image / 127.5) - 1.0

    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    min_size = config.patch_size

    # Aspect-preserving upscale guard for images smaller than the patch.
    def _upscale():
        scale = tf.cast(min_size, tf.float32) / tf.cast(
            tf.minimum(height, width), tf.float32
        )
        new_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w])

    image = tf.cond(
        tf.logical_or(height < min_size, width < min_size),
        true_fn=_upscale,
        false_fn=lambda: image,
    )

    return tf.image.random_crop(
        image, [config.patch_size, config.patch_size, config.channels]
    )


def make_curriculum_noise_fn(config: TrainingConfig, sigma_max_var: tf.Variable):
    """Build a noise function that samples per-image sigma from
    ``[noise_sigma_min, sigma_max_var]`` where the upper bound is a live
    ``tf.Variable`` widened per-epoch by the curriculum callback (D-003)."""

    sigma_min = float(config.noise_sigma_min)

    def add_curriculum_noise(patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # sigma_max_var is read at graph-execution time -> reflects the per-epoch
        # .assign performed by NoiseSigmaCurriculumCallback (risk spike confirmed).
        noise_level = tf.random.uniform([], sigma_min, sigma_max_var)
        noisy = patch + tf.random.normal(tf.shape(patch)) * noise_level
        return tf.clip_by_value(noisy, -1.0, 1.0), patch

    return add_curriculum_noise


def collect_training_paths(config: TrainingConfig) -> List[str]:
    """Weighted COCO+DIV2K path worklist (prevents COCO drowning DIV2K)."""
    pairs = select_weighted_image_paths(
        image_dirs=config.train_image_dirs,
        weights=config.dataset_weights,
        num_images=config.max_train_files,
        seed=config.seed,
    )
    return [path for _name, path in pairs]


def create_dataset(
    file_paths: List[str],
    config: TrainingConfig,
    noise_fn,
    is_training: bool,
) -> tf.data.Dataset:
    """Build a tf.data pipeline of (noisy, clean) [-1,+1] patch pairs."""
    if not file_paths:
        raise ValueError("No image files found for the dataset")

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(file_paths)),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    if is_training and config.patches_per_image > 1:
        dataset = dataset.flat_map(
            lambda p: tf.data.Dataset.from_tensors(p).repeat(config.patches_per_image)
        )

    dataset = dataset.map(
        lambda p: load_and_preprocess_image(p, config),
        num_parallel_calls=config.parallel_reads,
    )
    dataset = dataset.filter(lambda x: tf.reduce_sum(tf.abs(x)) > 0)
    dataset = dataset.map(
        lambda x: tf.ensure_shape(
            x, [config.patch_size, config.patch_size, config.channels]
        )
    )

    if is_training and config.augment_data:
        dataset = dataset.map(augment_patch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(noise_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda noisy, clean: (
            tf.ensure_shape(
                noisy, [config.patch_size, config.patch_size, config.channels]
            ),
            tf.ensure_shape(
                clean, [config.patch_size, config.patch_size, config.channels]
            ),
        )
    )

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------


def build_model(config: TrainingConfig) -> keras.Model:
    """Build the bias-free ConvNeXt denoiser from the variant config."""
    cfg = CONVUNEXT_CONFIGS[config.variant].copy()
    cfg.pop("description", None)
    cfg["convnext_version"] = config.convnext_version  # override variant default
    input_shape = (config.patch_size, config.patch_size, config.channels)
    return create_convunext_denoiser(
        input_shape=input_shape,
        use_gabor_stem=config.use_gabor_stem,
        gabor_filters=config.gabor_filters,
        gabor_kernel_size=config.gabor_kernel_size,
        enable_deep_supervision=config.enable_deep_supervision,
        final_activation="linear",  # MUST stay linear: bias-free homogeneity f(ax)=a*f(x)
        model_name=f"convunext_denoiser_{config.variant}",
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
    if offenders:
        logger.warning(
            f"Bias-free check: {len(offenders)} layer(s) carry bias/centering "
            f"(expected for ConvNeXt V2 GRN beta): {offenders[:10]}"
        )
    else:
        logger.info("Bias-free check: PASSED - all layers are bias-free")


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> keras.Model:
    """Train the bias-free ConvNeXt denoiser with the noise curriculum."""
    logger.info(f"Starting ConvNeXt denoiser training: {config.experiment_name}")
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    # Live, curriculum-controlled upper bound for the noise-sigma sampling range.
    sigma_max_var = tf.Variable(
        config.sigma_max_start, dtype=tf.float32, trainable=False, name="sigma_max"
    )
    noise_fn = make_curriculum_noise_fn(config, sigma_max_var)

    train_paths = collect_training_paths(config)
    val_paths = collect_image_paths(
        config.val_image_dirs,
        extensions=config.image_extensions,
        max_files=config.max_val_files,
    )
    logger.info(
        f"Sourced {len(train_paths)} train / {len(val_paths)} val image paths"
    )

    train_ds = create_dataset(train_paths, config, noise_fn, is_training=True)
    val_ds = create_dataset(val_paths, config, noise_fn, is_training=False)

    if config.steps_per_epoch is not None:
        steps_per_epoch = config.steps_per_epoch
    else:
        steps_per_epoch = max(
            100, (len(train_paths) * config.patches_per_image) // config.batch_size
        )
    validation_steps = config.validation_steps or max(
        10, len(val_paths) // config.batch_size
    )
    logger.info(
        f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )

    model = build_model(config)
    model.summary(print_fn=logger.info)
    verify_bias_free(model)

    lr_schedule = learning_rate_schedule_builder(
        {
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": steps_per_epoch * config.epochs,
            "warmup_steps": steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )

    # MSE loss -> least-squares-optimal (Miyasawa) denoiser.
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            "mae",
            PsnrMetric(max_val=2.0, name="psnr_metric"),
            SsimMetric(max_val=2.0, name="ssim_metric"),
        ],
    )
    logger.info(f"Model compiled with {model.count_params():,} parameters")

    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="convunext_denoiser",
        run_dir=str(output_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,
        include_analyzer=False,
    )
    callbacks.append(
        NoiseSigmaCurriculumCallback(
            sigma_max_var=sigma_max_var,
            sigma_max_start=config.sigma_max_start,
            sigma_max_end=config.sigma_max_end,
            total_epochs=config.curriculum_epochs,
            schedule=config.curriculum_schedule,
        )
    )

    start = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {time.time() - start:.2f}s")

    try:
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    gc.collect()
    return model


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bias-free ConvNeXt denoiser (Gabor stem + noise curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=list(CONVUNEXT_CONFIGS), default="base")
    parser.add_argument("--convnext-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-gabor-stem", action="store_true",
                        help="Disable the frozen Gabor depthwise stem")
    parser.add_argument("--gabor-filters", type=int, default=32)
    parser.add_argument("--sigma-max-start", type=float, default=0.05)
    parser.add_argument("--sigma-max-end", type=float, default=0.5)
    parser.add_argument("--curriculum-schedule",
                        choices=["linear", "cosine", "exp"], default="linear")
    parser.add_argument("--curriculum-epochs", type=int, default=None)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-val-files", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny end-to-end mechanism check (few steps/epochs, constant LR).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)

    if args.smoke:
        # Mechanism check: tiny, fast, constant LR (avoid cosine collapse at 2 epochs).
        config = TrainingConfig(
            variant="tiny",
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            gabor_filters=8,
            epochs=2,
            curriculum_epochs=2,
            batch_size=2,
            patch_size=64,
            channels=3,
            patches_per_image=2,
            max_train_files=8,
            max_val_files=4,
            steps_per_epoch=3,
            validation_steps=2,
            warmup_epochs=0,
            lr_schedule_type="constant",
            learning_rate=1e-3,
            sigma_max_start=0.05,
            sigma_max_end=0.5,
            curriculum_schedule="linear",
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "convunext_denoiser_smoke",
        )
    else:
        config = TrainingConfig(
            variant=args.variant,
            convnext_version=args.convnext_version,
            use_gabor_stem=not args.no_gabor_stem,
            gabor_filters=args.gabor_filters,
            enable_deep_supervision=args.deep_supervision,
            epochs=args.epochs,
            curriculum_epochs=args.curriculum_epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            channels=args.channels,
            patches_per_image=args.patches_per_image,
            learning_rate=args.learning_rate,
            sigma_max_start=args.sigma_max_start,
            sigma_max_end=args.sigma_max_end,
            curriculum_schedule=args.curriculum_schedule,
            max_train_files=args.max_train_files or 10000,
            max_val_files=args.max_val_files or 500,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )

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
