"""Training script for ConvNeXtPatchVAE — resolution-agnostic per-patch VAE.

The model uses ``add_loss`` internally; ``compile(loss=...)`` is bypassed.
All loss components (recon, KL, SIGReg) flow through ``model.metrics``:
``loss``, ``recon_loss``, ``kl_loss``, ``sigreg_loss`` — and their ``val_``
twins during evaluation.

Typical usage::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m \\
        train.convnext_patch_vae.train_convnext_patch_vae \\
        --dataset cifar10 --variant base --epochs 50

Smoke test (CPU, tiny model, synthetic data, <60 s)::

    python -m train.convnext_patch_vae.train_convnext_patch_vae --smoke

Training menu:
    T1a  CIFAR-10,  MSE, SIGReg ON  (~30 min RTX 4090)
    T1b  CIFAR-10,  MSE, SIGReg OFF (--lambda-sigreg 0)
    T2   ADE20K,   BCE, 256x256, patch=8, preset=base (~4-6 h RTX 4090)
    T3   COCO2017, BCE, 256x256, patch=8, preset=base (~8-12 h RTX 4090)

ADE20K quick-start (10 steps to validate pipeline)::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m \\
        train.convnext_patch_vae.train_convnext_patch_vae \\
        --dataset ade20k --image-size 256 --patch-size 8 \\
        --preset base --batch-size 32 --epochs 1 --steps-per-epoch 10

COCO 2017 quick-start::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg python -m \\
        train.convnext_patch_vae.train_convnext_patch_vae \\
        --dataset coco --image-size 256 --patch-size 8 \\
        --preset base --batch-size 32 --epochs 1 --steps-per-epoch 10
"""

# MPLBACKEND must be set before any matplotlib import — headless server guard.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import glob as _glob
import json
import dataclasses
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import keras

from dl_techniques.utils.logger import logger
from dl_techniques.models.convnext_patch_vae.config import (
    ConvNeXtPatchVAEConfig,
    HierarchicalConvNeXtPatchVAEConfig,
)
from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE
from dl_techniques.models.convnext_patch_vae.model_hierarchical import (
    HierarchicalConvNeXtPatchVAE,
    _L2ConditionedDecoder,
    _L2ConditionalPrior,
)
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from train.common import setup_gpu, create_base_argument_parser, create_callbacks
from train.convnext_patch_vae.callbacks import (
    LatentSpaceCallback,
    LatentInterpolationCallback,
)

CUSTOM_OBJECTS = {
    "ConvNeXtPatchVAE": ConvNeXtPatchVAE,
    "HierarchicalConvNeXtPatchVAE": HierarchicalConvNeXtPatchVAE,
    "_L2ConditionedDecoder": _L2ConditionedDecoder,
    "_L2ConditionalPrior": _L2ConditionalPrior,
}

# Per-channel statistics for MSE normalisation (mean/std per dataset)
_CIFAR10_MEAN  = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD   = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
_CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
_CIFAR100_STD  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

_CIFAR_STATS = {
    "cifar10":  (_CIFAR10_MEAN,  _CIFAR10_STD),
    "cifar100": (_CIFAR100_MEAN, _CIFAR100_STD),
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """All hyperparameters for one training run.

    Attributes:
        dataset: Dataset identifier. Supported: ``"cifar10"``, ``"cifar100"``,
            ``"ade20k"``, ``"coco"``. Derived from ``datasets`` when that list
            is provided.
        datasets: List of dataset identifiers for multi-dataset mixing.
            When non-empty, takes precedence over ``dataset``. Only
            ``"ade20k"`` and ``"coco"`` may be combined (identical
            normalisation). Single-element list behaves identically to
            the corresponding ``dataset`` string.
        img_size: Spatial resolution fed to the model (must be divisible
            by ``patch_size``).
        img_channels: Number of input/output channels (3 for RGB).
        augment_data: Whether to apply random horizontal flip + crop.
        augment_color: Whether to apply photometric augmentation
            (brightness, contrast, saturation) on top of geometric
            augmentation.  Has no effect when ``augment_data=False``.
            Saturation is unconditionally skipped for the CIFAR path
            (standardised data is not in ``[0, 1]`` regardless of loss type).
        model_variant: Optional preset shorthand (``"tiny"``, ``"base"``,
            ``"large"``).  When set, overrides ``embed_dim``,
            ``encoder_depth``, ``decoder_depth``, and ``latent_dim``.
        patch_size: Stem stride; must divide ``img_size``.
        embed_dim: ConvNeXt block width.
        encoder_depth: Number of ConvNextV2Block layers in encoder.
        decoder_depth: Number of ConvNextV2Block layers in decoder.
        kernel_size: Depthwise kernel size inside each block.
        latent_dim: Per-patch latent dimensionality.
        beta_kl: Weight on KL divergence term (beta-VAE style).
        lambda_sigreg: Weight on SIGReg anti-collapse term.  Set to ``0``
            for the T1b ablation.
        sigreg_knots: Integration grid points for SIGReg (>= 2).
        sigreg_num_proj: Random projections per SIGReg call (>= 64).
        recon_loss_type: ``"mse"`` or ``"bce"``.  BCE requires inputs in
            ``[0, 1]``; the dataset pipeline selects the correct scaling
            branch automatically.
        dropout_rate: Per-block dropout applied during training.
        gamma_clip: Symmetric gradient clip value (``None`` = disabled).
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 weight decay for AdamW (do NOT also set
            ``kernel_regularizer_config`` — double WD footgun).
        warmup_epochs: Linear warmup before cosine decay.
        beta_kl_start: Initial beta value at epoch 0 (annealing start).
        beta_anneal_epochs: Epochs over which beta ramps from ``beta_kl_start``
            to ``beta_kl``. Set to ``0`` to disable annealing.
        early_stopping_patience: EarlyStopping patience on ``val_loss``.
        success_threshold: ``val_loss <= threshold`` → convergence flag.
            Advisory only; does not block saving the model.
        output_dir: Root output directory (should be repo-root
            ``results/``).
        experiment_name: Run identifier. Auto-generated if ``None``.
    """

    # Dataset
    dataset: str = "cifar100"
    datasets: List[str] = dataclasses.field(default_factory=list)
    img_size: int = 32
    img_channels: int = 3
    augment_data: bool = True
    augment_color: bool = True
    patches_per_image: int = 4

    # Model shorthand (overrides per-field dims when set)
    model_variant: Optional[str] = None

    # Architecture
    patch_size: int = 4
    embed_dim: int = 128
    encoder_depth: int = 4
    decoder_depth: int = 4
    kernel_size: int = 7
    latent_dim: int = 16
    beta_kl: float = 0.5
    lambda_sigreg: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256
    recon_loss_type: str = "bce"
    dropout_rate: float = 0.0
    gamma_clip: float = 1.0

    # Training
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    beta_kl_start: float = 0.0001
    beta_anneal_epochs: int = 15
    early_stopping_patience: int = 10

    # Output / evaluation
    success_threshold: float = 0.02
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Large-dataset filesystem paths (used when dataset="ade20k" or "coco")
    ade20k_dir: str = "/media/arxwn/data0_4tb/datasets/ade20k"
    coco_dir: str = "/media/arxwn/data0_4tb/datasets/coco_2017"

    # Override computed steps_per_epoch (useful for quick pipeline validation)
    steps_per_epoch_override: Optional[int] = None

    # ------------------------------------------------------------------
    # Hierarchical (two-level) variant — opt-in via --hierarchical.
    # When False, the L1-* fields are ignored and the existing single-
    # scale architecture is used. When True, `patch_size` is the L2
    # patch size and `patch_size_l1` is required.
    # ------------------------------------------------------------------
    hierarchical: bool = False
    patch_size_l1: int = 32
    embed_dim_l1: int = 128
    encoder_depth_l1: int = 4
    decoder_depth_l1: int = 4
    latent_dim_l1: int = 64
    beta_kl_l1: float = 0.5
    beta_kl_l2: float = 0.5
    beta_kl_l1_start: float = 0.0001
    beta_kl_l2_start: float = 0.0001
    lambda_sigreg_l1: float = 0.05
    lambda_sigreg_l2: float = 0.1

    def __post_init__(self) -> None:
        # Normalize datasets/dataset: the list is canonical; the string is derived.
        if self.datasets:
            # Multi-dataset mode: derive composite string key for naming/checks.
            self.dataset = "+".join(self.datasets)
        else:
            # Single-dataset mode: derive list from the string.
            self.datasets = [self.dataset]

        # Validate multi-dataset combinations.
        if len(self.datasets) > 1:
            bad = [d for d in self.datasets if d not in ("ade20k", "coco")]
            if bad:
                raise ValueError(
                    f"Multi-dataset mixing only supports 'ade20k' and 'coco'. "
                    f"Unsupported entries: {bad}."
                )

        if self.experiment_name is None:
            variant = self.model_variant or "custom"
            self.experiment_name = f"{self.dataset}_{variant}"
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by "
                f"patch_size {self.patch_size}"
            )

    def to_model_config(self) -> ConvNeXtPatchVAEConfig:
        """Construct :class:`ConvNeXtPatchVAEConfig` from this config.

        When ``model_variant`` is set, preset dims override the per-field
        values for ``embed_dim``, depth, and ``latent_dim``.

        Returns:
            A fully-validated :class:`ConvNeXtPatchVAEConfig`.

        Raises:
            ValueError: If ``model_variant`` is not a recognised preset.
        """
        if self.model_variant is not None:
            presets = ConvNeXtPatchVAE.PRESETS
            if self.model_variant not in presets:
                raise ValueError(
                    f"Unknown model_variant '{self.model_variant}'. "
                    f"Choose from {list(presets.keys())}."
                )
            p = presets[self.model_variant]
            return ConvNeXtPatchVAEConfig(
                img_size=self.img_size,
                img_channels=self.img_channels,
                patch_size=self.patch_size,
                embed_dim=p["embed_dim"],
                encoder_depth=p["encoder_depth"],
                decoder_depth=p["decoder_depth"],
                kernel_size=self.kernel_size,
                latent_dim=p["latent_dim"],
                beta_kl=self.beta_kl,
                lambda_sigreg=self.lambda_sigreg,
                sigreg_knots=self.sigreg_knots,
                sigreg_num_proj=self.sigreg_num_proj,
                recon_loss_type=self.recon_loss_type,
                dropout_rate=self.dropout_rate,
                gamma_clip=self.gamma_clip,
            )
        return ConvNeXtPatchVAEConfig(
            img_size=self.img_size,
            img_channels=self.img_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            encoder_depth=self.encoder_depth,
            decoder_depth=self.decoder_depth,
            kernel_size=self.kernel_size,
            latent_dim=self.latent_dim,
            beta_kl=self.beta_kl,
            lambda_sigreg=self.lambda_sigreg,
            sigreg_knots=self.sigreg_knots,
            sigreg_num_proj=self.sigreg_num_proj,
            recon_loss_type=self.recon_loss_type,
            dropout_rate=self.dropout_rate,
            gamma_clip=self.gamma_clip,
        )

    def to_hierarchical_model_config(self) -> HierarchicalConvNeXtPatchVAEConfig:
        """Construct :class:`HierarchicalConvNeXtPatchVAEConfig` from this config.

        ``patch_size`` and ``latent_dim`` map to the L2 scale; the L1
        scale uses the dedicated ``patch_size_l1`` / ``latent_dim_l1`` /
        ``embed_dim_l1`` / ``encoder_depth_l1`` / ``decoder_depth_l1``
        fields.

        When ``model_variant`` is set, the L1+L2 width / depth / latent
        dims are taken from ``HierarchicalConvNeXtPatchVAE.PRESETS`` —
        symmetric with :meth:`to_model_config` for the single-scale path.
        Patch sizes and per-side L1/L2 KL/SIGReg weights remain governed
        by the dataclass fields.
        DECISION plan_2026-05-28_15256fe3/D-005
        """
        if self.model_variant is not None:
            presets = HierarchicalConvNeXtPatchVAE.PRESETS
            if self.model_variant not in presets:
                raise ValueError(
                    f"Unknown model_variant '{self.model_variant}' for "
                    f"hierarchical model. Choose from {list(presets.keys())}."
                )
            p = presets[self.model_variant]
            embed_dim_l1 = p["embed_dim_l1"]
            embed_dim_l2 = p["embed_dim_l2"]
            encoder_depth_l1 = p["encoder_depth_l1"]
            decoder_depth_l1 = p["decoder_depth_l1"]
            encoder_depth_l2 = p["encoder_depth_l2"]
            decoder_depth_l2 = p["decoder_depth_l2"]
            latent_dim_l1 = p["latent_dim_l1"]
            latent_dim_l2 = p["latent_dim_l2"]
        else:
            embed_dim_l1 = self.embed_dim_l1
            embed_dim_l2 = self.embed_dim
            encoder_depth_l1 = self.encoder_depth_l1
            decoder_depth_l1 = self.decoder_depth_l1
            encoder_depth_l2 = self.encoder_depth
            decoder_depth_l2 = self.decoder_depth
            latent_dim_l1 = self.latent_dim_l1
            latent_dim_l2 = self.latent_dim

        return HierarchicalConvNeXtPatchVAEConfig(
            img_size=self.img_size,
            img_channels=self.img_channels,
            patch_size_l1=self.patch_size_l1,
            patch_size_l2=self.patch_size,
            embed_dim_l1=embed_dim_l1,
            embed_dim_l2=embed_dim_l2,
            encoder_depth_l1=encoder_depth_l1,
            decoder_depth_l1=decoder_depth_l1,
            encoder_depth_l2=encoder_depth_l2,
            decoder_depth_l2=decoder_depth_l2,
            kernel_size=self.kernel_size,
            latent_dim_l1=latent_dim_l1,
            latent_dim_l2=latent_dim_l2,
            beta_kl_l1=self.beta_kl_l1,
            beta_kl_l2=self.beta_kl_l2,
            lambda_sigreg_l1=self.lambda_sigreg_l1,
            lambda_sigreg_l2=self.lambda_sigreg_l2,
            sigreg_knots=self.sigreg_knots,
            sigreg_num_proj=self.sigreg_num_proj,
            recon_loss_type=self.recon_loss_type,
            dropout_rate=self.dropout_rate,
            gamma_clip=self.gamma_clip,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_cifar_dataset(
    config: TrainingConfig,
) -> Tuple[Any, Any, int, int]:
    """Load CIFAR-10 or CIFAR-100 and build ``tf.data`` pipelines.

    Args:
        config: Training configuration. ``config.dataset`` must be one of
            ``"cifar10"`` or ``"cifar100"``.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
        Datasets emit ``(x, x)`` pairs — the label is the image itself;
        :meth:`ConvNeXtPatchVAE.train_step` ignores the second element.
    """
    import tensorflow as tf

    loader = (
        keras.datasets.cifar10 if config.dataset == "cifar10"
        else keras.datasets.cifar100
    )
    (x_train, _), (x_test, _) = loader.load_data()
    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")

    # Select normalisation branch based on recon_loss_type.
    # BCE requires inputs in [0, 1] → /255 only.
    # MSE applies per-dataset mean/std after /255 scaling.
    mean, std = _CIFAR_STATS[config.dataset]
    if config.recon_loss_type == "bce":
        x_train /= 255.0
        x_test  /= 255.0
    else:
        x_train /= 255.0
        x_test  /= 255.0
        x_test = (x_test - mean) / std
        if not config.augment_data:
            # No augmentation map runs, so standardise at numpy level.
            x_train = (x_train - mean) / std
        # else: deferred into _augment so jitter operates on [0, 1] data first.

    steps_per_epoch = max(1, len(x_train) // config.batch_size)
    val_steps       = max(1, len(x_test)  // config.batch_size)

    def _augment(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, [config.img_size, config.img_size, config.img_channels])
        if config.augment_color:
            x = tf.image.random_brightness(x, max_delta=0.1)
            x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        # Clip to [0,1] before the loss-type branch. BCE requires targets in
        # [0,1] and the jitter above can push pixels out of range; for MSE the
        # subsequent standardisation absorbs the (now-clipped) values. Mirrors
        # the filesystem patch path's clip.
        x = tf.clip_by_value(x, 0.0, 1.0)
        if config.recon_loss_type != "bce":
            mean_tf = tf.constant(mean, dtype=tf.float32)
            std_tf  = tf.constant(std,  dtype=tf.float32)
            x = (x - mean_tf) / std_tf
        return x

    train_ds = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(10_000)
        .map(_augment if config.augment_data else lambda x: x,
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(x_test)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(
        f"{config.dataset.upper()}: {len(x_train)} train / {len(x_test)} val | "
        f"steps_per_epoch={steps_per_epoch} | recon={config.recon_loss_type}"
    )
    return train_ds, val_ds, steps_per_epoch, val_steps


def _build_smoke_dataset(
    config: TrainingConfig,
) -> Tuple[Any, Any, int, int]:
    """Minimal synthetic dataset for smoke tests (no GPU needed).

    Args:
        config: Training configuration.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch=3, val_steps=2)``.
    """
    import tensorflow as tf

    n_train, n_val = config.batch_size * 3, config.batch_size * 2
    shape = (config.img_size, config.img_size, config.img_channels)

    rng = np.random.default_rng(42)
    x_train = rng.uniform(0, 1, (n_train,) + shape).astype("float32")
    x_val   = rng.uniform(0, 1, (n_val,)   + shape).astype("float32")

    if config.recon_loss_type == "mse":
        mean = np.mean(x_train, axis=(0, 1, 2), keepdims=False)
        std  = np.std(x_train,  axis=(0, 1, 2), keepdims=False) + 1e-8
        x_train = (x_train - mean) / std
        x_val   = (x_val   - mean) / std

    def _make_ds(x: np.ndarray, repeat: bool = False) -> Any:
        ds = (
            tf.data.Dataset.from_tensor_slices(x)
            .map(lambda xi: (xi, xi), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        if repeat:
            ds = ds.repeat()
        return ds

    return _make_ds(x_train, repeat=True), _make_ds(x_val), 3, 2


def _make_filesystem_patch_fn(
    img_size: int,
    img_channels: int,
    patches_per_image: int,
    augment_color: bool,
):
    """Train-path factory: decode one image, yield ``patches_per_image`` random crops.

    The original image is never resized (preserving natural scale and statistics).
    Each crop is an independent ``img_size × img_size`` patch taken at a random
    location. If either dimension of the source image is smaller than ``img_size``
    the image is upscaled to the minimum required size before cropping.

    Returns a callable ``path -> tf.data.Dataset`` suitable for ``flat_map``.

    Args:
        img_size: Crop size (square).
        img_channels: Number of channels.
        patches_per_image: Number of random patches to extract per source image.
        augment_color: Whether to apply brightness/contrast/saturation jitter.

    Returns:
        Callable ``(path: tf.Tensor) -> tf.data.Dataset``.
    """
    import tensorflow as tf

    def _to_patches(path: tf.Tensor) -> "tf.data.Dataset":
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=img_channels)
        img = tf.cast(img, tf.float32) / 255.0
        # Upscale only when source is smaller than the requested crop size.
        shape = tf.shape(img)
        new_h = tf.maximum(shape[0], img_size + 4)
        new_w = tf.maximum(shape[1], img_size + 4)
        img = tf.image.resize(img, [new_h, new_w])

        def _one_patch(_):
            patch = tf.image.random_crop(img, [img_size, img_size, img_channels])
            patch = tf.image.random_flip_left_right(patch)
            if augment_color:
                patch = tf.image.random_brightness(patch, max_delta=0.2)
                patch = tf.image.random_contrast(patch, lower=0.8, upper=1.2)
                if img_channels == 3:
                    patch = tf.image.random_saturation(patch, lower=0.8, upper=1.2)
                patch = tf.clip_by_value(patch, 0.0, 1.0)
            return patch

        patches = tf.stack([_one_patch(i) for i in range(patches_per_image)])
        return tf.data.Dataset.from_tensor_slices(patches)

    return _to_patches


def _make_filesystem_val_fn(img_size: int, img_channels: int):
    """Val-path factory: decode one image, return a single center-cropped tensor.

    Returns a callable ``path -> tf.Tensor`` suitable for ``.map()``.

    Args:
        img_size: Target square size.
        img_channels: Number of channels.

    Returns:
        Callable ``(path: tf.Tensor) -> tf.Tensor``.
    """
    import tensorflow as tf

    def _decode_val(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=img_channels)
        img = tf.cast(img, tf.float32) / 255.0
        shape = tf.shape(img)
        new_h = tf.maximum(shape[0], img_size)
        new_w = tf.maximum(shape[1], img_size)
        img = tf.image.resize(img, [new_h, new_w])
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
        return img

    return _decode_val


def _build_filesystem_dataset(
    train_glob: str,
    val_glob: str,
    img_size: int,
    img_channels: int,
    batch_size: int,
    augment: bool = True,
    augment_color: bool = True,
    patches_per_image: int = 4,
    seed: int = 42,
    dataset_label: str = "filesystem",
) -> Tuple[Any, Any, int, int]:
    """Build ``tf.data`` pipelines from raw filesystem JPEG directories.

    Training images are decoded at their original resolution; ``patches_per_image``
    random ``img_size × img_size`` crops are extracted from each via ``flat_map``.
    Validation images are center-cropped to ``img_size × img_size`` (one patch each).

    Args:
        train_glob: Glob pattern for training images (supports ``**``).
        val_glob: Glob pattern for validation images (supports ``**``).
        img_size: Patch size (square).
        img_channels: Number of output channels (3 for RGB).
        batch_size: Batch size.
        augment: Whether to apply random crop + flip augmentation.
        augment_color: Whether to apply photometric augmentation.
        patches_per_image: Number of random patches to extract per training image.
        seed: RNG seed for shuffle.
        dataset_label: Name used in log messages.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
        Both datasets emit ``(x, x)`` self-supervised pairs.
        ``val_steps`` is ``None`` — Keras exhausts the non-repeating val
        dataset naturally.

    Raises:
        FileNotFoundError: If no files match ``train_glob`` or ``val_glob``.
    """
    import tensorflow as tf

    train_files = sorted(_glob.glob(train_glob, recursive=True))
    val_files = sorted(_glob.glob(val_glob, recursive=True))

    if not train_files:
        raise FileNotFoundError(f"No training files matched: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"No validation files matched: {val_glob}")

    n_train = len(train_files)
    n_val = len(val_files)
    # Views-per-image depends on the augment branch: the augment path flat_maps
    # `patches_per_image` random crops per image, the no-augment path yields a
    # single centre-crop. Counting unconditionally overcounts steps_per_epoch
    # by `patches_per_image` on the --no-augment path, which also inflates the
    # cosine-LR horizon in _build_lr_schedule (total/warmup steps).
    # DECISION plan_2026-05-29_f1605e5a/D-001  (O3 + H11, single fix site)
    n_views = n_train * patches_per_image if augment else n_train
    steps_per_epoch = max(1, n_views // batch_size)

    _patch_fn = _make_filesystem_patch_fn(img_size, img_channels, patches_per_image, augment_color) if augment else None
    _val_fn   = _make_filesystem_val_fn(img_size, img_channels)

    if augment and _patch_fn is not None:
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_files)
            .shuffle(n_train, seed=seed, reshuffle_each_iteration=True)
            .flat_map(_patch_fn)
            .shuffle(batch_size * patches_per_image, reshuffle_each_iteration=True)
            .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        # No augmentation: single center-crop per image.
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_files)
            .shuffle(n_train, seed=seed, reshuffle_each_iteration=True)
            .map(_val_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_files)
        .map(_val_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    logger.info(
        f"{dataset_label}: {n_train} train / {n_val} val | "
        f"patches_per_image={patches_per_image} | "
        f"steps_per_epoch={steps_per_epoch} | img_size={img_size}"
    )
    return train_ds, val_ds, steps_per_epoch, None


def _build_mixed_filesystem_dataset(
    config: TrainingConfig,
) -> Tuple[Any, Any, int, None]:
    """Build a mixed ``tf.data`` pipeline from two or more filesystem datasets.

    Interleaves training images at the file-path level using size-proportional
    sampling weights so each batch draws from all sources proportionally.
    Validation images are concatenated into a single non-repeating dataset.

    Args:
        config: Training configuration.  ``config.datasets`` must contain two
            or more names from ``{"ade20k", "coco"}``.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch, None)``.
        ``val_steps`` is ``None`` — Keras exhausts the non-repeating val
        dataset naturally.

    Raises:
        FileNotFoundError: If no files match a glob for any dataset.
    """
    import tensorflow as tf

    _DATASET_GLOBS = {
        "ade20k": (
            os.path.join(config.ade20k_dir, "images", "ADE", "training", "**", "*.jpg"),
            os.path.join(config.ade20k_dir, "images", "ADE", "validation", "**", "*.jpg"),
        ),
        "coco": (
            os.path.join(config.coco_dir, "train2017", "*.jpg"),
            os.path.join(config.coco_dir, "val2017",   "*.jpg"),
        ),
    }

    all_train_files: List[List[str]] = []
    all_val_files: List[str] = []

    for ds_name in config.datasets:
        train_glob, val_glob = _DATASET_GLOBS[ds_name]
        train_files = sorted(_glob.glob(train_glob, recursive=True))
        val_files   = sorted(_glob.glob(val_glob,   recursive=True))
        if not train_files:
            raise FileNotFoundError(f"No training files matched ({ds_name}): {train_glob}")
        if not val_files:
            raise FileNotFoundError(f"No validation files matched ({ds_name}): {val_glob}")
        logger.info(
            f"{ds_name}: {len(train_files)} train / {len(val_files)} val"
        )
        all_train_files.append(train_files)
        all_val_files.extend(val_files)

    train_counts = [len(f) for f in all_train_files]
    total_train  = sum(train_counts)
    total_val    = len(all_val_files)
    weights      = [c / total_train for c in train_counts]
    steps_per_epoch = max(1, (total_train * config.patches_per_image) // config.batch_size)

    label = "+".join(config.datasets)
    logger.info(
        f"{label} (mixed): {total_train} train / {total_val} val | "
        f"patches_per_image={config.patches_per_image} | "
        f"steps_per_epoch={steps_per_epoch} | weights={[round(w, 3) for w in weights]}"
    )

    _patch_fn = _make_filesystem_patch_fn(
        config.img_size, config.img_channels,
        config.patches_per_image, config.augment_color,
    )
    _val_fn = _make_filesystem_val_fn(config.img_size, config.img_channels)

    # Per-source path datasets: shuffle + repeat so they never exhaust.
    path_datasets = [
        tf.data.Dataset.from_tensor_slices(files)
        .shuffle(len(files), seed=42, reshuffle_each_iteration=True)
        .repeat()
        for files in all_train_files
    ]

    train_ds = (
        tf.data.Dataset.sample_from_datasets(
            path_datasets,
            weights=weights,
            stop_on_empty_dataset=False,
            seed=42,
        )
        .flat_map(_patch_fn)
        .shuffle(config.batch_size * config.patches_per_image, reshuffle_each_iteration=True)
        .map(lambda x: (x, x),               num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(all_val_files)
        .map(_val_fn,          num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, steps_per_epoch, None


def build_dataset(
    config: TrainingConfig,
    smoke: bool = False,
) -> Tuple[Any, Any, int, int]:
    """Dispatch to the correct dataset builder.

    Args:
        config: Training configuration.
        smoke: If ``True``, returns a tiny synthetic dataset.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch, val_steps)``.

    Raises:
        ValueError: If ``config.dataset`` is not supported.
    """
    if smoke:
        return _build_smoke_dataset(config)
    if len(config.datasets) > 1:
        return _build_mixed_filesystem_dataset(config)
    if config.dataset in _CIFAR_STATS:
        return _build_cifar_dataset(config)
    if config.dataset == "ade20k":
        train_glob = os.path.join(config.ade20k_dir, "images", "ADE", "training", "**", "*.jpg")
        val_glob   = os.path.join(config.ade20k_dir, "images", "ADE", "validation", "**", "*.jpg")
        return _build_filesystem_dataset(
            train_glob, val_glob,
            img_size=config.img_size,
            img_channels=config.img_channels,
            batch_size=config.batch_size,
            augment=config.augment_data,
            augment_color=config.augment_color,
            patches_per_image=config.patches_per_image,
            dataset_label="ADE20K",
        )
    if config.dataset == "coco":
        train_glob = os.path.join(config.coco_dir, "train2017", "*.jpg")
        val_glob   = os.path.join(config.coco_dir, "val2017",   "*.jpg")
        return _build_filesystem_dataset(
            train_glob, val_glob,
            img_size=config.img_size,
            img_channels=config.img_channels,
            batch_size=config.batch_size,
            augment=config.augment_data,
            augment_color=config.augment_color,
            patches_per_image=config.patches_per_image,
            dataset_label="COCO2017",
        )
    raise ValueError(
        f"Unsupported dataset '{config.dataset}'. "
        "Supported: 'cifar10', 'cifar100', 'ade20k', 'coco'."
    )


# ---------------------------------------------------------------------------
# Reconstruction visualisation callback
# ---------------------------------------------------------------------------

class ReconVisualizationCallback(keras.callbacks.Callback):
    """Save side-by-side reconstruction grids every ``frequency`` epochs.

    Args:
        val_samples: Fixed validation batch, shape
            ``(N, H, W, C)`` in model-input space.
        save_dir: Directory for PNG files.
        frequency: Save every this many epochs (and on epoch 0).
        recon_loss_type: ``"mse"`` or ``"bce"`` — controls de-normalisation
            before clipping to ``[0, 1]`` for display.
        cifar_mean: Per-channel mean used during MSE normalisation.
        cifar_std: Per-channel std used during MSE normalisation.
    """

    def __init__(
        self,
        val_samples: np.ndarray,
        save_dir: str,
        frequency: int = 5,
        recon_loss_type: str = "mse",
        cifar_mean: Optional[np.ndarray] = None,
        cifar_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.val_samples = val_samples
        self.save_dir = save_dir
        self.frequency = frequency
        self.recon_loss_type = recon_loss_type
        self.cifar_mean = cifar_mean  # None = images already in [0,1], no denorm needed
        self.cifar_std  = cifar_std
        self._fixed_z: Optional[np.ndarray] = None  # fixed latents, lazy-init on first epoch
        os.makedirs(save_dir, exist_ok=True)

    def _to_display(self, x: np.ndarray) -> np.ndarray:
        """Undo normalisation and clip to ``[0, 1]``."""
        if self.recon_loss_type == "mse" and self.cifar_mean is not None:
            x = x * self.cifar_std + self.cifar_mean
        return np.clip(x, 0.0, 1.0)

    def _get_fixed_samples(self, n: int) -> np.ndarray:
        """Decode fixed latent codes through current model weights.

        Hierarchical models use ``model.sample(n)`` which already encodes
        the coherent joint-prior path via the learnable conditional prior
        ``p(z_l2 | z_l1)``. Hand-rolling ``decode(z_l1, z_l2)`` with
        independent N(0, I) is incoherent (see model_hierarchical.py docs).
        DECISION plan_2026-05-28_15256fe3/D-004
        """
        cfg = self.model.config
        if hasattr(cfg, "patches_per_side_l1"):
            decoded = self.model.sample(n, seed=42)
            return np.clip(np.array(decoded), 0.0, 1.0)
        if self._fixed_z is None:
            hp = wp = cfg.patches_per_side
            self._fixed_z = np.array(
                keras.random.normal((n, hp, wp, cfg.latent_dim), seed=42)
            )
        decoded = self.model.decode(keras.ops.convert_to_tensor(self._fixed_z))
        return np.clip(np.array(decoded), 0.0, 1.0)

    def _save_grid(self, path: str, originals: np.ndarray, recons: np.ndarray,
                   samples: np.ndarray, title: str) -> None:
        n = len(originals)
        cmap = "gray" if originals.shape[-1] == 1 else None
        fig, axes = plt.subplots(3, n, figsize=(n * 1.4, 4.8))
        row_labels = ["original", "recon", "sample"]
        for row, imgs in enumerate([originals, recons, samples]):
            for i in range(n):
                axes[row, i].imshow(imgs[i].squeeze(), cmap=cmap)
                axes[row, i].axis("off")
            axes[row, 0].set_ylabel(row_labels[row], fontsize=8)
        fig.suptitle(title, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Save a pre-training baseline grid (epoch 0000) before any weight updates."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            outputs = self.model(self.val_samples, training=False)
            originals = self._to_display(self.val_samples)
            recons    = self._to_display(np.array(outputs["reconstruction"]))
            samples   = self._get_fixed_samples(len(originals))
            path = os.path.join(self.save_dir, "recon_epoch_0000.png")
            self._save_grid(path, originals, recons, samples, "Epoch 0  |  pre-training baseline")
            logger.info(f"Pre-training reconstruction grid saved: {path}")
        except Exception as exc:
            logger.warning(f"ReconVisualizationCallback.on_train_begin failed: {exc}")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.frequency != 0 and epoch != 0:
            return
        try:
            outputs = self.model(self.val_samples, training=False)
            originals = self._to_display(self.val_samples)
            recons    = self._to_display(np.array(outputs["reconstruction"]))
            samples   = self._get_fixed_samples(len(originals))
            loss_val  = (logs or {}).get("loss", float("nan"))
            path = os.path.join(self.save_dir, f"recon_epoch_{epoch + 1:04d}.png")
            self._save_grid(path, originals, recons, samples,
                            f"Epoch {epoch + 1}  |  loss={loss_val:.4f}")
        except Exception as exc:
            logger.warning(f"ReconVisualizationCallback failed at epoch {epoch}: {exc}")

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Save a final reconstruction grid at end of training."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            outputs = self.model(self.val_samples, training=False)
            originals = self._to_display(self.val_samples)
            recons    = self._to_display(np.array(outputs["reconstruction"]))
            samples   = self._get_fixed_samples(len(originals))
            path = os.path.join(self.save_dir, "recon_final.png")
            self._save_grid(path, originals, recons, samples, "Final reconstruction")
            logger.info(f"Final reconstruction grid saved: {path}")
        except Exception as exc:
            logger.warning(f"ReconVisualizationCallback.on_train_end failed: {exc}")


# ---------------------------------------------------------------------------
# Beta annealing callback
# ---------------------------------------------------------------------------

class BetaAnnealingCallback(keras.callbacks.Callback):
    """Linearly ramps ``getattr(model, attr_name)`` from start to target.

    Mutation happens in ``on_epoch_begin`` (before any forward pass that epoch).
    Safe because the target attribute is a plain Python float evaluated
    eagerly each call (jit_compile=False enforced).

    Args:
        beta_start: Initial beta at epoch 0.
        beta_target: Beta after ``anneal_epochs`` epochs.
        anneal_epochs: Epochs to ramp over. ``<= 0`` disables.
        attr_name: Attribute on the model to mutate. Single-scale model
            uses ``"_beta_kl"``; hierarchical model uses ``"_beta_kl_l1"``
            or ``"_beta_kl_l2"`` for the two-scale staggered schedule.
    """

    def __init__(
        self,
        beta_start: float,
        beta_target: float,
        anneal_epochs: int,
        attr_name: str = "_beta_kl",
    ) -> None:
        super().__init__()
        self.beta_start = beta_start
        self.beta_target = beta_target
        self.anneal_epochs = anneal_epochs
        self.attr_name = attr_name

    def on_train_begin(self, logs=None) -> None:
        # Fast-forward beta to the correct value when resuming mid-anneal.
        # initial_epoch > 0 only when model.fit is called with initial_epoch=N.
        initial_epoch = int(self.params.get("initial_epoch", 0))
        if self.anneal_epochs > 0 and initial_epoch > 0:
            progress = min(1.0, initial_epoch / self.anneal_epochs)
            setattr(
                self.model,
                self.attr_name,
                self.beta_start + progress * (self.beta_target - self.beta_start),
            )

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        if self.anneal_epochs <= 0:
            return
        progress = min(1.0, epoch / self.anneal_epochs)
        new_beta = self.beta_start + progress * (self.beta_target - self.beta_start)
        setattr(self.model, self.attr_name, new_beta)
        logger.info(
            f"BetaAnnealing[{self.attr_name}] epoch={epoch}: beta={new_beta:.4f}"
        )


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def _build_lr_schedule(
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Any:
    """Build cosine-decay schedule with optional linear warmup.

    Args:
        config: Training configuration.
        steps_per_epoch: Number of gradient steps per epoch.

    Returns:
        A Keras learning-rate schedule object.
    """
    total_steps  = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    decay_steps  = max(1, total_steps - warmup_steps)

    if warmup_steps > 0:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,  # true linear ramp from 0 → warmup_target
            decay_steps=decay_steps,
            alpha=1e-6,
            warmup_target=config.learning_rate,
            warmup_steps=warmup_steps,
        )
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=decay_steps,
        alpha=1e-6,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: TrainingConfig, smoke: bool = False) -> None:
    """Run one training experiment.

    Args:
        config: Fully-validated :class:`TrainingConfig`.
        smoke: If ``True``, use tiny synthetic data and skip GPU.
    """
    logger.info(f"Starting ConvNeXtPatchVAE training: {config.experiment_name}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_ds, val_ds, steps_per_epoch, val_steps = build_dataset(config, smoke=smoke)
    if config.steps_per_epoch_override is not None:
        steps_per_epoch = config.steps_per_epoch_override
        logger.info(f"steps_per_epoch overridden to {steps_per_epoch}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if config.hierarchical:
        h_model_config = config.to_hierarchical_model_config()
        model = HierarchicalConvNeXtPatchVAE(
            config=h_model_config, name="hierarchical_convnext_patch_vae",
        )
        logger.info(
            "Model: HIERARCHICAL | "
            f"L1 patch={h_model_config.patch_size_l1} latent={h_model_config.latent_dim_l1} "
            f"embed={h_model_config.embed_dim_l1} depth={h_model_config.encoder_depth_l1} | "
            f"L2 patch={h_model_config.patch_size_l2} latent={h_model_config.latent_dim_l2} "
            f"embed={h_model_config.embed_dim_l2} depth={h_model_config.encoder_depth_l2} | "
            f"tile_factor={h_model_config.tile_factor}"
        )
    else:
        model_config = config.to_model_config()
        model = ConvNeXtPatchVAE(config=model_config, name="convnext_patch_vae")
        logger.info(
            f"Model: variant={config.model_variant or 'custom'} | "
            f"embed={model_config.embed_dim} | depth={model_config.encoder_depth} | "
            f"latent={model_config.latent_dim} | patch={model_config.patch_size}"
        )

    dummy = keras.ops.zeros(
        (1, config.img_size, config.img_size, config.img_channels)
    )
    model(dummy, training=False)
    model.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # Compile  —  losses live in add_loss; compile(loss=...) must be None.
    # jit_compile=False: XLA tracing fails on ops.reshape in _compute_sigreg.
    # ------------------------------------------------------------------
    lr_schedule = _build_lr_schedule(config, steps_per_epoch)
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
    )
    model.compile(optimizer=optimizer, loss=None, jit_compile=False)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    results_prefix = (
        "hierarchical_convnext_patch_vae" if config.hierarchical
        else "convnext_patch_vae"
    )
    callbacks, results_dir = create_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix=results_prefix,
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    curves_dir = os.path.join(results_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)
    callbacks.append(TrainingCurvesCallback(output_dir=curves_dir))

    if config.beta_anneal_epochs > 0:
        if config.hierarchical:
            # Shared schedule length across L1+L2; distinct targets/starts
            # preserved because coarse vs fine KL strength is an arch choice.
            # DECISION plan_2026-05-28_15256fe3/D-003
            callbacks.append(
                BetaAnnealingCallback(
                    beta_start=config.beta_kl_l1_start,
                    beta_target=config.beta_kl_l1,
                    anneal_epochs=config.beta_anneal_epochs,
                    attr_name="_beta_kl_l1",
                )
            )
            callbacks.append(
                BetaAnnealingCallback(
                    beta_start=config.beta_kl_l2_start,
                    beta_target=config.beta_kl_l2,
                    anneal_epochs=config.beta_anneal_epochs,
                    attr_name="_beta_kl_l2",
                )
            )
        else:
            callbacks.append(
                BetaAnnealingCallback(
                    beta_start=config.beta_kl_start,
                    beta_target=config.beta_kl,
                    anneal_epochs=config.beta_anneal_epochs,
                )
            )

    # Viz callbacks work for both single-scale and hierarchical models
    # (hierarchical branches in each callback dispatch on
    # `model.config.patches_per_side_l1`).
    # DECISION plan_2026-05-28_15256fe3/D-002
    recon_dir = os.path.join(results_dir, "reconstructions")
    # MSE + CIFAR: apply per-channel denorm in viz callback. All other
    # combos: pass None.
    if config.recon_loss_type == "mse" and config.dataset in _CIFAR_STATS:
        _ds_mean, _ds_std = _CIFAR_STATS[config.dataset]
    else:
        _ds_mean, _ds_std = None, None
    val_samples = None
    try:
        sample_batch = next(iter(val_ds))
        val_samples = np.array(sample_batch[0][:8])
        callbacks.append(
            ReconVisualizationCallback(
                val_samples=val_samples,
                save_dir=recon_dir,
                frequency=1,
                recon_loss_type=config.recon_loss_type,
                cifar_mean=_ds_mean,
                cifar_std=_ds_std,
            )
        )
    except Exception as exc:
        logger.warning(f"Could not set up ReconVisualizationCallback: {exc}")

    # Latent space PCA scatter and interpolation grids. Collect up to
    # 128 images from the validation set for the PCA scatter; re-use
    # the same 8 fixed images for interpolation pairs.
    if val_samples is not None:
        try:
            viz_frequency = 1
            latent_dir = os.path.join(results_dir, "latent_space")
            interp_dir = os.path.join(results_dir, "interpolations")

            latent_batches = [val_samples]
            for extra_batch in val_ds.take(15):
                latent_batches.append(np.array(extra_batch[0][:8]))
            latent_viz_images = np.concatenate(latent_batches, axis=0)[:128]

            callbacks.append(
                LatentSpaceCallback(
                    val_images=latent_viz_images,
                    save_dir=latent_dir,
                    frequency=viz_frequency,
                    cifar_mean=_ds_mean,
                    cifar_std=_ds_std,
                )
            )
            callbacks.append(
                LatentInterpolationCallback(
                    val_samples=val_samples,
                    save_dir=interp_dir,
                    frequency=viz_frequency,
                    num_steps=8,
                    cifar_mean=_ds_mean,
                    cifar_std=_ds_std,
                )
            )
        except Exception as exc:
            logger.warning(f"Could not set up latent visualization callbacks: {exc}")

    # ------------------------------------------------------------------
    # Persist config before fit (helps debug mid-run crashes)
    # ------------------------------------------------------------------
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, default=str)
    logger.info(f"Config saved to {config_path}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Save final model + reload check
    # ------------------------------------------------------------------
    final_path = os.path.join(results_dir, "final_model.keras")
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    try:
        reloaded = keras.models.load_model(final_path, custom_objects=CUSTOM_OBJECTS)
        # Compare deterministic encoder mu (not reconstruction — Sampling adds
        # per-call noise, so two independent calls always differ by design).
        # Hierarchical model returns (mu_l1, log_var_l1, mu_l2, log_var_l2);
        # single-scale returns (mu, log_var). Index 0 is the leading mu in
        # both cases; for hierarchical we additionally check mu_l2.
        if config.hierarchical:
            ref_mu_l1, _, ref_mu_l2, _ = model.encode(dummy)
            new_mu_l1, _, new_mu_l2, _ = reloaded.encode(dummy)
            delta_l1 = float(np.max(np.abs(np.array(ref_mu_l1) - np.array(new_mu_l1))))
            delta_l2 = float(np.max(np.abs(np.array(ref_mu_l2) - np.array(new_mu_l2))))
            max_delta = max(delta_l1, delta_l2)
            logger.info(
                f"Reload mu_l1 max|delta|={delta_l1:.2e}, "
                f"mu_l2 max|delta|={delta_l2:.2e}"
            )
        else:
            ref_mu = np.array(model.encode(dummy)[0])
            new_mu = np.array(reloaded.encode(dummy)[0])
            max_delta = float(np.max(np.abs(ref_mu - new_mu)))
        if max_delta < 1e-4:
            logger.info(f"Reload check PASSED: max|delta|={max_delta:.2e}")
        else:
            logger.error(f"Reload check FAILED: max|delta|={max_delta:.2e} >= 1e-4")
            sys.exit(1)
    except Exception as exc:
        logger.error(f"Reload check raised an exception: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Success guard — advisory; does not fail the run
    # ------------------------------------------------------------------
    val_losses = history.history.get("val_loss", [])
    if val_losses:
        best_val_loss = min(val_losses)
        epochs_run    = len(val_losses)
        # Loss-type-aware threshold: BCE loss on [0,1] images floors around
        # 0.1-0.2 for a converged model — the MSE default of 0.02 is
        # unreachable and produces a permanent false-negative advisory (H4).
        # Respect any explicit user override of --success-threshold.
        _loss_thresholds = {"bce": 0.15, "mse": 0.02}
        effective_threshold = (
            _loss_thresholds.get(config.recon_loss_type, config.success_threshold)
            if config.success_threshold == 0.02
            else config.success_threshold
        )
        converged = (
            best_val_loss <= effective_threshold
            and epochs_run >= 0.5 * config.epochs
        )
        if converged:
            logger.info(
                f"TRAINING CONVERGED: best val_loss={best_val_loss:.4f} "
                f"<= threshold={effective_threshold}"
            )
        else:
            logger.warning(
                f"TRAINING MAY NOT HAVE CONVERGED: "
                f"best val_loss={best_val_loss:.4f}, "
                f"threshold={effective_threshold}, "
                f"epochs_run={epochs_run}/{config.epochs}. "
                "Consider increasing --epochs or tuning --beta-kl / --lambda-sigreg."
            )

    # ------------------------------------------------------------------
    # Final metrics summary
    # ------------------------------------------------------------------
    h = history.history
    def _best(key: str) -> str:
        vals = h.get(key, [])
        return f"{min(vals):.4f}" if vals else "n/a"

    logger.info(
        "Training summary | "
        f"best val_loss={_best('val_loss')} | "
        f"best val_recon_loss={_best('val_recon_loss')} | "
        f"best val_kl_loss={_best('val_kl_loss')} | "
        f"best val_sigreg_loss={_best('val_sigreg_loss')} | "
        f"epochs_run={len(h.get('val_loss', []))}/{config.epochs}"
    )
    logger.info(f"Results written to: {results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = create_base_argument_parser(
        description="Train ConvNeXtPatchVAE (per-patch VAE with SIGReg anti-collapse)",
        default_dataset="cifar100",
        dataset_choices=["cifar10", "cifar100", "ade20k", "coco"],
    )

    # Smoke
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Run a tiny smoke test on synthetic data (CPU, <60 s). "
             "Overrides --epochs to 3 and uses a tiny model.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["cifar10", "cifar100", "ade20k", "coco"],
        metavar="DATASET",
        help="One or more datasets to mix (e.g. --datasets ade20k coco). "
             "Overrides --dataset when specified. Only 'ade20k' and 'coco' "
             "may be combined. Single value behaves like --dataset.",
    )

    # Model
    parser.add_argument(
        "--variant", type=str, default=None, dest="model_variant",
        choices=["tiny", "base", "large"],
        help="Preset variant (overrides --embed-dim / --depth / --latent-dim).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=4,
        help="Stem stride / patch size (img_size must be divisible).",
    )
    parser.add_argument("--embed-dim",      type=int,   default=128)
    parser.add_argument("--encoder-depth",  type=int,   default=4)
    parser.add_argument("--decoder-depth",  type=int,   default=4)
    parser.add_argument("--kernel-size",    type=int,   default=7)
    parser.add_argument("--latent-dim",     type=int,   default=16)
    parser.add_argument(
        "--beta-kl", type=float, default=0.5,
        help="KL divergence weight (beta-VAE style).",
    )
    parser.add_argument(
        "--lambda-sigreg", type=float, default=0.1,
        help="SIGReg anti-collapse weight. Set 0 for T1b ablation.",
    )
    parser.add_argument("--sigreg-knots",    type=int,   default=17)
    parser.add_argument("--sigreg-num-proj", type=int,   default=256)
    parser.add_argument(
        "--recon-loss-type", type=str, default="bce", choices=["mse", "bce"],
        help="Reconstruction loss family. BCE requires inputs in [0,1].",
    )
    parser.add_argument("--dropout",    type=float, default=0.0, dest="dropout_rate")
    parser.add_argument("--gamma-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_false", dest="augment_data",
                        default=True)
    parser.add_argument("--no-color-augment", action="store_false", dest="augment_color",
                        default=True,
                        help="Disable photometric augmentation (brightness/contrast/saturation).")
    parser.add_argument("--patches-per-image", type=int, default=4, dest="patches_per_image",
                        help="Random patches to extract per training image via flat_map.")

    # Training (extend base parser defaults)
    parser.add_argument("--warmup-epochs",      type=int,   default=5)
    parser.add_argument("--beta-kl-start",      type=float, default=0.0001,
                        help="Initial beta at epoch 0 for beta annealing.")
    parser.add_argument("--beta-anneal-epochs", type=int,   default=15,
                        help="Epochs to ramp beta from beta-kl-start to beta-kl. 0=disabled.")
    parser.add_argument("--success-threshold", type=float, default=0.02,
                        help="val_loss threshold for the convergence advisory.")

    # ------------------------------------------------------------------
    # Hierarchical (two-level) variant
    # ------------------------------------------------------------------
    parser.add_argument(
        "--hierarchical", action="store_true", default=False,
        help="Enable the two-level hierarchical variant "
             "(HierarchicalConvNeXtPatchVAE). --patch-size and --latent-dim "
             "become the L2 scale; --patch-size-l1 / --latent-dim-l1 / "
             "--embed-dim-l1 control the L1 (coarse) scale.",
    )
    parser.add_argument("--patch-size-l1", type=int, default=32,
                        help="L1 (coarse) patch size when --hierarchical.")
    parser.add_argument("--embed-dim-l1", type=int, default=128,
                        help="L1 ConvNeXt block width when --hierarchical.")
    parser.add_argument("--encoder-depth-l1", type=int, default=4)
    parser.add_argument("--decoder-depth-l1", type=int, default=4)
    parser.add_argument("--latent-dim-l1", type=int, default=64,
                        help="L1 per-patch latent dim when --hierarchical.")
    parser.add_argument("--beta-kl-l1", type=float, default=0.5)
    parser.add_argument("--beta-kl-l2", type=float, default=0.5)
    parser.add_argument("--beta-kl-l1-start", type=float, default=0.0001)
    parser.add_argument("--beta-kl-l2-start", type=float, default=0.0001)
    parser.add_argument("--lambda-sigreg-l1", type=float, default=0.05)
    parser.add_argument("--lambda-sigreg-l2", type=float, default=0.1)

    # Large-dataset filesystem paths
    parser.add_argument(
        "--ade20k-dir", type=str, default="/media/arxwn/data0_4tb/datasets/ade20k",
        help="Root directory of the ADE20K dataset (contains images/ADE/training/).",
    )
    parser.add_argument(
        "--coco-dir", type=str, default="/media/arxwn/data0_4tb/datasets/coco_2017",
        help="Root directory of the COCO 2017 dataset (contains train2017/ and val2017/).",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None, dest="steps_per_epoch_override",
        help="Override computed steps_per_epoch. Useful for quick pipeline validation.",
    )

    # Set script-appropriate base-parser defaults
    parser.set_defaults(
        epochs=50,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=10,
        image_size=32,   # CIFAR default; override to 128/256 for ade20k/coco
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    smoke = args.smoke
    if smoke:
        args.epochs        = 3
        args.image_size    = 32   # smoke uses synthetic data — 32×32 is sufficient
        args.model_variant = "tiny"
        args.embed_dim     = 16
        args.encoder_depth = 1
        args.decoder_depth = 1
        args.latent_dim    = 4
        args.sigreg_knots  = 5
        args.sigreg_num_proj = 32
        args.patch_size    = 4
        logger.info("Smoke mode: tiny model, 3 epochs, 32×32, synthetic data.")

    # GPU must be configured before any TF/Keras context is created.
    if not smoke:
        setup_gpu(args.gpu)

    # --datasets overrides --dataset when provided.
    datasets = getattr(args, "datasets", None) or [args.dataset]

    image_size = getattr(args, "image_size", 32)
    if any(d in ("ade20k", "coco") for d in datasets) and image_size == 32:
        logger.warning(
            f"--datasets {datasets} with default --image-size 32. "
            "Consider passing --image-size 128 or --image-size 256."
        )

    config = TrainingConfig(
        dataset=datasets[0],
        datasets=datasets,
        img_size=getattr(args, "image_size", 32),
        img_channels=3,
        augment_data=args.augment_data,
        augment_color=args.augment_color,
        patches_per_image=args.patches_per_image,
        model_variant=args.model_variant,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        kernel_size=args.kernel_size,
        latent_dim=args.latent_dim,
        beta_kl=args.beta_kl,
        lambda_sigreg=args.lambda_sigreg,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        recon_loss_type=args.recon_loss_type,
        dropout_rate=args.dropout_rate,
        gamma_clip=args.gamma_clip,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        beta_kl_start=args.beta_kl_start,
        beta_anneal_epochs=args.beta_anneal_epochs,
        early_stopping_patience=args.patience,
        success_threshold=args.success_threshold,
        ade20k_dir=args.ade20k_dir,
        coco_dir=args.coco_dir,
        steps_per_epoch_override=args.steps_per_epoch_override,
        hierarchical=args.hierarchical,
        patch_size_l1=args.patch_size_l1,
        embed_dim_l1=args.embed_dim_l1,
        encoder_depth_l1=args.encoder_depth_l1,
        decoder_depth_l1=args.decoder_depth_l1,
        latent_dim_l1=args.latent_dim_l1,
        beta_kl_l1=args.beta_kl_l1,
        beta_kl_l2=args.beta_kl_l2,
        beta_kl_l1_start=args.beta_kl_l1_start,
        beta_kl_l2_start=args.beta_kl_l2_start,
        lambda_sigreg_l1=args.lambda_sigreg_l1,
        lambda_sigreg_l2=args.lambda_sigreg_l2,
    )

    try:
        train(config, smoke=smoke)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
