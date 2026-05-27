"""Training script for CliffordNetPatchVAEV2 — hierarchical Clifford VAE.

Sibling of ``train_convnext_patch_vae_v2.py``. Same training loop, same
losses, same callbacks; the only differences are the model factory, the
hierarchical CLI flags (``--stage-dims``, ``--stage-depths``,
``--stage-shifts``, ``--downsample-kind``, ``--cli-mode``, ...) and the
output directory prefix.

Smoke (CPU, tiny model, synthetic data)::

    .venv/bin/python -m \\
        train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 --smoke

CIFAR-10 with VAE + LPIPS + MAE + classification head::

    CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg .venv/bin/python -m \\
        train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 \\
        --dataset cifar10 --variant base \\
        --recon-loss-type bce --lambda-lpips 0.1 \\
        --mae-mask-ratio 0.5 --lambda-mae 1.0 \\
        --use-classification-head --num-classes-cls 10 --lambda-cls 1.0 \\
        --epochs 50

Custom hierarchical config (override preset stages)::

    MPLBACKEND=Agg .venv/bin/python -m \\
        train.cliffordnet_patch_vae_v2.train_cliffordnet_patch_vae_v2 \\
        --dataset cifar10 --variant base \\
        --stage-dims 64 128 --stage-depths 2 2 \\
        --stage-shifts '[[1,2],[1,2,4]]' --downsample-kind blur
"""

# MPLBACKEND must be set before any matplotlib import — headless guard.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import dataclasses
import glob as _glob
import json
import sys
from typing import Any, List, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.cliffordnet_patch_vae_v2.config import (
    CliffordNetPatchVAEV2Config,
    PRESETS,
)
from dl_techniques.models.cliffordnet_patch_vae_v2.model import (
    CliffordNetPatchVAEV2,
)
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from train.common import setup_gpu, create_base_argument_parser, create_callbacks

from .callbacks import BetaAnnealingCallback, MaskedReconViz


CUSTOM_OBJECTS = {
    "CliffordNetPatchVAEV2": CliffordNetPatchVAEV2,
}

_CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
_CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
_CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
_CIFAR_STATS = {
    "cifar10": (_CIFAR10_MEAN, _CIFAR10_STD),
    "cifar100": (_CIFAR100_MEAN, _CIFAR100_STD),
}


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrainingConfig:
    """All hyperparameters for one CliffordNetPatchVAEV2 training run."""

    # Dataset
    dataset: str = "cifar10"
    img_size: int = 32
    img_channels: int = 3
    augment_data: bool = True
    augment_color: bool = True
    patches_per_image: int = 4

    # Model
    model_variant: Optional[str] = "base"
    patch_size: int = 4
    stage_dims: Optional[List[int]] = None
    stage_depths: Optional[List[int]] = None
    stage_shifts: Optional[List[List[int]]] = None
    downsample_kind: str = "blur"
    downsample_kernel_size: int = 7
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context_in_last_stage: bool = True
    layer_scale_init: float = 1e-5
    block_drop_path_rate: float = 0.0
    latent_dim: Optional[int] = None
    beta_kl: float = 0.5
    lambda_sigreg: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 256
    recon_loss_type: str = "bce"
    gamma_clip: float = 1.0

    # Multi-task
    mae_mask_ratio: float = 0.0
    lambda_mae: float = 1.0
    lambda_lpips: float = 0.0
    use_classification_head: bool = False
    num_classes_cls: int = 0
    cls_head_dropout: float = 0.0
    cls_head_num_heads: int = 4
    lambda_cls: float = 1.0
    use_segmentation_head: bool = False
    num_classes_seg: int = 0
    seg_head_dropout: float = 0.0
    lambda_seg: float = 1.0

    # Training
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    beta_kl_start: float = 0.0001
    beta_anneal_epochs: int = 15
    early_stopping_patience: int = 10

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Large-dataset filesystem paths
    ade20k_dir: str = "/media/arxwn/data0_4tb/datasets/ade20k"
    coco_dir: str = "/media/arxwn/data0_4tb/datasets/coco_2017"
    steps_per_epoch_override: Optional[int] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = (
                f"cliffordnet_v2_{self.dataset}_{self.model_variant or 'custom'}"
            )
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by patch_size "
                f"{self.patch_size}"
            )

    def to_model_config(self) -> CliffordNetPatchVAEV2Config:
        """Resolve a `CliffordNetPatchVAEV2Config` from this dataclass.

        Preset values fill in stage_dims / stage_depths / stage_shifts /
        latent_dim ONLY if the CLI did not override them (None sentinels).
        """
        if self.model_variant in PRESETS:
            preset = PRESETS[self.model_variant]
            stage_dims = (
                self.stage_dims if self.stage_dims is not None
                else preset["stage_dims"]
            )
            stage_depths = (
                self.stage_depths if self.stage_depths is not None
                else preset["stage_depths"]
            )
            stage_shifts = (
                self.stage_shifts if self.stage_shifts is not None
                else preset["stage_shifts"]
            )
            latent_dim = (
                self.latent_dim if self.latent_dim is not None
                else preset["latent_dim"]
            )
        else:
            stage_dims = self.stage_dims or [96, 192]
            stage_depths = self.stage_depths or [2, 4]
            stage_shifts = self.stage_shifts or [[1, 2], [1, 2, 4]]
            latent_dim = self.latent_dim or 16
        return CliffordNetPatchVAEV2Config(
            img_size=self.img_size,
            img_channels=self.img_channels,
            patch_size=self.patch_size,
            stage_dims=list(stage_dims),
            stage_depths=list(stage_depths),
            stage_shifts=[list(s) for s in stage_shifts],
            downsample_kind=self.downsample_kind,
            downsample_kernel_size=self.downsample_kernel_size,
            cli_mode=self.cli_mode,
            ctx_mode=self.ctx_mode,
            use_global_context_in_last_stage=self.use_global_context_in_last_stage,
            layer_scale_init=self.layer_scale_init,
            block_drop_path_rate=self.block_drop_path_rate,
            latent_dim=latent_dim,
            beta_kl=self.beta_kl,
            lambda_sigreg=self.lambda_sigreg,
            sigreg_knots=self.sigreg_knots,
            sigreg_num_proj=self.sigreg_num_proj,
            recon_loss_type=self.recon_loss_type,
            gamma_clip=self.gamma_clip,
            mae_mask_ratio=self.mae_mask_ratio,
            lambda_mae=self.lambda_mae,
            lambda_lpips=self.lambda_lpips,
            use_classification_head=self.use_classification_head,
            num_classes_cls=self.num_classes_cls,
            cls_head_dropout=self.cls_head_dropout,
            cls_head_num_heads=self.cls_head_num_heads,
            lambda_cls=self.lambda_cls,
            use_segmentation_head=self.use_segmentation_head,
            num_classes_seg=self.num_classes_seg,
            seg_head_dropout=self.seg_head_dropout,
            lambda_seg=self.lambda_seg,
        )


# ---------------------------------------------------------------------------
# Dataset builders (verbatim from v2 — they are model-agnostic)
# ---------------------------------------------------------------------------


def _build_cifar_dataset(config: TrainingConfig) -> Tuple[Any, Any, int, int]:
    import tensorflow as tf

    loader = (
        keras.datasets.cifar10 if config.dataset == "cifar10"
        else keras.datasets.cifar100
    )
    (x_train, y_train), (x_test, y_test) = loader.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.astype("int32").reshape(-1)
    y_test = y_test.astype("int32").reshape(-1)

    if config.recon_loss_type == "mse":
        mean, std = _CIFAR_STATS[config.dataset]
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

    def _augment(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(
            x, [config.img_size, config.img_size, config.img_channels]
        )
        if config.augment_color and config.recon_loss_type == "bce":
            x = tf.image.random_brightness(x, max_delta=0.2)
            x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
            x = tf.clip_by_value(x, 0.0, 1.0)
        return x

    def _make_pair(x: tf.Tensor, y: tf.Tensor):
        if config.use_classification_head:
            return ({"image": x, "label_cls": y}, x)
        return (x, x)

    steps_per_epoch = max(1, len(x_train) // config.batch_size)
    val_steps = max(1, len(x_test) // config.batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if config.augment_data:
        train_ds = train_ds.map(
            lambda x, y: (_augment(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = (
        train_ds
        .shuffle(10_000)
        .map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config.batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    logger.info(
        "%s: %d train / %d val | steps_per_epoch=%d | recon=%s | cls_head=%s",
        config.dataset.upper(), len(x_train), len(x_test),
        steps_per_epoch, config.recon_loss_type, config.use_classification_head,
    )
    return train_ds, val_ds, steps_per_epoch, val_steps


def _build_smoke_dataset(config: TrainingConfig) -> Tuple[Any, Any, int, int]:
    import tensorflow as tf

    n_train, n_val = config.batch_size * 3, config.batch_size * 2
    shape = (config.img_size, config.img_size, config.img_channels)
    rng = np.random.default_rng(42)
    x_train = rng.uniform(0.0, 1.0, (n_train,) + shape).astype("float32")
    x_val = rng.uniform(0.0, 1.0, (n_val,) + shape).astype("float32")

    if config.use_classification_head:
        n_cls = max(2, config.num_classes_cls)
        y_train = rng.integers(0, n_cls, size=(n_train,), dtype=np.int32)
        y_val = rng.integers(0, n_cls, size=(n_val,), dtype=np.int32)
    else:
        y_train = np.zeros((n_train,), dtype=np.int32)
        y_val = np.zeros((n_val,), dtype=np.int32)

    def _make_pair(x: tf.Tensor, y: tf.Tensor):
        if config.use_classification_head:
            return ({"image": x, "label_cls": y}, x)
        return (x, x)

    def _make(x, y, repeat: bool):
        ds = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .map(_make_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(config.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        if repeat:
            ds = ds.repeat()
        return ds

    return _make(x_train, y_train, True), _make(x_val, y_val, False), 3, 2


def _build_filesystem_dataset(
    train_glob: str,
    val_glob: str,
    img_size: int,
    img_channels: int,
    batch_size: int,
    patches_per_image: int,
    augment: bool,
    augment_color: bool,
    label: str,
) -> Tuple[Any, Any, int, None]:
    """Raw-filesystem JPEG dataset (ADE20K / COCO).

    Seg-mask loading is deferred (mirrors v2 D-006); training is
    VAE/LPIPS/MAE-only on these large datasets.
    """
    import tensorflow as tf

    train_files = sorted(_glob.glob(train_glob, recursive=True))
    val_files = sorted(_glob.glob(val_glob, recursive=True))
    if not train_files:
        raise FileNotFoundError(f"No training files matched: {train_glob}")
    if not val_files:
        raise FileNotFoundError(f"No validation files matched: {val_glob}")

    def _patch(path: tf.Tensor) -> tf.data.Dataset:
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=img_channels)
        img = tf.cast(img, tf.float32) / 255.0
        shape = tf.shape(img)
        new_h = tf.maximum(shape[0], img_size + 4)
        new_w = tf.maximum(shape[1], img_size + 4)
        img = tf.image.resize(img, [new_h, new_w])

        def _one(_):
            p = tf.image.random_crop(img, [img_size, img_size, img_channels])
            p = tf.image.random_flip_left_right(p)
            if augment_color:
                p = tf.image.random_brightness(p, max_delta=0.2)
                p = tf.image.random_contrast(p, lower=0.8, upper=1.2)
                if img_channels == 3:
                    p = tf.image.random_saturation(p, lower=0.8, upper=1.2)
                p = tf.clip_by_value(p, 0.0, 1.0)
            return p

        patches = tf.stack([_one(i) for i in range(patches_per_image)])
        return tf.data.Dataset.from_tensor_slices(patches)

    def _val(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=img_channels)
        img = tf.cast(img, tf.float32) / 255.0
        shape = tf.shape(img)
        new_h = tf.maximum(shape[0], img_size)
        new_w = tf.maximum(shape[1], img_size)
        img = tf.image.resize(img, [new_h, new_w])
        img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)
        return img

    n_train = len(train_files)
    steps_per_epoch = max(1, (n_train * patches_per_image) // batch_size)

    if augment:
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_files)
            .shuffle(n_train, seed=42, reshuffle_each_iteration=True)
            .flat_map(_patch)
            .shuffle(batch_size * patches_per_image, reshuffle_each_iteration=True)
            .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        train_ds = (
            tf.data.Dataset.from_tensor_slices(train_files)
            .shuffle(n_train, seed=42, reshuffle_each_iteration=True)
            .map(_val, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_files)
        .map(_val, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    logger.info(
        "%s: %d train / %d val | patches=%d | steps_per_epoch=%d",
        label, n_train, len(val_files), patches_per_image, steps_per_epoch,
    )
    return train_ds, val_ds, steps_per_epoch, None


def build_dataset(
    config: TrainingConfig, smoke: bool = False
) -> Tuple[Any, Any, int, Optional[int]]:
    if smoke:
        return _build_smoke_dataset(config)
    if config.dataset in _CIFAR_STATS:
        return _build_cifar_dataset(config)
    if config.dataset == "ade20k":
        if config.use_classification_head or config.use_segmentation_head:
            logger.warning(
                "ADE20K loader runs VAE/LPIPS/MAE-only — cls/seg labels are "
                "not wired. Disable --use-* flags or use CIFAR."
            )
        return _build_filesystem_dataset(
            os.path.join(config.ade20k_dir, "images", "ADE", "training", "**", "*.jpg"),
            os.path.join(config.ade20k_dir, "images", "ADE", "validation", "**", "*.jpg"),
            img_size=config.img_size,
            img_channels=config.img_channels,
            batch_size=config.batch_size,
            patches_per_image=config.patches_per_image,
            augment=config.augment_data,
            augment_color=config.augment_color,
            label="ADE20K",
        )
    if config.dataset == "coco":
        if config.use_classification_head or config.use_segmentation_head:
            logger.warning(
                "COCO loader runs VAE/LPIPS/MAE-only — cls/seg labels are "
                "not wired. Disable --use-* flags or use CIFAR."
            )
        return _build_filesystem_dataset(
            os.path.join(config.coco_dir, "train2017", "*.jpg"),
            os.path.join(config.coco_dir, "val2017", "*.jpg"),
            img_size=config.img_size,
            img_channels=config.img_channels,
            batch_size=config.batch_size,
            patches_per_image=config.patches_per_image,
            augment=config.augment_data,
            augment_color=config.augment_color,
            label="COCO2017",
        )
    raise ValueError(
        f"Unsupported dataset '{config.dataset}'. "
        "Supported: 'cifar10', 'cifar100', 'ade20k', 'coco'."
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _build_lr_schedule(config: TrainingConfig, steps_per_epoch: int):
    total = config.epochs * steps_per_epoch
    warmup = config.warmup_epochs * steps_per_epoch
    decay = max(1, total - warmup)
    if warmup > 0:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=decay,
            alpha=1e-6,
            warmup_target=config.learning_rate,
            warmup_steps=warmup,
        )
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=decay,
        alpha=1e-6,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(config: TrainingConfig, smoke: bool = False) -> None:
    logger.info(
        "Starting CliffordNetPatchVAEV2 training: %s", config.experiment_name
    )

    train_ds, val_ds, steps_per_epoch, val_steps = build_dataset(config, smoke=smoke)
    if config.steps_per_epoch_override is not None:
        steps_per_epoch = config.steps_per_epoch_override
        logger.info("steps_per_epoch overridden to %d", steps_per_epoch)

    model_config = config.to_model_config()
    model = CliffordNetPatchVAEV2(
        config=model_config, name="cliffordnet_patch_vae_v2"
    )
    logger.info(
        "Model: variant=%s | stage_dims=%s depths=%s shifts=%s | "
        "latent=%d patch=%d | mae=%.2f lpips=%.2f cls=%s seg=%s",
        config.model_variant or "custom",
        model_config.stage_dims,
        model_config.stage_depths,
        model_config.stage_shifts,
        model_config.latent_dim,
        model_config.patch_size,
        model_config.mae_mask_ratio,
        model_config.lambda_lpips,
        model_config.use_classification_head,
        model_config.use_segmentation_head,
    )

    dummy = keras.ops.zeros(
        (1, config.img_size, config.img_size, config.img_channels)
    )
    model(dummy, training=False)
    model.summary(print_fn=logger.info)

    lr_schedule = _build_lr_schedule(config, steps_per_epoch)
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=config.weight_decay,
    )
    # jit_compile=False matches v2 (XLA tends to fail on SIGReg reshape;
    # similar concern applies to Clifford's roll-based ops).
    model.compile(optimizer=optimizer, loss=None, jit_compile=False)

    callbacks, results_dir = create_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="cliffordnet_patch_vae_v2",
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
        callbacks.append(
            BetaAnnealingCallback(
                beta_start=config.beta_kl_start,
                beta_target=config.beta_kl,
                anneal_epochs=config.beta_anneal_epochs,
            )
        )

    if config.mae_mask_ratio > 0.0:
        try:
            sample_batch = next(iter(val_ds))
            head = sample_batch[0] if isinstance(sample_batch, tuple) else sample_batch
            if isinstance(head, dict):
                images = np.array(head["image"][:8])
            else:
                images = np.array(head[:8])
            recon_dir = os.path.join(results_dir, "masked_recon")
            callbacks.append(
                MaskedReconViz(val_samples=images, save_dir=recon_dir, frequency=1)
            )
        except Exception as exc:  # pragma: no cover — soft fail
            logger.warning("Could not set up MaskedReconViz: %s", exc)

    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, default=str)
    logger.info("Config saved to %s", config_path)

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(results_dir, "final_model.keras")
    model.save(final_path)
    logger.info("Final model saved: %s", final_path)

    # Reload check on deterministic mu (Sampling is stochastic).
    try:
        reloaded = keras.models.load_model(final_path, custom_objects=CUSTOM_OBJECTS)
        ref_mu = np.array(model.encode(dummy)[0])
        new_mu = np.array(reloaded.encode(dummy)[0])
        max_delta = float(np.max(np.abs(ref_mu - new_mu)))
        if max_delta < 1e-4:
            logger.info("Reload check PASSED: max|delta|=%.2e", max_delta)
        else:
            logger.error("Reload check FAILED: max|delta|=%.2e >= 1e-4", max_delta)
            sys.exit(1)
    except Exception as exc:
        logger.error("Reload check raised: %s", exc)
        sys.exit(1)

    h = history.history
    if h.get("val_loss"):
        logger.info(
            "Training summary | best val_loss=%.4f | epochs_run=%d/%d",
            min(h["val_loss"]), len(h["val_loss"]), config.epochs,
        )
    logger.info("Results written to: %s", results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_stage_shifts(s: str) -> List[List[int]]:
    """Parse --stage-shifts arg as a JSON list-of-lists of positive ints."""
    parsed = json.loads(s)
    if not isinstance(parsed, list) or not all(
        isinstance(inner, list) and all(isinstance(x, int) for x in inner)
        for inner in parsed
    ):
        raise argparse.ArgumentTypeError(
            "--stage-shifts must be a JSON list of lists of ints, e.g. "
            "'[[1,2],[1,2,4]]'"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = create_base_argument_parser(
        description=(
            "Train CliffordNetPatchVAEV2 (hierarchical Clifford-block VAE "
            "pretraining backbone)"
        ),
        default_dataset="cifar10",
        dataset_choices=["cifar10", "cifar100", "ade20k", "coco"],
    )
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument(
        "--variant", type=str, default="base", dest="model_variant",
        choices=list(PRESETS.keys()),
        help="Preset (tiny / base / large / xl).",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--stage-dims", type=int, nargs="+", default=None,
        help="Channel width per stage, e.g. --stage-dims 96 192. Overrides preset.",
    )
    parser.add_argument(
        "--stage-depths", type=int, nargs="+", default=None,
        help="Block count per stage. Overrides preset.",
    )
    parser.add_argument(
        "--stage-shifts", type=_parse_stage_shifts, default=None,
        help="Per-stage shifts as JSON list-of-lists, e.g. '[[1,2],[1,2,4]]'.",
    )
    parser.add_argument(
        "--downsample-kind", type=str, default="blur",
        choices=["avg", "max", "blur", "gaussian_dw", "pixel_unshuffle", "resnetd"],
    )
    parser.add_argument("--downsample-kernel-size", type=int, default=7)
    parser.add_argument(
        "--cli-mode", type=str, default="full",
        choices=["inner", "wedge", "full"],
    )
    parser.add_argument(
        "--ctx-mode", type=str, default="diff", choices=["diff", "abs"],
    )
    parser.add_argument(
        "--no-global-context", action="store_false",
        dest="use_global_context_in_last_stage", default=True,
    )
    parser.add_argument("--layer-scale-init", type=float, default=1e-5)
    parser.add_argument("--block-drop-path-rate", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Per-cell latent width. Overrides preset.")
    parser.add_argument("--beta-kl", type=float, default=0.5)
    parser.add_argument("--lambda-sigreg", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=256)
    parser.add_argument(
        "--recon-loss-type", type=str, default="bce", choices=["mse", "bce"],
    )
    parser.add_argument("--gamma-clip", type=float, default=1.0)
    parser.add_argument(
        "--no-augment", action="store_false", dest="augment_data", default=True,
    )
    parser.add_argument(
        "--no-color-augment", action="store_false",
        dest="augment_color", default=True,
    )
    parser.add_argument("--patches-per-image", type=int, default=4)

    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--beta-kl-start", type=float, default=0.0001)
    parser.add_argument("--beta-anneal-epochs", type=int, default=15)

    # Multi-task
    parser.add_argument("--mae-mask-ratio", type=float, default=0.0)
    parser.add_argument("--lambda-mae", type=float, default=1.0)
    parser.add_argument("--lambda-lpips", type=float, default=0.0)
    parser.add_argument(
        "--use-classification-head", action="store_true", default=False,
    )
    parser.add_argument("--num-classes-cls", type=int, default=0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument(
        "--use-segmentation-head", action="store_true", default=False,
    )
    parser.add_argument("--num-classes-seg", type=int, default=0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)

    parser.add_argument(
        "--ade20k-dir", type=str,
        default="/media/arxwn/data0_4tb/datasets/ade20k",
    )
    parser.add_argument(
        "--coco-dir", type=str,
        default="/media/arxwn/data0_4tb/datasets/coco_2017",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None,
        dest="steps_per_epoch_override",
    )

    parser.set_defaults(
        epochs=50, batch_size=256, learning_rate=3e-4, weight_decay=1e-4,
        patience=10, image_size=32,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    smoke = args.smoke
    if smoke:
        args.epochs = 2
        args.image_size = 16
        args.model_variant = "tiny"
        args.patch_size = 4
        args.batch_size = 4
        args.steps_per_epoch_override = 2
        args.sigreg_knots = 4
        args.sigreg_num_proj = 16
        logger.info(
            "Smoke mode: tiny variant, 2 epochs, 16x16, synthetic data."
        )

    if not smoke:
        setup_gpu(args.gpu)

    if args.use_classification_head and args.num_classes_cls == 0:
        if args.dataset == "cifar10":
            args.num_classes_cls = 10
        elif args.dataset == "cifar100":
            args.num_classes_cls = 100
        elif smoke:
            args.num_classes_cls = 5

    config = TrainingConfig(
        dataset=args.dataset,
        img_size=getattr(args, "image_size", 32),
        img_channels=3,
        augment_data=args.augment_data,
        augment_color=args.augment_color,
        patches_per_image=args.patches_per_image,
        model_variant=args.model_variant,
        patch_size=args.patch_size,
        stage_dims=args.stage_dims,
        stage_depths=args.stage_depths,
        stage_shifts=args.stage_shifts,
        downsample_kind=args.downsample_kind,
        downsample_kernel_size=args.downsample_kernel_size,
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context_in_last_stage=args.use_global_context_in_last_stage,
        layer_scale_init=args.layer_scale_init,
        block_drop_path_rate=args.block_drop_path_rate,
        latent_dim=args.latent_dim,
        beta_kl=args.beta_kl,
        lambda_sigreg=args.lambda_sigreg,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        recon_loss_type=args.recon_loss_type,
        gamma_clip=args.gamma_clip,
        mae_mask_ratio=args.mae_mask_ratio,
        lambda_mae=args.lambda_mae,
        lambda_lpips=args.lambda_lpips,
        use_classification_head=args.use_classification_head,
        num_classes_cls=args.num_classes_cls,
        lambda_cls=args.lambda_cls,
        use_segmentation_head=args.use_segmentation_head,
        num_classes_seg=args.num_classes_seg,
        lambda_seg=args.lambda_seg,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        beta_kl_start=args.beta_kl_start,
        beta_anneal_epochs=args.beta_anneal_epochs,
        early_stopping_patience=args.patience,
        ade20k_dir=args.ade20k_dir,
        coco_dir=args.coco_dir,
        steps_per_epoch_override=args.steps_per_epoch_override,
    )

    try:
        train(config, smoke=smoke)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
