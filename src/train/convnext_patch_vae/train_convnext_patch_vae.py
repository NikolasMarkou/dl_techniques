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

Training menu from README:
    T1a  CIFAR-10, MSE, SIGReg ON  (default, ~30 min RTX 4090)
    T1b  CIFAR-10, MSE, SIGReg OFF (--lambda-sigreg 0)
    T2   imagenette 128 px, patch=8, latent=32 (~2-3 h)
"""

# MPLBACKEND must be set before any matplotlib import — headless server guard.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import dataclasses
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import keras

from dl_techniques.utils.logger import logger
from dl_techniques.models.convnext_patch_vae.config import ConvNeXtPatchVAEConfig
from dl_techniques.models.convnext_patch_vae.model import ConvNeXtPatchVAE
from dl_techniques.callbacks.training_curves import TrainingCurvesCallback
from train.common import setup_gpu, create_base_argument_parser, create_callbacks

CUSTOM_OBJECTS = {"ConvNeXtPatchVAE": ConvNeXtPatchVAE}

# CIFAR-10 per-channel statistics (for MSE pipeline)
_CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainingConfig:
    """All hyperparameters for one training run.

    Attributes:
        dataset: Dataset identifier. Supported: ``"cifar10"``.
        img_size: Spatial resolution fed to the model (must be divisible
            by ``patch_size``).
        img_channels: Number of input/output channels (3 for RGB).
        augment_data: Whether to apply random horizontal flip + crop.
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
        early_stopping_patience: EarlyStopping patience on ``val_loss``.
        success_threshold: ``val_loss <= threshold`` → convergence flag.
            Advisory only; does not block saving the model.
        output_dir: Root output directory (should be repo-root
            ``results/``).
        experiment_name: Run identifier. Auto-generated if ``None``.
    """

    # Dataset
    dataset: str = "cifar10"
    img_size: int = 32
    img_channels: int = 3
    augment_data: bool = True

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
    recon_loss_type: str = "mse"
    dropout_rate: float = 0.0
    gamma_clip: float = 1.0

    # Training
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    early_stopping_patience: int = 10

    # Output / evaluation
    success_threshold: float = 0.02
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"convnext_patch_vae_{self.dataset}_{ts}"
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _build_cifar10_dataset(
    config: TrainingConfig,
) -> Tuple[Any, Any, int, int]:
    """Load CIFAR-10 and build ``tf.data`` pipelines.

    Args:
        config: Training configuration.

    Returns:
        Tuple of ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
        Datasets emit ``(x, x)`` pairs — the label is the image itself;
        :meth:`ConvNeXtPatchVAE.train_step` ignores the second element.
    """
    import tensorflow as tf

    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")

    # Select normalisation branch based on recon_loss_type.
    # BCE requires inputs in [0, 1] → /255 only.
    # MSE works with mean/std normalisation (ImageNet-style).
    if config.recon_loss_type == "bce":
        x_train /= 255.0
        x_test  /= 255.0
    else:
        x_train /= 255.0
        x_test  /= 255.0
        x_train = (x_train - _CIFAR10_MEAN) / _CIFAR10_STD
        x_test  = (x_test  - _CIFAR10_MEAN) / _CIFAR10_STD

    steps_per_epoch = max(1, len(x_train) // config.batch_size)
    val_steps       = max(1, len(x_test)  // config.batch_size)

    def _augment(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.random_flip_left_right(x)
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.image.random_crop(x, [config.img_size, config.img_size, config.img_channels])
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
        f"CIFAR-10: {len(x_train)} train / {len(x_test)} val | "
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
    if config.dataset == "cifar10":
        return _build_cifar10_dataset(config)
    raise ValueError(
        f"Unsupported dataset '{config.dataset}'. "
        "Supported: 'cifar10'. For imagenette, set --dataset cifar10 and "
        "--img-size 128 with an external image_folder loader (T2 task)."
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
        self.cifar_mean = cifar_mean if cifar_mean is not None else _CIFAR10_MEAN
        self.cifar_std  = cifar_std  if cifar_std  is not None else _CIFAR10_STD
        os.makedirs(save_dir, exist_ok=True)

    def _to_display(self, x: np.ndarray) -> np.ndarray:
        """Undo normalisation and clip to ``[0, 1]``."""
        if self.recon_loss_type == "mse":
            x = x * self.cifar_std + self.cifar_mean
        return np.clip(x, 0.0, 1.0)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if epoch % self.frequency != 0 and epoch != 0:
            return
        try:
            outputs = self.model(self.val_samples, training=False)
            originals = self._to_display(self.val_samples)
            recons    = self._to_display(
                np.array(outputs["reconstruction"])
            )

            n = len(originals)
            cmap = "gray" if originals.shape[-1] == 1 else None
            fig, axes = plt.subplots(2, n, figsize=(n * 1.4, 3.2))

            for i in range(n):
                axes[0, i].imshow(originals[i].squeeze(), cmap=cmap)
                axes[0, i].axis("off")
                if i == 0:
                    axes[0, i].set_ylabel("original", fontsize=8)
                axes[1, i].imshow(recons[i].squeeze(), cmap=cmap)
                axes[1, i].axis("off")
                if i == 0:
                    axes[1, i].set_ylabel("recon", fontsize=8)

            loss_val = (logs or {}).get("loss", float("nan"))
            fig.suptitle(
                f"Epoch {epoch + 1}  |  loss={loss_val:.4f}", fontsize=11
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            path = os.path.join(self.save_dir, f"recon_epoch_{epoch + 1:04d}.png")
            plt.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"ReconVisualizationCallback failed at epoch {epoch}: {exc}")


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
            initial_learning_rate=config.learning_rate,
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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = config.to_model_config()
    model = ConvNeXtPatchVAE(config=model_config, name="convnext_patch_vae")

    dummy = keras.ops.zeros(
        (1, config.img_size, config.img_size, config.img_channels)
    )
    model(dummy, training=False)
    model.summary(print_fn=logger.info)

    logger.info(
        f"Model: variant={config.model_variant or 'custom'} | "
        f"embed={model_config.embed_dim} | depth={model_config.encoder_depth} | "
        f"latent={model_config.latent_dim} | patch={model_config.patch_size}"
    )

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
    callbacks, results_dir = create_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="convnext_patch_vae",
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    curves_dir = os.path.join(results_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)
    callbacks.append(TrainingCurvesCallback(output_dir=curves_dir))

    # Grab a fixed validation sample for reconstruction visualisation
    recon_dir = os.path.join(results_dir, "reconstructions")
    try:
        sample_batch = next(iter(val_ds))
        val_samples = np.array(sample_batch[0][:8])
        callbacks.append(
            ReconVisualizationCallback(
                val_samples=val_samples,
                save_dir=recon_dir,
                frequency=max(1, config.epochs // 10),
                recon_loss_type=config.recon_loss_type,
            )
        )
    except Exception as exc:
        logger.warning(f"Could not set up ReconVisualizationCallback: {exc}")

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
        ref_out  = np.array(model(dummy, training=False)["reconstruction"])
        new_out  = np.array(reloaded(dummy, training=False)["reconstruction"])
        max_delta = float(np.max(np.abs(ref_out - new_out)))
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
        converged = (
            best_val_loss <= config.success_threshold
            and epochs_run >= 0.5 * config.epochs
        )
        if converged:
            logger.info(
                f"TRAINING CONVERGED: best val_loss={best_val_loss:.4f} "
                f"<= threshold={config.success_threshold}"
            )
        else:
            logger.warning(
                f"TRAINING MAY NOT HAVE CONVERGED: "
                f"best val_loss={best_val_loss:.4f}, "
                f"threshold={config.success_threshold}, "
                f"epochs_run={epochs_run}/{config.epochs}. "
                "Consider increasing --epochs or tuning --beta-kl / --lambda-sigreg."
            )

    logger.info(f"Results written to: {results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = create_base_argument_parser(
        description="Train ConvNeXtPatchVAE (per-patch VAE with SIGReg anti-collapse)",
        default_dataset="cifar10",
        dataset_choices=["cifar10"],
    )

    # Smoke
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Run a tiny smoke test on synthetic data (CPU, <60 s). "
             "Overrides --epochs to 3 and uses a tiny model.",
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
        "--recon-loss-type", type=str, default="mse", choices=["mse", "bce"],
        help="Reconstruction loss family. BCE requires inputs in [0,1].",
    )
    parser.add_argument("--dropout",    type=float, default=0.0, dest="dropout_rate")
    parser.add_argument("--gamma-clip", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_false", dest="augment_data",
                        default=True)

    # Training (extend base parser defaults)
    parser.add_argument("--warmup-epochs",    type=int,   default=5)
    parser.add_argument("--success-threshold", type=float, default=0.02,
                        help="val_loss threshold for the convergence advisory.")

    # Set script-appropriate base-parser defaults
    parser.set_defaults(
        epochs=50,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=10,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    smoke = args.smoke
    if smoke:
        args.epochs       = 3
        args.model_variant = "tiny"
        args.embed_dim    = 16
        args.encoder_depth = 1
        args.decoder_depth = 1
        args.latent_dim   = 4
        args.sigreg_knots = 5
        args.sigreg_num_proj = 32
        args.patch_size   = 4
        logger.info("Smoke mode: tiny model, 3 epochs, synthetic data.")

    # GPU must be configured before any TF/Keras context is created.
    if not smoke:
        setup_gpu(args.gpu)

    config = TrainingConfig(
        dataset=args.dataset,
        img_size=getattr(args, "image_size", 32),
        img_channels=3,
        augment_data=args.augment_data,
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
        early_stopping_patience=args.patience,
        success_threshold=args.success_threshold,
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
