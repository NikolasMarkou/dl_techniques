# DECISION plan_2026-06-11_f662207d/D-012
"""THERA arbitrary-scale super-resolution trainer (nested-tape ``train_step``).

This script trains a :class:`~dl_techniques.models.thera.model.Thera` model on
the arbitrary-scale SR pipeline (``train.thera.data``). The deliverable is the
:class:`TheraTrainingModel` wrapper plus a ``train_thera`` script (config
dataclass + argparse + ``main``).

The wrapper owns the parts THERA's reference ``run_train.py`` does outside the
model graph:

#. **Standardize** the raw ``[0, 1]`` ``source`` image with the THERA channel
   statistics ``MEAN``/``VAR``: ``(x - MEAN) / sqrt(VAR)``.
#. **Heat-kernel time** ``t = scale ** -2`` (inverse-square scale).
#. Run the inner ``Thera`` model -> raw residual field (the model stays a pure
   residual predictor, D-009).
#. **Denormalize** the field output ``out * sqrt(VAR) + MEAN`` and add the
   nearest-upsampled source residual ``+ source_nearest`` -> predicted HR pixels.
#. Reconstruction loss (MAE / Charbonnier) + ``tv_weight * TV`` aliasing penalty
   (exact spatial-Jacobian TV, D-010), gradient-clipped at ``max_grad_norm``.

# DECISION plan_2026-06-11_f662207d/D-012 (see decisions.md):
- Custom nested-tape ``train_step``: the OUTER ``tf.GradientTape`` differentiates
  the weights; the INNER persistent tape (inside ``decode_with_jac``) computes
  the exact per-pixel field Jacobian. Do NOT switch to a finite-difference TV
  (violates Q3) or drop TV to dodge the nested tape (real-finding rule).
- Manual ``tf.clip_by_global_norm(grads, max_grad_norm)`` in ``train_step``; do
  NOT ALSO set the optimizer's ``global_clipnorm`` (would double-clip).
- Standardize / denorm / ``+ source_nearest`` live HERE, not in the model
  (D-009: the model is a raw-residual-field predictor).
- Save the inner ``Thera`` (``thera_model.keras``) as the deployable artifact,
  NOT the ``TheraTrainingModel`` wrapper (the wrapper is a training harness).
- ``jit_compile=False``: the nested persistent tape + dynamic query shapes do
  not trace under XLA.

Typical usage::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m \\
        train.thera.train_thera \\
        --data-dir /path/to/DIV2K --backbone edsr-baseline --size pro \\
        --epochs 100 --batch-size 16
"""

# MPLBACKEND must be set before any matplotlib import — headless server guard.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import dataclasses
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.models.thera import build_thera
from dl_techniques.losses.thera_jacobian_tv import thera_tv_penalty
from dl_techniques.losses.image_restoration_loss import CharbonnierLoss
from dl_techniques.metrics.psnr_metric import PsnrMetric

from train.common import setup_gpu, set_seeds, create_callbacks, save_config_json
from train.common.callbacks import EpochMetricsPlotCallback
from train.thera.data import build_arbitrary_scale_dataset


# THERA per-channel image statistics (reference ``run_train.py``).
# ``standardize(x) = (x - MEAN) / sqrt(VAR)``.
THERA_MEAN = np.array([0.4488, 0.4371, 0.4040], dtype=np.float32)
THERA_VAR = np.array([0.25, 0.25, 0.25], dtype=np.float32)

# THERA reference field-init defaults. ``build_thera`` accepts ``None`` and
# falls back to the same values internally, but the config exposes them so they
# round-trip and a user can override.
DEFAULT_K_INIT = float(np.sqrt(np.log(4.0)) / (np.pi ** 2 * 2.0))
DEFAULT_COMPONENTS_INIT_SCALE = 16.0


# =====================================================================
# Training-model wrapper
# =====================================================================


@keras.saving.register_keras_serializable()
class TheraTrainingModel(keras.Model):
    """``keras.Model`` wrapper implementing THERA's nested-tape training step.

    Wraps an inner :class:`~dl_techniques.models.thera.model.Thera` (a raw
    residual-field predictor) and owns the standardize / denorm / residual-add
    and the reconstruction + Jacobian-TV objective. The inner ``Thera`` is the
    deployable artifact; this wrapper is a training harness.

    Args:
        thera_model: The inner :class:`Thera` model (raw residual predictor).
        tv_weight: Weight on the aliasing TV penalty. ``> 0`` activates the
            nested-tape Jacobian path; ``0`` disables it (no inner tape).
        loss_name: Reconstruction loss, ``"mae"`` (default) or ``"charbonnier"``.
        max_grad_norm: Global-norm gradient clip applied manually in
            ``train_step`` (THERA uses 10.0). Do NOT also set the optimizer's
            ``global_clipnorm``.
    """

    def __init__(
        self,
        thera_model: keras.Model,
        tv_weight: float = 1e-4,
        loss_name: str = "mae",
        max_grad_norm: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if loss_name not in ("mae", "charbonnier"):
            raise ValueError(
                f"loss_name must be 'mae' or 'charbonnier', got {loss_name!r}"
            )
        self.thera = thera_model
        self.tv_weight = float(tv_weight)
        self.loss_name = loss_name
        self.max_grad_norm = float(max_grad_norm)

        # THERA channel statistics as float32 constants (broadcast over B,H,W,C).
        self._mean = ops.convert_to_tensor(THERA_MEAN)
        self._std = ops.sqrt(ops.convert_to_tensor(THERA_VAR))

        self._charbonnier = (
            CharbonnierLoss() if loss_name == "charbonnier" else None
        )

        # Keras 3.8 does NOT auto-create a loss tracker for a custom train_step
        # (LESSONS): create it (and the rest) explicitly so `metrics` resets them
        # per epoch and prefixes the `val_` twin.
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_tracker = keras.metrics.Mean(name="mae")
        self.tv_tracker = keras.metrics.Mean(name="tv")
        self.psnr_metric = PsnrMetric(max_val=1.0, name="psnr")

    @property
    def metrics(self):
        # Returning the trackers here lets Keras reset them at epoch start and
        # auto-prefix `val_` during evaluate.
        return [
            self.loss_tracker,
            self.mae_tracker,
            self.tv_tracker,
            self.psnr_metric,
        ]

    # -----------------------------------------------------------------

    def _forward(
        self,
        batch: Dict[str, Any],
        training: bool,
        return_jac: bool,
    ) -> Tuple[Any, Any]:
        """Standardize -> inner Thera -> denorm + nearest-residual add.

        Args:
            batch: Dict with ``source`` ``(B, Hs, Ws, 3)``, ``target_coords``
                ``(B, Hq, Wq, 2)``, ``source_nearest`` ``(B, Hq, Wq, 3)`` and
                ``scale`` ``(B,)`` -- all raw ``[0, 1]`` (coords in ``[-0.5,
                0.5]``).
            training: Forwarded to the inner model.
            return_jac: When ``True``, also return the exact spatial Jacobian.

        Returns:
            ``(out, jac)`` -- predicted HR pixels ``(B, Hq, Wq, 3)`` and the
            Jacobian (``None`` when ``return_jac`` is ``False``).
        """
        source = (batch["source"] - self._mean) / self._std
        # Heat-kernel time t = scale ** -2 (inverse-square scale), shape (B, 1).
        t = ops.reshape(batch["scale"], (-1, 1)) ** -2.0

        result = self.thera(
            (source, batch["target_coords"], t),
            training=training,
            return_jac=return_jac,
        )
        if return_jac:
            out, jac = result
        else:
            out, jac = result, None

        # Denormalize the residual field, then add the nearest-upsampled source.
        out = out * self._std + self._mean + batch["source_nearest"]
        return out, jac

    def _recon_loss(self, out: Any, target: Any) -> Any:
        """Reconstruction loss scalar (MAE or Charbonnier)."""
        if self._charbonnier is not None:
            return self._charbonnier(target, out)
        return ops.mean(ops.abs(out - target))

    def _update_trackers(self, loss: Any, recon: Any, tv: Any, out: Any,
                         target: Any) -> Dict[str, Any]:
        """Update all metric trackers and return the result dict."""
        self.loss_tracker.update_state(loss)
        self.mae_tracker.update_state(recon)
        self.tv_tracker.update_state(tv)
        self.psnr_metric.update_state(target, ops.clip(out, 0.0, 1.0))
        return {m.name: m.result() for m in self.metrics}

    # -----------------------------------------------------------------

    def train_step(self, data: Any) -> Dict[str, Any]:
        batch = data
        return_jac = self.tv_weight > 0.0

        # OUTER tape -> weights. The inner persistent tape (decode_with_jac)
        # composes with this for the second-order TV term (D-010 / D-012).
        with tf.GradientTape() as tape:
            out, jac = self._forward(batch, training=True, return_jac=return_jac)
            recon = self._recon_loss(out, batch["target"])
            tv = thera_tv_penalty(jac) if jac is not None else 0.0
            loss = recon + self.tv_weight * tv

        grads = tape.gradient(loss, self.thera.trainable_variables)
        # Manual global-norm clip (THERA optax.clip_by_global_norm). Do NOT also
        # set the optimizer's global_clipnorm (D-012).
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.thera.trainable_variables))

        return self._update_trackers(loss, recon, tv, out, batch["target"])

    def test_step(self, data: Any) -> Dict[str, Any]:
        batch = data
        return_jac = self.tv_weight > 0.0
        # Mirror the train objective so val_loss = recon + tv_weight*TV and
        # monitor='val_loss' tracks the real objective.
        out, jac = self._forward(batch, training=False, return_jac=return_jac)
        recon = self._recon_loss(out, batch["target"])
        tv = thera_tv_penalty(jac) if jac is not None else 0.0
        loss = recon + self.tv_weight * tv
        return self._update_trackers(loss, recon, tv, out, batch["target"])

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        # Thin passthrough so the wrapper is a valid Model (used for build, not
        # by fit). Raw residual field of the inner Thera.
        return self.thera(inputs, training=training)

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "thera_model": keras.saving.serialize_keras_object(self.thera),
            "tv_weight": self.tv_weight,
            "loss_name": self.loss_name,
            "max_grad_norm": self.max_grad_norm,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TheraTrainingModel":
        config = dict(config)
        config["thera_model"] = keras.saving.deserialize_keras_object(
            config["thera_model"]
        )
        return cls(**config)


# =====================================================================
# Config
# =====================================================================


@dataclasses.dataclass
class TheraConfig:
    """All hyperparameters for one THERA training run.

    DIV2K subdir defaults mirror THERA (``DIV2K_train`` /
    ``DIV2K_dev_val``); any subdir name is accepted.
    """

    # Data
    data_dir: str = "/media/arxwn/data0_4tb/datasets/DIV2K"
    train_subdir: str = "DIV2K_train"
    val_subdir: str = "DIV2K_dev_val"
    source_size: int = 48
    target_samples: int = 48
    scale_range: Tuple[float, float] = (1.2, 4.0)
    augment_scale_range: Tuple[float, float] = (1.0, 2.0)
    augment_scale_prob: float = 0.5

    # Architecture
    backbone: str = "edsr-baseline"
    size: str = "pro"
    out_dim: int = 3
    k_init: Optional[float] = DEFAULT_K_INIT
    components_init_scale: Optional[float] = DEFAULT_COMPONENTS_INIT_SCALE

    # Objective
    tv_weight: float = 1e-4
    loss_name: str = "mae"
    max_grad_norm: float = 10.0

    # Training
    batch_size: int = 16
    epochs: int = 100
    steps_per_epoch: int = 1000
    val_steps: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 20

    # Runtime
    seed: int = 42
    gpu: int = 0
    result_dir: str = "results"
    visualize_every_n_epochs: int = 5

    def __post_init__(self) -> None:
        if self.backbone not in ("edsr-baseline", "rdn"):
            raise ValueError(
                f"backbone must be 'edsr-baseline' or 'rdn', got {self.backbone!r}"
            )
        if self.size not in ("air", "plus", "pro"):
            raise ValueError(
                f"size must be one of 'air', 'plus', 'pro', got {self.size!r}"
            )


# =====================================================================
# Training entrypoint
# =====================================================================


def _build_lr_schedule(config: TheraConfig) -> Any:
    """Cosine-decay schedule with Keras-3.8 native linear warmup."""
    total_steps = config.epochs * config.steps_per_epoch
    warmup_steps = config.warmup_epochs * config.steps_per_epoch
    decay_steps = max(1, total_steps - warmup_steps)
    if warmup_steps > 0:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=decay_steps,
            alpha=1e-6,
            warmup_target=config.lr,
            warmup_steps=warmup_steps,
        )
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.lr,
        decay_steps=decay_steps,
        alpha=1e-6,
    )


def train(config: TheraConfig) -> None:
    """Run one THERA training experiment.

    Args:
        config: Fully-validated :class:`TheraConfig`.
    """
    logger.info(
        f"Starting THERA training: backbone={config.backbone}, size={config.size}"
    )

    setup_gpu(config.gpu)
    set_seeds(config.seed)

    data_root = Path(config.data_dir)
    train_dir = data_root / config.train_subdir
    val_dir = data_root / config.val_subdir

    train_ds = build_arbitrary_scale_dataset(
        str(train_dir),
        source_size=config.source_size,
        target_samples=config.target_samples,
        scale_range=config.scale_range,
        augment_scale_range=config.augment_scale_range,
        augment_scale_prob=config.augment_scale_prob,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        repeat=True,
    )
    val_ds = build_arbitrary_scale_dataset(
        str(val_dir),
        source_size=config.source_size,
        target_samples=config.target_samples,
        scale_range=config.scale_range,
        augment_scale_range=config.augment_scale_range,
        augment_scale_prob=config.augment_scale_prob,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        repeat=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    thera = build_thera(
        config.out_dim,
        config.backbone,
        config.size,
        config.k_init,
        config.components_init_scale,
    )
    model = TheraTrainingModel(
        thera,
        tv_weight=config.tv_weight,
        loss_name=config.loss_name,
        max_grad_norm=config.max_grad_norm,
    )

    schedule = _build_lr_schedule(config)
    optimizer = keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=config.weight_decay,
        global_clipnorm=None,  # we clip manually in train_step (D-012)
    )
    # jit_compile=False: nested persistent tape + dynamic query shapes (D-012).
    model.compile(optimizer=optimizer, jit_compile=False)

    # Dummy forward to build the inner model (so trainable_variables exist).
    dummy_source = keras.ops.zeros((1, config.source_size, config.source_size, 3))
    dummy_coords = keras.ops.zeros(
        (1, config.target_samples, config.target_samples, 2)
    )
    dummy_t = keras.ops.ones((1, 1))
    model((dummy_source, dummy_coords, dummy_t), training=False)
    thera.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    model_name = f"thera_{config.backbone}_{config.size}"
    callbacks, results_dir = create_callbacks(
        model_name=model_name,
        results_dir_prefix="thera",
        output_root=config.result_dir,
        monitor="val_loss",
        patience=config.patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    viz_dir = os.path.join(results_dir, "metrics_curves")
    callbacks.append(
        EpochMetricsPlotCallback(
            viz_dir=viz_dir,
            metric_names=["psnr", "mae"],
            every_n=config.visualize_every_n_epochs,
        )
    )

    config_path = save_config_json(config, results_dir, "config.json")
    logger.info(f"Config saved to {config_path}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=config.val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Save the inner Thera (deployable artifact, NOT the wrapper) + reload check
    # ------------------------------------------------------------------
    thera_path = os.path.join(results_dir, "thera_model.keras")
    thera.save(thera_path)
    logger.info(f"Deployable Thera saved: {thera_path}")

    try:
        reloaded = keras.models.load_model(thera_path)
        ref = np.array(thera((dummy_source, dummy_coords, dummy_t), training=False))
        new = np.array(reloaded((dummy_source, dummy_coords, dummy_t), training=False))
        max_delta = float(np.max(np.abs(ref - new)))
        if max_delta < 1e-4:
            logger.info(f"Reload check PASSED: max|delta|={max_delta:.2e}")
        else:
            logger.error(f"Reload check FAILED: max|delta|={max_delta:.2e} >= 1e-4")
            sys.exit(1)
    except Exception as exc:
        if "out of memory" in str(exc).lower() or isinstance(
            exc, tf.errors.ResourceExhaustedError
        ):
            logger.warning(
                f"Reload check skipped — out of memory during load_model "
                f"(saved model at {thera_path} is intact): {exc}"
            )
        else:
            logger.error(f"Reload check raised an exception: {exc}", exc_info=True)
            sys.exit(1)

    val_losses = history.history.get("val_loss", [])
    if val_losses:
        logger.info(
            f"Training summary | best val_loss={min(val_losses):.4f} | "
            f"epochs_run={len(val_losses)}/{config.epochs}"
        )
    logger.info(f"Results written to: {results_dir}")


# =====================================================================
# CLI
# =====================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train THERA arbitrary-scale super-resolution model."
    )
    # Data
    parser.add_argument("--data-dir", type=str, default=TheraConfig.data_dir)
    parser.add_argument("--train-subdir", type=str, default=TheraConfig.train_subdir)
    parser.add_argument("--val-subdir", type=str, default=TheraConfig.val_subdir)
    parser.add_argument("--source-size", type=int, default=TheraConfig.source_size)
    parser.add_argument("--target-samples", type=int, default=TheraConfig.target_samples)
    parser.add_argument("--scale-min", type=float, default=TheraConfig.scale_range[0])
    parser.add_argument("--scale-max", type=float, default=TheraConfig.scale_range[1])
    parser.add_argument("--augment-scale-min", type=float,
                        default=TheraConfig.augment_scale_range[0])
    parser.add_argument("--augment-scale-max", type=float,
                        default=TheraConfig.augment_scale_range[1])
    parser.add_argument("--augment-scale-prob", type=float,
                        default=TheraConfig.augment_scale_prob)
    # Architecture
    parser.add_argument("--backbone", type=str, default=TheraConfig.backbone,
                        choices=["edsr-baseline", "rdn"])
    parser.add_argument("--size", type=str, default=TheraConfig.size,
                        choices=["air", "plus", "pro"])
    parser.add_argument("--k-init", type=float, default=DEFAULT_K_INIT)
    parser.add_argument("--components-init-scale", type=float,
                        default=DEFAULT_COMPONENTS_INIT_SCALE)
    # Objective
    parser.add_argument("--tv-weight", type=float, default=TheraConfig.tv_weight)
    parser.add_argument("--loss-name", type=str, default=TheraConfig.loss_name,
                        choices=["mae", "charbonnier"])
    parser.add_argument("--max-grad-norm", type=float, default=TheraConfig.max_grad_norm)
    # Training
    parser.add_argument("--batch-size", type=int, default=TheraConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TheraConfig.epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=TheraConfig.steps_per_epoch)
    parser.add_argument("--val-steps", type=int, default=TheraConfig.val_steps)
    parser.add_argument("--lr", type=float, default=TheraConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TheraConfig.weight_decay)
    parser.add_argument("--warmup-epochs", type=int, default=TheraConfig.warmup_epochs)
    parser.add_argument("--patience", type=int, default=TheraConfig.patience)
    # Runtime
    parser.add_argument("--seed", type=int, default=TheraConfig.seed)
    parser.add_argument("--gpu", type=int, default=TheraConfig.gpu)
    parser.add_argument("--result-dir", type=str, default=TheraConfig.result_dir)
    parser.add_argument("--visualize-every-n-epochs", type=int,
                        default=TheraConfig.visualize_every_n_epochs)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = TheraConfig(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        source_size=args.source_size,
        target_samples=args.target_samples,
        scale_range=(args.scale_min, args.scale_max),
        augment_scale_range=(args.augment_scale_min, args.augment_scale_max),
        augment_scale_prob=args.augment_scale_prob,
        backbone=args.backbone,
        size=args.size,
        k_init=args.k_init,
        components_init_scale=args.components_init_scale,
        tv_weight=args.tv_weight,
        loss_name=args.loss_name,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=args.seed,
        gpu=args.gpu,
        result_dir=args.result_dir,
        visualize_every_n_epochs=args.visualize_every_n_epochs,
    )

    try:
        train(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
