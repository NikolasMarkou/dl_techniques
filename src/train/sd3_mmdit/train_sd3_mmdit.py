"""Synthetic flow-matching smoke-trainer for the SD3 MMDiT transformer (tiny).

This script trains the :class:`~dl_techniques.models.sd3_mmdit.transformer.SD3MMDiT`
velocity head on a SYNTHETIC rectified-flow task so the architecture's training
path can be exercised end to end on a single GPU. It is a smoke oracle, NOT a
convergence run: the latents / conditioning are random Gaussians with no
learnable structure, so the loss should stay finite (and typically dips early)
without claiming generation quality.

Rectified-flow setup (SD3 / plan assumption A4)::

    x0       ~ N(0, I)                 clean image latents (B, S, S, in_ch)
    noise    ~ N(0, I)                 noise           (B, S, S, in_ch)
    t        ~ logit-normal(0,1)+shift per-sample flow time in (0, 1)
    x_t      = (1 - t) * x0 + t * noise  noised latent fed to the transformer
    v_target = noise - x0              the velocity the model must predict
    timestep = t * 1000.0             scalar diffusion time fed to the model

Logit-normal time weighting (HARD constraint: lives in the TRAINER, not the
loss). Each sample carries a precomputed SD3 Eq.(19) weight
``w(t) = 1 / pdf_logitnormal(t)`` from
:meth:`FlowMatchEulerScheduler.logit_normal_weight`. Because that raw weight
spikes near ``t -> 0/1``, the trainer normalizes the per-batch weight to mean 1
inside ``train_step`` (a smoke-stability measure, NOT a correctness change: it
only rescales the gradient magnitude per step, preserving the RELATIVE
per-sample weighting). The custom ``train_step`` noises + weights + computes the
MSE here (cleaner than slicing the loss), so ``compile`` needs only an
optimizer.

Run::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \
        .venv/bin/python -m train.sd3_mmdit.train_sd3_mmdit \
        --variant tiny --epochs 3 --steps-per-epoch 8 --batch-size 2 --gpu 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.models.sd3_mmdit.config import get_sd3_config
from dl_techniques.models.sd3_mmdit.scheduler import FlowMatchEulerScheduler
from dl_techniques.models.sd3_mmdit.transformer import (
    SD3MMDiT,
    create_sd3_mmdit,
)

from train.common import setup_gpu, set_seeds, create_callbacks, save_config_json


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for the synthetic SD3 MMDiT smoke-train.

    Args:
        variant: Config preset name (``"tiny"`` smoke-trainable / ``"full"``).
        batch_size: Samples per gradient step.
        steps_per_epoch: Synthetic batches drawn per epoch.
        epochs: Number of epochs.
        learning_rate: Adam learning rate.
        num_text_tokens: Synthetic caption length ``L`` for ``encoder_hidden_states``.
        output_dir: Results root (repo-root ``results/...`` by convention).
        results_prefix: Directory-name prefix inside ``output_dir``.
        mixed_bfloat16: If True, set the global ``mixed_bfloat16`` policy before
            building the model.
        gpu: GPU index forwarded to ``setup_gpu`` (also sets CUDA_VISIBLE_DEVICES).
        seed: RNG seed for reproducible synthetic data + init.
    """

    variant: str = "tiny"
    batch_size: int = 2
    steps_per_epoch: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    num_text_tokens: int = 7
    output_dir: str = "results/sd3_mmdit"
    results_prefix: str = "sd3_mmdit"
    mixed_bfloat16: bool = False
    gpu: Optional[int] = None
    seed: int = 42

    # --- derived (filled at build time; serialized for the run config) ---
    in_channels: int = field(default=0)
    joint_attention_dim: int = field(default=0)
    pooled_projection_dim: int = field(default=0)
    sample_size: int = field(default=0)


# ---------------------------------------------------------------------
# Trainer model: SD3MMDiT with a logit-normal-weighted custom train_step
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.train.sd3_mmdit")
class SD3FlowTrainer(keras.Model):
    """Thin trainer wrapping :class:`SD3MMDiT` with a custom flow-matching step.

    ``call`` delegates to the wrapped transformer (so ``predict`` and
    serialization work). ``train_step`` performs the rectified-flow noising,
    predicts the velocity, applies the logit-normal per-sample weighting (the
    HARD constraint keeps weighting out of :class:`FlowMatchingVelocityLoss`),
    and minimizes the weighted MSE. A per-batch mean-1 normalization of the
    weight keeps the smoke stable near ``t -> 0/1`` (relative weighting
    preserved; see module docstring).

    Keras 3.8 does NOT auto-create a loss tracker for a custom ``train_step``, so
    an explicit :class:`keras.metrics.Mean` is declared in ``__init__`` and
    exposed via the ``metrics`` property (SYSTEM.md "Keras 3.8 no auto-tracker").

    Args:
        transformer: The wrapped :class:`SD3MMDiT`.
        shift: SD3 static time-shift for the scheduler.
        logit_mean: Logit-normal time-sampling mean.
        logit_std: Logit-normal time-sampling std.
        **kwargs: Forwarded to ``keras.Model``.
    """

    def __init__(
        self,
        transformer: SD3MMDiT,
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.transformer = transformer
        self.shift = float(shift)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        # Stateless plain-Python scheduler (add_noise / velocity_target via keras.ops).
        self.scheduler = FlowMatchEulerScheduler(
            shift=self.shift,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )
        # Keras 3.8 has no auto-tracker for custom train_step -> declare explicitly.
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return self.transformer(inputs, training=training)

    @property
    def metrics(self) -> list:
        # Explicit tracker list so fit() resets it between epochs.
        return [self.loss_tracker]

    def train_step(self, data: Dict[str, keras.KerasTensor]) -> Dict[str, Any]:
        x0 = data["x0"]
        encoder_hidden_states = data["encoder_hidden_states"]
        pooled_projections = data["pooled_projections"]
        t = data["t"]              # (B,)
        noise = data["noise"]
        weight = data["weight"]    # (B,) raw logit-normal Eq.(19) weight

        # Per-sample time broadcast to the latent rank for add_noise.
        t_b = keras.ops.reshape(t, (-1, 1, 1, 1))
        x_t = self.scheduler.add_noise(x0, noise, t_b)
        v_target = self.scheduler.velocity_target(x0, noise)  # noise - x0
        timestep = t * 1000.0  # transformer expects t in [0, 1000]

        # Smoke-stability: normalize the per-batch weight to mean 1 (preserves the
        # RELATIVE logit-normal weighting; only rescales the per-step gradient
        # magnitude so a near-0/1 t cannot blow up the loss). NOT a correctness
        # change. See module docstring.
        weight = weight / (keras.ops.mean(weight) + 1e-8)

        with tf.GradientTape() as tape:
            v_pred = self.transformer(
                {
                    "latent": x_t,
                    "encoder_hidden_states": encoder_hidden_states,
                    "pooled_projections": pooled_projections,
                    "timestep": timestep,
                },
                training=True,
            )
            # Per-sample MSE over the non-batch (H, W, C) axes.
            per_sample = keras.ops.mean(
                keras.ops.square(v_pred - v_target), axis=[1, 2, 3]
            )
            loss = keras.ops.mean(weight * per_sample)

        grads = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.transformer.trainable_variables)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def compute_output_shape(
        self, input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        return self.transformer.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "transformer": keras.saving.serialize_keras_object(self.transformer),
                "shift": self.shift,
                "logit_mean": self.logit_mean,
                "logit_std": self.logit_std,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SD3FlowTrainer":
        config = dict(config)
        config["transformer"] = keras.saving.deserialize_keras_object(
            config["transformer"]
        )
        return cls(**config)


def build_trainer(config: TrainingConfig) -> Tuple[SD3FlowTrainer, Any]:
    """Build the tiny transformer + flow trainer for ``config.variant``.

    Returns:
        ``(trainer, sd3_config)`` where ``sd3_config`` is the ``SD3MMDiTConfig``.
    """
    sd3_config, _ae = get_sd3_config(config.variant)
    transformer = create_sd3_mmdit(config.variant)
    scheduler = FlowMatchEulerScheduler()
    trainer = SD3FlowTrainer(
        transformer=transformer,
        shift=scheduler.shift,
        logit_mean=scheduler.logit_mean,
        logit_std=scheduler.logit_std,
        name="sd3_flow_trainer",
    )
    return trainer, sd3_config


# ---------------------------------------------------------------------
# Synthetic rectified-flow dataset
# ---------------------------------------------------------------------


def make_synthetic_dataset(
    config: TrainingConfig,
    sd3_config: Any,
) -> tf.data.Dataset:
    """Build an infinite ``tf.data`` dataset of synthetic rectified-flow batches.

    Each element is a dict batch with the six keys consumed by
    :meth:`SD3FlowTrainer.train_step`: ``x0``, ``noise`` (both
    ``(B, S, S, in_ch)``), ``encoder_hidden_states`` ``(B, L, joint_dim)``,
    ``pooled_projections`` ``(B, pooled_dim)``, ``t`` ``(B,)``, ``weight``
    ``(B,)``.

    A Python generator (``tf.data.Dataset.from_generator``) yields fully-formed
    numpy batches so the SD3 logit-normal time sampling (which needs scipy's
    inverse-normal-CDF ``ndtri``, with no backend-agnostic in-graph form) stays
    host-side. The scheduler's ``sample_logit_normal_t`` / ``logit_normal_weight``
    produce ``t`` and ``weight``; everything else is ``N(0, 1)``.

    Args:
        config: Training configuration (batch size, text tokens, seed).
        sd3_config: The ``SD3MMDiTConfig`` (latent / conditioning dims).

    Returns:
        A repeating, prefetched ``tf.data.Dataset`` of dict batches.
    """
    B = config.batch_size
    S = sd3_config.sample_size
    in_ch = sd3_config.in_channels
    L = config.num_text_tokens
    joint_dim = sd3_config.joint_attention_dim
    pooled_dim = sd3_config.pooled_projection_dim

    scheduler = FlowMatchEulerScheduler()
    rng = np.random.default_rng(config.seed)

    def _gen():
        while True:
            x0 = rng.standard_normal((B, S, S, in_ch)).astype(np.float32)
            noise = rng.standard_normal((B, S, S, in_ch)).astype(np.float32)
            enc = rng.standard_normal((B, L, joint_dim)).astype(np.float32)
            pooled = rng.standard_normal((B, pooled_dim)).astype(np.float32)
            # Host-side logit-normal time + Eq.(19) weight (scipy ndtri/expit).
            t = scheduler.sample_logit_normal_t(B, seed=int(rng.integers(0, 2**31)))
            t = t.astype(np.float32)
            weight = scheduler.logit_normal_weight(t).astype(np.float32)
            yield {
                "x0": x0,
                "noise": noise,
                "encoder_hidden_states": enc,
                "pooled_projections": pooled,
                "t": t,
                "weight": weight,
            }

    output_signature = {
        "x0": tf.TensorSpec(shape=(B, S, S, in_ch), dtype=tf.float32),
        "noise": tf.TensorSpec(shape=(B, S, S, in_ch), dtype=tf.float32),
        "encoder_hidden_states": tf.TensorSpec(
            shape=(B, L, joint_dim), dtype=tf.float32
        ),
        "pooled_projections": tf.TensorSpec(shape=(B, pooled_dim), dtype=tf.float32),
        "t": tf.TensorSpec(shape=(B,), dtype=tf.float32),
        "weight": tf.TensorSpec(shape=(B,), dtype=tf.float32),
    }

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------


def train(config: TrainingConfig) -> Dict[str, Any]:
    """Run the synthetic flow-matching smoke-train.

    Args:
        config: The training configuration.

    Returns:
        A dict with the Keras ``history.history`` and the run results dir.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    if config.mixed_bfloat16:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logger.info("[train_sd3_mmdit] global policy set to mixed_bfloat16")

    trainer, sd3_config = build_trainer(config)

    # Populate the derived fields for the persisted run config.
    config.in_channels = int(sd3_config.in_channels)
    config.joint_attention_dim = int(sd3_config.joint_attention_dim)
    config.pooled_projection_dim = int(sd3_config.pooled_projection_dim)
    config.sample_size = int(sd3_config.sample_size)

    dataset = make_synthetic_dataset(config, sd3_config)

    # Custom train_step computes the (weighted) loss -> compile needs only an
    # optimizer (no loss arg).
    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
    )

    # No validation stream -> monitor the training loss; TerminateOnNaN is the
    # smoke gate. The analyzer needs a built/var-rich graph this dict-input
    # subclass does not expose cleanly, so it is disabled.
    callbacks, results_dir = create_callbacks(
        model_name=f"{config.variant}",
        results_dir_prefix=config.results_prefix,
        output_root=config.output_dir,
        monitor="loss",
        patience=config.epochs,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    save_config_json(config, results_dir)

    logger.info(
        "[train_sd3_mmdit] starting fit: variant=%s epochs=%d steps/epoch=%d "
        "batch=%d sample_size=%d in_channels=%d L=%d joint_dim=%d pooled_dim=%d",
        config.variant,
        config.epochs,
        config.steps_per_epoch,
        config.batch_size,
        config.sample_size,
        config.in_channels,
        config.num_text_tokens,
        config.joint_attention_dim,
        config.pooled_projection_dim,
    )

    history = trainer.fit(
        dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    final_loss = float(history.history["loss"][-1])
    logger.info(
        "[train_sd3_mmdit] done. final loss=%.6f results_dir=%s",
        final_loss,
        results_dir,
    )

    return {"history": history.history, "results_dir": results_dir}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Synthetic flow-matching smoke-train for SD3 MMDiT (tiny)."
    )
    parser.add_argument("--variant", type=str, default="tiny")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-3,
                        dest="learning_rate")
    parser.add_argument("--num-text-tokens", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="results/sd3_mmdit")
    parser.add_argument("--mixed-bfloat16", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainingConfig(
        variant=args.variant,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_text_tokens=args.num_text_tokens,
        output_dir=args.output_dir,
        mixed_bfloat16=args.mixed_bfloat16,
        gpu=args.gpu,
        seed=args.seed,
    )


def main() -> None:
    config = _parse_args()
    train(config)


if __name__ == "__main__":
    main()
