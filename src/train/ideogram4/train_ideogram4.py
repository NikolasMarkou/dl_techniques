"""Synthetic flow-matching smoke-trainer for the Ideogram4 DiT (tiny preset).

This script trains the :class:`Ideogram4Transformer` velocity head on a SYNTHETIC
rectified-flow task so the architecture's training path can be exercised end to
end on a single GPU. It is a smoke oracle, NOT a convergence run: the data is
random Gaussian latents with no learnable structure, so the loss should stay
finite (and typically dips early) without claiming generation quality.

Rectified-flow setup (assumption A8 in the plan):

    x0       ~ N(0, I)                 clean image latents (per image token)
    x1       ~ N(0, I)                 noise
    tau      ~ U(0, 1)                 per-sample flow time
    x_t      = (1 - tau) * x0 + tau * x1   noised latent fed at image positions
    v_target = x1 - x0                 the velocity the model must predict

The packed sequence layout mirrors the pipeline's ``_build_inputs``:
``[T text tokens][grid_h*grid_w image tokens]``. Text positions carry random
``llm_features`` (so conditioning is active); image positions carry ``x_t``.
Because the image tokens are the trailing ``N = grid_h*grid_w`` of the sequence,
the trainer model returns ``velocity[:, T:]`` and the dataset target is the
velocity at image positions ``(B, N, in_channels)`` -- a fixed-T slice -- so a
plain ``compile(loss=FlowMatchingVelocityLoss())`` trains ONLY on image tokens
without a custom ``train_step``.

Run::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \
        .venv/bin/python -m train.ideogram4.train_ideogram4 \
        --variant tiny --epochs 3 --steps-per-epoch 8 --batch-size 4 --gpu 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.losses import FlowMatchingVelocityLoss
from dl_techniques.models.ideogram4.config import get_ideogram4_config
from dl_techniques.models.ideogram4.transformer import (
    Ideogram4Transformer,
    create_ideogram4_transformer,
)
from dl_techniques.models.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)

from train.common import setup_gpu, set_seeds, create_callbacks, save_config_json


# ---------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for the synthetic Ideogram4 smoke-train.

    Args:
        variant: Config preset name (``"tiny"`` smoke-trainable / ``"full"``).
        batch_size: Sequences per gradient step.
        steps_per_epoch: Synthetic batches drawn per epoch.
        epochs: Number of epochs.
        num_text_tokens: Conditioning text tokens ``T`` per sequence.
        grid_h: Image-grid height in tokens.
        grid_w: Image-grid width in tokens (``grid_h*grid_w`` image tokens).
        learning_rate: Adam learning rate.
        output_dir: Results root (repo-root ``results/...`` by convention).
        results_prefix: Directory-name prefix inside ``output_dir``.
        mixed_bfloat16: If True, set the global ``mixed_bfloat16`` policy before
            building the model (velocity head stays float32 inside the model).
        gpu: GPU index forwarded to ``setup_gpu`` (also sets CUDA_VISIBLE_DEVICES).
        seed: RNG seed for reproducible synthetic data + init.
    """

    variant: str = "tiny"
    batch_size: int = 4
    steps_per_epoch: int = 8
    epochs: int = 3
    num_text_tokens: int = 8
    grid_h: int = 4
    grid_w: int = 4
    learning_rate: float = 1e-3
    output_dir: str = "results/ideogram4"
    results_prefix: str = "ideogram4"
    mixed_bfloat16: bool = False
    gpu: int = None
    seed: int = 42

    # --- derived (filled at build time; serialized for the run config) ---
    in_channels: int = field(default=0)
    llm_features_dim: int = field(default=0)
    num_image_tokens: int = field(default=0)
    seq_len: int = field(default=0)


# ---------------------------------------------------------------------
# Packed-sequence static index tensors (mirror pipeline._build_inputs)
# ---------------------------------------------------------------------


def build_packed_indices(
    num_text_tokens: int,
    grid_h: int,
    grid_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build the per-sample ``position_ids`` / ``segment_ids`` / ``indicator``.

    Layout ``[T text tokens][grid_h*grid_w image tokens]``, identical to the
    pipeline's ``_build_inputs`` (single segment, image positions offset by
    ``IMAGE_POSITION_OFFSET``). These arrays are static across the synthetic
    dataset; only the latents / llm_features / time vary per sample.

    Args:
        num_text_tokens: ``T`` text tokens.
        grid_h: Image-grid height in tokens.
        grid_w: Image-grid width in tokens.

    Returns:
        ``(position_ids, segment_ids, indicator, num_image, seq_len)`` where the
        three arrays have a leading length-1 axis for broadcasting over the batch.
    """
    num_image = grid_h * grid_w
    seq_len = num_text_tokens + num_image

    # Image position ids: t=0, h in [0,grid_h), w in [0,grid_w), offset to avoid
    # colliding with text positions.
    hh, ww = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    image_pos = np.stack(
        [
            np.zeros(num_image, dtype=np.int32),
            hh.reshape(-1).astype(np.int32),
            ww.reshape(-1).astype(np.int32),
        ],
        axis=-1,
    )
    image_pos = image_pos + IMAGE_POSITION_OFFSET

    text_arange = np.arange(num_text_tokens, dtype=np.int32)
    text_pos = np.stack([text_arange] * 3, axis=-1)

    pos_single = np.concatenate([text_pos, image_pos], axis=0)  # (L, 3)
    position_ids = pos_single[None].astype(np.int32)            # (1, L, 3)

    segment_ids = np.ones((1, seq_len), dtype=np.int32)

    indicator = np.empty((1, seq_len), dtype=np.int32)
    indicator[:, :num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[:, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    return position_ids, segment_ids, indicator, num_image, seq_len


# ---------------------------------------------------------------------
# Synthetic rectified-flow dataset
# ---------------------------------------------------------------------


def sample_flow_batch(
    batch_size: int,
    seq_len: int,
    in_channels: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Draw one synthetic rectified-flow batch.

    Args:
        batch_size: Sequences in the batch.
        seq_len: Packed sequence length ``L``.
        in_channels: Per-token latent channels.

    Returns:
        ``(x_t, v, tau, x0, x1)`` where ``x_t = (1-tau)*x0 + tau*x1`` is the
        noised latent path, ``v = x1 - x0`` is the velocity target (A8), and
        ``tau`` has shape ``(B, 1, 1)``. All tensors are float32. ``x0``/``x1``
        are returned so callers (and tests) can verify the target identity.
    """
    x0 = tf.random.normal((batch_size, seq_len, in_channels))
    x1 = tf.random.normal((batch_size, seq_len, in_channels))
    tau = tf.random.uniform((batch_size, 1, 1), minval=0.0, maxval=1.0)
    x_t = (1.0 - tau) * x0 + tau * x1
    v = x1 - x0
    return x_t, v, tau, x0, x1


def make_synthetic_dataset(
    config: TrainingConfig,
    in_channels: int,
    llm_features_dim: int,
) -> tf.data.Dataset:
    """Build an infinite ``tf.data`` dataset of synthetic rectified-flow batches.

    Each element is ``(inputs_dict, v_target_image)`` where ``inputs_dict`` has
    the six transformer keys and ``v_target_image`` is the velocity ``x1 - x0``
    at image positions only, shape ``(B, num_image, in_channels)``. Text-position
    velocities are never supervised because the trainer model slices them off.

    Args:
        config: Training configuration (batch size, grid, text tokens).
        in_channels: Per-token latent channels.
        llm_features_dim: Conditioning feature width.

    Returns:
        A repeating, prefetched ``tf.data.Dataset``.
    """
    B = config.batch_size
    T = config.num_text_tokens
    position_ids, segment_ids, indicator, num_image, seq_len = build_packed_indices(
        T, config.grid_h, config.grid_w
    )

    # Tile the static index arrays to the batch once (constant tensors).
    pos_b = tf.constant(np.broadcast_to(position_ids, (B, seq_len, 3)).copy())
    seg_b = tf.constant(np.broadcast_to(segment_ids, (B, seq_len)).copy())
    ind_b = tf.constant(np.broadcast_to(indicator, (B, seq_len)).copy())

    # Float mask selecting image positions in the packed sequence (B, L, 1).
    image_mask = tf.cast(
        tf.equal(ind_b, OUTPUT_IMAGE_INDICATOR)[..., None], tf.float32
    )
    text_mask = tf.cast(
        tf.equal(ind_b, LLM_TOKEN_INDICATOR)[..., None], tf.float32
    )

    def _gen(_):
        # Rectified-flow sampling. x0/x1 live at every position but only image
        # positions are fed/supervised; text positions are zeroed in x.
        x_t, v, tau, _x0, _x1 = sample_flow_batch(B, seq_len, in_channels)

        # Pack: image positions carry x_t, text positions carry 0.
        x_packed = x_t * image_mask

        # Random conditioning features at text positions (zero at image).
        llm_features = tf.random.normal((B, seq_len, llm_features_dim)) * text_mask

        # Per-sample flow time (B,) -- the transformer's ScalarSinusoidalEmbedding
        # broadcasts a (B,) time across L.
        t = tf.reshape(tau, (B,))

        inputs = {
            "x": x_packed,
            "llm_features": llm_features,
            "t": t,
            "position_ids": pos_b,
            "segment_ids": seg_b,
            "indicator": ind_b,
        }
        # Target: velocity at image positions only, (B, num_image, in_channels).
        v_image = v[:, T:, :]
        return inputs, v_image

    ds = tf.data.Dataset.range(1).repeat()
    ds = ds.map(_gen, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Trainer model: transformer velocity sliced to image positions
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.train.ideogram4")
class Ideogram4FlowTrainer(keras.Model):
    """Thin trainer wrapping :class:`Ideogram4Transformer`.

    ``call`` runs the transformer over the full packed sequence and returns the
    velocity at IMAGE positions only -- a fixed-T trailing slice
    ``velocity[:, num_text_tokens:]`` -- so a standard ``compile(loss=...)``
    supervises only image tokens (no custom ``train_step`` needed; the packed
    layout guarantees image tokens are the trailing ``N``).

    Args:
        transformer: The wrapped :class:`Ideogram4Transformer`.
        num_text_tokens: ``T`` -- the slice offset for image positions.
        **kwargs: Forwarded to ``keras.Model``.
    """

    def __init__(
        self,
        transformer: Ideogram4Transformer,
        num_text_tokens: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.transformer = transformer
        self.num_text_tokens = int(num_text_tokens)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: bool = None,
    ) -> keras.KerasTensor:
        velocity = self.transformer(inputs, training=training)  # (B, L, in_ch)
        return velocity[:, self.num_text_tokens:, :]            # (B, N, in_ch)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "transformer": keras.saving.serialize_keras_object(self.transformer),
                "num_text_tokens": self.num_text_tokens,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Ideogram4FlowTrainer":
        config = dict(config)
        config["transformer"] = keras.saving.deserialize_keras_object(
            config["transformer"]
        )
        return cls(**config)


def build_trainer(config: TrainingConfig) -> Tuple[Ideogram4FlowTrainer, int, int]:
    """Build the tiny transformer + flow trainer for ``config.variant``.

    Returns:
        ``(trainer, in_channels, llm_features_dim)``.
    """
    ic, _ae = get_ideogram4_config(config.variant)
    transformer = create_ideogram4_transformer(config.variant)
    trainer = Ideogram4FlowTrainer(
        transformer=transformer,
        num_text_tokens=config.num_text_tokens,
        name="ideogram4_flow_trainer",
    )
    return trainer, ic.in_channels, ic.llm_features_dim


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
        # Set the policy BEFORE building the model; the transformer pins its
        # velocity head to float32 internally, so this is safe for the smoke run.
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logger.info("[train_ideogram4] global policy set to mixed_bfloat16")

    trainer, in_channels, llm_features_dim = build_trainer(config)

    # Populate the derived fields for the persisted run config.
    config.in_channels = int(in_channels)
    config.llm_features_dim = int(llm_features_dim)
    config.num_image_tokens = config.grid_h * config.grid_w
    config.seq_len = config.num_text_tokens + config.num_image_tokens

    dataset = make_synthetic_dataset(config, in_channels, llm_features_dim)

    trainer.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=FlowMatchingVelocityLoss(),
    )

    # No validation stream -> monitor the training loss; TerminateOnNaN is the
    # smoke gate. The analyzer needs a built/var-rich model graph that this
    # dict-input subclass does not expose cleanly, so it is disabled.
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
        "[train_ideogram4] starting fit: variant=%s epochs=%d steps/epoch=%d "
        "batch=%d seq_len=%d (T=%d + %d image) in_channels=%d",
        config.variant,
        config.epochs,
        config.steps_per_epoch,
        config.batch_size,
        config.seq_len,
        config.num_text_tokens,
        config.num_image_tokens,
        config.in_channels,
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
        "[train_ideogram4] done. final loss=%.6f results_dir=%s",
        final_loss,
        results_dir,
    )

    return {"history": history.history, "results_dir": results_dir}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Synthetic flow-matching smoke-train for Ideogram4 (tiny)."
    )
    parser.add_argument("--variant", type=str, default="tiny")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-text-tokens", type=int, default=8)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--grid-w", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="results/ideogram4")
    parser.add_argument("--mixed-bfloat16", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainingConfig(
        variant=args.variant,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        num_text_tokens=args.num_text_tokens,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        learning_rate=args.learning_rate,
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
