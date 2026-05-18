"""
VQ-VAE model with the Rotation Trick quantizer (Fifty et al., ICLR 2025).

Mirrors the shape of ``VQVAEModel`` but routes through
``VectorQuantizerRotationTrick`` and uses the dl_techniques normalisation
factory (``create_normalization_layer``) inside its auto-built convolutional
encoder/decoder. Two construction paths:

1. **Bring-your-own encoder/decoder** — pass two ``keras.Model`` instances.
2. **Auto-build** — leave ``encoder=None, decoder=None`` and supply
   ``input_shape``; a small Conv2D encoder/decoder is built internally with
   the chosen norm type, hidden channel width, downsample factor, and residual
   block count.

**Architecture Overview:**

.. code-block:: text

    ┌────────────────────────────────────────┐
    │  Input image x [B, H, W, C_in]         │
    └──────────────┬─────────────────────────┘
                   ▼
    ┌────────────────────────────────────────┐
    │  Encoder  (BYO  OR  auto-built)        │
    │  ─ Conv2D(hidden, 4, s=2) × n_down     │
    │    + Norm (create_normalization_layer) │
    │    + swish                             │
    │  ─ ResBlock × num_res_blocks           │
    │  ─ Conv2D(embedding_dim, 1, 1)         │
    └──────────────┬─────────────────────────┘
                   ▼  z_e [B, H/f, W/f, D]
    ┌────────────────────────────────────────┐
    │  VectorQuantizerRotationTrick          │
    │    gradient_mode ∈                     │
    │      {rotation, reflection,            │
    │       no_grad_scale, ste}              │
    │    distance_mode ∈ {euclidean, cosine} │
    │    num_heads / use_ema / kmeans_init / │
    │    dead_code_reinit / diversity / orth │
    │    → aux losses via add_loss()         │
    └──────────────┬─────────────────────────┘
                   ▼  z_q [B, H/f, W/f, D]
    ┌────────────────────────────────────────┐
    │  Decoder  (BYO  OR  auto-built)        │
    │  ─ Conv2D(hidden, 1, 1)                │
    │  ─ ResBlock × num_res_blocks           │
    │  ─ Conv2DTranspose(hidden, 4, s=2)     │
    │    × n_down                            │
    │    + Norm (create_normalization_layer) │
    │    + swish                             │
    │  ─ Conv2D(C_in, 3, 1, padding=same)    │
    └──────────────┬─────────────────────────┘
                   ▼
    ┌────────────────────────────────────────┐
    │  Output x_rec [B, H, W, C_in]          │
    │  total_loss = recon(x, x_rec)          │
    │               + sum(layer.losses)      │
    └────────────────────────────────────────┘

References:
    - Fifty, C., Junkins, R., Duan, D., et al. (2025). Restructuring Vector
      Quantization with the Rotation Trick. ICLR 2025.
    - van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). Neural Discrete
      Representation Learning. NeurIPS 2017.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import tensorflow as tf
from keras import initializers, ops

from dl_techniques.layers.norms.factory import (
    NormalizationType,
    create_normalization_layer,
)
from dl_techniques.layers.vector_quantizer_rotation_trick import (
    VectorQuantizerRotationTrick,
)


# ---------------------------------------------------------------------


def _build_auto_encoder(
        input_shape: Tuple[int, int, int],
        hidden_channels: int,
        embedding_dim: int,
        downsample_factor: int,
        num_res_blocks: int,
        norm_type: NormalizationType,
) -> keras.Model:
    """Small Conv2D encoder with stride-2 downsamples + residual blocks."""
    import math
    n_down = int(math.log2(downsample_factor))
    if 2 ** n_down != downsample_factor:
        raise ValueError(
            f"downsample_factor must be a power of 2, got {downsample_factor}"
        )

    inp = keras.Input(shape=input_shape, name="encoder_input")
    x = inp
    for i in range(n_down):
        x = keras.layers.Conv2D(
            hidden_channels, 4, strides=2, padding="same",
            name=f"enc_down_{i}",
        )(x)
        x = create_normalization_layer(norm_type, name=f"enc_norm_{i}")(x)
        x = keras.layers.Activation("relu", name=f"enc_act_{i}")(x)

    for j in range(num_res_blocks):
        residual = x
        h = keras.layers.Conv2D(
            hidden_channels, 3, padding="same", name=f"enc_res_{j}_a"
        )(x)
        h = create_normalization_layer(norm_type, name=f"enc_res_{j}_norm_a")(h)
        h = keras.layers.Activation("relu", name=f"enc_res_{j}_act")(h)
        h = keras.layers.Conv2D(
            hidden_channels, 3, padding="same", name=f"enc_res_{j}_b"
        )(h)
        x = keras.layers.Add(name=f"enc_res_{j}_add")([residual, h])

    out = keras.layers.Conv2D(
        embedding_dim, 1, padding="same", name="enc_to_embedding"
    )(x)
    return keras.Model(inp, out, name="vqvae_rotation_encoder")


def _build_auto_decoder(
        latent_shape: Tuple[Optional[int], Optional[int], int],
        output_channels: int,
        hidden_channels: int,
        downsample_factor: int,
        num_res_blocks: int,
        norm_type: NormalizationType,
) -> keras.Model:
    """Small Conv2DTranspose decoder mirroring the auto encoder."""
    import math
    n_up = int(math.log2(downsample_factor))

    inp = keras.Input(shape=latent_shape, name="decoder_input")
    x = keras.layers.Conv2D(
        hidden_channels, 1, padding="same", name="dec_from_embedding"
    )(inp)

    for j in range(num_res_blocks):
        residual = x
        h = keras.layers.Conv2D(
            hidden_channels, 3, padding="same", name=f"dec_res_{j}_a"
        )(x)
        h = create_normalization_layer(norm_type, name=f"dec_res_{j}_norm_a")(h)
        h = keras.layers.Activation("relu", name=f"dec_res_{j}_act")(h)
        h = keras.layers.Conv2D(
            hidden_channels, 3, padding="same", name=f"dec_res_{j}_b"
        )(h)
        x = keras.layers.Add(name=f"dec_res_{j}_add")([residual, h])

    for i in range(n_up):
        x = keras.layers.Conv2DTranspose(
            hidden_channels, 4, strides=2, padding="same",
            name=f"dec_up_{i}",
        )(x)
        x = create_normalization_layer(norm_type, name=f"dec_norm_{i}")(x)
        x = keras.layers.Activation("relu", name=f"dec_act_{i}")(x)

    out = keras.layers.Conv2D(
        output_channels, 3, padding="same", activation="sigmoid",
        name="dec_to_output",
    )(x)
    return keras.Model(inp, out, name="vqvae_rotation_decoder")


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VQVAERotationTrick(keras.Model):
    """Rotation Trick VQ-VAE.

    Either:
      - pass ``encoder`` + ``decoder`` (bring-your-own), OR
      - leave both ``None`` and pass ``input_shape`` for the auto path.

    All quantizer flags are re-exposed at the model level and forwarded to
    ``VectorQuantizerRotationTrick``.

    :param encoder: Optional encoder model. If ``None``, auto-built.
    :param decoder: Optional decoder model. If ``None``, auto-built.
    :param num_embeddings: Codebook size.
    :param embedding_dim: Channel dim of latent (matches encoder output).
    :param commitment_cost: Beta for commitment loss.
    :param gradient_mode: ``rotation`` | ``reflection`` | ``no_grad_scale`` | ``ste``.
    :param distance_mode: ``euclidean`` | ``cosine``.
    :param use_ema: EMA codebook updates.
    :param ema_decay: EMA decay.
    :param num_heads: Multi-head codebook count.
    :param kmeans_init: Run one-shot k-means warm start.
    :param kmeans_init_steps: Number of batches accumulated before k-means.
    :param kmeans_seed: Deterministic numpy seed for k-means.
    :param dead_code_threshold: Consecutive unused-call count for re-init.
    :param diversity_coefficient: Aux loss weight.
    :param orthogonal_reg_coefficient: Aux loss weight.
    :param reconstruction_loss_weight: Weight for recon term.
    :param quantizer_initializer: Codebook initializer.
    :param input_shape: For auto-build path only — image shape (H, W, C).
    :param hidden_channels: Auto-build conv width.
    :param downsample_factor: Auto-build encoder/decoder stride compound.
    :param num_res_blocks: Number of residual blocks per stage.
    :param norm_type: Normalization layer type used in auto-built encoder/decoder.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            encoder: Optional[keras.Model] = None,
            decoder: Optional[keras.Model] = None,
            commitment_cost: float = 0.25,
            gradient_mode: str = "rotation",
            distance_mode: str = "euclidean",
            use_ema: bool = False,
            ema_decay: float = 0.99,
            num_heads: int = 1,
            kmeans_init: bool = False,
            kmeans_init_steps: int = 1,
            kmeans_seed: int = 42,
            dead_code_threshold: int = 0,
            diversity_coefficient: float = 0.0,
            orthogonal_reg_coefficient: float = 0.0,
            reconstruction_loss_weight: float = 1.0,
            quantizer_initializer: Union[str, initializers.Initializer] = "uniform",
            input_shape: Optional[Tuple[int, int, int]] = None,
            hidden_channels: int = 128,
            downsample_factor: int = 4,
            num_res_blocks: int = 2,
            norm_type: NormalizationType = "layer_norm",
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_embeddings <= 0:
            raise ValueError(f"num_embeddings must be positive, got {num_embeddings}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if reconstruction_loss_weight <= 0:
            raise ValueError(
                f"reconstruction_loss_weight must be positive, "
                f"got {reconstruction_loss_weight}"
            )

        # Auto-build if both encoder/decoder are None.
        if encoder is None and decoder is None:
            if input_shape is None:
                raise ValueError(
                    "Auto-build path requires input_shape=(H, W, C) when "
                    "encoder and decoder are not supplied."
                )
            encoder = _build_auto_encoder(
                input_shape=input_shape,
                hidden_channels=hidden_channels,
                embedding_dim=embedding_dim,
                downsample_factor=downsample_factor,
                num_res_blocks=num_res_blocks,
                norm_type=norm_type,
            )
            h_out = input_shape[0] // downsample_factor
            w_out = input_shape[1] // downsample_factor
            decoder = _build_auto_decoder(
                latent_shape=(h_out, w_out, embedding_dim),
                output_channels=input_shape[2],
                hidden_channels=hidden_channels,
                downsample_factor=downsample_factor,
                num_res_blocks=num_res_blocks,
                norm_type=norm_type,
            )
        elif encoder is None or decoder is None:
            raise ValueError(
                "Must supply BOTH encoder and decoder, or NEITHER (auto-build)."
            )

        self.encoder = encoder
        self.decoder = decoder

        self.quantizer = VectorQuantizerRotationTrick(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            gradient_mode=gradient_mode,
            distance_mode=distance_mode,
            initializer=quantizer_initializer,
            use_ema=use_ema,
            ema_decay=ema_decay,
            num_heads=num_heads,
            kmeans_init=kmeans_init,
            kmeans_init_steps=kmeans_init_steps,
            kmeans_seed=kmeans_seed,
            dead_code_threshold=dead_code_threshold,
            diversity_coefficient=diversity_coefficient,
            orthogonal_reg_coefficient=orthogonal_reg_coefficient,
            name="vector_quantizer_rotation_trick",
        )

        # Stash config for get_config.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.gradient_mode = gradient_mode
        self.distance_mode = distance_mode
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.num_heads = num_heads
        self.kmeans_init = kmeans_init
        self.kmeans_init_steps = kmeans_init_steps
        self.kmeans_seed = kmeans_seed
        self.dead_code_threshold = dead_code_threshold
        self.diversity_coefficient = diversity_coefficient
        self.orthogonal_reg_coefficient = orthogonal_reg_coefficient
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self._input_shape = input_shape
        self.hidden_channels = hidden_channels
        self.downsample_factor = downsample_factor
        self.num_res_blocks = num_res_blocks
        self.norm_type = norm_type

        if isinstance(quantizer_initializer, str):
            self.quantizer_initializer = initializers.get(quantizer_initializer)
        else:
            self.quantizer_initializer = quantizer_initializer

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        z_e = self.encoder(inputs, training=training)
        z_q = self.quantizer(z_e, training=training)
        return self.decoder(z_q, training=training)

    def train_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            x_recon = self(x, training=True)
            recon_loss = ops.mean(ops.square(x - x_recon))
            recon_loss = self.reconstruction_loss_weight * recon_loss
            vq_losses = self.quantizer.losses
            vq_loss = ops.sum(ops.stack(vq_losses)) if vq_losses else 0.0
            total = recon_loss + vq_loss
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        x = data[0] if isinstance(data, tuple) else data
        x_recon = self(x, training=False)
        recon_loss = ops.mean(ops.square(x - x_recon))
        recon_loss = self.reconstruction_loss_weight * recon_loss
        vq_losses = self.quantizer.losses
        vq_loss = ops.sum(ops.stack(vq_losses)) if vq_losses else 0.0
        total = recon_loss + vq_loss
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.vq_loss_tracker.update_state(vq_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    # ---- convenience helpers ----

    def encode(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        return self.encoder(inputs, training=False)

    def decode(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        return self.decoder(latents, training=False)

    def encode_to_indices(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        z_e = self.encode(inputs)
        return self.quantizer.get_codebook_indices(z_e)

    def decode_from_indices(self, indices: keras.KerasTensor) -> keras.KerasTensor:
        z_q = self.quantizer.quantize_from_indices(indices)
        return self.decode(z_q)

    # ---- config ----

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "decoder": keras.saving.serialize_keras_object(self.decoder),
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "commitment_cost": self.commitment_cost,
                "gradient_mode": self.gradient_mode,
                "distance_mode": self.distance_mode,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
                "num_heads": self.num_heads,
                "kmeans_init": self.kmeans_init,
                "kmeans_init_steps": self.kmeans_init_steps,
                "kmeans_seed": self.kmeans_seed,
                "dead_code_threshold": self.dead_code_threshold,
                "diversity_coefficient": self.diversity_coefficient,
                "orthogonal_reg_coefficient": self.orthogonal_reg_coefficient,
                "reconstruction_loss_weight": self.reconstruction_loss_weight,
                "quantizer_initializer": initializers.serialize(
                    self.quantizer_initializer
                ),
                "input_shape_config": self._input_shape,
                "hidden_channels": self.hidden_channels,
                "downsample_factor": self.downsample_factor,
                "num_res_blocks": self.num_res_blocks,
                "norm_type": self.norm_type,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQVAERotationTrick":
        encoder_cfg = config.pop("encoder")
        decoder_cfg = config.pop("decoder")
        encoder = keras.saving.deserialize_keras_object(encoder_cfg)
        decoder = keras.saving.deserialize_keras_object(decoder_cfg)
        config["input_shape"] = config.pop("input_shape_config", None)
        return cls(encoder=encoder, decoder=decoder, **config)
