"""Vector-Quantized VAE: an autoencoder with a discrete latent bottleneck.

Builds the model from a caller-supplied encoder and decoder plus a
``VectorQuantizer`` bottleneck between them. The encoder output is snapped
to the nearest vector in a learned codebook before decoding; a
straight-through estimator carries gradients through that non-differentiable
step. This replaces a standard VAE's continuous, Gaussian latent with a
finite set of codes, avoiding posterior collapse and giving a discrete
representation an autoregressive prior (PixelCNN, WaveNet) can later be
trained on.

The total loss is reconstruction plus codebook plus commitment, where the
codebook and commitment terms come from the quantizer's own ``add_loss``
calls. ``use_ema=True`` updates the codebook by exponential moving average
instead of by gradient descent.

References:
    - van den Oord et al., 2017. Neural Discrete Representation Learning.
      (https://arxiv.org/abs/1711.00937)
"""

import keras
import tensorflow as tf
from keras import ops, initializers
from typing import Optional, Tuple, Dict, Any, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.generative.vector_quantizer import VectorQuantizer
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------



@register_dl_technique("dl_techniques.models.vq_vae.model")
class VQVAEModel(keras.Model):
    """VQ-VAE: an encoder-decoder model with a discrete codebook bottleneck.

    Wraps a caller-supplied encoder and decoder around a
    :class:`~dl_techniques.layers.generative.vector_quantizer.VectorQuantizer`. Loss
    computation happens inside :meth:`train_step`, so ``compile`` only needs
    an optimizer.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input x                             │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  encoder → z_e(x), continuous        │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  VectorQuantizer                     │
        │  k* = argmin ||z_e - e_j||           │
        │  z_q = e_k*                          │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  decoder → x_recon                   │
        └───────────────┬──────────────────────┘
                        ▼
                  Output x_recon

        loss = MSE(x, x_recon)
             + ||sg[z_e] - e||^2          (codebook term)
             + beta * ||z_e - sg[e]||^2   (commitment term)

        sg[] is the stop-gradient operator; the codebook and
        commitment terms come from the quantizer's own add_loss calls.

    :param encoder: Network mapping inputs to continuous latents whose last
        axis is ``embedding_dim``.
    :type encoder: keras.Model
    :param decoder: Network mapping quantized latents back to the input
        space.
    :type decoder: keras.Model
    :param num_embeddings: Codebook size K. Must be positive.
    :type num_embeddings: int
    :param embedding_dim: Codebook vector dimension D. Must be positive and
        match the encoder's output channel count.
    :type embedding_dim: int
    :param commitment_cost: Weight beta of the commitment term. Defaults to
        0.25.
    :type commitment_cost: float
    :param use_ema: If True the codebook is updated by exponential moving
        average instead of gradient descent. Defaults to False.
    :type use_ema: bool
    :param ema_decay: EMA decay used when ``use_ema`` is True. Defaults to
        0.99.
    :type ema_decay: float
    :param reconstruction_loss_weight: Weight of the MSE reconstruction
        term. Must be positive. Defaults to 1.0.
    :type reconstruction_loss_weight: float
    :param quantizer_initializer: Initializer for the codebook embeddings.
        Defaults to ``'uniform'``.
    :type quantizer_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the ``Model`` base
        class.

    :raises ValueError: If ``num_embeddings``, ``embedding_dim`` or
        ``reconstruction_loss_weight`` is not positive.

    :ivar encoder: The encoder network.
    :ivar decoder: The decoder network.
    :ivar quantizer: The ``VectorQuantizer`` layer.
    :ivar total_loss_tracker: Metric tracking total loss.
    :ivar reconstruction_loss_tracker: Metric tracking reconstruction loss.
    :ivar vq_loss_tracker: Metric tracking codebook and commitment loss.

    Example:
        .. code-block:: python

            encoder = keras.Sequential([
                keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
                keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),
            ])
            decoder = keras.Sequential([
                keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
                keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid'),
            ])
            vqvae = VQVAEModel(
                encoder=encoder,
                decoder=decoder,
                num_embeddings=512,
                embedding_dim=64,
                commitment_cost=0.25,
            )
            vqvae.compile(optimizer='adam')
            vqvae.fit(train_data, epochs=100)

    Note:
        After training, a prior (PixelCNN, WaveNet) can be trained on the
        discrete codes from :meth:`encode_to_indices`, then sampled and
        decoded with :meth:`decode_from_indices`.

    References:
        - van den Oord et al., 2017. Neural Discrete Representation
          Learning. (https://arxiv.org/abs/1711.00937)
    """

    def __init__(
            self,
            encoder: keras.Model,
            decoder: keras.Model,
            num_embeddings: int,
            embedding_dim: int,
            commitment_cost: float = 0.25,
            use_ema: bool = False,
            ema_decay: float = 0.99,
            reconstruction_loss_weight: float = 1.0,
            quantizer_initializer: Union[str, initializers.Initializer] = "uniform",
            **kwargs: Any
    ) -> None:
        """Initialize the VQ-VAE model."""
        super().__init__(**kwargs)

        if num_embeddings <= 0:
            raise ValueError(
                f"num_embeddings must be positive, got {num_embeddings}"
            )
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        if reconstruction_loss_weight <= 0:
            raise ValueError(
                f"reconstruction_loss_weight must be positive, "
                f"got {reconstruction_loss_weight}"
            )

        self.encoder = encoder
        self.decoder = decoder

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            initializer=quantizer_initializer,
            use_ema=use_ema,
            ema_decay=ema_decay,
            name="vector_quantizer"
        )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.reconstruction_loss_weight = reconstruction_loss_weight

        if isinstance(quantizer_initializer, str):
            self.quantizer_initializer = initializers.get(quantizer_initializer)
        else:
            self.quantizer_initializer = quantizer_initializer

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method VQVAEModel inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        Args:
            input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the VQ-VAE: encode, quantize, decode.

        :param inputs: Input data to reconstruct.
        :type inputs: keras.KerasTensor
        :param training: Whether the model is in training mode; affects the
            quantizer's EMA updates.
        :type training: Optional[bool]
        :return: Reconstructed outputs, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        z_e = self.encoder(inputs, training=training)

        z_q = self.quantizer(z_e, training=training)

        reconstructed = self.decoder(z_q, training=training)

        return reconstructed

    def train_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        """Run one training step, computing and applying the VQ-VAE losses.

        :param data: Input data: a single tensor for unsupervised training,
            or a tuple ``(inputs, targets)`` / ``(inputs, targets,
            sample_weight)``.
        :type data: Union[keras.KerasTensor, Tuple]
        :return: Mapping from metric name to its current value.
        :rtype: Dict[str, Any]
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            x_recon = self(x, training=True)

            # DECISION plan-2026-08-19T163559-499b6f0e/D-011: cast the prediction up to
            # float32, never the data down; under mixed_float16 the subtraction otherwise
            # raises a dtype TypeError, and a float16 squared-error mean underflows. See decisions.md.
            x_recon = ops.cast(x_recon, "float32")

            # Compute reconstruction loss (MSE)
            reconstruction_loss = ops.mean((x - x_recon) ** 2)
            reconstruction_loss = (
                    self.reconstruction_loss_weight * reconstruction_loss
            )

            # DECISION plan-2026-08-18T140459-7991552f/D-026: sum self.losses, not
            # self.quantizer.losses; the narrow form silently drops regularizers on a
            # caller-supplied encoder or decoder. See decisions.md.
            aux_losses = self.losses
            # DECISION plan-2026-08-19T163559-499b6f0e/D-011: cast the aux-loss sum up to
            # float32 too, since add_loss terms carry compute_dtype. See decisions.md.
            vq_loss = (
                ops.cast(ops.sum(ops.stack(aux_losses)), "float32")
                if aux_losses else 0.0
            )

            total_loss = reconstruction_loss + vq_loss

            # DECISION plan-2026-08-19T163559-499b6f0e/D-089: call scale_loss inside the
            # tape; under mixed_float16 skipping it divides the whole weight update by the
            # loss scale with no warning. See decisions.md.
            scaled_loss = self.optimizer.scale_loss(total_loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(vq_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        """Run one evaluation step, reporting the same losses as :meth:`train_step`.

        :param data: Input data: a single tensor for unsupervised training,
            or a tuple ``(inputs, targets)`` / ``(inputs, targets,
            sample_weight)``.
        :type data: Union[keras.KerasTensor, Tuple]
        :return: Mapping from metric name to its current value.
        :rtype: Dict[str, Any]
        """
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        x_recon = self(x, training=False)

        # Compute reconstruction loss
        reconstruction_loss = ops.mean((x - x_recon) ** 2)
        reconstruction_loss = self.reconstruction_loss_weight * reconstruction_loss

        # DECISION plan-2026-08-18T140459-7991552f/D-026: sum self.losses, not
        # self.quantizer.losses, so val_loss stays comparable to the training objective. See decisions.md.
        aux_losses = self.losses
        vq_loss = ops.sum(ops.stack(aux_losses)) if aux_losses else 0.0

        total_loss = reconstruction_loss + vq_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(vq_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return the list of metrics tracked by the model.

        :return: The loss trackers.
        :rtype: List[keras.metrics.Metric]
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def encode(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Encode inputs to continuous latent representations.

        :param inputs: Input data.
        :type inputs: keras.KerasTensor
        :return: Continuous latent representations ``z_e(x)``.
        :rtype: keras.KerasTensor
        """
        return self.encoder(inputs, training=False)

    def quantize_latents(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        """Quantize continuous latents to discrete representations.

        Named ``quantize_latents``, not ``quantize``, to avoid colliding
        with the Keras quantization API.

        :param latents: Continuous latent representations ``z_e``.
        :type latents: keras.KerasTensor
        :return: Quantized latent representations ``z_q``.
        :rtype: keras.KerasTensor
        """
        return self.quantizer(latents, training=False)

    def decode(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        """Decode latent representations to reconstructed outputs.

        :param latents: Quantized latent representations ``z_q``.
        :type latents: keras.KerasTensor
        :return: Reconstructed outputs.
        :rtype: keras.KerasTensor
        """
        return self.decoder(latents, training=False)

    def encode_to_indices(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Encode inputs directly to discrete codebook indices.

        Useful for training autoregressive priors or compressing data.

        :param inputs: Input data.
        :type inputs: keras.KerasTensor
        :return: Integer tensor of codebook indices.
        :rtype: keras.KerasTensor

        Example:
            .. code-block:: python

                indices = vqvae.encode_to_indices(images)
                prior.fit(indices, epochs=100)
        """
        z_e = self.encode(inputs)
        indices = self.quantizer.get_codebook_indices(z_e)
        return indices

    def decode_from_indices(
            self,
            indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Decode discrete codebook indices to reconstructed outputs.

        Useful for sampling from autoregressive priors.

        :param indices: Integer tensor of codebook indices.
        :type indices: keras.KerasTensor
        :return: Reconstructed outputs.
        :rtype: keras.KerasTensor

        Example:
            .. code-block:: python

                sampled_indices = prior.sample(batch_size=16)
                generated = vqvae.decode_from_indices(sampled_indices)
        """
        z_q = self.quantizer.quantize_from_indices(indices)
        reconstructed = self.decode(z_q)
        return reconstructed

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "quantizer_initializer": initializers.serialize(
                self.quantizer_initializer
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VQVAEModel":
        """Recreate a model from its serialized configuration.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new ``VQVAEModel`` instance.
        :rtype: VQVAEModel
        """
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)

        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)

        return cls(encoder=encoder, decoder=decoder, **config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_vq_vae(
        encoder: keras.Model,
        decoder: keras.Model,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        reconstruction_loss_weight: float = 1.0,
        quantizer_initializer: Union[str, initializers.Initializer] = "uniform",
        **kwargs: Any
) -> VQVAEModel:
    """Create a VQ-VAE model over a caller-supplied encoder and decoder.

    VQ-VAE is a quantization scheme wrapped around an arbitrary autoencoder,
    with no backbone of its own, so there is no ``MODEL_VARIANTS`` table and
    ``encoder``/``decoder`` stay required arguments.

    :param encoder: Network mapping inputs to continuous latents whose last
        axis is ``embedding_dim``.
    :type encoder: keras.Model
    :param decoder: Network mapping quantized latents back to the input
        space.
    :type decoder: keras.Model
    :param num_embeddings: Codebook size K. Must be positive.
    :type num_embeddings: int
    :param embedding_dim: Codebook vector dimension D. Must be positive and
        match the encoder's output channel count.
    :type embedding_dim: int
    :param commitment_cost: Weight beta of the commitment term.
    :type commitment_cost: float
    :param use_ema: If True the codebook is updated by EMA rather than
        gradient descent.
    :type use_ema: bool
    :param ema_decay: EMA decay used when ``use_ema`` is True.
    :type ema_decay: float
    :param reconstruction_loss_weight: Weight of the MSE reconstruction
        term. Must be positive.
    :type reconstruction_loss_weight: float
    :param quantizer_initializer: Initializer for the codebook embeddings.
    :type quantizer_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments forwarded to the model
        constructor.
    :return: A configured ``VQVAEModel`` instance.
    :rtype: VQVAEModel
    :raises ValueError: If ``num_embeddings``, ``embedding_dim`` or
        ``reconstruction_loss_weight`` is not positive.

    Example:
        .. code-block:: python

            enc = keras.Sequential([keras.layers.Dense(16)])
            dec = keras.Sequential([keras.layers.Dense(8)])
            model = create_vq_vae(enc, dec, num_embeddings=32, embedding_dim=16)
            model(keras.random.normal((2, 8))).shape  # (2, 8)
    """
    return VQVAEModel(
        encoder=encoder,
        decoder=decoder,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost,
        use_ema=use_ema,
        ema_decay=ema_decay,
        reconstruction_loss_weight=reconstruction_loss_weight,
        quantizer_initializer=quantizer_initializer,
        **kwargs
    )

# ---------------------------------------------------------------------