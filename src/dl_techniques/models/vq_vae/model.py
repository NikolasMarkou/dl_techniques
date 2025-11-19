"""
Vector Quantised Variational AutoEncoder (VQ-VAE) Implementation.

This module implements the VQ-VAE model from "Neural Discrete Representation Learning"
(van den Oord et al., 2017). VQ-VAE learns discrete latent representations by combining
variational autoencoders with vector quantization, avoiding posterior collapse issues.

Reference:
    van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
    Neural Discrete Representation Learning. NeurIPS 2017.
    arXiv:1711.00937

Architecture Overview:

    Input x                                                     Output x_recon
      ↓                                                              ↑
    ┌─────────┐                                                ┌─────────┐
    │ Encoder │ → z_e(x) [continuous]                          │ Decoder │
    └─────────┘           ↓                                    └─────────┘
                    ┌──────────────┐                                ↑
                    │   Quantize   │ → Find nearest embedding       │
                    │   (L2 dist)  │   k = argmin ||z_e - e_j||²    │
                    └──────────────┘                                │
                          ↓                                         │
                    Embedding Table                                 │
                    e = [e_1, ..., e_K]                             │
                          ↓                                         │
                    z_q(x) = e_k [discrete] ────────────────────────┘

    Training losses (3 components):
    1. Reconstruction: log p(x|z_q(x))         - trains decoder & encoder
    2. Codebook:      ||sg[z_e(x)] - e||²      - trains embeddings
    3. Commitment:    β||z_e(x) - sg[e]||²     - trains encoder

    where sg[] is stop-gradient operator

Key Features:
    - Discrete latent space avoids posterior collapse
    - Straight-through gradient estimator for quantization
    - Optional exponential moving average (EMA) for codebook updates
    - Compatible with powerful autoregressive priors (PixelCNN, WaveNet)

Example:
    >>> # Define encoder and decoder
    >>> encoder = keras.Sequential([
    ...     keras.layers.Conv2D(64, 4, strides=2, padding='same', activation='relu'),
    ...     keras.layers.Conv2D(128, 4, strides=2, padding='same', activation='relu'),
    ... ])
    >>>
    >>> decoder = keras.Sequential([
    ...     keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', activation='relu'),
    ...     keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid'),
    ... ])
    >>>
    >>> # Create VQ-VAE model
    >>> vqvae = VQVAEModel(
    ...     encoder=encoder,
    ...     decoder=decoder,
    ...     num_embeddings=512,
    ...     embedding_dim=64,
    ...     commitment_cost=0.25
    ... )
    >>>
    >>> # Compile and train
    >>> vqvae.compile(optimizer='adam')
    >>> vqvae.fit(train_data, epochs=100)
"""

import keras
import tensorflow as tf
from keras import ops, initializers
from typing import Optional, Tuple, Dict, Any, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.vector_quantizer import VectorQuantizer

# ---------------------------------------------------------------------



@keras.saving.register_keras_serializable()
class VQVAEModel(keras.Model):
    """
    Complete VQ-VAE model combining encoder, quantizer, and decoder.

    This model implements the full VQ-VAE architecture that learns discrete latent
    representations. It can be used for various tasks including image generation,
    compression, and representation learning.

    **Architecture**:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      VQ-VAE Pipeline                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input x                                                    │
    │    ↓                                                        │
    │  ┌──────────┐                                               │
    │  │ Encoder  │ → z_e(x) [continuous, shape: H×W×D]           │
    │  └──────────┘                                               │
    │       ↓                                                     │
    │  ┌────────────────┐                                         │
    │  │  Quantizer     │ → z_q(x) [discrete, shape: H×W×D]       │
    │  │  - Find k* =   │    using codebook of K embeddings       │
    │  │    argmin||·|| │                                         │
    │  │  - z_q = e_k*  │                                         │
    │  └────────────────┘                                         │
    │       ↓                                                     │
    │  ┌──────────┐                                               │
    │  │ Decoder  │ → x_recon [reconstructed input]               │
    │  └──────────┘                                               │
    │       ↓                                                     │
    │  Output x_recon                                             │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Loss = reconstruction_loss + codebook_loss + commitment_loss
         = MSE(x, x_recon) + ||sg[z_e] - e||² + β||z_e - sg[e]||²
    ```

    **Training Process**:
    1. Encoder produces continuous latent z_e(x)
    2. Quantizer maps z_e to nearest codebook entry z_q
    3. Decoder reconstructs from z_q
    4. Three losses train different components:
       - Reconstruction: trains encoder + decoder
       - Codebook: trains embeddings
       - Commitment: trains encoder

    Args:
        encoder: Encoder network that maps inputs to continuous latents.
            Should output shape `(..., embedding_dim)`.
        decoder: Decoder network that reconstructs from quantized latents.
            Should accept input shape `(..., embedding_dim)`.
        num_embeddings: Size of discrete codebook (K). Typical values: 128-512.
        embedding_dim: Dimensionality of embeddings (D). Should match encoder output.
        commitment_cost: Weight for commitment loss (β). Prevents encoder from growing
            unbounded. Typical values: 0.25-0.5. Default: 0.25.
        use_ema: Whether to use EMA for codebook updates instead of gradients.
            EMA can be more stable but requires tuning. Default: False.
        ema_decay: Decay rate for EMA updates. Only used if use_ema=True.
            Default: 0.99.
        reconstruction_loss_weight: Weight for reconstruction loss. Can be used to
            balance reconstruction quality vs. codebook learning. Default: 1.0.
        quantizer_initializer: Initializer for embedding vectors. Default: 'uniform'.
        **kwargs: Additional arguments for Model base class.

    Attributes:
        encoder: The encoder network.
        decoder: The decoder network.
        quantizer: The VectorQuantizer layer.
        total_loss_tracker: Metric tracking total loss.
        reconstruction_loss_tracker: Metric tracking reconstruction loss.
        vq_loss_tracker: Metric tracking quantization losses.

    Example:
        >>> # Simple 2D convolution example for images
        >>> encoder = keras.Sequential([
        ...     keras.layers.Conv2D(64, 4, strides=2, padding='same'),
        ...     keras.layers.ReLU(),
        ...     keras.layers.Conv2D(128, 4, strides=2, padding='same'),
        ...     keras.layers.ReLU(),
        ...     keras.layers.Conv2D(64, 3, padding='same'),  # embedding_dim=64
        ... ])
        >>>
        >>> decoder = keras.Sequential([
        ...     keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        ...     keras.layers.ReLU(),
        ...     keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
        ...     keras.layers.ReLU(),
        ...     keras.layers.Conv2D(3, 3, padding='same', activation='sigmoid'),
        ... ])
        >>>
        >>> vqvae = VQVAEModel(
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     num_embeddings=512,
        ...     embedding_dim=64,
        ... )
        >>>
        >>> # Compile with optimizer only (loss is computed internally)
        >>> vqvae.compile(optimizer=keras.optimizers.Adam(1e-3))
        >>>
        >>> # Train on images
        >>> vqvae.fit(train_images, epochs=100, batch_size=64)
        >>>
        >>> # Reconstruct images
        >>> reconstructed = vqvae(test_images)
        >>>
        >>> # Get discrete codes for prior training
        >>> z_e = vqvae.encoder(test_images)
        >>> indices = vqvae.quantizer.get_codebook_indices(z_e)
        >>>
        >>> # Generate by sampling from prior and decoding
        >>> # (assumes you've trained a PixelCNN prior)
        >>> sampled_indices = prior.sample()
        >>> z_q = vqvae.quantizer.quantize_from_indices(sampled_indices)
        >>> generated = vqvae.decoder(z_q)

    Notes:
        - The model handles loss computation internally during training
        - Use separate encoder/decoder for flexible architectures
        - After training VQ-VAE, train a prior (PixelCNN, WaveNet) on discrete codes
        - The reconstruction loss uses MSE by default
        - Consider normalizing inputs to [0, 1] or [-1, 1]

    References:
        van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
        Neural Discrete Representation Learning. NeurIPS 2017.
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

        # Validate inputs
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

        # Store networks
        self.encoder = encoder
        self.decoder = decoder

        # Create quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            initializer=quantizer_initializer,
            use_ema=use_ema,
            ema_decay=ema_decay,
            name="vector_quantizer"
        )

        # Store configuration
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

        # Create metrics for tracking losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through VQ-VAE: encode, quantize, decode.

        Args:
            inputs: Input data to reconstruct.
            training: Whether in training mode. Affects quantizer EMA updates.

        Returns:
            Reconstructed outputs with same shape as inputs.
        """
        # Encode to continuous latents
        z_e = self.encoder(inputs, training=training)

        # Quantize to discrete latents
        z_q = self.quantizer(z_e, training=training)

        # Decode from quantized latents
        reconstructed = self.decoder(z_q, training=training)

        return reconstructed

    def train_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        """
        Custom training step that computes VQ-VAE losses.

        Args:
            data: Input data. Can be:
                - Single tensor: inputs (unsupervised)
                - Tuple: (inputs, targets) or (inputs, targets, sample_weight)

        Returns:
            Dictionary mapping metric names to their current values.
        """
        # Unpack data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            # Forward pass
            x_recon = self(x, training=True)

            # Compute reconstruction loss (MSE)
            reconstruction_loss = ops.mean((x - x_recon) ** 2)
            reconstruction_loss = (
                    self.reconstruction_loss_weight * reconstruction_loss
            )

            # Get VQ losses from quantizer
            vq_losses = self.quantizer.losses
            # Sum vq_losses if multiple (e.g. codebook + commitment)
            vq_loss = ops.sum(ops.stack(vq_losses))

            # Total loss
            total_loss = reconstruction_loss + vq_loss

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(vq_loss)

        # Return metrics
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vq_loss": self.vq_loss_tracker.result(),
        }

    def test_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, Any]:
        """
        Custom test step for evaluation.

        Args:
            data: Input data. Can be:
                - Single tensor: inputs (unsupervised)
                - Tuple: (inputs, targets) or (inputs, targets, sample_weight)

        Returns:
            Dictionary mapping metric names to their current values.
        """
        # Unpack data
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Forward pass
        x_recon = self(x, training=False)

        # Compute reconstruction loss
        reconstruction_loss = ops.mean((x - x_recon) ** 2)
        reconstruction_loss = self.reconstruction_loss_weight * reconstruction_loss

        # Get VQ losses from quantizer
        vq_losses = self.quantizer.losses
        vq_loss = ops.sum(ops.stack(vq_losses))

        # Total loss
        total_loss = reconstruction_loss + vq_loss

        # Update metrics
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
        """
        List of metrics tracked by the model.

        Returns:
            List of metric objects.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    def encode(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Encode inputs to continuous latent representations.

        Args:
            inputs: Input data.

        Returns:
            Continuous latent representations z_e(x).
        """
        return self.encoder(inputs, training=False)

    def quantize_latents(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        """
        Quantize continuous latents to discrete representations.

        Renamed from `quantize` to avoid collision with Keras quantization API.

        Args:
            latents: Continuous latent representations z_e.

        Returns:
            Quantized latent representations z_q.
        """
        return self.quantizer(latents, training=False)

    def decode(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        """
        Decode latent representations to reconstructed outputs.

        Args:
            latents: Quantized latent representations z_q.

        Returns:
            Reconstructed outputs.
        """
        return self.decoder(latents, training=False)

    def encode_to_indices(
            self,
            inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Encode inputs directly to discrete codebook indices.

        Useful for training autoregressive priors or compressing data.

        Args:
            inputs: Input data.

        Returns:
            Integer tensor of codebook indices.

        Example:
            >>> indices = vqvae.encode_to_indices(images)
            >>> # Train PixelCNN prior
            >>> prior.fit(indices, epochs=100)
        """
        z_e = self.encode(inputs)
        indices = self.quantizer.get_codebook_indices(z_e)
        return indices

    def decode_from_indices(
            self,
            indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Decode discrete codebook indices to reconstructed outputs.

        Useful for sampling from autoregressive priors.

        Args:
            indices: Integer tensor of codebook indices.

        Returns:
            Reconstructed outputs.

        Example:
            >>> # Sample from prior
            >>> sampled_indices = prior.sample(batch_size=16)
            >>> # Decode to images
            >>> generated = vqvae.decode_from_indices(sampled_indices)
        """
        z_q = self.quantizer.quantize_from_indices(indices)
        reconstructed = self.decode(z_q)
        return reconstructed

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters.
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
        """
        Create model from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New VQVAEModel instance.
        """
        # Deserialize encoder and decoder models
        encoder_config = config.pop("encoder")
        encoder = keras.saving.deserialize_keras_object(encoder_config)

        decoder_config = config.pop("decoder")
        decoder = keras.saving.deserialize_keras_object(decoder_config)

        # Pass reconstituted models to init
        return cls(encoder=encoder, decoder=decoder, **config)

# ---------------------------------------------------------------------