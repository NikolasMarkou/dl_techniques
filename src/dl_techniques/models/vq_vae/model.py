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
from keras import ops, layers, initializers
from typing import Optional, Tuple, Dict, Any, Union


@keras.saving.register_keras_serializable()
class VectorQuantizer(layers.Layer):
    """
    Vector Quantization layer for discrete latent representations.

    This layer implements the vector quantization (VQ) mechanism that maps continuous
    encoder outputs to discrete codes by finding the nearest embedding vector in a
    learned codebook. It uses straight-through gradient estimation to enable
    end-to-end training.

    **Architecture**:
    ```
    Input: z_e(x) ∈ ℝ^(H×W×D)
         ↓
    ┌─────────────────────────────────────┐
    │  For each spatial position (h,w):   │
    │                                     │
    │  1. Compute distances to all codes: │
    │     d_j = ||z_e[h,w] - e_j||²       │
    │                                     │
    │  2. Find nearest:                   │
    │     k* = argmin_j d_j               │
    │                                     │
    │  3. Replace with embedding:         │
    │     z_q[h,w] = e_k*                 │
    └─────────────────────────────────────┘
         ↓
    Output: z_q(x) ∈ ℝ^(H×W×D)

    Gradient flow (straight-through estimator):
         Forward:  z_q = embedding_lookup(k*)
         Backward: ∇z_e = ∇z_q  (gradients copied)
    ```

    **Loss Components**:
    - Codebook loss: Moves embeddings toward encoder outputs
    - Commitment loss: Encourages encoder to commit to embeddings

    Args:
        num_embeddings: Size of the discrete codebook (K). Number of possible codes.
        embedding_dim: Dimensionality of each embedding vector (D).
        commitment_cost: Weight for commitment loss (β). Prevents encoder output
            from growing unbounded. Typically 0.25. Default: 0.25.
        initializer: Initializer for embedding vectors. Default: 'uniform'.
        use_ema: Whether to use exponential moving average for codebook updates
            instead of gradient-based updates. EMA is more stable but requires
            careful tuning. Default: False.
        ema_decay: Decay rate for EMA updates (γ). Only used if use_ema=True.
            Default: 0.99.
        epsilon: Small constant for numerical stability in EMA updates.
            Default: 1e-5.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tensor with shape: `(batch_size, height, width, embedding_dim)` for 2D data,
        or `(batch_size, sequence_length, embedding_dim)` for 1D data.
        The last dimension must match `embedding_dim`.

    Output shape:
        Same as input shape. The quantized representation z_q(x).

    Attributes:
        embeddings: The codebook of embedding vectors with shape
            `(num_embeddings, embedding_dim)`.
        ema_cluster_size: Cluster sizes for EMA updates. Only created if use_ema=True.
        ema_embeddings: EMA accumulator for embeddings. Only created if use_ema=True.

    Example:
        >>> # Quantize 2D feature maps
        >>> quantizer = VectorQuantizer(
        ...     num_embeddings=512,
        ...     embedding_dim=64,
        ...     commitment_cost=0.25
        ... )
        >>> z_e = ops.random.normal((8, 32, 32, 64))  # Encoder output
        >>> z_q = quantizer(z_e)  # Quantized output
        >>> print(z_q.shape)  # (8, 32, 32, 64)
        >>>
        >>> # Get quantization losses
        >>> losses = quantizer.losses
        >>> print(f"Codebook loss: {losses[0]}, Commitment loss: {losses[1]}")
        >>>
        >>> # Use with EMA updates
        >>> quantizer_ema = VectorQuantizer(
        ...     num_embeddings=512,
        ...     embedding_dim=64,
        ...     use_ema=True,
        ...     ema_decay=0.99
        ... )

    Notes:
        - The layer adds two losses during the call: codebook loss and commitment loss
        - With EMA updates, embeddings are updated via moving averages during training
        - The straight-through estimator copies gradients through the quantization
        - Codebook usage statistics are tracked for monitoring dead codes

    References:
        van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).
        Neural Discrete Representation Learning. NeurIPS 2017.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        initializer: Union[str, initializers.Initializer] = "uniform",
        use_ema: bool = False,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        **kwargs: Any
    ) -> None:
        """Initialize the VectorQuantizer layer."""
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
        if commitment_cost < 0:
            raise ValueError(
                f"commitment_cost must be non-negative, got {commitment_cost}"
            )
        if use_ema and not (0 < ema_decay < 1):
            raise ValueError(
                f"ema_decay must be in (0, 1), got {ema_decay}"
            )

        # Store configuration
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        # Store initializer for serialization
        if isinstance(initializer, str):
            self.initializer = initializers.get(initializer)
        else:
            self.initializer = initializer

        # Embeddings will be created in build()
        self.embeddings = None
        self.ema_cluster_size = None
        self.ema_embeddings = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the embedding codebook and EMA variables if needed.

        Args:
            input_shape: Shape of input tensor. Last dimension must match
                embedding_dim.

        Raises:
            ValueError: If last dimension doesn't match embedding_dim.
        """
        if input_shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Input last dimension {input_shape[-1]} must match "
                f"embedding_dim {self.embedding_dim}"
            )

        # Create embedding codebook
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=self.initializer,
            trainable=not self.use_ema,  # Not trainable if using EMA
        )

        # Create EMA variables if needed
        if self.use_ema:
            # Track cluster sizes for each embedding
            self.ema_cluster_size = self.add_weight(
                name="ema_cluster_size",
                shape=(self.num_embeddings,),
                initializer="zeros",
                trainable=False,
            )

            # Track sum of assigned encoder outputs for each embedding
            self.ema_embeddings = self.add_weight(
                name="ema_embeddings",
                shape=(self.num_embeddings, self.embedding_dim),
                initializer=self.initializer,
                trainable=False,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Quantize inputs using nearest neighbor lookup in embedding space.

        This method:
        1. Flattens spatial dimensions
        2. Computes distances to all embeddings
        3. Finds nearest embedding for each position
        4. Looks up corresponding embedding vectors
        5. Reshapes back to original spatial structure
        6. Applies straight-through estimator for gradients
        7. Computes and adds quantization losses

        Args:
            inputs: Encoder outputs z_e(x) with shape
                `(batch, height, width, embedding_dim)` or
                `(batch, sequence_length, embedding_dim)`.
            training: Whether in training mode. Affects EMA updates.

        Returns:
            Quantized outputs z_q(x) with same shape as inputs.
        """
        # Get input shape for later reshaping
        input_shape = ops.shape(inputs)

        # Flatten spatial dimensions: (batch, height, width, dim) -> (batch*height*width, dim)
        flat_inputs = ops.reshape(inputs, (-1, self.embedding_dim))

        # Compute L2 distances to all embeddings
        # ||z_e - e_j||² = ||z_e||² + ||e_j||² - 2*z_e·e_j
        distances = (
            ops.sum(flat_inputs ** 2, axis=1, keepdims=True)  # ||z_e||²
            + ops.sum(self.embeddings ** 2, axis=1)  # ||e_j||²
            - 2 * ops.matmul(flat_inputs, ops.transpose(self.embeddings))  # 2*z_e·e_j
        )

        # Find nearest embedding indices: k* = argmin_j ||z_e - e_j||²
        encoding_indices = ops.argmin(distances, axis=1)

        # Convert indices to one-hot for gathering
        encodings = ops.one_hot(
            encoding_indices,
            self.num_embeddings,
        )

        # Quantize: z_q = e_k*
        quantized = ops.matmul(encodings, self.embeddings)
        quantized = ops.reshape(quantized, input_shape)

        # Update embeddings with EMA if enabled and training
        if self.use_ema and training:
            self._update_ema(flat_inputs, encodings)

        # Compute losses
        # Codebook loss: ||sg[z_e(x)] - e||² (updates embeddings only)
        codebook_loss = ops.mean(
            (ops.stop_gradient(inputs) - quantized) ** 2
        )

        # Commitment loss: β||z_e(x) - sg[e]||² (updates encoder only)
        commitment_loss = self.commitment_cost * ops.mean(
            (inputs - ops.stop_gradient(quantized)) ** 2
        )

        # Add losses to layer
        self.add_loss(codebook_loss)
        self.add_loss(commitment_loss)

        # Straight-through estimator: copy gradients from quantized to inputs
        # Forward: use quantized values
        # Backward: gradients flow through as if quantized = inputs
        quantized = inputs + ops.stop_gradient(quantized - inputs)

        return quantized

    def _update_ema(
        self,
        flat_inputs: keras.KerasTensor,
        encodings: keras.KerasTensor
    ) -> None:
        """
        Update embeddings using exponential moving averages.

        EMA update equations:
            N_i^(t) = γ * N_i^(t-1) + (1 - γ) * n_i^(t)
            m_i^(t) = γ * m_i^(t-1) + (1 - γ) * Σ_j z_j (where cluster(z_j) = i)
            e_i^(t) = m_i^(t) / N_i^(t)

        where:
            N_i: cluster size for embedding i
            m_i: sum of encoder outputs assigned to embedding i
            γ: decay rate (ema_decay)
            n_i: number of vectors assigned to i in current batch

        Args:
            flat_inputs: Flattened encoder outputs with shape
                `(batch*spatial, embedding_dim)`.
            encodings: One-hot encoding of nearest embeddings with shape
                `(batch*spatial, num_embeddings)`.
        """
        # Count how many vectors were assigned to each embedding in this batch
        cluster_size = ops.sum(encodings, axis=0)  # (num_embeddings,)

        # Sum of all encoder outputs assigned to each embedding
        embed_sums = ops.matmul(
            ops.transpose(encodings),  # (num_embeddings, batch*spatial)
            flat_inputs  # (batch*spatial, embedding_dim)
        )  # (num_embeddings, embedding_dim)

        # Update cluster size with EMA
        # N^(t) = γ * N^(t-1) + (1 - γ) * n^(t)
        new_cluster_size = (
            self.ema_decay * self.ema_cluster_size
            + (1 - self.ema_decay) * cluster_size
        )
        self.ema_cluster_size.assign(new_cluster_size)

        # Update embedding sums with EMA
        # m^(t) = γ * m^(t-1) + (1 - γ) * sum^(t)
        new_ema_embeddings = (
            self.ema_decay * self.ema_embeddings
            + (1 - self.ema_decay) * embed_sums
        )
        self.ema_embeddings.assign(new_ema_embeddings)

        # Update embeddings: e^(t) = m^(t) / N^(t)
        # Add epsilon for numerical stability
        normalized_embeddings = (
            self.ema_embeddings
            / ops.reshape(self.ema_cluster_size + self.epsilon, (-1, 1))
        )
        self.embeddings.assign(normalized_embeddings)

    def get_codebook_indices(
        self,
        inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Get discrete codebook indices for given inputs.

        This is useful for:
        - Training autoregressive priors over discrete codes
        - Analyzing codebook usage
        - Compressing representations

        Args:
            inputs: Encoder outputs with shape
                `(batch, height, width, embedding_dim)` or
                `(batch, sequence_length, embedding_dim)`.

        Returns:
            Integer tensor of codebook indices with shape
                `(batch, height, width)` or `(batch, sequence_length)`.

        Example:
            >>> z_e = encoder(images)
            >>> indices = quantizer.get_codebook_indices(z_e)
            >>> # Train PixelCNN prior on indices
            >>> prior = PixelCNN(num_classes=quantizer.num_embeddings)
            >>> prior.fit(indices)
        """
        # Get spatial shape
        input_shape = ops.shape(inputs)
        spatial_shape = input_shape[:-1]

        # Flatten spatial dimensions
        flat_inputs = ops.reshape(inputs, (-1, self.embedding_dim))

        # Compute distances to all embeddings
        distances = (
            ops.sum(flat_inputs ** 2, axis=1, keepdims=True)
            + ops.sum(self.embeddings ** 2, axis=1)
            - 2 * ops.matmul(flat_inputs, ops.transpose(self.embeddings))
        )

        # Find nearest embedding indices
        encoding_indices = ops.argmin(distances, axis=1)

        # Reshape back to spatial dimensions
        encoding_indices = ops.reshape(encoding_indices, spatial_shape)

        return encoding_indices

    def quantize_from_indices(
        self,
        indices: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Convert discrete indices back to continuous embeddings.

        This is the inverse of get_codebook_indices() and is useful for:
        - Sampling from autoregressive priors
        - Decoding discrete representations

        Args:
            indices: Integer tensor of codebook indices with shape
                `(batch, height, width)` or `(batch, sequence_length)`.

        Returns:
            Quantized embeddings with shape
                `(batch, height, width, embedding_dim)` or
                `(batch, sequence_length, embedding_dim)`.

        Example:
            >>> # Sample indices from prior
            >>> sampled_indices = prior.sample(batch_size=4)
            >>> # Convert to embeddings
            >>> z_q = quantizer.quantize_from_indices(sampled_indices)
            >>> # Decode to images
            >>> generated_images = decoder(z_q)
        """
        # Get spatial shape
        spatial_shape = ops.shape(indices)

        # Flatten indices
        flat_indices = ops.reshape(indices, (-1,))

        # One-hot encode
        encodings = ops.one_hot(flat_indices, self.num_embeddings)

        # Look up embeddings
        quantized = ops.matmul(encodings, self.embeddings)

        # Reshape to include spatial dimensions and embedding dimension
        output_shape = ops.concatenate([spatial_shape, [self.embedding_dim]], axis=0)
        quantized = ops.reshape(quantized, output_shape)

        return quantized

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed to
            reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
            "initializer": initializers.serialize(self.initializer),
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "epsilon": self.epsilon,
        })
        return config


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
    def metrics(self) -> list:
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

    def quantize(self, latents: keras.KerasTensor) -> keras.KerasTensor:
        """
        Quantize continuous latents to discrete representations.

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

        Note:
            The encoder and decoder are saved as part of the model structure,
            not in the config dictionary.
        """
        config = super().get_config()
        config.update({
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

        Note:
            This is called automatically during model loading.
            The encoder and decoder are loaded separately.
        """
        # Note: encoder and decoder are handled by Keras automatically
        # They don't need to be in config
        return cls(**config)


# Import tensorflow for GradientTape
