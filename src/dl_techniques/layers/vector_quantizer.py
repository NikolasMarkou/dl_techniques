import keras
from keras import ops, layers, initializers
from typing import Optional, Tuple, Dict, Any, Union


# ---------------------------------------------------------------------

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
        >>> z_e = keras.random.normal((8, 32, 32, 64))  # Encoder output
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
        # Ensure layer is built if called directly without prior build
        if not self.built:
            self.build(ops.shape(inputs))

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
        # Ensure layer is built
        if not self.built:
            raise ValueError("Layer must be built before calling quantize_from_indices")

        # Get spatial shape
        spatial_shape = ops.shape(indices)

        # Flatten indices
        flat_indices = ops.reshape(indices, (-1,))

        # One-hot encode
        encodings = ops.one_hot(flat_indices, self.num_embeddings)

        # Look up embeddings
        quantized = ops.matmul(encodings, self.embeddings)

        # Reshape to include spatial dimensions and embedding dimension
        # Convert dim to tensor to avoid list/tuple issues in concatenate
        dim_tensor = ops.convert_to_tensor([self.embedding_dim], dtype="int32")
        # Ensure spatial_shape is compatible (usually int32/int64 depending on backend)
        spatial_shape = ops.cast(spatial_shape, "int32")

        output_shape = ops.concatenate([spatial_shape, dim_tensor], axis=0)
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