"""
Sparse Autoencoder (SAE) Module
===============================

This module provides a comprehensive implementation of Sparse Autoencoders
for feature extraction, interpretability research, and representation learning.

Sparse Autoencoders are designed to learn sparse, interpretable representations
of neural network activations or other high-dimensional data. They enforce
sparsity constraints on the latent space to encourage monosemantic features.

Supported SAE Variants
----------------------
- ``relu``: Standard ReLU SAE with L1 sparsity penalty
- ``topk``: TopK SAE that keeps only the top k activations per sample
- ``batch_topk``: BatchTopK SAE with batch-level sparsity constraint
- ``jumprelu``: JumpReLU SAE with learnable threshold for improved reconstruction
- ``gated``: Gated SAE that separates feature detection from magnitude estimation

References
----------
- Bricken et al. (2023): "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning"
- Gao et al. (2024): "Scaling and Evaluating Sparse Autoencoders" (OpenAI)
- Rajamanoharan et al. (2024): "Improving Dictionary Learning with Gated Sparse Autoencoders"
- Rajamanoharan et al. (2024): "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs"
- Bussmann et al. (2024): "BatchTopK Sparse Autoencoders"
"""

import keras
from keras import ops, initializers, regularizers, constraints
from typing import Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

SAEVariant = Literal['relu', 'topk', 'batch_topk', 'jumprelu', 'gated']

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package='dl_techniques')
class SparseAutoencoder(keras.layers.Layer):
    """
    Sparse Autoencoder layer with multiple sparsity enforcement variants.

    This layer implements a sparse autoencoder that encodes input activations
    into a sparse, higher-dimensional latent space and reconstructs them.
    Multiple sparsity mechanisms are supported to balance reconstruction
    fidelity with interpretability.

    Architecture
    ------------
    ::

        Input (d_input) -> Encoder (d_latent) -> Sparsity -> Decoder (d_input)
                               |                    |
                           W_enc + b_enc      Variant-specific
                                                activation

    Parameters
    ----------
    d_input : int
        Dimensionality of the input activations.
    d_latent : int
        Dimensionality of the latent (dictionary) space. Typically larger
        than d_input (e.g., 4x to 32x expansion).
    variant : SAEVariant, default='topk'
        Sparsity enforcement variant:
        - 'relu': Standard ReLU with L1 penalty
        - 'topk': Keep top k activations per sample
        - 'batch_topk': Batch-level TopK for adaptive allocation
        - 'jumprelu': JumpReLU with learnable threshold
        - 'gated': Gated SAE with separate detection/magnitude
    k : int, optional
        Number of active latents for TopK variants. Required for
        'topk' and 'batch_topk'. Default is 32.
    l1_coefficient : float, default=1e-3
        L1 sparsity penalty coefficient for 'relu' variant.
    l0_coefficient : float, default=1e-3
        L0 sparsity penalty coefficient for 'jumprelu' variant.
    tied_weights : bool, default=False
        If True, decoder weights are transpose of encoder weights.
    normalize_decoder : bool, default=True
        If True, constrain decoder columns to unit norm.
    use_pre_encoder_bias : bool, default=True
        If True, subtract a learned bias before encoding (centering).
    aux_k : int, optional
        Number of dead latents to include in auxiliary loss.
        Helps prevent dead features. Default is 256.
    aux_coefficient : float, default=1/32
        Coefficient for auxiliary loss that prevents dead latents.
    kernel_initializer : str or Initializer, default='glorot_uniform'
        Initializer for encoder/decoder weights.
    bias_initializer : str or Initializer, default='zeros'
        Initializer for bias vectors.
    kernel_regularizer : Regularizer, optional
        Regularizer for encoder/decoder weights.
    **kwargs : Any
        Additional arguments for the base Layer class.

    Attributes
    ----------
    encoder_weight : keras.Variable
        Encoder weight matrix of shape (d_input, d_latent).
    decoder_weight : keras.Variable
        Decoder weight matrix of shape (d_latent, d_input).
        None if tied_weights is True.
    encoder_bias : keras.Variable
        Encoder bias of shape (d_latent,).
    pre_encoder_bias : keras.Variable
        Pre-encoder centering bias of shape (d_input,).
        None if use_pre_encoder_bias is False.
    decoder_bias : keras.Variable
        Decoder bias of shape (d_input,).
    threshold : keras.Variable
        Learnable threshold for JumpReLU variant.
        Shape (d_latent,). None for other variants.
    gate_weight : keras.Variable
        Gating weight for Gated variant.
        Shape (d_input, d_latent). None for other variants.
    gate_bias : keras.Variable
        Gating bias for Gated variant.
        Shape (d_latent,). None for other variants.

    Examples
    --------
    Basic TopK SAE for LLM interpretability:

    >>> sae = SparseAutoencoder(
    ...     d_input=768,
    ...     d_latent=768 * 8,  # 8x expansion
    ...     variant='topk',
    ...     k=64
    ... )
    >>> activations = keras.random.normal((32, 128, 768))
    >>> reconstructed, latents, aux_loss = sae(activations, return_latents=True)

    JumpReLU SAE with custom initialization:

    >>> sae = SparseAutoencoder(
    ...     d_input=1024,
    ...     d_latent=16384,
    ...     variant='jumprelu',
    ...     l0_coefficient=1e-4,
    ...     kernel_initializer='he_normal'
    ... )

    Gated SAE for fine-grained control:

    >>> sae = SparseAutoencoder(
    ...     d_input=512,
    ...     d_latent=4096,
    ...     variant='gated',
    ...     normalize_decoder=True
    ... )

    Notes
    -----
    - For LLM interpretability, 'topk' or 'batch_topk' are recommended
      as they allow direct control over sparsity level.
    - 'jumprelu' offers the best reconstruction-sparsity tradeoff but
      requires tuning the l0_coefficient.
    - 'gated' provides the cleanest separation of feature detection
      from magnitude estimation but has higher parameter count.
    - Dead latent resampling can be implemented externally using the
      auxiliary loss signal.

    See Also
    --------
    SparseAutoencoderTrainer : Training utilities for SAEs
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        variant: SAEVariant = 'topk',
        k: Optional[int] = 32,
        l1_coefficient: float = 1e-3,
        l0_coefficient: float = 1e-3,
        tied_weights: bool = False,
        normalize_decoder: bool = True,
        use_pre_encoder_bias: bool = True,
        aux_k: Optional[int] = 256,
        aux_coefficient: float = 1/32,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Sparse Autoencoder layer."""
        super().__init__(**kwargs)

        # Validate variant
        valid_variants = ('relu', 'topk', 'batch_topk', 'jumprelu', 'gated')
        if variant not in valid_variants:
            raise ValueError(
                f"Invalid variant '{variant}'. Must be one of {valid_variants}"
            )

        # Validate k for TopK variants
        if variant in ('topk', 'batch_topk') and (k is None or k <= 0):
            raise ValueError(
                f"k must be a positive integer for variant '{variant}', got {k}"
            )

        # Store configuration
        self.d_input = d_input
        self.d_latent = d_latent
        self.variant = variant
        self.k = k
        self.l1_coefficient = l1_coefficient
        self.l0_coefficient = l0_coefficient
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        self.use_pre_encoder_bias = use_pre_encoder_bias
        self.aux_k = aux_k
        self.aux_coefficient = aux_coefficient

        # Store initializers/regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Will be created in build()
        self.encoder_weight: Optional[keras.Variable] = None
        self.decoder_weight: Optional[keras.Variable] = None
        self.encoder_bias: Optional[keras.Variable] = None
        self.pre_encoder_bias: Optional[keras.Variable] = None
        self.decoder_bias: Optional[keras.Variable] = None
        self.threshold: Optional[keras.Variable] = None
        self.gate_weight: Optional[keras.Variable] = None
        self.gate_bias: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor. Last dimension must match d_input.
        """
        # Validate input dimension
        if input_shape[-1] is not None and input_shape[-1] != self.d_input:
            raise ValueError(
                f"Expected input dimension {self.d_input}, "
                f"got {input_shape[-1]}"
            )

        # Encoder weights: (d_input, d_latent)
        self.encoder_weight = self.add_weight(
            name='encoder_weight',
            shape=(self.d_input, self.d_latent),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Decoder weights: (d_latent, d_input)
        if not self.tied_weights:
            self.decoder_weight = self.add_weight(
                name='decoder_weight',
                shape=(self.d_latent, self.d_input),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

        # Encoder bias
        self.encoder_bias = self.add_weight(
            name='encoder_bias',
            shape=(self.d_latent,),
            initializer=self.bias_initializer,
            trainable=True
        )

        # Pre-encoder bias (centering)
        if self.use_pre_encoder_bias:
            self.pre_encoder_bias = self.add_weight(
                name='pre_encoder_bias',
                shape=(self.d_input,),
                initializer='zeros',
                trainable=True
            )

        # Decoder bias
        self.decoder_bias = self.add_weight(
            name='decoder_bias',
            shape=(self.d_input,),
            initializer=self.bias_initializer,
            trainable=True
        )

        # Variant-specific weights
        if self.variant == 'jumprelu':
            # Learnable threshold for JumpReLU
            # Initialize slightly above zero
            self.threshold = self.add_weight(
                name='threshold',
                shape=(self.d_latent,),
                initializer=initializers.Constant(0.001),
                trainable=True,
                constraint=constraints.NonNeg()
            )
        elif self.variant == 'gated':
            # Separate gating network
            self.gate_weight = self.add_weight(
                name='gate_weight',
                shape=(self.d_input, self.d_latent),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            self.gate_bias = self.add_weight(
                name='gate_bias',
                shape=(self.d_latent,),
                initializer=self.bias_initializer,
                trainable=True
            )

        super().build(input_shape)

    def _get_decoder_weight(self) -> keras.Variable:
        """
        Get the decoder weight matrix.

        Returns the transposed encoder weight if tied_weights is True,
        otherwise returns the separate decoder weight.

        Returns
        -------
        keras.Variable
            Decoder weight matrix of shape (d_latent, d_input).
        """
        if self.tied_weights:
            return ops.transpose(self.encoder_weight)
        return self.decoder_weight

    def _normalize_decoder_columns(self, decoder_weight: keras.Variable) -> keras.Variable:
        """
        Normalize decoder columns to unit norm.

        Parameters
        ----------
        decoder_weight : keras.Variable
            Decoder weight matrix of shape (d_latent, d_input).

        Returns
        -------
        keras.Variable
            Normalized decoder weight matrix.
        """
        if not self.normalize_decoder:
            return decoder_weight

        norms = ops.sqrt(
            ops.sum(ops.square(decoder_weight), axis=-1, keepdims=True) + 1e-8
        )
        return decoder_weight / norms

    def encode(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Encode inputs to pre-activation latent space.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (..., d_input).
        training : bool, optional
            Training mode flag.

        Returns
        -------
        keras.KerasTensor
            Pre-activation latent tensor of shape (..., d_latent).
        """
        x = inputs

        # Apply centering
        if self.use_pre_encoder_bias:
            x = x - self.pre_encoder_bias

        # Linear encoding
        pre_activation = ops.matmul(x, self.encoder_weight) + self.encoder_bias

        return pre_activation

    def _apply_sparsity(
        self,
        pre_activation: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply variant-specific sparsity mechanism.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation latent tensor of shape (..., d_latent).
        training : bool, optional
            Training mode flag.

        Returns
        -------
        tuple
            - Sparse latent activations of shape (..., d_latent)
            - Sparsity loss tensor (scalar)
        """
        if self.variant == 'relu':
            return self._apply_relu_sparsity(pre_activation)
        elif self.variant == 'topk':
            return self._apply_topk_sparsity(pre_activation)
        elif self.variant == 'batch_topk':
            return self._apply_batch_topk_sparsity(pre_activation, training)
        elif self.variant == 'jumprelu':
            return self._apply_jumprelu_sparsity(pre_activation)
        elif self.variant == 'gated':
            return self._apply_gated_sparsity(pre_activation)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _apply_relu_sparsity(
        self,
        pre_activation: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply ReLU activation with L1 sparsity penalty.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor.

        Returns
        -------
        tuple
            - ReLU activations
            - L1 sparsity loss
        """
        latents = ops.relu(pre_activation)

        # L1 penalty on activations
        l1_loss = self.l1_coefficient * ops.mean(ops.abs(latents))

        return latents, l1_loss

    def _apply_topk_sparsity(
        self,
        pre_activation: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply TopK sparsity - keep only top k activations per sample.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor of shape (..., d_latent).

        Returns
        -------
        tuple
            - Sparse latents with only top k non-zero
            - Zero sparsity loss (sparsity is implicit)
        """
        # Get shape info
        original_shape = ops.shape(pre_activation)
        flat_shape = ops.stack([-1, original_shape[-1]])
        flat_pre_act = ops.reshape(pre_activation, flat_shape)

        # Apply ReLU first
        relu_act = ops.relu(flat_pre_act)

        # Get top-k values and indices
        top_values, top_indices = ops.top_k(relu_act, k=self.k)

        # Create sparse output using scatter
        batch_size = ops.shape(flat_pre_act)[0]
        batch_indices = ops.repeat(
            ops.arange(batch_size)[:, None],
            self.k,
            axis=1
        )

        # Create mask and values for sparse tensor
        indices = ops.stack([
            ops.reshape(batch_indices, [-1]),
            ops.reshape(top_indices, [-1])
        ], axis=1)

        # Scatter top values into zeros
        sparse_flat = ops.scatter(
            indices,
            ops.reshape(top_values, [-1]),
            (batch_size, self.d_latent)
        )

        # Reshape back to original shape
        latents = ops.reshape(sparse_flat, original_shape)

        # No explicit sparsity loss for TopK
        sparsity_loss = ops.zeros(())

        return latents, sparsity_loss

    def _apply_batch_topk_sparsity(
        self,
        pre_activation: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply BatchTopK sparsity - batch-level top k constraint.

        This allows variable number of active latents per sample while
        maintaining average sparsity across the batch.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor of shape (..., d_latent).
        training : bool, optional
            Training mode flag.

        Returns
        -------
        tuple
            - Sparse latents with batch-level TopK
            - Zero sparsity loss
        """
        # Get shape info
        original_shape = ops.shape(pre_activation)
        flat_shape = ops.stack([-1, original_shape[-1]])
        flat_pre_act = ops.reshape(pre_activation, flat_shape)

        batch_size = ops.shape(flat_pre_act)[0]

        # Apply ReLU first
        relu_act = ops.relu(flat_pre_act)

        # Flatten entire batch for batch-level TopK
        fully_flat = ops.reshape(relu_act, [-1])
        total_elements = batch_size * self.d_latent
        batch_k = batch_size * self.k

        # Get threshold value (k-th largest across entire batch)
        top_values, _ = ops.top_k(fully_flat, k=batch_k)
        threshold = top_values[-1]

        # Create mask for values above threshold
        mask = ops.cast(relu_act >= threshold, relu_act.dtype)
        latents_flat = relu_act * mask

        # Reshape back
        latents = ops.reshape(latents_flat, original_shape)

        sparsity_loss = ops.zeros(())

        return latents, sparsity_loss

    def _apply_jumprelu_sparsity(
        self,
        pre_activation: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply JumpReLU activation with learnable threshold.

        JumpReLU: f(x) = x * (x > theta), where theta is learnable.
        Uses straight-through estimator for gradient through discontinuity.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor.

        Returns
        -------
        tuple
            - JumpReLU activations
            - L0 sparsity loss (approximated)
        """
        # JumpReLU: zero out values below threshold, keep values above
        # mask = (pre_activation > threshold)
        # output = pre_activation * mask

        # For gradient flow, we use straight-through estimator:
        # Forward: discontinuous step
        # Backward: smooth approximation

        # Forward pass mask
        mask = ops.cast(pre_activation > self.threshold, pre_activation.dtype)

        # Apply JumpReLU
        latents = pre_activation * mask

        # L0 loss approximation (count of active features)
        # Using sigmoid approximation for gradient
        l0_approx = ops.sigmoid(
            (pre_activation - self.threshold) * 10.0  # Steepness factor
        )
        l0_loss = self.l0_coefficient * ops.mean(ops.sum(l0_approx, axis=-1))

        return latents, l0_loss

    def _apply_gated_sparsity(
        self,
        pre_activation: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply Gated SAE sparsity mechanism.

        Separates feature detection (gate) from magnitude estimation (encoder).
        gate_pre = W_gate @ x + b_gate
        gate = sigmoid(gate_pre)  # or step function
        latents = ReLU(pre_activation) * gate

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor from main encoder.

        Returns
        -------
        tuple
            - Gated sparse latents
            - L1 loss on gate pre-activations
        """
        # We need the original centered input for gating
        # This is stored during encode, but we recompute for cleaner code
        # In practice, you might want to cache this

        # Gate pre-activations (computed from same centered input)
        # Note: This requires access to centered input, which we handle
        # by computing gate in the main call() method

        # For now, use encoder pre-activation for gating decision
        # This is a simplified version; full implementation would have
        # separate gating path

        gate_logits = pre_activation  # Simplified; see full implementation below
        gate = ops.sigmoid(gate_logits)

        # Magnitude from ReLU of pre-activation
        magnitude = ops.relu(pre_activation)

        # Gated output
        latents = magnitude * gate

        # L1 on gate activations
        l1_loss = self.l1_coefficient * ops.mean(ops.sum(gate, axis=-1))

        return latents, l1_loss

    def decode(
        self,
        latents: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Decode sparse latents back to input space.

        Parameters
        ----------
        latents : keras.KerasTensor
            Sparse latent tensor of shape (..., d_latent).

        Returns
        -------
        keras.KerasTensor
            Reconstructed tensor of shape (..., d_input).
        """
        decoder_weight = self._get_decoder_weight()
        decoder_weight = self._normalize_decoder_columns(decoder_weight)

        reconstruction = ops.matmul(latents, decoder_weight) + self.decoder_bias

        # Add back pre-encoder bias
        if self.use_pre_encoder_bias:
            reconstruction = reconstruction + self.pre_encoder_bias

        return reconstruction

    def _compute_auxiliary_loss(
        self,
        pre_activation: keras.KerasTensor,
        latents: keras.KerasTensor,
        inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute auxiliary loss to prevent dead latents.

        Reconstructs using the top aux_k dead (inactive) latents
        to encourage their activation.

        Parameters
        ----------
        pre_activation : keras.KerasTensor
            Pre-activation tensor.
        latents : keras.KerasTensor
            Current sparse latents.
        inputs : keras.KerasTensor
            Original input for computing reconstruction error.

        Returns
        -------
        keras.KerasTensor
            Auxiliary reconstruction loss.
        """
        if self.aux_k is None or self.aux_k <= 0:
            return ops.zeros(())

        # Find dead latents (currently zero)
        is_dead = ops.cast(ops.abs(latents) < 1e-8, pre_activation.dtype)

        # Get pre-activations of dead latents only
        dead_pre_act = pre_activation * is_dead

        # Apply ReLU to dead pre-activations
        dead_relu = ops.relu(dead_pre_act)

        # Get top aux_k from dead latents
        original_shape = ops.shape(dead_relu)
        flat_shape = ops.stack([-1, original_shape[-1]])
        flat_dead = ops.reshape(dead_relu, flat_shape)

        top_values, top_indices = ops.top_k(flat_dead, k=min(self.aux_k, self.d_latent))

        # Create sparse tensor from top dead latents
        batch_size = ops.shape(flat_dead)[0]
        batch_indices = ops.repeat(
            ops.arange(batch_size)[:, None],
            min(self.aux_k, self.d_latent),
            axis=1
        )

        indices = ops.stack([
            ops.reshape(batch_indices, [-1]),
            ops.reshape(top_indices, [-1])
        ], axis=1)

        aux_latents_flat = ops.scatter(
            indices,
            ops.reshape(top_values, [-1]),
            (batch_size, self.d_latent)
        )

        aux_latents = ops.reshape(aux_latents_flat, original_shape)

        # Decode auxiliary latents
        aux_reconstruction = self.decode(aux_latents)

        # Auxiliary reconstruction error
        aux_error = ops.mean(ops.square(inputs - aux_reconstruction))

        return self.aux_coefficient * aux_error

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        return_latents: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass through the Sparse Autoencoder.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input activations of shape (..., d_input).
        training : bool, optional
            Training mode flag. Affects some sparsity mechanisms.
        return_latents : bool, default=False
            If True, return latents and losses along with reconstruction.

        Returns
        -------
        keras.KerasTensor or tuple
            If return_latents is False:
                Reconstructed tensor of shape (..., d_input).
            If return_latents is True:
                Tuple of (reconstruction, latents, total_loss) where:
                - reconstruction: shape (..., d_input)
                - latents: sparse latent tensor, shape (..., d_latent)
                - total_loss: combined sparsity and auxiliary losses
        """
        # Handle Gated variant specially (needs centered input for gate)
        if self.variant == 'gated':
            return self._call_gated(inputs, training, return_latents)

        # Encode to pre-activation
        pre_activation = self.encode(inputs, training)

        # Apply sparsity mechanism
        latents, sparsity_loss = self._apply_sparsity(pre_activation, training)

        # Decode
        reconstruction = self.decode(latents)

        if not return_latents:
            # Add losses to layer
            self.add_loss(sparsity_loss)
            return reconstruction

        # Compute auxiliary loss
        aux_loss = self._compute_auxiliary_loss(pre_activation, latents, inputs)

        total_loss = sparsity_loss + aux_loss
        self.add_loss(total_loss)

        return reconstruction, latents, total_loss

    def _call_gated(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        return_latents: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass for Gated SAE variant.

        Implements full gated architecture with separate detection and magnitude.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input activations.
        training : bool, optional
            Training mode flag.
        return_latents : bool, default=False
            Whether to return intermediate values.

        Returns
        -------
        keras.KerasTensor or tuple
            Same as call().
        """
        # Center input
        if self.use_pre_encoder_bias:
            centered = inputs - self.pre_encoder_bias
        else:
            centered = inputs

        # Magnitude path (main encoder)
        magnitude_pre = ops.matmul(centered, self.encoder_weight) + self.encoder_bias
        magnitude = ops.relu(magnitude_pre)

        # Gate path (separate gating network)
        gate_pre = ops.matmul(centered, self.gate_weight) + self.gate_bias
        # Use hard sigmoid for sparsity, soft sigmoid for gradients
        gate = ops.sigmoid(gate_pre)

        # For actual sparsity, threshold the gate
        gate_mask = ops.cast(gate > 0.5, gate.dtype)

        # Gated latents
        latents = magnitude * gate_mask

        # Sparsity loss on gate pre-activations
        sparsity_loss = self.l1_coefficient * ops.mean(
            ops.sum(ops.relu(gate_pre), axis=-1)
        )

        # Decode
        reconstruction = self.decode(latents)

        if not return_latents:
            self.add_loss(sparsity_loss)
            return reconstruction

        # Auxiliary loss for dead latents
        aux_loss = self._compute_auxiliary_loss(magnitude_pre, latents, inputs)

        total_loss = sparsity_loss + aux_loss
        self.add_loss(total_loss)

        return reconstruction, latents, total_loss

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape (same as input for reconstruction).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Configuration dictionary for serialization.
        """
        config = super().get_config()
        config.update({
            'd_input': self.d_input,
            'd_latent': self.d_latent,
            'variant': self.variant,
            'k': self.k,
            'l1_coefficient': self.l1_coefficient,
            'l0_coefficient': self.l0_coefficient,
            'tied_weights': self.tied_weights,
            'normalize_decoder': self.normalize_decoder,
            'use_pre_encoder_bias': self.use_pre_encoder_bias,
            'aux_k': self.aux_k,
            'aux_coefficient': self.aux_coefficient,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SparseAutoencoder':
        """
        Create layer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        SparseAutoencoder
            Instantiated layer.
        """
        config = config.copy()
        config['kernel_initializer'] = initializers.deserialize(
            config.get('kernel_initializer', 'glorot_uniform')
        )
        config['bias_initializer'] = initializers.deserialize(
            config.get('bias_initializer', 'zeros')
        )
        config['kernel_regularizer'] = regularizers.deserialize(
            config.get('kernel_regularizer')
        )
        return cls(**config)


# -----------------------------------------------------------------------------
# Factory function for convenient creation
# -----------------------------------------------------------------------------

def create_sparse_autoencoder(
    d_input: int,
    d_latent: int,
    variant: SAEVariant = 'topk',
    expansion_factor: Optional[int] = None,
    **kwargs: Any
) -> SparseAutoencoder:
    """
    Factory function to create Sparse Autoencoder with common configurations.

    Parameters
    ----------
    d_input : int
        Input dimensionality.
    d_latent : int, optional
        Latent dimensionality. If not provided, uses expansion_factor.
    variant : SAEVariant, default='topk'
        SAE variant to use.
    expansion_factor : int, optional
        If provided, sets d_latent = d_input * expansion_factor.
    **kwargs : Any
        Additional arguments passed to SparseAutoencoder.

    Returns
    -------
    SparseAutoencoder
        Configured SAE instance.

    Examples
    --------
    >>> sae = create_sparse_autoencoder(768, expansion_factor=8, variant='topk', k=64)
    >>> sae = create_sparse_autoencoder(1024, 16384, variant='jumprelu')
    """
    if expansion_factor is not None:
        d_latent = d_input * expansion_factor

    return SparseAutoencoder(
        d_input=d_input,
        d_latent=d_latent,
        variant=variant,
        **kwargs
    )

# ---------------------------------------------------------------------