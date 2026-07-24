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

@keras.saving.register_keras_serializable()
class SparseAutoencoder(keras.layers.Layer):
    """
    Sparse Autoencoder layer with multiple sparsity enforcement variants.

    This layer implements a sparse autoencoder that encodes input activations
    into a sparse, higher-dimensional latent space and reconstructs them.
    Multiple sparsity mechanisms are supported (ReLU with L1 penalty, TopK,
    BatchTopK, JumpReLU with learnable threshold, and Gated SAE) to balance
    reconstruction fidelity with interpretability. An auxiliary loss mechanism
    helps prevent dead latents.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [..., d_input]           │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Pre-encoder Bias (optional)    │
        │  x = x - pre_encoder_bias       │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Encoder: W_enc @ x + b_enc     │
        │  → pre_activation [..., d_lat]  │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Sparsity (variant-specific)    │
        │  relu / topk / batch_topk /     │
        │  jumprelu / gated               │
        │  → sparse latents + loss        │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Decoder: W_dec @ latents       │
        │  + decoder_bias                 │
        │  + pre_encoder_bias (optional)  │
        │  → reconstruction [..., d_in]   │
        └─────────────────────────────────┘

    :param d_input: Dimensionality of the input activations.
    :type d_input: int
    :param d_latent: Dimensionality of the latent (dictionary) space. Typically
        larger than d_input (e.g., 4x to 32x expansion).
    :type d_latent: int
    :param variant: Sparsity enforcement variant. One of ``'relu'``, ``'topk'``,
        ``'batch_topk'``, ``'jumprelu'``, ``'gated'``. Defaults to ``'topk'``.
    :type variant: SAEVariant
    :param k: Number of active latents for TopK variants. Required for
        ``'topk'`` and ``'batch_topk'``. Defaults to 32.
    :type k: int or None
    :param l1_coefficient: L1 sparsity penalty coefficient for ``'relu'`` variant.
        Defaults to 1e-3.
    :type l1_coefficient: float
    :param l0_coefficient: L0 sparsity penalty coefficient for ``'jumprelu'`` variant.
        Defaults to 1e-3.
    :type l0_coefficient: float
    :param tied_weights: If True, decoder weights are transpose of encoder weights.
        Defaults to False.
    :type tied_weights: bool
    :param normalize_decoder: If True, constrain decoder columns to unit norm.
        Defaults to True.
    :type normalize_decoder: bool
    :param use_pre_encoder_bias: If True, subtract a learned bias before encoding.
        Defaults to True.
    :type use_pre_encoder_bias: bool
    :param aux_k: Number of dead latents to include in auxiliary loss. Defaults to 256.
    :type aux_k: int or None
    :param aux_coefficient: Coefficient for auxiliary loss. Defaults to 1/32.
    :type aux_coefficient: float
    :param kernel_initializer: Initializer for encoder/decoder weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for encoder/decoder weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional arguments for the base Layer class.
    :type kwargs: Any
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        variant: SAEVariant = 'topk',
        k: Optional[int] = 32,
        batch_topk_decay: float = 0.99,
        l1_coefficient: float = 1e-3,
        l0_coefficient: float = 1e-3,
        tied_weights: bool = False,
        normalize_decoder: bool = True,
        use_pre_encoder_bias: bool = True,
        aux_k: Optional[int] = 256,
        aux_coefficient: float = 1/32,
        dead_steps_threshold: int = 1000,
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

        # Validate dimensions and auxiliary-loss width
        if d_input <= 0:
            raise ValueError(
                f"d_input must be a positive integer, got {d_input}"
            )
        if d_latent <= 0:
            raise ValueError(
                f"d_latent must be a positive integer, got {d_latent}"
            )
        # NOTE: the aux_k > d_latent upper bound is intentionally NOT enforced
        # here — the default aux_k=256 exceeds small d_latent (e.g. the tests'
        # d_latent=20) and _compute_auxiliary_loss already clamps at use via
        # min(aux_k, d_latent). Only the genuine error (non-positive) is caught.
        if aux_k is not None and aux_k <= 0:
            raise ValueError(
                f"aux_k must be a positive integer, got {aux_k}"
            )

        # Store configuration
        self.d_input = d_input
        self.d_latent = d_latent
        self.variant = variant
        self.k = k
        self.batch_topk_decay = batch_topk_decay
        self.l1_coefficient = l1_coefficient
        self.l0_coefficient = l0_coefficient
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder
        self.use_pre_encoder_bias = use_pre_encoder_bias
        self.aux_k = aux_k
        self.aux_coefficient = aux_coefficient
        self.dead_steps_threshold = dead_steps_threshold

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
        self.batch_topk_threshold: Optional[keras.Variable] = None
        self.dead_steps: Optional[keras.Variable] = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights.

        :param input_shape: Shape of the input tensor. Last dimension must match d_input.
        :type input_shape: tuple
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
        elif self.variant == 'batch_topk':
            # Non-trainable EMA of the batch-relative top-k threshold. Updated
            # during training and used verbatim at inference (see D-002).
            self.batch_topk_threshold = self.add_weight(
                name='batch_topk_threshold',
                shape=(),
                initializer='zeros',
                trainable=False,
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

        # Non-trainable per-latent counter of consecutive training steps a
        # latent has not fired. Drives stateful dead-latent detection in the
        # AuxK auxiliary loss (persists across .keras round-trip; not in config).
        # The 'gated' variant uses the frozen-decoder L_aux (computed inline in
        # _call_gated) and never touches the AuxK dead-latent path, so it does
        # not allocate dead_steps.
        if (
            self.aux_k is not None
            and self.aux_k > 0
            and self.variant != 'gated'
        ):
            self.dead_steps = self.add_weight(
                name='dead_steps',
                shape=(self.d_latent,),
                initializer='zeros',
                trainable=False,
            )

        super().build(input_shape)

    def _get_decoder_weight(self) -> keras.Variable:
        """
        Get the decoder weight matrix.

        Returns the transposed encoder weight if tied_weights is True,
        otherwise returns the separate decoder weight.

        :return: Decoder weight matrix of shape ``(d_latent, d_input)``.
        :rtype: keras.Variable
        """
        if self.tied_weights:
            return ops.transpose(self.encoder_weight)
        return self.decoder_weight

    def _normalize_decoder_columns(self, decoder_weight: keras.Variable) -> keras.Variable:
        """
        Normalize each latent's decoder vector to unit L2 norm.

        The decoder weight has shape ``(d_latent, d_input)``, so ``axis=-1``
        normalizes each ROW ``decoder_weight[i, :]`` — i.e. the decoder /
        dictionary vector for latent ``i`` — to unit norm. This is the standard
        SAE dictionary-atom convention (each latent maps to a unit-norm direction
        in input space). Despite the method name, these are per-latent ROWS of
        this ``(d_latent, d_input)`` layout, not columns.

        :param decoder_weight: Decoder weight matrix of shape ``(d_latent, d_input)``.
        :type decoder_weight: keras.Variable
        :return: Decoder weight with each latent's row normalized to unit L2 norm.
        :rtype: keras.Variable
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

        :param inputs: Input tensor of shape ``(..., d_input)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: bool or None
        :return: Pre-activation latent tensor of shape ``(..., d_latent)``.
        :rtype: keras.KerasTensor
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

        :param pre_activation: Pre-activation latent tensor of shape ``(..., d_latent)``.
        :type pre_activation: keras.KerasTensor
        :param training: Training mode flag.
        :type training: bool or None
        :return: Tuple of (sparse latent activations, sparsity loss scalar).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if self.variant == 'relu':
            return self._apply_relu_sparsity(pre_activation)
        elif self.variant == 'topk':
            return self._apply_topk_sparsity(pre_activation)
        elif self.variant == 'batch_topk':
            return self._apply_batch_topk_sparsity(pre_activation, training)
        elif self.variant == 'jumprelu':
            return self._apply_jumprelu_sparsity(pre_activation)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _apply_relu_sparsity(
        self,
        pre_activation: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Apply ReLU activation with L1 sparsity penalty.

        :param pre_activation: Pre-activation tensor.
        :type pre_activation: keras.KerasTensor
        :return: Tuple of (ReLU activations, L1 sparsity loss).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
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
        Apply TopK sparsity -- keep only top k activations per sample.

        :param pre_activation: Pre-activation tensor of shape ``(..., d_latent)``.
        :type pre_activation: keras.KerasTensor
        :return: Tuple of (sparse latents with only top k non-zero, zero sparsity loss).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
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
        Apply BatchTopK sparsity -- batch-level top k constraint.

        Allows variable number of active latents per sample while
        maintaining average sparsity across the batch.

        Inference (``training=False``) masks with the STORED EMA threshold,
        which is seeded directly with the batch threshold on the first training
        step and smoothed thereafter via ``batch_topk_decay`` (avoids hundreds
        of steps of under-sparse inference while the EMA ramps up from zero).

        :param pre_activation: Pre-activation tensor of shape ``(..., d_latent)``.
        :type pre_activation: keras.KerasTensor
        :param training: Training mode flag.
        :type training: bool or None
        :return: Tuple of (sparse latents with batch-level TopK, zero sparsity loss).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Get shape info
        original_shape = ops.shape(pre_activation)
        flat_shape = ops.stack([-1, original_shape[-1]])
        flat_pre_act = ops.reshape(pre_activation, flat_shape)

        batch_size = ops.shape(flat_pre_act)[0]

        # Apply ReLU first
        relu_act = ops.relu(flat_pre_act)

        if training:
            # Training: compute the batch-relative threshold (k-th largest across
            # the entire batch) and use it for THIS pass's mask (paper behavior).
            fully_flat = ops.reshape(relu_act, [-1])
            batch_k = batch_size * self.k  # dynamic int under eager scope (A6)
            top_values, _ = ops.top_k(fully_flat, k=batch_k)
            batch_threshold = top_values[-1]
            # Cold-start: if the stored threshold is still 0 (first training
            # step), seed it with the batch threshold instead of EMA-blending up
            # from zero — otherwise inference stays under-sparse for hundreds of
            # steps while the EMA ramps. Thereafter smooth via `batch_topk_decay`.
            stored = self.batch_topk_threshold
            seed = ops.cast(stored <= 0.0, stored.dtype)  # 1.0 first step, else 0.0
            ema = (
                self.batch_topk_decay * stored
                + (1.0 - self.batch_topk_decay) * batch_threshold
            )
            self.batch_topk_threshold.assign(
                seed * batch_threshold + (1.0 - seed) * ema
            )
            threshold = batch_threshold
        else:
            # DECISION plan-2026-07-24T071243-9325b386/D-002: at inference use the STORED
            # EMA threshold, never recompute from the batch — recomputing makes a
            # fixed example's output depend on its batch-mates (invariant #1). See
            # decisions.md D-002.
            threshold = self.batch_topk_threshold

        # Mask requires meeting the threshold AND strict positivity, so a
        # threshold of 0 (early training / empty EMA) does not select every
        # non-negative latent (zero-threshold edge fix).
        mask = ops.cast(
            ops.logical_and(relu_act >= threshold, relu_act > 0),
            relu_act.dtype,
        )
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
        Apply JumpReLU activation with a learnable per-latent threshold.

        **Actual mechanism (what this code does):**

        - *Forward*: a hard gate ``f(x) = x * (x > theta)`` with a learnable,
          ``NonNeg``-constrained per-latent threshold ``theta`` (``self.threshold``).
        - *Backward*: this is ORDINARY autodiff through a ``cast`` + multiply.
          The mask ``cast(x > theta)`` is gradient-blocked, so the encoder
          receives gradient only through the currently-active (above-threshold)
          latents — exactly like a hard ReLU gate. There is NO custom-gradient
          mechanism here.
        - ``theta`` receives ZERO gradient from the reconstruction path (the
          comparison/cast blocks it); it is trained SOLELY by the separate
          sigmoid-surrogate L0 penalty term (``sigmoid((x - theta) * steepness)``)
          returned as the sparsity loss below.

        **Deviation from the paper.** This differs from Rajamanoharan et al.
        (2024) "Jumping Ahead", whose JumpReLU uses a kernel-density-estimator
        surrogate gradient for the threshold, giving ``theta`` a
        reconstruction-fidelity gradient in addition to the L0 signal. That
        surrogate-gradient mechanism is intentionally NOT implemented here: it
        would be a net-new custom-gradient path, out of scope for this reference
        layer. The existing L0-surrogate already trains ``theta``; the deviation
        is documented rather than silently misrepresented.

        :param pre_activation: Pre-activation tensor.
        :type pre_activation: keras.KerasTensor
        :return: Tuple of (JumpReLU activations, approximated L0 sparsity loss).
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Hard gate: keep values strictly above the learnable threshold, zero the
        # rest. The cast blocks gradient to `self.threshold` (see docstring) — the
        # encoder still receives gradient through the surviving (unmasked) latents.
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

    def decode(
        self,
        latents: keras.KerasTensor,
        freeze: bool = False
    ) -> keras.KerasTensor:
        """
        Decode sparse latents back to input space.

        :param latents: Sparse latent tensor of shape ``(..., d_latent)``.
        :type latents: keras.KerasTensor
        :param freeze: When True, the decoder weight and all reconstruction
            biases are wrapped in ``stop_gradient`` so no gradient flows to the
            decoder parameters through this path (used by the Gated ``L_aux``,
            which must isolate the reconstruction gradient to the gate encoder).
            The default (``False``) path is behaviorally identical to a plain
            decode. Defaults to False.
        :type freeze: bool
        :return: Reconstructed tensor of shape ``(..., d_input)``.
        :rtype: keras.KerasTensor
        """
        decoder_weight = self._get_decoder_weight()
        decoder_weight = self._normalize_decoder_columns(decoder_weight)
        if freeze:
            decoder_weight = ops.stop_gradient(decoder_weight)

        decoder_bias = (
            ops.stop_gradient(self.decoder_bias) if freeze else self.decoder_bias
        )
        reconstruction = ops.matmul(latents, decoder_weight) + decoder_bias

        # Add back pre-encoder bias
        if self.use_pre_encoder_bias:
            pre_bias = (
                ops.stop_gradient(self.pre_encoder_bias)
                if freeze else self.pre_encoder_bias
            )
            reconstruction = reconstruction + pre_bias

        return reconstruction

    def _compute_auxiliary_loss(
        self,
        pre_activation: keras.KerasTensor,
        latents: keras.KerasTensor,
        inputs: keras.KerasTensor,
        main_reconstruction: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute auxiliary loss to prevent dead latents (OpenAI AuxK).

        Reconstructs the *residual* the live latents failed to explain using
        the top ``aux_k`` dead latents, encouraging dead latents to specialize
        on what the main reconstruction misses. "Dead" is determined by a
        stateful per-latent counter (``dead_steps``): a latent that has not
        fired for more than ``dead_steps_threshold`` training steps is dead.

        :param pre_activation: Pre-activation tensor of shape ``(..., d_latent)``.
        :type pre_activation: keras.KerasTensor
        :param latents: Current sparse latents of shape ``(..., d_latent)``.
        :type latents: keras.KerasTensor
        :param inputs: Original input of shape ``(..., d_input)``.
        :type inputs: keras.KerasTensor
        :param main_reconstruction: Reconstruction from the main (live) latents,
            of shape ``(..., d_input)``.
        :type main_reconstruction: keras.KerasTensor
        :param training: Training mode flag. When truthy, the ``dead_steps``
            counter is updated in place.
        :type training: bool or None
        :return: Auxiliary reconstruction loss.
        :rtype: keras.KerasTensor
        """
        if self.aux_k is None or self.aux_k <= 0:
            return ops.zeros(())

        # DECISION plan-2026-07-24T071243-9325b386/D-003: target the main-reconstruction residual, NOT the raw input (OpenAI AuxK) — dead latents specialize on what live latents miss. See decisions.md D-003.
        residual = ops.stop_gradient(inputs - main_reconstruction)

        # Stateful dead-latent detection: which latents fired THIS batch,
        # reduced over all leading (batch) axes to shape (d_latent,).
        flat_latents = ops.reshape(latents, (-1, self.d_latent))
        fired = ops.max(
            ops.cast(ops.abs(flat_latents) > 1e-8, self.dead_steps.dtype),
            axis=0,
        )
        if training:
            new_dead_steps = ops.where(
                fired > 0,
                ops.zeros_like(self.dead_steps),
                self.dead_steps + 1.0,
            )
            self.dead_steps.assign(new_dead_steps)

        # A latent is dead if it has not fired for more than the threshold.
        is_dead_vec = ops.cast(
            self.dead_steps > self.dead_steps_threshold, pre_activation.dtype
        )

        # Get pre-activations of dead latents only (broadcast over batch axes)
        dead_pre_act = pre_activation * is_dead_vec

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

        # Auxiliary reconstruction error against the residual (AuxK)
        aux_error = ops.mean(ops.square(residual - aux_reconstruction))

        return self.aux_coefficient * aux_error

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        return_latents: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass through the Sparse Autoencoder.

        :param inputs: Input activations of shape ``(..., d_input)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag. Affects some sparsity mechanisms.
        :type training: bool or None
        :param return_latents: If True, return latents and losses along with
            reconstruction. Defaults to False.
        :type return_latents: bool
        :return: Reconstructed tensor if return_latents is False, otherwise
            tuple of (reconstruction, latents, total_loss).
        :rtype: keras.KerasTensor or tuple

        **Return-shape contract.** With ``return_latents=False`` (the default)
        this returns *only* the ``reconstruction`` tensor. With
        ``return_latents=True`` it returns the tuple
        ``(reconstruction, latents, total_loss)``, where ``total_loss`` is the
        SAME scalar (``sparsity_loss + aux_loss``) that has already been
        registered on the layer via ``self.add_loss(...)`` on this same call —
        it is exposed purely for inspection/debugging. The auxiliary + sparsity
        training signal is added to ``self.losses`` on EVERY forward pass,
        regardless of ``return_latents``; the flag controls only the return
        shape, never whether the model is trained. Callers using stock
        ``keras`` ``fit()`` should therefore rely on ``model.losses`` (which
        already includes this scalar) and must NOT also add the returned
        ``total_loss`` to their own loss, or the aux/sparsity term is
        double-counted. The return type deliberately varies by flag — this is
        the idiomatic Keras "add_loss + optional aux outputs" pattern, not an
        inconsistency.
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

        # Compute auxiliary loss unconditionally — it is a TRAINING signal, not
        # an output-format concern. _compute_auxiliary_loss early-returns
        # ops.zeros(()) when aux_k is disabled, so this is cheap when unused.
        aux_loss = self._compute_auxiliary_loss(
            pre_activation, latents, inputs, reconstruction, training
        )

        # DECISION plan-2026-07-24T071243-9325b386/D-005: add the aux+sparsity loss to
        # self.losses on EVERY forward pass, independent of return_latents.
        # Gating add_loss on an output-format flag (as before) meant the
        # dead-latent aux only trained the model when the caller happened to
        # request the tuple — a bug. return_latents controls ONLY the return
        # shape. The returned total_loss is the SAME scalar registered here
        # (inspection only); callers under fit() rely on model.losses and must
        # NOT re-sum it, or it double-counts. Do NOT move add_loss back inside
        # the return_latents branch. See decisions.md D-005.
        total_loss = sparsity_loss + aux_loss
        self.add_loss(total_loss)

        if not return_latents:
            return reconstruction

        return reconstruction, latents, total_loss

    def _call_gated(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        return_latents: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass for Gated SAE variant.

        Implements the gated architecture with separate detection (gate) and
        magnitude encoders. The gate encoder receives a reconstruction-quality
        gradient via the paper's frozen-decoder auxiliary loss ``L_aux``
        (Rajamanoharan et al. 2024): ``relu(gate_pre)`` is routed through a
        decoder whose weight/biases are ``stop_gradient``-frozen and compared to
        the raw input, so ``gate_weight``/``gate_bias`` learn whether opening a
        gate helps reconstruction (the hard Heaviside gate mask itself blocks
        that gradient). Deviation from the paper: the magnitude and gate encoders
        are kept UNTIED (no ``W_mag = exp(r) * W_gate`` weight-tying) — out of
        scope; documented as a deliberate simplification.

        :param inputs: Input activations.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: bool or None
        :param return_latents: Whether to return intermediate values. Defaults to False.
        :type return_latents: bool
        :return: Same as ``call()``.
        :rtype: keras.KerasTensor or tuple
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

        # Gated SAE auxiliary loss (Rajamanoharan et al. 2024):
        # route the gate's ReLU pre-activations through a FROZEN decoder and
        # reconstruct the input, giving the gate encoder a fidelity gradient.
        # DECISION plan-2026-07-24T071243-9325b386/D-004: the gate encoder needs a
        # reconstruction gradient that the hard gate-mask (cast(sigmoid>0.5))
        # blocks; freezing the decoder (stop_gradient on weight+biases) isolates
        # that signal to gate_weight/gate_bias WITHOUT moving the decoder. Do NOT
        # use the AuxK dead-latent residual here (that is the non-gated variants'
        # loss) and do NOT drop the freeze (a live decoder would let this term
        # train the decoder instead of the gate). Encoder weight-tying
        # (W_mag = exp(r)*W_gate) is deliberately NOT implemented (out of scope;
        # untied encoders are a documented deviation). See decisions.md D-004.
        gate_relu = ops.relu(gate_pre)
        aux_reconstruction = self.decode(gate_relu, freeze=True)
        aux_loss = self.aux_coefficient * ops.mean(
            ops.square(inputs - aux_reconstruction)
        )

        # Add aux+sparsity to self.losses unconditionally (see call() D-005):
        # the L_aux training signal must not depend on return_latents. Exactly
        # one add_loss per forward pass; the returned total_loss is the SAME
        # scalar (inspection only — do not re-sum under fit()).
        total_loss = sparsity_loss + aux_loss
        self.add_loss(total_loss)

        if not return_latents:
            return reconstruction

        return reconstruction, latents, total_loss

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Output shape (same as input for reconstruction).
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.

        :return: Configuration dictionary for serialization.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'd_input': self.d_input,
            'd_latent': self.d_latent,
            'variant': self.variant,
            'k': self.k,
            'batch_topk_decay': self.batch_topk_decay,
            'l1_coefficient': self.l1_coefficient,
            'l0_coefficient': self.l0_coefficient,
            'tied_weights': self.tied_weights,
            'normalize_decoder': self.normalize_decoder,
            'use_pre_encoder_bias': self.use_pre_encoder_bias,
            'aux_k': self.aux_k,
            'aux_coefficient': self.aux_coefficient,
            'dead_steps_threshold': self.dead_steps_threshold,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SparseAutoencoder':
        """
        Create layer from configuration dictionary.

        :param config: Configuration dictionary.
        :type config: dict
        :return: Instantiated layer.
        :rtype: SparseAutoencoder
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

    :param d_input: Input dimensionality.
    :type d_input: int
    :param d_latent: Latent dimensionality. If not provided, uses expansion_factor.
    :type d_latent: int
    :param variant: SAE variant to use. Defaults to ``'topk'``.
    :type variant: SAEVariant
    :param expansion_factor: If provided, sets ``d_latent = d_input * expansion_factor``.
    :type expansion_factor: int or None
    :param kwargs: Additional arguments passed to SparseAutoencoder.
    :type kwargs: Any
    :return: Configured SAE instance.
    :rtype: SparseAutoencoder
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