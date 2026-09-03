"""
ByteLatentTransformer is a tokenizer-free byte-level language model, defined by the
`ByteLatentTransformer` class and its `create_blt_model` factory. It runs as a funnel and a
fan-out instead of paying full transformer width on every byte: `LocalEncoder` pools each
patch's byte states into one vector, `GlobalTransformer` runs the expensive layers once per
patch instead of once per byte, and `LocalDecoder` cross-attends back to patch context to
produce per-byte logits. `DynamicPatcher` opens a new patch wherever a small entropy model
finds a byte surprising, so predictable runs collapse into long patches and surprising
regions get short ones. This implementation diverges from the paper: patches beyond
`max_patches` are kept by position rather than entropy rank, `entropy_threshold` is a fixed
constant rather than adaptive, and byte embeddings are plain learned embeddings rather than
hash n-grams. `train_step` masks padded positions (byte id 0) even when a caller compiles a
plain `sparse_categorical_crossentropy` loss. `generate` and its filtering helpers are
eager-only and do not trace under `tf.function`.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better Than
      Tokens. (https://arxiv.org/abs/2412.09871)
    - Yu et al., 2023. MEGABYTE: Predicting Million-byte Sequences with Multiscale
      Transformers. (https://arxiv.org/abs/2305.07185)
    - Xue et al., 2022. ByT5: Towards a Token-Free Future with Pre-trained
      Byte-to-Byte Models. (https://arxiv.org/abs/2105.13626)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.blt_blocks import (
    ByteTokenizer, EntropyModel,
    DynamicPatcher, LocalDecoder,
    LocalEncoder, GlobalTransformer
)
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.byte_latent_transformer.model")
class ByteLatentTransformer(keras.Model):
    """
    Byte Latent Transformer: hierarchical byte-level model with dynamic patching.

    Operates directly on UTF-8 bytes. Patch boundaries are computed from an entropy
    model unless `patch_ids` are supplied directly.

    Architecture:

    .. code-block:: text

        bytes [B, S] ──► EntropyModel ──► DynamicPatcher ──► patch_ids [B, S]
              │                                                    │
              ▼                                                    │
        LocalEncoder ◄──────────────────────────────────────────────
              │  patches [B, P, global_dim]
              ▼
        GlobalTransformer
              │  context [B, P, global_dim]
              ▼
        LocalDecoder (cross-attends to context, gathered by patch_ids)
              │
              ▼
        logits [B, S, vocab_size]

    `patch_ids` may instead arrive as an input, skipping the entropy model and the
    patcher; this is the intended path when boundaries are precomputed once.

    :param vocab_size: Size of the byte vocabulary (typically 256 + special tokens).
    :param local_dim: Hidden dimension for the local encoder and decoder.
    :param global_dim: Hidden dimension for the global transformer.
    :param num_local_layers: Number of transformer layers in local encoder/decoder.
    :param num_global_layers: Number of transformer layers in the global processor.
    :param num_heads_local: Number of attention heads in the local transformers.
    :param num_heads_global: Number of attention heads in the global transformer.
    :param max_sequence_length: Maximum sequence length in bytes.
    :param max_patches: Maximum number of patches per sequence.
    :param entropy_threshold: Nats threshold above which a byte opens a new patch.
    :param cross_attention_queries: Number of learnable queries for attention pooling.
    :param dropout_rate: Dropout rate applied throughout.
    :param patch_pooling_method: One of ``'attention'``, ``'mean'``, ``'max'``.
    :param entropy_model: Optional pre-built entropy model; a default one is built if
        omitted.
    :param kwargs: Additional ``keras.Model`` arguments.

    Input shape:
        2D tensor ``(batch_size, sequence_length)`` of byte tokens.

    Output shape:
        3D tensor ``(batch_size, sequence_length, vocab_size)`` of next-byte logits.

    Example:
        ```python
        model = ByteLatentTransformer.from_variant("base", vocab_size=260)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        text = model.generate(prompt="The future of language models", max_new_tokens=100)
        ```

    Note:
        Causality holds in five places: the entropy model, the local encoder, the
        global transformer (over the patch axis), the decoder's self-attention, and
        the decoder's cross-attention, which gathers the preceding patch's
        representation rather than the byte's own.
    """

    # Model variant configurations following ConvNeXtV2 pattern
    MODEL_VARIANTS = {
        "micro": {
            "local_dim": 256,
            "global_dim": 384,
            "num_local_layers": 3,
            "num_global_layers": 6,
            "num_heads_local": 4,
            "num_heads_global": 6,
            "max_patches": 128
        },
        "tiny": {
            "local_dim": 384,
            "global_dim": 512,
            "num_local_layers": 4,
            "num_global_layers": 8,
            "num_heads_local": 6,
            "num_heads_global": 8,
            "max_patches": 256
        },
        "small": {
            "local_dim": 512,
            "global_dim": 768,
            "num_local_layers": 6,
            "num_global_layers": 12,
            "num_heads_local": 8,
            "num_heads_global": 12,
            "max_patches": 512
        },
        "base": {
            "local_dim": 768,
            "global_dim": 1024,
            "num_local_layers": 8,
            "num_global_layers": 16,
            "num_heads_local": 12,
            "num_heads_global": 16,
            "max_patches": 768
        },
        "large": {
            "local_dim": 1024,
            "global_dim": 1536,
            "num_local_layers": 12,
            "num_global_layers": 24,
            "num_heads_local": 16,
            "num_heads_global": 24,
            "max_patches": 1024
        },
        "huge": {
            "local_dim": 1536,
            "global_dim": 2048,
            "num_local_layers": 16,
            "num_global_layers": 32,
            "num_heads_local": 24,
            "num_heads_global": 32,
            "max_patches": 1536
        }
    }

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            global_dim: int = 768,
            num_local_layers: int = 6,
            num_global_layers: int = 12,
            num_heads_local: int = 8,
            num_heads_global: int = 12,
            max_sequence_length: int = 2048,
            max_patches: int = 512,
            entropy_threshold: float = 1.5,
            cross_attention_queries: int = 4,
            dropout_rate: float = 0.1,
            patch_pooling_method: str = 'attention',
            entropy_model: Optional[keras.layers.Layer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_local_layers = num_local_layers
        self.num_global_layers = num_global_layers
        self.num_heads_local = num_heads_local
        self.num_heads_global = num_heads_global
        self.max_sequence_length = max_sequence_length
        self.max_patches = max_patches
        self.entropy_threshold = entropy_threshold
        self.cross_attention_queries = cross_attention_queries
        self.dropout_rate = dropout_rate
        self.patch_pooling_method = patch_pooling_method
        self._custom_entropy_model = entropy_model is not None

        # Validate configuration before constructing sub-layers
        self._validate_config()

        # DECISION plan-2026-08-14T183218-f4c612aa/D-018: no construction-time
        # degeneracy warning; a vocab-only threshold check warned on 100% of
        # shipped configs. See `DynamicPatcher.warn_if_segmentation_is_degenerate`.

        # Create all sub-layers in __init__ following modern Keras pattern
        self.tokenizer = ByteTokenizer(
            vocab_size=vocab_size,
            name='tokenizer'
        )

        # Create entropy model
        if entropy_model is None:
            self.entropy_model = EntropyModel(
                vocab_size=vocab_size,
                hidden_dim=256,
                num_layers=4,
                num_heads=8,
                max_seq_len=max_sequence_length,
                name='entropy_model'
            )
        else:
            self.entropy_model = entropy_model

        # Create patcher
        self.patcher = DynamicPatcher(
            entropy_threshold=entropy_threshold,
            max_patches=max_patches,
            name='patcher'
        )

        # Create main model components
        self.local_encoder = LocalEncoder(
            vocab_size=vocab_size,
            local_dim=local_dim,
            num_local_layers=num_local_layers,
            num_heads_local=num_heads_local,
            max_sequence_length=max_sequence_length,
            max_patches=max_patches,
            dropout_rate=dropout_rate,
            patch_pooling_method=patch_pooling_method,
            global_dim=global_dim,
            cross_attention_queries=cross_attention_queries,
            name='local_encoder'
        )

        self.global_transformer = GlobalTransformer(
            global_dim=global_dim,
            num_global_layers=num_global_layers,
            num_heads_global=num_heads_global,
            max_patches=max_patches,
            dropout_rate=dropout_rate,
            name='global_transformer'
        )

        self.local_decoder = LocalDecoder(
            vocab_size=vocab_size,
            local_dim=local_dim,
            global_dim=global_dim,
            num_local_layers=num_local_layers,
            num_heads_local=num_heads_local,
            max_sequence_length=max_sequence_length,
            dropout_rate=dropout_rate,
            name='local_decoder'
        )

    def _validate_config(self) -> None:
        """
        Validate constructor arguments.

        :raises ValueError: If any dimension/layer/head count is non-positive, if
            head counts do not divide their corresponding model dimensions, or if
            ``patch_pooling_method`` is not a recognized value.
        """
        positive_args = {
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_global_layers': self.num_global_layers,
            'num_heads_local': self.num_heads_local,
            'num_heads_global': self.num_heads_global,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'cross_attention_queries': self.cross_attention_queries,
        }
        for name, value in positive_args.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.local_dim % self.num_heads_local != 0:
            raise ValueError(
                f"local_dim ({self.local_dim}) must be divisible by "
                f"num_heads_local ({self.num_heads_local})"
            )
        if self.global_dim % self.num_heads_global != 0:
            raise ValueError(
                f"global_dim ({self.global_dim}) must be divisible by "
                f"num_heads_global ({self.num_heads_global})"
            )

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {self.dropout_rate}"
            )

        valid_pooling = {'attention', 'mean', 'max'}
        if self.patch_pooling_method not in valid_pooling:
            raise ValueError(
                f"patch_pooling_method must be one of {valid_pooling}, "
                f"got '{self.patch_pooling_method}'"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model and all sub-layers.

        :param input_shape: Shape ``(batch_size, sequence_length)``.
        """
        # Remember the shape so the model can be rebuilt on .keras deserialization
        self._build_input_shape = input_shape

        # Build sub-layers explicitly (skip if already built, e.g. pre-trained entropy model)
        if not self.tokenizer.built:
            self.tokenizer.build(input_shape)
        if not self.entropy_model.built:
            self.entropy_model.build(input_shape)
        if not self.patcher.built:
            self.patcher.build(input_shape[:-1] + (1,))

        # Build main components
        if not self.local_encoder.built:
            self.local_encoder.build(input_shape)

        # Compute patch representation shape for global transformer
        patch_shape = input_shape[:-1] + (self.max_patches, self.global_dim)
        if not self.global_transformer.built:
            self.global_transformer.build(patch_shape)

        # Build decoder with global context shape
        global_context_shape = patch_shape
        if not self.local_decoder.built:
            self.local_decoder.build([input_shape, global_context_shape, input_shape])

        logger.info(f"Built BLT model with input shape {input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the BLT forward pass.

        :param inputs: Either a byte token tensor ``(batch_size, seq_len)`` or a dict
            with ``'tokens'`` and optionally ``'patch_ids'``.
        :param training: Whether the call is in training mode.
        :return: Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            byte_tokens = inputs['tokens']
            patch_ids = inputs.get('patch_ids')
        else:
            byte_tokens = inputs
            patch_ids = None

        # If patch information not provided, compute it
        if patch_ids is None:
            # Compute entropy
            entropy_logits = self.entropy_model(byte_tokens, training=training)
            entropy = self.entropy_model.compute_entropy(entropy_logits)

            # Create dynamic patches
            patch_lengths = self.patcher(entropy, training=training)
            # Pass the STATIC sequence length; see the D-034 anchor on
            # `PatchingLayer.compute_patch_ids`.
            patch_ids = self.patcher.compute_patch_ids(
                patch_lengths, seq_len=ops.shape(byte_tokens)[1])

        # Encode bytes to patch representations
        patch_representations = self.local_encoder(
            byte_tokens, patch_ids, training=training
        )

        # Process patches through global transformer
        global_context = self.global_transformer(
            patch_representations, training=training
        )

        # Decode with global context to generate next-byte predictions
        logits = self.local_decoder(
            byte_tokens, global_context, patch_ids, training=training
        )

        return logits

    def train_step(self, data: Tuple[keras.KerasTensor, ...]) -> Dict[str, keras.KerasTensor]:
        """
        Run one training step with masked-loss handling.

        :param data: ``(x, y)`` or ``(x, y, sample_weight)``, input and target tokens.
        :return: Dict mapping metric names to their current values.
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        with tf.GradientTape() as tape:
            # Forward pass
            logits = self(x, training=True)

            # Compute loss with proper masking
            loss = self._compute_masked_loss(y, logits, sample_weight)
            # DECISION plan-2026-08-19T163559-499b6f0e/D-036: scale_loss must stay
            # inside the tape; omitting it divides the whole update by the fp16
            # loss scale (measured ratio 1.666e+04). See decisions.md.
            scaled_loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, logits, sample_weight=sample_weight)

        # Return metrics
        return {m.name: m.result() for m in self.metrics}

    def _compute_masked_loss(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            sample_weight: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Compute cross-entropy loss over non-padded (nonzero) tokens only.

        :param y_true: Target byte tokens.
        :param y_pred: Predicted logits.
        :param sample_weight: Optional per-example weights.
        :return: Scalar loss.
        """
        # Create mask for non-padded tokens (assuming 0 is pad token)
        mask = ops.cast(ops.not_equal(y_true, 0), dtype=y_pred.dtype)

        # Compute cross-entropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        # Apply mask
        loss = loss * mask

        # Apply sample weights if provided
        if sample_weight is not None:
            loss = loss * sample_weight

        # Return mean loss over non-padded tokens
        return ops.sum(loss) / ops.maximum(ops.sum(mask), 1.0)

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            temperature: float = 1.0,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            do_sample: bool = True
    ) -> str:
        """
        Generate text autoregressively, byte by byte. Eager-only.

        :param prompt: Input text prompt.
        :param max_new_tokens: Maximum number of new tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Optional nucleus-sampling threshold.
        :param top_k: Optional top-k sampling threshold.
        :param do_sample: Sample if True, else decode greedily.
        :return: Generated text, excluding the prompt.
        """
        # Convert prompt to byte tokens
        tokens = self.tokenizer.text_to_bytes(prompt, add_bos=True, add_eos=False)

        # Convert to tensor
        input_ids = ops.array([tokens], dtype='int32')

        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass with dynamic patching
            logits = self(input_ids, training=False)

            # Logits at the last position, shape (batch_size, vocab_size).
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Sample next token
            if do_sample:
                if top_k is not None:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                if top_p is not None:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)

                # Sample from distribution
                probs = keras.activations.softmax(next_token_logits, axis=-1)
                next_token = keras.random.categorical(ops.log(probs), num_samples=1)
            else:
                # Greedy decoding
                next_token = ops.argmax(next_token_logits, axis=-1, keepdims=True)

            # Append to sequence
            input_ids = ops.concatenate([input_ids, next_token], axis=1)

            # Check for end token
            if next_token[0, 0] == self.tokenizer.eos_id:
                break

        # Convert back to text
        generated_tokens = input_ids[0].numpy().tolist()
        generated_text = self.tokenizer.tokens_to_text(generated_tokens)

        # Remove prompt from generated text
        return generated_text[len(prompt):]

    def _top_k_filtering(self, logits: keras.KerasTensor, k: int) -> keras.KerasTensor:
        """Apply top-k filtering to logits."""
        # Get top-k values and indices
        top_k_logits, top_k_indices = ops.top_k(logits, k=k)

        # Create mask for top-k positions
        mask = ops.zeros_like(logits, dtype='bool')
        batch_size = ops.shape(logits)[0]

        for i in range(batch_size):
            for j in range(k):
                idx = top_k_indices[i, j]
                # Use slice_update for compatibility
                mask = ops.slice_update(mask, [i, idx], True)

        # Set non-top-k positions to negative infinity
        return ops.where(mask, logits, ops.full_like(logits, float('-inf')))

    def _top_p_filtering(self, logits: keras.KerasTensor, p: float) -> keras.KerasTensor:
        """Apply top-p (nucleus) filtering to logits."""
        # Sort logits in descending order
        sorted_logits, sorted_indices = ops.top_k(logits, k=ops.shape(logits)[-1])

        # Compute cumulative probabilities
        sorted_probs = keras.activations.softmax(sorted_logits, axis=-1)
        cumulative_probs = ops.cumsum(sorted_probs, axis=-1)

        # Create mask for positions to keep
        sorted_indices_to_remove = cumulative_probs > p
        # Keep at least one token
        sorted_indices_to_remove = ops.concatenate([
            ops.zeros_like(sorted_indices_to_remove[:, :1]),
            sorted_indices_to_remove[:, :-1]
        ], axis=-1)

        # Set filtered positions to negative infinity
        filtered_logits = ops.where(
            sorted_indices_to_remove,
            ops.full_like(sorted_logits, float('-inf')),
            sorted_logits
        )

        # Scatter back to original positions
        output_logits = ops.zeros_like(logits)
        batch_size = ops.shape(sorted_indices)[0]
        vocab_size = ops.shape(sorted_indices)[1]

        for i in range(batch_size):
            for j in range(vocab_size):
                idx = sorted_indices[i, j]
                value = filtered_logits[i, j]
                output_logits = ops.slice_update(
                    output_logits, [i, idx], value
                )

        return output_logits

    @classmethod
    def from_variant(
            cls,
            variant: str,
            vocab_size: int = 260,
            max_sequence_length: int = 2048,
            entropy_threshold: float = 1.5,
            **kwargs: Any
    ) -> 'ByteLatentTransformer':
        """
        Build a model from a named entry in ``MODEL_VARIANTS``.

        :param variant: One of ``"micro"``, ``"tiny"``, ``"small"``, ``"base"``,
            ``"large"``, ``"huge"``.
        :param vocab_size: Size of the byte vocabulary.
        :param max_sequence_length: Maximum sequence length in bytes.
        :param entropy_threshold: Nats threshold for dynamic patching.
        :param kwargs: Additional constructor arguments, overriding the variant.
        :return: A configured :class:`ByteLatentTransformer`.
        :raises ValueError: If ``variant`` is not in ``MODEL_VARIANTS``.

        Example:
            >>> # Micro model for experimentation
            >>> model = ByteLatentTransformer.from_variant("micro", vocab_size=260)
            >>>
            >>> # Base model for production
            >>> model = ByteLatentTransformer.from_variant("base", max_sequence_length=4096)
            >>>
            >>> # Large model with custom settings
            >>> model = ByteLatentTransformer.from_variant("large", dropout_rate=0.2)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()

        # Override with provided arguments
        config.update({
            'vocab_size': vocab_size,
            'max_sequence_length': max_sequence_length,
            'entropy_threshold': entropy_threshold,
            **kwargs
        })

        logger.info(f"Creating BLT-{variant.upper()} model")
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor configuration for serialization.

        :return: Config dict, including the entropy model when it was custom-built.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_global_layers': self.num_global_layers,
            'num_heads_local': self.num_heads_local,
            'num_heads_global': self.num_heads_global,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'entropy_threshold': self.entropy_threshold,
            'cross_attention_queries': self.cross_attention_queries,
            'dropout_rate': self.dropout_rate,
            'patch_pooling_method': self.patch_pooling_method,
            # DECISION plan-2026-08-18T140459-7991552f/D-030: serialize explicitly
            # rather than storing the live layer, which reloads as a plain dict
            # and crashes `build()` on the first custom entropy model. See decisions.md.
            'entropy_model': (
                keras.saving.serialize_keras_object(self.entropy_model)
                if self._custom_entropy_model else None
            ),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Return the build configuration so the model can be rebuilt on load.

        :returns: A dict carrying the stored ``input_shape``, or an empty dict if
            the model was never built.
        """
        if hasattr(self, "_build_input_shape") and self._build_input_shape is not None:
            return {"input_shape": list(self._build_input_shape)}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Rebuild the model (and all sub-layers) from a saved build configuration.

        :param config: Dict produced by :meth:`get_build_config`.
        """
        if "input_shape" in config and config["input_shape"] is not None:
            self.build(tuple(config["input_shape"]))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ByteLatentTransformer':
        """
        Reconstruct a model from a :meth:`get_config` dictionary.

        :param config: Configuration dictionary from :meth:`get_config`.
        :return: A new :class:`ByteLatentTransformer` instance.
        """
        config = dict(config)
        # DECISION plan-2026-08-18T140459-7991552f/D-030: mirrors the serialize
        # call in `get_config`; without it `build()` raises on a reloaded custom
        # entropy model. See decisions.md.
        if config.get('entropy_model') is not None:
            config['entropy_model'] = keras.saving.deserialize_keras_object(
                config['entropy_model']
            )
        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with BLT-specific information."""
        super().summary(**kwargs)

        # Print additional BLT information
        logger.info("BLT Configuration:")
        logger.info(f"  - Local dimension: {self.local_dim}")
        logger.info(f"  - Global dimension: {self.global_dim}")
        logger.info(f"  - Local layers: {self.num_local_layers}")
        logger.info(f"  - Global layers: {self.num_global_layers}")
        logger.info(f"  - Max sequence length: {self.max_sequence_length}")
        logger.info(f"  - Max patches: {self.max_patches}")
        logger.info(f"  - Entropy threshold: {self.entropy_threshold}")
        logger.info(f"  - Patch pooling: {self.patch_pooling_method}")


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def create_blt_model(
        variant: str = "base",
        vocab_size: int = 260,
        max_sequence_length: int = 2048,
        entropy_threshold: float = 1.5,
        **kwargs: Any
) -> ByteLatentTransformer:
    """
    Create a BLT model for the given variant.

    :param variant: One of ``"micro"``, ``"tiny"``, ``"small"``, ``"base"``,
        ``"large"``, ``"huge"``.
    :param vocab_size: Size of the byte vocabulary.
    :param max_sequence_length: Maximum sequence length in bytes.
    :param entropy_threshold: Nats threshold for dynamic patching.
    :param kwargs: Additional arguments passed to the constructor.
    :return: A configured :class:`ByteLatentTransformer`.

    Example:
        >>> # Create and compile base model
        >>> model = create_blt_model("base", compile_model=True)
        >>>
        >>> # Create large model for inference
        >>> model = create_blt_model("large", max_sequence_length=4096)
        >>>
        >>> # Create custom configured model
        >>> model = create_blt_model("tiny", dropout_rate=0.2, compile_model=True)
    """
    model = ByteLatentTransformer.from_variant(
        variant=variant,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        entropy_threshold=entropy_threshold,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------
