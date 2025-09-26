"""
Byte Latent Transformer (BLT): Patches Scale Better Than Tokens

This module implements the Byte Latent Transformer (BLT), a revolutionary byte-level
large language model architecture that, for the first time, matches tokenization-based
LLM performance at scale while achieving significant improvements in inference efficiency
and robustness.

Architecture Overview
====================

BLT introduces a paradigm shift from fixed-vocabulary tokenization to dynamic, learnable
byte grouping through three core innovations:

1. **Dynamic Entropy-Based Patching**: Instead of static tokens, BLT segments bytes into
   patches based on the entropy of next-byte predictions, allocating more compute where
   data complexity demands it. This allows for contextual groupings with uniform
   information density.

2. **Hierarchical Processing Architecture**:
   - **Local Encoder**: Lightweight transformer that efficiently maps byte sequences
     into expressive patch representations using cross-attention pooling
   - **Global Latent Transformer**: Computationally intensive autoregressive model
     operating on patch representations (consumes bulk of FLOPs)
   - **Local Decoder**: Lightweight transformer decoding patch representations back
     to raw bytes using cross-attention mechanisms

3. **Enhanced Byte Representations**: Incorporates hash n-gram embeddings (n=3-8) to
   capture contextual byte patterns and improve representation quality.

Based on "Byte Latent Transformer: Patches Scale Better Than Tokens"
by Pagnoni et al. (2024), arXiv:2412.09871v1 [cs.CL] 13 Dec 2024
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


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ByteLatentTransformer(keras.Model):
    """
    Complete Byte Latent Transformer model with hierarchical processing architecture.

    This model implements the full BLT architecture with dynamic patching,
    hierarchical processing, and autoregressive generation capabilities.
    It operates directly on UTF-8 bytes for language-agnostic processing.

    **Architecture**:
    ```
    Input(bytes) → TokenizerComponent → EntropyModel → DynamicPatcher
                    ↓
    LocalEncoder → GlobalTransformer → LocalDecoder → Output(logits)
    ```

    **Key Features**:
    - Dynamic entropy-based patching for adaptive compute allocation
    - Hierarchical processing with local and global attention mechanisms
    - Byte-level processing for language-agnostic capabilities
    - Cross-attention between patch and byte representations
    - Autoregressive generation with sampling strategies

    **Performance Characteristics**:
    - Up to 50% reduction in inference FLOPs vs tokenization-based models
    - Superior robustness to noise and character-level corruption
    - Enhanced multilingual performance, especially low-resource languages
    - Matches Llama 3 performance up to 8B parameters

    Args:
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        local_dim: Hidden dimension for local encoder/decoder.
        global_dim: Hidden dimension for global transformer.
        num_local_layers: Number of transformer layers in local encoder/decoder.
        num_global_layers: Number of transformer layers in global processor.
        num_heads_local: Number of attention heads for local transformers.
        num_heads_global: Number of attention heads for global transformer.
        max_sequence_length: Maximum sequence length in bytes.
        max_patches: Maximum number of patches per sequence.
        entropy_threshold: Threshold for dynamic patching decisions.
        cross_attention_queries: Number of queries for patch representation.
        dropout_rate: Dropout rate for all layers.
        patch_pooling_method: Method for patch pooling ('max', 'mean', 'attention').
        entropy_model: Optional pre-trained entropy model for dynamic patching.
        **kwargs: Additional model arguments.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)` containing byte tokens.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)` containing
        next-byte prediction logits.

    Example:
        ```python
        # Create BLT model for text generation
        model = ByteLatentTransformer.from_variant(
            "base",
            vocab_size=260,
            max_sequence_length=2048
        )

        # Compile for training
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate text
        generated_text = model.generate(
            prompt="The future of language models",
            max_new_tokens=100,
            temperature=0.8
        )
        ```

    Note:
        The model automatically handles dynamic patching during inference.
        For training, provide preprocessed patch information for efficiency,
        or let the model compute patches dynamically.
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model and all sub-layers.

        Args:
            input_shape: Shape of input tensor (batch_size, sequence_length).
        """
        # Build sub-layers explicitly for serialization robustness
        self.tokenizer.build(input_shape)
        self.entropy_model.build(input_shape)
        self.patcher.build(input_shape[:-1] + (1,))  # Entropy has 1 feature per token

        # Build main components
        self.local_encoder.build(input_shape)

        # Compute patch representation shape for global transformer
        patch_shape = input_shape[:-1] + (self.max_patches, self.global_dim)
        self.global_transformer.build(patch_shape)

        # Build decoder with global context shape
        global_context_shape = patch_shape
        self.local_decoder.build([input_shape, global_context_shape, input_shape])

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(f"Built BLT model with input shape {input_shape}")

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the BLT model.

        Args:
            inputs: Either byte token tensor of shape (batch_size, seq_len) or
                   dictionary with 'tokens' and optionally 'patch_lengths', 'patch_ids'.
            training: Whether in training mode.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
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
            patch_ids = self.patcher.compute_patch_ids(patch_lengths)

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
        Custom training step for BLT.

        Args:
            data: Tuple of (x, y) where x is input tokens and y is target tokens.

        Returns:
            Dictionary of metrics.
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

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

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
        Compute loss with proper masking for padded tokens.

        Args:
            y_true: Target tokens.
            y_pred: Predicted logits.
            sample_weight: Optional sample weights.

        Returns:
            Scalar loss value.
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
        Generate text autoregressively using the BLT model.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling threshold.
            top_k: Top-k sampling threshold.
            do_sample: Whether to use sampling or greedy decoding.

        Returns:
            Generated text string.
        """
        # Convert prompt to byte tokens
        tokens = self.tokenizer.text_to_bytes(prompt, add_bos=True, add_eos=False)

        # Convert to tensor
        input_ids = ops.array([tokens], dtype='int32')

        # Generation loop
        for _ in range(max_new_tokens):
            # Forward pass with dynamic patching
            logits = self(input_ids, training=False)

            # Get next token logits
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

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
        Create a BLT model from a predefined variant configuration.

        This method provides convenient access to well-tested model configurations
        that have been optimized for different computational budgets and use cases.

        Args:
            variant: String, one of "micro", "tiny", "small", "base", "large", "huge".
            vocab_size: Size of byte vocabulary (typically 256 + special tokens).
            max_sequence_length: Maximum sequence length in bytes.
            entropy_threshold: Threshold for dynamic patching decisions.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ByteLatentTransformer model instance.

        Raises:
            ValueError: If variant is not recognized.

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
        Get model configuration for serialization.

        Returns:
            Configuration dictionary containing all constructor parameters.
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
            'entropy_model': None if not self._custom_entropy_model else self.entropy_model,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ByteLatentTransformer':
        """
        Create model from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            ByteLatentTransformer model instance.
        """
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
# Convenience functions following ConvNeXtV2 pattern
# ---------------------------------------------------------------------

def create_blt_model(
        variant: str = "base",
        vocab_size: int = 260,
        max_sequence_length: int = 2048,
        entropy_threshold: float = 1.5,
        compile_model: bool = False,
        optimizer: str = 'adam',
        learning_rate: float = 1e-4,
        **kwargs: Any
) -> ByteLatentTransformer:
    """
    Create a BLT model with the specified variant and configuration.

    This is the primary convenience function for creating BLT models with
    sensible defaults and optional compilation.

    Args:
        variant: String, model variant ("micro", "tiny", "small", "base", "large", "huge").
        vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        max_sequence_length: Maximum sequence length in bytes.
        entropy_threshold: Threshold for dynamic patching decisions.
        compile_model: Whether to compile the model for training.
        optimizer: Optimizer to use if compiling.
        learning_rate: Learning rate if compiling.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Configured BLT model, optionally compiled.

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

    if compile_model:
        model.compile(
            optimizer=keras.optimizers.get({
                'class_name': optimizer,
                'config': {'learning_rate': learning_rate}
            }),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Compiled BLT-{variant.upper()} model for training")

    return model

# ---------------------------------------------------------------------

def train_entropy_model(
        vocab_size: int = 260,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dataset: Optional[Any] = None,
        epochs: int = 10,
        batch_size: int = 32,
        **kwargs: Any
) -> EntropyModel:
    """
    Train a separate entropy model for dynamic patching in BLT.

    This function creates and trains a lightweight causal transformer
    to serve as the entropy model for BLT's dynamic patching mechanism.
    The entropy model learns to predict next-byte probabilities and
    provides entropy estimates for patch boundary detection.

    Args:
        vocab_size: Size of byte vocabulary.
        hidden_dim: Hidden dimension of the entropy model.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        dataset: Training dataset yielding byte token sequences.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        **kwargs: Additional arguments for model creation.

    Returns:
        Trained entropy model ready for use in BLT.

    Example:
        >>> # Train entropy model on dataset
        >>> entropy_model = train_entropy_model(
        ...     dataset=byte_dataset,
        ...     epochs=5,
        ...     batch_size=16
        ... )
        >>>
        >>> # Use in BLT model
        >>> blt_model = create_blt_model("base", entropy_model=entropy_model)
    """
    entropy_model = EntropyModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        **kwargs
    )

    # Compile for language modeling
    entropy_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train if dataset provided
    if dataset is not None:
        logger.info("Training entropy model for BLT dynamic patching...")
        entropy_model.fit(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        logger.info("Entropy model training completed successfully")
    else:
        logger.warning("No dataset provided. Returning untrained entropy model.")

    return entropy_model

# ---------------------------------------------------------------------
