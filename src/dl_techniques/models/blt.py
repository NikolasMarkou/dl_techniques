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

Key Technical Innovations
========================

**Dynamic Patching Schemes**:
- Entropy-based patching using small byte-level language models
- Global and approximate monotonic constraints for patch boundary detection
- Incremental patching capability for autoregressive generation
- Context-aware patch size adaptation

**Cross-Attention Architecture**:
- Perceiver-style cross-attention in encoder (patches query, bytes as keys/values)
- Reverse cross-attention in decoder (bytes query, patches as keys/values)
- Masked attention ensuring patches only attend to constituent bytes

**Byte-Level Enhancements**:
- Rolling polynomial hash functions for n-gram embedding lookup
- Multi-scale byte context incorporation (3-8 gram embeddings)
- UTF-8 aware processing for language-agnostic capabilities

Performance Characteristics
==========================

**Efficiency Gains**:
- Up to 50% reduction in inference FLOPs compared to tokenization-based models
- Simultaneous scaling of model and patch size within fixed inference budgets
- Dynamic compute allocation based on prediction complexity

**Quality Improvements**:
- Matches Llama 3 training flop-controlled performance up to 8B parameters
- Superior scaling trends in fixed-inference budget scenarios
- Enhanced performance on structured and repetitive content

**Robustness Advantages**:
- Significantly improved noise robustness (8+ point advantage on corrupted inputs)
- Enhanced character-level manipulation capabilities (99.9% on spelling tasks)
- Better multilingual performance, especially for low-resource languages
- Superior orthographic and phonological understanding

Scaling Properties
=================

**Training Scale**: Successfully scaled to 8B parameters on 4T bytes of training data,
demonstrating the first flop-controlled scaling study of byte-level models at this scale.

**Inference Scaling**: Introduces new scaling dimension where both model size and patch
size can increase simultaneously while maintaining constant inference budget.

**Crossover Points**: BLT surpasses BPE models beyond compute-optimal training regimes,
typically at 2-3x compute-optimal budgets.

Implementation Details
=====================

**Entropy Model**: Small transformer (14 layers, 512 hidden dim, 100M parameters) with
sliding window attention for entropy computation and patch boundary identification.

**Hash Embeddings**: 500K hash functions with rolling polynomial hashing for efficient
n-gram lookup without explicit vocabulary storage.

**Attention Mechanisms**:
- Block-causal attention in global transformer
- Local windowed attention in byte-level components
- Cross-attention with query masking for patch-byte interactions

Applications and Use Cases
=========================

**Ideal For**:
- Language-agnostic text processing
- Robust text understanding in noisy environments
- Character-level text manipulation tasks
- Multilingual applications with diverse scripts
- Long-context processing with variable complexity
- Applications requiring tokenizer-free processing

**Research Applications**:
- Investigating compute-efficient architectures
- Studying hierarchical text representations
- Developing robust NLP systems
- Multilingual model development

Limitations and Considerations
============================

- Requires careful hyperparameter tuning for entropy thresholds
- Implementation complexity higher than standard transformers
- May require specialized optimization for deployment efficiency
- Entropy model training adds preprocessing overhead

References
==========

Based on "Byte Latent Transformer: Patches Scale Better Than Tokens"
by Pagnoni et al. (2024), introducing the first byte-level architecture
to match token-based performance at scale.

arXiv:2412.09871v1 [cs.CL] 13 Dec 2024
https://github.com/facebookresearch/blt

Example Usage
=============

```python
from dl_techniques.models.blt import create_blt_model

# Create base BLT model
model = create_blt_model(
    vocab_size=260,
    local_dim=512,
    global_dim=768,
    num_local_layers=6,
    num_global_layers=12,
    max_sequence_length=2048,
    entropy_threshold=1.5
)

# Generate text
generated = model.generate(
    prompt="The future of language models",
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True
)

# Fine-tune on domain-specific data
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_data, epochs=3, batch_size=8)
```

The BLT architecture represents a fundamental advancement in language modeling,
offering a scalable, robust, and efficient alternative to tokenization-based
approaches while maintaining competitive performance across diverse tasks.
"""

import keras
from keras import ops
from typing import Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.byte_latent_transformer_blocks import (
    ByteTokenizer, EntropyModel,
    DynamicPatcher, LocalDecoder,
    LocalEncoder, GlobalTransformer
)

# ---------------------------------------------------------------------

class ByteLatentTransformer(keras.Model):
    """
    Complete Byte Latent Transformer model.

    This model implements the full BLT architecture with dynamic patching,
    hierarchical processing, and autoregressive generation capabilities.
    It operates directly on UTF-8 bytes for language-agnostic processing.

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
        entropy_threshold: Threshold for dynamic patching.
        cross_attention_queries: Number of queries for patch representation.
        dropout_rate: Dropout rate for all layers.
        patch_pooling_method: Method for patch pooling ('max', 'mean', 'attention').
        entropy_model: Optional pre-trained entropy model for dynamic patching.
        **kwargs: Additional model arguments.

    Example:
        ```python
        from dl_techniques.models.blt import create_blt_model

        # Create model
        model = create_blt_model(
            vocab_size=260,
            local_dim=512,
            global_dim=768,
            num_local_layers=6,
            num_global_layers=12
        )

        # Compile for training
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """

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

        # Initialize tokenizer and data processing
        self.tokenizer = ByteTokenizer(
            vocab_size=vocab_size,
            name='tokenizer'
        )

        # Initialize entropy model
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

        # Initialize dynamic patcher
        self.patcher = DynamicPatcher(
            entropy_threshold=entropy_threshold,
            max_patches=max_patches,
            name='patcher'
        )

        # Initialize main model components
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

    def train_step(self, data):
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

        with keras.utils.GradientTape() as tape:
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
                next_token = ops.random.categorical(ops.log(probs), num_samples=1)
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
        for i in range(ops.shape(logits)[0]):
            for j in range(k):
                idx = top_k_indices[i, j]
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
        for i in range(ops.shape(sorted_indices)[0]):
            indices = sorted_indices[i]
            values = filtered_logits[i]
            for j in range(ops.shape(indices)[0]):
                output_logits = ops.slice_update(
                    output_logits, [i, indices[j]], values[j]
                )

        return output_logits

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
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
            'patch_pooling_method': self.patch_pooling_method
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ByteLatentTransformer':
        """Create model from configuration."""
        return cls(**config)

# ---------------------------------------------------------------------
# Convenience functions for creating BLT models
# ---------------------------------------------------------------------

def create_blt_model(
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
        compile_model: bool = False,
        optimizer: str = 'adam',
        learning_rate: float = 1e-4
) -> ByteLatentTransformer:
    """
    Create a BLT model with the specified configuration.

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
        entropy_threshold: Threshold for dynamic patching.
        cross_attention_queries: Number of queries for patch representation.
        dropout_rate: Dropout rate for all layers.
        patch_pooling_method: Method for patch pooling ('max', 'mean', 'attention').
        entropy_model: Optional pre-trained entropy model.
        compile_model: Whether to compile the model.
        optimizer: Optimizer to use if compiling.
        learning_rate: Learning rate if compiling.

    Returns:
        Configured BLT model.
    """
    model = ByteLatentTransformer(
        vocab_size=vocab_size,
        local_dim=local_dim,
        global_dim=global_dim,
        num_local_layers=num_local_layers,
        num_global_layers=num_global_layers,
        num_heads_local=num_heads_local,
        num_heads_global=num_heads_global,
        max_sequence_length=max_sequence_length,
        max_patches=max_patches,
        entropy_threshold=entropy_threshold,
        cross_attention_queries=cross_attention_queries,
        dropout_rate=dropout_rate,
        patch_pooling_method=patch_pooling_method,
        entropy_model=entropy_model
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

    return model


def create_blt_small() -> ByteLatentTransformer:
    """Create a small BLT model for experimentation."""
    return create_blt_model(
        vocab_size=260,
        local_dim=256,
        global_dim=384,
        num_local_layers=4,
        num_global_layers=6,
        num_heads_local=4,
        num_heads_global=6,
        max_sequence_length=1024,
        max_patches=256
    )


def create_blt_base() -> ByteLatentTransformer:
    """Create a base-sized BLT model."""
    return create_blt_model(
        vocab_size=260,
        local_dim=512,
        global_dim=768,
        num_local_layers=6,
        num_global_layers=12,
        num_heads_local=8,
        num_heads_global=12,
        max_sequence_length=2048,
        max_patches=512
    )


def create_blt_large() -> ByteLatentTransformer:
    """Create a large BLT model."""
    return create_blt_model(
        vocab_size=260,
        local_dim=768,
        global_dim=1024,
        num_local_layers=8,
        num_global_layers=16,
        num_heads_local=12,
        num_heads_global=16,
        max_sequence_length=4096,
        max_patches=1024
    )


def train_entropy_model(
        vocab_size: int = 260,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        dataset=None,  # Training dataset
        epochs: int = 10,
        batch_size: int = 32
) -> EntropyModel:
    """
    Train a separate entropy model for dynamic patching.

    This function creates and trains a small causal transformer
    to serve as the entropy model for BLT's dynamic patching.

    Args:
        vocab_size: Size of byte vocabulary.
        hidden_dim: Hidden dimension of the entropy model.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
        dataset: Training dataset (should yield byte token sequences).
        epochs: Number of training epochs.
        batch_size: Batch size for training.

    Returns:
        Trained entropy model.
    """
    entropy_model = EntropyModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len
    )

    # Compile for language modeling
    entropy_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train if dataset provided
    if dataset is not None:
        logger.info("Training entropy model...")
        entropy_model.fit(
            dataset,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        logger.info("Entropy model training completed")

    return entropy_model