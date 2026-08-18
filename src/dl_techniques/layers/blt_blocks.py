"""
Byte Latent Transformer (BLT) Core Layer Components

This module implements the fundamental building blocks of the Byte Latent Transformer
architecture, providing a tokenization-free approach to language modeling that operates
directly on raw UTF-8 bytes with dynamic compute allocation.

Architecture Components Overview
================================

The BLT architecture consists of seven core layer components that work together to
achieve efficient byte-level language modeling:

1. **ByteTokenizer**: Converts raw text to byte token sequences and back
2. **EntropyModel**: Small causal transformer for computing next-byte entropy
3. **DynamicPatcher**: Segments bytes into patches based on entropy thresholds
4. **PatchPooling**: Reduces byte sequences to patch representations
5. **LocalEncoder**: Processes bytes within their patches with causal attention
6. **GlobalTransformer**: Models long-range dependencies across patch representations
7. **LocalDecoder**: Generates next-byte predictions with global context

Technical Innovation: Dynamic Patching
=====================================

Unlike traditional tokenization with fixed vocabularies, BLT uses entropy-driven
dynamic patching that:

- **Adapts to Content Complexity**: High-entropy regions (unpredictable) get more
  compute via new patch boundaries, while low-entropy regions (predictable) are
  grouped into larger patches

- **Preserves Byte-Level Information**: No information loss through heuristic
  tokenization, maintaining access to character-level patterns

- **Enables Flexible Scaling**: Patch size can be adjusted independently of model
  size, allowing simultaneous scaling of both dimensions

Hierarchical Processing Pipeline
===============================

The BLT processing pipeline follows this flow:

```
Raw Text → ByteTokenizer → Byte Tokens
                              ↓
EntropyModel → Entropy Values → DynamicPatcher → Patch Boundaries
                                                      ↓
Byte Tokens + Patch IDs → LocalEncoder → Patch Representations
                                              ↓
                          GlobalTransformer → Contextual Patches
                                              ↓
Byte Tokens + Global Context → LocalDecoder → Next-Byte Logits
```

Key Technical Features
=====================

**Entropy-Based Segmentation**:
- Uses Shannon entropy H(x_i) = -Σ p(x_i|context) * log(p(x_i|context))
- Global threshold: H(x_t) > θ_g creates new patch boundary
- Approximate monotonic: H(x_t) - H(x_t-1) > θ_r for trend breaks

**Cross-Attention Mechanisms**:
- Encoder: Patches query, bytes provide keys/values for pooling
- Decoder: Bytes query, patches provide keys/values for context
- Masked attention ensures bytes only attend to their patch context

**Hash N-gram Embeddings**:
- Rolling polynomial hash for n-grams (n=3-8): Hash(g_i,n) = Σ b_i * a^j
- 500K hash functions map arbitrary n-grams to embedding space
- Captures multi-scale byte context without explicit vocabulary

**Causal Attention Patterns**:
- Local models use windowed causal attention within patches
- Global model uses standard causal attention across patches
- Maintains autoregressive properties for generation

Performance Characteristics
==========================

**Computational Efficiency**:
- Reduces inference FLOPs by up to 50% through larger patch sizes
- Dynamic compute allocation: O(patches) vs O(tokens) complexity
- Lightweight local models (6-8 layers) vs heavy global model (12+ layers)

**Memory Efficiency**:
- No large embedding matrices for fixed vocabularies
- Hash embeddings provide compact n-gram representation
- Variable-length patches reduce sequence processing overhead

**Scaling Properties**:
- Enables simultaneous model and patch size scaling
- Better scaling trends beyond compute-optimal training regimes
- Crossover points typically at 2-3x compute-optimal budgets

Robustness Advantages
====================

**Noise Resilience**:
- Direct byte processing handles character-level corruptions
- 8+ point advantage on noisy input benchmarks
- Maintains performance with case changes, character drops, repetitions

**Linguistic Capabilities**:
- 99.9% accuracy on character manipulation tasks
- Superior orthographic and phonological understanding
- Better multilingual performance, especially low-resource languages

**Long-tail Generalization**:
- No out-of-vocabulary issues with fixed tokenizers
- Handles arbitrary character sequences and scripts
- Improved performance on rare byte combinations

Implementation Details
=====================

**Layer Architecture**:
- All transformers use SwiGLU activations and RMSNorm
- RoPE positional embeddings in self-attention layers
- Flash Attention for standard masks, Flex Attention for dynamic masks

**Training Considerations**:
- Entropy model requires separate pre-training on byte sequences
- Patch-aware batching to maintain consistent compute per batch
- Gradient accumulation across variable patch sizes

**Memory Management**:
- Efficient cross-attention implementation with patch masking
- Dynamic tensor shapes handled through careful padding strategies
- Context length normalization to maintain fair comparisons

Usage Patterns
==============

**Research Applications**:
```python
# Study entropy-based segmentation
entropy_model = EntropyModel(vocab_size=260, hidden_dim=256)
patcher = DynamicPatcher(entropy_threshold=1.5)

# Analyze dynamic patching behavior
entropy = entropy_model(byte_tokens)
patch_lengths = patcher(entropy)
```

**Production Deployment**:
```python
# Create efficient BLT model
encoder = LocalEncoder(local_dim=512, global_dim=768)
global_transformer = GlobalTransformer(global_dim=768, num_layers=12)
decoder = LocalDecoder(vocab_size=260, local_dim=512)

# Process with automatic patching
patches = encoder(tokens, patch_ids)
context = global_transformer(patches)
logits = decoder(tokens, context, patch_ids)
```

**Custom Patching Strategies**:
```python
# Implement custom pooling method
pooling = PatchPooling(
    pooling_method='attention',
    output_dim=768,
    num_queries=4
)

# Use with different entropy thresholds
patcher = DynamicPatcher(entropy_threshold=2.0)  # Larger patches
```

Integration with dl-techniques Framework
=======================================

These layers integrate seamlessly with the broader dl-techniques ecosystem:

- **Optimizers**: Use with advanced scheduling from `optimization` module
- **Regularizers**: Apply weight decay and dropout from `regularizers`
- **Losses**: Compatible with standard language modeling objectives
- **Metrics**: Works with perplexity and custom byte-level metrics

The modular design allows for easy experimentation with different:
- Entropy models and thresholds
- Pooling strategies for patch creation
- Cross-attention mechanisms
- Local vs global architecture balance

Future Extensions
=================

The layer design supports extension for:
- **Multimodal Processing**: Extend entropy models to handle image/audio bytes
- **Sparse Attention**: Implement more efficient attention patterns
- **Adaptive Thresholding**: Learn entropy thresholds during training
- **Hierarchical Patches**: Multi-level patch hierarchies for longer contexts

References
==========

Implementation based on:
"Byte Latent Transformer: Patches Scale Better Than Tokens"
Pagnoni et al., 2024
arXiv:2412.09871v1 [cs.CL]

Key innovations:
- Dynamic entropy-based patching algorithm
- Hierarchical byte-patch-global processing
- Cross-attention pooling mechanisms
- Hash n-gram embeddings for byte context

These layers represent a fundamental shift from tokenization-based language
modeling toward more flexible, robust, and efficient byte-level processing
that maintains competitive performance while offering significant advantages
in efficiency, robustness, and multilingual capabilities.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformers.transformer import TransformerLayer
from .attention.multi_head_attention import MultiHeadAttention
from .embedding.positional_embedding import PositionalEmbedding
from ..utils.masking import create_mask
from ..utils.logger import logger

# ---------------------------------------------------------------------


def causal_attend_mask(hidden_states: keras.KerasTensor) -> keras.KerasTensor:
    """Build the lower-triangular self-attention mask for a BLT stack.

    Every stack in BLT -- the entropy model, the local encoder, the global
    transformer over patches and the local decoder's self-attention -- is
    consumed under a next-byte objective, and none of them constructed a mask:
    each called ``TransformerLayer(x, training=...)``, ``TransformerLayer``
    defaults ``attention_mask=None`` and the attention layers mask only with
    what they are handed. Position ``i`` therefore attended to the very byte it
    was asked to predict.

    The mask is built in the masking factory's BLOCK semantics (``True`` means
    "mask out") and inverted once to the ATTEND semantics the attention layers
    expect. It is returned at rank 3 on purpose: a rank-2 mask is interpreted
    by the attention layers as a ``(batch, seq_len)`` *padding* mask, not as a
    ``(seq_len, seq_len)`` score mask, so a rank-2 causal mask would be
    silently misread.

    :param hidden_states: Sequence tensor of shape ``(batch, seq_len, dim)``.
    :type hidden_states: keras.KerasTensor
    :return: Boolean mask ``(batch, seq_len, seq_len)``, ``True`` = may attend.
    :rtype: keras.KerasTensor
    """
    batch_size = ops.shape(hidden_states)[0]
    seq_len = ops.shape(hidden_states)[1]
    blocked = create_mask('causal', seq_len=seq_len, dtype='bool')
    blocked = ops.broadcast_to(
        ops.expand_dims(blocked, axis=0), (batch_size, seq_len, seq_len)
    )
    return ops.logical_not(blocked)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques.blt")
class ByteTokenizer(keras.layers.Layer):
    """
    Converts text to byte tokens for BLT processing.

    This layer handles the conversion of text strings to byte sequences,
    with proper handling of special tokens and padding. It operates at the
    byte level to achieve true language-agnostic processing.

    **Intent**: Provide byte-level tokenization that maintains full character
    information without vocabulary limitations, enabling robust processing of
    any UTF-8 text input.

    **Architecture Overview:**

    .. code-block:: text

    Text String → UTF-8 Encoding → Byte Values + Offset → Special Tokens
    "Hello" → [72, 101, 108, 108, 111] → [76, 105, 112, 112, 115] → [1, 76, ..., 2]
    ```
        :param vocab_size: Size of the vocabulary including special tokens.
        :param byte_offset: Offset added to raw byte values for special tokens.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            byte_offset: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset

        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3

    def text_to_bytes(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Convert text string to byte token sequence.

            :param text: Input text string.
            :param add_bos: Whether to add begin-of-sequence token.
            :param add_eos: Whether to add end-of-sequence token.

            :return: List of byte token IDs.
        """
        # Convert to UTF-8 bytes
        byte_sequence = text.encode('utf-8', errors='ignore')

        # Map bytes to tokens with offset
        tokens = [byte + self.byte_offset for byte in byte_sequence]

        # Add special tokens
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """
        Convert byte token sequence back to text.

            :param tokens: List of byte token IDs.

            :return: Decoded text string.
        """
        # Filter out special tokens and convert back to bytes
        byte_values = []
        for token in tokens:
            if token >= self.byte_offset:
                byte_values.append(token - self.byte_offset)

        # Convert bytes back to string
        try:
            text = bytes(byte_values).decode('utf-8', errors='ignore')
        except (ValueError, UnicodeDecodeError):
            text = ""

        return text

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        ByteTokenizer processes text strings to byte sequences.
        Output shape depends on the text length, so the sequence
        dimension is dynamic (None).

            :param input_shape: Input shape tuple (ignored for this utility layer).

            :return: Output shape tuple: (batch_size, None) for variable-length byte sequences.
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 1:
            return (input_shape[0], None)
        return (None, None)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'byte_offset': self.byte_offset
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class EntropyModel(keras.layers.Layer):
    """
    Small causal transformer for computing next-byte entropy.

    This model predicts the probability distribution of the next byte
    at each position, which is used for dynamic patching. The entropy
    computed from these distributions indicates information density.

    **Intent**: Provide lightweight entropy computation for dynamic patch
    boundary detection, enabling adaptive compute allocation based on
    information content.

    **Architecture Overview:**

    .. code-block:: text

    Byte Tokens → Embedding → Positional → Transformer Layers → LayerNorm → Dense → Logits
    [B,S] → [B,S,H] → [B,S,H] → [B,S,H] → [B,S,H] → [B,S,V] → Shannon Entropy
    ```

    **Mathematical Operations**:
    1. **Token Embedding**: E(x) ∈ ℝ^(V×H)
    2. **Position Encoding**: PE(pos, 2i) = sin(pos/10000^(2i/H))
    3. **Causal Self-Attention**: Att(Q,K,V) with lower triangular mask
    4. **Entropy Calculation**: H(x) = -Σ p(x) log p(x)

        :param vocab_size: Size of byte vocabulary.
        :param hidden_dim: Hidden dimension of the transformer.
        :param num_layers: Number of transformer layers.
        :param num_heads: Number of attention heads.
        :param max_seq_len: Maximum sequence length.
        :param dropout_rate: Dropout rate.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            hidden_dim: int = 256,
            num_layers: int = 6,
            num_heads: int = 8,
            max_seq_len: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Create all sub-layers in __init__
        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_dim,
            name='token_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name='positional_embedding'
        )

        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_size=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_size=self.hidden_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'transformer_layer_{i}'
            )
            self.transformer_layers.append(layer)

        self.layer_norm = keras.layers.LayerNormalization(name='final_layer_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the entropy model layers."""
        # Explicitly build sub-layers for serialization
        self.embedding.build(input_shape)

        # Compute shapes for building transformer layers
        embedded_shape = self.embedding.compute_output_shape(input_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        self.positional_embedding.build(embedded_shape)

        # Build transformer layers
        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        # Build final layers
        self.layer_norm.build(current_shape)
        norm_shape = current_shape
        self.output_projection.build(norm_shape)

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the entropy model.

            :param inputs: Input token tensor of shape (batch_size, seq_len).
            :param training: Whether in training mode.

            :return: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Token embedding
        x = self.embedding(inputs)

        # Add positional embedding
        x = self.positional_embedding(x, training=training)

        # Apply transformer layers. The entropy model predicts the NEXT byte,
        # so it is causal: without the mask its "surprise" at position i is
        # computed from a state that has already read byte i+1.
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        # Final layer norm and projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits

    def compute_entropy(self, logits: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute Shannon entropy from logits.

            :param logits: Logits tensor of shape (batch_size, seq_len, vocab_size).

            :return: Entropy tensor of shape (batch_size, seq_len).
        """
        # Apply softmax to get probabilities
        probs = keras.activations.softmax(logits, axis=-1)

        # Compute log probabilities for numerical stability
        log_probs = ops.log(ops.maximum(probs, 1e-12))

        # Shannon entropy: H = -sum(p * log(p))
        entropy = -ops.sum(probs * log_probs, axis=-1)

        return entropy

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape + (self.vocab_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DynamicPatcher(keras.layers.Layer):
    """
    Creates dynamic patches based on entropy thresholding.

    This layer implements the core dynamic patching algorithm that segments
    byte sequences based on information density measured by entropy. When
    entropy exceeds a threshold, it indicates an unpredictable transition
    where a new patch boundary should be created.

    **Intent**: Enable adaptive sequence segmentation based on information
    content, allowing more compute for complex regions while grouping
    predictable content efficiently.

    **Architecture Overview:**

    .. code-block:: text

    Entropy Values → Threshold Detection → Boundary Creation → Patch Lengths
    [B,S] → Compare > θ → Boundary Mask → [B,max_patches]
    ```

    **Patching Algorithm**:

    A position ``t`` opens a new patch when ``H(x_t) > entropy_threshold``.
    Each byte is then assigned the number of boundaries at or before it,
    saturated at ``max_patches - 1``; the patch lengths are the occupancy
    counts of that assignment. Two consequences are deliberate, not
    incidental:

    - Rows sum to ``seq_len`` **by construction** — every byte is counted
      into exactly one patch. This matters because ``compute_patch_ids``
      does not validate the sum; a row that summed to less would silently
      misassign ids rather than raise.
    - The cap is a POSITION-ordered truncation, not a magnitude ranking.
      Everything after the ``(max_patches - 1)``-th boundary merges into
      the final patch. See the ``call`` anchor for why the alternative is
      inadmissible rather than merely worse.

    A leading zero-length patch is legal and occurs whenever position 0 is
    itself a boundary; trailing patches are zero-length whenever a sequence
    produces fewer boundaries than ``max_patches - 1``. Both leave the patch
    ids non-decreasing, which is what ``LocalDecoder``'s preceding-patch
    gather requires.

        :param entropy_threshold: Entropy (in nats) above which a byte opens a
            new patch. Note the scale: for a vocabulary of size ``V`` the
            entropy of a uniform distribution is ``ln(V)``, so a threshold at
            or below the model's typical entropy makes EVERY position a
            boundary and a threshold above ``ln(V)`` makes none.
        :param max_patches: Maximum number of patches to create.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            entropy_threshold: float = 1.5,
            max_patches: int = 512,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.entropy_threshold = entropy_threshold
        self.max_patches = max_patches

    def call(
            self,
            entropy: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Derive patch lengths from the entropy values, per row.

        Each row is segmented independently: sequences with different content
        get different boundaries. This is what makes the shipped
        ``EntropyModel`` and the trainer's entropy-pretraining stage
        load-bearing — before this, only ``ops.shape(entropy)`` was read and
        every row of the batch received the same equal-length partition.

            :param entropy: Entropy tensor of shape (batch_size, seq_len), in
                the same units as ``entropy_threshold`` (nats, as produced by
                ``EntropyModel.compute_entropy``).
            :param training: Whether in training mode. Unused — the
                segmentation is deterministic.

            :return: Patch lengths tensor of shape (batch_size, max_patches),
                ``int32``, non-negative, each row summing to exactly
                ``seq_len``.
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-012
        # ---------------------------------------------------------------
        # The cap is applied BY POSITION (`ops.minimum` on a running count),
        # never by entropy MAGNITUDE. DO NOT "improve" this into an
        # `ops.top_k` over the entropy values to keep the max_patches-1
        # "most informative" boundaries. That variant is INADMISSIBLE, not
        # merely a different trade-off: a late high-entropy position would
        # displace an earlier boundary, so an EARLIER byte's patch id would
        # depend on a LATER byte. BLT is trained under a next-byte
        # objective, and
        # tests/test_models/test_byte_latent_transformer/test_model.py
        # ::TestCausality::test_future_byte_does_not_change_the_past
        # requires the logits before the perturbed byte to be EXACTLY
        # unchanged. The top-k variant was written and measured: it moves
        # them by 4.85e-01. The price paid here is real and accepted — a
        # sequence whose most informative positions are all late gets one
        # long final patch — but it buys causality structurally rather than
        # by test.
        #
        # Two further spellings are load-bearing:
        #  * The count runs in INT32, not in the compute dtype. A float
        #    count reduction is the failure `activations/sparsemax.py`'s
        #    D-017 anchor records as Defect E (an fp16 tree reduction
        #    counted 2049 as 2048 and 2051 as 2052), and this count is the
        #    patch id itself.
        #  * The lengths are OCCUPANCY COUNTS of a per-byte assignment, not
        #    differences of boundary positions. That is what makes the row
        #    sum exactly `seq_len` structurally: every byte is counted into
        #    exactly one patch, for any threshold, including the degenerate
        #    ends (no boundary at all, or a boundary at every position).
        #    `compute_patch_ids` does NOT validate the sum, so a
        #    construction that merely usually sums correctly would fail
        #    silently.
        # ---------------------------------------------------------------
        is_boundary = ops.cast(entropy > self.entropy_threshold, 'int32')

        # Patch id of byte t = number of boundaries at positions <= t,
        # saturated so that everything past the last admissible boundary
        # merges into the final patch. Depends only on entropy[..., :t + 1].
        patch_index = ops.cumsum(is_boundary, axis=1)
        patch_index = ops.minimum(patch_index, self.max_patches - 1)

        # (batch, seq_len, max_patches) -> (batch, max_patches) occupancy.
        # `compute_patch_ids` already materializes a tensor of this exact
        # shape, so this is not a new memory regime.
        occupancy = ops.one_hot(patch_index, self.max_patches, dtype='int32')

        return ops.sum(occupancy, axis=1)

    def warn_if_segmentation_is_degenerate(
            self,
            entropy: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
    ) -> bool:
        """Report a degenerate segmentation MEASURED on a concrete batch.

        **Contract**: pure except for the log record; returns ``True`` iff a
        warning was emitted, so a caller (or a test) asserts on the decision
        rather than on log text. Never raises and never changes behaviour.
        Requires an EAGER tensor — it reads the entropy values.

        **Pass a ``mask`` whenever the batch is padded.** The rate is a MEAN
        over positions, so padding dilutes it toward the padding's own
        behaviour and can put the informative end structurally out of reach: a
        batch that is 87.5% right-padding (a 256-byte cap padded to a
        2048-position window — the shipped ``large`` BLT preset) caps the
        observed rate at ~0.125 even when EVERY real byte is a boundary, so the
        ``rate == 1.0`` arm can never fire. Measured on that shape with
        all-boundary real content: **0.1250 unmasked** (silent) against
        **1.0000 masked** (warns). A trained entropy model drives pad-after-pad
        to near-zero entropy, which is why the dilution is systematic rather
        than incidental.

        Degenerate means one of the two ends, and only those:

        - **boundary rate 1.0** — every position opens a patch, so patch 0 is
          empty, one byte lands in each patch after it, and the whole remaining
          tail merges into the final patch. This is what an untrained entropy
          model produces (its output sits near the uniform ceiling
          ``ln(vocab_size)``) against any threshold below that ceiling.
        - **boundary rate 0.0** — no position opens a patch, so the entire
          sequence is one patch and ``max_patches`` is inert.

        Both are legal, silent, and worse than the fixed equal-length split
        this layer replaced. Rates strictly between the ends are NOT reported:
        that is an ordinary segmentation, and how coarse or fine it should be
        is a modelling choice this layer has no basis to second-guess.

        :param entropy: Concrete (eager) entropy tensor, ``(batch, seq_len)``,
            in nats — the same tensor ``call`` consumes.
        :param mask: Optional concrete (eager) tensor broadcastable to
            ``entropy``, non-zero at REAL positions and zero at padding. When
            given, the rate is computed over the non-zero positions only. When
            omitted, every position counts — correct only for an unpadded
            batch.
        :return: ``True`` if a warning was logged. Note this includes the
            no-real-positions case (an all-zero ``mask``), which is a defect in
            the CALLER's probe rather than a degenerate segmentation, and is
            reported as such.
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-018
        # This is an EXPLICIT, opt-in diagnostic and it is deliberately NOT
        # called from `call()`. Two reasons, the second measured:
        #  * `call()` is `keras.ops`-only with a static `max_patches` (I-1/I-2/
        #    I-3). A Python-level branch on a tensor value is not expressible
        #    there without raw `tf`.
        #  * A once-per-instance warning inside `call()` cannot observe the
        #    rate at all under tracing: reading it raises
        #    `NotImplementedError: Cannot convert a symbolic tf.Tensor
        #    (Mean:0) to a numpy array`, and a Python side effect in a traced
        #    `call` fires once per RETRACE regardless of the data.
        # It replaces `warn_if_entropy_threshold_is_degenerate`, which compared
        # the threshold to `0.5 * ln(vocab_size)` — vocabulary arithmetic that
        # never looked at the entropy and therefore fired on 100% of shipped
        # configurations (1.5 and 1.3 against a 2.78-nat floor at
        # vocab_size=260), including the one D-015 argues is probably right.
        # Do NOT reintroduce a construction-time variant: at construction there
        # is no entropy to measure. See decisions.md D-018.
        # DECISION plan-2026-08-14T183218-f4c612aa/D-024
        # The rate is computed over REAL positions when a mask is supplied, and
        # the only shipped caller supplies one. Do NOT drop the mask parameter
        # "because the unmasked mean is the same thing": it is not. Padding is
        # not a neutral filler here -- a trained entropy model predicts
        # pad-after-pad with near-zero entropy, so padded positions are
        # systematically NON-boundaries, and the mean over them is an average of
        # the signal with a constant. MEASURED on the shipped `large` shape
        # (256 real bytes padded to 2048) with every real byte a boundary:
        # unmasked 0.1250 -> silent; masked 1.0000 -> warns. The informative arm
        # of this diagnostic was unreachable at its own call site. See
        # decisions.md D-024.
        is_boundary = ops.cast(entropy > self.entropy_threshold, 'float32')

        if mask is None:
            rate = float(ops.convert_to_numpy(ops.mean(is_boundary)))
            scope = "this batch"
        else:
            weights = ops.cast(ops.cast(mask, 'bool'), 'float32')
            counted = float(ops.convert_to_numpy(ops.sum(weights)))
            if counted == 0.0:
                logger.warning(
                    f"{type(self).__name__}: the supplied mask selects NO "
                    f"positions, so no boundary rate could be measured and this "
                    f"diagnostic saw nothing. Check the caller's mask."
                )
                return True
            rate = float(
                ops.convert_to_numpy(ops.sum(is_boundary * weights))
            ) / counted
            scope = (
                f"this batch (measured over its {int(counted)} non-padding "
                f"positions; padding excluded)"
            )

        if rate == 1.0:
            logger.warning(
                f"{type(self).__name__}: entropy_threshold="
                f"{self.entropy_threshold:.4g} nats is below the entropy at "
                f"EVERY position of {scope} (observed boundary rate 1.0), "
                f"so patch 0 is empty, each patch after it holds one byte, and "
                f"the whole remaining sequence collapses into the final patch "
                f"of max_patches={self.max_patches}. Raise the threshold, or "
                f"pretrain the entropy model before relying on the "
                f"segmentation."
            )
            return True

        if rate == 0.0:
            logger.warning(
                f"{type(self).__name__}: entropy_threshold="
                f"{self.entropy_threshold:.4g} nats is above the entropy at "
                f"every position of {scope} (observed boundary rate 0.0), "
                f"so the whole sequence is a single patch and "
                f"max_patches={self.max_patches} is inert. Lower the threshold."
            )
            return True

        return False

    def compute_patch_ids(
            self,
            patch_lengths: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Convert patch lengths to patch IDs for each position.

            :param patch_lengths: Patch lengths tensor of shape (batch_size, max_patches).

            :return: Patch IDs tensor of shape (batch_size, seq_len).
        """
        batch_size = ops.shape(patch_lengths)[0]
        max_patches = ops.shape(patch_lengths)[1]

        # Calculate total sequence length
        total_lengths = ops.sum(patch_lengths, axis=1)
        max_seq_len = ops.max(total_lengths)

        # Vectorized patch ID computation using cumulative sums
        # cumulative_lengths[b, p] = sum of patch_lengths[b, :p+1]
        cumulative_lengths = ops.cumsum(patch_lengths, axis=1)

        # For each position in the sequence, find which patch it belongs to
        # Position i belongs to patch p if cumulative_lengths[b, p-1] <= i < cumulative_lengths[b, p]
        # This is equivalent to: patch_id[i] = number of patches whose cumulative length <= i
        positions = ops.arange(max_seq_len)  # (seq_len,)
        positions = ops.expand_dims(ops.expand_dims(positions, 0), -1)  # (1, seq_len, 1)
        cum_expanded = ops.expand_dims(cumulative_lengths, 1)  # (batch, 1, max_patches)

        # For each position, count how many patch boundaries are <= position
        # patch_id = sum(cumulative_lengths <= position) - 1, clamped to [0, max_patches-1]
        boundary_passed = ops.cast(cum_expanded <= positions, 'int32')
        patch_ids = ops.sum(boundary_passed, axis=-1)  # (batch, seq_len)
        patch_ids = ops.minimum(patch_ids, max_patches - 1)

        return ops.cast(patch_ids, 'int32')

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], self.max_patches)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'entropy_threshold': self.entropy_threshold,
            'max_patches': self.max_patches
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PatchPooling(keras.layers.Layer):
    """
    Pools byte representations within patches to create patch representations.

    This layer reduces the sequence of byte hidden states to patch hidden states
    using various pooling strategies. The attention-based pooling method uses
    learnable query vectors to extract the most relevant information from each patch.

    **Intent**: Aggregate byte-level information within patches into compact
    representations while preserving the most important features for global
    processing.

    **Architecture** (Attention Pooling):
    ```
    Byte Hidden States + Patch IDs → Patch Grouping → Query Attention → Patch Representations
    [B,S,H] + [B,S] → {Patch_i} → Q @ {K,V}_i → [B,P,D]
    ```

    **Pooling Methods**:
    1. **Max Pooling**: max(h_bytes) per patch; an EMPTY patch pools to a zero
       vector, NOT to the internal `-1e9` masking sentinel (D-039).
    2. **Mean Pooling**: mean(h_bytes) per patch
    3. **Attention Pooling**: Learnable queries attend to patch bytes

        :param pooling_method: Method for pooling ('max', 'mean', 'attention').
        :param output_dim: Output dimension for patch representations.
        :param num_queries: Number of query vectors for attention pooling.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            pooling_method: str = 'attention',
            output_dim: int = 768,
            num_queries: int = 4,
            max_patches: int = 64,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.pooling_method = pooling_method
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.max_patches = max_patches

        # Sub-layers that depend on input_dim are created in build()
        self.attention_layer = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build pooling layers."""
        input_dim = input_shape[-1]

        if self.pooling_method == 'attention':
            # Use Keras built-in MHA for cross-attention (query → key/value)
            num_heads = min(8, input_dim)
            key_dim = max(input_dim // num_heads, 1)
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                name='patch_attention'
            )

            # Create learnable query embeddings
            self.query_embeddings = self.add_weight(
                shape=(self.num_queries, input_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='query_embeddings'
            )

            # Explicitly build the attention layer so its weights materialize on
            # .keras reload (lazy first-call build leaves weights unloadable).
            # call() uses query=(B, num_queries, input_dim), key/value=(B, T, input_dim).
            query_shape = (input_shape[0], self.num_queries, input_dim)
            kv_shape = (input_shape[0], None, input_dim)
            self.attention_layer.build(query_shape, kv_shape, kv_shape)

        # Create and build output projection
        self.output_projection = keras.layers.Dense(
            self.output_dim,
            name='output_projection'
        )
        projection_input_shape = (input_shape[0], None, input_dim)
        self.output_projection.build(projection_input_shape)

        super().build(input_shape)

    def call(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Pool byte representations into patch representations.

            :param byte_hiddens: Byte hidden states of shape (batch_size, seq_len, hidden_dim).
            :param patch_ids: Patch IDs of shape (batch_size, seq_len).
            :param training: Whether in training mode.

            :return: Patch representations of shape (batch_size, num_patches, output_dim).
        """
        batch_size = ops.shape(byte_hiddens)[0]
        seq_len = ops.shape(byte_hiddens)[1]
        hidden_dim = ops.shape(byte_hiddens)[2]

        # Use static max_patches for graph-mode compatibility
        num_patches = self.max_patches

        if self.pooling_method == 'max':
            return self._max_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'mean':
            return self._mean_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'attention':
            return self._attention_pooling(byte_hiddens, patch_ids, num_patches, training)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def _max_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Max pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]
        hidden_dim = ops.shape(byte_hiddens)[2]

        patch_reps = []

        for p in range(num_patches):
            # Create mask for positions belonging to this patch
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            # Apply mask and get max (set masked positions to large negative value)
            masked_hiddens = ops.where(mask_expanded, byte_hiddens, -1e9)
            patch_max = ops.max(masked_hiddens, axis=1)  # (batch_size, hidden_dim)

            # DECISION plan-2026-08-18T140459-7991552f/D-039: EMPTY patches are
            # the NORM here, not an edge case -- `DynamicPatcher` always emits
            # `max_patches` slots and fills only as many as there were entropy
            # crossings (~120 of 128 empty for a 16-byte sequence at `micro`).
            # For an empty patch `mask` is all-False, so the `-1e9` sentinel
            # above SURVIVES the max and the slot becomes `[-1e9] * hidden_dim`,
            # which the output `Dense` turns into O(1e9) activations and the
            # downstream `GlobalTransformer`'s LayerNorm then normalizes almost
            # entirely against, annihilating the real patches. Do NOT drop this
            # `where` and do NOT "fix" it by making the sentinel smaller (a
            # finite sentinel is still an arbitrary non-zero constant in every
            # empty slot). The neutral value is 0.0, matching `_mean_pooling`
            # (which divides a zero sum by `max(count, 1)`) and
            # `_attention_pooling` (zeroed keys/values).
            has_any = ops.any(mask, axis=1, keepdims=True)  # (batch_size, 1)
            patch_max = ops.where(has_any, patch_max, ops.zeros_like(patch_max))

            patch_reps.append(patch_max)

        # Stack all patches
        result = ops.stack(patch_reps, axis=1)  # (batch_size, num_patches, hidden_dim)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _mean_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Mean pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            # Create mask for positions belonging to this patch
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            # Apply mask and compute mean
            masked_hiddens = byte_hiddens * mask_expanded
            patch_sum = ops.sum(masked_hiddens, axis=1)  # (batch_size, hidden_dim)
            patch_count = ops.sum(ops.cast(mask, byte_hiddens.dtype), axis=1, keepdims=True)
            patch_mean = patch_sum / ops.maximum(patch_count, 1.0)

            patch_reps.append(patch_mean)

        # Stack all patches
        result = ops.stack(patch_reps, axis=1)  # (batch_size, num_patches, hidden_dim)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _attention_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Attention-based pooling within patches."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            # Find positions belonging to this patch
            mask = ops.equal(patch_ids, p)

            # Get patch-specific hidden states
            mask_expanded = ops.expand_dims(mask, axis=-1)
            patch_hiddens = ops.where(
                mask_expanded,
                byte_hiddens,
                ops.zeros_like(byte_hiddens)
            )

            # Use learnable queries to attend to patch hidden states
            queries = ops.expand_dims(self.query_embeddings, axis=0)
            queries = ops.tile(queries, [batch_size, 1, 1])

            # Cross-attention: queries attend to patch hidden states
            attended = self.attention_layer(
                query=queries,
                value=patch_hiddens,
                key=patch_hiddens,
                training=training
            )

            # Flatten and average the attended queries
            patch_rep = ops.mean(attended, axis=1)  # (batch_size, hidden_dim)

            patch_reps.append(patch_rep)

        # Stack patch representations
        result = ops.stack(patch_reps, axis=1)

        # Project to output dimension if needed
        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, None, self.output_dim)  # num_patches is dynamic

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'pooling_method': self.pooling_method,
            'output_dim': self.output_dim,
            'num_queries': self.num_queries,
            'max_patches': self.max_patches
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LocalEncoder(keras.layers.Layer):
    """
    Local Encoder for BLT that processes bytes within their patches.

    This encoder applies causal self-attention to bytes within patches,
    learning local patterns and dependencies. It then pools the byte
    representations to create a single representation for each patch.

    **Intent**: Process byte sequences with local causal attention to capture
    short-range dependencies, then aggregate into patch representations for
    hierarchical global processing.

    **Architecture Overview:**

    .. code-block:: text

    Byte Tokens → Embedding → Positional → Local Transformers → Patch Pooling
    [B,S] → [B,S,D_l] → [B,S,D_l] → [B,S,D_l] → [B,P,D_g]
    ```

    **Processing Flow**:
    1. **Byte Embedding**: Map tokens to dense vectors
    2. **Position Encoding**: Add positional information
    3. **Local Attention**: Causal self-attention within patches
    4. **Cross-Attention Pooling**: Aggregate bytes to patch representations

        :param vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        :param local_dim: Hidden dimension for local encoder.
        :param num_local_layers: Number of transformer layers in local encoder.
        :param num_heads_local: Number of attention heads for local transformer.
        :param max_sequence_length: Maximum sequence length in bytes.
        :param max_patches: Maximum number of patches per sequence.
        :param dropout_rate: Dropout rate for all layers.
        :param patch_pooling_method: Method for patch pooling ('max', 'mean', 'attention').
        :param global_dim: Hidden dimension for global transformer (output dimension).
        :param cross_attention_queries: Number of queries for patch representation.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            patch_pooling_method: str = 'attention',
            global_dim: int = 768,
            cross_attention_queries: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate
        self.patch_pooling_method = patch_pooling_method
        self.global_dim = global_dim
        self.cross_attention_queries = cross_attention_queries

        # Create all sub-layers in __init__
        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='byte_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout_rate=self.dropout_rate,
            name='positional_embedding'
        )

        self.transformer_layers = []
        for i in range(self.num_local_layers):
            layer = TransformerLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'local_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        self.patch_pooling = PatchPooling(
            pooling_method=self.patch_pooling_method,
            output_dim=self.global_dim,
            num_queries=self.cross_attention_queries,
            max_patches=self.max_patches,
            name='patch_pooling'
        )

        self.layer_norm = keras.layers.LayerNormalization(name='local_encoder_norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build local encoder layers."""
        # Build embedding layer
        self.byte_embedding.build(input_shape)

        # Compute shape after embedding
        embedded_shape = self.byte_embedding.compute_output_shape(input_shape)

        # Build positional embedding
        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        # Build transformer layers
        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        # Build layer norm
        self.layer_norm.build(current_shape)
        norm_shape = current_shape

        # Build patch pooling
        self.patch_pooling.build(norm_shape)

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of local encoder.

            :param byte_tokens: Byte token tensor of shape (batch_size, seq_len).
            :param patch_ids: Patch ID tensor of shape (batch_size, seq_len).
            :param training: Whether in training mode.

            :return: Patch representations of shape (batch_size, num_patches, global_dim).
        """
        # Embed byte tokens
        x = self.byte_embedding(byte_tokens)

        # Add positional embeddings
        x = self.positional_embedding(x, training=training)

        # Apply causal transformer layers
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Pool bytes into patch representations
        patch_representations = self.patch_pooling(x, patch_ids, training=training)

        return patch_representations

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.max_patches, self.global_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate,
            'patch_pooling_method': self.patch_pooling_method,
            'global_dim': self.global_dim,
            'cross_attention_queries': self.cross_attention_queries
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GlobalTransformer(keras.layers.Layer):
    """
    Global Transformer for BLT that processes patch sequences.

    This transformer applies self-attention across patch representations
    to model long-range dependencies in the hierarchical structure.

    **Intent**: Model long-range dependencies between patches using standard
    causal self-attention, enabling global context understanding while
    maintaining computational efficiency through reduced sequence length.

    **Architecture Overview:**

    .. code-block:: text

    Patch Representations → Positional → Global Transformers → Contextualized Patches
    [B,P,D_g] → [B,P,D_g] → [B,P,D_g] → [B,P,D_g]
    ```

    **Global Processing**:
    1. **Patch Positions**: Add positional encoding to patch sequence
    2. **Causal Attention**: Standard transformer self-attention across patches
    3. **Deep Processing**: Multiple layers for complex dependency modeling
    4. **Context Integration**: Rich patch representations with global awareness

        :param global_dim: Hidden dimension for global transformer.
        :param num_global_layers: Number of transformer layers in global processor.
        :param num_heads_global: Number of attention heads for global transformer.
        :param max_patches: Maximum number of patches per sequence.
        :param dropout_rate: Dropout rate for all layers.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            global_dim: int = 768,
            num_global_layers: int = 12,
            num_heads_global: int = 12,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.global_dim = global_dim
        self.num_global_layers = num_global_layers
        self.num_heads_global = num_heads_global
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate

        # Create sub-layers in __init__
        self.patch_positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_patches,
            dim=self.global_dim,
            dropout_rate=self.dropout_rate,
            name='patch_positional_embedding'
        )

        self.transformer_layers = []
        for i in range(self.num_global_layers):
            layer = TransformerLayer(
                hidden_size=self.global_dim,
                num_heads=self.num_heads_global,
                intermediate_size=self.global_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'global_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        self.layer_norm = keras.layers.LayerNormalization(name='global_transformer_norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build global transformer layers."""
        # Build positional embedding
        self.patch_positional_embedding.build(input_shape)
        pos_embedded_shape = self.patch_positional_embedding.compute_output_shape(input_shape)

        # Build transformer layers
        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        # Build final layer norm
        self.layer_norm.build(current_shape)

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
            self,
            patch_representations: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of global transformer.

            :param patch_representations: Patch representations of shape (batch_size, num_patches, global_dim).
            :param training: Whether in training mode.

            :return: Contextual patch representations of shape (batch_size, num_patches, global_dim).
        """
        # Add patch positional embeddings
        x = self.patch_positional_embedding(patch_representations, training=training)

        # Apply global transformer layers, causal over the PATCH axis: patch p's
        # contextualized representation must not depend on patches after it.
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        # Apply final layer norm
        x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'global_dim': self.global_dim,
            'num_global_layers': self.num_global_layers,
            'num_heads_global': self.num_heads_global,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class LocalDecoder(keras.layers.Layer):
    """
    Local Decoder for BLT that generates next byte predictions.

    This decoder processes byte sequences with causal self-attention
    and uses cross-attention to incorporate global patch context.

    **Intent**: Generate next-byte predictions by combining local causal
    modeling with global patch context, enabling both local coherence
    and global consistency in generation.

    **Architecture Overview:**

    .. code-block:: text

    Byte Tokens + Global Context → Self-Attention → Cross-Attention → Output Logits
    [B,S] + [B,P,D_g] → [B,S,D_l] → [B,S,D_l] → [B,S,V]
    ```

    **Decoder Flow**:
    1. **Byte Embedding**: Map input tokens to local dimension
    2. **Causal Self-Attention**: Model local byte dependencies
    3. **Cross-Attention**: Incorporate global patch context
    4. **Output Projection**: Generate vocabulary logits

    **Cross-Attention Mechanism**:
    - Bytes query their corresponding patch representations
    - Masked to ensure bytes only see relevant patch context
    - Combines local patterns with global understanding

        :param vocab_size: Size of byte vocabulary (typically 256 + special tokens).
        :param local_dim: Hidden dimension for local decoder.
        :param global_dim: Hidden dimension for global transformer.
        :param num_local_layers: Number of transformer layers in local decoder.
        :param num_heads_local: Number of attention heads for local transformers.
        :param max_sequence_length: Maximum sequence length in bytes.
        :param dropout_rate: Dropout rate for all layers.
            **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            global_dim: int = 768,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate

        # Create sub-layers in __init__
        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='decoder_byte_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout_rate=self.dropout_rate,
            name='decoder_positional_embedding'
        )

        # Context projection if dimensions don't match
        self.context_projection = None
        if self.global_dim != self.local_dim:
            self.context_projection = keras.layers.Dense(
                self.local_dim,
                name='context_projection'
            )

        # Decoder layers with cross-attention
        self.decoder_layers = []
        self.cross_attention_layers = []
        self.cross_attention_norms = []

        for i in range(self.num_local_layers):
            # Self-attention layer
            decoder_layer = TransformerLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'decoder_transformer_{i}'
            )
            self.decoder_layers.append(decoder_layer)

            # Cross-attention to global patch context (Keras built-in for cross-attn)
            cross_attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads_local,
                key_dim=max(self.local_dim // self.num_heads_local, 1),
                dropout=self.dropout_rate,
                name=f'cross_attention_{i}'
            )
            self.cross_attention_layers.append(cross_attention)

            # Layer norm for cross-attention
            cross_norm = keras.layers.LayerNormalization(name=f'cross_attention_norm_{i}')
            self.cross_attention_norms.append(cross_norm)

        # Final layers
        self.layer_norm = keras.layers.LayerNormalization(name='decoder_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build local decoder layers."""
        # Build byte embedding
        byte_input_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        self.byte_embedding.build(byte_input_shape)

        # Compute embedded shape
        embedded_shape = self.byte_embedding.compute_output_shape(byte_input_shape)

        # Build positional embedding
        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        # Build context projection if needed
        if self.context_projection is not None:
            global_context_shape = (embedded_shape[0], None, self.global_dim)
            self.context_projection.build(global_context_shape)

        # Build decoder layers and cross-attention
        current_shape = pos_embedded_shape
        cross_attention_kv_shape = (current_shape[0], current_shape[1], self.local_dim)

        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            # Build self-attention decoder layer
            decoder_layer.build(current_shape)
            decoder_output_shape = decoder_layer.compute_output_shape(current_shape)

            # Explicitly build cross-attention so its weights materialize on
            # .keras reload (lazy first-call build leaves weights unloadable).
            # call() uses query=decoder_hidden (local_dim), key/value=local_dim context.
            cross_attention.build(
                decoder_output_shape,
                cross_attention_kv_shape,
                cross_attention_kv_shape,
            )

            # Build cross-attention norm
            cross_norm.build(decoder_output_shape)

            current_shape = decoder_output_shape

        # Build final layers
        self.layer_norm.build(current_shape)
        self.output_projection.build(current_shape)

        # Always call parent build at the end (MUST be last)
        super().build(input_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of local decoder.

            :param byte_tokens: Byte token tensor of shape (batch_size, seq_len).
            :param global_context: Global patch representations of shape (batch_size, num_patches, global_dim).
            :param patch_ids: Patch ID tensor of shape (batch_size, seq_len).
            :param training: Whether in training mode.

            :return: Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Embed byte tokens
        x = self.byte_embedding(byte_tokens)

        # Add positional embeddings
        x = self.positional_embedding(x, training=training)

        # Project global context to local dimension if needed
        if self.context_projection is not None:
            global_context = self.context_projection(global_context)

        # Apply decoder layers with cross-attention
        attend_mask = causal_attend_mask(x)
        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            # Self-attention within byte sequence (causal)
            x = decoder_layer(x, attention_mask=attend_mask, training=training)

            # Cross-attention to global context
            cross_attended = self._masked_cross_attention(
                x, global_context, patch_ids, cross_attention, training
            )

            # Residual connection and layer norm for cross-attention
            x = x + cross_attended
            x = cross_norm(x)

        # Apply final layer norm
        x = self.layer_norm(x)

        # Project to vocabulary logits
        logits = self.output_projection(x)

        return logits

    def _masked_cross_attention(
            self,
            decoder_hidden: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            cross_attention: keras.layers.Layer,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply cross-attention to the PRECEDING patch's global representation.

        Byte ``i`` reads the contextualized representation of patch
        ``patch_ids[i] - 1``, not of its own patch, and the attention is
        additionally masked causally over the gathered byte-length key
        sequence. Both restrictions are needed for the decoder to be causal:

        * gathering the byte's *own* patch leaks the future, because a patch
          representation is pooled over every byte of that patch, including
          the bytes after ``i`` -- and including the target byte itself;
        * even with the previous-patch gather, key ``j`` for ``j > i`` may
          carry patch ``patch_ids[j] - 1``, which can be ``patch_ids[i]`` or
          later, so the causal mask over the key axis is not redundant.

        Bytes in patch 0 have no preceding patch. Their gather index is clamped
        to 0 and the gathered vector is then zeroed, so they receive no global
        context at all rather than reading their own patch. Zeroing the key
        rather than masking the query row also avoids a fully-masked softmax
        row.
        """
        batch_size = ops.shape(decoder_hidden)[0]
        seq_len = ops.shape(decoder_hidden)[1]

        # Previous-patch gather, clamped at 0 for the first patch.
        prev_patch_ids = ops.maximum(patch_ids - 1, 0)  # (batch, seq_len)
        gather_idx = ops.expand_dims(prev_patch_ids, axis=-1)  # (batch, seq_len, 1)
        global_dim = ops.shape(global_context)[-1]
        gather_idx = ops.broadcast_to(gather_idx, (batch_size, seq_len, global_dim))
        position_context = ops.take_along_axis(global_context, gather_idx, axis=1)

        # Patch-0 bytes get a zero context vector instead of their own patch.
        has_prev = ops.cast(
            ops.expand_dims(ops.greater(patch_ids, 0), axis=-1),
            position_context.dtype,
        )
        position_context = position_context * has_prev

        # Causal mask over the gathered key sequence. keras MultiHeadAttention
        # takes ATTEND semantics (True = may attend) at shape (B, T_q, T_k).
        blocked = create_mask('causal', seq_len=seq_len, dtype='bool')
        blocked = ops.broadcast_to(
            ops.expand_dims(blocked, axis=0), (batch_size, seq_len, seq_len)
        )
        cross_attend_mask = ops.logical_not(blocked)

        attended = cross_attention(
            query=decoder_hidden,
            value=position_context,
            key=position_context,
            attention_mask=cross_attend_mask,
            training=training
        )

        return attended

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if isinstance(input_shape, list):
            # Multiple inputs - use first for batch and seq dimensions
            batch_size = input_shape[0][0]
            seq_len = input_shape[0][1]
        else:
            batch_size = input_shape[0]
            seq_len = input_shape[1]
        return (batch_size, seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------