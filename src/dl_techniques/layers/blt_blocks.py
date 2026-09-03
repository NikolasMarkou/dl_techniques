"""Seven layers that make up the Byte Latent Transformer (BLT): ByteTokenizer,
EntropyModel, DynamicPatcher, PatchPooling, LocalEncoder, GlobalTransformer,
and LocalDecoder.

BLT replaces a fixed subword vocabulary with entropy-driven patching over raw
UTF-8 bytes. A small causal EntropyModel scores each byte's next-byte
surprise; DynamicPatcher opens a new patch wherever that surprise crosses a
threshold, so predictable stretches merge into large patches and hard-to-predict
stretches get finer-grained compute. LocalEncoder attends over bytes within
their patch and pools each patch to one vector; GlobalTransformer attends
across patches; LocalDecoder generates next-byte logits by combining local
byte context with the preceding patch's global representation.

Every stack here is causal only because each layer is given an explicit
attention mask (`causal_attend_mask`) at every call site; passing no mask lets
a position attend to the byte it is meant to predict. `DynamicPatcher` needs
its `seq_len` passed explicitly when used inside a traced or XLA-compiled
graph, since the alternative (deriving it from the data) makes the layer's
output shape data-dependent.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. (https://arxiv.org/abs/2412.09871)
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.masking import create_mask
from dl_techniques.utils.keras_registration import register_dl_technique

from .transformers.transformer import TransformerLayer
from .embedding.positional_embedding import PositionalEmbedding

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

    The mask is built in the masking factory's block semantics (``True`` means
    "mask out") and inverted once to the attend semantics the attention layers
    expect. It is returned at rank 3: a rank-2 mask is interpreted by the
    attention layers as a ``(batch, seq_len)`` padding mask, not as a
    ``(seq_len, seq_len)`` score mask, so a rank-2 causal mask would be
    misread.

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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class ByteTokenizer(keras.layers.Layer):
    """Converts text strings to and from byte token sequences.

    Operates at the byte level, so there is no fixed subword vocabulary and no
    out-of-vocabulary case: any UTF-8 text round-trips through
    ``text_to_bytes`` / ``tokens_to_text``.

    Architecture:

    .. code-block:: text

        "Hello" (text)
              |
              v
        UTF-8 encode -> raw bytes
              |
              v
        + byte_offset, + BOS/EOS ids
              |
              v
        [1, 76, 105, 112, 112, 115, 2]  (token ids)

    :param vocab_size: Size of the vocabulary including special tokens.
    :type vocab_size: int
    :param byte_offset: Offset added to raw byte values, reserving IDs below
        it for special tokens (pad, BOS, EOS, sep).
    :type byte_offset: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Convert a text string to a byte token sequence.

        :param text: Input text string.
        :type text: str
        :param add_bos: Whether to prepend the begin-of-sequence token.
        :type add_bos: bool
        :param add_eos: Whether to append the end-of-sequence token.
        :type add_eos: bool
        :return: List of byte token IDs.
        :rtype: List[int]
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
        """Convert a byte token sequence back to text.

        :param tokens: List of byte token IDs.
        :type tokens: List[int]
        :return: Decoded text string.
        :rtype: str
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
        """Compute the output shape.

        The sequence dimension is dynamic, since output length depends on
        text length.

        :param input_shape: Input shape tuple (ignored for this utility layer).
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, None)``.
        :rtype: Tuple[Optional[int], ...]
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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class EntropyModel(keras.layers.Layer):
    """Small causal transformer predicting next-byte entropy for patching.

    Predicts the next-byte probability distribution at every position; the
    Shannon entropy of that distribution (`compute_entropy`) is what
    `DynamicPatcher` thresholds to place patch boundaries.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]
              |
              v
        token embedding + positional embedding
              |
              v
        causal TransformerLayer x num_layers
              |
              v
        LayerNorm -> Dense(vocab_size)
              |
              v
        logits [B, S, V] -> Shannon entropy [B, S]

    :param vocab_size: Size of the byte vocabulary.
    :type vocab_size: int
    :param hidden_dim: Hidden dimension of the transformer.
    :type hidden_dim: int
    :param num_layers: Number of transformer layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Run the entropy model forward.

        :param inputs: Input token tensor, shape ``(batch_size, seq_len)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
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
        """Compute Shannon entropy ``H = -sum(p * log(p))`` from logits.

        :param logits: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :type logits: keras.KerasTensor
        :return: Entropy in nats, shape ``(batch_size, seq_len)``.
        :rtype: keras.KerasTensor
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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class DynamicPatcher(keras.layers.Layer):
    """Segments a byte sequence into patches by thresholding entropy.

    A position ``t`` opens a new patch when ``H(x_t) > entropy_threshold``.
    Each byte is assigned the number of boundaries at or before it, saturated
    at ``max_patches - 1``; the returned patch lengths are the occupancy
    counts of that assignment.

    Architecture:

    .. code-block:: text

        entropy [B, S]
              |
              v
        is_boundary = entropy > threshold
              |
              v
        cumsum, saturate at max_patches - 1
              |
              v
        one-hot occupancy -> sum
              |
              v
        patch_lengths [B, max_patches]

    Rows sum to ``seq_len`` by construction, since every byte is counted into
    exactly one patch; ``compute_patch_ids`` does not re-validate this sum. The
    cap truncates by position, not by entropy magnitude — everything after the
    ``(max_patches - 1)``-th boundary merges into the final patch. A
    magnitude-based cap (keeping the highest-entropy boundaries via top-k)
    would let a late high-entropy byte displace an earlier boundary, making an
    earlier byte's patch id depend on a later byte — measured to move
    pre-perturbation logits by 4.85e-01 under
    ``test_future_byte_does_not_change_the_past``, which the position-ordered
    cap keeps exactly unchanged.

    A leading zero-length patch is legal when position 0 is itself a
    boundary; trailing patches are zero-length whenever a sequence produces
    fewer boundaries than ``max_patches - 1``. Both leave patch ids
    non-decreasing, which ``LocalDecoder``'s preceding-patch gather requires.

    :param entropy_threshold: Entropy in nats above which a byte opens a new
        patch. For a vocabulary of size ``V`` the entropy of a uniform
        distribution is ``ln(V)``; a threshold at or below the model's
        typical entropy makes every position a boundary, and one above
        ``ln(V)`` makes none.
    :type entropy_threshold: float
    :param max_patches: Maximum number of patches to create.
    :type max_patches: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Derive patch lengths from entropy values, per row.

        Each row is segmented independently, so sequences with different
        content get different boundaries.

        :param entropy: Entropy, shape ``(batch_size, seq_len)``, in nats (as
            produced by ``EntropyModel.compute_entropy``).
        :type entropy: keras.KerasTensor
        :param training: Unused — the segmentation is deterministic.
        :type training: Optional[bool]
        :return: Patch lengths, shape ``(batch_size, max_patches)``,
            ``int32``, non-negative, each row summing to exactly ``seq_len``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-012: cap by position, not entropy
        # magnitude (no top-k) -- a magnitude cap breaks causality (measured: moves logits 4.85e-01). See decisions.md.
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
        """Warn if a concrete batch's segmentation is degenerate.

        Pure except for the log record; returns ``True`` only if a warning
        was emitted, so a caller or test can assert on the decision rather
        than on log text. Never raises. Requires an eager tensor, since it
        reads the entropy values.

        Pass ``mask`` whenever the batch is padded: the rate is a mean over
        positions, and padding dilutes it toward the padding's own behavior.
        A trained entropy model drives padding to near-zero entropy, so a
        batch that is mostly padding can hide a fully-degenerate real region —
        measured on a 256-real-byte sequence padded to 2048 with every real
        byte a boundary: rate 0.1250 unmasked (silent) versus 1.0000 masked
        (warns).

        Degenerate means one of two ends:

        - boundary rate 1.0 — every position opens a patch, so patch 0 is
          empty, one byte lands in each patch after it, and the remaining
          tail merges into the final patch. An untrained entropy model
          (output near the uniform ceiling ``ln(vocab_size)``) produces this
          against any threshold below that ceiling.
        - boundary rate 0.0 — no position opens a patch, so the whole
          sequence is one patch and ``max_patches`` is inert.

        Rates strictly between the ends are not reported — that is an
        ordinary segmentation.

        :param entropy: Concrete (eager) entropy tensor, ``(batch, seq_len)``,
            in nats — the same tensor ``call`` consumes.
        :type entropy: keras.KerasTensor
        :param mask: Optional concrete (eager) tensor broadcastable to
            ``entropy``, non-zero at real positions and zero at padding. When
            omitted, every position counts, which is correct only for an
            unpadded batch.
        :type mask: Optional[keras.KerasTensor]
        :return: ``True`` if a warning was logged. This includes the
            no-real-positions case (an all-zero ``mask``), reported as a
            defect in the caller's probe rather than a degenerate
            segmentation.
        :rtype: bool
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-018: opt-in, not called from call() --
        # call() is keras.ops-only with static shapes; a Python branch on a tensor value needs eager data. See decisions.md.
        # DECISION plan-2026-08-14T183218-f4c612aa/D-024: rate is computed over real positions when
        # mask is given -- unmasked mean hides degenerate content behind padding (measured 0.1250 vs 1.0000). See decisions.md.
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
            patch_lengths: keras.KerasTensor,
            seq_len: Optional[int] = None
    ) -> keras.KerasTensor:
        """Convert patch lengths to a patch id for each byte position.

        :param patch_lengths: Patch lengths, shape ``(batch_size, max_patches)``.
        :type patch_lengths: keras.KerasTensor
        :param seq_len: Sequence length to expand to. Pass this from the
            caller's own byte tensor whenever known. When ``None``, it is
            recovered from the data as ``max(sum(patch_lengths))``, which
            makes the layer's output shape data-dependent and XLA-incompatible.
        :type seq_len: Optional[int]
        :return: Patch ids, shape ``(batch_size, seq_len)``.
        :rtype: keras.KerasTensor
        """
        max_patches = ops.shape(patch_lengths)[1]

        # DECISION plan-2026-08-19T163559-499b6f0e/D-034: pass seq_len in; deriving it from
        # patch_lengths makes output shape data-dependent, which XLA rejects (broke src/train/blt/). See decisions.md.
        if seq_len is None:
            max_seq_len = ops.max(ops.sum(patch_lengths, axis=1))
        else:
            max_seq_len = seq_len

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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class PatchPooling(keras.layers.Layer):
    """Pools byte hidden states within each patch into one patch vector.

    Architecture (attention pooling):

    .. code-block:: text

        byte hiddens [B, S, H], patch_ids [B, S]
              |
              v
        group bytes by patch id
              |
              v
        learnable queries attend to each patch's bytes
              |
              v
        patch representations [B, P, D]

    Three pooling methods: ``max`` (per-patch max, an empty patch pools to a
    zero vector, not to the internal ``-1e9`` masking sentinel), ``mean``
    (per-patch mean), and ``attention`` (learnable queries attend to each
    patch's bytes).

    :param pooling_method: One of ``'max'``, ``'mean'``, ``'attention'``.
    :type pooling_method: str
    :param output_dim: Output dimension of the patch representations.
    :type output_dim: int
    :param num_queries: Number of query vectors for attention pooling.
    :type num_queries: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Pool byte hidden states into patch representations.

        :param byte_hiddens: Byte hidden states, shape ``(batch_size,
            seq_len, hidden_dim)``.
        :type byte_hiddens: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Patch representations, shape ``(batch_size, num_patches,
            output_dim)``.
        :rtype: keras.KerasTensor
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

            # DECISION plan-2026-08-18T140459-7991552f/D-039: rescue empty patches to zero,
            # not the -1e9 sentinel -- most patch slots are empty by construction; the sentinel would
            # dominate downstream LayerNorm and annihilate the real patches. See decisions.md.
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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class LocalEncoder(keras.layers.Layer):
    """Processes bytes within patches with causal attention, then pools to patches.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]
              |
              v
        byte embedding + positional embedding
              |
              v
        causal TransformerLayer x num_local_layers
              |
              v
        LayerNorm
              |
              v
        PatchPooling (patch_ids) -> patch representations [B, P, D_g]

    :param vocab_size: Size of the byte vocabulary (typically 256 plus
        special tokens).
    :type vocab_size: int
    :param local_dim: Hidden dimension of the local encoder.
    :type local_dim: int
    :param num_local_layers: Number of transformer layers in the local encoder.
    :type num_local_layers: int
    :param num_heads_local: Number of attention heads in the local transformer.
    :type num_heads_local: int
    :param max_sequence_length: Maximum sequence length in bytes.
    :type max_sequence_length: int
    :param max_patches: Maximum number of patches per sequence.
    :type max_patches: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param patch_pooling_method: One of ``'max'``, ``'mean'``, ``'attention'``.
    :type patch_pooling_method: str
    :param global_dim: Output dimension, matching the global transformer's
        hidden dimension.
    :type global_dim: int
    :param cross_attention_queries: Number of queries for attention pooling.
    :type cross_attention_queries: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Run the local encoder forward.

        :param byte_tokens: Byte tokens, shape ``(batch_size, seq_len)``.
        :type byte_tokens: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Patch representations, shape ``(batch_size, num_patches,
            global_dim)``.
        :rtype: keras.KerasTensor
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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class GlobalTransformer(keras.layers.Layer):
    """Applies causal self-attention across patch representations.

    Models long-range dependencies between patches, over a sequence that is
    much shorter than the underlying byte sequence.

    Architecture:

    .. code-block:: text

        patch representations [B, P, D_g]
              |
              v
        patch positional embedding
              |
              v
        causal TransformerLayer x num_global_layers
              |
              v
        LayerNorm -> contextualized patches [B, P, D_g]

    :param global_dim: Hidden dimension of the global transformer.
    :type global_dim: int
    :param num_global_layers: Number of transformer layers.
    :type num_global_layers: int
    :param num_heads_global: Number of attention heads.
    :type num_heads_global: int
    :param max_patches: Maximum number of patches per sequence.
    :type max_patches: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Run the global transformer forward.

        :param patch_representations: Patch representations, shape
            ``(batch_size, num_patches, global_dim)``.
        :type patch_representations: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Contextualized patch representations, same shape as input.
        :rtype: keras.KerasTensor
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

@register_dl_technique("dl_techniques.layers.blt_blocks")
class LocalDecoder(keras.layers.Layer):
    """Generates next-byte logits from causal self-attention and patch context.

    Each decoder layer alternates causal self-attention over bytes with
    cross-attention to the preceding patch's global representation, so a
    prediction combines local byte history with global context without
    leaking the future.

    Architecture:

    .. code-block:: text

        byte tokens [B, S], global context [B, P, D_g]
              |
              v
        byte embedding + positional embedding
              |
              v
        (self-attention -> cross-attention to preceding patch -> norm)
              x num_local_layers
              |
              v
        LayerNorm -> Dense(vocab_size) -> logits [B, S, V]

    :param vocab_size: Size of the byte vocabulary (typically 256 plus
        special tokens).
    :type vocab_size: int
    :param local_dim: Hidden dimension of the local decoder.
    :type local_dim: int
    :param global_dim: Hidden dimension of the global transformer's output.
    :type global_dim: int
    :param num_local_layers: Number of transformer layers in the local decoder.
    :type num_local_layers: int
    :param num_heads_local: Number of attention heads.
    :type num_heads_local: int
    :param max_sequence_length: Maximum sequence length in bytes.
    :type max_sequence_length: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
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
        """Run the local decoder forward.

        :param byte_tokens: Byte tokens, shape ``(batch_size, seq_len)``.
        :type byte_tokens: keras.KerasTensor
        :param global_context: Global patch representations, shape
            ``(batch_size, num_patches, global_dim)``.
        :type global_context: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
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
        """Cross-attend to the preceding patch's global representation.

        Byte ``i`` reads the contextualized representation of patch
        ``patch_ids[i] - 1``, not of its own patch, and the attention is
        additionally masked causally over the gathered byte-length key
        sequence. Both restrictions are needed for the decoder to be causal:
        gathering a byte's own patch leaks the future, since that patch's
        representation is pooled over every byte in it including the target
        byte itself; and even with the previous-patch gather, a later key
        ``j`` may carry a patch index at or after ``patch_ids[i]``, so the
        causal mask over the key axis is not redundant.

        Bytes in patch 0 have no preceding patch. Their gather index is
        clamped to 0 and the gathered vector is then zeroed, so they receive
        no global context at all rather than reading their own patch. Zeroing
        the key rather than masking the query row also avoids a
        fully-masked softmax row.

        :param decoder_hidden: Decoder hidden states, shape ``(batch_size,
            seq_len, local_dim)``.
        :type decoder_hidden: keras.KerasTensor
        :param global_context: Global patch representations, shape
            ``(batch_size, num_patches, local_dim)``.
        :type global_context: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param cross_attention: The ``MultiHeadAttention`` layer to apply.
        :type cross_attention: keras.layers.Layer
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Cross-attended output, shape ``(batch_size, seq_len,
            local_dim)``.
        :rtype: keras.KerasTensor
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