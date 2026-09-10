"""Seven layers that make up the Byte Latent Transformer (BLT): ByteTokenizer,
EntropyModel, DynamicPatcher, PatchPooling, LocalEncoder, GlobalTransformer,
and LocalDecoder, plus the shared `causal_attend_mask` helper.

BLT replaces a fixed subword vocabulary with entropy-driven patching over raw
UTF-8 bytes. A small causal EntropyModel scores each byte's next-byte
surprise; DynamicPatcher opens a new patch wherever that surprise crosses a
threshold, so predictable stretches merge into large patches and
hard-to-predict stretches get finer-grained compute. LocalEncoder attends over
bytes and pools each patch to one vector; GlobalTransformer attends across
patches; LocalDecoder combines local byte context with the preceding patch's
global representation to produce next-byte logits. Each stack is causal
because every call site hands its `TransformerLayer`s an explicit
`causal_attend_mask`; the attention layers mask only with what they are given.
`DynamicPatcher.compute_patch_ids` needs its `seq_len` passed explicitly under
a traced or XLA-compiled graph, since recovering it from the data makes the
output shape data-dependent. Patch slots beyond a sequence's boundary count
are empty rather than masked, and nothing here carries pretrained weights.

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

from ..transformers.transformer import TransformerLayer
from ..embedding.positional_embedding import PositionalEmbedding

# ---------------------------------------------------------------------


def causal_attend_mask(hidden_states: keras.KerasTensor) -> keras.KerasTensor:
    """Build the lower-triangular self-attention mask for a BLT stack.

    Only the batch and sequence sizes of ``hidden_states`` are read; its values
    and dtype are ignored. Every stack in BLT is consumed under a next-byte
    objective, so each call site passes this mask to its ``TransformerLayer``s.

    Mask semantics:

    .. code-block:: text

        create_mask('causal')   True = mask out    (block semantics)
              │
              ▼
        logical_not             True = may attend  (attend semantics)
              │
              ▼
        broadcast to [B, S, S]

    Rank 3 matters: the attention layers read a rank-2 mask as a
    ``(batch, seq_len)`` padding mask rather than a ``(seq_len, seq_len)``
    score mask, so a rank-2 causal mask would be misread.

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

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class ByteTokenizer(keras.layers.Layer):
    """Convert text strings to and from byte token sequences.

    Operates at the byte level, so there is no fixed subword vocabulary and no
    out-of-vocabulary case: any UTF-8 text round-trips through
    ``text_to_bytes`` / ``tokens_to_text``. The layer has no ``call`` and no
    weights; both directions are plain Python over lists of ints.

    Architecture:

    .. code-block:: text

        "Hello"
              │
              ▼
        ┌──────────────────────────────┐
        │ utf-8 encode                 │
        └──────────────────────────────┘
              │ 72, 101, 108, 108, 111
              ▼
        ┌──────────────────────────────┐
        │ add byte_offset              │
        └──────────────────────────────┘
              │ 76, 105, 112, 112, 115
              ▼
        ┌──────────────────────────────┐
        │ prepend bos, append eos      │
        └──────────────────────────────┘
              │
              ▼
        [1, 76, 105, 112, 112, 115, 2]

    Special ids are fixed at pad 0, bos 1, eos 2 and sep 3, so a
    ``byte_offset`` below 4 would collide with them. Nothing checks this, and
    nothing checks a token against ``vocab_size``, which is carried for the
    config only.

    :param vocab_size: Size of the vocabulary including special tokens. Stored
        for serialization; no method reads it.
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

        # A byte_offset at or below 3 would collide with these four ids.
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3

    def text_to_bytes(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a text string to a byte token sequence.

        Undecodable input is dropped rather than raising, since the encode uses
        ``errors='ignore'``.

        :param text: Input text string.
        :type text: str
        :param add_bos: Whether to prepend the begin-of-sequence token.
        :type add_bos: bool
        :param add_eos: Whether to append the end-of-sequence token.
        :type add_eos: bool
        :return: List of byte token IDs.
        :rtype: List[int]
        """
        byte_sequence = text.encode('utf-8', errors='ignore')

        tokens = [byte + self.byte_offset for byte in byte_sequence]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert a byte token sequence back to text.

        Tokens below ``byte_offset`` are dropped, which removes the special
        ids. A token that leaves a value above 255 after the offset is removed
        cannot form a byte, and the whole call then returns an empty string.

        :param tokens: List of byte token IDs.
        :type tokens: List[int]
        :return: Decoded text string, empty if the byte values are not a valid
            sequence.
        :rtype: str
        """
        byte_values = []
        for token in tokens:
            if token >= self.byte_offset:
                byte_values.append(token - self.byte_offset)

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
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'byte_offset': self.byte_offset
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class EntropyModel(keras.layers.Layer):
    """Predict next-byte logits with a small causal transformer.

    ``call`` returns logits. Their Shannon entropy comes from a separate
    ``compute_entropy`` call, and that entropy is what ``DynamicPatcher``
    thresholds to place patch boundaries.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]
              │
              ▼
        ┌──────────────────────────────┐
        │ token embedding              │
        └──────────────────────────────┘
              │ [B, S, H]
              ▼
        ┌──────────────────────────────┐
        │ positional embedding         │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4H                 │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to vocab_size          │
        └──────────────────────────────┘
              │
              ▼
        logits [B, S, V]
              │
              ▼
        compute_entropy ──► [B, S] in nats

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
    :param dropout_rate: Dropout rate, shared by the positional embedding and
        the transformer layers.
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
        """Build the entropy model layers.

        :param input_shape: Token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.embedding.build(input_shape)

        embedded_shape = self.embedding.compute_output_shape(input_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        self.positional_embedding.build(embedded_shape)

        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)
        norm_shape = current_shape
        self.output_projection.build(norm_shape)

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
        x = self.embedding(inputs)

        x = self.positional_embedding(x, training=training)

        # Without the mask, the surprise at position i is computed from a state
        # that has already read byte i+1.
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits

    def compute_entropy(self, logits: keras.KerasTensor) -> keras.KerasTensor:
        """Compute Shannon entropy ``H = -sum(p * log(p))`` from logits.

        Probabilities are floored at 1e-12 before the log, so the result stays
        finite and its ceiling is ``ln(vocab_size)``.

        :param logits: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :type logits: keras.KerasTensor
        :return: Entropy in nats, shape ``(batch_size, seq_len)``.
        :rtype: keras.KerasTensor
        """
        probs = keras.activations.softmax(logits, axis=-1)

        log_probs = ops.log(ops.maximum(probs, 1e-12))

        entropy = -ops.sum(probs * log_probs, axis=-1)

        return entropy

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Token shape ``(batch, seq_len)`` as a tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Input shape with ``vocab_size`` appended.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape + (self.vocab_size,)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class DynamicPatcher(keras.layers.Layer):
    """Segment a byte sequence into patches by thresholding entropy.

    A position ``t`` opens a new patch when ``H(x_t) > entropy_threshold``.
    Each byte is assigned the number of boundaries at or before it, saturated
    at ``max_patches - 1``; the returned patch lengths are the occupancy
    counts of that assignment. The layer holds no weights.

    Architecture:

    .. code-block:: text

        entropy [B, S]
              │
              ▼
        ┌──────────────────────────────┐
        │ entropy > threshold          │
        └──────────────────────────────┘
              │ is_boundary [B, S]
              ▼
        ┌──────────────────────────────┐
        │ cumsum, min at P-1           │
        └──────────────────────────────┘
              │ patch_index [B, S]
              ▼
        ┌──────────────────────────────┐
        │ one-hot, then sum over S     │
        └──────────────────────────────┘
              │
              ▼
        patch_lengths [B, max_patches]

    Segmentation, with max_patches 4:

    .. code-block:: text

        position           0   1   2   3   4   5
        is_boundary        0   1   0   0   1   0
        patch_index        0   1   1   1   2   2
        patch_lengths      1   3   2   0

    Rows sum to ``seq_len`` by construction, since every byte is counted into
    exactly one patch; ``compute_patch_ids`` does not re-validate that sum. The
    cap truncates by position, not by entropy magnitude, so everything after
    the ``(max_patches - 1)``-th boundary merges into the final patch. Keeping
    the highest-entropy boundaries instead would let a late high-entropy byte
    displace an earlier one, making an earlier byte's patch id depend on a
    later byte.

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
        # DECISION plan-2026-08-14T183218-f4c612aa/D-012: cap by position, not by
        # entropy magnitude; a top-k cap breaks causality, moving logits 4.85e-01.
        # See decisions.md.
        is_boundary = ops.cast(entropy > self.entropy_threshold, 'int32')

        # The saturating cumsum depends only on entropy[..., :t + 1].
        patch_index = ops.cumsum(is_boundary, axis=1)
        patch_index = ops.minimum(patch_index, self.max_patches - 1)

        # compute_patch_ids already materializes a tensor of this exact shape, so
        # the one-hot is not a new memory regime.
        occupancy = ops.one_hot(patch_index, self.max_patches, dtype='int32')

        return ops.sum(occupancy, axis=1)

    def warn_if_segmentation_is_degenerate(
            self,
            entropy: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
    ) -> bool:
        """Warn if a concrete batch's segmentation is degenerate.

        Pure except for the log record; returns ``True`` only if a warning
        was emitted, so a caller can assert on the decision rather than on log
        text. Never raises. Requires an eager tensor, since it reads the
        entropy values.

        Pass ``mask`` whenever the batch is padded. The rate is a mean over
        positions, and a trained entropy model drives padding to near-zero
        entropy, so a mostly-padding batch can hide a fully degenerate real
        region behind a low unmasked rate.

        Degenerate means one of two ends:

        - boundary rate 1.0 — every position opens a patch, so patch 0 is
          empty, one byte lands in each patch after it, and the remaining
          tail merges into the final patch. An untrained entropy model
          (output near the uniform ceiling ``ln(vocab_size)``) produces this
          against any threshold below that ceiling.
        - boundary rate 0.0 — no position opens a patch, so the whole
          sequence is one patch and ``max_patches`` is inert.

        Rates strictly between the ends are an ordinary segmentation and are
        not reported.

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
        # DECISION plan-2026-08-14T183218-f4c612aa/D-018: opt-in, never called from
        # call(), which is keras.ops-only; a branch on a value needs eager data.
        # DECISION plan-2026-08-14T183218-f4c612aa/D-024: measure over real positions
        # when a mask is given; unmasked hides content behind padding (0.1250 vs
        # 1.0000). See decisions.md.
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

        A position's id is the number of cumulative patch lengths at or below
        it, clamped to ``max_patches - 1``, so a zero-length patch consumes no
        position.

        :param patch_lengths: Patch lengths, shape ``(batch_size, max_patches)``.
        :type patch_lengths: keras.KerasTensor
        :param seq_len: Sequence length to expand to. Pass this from the
            caller's own byte tensor whenever known. When ``None``, it is
            recovered from the data as ``max(sum(patch_lengths))``, which
            makes the layer's output shape data-dependent and XLA-incompatible.
        :type seq_len: Optional[int]
        :return: Patch ids, shape ``(batch_size, seq_len)``, ``int32``.
        :rtype: keras.KerasTensor
        """
        max_patches = ops.shape(patch_lengths)[1]

        # DECISION plan-2026-08-19T163559-499b6f0e/D-034: pass seq_len in; deriving it
        # makes the output shape data-dependent, which XLA rejects. See decisions.md.
        if seq_len is None:
            max_seq_len = ops.max(ops.sum(patch_lengths, axis=1))
        else:
            max_seq_len = seq_len

        cumulative_lengths = ops.cumsum(patch_lengths, axis=1)

        positions = ops.arange(max_seq_len)
        positions = ops.expand_dims(ops.expand_dims(positions, 0), -1)
        cum_expanded = ops.expand_dims(cumulative_lengths, 1)

        boundary_passed = ops.cast(cum_expanded <= positions, 'int32')
        patch_ids = ops.sum(boundary_passed, axis=-1)
        patch_ids = ops.minimum(patch_ids, max_patches - 1)

        return ops.cast(patch_ids, 'int32')

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Entropy shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, max_patches)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.max_patches)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'entropy_threshold': self.entropy_threshold,
            'max_patches': self.max_patches
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class PatchPooling(keras.layers.Layer):
    """Pool byte hidden states within each patch into one patch vector.

    The output always has ``max_patches`` slots, built by looping over patch
    indices in Python, so the layer emits that many sub-graphs whatever the
    sequence contains. Every method ends with a Dense projection to
    ``output_dim``.

    Architecture (attention pooling):

    .. code-block:: text

        byte hiddens [B, S, H]        patch_ids [B, S]
              │                             │
              ▼                             ▼
        ┌──────────────────────────────────────┐
        │ for p in range(max_patches):         │
        │   keep bytes with patch_ids == p     │
        │   zero the rest                      │
        │   learnable queries cross-attend     │
        │   mean over the queries              │
        └──────────────────────────────────────┘
              │ stack over p, [B, P, H]
              ▼
        ┌──────────────────────────────────────┐
        │ dense to output_dim                  │
        └──────────────────────────────────────┘
              │
              ▼
        patch representations [B, P, output_dim]

    Attention pooling passes no attention mask, so the zeroed out-of-patch
    positions still take part as keys and values.

    Methods and empty patches:

    .. code-block:: text

        method     non-empty patch          empty patch
        max        per-patch maximum        zero vector
        mean       per-patch mean           zero vector
        attention  queries attend, then     attends over an
                   mean over queries        all-zero sequence

    The ``max`` path masks with an internal ``-1e9`` sentinel and rescues empty
    patches to zero rather than leaving the sentinel in place.

    :param pooling_method: One of ``'max'``, ``'mean'``, ``'attention'``. An
        unknown value raises ``ValueError`` from ``call``, not from the
        constructor.
    :type pooling_method: str
    :param output_dim: Output dimension of the patch representations.
    :type output_dim: int
    :param num_queries: Number of query vectors for attention pooling. Unused
        by the other two methods.
    :type num_queries: int
    :param max_patches: Number of patch slots emitted, used as the static
        patch count in ``call``.
    :type max_patches: int
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

        # Both depend on the input width, so build() creates them.
        self.attention_layer = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the pooling and projection layers.

        The attention head count and key width are derived from the input
        width, not from ``output_dim``.

        :param input_shape: Byte hidden shape ``(batch, seq_len, hidden_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_dim = input_shape[-1]

        if self.pooling_method == 'attention':
            num_heads = min(8, input_dim)
            key_dim = max(input_dim // num_heads, 1)
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                name='patch_attention'
            )

            self.query_embeddings = self.add_weight(
                shape=(self.num_queries, input_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='query_embeddings'
            )

            # Explicit build, because a lazy first-call build leaves the
            # attention weights unloadable on a .keras reload.
            query_shape = (input_shape[0], self.num_queries, input_dim)
            kv_shape = (input_shape[0], None, input_dim)
            self.attention_layer.build(query_shape, kv_shape, kv_shape)

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
        :param training: Whether in training mode. Reaches the attention
            sub-layer only.
        :type training: Optional[bool]
        :return: Patch representations, shape ``(batch_size, max_patches,
            output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``pooling_method`` is not one of the three names.
        """
        batch_size = ops.shape(byte_hiddens)[0]
        seq_len = ops.shape(byte_hiddens)[1]
        hidden_dim = ops.shape(byte_hiddens)[2]

        # A static count keeps the loop and the output shape graph-safe.
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
        """Take the per-channel maximum over each patch's bytes."""
        batch_size = ops.shape(byte_hiddens)[0]
        hidden_dim = ops.shape(byte_hiddens)[2]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            masked_hiddens = ops.where(mask_expanded, byte_hiddens, -1e9)
            patch_max = ops.max(masked_hiddens, axis=1)

            # DECISION plan-2026-08-18T140459-7991552f/D-039: rescue empty patches to
            # zero; the -1e9 sentinel would dominate the downstream LayerNorm.
            # See decisions.md.
            has_any = ops.any(mask, axis=1, keepdims=True)
            patch_max = ops.where(has_any, patch_max, ops.zeros_like(patch_max))

            patch_reps.append(patch_max)

        result = ops.stack(patch_reps, axis=1)

        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _mean_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Take the mean over each patch's bytes, empty patches giving zero."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            masked_hiddens = byte_hiddens * mask_expanded
            patch_sum = ops.sum(masked_hiddens, axis=1)
            # The floor of 1 turns an empty patch into a zero vector.
            patch_count = ops.sum(ops.cast(mask, byte_hiddens.dtype), axis=1, keepdims=True)
            patch_mean = patch_sum / ops.maximum(patch_count, 1.0)

            patch_reps.append(patch_mean)

        result = ops.stack(patch_reps, axis=1)

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
        """Let learnable queries cross-attend to each patch's zeroed sequence."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)

            # Out-of-patch positions are zeroed rather than masked, so they
            # remain in the key and value sequence.
            mask_expanded = ops.expand_dims(mask, axis=-1)
            patch_hiddens = ops.where(
                mask_expanded,
                byte_hiddens,
                ops.zeros_like(byte_hiddens)
            )

            queries = ops.expand_dims(self.query_embeddings, axis=0)
            queries = ops.tile(queries, [batch_size, 1, 1])

            attended = self.attention_layer(
                query=queries,
                value=patch_hiddens,
                key=patch_hiddens,
                training=training
            )

            patch_rep = ops.mean(attended, axis=1)

            patch_reps.append(patch_rep)

        result = ops.stack(patch_reps, axis=1)

        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Byte hidden shape ``(batch, seq_len, hidden_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, None, output_dim)``. The patch axis is reported
            as dynamic even though ``call`` always emits ``max_patches``;
            ``LocalEncoder.compute_output_shape`` reports the static count.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, None, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'pooling_method': self.pooling_method,
            'output_dim': self.output_dim,
            'num_queries': self.num_queries,
            'max_patches': self.max_patches
        })
        return config


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class LocalEncoder(keras.layers.Layer):
    """Process bytes with causal attention, then pool them into patches.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]            patch_ids [B, S]
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ byte embedding               │    │
        └──────────────────────────────┘    │
              │ [B, S, D_l]                 │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ positional embedding         │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4 * D_l            │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ layer norm                   │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             ▼
        ┌──────────────────────────────────────┐
        │ patch pooling                        │
        └──────────────────────────────────────┘
              │
              ▼
        patch representations [B, max_patches, D_g]

    The attention is causal over bytes but carries no padding mask, so padded
    positions are attended to as ordinary bytes.

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
    :param max_patches: Maximum number of patches per sequence, forwarded to
        the pooling layer as its patch-slot count.
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
        """Build the local encoder layers.

        :param input_shape: Byte token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.byte_embedding.build(input_shape)

        embedded_shape = self.byte_embedding.compute_output_shape(input_shape)

        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)
        norm_shape = current_shape

        self.patch_pooling.build(norm_shape)

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
        :return: Patch representations, shape ``(batch_size, max_patches,
            global_dim)``.
        :rtype: keras.KerasTensor
        """
        x = self.byte_embedding(byte_tokens)

        x = self.positional_embedding(x, training=training)

        # The pooled patch vectors feed a next-byte objective, so byte i must
        # not attend past itself.
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        x = self.layer_norm(x)

        patch_representations = self.patch_pooling(x, patch_ids, training=training)

        return patch_representations

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Byte token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, max_patches, global_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.max_patches, self.global_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class GlobalTransformer(keras.layers.Layer):
    """Apply causal self-attention across patch representations.

    Models long-range dependencies between patches, over a sequence that is
    much shorter than the underlying byte sequence.

    Architecture:

    .. code-block:: text

        patch representations [B, P, D_g]
              │
              ▼
        ┌──────────────────────────────┐
        │ patch positional embedding   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4 * D_g            │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        contextualized patches [B, P, D_g]

    The mask is causal over the patch axis only. Empty patch slots carry no
    padding mask, so they take part in attention like any other patch.

    :param global_dim: Hidden dimension of the global transformer.
    :type global_dim: int
    :param num_global_layers: Number of transformer layers.
    :type num_global_layers: int
    :param num_heads_global: Number of attention heads.
    :type num_heads_global: int
    :param max_patches: Maximum number of patches per sequence, and the
        positional embedding's length.
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
        """Build the global transformer layers.

        :param input_shape: Patch representation shape ``(batch, P, D_g)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.patch_positional_embedding.build(input_shape)
        pos_embedded_shape = self.patch_positional_embedding.compute_output_shape(input_shape)

        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)

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
        x = self.patch_positional_embedding(patch_representations, training=training)

        # Patch p's representation must not depend on the patches after it.
        attend_mask = causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Patch representation shape ``(batch, P, D_g)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape, since the layer preserves it.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class LocalDecoder(keras.layers.Layer):
    """Generate next-byte logits from causal self-attention and patch context.

    Each decoder layer runs causal self-attention over bytes, then
    cross-attends to the preceding patch's global representation, adds that as
    a residual and normalizes. A prediction therefore combines local byte
    history with global context without reading the future.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]        global context [B, P, D_g]
              │                             │
              ▼                             ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ byte embedding           │  │ dense to D_l             │
        └──────────────────────────┘  │ (only if D_g != D_l)     │
              │                       └──────────────────────────┘
              ▼                             │
        ┌──────────────────────────┐        │
        │ positional embedding     │        │
        └──────────────────────────┘        │
              │ [B, S, D_l]                 │
              ▼                             │
        ┌──────────────────────────────────────────┐
        │ x N:                                     │
        │   transformer layer ◄── causal mask      │
        │   cross-attention   ◄── prev-patch keys  │
        │   residual add, then layer norm          │
        └──────────────────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to vocab_size          │
        └──────────────────────────────┘
              │
              ▼
        logits [B, S, V]

    Cross-attention gather:

    .. code-block:: text

        patch_ids        0    0    1    1    2
        gather index     0    0    0    0    1    (clamped at 0)
        has_prev         0    0    1    1    1
        key vector       0    0   g_0  g_0  g_1

    :param vocab_size: Size of the byte vocabulary (typically 256 plus
        special tokens).
    :type vocab_size: int
    :param local_dim: Hidden dimension of the local decoder.
    :type local_dim: int
    :param global_dim: Hidden dimension of the global transformer's output. A
        value different from ``local_dim`` adds the context projection.
    :type global_dim: int
    :param num_local_layers: Number of transformer layers in the local decoder.
    :type num_local_layers: int
    :param num_heads_local: Number of attention heads, in both the
        self-attention and the cross-attention.
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

        # Matching widths need no projection, so the attribute stays None.
        self.context_projection = None
        if self.global_dim != self.local_dim:
            self.context_projection = keras.layers.Dense(
                self.local_dim,
                name='context_projection'
            )

        self.decoder_layers = []
        self.cross_attention_layers = []
        self.cross_attention_norms = []

        for i in range(self.num_local_layers):
            decoder_layer = TransformerLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'decoder_transformer_{i}'
            )
            self.decoder_layers.append(decoder_layer)

            cross_attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads_local,
                key_dim=max(self.local_dim // self.num_heads_local, 1),
                dropout=self.dropout_rate,
                name=f'cross_attention_{i}'
            )
            self.cross_attention_layers.append(cross_attention)

            cross_norm = keras.layers.LayerNormalization(name=f'cross_attention_norm_{i}')
            self.cross_attention_norms.append(cross_norm)

        self.layer_norm = keras.layers.LayerNormalization(name='decoder_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the local decoder layers.

        :param input_shape: Byte token shape ``(batch, seq_len)``, or a list
            whose first entry is that shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        byte_input_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        self.byte_embedding.build(byte_input_shape)

        embedded_shape = self.byte_embedding.compute_output_shape(byte_input_shape)

        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        if self.context_projection is not None:
            global_context_shape = (embedded_shape[0], None, self.global_dim)
            self.context_projection.build(global_context_shape)

        # The gathered keys are one per byte, so their length is the byte length.
        current_shape = pos_embedded_shape
        cross_attention_kv_shape = (current_shape[0], current_shape[1], self.local_dim)

        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            decoder_layer.build(current_shape)
            decoder_output_shape = decoder_layer.compute_output_shape(current_shape)

            # Explicit build, because a lazy first-call build leaves the
            # cross-attention weights unloadable on a .keras reload.
            cross_attention.build(
                decoder_output_shape,
                cross_attention_kv_shape,
                cross_attention_kv_shape,
            )

            cross_norm.build(decoder_output_shape)

            current_shape = decoder_output_shape

        self.layer_norm.build(current_shape)
        self.output_projection.build(current_shape)

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
        x = self.byte_embedding(byte_tokens)

        x = self.positional_embedding(x, training=training)

        if self.context_projection is not None:
            global_context = self.context_projection(global_context)

        attend_mask = causal_attend_mask(x)
        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            x = decoder_layer(x, attention_mask=attend_mask, training=training)

            cross_attended = self._masked_cross_attention(
                x, global_context, patch_ids, cross_attention, training
            )

            # Post-norm around the cross-attention residual.
            x = x + cross_attended
            x = cross_norm(x)

        x = self.layer_norm(x)

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

        prev_patch_ids = ops.maximum(patch_ids - 1, 0)
        gather_idx = ops.expand_dims(prev_patch_ids, axis=-1)
        global_dim = ops.shape(global_context)[-1]
        gather_idx = ops.broadcast_to(gather_idx, (batch_size, seq_len, global_dim))
        position_context = ops.take_along_axis(global_context, gather_idx, axis=1)

        # Zeroing beats masking the row here: a fully masked softmax row is not.
        has_prev = ops.cast(
            ops.expand_dims(ops.greater(patch_ids, 0), axis=-1),
            position_context.dtype,
        )
        position_context = position_context * has_prev

        # keras MultiHeadAttention takes attend semantics at (B, T_q, T_k), and
        # the key axis here is the byte axis.
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
        """Compute output shape.

        :param input_shape: Byte token shape ``(batch, seq_len)``, or a list
            whose first entry is that shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, seq_len, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, list):
            # The byte-token shape carries the batch and sequence axes.
            batch_size = input_shape[0][0]
            seq_len = input_shape[0][1]
        else:
            batch_size = input_shape[0]
            seq_len = input_shape[1]
        return (batch_size, seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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