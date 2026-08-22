"""
SAM 2 memory attention: the stack that conditions a frame on its memory bank.
=============================================================================

Two public classes -- :class:`SAM2MemoryAttentionLayer` and
:class:`SAM2MemoryAttention` -- plus the private rotary attention primitive
they are composed from.

Based on:
---------
- Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos."
- Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
  Embedding."

Key Features:
------------
- Pre-norm blocks, each running self-attention (the frame's own tokens), then
  cross-attention (the memory sequence), then a feed-forward sub-block.
- 2D axial RoPE applied AFTER the head split, on both attention sub-blocks, via
  :class:`~dl_techniques.layers.embedding.axial_rope_2d.AxialRoPE2D`.
- Cross-attention consumes keys and values of width ``kv_in_dim`` (SAM 2's
  compressed ``mem_dim=64`` memory channels) while queries stay at
  ``d_model=256``, and enables ``repeat_k``, broadcasting ONE spatial angle
  table across the ``r`` memory frames stacked along the key axis.

Architecture Overview:
---------------------
1. **SAM2MemoryAttentionLayer** -- self-attention -> cross-attention -> MLP.
2. **SAM2MemoryAttention** -- ``num_layers`` identical blocks, then a final
   layer normalization.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM2.memory_attention import SAM2MemoryAttention
attention = SAM2MemoryAttention(d_model=256, num_layers=4)
conditioned = attention(tokens, memory, memory_pos, num_obj_ptr_tokens=4)
```

Measured caveats:
----------------
- **Positional encoding is re-added at four independently configurable points,
  and the shipped SAM 2.1 setting is ASYMMETRIC between queries and keys.**
  ``pos_enc_at_input=True``, ``pos_enc_at_attn=False``,
  ``pos_enc_at_cross_attn_queries=False``, ``pos_enc_at_cross_attn_keys=True``.
  Every combination runs without a shape error, which is why these are four
  explicit configuration fields rather than one assumed-uniform "add positional
  encoding everywhere".
- **RoPE here is SPATIAL ONLY.** The same angle table is reused for every
  memory frame. A memory frame's temporal identity is carried exclusively by an
  additive per-slot embedding owned by the top-level ``SAM2`` model and folded
  into ``memory_pos`` before it reaches this module. Nothing in this file may
  introduce a temporal term; if it did, no test could tell a spatial mechanism
  from a temporal one.
"""

import math
import keras
from keras import ops
from typing import Any, Dict, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.axial_rope_2d import AxialRoPE2D

# ---------------------------------------------------------------------

# Scale applied to the current frame's positional encoding when it is folded
# into the memory-attention INPUT (`pos_enc_at_input=True`). This is a fixed
# architectural constant of SAM 2, not a tunable, and it is intentionally NOT an
# `__init__` parameter -- exposing it would invite a "harmless" retune that
# silently changes what every downstream layer sees.
#
# NOTE ON THE RATIONALE. An earlier version of this comment justified the 0.1 by
# claiming "the same positional tensor is re-added at full strength inside the
# attention sub-blocks". That is FALSE under the shipped configuration:
# `pos_enc_at_attn=False` and `pos_enc_at_cross_attn_queries=False`, so
# `query_pos` is re-added at NO attention sub-block and this attenuated
# injection is the only place the current frame's positional encoding enters at
# all. The constant's VALUE is unchanged (it is the reference's); only the
# reason given for it was wrong, and a wrong reason is what licenses a later
# "simplification" to 1.0.
_INPUT_POS_ENC_SCALE = 0.1

# DECISION plan-2026-08-22T035419-a11304c8/D-090
#: The SHIPPED memory-attention dropout rate, and the single home of the
#: number. Every class below takes it as its constructor default and
#: ``SAM2.MODEL_VARIANTS`` reads it from here rather than restating ``0.1``.
#: Do NOT write a bare ``0.1`` into the variant table or into any of the three
#: signatures below: a rate restated in two homes drifts silently, and this
#: file already had one hard-wired copy per class with no way to reach it. The
#: model-level knob (``SAM2.from_variant(dropout_rate=...)``, D-090) overrides
#: this per construction; the default stays 0.1 so shipped behaviour is
#: bit-identical. The constructor PARAMETER on all three classes below is now
#: ``dropout_rate`` too (D-130, the `*_dropout_rate` convention wave), so the
#: constant, the parameter, the attribute and the ``get_config`` key all carry
#: one spelling. The old ``dropout`` spelling is gone with no alias: the user
#: ruled that no checkpoint depends on it.
DEFAULT_DROPOUT_RATE: float = 0.1

# ---------------------------------------------------------------------


# DECISION plan-2026-08-04T044628-4c240b4c/D-008
# This class is deliberately PRIVATE and deliberately UNREGISTERED.
#
# Do NOT "promote" it to a public, `@keras.saving.register_keras_serializable()`
# sibling of the two classes below. Two reasons, in order of force:
#   1. It would take the plan's public-class count from 15 to 16, which is that
#      budget's hard STOP trigger -- with six components still unbuilt. A
#      registered key is public surface whether or not the name has a leading
#      underscore.
#   2. Registration buys nothing here. It is constructed unconditionally by its
#      owner's `__init__` from the owner's own config, so it never appears in a
#      serialized config dict and is never rebuilt from one. Its weights are
#      saved and restored by attribute path like any other sub-layer.
# It still carries a full `get_config()` / `compute_output_shape()` so that
# promoting it later, if a second owner ever appears, is a one-line change.
# See decisions.md D-008.
class _SAM2RoPEAttention(keras.layers.Layer):
    """Multi-head attention with 2D axial RoPE applied post-head-split.

    Private implementation detail of :class:`SAM2MemoryAttentionLayer`. It is
    deliberately not part of the package's public surface and is deliberately
    NOT registered for serialization: it is always constructed unconditionally
    by its owning layer's ``__init__`` from that layer's own configuration, so
    it never needs to be rebuilt from a config dict of its own.

    Queries are projected from ``embedding_dim``; keys and values are projected
    from ``kv_in_dim``, which differs from ``embedding_dim`` in SAM 2's memory
    cross-attention (queries are 256-wide frame features, keys/values are
    64-wide compressed memory features).

    **Data flow:**

    .. code-block:: text

        query (B, Nq, E)   key (B, Nk, KV)   value (B, Nk, KV)
             │                  │                  │
          q_proj             k_proj             v_proj      -> internal_dim
             │                  │                  │
          split heads        split heads        split heads
             │                  │                  │
             └──── AxialRoPE2D(q, k, num_k_exclude) ┘        (v is NOT rotated)
                        │                  │                  │
                  softmax(q k^T / sqrt(head_dim)) · v
                                  │
                            merge heads -> out_proj -> (B, Nq, E)

    :param embedding_dim: Width of the query tensor.
    :type embedding_dim: int
    :param num_heads: Number of attention heads. Must divide ``internal_dim``.
    :type num_heads: int
    :param downsample_rate: ``internal_dim = embedding_dim // downsample_rate``.
    :type downsample_rate: int
    :param dropout_rate: Dropout rate applied to the attention weights.
    :type dropout_rate: float
    :param kv_in_dim: Width of the key/value tensors. ``None`` means
        ``embedding_dim``.
    :type kv_in_dim: Optional[int]
    :param rope_theta: Base of the RoPE frequency ladder.
    :type rope_theta: float
    :param feat_sizes: Spatial grid ``(H, W)`` of the query tokens.
    :type feat_sizes: Tuple[int, int]
    :param repeat_k: Broadcast the query grid's angle table across an integer
        number of key blocks (SAM 2's stacked memory frames).
    :type repeat_k: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``embedding_dim`` is not divisible by
        ``downsample_rate``, if ``internal_dim`` is not divisible by
        ``num_heads``, or if the resulting head width is not a multiple of 4
        (required by 2D axial RoPE).
    """

    def __init__(
            self,
            embedding_dim: int = 256,
            num_heads: int = 1,
            downsample_rate: int = 1,
            dropout_rate: float = DEFAULT_DROPOUT_RATE,
            kv_in_dim: Optional[int] = None,
            rope_theta: float = 10000.0,
            feat_sizes: Sequence[int] = (64, 64),
            repeat_k: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embedding_dim % downsample_rate != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"downsample_rate ({downsample_rate})"
            )
        internal_dim = embedding_dim // downsample_rate
        if internal_dim % num_heads != 0:
            raise ValueError(
                f"internal_dim ({internal_dim} = {embedding_dim} // "
                f"{downsample_rate}) must be divisible by num_heads "
                f"({num_heads})"
            )
        head_dim = internal_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"head width ({head_dim}) must be divisible by 4 for 2D axial "
                f"RoPE (head_dim // 4 frequency bands per axis)"
            )

        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.downsample_rate = int(downsample_rate)
        self.dropout_rate = float(dropout_rate)
        self.kv_in_dim = int(kv_in_dim) if kv_in_dim is not None else None
        self.rope_theta = float(rope_theta)
        self.feat_sizes = (int(feat_sizes[0]), int(feat_sizes[1]))
        self.repeat_k = bool(repeat_k)

        # Derived, non-config.
        self.internal_dim = internal_dim
        self.head_dim = head_dim
        self.kv_dim = self.kv_in_dim if self.kv_in_dim is not None else self.embedding_dim
        self._scale = 1.0 / math.sqrt(float(head_dim))

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.q_proj = keras.layers.Dense(self.internal_dim, name="q_proj")
        self.k_proj = keras.layers.Dense(self.internal_dim, name="k_proj")
        self.v_proj = keras.layers.Dense(self.internal_dim, name="v_proj")
        self.out_proj = keras.layers.Dense(self.embedding_dim, name="out_proj")
        self.attn_dropout = keras.layers.Dropout(self.dropout_rate, name="attn_dropout")
        self.rope = AxialRoPE2D(
            head_dim=self.head_dim,
            feat_shape=self.feat_sizes,
            theta=self.rope_theta,
            repeat_k=self.repeat_k,
            name="rope",
        )

    def build(
            self,
            query_shape: Tuple[Optional[int], ...],
            key_shape: Optional[Tuple[Optional[int], ...]] = None,
            value_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> None:
        """Build every projection and the RoPE table.

        :param query_shape: ``(batch, num_query_tokens, embedding_dim)``.
        :type query_shape: Tuple[Optional[int], ...]
        :param key_shape: ``(batch, num_key_tokens, kv_in_dim)``. Defaults to
            ``query_shape`` when omitted (self-attention).
        :type key_shape: Optional[Tuple[Optional[int], ...]]
        :param value_shape: Same shape contract as ``key_shape``.
        :type value_shape: Optional[Tuple[Optional[int], ...]]
        """
        if self.built:
            return

        key_shape = tuple(key_shape) if key_shape is not None else tuple(query_shape)
        value_shape = tuple(value_shape) if value_shape is not None else key_shape

        self.q_proj.build(tuple(query_shape))
        self.k_proj.build(key_shape)
        self.v_proj.build(value_shape)
        self.out_proj.build((*tuple(query_shape)[:-1], self.internal_dim))

        num_q = query_shape[-2]
        num_k = key_shape[-2]
        self.rope.build(
            (None, self.num_heads, num_q, self.head_dim),
            (None, self.num_heads, num_k, self.head_dim),
        )
        self.attn_dropout.build(
            (None, self.num_heads, num_q, num_k)
        )

        super().build(tuple(query_shape))

    # -----------------------------------------------------------------
    # head plumbing
    # -----------------------------------------------------------------

    def _split_heads(self, x: Any, num_tokens: int) -> Any:
        """Reshape ``(B, N, internal_dim)`` to ``(B, heads, N, head_dim)``.

        :param x: Projected tensor.
        :type x: Any
        :param num_tokens: Static token count ``N``.
        :type num_tokens: int
        :return: Head-split tensor.
        :rtype: Any
        """
        x = ops.reshape(x, (-1, num_tokens, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: Any, num_tokens: int) -> Any:
        """Reshape ``(B, heads, N, head_dim)`` back to ``(B, N, internal_dim)``.

        :param x: Head-split tensor.
        :type x: Any
        :param num_tokens: Static token count ``N``.
        :type num_tokens: int
        :return: Merged tensor.
        :rtype: Any
        """
        x = ops.transpose(x, (0, 2, 1, 3))
        return ops.reshape(x, (-1, num_tokens, self.internal_dim))

    @staticmethod
    def _static_tokens(x: Any, name: str) -> int:
        """Read a statically-known token count from a rank-3 tensor.

        :param x: Tensor of shape ``(batch, num_tokens, channels)``.
        :type x: Any
        :param name: Human-readable tensor name for the error message.
        :type name: str
        :return: The token count.
        :rtype: int
        :raises ValueError: If the token axis is dynamic.
        """
        num_tokens = x.shape[-2]
        if num_tokens is None:
            raise ValueError(
                f"SAM 2 memory attention requires a STATIC token axis on "
                f"{name}; got a dynamic dimension in shape {x.shape}. Trace "
                f"with a static input signature."
            )
        return int(num_tokens)

    def call(
            self,
            query: Any,
            key: Any,
            value: Any,
            num_k_exclude: int = 0,
            training: Optional[bool] = None,
    ) -> Any:
        """Run rotary multi-head attention.

        :param query: ``(batch, num_query_tokens, embedding_dim)``.
        :type query: Any
        :param key: ``(batch, num_key_tokens, kv_in_dim)``.
        :type key: Any
        :param value: ``(batch, num_key_tokens, kv_in_dim)``.
        :type value: Any
        :param num_k_exclude: Number of trailing key rows to leave unrotated
            (SAM 2's object-pointer tokens carry no spatial position).
        :type num_k_exclude: int
        :param training: Keras training flag; gates the attention dropout.
        :type training: Optional[bool]
        :return: ``(batch, num_query_tokens, embedding_dim)``.
        :rtype: Any
        """
        num_q = self._static_tokens(query, "query")
        num_k = self._static_tokens(key, "key")

        q = self._split_heads(self.q_proj(query), num_q)
        k = self._split_heads(self.k_proj(key), num_k)
        v = self._split_heads(self.v_proj(value), num_k)

        # Rotate q and k only. Values carry content, not position -- rotating
        # them would make the output basis position-dependent.
        q, k = self.rope(q, k, num_k_exclude=num_k_exclude)

        attn = ops.matmul(q * self._scale, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, training=training)

        out = ops.matmul(attn, v)
        return self.out_proj(self._merge_heads(out, num_q))

    def compute_output_shape(
            self,
            query_shape: Tuple[Optional[int], ...],
            key_shape: Optional[Tuple[Optional[int], ...]] = None,
            value_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param query_shape: Query shape.
        :type query_shape: Tuple[Optional[int], ...]
        :param key_shape: Unused; present for the call-signature contract.
        :type key_shape: Optional[Tuple[Optional[int], ...]]
        :param value_shape: Unused; present for the call-signature contract.
        :type value_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(*query_shape[:-1], embedding_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*tuple(query_shape)[:-1], self.embedding_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "downsample_rate": self.downsample_rate,
            "dropout_rate": self.dropout_rate,
            "kv_in_dim": self.kv_in_dim,
            "rope_theta": self.rope_theta,
            "feat_sizes": self.feat_sizes,
            "repeat_k": self.repeat_k,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2MemoryAttentionLayer(keras.layers.Layer):
    """One SAM 2 memory-attention block: self-attn, memory cross-attn, FFN.

    All three sub-blocks are pre-norm with a residual add. Positional encoding
    is re-added at three of this layer's own configurable points (the fourth,
    ``pos_enc_at_input``, belongs to the enclosing
    :class:`SAM2MemoryAttention`).

    **Architecture:**

    .. code-block:: text

        target ──► norm1 ─► (+query_pos if pos_enc_at_attn) ─► self-attn ─┐
           └──────────────────────── + ◄──── dropout1 ◄───────────────────┘
           │
           ├─► norm2 ─► (+query_pos if pos_enc_at_cross_attn_queries) = q ─┐
           │   memory ─► (+memory_pos if pos_enc_at_cross_attn_keys)  = k  ├─► cross-attn
           │   memory ─────────────────────────────────────────────── = v ─┘
           └──────────────────────── + ◄──── dropout2 ◄──────────────────────┘
           │
           ├─► norm3 ─► Dense(dim_feedforward) ─► act ─► dropout ─► Dense(d_model)
           └──────────────────────── + ◄──── dropout3 ◄──────────────────────┘

    Note that the cross-attention VALUES never receive positional encoding under
    any setting — only the keys do. Adding it to values as well would leak
    position into the content the block writes back.

    :param d_model: Query/residual width. Defaults to ``256``.
    :type d_model: int
    :param dim_feedforward: FFN hidden width. Defaults to ``2048``.
    :type dim_feedforward: int
    :param dropout_rate: Dropout rate used by the FFN, the three residual dropouts,
        and both attention sub-blocks. Defaults to ``0.1``.
    :type dropout_rate: float
    :param activation: FFN hidden activation. SAM 2.1 ships ``'relu'``, not the
        transformer-default ``'gelu'``. Defaults to ``'relu'``.
    :type activation: str
    :param pos_enc_at_attn: Add ``query_pos`` to the self-attention queries and
        keys. Shipped value ``False``.
    :type pos_enc_at_attn: bool
    :param pos_enc_at_cross_attn_queries: Add ``query_pos`` to the
        cross-attention queries. Shipped value ``False``.
    :type pos_enc_at_cross_attn_queries: bool
    :param pos_enc_at_cross_attn_keys: Add ``memory_pos`` to the
        cross-attention keys. Shipped value ``True``.
    :type pos_enc_at_cross_attn_keys: bool
    :param num_heads: Attention heads in both sub-blocks. Defaults to ``1``.
    :type num_heads: int
    :param downsample_rate: Attention internal-width divisor. Defaults to ``1``.
    :type downsample_rate: int
    :param rope_theta: RoPE frequency-ladder base. Defaults to ``10000.0``.
    :type rope_theta: float
    :param feat_sizes: Query token grid ``(H, W)``. Defaults to ``(64, 64)``.
    :type feat_sizes: Sequence[int]
    :param kv_in_dim: Width of the memory keys/values consumed by the
        cross-attention. Defaults to ``64`` (SAM 2's ``mem_dim``).
    :type kv_in_dim: int
    :param layer_norm_epsilon: Epsilon of the three layer normalizations.
        Defaults to ``1e-5``.
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim_feedforward`` is not positive, or if the
        attention geometry is invalid (propagated from the attention block).

    Example:
        >>> import numpy as np
        >>> layer = SAM2MemoryAttentionLayer(d_model=32, dim_feedforward=64,
        ...                                  feat_sizes=(2, 2), kv_in_dim=8)
        >>> target = np.zeros((1, 4, 32), dtype="float32")
        >>> memory = np.zeros((1, 8, 8), dtype="float32")
        >>> layer(target, memory).shape
        (1, 4, 32)
    """

    def __init__(
            self,
            d_model: int = 256,
            dim_feedforward: int = 2048,
            dropout_rate: float = DEFAULT_DROPOUT_RATE,
            activation: str = "relu",
            pos_enc_at_attn: bool = False,
            pos_enc_at_cross_attn_queries: bool = False,
            pos_enc_at_cross_attn_keys: bool = True,
            num_heads: int = 1,
            downsample_rate: int = 1,
            rope_theta: float = 10000.0,
            feat_sizes: Sequence[int] = (64, 64),
            kv_in_dim: int = 64,
            layer_norm_epsilon: float = 1e-5,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim_feedforward <= 0:
            raise ValueError(
                f"dim_feedforward must be positive, got {dim_feedforward}"
            )

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_rate = float(dropout_rate)
        self.activation = activation
        self.pos_enc_at_attn = bool(pos_enc_at_attn)
        self.pos_enc_at_cross_attn_queries = bool(pos_enc_at_cross_attn_queries)
        self.pos_enc_at_cross_attn_keys = bool(pos_enc_at_cross_attn_keys)
        self.num_heads = int(num_heads)
        self.downsample_rate = int(downsample_rate)
        self.rope_theta = float(rope_theta)
        self.feat_sizes = (int(feat_sizes[0]), int(feat_sizes[1]))
        self.kv_in_dim = int(kv_in_dim)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        # Resolved once so a test can read the SAME callable the FFN uses,
        # rather than re-deriving it from the config string.
        self.activation_fn = keras.activations.get(self.activation)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.self_attn = _SAM2RoPEAttention(
            embedding_dim=self.d_model,
            num_heads=self.num_heads,
            downsample_rate=self.downsample_rate,
            dropout_rate=self.dropout_rate,
            kv_in_dim=None,
            rope_theta=self.rope_theta,
            feat_sizes=self.feat_sizes,
            repeat_k=False,
            name="self_attn",
        )
        self.cross_attn_image = _SAM2RoPEAttention(
            embedding_dim=self.d_model,
            num_heads=self.num_heads,
            downsample_rate=self.downsample_rate,
            dropout_rate=self.dropout_rate,
            kv_in_dim=self.kv_in_dim,
            rope_theta=self.rope_theta,
            feat_sizes=self.feat_sizes,
            # The memory key sequence stacks `r` spatial frames, so ONE spatial
            # angle table is broadcast across all of them. Frames are told apart
            # by an additive temporal embedding folded into `memory_pos`, never
            # by the rotation.
            repeat_k=True,
            name="cross_attn_image",
        )
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm1")
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm2")
        self.norm3 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm3")
        self.linear1 = keras.layers.Dense(self.dim_feedforward, name="linear1")
        self.linear2 = keras.layers.Dense(self.d_model, name="linear2")
        self.ffn_dropout = keras.layers.Dropout(self.dropout_rate, name="ffn_dropout")
        self.dropout1 = keras.layers.Dropout(self.dropout_rate, name="dropout1")
        self.dropout2 = keras.layers.Dropout(self.dropout_rate, name="dropout2")
        self.dropout3 = keras.layers.Dropout(self.dropout_rate, name="dropout3")

    def build(
            self,
            target_shape: Tuple[Optional[int], ...],
            memory_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> None:
        """Build every sub-layer explicitly.

        :param target_shape: ``(batch, num_query_tokens, d_model)``.
        :type target_shape: Tuple[Optional[int], ...]
        :param memory_shape: ``(batch, num_memory_tokens, kv_in_dim)``. Defaults
            to a single-frame memory of the query grid's size when omitted.
        :type memory_shape: Optional[Tuple[Optional[int], ...]]
        :raises ValueError: If ``target_shape`` is not rank-3, if its width does
            not equal ``d_model``, or if ``memory_shape``'s width does not equal
            ``kv_in_dim``.
        """
        if self.built:
            return

        target_shape = tuple(target_shape)
        if len(target_shape) != 3:
            raise ValueError(
                f"target must be rank-3 (batch, num_tokens, d_model), got "
                f"shape {target_shape}"
            )
        if target_shape[-1] is not None and target_shape[-1] != self.d_model:
            raise ValueError(
                f"target width ({target_shape[-1]}) must equal d_model "
                f"({self.d_model})"
            )

        if memory_shape is None:
            memory_shape = (
                target_shape[0], target_shape[1], self.kv_in_dim)
        memory_shape = tuple(memory_shape)
        if memory_shape[-1] is not None and memory_shape[-1] != self.kv_in_dim:
            raise ValueError(
                f"memory width ({memory_shape[-1]}) must equal kv_in_dim "
                f"({self.kv_in_dim})"
            )

        self.norm1.build(target_shape)
        self.self_attn.build(target_shape, target_shape, target_shape)
        self.dropout1.build(target_shape)

        self.norm2.build(target_shape)
        self.cross_attn_image.build(target_shape, memory_shape, memory_shape)
        self.dropout2.build(target_shape)

        self.norm3.build(target_shape)
        self.linear1.build(target_shape)
        hidden_shape = (*target_shape[:-1], self.dim_feedforward)
        self.ffn_dropout.build(hidden_shape)
        self.linear2.build(hidden_shape)
        self.dropout3.build(target_shape)

        super().build(target_shape)

    def call(
            self,
            target: Any,
            memory: Any,
            query_pos: Optional[Any] = None,
            memory_pos: Optional[Any] = None,
            num_k_exclude: int = 0,
            training: Optional[bool] = None,
    ) -> Any:
        """Run the three sub-blocks.

        :param target: Current-frame tokens, ``(batch, num_query_tokens,
            d_model)``.
        :type target: Any
        :param memory: Memory sequence, ``(batch, num_memory_tokens,
            kv_in_dim)``.
        :type memory: Any
        :param query_pos: Positional encoding for ``target``, same shape.
            ``None`` disables every ``query_pos`` injection regardless of the
            boolean settings.
        :type query_pos: Optional[Any]
        :param memory_pos: Positional encoding for ``memory``, same shape.
        :type memory_pos: Optional[Any]
        :param num_k_exclude: Trailing memory tokens excluded from RoPE (the
            object-pointer tail).
        :type num_k_exclude: int
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(batch, num_query_tokens, d_model)``.
        :rtype: Any
        """
        # --- self-attention -------------------------------------------------
        normed = self.norm1(target)
        if self.pos_enc_at_attn and query_pos is not None:
            attn_in = normed + query_pos
        else:
            attn_in = normed
        # Queries and keys share the positional treatment; the VALUE is always
        # the un-positioned normalized tensor.
        delta = self.self_attn(attn_in, attn_in, normed, training=training)
        target = target + self.dropout1(delta, training=training)

        # --- memory cross-attention ------------------------------------------
        normed = self.norm2(target)
        if self.pos_enc_at_cross_attn_queries and query_pos is not None:
            cross_q = normed + query_pos
        else:
            cross_q = normed
        if self.pos_enc_at_cross_attn_keys and memory_pos is not None:
            cross_k = memory + memory_pos
        else:
            cross_k = memory
        delta = self.cross_attn_image(
            cross_q, cross_k, memory,
            num_k_exclude=num_k_exclude, training=training,
        )
        target = target + self.dropout2(delta, training=training)

        # --- feed-forward -----------------------------------------------------
        normed = self.norm3(target)
        hidden = self.ffn_dropout(
            self.activation_fn(self.linear1(normed)), training=training)
        target = target + self.dropout3(self.linear2(hidden), training=training)
        return target

    def compute_output_shape(
            self,
            target_shape: Tuple[Optional[int], ...],
            memory_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param target_shape: Target shape.
        :type target_shape: Tuple[Optional[int], ...]
        :param memory_shape: Unused; present for the call-signature contract.
        :type memory_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(*target_shape[:-1], d_model)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*tuple(target_shape)[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dim_feedforward": self.dim_feedforward,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "pos_enc_at_attn": self.pos_enc_at_attn,
            "pos_enc_at_cross_attn_queries": self.pos_enc_at_cross_attn_queries,
            "pos_enc_at_cross_attn_keys": self.pos_enc_at_cross_attn_keys,
            "num_heads": self.num_heads,
            "downsample_rate": self.downsample_rate,
            "rope_theta": self.rope_theta,
            "feat_sizes": self.feat_sizes,
            "kv_in_dim": self.kv_in_dim,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SAM2MemoryAttention(keras.layers.Layer):
    """The full SAM 2 memory-attention stack.

    Conditions a frame's features on a memory sequence built from previous
    frames and object-pointer tokens. It stacks ``num_layers`` identical
    :class:`SAM2MemoryAttentionLayer` blocks and applies a final layer
    normalization.

    **Architecture:**

    .. code-block:: text

        current_features ─(+ 0.1 * current_pos if pos_enc_at_input)─┐
                                                                    ▼
            memory, memory_pos ──────────►  layer 1 ... layer N  ──► norm ──► out

    ``num_obj_ptr_tokens`` is threaded to every block as the RoPE exclusion
    count: object pointers sit at the TAIL of the memory sequence and carry no
    spatial position, so rotating them would encode a position they do not have.

    :param d_model: Feature width of the current frame. Defaults to ``256``.
    :type d_model: int
    :param num_layers: Number of stacked blocks. Defaults to ``4``.
    :type num_layers: int
    :param pos_enc_at_input: Fold ``current_pos`` into the stack input (scaled
        by a fixed ``0.1``). Shipped value ``True``.
    :type pos_enc_at_input: bool
    :param dim_feedforward: Per-block FFN hidden width. Defaults to ``2048``.
    :type dim_feedforward: int
    :param dropout_rate: Per-block dropout rate. Defaults to ``0.1``.
    :type dropout_rate: float
    :param activation: Per-block FFN activation. Defaults to ``'relu'``.
    :type activation: str
    :param pos_enc_at_attn: Per-block self-attention positional injection.
        Shipped value ``False``.
    :type pos_enc_at_attn: bool
    :param pos_enc_at_cross_attn_queries: Per-block cross-attention query
        injection. Shipped value ``False``.
    :type pos_enc_at_cross_attn_queries: bool
    :param pos_enc_at_cross_attn_keys: Per-block cross-attention key injection.
        Shipped value ``True``.
    :type pos_enc_at_cross_attn_keys: bool
    :param num_heads: Per-block attention heads. Defaults to ``1``.
    :type num_heads: int
    :param downsample_rate: Per-block attention width divisor. Defaults to ``1``.
    :type downsample_rate: int
    :param rope_theta: RoPE frequency-ladder base. Defaults to ``10000.0``.
    :type rope_theta: float
    :param feat_sizes: Query token grid ``(H, W)``. Defaults to ``(64, 64)``.
    :type feat_sizes: Sequence[int]
    :param kv_in_dim: Memory key/value width. Defaults to ``64``.
    :type kv_in_dim: int
    :param layer_norm_epsilon: Epsilon of every layer normalization. Defaults to
        ``1e-5``.
    :type layer_norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``num_layers`` is not positive, or if a per-block
        parameter is invalid (propagated from the block).

    Example:
        >>> import numpy as np
        >>> stack = SAM2MemoryAttention(d_model=32, num_layers=2,
        ...                             dim_feedforward=64, feat_sizes=(2, 2),
        ...                             kv_in_dim=8)
        >>> features = np.zeros((1, 4, 32), dtype="float32")
        >>> memory = np.zeros((1, 8, 8), dtype="float32")
        >>> stack(features, memory).shape
        (1, 4, 32)
    """

    def __init__(
            self,
            d_model: int = 256,
            num_layers: int = 4,
            pos_enc_at_input: bool = True,
            dim_feedforward: int = 2048,
            dropout_rate: float = DEFAULT_DROPOUT_RATE,
            activation: str = "relu",
            pos_enc_at_attn: bool = False,
            pos_enc_at_cross_attn_queries: bool = False,
            pos_enc_at_cross_attn_keys: bool = True,
            num_heads: int = 1,
            downsample_rate: int = 1,
            rope_theta: float = 10000.0,
            feat_sizes: Sequence[int] = (64, 64),
            kv_in_dim: int = 64,
            layer_norm_epsilon: float = 1e-5,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        # Store ALL configuration parameters.
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.pos_enc_at_input = bool(pos_enc_at_input)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_rate = float(dropout_rate)
        self.activation = activation
        self.pos_enc_at_attn = bool(pos_enc_at_attn)
        self.pos_enc_at_cross_attn_queries = bool(pos_enc_at_cross_attn_queries)
        self.pos_enc_at_cross_attn_keys = bool(pos_enc_at_cross_attn_keys)
        self.num_heads = int(num_heads)
        self.downsample_rate = int(downsample_rate)
        self.rope_theta = float(rope_theta)
        self.feat_sizes = (int(feat_sizes[0]), int(feat_sizes[1]))
        self.kv_in_dim = int(kv_in_dim)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.layers = [
            SAM2MemoryAttentionLayer(
                d_model=self.d_model,
                dim_feedforward=self.dim_feedforward,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                pos_enc_at_attn=self.pos_enc_at_attn,
                pos_enc_at_cross_attn_queries=self.pos_enc_at_cross_attn_queries,
                pos_enc_at_cross_attn_keys=self.pos_enc_at_cross_attn_keys,
                num_heads=self.num_heads,
                downsample_rate=self.downsample_rate,
                rope_theta=self.rope_theta,
                feat_sizes=self.feat_sizes,
                kv_in_dim=self.kv_in_dim,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name=f"layer_{index}",
            )
            for index in range(self.num_layers)
        ]
        self.norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="norm")

    def build(
            self,
            features_shape: Tuple[Optional[int], ...],
            memory_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> None:
        """Build every block and the output normalization.

        :param features_shape: ``(batch, num_query_tokens, d_model)``.
        :type features_shape: Tuple[Optional[int], ...]
        :param memory_shape: ``(batch, num_memory_tokens, kv_in_dim)``.
        :type memory_shape: Optional[Tuple[Optional[int], ...]]
        """
        if self.built:
            return

        features_shape = tuple(features_shape)
        memory_shape = tuple(memory_shape) if memory_shape is not None else None
        for block in self.layers:
            block.build(features_shape, memory_shape)
        self.norm.build(features_shape)

        logger.debug(
            "SAM2MemoryAttention built: d_model=%d num_layers=%d "
            "pos_enc_at_input=%s feat_sizes=%s kv_in_dim=%d",
            self.d_model, self.num_layers, self.pos_enc_at_input,
            self.feat_sizes, self.kv_in_dim,
        )
        super().build(features_shape)

    def call(
            self,
            features: Any,
            memory: Any,
            features_pos: Optional[Any] = None,
            memory_pos: Optional[Any] = None,
            num_obj_ptr_tokens: int = 0,
            training: Optional[bool] = None,
    ) -> Any:
        """Condition ``features`` on ``memory``.

        :param features: Current-frame tokens, ``(batch, H * W, d_model)``.
        :type features: Any
        :param memory: Memory sequence, ``(batch, num_memory_tokens,
            kv_in_dim)``, with any object-pointer tokens at the TAIL.
        :type memory: Any
        :param features_pos: Positional encoding for ``features``, same shape.
        :type features_pos: Optional[Any]
        :param memory_pos: Positional encoding for ``memory``, same shape. Any
            temporal embedding must already be folded into this tensor.
        :type memory_pos: Optional[Any]
        :param num_obj_ptr_tokens: Number of trailing object-pointer tokens in
            ``memory``; threaded to every block as the RoPE exclusion count.
        :type num_obj_ptr_tokens: int
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``(batch, H * W, d_model)``.
        :rtype: Any
        """
        output = features
        if self.pos_enc_at_input and features_pos is not None:
            output = output + _INPUT_POS_ENC_SCALE * features_pos

        for block in self.layers:
            output = block(
                output,
                memory,
                query_pos=features_pos,
                memory_pos=memory_pos,
                num_k_exclude=num_obj_ptr_tokens,
                training=training,
            )
        return self.norm(output)

    def compute_output_shape(
            self,
            features_shape: Tuple[Optional[int], ...],
            memory_shape: Optional[Tuple[Optional[int], ...]] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, derived from stored config.

        :param features_shape: Feature shape.
        :type features_shape: Tuple[Optional[int], ...]
        :param memory_shape: Unused; present for the call-signature contract.
        :type memory_shape: Optional[Tuple[Optional[int], ...]]
        :return: ``(*features_shape[:-1], d_model)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*tuple(features_shape)[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "pos_enc_at_input": self.pos_enc_at_input,
            "dim_feedforward": self.dim_feedforward,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "pos_enc_at_attn": self.pos_enc_at_attn,
            "pos_enc_at_cross_attn_queries": self.pos_enc_at_cross_attn_queries,
            "pos_enc_at_cross_attn_keys": self.pos_enc_at_cross_attn_keys,
            "num_heads": self.num_heads,
            "downsample_rate": self.downsample_rate,
            "rope_theta": self.rope_theta,
            "feat_sizes": self.feat_sizes,
            "kv_in_dim": self.kv_in_dim,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------
