"""Reference-conditioned set fusion blocks for BurstDP.

Two sibling fusion-block kinds — both share the contract
``call(ref_tokens, aux_tokens, aux_mask, training=None) -> ref_tokens``:

1. :class:`BurstFusionBlock` (``fusion_type="custom"``, the default):
   masked cross-attention from ref-as-query to flat(aux) key/values, plus
   a self-attention pre-step and FFN post-step. Per-token attention,
   variable-N safe.

2. :class:`BurstFusionBlockAdaLN` (``fusion_type="adaln"``): modulation-
   based fusion via :class:`AdaLNZeroConditionalBlock`. Aux tokens are
   masked-mean-pooled to a single ``(B, 1, D)`` summary vector, which
   conditions self-attention on ``ref`` through AdaLN shift/scale/gate.
   Lower capacity than full cross-attention — exists as an ablation lever
   for the H1/H3 hypothesis pair (see ``research/new/
   2026_burst_conditioned_dense_prediction.md``).

Both blocks are permutation-invariant over the auxiliary set: the first
by attention-with-flat-mask, the second by mask-aware mean pooling.

Note: ``PerceiverTransformerLayer`` was considered as a third variant and
rejected — its public ``call(query, kv_input, training)`` accepts no
``attention_mask`` parameter, so padded aux slots would leak into the
softmax. See ``plans/plan_2026-05-19_39a6a454/findings/
01-perceiver-mask-blocker.md`` (decision D-003).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras
from keras import layers, ops

from dl_techniques.layers.attention.multi_head_cross_attention import MultiHeadCrossAttention
from dl_techniques.layers.transformers.adaln_zero import AdaLNZeroConditionalBlock


@keras.saving.register_keras_serializable()
class BurstFusionBlock(keras.layers.Layer):
    """One ref-conditioned set fusion block.

    Args:
        dim: Token embedding dimension. Must match the shared encoder.
        num_heads: Attention heads for both self and cross attention.
        mlp_ratio: FFN expansion ratio.
        dropout_rate: Generic dropout applied inside FFN and on attention output.
        attention_dropout_rate: Dropout on attention weights.
        activation: FFN activation, defaults to ``gelu``.
        **kwargs: Layer base kwargs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be a positive multiple of num_heads ({num_heads})"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.activation = activation

        self.norm_self = layers.LayerNormalization(name="norm_self")
        self.self_attn = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            name="self_attn",
        )

        self.norm_cross_q = layers.LayerNormalization(name="norm_cross_q")
        self.norm_cross_kv = layers.LayerNormalization(name="norm_cross_kv")
        self.cross_attn = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            name="cross_attn",
        )

        self.norm_ffn = layers.LayerNormalization(name="norm_ffn")
        hidden = max(1, int(self.dim * self.mlp_ratio))
        self.ffn_dense_1 = layers.Dense(hidden, name="ffn_1")
        self.ffn_act = layers.Activation(self.activation, name="ffn_act")
        self.ffn_dense_2 = layers.Dense(self.dim, name="ffn_2")
        self.dropout = (
            layers.Dropout(self.dropout_rate, name="dropout")
            if self.dropout_rate > 0.0
            else None
        )

    def build(self, input_shape: Any) -> None:
        # input_shape is expected to be a list/tuple of three shapes:
        # ref_shape (B, T, D), aux_shape (B, N, T, D), mask_shape (B, N).
        # We don't strictly need to consume input_shape — Keras will build
        # sublayers lazily on first call — but call super at the end for
        # checkpoint correctness.
        super().build(input_shape)

    def _build_flat_aux_mask(
        self,
        aux_mask: keras.KerasTensor,
        token_count: int,
    ) -> keras.KerasTensor:
        """Repeat per-view validity to per-token validity.

        aux_mask: (B, N)   in {0, 1}
        return:   (B, N * T) where each view's validity is repeated T times.
        """
        # (B, N) -> (B, N, 1) -> (B, N, T) -> (B, N*T)
        expanded = ops.expand_dims(aux_mask, axis=-1)
        expanded = ops.broadcast_to(
            expanded,
            (ops.shape(aux_mask)[0], ops.shape(aux_mask)[1], token_count),
        )
        return ops.reshape(expanded, (ops.shape(aux_mask)[0], -1))

    def _any_valid(self, aux_mask: keras.KerasTensor) -> keras.KerasTensor:
        """Per-sample 0/1 indicating whether any aux view is valid in the batch."""
        # (B, N) -> (B,)
        any_v = ops.cast(ops.sum(aux_mask, axis=-1) > 0, dtype="float32")
        return any_v

    def call(
        self,
        ref_tokens: keras.KerasTensor,
        aux_tokens: keras.KerasTensor,
        aux_mask: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run one fusion block.

        Args:
            ref_tokens: ``(B, T, D)`` reference tokens.
            aux_tokens: ``(B, N, T, D)`` auxiliary tokens.
            aux_mask:   ``(B, N)`` float 0/1 view validity (1 = real view).
            training:   Keras training flag.

        Returns:
            Updated reference tokens, shape ``(B, T, D)``.
        """
        # --- 1. Self-attention on reference tokens ---
        x = ref_tokens
        h = self.norm_self(x)
        h = self.self_attn(h, training=training)
        if self.dropout is not None:
            h = self.dropout(h, training=training)
        x = x + h

        # --- 2. Cross-attention reference -> aux ---
        b = ops.shape(aux_tokens)[0]
        n = ops.shape(aux_tokens)[1]
        t = ops.shape(aux_tokens)[2]
        d = ops.shape(aux_tokens)[3]

        kv_flat = ops.reshape(aux_tokens, (b, n * t, d))
        # Apply mask in feature space too: padded views were zeros from the
        # encoder gate already, but normalising over them would still inject
        # learned bias. Multiply by the broadcast mask once more for safety.
        flat_mask = self._build_flat_aux_mask(aux_mask, token_count=ref_tokens.shape[1])
        # flat_mask: (B, N*T)
        kv_flat = kv_flat * ops.expand_dims(flat_mask, axis=-1)

        q = self.norm_cross_q(x)
        kv = self.norm_cross_kv(kv_flat)
        attended = self.cross_attn(q, kv_input=kv, attention_mask=flat_mask, training=training)

        # If a sample has zero valid aux views, the cross-attn output is
        # undefined (all-masked softmax -> NaN). Gate by per-sample any-valid.
        any_v = self._any_valid(aux_mask)  # (B,)
        any_v = ops.reshape(any_v, (-1, 1, 1))
        attended = attended * ops.cast(any_v, attended.dtype)

        if self.dropout is not None:
            attended = self.dropout(attended, training=training)
        x = x + attended

        # --- 3. Position-wise FFN ---
        h = self.norm_ffn(x)
        h = self.ffn_dense_1(h)
        h = self.ffn_act(h)
        if self.dropout is not None:
            h = self.dropout(h, training=training)
        h = self.ffn_dense_2(h)
        if self.dropout is not None:
            h = self.dropout(h, training=training)
        x = x + h

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "activation": self.activation,
            }
        )
        return config


# ---------------------------------------------------------------------------
# AdaLN-conditioned sibling block
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BurstFusionBlockAdaLN(keras.layers.Layer):
    """AdaLN-Zero conditioned fusion (sibling of :class:`BurstFusionBlock`).

    Aux tokens are summarised to a single ``(B, 1, D)`` mean vector
    (mask-aware over the ``(N, T)`` flattened axes) and used as the
    conditioning input ``c`` to an :class:`AdaLNZeroConditionalBlock`. The
    block performs self-attention on ``ref_tokens`` modulated by ``c``,
    rather than cross-attention to per-token aux features.

    This is a deliberately lower-capacity fusion path than
    :class:`BurstFusionBlock` — useful as an ablation lever for the
    "attention collapse" hypothesis (H3): if AdaLN matches `custom` on
    benchmarks, per-token cross-attention to aux is decorative rather than
    load-bearing.

    Same call contract as :class:`BurstFusionBlock`:
    ``call(ref_tokens, aux_tokens, aux_mask, training=None) -> ref_tokens``.

    Args:
        dim: Token embedding dimension. Must match the shared encoder.
        num_heads: Heads for the inner AdaLN self-attention.
        mlp_ratio: FFN expansion ratio for the AdaLN MLP sub-block.
        dropout_rate: Generic dropout applied inside the AdaLN block.
        attention_dropout_rate: Currently NOT used (kept for ctor signature
            parity with :class:`BurstFusionBlock`; AdaLN's default MHA
            applies a single ``dropout`` across attention + projection).
        activation: Currently NOT used (the AdaLN block uses the default
            GELU FFN activation when ``ffn_type=None``). Kept for ctor
            signature parity.
        **kwargs: Layer base kwargs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        activation: str = "gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be a positive multiple of num_heads ({num_heads})"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.activation = activation

        # DECISION plan_2026-05-19_39a6a454/D-002
        # `use_causal_mask=False` is REQUIRED for BurstDP — reference patches
        # have no causal / temporal order. AdaLNZeroConditionalBlock defaults
        # `use_causal_mask=True` (matching its DiT/LeWM origin); accepting that
        # default here would silently triangle-mask half the ref tokens with
        # no error, producing a model that trains but is semantically wrong.
        # See plans/plan_2026-05-19_39a6a454/decisions.md D-002.
        self.adaln = AdaLNZeroConditionalBlock(
            dim=self.dim,
            num_heads=self.num_heads,
            dim_head=self.dim // self.num_heads,
            mlp_dim=max(1, int(self.dim * self.mlp_ratio)),
            dropout=self.dropout_rate,
            use_causal_mask=False,
            name="adaln",
        )

    def build(self, input_shape: Any) -> None:
        # Inputs come as positional kwargs to call(); Keras may pass either
        # a single shape or a list. We do not strictly need to consume
        # input_shape — sublayers build on first call. Super at the end for
        # checkpoint correctness.
        super().build(input_shape)

    def _masked_aux_mean(
        self,
        aux_tokens: keras.KerasTensor,
        aux_mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Mean-pool ``aux_tokens (B, N, T, D)`` over ``(N, T)`` honoring ``aux_mask (B, N)``.

        Padded views (mask=0) contribute zero. Per-sample any-valid gate
        zeros the output when N=0 so AdaLN at training time still produces
        ``shift=scale=gate≈0`` and the block behaves as identity in ref
        (matching the cross-attention block's N=0 contract).

        Returns: ``(B, 1, D)`` — broadcastable conditioning vector for AdaLN.
        """
        # Mask in feature space: zero out padded views.
        m_b_n_1_1 = ops.reshape(aux_mask, (ops.shape(aux_tokens)[0], ops.shape(aux_tokens)[1], 1, 1))
        aux_zeroed = aux_tokens * ops.cast(m_b_n_1_1, aux_tokens.dtype)  # (B, N, T, D)

        # Sum over N and T -> (B, D).
        summed = ops.sum(ops.sum(aux_zeroed, axis=2), axis=1)

        # Divide by valid token count = sum(mask) * T (floor 1 to guard N=0).
        t = ops.cast(ops.shape(aux_tokens)[2], summed.dtype)
        n_valid_views = ops.sum(aux_mask, axis=-1)            # (B,)
        n_valid_tokens = ops.cast(n_valid_views, summed.dtype) * t
        n_valid_tokens = ops.maximum(n_valid_tokens, 1.0)
        c = summed / ops.expand_dims(n_valid_tokens, axis=-1)  # (B, D)

        # Any-valid gate (zero c entirely when no aux view is valid).
        any_valid = ops.cast(n_valid_views > 0, c.dtype)       # (B,)
        c = c * ops.expand_dims(any_valid, axis=-1)            # (B, D)

        # Broadcast to (B, 1, D).
        return ops.expand_dims(c, axis=1)

    def call(
        self,
        ref_tokens: keras.KerasTensor,
        aux_tokens: keras.KerasTensor,
        aux_mask: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward.

        Args:
            ref_tokens: ``(B, T, D)`` reference tokens.
            aux_tokens: ``(B, N, T, D)`` auxiliary tokens.
            aux_mask:   ``(B, N)`` float 0/1 view validity.
            training:   Keras training flag.

        Returns:
            Updated reference tokens, shape ``(B, T, D)``.
        """
        c = self._masked_aux_mean(aux_tokens, aux_mask)  # (B, 1, D)
        return self.adaln([ref_tokens, c], training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "activation": self.activation,
            }
        )
        return config
