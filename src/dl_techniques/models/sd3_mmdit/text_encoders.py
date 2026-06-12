"""
Faithful from-scratch SD3 text encoders (Keras 3 port): CLIP / OpenCLIP / T5.

This module provides architecture-faithful, **from-scratch** re-implementations of
the three text towers Stable Diffusion 3 conditions on:

- :class:`CLIPTextEncoder`      -- the OpenAI CLIP ViT-L/14 text tower
  (token + learned-absolute positional embedding, a stack of pre-LN transformer
  encoder layers with QuickGELU MLPs, a CAUSAL self-attention mask, a final
  LayerNorm, and a pooled-EOS text projection).
- :class:`OpenCLIPTextEncoder` -- the larger OpenCLIP ViT-bigG text tower
  (identical structure, larger dims, standard GELU instead of QuickGELU). It is a
  real subclass of :class:`CLIPTextEncoder` (mirroring the PyTorch
  ``class OpenCLIP(CLIP)``) that only overrides the constructor defaults and the
  activation.
- :class:`T5Encoder`           -- the T5-v1.1 XXL encoder
  (shared token embedding, a stack of pre-RMSNorm self-attention + gated-GELU FFN
  blocks, T5 RELATIVE-POSITION-BUCKET attention bias shared from the first block,
  NO ``1/sqrt(d)`` attention scaling, a final RMSNorm). Bidirectional bucketing
  (this is an *encoder*).

**Faithfulness scope (DECISION plan_2026-06-12_dfce0712/D-009).** "Faithful" here
means *architecture + correct forward shapes + serialization round-trip*, NOT
weight parity: there is no pretrained-weight loading path in this repo, so the
encoders are randomly initialized. All three consume INTEGER token-id tensors
``(B, L)`` and an optional ``attention_mask (B, L)`` of 1/0 (1 = keep). No
tokenizer is provided (out of scope per state.md); callers pass raw ids.

**Return contracts** (consumed by the step-10 pipeline):

- ``CLIPTextEncoder`` / ``OpenCLIPTextEncoder`` return a ``dict``:
    - ``"pooled"``       ``(B, embed_dim)``  -- ``text_projection`` applied to the
      EOS-token hidden state (EOS position = ``argmax(token_ids, axis=-1)``, the
      OpenAI-CLIP convention where the EOS id is the largest id in the vocab).
    - ``"last_hidden"``  ``(B, L, embed_dim)`` -- the full sequence AFTER the final
      LayerNorm (pre-projection).
    - ``"penultimate"``  ``(B, L, embed_dim)`` -- the output of the SECOND-TO-LAST
      encoder layer (what SD3 actually conditions on, alongside ``pooled``).
- ``T5Encoder`` returns the sequence tensor ``(B, L, embed_dim)`` directly.

PyTorch references (faithfully ported, structure only):
``openai/CLIP`` ``model.py`` (``Transformer`` / ``ResidualAttentionBlock``),
``open_clip`` ``transformer.py``, HuggingFace ``transformers`` ``T5Stack`` /
``T5Attention`` / ``T5LayerFF`` (gated-GELU ``T5DenseGatedActDense``).
"""

import keras
from keras import ops
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------


def _quick_gelu(x: keras.KerasTensor) -> keras.KerasTensor:
    """QuickGELU: ``x * sigmoid(1.702 * x)`` (the OpenAI-CLIP activation).

    :param x: Input tensor.
    :type x: keras.KerasTensor
    :return: Activated tensor (same shape).
    :rtype: keras.KerasTensor
    """
    return x * ops.sigmoid(1.702 * x)


def _resolve_act(act_fn: str):
    """Map an ``act_fn`` string to a callable.

    :param act_fn: ``"quick_gelu"`` or ``"gelu"``.
    :type act_fn: str
    :return: The activation callable.
    :raises ValueError: If ``act_fn`` is not a known key.
    """
    if act_fn == "quick_gelu":
        return _quick_gelu
    if act_fn == "gelu":
        return lambda x: keras.activations.gelu(x, approximate=False)
    raise ValueError(
        f"act_fn must be 'quick_gelu' or 'gelu', got {act_fn!r}"
    )


# =====================================================================
# A) CLIPTextEncoder (+ OpenCLIPTextEncoder subclass)
# =====================================================================


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class CLIPTextEncoder(keras.Model):
    """OpenAI-CLIP text tower (from-scratch, token-id input, causal-masked).

    **Intent.** Given integer token ids ``(B, L)`` (and an optional padding
    ``attention_mask (B, L)`` of 1/0), produce the three tensors SD3's text
    conditioning needs: a pooled, projected EOS vector ``(B, embed_dim)``, the
    final-LN sequence ``(B, L, embed_dim)``, and the penultimate-layer sequence
    ``(B, L, embed_dim)``.

    **Architecture.**

    .. code-block:: text

        ids (B, L) int
          token_embedding(ids)  +  position_embedding(arange(L))   -> x (B, L, D)
          for layer in num_layers x:
              x = x + attn( LN1(x) )      # MHA, CAUSAL + padding mask
              x = x + mlp( LN2(x) )       # Dense(4D) -> act -> Dense(D)
              (capture x after the (num_layers-2)-th layer as `penultimate`)
          last_hidden = final_layer_norm(x)
          eos = last_hidden[ batch, argmax(ids, axis=-1) ]          # (B, D)
          pooled = text_projection(eos)                             # (B, D)

    The self-attention uses a lower-triangular CAUSAL mask (CLIP text is
    autoregressively masked) combined additively with the padding mask derived
    from ``attention_mask``. Attention is the standard
    :class:`keras.layers.MultiHeadAttention` (a manual SDPA would duplicate it
    for no gain); the causal + padding mask is supplied as a boolean
    ``attention_mask`` of shape ``(B, L, L)``.

    :param vocab_size: Token vocabulary size.
    :type vocab_size: int
    :param embed_dim: Model / embedding width.
    :type embed_dim: int
    :param num_layers: Number of transformer encoder layers.
    :type num_layers: int
    :param num_heads: Attention heads (``embed_dim % num_heads == 0``).
    :type num_heads: int
    :param max_seq_len: Maximum sequence length (learned positional table size).
    :type max_seq_len: int
    :param eps: LayerNorm epsilon.
    :type eps: float
    :param act_fn: MLP activation, ``"quick_gelu"`` (CLIP) or ``"gelu"`` (OpenCLIP).
    :type act_fn: str
    :param kwargs: Forwarded to :class:`keras.Model`.

    :raises ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 77,
        eps: float = 1e-5,
        act_fn: str = "quick_gelu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.eps = eps
        self.act_fn = act_fn
        self._act = _resolve_act(act_fn)

        # --- embeddings -------------------------------------------------
        self.token_embedding = keras.layers.Embedding(
            vocab_size, embed_dim, name="token_embedding"
        )
        self.position_embedding = keras.layers.Embedding(
            max_seq_len, embed_dim, name="position_embedding"
        )

        # --- per-layer sublayers (flat lists, indexed by layer) --------
        self.ln1: List[keras.layers.Layer] = []
        self.attn: List[keras.layers.Layer] = []
        self.ln2: List[keras.layers.Layer] = []
        self.mlp_fc: List[keras.layers.Layer] = []
        self.mlp_proj: List[keras.layers.Layer] = []
        head_dim = embed_dim // num_heads
        for i in range(num_layers):
            self.ln1.append(
                keras.layers.LayerNormalization(epsilon=eps, name=f"ln1_{i}")
            )
            self.attn.append(
                keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=head_dim,
                    name=f"attn_{i}",
                )
            )
            self.ln2.append(
                keras.layers.LayerNormalization(epsilon=eps, name=f"ln2_{i}")
            )
            self.mlp_fc.append(
                keras.layers.Dense(4 * embed_dim, name=f"mlp_fc_{i}")
            )
            self.mlp_proj.append(
                keras.layers.Dense(embed_dim, name=f"mlp_proj_{i}")
            )

        # --- final norm + projection -----------------------------------
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=eps, name="final_layer_norm"
        )
        self.text_projection = keras.layers.Dense(
            embed_dim, use_bias=False, name="text_projection"
        )

        logger.debug(
            f"Initialized CLIPTextEncoder(vocab_size={vocab_size}, "
            f"embed_dim={embed_dim}, num_layers={num_layers}, "
            f"num_heads={num_heads}, max_seq_len={max_seq_len}, "
            f"act_fn={act_fn})"
        )

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer with concrete shapes.

        :param input_shape: Shape of the token-id input ``(B, L)``. May also be a
            list/dict (Functional-model wrapping) -- only the token-id shape is
            used; ``L`` is treated as dynamic.
        :type input_shape: Any
        """
        seq_shape = (None, None, self.embed_dim)  # (B, L, D)

        self.token_embedding.build((None, None))
        self.position_embedding.build((None, None))

        for i in range(self.num_layers):
            self.ln1[i].build(seq_shape)
            # MultiHeadAttention: query/value both the LN1 output.
            self.attn[i].build(query_shape=seq_shape, value_shape=seq_shape)
            self.ln2[i].build(seq_shape)
            self.mlp_fc[i].build(seq_shape)
            self.mlp_proj[i].build((None, None, 4 * self.embed_dim))

        self.final_layer_norm.build(seq_shape)
        self.text_projection.build((None, self.embed_dim))

        super().build(input_shape)

    def _build_attention_mask(
        self, token_ids: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Combine the causal mask with the padding mask into ``(B, L, L)`` bool.

        ``True`` = attend, ``False`` = block (the boolean convention
        :class:`keras.layers.MultiHeadAttention` expects for ``attention_mask``).

        :param token_ids: ``(B, L)`` int ids (used only for L / batch shape).
        :type token_ids: keras.KerasTensor
        :param attention_mask: Optional ``(B, L)`` 1/0 padding mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :return: ``(B, L, L)`` boolean attention mask.
        :rtype: keras.KerasTensor
        """
        L = ops.shape(token_ids)[1]
        # Lower-triangular causal mask (L, L): query i may attend to key j <= i.
        # Built from an arange index comparison rather than ops.tril: tril routes
        # through a tf cond that rejects the Python-bool dtype on a dynamic-L
        # symbolic tensor during the .keras save/load trace. The index form is
        # fully graph-safe (no cond, no Python bool).
        rows = ops.arange(0, L, dtype="int32")[:, None]  # (L, 1) query index
        cols = ops.arange(0, L, dtype="int32")[None, :]  # (1, L) key index
        causal = ops.greater_equal(rows, cols)  # (L, L) bool, True where j <= i
        causal = ops.expand_dims(causal, axis=0)  # (1, L, L)

        if attention_mask is not None:
            # Key padding: a key position j is visible to every query i.
            key_mask = ops.cast(attention_mask, "bool")  # (B, L)
            key_mask = ops.expand_dims(key_mask, axis=1)  # (B, 1, L)
            return ops.logical_and(causal, key_mask)  # (B, L, L)
        return causal

    def call(
        self,
        token_ids: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Encode token ids into pooled / last-hidden / penultimate tensors.

        :param token_ids: Integer ids ``(B, L)``.
        :type token_ids: keras.KerasTensor
        :param attention_mask: Optional ``(B, L)`` 1/0 padding mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: ``{"pooled", "last_hidden", "penultimate"}`` (see class doc).
        :rtype: Dict[str, keras.KerasTensor]
        """
        token_ids = ops.cast(token_ids, "int32")
        L = ops.shape(token_ids)[1]

        # --- token + learned absolute positional embedding -------------
        x = self.token_embedding(token_ids)  # (B, L, D)
        positions = ops.arange(0, L, dtype="int32")
        pos = self.position_embedding(positions)  # (L, D)
        x = x + ops.expand_dims(pos, axis=0)  # broadcast over batch

        attn_mask = self._build_attention_mask(token_ids, attention_mask)

        # --- transformer encoder stack ---------------------------------
        penultimate = None
        for i in range(self.num_layers):
            residual = x
            h = self.ln1[i](x, training=training)
            h = self.attn[i](
                query=h,
                value=h,
                attention_mask=attn_mask,
                training=training,
            )
            x = residual + h

            residual = x
            h = self.ln2[i](x, training=training)
            h = self.mlp_fc[i](h, training=training)
            h = self._act(h)
            h = self.mlp_proj[i](h, training=training)
            x = residual + h

            if i == self.num_layers - 2:
                penultimate = x

        if self.num_layers == 1:
            # Degenerate (test) case: no second-to-last layer exists.
            penultimate = x

        last_hidden = self.final_layer_norm(x, training=training)  # (B, L, D)

        # --- pooled EOS projection -------------------------------------
        # EOS token = the position of the largest token id per row (OpenAI CLIP
        # convention: the EOS id is the max id in the vocabulary).
        eos_idx = ops.argmax(token_ids, axis=-1)  # (B,)
        B = ops.shape(token_ids)[0]
        batch_idx = ops.arange(0, B, dtype=eos_idx.dtype)
        gather_idx = ops.stack([batch_idx, eos_idx], axis=-1)  # (B, 2)
        eos_hidden = ops.take_along_axis(
            last_hidden,
            ops.reshape(eos_idx, (-1, 1, 1)),
            axis=1,
        )  # (B, 1, D)
        eos_hidden = ops.squeeze(eos_hidden, axis=1)  # (B, D)
        pooled = self.text_projection(eos_hidden, training=training)  # (B, D)

        return {
            "pooled": pooled,
            "last_hidden": last_hidden,
            "penultimate": penultimate,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return per-key output shapes (works before ``build``).

        :param input_shape: ``(B, L)`` token-id shape (list/dict tolerated).
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``{"pooled": (B, D), "last_hidden": (B, L, D),
            "penultimate": (B, L, D)}``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        ids_shape = _first_shape(input_shape)
        B = ids_shape[0]
        L = ids_shape[1] if len(ids_shape) > 1 else None
        return {
            "pooled": (B, self.embed_dim),
            "last_hidden": (B, L, self.embed_dim),
            "penultimate": (B, L, self.embed_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return all constructor arguments for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_seq_len": self.max_seq_len,
                "eps": self.eps,
                "act_fn": self.act_fn,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class OpenCLIPTextEncoder(CLIPTextEncoder):
    """OpenCLIP (ViT-bigG) text tower: larger dims, standard GELU.

    A real subclass of :class:`CLIPTextEncoder` (mirroring the PyTorch
    ``class OpenCLIP(CLIP)``). It only overrides the constructor defaults
    (``embed_dim=1280``, ``num_layers=32``, ``num_heads=16``) and forces
    ``act_fn="gelu"`` (standard, non-approximate GELU instead of QuickGELU).
    Everything else -- structure, masking, pooled-EOS contract, serialization --
    is inherited unchanged.

    :param vocab_size: Token vocabulary size.
    :type vocab_size: int
    :param embed_dim: Model width (default ``1280``).
    :type embed_dim: int
    :param num_layers: Number of encoder layers (default ``32``).
    :type num_layers: int
    :param num_heads: Attention heads (default ``16``).
    :type num_heads: int
    :param max_seq_len: Maximum sequence length.
    :type max_seq_len: int
    :param eps: LayerNorm epsilon.
    :type eps: float
    :param kwargs: Forwarded to :class:`CLIPTextEncoder` (``act_fn`` is forced
        to ``"gelu"`` and must not be overridden).
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 1280,
        num_layers: int = 32,
        num_heads: int = 16,
        max_seq_len: int = 77,
        eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        # Force standard GELU; OpenCLIP does not use QuickGELU.
        kwargs.pop("act_fn", None)
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            eps=eps,
            act_fn="gelu",
            **kwargs,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return constructor args; drop ``act_fn`` (fixed to ``"gelu"``)."""
        config = super().get_config()
        config.pop("act_fn", None)
        return config


# =====================================================================
# C) T5Encoder
# =====================================================================


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class T5Encoder(keras.Model):
    """T5-v1.1 encoder (from-scratch, relative-position-bucket bias).

    **Intent.** Given integer token ids ``(B, L)`` (+ optional padding
    ``attention_mask (B, L)`` of 1/0), produce a sequence of hidden states
    ``(B, L, embed_dim)`` -- the T5 conditioning stream SD3 uses.

    **Architecture.**

    .. code-block:: text

        x = shared_embedding(ids)                          # (B, L, D)
        position_bias = block_0.compute_bias(L) + pad_mask # (1/ B, heads, L, L)
        for block in num_layers x:
            # self-attention (NO 1/sqrt(d) scaling; T5 omits it):
            n = RMSNorm(x)
            scores = q @ k^T + position_bias               # (B, heads, L, L)
            x = x + Wo( softmax(scores) @ v )
            # gated-GELU FFN:
            n = RMSNorm(x)
            x = x + wo( gelu(wi_0(n)) * wi_1(n) )
        x = final RMSNorm(x)

    **Relative-position bias (shared).** Only the FIRST block owns the
    ``relative_attention_bias = Embedding(num_buckets, num_heads)`` and computes
    the ``(1, heads, L, L)`` bias via :meth:`_compute_bias`; that bias is threaded
    (passed) to every subsequent block. Bucketing is **bidirectional** (this is an
    encoder): half the buckets cover future positions, half the past; large
    distances are log-scaled. See :meth:`_relative_position_bucket`.

    **Dynamic vs fixed L (DECISION plan_2026-06-12_dfce0712/D-009).** The bias is
    built for the dynamic ``L`` from ``ops.shape`` at call time using ``keras.ops``
    throughout (``arange`` arithmetic + an ``Embedding`` gather), so it handles
    variable sequence length without a fixed ``max_seq_len`` -- no fixed-length
    fallback was needed.

    :param vocab_size: Token vocabulary size.
    :type vocab_size: int
    :param embed_dim: Model width.
    :type embed_dim: int
    :param num_layers: Number of T5 blocks.
    :type num_layers: int
    :param num_heads: Attention heads.
    :type num_heads: int
    :param ff_dim: Gated-FFN inner width.
    :type ff_dim: int
    :param rel_attention_num_buckets: Number of relative-position buckets.
    :type rel_attention_num_buckets: int
    :param rel_attention_max_distance: Max distance for log-scaled buckets.
    :type rel_attention_max_distance: int
    :param eps: RMSNorm epsilon.
    :type eps: float
    :param kwargs: Forwarded to :class:`keras.Model`.
    """

    def __init__(
        self,
        vocab_size: int = 32128,
        embed_dim: int = 4096,
        num_layers: int = 24,
        num_heads: int = 64,
        ff_dim: int = 10240,
        rel_attention_num_buckets: int = 32,
        rel_attention_max_distance: int = 128,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rel_attention_num_buckets = rel_attention_num_buckets
        self.rel_attention_max_distance = rel_attention_max_distance
        self.eps = eps
        # T5 uses a fixed inner attention dim = num_heads * key_value_proj_dim.
        # For T5-v1.1 the inner dim equals embed_dim (key_dim = embed_dim/heads).
        self.inner_dim = embed_dim

        # --- shared token embedding ------------------------------------
        self.shared = keras.layers.Embedding(
            vocab_size, embed_dim, name="shared"
        )

        # --- relative-position bias (owned by the first block only) ----
        self.relative_attention_bias = keras.layers.Embedding(
            rel_attention_num_buckets, num_heads, name="relative_attention_bias"
        )

        # --- per-block sublayers (flat lists) --------------------------
        self.attn_norm: List[keras.layers.Layer] = []
        self.q: List[keras.layers.Layer] = []
        self.k: List[keras.layers.Layer] = []
        self.v: List[keras.layers.Layer] = []
        self.o: List[keras.layers.Layer] = []
        self.ff_norm: List[keras.layers.Layer] = []
        self.wi_0: List[keras.layers.Layer] = []
        self.wi_1: List[keras.layers.Layer] = []
        self.wo: List[keras.layers.Layer] = []
        for i in range(num_layers):
            self.attn_norm.append(RMSNorm(axis=-1, epsilon=eps, name=f"attn_norm_{i}"))
            self.q.append(keras.layers.Dense(self.inner_dim, use_bias=False, name=f"q_{i}"))
            self.k.append(keras.layers.Dense(self.inner_dim, use_bias=False, name=f"k_{i}"))
            self.v.append(keras.layers.Dense(self.inner_dim, use_bias=False, name=f"v_{i}"))
            self.o.append(keras.layers.Dense(embed_dim, use_bias=False, name=f"o_{i}"))
            self.ff_norm.append(RMSNorm(axis=-1, epsilon=eps, name=f"ff_norm_{i}"))
            self.wi_0.append(keras.layers.Dense(ff_dim, use_bias=False, name=f"wi_0_{i}"))
            self.wi_1.append(keras.layers.Dense(ff_dim, use_bias=False, name=f"wi_1_{i}"))
            self.wo.append(keras.layers.Dense(embed_dim, use_bias=False, name=f"wo_{i}"))

        # --- final norm ------------------------------------------------
        self.final_norm = RMSNorm(axis=-1, epsilon=eps, name="final_norm")

        logger.debug(
            f"Initialized T5Encoder(vocab_size={vocab_size}, "
            f"embed_dim={embed_dim}, num_layers={num_layers}, "
            f"num_heads={num_heads}, ff_dim={ff_dim}, "
            f"num_buckets={rel_attention_num_buckets}, "
            f"max_distance={rel_attention_max_distance})"
        )

    def build(self, input_shape: Any) -> None:
        """Build every sub-layer with concrete shapes."""
        seq_shape = (None, None, self.embed_dim)
        inner_shape = (None, None, self.inner_dim)

        self.shared.build((None, None))
        self.relative_attention_bias.build((None, None))

        for i in range(self.num_layers):
            self.attn_norm[i].build(seq_shape)
            self.q[i].build(seq_shape)
            self.k[i].build(seq_shape)
            self.v[i].build(seq_shape)
            self.o[i].build(inner_shape)
            self.ff_norm[i].build(seq_shape)
            self.wi_0[i].build(seq_shape)
            self.wi_1[i].build(seq_shape)
            self.wo[i].build((None, None, self.ff_dim))

        self.final_norm.build(seq_shape)
        super().build(input_shape)

    @staticmethod
    def _relative_position_bucket(
        relative_position: keras.KerasTensor,
        num_buckets: int,
        max_distance: int,
    ) -> keras.KerasTensor:
        """Bidirectional T5 relative-position bucketing.

        Faithful port of HuggingFace ``T5Attention._relative_position_bucket``
        (bidirectional branch): half the buckets are reserved for the sign of the
        relative position; within each half, small distances are exact and large
        distances are grouped log-scaled up to ``max_distance``.

        :param relative_position: ``(L, L)`` integer ``key_pos - query_pos``.
        :type relative_position: keras.KerasTensor
        :param num_buckets: Total buckets (split in half for bidirectional).
        :type num_buckets: int
        :param max_distance: Distance beyond which all positions share a bucket.
        :type max_distance: int
        :return: ``(L, L)`` integer bucket index in ``[0, num_buckets)``.
        :rtype: keras.KerasTensor
        """
        relative_buckets = 0
        num_buckets = num_buckets // 2
        # Bidirectional: positive (key after query) goes to the upper half.
        relative_buckets = relative_buckets + ops.cast(
            ops.greater(relative_position, 0), "int32"
        ) * num_buckets
        relative_position = ops.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = ops.less(relative_position, max_exact)

        # Log-scaled bucket for large distances.
        rp_f = ops.cast(relative_position, "float32")
        relative_position_if_large = max_exact + ops.cast(
            ops.log(rp_f / float(max_exact))
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            "int32",
        )
        relative_position_if_large = ops.minimum(
            relative_position_if_large,
            ops.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets = relative_buckets + ops.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def _compute_bias(self, L: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the ``(1, num_heads, L, L)`` relative-position attention bias.

        :param L: Dynamic sequence length (scalar tensor from ``ops.shape``).
        :type L: keras.KerasTensor
        :return: ``(1, num_heads, L, L)`` additive bias.
        :rtype: keras.KerasTensor
        """
        context_position = ops.arange(0, L, dtype="int32")[:, None]  # (L, 1)
        memory_position = ops.arange(0, L, dtype="int32")[None, :]  # (1, L)
        relative_position = memory_position - context_position  # (L, L)
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.rel_attention_num_buckets,
            max_distance=self.rel_attention_max_distance,
        )  # (L, L)
        values = self.relative_attention_bias(rp_bucket)  # (L, L, heads)
        # (1, heads, L, L)
        values = ops.expand_dims(ops.transpose(values, (2, 0, 1)), axis=0)
        return values

    def call(
        self,
        token_ids: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode token ids into a ``(B, L, embed_dim)`` sequence.

        :param token_ids: Integer ids ``(B, L)``.
        :type token_ids: keras.KerasTensor
        :param attention_mask: Optional ``(B, L)`` 1/0 padding mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: Sequence ``(B, L, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        token_ids = ops.cast(token_ids, "int32")
        L = ops.shape(token_ids)[1]
        H = self.num_heads
        head_dim = self.inner_dim // H

        x = self.shared(token_ids)  # (B, L, D)

        # --- shared relative-position bias (+ additive padding mask) ----
        position_bias = self._compute_bias(L)  # (1, H, L, L)
        if attention_mask is not None:
            # Additive key mask: 0 where keep, large-negative where padded.
            key_mask = ops.cast(attention_mask, "float32")  # (B, L)
            min_val = float(np.finfo(np.float32).min)
            add_mask = (1.0 - key_mask) * min_val  # (B, L)
            add_mask = add_mask[:, None, None, :]  # (B, 1, 1, L)
            position_bias = position_bias + add_mask  # (B, H, L, L)

        def split_heads(t: keras.KerasTensor) -> keras.KerasTensor:
            B = ops.shape(t)[0]
            t = ops.reshape(t, (B, L, H, head_dim))
            return ops.transpose(t, (0, 2, 1, 3))  # (B, H, L, hd)

        for i in range(self.num_layers):
            # --- self-attention (NO 1/sqrt(d) scaling) -----------------
            residual = x
            n = self.attn_norm[i](x, training=training)
            q = split_heads(self.q[i](n, training=training))
            k = split_heads(self.k[i](n, training=training))
            v = split_heads(self.v[i](n, training=training))
            # DECISION plan_2026-06-12_dfce0712/D-009: T5 omits the 1/sqrt(d)
            # attention scaling (it is folded into the initializer scale). Do NOT
            # divide scores by sqrt(head_dim) here -- adding the scale would
            # break faithfulness with the T5 reference. See decisions.md D-009.
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B,H,L,L)
            scores = scores + ops.cast(position_bias, scores.dtype)
            weights = ops.softmax(scores, axis=-1)
            attn = ops.matmul(weights, v)  # (B, H, L, hd)
            attn = ops.transpose(attn, (0, 2, 1, 3))  # (B, L, H, hd)
            B = ops.shape(attn)[0]
            attn = ops.reshape(attn, (B, L, self.inner_dim))
            attn = self.o[i](attn, training=training)
            x = residual + attn

            # --- gated-GELU FFN ----------------------------------------
            residual = x
            n = self.ff_norm[i](x, training=training)
            gated = keras.activations.gelu(self.wi_0[i](n, training=training)) * \
                self.wi_1[i](n, training=training)
            x = residual + self.wo[i](gated, training=training)

        x = self.final_norm(x, training=training)
        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, L, embed_dim)`` (works before ``build``).

        :param input_shape: ``(B, L)`` token-id shape (list/dict tolerated).
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, L, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        ids_shape = _first_shape(input_shape)
        B = ids_shape[0]
        L = ids_shape[1] if len(ids_shape) > 1 else None
        return (B, L, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return all constructor arguments for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rel_attention_num_buckets": self.rel_attention_num_buckets,
                "rel_attention_max_distance": self.rel_attention_max_distance,
                "eps": self.eps,
            }
        )
        return config


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _first_shape(input_shape: Any) -> Tuple[Optional[int], ...]:
    """Extract the token-id shape from a possibly nested input_shape.

    Functional-model wrapping can pass a list (or dict) of shapes; the token-id
    shape is the first / only ``(B, L)`` entry.

    :param input_shape: A shape tuple, list of shapes, or dict of shapes.
    :type input_shape: Any
    :return: The ``(B, L)`` token-id shape tuple.
    :rtype: Tuple[Optional[int], ...]
    """
    if isinstance(input_shape, dict):
        input_shape = list(input_shape.values())[0]
    if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and \
            isinstance(input_shape[0], (list, tuple)):
        input_shape = input_shape[0]
    return tuple(input_shape)

# ---------------------------------------------------------------------

