"""
SD3 MMDiT joint attention: dual-stream (image + text) joint self-attention.

This layer is the Keras 3 port of the diffusers ``JointAttnProcessor`` semantics
used inside SD3 MMDiT blocks. Two token streams -- the image (``hidden_states``)
and the text/context (``encoder_hidden_states``) -- are independently projected to
Q/K/V, optionally per-head RMS-normed on Q/K, then **concatenated along the
sequence axis** so the two streams attend *jointly* in a single scaled-dot-product
attention. The joint output is then **split back** to the original per-stream
lengths and each stream is run through its own output projection.

This faithfully reproduces the MMDiT joint-attention math while deliberately
DROPPING the PyTorch source's paging / KV-cache / deque machinery (the source
carries a ``# TODO: REVISIT THE PAGING LOGIC``): that cache is an inference-time
hack that is irrelevant to -- and unused by -- correct training-time joint
attention. See the ``# DECISION`` anchor in :meth:`call`.

**Intent**

Provide the dual-stream joint attention primitive for SD3 MMDiT blocks: image and
text tokens attend to each other (and themselves) in one shared attention, which
is the structural difference from a single-stream DiT. The layer owns only the
projection + norm + SDPA math; AdaLN modulation lives in the surrounding block.

**Architecture**

::

    hidden_states (B, N_img, dim)        encoder_hidden_states (B, N_txt, dim)
          │                                        │
    to_q/to_k/to_v (Dense dim)            add_q_proj/add_k_proj/add_v_proj (Dense dim)
          │ reshape (B, H, N_img, hd)              │ reshape (B, H, N_txt, hd)
    norm_q / norm_k (RMSNorm head_dim)    norm_added_q / norm_added_k (RMSNorm head_dim)
          └──────────────┬─────────────────────────┘
                         │  concat along seq axis (axis=2)
                         ▼
              Q,K,V : (B, H, N_img+N_txt, hd)
                         │  scores = Q·Kᵀ · head_dim^-0.5
                         │  softmax (float32) · V
                         ▼
              out : (B, H, N_img+N_txt, hd) -> (B, N_img+N_txt, dim)
                         │  split at N_img
          ┌──────────────┴─────────────────────────┐
    image_out (B, N_img, dim)             text_out (B, N_txt, dim)
          │ to_out (Dense dim)                     │ to_add_out (Dense dim)*
          ▼                                        ▼
    image_out                              text_out  (*omitted if context_pre_only)

When ``context_pre_only`` is True (the final MMDiT block discards the text path),
``to_add_out`` is not created and ``call`` returns only the image stream.

PyTorch reference semantics (diffusers ``JointAttnProcessor2_0``, paging stripped)::

    q = to_q(x); k = to_k(x); v = to_v(x)
    eq = add_q_proj(c); ek = add_k_proj(c); ev = add_v_proj(c)
    # per-head reshape + optional RMS qk-norm
    q = cat([q, eq], dim=seq); k = cat([k, ek], dim=seq); v = cat([v, ev], dim=seq)
    out = scaled_dot_product_attention(q, k, v)
    img, txt = split(out, [N_img, N_txt], dim=seq)
    img = to_out(img); txt = to_add_out(txt)  # txt dropped if context_pre_only
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class MMDiTJointAttention(keras.layers.Layer):
    """SD3 MMDiT dual-stream joint attention (image + text).

    Projects an image token stream and a text/context token stream to Q/K/V,
    applies optional per-head RMS QK-norm to each stream, concatenates the two
    streams along the sequence axis for a single joint scaled-dot-product
    attention, then splits the result back to the original per-stream lengths
    and projects each stream out.

    :param dim: Model / embedding dimensionality. Must be divisible by
        ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param qk_norm: If True, apply per-head :class:`RMSNorm` (over ``head_dim``)
        to image and text Q/K. Defaults to True.
    :type qk_norm: bool
    :param use_bias: Whether the Q/K/V/output ``Dense`` projections use a bias.
        Defaults to True.
    :type use_bias: bool
    :param context_pre_only: If True, the text/context output projection
        (``to_add_out``) is not created and :meth:`call` returns only the image
        stream (the final MMDiT block discards the text path). Defaults to False.
    :type context_pre_only: bool
    :param eps: Epsilon for the per-head RMS QK-norm. Defaults to ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, or if
        ``dim`` / ``num_heads`` are not positive integers.

    Input/Output:
        ``call([hidden_states, encoder_hidden_states])`` with
        ``hidden_states: (B, N_img, dim)`` and
        ``encoder_hidden_states: (B, N_txt, dim)``. Returns
        ``[image_out (B, N_img, dim), text_out (B, N_txt, dim)]`` when
        ``context_pre_only`` is False, else ``image_out (B, N_img, dim)``.

    Example:
        >>> attn = MMDiTJointAttention(dim=256, num_heads=4)
        >>> img = keras.random.normal((2, 16, 256))
        >>> txt = keras.random.normal((2, 7, 256))
        >>> img_out, txt_out = attn([img, txt])
        >>> img_out.shape, txt_out.shape
        ((2, 16, 256), (2, 7, 256))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        use_bias: bool = True,
        context_pre_only: bool = False,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- validation -------------------------------------------------
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # --- store config ----------------------------------------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = bool(qk_norm)
        self.use_bias = bool(use_bias)
        self.context_pre_only = bool(context_pre_only)
        self.eps = float(eps)
        self._scale = self.head_dim ** -0.5

        # --- image-stream projections (created here, built in build()) --
        self.to_q = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_q")
        self.to_k = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_k")
        self.to_v = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_v")
        self.to_out = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="to_out"
        )

        # --- text/context-stream projections ---------------------------
        self.add_q_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_q_proj"
        )
        self.add_k_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_k_proj"
        )
        self.add_v_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_v_proj"
        )
        # Text output projection only when the text stream is kept.
        self.to_add_out = (
            None
            if self.context_pre_only
            else keras.layers.Dense(
                dim, use_bias=self.use_bias, name="to_add_out"
            )
        )

        # --- per-head QK-RMSNorm ---------------------------------------
        if self.qk_norm:
            self.norm_q = RMSNorm(axis=-1, epsilon=self.eps, name="norm_q")
            self.norm_k = RMSNorm(axis=-1, epsilon=self.eps, name="norm_k")
            self.norm_added_q = RMSNorm(
                axis=-1, epsilon=self.eps, name="norm_added_q"
            )
            self.norm_added_k = RMSNorm(
                axis=-1, epsilon=self.eps, name="norm_added_k"
            )
        else:
            self.norm_q = None
            self.norm_k = None
            self.norm_added_q = None
            self.norm_added_k = None

        logger.debug(
            f"Initialized MMDiTJointAttention(dim={dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}, qk_norm={self.qk_norm}, "
            f"use_bias={self.use_bias}, context_pre_only={self.context_pre_only}, "
            f"eps={self.eps})"
        )

    def build(
        self, input_shape: List[Tuple[Optional[int], ...]]
    ) -> None:
        """Build the per-stream Q/K/V, QK-norm, and output projections.

        :param input_shape: List ``[img_shape, txt_shape]`` where
            ``img_shape = (B, N_img, dim)`` and ``txt_shape = (B, N_txt, dim)``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If ``input_shape`` is not a list/tuple of two shapes,
            or either stream's last dim is not ``dim``.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "MMDiTJointAttention expects input_shape to be a list of two "
                f"shapes [img_shape, txt_shape], got {input_shape}"
            )
        img_shape, txt_shape = input_shape[0], input_shape[1]
        if len(img_shape) != 3 or img_shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states must have shape (B, N_img, dim={self.dim}), "
                f"got {img_shape}"
            )
        if len(txt_shape) != 3 or txt_shape[-1] != self.dim:
            raise ValueError(
                f"encoder_hidden_states must have shape (B, N_txt, dim="
                f"{self.dim}), got {txt_shape}"
            )

        # Image-stream Q/K/V consume (B, N_img, dim).
        self.to_q.build(img_shape)
        self.to_k.build(img_shape)
        self.to_v.build(img_shape)
        # Image output projection consumes (B, N_img, dim).
        self.to_out.build(img_shape)

        # Text-stream Q/K/V consume (B, N_txt, dim).
        self.add_q_proj.build(txt_shape)
        self.add_k_proj.build(txt_shape)
        self.add_v_proj.build(txt_shape)
        if self.to_add_out is not None:
            self.to_add_out.build(txt_shape)

        # QK-norm normalizes over the per-head dim; build with a shape whose
        # last axis is head_dim so each scale parameter is (head_dim,).
        if self.qk_norm:
            qk_norm_shape = (None, self.num_heads, None, self.head_dim)
            self.norm_q.build(qk_norm_shape)
            self.norm_k.build(qk_norm_shape)
            self.norm_added_q.build(qk_norm_shape)
            self.norm_added_k.build(qk_norm_shape)

        super().build(input_shape)

    def _to_heads(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape ``(B, N, dim)`` -> ``(B, num_heads, N, head_dim)``."""
        shape = keras.ops.shape(x)
        batch, length = shape[0], shape[1]
        x = keras.ops.reshape(
            x, (batch, length, self.num_heads, self.head_dim)
        )
        return keras.ops.transpose(x, (0, 2, 1, 3))

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """Run dual-stream joint attention.

        :param inputs: ``[hidden_states, encoder_hidden_states]`` with shapes
            ``(B, N_img, dim)`` and ``(B, N_txt, dim)``.
        :type inputs: List[keras.KerasTensor]
        :param training: Forwarded to the QK-norm sub-layers.
        :type training: Optional[bool]
        :return: ``[image_out, text_out]`` if ``context_pre_only`` is False,
            else ``image_out``.
        :rtype: Union[keras.KerasTensor, List[keras.KerasTensor]]
        """
        hidden_states, encoder_hidden_states = inputs[0], inputs[1]

        # Dynamic image sequence length used for the post-attention split.
        n_img = keras.ops.shape(hidden_states)[1]

        # --- project both streams to (B, heads, seq, head_dim) ----------
        img_q = self._to_heads(self.to_q(hidden_states))
        img_k = self._to_heads(self.to_k(hidden_states))
        img_v = self._to_heads(self.to_v(hidden_states))

        txt_q = self._to_heads(self.add_q_proj(encoder_hidden_states))
        txt_k = self._to_heads(self.add_k_proj(encoder_hidden_states))
        txt_v = self._to_heads(self.add_v_proj(encoder_hidden_states))

        # --- per-head QK-RMSNorm over head_dim --------------------------
        if self.qk_norm:
            img_q = self.norm_q(img_q, training=training)
            img_k = self.norm_k(img_k, training=training)
            txt_q = self.norm_added_q(txt_q, training=training)
            txt_k = self.norm_added_k(txt_k, training=training)

        # --- joint attention: concat image + text along the seq axis ----
        q = keras.ops.concatenate([img_q, txt_q], axis=2)
        k = keras.ops.concatenate([img_k, txt_k], axis=2)
        v = keras.ops.concatenate([img_v, txt_v], axis=2)

        # DECISION plan_2026-06-12_dfce0712/D-004: implement plain joint SDPA and
        # DROP the PyTorch source's paging / KV-cache / deque logic (the source
        # carries `# TODO: REVISIT THE PAGING LOGIC`). The cache is an
        # inference-time micro-opt that is UNUSED by and irrelevant to the MMDiT
        # joint-attention math; porting it would add stateful, non-graph-safe,
        # untested machinery for zero training-time benefit. Do NOT re-introduce a
        # KV cache here -- if paged inference is ever needed it belongs in the
        # pipeline, not in this stateless layer. Also: SDPA is computed manually
        # with keras.ops (no fused op) for backend portability and bf16 safety,
        # mirroring ideogram4_attention. See decisions.md D-004.
        scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        )
        scores = scores * self._scale

        # Softmax in float32 for bf16 stability, then cast back.
        attn = keras.ops.softmax(
            keras.ops.cast(scores, "float32"), axis=-1
        )
        attn = keras.ops.cast(attn, v.dtype)
        out = keras.ops.matmul(attn, v)  # (B, heads, N_img+N_txt, head_dim)

        # --- merge heads -> (B, N_img+N_txt, dim) -----------------------
        out_shape = keras.ops.shape(out)
        batch, total_len = out_shape[0], out_shape[2]
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch, total_len, self.dim))

        # --- split back to per-stream lengths ---------------------------
        image_out = out[:, :n_img, :]
        text_out = out[:, n_img:, :]

        image_out = self.to_out(image_out)

        if self.context_pre_only:
            return image_out

        text_out = self.to_add_out(text_out)
        return [image_out, text_out]

    def compute_output_shape(
        self, input_shape: List[Tuple[Optional[int], ...]]
    ) -> Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]:
        """Return the per-stream output shape(s) from stored config.

        Works before :meth:`build`.

        :param input_shape: List ``[img_shape, txt_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: ``[img_shape, txt_shape]`` if ``context_pre_only`` is False,
            else ``img_shape`` (each ``(B, N, dim)``).
        :rtype: Union[Tuple, List[Tuple]]
        """
        img_shape, txt_shape = input_shape[0], input_shape[1]
        image_out_shape = (img_shape[0], img_shape[1], self.dim)
        if self.context_pre_only:
            return image_out_shape
        text_out_shape = (txt_shape[0], txt_shape[1], self.dim)
        return [image_out_shape, text_out_shape]

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Dictionary with all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "qk_norm": self.qk_norm,
                "use_bias": self.use_bias,
                "context_pre_only": self.context_pre_only,
                "eps": self.eps,
            }
        )
        return config

# ---------------------------------------------------------------------
