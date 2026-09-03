"""MMDiTJointAttention, dual-stream joint attention for SD3 MMDiT blocks.

An image token stream and a text context stream are each projected to
Q/K/V by their own weights, optionally per-head RMS-normed, then
concatenated along the sequence axis. One scaled-dot-product attention
runs over the concatenation, so image tokens attend to text tokens and
text tokens attend to image tokens; the result is split back at the
image length and each stream goes through its own output projection.
That single joint attention over per-modality weights is the structural
difference from a single-stream DiT.

The layer owns the projections, the norms, and the SDPA math; AdaLN
modulation lives in the surrounding block. When ``context_pre_only`` is
True (the final MMDiT block), ``to_add_out`` is not created and
:meth:`call` returns the image stream alone rather than a two-element
list.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.attention.mmdit_joint_attention")
class MMDiTJointAttention(keras.layers.Layer):
    """Compute dual-stream joint attention over an image stream and a text
    context stream.

    Projects an image token stream and a text context stream to Q/K/V with
    separate per-modality weights, applies optional per-head RMS QK-norm to
    each, concatenates the two along the sequence axis for one joint
    scaled-dot-product attention, then splits the result back at the image
    length and projects each stream out through its own weights.

    Shapes below use ``B`` = batch, ``N_img`` and ``N_txt`` = per-stream
    sequence lengths, ``H`` = num_heads, ``hd`` = head_dim, and
    ``N = N_img + N_txt``.

    Architecture:

    .. code-block:: text

          image [B, N_img, dim]        text [B, N_txt, dim]
                    │                            │
                    ▼                            ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ to_q, to_k, to_v   │      │ add_q_proj,        │
          │ image weights      │      │ add_k_proj,        │
          │                    │      │ add_v_proj         │
          │                    │      │ text weights       │
          └─────────┬──────────┘      └─────────┬──────────┘
                    ▼                           ▼
          reshape to heads              reshape to heads
          [B, H, N_img, hd]             [B, H, N_txt, hd]
                    │                           │
                    ▼                           ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ norm_q, norm_k     │      │ norm_added_q,      │
          │ (optional)         │      │ norm_added_k       │
          │                    │      │ (optional)         │
          └─────────┬──────────┘      └─────────┬──────────┘
                    └────────────┬──────────────┘
                                 ▼
                    concat on the sequence axis
                    Q, K, V  [B, H, N, hd],  N = N_img + N_txt
                                 │
                                 ▼
          ┌───────────────────────────────────────────────┐
          │ one joint attention over the concatenation    │
          │   S = Q . K^T * head_dim^-0.5   [B, H, N, N]  │
          │   softmax in float32, cast back to V's dtype  │
          │   A . V                                       │
          │ This is where image tokens see text tokens    │
          │ and text tokens see image tokens.             │
          └───────────────────────┬───────────────────────┘
                                  ▼
                    merge heads  [B, N, dim]
                                  │
                    split at N_img
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌────────────────────┐      ┌────────────────────┐
          │ to_out             │      │ to_add_out         │
          │                    │      │ (omitted entirely  │
          │                    │      │  when              │
          │                    │      │  context_pre_only) │
          └─────────┬──────────┘      └─────────┬──────────┘
                    ▼                           ▼
          image_out [B, N_img, dim]    text_out [B, N_txt, dim]

        With context_pre_only=True, to_add_out is never created and
        call() returns the image stream alone, not a two-element list.
        The right branch above stops at the split.

        This layer accepts no attention mask.

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
        """Validate the configuration and create every projection and norm.

        :param dim: Model dimensionality, divisible by ``num_heads``.
        :type dim: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param qk_norm: Whether to build the four per-head RMS QK-norms.
        :type qk_norm: bool
        :param use_bias: Whether the Dense projections carry a bias.
        :type use_bias: bool
        :param context_pre_only: If True, ``to_add_out`` is not created.
        :type context_pre_only: bool
        :param eps: Epsilon for the QK-norms.
        :type eps: float
        :param kwargs: Additional ``keras.layers.Layer`` arguments.
        :type kwargs: Any

        :raises ValueError: If ``dim`` or ``num_heads`` is not a positive
            integer, if ``dim`` is not divisible by ``num_heads``, or if
            ``eps`` is not positive.
        """
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        # Message text kept byte-identical to what stood here, so this does
        # not call the shared validate_head_divisibility (its message has
        # no trailing period, unlike this one) -- see wave_field_attention.py.
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = bool(qk_norm)
        self.use_bias = bool(use_bias)
        self.context_pre_only = bool(context_pre_only)
        self.eps = float(eps)
        # Not common.compute_attention_scale(head_dim): that helper's float64
        # bit pattern diverges from head_dim ** -0.5 in the last ULP.
        self._scale = self.head_dim ** -0.5

        self.to_q = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_q")
        self.to_k = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_k")
        self.to_v = keras.layers.Dense(dim, use_bias=self.use_bias, name="to_v")
        self.to_out = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="to_out"
        )

        self.add_q_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_q_proj"
        )
        self.add_k_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_k_proj"
        )
        self.add_v_proj = keras.layers.Dense(
            dim, use_bias=self.use_bias, name="add_v_proj"
        )
        # The text output projection exists only when the text stream is kept.
        # `call()` returns a single tensor in the other case.
        self.to_add_out = (
            None
            if self.context_pre_only
            else keras.layers.Dense(
                dim, use_bias=self.use_bias, name="to_add_out"
            )
        )

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
        """Build the per-stream Q/K/V, QK-norm and output projections.

        The four QK-norms are built at a shape whose last axis is ``head_dim``,
        so each scale parameter comes out ``(head_dim,)``.

        :param input_shape: List ``[img_shape, txt_shape]`` where
            ``img_shape = (B, N_img, dim)`` and ``txt_shape = (B, N_txt, dim)``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If ``input_shape`` is not a list/tuple of two shapes,
            or either stream's last dim is not ``dim``.
        """
        if self.built:
            return

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

        # The two streams have separate weights, so each is built at its own
        # shape. Image stream first.
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

        # QK-norm normalizes over the per-head dim. Building with a 4D shape
        # whose last axis is head_dim is what makes each scale (head_dim,).
        if self.qk_norm:
            qk_norm_shape = (None, self.num_heads, None, self.head_dim)
            self.norm_q.build(qk_norm_shape)
            self.norm_k.build(qk_norm_shape)
            self.norm_added_q.build(qk_norm_shape)
            self.norm_added_k.build(qk_norm_shape)

        super().build(input_shape)

    def _to_heads(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Split the channel axis into heads and move the head axis forward.

        :param x: Tensor of shape ``(B, N, dim)``.
        :type x: keras.KerasTensor
        :return: Tensor of shape ``(B, num_heads, N, head_dim)``.
        :rtype: keras.KerasTensor
        """
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

        Project each stream with its own weights, concatenate on the sequence
        axis, run one attention over the concatenation, split the result back
        at the image length, and project each stream out.

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

        # Captured before the concatenation, since the split index is no
        # longer recoverable from the concatenated tensor.
        n_img = keras.ops.shape(hidden_states)[1]

        img_q = self._to_heads(self.to_q(hidden_states))
        img_k = self._to_heads(self.to_k(hidden_states))
        img_v = self._to_heads(self.to_v(hidden_states))

        txt_q = self._to_heads(self.add_q_proj(encoder_hidden_states))
        txt_k = self._to_heads(self.add_k_proj(encoder_hidden_states))
        txt_v = self._to_heads(self.add_v_proj(encoder_hidden_states))

        if self.qk_norm:
            img_q = self.norm_q(img_q, training=training)
            img_k = self.norm_k(img_k, training=training)
            txt_q = self.norm_added_q(txt_q, training=training)
            txt_k = self.norm_added_k(txt_k, training=training)

        # This concatenation is the "joint" in joint attention: one attention
        # problem instead of two, so each stream attends to the other.
        # Shape: (B, H, N_img, hd) + (B, H, N_txt, hd) -> (B, H, N_img+N_txt, hd)
        q = keras.ops.concatenate([img_q, txt_q], axis=2)
        k = keras.ops.concatenate([img_k, txt_k], axis=2)
        v = keras.ops.concatenate([img_v, txt_v], axis=2)

        # DECISION plan_2026-06-12_dfce0712/D-004: no KV cache here -- the PyTorch
        # source's paging is an inference-time device the joint-attention math does not need. SDPA is written out by hand in keras.ops, matching ideogram4_attention. See decisions.md.
        # Shape: (B, H, N, hd) @ (B, H, hd, N) -> (B, H, N, N),  N = N_img + N_txt
        scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 3, 2))
        )
        scores = scores * self._scale

        # Softmax in float32 for bf16 stability, then back to V's dtype.
        attn = keras.ops.softmax(
            keras.ops.cast(scores, "float32"), axis=-1
        )
        attn = keras.ops.cast(attn, v.dtype)
        # Shape: (B, H, N, N) @ (B, H, N, hd) -> (B, H, N, hd)
        out = keras.ops.matmul(attn, v)

        out_shape = keras.ops.shape(out)
        batch, total_len = out_shape[0], out_shape[2]
        # Shape: (B, H, N, hd) -> (B, N, H, hd) -> (B, N, dim)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch, total_len, self.dim))

        # Split index is the image length captured before the concat.
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

        Everything it needs is in ``self.dim`` and ``context_pre_only``, so it
        works before :meth:`build`.

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

        Every ``__init__`` argument is included, so a reloaded layer rebuilds
        the same set of projections and norms.

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
