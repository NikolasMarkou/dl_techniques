"""
SD3 MMDiT dual-stream transformer block + final projection layer.

Keras 3 port of the Stable Diffusion 3 MMDiT ``DiTBlock`` (the diffusers
``JointTransformerBlock``) and the final output head. The block is the
dual-stream container that wires together the three step-1..3 primitives:

- :class:`~dl_techniques.layers.attention.mmdit_joint_attention.MMDiTJointAttention`
  -- the joint image+text scaled-dot-product attention,
- the SD3 AdaLN modulation trio
  (:class:`~dl_techniques.layers.transformers.sd3_adaln.AdaLayerNormZero` /
  :class:`AdaLayerNormZeroX` / :class:`AdaLayerNormContinuous`),
- :class:`~dl_techniques.layers.ffn.gelu_mlp_ffn.GELUMLPFFN` -- the
  GELU-tanh FeedForward.

**Intent**

Provide one stackable MMDiT block that, given an image token stream
``hidden_states (B, N_img, dim)``, a text token stream
``encoder_hidden_states (B, N_txt, dim)`` and a per-sample conditioning vector
``time_emb (B, dim)``, updates both streams via AdaLN-modulated joint attention
and per-stream gated FFNs. Three structural flavors are selected by constructor
flags:

- ``use_dual_attention`` -- image stream gets a second (self) attention path
  (SD3.5-medium); the image AdaLN becomes the 9-way :class:`AdaLayerNormZeroX`.
- ``context_pre_only`` -- the LAST block discards the text path: the text AdaLN
  becomes the 2-way :class:`AdaLayerNormContinuous` (no gates), the joint
  attention drops the text output projection, and :meth:`MMDiTBlock.call`
  returns ONLY the updated image stream (a single tensor).

**Dual-return contract.** When ``context_pre_only`` is ``False`` the block
returns ``(hidden_states, encoder_hidden_states)`` -- both ``(B, N, dim)``.
When ``context_pre_only`` is ``True`` it returns the single image tensor
``hidden_states (B, N_img, dim)`` (the text stream is final and not propagated).
This mirrors the PyTorch source where the final block's ``encoder_hidden_states``
return is ``None``; returning a single tensor (rather than a ``(tensor, None)``
pair) keeps the contract graph-safe and unambiguous for the model's block stack.

**Architecture** (``context_pre_only=False``, ``use_dual_attention=False``)::

    norm_h, g_msa, sh_mlp, sc_mlp, g_mlp        = norm1([h, t])
    norm_e, cg_msa, csh_mlp, csc_mlp, cg_mlp    = norm1_context([e, t])
    attn_h, attn_e = attn([norm_h, norm_e])
    h = h + g_msa[:,None,:]  * attn_h
    h = h + g_mlp[:,None,:]  * ff(norm2(h)*(1+sc_mlp[:,None,:]) + sh_mlp[:,None,:])
    e = e + cg_msa[:,None,:] * attn_e
    e = e + cg_mlp[:,None,:] * ff_context(norm2_context(e)*(1+csc_mlp)+csh_mlp)
    return (h, e)

PyTorch reference: diffusers ``models/transformers/transformer_sd3.py``
(``JointTransformerBlock.forward``).
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.mmdit_joint_attention import (
    MMDiTJointAttention,
)
from dl_techniques.layers.transformers.sd3_adaln import (
    AdaLayerNormZero,
    AdaLayerNormZeroX,
    AdaLayerNormContinuous,
)
from dl_techniques.layers.ffn.gelu_mlp_ffn import GELUMLPFFN

# ---------------------------------------------------------------------


def _unpack_triple_shape(
    input_shape: Any,
) -> Tuple[
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
    Tuple[Optional[int], ...],
]:
    """Split a ``[img_shape, txt_shape, cond_shape]`` input into its parts.

    :param input_shape: list/tuple of exactly three shapes
        ``[img_shape, txt_shape, cond_shape]`` where ``img_shape = (B, N_img,
        dim)``, ``txt_shape = (B, N_txt, dim)`` and ``cond_shape = (B, dim)``.
    :type input_shape: Any
    :return: ``(img_shape, txt_shape, cond_shape)``.
    :rtype: Tuple[Tuple, Tuple, Tuple]
    :raises ValueError: If ``input_shape`` is not a triple of shapes.
    """
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
        raise ValueError(
            "Expected input_shape to be a triple "
            "[img_shape, txt_shape, cond_shape], got "
            f"{input_shape}"
        )
    img_shape, txt_shape, cond_shape = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
    )
    for name, s in (
        ("img_shape", img_shape),
        ("txt_shape", txt_shape),
        ("cond_shape", cond_shape),
    ):
        if not isinstance(s, (list, tuple)):
            raise ValueError(
                f"Each element of input_shape must be a shape tuple; "
                f"{name}={s}"
            )
    return tuple(img_shape), tuple(txt_shape), tuple(cond_shape)


def _gate(
    x: keras.KerasTensor, gate: keras.KerasTensor
) -> keras.KerasTensor:
    """Apply a ``(B, dim)`` gate to a ``(B, N, dim)`` tensor (broadcast)."""
    return keras.ops.expand_dims(gate, axis=1) * x


def _modulate(
    x: keras.KerasTensor,
    shift: keras.KerasTensor,
    scale: keras.KerasTensor,
) -> keras.KerasTensor:
    """AdaLN MLP-branch modulation ``x*(1+scale[:,None,:]) + shift[:,None,:]``."""
    scale = keras.ops.expand_dims(scale, axis=1)
    shift = keras.ops.expand_dims(shift, axis=1)
    return x * (1.0 + scale) + shift


# =====================================================================
# MMDiTBlock
# =====================================================================


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class MMDiTBlock(keras.layers.Layer):
    """SD3 MMDiT dual-stream transformer block.

    Wires the joint attention, the per-stream AdaLN modulation, and the
    per-stream GELU-tanh FFN into one stackable block. Updates both the image
    stream ``hidden_states`` and the text stream ``encoder_hidden_states``
    conditioned on a per-sample vector ``time_emb``.

    :param dim: Model / embedding dimensionality. Must be divisible by
        ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: FFN expansion ratio; the FFN hidden dim is
        ``int(dim * mlp_ratio)``. Defaults to ``4.0``.
    :type mlp_ratio: float
    :param context_pre_only: If True this is the final block: the text stream is
        normalized with :class:`AdaLayerNormContinuous` (no gates), the joint
        attention drops its text output projection, and :meth:`call` returns the
        single image tensor (the text path is not propagated). Defaults to False.
    :type context_pre_only: bool
    :param use_dual_attention: If True the image stream gets a second (self)
        attention path and the image AdaLN becomes the 9-way
        :class:`AdaLayerNormZeroX`. Defaults to False.
    :type use_dual_attention: bool
    :param qk_norm: Forwarded to :class:`MMDiTJointAttention`. Defaults to True.
    :type qk_norm: bool
    :param eps: Epsilon for the affine-free LayerNorms / AdaLN. Defaults to
        ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` is not a positive integer or ``eps <= 0``.

    Input/Output:
        ``call([hidden_states, encoder_hidden_states, time_emb])`` with
        ``hidden_states: (B, N_img, dim)``, ``encoder_hidden_states: (B, N_txt,
        dim)`` and ``time_emb: (B, dim)``. Returns ``(hidden_states,
        encoder_hidden_states)`` (both ``(B, N, dim)``) when ``context_pre_only``
        is False, else the single image tensor ``(B, N_img, dim)``.

    Example:
        >>> blk = MMDiTBlock(dim=64, num_heads=4)
        >>> h = keras.random.normal((2, 16, 64))
        >>> e = keras.random.normal((2, 7, 64))
        >>> t = keras.random.normal((2, 64))
        >>> h_out, e_out = blk([h, e, t])
        >>> h_out.shape, e_out.shape
        ((2, 16, 64), (2, 7, 64))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        context_pre_only: bool = False,
        use_dual_attention: bool = False,
        qk_norm: bool = True,
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
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # --- store config ----------------------------------------------
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.context_pre_only = bool(context_pre_only)
        self.use_dual_attention = bool(use_dual_attention)
        self.qk_norm = bool(qk_norm)
        self.eps = float(eps)
        self._ffn_hidden = int(dim * self.mlp_ratio)

        # --- image-stream AdaLN ----------------------------------------
        if self.use_dual_attention:
            self.norm1 = AdaLayerNormZeroX(dim, eps=self.eps, name="norm1")
        else:
            self.norm1 = AdaLayerNormZero(dim, eps=self.eps, name="norm1")

        # --- text-stream AdaLN -----------------------------------------
        if self.context_pre_only:
            self.norm1_context = AdaLayerNormContinuous(
                dim, eps=self.eps, name="norm1_context"
            )
        else:
            self.norm1_context = AdaLayerNormZero(
                dim, eps=self.eps, name="norm1_context"
            )

        # --- joint attention -------------------------------------------
        self.attn = MMDiTJointAttention(
            dim,
            num_heads,
            qk_norm=self.qk_norm,
            context_pre_only=self.context_pre_only,
            name="attn",
        )

        # --- optional dual self-attention on the image stream ----------
        # DECISION plan_2026-06-12_dfce0712/D-005: use keras.layers.MultiHeadAttention
        # for the dual self-attention (attn2) rather than building a 4th custom
        # QK-normed attention class. The SD3.5-medium dual path is a standard
        # single-stream self-attention; reusing MHA avoids a speculative 4th
        # attention abstraction (earned-abstraction / YAGNI) AT THE COST OF
        # omitting attn2's per-head QK-RMSNorm (a minor SD3.5-medium detail).
        # Do NOT re-implement a QK-normed attention here just to mirror that
        # detail -- if exact QK-norm parity on the dual path is ever required it
        # is a focused follow-up, not a reason to fork a new attention class.
        # See decisions.md D-005.
        if self.use_dual_attention:
            self.attn2 = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=dim // num_heads,
                name="attn2",
            )
        else:
            self.attn2 = None

        # --- image FFN + pre-norm --------------------------------------
        self.ff = GELUMLPFFN(
            hidden_dim=self._ffn_hidden, output_dim=dim, name="ff"
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.eps, center=False, scale=False, name="norm2"
        )

        # --- text FFN + pre-norm (only when the text stream is kept) ----
        if self.context_pre_only:
            self.ff_context = None
            self.norm2_context = None
        else:
            self.ff_context = GELUMLPFFN(
                hidden_dim=self._ffn_hidden, output_dim=dim, name="ff_context"
            )
            self.norm2_context = keras.layers.LayerNormalization(
                epsilon=self.eps,
                center=False,
                scale=False,
                name="norm2_context",
            )

        logger.debug(
            f"Initialized MMDiTBlock(dim={dim}, num_heads={num_heads}, "
            f"mlp_ratio={self.mlp_ratio}, context_pre_only={self.context_pre_only}, "
            f"use_dual_attention={self.use_dual_attention}, qk_norm={self.qk_norm}, "
            f"eps={self.eps})"
        )

    def build(self, input_shape: Any) -> None:
        """Build every (non-None) sub-layer with its correct input shape.

        :param input_shape: ``[img_shape, txt_shape, cond_shape]``.
        :type input_shape: Any
        """
        img_shape, txt_shape, cond_shape = _unpack_triple_shape(input_shape)

        # AdaLN layers consume [stream_shape, cond_shape].
        self.norm1.build([img_shape, cond_shape])
        self.norm1_context.build([txt_shape, cond_shape])

        # Joint attention consumes [img_shape, txt_shape].
        self.attn.build([img_shape, txt_shape])

        # Dual self-attention: query == value == norm_h2 (image stream).
        if self.attn2 is not None:
            self.attn2.build(query_shape=img_shape, value_shape=img_shape)

        # Image FFN pre-norm + FFN both consume (B, N_img, dim).
        self.norm2.build(img_shape)
        self.ff.build(img_shape)

        # Text FFN pre-norm + FFN both consume (B, N_txt, dim).
        if self.norm2_context is not None:
            self.norm2_context.build(txt_shape)
            self.ff_context.build(txt_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Run the dual-stream block.

        :param inputs: ``[hidden_states, encoder_hidden_states, time_emb]`` with
            shapes ``(B, N_img, dim)``, ``(B, N_txt, dim)`` and ``(B, dim)``.
        :type inputs: List[keras.KerasTensor]
        :param training: Forwarded to the attention / FFN sub-layers.
        :type training: Optional[bool]
        :return: ``(hidden_states, encoder_hidden_states)`` if
            ``context_pre_only`` is False, else the single image tensor.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]
        """
        hidden_states, encoder_hidden_states, time_emb = (
            inputs[0],
            inputs[1],
            inputs[2],
        )

        # --- image-stream AdaLN ----------------------------------------
        if self.use_dual_attention:
            (
                norm_h,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                norm_h2,
                gate_msa2,
            ) = self.norm1([hidden_states, time_emb], training=training)
        else:
            (
                norm_h,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.norm1([hidden_states, time_emb], training=training)

        # --- text-stream AdaLN -----------------------------------------
        if self.context_pre_only:
            norm_enc = self.norm1_context(
                [encoder_hidden_states, time_emb], training=training
            )
        else:
            (
                norm_enc,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(
                [encoder_hidden_states, time_emb], training=training
            )

        # --- joint attention -------------------------------------------
        if self.context_pre_only:
            attn_out = self.attn([norm_h, norm_enc], training=training)
            enc_out = None
        else:
            attn_out, enc_out = self.attn(
                [norm_h, norm_enc], training=training
            )

        # --- image residual (attention) --------------------------------
        hidden_states = hidden_states + _gate(attn_out, gate_msa)

        # --- optional dual self-attention ------------------------------
        if self.use_dual_attention:
            attn2_out = self.attn2(norm_h2, norm_h2, training=training)
            hidden_states = hidden_states + _gate(attn2_out, gate_msa2)

        # --- image FFN -------------------------------------------------
        nh = _modulate(self.norm2(hidden_states), shift_mlp, scale_mlp)
        ff_out = self.ff(nh, training=training)
        hidden_states = hidden_states + _gate(ff_out, gate_mlp)

        # --- context branch --------------------------------------------
        if self.context_pre_only:
            return hidden_states

        encoder_hidden_states = encoder_hidden_states + _gate(
            enc_out, c_gate_msa
        )
        ne = _modulate(
            self.norm2_context(encoder_hidden_states),
            c_shift_mlp,
            c_scale_mlp,
        )
        ff_c_out = self.ff_context(ne, training=training)
        encoder_hidden_states = encoder_hidden_states + _gate(
            ff_c_out, c_gate_mlp
        )

        return hidden_states, encoder_hidden_states

    def compute_output_shape(
        self, input_shape: Any
    ) -> Union[
        Tuple[Optional[int], ...],
        Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]],
    ]:
        """Return the dual (or single) output shape from stored config.

        :param input_shape: ``[img_shape, txt_shape, cond_shape]``.
        :type input_shape: Any
        :return: ``img_shape`` if ``context_pre_only``, else ``(img_shape,
            txt_shape)``.
        :rtype: Union[Tuple, Tuple[Tuple, Tuple]]
        """
        img_shape, txt_shape, _ = _unpack_triple_shape(input_shape)
        if self.context_pre_only:
            return img_shape
        return img_shape, txt_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config (all ``__init__`` parameters)."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "context_pre_only": self.context_pre_only,
                "use_dual_attention": self.use_dual_attention,
                "qk_norm": self.qk_norm,
                "eps": self.eps,
            }
        )
        return config


# =====================================================================
# MMDiTFinalLayer
# =====================================================================


@keras.saving.register_keras_serializable(package="dl_techniques.models")
class MMDiTFinalLayer(keras.layers.Layer):
    """SD3 MMDiT final projection head.

    Applies a conditioned :class:`AdaLayerNormContinuous` (scale + shift, no
    gate) to the image stream, then a linear projection to ``out_channels``.
    Used as the model's output head to map ``(B, N, dim)`` patch tokens to the
    per-patch velocity channels ``(B, N, out_channels)``.

    :param dim: Model / embedding dimensionality.
    :type dim: int
    :param out_channels: Output channel count (per patch token).
    :type out_channels: int
    :param eps: Epsilon for the AdaLN LayerNorm. Defaults to ``1e-6``.
    :type eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If ``dim`` / ``out_channels`` are not positive integers
        or ``eps <= 0``.

    Input/Output:
        ``call([hidden_states, time_emb])`` with ``hidden_states: (B, N, dim)``
        and ``time_emb: (B, dim)``. Returns ``(B, N, out_channels)``.

    Example:
        >>> head = MMDiTFinalLayer(dim=64, out_channels=16)
        >>> h = keras.random.normal((2, 16, 64))
        >>> t = keras.random.normal((2, 64))
        >>> head([h, t]).shape
        (2, 16, 16)
    """

    def __init__(
        self,
        dim: int,
        out_channels: int,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive integer, got {out_channels}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.out_channels = out_channels
        self.eps = float(eps)

        self.norm = AdaLayerNormContinuous(dim, eps=self.eps, name="norm")
        self.proj = keras.layers.Dense(out_channels, name="proj")

        logger.debug(
            f"Initialized MMDiTFinalLayer(dim={dim}, "
            f"out_channels={out_channels}, eps={self.eps})"
        )

    def build(self, input_shape: Any) -> None:
        """Build the AdaLN + projection sub-layers.

        :param input_shape: ``[hidden_shape, cond_shape]`` where
            ``hidden_shape = (B, N, dim)`` and ``cond_shape = (B, dim)``.
        :type input_shape: Any
        :raises ValueError: If ``input_shape`` is not a pair of shapes.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "MMDiTFinalLayer expects input_shape to be a pair "
                f"[hidden_shape, cond_shape], got {input_shape}"
            )
        hidden_shape, cond_shape = tuple(input_shape[0]), tuple(input_shape[1])
        self.norm.build([hidden_shape, cond_shape])
        # AdaLN preserves (B, N, dim); proj consumes that.
        self.proj.build(self.norm.compute_output_shape([hidden_shape, cond_shape]))
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply conditioned norm + projection.

        :param inputs: ``[hidden_states, time_emb]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: ``(B, N, out_channels)``.
        :rtype: keras.KerasTensor
        """
        hidden_states, time_emb = inputs[0], inputs[1]
        x = self.norm([hidden_states, time_emb], training=training)
        return self.proj(x)

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, N, out_channels)`` from stored config."""
        hidden_shape = tuple(input_shape[0])
        return (hidden_shape[0], hidden_shape[1], self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config (all ``__init__`` parameters)."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "out_channels": self.out_channels,
                "eps": self.eps,
            }
        )
        return config

# ---------------------------------------------------------------------
