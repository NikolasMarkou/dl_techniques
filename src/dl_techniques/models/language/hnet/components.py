"""H-Net's isotropic stack: the mixer builder, the pre-norm block, and the block stack.

Every H-Net stage -- encoder, innermost main network, decoder -- is one *isotropic*
stack: a flat run of identical-width residual blocks followed by a single RMSNorm. The
stack's shape is read off a layout string (``"m4"``, ``"T22"``, ``"m4T1"``) whose grammar
and parser live in :mod:`dl_techniques.models.language.hnet.config`; this module turns a
parsed :class:`~dl_techniques.models.language.hnet.config.IsotropicSpec` into Keras
layers and nothing else. It builds no chunking, no recursion and no model.

The letter selects the mixer and its CASE selects whether a SwiGLU MLP accompanies it::

    m  Mamba-2                      M  Mamba-2 + SwiGLU
    t  causal attention             T  causal attention + SwiGLU

Residual convention
-------------------
The reference's blocks use the *fused* residual API of ``flash_attn``'s Triton RMSNorm
(``hnet/modules/block.py:104-135``): ``norm(h, residual=r, prenorm=True)`` returns
``(norm(r + h), r + h)``, and ``Isotropic.forward`` closes the chain with
``norm(h, residual=r, prenorm=False)``. Unrolled, that is exactly ordinary pre-norm::

    x = x + mixer(norm1(x))
    x = x + mlp(norm2(x))          # uppercase letters only
    ...
    out = final_norm(x)            # hnet/modules/isotropic.py:96,167

so this port writes the plain form. There is no fused-residual layer to reproduce and no
numerical divergence to record: the two are the same arithmetic in the same order.

The reference additionally keeps the residual stream in fp32 (``residual_in_fp32=True``).
That is a mixed-precision *stability* choice about the accumulator dtype, not part of the
architecture; this port lets Keras's mixed-precision policy own the residual dtype, as
every other model in this repository does.

Three values are passed EXPLICITLY here that a default would otherwise supply, because in
each case the default is the wrong number or a number that has moved:

* ``epsilon=1e-5`` on every RMSNorm. :func:`create_normalization_layer` defaults to
  ``1e-6``; the reference uses ``1e-5`` (``block.py:41``, ``isotropic.py:96``). A 100x
  change in the denominator of every normalized activation has no shape symptom and
  raises no warning.
* ``norm_before_gate=False`` on every :class:`Mamba2Layer`. That default flipped once
  already in this repository (2026-08-15) and a value that has moved is not a value to
  rely on.
* ``ffn_expansion_factor=4, ffn_multiple_of=128`` on every SwiGLU, and **no**
  ``hidden_dim``. Plan step 2(f) measured this to be the unique pair of twenty that
  reproduces the reference's ``round_up(8 * d_model / 3, 128)``
  (``hnet/modules/mlp.py:19-23``) at every shipped width, with zero mismatches over 64
  widths; the factory's own defaults ``(4, 256)`` agree at ``d_model`` 1024 and 1536 and
  silently disagree at 2048 (5632 vs 5504). Passing ``hidden_dim`` would bypass the
  derivation entirely and is a documented silent behaviour change.

SwiGLU gate pairing: the reference splits one fused ``fc1`` as ``(y, gate) = chunk(2)``
and computes ``silu(gate) * y`` (``mlp.py:29-31``); this repository's ``SwiGLUFFN`` uses
two independent projections and computes ``silu(gate_proj(x)) * up_proj(x)``. From random
init -- and no reference checkpoint is ever loaded -- these are the same function with the
two projections named differently, and step 2(f) confirmed the parameter counts agree
exactly at all three widths.

Masking
-------
:func:`build_causal_keep_mask` emits a rank-3 ``(1, L, L)`` int32 **keep predicate**
(``1 = attend``), never an additive bias. Both facts are load-bearing and both were
measured (D-010(e)): a rank-2 mask reaches ``GroupedQueryAttention._apply_mask`` as a
*key-padding* mask and encodes no causal structure at all, and an additive ``-1e9`` bias
under ``mixed_float16`` yields ``0 * -inf = NaN`` at every unmasked position.

The Mamba-2 mixer takes no mask. That is not an omission: the reference does not mask it
either in unpacked mode (``isotropic.py:126-146`` puts nothing mask-shaped into
``ssm_mixer_kwargs``), and this repository's :class:`Mamba2Layer` has no mask knob to
pass one to.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: ``hnet/modules/{isotropic,block,mha,mlp}.py``.
"""

from typing import Any, Dict, List, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.models.language.hnet.config import (
    LAYOUT_LETTERS,
    MIXER_ATTENTION,
    MIXER_MAMBA2,
    IsotropicSpec,
    parse_arch_layout,
)
from dl_techniques.models.language.mamba.components_v2 import Mamba2Layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = [
    "HNetBlock",
    "HNetIsotropic",
    "NORM_EPSILON",
    "ROPE_THETA",
    "SWIGLU_EXPANSION_FACTOR",
    "SWIGLU_MULTIPLE_OF",
    "build_causal_keep_mask",
    "build_mixer",
    "build_mlp",
    "parse_isotropic_layout",
]

#: RMSNorm epsilon everywhere in H-Net (``block.py:41``, ``isotropic.py:96``). The
#: normalization factory defaults to ``1e-6``, so every call site passes this explicitly.
NORM_EPSILON: float = 1e-5

#: RoPE base frequency (``mha.py:206`` ``rotary_emb_base=10000.0``).
ROPE_THETA: float = 10000.0

#: SwiGLU sizing knobs. ``create_ffn_layer('swiglu', ...)`` derives its hidden width as
#: ``round_up(int(output_dim * factor * 2/3), multiple_of)``, which at ``factor=4`` is the
#: identical float expression ``8 * d_model / 3`` the reference evaluates
#: (``hnet/modules/mlp.py:19-23``). Measured in plan step 2(f) as the ONLY pair of the
#: twenty tried that agrees at every shipped ``d_model``; the factory defaults ``(4, 256)``
#: disagree at ``d_model=2048``.
SWIGLU_EXPANSION_FACTOR: int = 4
SWIGLU_MULTIPLE_OF: int = 128


# ---------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------


def build_causal_keep_mask(
        seq_len: Any,
        window_size: int = -1,
        dtype: str = "int32",
) -> keras.KerasTensor:
    """Build the rank-3 causal (optionally sliding-window) attention keep predicate.

    The predicate is ``j <= i AND i - j <= window_size``: query ``i`` may attend to key
    ``j`` when ``j`` is not in the future and is no further than ``window_size`` positions
    behind. ``window_size = -1`` drops the band and leaves plain causality, matching the
    reference's ``window_size=-1`` sentinel for global attention
    (``hnet/modules/mha.py:203``) and its ``flash_attn`` ``window_size=(w, -1)``
    left-only band (``mha.py:69``).

    :param seq_len: Sequence length. A Python ``int`` or a scalar tensor.
    :type seq_len: Any
    :param window_size: Maximum lookback distance, or ``-1`` for unlimited context.
    :type window_size: int
    :param dtype: Output dtype. The predicate is integral, never floating bias.
    :type dtype: str
    :returns: ``(1, seq_len, seq_len)`` tensor, ``1 = attend``, ``0 = masked``. The
        leading axis is 1 so the same predicate broadcasts across the batch.
    :rtype: keras.KerasTensor
    :raises ValueError: if ``window_size`` is below ``-1``.
    """
    if not isinstance(window_size, int) or isinstance(window_size, bool):
        raise TypeError(
            f"window_size must be an int, got {type(window_size).__name__}: "
            f"{window_size!r}"
        )
    if window_size < -1:
        raise ValueError(
            f"window_size must be -1 (unlimited context) or a non-negative lookback "
            f"distance, got {window_size}"
        )

    # DECISION plan-2026-09-09T042752-6d66ac56/D-010: build the triangle from `arange`
    # broadcasts. Do NOT reach for `keras.ops.tril`/`triu` here -- both RAISE
    # `TypeError: ('pred must not be a Python bool', True)` the moment the function is
    # traced, at `jit_compile=False` as well as under XLA, and their EAGER result is
    # bitwise equal to this form, so the wrong call looks correct until it is traced.
    # Measured plan step 2(b). See decisions.md D-010(b).
    positions = keras.ops.arange(seq_len)
    query_pos = positions[:, None]
    key_pos = positions[None, :]

    keep = key_pos <= query_pos
    if window_size >= 0:
        keep = keras.ops.logical_and(keep, (query_pos - key_pos) <= window_size)

    return keras.ops.cast(keep[None, ...], dtype)


def _combine_keep_masks(
        causal_keep: keras.KerasTensor,
        key_validity: Optional[keras.KerasTensor],
) -> keras.KerasTensor:
    """AND a rank-2 key-validity predicate into the rank-3 causal predicate.

    Both operands are ``1 = keep``, so composition is a logical AND -- exactly how
    ``window_attention.py`` composes a band with a causal mask. Padded keys are removed
    for every query; nothing about the causal structure changes.

    :param causal_keep: ``(1, L, L)`` predicate from :func:`build_causal_keep_mask`.
    :type causal_keep: keras.KerasTensor
    :param key_validity: Optional ``(B, L)`` predicate, ``1`` for a real token. ``None``
        returns ``causal_keep`` unchanged.
    :type key_validity: Optional[keras.KerasTensor]
    :returns: ``(1, L, L)`` when ``key_validity`` is ``None``, else ``(B, L, L)``.
    :rtype: keras.KerasTensor
    """
    if key_validity is None:
        return causal_keep
    combined = keras.ops.logical_and(
        keras.ops.cast(causal_keep, "bool"),
        keras.ops.cast(key_validity, "bool")[:, None, :],
    )
    return keras.ops.cast(combined, causal_keep.dtype)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def parse_isotropic_layout(layout: str) -> IsotropicSpec:
    """Parse one layout string into its :class:`IsotropicSpec`.

    A one-element ``arch_layout`` list is the innermost-stage form, so wrapping the
    string in a list and taking ``.main`` reuses
    :func:`~dl_techniques.models.language.hnet.config.parse_arch_layout` verbatim rather
    than re-implementing the grammar here. The grammar has exactly one implementation.

    :param layout: A layout string such as ``"m4T1"``.
    :type layout: str
    :returns: The parsed stack description.
    :rtype: IsotropicSpec
    :raises TypeError: if ``layout`` is not a string.
    :raises ValueError: if the string is empty or malformed.
    """
    spec = parse_arch_layout([layout]).main
    assert isinstance(spec, IsotropicSpec)  # a 1-element layout is always innermost
    return spec


def build_mixer(
        kind: str,
        d_model: int,
        num_heads: int = 0,
        rotary_emb_dim: int = 0,
        max_seq_len: int = 2048,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        name: Optional[str] = None,
) -> keras.layers.Layer:
    """Build the sequence mixer selected by a layout letter.

    ``m``/``M`` build the Mamba-2 selective state-space layer; ``t``/``T`` build causal
    multi-head attention via the ``group_query`` factory entry with
    ``num_kv_heads == num_heads`` (the MHA special case). Case is irrelevant here -- it
    selects the MLP, which is :func:`build_mlp`'s business.

    :param kind: One of ``"m"``, ``"M"``, ``"t"``, ``"T"``.
    :type kind: str
    :param d_model: Model width.
    :type d_model: int
    :param num_heads: Attention heads. Required for ``t``/``T``, ignored otherwise.
    :type num_heads: int
    :param rotary_emb_dim: Number of head dimensions RoPE rotates. ``0`` disables RoPE;
        must not exceed ``d_model // num_heads``. Ignored for ``m``/``M``.
    :type rotary_emb_dim: int
    :param max_seq_len: Largest position the RoPE tables cover. Ignored for ``m``/``M``.
    :type max_seq_len: int
    :param d_state: Mamba-2 SSM state width. Ignored for ``t``/``T``.
    :type d_state: int
    :param d_conv: Mamba-2 depthwise causal convolution width. Ignored for ``t``/``T``.
    :type d_conv: int
    :param expand: Mamba-2 inner-width expansion factor. Ignored for ``t``/``T``.
    :type expand: int
    :param headdim: Mamba-2 SSM head width. Ignored for ``t``/``T``.
    :type headdim: int
    :param name: Layer name.
    :type name: Optional[str]
    :returns: The mixer layer.
    :rtype: keras.layers.Layer
    :raises ValueError: if ``kind`` is not a layout letter, if an attention mixer is
        asked for without a positive ``num_heads``, if ``d_model`` is not divisible by
        ``num_heads``, or if ``rotary_emb_dim`` exceeds the head width.
    """
    if kind not in LAYOUT_LETTERS:
        raise ValueError(
            f"unknown layout letter {kind!r}; expected one of {LAYOUT_LETTERS}"
        )

    mixer = MIXER_MAMBA2 if kind.lower() == "m" else MIXER_ATTENTION

    if mixer == MIXER_MAMBA2:
        return Mamba2Layer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            # Both passed explicitly and neither is decorative. `norm_epsilon` because
            # the reference normalizes at 1e-5 (H5); `norm_before_gate` because this
            # repository's default for it FLIPPED on 2026-08-15, and a value that has
            # moved once is not a value to inherit. The reference Mamba-2 computes
            # `norm(y * silu(z))`, which is `norm_before_gate=False`.
            norm_epsilon=NORM_EPSILON,
            norm_before_gate=False,
            name=name,
        )

    if num_heads <= 0:
        raise ValueError(
            f"an attention mixer ({kind!r}) needs a positive num_heads, got {num_heads}"
        )
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
    head_dim = d_model // num_heads
    if not 0 <= rotary_emb_dim <= head_dim:
        raise ValueError(
            f"rotary_emb_dim ({rotary_emb_dim}) must be in [0, head_dim] where "
            f"head_dim = d_model // num_heads = {d_model} // {num_heads} = {head_dim}"
        )

    # DECISION plan-2026-09-09T042752-6d66ac56/D-005: partial rotary is expressed as the
    # FRACTION `rotary_emb_dim / head_dim`, never hardcoded and never rounded to 1.0 --
    # every shipped variant asks for half its head width (32 of 64, 48 of 96, 64 of 128),
    # so a hardcoded 1.0 doubles the rotated span at every stage with no shape symptom.
    # This factory entry's RoPE is the INTERLEAVED (GPT-J) pairing while the reference
    # uses split-half (GPT-NeoX); that divergence is deliberate and is only absorbable
    # because no reference checkpoint is ever loaded. Do NOT "fix" it by swapping in the
    # split-half layer without also permuting the q/k projection rows -- and do not
    # assume RoPE is inert here: it was MEASURED live at 7.33e-01 (step 2(d)), against a
    # rope_percentage=0.0 floor of ~3e-07. See decisions.md D-005 and D-010(d).
    rope_percentage = rotary_emb_dim / head_dim

    return create_attention_layer(
        "group_query",
        name=name,
        dim=d_model,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=max_seq_len,
        rope_percentage=rope_percentage,
        rope_theta=ROPE_THETA,
        dropout_rate=0.0,
        use_bias=False,
    )


def build_mlp(d_model: int, name: Optional[str] = None) -> keras.layers.Layer:
    """Build the SwiGLU MLP that accompanies an uppercase layout letter.

    :param d_model: Model width; also the MLP's output width.
    :type d_model: int
    :param name: Layer name.
    :type name: Optional[str]
    :returns: The SwiGLU layer, sized ``round_up(8 * d_model / 3, 128)``.
    :rtype: keras.layers.Layer
    """
    return create_ffn_layer(
        "swiglu",
        name=name,
        output_dim=d_model,
        # No `hidden_dim`: an explicit hidden width is used verbatim and DISCARDS the
        # 2/3-rule derivation these two knobs exist to drive.
        ffn_expansion_factor=SWIGLU_EXPANSION_FACTOR,
        ffn_multiple_of=SWIGLU_MULTIPLE_OF,
        use_bias=False,
    )


# ---------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.hnet.components")
class HNetBlock(keras.layers.Layer):
    """One pre-norm residual block: a mixer, and for uppercase letters a SwiGLU MLP.

    .. code-block:: text

        x ──┬──────────────────────────────► + ──┬────────────────────────► + ──► out
            │                                ▲   │                          ▲
            └─ norm1 ─► mixer ───────────────┘   └─ norm2 ─► mlp (M/T only)─┘

    Both norms are RMSNorm at ``epsilon=1e-5``. The block owns no residual-scaling
    initializer: the reference's depth-scaled ``0.02/sqrt(n_residuals)`` init is applied
    from the outside, over the whole nested hierarchy, and belongs to the model.

    :param d_model: Model width.
    :type d_model: int
    :param kind: Layout letter -- ``"m"``, ``"M"``, ``"t"`` or ``"T"``. Case selects
        whether the MLP branch exists.
    :type kind: str
    :param num_heads: Attention heads, for ``t``/``T``.
    :type num_heads: int
    :param rotary_emb_dim: Rotated head dimensions, for ``t``/``T``.
    :type rotary_emb_dim: int
    :param max_seq_len: Largest position the RoPE tables cover.
    :type max_seq_len: int
    :param d_state: Mamba-2 state width, for ``m``/``M``.
    :type d_state: int
    :param d_conv: Mamba-2 convolution width, for ``m``/``M``.
    :type d_conv: int
    :param expand: Mamba-2 expansion factor, for ``m``/``M``.
    :type expand: int
    :param headdim: Mamba-2 SSM head width, for ``m``/``M``.
    :type headdim: int
    :param norm_epsilon: RMSNorm epsilon. Defaults to :data:`NORM_EPSILON`; the
        normalization factory's own default (``1e-6``) is never used.
    :type norm_epsilon: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :raises ValueError: if ``kind`` is not a layout letter, or if an attention block's
        head configuration is inconsistent.

    Example:
        >>> block = HNetBlock(d_model=64, kind="T", num_heads=4, rotary_emb_dim=8)
        >>> y = block(keras.random.normal((2, 16, 64)))
        >>> y.shape
        (2, 16, 64)
    """

    def __init__(
            self,
            d_model: int,
            kind: str,
            num_heads: int = 0,
            rotary_emb_dim: int = 0,
            max_seq_len: int = 2048,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            norm_epsilon: float = NORM_EPSILON,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if kind not in LAYOUT_LETTERS:
            raise ValueError(
                f"unknown layout letter {kind!r}; expected one of {LAYOUT_LETTERS}"
            )
        if d_model < 1:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model
        self.kind = kind
        self.num_heads = num_heads
        self.rotary_emb_dim = rotary_emb_dim
        self.max_seq_len = max_seq_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.norm_epsilon = norm_epsilon

        self.mixer_family = MIXER_MAMBA2 if kind.lower() == "m" else MIXER_ATTENTION
        self.has_mlp = kind.isupper()

        self.norm1 = create_normalization_layer(
            "rms_norm", name="norm1", epsilon=norm_epsilon
        )
        self.mixer = build_mixer(
            kind=kind,
            d_model=d_model,
            num_heads=num_heads,
            rotary_emb_dim=rotary_emb_dim,
            max_seq_len=max_seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            name="mixer",
        )
        if self.has_mlp:
            self.norm2 = create_normalization_layer(
                "rms_norm", name="norm2", epsilon=norm_epsilon
            )
            self.mlp = build_mlp(d_model, name="mlp")
        else:
            self.norm2 = None
            self.mlp = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly.

        A sub-layer left to lazy construction is not restored by ``load_weights`` on a
        freshly constructed model, so each one is built here against the shape it will
        actually see -- all of them ``(batch, seq_len, d_model)``.

        :param input_shape: ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: if the input width disagrees with ``d_model``.
        """
        if self.built:
            return
        if len(input_shape) != 3:
            raise ValueError(
                f"HNetBlock expects a rank-3 (batch, seq_len, d_model) input, got "
                f"shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.d_model:
            raise ValueError(
                f"HNetBlock was configured for d_model={self.d_model} but the input "
                f"has width {input_shape[-1]}"
            )

        self.norm1.build(input_shape)
        self.mixer.build(input_shape)
        if self.has_mlp:
            self.norm2.build(input_shape)
            self.mlp.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the pre-norm block.

        :param inputs: ``(batch, seq_len, d_model)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Keep predicate, ``1 = attend``, of rank 2, 3 or 4. Used
            only by ``t``/``T`` blocks; the Mamba-2 mixer takes no mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :returns: ``(batch, seq_len, d_model)``.
        :rtype: keras.KerasTensor
        """
        normed = self.norm1(inputs, training=training)
        if self.mixer_family == MIXER_ATTENTION:
            mixed = self.mixer(
                normed, attention_mask=attention_mask, training=training
            )
        else:
            mixed = self.mixer(normed, training=training)
        hidden = inputs + mixed

        if self.has_mlp:
            hidden = hidden + self.mlp(
                self.norm2(hidden, training=training), training=training
            )
        return hidden

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """:param input_shape: ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :returns: The same shape -- a residual block is width- and length-preserving.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """:returns: Every constructor argument, for a value round trip.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "kind": self.kind,
            "num_heads": self.num_heads,
            "rotary_emb_dim": self.rotary_emb_dim,
            "max_seq_len": self.max_seq_len,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "norm_epsilon": self.norm_epsilon,
        })
        return config


@register_dl_technique("dl_techniques.models.hnet.components")
class HNetIsotropic(keras.layers.Layer):
    """A flat stack of :class:`HNetBlock` s followed by one RMSNorm.

    This is the port of ``hnet/modules/isotropic.py``'s ``Isotropic`` on its
    padded/masked path. The packed (``cu_seqlens``) path is deliberately absent: it is a
    CUDA kernel-API artifact, and dropping it removes the branch this class would
    otherwise carry in every method (plan decision D-007, G3).

    The causal keep predicate is built ONCE per call from the current sequence length and
    shared by every attention block in the stack, since every block at a stage shares the
    stage's ``window_size``. It is not built at all when the layout has no attention
    letter.

    :param d_model: Model width for this stage.
    :type d_model: int
    :param layout: The layout string, e.g. ``"m4"``, ``"T22"``, ``"m4T1"``.
    :type layout: str
    :param num_heads: Attention heads at this stage.
    :type num_heads: int
    :param rotary_emb_dim: Rotated head dimensions at this stage.
    :type rotary_emb_dim: int
    :param window_size: Sliding-window lookback at this stage; ``-1`` for full causal
        context.
    :type window_size: int
    :param max_seq_len: Largest position the RoPE tables cover.
    :type max_seq_len: int
    :param d_state: Mamba-2 state width.
    :type d_state: int
    :param d_conv: Mamba-2 convolution width.
    :type d_conv: int
    :param expand: Mamba-2 expansion factor.
    :type expand: int
    :param headdim: Mamba-2 SSM head width.
    :type headdim: int
    :param norm_epsilon: RMSNorm epsilon, for the blocks and the final norm alike.
    :type norm_epsilon: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.

    :raises ValueError: if ``layout`` is malformed, or a block's configuration is
        inconsistent.

    Example:
        >>> stack = HNetIsotropic(d_model=64, layout="m2T1", num_heads=4,
        ...                       rotary_emb_dim=8, window_size=3)
        >>> stack(keras.random.normal((2, 16, 64))).shape
        (2, 16, 64)
    """

    def __init__(
            self,
            d_model: int,
            layout: str,
            num_heads: int = 0,
            rotary_emb_dim: int = 0,
            window_size: int = -1,
            max_seq_len: int = 2048,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            norm_epsilon: float = NORM_EPSILON,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        spec = parse_isotropic_layout(layout)
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError(
                f"window_size must be an int, got {type(window_size).__name__}: "
                f"{window_size!r}"
            )
        if window_size < -1:
            raise ValueError(
                f"window_size must be -1 (unlimited context) or a non-negative "
                f"lookback distance, got {window_size}"
            )

        self.d_model = d_model
        self.layout = layout
        self.num_heads = num_heads
        self.rotary_emb_dim = rotary_emb_dim
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.norm_epsilon = norm_epsilon

        self.spec = spec
        self.arch_full: Tuple[str, ...] = spec.arch_full
        self.has_attention = any(
            letter.lower() == "t" for letter in self.arch_full
        )

        # A FLAT list of blocks. Nesting layers one list deeper loses their weights on
        # save/restore, so the expansion of "m4" into four blocks is flattened here.
        self.blocks: List[HNetBlock] = [
            HNetBlock(
                d_model=d_model,
                kind=letter,
                num_heads=num_heads,
                rotary_emb_dim=rotary_emb_dim,
                max_seq_len=max_seq_len,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                norm_epsilon=norm_epsilon,
                name=f"block_{index}",
            )
            for index, letter in enumerate(self.arch_full)
        ]
        self.final_norm = create_normalization_layer(
            "rms_norm", name="final_norm", epsilon=norm_epsilon
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every block and the final norm.

        :param input_shape: ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return
        for block in self.blocks:
            block.build(input_shape)
        self.final_norm.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            padding_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the stack.

        :param inputs: ``(batch, seq_len, d_model)``.
        :type inputs: keras.KerasTensor
        :param padding_mask: Optional rank-2 ``(batch, seq_len)`` key-validity keep
            predicate,
            ``1`` for a real token. It is AND-ed into the causal predicate so padded
            keys are not attended. It does NOT reach the Mamba-2 mixer, which has no
            mask knob -- exactly as in the reference's unpacked path. It is spelled
            ``padding_mask`` rather than ``mask`` on purpose: Keras reserves the name
            ``mask`` on ``call`` and would auto-populate it from an upstream layer's
            ``_keras_mask``, silently substituting a mask this stack never asked for.
        :type padding_mask: Optional[keras.KerasTensor]
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :returns: ``(batch, seq_len, d_model)``.
        :rtype: keras.KerasTensor
        """
        attention_mask = None
        if self.has_attention:
            seq_len = keras.ops.shape(inputs)[1]
            attention_mask = _combine_keep_masks(
                build_causal_keep_mask(seq_len, self.window_size), padding_mask
            )

        hidden = inputs
        for block in self.blocks:
            hidden = block(
                hidden, attention_mask=attention_mask, training=training
            )
        return self.final_norm(hidden, training=training)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """:param input_shape: ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :returns: The same shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """:returns: Every constructor argument, for a value round trip.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "layout": self.layout,
            "num_heads": self.num_heads,
            "rotary_emb_dim": self.rotary_emb_dim,
            "window_size": self.window_size,
            "max_seq_len": self.max_seq_len,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "norm_epsilon": self.norm_epsilon,
        })
        return config
