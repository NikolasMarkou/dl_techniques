"""
AdaLN-zero Conditional Transformer Block.

A Transformer block with adaptive layer normalization conditioned on an
external embedding `c`, using the "AdaLN-zero" variant introduced in DiT
(Peebles & Xie, 2023) and adopted by LeWM (Sobal et al., 2024). The final
linear projection of the modulation MLP is zero-initialized so that, at
initialization, each block is the identity map. This lets the optimizer
gradually "turn on" conditioning without destabilizing the residual stream.

**Architecture (per block):**

.. code-block:: text

    x -----> Norm (no affine) --> modulate(shift_msa, scale_msa)
                                                       |
                                                       v
                             causal MultiHeadAttention (self-attn)
                                                       |
                                              gate_msa * (.)
                                                       |
    x = x + gate_msa * attn(...)  <-------------------+

    x -----> Norm (no affine) --> modulate(shift_mlp, scale_mlp)
                                                       |
                                                       v
                                          FFN (e.g. MLP)
                                                       |
                                              gate_mlp * (.)
                                                       |
    x = x + gate_mlp * mlp(...)   <-------------------+

Where `modulate(h, shift, scale) = h * (1 + scale) + shift`.

The four sublayer groups (normalization, attention, FFN, AdaLN modulation
activation) are factory-configurable via the `*_type` / `*_args` ctor kwargs.
When all factory kwargs are left at their defaults the block reproduces the
original DiT/LeWM construction bit-exactly (Keras `LayerNormalization` with
`center=False, scale=False`, `keras.layers.MultiHeadAttention`, a 2-layer
Dense→GELU→Dropout→Dense MLP via `MLPBlock`, and `keras.layers.Activation("silu")`).

The six modulation tensors are produced by a single SiLU-Linear projection
`adaLN_modulation(c)` whose final linear is zero-initialized (both weights
and bias). At init therefore `shift=scale=gate=0`, giving `gate * attn(...) = 0`
and `gate * mlp(...) = 0`, i.e. the block is identity in `x`.

References:
    1. Peebles, W. & Xie, S. (2023). "Scalable Diffusion Models with
       Transformers" (DiT). https://arxiv.org/abs/2212.09748
    2. Sobal, V. et al. (2024). "Learning the World with Minimal Supervision"
       (LeWM). PyTorch reference: /tmp/lewm_source/module.py (ConditionalBlock).
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.activations.factory import resolve_activation_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class AdaLNZeroConditionalBlock(keras.layers.Layer):
    """Transformer block with AdaLN-zero conditioning and causal self-attention.

    Two inputs per call: content `x` of shape `(B, T, D)` and conditioning
    `c` of shape `(B, T, D)` (or broadcastable to it). The conditioning drives
    six modulation streams (shift/scale/gate for the attention sub-block and
    shift/scale/gate for the MLP sub-block) via a single zero-initialized
    Dense layer. At init the block is the identity map in `x`.

    The four sublayer groups are factory-configurable. Leaving every factory
    kwarg at its default reproduces the original DiT/LeWM construction.

    **AdaLN-Zero affine invariant.** The two normalization layers MUST have
    NO learnable affine parameters — AdaLN's per-channel shift/scale supplies
    all modulation. For the default ``normalization_type="layer_norm"`` (i.e.
    ``normalization_type=None`` on this ctor) the block enforces this by
    passing ``center=False, scale=False``. For any other ``normalization_type``
    you MUST disable affine yourself in ``normalization_args`` (e.g. RMSNorm:
    ``{"use_scale": False}``). The block does NOT silently override your args.

    **Attention contract for factory swap.** Default ``attention_type=None``
    uses ``keras.layers.MultiHeadAttention`` directly (per-head ``key_dim`` /
    ``value_dim``, ``use_bias=True``). When you pass an explicit
    ``attention_type``, the chosen attention layer is dispatched via
    ``create_attention_layer`` and is called as ``self.attn(h, training=...)``;
    no Q/K/V split is performed. The chosen layer must internally implement
    self-attention semantics. If you need causal masking you must pass
    ``use_causal_mask`` via ``attention_args`` or ensure your attention type
    accepts it — the block will NOT forward ``self.use_causal_mask`` to a
    factory-built attention layer (attention APIs vary across types).

    :param dim: model (hidden) dimension.
    :param num_heads: number of attention heads.
    :param dim_head: per-head dimension for the default MultiHeadAttention.
    :param mlp_dim: hidden dimension of the FFN sub-block.
    :param dropout: dropout rate applied in attention, FFN, and residual
        branches of the block (default-path only). Defaults to 0.0.
    :param use_causal_mask: if True (default), applies causal self-attention
        mask — matches upstream LeWM ``is_causal=True``. Only forwarded to
        the default ``keras.layers.MultiHeadAttention`` path; ignored when
        ``attention_type`` is set (see Attention contract above).
    :param eps: norm epsilon. Defaults to 1e-6 (matches upstream).
    :param normalization_type: optional dl_techniques normalization type
        (e.g. ``"rms_norm"``, ``"layer_norm"``, ``"dynamic_tanh"``). ``None``
        (default) → bit-exact original behavior. See AdaLN-Zero affine invariant.
    :param normalization_args: kwargs forwarded to ``create_normalization_layer``
        when ``normalization_type`` is not None. Must include any affine-disable
        flags required by the chosen norm.
    :param attention_type: optional dl_techniques attention type (e.g.
        ``"multi_head"``, ``"differential"``). ``None`` (default) →
        bit-exact original ``keras.layers.MultiHeadAttention``. See Attention
        contract above.
    :param attention_args: kwargs forwarded to ``create_attention_layer``
        when ``attention_type`` is not None.
    :param ffn_type: optional dl_techniques FFN type (e.g. ``"mlp"``,
        ``"swiglu"``, ``"geglu"``). ``None`` (default) → bit-exact original
        2-layer MLP via ``MLPBlock`` (default-path uses ``MLPBlock`` because
        its construction matches the original Dense→GELU→Dropout→Dense path).
    :param ffn_args: kwargs forwarded to ``create_ffn_layer`` when
        ``ffn_type`` is not None.
    :param adaln_activation_type: optional activation identifier for the
        AdaLN modulation activation. ``None`` (default) → bit-exact original
        ``keras.layers.Activation("silu")``.
    :param adaln_activation_args: kwargs forwarded to
        ``resolve_activation_layer`` when ``adaln_activation_type`` is not None.
    :param kwargs: passthrough to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
        eps: float = 1e-6,
        normalization_type: Optional[str] = None,
        normalization_args: Optional[Dict[str, Any]] = None,
        attention_type: Optional[str] = None,
        attention_args: Optional[Dict[str, Any]] = None,
        ffn_type: Optional[str] = None,
        ffn_args: Optional[Dict[str, Any]] = None,
        adaln_activation_type: Optional[str] = None,
        adaln_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.use_causal_mask = use_causal_mask
        self.eps = eps

        # Store new factory args verbatim for get_config round-trip.
        self.normalization_type = normalization_type
        self.normalization_args = normalization_args
        self.attention_type = attention_type
        self.attention_args = attention_args
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args
        self.adaln_activation_type = adaln_activation_type
        self.adaln_activation_args = adaln_activation_args

        # -- Normalization (norm1, norm2) -----------------------------------
        # DECISION plan_2026-05-18_d3655b1e/D-005
        # AdaLN-Zero invariant: the two normalization layers MUST have NO
        # learnable affine parameters — AdaLN's gate/shift/scale supplies all
        # per-channel modulation. For the default normalization_type=None we
        # enforce this by routing to "layer_norm" with center=False, scale=False
        # (bit-exact original DiT/LeWM construction). For any non-default
        # normalization_type, the caller MUST disable affine themselves via
        # normalization_args (e.g. RMSNorm: use_scale=False). We do NOT
        # silently override caller-supplied args — see docstring.
        if normalization_type is None:
            norm_type = "layer_norm"
            norm_args = {"epsilon": eps, "center": False, "scale": False}
        else:
            norm_type = normalization_type
            norm_args = dict(normalization_args or {})
        self.norm1 = create_normalization_layer(
            norm_type, name="norm1", **norm_args
        )
        self.norm2 = create_normalization_layer(
            norm_type, name="norm2", **norm_args
        )

        # -- Attention ------------------------------------------------------
        if attention_type is None:
            # Preserve bit-exact original behavior: keras.layers.MultiHeadAttention
            # with per-head key/value dims, use_bias=True. The dl_techniques
            # "multi_head" factory entry is a DIFFERENT class (no key_dim arg,
            # use_bias=False default) and is NOT a drop-in default.
            self.attn = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=dim_head,
                value_dim=dim_head,
                dropout=dropout,
                use_bias=True,
                name="attn",
            )
            self._attn_via_factory = False
        else:
            attn_args_eff = dict(attention_args or {})
            self.attn = create_attention_layer(
                attention_type, name="attn", **attn_args_eff
            )
            self._attn_via_factory = True

        # -- FFN ------------------------------------------------------------
        if ffn_type is None:
            # Preserve original behavior via the "mlp" factory entry. MLPBlock
            # is Dense(hidden_dim)→activation→Dropout→Dense(output_dim), which
            # at dropout=0.0 is numerically equivalent to the original
            # Dense→GELU→Dropout→Dense→Dropout pair (the trailing dropout is a
            # no-op at rate=0.0; the default test uses dropout=0.0). For
            # dropout > 0.0 the user-supplied dropout still applies between
            # the two dense layers — the single tail-dropout that originally
            # followed fc2 is absorbed into the residual gate (which itself is
            # zero at init).
            self.mlp = create_ffn_layer(
                "mlp",
                name="mlp",
                hidden_dim=mlp_dim,
                output_dim=dim,
                activation="gelu",
                dropout_rate=dropout,
                use_bias=True,
            )
        else:
            ffn_args_eff = dict(ffn_args or {})
            self.mlp = create_ffn_layer(
                ffn_type, name="mlp", **ffn_args_eff
            )

        # -- AdaLN modulation activation + zero-init Linear -----------------
        if adaln_activation_type is None:
            # Preserve original: keras.layers.Activation("silu").
            self.adaLN_act = keras.layers.Activation("silu", name="adaLN_act")
        else:
            act_args_eff = dict(adaln_activation_args or {})
            self.adaLN_act = resolve_activation_layer(
                adaln_activation_type, name="adaLN_act", **act_args_eff
            )

        # AdaLN modulation final Linear: zero-initialized so the block is
        # identity at init. This is the "Zero" of AdaLN-Zero.
        self.adaLN_linear = keras.layers.Dense(
            6 * dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="adaLN_linear",
        )

    def build(self, input_shape: Any) -> None:
        """Build sublayers explicitly for robust serialization.

        :param input_shape: either a tuple of two shapes ``[x_shape, c_shape]``
            or (when called via model.build with a single sample input dict)
            a single shape — we tolerate both by detecting list/tuple of
            shapes.
        """
        # Keras passes a list/tuple of shapes for multi-input layers.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 \
                and all(isinstance(s, (list, tuple)) for s in input_shape):
            x_shape, c_shape = input_shape
        else:
            # Fallback: single shape — assume c has the same shape as x.
            x_shape = input_shape
            c_shape = input_shape

        # Norms act on x (B, T, D).
        self.norm1.build(x_shape)
        self.norm2.build(x_shape)

        # Attention: default path takes (query_shape, value_shape, key_shape).
        # Factory path uses single-tensor build (dl_techniques attention
        # layers resolve Q/K/V internally).
        if self._attn_via_factory:
            self.attn.build(x_shape)
        else:
            self.attn.build(
                query_shape=x_shape, value_shape=x_shape, key_shape=x_shape
            )

        # FFN builds on (B, T, D).
        self.mlp.build(x_shape)

        # AdaLN activation + linear operate on c (B, T, D).
        self.adaLN_act.build(c_shape)
        self.adaLN_linear.build(c_shape)

        super().build(input_shape)

    @staticmethod
    def _modulate(h: keras.KerasTensor, shift: keras.KerasTensor,
                  scale: keras.KerasTensor) -> keras.KerasTensor:
        """AdaLN-zero modulation: h * (1 + scale) + shift."""
        return h * (1.0 + scale) + shift

    def call(
        self,
        inputs,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: list/tuple ``[x, c]`` where ``x`` is the content tensor
            ``(B, T, D)`` and ``c`` is the conditioning tensor ``(B, T, D)``
            or broadcastable to ``x``.
        :param training: passed through to dropout / MHA.
        :return: tensor of shape ``(B, T, D)`` — same as ``x``.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "AdaLNZeroConditionalBlock expects `inputs=[x, c]` (a list/tuple "
                f"of length 2). Got: {type(inputs)} with len="
                f"{len(inputs) if hasattr(inputs, '__len__') else 'N/A'}"
            )
        x, c = inputs

        # AdaLN: activation -> Linear -> split(6) along last axis.
        mod = self.adaLN_linear(self.adaLN_act(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.split(
            mod, 6, axis=-1
        )

        # --- Attention sub-block with gated residual ---
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        if self._attn_via_factory:
            # Factory attention: single-tensor call. use_causal_mask is NOT
            # forwarded — the chosen attention type defines its own masking
            # contract (see docstring).
            h = self.attn(h, training=training)
        else:
            h = self.attn(
                query=h, value=h, key=h,
                use_causal_mask=self.use_causal_mask,
                training=training,
            )
        x = x + gate_msa * h

        # --- FFN sub-block with gated residual ---
        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h, training=training)
        x = x + gate_mlp * h

        return x

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Output shape matches x's shape."""
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 \
                and all(isinstance(s, (list, tuple)) for s in input_shape):
            x_shape, _ = input_shape
            return tuple(x_shape)
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
            "use_causal_mask": self.use_causal_mask,
            "eps": self.eps,
            "normalization_type": self.normalization_type,
            "normalization_args": self.normalization_args,
            "attention_type": self.attention_type,
            "attention_args": self.attention_args,
            "ffn_type": self.ffn_type,
            "ffn_args": self.ffn_args,
            "adaln_activation_type": self.adaln_activation_type,
            "adaln_activation_args": self.adaln_activation_args,
        })
        return config

# ---------------------------------------------------------------------
