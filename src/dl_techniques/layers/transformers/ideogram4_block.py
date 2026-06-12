"""
Ideogram4 DiT transformer block + final layer (4-stream tanh-gated AdaLN).

This module ports the Ideogram4 ``Ideogram4TransformerBlock`` and
``Ideogram4FinalLayer`` to Keras 3. The block is structurally distinct from the
repository's existing ``AdaLNZeroConditionalBlock`` (which is 6-stream,
shift+scale, SiLU-gated, pre-norm-only); see decisions D-002.

Block conditioning (the structurally-novel bits, ported exactly):

- ``adaln_modulation`` is a single ``Dense(4 * hidden_size, use_bias=True)``
  applied to ``adaln_input``; its output is split along the last axis into four
  equal chunks ``(scale_msa, gate_msa, scale_mlp, gate_mlp)``.
- **Gates** are passed through ``tanh``; **scales** are ``1 + x``. This is
  *scale-only* AdaLN (NO shift) -- unlike standard DiT AdaLN-zero which also has
  a shift term. Do NOT add a shift.
- ``adaln_input`` may be ``(B, 1, adaln_dim)`` (per-sample, broadcast over L) or
  ``(B, L, adaln_dim)``. The four modulation tensors are kept rank-3 and
  broadcast over L accordingly.
- **Four RMSNorms** form a sandwich: ``norm1`` is the PRE-norm (on the sublayer
  input) and ``norm2`` is a POST-norm applied to the sublayer *output* before the
  gate and the residual add::

      x = x + tanh(gate_msa) * attn_norm2(attn(attn_norm1(x) * (1 + scale_msa)))
      x = x + tanh(gate_mlp) * ffn_norm2(ffn(ffn_norm1(x) * (1 + scale_mlp)))

  The post-norm-inside-the-residual is unusual but replicated exactly.

Final layer:

- ``LayerNormalization(center=False, scale=False, epsilon=1e-6)`` (no affine),
  then ``* (1 + adaln_modulation(silu(c)))`` (silu applied to ``c`` BEFORE the
  Dense), then ``Dense(out_channels, use_bias=True)``.

PyTorch reference (faithfully ported)::

    # block
    mod = self.adaln_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    gate_msa = tanh(gate_msa); gate_mlp = tanh(gate_mlp)
    scale_msa = 1 + scale_msa; scale_mlp = 1 + scale_mlp
    attn_out = self.attention(self.attention_norm1(x) * scale_msa, ...)
    x = x + gate_msa * self.attention_norm2(attn_out)
    x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))

    # final
    scale = 1 + self.adaln_modulation(F.silu(c))
    return self.linear(self.norm_final(x) * scale)

MLP note: the PyTorch ``Ideogram4MLP`` is a plain bias-free SwiGLU
``w2(silu(w1 x) * w3 x)`` with ``hidden_dim = intermediate_size``. The reused
``SwiGLUFFN`` is configured with ``ffn_expansion_factor=1`` and
``ffn_multiple_of=intermediate_size`` so its rounded hidden dim equals
``intermediate_size`` exactly, and ``use_bias=False`` for the bias-free match.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.ffn.swiglu_ffn import SwiGLUFFN
from dl_techniques.layers.attention.ideogram4_attention import Ideogram4Attention

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class Ideogram4TransformerBlock(keras.layers.Layer):
    """Ideogram4 DiT block: 4-stream tanh-gated AdaLN with an RMSNorm sandwich.

    :param hidden_size: Model / embedding dimensionality. Must be divisible by
        ``num_heads``.
    :type hidden_size: int
    :param intermediate_size: SwiGLU MLP hidden dimension (exact). Must be
        ``>= ceil(hidden_size * 2 / 3)`` so the reused ``SwiGLUFFN`` can express
        it exactly via ``ffn_multiple_of=intermediate_size``.
    :type intermediate_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param norm_eps: Epsilon for the four block RMSNorms. Defaults to ``1e-5``.
    :type norm_eps: float
    :param adaln_dim: Feature dimension of ``adaln_input`` (the conditioning).
    :type adaln_dim: int
    :param attention_eps: Epsilon for the attention's per-head QK RMSNorm.
        Defaults to ``1e-5`` (matching the PyTorch reference).
    :type attention_eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If any size is non-positive, ``hidden_size`` is not
        divisible by ``num_heads``, or ``intermediate_size`` is too small to be
        expressed exactly by ``SwiGLUFFN``.

    Input/Output:
        ``call(x, segment_ids, cos, sin, adaln_input)`` with
        ``x: (B, L, hidden_size)``, ``segment_ids: (B, L)``,
        ``cos / sin: (B, L, head_dim)``,
        ``adaln_input: (B, 1, adaln_dim)`` or ``(B, L, adaln_dim)``
        returns ``(B, L, hidden_size)``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        adaln_dim: int,
        norm_eps: float = 1e-5,
        attention_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- validation -------------------------------------------------
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(intermediate_size, int) or intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be a positive integer, got "
                f"{intermediate_size}"
            )
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}"
            )
        if not isinstance(adaln_dim, int) or adaln_dim <= 0:
            raise ValueError(
                f"adaln_dim must be a positive integer, got {adaln_dim}"
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        # SwiGLUFFN computes hidden = ceil(int(output_dim*factor*2/3) / mult)*mult.
        # With factor=1 and mult=intermediate_size, the rounded hidden equals
        # intermediate_size iff int(hidden_size*2/3) <= intermediate_size.
        min_intermediate = int(hidden_size * 2 / 3)
        if intermediate_size < min_intermediate:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be >= "
                f"int(hidden_size * 2/3) = {min_intermediate} so SwiGLUFFN can "
                f"express the exact MLP hidden dim."
            )
        if norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}")
        if attention_eps <= 0:
            raise ValueError(f"attention_eps must be positive, got {attention_eps}")

        # --- store config ----------------------------------------------
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.adaln_dim = adaln_dim
        self.norm_eps = float(norm_eps)
        self.attention_eps = float(attention_eps)

        # --- sub-layers (created here, built in build()) ---------------
        self.attention = Ideogram4Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            eps=self.attention_eps,
            name="attention",
        )
        # Plain bias-free SwiGLU with hidden_dim == intermediate_size exactly.
        self.feed_forward = SwiGLUFFN(
            output_dim=hidden_size,
            ffn_expansion_factor=1,
            ffn_multiple_of=intermediate_size,
            use_bias=False,
            name="feed_forward",
        )
        self.attention_norm1 = RMSNorm(
            axis=-1, epsilon=self.norm_eps, name="attention_norm1"
        )
        self.attention_norm2 = RMSNorm(
            axis=-1, epsilon=self.norm_eps, name="attention_norm2"
        )
        self.ffn_norm1 = RMSNorm(
            axis=-1, epsilon=self.norm_eps, name="ffn_norm1"
        )
        self.ffn_norm2 = RMSNorm(
            axis=-1, epsilon=self.norm_eps, name="ffn_norm2"
        )
        # adaln_input -> (scale_msa, gate_msa, scale_mlp, gate_mlp).
        self.adaln_modulation = keras.layers.Dense(
            4 * hidden_size, use_bias=True, name="adaln_modulation"
        )

        logger.debug(
            f"Initialized Ideogram4TransformerBlock(hidden_size={hidden_size}, "
            f"intermediate_size={intermediate_size}, num_heads={num_heads}, "
            f"adaln_dim={adaln_dim})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers explicitly.

        :param input_shape: Shape of ``x``, expected ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is not ``hidden_size``.
        """
        if len(input_shape) != 3 or input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Ideogram4TransformerBlock expects x of shape "
                f"(B, L, hidden_size={self.hidden_size}), got {input_shape}"
            )

        feature_shape = tuple(input_shape)

        self.attention.build(feature_shape)
        self.feed_forward.build(feature_shape)
        self.attention_norm1.build(feature_shape)
        self.attention_norm2.build(feature_shape)
        self.ffn_norm1.build(feature_shape)
        self.ffn_norm2.build(feature_shape)
        # adaln_modulation consumes (B, *, adaln_dim).
        self.adaln_modulation.build((input_shape[0], None, self.adaln_dim))

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        segment_ids: keras.KerasTensor,
        cos: keras.KerasTensor,
        sin: keras.KerasTensor,
        adaln_input: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the 4-stream tanh-gated AdaLN block.

        :param x: Token features ``(B, L, hidden_size)``.
        :type x: keras.KerasTensor
        :param segment_ids: Integer segment id per token ``(B, L)``.
        :type segment_ids: keras.KerasTensor
        :param cos: mRoPE cosine table ``(B, L, head_dim)``.
        :type cos: keras.KerasTensor
        :param sin: mRoPE sine table ``(B, L, head_dim)``.
        :type sin: keras.KerasTensor
        :param adaln_input: Conditioning ``(B, 1, adaln_dim)`` or
            ``(B, L, adaln_dim)``.
        :type adaln_input: keras.KerasTensor
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: ``(B, L, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        # adaln_input -> 4 equal chunks along the last axis. Kept rank-3 so a
        # (B, 1, adaln_dim) conditioning broadcasts over L.
        mod = self.adaln_modulation(adaln_input)  # (B, *, 4 * hidden_size)
        scale_msa, gate_msa, scale_mlp, gate_mlp = keras.ops.split(
            mod, 4, axis=-1
        )

        # Gates: tanh. Scales: 1 + x (scale-only, NO shift).
        gate_msa = keras.ops.tanh(gate_msa)
        gate_mlp = keras.ops.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        # --- attention sublayer: pre-norm * scale, attn, post-norm, gate ---
        attn_in = self.attention_norm1(x, training=training) * scale_msa
        attn_out = self.attention(
            attn_in, segment_ids, cos, sin, training=training
        )
        x = x + gate_msa * self.attention_norm2(attn_out, training=training)

        # --- ffn sublayer: pre-norm * scale, ffn, post-norm, gate ----------
        ffn_in = self.ffn_norm1(x, training=training) * scale_mlp
        ffn_out = self.feed_forward(ffn_in, training=training)
        x = x + gate_mlp * self.ffn_norm2(ffn_out, training=training)

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identical to ``x``'s shape).

        :param input_shape: Shape of ``x`` ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, L, hidden_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Dictionary with all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_heads": self.num_heads,
                "adaln_dim": self.adaln_dim,
                "norm_eps": self.norm_eps,
                "attention_eps": self.attention_eps,
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.layers")
class Ideogram4FinalLayer(keras.layers.Layer):
    """Ideogram4 DiT final layer: no-affine LayerNorm, scale-AdaLN, linear head.

    Applies ``LayerNormalization(center=False, scale=False, epsilon=1e-6)`` to
    ``x``, modulates by ``1 + adaln_modulation(silu(c))`` (silu applied BEFORE
    the Dense), then projects to ``out_channels`` with a biased Dense.

    :param hidden_size: Input feature dimensionality.
    :type hidden_size: int
    :param out_channels: Output channel count of the linear head.
    :type out_channels: int
    :param adaln_dim: Feature dimension of the conditioning ``c``.
    :type adaln_dim: int
    :param layernorm_eps: Epsilon for the no-affine LayerNorm. Defaults to
        ``1e-6`` (matching the PyTorch reference).
    :type layernorm_eps: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If any size is non-positive.

    Input/Output:
        ``call(x, c)`` with ``x: (B, L, hidden_size)`` and
        ``c: (B, 1, adaln_dim)`` or ``(B, L, adaln_dim)`` returns
        ``(B, L, out_channels)``.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        adaln_dim: int,
        layernorm_eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive integer, got {hidden_size}"
            )
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError(
                f"out_channels must be a positive integer, got {out_channels}"
            )
        if not isinstance(adaln_dim, int) or adaln_dim <= 0:
            raise ValueError(
                f"adaln_dim must be a positive integer, got {adaln_dim}"
            )
        if layernorm_eps <= 0:
            raise ValueError(
                f"layernorm_eps must be positive, got {layernorm_eps}"
            )

        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.adaln_dim = adaln_dim
        self.layernorm_eps = float(layernorm_eps)

        # No-affine LayerNorm (center=False, scale=False).
        self.norm_final = keras.layers.LayerNormalization(
            epsilon=self.layernorm_eps,
            center=False,
            scale=False,
            name="norm_final",
        )
        self.linear = keras.layers.Dense(
            out_channels, use_bias=True, name="linear"
        )
        self.adaln_modulation = keras.layers.Dense(
            hidden_size, use_bias=True, name="adaln_modulation"
        )

        logger.debug(
            f"Initialized Ideogram4FinalLayer(hidden_size={hidden_size}, "
            f"out_channels={out_channels}, adaln_dim={adaln_dim})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers explicitly.

        :param input_shape: Shape of ``x``, expected ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last input dimension is not ``hidden_size``.
        """
        if len(input_shape) != 3 or input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Ideogram4FinalLayer expects x of shape "
                f"(B, L, hidden_size={self.hidden_size}), got {input_shape}"
            )

        feature_shape = tuple(input_shape)
        self.norm_final.build(feature_shape)
        self.linear.build(feature_shape)
        self.adaln_modulation.build((input_shape[0], None, self.adaln_dim))

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        c: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the final layer.

        :param x: Token features ``(B, L, hidden_size)``.
        :type x: keras.KerasTensor
        :param c: Conditioning ``(B, 1, adaln_dim)`` or ``(B, L, adaln_dim)``.
        :type c: keras.KerasTensor
        :param training: Forwarded to sub-layers.
        :type training: Optional[bool]
        :return: ``(B, L, out_channels)``.
        :rtype: keras.KerasTensor
        """
        # silu applied to c BEFORE the Dense; scale is 1 + modulation.
        scale = 1.0 + self.adaln_modulation(keras.ops.silu(c))
        normed = self.norm_final(x, training=training)
        return self.linear(normed * scale)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``(B, L, out_channels)``.

        :param input_shape: Shape of ``x`` ``(B, L, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(B, L, out_channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.out_channels,)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        :return: Dictionary with all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "out_channels": self.out_channels,
                "adaln_dim": self.adaln_dim,
                "layernorm_eps": self.layernorm_eps,
            }
        )
        return config

# ---------------------------------------------------------------------
