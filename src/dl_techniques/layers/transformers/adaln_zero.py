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

    x -----> LayerNorm (no affine) --> modulate(shift_msa, scale_msa)
                                                       |
                                                       v
                             causal MultiHeadAttention (self-attn)
                                                       |
                                              gate_msa * (.)
                                                       |
    x = x + gate_msa * attn(...)  <-------------------+

    x -----> LayerNorm (no affine) --> modulate(shift_mlp, scale_mlp)
                                                       |
                                                       v
                                 2-layer MLP (Dense-GELU-Dense)
                                                       |
                                              gate_mlp * (.)
                                                       |
    x = x + gate_mlp * mlp(...)   <-------------------+

Where `modulate(h, shift, scale) = h * (1 + scale) + shift`.

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


@keras.saving.register_keras_serializable()
class AdaLNZeroConditionalBlock(keras.layers.Layer):
    """Transformer block with AdaLN-zero conditioning and causal self-attention.

    Two inputs per call: content `x` of shape `(B, T, D)` and conditioning
    `c` of shape `(B, T, D)` (or broadcastable to it). The conditioning drives
    six modulation streams (shift/scale/gate for the attention sub-block and
    shift/scale/gate for the MLP sub-block) via a single zero-initialized
    Dense layer. At init the block is the identity map in `x`.

    :param dim: model (hidden) dimension.
    :param num_heads: number of attention heads.
    :param dim_head: per-head dimension for MultiHeadAttention.
    :param mlp_dim: hidden dimension of the MLP sub-block.
    :param dropout: dropout rate applied in attention, MLP, and the residual
        branches of the block. Defaults to 0.0.
    :param use_causal_mask: if True (default), applies causal self-attention
        mask — matches upstream LeWM `is_causal=True`.
    :param eps: LayerNorm epsilon. Defaults to 1e-6 (matches upstream).
    :param kwargs: passthrough to `keras.layers.Layer`.
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

        # Two elementwise-affine=False LayerNorms (gamma/beta come from AdaLN).
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=eps, center=False, scale=False, name="norm1"
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=eps, center=False, scale=False, name="norm2"
        )

        # Multi-head self-attention. We rely on Keras' built-in MHA which
        # matches PyTorch SDPA semantics for the decoder self-attention case.
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_head,
            value_dim=dim_head,
            dropout=dropout,
            use_bias=True,
            name="attn",
        )

        # Two-layer MLP: Dense(mlp_dim) -> GELU -> Dropout -> Dense(dim) -> Dropout
        self.mlp_fc1 = keras.layers.Dense(mlp_dim, activation=None, name="mlp_fc1")
        self.mlp_act = keras.layers.Activation("gelu", name="mlp_act")
        self.mlp_drop1 = keras.layers.Dropout(dropout, name="mlp_drop1")
        self.mlp_fc2 = keras.layers.Dense(dim, activation=None, name="mlp_fc2")
        self.mlp_drop2 = keras.layers.Dropout(dropout, name="mlp_drop2")

        # AdaLN modulation: SiLU -> zero-initialized Linear(6 * dim).
        # Zero-init ensures the block is identity at initialization.
        self.adaLN_act = keras.layers.Activation("silu", name="adaLN_act")
        self.adaLN_linear = keras.layers.Dense(
            6 * dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="adaLN_linear",
        )

    def build(self, input_shape: Any) -> None:
        """Build sublayers explicitly for robust serialization.

        :param input_shape: either a tuple of two shapes `[x_shape, c_shape]`
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

        # MHA: query=key=value=x — build with (x_shape, x_shape).
        self.attn.build(query_shape=x_shape, value_shape=x_shape, key_shape=x_shape)

        # MLP builds on (B, T, D).
        self.mlp_fc1.build(x_shape)
        mlp_hidden_shape = tuple(list(x_shape[:-1]) + [self.mlp_dim])
        self.mlp_act.build(mlp_hidden_shape)
        self.mlp_drop1.build(mlp_hidden_shape)
        self.mlp_fc2.build(mlp_hidden_shape)
        self.mlp_drop2.build(x_shape)

        # AdaLN linear operates on c (B, T, D).
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

        :param inputs: list/tuple `[x, c]` where `x` is the content tensor
            `(B, T, D)` and `c` is the conditioning tensor `(B, T, D)` or
            broadcastable to `x`.
        :param training: passed through to dropout / MHA.
        :return: tensor of shape `(B, T, D)` — same as `x`.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "AdaLNZeroConditionalBlock expects `inputs=[x, c]` (a list/tuple "
                f"of length 2). Got: {type(inputs)} with len="
                f"{len(inputs) if hasattr(inputs, '__len__') else 'N/A'}"
            )
        x, c = inputs

        # AdaLN: SiLU -> Linear -> chunk(6) along last axis.
        mod = self.adaLN_linear(self.adaLN_act(c))
        # Split into 6 tensors of shape (B, T, D).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.split(
            mod, 6, axis=-1
        )

        # --- Attention sub-block with gated residual ---
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(
            query=h, value=h, key=h,
            use_causal_mask=self.use_causal_mask,
            training=training,
        )
        x = x + gate_msa * h

        # --- MLP sub-block with gated residual ---
        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp_fc1(h)
        h = self.mlp_act(h)
        h = self.mlp_drop1(h, training=training)
        h = self.mlp_fc2(h)
        h = self.mlp_drop2(h, training=training)
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
        })
        return config
