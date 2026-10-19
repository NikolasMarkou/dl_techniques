"""Transformer layers for Timestral inference in Keras 3.

Parameter names match the released weights so they load with strict=True.
All tensors use the packed layout: one row per variate of every series in
the batch, (R, N, D) with N patches. Time attention runs per row over patches;
variate attention runs per patch over the rows of the same series.
"""

from dataclasses import dataclass

import keras
from keras import ops
from keras.layers import Layer
from typing import Optional, Tuple

from dl_techniques.models.time_series.timestral.config import LAYER_KIND_AXES


class RMSNorm(Layer):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x_f = ops.cast(x, "float32")
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * scale
        rms = ops.rsqrt(ops.mean(ops.square(x_f), axis=-1, keepdims=True) + self.eps)
        out = ops.cast(x_f * rms, x.dtype)
        return out * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "eps": self.eps})
        return config


class ResidualBlock(Layer):
    """Residual block: output(silu(hidden(x))) + residual(x).

    Used by the patch embedding and the quantile head.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

    def build(self, input_shape):
        self.hidden_layer = keras.layers.Dense(
            self.hidden_dims,
            use_bias=self.use_bias,
            name="hidden_layer",
        )
        self.output_layer = keras.layers.Dense(
            self.output_dims,
            use_bias=self.use_bias,
            name="output_layer",
        )
        self.residual_layer = keras.layers.Dense(
            self.output_dims,
            use_bias=self.use_bias,
            name="residual_layer",
        )
        self.activation = keras.activations.silu
        super().build(input_shape)

    def call(self, x):
        hidden = self.activation(self.hidden_layer(x))
        out = self.output_layer(hidden)
        return out + self.residual_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dims": self.input_dims,
            "hidden_dims": self.hidden_dims,
            "output_dims": self.output_dims,
            "use_bias": self.use_bias,
        })
        return config


class FusedSwiGLU(Layer):
    """SwiGLU MLP with fused input projection: gate * silu(up)."""

    def __init__(self, embed_dim: int, hidden_dim: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w12 = keras.layers.Dense(
            2 * self.hidden_dim,
            use_bias=self.use_bias,
            name="w12",
        )
        self.w3 = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            name="w3",
        )
        super().build(input_shape)

    def call(self, x):
        gate_up = self.w12(x)
        gate, up = ops.split(gate_up, 2, axis=-1)
        return self.w3(gate * ops.silu(up))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "use_bias": self.use_bias,
        })
        return config


class SinCosRoPE(Layer):
    """Rotary Position Embedding (sin/cos concatenation) with partial rotation.

    Only the first `rotary_dim` dimensions of each head rotate. The inverse-
    frequency table is zero-padded past `rotary_dim / 2`, so remaining dims get
    cos=1, sin=0 (identity rotation).
    """

    def __init__(
        self,
        dim: int,
        base: float,
        rotary_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not (0 < rotary_dim <= dim and rotary_dim % 2 == 0):
            raise ValueError(f"Invalid rotary_dim={rotary_dim} for dim={dim}")
        self.dim = dim
        self.base = base
        self.rotary_dim = rotary_dim

    def build(self, input_shape):
        # inv_freq shape: (dim // 2,)
        inv_freq = ops.zeros(self.dim // 2, dtype="float32")
        idx = ops.arange(0, self.rotary_dim, 2, dtype="float32")
        # Only first rotary_dim//2 entries are non-zero
        inv_freq = ops.slice_update(
            inv_freq,
            (0,),
            1.0 / (self.base ** (idx / self.rotary_dim)),
        )
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(self.dim // 2,),
            initializer=keras.initializers.Constant(inv_freq),
            trainable=False,
        )
        super().build(input_shape)

    def rebuild_inv_freq(self):
        """Recompute the table on the current device (for device transfers)."""
        inv_freq = ops.zeros(self.dim // 2, dtype="float32")
        idx = ops.arange(0, self.rotary_dim, 2, dtype="float32")
        inv_freq = ops.slice_update(
            inv_freq,
            (0,),
            1.0 / (self.base ** (idx / self.rotary_dim)),
        )
        self.inv_freq.assign(inv_freq)

    def call(self, x, position_ids):
        """Return (cos, sin) tables, each (R, N, head_dim).

        Args:
            x: Input tensor (R, N, D) - used for dtype/device.
            position_ids: Integer positions (R, N).
        """
        # position_ids: (R, N) -> (R, 1, N)
        # inv_freq: (dim//2,) -> (1, dim//2, 1)
        inv_freq_expanded = ops.expand_dims(self.inv_freq, axis=0)  # (1, dim//2)
        inv_freq_expanded = ops.expand_dims(inv_freq_expanded, axis=-1)  # (1, dim//2, 1)
        pos_expanded = ops.expand_dims(position_ids, axis=1)  # (R, 1, N)

        # freqs: (R, dim//2, N)
        freqs = ops.matmul(inv_freq_expanded, ops.cast(pos_expanded, "float32"))
        # Transpose to (R, N, dim//2)
        freqs = ops.transpose(freqs, (0, 2, 1))

        # emb: (R, N, dim) - concatenate freqs with itself
        emb = ops.concatenate([freqs, freqs], axis=-1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)

        return ops.cast(cos, x.dtype), ops.cast(sin, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "base": self.base,
            "rotary_dim": self.rotary_dim,
        })
        return config


def _rotate_half(x):
    """Rotate half the dimensions for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def _apply_rotary(q, k, cos, sin):
    """Apply rotary embeddings to q, k shaped (batch, seq, heads, head_dim)."""
    # cos, sin: (R, N, head_dim) -> broadcast over heads: (R, N, 1, head_dim)
    cos = ops.expand_dims(cos, axis=2)
    sin = ops.expand_dims(sin, axis=2)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


@dataclass(frozen=True)
class TwoStreamMask:
    """Time mask over [Enc; Dec] sequence plus its split point.

    Attention with this mask runs two query passes: context queries [:enc_len]
    over context keys only, and horizon queries [enc_len:] over all keys.
    This skips the always-empty context-to-horizon quadrant.
    """
    mask: keras.KerasTensor  # (R, 1, S, S) bool, True = attend
    enc_len: int


def _sdpa(query, key, value, mask=None):
    """Scaled dot-product attention on (batch, seq, heads, head_dim) tensors.

    Scores (B, H, S, T); `mask` (B, 1, S, T) broadcasts over heads; True =
    attend. Manual softmax path — `ops.attention` does not exist in this Keras.
    """
    head_dim = int(key.shape[-1])
    scores = ops.einsum("bshd,bthd->bhst", query, key) * (1.0 / head_dim ** 0.5)
    if mask is not None:
        mask_h = ops.broadcast_to(mask, ops.shape(scores))
        scores = ops.where(mask_h, scores, -1e30 * ops.ones_like(scores))
    attn = ops.exp(scores - ops.max(scores, axis=-1, keepdims=True))
    attn = attn / ops.sum(attn, axis=-1, keepdims=True)
    return ops.einsum("bhst,bthd->bshd", attn, value)


class Attention(Layer):
    """Multi-head attention over time axis (is_time=True) or variate axis.

    Time attention applies RoPE; variate attention has no positional encoding.
    With sdpa_sigmoid gate, attention output is multiplied by sigmoid(x @ W_g).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_time: bool,
        use_bias: bool,
        gate: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_time = is_time
        self.use_bias = use_bias
        self.gate_type = gate

    def build(self, input_shape):
        self.wqkv = keras.layers.Dense(
            3 * self.embed_dim,
            use_bias=self.use_bias,
            name="wqkv",
        )
        self.wo = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            name="wo",
        )
        if self.gate_type == "sdpa_sigmoid":
            self.gate = keras.layers.Dense(
                self.embed_dim,
                use_bias=self.use_bias,
                name="gate",
            )
        else:
            self.gate = None
        super().build(input_shape)

    def _qkv(self, x):
        """Split fused QKV projection. Weight layout: (qkv, head_dim, n_heads)."""
        batch, seq = int(ops.shape(x)[0]), int(ops.shape(x)[1])
        qkv = self.wqkv(x)
        qkv = ops.reshape(qkv, (batch, seq, 3, self.num_heads, self.head_dim))
        # Permute to (3, batch, seq, heads, head_dim)
        qkv = ops.transpose(qkv, (2, 0, 1, 3, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        return q, k, v

    def call(self, x, mask, cos, sin):
        rows, num_patches, _ = ops.shape(x)

        # Time attention: (R, N, D); Variate attention: (N, R, D)
        inputs = x if self.is_time else ops.transpose(x, (1, 0, 2))
        q, k, v = self._qkv(inputs)

        if self.is_time:
            q, k = _apply_rotary(q, k, cos, sin)

        if isinstance(mask, TwoStreamMask):
            # Two-stream attention for DHF
            assert self.is_time, "TwoStreamMask only for time attention"
            n = mask.enc_len
            # Encoder queries: causal over encoder keys
            out_enc = _sdpa(q[:, :n], k[:, :n], v[:, :n], mask.mask[:, :, :n, :n])
            # Decoder queries: full attention over all keys
            out_dec = _sdpa(q[:, n:], k, v, mask.mask[:, :, n:, :])
            out = ops.concatenate([out_enc, out_dec], axis=1)
        else:
            out = _sdpa(q, k, v, mask)

        if self.is_time:
            out = ops.reshape(out, ops.shape(inputs))
        else:
            out = ops.reshape(out, (ops.shape(x)[1], ops.shape(x)[0], self.embed_dim))
            out = ops.transpose(out, (1, 0, 2))

        if self.gate is not None:
            out = out * ops.sigmoid(self.gate(x))

        return self.wo(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "is_time": self.is_time,
            "use_bias": self.use_bias,
            "gate": self.gate_type,
        })
        return config


class TransformerBlock(Layer):
    """One layer kind: attention(s) in order, each pre-norm residual, then MLP."""

    def __init__(
        self,
        kind: str,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kind = kind
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.norm_eps = norm_eps
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.time_attn_gate = time_attn_gate
        self.variate_attn_gate = variate_attn_gate

    def build(self, input_shape):
        axes = LAYER_KIND_AXES[self.kind]
        self.attn_norms = []
        self.attns = []
        for i, is_time in enumerate(axes):
            norm = RMSNorm(self.embed_dim, self.norm_eps, name=f"attn_norm_{i}")
            attn = Attention(
                self.embed_dim,
                self.num_heads,
                is_time,
                self.attn_bias,
                self.time_attn_gate if is_time else self.variate_attn_gate,
                name=f"attn_{i}",
            )
            self.attn_norms.append(norm)
            self.attns.append(attn)

        self.mlp_norm = RMSNorm(self.embed_dim, self.norm_eps, name="mlp_norm")
        self.mlp = FusedSwiGLU(
            self.embed_dim,
            self.mlp_hidden_dim,
            self.mlp_bias,
            name="mlp",
        )
        super().build(input_shape)

    def call(self, x, time_mask=None, variate_mask=None, cos=None, sin=None):
        for norm, attn in zip(self.attn_norms, self.attns):
            residual = x
            x = norm(x=x)
            mask = time_mask if attn.is_time else variate_mask
            x = attn(x, mask=mask, cos=cos, sin=sin)
            x = x + residual
        # MLP
        x = x + self.mlp(self.mlp_norm(x=x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "kind": self.kind,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "norm_eps": self.norm_eps,
            "attn_bias": self.attn_bias,
            "mlp_bias": self.mlp_bias,
            "time_attn_gate": self.time_attn_gate,
            "variate_attn_gate": self.variate_attn_gate,
        })
        return config


class Transformer(Layer):
    """Stack of TransformerBlocks with shared RoPE and output norm."""

    def __init__(
        self,
        block_schedule: tuple[str, ...],
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        rope_theta: float,
        rope_dim: int,
        norm_eps: float,
        attn_bias: bool,
        mlp_bias: bool,
        time_attn_gate: str,
        variate_attn_gate: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_schedule = block_schedule
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.rope_theta = rope_theta
        self.rope_dim = rope_dim
        self.norm_eps = norm_eps
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.time_attn_gate = time_attn_gate
        self.variate_attn_gate = variate_attn_gate

    def build(self, input_shape):
        self.layers = []
        for i, kind in enumerate(self.block_schedule):
            block = TransformerBlock(
                kind=kind,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                norm_eps=self.norm_eps,
                attn_bias=self.attn_bias,
                mlp_bias=self.mlp_bias,
                time_attn_gate=self.time_attn_gate,
                variate_attn_gate=self.variate_attn_gate,
                name=f"block_{i}",
            )
            self.layers.append(block)

        self.rope = SinCosRoPE(
            dim=self.embed_dim // self.num_heads,
            base=self.rope_theta,
            rotary_dim=self.rope_dim,
            name="rope",
        )
        self.out_norm = RMSNorm(self.embed_dim, self.norm_eps, name="out_norm")
        super().build(input_shape)

    def call(self, x, time_mask=None, variate_mask=None, positions=None):
        """Run stack on x (R, N, D) with integer time positions (R, N)."""
        cos, sin = self.rope(x=x, position_ids=positions)
        for layer in self.layers:
            x = layer(x, time_mask=time_mask, variate_mask=variate_mask, cos=cos, sin=sin)
        return self.out_norm(x=x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "block_schedule": list(self.block_schedule),
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "rope_theta": self.rope_theta,
            "rope_dim": self.rope_dim,
            "norm_eps": self.norm_eps,
            "attn_bias": self.attn_bias,
            "mlp_bias": self.mlp_bias,
            "time_attn_gate": self.time_attn_gate,
            "variate_attn_gate": self.variate_attn_gate,
        })
        return config