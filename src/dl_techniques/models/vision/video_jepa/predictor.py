"""Video-JEPA-Clifford predictor: factorized spatial and causal-temporal stack.

`VideoJEPAPredictor` takes per-frame patch latents `z` of shape `(B, T, H_p,
W_p, D)` and predicts the next-frame latents. Each of its `depth` pairs
alternates a spatial pass (`CliffordNetBlock` applied independently per
frame) with a causal temporal pass (causal self-attention plus MLP, then
`CausalCliffordNetBlock`, both over the `T` axis). A learned 1D temporal
position embedding is added to `z` once before the first pair. The model
takes pixels only: no telemetry conditioning.

A perturbation at frame `k` never alters any output at frame `< k`: the
spatial pass is independent per frame, the temporal attention uses a causal
mask, and `CausalCliffordNetBlock` uses only left-context over `T`.

At initialization every residual branch (attention, MLP, causal Clifford,
spatial Clifford) is scaled by a LayerScale gamma of `1e-5`, so the predictor
starts as `z + pos_t` plus a near-zero correction.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CliffordNetBlock,
)
from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.utils.keras_registration import register_dl_technique

#: Variance epsilon for every ``LayerNormalization`` authored in this module.
#:
#: Matches the ``CliffordNetBlock`` / ``CausalCliffordNetBlock`` this predictor
#: is assembled from, which build ``LayerNormalization(epsilon=1e-6)`` directly
#: (``layers/geometric/clifford_block.py:1443``) to agree with
#: ``layers/norms/factory.py``'s ``setdefault('epsilon', 1e-6)``. Keras'
#: own default is 1e-3, i.e. 1000x this, which is what these two norms silently
#: used before. Import this name at any new construction site here rather than
#: writing the literal a third time.
_NORM_EPSILON: float = 1e-6


@register_dl_technique("dl_techniques.models.video_jepa.predictor")
class CausalSelfAttnMLPBlock(keras.layers.Layer):
    """Causal self-attention and MLP block, pre-norm, LayerScale-identity init.

    Architecture:

    .. code-block:: text

        x [B, T, D]
        ├─────────────────────────────┐ (residual)
        ▼                              │
        ┌─────────────┐                │
        │ layernorm     │                │
        └─────┬───────┘                │
              ▼                        │
        ┌─────────────────────┐        │
        │ causal self-attention │        │
        └─────┬───────────────┘        │
              ▼                        │
        gamma_a (LayerScale)            │
              ▼                        │
             add ◄──────────────────────┘
              │
              ├─────────────────────────────┐ (residual)
              ▼                              │
        ┌─────────────┐                      │
        │ layernorm     │                      │
        └─────┬───────┘                      │
              ▼                              │
        ┌─────────────┐                      │
        │ MLP           │                      │
        └─────┬───────┘                      │
              ▼                              │
        gamma_m (LayerScale)                  │
              ▼                              │
             add ◄──────────────────────────┘
              ▼
        out [B, T, D]

    `gamma_a` and `gamma_m` are per-channel learnable scales initialized to
    `layer_scale_init`, so the block is near-identity at initialization.

    :param dim: Channel dimension `D`.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dim_head: Per-head dimension, the `key_dim` of the attention layer.
    :type dim_head: int
    :param mlp_dim: Hidden dimension of the MLP.
    :type mlp_dim: int
    :param dropout_rate: Dropout rate inside both attention and MLP.
    :type dropout_rate: float
    :param layer_scale_init: Initial value of the LayerScale gamma.
    :type layer_scale_init: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 128,
        dropout_rate: float = 0.0,
        layer_scale_init: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim_head <= 0:
            raise ValueError(f"dim_head must be positive, got {dim_head}")
        if mlp_dim <= 0:
            raise ValueError(f"mlp_dim must be positive, got {mlp_dim}")

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.layer_scale_init = layer_scale_init

        # DECISION plan-2026-08-17T183311-79c63e38/D-028: epsilon is explicit on
        # both LayerNorms, matching the 1e-6 the spatial/causal CliffordNetBlocks
        # use, not Keras' 1e-3 default -- an epsilon mismatch is invisible to shape/dtype/finiteness tests. See decisions.md.
        self.ln1 = keras.layers.LayerNormalization(
            epsilon=_NORM_EPSILON, name="ln_attn"
        )
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_head,
            dropout=dropout_rate,
            name="mha",
        )
        self.ln2 = keras.layers.LayerNormalization(
            epsilon=_NORM_EPSILON, name="ln_mlp"
        )
        self.mlp_hidden = keras.layers.Dense(mlp_dim, activation="gelu",
                                             name="mlp_hidden")
        # DECISION plan_2026-05-24_ca745a6c/D-005: skip instantiating Dropout at
        # rate=0.0 -- Dropout.call(training=<symbolic tensor>) raises under
        # @tf.function even at rate 0.0. See decisions.md.
        self.mlp_drop = (
            keras.layers.Dropout(dropout_rate, name="mlp_dropout")
            if dropout_rate > 0.0
            else None
        )
        self.mlp_out = keras.layers.Dense(dim, name="mlp_out")

        # Per-channel LayerScale gamma; initializer= and constraint=None are
        # both required here, per the D-005 anchor in layers/geometric/clifford_block.py.
        self.gamma_a = LayerScale(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.layer_scale_init),
            constraint=None,
            name="gamma_attn",
        )
        self.gamma_m = LayerScale(
            multiplier_type="CHANNEL",
            initializer=keras.initializers.Constant(self.layer_scale_init),
            constraint=None,
            name="gamma_mlp",
        )

    def build(self, input_shape: Any) -> None:
        """Build sub-layers with shape ``(B, T, D)``."""
        if len(input_shape) != 3:
            raise ValueError(
                f"CausalSelfAttnMLPBlock expects 3D input (B, T, D), got "
                f"shape {input_shape}."
            )
        self.ln1.build(input_shape)
        # MHA builds on (query_shape, value_shape).
        self.attn.build(input_shape, input_shape)
        self.ln2.build(input_shape)
        self.mlp_hidden.build(input_shape)
        mlp_hidden_shape = tuple(input_shape[:-1]) + (self.mlp_dim,)
        if self.mlp_drop is not None:
            self.mlp_drop.build(mlp_hidden_shape)
        self.mlp_out.build(mlp_hidden_shape)

        self.gamma_a.build(input_shape)
        self.gamma_m.build(input_shape)
        super().build(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the causal-attention and MLP residual branches.

        :param x: Input tensor, shape `(B, T, D)`.
        :param training: Forwarded to attention and dropout.
        :return: Output tensor, shape `(B, T, D)`.
        :rtype: keras.KerasTensor
        """
        h = self.ln1(x)
        a = self.attn(h, h, use_causal_mask=True, training=training)
        x = x + self.gamma_a(a)

        h = self.ln2(x)
        h = self.mlp_hidden(h)
        if self.mlp_drop is not None:
            h = self.mlp_drop(h, training=training)
        h = self.mlp_out(h)
        x = x + self.gamma_m(h)
        return x

    def compute_output_shape(
        self, input_shape: Any,
    ) -> Tuple[Optional[int], ...]:
        """Return the input shape unchanged."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "layer_scale_init": self.layer_scale_init,
        })
        return config


@register_dl_technique("dl_techniques.models.video_jepa.predictor")
class VideoJEPAPredictor(keras.layers.Layer):
    """Factorized spatial and causal-temporal Clifford predictor, pixels-only.

    Architecture, one of `depth` pairs:

    .. code-block:: text

        z [B, T, H_p, W_p, D]
        │ reshape
        ▼
        z_s [B*T, H_p, W_p, D]
        │
        ┌─────▼─────┐
        │ CliffordNetBlock│  spatial, per-frame
        └─────┬─────┘
              ▼  + residual, reshape back
        z [B, T, H_p, W_p, D]
        │ transpose + reshape
        ▼
        z_t [B*H_p*W_p, T, D]
        │
        ┌─────▼─────┐
        │ CausalSelfAttnMLPBlock│
        └─────┬─────┘
              ▼
        ┌─────────────────────┐
        │ CausalCliffordNetBlock│  left-context only over T
        └─────┬───────────────┘
              ▼  + residual
        z [B, T, H_p, W_p, D]  transpose + reshape back

    :param embed_dim: Latent dimension `D`, must equal the encoder's `embed_dim`.
    :type embed_dim: int
    :param num_frames_max: Maximum window length `T_max` for the learned 1D
        temporal position embedding.
    :type num_frames_max: int
    :param patches_per_side: `H_p = W_p`, used to build static shapes.
    :type patches_per_side: int
    :param depth: Number of spatial/temporal pairs.
    :type depth: int
    :param num_heads: Heads for the temporal self-attention.
    :type num_heads: int
    :param dim_head: Per-head dimension for the temporal attention.
    :type dim_head: int
    :param mlp_dim: MLP hidden dimension inside the temporal block.
    :type mlp_dim: int
    :param shifts: Channel-shift offsets for predictor Clifford blocks.
    :type shifts: Iterable[int]
    :param dropout_rate: Dropout rate inside the temporal block.
    :type dropout_rate: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frames_max: int,
        patches_per_side: int,
        depth: int = 2,
        num_heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 128,
        shifts: Iterable[int] = (1, 2),
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_frames_max <= 0:
            raise ValueError(
                f"num_frames_max must be positive, got {num_frames_max}"
            )
        if patches_per_side <= 0:
            raise ValueError(
                f"patches_per_side must be positive, got {patches_per_side}"
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.embed_dim = embed_dim
        self.num_frames_max = num_frames_max
        self.patches_per_side = patches_per_side
        self.depth = depth
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.shifts = list(shifts)
        self.dropout_rate = dropout_rate

        # Per-pair sub-layers.
        self.spatial_blocks: List[CliffordNetBlock] = [
            CliffordNetBlock(
                channels=embed_dim,
                shifts=self.shifts,
                cli_mode="full",
                ctx_mode="diff",
                use_global_context=False,
                name=f"spatial_block_{i}",
            )
            for i in range(depth)
        ]
        self.attn_blocks: List[CausalSelfAttnMLPBlock] = [
            CausalSelfAttnMLPBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                name=f"attn_block_{i}",
            )
            for i in range(depth)
        ]
        self.causal_blocks: List[CausalCliffordNetBlock] = [
            CausalCliffordNetBlock(
                channels=embed_dim,
                shifts=self.shifts,
                cli_mode="full",
                ctx_mode="diff",
                use_global_context=False,
                name=f"causal_temp_block_{i}",
            )
            for i in range(depth)
        ]

    # ------------------------------------------------------------------
    def build(self, input_shape: Any) -> None:
        """Build sub-layers.

        :param input_shape: `z_shape = (B, T, H_p, W_p, D)`.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
            raise ValueError(
                "VideoJEPAPredictor expects input_shape = "
                "(B, T, H_p, W_p, D) (5D). "
                f"Got: {input_shape}"
            )
        z_shape = input_shape

        self.pos_t = self.add_weight(
            name="pos_t",
            shape=(1, self.num_frames_max, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        spatial_in = (None, self.patches_per_side, self.patches_per_side,
                      self.embed_dim)
        for blk in self.spatial_blocks:
            blk.build(spatial_in)

        attn_in = (None, z_shape[1], self.embed_dim)
        for blk in self.attn_blocks:
            blk.build(attn_in)

        # CausalCliffordNetBlock runs in sequence mode and consumes this
        # shape natively.
        causal_in = (None, z_shape[1], self.embed_dim)
        for blk in self.causal_blocks:
            blk.build(causal_in)

        super().build(input_shape)

    def call(
        self,
        z: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the alternating spatial and causal-temporal pairs.

        :param z: `(B, T, H_p, W_p, D)` per-frame patch latents.
        :param training: Forwarded to all sub-layers.
        :return: `(B, T, H_p, W_p, D)` predicted next-frame patch latents.
        :rtype: keras.KerasTensor
        :raises ValueError: If `z` is a list or tuple; the predictor takes a
            single tensor.
        """
        if isinstance(z, (list, tuple)):
            raise ValueError(
                "VideoJEPAPredictor expects a single tensor z "
                "(B, T, H_p, W_p, D). Got a list/tuple input."
            )

        shape = ops.shape(z)
        B, T, Hp, Wp, D = shape[0], shape[1], shape[2], shape[3], shape[4]
        N = Hp * Wp

        # Temporal PE is added once, before the first pair, and broadcasts
        # over the spatial axes.
        pos_t = self.pos_t[:, :T, :]
        pos_t = ops.reshape(pos_t, (1, T, 1, 1, D))
        z = z + pos_t

        for i in range(self.depth):
            z_s = ops.reshape(z, (B * T, Hp, Wp, D))
            z_s = z_s + self.spatial_blocks[i](z_s, training=training)
            z = ops.reshape(z_s, (B, T, Hp, Wp, D))

            z_t = ops.transpose(z, (0, 2, 3, 1, 4))
            z_t = ops.reshape(z_t, (B * N, T, D))

            z_t = self.attn_blocks[i](z_t, training=training)
            z_t = z_t + self.causal_blocks[i](z_t, training=training)

            z_t = ops.reshape(z_t, (B, Hp, Wp, T, D))
            z = ops.transpose(z_t, (0, 3, 1, 2, 4))

        return z

    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Return `z_shape` unchanged."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_frames_max": self.num_frames_max,
            "patches_per_side": self.patches_per_side,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "mlp_dim": self.mlp_dim,
            "shifts": self.shifts,
            "dropout_rate": self.dropout_rate,
        })
        return config
