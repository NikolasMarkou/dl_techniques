"""Video-JEPA-Clifford predictor (D-002, D-013).

Factorized spatial + causal-temporal stack. Pixels-only — no telemetry
conditioning (D-013, iter-3).

Architecture
------------

Input:
    z  : (B, T, H_p, W_p, D)  — per-frame patch latents

Per predictor "pair", we alternate:

1. **Spatial pass** — (B, T, H_p, W_p, D) → reshape (B*T, H_p, W_p, D) →
   ``CliffordNetBlock`` → reshape back. Acts independently per frame;
   strictly causal (no cross-frame info).

2. **Temporal pass** — transpose (B, T, H_p, W_p, D) → (B, H_p, W_p, T, D) →
   reshape (B*H_p*W_p, T, D) → ``CausalSelfAttnMLPBlock`` (causal MHA + MLP
   wrapped in LayerScale γ=1e-5 residual for identity-at-init)
   → reshape (B*H_p*W_p, 1, T, D) → ``CausalCliffordNetBlock``
   → reshape (B*H_p*W_p, T, D) → reshape/transpose back to
   (B, T, H_p, W_p, D).

A learned 1D temporal positional embedding ``pos_t: (1, T_max, D)``
is added to ``z`` **once** before block 0.

Causality invariant
-------------------
A perturbation at frame ``k`` must not alter any output at frame ``< k``.

- Spatial pass: trivially independent across ``T``.
- Temporal attention: ``MultiHeadAttention(use_causal_mask=True)``.
- CausalCliffordNetBlock: left-padded valid convs over the W (=T) axis.
- Temporal PE: applied once, additive — causal-safe by construction.

See ``tests/test_models/test_video_jepa/test_video_jepa.py::TestPredictor::
test_causality``.

Identity-at-init invariant
--------------------------
At init, every ``CausalSelfAttnMLPBlock`` residual path is scaled by
LayerScale γ=1e-5 on both the attention and MLP branches, making the
block near-identity. ``CausalCliffordNetBlock`` is near-identity via
its own LayerScale γ=1e-5, and each spatial ``CliffordNetBlock`` is
similarly near-identity. So at init the predictor is ``z + pos_t + eps``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CliffordNetBlock,
)


@keras.saving.register_keras_serializable()
class CausalSelfAttnMLPBlock(keras.layers.Layer):
    """Plain causal self-attention + MLP block with LayerScale-identity init.

    Structure (pre-norm + residual):
        y = x + gamma_a * Attn(LN(x))
        out = y + gamma_m * MLP(LN(y))

    where ``gamma_a``, ``gamma_m`` are per-channel learnable scales
    initialized to ``1e-5`` so the block is near-identity at init.

    :param dim: Channel dimension ``D``.
    :param num_heads: Number of attention heads.
    :param dim_head: Per-head dimension (``key_dim`` of MHA).
    :param mlp_dim: Hidden dimension of the MLP.
    :param dropout: Dropout rate inside both attention and MLP.
    :param layer_scale_init: Initial value of the LayerScale γ (default 1e-5).
    :param kwargs: passthrough.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 128,
        dropout: float = 0.0,
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
        self.dropout_rate = dropout
        self.layer_scale_init = layer_scale_init

        # Sub-layers.
        self.ln1 = keras.layers.LayerNormalization(name="ln_attn")
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_head,
            dropout=dropout,
            name="mha",
        )
        self.ln2 = keras.layers.LayerNormalization(name="ln_mlp")
        self.mlp_hidden = keras.layers.Dense(mlp_dim, activation="gelu",
                                             name="mlp_hidden")
        self.mlp_drop = keras.layers.Dropout(dropout, name="mlp_dropout")
        self.mlp_out = keras.layers.Dense(dim, name="mlp_out")

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
        self.mlp_drop.build(mlp_hidden_shape)
        self.mlp_out.build(mlp_hidden_shape)

        # LayerScale γ vectors: per-channel, initialized to layer_scale_init.
        self.gamma_a = self.add_weight(
            name="gamma_attn",
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        self.gamma_m = self.add_weight(
            name="gamma_mlp",
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self, x: keras.KerasTensor, training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # --- Attention branch (causal) ---
        h = self.ln1(x)
        a = self.attn(h, h, use_causal_mask=True, training=training)
        x = x + a * self.gamma_a

        # --- MLP branch ---
        h = self.ln2(x)
        h = self.mlp_hidden(h)
        h = self.mlp_drop(h, training=training)
        h = self.mlp_out(h)
        x = x + h * self.gamma_m
        return x

    def compute_output_shape(
        self, input_shape: Any,
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate,
            "layer_scale_init": self.layer_scale_init,
        })
        return config


@keras.saving.register_keras_serializable()
class VideoJEPAPredictor(keras.layers.Layer):
    """Factorized spatial + causal-temporal Clifford predictor (pixels-only).

    :param embed_dim: Latent dimension ``D`` (must equal encoder ``embed_dim``).
    :param num_frames_max: Maximum window length ``T_max`` for the learned
        1D temporal PE (usually ``num_frames`` from config).
    :param patches_per_side: ``H_p = W_p`` — used to build static shapes.
    :param depth: Number of (spatial, temporal) pairs.
    :param num_heads: Heads for the temporal self-attention.
    :param dim_head: Per-head dimension for the temporal MHA.
    :param mlp_dim: MLP hidden dim inside the temporal block.
    :param shifts: Channel-shift offsets for predictor Clifford blocks.
    :param dropout: Dropout rate inside the temporal block.
    :param kwargs: passthrough.
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
        dropout: float = 0.0,
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
        self.dropout_rate = dropout

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
                dropout=dropout,
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

        :param input_shape: ``z_shape = (B, T, H_p, W_p, D)``.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 5:
            raise ValueError(
                "VideoJEPAPredictor expects input_shape = "
                "(B, T, H_p, W_p, D) (5D). "
                f"Got: {input_shape}"
            )
        z_shape = input_shape

        # Learned 1D temporal PE: (1, T_max, D) — expanded on add.
        self.pos_t = self.add_weight(
            name="pos_t",
            shape=(1, self.num_frames_max, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        # Spatial blocks consume (B*T, H_p, W_p, D).
        spatial_in = (None, self.patches_per_side, self.patches_per_side,
                      self.embed_dim)
        for blk in self.spatial_blocks:
            blk.build(spatial_in)

        # Attention blocks consume (B*H_p*W_p, T, D).
        attn_in = (None, z_shape[1], self.embed_dim)
        for blk in self.attn_blocks:
            blk.build(attn_in)

        # Causal Clifford blocks consume (B*H_p*W_p, 1, T, D).
        causal_in = (None, 1, z_shape[1], self.embed_dim)
        for blk in self.causal_blocks:
            blk.build(causal_in)

        super().build(input_shape)

    # ------------------------------------------------------------------
    def call(
        self,
        z: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param z: ``(B, T, H_p, W_p, D)`` per-frame patch latents.
        :param training: Forwarded to all sub-layers.
        :return: ``(B, T, H_p, W_p, D)`` predicted next-frame patch latents.
        """
        # Backward-compat guard: if a caller passes ``[z]`` or ``[z, c]`` we
        # raise a clear error rather than silently unpack.
        if isinstance(z, (list, tuple)):
            raise ValueError(
                "VideoJEPAPredictor now expects a single tensor z "
                "(B, T, H_p, W_p, D); telemetry conditioning was removed "
                "in iter-3 (D-013). Got a list/tuple input."
            )

        # --- Shapes ---
        shape = ops.shape(z)
        B, T, Hp, Wp, D = shape[0], shape[1], shape[2], shape[3], shape[4]
        N = Hp * Wp  # patches per frame

        # --- Temporal PE (add once, broadcast over spatial) ---
        # pos_t : (1, T_max, D) → slice to (1, T, D) → (1, T, 1, 1, D)
        pos_t = self.pos_t[:, :T, :]                          # (1, T, D)
        pos_t = ops.reshape(pos_t, (1, T, 1, 1, D))
        z = z + pos_t

        # --- Alternating pairs ---
        for i in range(self.depth):
            # ============ Spatial pass ============
            # (B, T, H_p, W_p, D) → (B*T, H_p, W_p, D)
            z_s = ops.reshape(z, (B * T, Hp, Wp, D))
            z_s = self.spatial_blocks[i](z_s, training=training)
            z = ops.reshape(z_s, (B, T, Hp, Wp, D))

            # ============ Temporal pass ============
            # (B, T, H_p, W_p, D) → transpose → (B, H_p, W_p, T, D)
            z_t = ops.transpose(z, (0, 2, 3, 1, 4))
            # → reshape (B*H_p*W_p, T, D)
            z_t = ops.reshape(z_t, (B * N, T, D))

            # Causal self-attention + MLP block (identity at init via
            # LayerScale γ=1e-5).
            z_t = self.attn_blocks[i](z_t, training=training)

            # → reshape (B*N, 1, T, D) for CausalCliffordNetBlock.
            z_t = ops.reshape(z_t, (B * N, 1, T, D))
            z_t = self.causal_blocks[i](z_t, training=training)

            # → reshape back (B*N, T, D) → (B, H_p, W_p, T, D) →
            #   transpose to (B, T, H_p, W_p, D).
            z_t = ops.reshape(z_t, (B, Hp, Wp, T, D))
            z = ops.transpose(z_t, (0, 3, 1, 2, 4))

        return z

    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Any
    ) -> Tuple[Optional[int], ...]:
        """Output matches ``z_shape``."""
        return tuple(input_shape)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
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
            "dropout": self.dropout_rate,
        })
        return config
