"""Video-JEPA-Clifford predictor (D-002, D-004).

Factorized spatial + causal-temporal stack with AdaLN-zero telemetry
conditioning.

Architecture
------------

Input:
    z  : (B, T, H_p, W_p, D)  — per-frame patch latents
    c  : (B, T, cond_dim)     — per-frame telemetry conditioning (cond_dim == D)

Per predictor "pair", we alternate:

1. **Spatial pass** — (B, T, H_p, W_p, D) → reshape (B*T, H_p, W_p, D) →
   ``CliffordNetBlock`` → reshape back. Acts independently per frame;
   strictly causal (no cross-frame info).

2. **Temporal pass** — transpose (B, T, H_p, W_p, D) → (B, H_p, W_p, T, D) →
   reshape (B*H_p*W_p, T, D) → ``AdaLNZeroConditionalBlock([x, c_tile])``
   with causal self-attention, tiling c: (B, T, D) → (B*H_p*W_p, T, D)
   → reshape (B*H_p*W_p, 1, T, D) → ``CausalCliffordNetBlock`` → reshape
   (B*H_p*W_p, T, D) → reshape/transpose back to (B, T, H_p, W_p, D).

A learned 1D temporal positional embedding ``pos_t: (1, T_max, 1, 1, D)``
is added to ``z`` **once** before block 0.

Causality invariant
-------------------
A perturbation at frame ``k`` must not alter any output at frame ``< k``.

- Spatial pass: trivially independent across ``T``.
- Temporal AdaLN block: ``MultiHeadAttention(use_causal_mask=True)``.
- CausalCliffordNetBlock: left-padded valid convs over the W (=T) axis.
- Temporal PE: applied once, additive — causal-safe by construction.

See ``tests/test_models/test_video_jepa/test_video_jepa.py::TestPredictor::
test_causality``.

Identity-at-init invariant
--------------------------
At init, every ``AdaLNZeroConditionalBlock`` is identity in x (zero-initialized
modulation MLP ⇒ gate=0). Each ``CausalCliffordNetBlock`` is near-identity
via LayerScale γ=1e-5. Each spatial ``CliffordNetBlock`` is near-identity
via LayerScale γ=1e-5. So at init the predictor is ``z + pos_t + eps`` ,
and ``predictor(z, c) - predictor(z, c') ≈ 0`` independent of ``c``
because only the AdaLN blocks read ``c`` and their output is zero-gated.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.adaln_zero import AdaLNZeroConditionalBlock
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CliffordNetBlock,
)


@keras.saving.register_keras_serializable()
class VideoJEPAPredictor(keras.layers.Layer):
    """Factorized spatial + causal-temporal Clifford predictor.

    :param embed_dim: Latent dimension ``D`` (must equal encoder ``embed_dim``).
    :param num_frames_max: Maximum window length ``T_max`` for the learned
        1D temporal PE (usually ``num_frames`` from config).
    :param patches_per_side: ``H_p = W_p`` — used to build static shapes.
    :param depth: Number of (spatial, temporal) pairs.
    :param num_heads: Heads for the AdaLN block's MHA.
    :param dim_head: Per-head dimension for the AdaLN block's MHA.
    :param mlp_dim: MLP hidden dim inside the AdaLN block.
    :param shifts: Channel-shift offsets for predictor Clifford blocks.
    :param dropout: Dropout rate inside the AdaLN block.
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
        self.adaln_blocks: List[AdaLNZeroConditionalBlock] = [
            AdaLNZeroConditionalBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_causal_mask=True,
                name=f"adaln_block_{i}",
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

        :param input_shape: list/tuple ``[z_shape, c_shape]`` where
            ``z_shape = (B, T, H_p, W_p, D)`` and ``c_shape = (B, T, D)``.
        """
        if (
            not isinstance(input_shape, (list, tuple))
            or len(input_shape) != 2
            or not all(isinstance(s, (list, tuple)) for s in input_shape)
        ):
            raise ValueError(
                "VideoJEPAPredictor expects input_shape = [z_shape, c_shape]. "
                f"Got: {input_shape}"
            )
        z_shape, c_shape = input_shape
        if len(z_shape) != 5 or len(c_shape) != 3:
            raise ValueError(
                f"z_shape must be 5D (B,T,H_p,W_p,D), c_shape must be 3D "
                f"(B,T,D). Got z_shape={z_shape}, c_shape={c_shape}"
            )

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

        # AdaLN blocks consume [x:(B*H_p*W_p, T, D), c_tile:(B*H_p*W_p, T, D)].
        adaln_x_shape = (None, z_shape[1], self.embed_dim)
        adaln_c_shape = (None, z_shape[1], self.embed_dim)
        for blk in self.adaln_blocks:
            blk.build([adaln_x_shape, adaln_c_shape])

        # Causal Clifford blocks consume (B*H_p*W_p, 1, T, D).
        causal_in = (None, 1, z_shape[1], self.embed_dim)
        for blk in self.causal_blocks:
            blk.build(causal_in)

        super().build(input_shape)

    # ------------------------------------------------------------------
    def call(
        self,
        inputs,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: list/tuple ``[z, c]`` where
            ``z : (B, T, H_p, W_p, D)`` and ``c : (B, T, D)``.
        :param training: Forwarded to all sub-layers.
        :return: ``(B, T, H_p, W_p, D)`` predicted next-frame patch latents.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "VideoJEPAPredictor expects inputs = [z, c] (list/tuple of "
                f"length 2). Got: {type(inputs)}"
            )
        z, c = inputs

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

            # Tile c: (B, T, D) → (B, 1, T, D) → (B, N, T, D) → (B*N, T, D)
            c_e = ops.expand_dims(c, axis=1)                  # (B, 1, T, D)
            c_t = ops.broadcast_to(c_e, (B, N, T, D))
            c_t = ops.reshape(c_t, (B * N, T, D))

            # AdaLN-zero block (identity at init, causal self-attn).
            z_t = self.adaln_blocks[i]([z_t, c_t], training=training)

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
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 2
            and all(isinstance(s, (list, tuple)) for s in input_shape)
        ):
            z_shape, _ = input_shape
            return tuple(z_shape)
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
