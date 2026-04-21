"""Hybrid Video-JEPA-Clifford encoder (D-001).

Forward path (per-frame):

.. code-block:: text

    pixels_flat : (B*T, H, W, C)
            │
            ▼  PatchEmbedding2D(patch=P, embed_dim=D)
    tokens    : (B*T, N, D)                    [N = H_p * W_p]
            │
            ▼  reshape
    grid      : (B*T, H_p, W_p, D)
            │
            ▼  + PositionEmbeddingSine2D (channels-first → last, broadcast)
    grid_pe   : (B*T, H_p, W_p, D)
            │
            ▼  N_enc × CliffordNetBlock(channels=D, shifts=encoder_shifts)
    latents   : (B*T, H_p, W_p, D)

A time dimension is not introduced here; callers reshape
``(B, T, H, W, C) → (B*T, H, W, C)`` before calling. See
:meth:`VideoJEPA.encode_frames`.

Notes
-----
* ``PositionEmbeddingSine2D`` emits channels-first ``(B, 2*num_pos_feats, H_p,
  W_p)``. We transpose to channels-last and assert the feature dim is ``D``
  (requires ``num_pos_feats = D // 2``, which the constructor enforces).
* ``CliffordNetBlock`` uses ``BatchNormalization`` inside the context stream;
  smoke tests **must** use batch size ``B*T >= 2`` (plan § Hard constraints).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock


@keras.saving.register_keras_serializable()
class VideoJEPACliffordEncoder(keras.layers.Layer):
    """Hybrid per-frame encoder: PatchEmbedding2D → sine2D PE → N × CliffordNetBlock.

    :param embed_dim: Embedding dimension ``D``. Must be even
        (``num_pos_feats = D // 2``).
    :param patch_size: Non-overlapping patch edge length ``P``.
    :param img_size: Square input edge length ``H = W``. Must be divisible
        by ``P``.
    :param img_channels: Number of pixel channels ``C``.
    :param depth: Number of stacked :class:`CliffordNetBlock` layers
        (``encoder_clifford_depth`` in config).
    :param shifts: Channel-shift offsets for the encoder Clifford blocks.
    :param dropout: Dropout rate applied to the post-patch-embed features
        (kept as a layer for potential future use; 0.0 by default).
    :param kwargs: passthrough to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        img_size: int,
        img_channels: int = 3,
        depth: int = 2,
        shifts: Iterable[int] = (1, 2),
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be positive and even (D/2 used by sine2D PE); "
                f"got {embed_dim}"
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size "
                f"({patch_size})."
            )
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.depth = depth
        self.shifts = list(shifts)
        self.dropout_rate = dropout

        self._patches_per_side = img_size // patch_size
        self._num_patches = self._patches_per_side ** 2

        # --- Sub-layers (created in __init__ per modern Keras 3 pattern) ---
        self.patch_embed = PatchEmbedding2D(
            patch_size=patch_size,
            embed_dim=embed_dim,
            name="patch_embed",
        )
        self.pos_embed = PositionEmbeddingSine2D(
            num_pos_feats=embed_dim // 2,
            name="pos_embed",
        )
        self.dropout = (
            keras.layers.Dropout(dropout, name="drop")
            if dropout > 0.0
            else None
        )
        self.blocks: List[CliffordNetBlock] = [
            CliffordNetBlock(
                channels=embed_dim,
                shifts=self.shifts,
                cli_mode="full",
                ctx_mode="diff",
                use_global_context=False,
                name=f"clifford_block_{i}",
            )
            for i in range(depth)
        ]

    # ------------------------------------------------------------------
    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers in dependency order.

        :param input_shape: ``(B_total, H, W, C)`` where ``B_total = B*T``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"VideoJEPACliffordEncoder expects 4D input (B_total, H, W, C); "
                f"got rank {len(input_shape)} shape {input_shape}"
            )

        self.patch_embed.build(input_shape)
        # PositionEmbeddingSine2D expects 4D input and emits channels-first;
        # we build it with a dummy shape matching what it would receive had
        # we called it on the post-patch-embed 4D tensor — but in our call()
        # we synthesise positions from H_p, W_p directly, so PE2D.build just
        # needs a 4D shape.
        pe_input_shape = (
            input_shape[0], self._patches_per_side, self._patches_per_side,
            self.embed_dim,
        )
        self.pos_embed.build(pe_input_shape)

        if self.dropout is not None:
            self.dropout.build(pe_input_shape)

        for blk in self.blocks:
            blk.build(pe_input_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------
    def call(
        self,
        pixels_flat: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode a flat pixel batch to a 4D patch grid.

        :param pixels_flat: ``(B_total, H, W, C)`` where
            ``B_total = B * T`` for training or ``B * 1`` for streaming.
        :param training: Forwarded to dropout and Clifford BN.
        :return: ``(B_total, H_p, W_p, D)`` patch-grid latents.
        """
        # 1. Patchify: (B_total, H, W, C) → (B_total, N, D)
        tokens = self.patch_embed(pixels_flat, training=training)

        # 2. Reshape to 4D grid: (B_total, N, D) → (B_total, H_p, W_p, D)
        B_total = ops.shape(tokens)[0]
        Hp = self._patches_per_side
        grid = ops.reshape(tokens, (B_total, Hp, Hp, self.embed_dim))

        # 3. Positional embedding. PositionEmbeddingSine2D outputs channels-
        #    first (B, 2*num_pos_feats, H_p, W_p). We transpose to channels-
        #    last and add. Note: PE2D.call only uses the input shape to
        #    derive B/H/W — the content is ignored, so any 4D tensor with
        #    the right spatial layout will do.
        pe_cf = self.pos_embed(grid)  # (B_total, D, H_p, W_p)
        pe_cl = ops.transpose(pe_cf, (0, 2, 3, 1))  # → (B_total, H_p, W_p, D)
        grid = grid + pe_cl

        if self.dropout is not None:
            grid = self.dropout(grid, training=training)

        # 4. Stacked CliffordNetBlocks (residual, shape-preserving).
        for blk in self.blocks:
            grid = blk(grid, training=training)

        return grid

    # ------------------------------------------------------------------
    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """``(B_total, H_p, W_p, D)``."""
        B_total = input_shape[0]
        return (B_total, self._patches_per_side, self._patches_per_side,
                self.embed_dim)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
            "img_channels": self.img_channels,
            "depth": self.depth,
            "shifts": self.shifts,
            "dropout": self.dropout_rate,
        })
        return config
