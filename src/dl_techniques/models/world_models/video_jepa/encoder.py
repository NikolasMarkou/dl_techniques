"""Hybrid Video-JEPA-Clifford per-frame encoder.

`VideoJEPACliffordEncoder` patchifies a batch of frames, adds a 2D sine
position embedding, and refines the result through a stack of
`CliffordNetBlock` layers. No time dimension is introduced here: callers
reshape `(B, T, H, W, C)` to `(B*T, H, W, C)` before calling, and time is
handled by the predictor instead.

`embed_dim` must be a multiple of 4, not merely even: `PositionEmbeddingSine2D`
receives `num_pos_feats = embed_dim // 2` and splits that value between its
sine and cosine halves, so it must itself be even. `CliffordNetBlock` uses
batch normalization inside its context stream, so a batch size of at least 2
is required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import keras
from keras import ops

from dl_techniques.utils.logger import logger
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.embedding.positional_embedding_sine_2d import (
    PositionEmbeddingSine2D,
)
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.video_jepa.encoder")
class VideoJEPACliffordEncoder(keras.layers.Layer):
    """Hybrid per-frame encoder: patch embed, sine position embed, Clifford blocks.

    Architecture:

    .. code-block:: text

        pixels_flat [B*T, H, W, C]
        │
        ┌─────▼─────┐
        │ PatchEmbedding2D│  patch=P, embed_dim=D
        └─────┬───────┘
              ▼
        tokens [B*T, N, D]  (N = H_p * W_p)
              │ reshape
              ▼
        grid [B*T, H_p, W_p, D]
              ├─────────────────────────────┐
              ▼                              │
        ┌─────────────────────┐              │
        │ PositionEmbeddingSine2D│              │
        └─────┬───────────────┘              │
              ▼                              │
             add ◄───────────────────────────┘
              ▼
        ┌─────────────────────┐
        │ CliffordNetBlock x N  │  channels=D, shifts=shifts
        └─────┬───────────────┘  identity residual added per block
              ▼
        latents [B*T, H_p, W_p, D]

    :param embed_dim: Embedding dimension `D`. Must be a positive multiple
        of 4: `num_pos_feats = D // 2` is handed to
        :class:`PositionEmbeddingSine2D`, and that value must itself be even.
    :type embed_dim: int
    :param patch_size: Non-overlapping patch edge length `P`.
    :type patch_size: int
    :param img_size: Square input edge length `H = W`. Must be divisible by `P`.
    :type img_size: int
    :param img_channels: Number of pixel channels `C`.
    :type img_channels: int
    :param depth: Number of stacked :class:`CliffordNetBlock` layers.
    :type depth: int
    :param shifts: Channel-shift offsets for the encoder Clifford blocks.
    :type shifts: Iterable[int]
    :param dropout_rate: Dropout rate applied to the post-patch-embed features.
    :type dropout_rate: float
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        img_size: int,
        img_channels: int = 3,
        depth: int = 2,
        shifts: Iterable[int] = (1, 2),
        dropout_rate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # DECISION plan-2026-08-28T181715-3870472c/D-007: the constraint is % 4,
        # not % 2 -- embed_dim=10 passed the old % 2 check and built a position
        # encoder that could never complete a forward pass. See decisions.md.
        if embed_dim <= 0 or embed_dim % 4 != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be a positive multiple of 4, "
                f"not merely even: the sine position encoding receives "
                f"num_pos_feats = embed_dim // 2 = {embed_dim // 2}, and that "
                f"value must ITSELF be even because PositionEmbeddingSine2D "
                f"splits it between its sine and cosine halves. Use "
                f"embed_dim = "
                f"{((embed_dim + 3) // 4) * 4 if embed_dim > 0 else 4}."
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
        self.dropout_rate = dropout_rate

        self._patches_per_side = img_size // patch_size
        self._num_patches = self._patches_per_side ** 2

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
            keras.layers.Dropout(dropout_rate, name="drop")
            if dropout_rate > 0.0
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

        :param input_shape: `(B_total, H, W, C)` where `B_total = B*T`.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"VideoJEPACliffordEncoder expects 4D input (B_total, H, W, C); "
                f"got rank {len(input_shape)} shape {input_shape}"
            )

        self.patch_embed.build(input_shape)
        # PositionEmbeddingSine2D just needs a 4D shape here: call() synthesizes
        # positions from H_p, W_p directly, ignoring this shape's content.
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

        :param pixels_flat: `(B_total, H, W, C)`, where `B_total = B * T`
            for training or `B * 1` for streaming.
        :param training: Forwarded to dropout and Clifford batch normalization.
        :return: `(B_total, H_p, W_p, D)` patch-grid latents.
        :rtype: keras.KerasTensor
        """
        tokens = self.patch_embed(pixels_flat, training=training)

        B_total = ops.shape(tokens)[0]
        Hp = self._patches_per_side
        grid = ops.reshape(tokens, (B_total, Hp, Hp, self.embed_dim))

        # PositionEmbeddingSine2D outputs channels-first; transpose to
        # channels-last before adding. Its call() only reads the input shape
        # to derive B/H/W, so any 4D tensor with the right spatial layout works.
        pe_cf = self.pos_embed(grid)
        pe_cl = ops.transpose(pe_cf, (0, 2, 3, 1))
        grid = grid + pe_cl

        if self.dropout is not None:
            grid = self.dropout(grid, training=training)

        # Blocks are transform-only; the identity residual is added here.
        for blk in self.blocks:
            grid = grid + blk(grid, training=training)

        return grid

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return `(B_total, H_p, W_p, D)`."""
        B_total = input_shape[0]
        return (B_total, self._patches_per_side, self._patches_per_side,
                self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
            "img_channels": self.img_channels,
            "depth": self.depth,
            "shifts": self.shifts,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoJEPACliffordEncoder":
        """Rebuild from a config, tolerating a pre-`% 4` stored `embed_dim`.

        A non-conforming `embed_dim` is rounded up with a warning rather than
        raised on, so an archive from before the multiple-of-4 rule still
        loads. Such an encoder's position encoding could never complete a
        forward pass, so no model carrying one was ever trainable.

        :param config: Serialized configuration.
        :return: The reconstructed encoder.
        :rtype: VideoJEPACliffordEncoder
        """
        config = dict(config)
        embed_dim = config.get("embed_dim")
        if isinstance(embed_dim, int) and embed_dim > 0 and embed_dim % 4 != 0:
            substitute = ((embed_dim + 3) // 4) * 4
            logger.warning(
                "VideoJEPACliffordEncoder config carries embed_dim=%d, whose "
                "sine width num_pos_feats=%d is odd; this archive predates the "
                "multiple-of-4 requirement and its position encoder could "
                "never run a forward pass. Substituting embed_dim=%d "
                "(num_pos_feats=%d). The encoder output width changes from %d "
                "to %d, so stored weights for this layer will not match.",
                embed_dim, embed_dim // 2, substitute, substitute // 2,
                embed_dim, substitute,
            )
            config["embed_dim"] = substitute
        return cls(**config)
