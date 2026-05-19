"""Task heads for BurstDP.

Two light wrappers around :class:`DPTDecoder` from the existing
``depth_anything`` module. Each takes a 2D feature map of fused reference
tokens and produces a dense per-pixel output at the original image
resolution (DPT decoder handles the bilinear upsample).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import keras
from keras import layers

from dl_techniques.models.depth_anything.components import DPTDecoder


def _upsample_factor_for_patch(patch_size: int) -> int:
    """Patch_size is required to be a power of 2; upsample by that factor."""
    if patch_size <= 0 or (patch_size & (patch_size - 1)) != 0:
        raise ValueError(f"patch_size must be a positive power of 2, got {patch_size}")
    return patch_size


@keras.saving.register_keras_serializable()
class ReconstructionHead(keras.layers.Layer):
    """Per-pixel reconstruction (RGB)."""

    def __init__(
        self,
        decoder_dims: Tuple[int, ...] = (256, 128, 64, 32),
        patch_size: int = 16,
        out_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decoder_dims = tuple(decoder_dims)
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.decoder = DPTDecoder(
            dims=list(self.decoder_dims),
            output_channels=self.out_channels,
            output_activation="sigmoid",
            upsample_factor=_upsample_factor_for_patch(self.patch_size),
            name="recon_dpt",
        )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        return self.decoder(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "decoder_dims": list(self.decoder_dims),
                "patch_size": self.patch_size,
                "out_channels": self.out_channels,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class SegmentationHead(keras.layers.Layer):
    """Per-pixel class logits."""

    def __init__(
        self,
        decoder_dims: Tuple[int, ...] = (256, 128, 64, 32),
        patch_size: int = 16,
        num_classes: int = 81,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decoder_dims = tuple(decoder_dims)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.decoder = DPTDecoder(
            dims=list(self.decoder_dims),
            output_channels=self.num_classes,
            output_activation="linear",
            upsample_factor=_upsample_factor_for_patch(self.patch_size),
            name="seg_dpt",
        )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        return self.decoder(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "decoder_dims": list(self.decoder_dims),
                "patch_size": self.patch_size,
                "num_classes": self.num_classes,
            }
        )
        return config


