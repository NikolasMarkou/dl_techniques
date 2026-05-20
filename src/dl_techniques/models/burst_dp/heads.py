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


# DECISION plan_2026-05-20_b8f8df89/D-001
# ReconstructionHead produces a *signed residual delta*, NOT the final image.
# The caller (BurstDP.call) computes `recon = clip(ref + delta)`. This residual
# parameterization is REQUIRED: without the input skip the recon path has no
# identity gradient route and collapses to a blurry mean (overfit diagnostic:
# gradient global-norm 17.1 -> 0.006 in 50 steps). Two consequences:
#   - DPTDecoder output_activation MUST be `linear` (delta is signed; `sigmoid`
#     would force a non-negative, [0,1]-bounded output — a ghost constraint from
#     the old plain-autoencoder framing).
#   - `residual_proj` is zero-initialized so delta == 0 at init => recon == ref
#     (output starts at the identity-copy PSNR, ~20 dB, instead of garbage).
# See plans/plan_2026-05-20_b8f8df89/decisions.md D-001.
@keras.saving.register_keras_serializable()
class ReconstructionHead(keras.layers.Layer):
    """Per-pixel reconstruction residual (signed RGB delta).

    Produces a signed delta to be added to the reference image by the caller
    (`recon = clip(ref + delta)`), not the reconstructed image itself.
    """

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
            output_activation="linear",
            upsample_factor=_upsample_factor_for_patch(self.patch_size),
            name="recon_dpt",
        )
        # Zero-initialized 1x1 projection: delta == 0 at init (ControlNet /
        # AdaLN-Zero pattern). Gradient revives after the first optimizer step.
        self.residual_proj = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="residual_proj",
        )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        delta = self.decoder(x, training=training)
        return self.residual_proj(delta)

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


