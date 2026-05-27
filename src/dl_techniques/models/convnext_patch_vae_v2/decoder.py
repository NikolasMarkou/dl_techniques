"""Decoder for :class:`ConvNeXtPatchVAEV2`.

V2 reuses V1's decoder as-is (`ConvNeXtPatchDecoder`). The decoder is
purely ``z (B, Hp, Wp, latent_dim) -> x_hat (B, H, W, C)`` and has no
multi-task concerns — its API matches V2's needs exactly.

This thin shim re-exports it under a V2 name for symmetry with the
``encoder.py`` / ``mae_mask.py`` / ``heads.py`` siblings.

DECISION plan_2026-05-27_4a444b14/D-004: V2 keeps V1 untouched. The
decoder is identical between V1 and V2; subclassing or copying would
duplicate code without payoff.
"""

from dl_techniques.models.convnext_patch_vae.decoder import (
    ConvNeXtPatchDecoder as ConvNeXtPatchDecoderV2,
)

__all__ = ["ConvNeXtPatchDecoderV2"]
