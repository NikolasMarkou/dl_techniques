"""SimMIM-style MAE masking utilities for ConvNeXtPatchVAEV2.

We mask features *after the patchifying stem* (still on the full grid)
rather than dropping patches from the encoder input. Reasons:

- ConvNeXt blocks rely on full ``(Hp, Wp)`` spatial grids for the 7×7
  depthwise convolutions; canonical MAE (visible-only sequence) would
  require swapping ConvNeXt for a transformer.
- ConvNeXt-V2's FCMAE paper uses exactly this recipe — masked patches
  are replaced with a learnable mask token, the encoder runs at full
  resolution, and the recon loss is weighted toward masked patches.

DECISION plan_2026-05-27_4a444b14/D-002: SimMIM-style masking. Mask
generation, token application, and pixel-space mask upsampling live
here so the encoder stays focused on the ConvNeXt stack.
"""

from __future__ import annotations

from typing import Optional, Tuple

import keras
from keras import ops


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------


def generate_patch_mask(
    batch_size: int,
    hp: int,
    wp: int,
    ratio: float,
    seed: Optional[int] = None,
) -> keras.KerasTensor:
    """Generate a random per-patch boolean mask.

    Per-sample independent: each sample in the batch is given an
    independent uniform random rank over its patches, and the top
    ``round(ratio * Hp * Wp)`` ranks become masked. Exactly the same
    number of patches per sample are masked.

    Args:
        batch_size: Batch size. Pass a static int (dynamic batch sizes
            are common in Keras; the caller should resolve via
            ``ops.shape(x)[0]`` first).
        hp: Patches per side along H.
        wp: Patches per side along W.
        ratio: Fraction of patches to mask. ``0 <= ratio <= 1``.
        seed: Optional integer seed for reproducibility.

    Returns:
        Tensor of shape ``(batch_size, hp, wp, 1)`` with values in
        ``{0.0, 1.0}`` (1.0 = masked).
    """
    if ratio <= 0.0:
        return ops.zeros((batch_size, hp, wp, 1), dtype="float32")
    if ratio >= 1.0:
        return ops.ones((batch_size, hp, wp, 1), dtype="float32")

    n = hp * wp
    n_mask = max(1, int(round(ratio * n)))

    # Uniform [0,1) ranks per (sample, patch); sort within each sample;
    # top n_mask indices become masked. Using keras.random.uniform → top_k
    # avoids backend-specific shuffle ops.
    noise = keras.random.uniform((batch_size, n), seed=seed)
    # `ops.argsort` is the canonical way to grab the lowest-N indices in
    # backend-agnostic Keras 3. Construct one-hot mask by comparing
    # against a threshold per sample — cheaper than scatter on TF.
    threshold = ops.take(
        ops.sort(noise, axis=-1),
        indices=n_mask - 1,
        axis=-1,
    )
    threshold = ops.reshape(threshold, (batch_size, 1))
    mask_flat = ops.cast(noise <= threshold, "float32")
    mask = ops.reshape(mask_flat, (batch_size, hp, wp, 1))
    return mask


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------


def apply_mask_with_token(
    features: keras.KerasTensor,
    mask: keras.KerasTensor,
    mask_token: keras.KerasTensor,
) -> keras.KerasTensor:
    """Replace masked positions with a broadcastable learnable mask token.

    Args:
        features: ``(B, Hp, Wp, E)`` post-stem feature map.
        mask: ``(B, Hp, Wp, 1)`` binary mask (1 = masked).
        mask_token: ``(1, 1, 1, E)`` learnable token, broadcast across
            ``(B, Hp, Wp)``.

    Returns:
        ``(B, Hp, Wp, E)`` mixed feature map.
    """
    return features * (1.0 - mask) + mask_token * mask


# ---------------------------------------------------------------------------
# Pixel-space mask
# ---------------------------------------------------------------------------


def upsample_mask_to_pixels(
    mask: keras.KerasTensor,
    patch_size: int,
) -> keras.KerasTensor:
    """Upsample a per-patch mask to pixel space via nearest-neighbor.

    Args:
        mask: ``(B, Hp, Wp, 1)`` per-patch mask.
        patch_size: Integer patch edge length.

    Returns:
        ``(B, H, W, 1)`` pixel-space mask with ``H = Hp * patch_size``.
    """
    # `keras.ops.repeat` on axis 1 then axis 2 is the backend-agnostic
    # equivalent of `tf.image.resize(..., method="nearest")` here.
    mask = ops.repeat(mask, patch_size, axis=1)
    mask = ops.repeat(mask, patch_size, axis=2)
    return mask


# ---------------------------------------------------------------------------
# Convenience type
# ---------------------------------------------------------------------------

MaskedFeatures = Tuple[keras.KerasTensor, Optional[keras.KerasTensor]]
"""``(features_after_mask, mask)`` — mask is ``None`` when ratio == 0."""
