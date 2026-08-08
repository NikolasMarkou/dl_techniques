"""
Input transforms that reference SAM applies **before** the model.

`SAM.preprocess` normalizes and pads an image up to the encoder's `img_size`;
it cannot shrink one. Reference SAM never asks it to: the official pipeline runs
a `ResizeLongestSide` transform on the raw image first, so what reaches the
model is always at most `img_size` on its longest side and the pad is
non-negative by construction. No equivalent transform existed anywhere in this
repository (a repo-wide grep for `ResizeLongest|longest` returned zero hits),
which is why `SAM.preprocess`'s oversize `ValueError` would otherwise be a dead
end rather than a remedy.

This module is `keras.ops`-only (no raw TensorFlow ops), so it stays inside the
package's backend-purity invariant (I-1).
"""

import keras
from keras import ops

__all__ = [
    "resize_longest_side",
]


def resize_longest_side(
    image: keras.KerasTensor,
    target_length: int,
    interpolation: str = "bilinear",
) -> keras.KerasTensor:
    """
    Resize an image so that its LONGEST side equals ``target_length``.

    The aspect ratio is preserved. This is the transform reference SAM applies
    before the model (its ``ResizeLongestSide``), and it is the remedy named by
    the ``ValueError`` that :meth:`SAM.preprocess` raises on an oversize input.

    The transform is symmetric in the two spatial axes --- a landscape image and
    its transpose produce transposed results --- so whichever axis is longer is
    the one pinned to ``target_length``.

    Args:
        image: Image tensor of shape ``(H, W, C)`` or ``(B, H, W, C)``, channels
            last. Its spatial extents must be statically known.
        target_length: Desired length of the longest side, in pixels. Must be
            positive.
        interpolation: Interpolation mode forwarded to
            ``keras.ops.image.resize``. Defaults to ``"bilinear"``, matching
            reference SAM's PIL ``BILINEAR`` resize.

    Returns:
        The resized image, same rank as the input, with
        ``max(new_h, new_w) == target_length``. A square image already at
        ``target_length`` keeps its shape (the resize is a no-op).

    Raises:
        ValueError: If ``image`` is not rank 3 or 4, if either spatial extent is
            statically unknown or non-positive, or if ``target_length`` is not
            positive.

    Example:
        A ``(300, 900, 3)`` image at ``target_length=1024`` becomes
        ``(341, 1024, 3)``; its transpose ``(900, 300, 3)`` becomes
        ``(1024, 341, 3)``.
    """
    if target_length <= 0:
        raise ValueError(
            f"target_length must be positive; got {target_length}"
        )

    shape = tuple(image.shape)
    if len(shape) not in (3, 4):
        raise ValueError(
            f"resize_longest_side expects a channels-last image of rank 3 "
            f"(H, W, C) or rank 4 (B, H, W, C); got shape {shape}"
        )

    h_axis = 0 if len(shape) == 3 else 1
    old_h, old_w = shape[h_axis], shape[h_axis + 1]
    if old_h is None or old_w is None:
        raise ValueError(
            f"resize_longest_side needs statically-known spatial extents; got "
            f"shape {shape}. Resize the image before it enters a symbolic graph."
        )
    old_h, old_w = int(old_h), int(old_w)
    if old_h <= 0 or old_w <= 0:
        raise ValueError(
            f"image spatial extents must be positive; got height={old_h}, "
            f"width={old_w}"
        )

    # DECISION plan-2026-08-03T191222-1d751f81/D-005: reference SAM's
    # `ResizeLongestSide.get_preprocess_shape` rounds with `int(x + 0.5)`. Do
    # NOT "modernize" this to `round()` -- Python's `round` is banker's rounding
    # and disagrees on every exact .5 (`round(0.5) == 0`), which shifts the
    # padded coordinate frame by a pixel relative to every official SAM
    # checkpoint and every published prompt coordinate. Do NOT truncate either.
    # This helper is also deliberately a plain module-level function, NOT a
    # class and NOT a Keras layer: it carries no state and no weights, and it is
    # charged as a single-use abstraction until iteration 2's trainer adopts it.
    scale = target_length * 1.0 / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)

    return ops.image.resize(
        image,
        (new_h, new_w),
        interpolation=interpolation,
        data_format="channels_last",
    )
