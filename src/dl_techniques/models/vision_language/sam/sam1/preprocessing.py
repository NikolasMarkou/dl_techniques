"""
SAM 1 input transform, built by ``resize_longest_side``, applied before the
model.

Scales an image so its longest side equals the encoder's ``img_size``,
preserving aspect ratio. This is the transform reference SAM runs on the raw
image before its pipeline, and what makes ``SAM.preprocess`` well defined:
``SAM.preprocess`` only pads up to ``img_size`` and cannot shrink an image,
so it raises on an oversize input unless this transform runs first.

References:
    - Kirillov et al., 2023. Segment Anything. (https://arxiv.org/abs/2304.02643)
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
    Resize an image so its longest side equals ``target_length``.

    The aspect ratio is preserved, and the transform is symmetric in the two
    spatial axes, so whichever axis is longer is the one pinned to
    ``target_length``.

    :param image: Image tensor of shape ``(H, W, C)`` or ``(B, H, W, C)``,
        channels last, with statically known spatial extents.
    :type image: keras.KerasTensor
    :param target_length: Desired length of the longest side, in pixels.
        Must be positive.
    :type target_length: int
    :param interpolation: Interpolation mode forwarded to
        ``keras.ops.image.resize``. Defaults to ``'bilinear'``, matching
        reference SAM's PIL ``BILINEAR`` resize.
    :type interpolation: str
    :return: The resized image, same rank as the input, with
        ``max(new_h, new_w) == target_length``.
    :rtype: keras.KerasTensor
    :raises ValueError: If ``image`` is not rank 3 or 4, if either spatial
        extent is statically unknown or non-positive, or if
        ``target_length`` is not positive.

    Example:
        A ``(300, 900, 3)`` image at ``target_length=1024`` becomes
        ``(341, 1024, 3)``.
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

    # DECISION plan-2026-08-03T191222-1d751f81/D-005: round with int(x + 0.5),
    # never Python's round() -- banker's rounding disagrees on every exact .5 and shifts the padded coordinate frame by a pixel. See decisions.md.
    scale = target_length * 1.0 / max(old_h, old_w)
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)

    return ops.image.resize(
        image,
        (new_h, new_w),
        interpolation=interpolation,
        data_format="channels_last",
    )
