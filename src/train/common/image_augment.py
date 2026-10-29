"""Seeded, size-agnostic image augmentation shared by every vision trainer.

One home for what trainers used to hand-roll (``train_cliffordnet``'s AutoAugment and random
erasing, ``train_vit``'s brightness/contrast, the flips of alexnet, clip, the SOM and the CLIP
image-text path). Every operation:

* sees a ``[0, 1]`` float ``(H, W, C)`` image, BEFORE normalization (the ``train_vit``
  D-006/D-007 rule: no ``clip_by_value`` on normalized data, none here either);
* takes an ``int32`` seed pair first, ``op(seed, image) -> image``, and draws all its
  randomness from it (``tf.random.stateless_*``, or ``np.random.default_rng(seed)`` inside
  AutoAugment's PIL step), so a run is reproducible from the loader's seed;
* works at any ``H x W`` (nothing is hard-coded to 32).

:func:`as_hook` turns an operation into the loader's ``augment=`` hook
(:mod:`dl_techniques.datasets.vision.augment_hook`), :func:`build_augment` builds a named preset
(:data:`PRESETS`) and :func:`add_augmentation_arguments` declares ``--augment``. Outside a
seeded pipeline, :func:`unseeded` wraps an operation to draw its seed from the global TF
generator (the CLIP image-text map).
"""

import argparse
import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps

from dl_techniques.datasets.vision.augment_hook import Augment, seeded
from dl_techniques.datasets.vision.crops import flip_and_pad_crop

#: ``op(seed, image) -> image``: ``seed`` is an ``int32[2]`` tensor, ``image`` a ``[0, 1]`` float
#: ``(H, W, C)`` tensor.
ImageOp = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

# ---------------------------------------------------------------------
# AutoAugment policies
# Each sub-policy is a tuple of (op_name, probability, magnitude_index); magnitude indices map
# to operation-specific ranges over NUM_MAGNITUDE_BINS bins (torchvision's mapping).
# ---------------------------------------------------------------------

CIFAR10_POLICY: Tuple[Tuple[Tuple[str, float, int], ...], ...] = (
    (("Invert", 0.1, 7), ("Contrast", 0.2, 6)),
    (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
    (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
    (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)),
    (("ShearY", 0.7, 2), ("Invert", 0.4, 7)),
    (("Color", 0.6, 1), ("Equalize", 1.0, 2)),
    (("Invert", 0.6, 4), ("Rotate", 0.6, 4)),
    (("Solarize", 0.3, 5), ("AutoContrast", 0.1, 12)),
    (("Equalize", 0.2, 4), ("Rotate", 0.6, 8)),
    (("Color", 0.1, 0), ("Brightness", 0.7, 2)),
    (("Sharpness", 0.4, 7), ("TranslateX", 0.9, 9)),
    (("ShearX", 0.6, 5), ("Rotate", 0.7, 2)),
    (("AutoContrast", 0.8, 4), ("Solarize", 0.4, 5)),
    (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)),
    (("Equalize", 0.8, 8), ("Invert", 0.1, 3)),
    (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)),
    (("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)),
    (("TranslateY", 0.7, 9), ("TranslateY", 0.7, 9)),
    (("AutoContrast", 0.9, 0), ("Solarize", 0.4, 3)),
    (("Equalize", 0.7, 5), ("Invert", 0.1, 3)),
    (("Equalize", 0.7, 8), ("TranslateY", 0.3, 9)),
    (("Rotate", 0.8, 7), ("TranslateY", 0.7, 9)),
    (("ShearX", 0.7, 5), ("Equalize", 0.8, 5)),
)


NUM_MAGNITUDE_BINS = 31

#: The AutoAugment policies by name.
AUTOAUGMENT_POLICIES: Dict[str, Tuple[Tuple[Tuple[str, float, int], ...], ...]] = {
    "cifar10": CIFAR10_POLICY,
}

#: Fill of the area a rotation, shear or translation uncovers (mid-grey, as torchvision).
_AUTOAUGMENT_FILL = 128

# ---------------------------------------------------------------------
# AutoAugment (PIL, seeded numpy generator)
# ---------------------------------------------------------------------


def _apply_autoaugment_op(
        pil_img: Image.Image, op_name: str, magnitude_idx: int, rng: np.random.Generator,
) -> Image.Image:
    """Apply one AutoAugment operation to a PIL image.

    The magnitude mapping is torchvision's AutoAugment with ``NUM_MAGNITUDE_BINS`` bins; the
    sign of a signed operation is drawn from ``rng``.
    """
    n = NUM_MAGNITUDE_BINS - 1
    m = magnitude_idx
    width = pil_img.size[0]
    fill = _AUTOAUGMENT_FILL if pil_img.mode == "L" else (_AUTOAUGMENT_FILL,) * 3

    # Operations with no sign.
    if op_name == "Invert":
        return ImageOps.invert(pil_img)
    if op_name == "AutoContrast":
        return ImageOps.autocontrast(pil_img)
    if op_name == "Equalize":
        return ImageOps.equalize(pil_img)
    if op_name == "Posterize":
        return ImageOps.posterize(pil_img, max(1, int(8 - round(m / (n / 4.0)))))
    if op_name == "Solarize":
        return ImageOps.solarize(pil_img, int(255 - m / n * 255))

    if op_name in ("ShearX", "ShearY"):
        magnitude = m / n * 0.3
    elif op_name in ("TranslateX", "TranslateY"):
        magnitude = m / n * (150.0 / 331.0 * width)
    elif op_name == "Rotate":
        magnitude = m / n * 30.0
    elif op_name in ("Brightness", "Color", "Contrast", "Sharpness"):
        magnitude = m / n * 0.9
    else:
        raise ValueError(f"Unknown AutoAugment operation {op_name!r}")
    if rng.random() > 0.5:
        magnitude = -magnitude

    enhancers = {"Brightness": ImageEnhance.Brightness, "Color": ImageEnhance.Color,
                 "Contrast": ImageEnhance.Contrast, "Sharpness": ImageEnhance.Sharpness}
    if op_name in enhancers:
        return enhancers[op_name](pil_img).enhance(1.0 + magnitude)
    if op_name == "Rotate":
        return pil_img.rotate(magnitude, fillcolor=fill)
    affine = {
        "ShearX": (1, magnitude, 0, 0, 1, 0),
        "ShearY": (1, 0, 0, magnitude, 1, 0),
        "TranslateX": (1, 0, int(round(magnitude)), 0, 1, 0),
        "TranslateY": (1, 0, 0, 0, 1, int(round(magnitude))),
    }[op_name]
    return pil_img.transform(pil_img.size, Image.AFFINE, affine, fillcolor=fill)


def _autoaugment_array(
        image: np.ndarray, policy: Sequence[Sequence[Tuple[str, float, int]]],
        rng: np.random.Generator,
) -> np.ndarray:
    """One random sub-policy of ``policy`` on a float32 ``[0, 1]`` HWC array."""
    channels = image.shape[-1]
    pixels = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(pixels[..., 0] if channels == 1 else pixels)
    for op_name, probability, magnitude_idx in policy[rng.integers(len(policy))]:
        if rng.random() > probability:
            continue
        pil_img = _apply_autoaugment_op(pil_img, op_name, magnitude_idx, rng)
    out = np.asarray(pil_img, dtype=np.float32) / 255.0
    return out.reshape(image.shape)


def _generator(seed: np.ndarray) -> np.random.Generator:
    """A numpy generator fixed by an ``int32``/``int64`` seed pair."""
    return np.random.default_rng([int(value) & 0xFFFFFFFF for value in np.ravel(seed)])


def autoaugment(policy: str = "cifar10") -> ImageOp:
    """AutoAugment: one random sub-policy of ``policy`` per image (PIL, 1 or 3 channels).

    :param policy: A key of :data:`AUTOAUGMENT_POLICIES`.
    :raises ValueError: Unknown ``policy``.
    """
    if policy not in AUTOAUGMENT_POLICIES:
        raise ValueError(
            f"Unknown AutoAugment policy {policy!r}; known: {sorted(AUTOAUGMENT_POLICIES)}")
    table = AUTOAUGMENT_POLICIES[policy]

    def run(image: np.ndarray, seed: np.ndarray) -> np.ndarray:
        return _autoaugment_array(image, table, _generator(seed))

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        image = tf.cast(image, tf.float32)
        out = tf.numpy_function(run, [image, tf.cast(seed, tf.int64)], tf.float32)
        out.set_shape(image.shape)
        return out

    return op


# ---------------------------------------------------------------------
# Tensor operations
# ---------------------------------------------------------------------


def flip_pad_crop(pad: int = 4, mode: str = "CONSTANT", flip: bool = True) -> ImageOp:
    """Random horizontal flip, then a ``pad``-pixel pad and a random crop back to the size.

    :param pad: Pixels added on each side.
    :param mode: ``tf.pad`` mode (``"CONSTANT"``, ``"REFLECT"``, ``"SYMMETRIC"``).
    :param flip: Whether to flip horizontally (not for digits).
    """
    return lambda seed, image: flip_and_pad_crop(image, pad=pad, mode=mode, flip=flip, seed=seed)


def random_flip() -> ImageOp:
    """Random horizontal flip."""
    return lambda seed, image: tf.image.stateless_random_flip_left_right(image, seed)


def random_crop(height: int, width: int) -> ImageOp:
    """A random ``height x width`` window of a larger image (the train crop of a resize-then-crop).

    :raises ValueError: A non-positive size. A window larger than the image fails at run time.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"crop size must be positive, got {height}x{width}")

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        out = tf.image.stateless_random_crop(image, (height, width, image.shape[-1]), seed)
        out.set_shape((height, width, image.shape[-1]))
        return out

    return op


def random_dihedral() -> ImageOp:
    """Independent random horizontal and vertical flips and a random multiple of 90 degrees.

    For square patches (a rotation of a non-square image swaps its sides).
    """

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        s = tf.unstack(tf.random.experimental.stateless_split(seed, 3))
        image = tf.image.stateless_random_flip_left_right(image, s[0])
        image = tf.image.stateless_random_flip_up_down(image, s[1])
        k = tf.random.stateless_uniform([], s[2], 0, 4, dtype=tf.int32)
        return tf.image.rot90(image, k)

    return op


def photometric(
        brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0, hue: float = 0.0,
) -> ImageOp:
    """Random brightness, contrast, saturation and hue on a ``[0, 1]`` image, no clipping.

    :param brightness: Maximum absolute brightness delta (``0`` skips it).
    :param contrast: Contrast factor drawn from ``[1 - contrast, 1 + contrast]``.
    :param saturation: Saturation factor drawn from ``[1 - saturation, 1 + saturation]``
        (3-channel images only).
    :param hue: Maximum absolute hue delta in ``[0, 0.5]`` (3-channel images only).
    :raises ValueError: A negative amount, ``contrast`` or ``saturation`` of 1 or more, or
        ``hue`` above 0.5.
    """
    if min(brightness, contrast, saturation, hue) < 0 or max(contrast, saturation) >= 1 or hue > 0.5:
        raise ValueError(
            f"photometric amounts must be >= 0, contrast and saturation < 1, hue <= 0.5; got "
            f"brightness={brightness}, contrast={contrast}, saturation={saturation}, hue={hue}")

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
        if brightness:
            image = tf.image.stateless_random_brightness(image, brightness, seeds[0])
        if contrast:
            image = tf.image.stateless_random_contrast(
                image, 1.0 - contrast, 1.0 + contrast, seeds[1])
        if image.shape[-1] == 3:
            if saturation:
                image = tf.image.stateless_random_saturation(
                    image, 1.0 - saturation, 1.0 + saturation, seeds[2])
            if hue:
                image = tf.image.stateless_random_hue(image, hue, seeds[3])
        return image

    return op


def random_affine(
        rotation: float = 0.0, zoom: float = 0.0, translation: float = 0.0,
        fill_mode: str = "REFLECT",
) -> ImageOp:
    """Random rotation, zoom and translation about the image centre, in one bilinear resample.

    :param rotation: Maximum absolute rotation in degrees.
    :param zoom: Zoom factor drawn from ``[1 - zoom, 1 + zoom]`` (above 1 magnifies).
    :param translation: Maximum shift as a fraction of the height and of the width.
    :param fill_mode: What fills the uncovered border: ``"REFLECT"``, ``"NEAREST"``,
        ``"WRAP"`` or ``"CONSTANT"`` (zeros).
    :raises ValueError: A negative amount, ``zoom`` of 1 or more, or an unknown ``fill_mode``.
    """
    if min(rotation, zoom, translation) < 0 or zoom >= 1:
        raise ValueError(f"random_affine amounts must be >= 0 and zoom < 1; got "
                         f"rotation={rotation}, zoom={zoom}, translation={translation}")
    if fill_mode not in ("REFLECT", "NEAREST", "WRAP", "CONSTANT"):
        raise ValueError(f"Unknown fill_mode {fill_mode!r}")

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        h_f, w_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        s = tf.unstack(tf.random.experimental.stateless_split(seed, 3))
        angle = tf.random.stateless_uniform([], s[0], -rotation, rotation) * (math.pi / 180.0)
        inverse_zoom = 1.0 / tf.random.stateless_uniform([], s[1], 1.0 - zoom, 1.0 + zoom)
        shift = tf.random.stateless_uniform([2], s[2], -translation, translation) * [w_f, h_f]
        cos, sin = tf.cos(angle) * inverse_zoom, tf.sin(angle) * inverse_zoom
        cx, cy = (w_f - 1.0) / 2.0, (h_f - 1.0) / 2.0
        # The projective transform maps an OUTPUT pixel to the INPUT pixel it samples.
        matrix = tf.stack([
            cos, -sin, cx - (cos * cx - sin * cy) - shift[0],
            sin, cos, cy - (sin * cx + cos * cy) - shift[1],
            0.0, 0.0])
        out = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.cast(image, tf.float32)[None], transforms=matrix[None],
            output_shape=tf.stack([height, width]), fill_value=0.0,
            interpolation="BILINEAR", fill_mode=fill_mode)[0]
        out.set_shape(image.shape)
        return tf.cast(out, image.dtype)

    return op


def random_erasing(
        prob: float, fill: Sequence[float] = (0.0,), area: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3), trials: int = 100,
) -> ImageOp:
    """Erase one random rectangle with probability ``prob`` (torchvision ``RandomErasing``).

    The rectangle's area fraction is drawn from ``area`` and its aspect ratio log-uniformly from
    ``ratio``; the first of ``trials`` draws that fits inside the image is used, and none that
    fits leaves the image as it is.

    :param prob: Probability of erasing.
    :param fill: The value written, a scalar or one per channel. In ``[0, 1]`` space the
        per-channel dataset mean is the equivalent of "0 after normalization".
    :param area: Erased fraction of the image area, ``(low, high)``.
    :param ratio: Aspect ratio ``w / h``, ``(low, high)``.
    :param trials: Candidate rectangles drawn per image.
    :raises ValueError: ``prob`` outside ``[0, 1]`` or an empty range.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if not 0.0 < area[0] <= area[1] <= 1.0 or not 0.0 < ratio[0] <= ratio[1]:
        raise ValueError(f"bad erasing ranges: area={area}, ratio={ratio}")
    value = tf.constant(np.asarray(fill, dtype=np.float32).reshape(1, 1, -1))

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        h_f, w_f = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        s = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
        wanted = tf.random.stateless_uniform([], s[0]) < prob
        target = tf.random.stateless_uniform([trials], s[1], area[0], area[1]) * h_f * w_f
        aspect = tf.exp(tf.random.stateless_uniform(
            [trials], s[2], math.log(ratio[0]), math.log(ratio[1])))
        erase_h = tf.cast(tf.round(tf.sqrt(target * aspect)), tf.int32)
        erase_w = tf.cast(tf.round(tf.sqrt(target / aspect)), tf.int32)
        fits = (erase_h > 0) & (erase_h < height) & (erase_w > 0) & (erase_w < width)
        first = tf.argmax(tf.cast(fits, tf.int32), output_type=tf.int32)
        erase = wanted & tf.reduce_any(fits)
        eh, ew = erase_h[first], erase_w[first]
        where = tf.random.stateless_uniform([2], s[3])
        top = tf.cast(tf.floor(where[0] * tf.cast(height - eh, tf.float32)), tf.int32)
        left = tf.cast(tf.floor(where[1] * tf.cast(width - ew, tf.float32)), tf.int32)
        rows = tf.range(height)[:, None]
        cols = tf.range(width)[None, :]
        inside = ((rows >= top) & (rows < top + eh) & (cols >= left) & (cols < left + ew)
                  & erase)[..., None]
        return tf.where(inside, tf.cast(value, image.dtype), image)

    return op


def compose(*ops: ImageOp) -> ImageOp:
    """Apply ``ops`` in order, each with its own seed split off the element's seed."""
    if not ops:
        raise ValueError("compose needs at least one operation")
    if len(ops) == 1:
        return ops[0]

    def op(seed: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
        seeds = tf.unstack(tf.random.experimental.stateless_split(seed, len(ops)))
        for one, one_seed in zip(ops, seeds):
            image = one(one_seed, image)
        return image

    return op


# ---------------------------------------------------------------------
# Hooks and presets
# ---------------------------------------------------------------------


def as_hook(op: ImageOp) -> Augment:
    """The loader's seeded ``augment=`` hook that applies ``op`` and keeps the label."""

    @seeded
    def hook(seed: tf.Tensor, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return op(seed, image), label

    return hook


def unseeded(op: ImageOp) -> Callable[[tf.Tensor], tf.Tensor]:
    """``op`` as an ``image -> image`` map for a pipeline that carries no per-element seed.

    The seed is drawn from the global TF generator, so the result follows ``tf.random.set_seed``
    but, under a parallel ``tf.data`` map, not the element order.
    """
    return lambda image: op(tf.random.uniform([2], 0, 2 ** 31 - 1, dtype=tf.int32), image)


#: The preset names of :func:`build_op`.
PRESETS = ("none", "flip", "flip_pad_crop", "cifar_light", "cifar_autoaugment", "photometric")


def build_op(
        name: str, *, random_erasing_prob: float = 0.0, fill: Sequence[float] = (0.0,),
        pad: int = 4,
) -> Optional[ImageOp]:
    """The operation of a preset, or ``None`` for ``"none"``.

    * ``flip``: horizontal flip.
    * ``flip_pad_crop``: flip, then a ``pad``-pixel zero pad and a random crop back.
    * ``cifar_light``: flip, ``pad``-pixel REFLECT pad-crop, brightness 0.1, contrast 0.9-1.1
      (``train_vit``'s CIFAR recipe).
    * ``cifar_autoaugment``: flip, ``pad``-pixel zero pad-crop, the CIFAR-10 AutoAugment policy,
      then random erasing with ``random_erasing_prob`` and ``fill`` (arXiv:2601.06793v2's recipe).
    * ``photometric``: flip, brightness 0.1, contrast 0.9-1.1, saturation 0.9-1.1, hue 0.05.

    :param name: One of :data:`PRESETS`.
    :param random_erasing_prob: ``cifar_autoaugment`` only.
    :param fill: ``cifar_autoaugment`` only: the erased value, the dataset's per-channel mean.
    :param pad: Pad of the pad-crop presets.
    :raises ValueError: Unknown ``name``, or ``random_erasing_prob`` given to another preset.
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown augmentation preset {name!r}; known: {list(PRESETS)}")
    if random_erasing_prob and name != "cifar_autoaugment":
        raise ValueError(f"random_erasing_prob does not apply to the {name!r} preset")
    if name == "none":
        return None
    if name == "flip":
        return random_flip()
    if name == "flip_pad_crop":
        return flip_pad_crop(pad)
    if name == "cifar_light":
        return compose(flip_pad_crop(pad, mode="REFLECT"), photometric(brightness=0.1, contrast=0.1))
    if name == "photometric":
        return compose(random_flip(), photometric(0.1, 0.1, 0.1, 0.05))
    ops = [flip_pad_crop(pad), autoaugment("cifar10")]
    if random_erasing_prob:
        ops.append(random_erasing(random_erasing_prob, fill))
    return compose(*ops)


def build_augment(name: str, **kwargs) -> Optional[Augment]:
    """The loader hook of a preset (see :func:`build_op`), or ``None`` for ``"none"``."""
    op = build_op(name, **kwargs)
    return None if op is None else as_hook(op)


def add_augmentation_arguments(
        parser: argparse.ArgumentParser, default: Optional[str] = None,
) -> None:
    """Declare ``--augment`` (a preset of :data:`PRESETS`).

    :param parser: The parser to extend.
    :param default: The preset when the flag is absent; ``None`` lets the trainer pick one per
        dataset and refuse a combination that does not apply.
    """
    parser.add_argument(
        "--augment", type=str, default=default, choices=PRESETS,
        help="Train augmentation preset on the [0, 1] image before normalization "
             "(train.common.image_augment). 'none' adds nothing to the loader's own crop. "
             + ("" if default is None else f"Default: {default}."),
    )
