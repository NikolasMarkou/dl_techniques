"""Image geometry shared by the labelled-image loaders: crops and the constants behind them.

One home for the numbers and the one operation that more than one dataset module needs:

* :data:`EVAL_CROP_FRACTION`, :data:`RANDOM_RESIZED_CROP_AREA` and
  :data:`RANDOM_RESIZED_CROP_RATIO`, the ImageNet-classifier recipe (Inception-style
  random-resized crop for training, shorter side to ``size / 0.875`` then a centre crop for
  evaluation). ``imagenet2012`` and ``imagenette`` both import them.
* :func:`random_resized_crop` and :func:`resize_and_centre_crop`, the stateful tf.data forms
  Imagenette uses (ImageNet keeps its own stateless, antialiased forms: they differ on purpose).
* :func:`flip_and_pad_crop`, the flip plus pad-and-crop of the small-image datasets, which
  ``train.common.classification_data.flip_pad_crop`` and the Imagenette ``"resize"`` policy
  both build on.
"""

from typing import Optional, Tuple

import tensorflow as tf

#: Evaluation centre-crop fraction: the shorter side is resized to ``image_size /
#: EVAL_CROP_FRACTION`` and the centre ``image_size`` square is kept (the torchvision
#: 256 -> 224 ratio).
EVAL_CROP_FRACTION = 0.875

#: Area and aspect-ratio ranges of the training random-resized crop (the Inception /
#: torchvision ``RandomResizedCrop`` defaults).
RANDOM_RESIZED_CROP_AREA: Tuple[float, float] = (0.08, 1.0)
RANDOM_RESIZED_CROP_RATIO: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)


def random_resized_crop(
        image: tf.Tensor, image_size: int, seed: Optional[int] = None,
) -> tf.Tensor:
    """Random-resized crop plus horizontal flip of one ``[0, 1]`` image.

    Samples a box covering :data:`RANDOM_RESIZED_CROP_AREA` of the image at an
    aspect ratio in :data:`RANDOM_RESIZED_CROP_RATIO` (falling back to the whole
    image when no box fits), resizes it to ``image_size`` square, then flips.
    """
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4], tf.float32),
        min_object_covered=0.0,
        aspect_ratio_range=RANDOM_RESIZED_CROP_RATIO,
        area_range=RANDOM_RESIZED_CROP_AREA,
        max_attempts=10,
        use_image_if_no_bounding_boxes=True,
        seed=seed,
    )
    image = tf.slice(image, begin, size)
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return tf.image.random_flip_left_right(image, seed=seed)


def resize_and_centre_crop(image: tf.Tensor, image_size: int) -> tf.Tensor:
    """Resize the shorter side to ``image_size / EVAL_CROP_FRACTION``, keep the centre square."""
    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = (image_size / EVAL_CROP_FRACTION) / tf.reduce_min(shape)
    resized = tf.cast(tf.round(shape * scale), tf.int32)
    image = tf.image.resize(image, resized, method="bilinear")
    return tf.image.resize_with_crop_or_pad(image, image_size, image_size)


def flip_and_pad_crop(
        image: tf.Tensor, pad: int = 4, mode: str = "CONSTANT", flip: bool = True,
        seed: Optional[int] = None,
) -> tf.Tensor:
    """Random horizontal flip, then a ``pad``-pixel pad and a random crop back to the size.

    Runs on the ``[0, 1]`` image, BEFORE normalization, on purpose (the ``train_vit``
    D-006/D-007 lesson: augmenting normalized data and then clipping to ``[0, 1]`` saturates
    most pixels and silently creates a train/validation distribution mismatch). No
    ``clip_by_value`` here.

    :param image: ``(H, W, C)`` image.
    :param pad: Pixels added on each side.
    :param mode: ``tf.pad`` mode (``"CONSTANT"``, ``"REFLECT"``, ``"SYMMETRIC"``).
    :param flip: Whether to flip horizontally (not for digits).
    :param seed: Op-level seed of the flip and the crop; ``None`` draws from the global
        generator.
    """
    if flip:
        image = tf.image.random_flip_left_right(image, seed=seed)
    size = tf.shape(image)
    image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode=mode)
    return tf.image.random_crop(image, size, seed=seed)
