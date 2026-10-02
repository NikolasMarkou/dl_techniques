"""Homography pair data pipeline for LightGlue training.

Builds a ``tf.data`` pipeline of ``(image0, image1, H0to1)`` pairs from a folder
of ordinary photographs (COCO ``train2017``): each source image is decoded to
grayscale, centre-cropped to the target aspect ratio and resized; ``image1`` is
``image0`` warped by a randomly sampled homography, with an optional photometric
jitter.

Homography direction convention
-------------------------------
``H0to1`` is the FORWARD homography of :mod:`dl_techniques.utils.homography`:
a pixel ``p = (x, y)`` (x to the right, y downward, origin top-left) in
``image0`` appears at ``dehomogenise(H0to1 @ [x, y, 1])`` in ``image1``.
``warp_image_tf`` inverts ``H`` internally because the underlying op takes the
output-to-input transform, so ``H0to1`` is passed to it unchanged and is also
what is emitted. The tests verify this with a marked pixel.

Out-of-frame pixels
-------------------
Pixels of ``image1`` whose preimage lies outside ``image0`` are exactly zero
(constant fill). Photometric jitter is applied to the source copy BEFORE the
warp, so the fill stays exactly zero.

Image decoding
--------------
``train.superpoint.train_superpoint._decode_grayscale`` is NOT imported: it
sits in a module that pulls in the whole ``train.common`` stack (about 4 s and
a large part of the model package), and it resizes to a square without
cropping and threads a label through. A short local decode (same
``read_file -> decode_image -> rgb_to_grayscale -> bilinear resize`` chain)
that also supports a ``(H, W)`` target with a centre crop is used instead.

Everything here is CPU-only ``tf.data`` work; importing this module creates no
tensor and does not initialise any device.
"""

import os
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

from dl_techniques.utils.homography import sample_homography_tf, warp_image_tf
from dl_techniques.utils.logger import logger

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

# Wider than the SuperPoint training defaults (rotation 25 deg, scale
# 0.8-1.2, perspective 0.0008, translation 0.1), closer to the glue-factory
# homography benchmark difficulty.
DEFAULT_HOMOGRAPHY_PARAMS: Dict[str, Union[float, Tuple[float, float]]] = {
    "rotation": float(np.deg2rad(30.0)),
    "scale": (0.7, 1.3),
    "perspective": 0.002,
    "translation": 0.15,
    "shear": 0.0,
}

# Photometric jitter magnitudes at strength 1.0.
_BRIGHTNESS = 0.15
_CONTRAST = 0.3
_GAMMA = 0.25
_NOISE_STD = 0.02


def list_images(root: str) -> Sequence[str]:
    """Return the sorted image files directly inside a directory.

    :param root: directory holding ``.jpg``, ``.jpeg`` or ``.png`` files
        (extension match is case-insensitive, no recursion).
    :return: sorted list of absolute-or-as-given file paths.
    :raises FileNotFoundError: if ``root`` is not a directory.
    :raises ValueError: if the directory holds no image file.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"image directory does not exist: {root}")
    paths = sorted(
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not paths:
        raise ValueError(
            f"no image files {IMAGE_EXTENSIONS} found in directory: {root}"
        )
    return paths


def _decode_grayscale_crop(path: "tf.Tensor", height: int, width: int) -> "tf.Tensor":
    """Decode one image to ``(height, width, 1)`` float32 in ``[0, 1]``.

    A centre crop to the target aspect ratio (cropping the longer side, so the
    shorter side is kept whole) precedes the bilinear resize, so the content is
    not stretched.

    :param path: scalar string tensor.
    :param height: target height.
    :param width: target width.
    :return: grayscale image tensor with a static shape.
    """
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_grayscale(img)
    shp = tf.shape(img)
    ih, iw = shp[0], shp[1]
    # Largest crop of the target aspect ratio (width / height) that fits.
    crop_h = tf.minimum(ih, (iw * height) // width)
    crop_w = tf.minimum(iw, (ih * width) // height)
    crop_h = tf.maximum(crop_h, 1)
    crop_w = tf.maximum(crop_w, 1)
    off_h = (ih - crop_h) // 2
    off_w = (iw - crop_w) // 2
    img = tf.image.crop_to_bounding_box(img, off_h, off_w, crop_h, crop_w)
    img = tf.image.resize(img, (height, width), method="bilinear")
    img = tf.clip_by_value(img, 0.0, 1.0)
    img.set_shape((height, width, 1))
    return img


def _stateless_seed(base_seed: int, idx: "tf.Tensor", stream: int) -> "tf.Tensor":
    """Length-2 int32 seed ``[base_seed + stream, idx]`` (per-element stateless)."""
    return tf.stack(
        [tf.constant(base_seed + stream, dtype=tf.int32), tf.cast(idx, tf.int32)]
    )


def _photometric_jitter(
    image: "tf.Tensor", seed: "tf.Tensor", strength: float
) -> "tf.Tensor":
    """Brightness, contrast, gamma and light gaussian noise, clipped to [0, 1].

    :param image: ``(H, W, 1)`` float32 in ``[0, 1]``.
    :param seed: length-2 int32 stateless seed.
    :param strength: multiplier on the jitter magnitudes (``0`` is a no-op).
    :return: jittered image, same shape.
    """
    if strength <= 0.0:
        return image

    def _u(lo: float, hi: float, sub: int) -> "tf.Tensor":
        s = seed + tf.constant([sub, sub * 5 + 3], dtype=tf.int32)
        return tf.random.stateless_uniform([], seed=s, minval=lo, maxval=hi)

    gamma = tf.exp(_u(-_GAMMA, _GAMMA, 0) * strength)
    out = tf.pow(tf.clip_by_value(image, 0.0, 1.0), gamma)
    contrast = 1.0 + _u(-_CONTRAST, _CONTRAST, 1) * strength
    mean = tf.reduce_mean(out)
    out = (out - mean) * contrast + mean
    out = out + _u(-_BRIGHTNESS, _BRIGHTNESS, 2) * strength
    noise_std = _u(0.0, _NOISE_STD, 3) * strength
    noise = tf.random.stateless_normal(
        tf.shape(out), seed=seed + tf.constant([4, 23], dtype=tf.int32)
    )
    out = out + noise * noise_std
    return tf.clip_by_value(out, 0.0, 1.0)


def _resolve_homography_params(
    homography_params: Optional[Mapping[str, object]]
) -> Dict[str, object]:
    """Merge user ranges over :data:`DEFAULT_HOMOGRAPHY_PARAMS` (unknown keys raise)."""
    params = dict(DEFAULT_HOMOGRAPHY_PARAMS)
    if homography_params:
        unknown = set(homography_params) - set(params)
        if unknown:
            raise ValueError(
                f"unknown homography_params keys {sorted(unknown)}; "
                f"valid keys: {sorted(params)}"
            )
        params.update(homography_params)
    params["scale"] = (float(params["scale"][0]), float(params["scale"][1]))
    return params


def make_pair_dataset(
    image_paths: Sequence[str],
    image_size: Union[int, Tuple[int, int]],
    batch_size: int,
    seed: int = 0,
    homography_params: Optional[Mapping[str, object]] = None,
    photometric_jitter: Union[bool, float] = True,
    shuffle: bool = True,
    repeat: bool = False,
    jitter_image0: bool = False,
    drop_remainder: bool = True,
    ignore_errors: bool = False,
) -> "tf.data.Dataset":
    """Build the homography pair dataset.

    Each element is a batch of::

        {"image0": (B, H, W, 1) float32 in [0, 1],
         "image1": (B, H, W, 1) float32 in [0, 1],
         "H0to1": (B, 3, 3) float32,
         "image_size0": (B, 2) float32 = (w, h),
         "image_size1": (B, 2) float32 = (w, h)}

    ``image1`` is ``image0`` warped by ``H0to1`` (a pixel ``p`` of ``image0``
    appears at ``H0to1 @ p`` in ``image1``, see the module docstring); pixels
    with no source become zero. ``image_size0/1`` are the constant ``(w, h)``
    of the images (no padding is applied, so they equal the full image size).

    Randomness is stateless per element: the homography uses the seed
    ``[seed, i]`` and the jitters ``[seed + 1, i]`` and ``[seed + 2, i]``,
    where ``i`` is the element's position in the (shuffled, repeated) stream.
    A fixed ``seed`` is therefore deterministic, element ``i`` of different
    epochs gets a different homography, and different ``seed`` values give
    different pairs. Runs on CPU only.

    :param image_paths: image files (see :func:`list_images`).
    :param image_size: ``int`` (square) or ``(H, W)``.
    :param batch_size: batch size.
    :param seed: base seed (also seeds the shuffle).
    :param homography_params: overrides for ``rotation`` (radians),
        ``scale`` ((lo, hi)), ``perspective``, ``translation`` (fraction of
        the size) and ``shear`` (radians); see
        :data:`DEFAULT_HOMOGRAPHY_PARAMS` for the defaults, which are wider
        than the SuperPoint training defaults.
    :param photometric_jitter: ``False``/``0`` off; ``True`` or a float is the
        strength multiplier on the jitter applied to ``image1``.
    :param shuffle: shuffle the file order (reshuffled each epoch, seeded).
    :param repeat: repeat the dataset indefinitely.
    :param jitter_image0: also jitter ``image0`` (independent seed).
    :param drop_remainder: drop the last partial batch (static batch dim).
    :param ignore_errors: skip undecodable files instead of raising.
    :return: the batched, prefetched ``tf.data.Dataset``.
    :raises ValueError: for an empty path list, a non-positive batch or size,
        or unknown ``homography_params`` keys.
    """
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if isinstance(image_size, int):
        height = width = int(image_size)
    else:
        height, width = int(image_size[0]), int(image_size[1])
    if height < 1 or width < 1:
        raise ValueError(f"image_size must be positive, got {image_size}")
    params = _resolve_homography_params(homography_params)
    strength = float(photometric_jitter)
    seed = int(seed)
    size_vec = np.array([width, height], dtype=np.float32)

    def _make(idx: "tf.Tensor", path: "tf.Tensor") -> Dict[str, "tf.Tensor"]:
        image0 = _decode_grayscale_crop(path, height, width)
        h_mat = sample_homography_tf(
            (height, width), _stateless_seed(seed, idx, 0), **params
        )
        source = _photometric_jitter(
            image0, _stateless_seed(seed, idx, 1), strength
        )
        image1 = warp_image_tf(source, h_mat)
        if jitter_image0:
            image0 = _photometric_jitter(
                image0, _stateless_seed(seed, idx, 2), strength
            )
        image1.set_shape((height, width, 1))
        h_mat.set_shape((3, 3))
        return {
            "image0": image0,
            "image1": image1,
            "H0to1": h_mat,
            "image_size0": tf.constant(size_vec),
            "image_size1": tf.constant(size_vec),
        }

    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in image_paths])
    if shuffle:
        ds = ds.shuffle(
            len(image_paths), seed=seed, reshuffle_each_iteration=True
        )
    if repeat:
        ds = ds.repeat()
    ds = ds.enumerate()
    ds = ds.map(_make, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if ignore_errors:
        ds = ds.ignore_errors(log_warning=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    logger.info(
        f"lightglue pair dataset: n={len(image_paths)} size=({height},{width}) "
        f"batch={batch_size} seed={seed} shuffle={shuffle} repeat={repeat} "
        f"jitter={strength} params={params}"
    )
    return ds
