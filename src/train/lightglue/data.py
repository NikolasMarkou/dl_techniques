"""Homography pair data pipeline for LightGlue training.

Builds a ``tf.data`` pipeline of ``(image0, image1, H0to1)`` pairs from a folder
of ordinary photographs (COCO ``train2017``). Each source image is decoded to
grayscale, centre-cropped to the target aspect ratio and resized to a SOURCE
frame ``source_scale`` times larger than the view (a source smaller than that
is simply resized up). ``image0`` and ``image1`` are then two independent
warped patches of that source, as in glue-factory's homography dataset, each
with an optional photometric jitter.

Border-free views
-----------------
Both views are sampled so that NO output pixel reads outside the source frame
(no zero fill, no black wedge). Glue-factory
(``gluefactory/datasets/homographies.py``) says it "yields an image pair
without border artifacts" and does it with ``sample_homography_corners``: both
views are patches whose corner quad lies inside the source. Here, for each
view, ``sample_homography_tf`` draws a perturbation ``P`` of the view
rectangle; the resulting quad is centred in the source and shrunk about the
source centre by the largest factor ``k`` that keeps all four corners inside
the frame (``k`` is capped so the view is never magnified beyond the source
resolution). Because the quad is convex and inside the frame, so is every
pixel of the view. The view-to-source map is ``M = A(k) @ P``.

Remaining differences to glue-factory, kept on purpose: the quad is a
perturbed rectangle shrunk to fit (glue-factory samples the corners directly
with a difficulty parameter and a convexity floor), so large rotations or
scales cost zoom rather than being re-drawn; the default ranges are per view,
and the relative homography between the views spans roughly twice them; there
is no photometric ``dark`` mode. Both views are warped (glue-factory has a
``right_only`` mode that leaves view 0 un-warped, not offered here).

Homography direction convention
-------------------------------
``H0to1 = M1^-1 @ M0`` is the FORWARD homography of
:mod:`dl_techniques.utils.homography`: a pixel ``p = (x, y)`` (x to the right,
y downward, origin top-left) in ``image0`` appears at
``dehomogenise(H0to1 @ [x, y, 1])`` in ``image1``, because both views show the
same source point ``M0 p = M1 H0to1 p``. ``warp_image_tf`` inverts its ``H``
internally, so each view is produced with ``M^-1`` as the source-to-view
transform. The tests verify this with a marked pixel.

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

# Per-VIEW ranges (both views get an independent perturbation, so the relative
# homography between them spans about twice these). Wider than the SuperPoint
# training defaults (rotation 25 deg, scale 0.8-1.2, perspective 0.0008,
# translation 0.1) in relative terms, shrunk per view to keep the pair
# difficulty near the previous single-warp defaults.
DEFAULT_HOMOGRAPHY_PARAMS: Dict[str, Union[float, Tuple[float, float]]] = {
    "rotation": float(np.deg2rad(20.0)),
    "scale": (0.8, 1.2),
    "perspective": 0.001,
    "translation": 0.08,
    "shear": 0.0,
}

# Source frame is this many times the view size (identity views are then a
# resize of the whole source, and perturbed quads fit with little zoom loss).
DEFAULT_SOURCE_SCALE = 1.5

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


def _view_to_source(
    view_hw: Tuple[int, int],
    source_hw: Tuple[int, int],
    seed: "tf.Tensor",
    params: Mapping[str, object],
) -> "tf.Tensor":
    """Border-free view-to-source homography ``M`` (see the module docstring).

    :param view_hw: ``(height, width)`` of the view.
    :param source_hw: ``(height, width)`` of the source frame.
    :param seed: length-2 int32 stateless seed for ``sample_homography_tf``.
    :param params: resolved homography ranges.
    :return: ``(3, 3)`` float64 map taking a view pixel to the source pixel it
        samples. All four view corners map into ``[1, S - 2]`` of the source
        (a one pixel margin, so bilinear taps stay in the frame).
    """
    # DECISION plan-2026-10-02T084508-dd2c07ac/D-017
    # Both views must be warped patches that stay inside the source frame. Do
    # NOT go back to warping one full frame with constant-0 fill (the old
    # image1): about a quarter of image1 was black and a third of its keypoints
    # sat on or next to the fill, all labelled dustbin, a learnable shortcut.
    # Guard: test_views_are_border_free in tests/test_train/test_lightglue/test_data.py.
    vh, vw = view_hw
    sh, sw = source_hw
    perturb = tf.cast(sample_homography_tf((vh, vw), seed, **params), tf.float64)
    corners = tf.constant(
        [[0.0, vw - 1.0, 0.0, vw - 1.0],
         [0.0, 0.0, vh - 1.0, vh - 1.0],
         [1.0, 1.0, 1.0, 1.0]],
        dtype=tf.float64,
    )
    quad = perturb @ corners
    quad_xy = quad[:2] / tf.maximum(quad[2:3], 1e-3)
    centre_v = tf.constant([(vw - 1.0) / 2.0, (vh - 1.0) / 2.0], dtype=tf.float64)
    centre_s = tf.constant([(sw - 1.0) / 2.0, (sh - 1.0) / 2.0], dtype=tf.float64)
    half_s = centre_s - 1.0
    extent = tf.maximum(tf.abs(quad_xy - centre_v[:, None]), 1e-6)
    k_fit = tf.reduce_min(half_s[:, None] / extent)
    k_cap = tf.constant((sw - 1.0) / (vw - 1.0 if vw > 1 else 1.0), tf.float64)
    k = tf.minimum(k_fit, k_cap)
    zero = tf.constant(0.0, tf.float64)
    one = tf.constant(1.0, tf.float64)
    offset = centre_s - k * centre_v
    affine = tf.stack([
        tf.stack([k, zero, offset[0]]),
        tf.stack([zero, k, offset[1]]),
        tf.stack([zero, zero, one]),
    ])
    return affine @ perturb


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
    source_scale: float = DEFAULT_SOURCE_SCALE,
) -> "tf.data.Dataset":
    """Build the homography pair dataset.

    Each element is a batch of::

        {"image0": (B, H, W, 1) float32 in [0, 1],
         "image1": (B, H, W, 1) float32 in [0, 1],
         "H0to1": (B, 3, 3) float32,
         "image_size0": (B, 2) float32 = (w, h),
         "image_size1": (B, 2) float32 = (w, h)}

    Both images are border-free warped patches of one source (see the module
    docstring): every pixel of both reads inside the source frame, none is
    zero-filled. A pixel ``p`` of ``image0`` appears at ``H0to1 @ p`` in
    ``image1``. ``image_size0/1`` are the constant ``(w, h)`` of the images (no
    padding is applied, so they equal the full image size).

    Randomness is stateless per element: view 1 uses the seed ``[seed, i]``,
    view 0 ``[seed + 16, i]`` and the jitters ``[seed + 1, i]`` and
    ``[seed + 2, i]``, where ``i`` is the element's position in the (shuffled,
    repeated) stream.
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
        than the SuperPoint training defaults. They apply to EACH view.
    :param photometric_jitter: ``False``/``0`` off; ``True`` or a float is the
        strength multiplier on the jitter applied to ``image1`` (after the
        warp, so nothing is ever filled).
    :param shuffle: shuffle the file order (reshuffled each epoch, seeded).
    :param repeat: repeat the dataset indefinitely.
    :param jitter_image0: also jitter ``image0`` (independent seed).
    :param drop_remainder: drop the last partial batch (static batch dim).
    :param ignore_errors: skip undecodable files instead of raising.
    :param source_scale: the decoded source frame is this many times the view
        size (``>= 1``; a smaller photograph is resized up to it).
    :return: the batched, prefetched ``tf.data.Dataset``.
    :raises ValueError: for an empty path list, a non-positive batch or size,
        ``source_scale < 1``, or unknown ``homography_params`` keys.
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
    if source_scale < 1.0:
        raise ValueError(f"source_scale must be >= 1, got {source_scale}")
    src_h = int(round(height * source_scale))
    src_w = int(round(width * source_scale))
    params = _resolve_homography_params(homography_params)
    strength = float(photometric_jitter)
    seed = int(seed)
    size_vec = np.array([width, height], dtype=np.float32)

    def _make(idx: "tf.Tensor", path: "tf.Tensor") -> Dict[str, "tf.Tensor"]:
        source = _decode_grayscale_crop(path, src_h, src_w)
        m0 = _view_to_source((height, width), (src_h, src_w),
                             _stateless_seed(seed, idx, 16), params)
        m1 = _view_to_source((height, width), (src_h, src_w),
                             _stateless_seed(seed, idx, 0), params)
        image0 = warp_image_tf(source, tf.linalg.inv(m0), (height, width))
        image1 = warp_image_tf(source, tf.linalg.inv(m1), (height, width))
        image1 = _photometric_jitter(
            image1, _stateless_seed(seed, idx, 1), strength
        )
        if jitter_image0:
            image0 = _photometric_jitter(
                image0, _stateless_seed(seed, idx, 2), strength
            )
        h_mat = tf.linalg.inv(m1) @ m0
        h_mat = tf.cast(h_mat / h_mat[2, 2], tf.float32)
        image0.set_shape((height, width, 1))
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
        f"jitter={strength} source_scale={source_scale} params={params}"
    )
    return ds
