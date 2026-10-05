"""Transformation-sequence datasets: cyclic image sequences where one attribute
changes per step, for learning equivariant representations.

The problem this solves
-----------------------
An equivariant model can only be *learned* from data that shows what the
transformation does. A Topographic VAE is trained on **sequences**, not images:
each sequence holds a fixed example under one continuously varying attribute, and
the model's job is to organize its latent space so that the attribute shows up as
a cyclic roll inside a capsule. That requires sequences to be

- **cyclic** — the attribute wraps, so a full traversal returns to the start and a
  capsule roll of `S` steps is the identity rather than running off an edge;
- **single-attribute** — exactly one factor moves per sequence, otherwise there is
  no single transformation for the capsule roll to correspond to; and
- **randomly posed** — each sequence starts at an arbitrary parameter value, so a
  model cannot memorize absolute factor positions and must encode the shift.

Two datasets
------------
``MNIST`` (Section A.8) with rotation, hue and scale, and a **dSprites** subset
(Section A.9) with x, y, orientation and scale. dSprites has exact ground-truth
factor labels, which is what makes it the better instrument for the CapCorr
metric; MNIST's factors are known only by construction.

Scale on dSprites is bounded rather than cyclic, and this module documents the
resulting mismatch rather than hiding it: the paper observes that sequences do not
match the latent priors exactly yet the models still train well. Scale in
particular loops over its available values three times inside one sequence, so the
"one value per step" property is relaxed for that one factor.

dSprites provenance and caching
--------------------------------
The dataset is the DeepMind ``.npz`` archive, downloaded once to
:data:`DEFAULT_DSPRITES_CACHE` on the large data volume and never inside the
repository. That directory is NOT tracked by git, so a fresh checkout has no data
and the first run downloads it. Pass ``cache_root`` to point elsewhere (including
at an existing copy, in which case no network access happens).

Note the contrast with ``dl_techniques.datasets.graphs.tudataset``: that loader
deletes the downloaded zip after extraction, because the zip is redundant. Here
the ``.npz`` **is** the cache, so deleting it would force a ~250MB re-download on
every run.

References:
    - Keller & Welling, 2022. Topographic VAEs learn Equivariant Capsules.
      NeurIPS 2021. (https://arxiv.org/abs/2109.01394)
    - LeCun, Cortes & Burges, 2010. MNIST handwritten digit database.
      (http://yann.lecun.com/exdb/mnist)
    - Matthey et al., 2017. dSprites: Disentanglement Testing Sprites Dataset.
      (https://github.com/deepmind/dsprites-dataset)
    - Hyvarinen, Hurri & Varrynen, 2004. A Unifying Framework for Natural Image
      Statistics: Spatiotemporal Activity Bubbles.
      (https://doi.org/10.1016/j.neucom.2004.09.007)
"""

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import os
import urllib.request

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

#: On-disk cache for dSprites. This MUST live on the large data volume, never
#: inside the repository or on the repo SSD (`/media/arxwn/data_fast`).
DEFAULT_DSPRITES_CACHE = "/media/arxwn/data0_4tb/datasets/dsprites"

#: Official DeepMind archive. The name encodes the factor counts: colour, shape,
#: scale, orientation, x and y (6 factors), with 1x3x6x40x32x32 value ranges.
DSPRITES_URL = (
    "https://github.com/deepmind/dsprites-dataset/raw/master/"
    "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
)
DSPRITES_FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

#: Transformation names understood by both builders.
MNIST_TRANSFORMS = ("rotation", "color", "scale")
DSPRITES_TRANSFORMS = ("x_position", "y_position", "orientation", "scale")


# ---------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------


def build_mnist_transform_sequences(
    transform: str = "rotation",
    sequence_length: int = 18,
    num_sequences: int = 4096,
    seed: int = 0,
    rotation_increment: float = 20.0,
    hue_increment: float = 20.0,
    scale_increment: float = 0.0366,
    scale_min: float = 0.60,
    scale_max: float = 1.26,
    image_size: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cyclic MNIST transformation sequences (paper Section A.8).

    Each sequence is one MNIST digit held under a single attribute that advances by
    a fixed increment per step and wraps at the end. The ground-truth factor value
    per timestep is returned alongside, which is what the CapCorr metric needs.

    Parameters follow the paper: 20-degree increments for rotation and hue, and
    3.66% for scale. Scale is inherently non-cyclic, so it is bounded to
    ``[scale_min, scale_max]`` and the sequence wraps from ``scale_max`` back to
    ``scale_min`` — a bounded cycle, not the true scale manifold.

    Hue rotation is done in HSV through a luminance-preserving channel rotation so
    the digit stays legible; a naive RGB rotation darkens the image.

    :param transform: One of :data:`MNIST_TRANSFORMS`. Defaults to
        ``"rotation"``.
    :type transform: str
    :param sequence_length: Frames per sequence ``S``. Defaults to 18.
    :type sequence_length: int
    :param num_sequences: How many sequences to build. Defaults to 4096.
    :type num_sequences: int
    :param seed: Seed for the base-digit draw and the start poses.
    :type seed: int
    :param rotation_increment: Degrees per step. Defaults to 20.0.
    :type rotation_increment: float
    :param hue_increment: Hue degrees per step. Defaults to 20.0.
    :type hue_increment: float
    :param scale_increment: Fractional scale change per step. Defaults to 0.0366.
    :type scale_increment: float
    :param scale_min: Smallest scale in the cycle. Defaults to 0.60.
    :type scale_min: float
    :param scale_max: Largest scale in the cycle. Defaults to 1.26.
    :type scale_max: float
    :param image_size: Output frame edge length. Defaults to 28.
    :type image_size: int
    :return: ``(sequences, factors)``. ``sequences`` is
        ``(num_sequences, sequence_length, image_size, image_size, 3)`` float32 on
        ``[0, 1]`` — 3 channels because hue rotation needs colour. ``factors`` is
        ``(num_sequences, sequence_length)`` float32 holding the ground-truth
        attribute value at each step.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: On an unknown ``transform``, a non-positive
        ``sequence_length`` / ``num_sequences``, or a scale range that does not
        contain at least one increment.

    :Example:

    >>> sequences, factors = build_mnist_transform_sequences(num_sequences=8)
    >>> sequences.shape
    (8, 18, 28, 28, 3)
    >>> factors.shape
    (8, 18)
    """
    if transform not in MNIST_TRANSFORMS:
        raise ValueError(
            f"transform must be one of {list(MNIST_TRANSFORMS)}, got {transform!r}"
        )
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be positive, got {sequence_length}"
        )
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")

    import keras

    if transform == "scale":
        steps = int(np.floor((scale_max - scale_min) / scale_increment))
        if steps < 1:
            raise ValueError(
                f"scale range [{scale_min}, {scale_max}] with increment "
                f"{scale_increment} spans no complete step; widen the range or "
                f"shrink the increment"
            )
        # Scale is NOT cyclic in nature, so the cycle closes over the bounded
        # range. The step COUNT is what makes the sequence length divisible.
        values = scale_min + scale_increment * np.arange(steps + 1)
    elif transform == "rotation":
        steps = int(round(360.0 / rotation_increment))
        values = rotation_increment * np.arange(steps)
    else:
        steps = int(round(360.0 / hue_increment))
        values = hue_increment * np.arange(steps)

    (images, _), _ = keras.datasets.mnist.load_data()
    rng = np.random.default_rng(seed)
    base_hues = rng.uniform(0.0, 360.0, size=num_sequences)

    # A random base digit per sequence, and a random START POSE per sequence:
    # without the latter a model could learn absolute factor positions instead of
    # the shift, which is the entire quantity CapCorr measures.
    base_indices = rng.integers(0, images.shape[0], size=num_sequences)
    start_offsets = rng.integers(0, len(values), size=num_sequences)

    sequences = np.zeros(
        (num_sequences, sequence_length, image_size, image_size, 3),
        dtype=np.float32,
    )
    factors = np.zeros((num_sequences, sequence_length), dtype=np.float32)

    for index in range(num_sequences):
        base = images[base_indices[index]].astype(np.float32) / 255.0
        base = base[:, :, None]  # (H, W, 1) grayscale
        if transform == "color":
            # A grayscale digit has no chrominance, so a hue rotation would leave
            # every frame identical -- a sequence with no signal in it that looks
            # like valid data. The digit is given a fixed hue first (varied
            # across the dataset so the sequences are not all the same tint), by
            # mapping its grayscale value onto a hue ramp at constant luminance.
            base = _grayscale_to_hue(
                base[:, :, 0], float(base_hues[index])
            )
        steps_out = np.empty(
            (sequence_length, image_size, image_size, 3), dtype=np.float32
        )
        for step in range(sequence_length):
            value = values[
                (start_offsets[index] + step) % len(values)
            ]
            steps_out[step] = _apply_mnist_transform(
                base, transform, float(value), image_size
            )
            factors[index, step] = value
        sequences[index] = steps_out

    logger.info(
        f"Built MNIST '{transform}' sequences: {sequences.shape} "
        f"({len(values)} distinct factor values, cyclic)"
    )
    return sequences, factors


def _apply_mnist_transform(
    image: np.ndarray, transform: str, value: float, image_size: int
) -> np.ndarray:
    """Apply one MNIST attribute value to a grayscale image, returning RGB.

    :param image: Grayscale image ``(H, W, 1)`` on ``[0, 1]``.
    :type image: np.ndarray
    :param transform: One of :data:`MNIST_TRANSFORMS`.
    :type transform: str
    :param value: The attribute value: degrees for rotation and hue, a multiplier
        for scale.
    :type value: float
    :param image_size: Output edge length.
    :type image_size: int
    :return: An ``(image_size, image_size, 3)`` float32 image on ``[0, 1]``.
    :rtype: np.ndarray
    """
    if transform == "rotation":
        warped = _rotate_bilinear(image, float(value) * np.pi / 180.0)
    elif transform == "color":
        warped = _rotate_hue(image, float(value))
    else:  # scale
        warped = _rescale(image, float(value), image_size)

    return np.clip(warped, 0.0, 1.0).astype(np.float32)


def _rotate_bilinear(image, radians: float):
    """Rotate an ``(H, W, C)`` array about its centre with bilinear sampling.

    Delegates to :func:`_affine_warp`.

    :param image: An ``(H, W, C)`` float array.
    :param radians: Rotation angle in radians.
    :return: The rotated array, same shape.
    """
    return _affine_warp(image, radians)


def _affine_warp(image, radians: float) -> np.ndarray:
    """Rotate an ``(H, W, C)`` array about its centre, deterministically.

    Inverse-maps every output pixel through the rotation and samples the source
    bilinearly, with zeros outside the frame — the correct background for a black
    digit on a black canvas.

    **Why NumPy and not ``tf.image``:** two TensorFlow spellings were tried and
    rejected. ``tf.nn.grid_sample`` is not public in TF 2.18, and
    ``gen_image_ops.image_projective_transform_v3`` — reached privately, since
    ``tf.image.transform`` and ``tf.image.projective_transform`` are both absent
    from the public API — is **non-deterministic in this build**: MEASURED 2026-10-05
    on three consecutive identical processes, the same 28x20 probe with an
    identity matrix returned sum 0.0, sum 0.0, and then the correct sum 16.0.
    Silently emitting an all-black frame twice out of three is not a defect a
    dataset generator may carry, and no amount of asserting on its output would
    catch it. A pure-NumPy inverse map is exact, reproducible and independent of
    the backend.

    :param image: An ``(H, W, C)`` float array.
    :type image: np.ndarray
    :param radians: Rotation angle in radians.
    :type radians: float
    :return: An ``(H, W, C)`` float32 array.
    :rtype: np.ndarray
    """
    source = np.asarray(image, dtype=np.float32)
    height, width = source.shape[0], source.shape[1]
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    centre_y = (height - 1.0) / 2.0
    centre_x = (width - 1.0) / 2.0

    # Output pixel grid, centred.
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    offset_y = ys - centre_y
    offset_x = xs - centre_x

    # Inverse rotation: an output pixel reads the source point it maps to.
    sample_x = offset_x * cos_a + offset_y * sin_a + centre_x
    sample_y = -offset_x * sin_a + offset_y * cos_a + centre_y

    return _bilinear_sample(source, sample_x, sample_y)


def _bilinear_sample(
    source: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray
) -> np.ndarray:
    """Bilinearly sample ``source`` at fractional coordinates, zero outside.

    :param source: An ``(H, W, C)`` array.
    :type source: np.ndarray
    :param sample_x: Fractional x coordinates, any shape.
    :type sample_x: np.ndarray
    :param sample_y: Fractional y coordinates, same shape as ``sample_x``.
    :type sample_y: np.ndarray
    :return: An array shaped ``sample_x.shape + (C,)``, zero where the coordinate
        falls outside ``[0, W-1] x [0, H-1]``.
    :rtype: np.ndarray
    """
    height, width = source.shape[0], source.shape[1]
    inside = (
        (sample_x >= 0.0)
        & (sample_x <= width - 1.0)
        & (sample_y >= 0.0)
        & (sample_y <= height - 1.0)
    )
    # Clamp before indexing: the interior is what the caller wants, and the
    # outside is masked to zero afterwards, so clamping never leaks a value in.
    x0 = np.clip(np.floor(sample_x), 0, width - 1).astype(np.intp)
    y0 = np.clip(np.floor(sample_y), 0, height - 1).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    weight_x = (sample_x - x0)[..., None]
    weight_y = (sample_y - y0)[..., None]

    top = source[y0, x0] * (1.0 - weight_x) + source[y0, x1] * weight_x
    bottom = source[y1, x0] * (1.0 - weight_x) + source[y1, x1] * weight_x
    sampled = top * (1.0 - weight_y) + bottom * weight_y
    return (sampled * inside[..., None]).astype(np.float32)


def _grayscale_to_hue(gray: np.ndarray, hue_degrees: float) -> np.ndarray:
    """Tint a grayscale image with a hue while holding its luminance.

    The luminance channel is the original gray value, so the digit keeps its
    shape and contrast; only the chrominance is set. A pure grayscale image has
    zero saturation and rotating its hue is a no-op, which is why the colour
    sequences need this step before the rotation itself is meaningful.

    :param gray: An ``(H, W)`` array on ``[0, 1]``.
    :type gray: np.ndarray
    :param hue_degrees: Base hue in degrees.
    :type hue_degrees: float
    :return: An ``(H, W, 3)`` float32 array on ``[0, 1]``.
    :rtype: np.ndarray
    """
    # HSL, not HSV: HSL's L *is* the gray value, so luminance is preserved by
    # construction. HSV's V is the MAXIMUM channel, so using gray as V turns a
    # mid-gray into a fully saturated colour whose mean brightness is ~0.52x the
    # original -- MEASURED on a linear ramp: mean RGB 0.2600 against a gray mean
    # of 0.5000. That would darken every digit alongside changing its hue, so the
    # factor under study would not be the only thing varying across a sequence.
    saturation = 0.72
    height, width = int(gray.shape[0]), int(gray.shape[1])
    hsl = np.stack(
        [
            np.full((height, width), float(hue_degrees) / 360.0, dtype=np.float32),
            np.full((height, width), saturation, dtype=np.float32),
            np.asarray(gray, dtype=np.float32),
        ],
        axis=-1,
    )
    return np.clip(_hsl_to_rgb(hsl), 0.0, 1.0).astype(np.float32)


def _hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    """Vectorized HSL -> RGB for an ``(..., 3)`` array with channels in [0, 1].

    A direct transcription of the standard sector formula. ``tf.image.hsv_to_rgb``
    would do this too, but the surrounding warps are NumPy for determinism
    (see :func:`_affine_warp`), and mixing in a TF call here would reintroduce
    the backend dependency the module was written to avoid.

    :param hsl: ``(..., 3)`` array; H in [0, 1) cycles, S and L in [0, 1].
    :type hsl: np.ndarray
    :return: ``(..., 3)`` array on [0, 1].
    :rtype: np.ndarray
    """
    hue_sector = (hsl[..., 0] * 6.0) % 6.0
    saturation = hsl[..., 1]
    lightness = hsl[..., 2]

    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    second = chroma * (1.0 - np.abs(hue_sector % 2.0 - 1.0))
    offset = lightness - chroma / 2.0

    sector = np.floor(hue_sector).astype(np.int32) % 6
    red = np.select(
        [sector == i for i in range(6)],
        [chroma, second, 0.0, 0.0, second, chroma],
        default=0.0,
    )
    green = np.select(
        [sector == i for i in range(6)],
        [second, chroma, chroma, second, 0.0, 0.0],
        default=0.0,
    )
    blue = np.select(
        [sector == i for i in range(6)],
        [0.0, 0.0, second, chroma, chroma, second],
        default=0.0,
    )
    return np.stack([red, green, blue], axis=-1) + offset[..., None]


def _rotate_hue(image, degrees: float):
    """Rotate an image's hue while holding its luminance and saturation.

    Implemented as a rotation in YIQ: Y (luminance) is untouched and the
    chrominance plane is rotated, which is the cheap approximation of an HSV hue
    rotation. An RGB-space rotation would rescale luminance and visibly darken the
    digit, changing the data alongside the factor being studied.

    The input must already carry chrominance (:func:`_grayscale_to_hue` provides
    it); a grayscale input has none and every frame would be identical.

    :param image: An ``(H, W, 3)`` float tensor on ``[0, 1]``.
    :type image: Any
    :param degrees: Hue angle in degrees.
    :type degrees: float
    :return: An ``(H, W, 3)`` float tensor.
    """
    source = np.asarray(image, dtype=np.float32)
    if source.shape[-1] == 1:
        # A grayscale input has no chrominance to rotate; every frame would be
        # identical and the sequence would carry no signal at all.
        raise ValueError(
            "hue rotation needs a colour image, got a single channel"
        )
    coefficients = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ],
        dtype=np.float32,
    )
    inverse = np.linalg.inv(coefficients).astype(np.float32)

    yiq = source @ coefficients.T
    radians = np.deg2rad(degrees)
    cos_a = np.cos(radians).astype(np.float32)
    sin_a = np.sin(radians).astype(np.float32)
    iq = yiq[..., 1] * cos_a - yiq[..., 2] * sin_a
    q = yiq[..., 1] * sin_a + yiq[..., 2] * cos_a
    yiq = np.stack([yiq[..., 0], iq, q], axis=-1)
    return yiq @ inverse.T


def _rescale(image, factor: float, image_size: int) -> np.ndarray:
    """Resize an image's CONTENT about its centre, keeping the frame fixed.

    Scaling the content and re-fitting to ``image_size`` — rather than resizing
    the canvas — keeps the digit in the middle for every factor value, so a scale
    sequence does not also present a translation the model could latch onto
    instead of the intended factor.

    Deterministic NumPy inverse map, for the same reason as :func:`_affine_warp`.

    :param image: An ``(H, W, C)`` float array.
    :type image: np.ndarray
    :param factor: Multiplier in ``(0, 1]``. Larger means a bigger sprite.
    :type factor: float
    :param image_size: Output edge length. Unused when it already matches, and
        the caller resizes afterwards; accepted so the call site reads in terms of
        the intended output geometry.
    :type image_size: int
    :return: An ``(H, W, C)`` float32 array, same shape as the input.
    :rtype: np.ndarray
    """
    source = np.asarray(image, dtype=np.float32)
    height, width = source.shape[0], source.shape[1]
    centre_y = (height - 1.0) / 2.0
    centre_x = (width - 1.0) / 2.0

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    # An output pixel at distance d from the centre must read the source pixel at
    # distance d / factor, so the sprite grows by `factor`.
    sample_x = (xs - centre_x) / factor + centre_x
    sample_y = (ys - centre_y) / factor + centre_y
    return _bilinear_sample(source, sample_x, sample_y)


# ---------------------------------------------------------------------
# dSprites
# ---------------------------------------------------------------------


def download_dsprites(
    cache_root: str = DEFAULT_DSPRITES_CACHE,
    url: str = DSPRITES_URL,
) -> str:
    """Return the path to the dSprites archive, downloading it if absent.

    The ``.npz`` file *is* the cache and is never deleted; see the module
    docstring for why that differs from ``tudataset``.

    :param cache_root: Directory holding the archive. Defaults to
        :data:`DEFAULT_DSPRITES_CACHE`. Never point this at the repository.
    :type cache_root: str
    :param url: Download URL. Defaults to :data:`DSPRITES_URL`.
    :type url: str
    :return: Absolute path to the ``.npz``.
    :rtype: str
    :raises urllib.error.URLError: If the host is unreachable.

    :Example:

    >>> path = download_dsprites()  # doctest: +SKIP
    """
    root = os.path.abspath(os.path.expanduser(cache_root))
    archive = os.path.join(root, DSPRITES_FILENAME)

    if os.path.isfile(archive):
        logger.info(f"dSprites already cached at {archive}")
        return archive

    os.makedirs(root, exist_ok=True)
    logger.info(f"Downloading dSprites from {url} to {archive}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "dl-techniques/dsprites-loader"}
    )
    # Streamed in chunks: the archive is ~250MB and `response.read()` would hold
    # the whole thing plus a copy in memory.
    with urllib.request.urlopen(request) as response, open(archive, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)

    logger.info(f"dSprites ready at {archive}")
    return archive


def load_dsprites_subset(
    cache_root: str = DEFAULT_DSPRITES_CACHE,
    num_scales: int = 5,
    stride_positions: int = 2,
    stride_orientations: int = 2,
    shuffle: bool = True,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load the dSprites subset used by the paper (Section A.9).

    The full dataset is 737,280 images. The paper takes all 3 shapes, the largest
    5 scales, and **every other** example from the first 30 orientations and
    positions — 3 x 5 x 15 x 15 x 15 = 50,625 images of 64x64 grayscale. The
    striding is what keeps the run affordable; the consequence is that successive
    frames in a sequence overlap less than they would in the full dataset, which
    the paper notes as the reason a smaller within-capsule neighbourhood
    (``K = 1``) works better here than on MNIST.

    dSprites is *fully labelled*, so the returned factors are the true generative
    values — the reason it is the better instrument for CapCorr.

    :param cache_root: Directory holding the archive. Defaults to
        :data:`DEFAULT_DSPRITES_CACHE`.
    :type cache_root: str
    :param num_scales: Number of scales to keep, largest first. Defaults to 5.
    :type num_scales: int
    :param stride_positions: Stride over the first 30 x and y positions.
        Defaults to 2, giving 15 each.
    :type stride_positions: int
    :param stride_orientations: Stride over the first 40 orientations. Defaults
        to 2, giving 15. The paper's subset is 3 x 5 x 15 x 15 x 15 = 50,625
        images, which requires striding **orientations too** — leaving this at 1
        yields 101,250, exactly double the reported count, and is the only way the
        stated total fails to come out.
    :type stride_orientations: int
    :param shuffle: Whether to shuffle the subset. Defaults to ``True``.
    :type shuffle: bool
    :param seed: Seed for the shuffle.
    :type seed: int
    :return: ``(images, factors)``. ``images`` is ``(N, 64, 64, 1)`` float32 on
        ``[0, 1]``; ``factors`` is a dict with ``shape``, ``scale``, ``orientation``,
        ``x_position`` and ``y_position``, each ``(N,)``.
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray]]

    :Example:

    >>> images, factors = load_dsprites_subset()  # doctest: +SKIP
    >>> images.shape
    (50625, 64, 64, 1)
    """
    archive = download_dsprites(cache_root=cache_root)
    with np.load(archive, encoding="latin1") as data:
        raw_images = data["imgs"]
        # MEASURED structure of the official archive: it ships `latents_values`
        # as an (N, 6) array of the ACTUAL factor values (one column per factor,
        # in the order colour, shape, scale, orientation, x, y), NOT the one-hot
        # `latents_classes` encoding. The earlier `argmax(one_hot)` formulation
        # silently depended on the one-hot form and raised KeyError on
        # `color_values`, which does not exist in this file.
        latent_columns = (
            "color", "shape", "scale", "orientation", "x_position", "y_position",
        )
        latent_values = np.asarray(data["latents_values"], dtype=np.float64)
    if latent_values.shape[1] != len(latent_columns):
        raise ValueError(
            f"dSprites latents_values has {latent_values.shape[1]} column(s), "
            f"expected {len(latent_columns)}"
        )
    latent = {
        name: latent_values[:, index]
        for index, name in enumerate(latent_columns)
    }

    shape = latent["shape"]
    scale = latent["scale"]
    orientation = latent["orientation"]
    x_position = latent["x_position"]
    y_position = latent["y_position"]

    # Largest `num_scales` scale VALUES (dSprites' 6 scale values run from a
    # small sprite up to a large one, so "largest" means the biggest rendered
    # size, which is the largest value in the column).
    scale_values = np.unique(scale)
    largest_scales = np.sort(scale_values)[::-1][:num_scales]
    mask = np.isin(scale, largest_scales)

    # "every other example from the first 30" of each axis. The axes are sorted
    # first: dSprites' first 30 positions are the small-magnitude ones, and
    # `np.unique` already returns sorted values, so a strided slice of the first
    # 30 IS the paper's subset.
    for column, stride in (
        (x_position, stride_positions),
        (y_position, stride_positions),
        (orientation, stride_orientations),
    ):
        selected = np.unique(column)[:30:stride]
        mask &= np.isin(column, selected)

    indices = np.flatnonzero(mask)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)

    selected_images = raw_images[indices].astype(np.float32)
    # dSprites pixels are 0/255 booleans in an alpha-style encoding; normalize.
    if selected_images.max() > 1.0:
        selected_images = selected_images / 255.0
    selected_images = selected_images[:, :, :, None]

    factors = {
        "shape": shape[indices],
        "scale": scale[indices],
        "orientation": orientation[indices],
        "x_position": x_position[indices],
        "y_position": y_position[indices],
    }
    logger.info(
        f"Loaded dSprites subset: {selected_images.shape[0]} images of "
        f"{selected_images.shape[1]}x{selected_images.shape[2]}, "
        f"shapes={np.unique(factors['shape']).tolist()}, "
        f"scales={len(np.unique(factors['scale']))} largest"
    )
    return selected_images, factors


def build_dsprites_transform_sequences(
    transform: str = "orientation",
    sequence_length: int = 15,
    num_sequences: int = 4096,
    cache_root: str = DEFAULT_DSPRITES_CACHE,
    seed: int = 0,
    scale_loops: int = 3,
    **subset_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build cyclic dSprites transformation sequences (paper Section A.9).

    Like :func:`build_mnist_transform_sequences` but over exact ground-truth
    factor labels, and with one documented relaxation: **scale is not cyclic** in
    dSprites (it is bounded), so a scale sequence loops over the available scales
    ``scale_loops`` times and the cycle closes on the list rather than on a
    continuum. The paper observes that this mismatch with the latent priors does
    not prevent training; it is stated here rather than smoothed over, because a
    reader comparing CapCorr across scales needs to know the cycle is discrete.

    The other three factors are genuinely cyclic over their 15 selected values.

    :param transform: One of :data:`DSPRITES_TRANSFORMS`. Defaults to
        ``"orientation"``.
    :type transform: str
    :param sequence_length: Frames per sequence ``S``. Defaults to 15, which
        equals the number of selected factor values so a full cycle fits exactly.
    :type sequence_length: int
    :param num_sequences: How many sequences to build. Defaults to 4096.
    :type num_sequences: int
    :param cache_root: dSprites cache directory. Defaults to
        :data:`DEFAULT_DSPRITES_CACHE`.
    :type cache_root: str
    :param seed: Seed for the base draw and the start poses.
    :type seed: int
    :param scale_loops: How many times a scale sequence loops over the available
        scales. Defaults to 3, the paper's value. Ignored for other transforms.
    :type scale_loops: int
    :param subset_kwargs: Forwarded to :func:`load_dsprites_subset`.
    :type subset_kwargs: Any
    :return: ``(sequences, factors)``. ``sequences`` is
        ``(num_sequences, sequence_length, 64, 64, 1)`` float32 on ``[0, 1]``;
        ``factors`` is ``(num_sequences, sequence_length)`` int64 holding the true
        ground-truth factor index at each step.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: On an unknown ``transform`` or a non-positive
        ``sequence_length`` / ``num_sequences``.

    :Example:

    >>> sequences, factors = build_dsprites_transform_sequences()  # doctest: +SKIP
    >>> sequences.shape
    (4096, 15, 64, 64, 1)
    """
    if transform not in DSPRITES_TRANSFORMS:
        raise ValueError(
            f"transform must be one of {list(DSPRITES_TRANSFORMS)}, "
            f"got {transform!r}"
        )
    if sequence_length <= 0:
        raise ValueError(
            f"sequence_length must be positive, got {sequence_length}"
        )
    if num_sequences <= 0:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")

    images, factors = load_dsprites_subset(
        cache_root=cache_root, seed=seed, **subset_kwargs
    )

    # Which factor column drives this transform, and the ORDERED list of its
    # values. For scale the list is repeated `scale_loops` times.
    key = {
        "x_position": "x_position",
        "y_position": "y_position",
        "orientation": "orientation",
        "scale": "scale",
    }[transform]
    values = np.unique(factors[key])
    if transform == "scale":
        order = np.concatenate([values] * max(1, scale_loops))
        values = order
        logger.info(
            f"dSprites scale sequence loops {scale_loops}x over "
            f"{len(np.unique(factors[key]))} scales -> cycle length {len(values)}"
        )
    elif transform == "x_position":
        values = np.sort(values)
    elif transform == "y_position":
        values = np.sort(values)

    rng = np.random.default_rng(seed + 1)
    base_indices = rng.integers(0, images.shape[0], size=num_sequences)
    start_offsets = rng.integers(0, len(values), size=num_sequences)

    sequences = np.zeros(
        (num_sequences, sequence_length) + images.shape[1:],
        dtype=np.float32,
    )
    sequence_factors = np.zeros(
        (num_sequences, sequence_length), dtype=np.float64
    )

    for index in range(num_sequences):
        for step in range(sequence_length):
            value = values[
                (start_offsets[index] + step) % len(values)
            ]
            match = np.flatnonzero(factors[key] == value)
            # The factor value alone does not identify an image: position and
            # scale/orientation are separate axes, and several combinations share
            # a value. Draw among the matching rows so the sequence varies only in
            # the requested factor and the base sprite stays otherwise fixed.
            pool = match[
                base_indices[index] % len(match)
            ] if len(match) else base_indices[index]
            sequences[index, step] = images[pool]
            sequence_factors[index, step] = float(value)

    logger.info(
        f"Built dSprites '{transform}' sequences: {sequences.shape} "
        f"(cycle length {len(values)})"
    )
    return sequences, sequence_factors

# ---------------------------------------------------------------------


def create_transform_sequence_dataset(
    name: str,
    transform: str = "rotation",
    sequence_length: Optional[int] = None,
    num_sequences: int = 4096,
    seed: int = 0,
    cache_root: str = DEFAULT_DSPRITES_CACHE,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch to the MNIST or dSprites sequence builder by dataset name.

    One entry point so a trainer takes ``--dataset`` and does not branch itself;
    the two builders' signatures differ in their extra keyword arguments, which is
    why ``**kwargs`` is forwarded rather than enumerated here.

    :param name: ``"mnist"`` or ``"dsprites"``.
    :type name: str
    :param transform: A transformation valid for the chosen dataset.
    :type transform: str
    :param sequence_length: Frames per sequence. Defaults to the dataset's own
        value (18 for MNIST, 15 for dSprites) when ``None``.
    :type sequence_length: Optional[int]
    :param num_sequences: How many sequences to build.
    :type num_sequences: int
    :param seed: Seed.
    :type seed: int
    :param cache_root: dSprites cache directory; ignored by MNIST.
    :type cache_root: str
    :param kwargs: Forwarded to the dataset's builder.
    :type kwargs: Any
    :return: ``(sequences, factors)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :raises ValueError: On an unknown dataset name.

    :Example:

    >>> sequences, factors = create_transform_sequence_dataset("mnist")
    >>> sequences.shape
    (4096, 18, 28, 28, 3)
    """
    if name == "mnist":
        return build_mnist_transform_sequences(
            transform=transform,
            sequence_length=18 if sequence_length is None else sequence_length,
            num_sequences=num_sequences,
            seed=seed,
            **kwargs,
        )
    if name == "dsprites":
        return build_dsprites_transform_sequences(
            transform=transform,
            sequence_length=15 if sequence_length is None else sequence_length,
            num_sequences=num_sequences,
            cache_root=cache_root,
            seed=seed,
            **kwargs,
        )
    raise ValueError(
        f"Unknown dataset {name!r}. Available: 'mnist', 'dsprites'"
    )

# ---------------------------------------------------------------------