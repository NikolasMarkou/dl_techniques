"""DTSPrompt generators for DocRes — the classical-CV prompt channels.

DocRes conditions a single Restormer backbone on a task by *concatenating three
extra input channels* onto the RGB image; the network itself has no
task-conditioning at all. Those three channels are what this module produces.

Upstream (``DocRes/inference.py:19-94``, ``DocRes/utils.py:84-140``,
``DocRes/loaders/docres_loader.py:201-227,353-371``) writes every generator in
OpenCV plus a ``skimage`` Sauvola, and duplicates the same recipe three times.
This port makes two deliberate departures:

**1. numpy + scipy only — no ``cv2``, no ``skimage``.**
``opencv-python`` and ``scikit-image`` are NOT declared in ``pyproject.toml``,
in the core dependency list or in any extra; ``scipy`` is core. Both happen to
be importable in this repository's ``.venv`` (measured: cv2 4.13.0, skimage
0.26.0), which is exactly the accident that lets the four existing ``cv2``
importers in this tree pass CI while being undeclared. Importing them here
would make this module work on this machine and fail on a clean install of the
declared dependency set. **Do not "simplify" this file back to OpenCV.** The
tests are free to import cv2/skimage — they use them as the parity oracle —
and a suite guard asserts that the shipped package does not.

The numpy/scipy equivalences below were measured against OpenCV 4.13.0 /
scikit-image 0.26.0 rather than assumed (see the module's test suite, which
re-derives every one of them):

===========================  ==================================  ==============
upstream call                replacement                          agreement
===========================  ==================================  ==============
``cv2.dilate(p, ones(7,7))`` ``ndimage.grey_dilation``            bit-exact
``cv2.medianBlur(p, 21)``    ``ndimage.median_filter(21)``        bit-exact
``cv2.Sobel(CV_16S,...)``    ``ndimage.correlate`` + |.|          bit-exact
``cv2.cvtColor(BGR2GRAY)``   15-bit fixed point (see below)       bit-exact
``cv2.normalize(MINMAX)``    affine rescale to [0, 255]           bit-exact
``skimage.threshold_sauvola``integral-image local mean/std        bit-exact
``cv2.resize(INTER_LINEAR)`` :func:`_resize_bilinear`             max|d| = 1
===========================  ==================================  ==============

The single approximation is the bilinear resize: OpenCV evaluates
``INTER_LINEAR`` for 8-bit input in 11-bit fixed point, this module evaluates
it in float64 and rounds once. Same half-pixel sampling convention, same
replicate border, so the two differ only by OpenCV's intermediate
quantisation — measured max|difference| of exactly 1 grey level on ~12% of
pixels, 0 on the rest. It is a documented approximation, not a silent one.
``scipy.ndimage.zoom(order=1, grid_mode=True, mode='nearest')`` was measured to
produce *bit-identical* output to :func:`_resize_bilinear`, so nothing is lost
by preferring the explicit implementation, which takes an exact output shape
instead of a zoom factor.

**2. Channel order is RGB, not BGR.**
Upstream reads images with ``cv2.imread`` and therefore works in BGR. This
library is RGB throughout. The per-channel background pipeline is
order-agnostic, and the greyscale conversion uses the RGB-ordered weights, so
this module applied to an RGB array computes exactly what upstream computes
when applied to the same image in BGR.

**Cost note.** ``median_filter(size=21)`` over the 1024x1024 working resolution
costs ~2.7 s per channel (~8 s per call) against OpenCV's ~0.1 s. That is the
price of the dependency decision above. It is paid once per image by the
offline sidecar-precompute step, never inside a training loop.

Public surface:
    * :func:`estimate_background` — the ONE shared dilate/median/absdiff/
      normalise primitive that upstream duplicates three times.
    * :func:`deshadow_prompt`, :func:`appearance_prompt`, :func:`deblur_prompt`,
      :func:`binarization_prompt`, :func:`dewarp_prompt` — the five generators.
    * :func:`sauvola_mod_binarization` — the two-pass Sauvola used by
      :func:`binarization_prompt`.
    * :func:`base_coordinate_grid`, :func:`apply_document_mask` — dewarping
      helpers.
"""

from typing import Tuple

import numpy as np
from scipy import ndimage

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Upstream constants. Every one of these is an unexplained magic number in
# DocRes -- no comment, no paper cross-reference. They are reproduced exactly.
# ---------------------------------------------------------------------------

BACKGROUND_WORKING_SIZE: int = 1024
"""Square resolution the background estimate is computed at (``inference.py:29``)."""

DILATE_KERNEL_SIZE: int = 7
"""Side of the square dilation structuring element (``inference.py:34``)."""

MEDIAN_KERNEL_SIZE: int = 21
"""Side of the median-blur window (``inference.py:35``)."""

SAUVOLA_BINARY_THRESHOLD: int = 155
"""Re-threshold applied to the Sauvola binary map (``inference.py:85-86``)."""

SAUVOLA_N1_FRACTION: float = 0.05
"""First-pass window as a fraction of ``min(H, W)`` (``utils.py:108``)."""

SAUVOLA_N2_FRACTION: float = 0.10
"""Second-pass window as a fraction of ``min(H, W)`` (``utils.py:111``)."""

SAUVOLA_K1: float = 0.5
"""First-pass Sauvola ``k`` (``utils.py:114``)."""

SAUVOLA_K2: float = 0.5
"""Second-pass Sauvola ``k`` (``utils.py:115``)."""

_UINT8_SAUVOLA_R: float = 127.5
"""``skimage``'s dynamic-range normaliser for uint8: ``0.5 * (255 - 0)``."""

# DECISION plan-2026-09-08T111844-de235227/D-020
# Do NOT "simplify" this to round(0.299*R + 0.587*G + 0.114*B), and do NOT
# substitute the 14-bit triple (4899, 9617, 1868) that most references quote
# for OpenCV. Both were MEASURED wrong on ~0.2% of pixels by one grey level
# against OpenCV 4.13.0; only the triple below is exact. The error is small
# but it feeds the Sauvola binarization, where a one-level shift can flip a
# pixel's class. See D-020 in decisions.md.
# cv2's COLOR_BGR2GRAY for 8-bit input is integer arithmetic, not float. The
# exact coefficients and shift were found by exhaustive search against OpenCV
# 4.13.0 over 90k random pixels: this triple at shift 15 reproduces cvtColor
# bit-for-bit (100.0000%), while the widely-quoted 14-bit triple
# (1868, 9617, 4899) and the float form round(0.114*B + 0.587*G + 0.299*R) are
# each wrong on ~0.2% of pixels by one grey level. Weights are listed here in
# R, G, B order because this module is RGB.
_GRAY_R: int = 9798
_GRAY_G: int = 19235
_GRAY_B: int = 3735
_GRAY_SHIFT: int = 15

_SOBEL_KX: np.ndarray = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64
)
"""The ksize=3 Sobel x-derivative kernel ``cv2.getDerivKernels`` produces."""


# ---------------------------------------------------------------------------
# Private primitives -- each one a measured stand-in for a specific cv2 call.
# ---------------------------------------------------------------------------


def _check_rgb_uint8(img: np.ndarray, name: str = "img") -> np.ndarray:
    """Validate a page image and return it unchanged.

    Args:
        img: Candidate array.
        name: Parameter name, used in the error message.

    Returns:
        ``img`` itself (no copy, no cast).

    Raises:
        ValueError: If ``img`` is not a ``(H, W, 3)`` uint8 array with both
            spatial extents non-zero.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(img)!r}")
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape (H, W, 3), got {img.shape}. DTSPrompt "
            f"generators take a 3-channel RGB page image."
        )
    if img.dtype != np.uint8:
        raise ValueError(
            f"{name} must be uint8 in [0, 255], got dtype {img.dtype}. The "
            f"upstream pipeline is uint8 end to end; casting here would hide "
            f"a caller that already divided by 255."
        )
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError(f"{name} has a zero spatial extent: {img.shape}")
    return img


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resample with OpenCV's ``INTER_LINEAR`` conventions.

    Reproduces ``cv2.resize(img, (out_w, out_h))``: half-pixel sample centres
    (``src = (dst + 0.5) * scale - 0.5``) and a replicated border. Arithmetic
    is float64 and the single rounding happens at the uint8 cast, where OpenCV
    instead works in 11-bit fixed point -- the sole reason the two can differ,
    and then by at most one grey level.

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` uint8 array.
        out_h: Target height in pixels.
        out_w: Target width in pixels.

    Returns:
        uint8 array of shape ``(out_h, out_w)`` or ``(out_h, out_w, C)``.

    Raises:
        ValueError: If either target extent is not a positive integer.
    """
    # DECISION plan-2026-09-08T111844-de235227/D-019
    # Do NOT replace this with scipy.ndimage.zoom "for brevity", and do NOT
    # reach for cv2.resize "for exactness". zoom(order=1) DEFAULTS to
    # grid_mode=False -- the align-corners convention, which is NOT what
    # OpenCV does and would shift every sample; only zoom(order=1,
    # grid_mode=True, mode='nearest') matches, and it was MEASURED to produce
    # output bit-identical to this function while taking a float zoom factor
    # instead of an exact output shape (so a 1024 target can come back 1023).
    # cv2 is not an option at all: it is undeclared in pyproject.toml (D-002).
    # The residual vs OpenCV is one grey level on ~12% of pixels, caused only
    # by OpenCV's 11-bit fixed-point intermediate. See D-019 in decisions.md.
    if out_h < 1 or out_w < 1:
        raise ValueError(f"resize target must be positive, got {(out_h, out_w)}")
    in_h, in_w = img.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return img.copy()

    scale_y = in_h / out_h
    scale_x = in_w / out_w
    y = (np.arange(out_h, dtype=np.float64) + 0.5) * scale_y - 0.5
    x = (np.arange(out_w, dtype=np.float64) + 0.5) * scale_x - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = (y - y0)[:, None]
    fx = (x - x0)[None, :]
    y0c = np.clip(y0, 0, in_h - 1)
    y1c = np.clip(y0 + 1, 0, in_h - 1)
    x0c = np.clip(x0, 0, in_w - 1)
    x1c = np.clip(x0 + 1, 0, in_w - 1)

    src = img.astype(np.float64)
    if src.ndim == 3:
        fy = fy[..., None]
        fx = fx[..., None]
    top = src[y0c][:, x0c] * (1.0 - fx) + src[y0c][:, x1c] * fx
    bottom = src[y1c][:, x0c] * (1.0 - fx) + src[y1c][:, x1c] * fx
    out = top * (1.0 - fy) + bottom * fy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Luma conversion bit-identical to ``cv2.cvtColor(..., BGR2GRAY)``.

    Args:
        img: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        ``(H, W)`` uint8 greyscale array.
    """
    r = img[..., 0].astype(np.int64)
    g = img[..., 1].astype(np.int64)
    b = img[..., 2].astype(np.int64)
    acc = r * _GRAY_R + g * _GRAY_G + b * _GRAY_B + (1 << (_GRAY_SHIFT - 1))
    return (acc >> _GRAY_SHIFT).astype(np.uint8)


def _normalize_minmax_uint8(plane: np.ndarray) -> np.ndarray:
    """Affine rescale to ``[0, 255]``, matching ``cv2.normalize(NORM_MINMAX)``.

    OpenCV guards the degenerate case by setting the scale to zero rather than
    dividing by zero, which maps a constant plane to all-zeros. That behaviour
    is reproduced here: a blank scan must not produce NaN.

    Args:
        plane: ``(H, W)`` uint8 array.

    Returns:
        ``(H, W)`` uint8 array spanning ``[0, 255]``, or all-zeros if ``plane``
        is constant.
    """
    lo = int(plane.min())
    hi = int(plane.max())
    if hi == lo:
        return np.zeros_like(plane)
    scaled = (plane.astype(np.float64) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _sobel_magnitude_gray(img: np.ndarray) -> np.ndarray:
    """The shared ``0.5*|Sobel_x| + 0.5*|Sobel_y|`` greyscale gradient map.

    Reproduces upstream's ``cv2.Sobel(CV_16S) -> convertScaleAbs ->
    addWeighted(0.5, 0.5) -> BGR2GRAY`` chain, including the saturating uint8
    cast after each ``|.|`` (which is why the two absolute values are clipped
    to 255 *before* being averaged, not after). Border handling is OpenCV's
    default ``BORDER_REFLECT_101``, i.e. scipy's ``'mirror'``.

    Args:
        img: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        ``(H, W)`` uint8 gradient-magnitude map.
    """
    src = img.astype(np.float64)
    gx = np.stack(
        [ndimage.correlate(src[..., c], _SOBEL_KX, mode="mirror") for c in range(3)],
        axis=-1,
    )
    gy = np.stack(
        [ndimage.correlate(src[..., c], _SOBEL_KX.T, mode="mirror") for c in range(3)],
        axis=-1,
    )
    abs_x = np.clip(np.rint(np.abs(gx)), 0, 255)
    abs_y = np.clip(np.rint(np.abs(gy)), 0, 255)
    blended = np.clip(np.rint(0.5 * abs_x + 0.5 * abs_y), 0, 255).astype(np.uint8)
    return _rgb_to_gray(blended)


def _local_mean_std(image: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed mean and (population) standard deviation via integral images.

    Reproduces ``skimage.filters.thresholding._mean_std``. skimage pads
    asymmetrically by ``(w//2 + 1, w//2)`` and then shifts the box by one; a
    symmetric ``w//2`` reflect pad with no shift is the identical window, and
    is what this function does. ``mode='reflect'`` here is numpy's, which is
    skimage's -- the edge sample is not repeated.

    Args:
        image: 2-D array.
        window_size: Odd window side.

    Returns:
        ``(mean, std)``, both float64 and the same shape as ``image``. The
        variance is clipped at zero before the square root, so float error
        cannot produce NaN on a constant region.
    """
    src = image.astype(np.float64)
    half = window_size // 2
    padded = np.pad(src, half, mode="reflect")
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)))
    integral_sq = np.pad(
        np.cumsum(np.cumsum(padded * padded, axis=0), axis=1), ((1, 0), (1, 0))
    )
    h, w = src.shape
    ws = window_size

    def _box(acc: np.ndarray) -> np.ndarray:
        return (
            acc[ws : ws + h, ws : ws + w]
            - acc[0:h, ws : ws + w]
            - acc[ws : ws + h, 0:w]
            + acc[0:h, 0:w]
        ) / float(ws * ws)

    mean = _box(integral)
    mean_sq = _box(integral_sq)
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))
    return mean, std


def _threshold_sauvola(gray: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Sauvola's local threshold, matching ``skimage.filters.threshold_sauvola``.

    ``T = m * (1 + k * (s / R - 1))`` with ``R = 127.5``, the half dynamic
    range skimage derives from the uint8 dtype limits.

    Args:
        gray: ``(H, W)`` uint8 array.
        window_size: Odd window side.
        k: Sauvola's ``k``.

    Returns:
        ``(H, W)`` float64 threshold map.
    """
    mean, std = _local_mean_std(gray, window_size)
    return mean * (1.0 + k * ((std / _UINT8_SAUVOLA_R) - 1.0))


def _odd_window(fraction: float, min_extent: int) -> int:
    """Upstream's window rule: ``int(fraction * min(H, W))``, forced odd, min 1.

    ``utils.py:108-113`` adds one to an even result, so a fraction that
    truncates to 0 becomes 1 rather than an illegal window of 0.

    Args:
        fraction: 0.05 or 0.10.
        min_extent: ``min(H, W)`` of the page.

    Returns:
        An odd window side of at least 1.
    """
    n = int(fraction * min_extent)
    if n % 2 == 0:
        n = n + 1
    return n


# ---------------------------------------------------------------------------
# The ONE shared background-estimation primitive.
# ---------------------------------------------------------------------------


def estimate_background(
    img: np.ndarray, working_size: int = BACKGROUND_WORKING_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a page's illumination background and its normalised image.

    This is the recipe DocRes carries in three byte-identical copies
    (``inference.py:26-52`` and ``:64-80``, ``eval.py:20-95``,
    ``loaders/docres_loader.py:212-227,353-371``). The copies differ in exactly
    one respect: which intermediate they keep. Both are returned here, and the
    two generators that need them select. Do not re-derive either one.

    The pipeline, per colour channel, at ``working_size`` square::

        dilate(7x7) -> median(21)                       -> bg_img
        255 - |plane - bg_img| -> rescale to [0, 255]   -> norm_img

    Both outputs are resized back to the input's resolution before returning,
    so the working resolution is invisible to callers.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.
        working_size: Square resolution the estimate is computed at. Defaults
            to upstream's 1024. Exposed only so tests can run cheaply; changing
            it changes the output, because the 7x7 and 21x21 kernels are
            absolute pixel sizes at this resolution and therefore cover a
            different fraction of the page.

    Returns:
        Tuple ``(bg_img, norm_img)``, both ``(H, W, 3)`` uint8:

        * ``bg_img`` — the smooth background/illumination estimate. This is
          the deshadowing prompt.
        * ``norm_img`` — the background-normalised page. This is the
          appearance prompt.

    Raises:
        ValueError: If ``img`` is not a ``(H, W, 3)`` uint8 array, or
            ``working_size`` is not positive.
    """
    _check_rgb_uint8(img)
    if working_size < 1:
        raise ValueError(f"working_size must be positive, got {working_size}")

    h, w = img.shape[:2]
    work = _resize_bilinear(img, working_size, working_size)

    bg_planes = []
    norm_planes = []
    for c in range(3):
        plane = work[..., c]
        dilated = ndimage.grey_dilation(
            plane, size=(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), mode="nearest"
        )
        bg_plane = ndimage.median_filter(dilated, size=MEDIAN_KERNEL_SIZE, mode="nearest")
        bg_planes.append(bg_plane)
        # 255 - absdiff, in uint8 arithmetic without wraparound.
        diff = 255 - np.abs(plane.astype(np.int16) - bg_plane.astype(np.int16))
        norm_planes.append(_normalize_minmax_uint8(diff.astype(np.uint8)))

    bg_img = _resize_bilinear(np.stack(bg_planes, axis=-1), h, w)
    norm_img = _resize_bilinear(np.stack(norm_planes, axis=-1), h, w)
    return bg_img, norm_img


# ---------------------------------------------------------------------------
# The five prompt generators.
# ---------------------------------------------------------------------------


def deshadow_prompt(
    img: np.ndarray, working_size: int = BACKGROUND_WORKING_SIZE
) -> np.ndarray:
    """Deshadowing prompt: the 3-channel background estimate.

    Upstream ``inference.py:26-52`` computes a ``shadow_map`` after the
    background and then **never returns it** (``return bg_imgs`` at line 52
    discards it) -- that is dead code and is deliberately not ported.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.
        working_size: See :func:`estimate_background`.

    Returns:
        ``(H, W, 3)`` uint8 array in ``[0, 255]``.
    """
    bg_img, _ = estimate_background(img, working_size=working_size)
    return bg_img


def appearance_prompt(
    img: np.ndarray, working_size: int = BACKGROUND_WORKING_SIZE
) -> np.ndarray:
    """Appearance-enhancement prompt: the 3-channel background-normalised page.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.
        working_size: See :func:`estimate_background`.

    Returns:
        ``(H, W, 3)`` uint8 array in ``[0, 255]``.
    """
    _, norm_img = estimate_background(img, working_size=working_size)
    return norm_img


def deblur_prompt(img: np.ndarray) -> np.ndarray:
    """Deblurring prompt: the Sobel gradient magnitude, replicated to 3 channels.

    ``0.5 * |Sobel_x| + 0.5 * |Sobel_y|``, converted to greyscale and then
    stacked back to three identical channels -- upstream really does round-trip
    through ``COLOR_BGR2GRAY`` / ``COLOR_GRAY2BGR`` (``inference.py:60-61``),
    so the three prompt channels carry no independent information.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.

    Returns:
        ``(H, W, 3)`` uint8 array in ``[0, 255]``, all three channels equal.
    """
    _check_rgb_uint8(img)
    gray = _sobel_magnitude_gray(img)
    return np.repeat(gray[..., None], 3, axis=-1)


def sauvola_mod_binarization(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The two-pass modified Sauvola binarization (``utils.py:84-133``).

    Pass 1 thresholds the greyscale page at window ``n1`` to build a contrast
    map ``C``; pass 2 thresholds ``C`` at the wider window ``n2`` to produce
    the binary image. Both windows track ``min(H, W)``, so the algorithm is
    scale-relative rather than fixed-pixel.

    The float threshold map is cast to uint8 by **truncation**, matching
    upstream's ``thresh.astype(np.uint8)`` (``inference.py:84``). The cast is
    safe because ``s <= R`` for uint8 input forces ``T <= m <= 255``.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image, or a ``(H, W)`` uint8
            greyscale page.

    Returns:
        Tuple ``(binary, threshold)``:

        * ``binary`` — ``(H, W)`` uint8, values in ``{0, 255}``.
        * ``threshold`` — ``(H, W)`` uint8, the second-pass threshold map.

    Raises:
        ValueError: If ``img`` is neither ``(H, W, 3)`` nor ``(H, W)`` uint8.
    """
    if isinstance(img, np.ndarray) and img.ndim == 2 and img.dtype == np.uint8:
        gray = np.copy(img)
    else:
        _check_rgb_uint8(img)
        gray = _rgb_to_gray(img)

    min_extent = min(gray.shape[0], gray.shape[1])
    n1 = _odd_window(SAUVOLA_N1_FRACTION, min_extent)
    n2 = _odd_window(SAUVOLA_N2_FRACTION, min_extent)

    t1 = _threshold_sauvola(gray, window_size=n1, k=SAUVOLA_K1)
    max_val = float(np.amax(gray))
    contrast = np.zeros(gray.shape, dtype=np.float32)
    above = gray > t1
    # Safe: gray[above] <= max_val and gray[above] > t1[above], so the
    # denominator is strictly positive wherever it is evaluated.
    contrast[above] = (
        (gray[above] - t1[above]) / (max_val - t1[above])
    ).astype(np.float32)
    contrast = contrast * 255.0
    second_input = contrast.astype(np.uint8)

    t2 = _threshold_sauvola(second_input, window_size=n2, k=SAUVOLA_K2)
    binary = np.copy(gray)
    binary[second_input <= t2] = 0
    binary[second_input > t2] = 255
    return binary, t2.astype(np.uint8)


def binarization_prompt(img: np.ndarray) -> np.ndarray:
    """Binarization prompt: three DIFFERENT single-channel maps, stacked.

    Unlike :func:`deblur_prompt`, the three channels here are not a replicated
    map -- they are ``[Sauvola threshold, Sobel gradient, Sauvola binary]``
    (``inference.py:82-94``), in that order.

    The ``> 155`` re-threshold upstream applies to the binary map is a no-op
    (the map is already ``{0, 255}``); it is reproduced anyway so the two
    implementations cannot diverge if upstream's Sauvola ever stops saturating.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.

    Returns:
        ``(H, W, 3)`` uint8 array in ``[0, 255]``. Channel 0 is the threshold
        map, channel 1 the gradient magnitude, channel 2 the ``{0, 255}``
        binary.
    """
    _check_rgb_uint8(img)
    binary, threshold = sauvola_mod_binarization(img)
    binary = np.where(binary > SAUVOLA_BINARY_THRESHOLD, 255, 0).astype(np.uint8)
    gradient = _sobel_magnitude_gray(img)
    return np.stack([threshold, gradient, binary], axis=-1)


def base_coordinate_grid(height: int, width: int) -> np.ndarray:
    """The normalised base-coordinate grid (``utils.getBasecoord``, normalised).

    ``getBasecoord`` builds a row-index plane and a column-index plane and
    concatenates them **column-plane first** (``utils.py:139`` puts
    ``base_coord1``, the ``arange(w)`` tiled down the rows, at channel 0). So
    channel 0 is **x** and varies along the width; channel 1 is **y** and
    varies down the height. Swapping them is a silent, plausible-looking
    defect -- the array keeps its shape, its dtype and its value range, and the
    dewarping remap simply transposes the page.

    Upstream only ever calls this at 256x256 and divides by the single scalar
    256, which leaves the non-square normalisation unspecified. This function
    normalises each axis by its own extent (x by ``width``, y by ``height``),
    which agrees with upstream exactly on every square grid and is the only
    choice that keeps both channels in ``[0, 1)``.

    Args:
        height: Grid height in pixels.
        width: Grid width in pixels.

    Returns:
        ``(height, width, 2)`` float32 array; channel 0 is ``x / width``,
        channel 1 is ``y / height``. Both span ``[0, 1)``.

    Raises:
        ValueError: If either extent is not positive.
    """
    if height < 1 or width < 1:
        raise ValueError(f"grid extents must be positive, got {(height, width)}")
    # DECISION plan-2026-09-08T111844-de235227/D-021
    # Channel 0 is x (varies across the WIDTH), channel 1 is y. Do NOT reorder
    # them to the more natural-looking (row, col): upstream's getBasecoord
    # concatenates the COLUMN plane first (utils.py:139), and a swap keeps the
    # shape, dtype and range identical while transposing every dewarped page.
    # Each axis is normalised by its OWN extent; upstream only ever divides by
    # the single scalar 256 because its grid is always square, which leaves
    # the non-square case unspecified. Do NOT collapse this back to one
    # divisor. See D-021 in decisions.md.
    y_plane = np.tile(np.arange(height, dtype=np.float32).reshape(height, 1), (1, width))
    x_plane = np.tile(np.arange(width, dtype=np.float32).reshape(1, width), (height, 1))
    return np.stack([x_plane / float(width), y_plane / float(height)], axis=-1).astype(
        np.float32
    )


def apply_document_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero the RGB page outside the document mask (``inference.py:22``).

    The dewarping RGB input is the *masked* page, not the raw one. This lives
    beside :func:`dewarp_prompt` so the inference and staging paths share one
    definition instead of each writing ``img[mask == 0] = 0``.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image.
        mask: ``(H, W)`` uint8 document mask; any non-zero value is inside.

    Returns:
        A new ``(H, W, 3)`` uint8 array, zeroed outside the mask.

    Raises:
        ValueError: If shapes or dtypes do not match the contract.
    """
    _check_rgb_uint8(img)
    _check_mask(mask, img.shape[0], img.shape[1])
    out = img.copy()
    out[mask == 0] = 0
    return out


def _check_mask(mask: np.ndarray, height: int, width: int) -> None:
    """Validate a document mask against a page's spatial shape.

    Args:
        mask: Candidate mask.
        height: Expected height.
        width: Expected width.

    Raises:
        ValueError: If ``mask`` is not a ``(height, width)`` uint8 array.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"mask must be a numpy array, got {type(mask)!r}")
    if mask.dtype != np.uint8:
        raise ValueError(
            f"mask must be uint8 in [0, 255], got dtype {mask.dtype}. Upstream "
            f"divides the mask by 255 to build the prompt channel, so a mask "
            f"already scaled to [0, 1] would be silently 255x too dark."
        )
    if mask.shape != (height, width):
        raise ValueError(
            f"mask shape {mask.shape} does not match the page ({height}, {width})"
        )


def dewarp_prompt(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dewarping prompt: ``[base_x, base_y, mask]``.

    .. warning::

       **The mask is an input here, and this port does not produce it.**
       Upstream obtains it from a *separate trained network* --- MBD
       ("Mask-Based Dewarper", ``DocRes/data/MBD/``, checkpoint ``mbd.pkl``),
       invoked as ``net1_net2_infer_single_im`` at ``inference.py:20``. MBD is
       not part of this port, so a caller wanting dewarping prompts must supply
       a document mask from somewhere else (MBD, a segmentation model, or a
       hand annotation). Without one, dewarping cannot be run end to end.

    See :func:`base_coordinate_grid` for the x/y channel order, which is the
    one thing here that can be wrong without changing the array's shape.

    Args:
        img: ``(H, W, 3)`` uint8 RGB page image. Used for its spatial shape
            and validated; its pixel values do not enter the prompt. Pass the
            page through :func:`apply_document_mask` to build the matching RGB
            input.
        mask: ``(H, W)`` uint8 document mask, non-zero inside the page.

    Returns:
        ``(H, W, 3)`` **float32** array in ``[0, 1]`` -- unlike the other four
        generators, which are uint8. Channel 0 is ``x / W``, channel 1 is
        ``y / H``, channel 2 is ``mask / 255``.

    Raises:
        ValueError: If ``img`` or ``mask`` violates the contract above.
    """
    _check_rgb_uint8(img)
    h, w = img.shape[:2]
    _check_mask(mask, h, w)
    grid = base_coordinate_grid(h, w)
    mask_channel = (mask.astype(np.float32) / 255.0)[..., None]
    return np.concatenate([grid, mask_channel], axis=-1).astype(np.float32)


logger.debug(
    "dtsprompt loaded (numpy+scipy only; background working size %d, "
    "dilate %d, median %d)",
    BACKGROUND_WORKING_SIZE,
    DILATE_KERNEL_SIZE,
    MEDIAN_KERNEL_SIZE,
)
