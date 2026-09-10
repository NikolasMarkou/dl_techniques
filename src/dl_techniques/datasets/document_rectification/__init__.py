"""Document-*rectification* (dewarping) data utilities for the DocScanner port.

This is a sibling of
:mod:`dl_techniques.datasets.document_restoration`, not an extension of it, and
the split is deliberate. ``document_restoration`` owns DocRes's *photometric*
task family -- deshadow, appearance, deblur, binarisation -- whose data unit is
``(image, prompt channels, restored image)`` and whose ``TASKS`` table is
consumed by exactly one model. This package owns *geometric* ground truth: a
warped page paired with the backward warping flow that undoes the warp, whose
data unit is ``(image, f_gt, mask)`` and whose consumers are the DocScanner
segmenter and rectifier. Nothing here branches on a DocRes task string and
nothing there knows what a backward map is; merging them would put two
unrelated ground-truth contracts behind one import.

The one thing that *is* shared is re-used rather than copied:
:func:`~dl_techniques.datasets.document_restoration.dtsprompt.base_coordinate_grid`
supplies the normalised ``(x/W, y/H)`` flat-UV grid this package's warp math is
expressed in, including its channel-order and per-axis-normalisation decisions.

Modules:
    * :mod:`~dl_techniques.datasets.document_rectification.synthetic_warp` --
      the synthetic warped-page generator. **numpy + scipy only** at module
      scope (``cv2`` and ``skimage`` are undeclared dependencies; ``Pillow``
      lives in the ``data`` extra and is imported lazily by the single
      file-touching helper). Its warp family is composed exclusively of
      closed-form-invertible primitives, so the backward map ``f_gt`` *and* the
      forward map ``g`` are both exact and neither is ever fitted -- read that
      module's docstring before changing the warp family, because the usual
      "render through the map you supervise on" recipe silently inverts the
      two directions.

Public surface:
    * :class:`InvertibleWarp`, :func:`sample_warp` -- the warp itself.
    * :func:`render_sample`, :func:`generate_sample` -- ``(image, f_gt, mask)``.
    * :func:`assess_warp`, :func:`is_degenerate`, :func:`map_jacobian_stats`,
      :class:`WarpQuality`, :class:`DegenerateWarpError` -- the
      reject-and-resample policy.
    * :func:`rgb_to_hsv`, :func:`hsv_to_rgb`, :func:`jitter_hsv` -- the
      photometric jitter.
    * :func:`load_rgb` -- lazy-Pillow file reader.
"""

from .synthetic_warp import (
    BACKGROUND_CROP_SCALE_RANGE,
    DEFAULT_CONTROL_POINTS,
    DEFAULT_PAGE_MARGIN,
    DEFAULT_PERSPECTIVE_AMPLITUDE,
    DEFAULT_SHEAR_AMPLITUDE,
    DEFAULT_SHEAR_STAGES,
    HUE_SHIFT_RANGE,
    MAX_JACOBIAN_CONDITION,
    MAX_WARP_RESAMPLE_ATTEMPTS,
    MIN_JACOBIAN_DET,
    MIN_PAGE_COVERAGE,
    SATURATION_SCALE_RANGE,
    VALUE_SCALE_RANGE,
    WARP_PROBE_SIZE,
    DegenerateWarpError,
    InvertibleWarp,
    WarpQuality,
    assess_warp,
    generate_sample,
    hsv_to_rgb,
    is_degenerate,
    jitter_hsv,
    load_rgb,
    map_jacobian_stats,
    render_sample,
    rgb_to_hsv,
    sample_warp,
)

__all__ = [
    "BACKGROUND_CROP_SCALE_RANGE",
    "DEFAULT_CONTROL_POINTS",
    "DEFAULT_PAGE_MARGIN",
    "DEFAULT_PERSPECTIVE_AMPLITUDE",
    "DEFAULT_SHEAR_AMPLITUDE",
    "DEFAULT_SHEAR_STAGES",
    "HUE_SHIFT_RANGE",
    "MAX_JACOBIAN_CONDITION",
    "MAX_WARP_RESAMPLE_ATTEMPTS",
    "MIN_JACOBIAN_DET",
    "MIN_PAGE_COVERAGE",
    "SATURATION_SCALE_RANGE",
    "VALUE_SCALE_RANGE",
    "WARP_PROBE_SIZE",
    "DegenerateWarpError",
    "InvertibleWarp",
    "WarpQuality",
    "assess_warp",
    "generate_sample",
    "hsv_to_rgb",
    "is_degenerate",
    "jitter_hsv",
    "load_rgb",
    "map_jacobian_stats",
    "render_sample",
    "rgb_to_hsv",
    "sample_warp",
]
