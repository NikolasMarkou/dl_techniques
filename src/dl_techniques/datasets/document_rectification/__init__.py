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
    * :mod:`~dl_techniques.datasets.document_rectification.uvdoc` -- the UVDoc
      reader and densifier. UVDoc's ground truth is a **coarse 89x61
      correspondence lattice** in MATLAB v7.3 (HDF5) ``.mat`` files, already in
      the backward direction (rectified domain, distorted-pixel values), so it
      is densified with an interpolating tensor-product spline rather than
      inverted. Its forward map ``g`` IS fitted, by ``scipy.interpolate``, and
      is the one genuinely approximate step in this whole port -- bounded,
      measured and gated by a rejection path. numpy + scipy at module scope;
      ``h5py`` and ``Pillow`` (both ``data``-extra only) are imported lazily
      inside the two functions that need them.
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
    * :class:`UVDocSource`, :func:`load_geometry`, :func:`load_image` -- UVDoc.
    * :func:`read_grid2d`, :func:`read_segmentation`, :func:`read_uvmap`,
      :func:`control_point_uv`, :func:`densify_backward_map`,
      :func:`invert_backward_map`, :func:`rectified_pixel_grid` -- the UVDoc
      ground-truth path, piece by piece.
    * :func:`assess_densification`, :func:`is_unusable`,
      :func:`held_out_densification_error`, :func:`roundtrip_error`,
      :class:`UVDocQuality`, :class:`UVDocGeometry`,
      :class:`UVDocDensificationError`, :class:`UVDocError` -- UVDoc's measured
      accuracy and its rejection path (the synthetic side's counterpart is
      :func:`is_degenerate`, which can re-draw instead of rejecting).
    * :class:`InvertibleWarp`, :func:`sample_warp` -- the warp itself.
    * :func:`render_sample`, :func:`generate_sample` -- ``(image, f_gt, mask)``.
    * :func:`assess_warp`, :func:`is_degenerate`, :func:`map_jacobian_stats`,
      :class:`WarpQuality`, :class:`DegenerateWarpError` -- the
      reject-and-resample policy.
    * :func:`rgb_to_hsv`, :func:`hsv_to_rgb`, :func:`jitter_hsv` -- the
      photometric jitter.
    * :func:`load_rgb` -- lazy-Pillow file reader.
"""

from .uvdoc import (
    MAX_HELD_OUT_DENSIFICATION_PIXELS,
    MAX_HULL_MISS_FRACTION,
    MAX_INTERIOR_ROUNDTRIP_PIXELS,
    ROUNDTRIP_BORDER_PIXELS,
    UVDocDensificationError,
    UVDocError,
    UVDocGeometry,
    UVDocQuality,
    UVDocSource,
    assess_densification,
    control_point_uv,
    densify_backward_map,
    held_out_densification_error,
    invert_backward_map,
    is_unusable,
    load_geometry,
    load_image,
    read_grid2d,
    read_segmentation,
    read_uvmap,
    rectified_pixel_grid,
    roundtrip_error,
)
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
    "MAX_HELD_OUT_DENSIFICATION_PIXELS",
    "MAX_HULL_MISS_FRACTION",
    "MAX_INTERIOR_ROUNDTRIP_PIXELS",
    "ROUNDTRIP_BORDER_PIXELS",
    "UVDocDensificationError",
    "UVDocError",
    "UVDocGeometry",
    "UVDocQuality",
    "UVDocSource",
    "assess_densification",
    "control_point_uv",
    "densify_backward_map",
    "held_out_densification_error",
    "invert_backward_map",
    "load_geometry",
    "load_image",
    "read_grid2d",
    "read_segmentation",
    "read_uvmap",
    "rectified_pixel_grid",
    "roundtrip_error",
    "is_unusable",
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
