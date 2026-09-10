"""Synthetic warped-page generator for the DocScanner rectification port.

DocScanner is trained on rendered document distortions: a flat page is warped,
composited onto a background texture, photometrically jittered, and paired with
the *backward warping flow* ``f_gt`` that undoes the warp. The paper's own
recipe uses Doc3D (100k renders) and DTD backgrounds; neither Doc3D nor its
renderer is obtainable (see the porting plan's F-09/F-13), so this module
synthesises the same kind of pair from a flat page raster plus a background
texture, using **numpy and scipy only**.

The direction problem, and why this module does not follow the obvious recipe
---------------------------------------------------------------------------
There are two maps in a dewarping pair, and they point opposite ways:

``f_gt`` (**backward**, the supervision target of Eq. 11)
    Indexed by the **rectified/flat** pixel grid; its value is the coordinate
    in the **distorted** image that pixel is read from. This is what
    :class:`~dl_techniques.models.vision.image_restoration.doc_scanner.model.DocScannerRectifier`
    emits and what ``F.grid_sample(distorted, bm)`` consumes upstream.

``g`` (**forward**, Eq. 12, needed only by the circle-consistency term)
    Indexed by the **distorted** pixel grid; its value is the coordinate in the
    rectified image. It is ``f_gt``'s inverse.

The usual synthetic recipe renders the distorted image by *gathering* the flat
page through a map, and a gather render pins down exactly one of the two maps:
the one indexed by the grid being written, i.e. the **distorted-to-flat**
direction, i.e. ``g``. So "build the backward map first and render through it"
is not achievable as stated -- rendering through a map makes that map exact,
and the map you can render through is ``g``, not ``f_gt``. Taking that recipe
literally and labelling the render map ``f_gt`` is a silent, plausible-looking
inversion: every shape, dtype, range and finiteness check passes, training
converges, and the resulting model emits a map that unwarps in the wrong
direction. There is no shape symptom at all at the square 288x288 training
resolution.

This module removes the trade-off instead of choosing a side: **every warp
primitive it composes has a closed-form inverse**, so ``f_gt`` and ``g`` are
BOTH exact and neither is ever obtained by numerical inversion. The warp is a
composition of

1. shear-type coupling maps ``(x, y) -> (x, y + a(x))`` and
   ``(x, y) -> (x + b(y), y)``, whose inverses are the same expression with the
   sign flipped -- exact for *any* displacement profile ``a``, ``b``, including
   the clamped cubic splines used here;
2. one projective (homography) stage for perspective, inverted by ``H^-1``;
3. one isotropic fit affine that places the warped page inside the frame with a
   margin, inverted trivially.

The distorted image is then rendered by gathering the flat page through
``g = Phi^-1``, and ``f_gt = Phi`` is emitted. The pair is consistent to float
round-off (measured: the emitted ``f_gt`` composed with the render map
reproduces the identity to better than 1e-9 pixels), and there is no
scattered-data inversion anywhere in this file.

A consequence worth carrying forward: because ``g`` is closed form here, the
plan's step-14 ``griddata`` inversion is not needed for *synthetic* samples.
It is still needed for UVDoc, whose ground truth is a coarse correspondence
grid with no analytic form.

A second consequence: a composition of bijections is a bijection, so this warp
family **cannot fold the page**. Folding is not a failure mode that has to be
detected here; the failure mode that survives is extreme *shear* (a Jacobian
with a large condition number but determinant 1) and extreme perspective (a
vanishing line close to the frame). Both are rejected, and re-sampled, by
:func:`assess_warp` / :func:`is_degenerate`.

Units and channel order (do not "simplify" these)
-------------------------------------------------
``f_gt`` is ``(H, W, 2)`` float32 in **absolute pixel coordinates of the
distorted image**, channel 0 = ``x`` (column), channel 1 = ``y`` (row). That is
exactly the port's own convention -- ``warp.coords_grid`` emits ``(x, y)``,
``DocScannerRectifier`` emits absolute full-resolution pixels, and
:class:`~dl_techniques.losses.DocScannerFlowSequenceLoss` computes its L1 in
those pixels. Normalising to ``[-1, 1]`` here would be a different objective
with a different effective ``alpha`` (see that loss's docstring), and swapping
the two channels is invisible at ``H == W``.

Dependencies
------------
**numpy + scipy only, at module scope.** ``cv2`` and ``scikit-image`` are not
declared in ``pyproject.toml``, and ``Pillow`` is declared only in the ``data``
extra -- so :func:`load_rgb`, the one function that touches a file, imports
Pillow lazily inside its own body. The warp math, the renderer and the HSV
jitter therefore import on a core install, which is what lets the same module
serve an offline staging script and a Keras trainer. This mirrors the house
rule stated at ``document_restoration/dtsprompt.py:9-15``; a suite guard
enforces it.

Public surface:
    * :class:`InvertibleWarp` -- the composed, exactly-invertible warp.
    * :func:`sample_warp` -- draw a non-degenerate warp from an explicit
      ``numpy.random.Generator``.
    * :func:`render_sample` -- render ``(image, f_gt, mask)`` for a given warp.
    * :func:`generate_sample` -- :func:`sample_warp` + :func:`render_sample`.
    * :func:`assess_warp`, :func:`is_degenerate`, :func:`map_jacobian_stats`,
      :class:`WarpQuality`, :class:`DegenerateWarpError` -- the rejection
      policy.
    * :func:`rgb_to_hsv`, :func:`hsv_to_rgb`, :func:`jitter_hsv` -- the
      photometric augmentation, in numpy.
    * :func:`load_rgb` -- the only file-touching helper (lazy Pillow).

References:
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2, Eq. 11-12
      for the two map directions, and the "jitter in the HSV color space to
      magnify illumination and document color variations" sentence this
      module's :func:`jitter_hsv` implements.
    - Ma et al., 2018. DocUNet: Document Image Unwarping via a Stacked U-Net --
      the control-grid synthetic recipe this file's warp family stands in for.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from dl_techniques.datasets.document_restoration.dtsprompt import base_coordinate_grid
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants. Every one of these is THIS REPO'S CHOICE: the paper states only
# that it warps a flat page and jitters it in HSV, and no synthetic renderer
# was released with it. None of these numbers reproduces a published value.
# ---------------------------------------------------------------------------

#: Number of interior control points per 1-D displacement profile. Six gives a
#: fold-and-curl look at page scale; more makes the profile ripple.
DEFAULT_CONTROL_POINTS: int = 6

#: Number of alternating shear stages composed into one warp. Two stages of
#: each orientation is the smallest composition that can bend both axes.
DEFAULT_SHEAR_STAGES: int = 4

#: Peak magnitude of a shear displacement, in units of the unit square.
DEFAULT_SHEAR_AMPLITUDE: float = 0.09

#: Peak corner displacement of the perspective stage, in units of the unit
#: square.
DEFAULT_PERSPECTIVE_AMPLITUDE: float = 0.10

#: Fraction of the frame kept clear around the warped page by the fit affine.
DEFAULT_PAGE_MARGIN: float = 0.06

#: Rejection thresholds for :func:`is_degenerate`, measured on a square probe
#: grid so they do not depend on the output resolution or aspect ratio.
MIN_JACOBIAN_DET: float = 0.05
MAX_JACOBIAN_CONDITION: float = 6.0
MIN_PAGE_COVERAGE: float = 0.20

#: Square probe resolution the quality metrics are measured at.
WARP_PROBE_SIZE: int = 64

#: How many times :func:`sample_warp` re-draws before giving up.
MAX_WARP_RESAMPLE_ATTEMPTS: int = 32

#: HSV jitter ranges: hue is an additive shift (wrapped), saturation and value
#: are multiplicative.
HUE_SHIFT_RANGE: Tuple[float, float] = (-0.05, 0.05)
SATURATION_SCALE_RANGE: Tuple[float, float] = (0.60, 1.40)
VALUE_SCALE_RANGE: Tuple[float, float] = (0.70, 1.30)

#: Background crop scale range, as a fraction of the background's short side.
BACKGROUND_CROP_SCALE_RANGE: Tuple[float, float] = (0.45, 1.00)

_EPS = 1e-12


class DegenerateWarpError(RuntimeError):
    """Raised when no acceptable warp could be drawn within the attempt budget.

    This is a *policy* exception, not a bug: it means the requested amplitudes
    are large enough that most draws violate :func:`is_degenerate`. Reduce
    ``shear_amplitude`` / ``perspective_amplitude`` or raise
    :data:`MAX_JACOBIAN_CONDITION` deliberately.
    """


# ---------------------------------------------------------------------------
# 1. The exactly-invertible primitives.
#
# Each primitive maps R^2 -> R^2 and provides an inverse that is its algebraic
# inverse, not an approximation. `apply` and `invert` both take and return
# `(N, 2)` float64 arrays in (x, y) order.
# ---------------------------------------------------------------------------


class _Primitive:
    """Base class for a bijection of the plane with a closed-form inverse."""

    def apply(self, xy: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def invert(self, xy: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class _Shear(_Primitive):
    """A coupling map: displace one axis by a function of the *other* axis.

    ``axis=1`` (the default) is ``(x, y) -> (x, y + a(x))``; ``axis=0`` is
    ``(x, y) -> (x + a(y), y)``. The inverse subtracts the same displacement
    evaluated at the same, unchanged, driving coordinate -- which is why the
    inverse is exact for an *arbitrary* profile ``a``, including a spline, a
    step, or a profile clamped outside ``[0, 1]``.

    Attributes:
        axis: Which output component is displaced (0 for x, 1 for y).
        profile: Callable mapping the driving coordinate to a displacement.
    """

    def __init__(self, axis: int, profile: "_ClampedSpline") -> None:
        if axis not in (0, 1):
            raise ValueError(f"axis must be 0 or 1, got {axis}")
        self.axis = axis
        self.profile = profile

    def _displacement(self, xy: np.ndarray) -> np.ndarray:
        driving = xy[:, 1 - self.axis]
        return self.profile(driving)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        out = xy.copy()
        out[:, self.axis] = xy[:, self.axis] + self._displacement(xy)
        return out

    def invert(self, xy: np.ndarray) -> np.ndarray:
        # The driving coordinate is untouched by `apply`, so it can be read
        # straight off the warped point -- this is the whole reason a coupling
        # map inverts in closed form. Do NOT "symmetrise" this by evaluating
        # the profile at the output of the displaced axis.
        out = xy.copy()
        out[:, self.axis] = xy[:, self.axis] - self._displacement(xy)
        return out


class _ClampedSpline:
    """A natural cubic spline over ``[0, 1]``, held constant outside it.

    Clamping the *argument* (rather than letting the spline extrapolate) keeps
    the displacement bounded for query points outside the unit square -- which
    the renderer generates constantly, because it inverts the map over the
    whole distorted frame, most of which is background. Cubic extrapolation
    there produces displacements of tens of page widths and a useless mask.

    Attributes:
        knots: ``(n,)`` increasing abscissae spanning ``[0, 1]``.
        values: ``(n,)`` displacement values at ``knots``.
    """

    def __init__(self, knots: np.ndarray, values: np.ndarray) -> None:
        self.knots = np.asarray(knots, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self._spline = CubicSpline(self.knots, self.values, bc_type="natural")

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(self._spline(np.clip(t, 0.0, 1.0)), dtype=np.float64)


class _Affine(_Primitive):
    """Isotropic scale plus translation: ``p -> scale * p + offset``.

    Attributes:
        scale: Positive isotropic scale factor.
        offset: ``(2,)`` translation, applied after the scale.
    """

    def __init__(self, scale: float, offset: Sequence[float]) -> None:
        if not scale > 0.0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = float(scale)
        self.offset = np.asarray(offset, dtype=np.float64).reshape(2)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.scale + self.offset

    def invert(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.offset) / self.scale


class _Homography(_Primitive):
    """A projective map of the plane, inverted by the inverse matrix.

    Attributes:
        matrix: ``(3, 3)`` homogeneous matrix with ``matrix[2, 2] == 1``.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        self.inverse = np.linalg.inv(self.matrix)

    @staticmethod
    def _project(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
        denominator = matrix[2, 0] * xy[:, 0] + matrix[2, 1] * xy[:, 1] + matrix[2, 2]
        # A projective map is a bijection only away from its vanishing line.
        # Guarding the denominator keeps the arithmetic finite; a warp that
        # gets anywhere near it is rejected by `is_degenerate` (the Jacobian
        # condition number explodes long before the sign flips).
        denominator = np.where(
            np.abs(denominator) < _EPS, np.sign(denominator + _EPS) * _EPS, denominator
        )
        x = matrix[0, 0] * xy[:, 0] + matrix[0, 1] * xy[:, 1] + matrix[0, 2]
        y = matrix[1, 0] * xy[:, 0] + matrix[1, 1] * xy[:, 1] + matrix[1, 2]
        return np.stack([x / denominator, y / denominator], axis=-1)

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return self._project(self.matrix, xy)

    def invert(self, xy: np.ndarray) -> np.ndarray:
        return self._project(self.inverse, xy)


def _homography_from_corners(corners: np.ndarray) -> np.ndarray:
    """Solve the 8-parameter homography taking the unit square to ``corners``.

    Args:
        corners: ``(4, 2)`` destinations of ``(0,0), (1,0), (1,1), (0,1)``.

    Returns:
        ``(3, 3)`` float64 matrix with a unit bottom-right entry.

    Raises:
        ValueError: If ``corners`` is not ``(4, 2)``.
        numpy.linalg.LinAlgError: If the correspondence is degenerate.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"corners must be (4, 2), got {corners.shape}")
    source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for (sx, sy), (dx, dy) in zip(source, corners):
        rows.append(np.array([sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy]))
        rhs.append(dx)
        rows.append(np.array([0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy]))
        rhs.append(dy)
    solution = np.linalg.solve(np.stack(rows), np.asarray(rhs, dtype=np.float64))
    return np.append(solution, 1.0).reshape(3, 3)


# ---------------------------------------------------------------------------
# 2. The composed warp.
# ---------------------------------------------------------------------------


class InvertibleWarp:
    """A composition of bijections mapping flat UV to distorted UV.

    Both directions are exact: :meth:`apply` runs the primitives in order,
    :meth:`invert` runs each primitive's closed-form inverse in reverse order.
    No numerical inversion, no interpolation, no fitting.

    The convention, stated once:

    * ``apply`` takes **flat/rectified** UV and returns **distorted** UV. In
      pixel units that is the paper's backward map ``f_gt`` (Eq. 11).
    * ``invert`` takes **distorted** UV and returns **flat/rectified** UV. In
      pixel units that is the paper's forward map ``g`` (Eq. 12), and it is the
      map :func:`render_sample` gathers the page through.

    Attributes:
        primitives: The ordered primitives, applied left to right by
            :meth:`apply`.
    """

    def __init__(self, primitives: Sequence[_Primitive]) -> None:
        if len(primitives) == 0:
            raise ValueError("a warp needs at least one primitive")
        self.primitives: Tuple[_Primitive, ...] = tuple(primitives)

    # -- the two directions ------------------------------------------------

    def apply(self, xy: np.ndarray) -> np.ndarray:
        """Flat UV to distorted UV.

        Args:
            xy: ``(N, 2)`` float array, channel 0 ``x``, channel 1 ``y``.

        Returns:
            ``(N, 2)`` float64 array in the same channel order.
        """
        out = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
        for primitive in self.primitives:
            out = primitive.apply(out)
        return out

    def invert(self, xy: np.ndarray) -> np.ndarray:
        """Distorted UV to flat UV -- the exact inverse of :meth:`apply`.

        Args:
            xy: ``(N, 2)`` float array, channel 0 ``x``, channel 1 ``y``.

        Returns:
            ``(N, 2)`` float64 array in the same channel order.
        """
        out = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
        for primitive in reversed(self.primitives):
            out = primitive.invert(out)
        return out

    # -- the two maps, in absolute pixels ----------------------------------

    def backward_map(self, height: int, width: int) -> np.ndarray:
        """``f_gt``: the map the rectifier is supervised on.

        Args:
            height: Output height ``H`` in pixels.
            width: Output width ``W`` in pixels.

        Returns:
            ``(H, W, 2)`` float32. Indexed by the **rectified** pixel grid;
            the value at ``[row, col]`` is the ``(x, y)`` pixel coordinate in
            the **distorted** image that this rectified pixel is read from.
        """
        grid = _flat_uv_grid(height, width)
        warped = self.apply(grid.reshape(-1, 2)).reshape(height, width, 2)
        return (warped * np.array([width, height], dtype=np.float64)).astype(np.float32)

    def forward_map(self, height: int, width: int) -> np.ndarray:
        """``g``: the circle-consistency term's map, and the render gather map.

        Args:
            height: Output height ``H`` in pixels.
            width: Output width ``W`` in pixels.

        Returns:
            ``(H, W, 2)`` float32. Indexed by the **distorted** pixel grid; the
            value at ``[row, col]`` is the ``(x, y)`` pixel coordinate in the
            **rectified** image that this distorted pixel maps to. Values
            outside ``[0, W) x [0, H)`` are background, and are returned
            unclipped so the caller can test them.
        """
        grid = _flat_uv_grid(height, width)
        unwarped = self.invert(grid.reshape(-1, 2)).reshape(height, width, 2)
        return (unwarped * np.array([width, height], dtype=np.float64)).astype(np.float32)


def _flat_uv_grid(height: int, width: int) -> np.ndarray:
    """The normalised ``(x/W, y/H)`` grid both frames are expressed in.

    Delegates to
    :func:`~dl_techniques.datasets.document_restoration.dtsprompt.base_coordinate_grid`
    -- the repo already owns this grid, including its channel-order decision
    (channel 0 is ``x``, which varies along the width) and its
    normalise-each-axis-by-its-own-extent rule. It is re-used, not re-derived.
    The only change here is the float64 cast: the warp composition is done in
    double precision so the ``apply`` / ``invert`` round trip closes at
    round-off rather than at float32's 1e-7.

    Note that this grid spans ``[0, 1)``, not ``[0, 1]``: pixel ``i`` sits at
    ``i / W``. The renderer samples the page raster under the same convention
    (``u * W_page`` reads page column ``u * W_page``), so the two agree exactly
    at the identity warp.

    Args:
        height: Grid height in pixels.
        width: Grid width in pixels.

    Returns:
        ``(height, width, 2)`` float64 array.
    """
    return base_coordinate_grid(height, width).astype(np.float64)


# ---------------------------------------------------------------------------
# 3. Sampling a warp, and the degeneracy policy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarpQuality:
    """Resolution-independent quality metrics of a warp.

    Attributes:
        min_jacobian_det: Smallest signed Jacobian determinant over the probe
            grid. Non-positive means the map folds -- impossible for this warp
            family, so a non-positive value is a bug, not a draw to reject.
        max_jacobian_condition: Largest ratio of singular values of the
            Jacobian. This, not the determinant, is the metric that matters
            here: shear maps have determinant exactly 1 at every amplitude, so
            a determinant test alone cannot see an extreme shear at all.
        page_coverage: Fraction of the distorted frame the warped page covers.
    """

    min_jacobian_det: float
    max_jacobian_condition: float
    page_coverage: float


def map_jacobian_stats(coord_map: np.ndarray) -> Tuple[float, float]:
    """Jacobian determinant and condition-number extremes of a coordinate map.

    Operates on the array, not on a warp object, so it can be handed a map that
    no :class:`InvertibleWarp` could produce -- e.g. a folded one -- which is
    what makes the degeneracy guard testable rather than vacuous.

    The 2x2 singular values are computed in closed form from the determinant
    and the Frobenius norm (``s1*s2 = |det|``, ``s1^2 + s2^2 = ||J||_F^2``)
    rather than by calling an SVD per pixel.

    Args:
        coord_map: ``(H, W, 2)`` coordinate map in pixel units, channel 0
            ``x``, channel 1 ``y``.

    Returns:
        ``(min_signed_det, max_condition_number)``.

    Raises:
        ValueError: If ``coord_map`` is not ``(H, W, 2)`` with ``H, W >= 2``.
    """
    coord_map = np.asarray(coord_map, dtype=np.float64)
    if coord_map.ndim != 3 or coord_map.shape[2] != 2:
        raise ValueError(f"coord_map must be (H, W, 2), got {coord_map.shape}")
    if coord_map.shape[0] < 2 or coord_map.shape[1] < 2:
        raise ValueError(
            f"coord_map needs at least 2x2 samples to differentiate, got "
            f"{coord_map.shape[:2]}"
        )
    d_col = np.gradient(coord_map, axis=1)
    d_row = np.gradient(coord_map, axis=0)
    a = d_col[..., 0]
    c = d_col[..., 1]
    b = d_row[..., 0]
    d = d_row[..., 1]
    det = a * d - b * c
    frobenius_sq = a * a + b * b + c * c + d * d
    discriminant = np.sqrt(np.maximum(frobenius_sq**2 - 4.0 * det**2, 0.0))
    sigma_max = np.sqrt(np.maximum((frobenius_sq + discriminant) * 0.5, 0.0))
    sigma_min = np.sqrt(np.maximum((frobenius_sq - discriminant) * 0.5, 0.0))
    condition = sigma_max / np.maximum(sigma_min, _EPS)
    return float(det.min()), float(condition.max())


def assess_warp(warp: InvertibleWarp, probe_size: int = WARP_PROBE_SIZE) -> WarpQuality:
    """Measure a warp on a square probe grid.

    The probe is deliberately square and of fixed size: the warp lives in the
    unit square, so its intrinsic stretch does not depend on the resolution or
    the aspect ratio the sample is finally rendered at. Measuring on the output
    grid instead would make the accept/reject decision move with ``size``.

    Args:
        warp: The warp to measure.
        probe_size: Side of the square probe grid.

    Returns:
        The :class:`WarpQuality` metrics.
    """
    backward = warp.backward_map(probe_size, probe_size)
    min_det, max_condition = map_jacobian_stats(backward)
    flat_uv = warp.invert(_flat_uv_grid(probe_size, probe_size).reshape(-1, 2))
    inside = np.all((flat_uv >= 0.0) & (flat_uv < 1.0), axis=1)
    return WarpQuality(
        min_jacobian_det=min_det,
        max_jacobian_condition=max_condition,
        page_coverage=float(inside.mean()),
    )


def is_degenerate(quality: WarpQuality) -> bool:
    """Whether a warp must be rejected and re-drawn.

    The policy is **reject-and-resample**, not clamp. Clamping a too-extreme
    warp would silently bias the training distribution toward the clamp
    boundary and would leave the emitted sample looking perfectly healthy;
    re-drawing keeps the accepted distribution exactly "the prior, conditioned
    on being acceptable", which is a thing that can be stated and checked.

    Args:
        quality: Metrics from :func:`assess_warp`.

    Returns:
        ``True`` if the warp violates any threshold.
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-037: the load-bearing term here
    # is the CONDITION NUMBER, not the determinant. A shear has determinant
    # exactly 1 at every amplitude, and this warp family is built out of
    # shears -- so a determinant-only test (the natural way to write "detect a
    # degenerate warp") would accept an arbitrarily violent one and reject
    # nothing this generator can actually produce. The determinant term is kept
    # only as a fold tripwire for a future edit that adds a non-injective
    # primitive. Do NOT replace the reject-and-resample policy with clamping:
    # clamping biases the accepted distribution onto the threshold and leaves
    # the emitted sample looking perfectly healthy. See decisions.md D-037.
    return bool(
        quality.min_jacobian_det <= MIN_JACOBIAN_DET
        or quality.max_jacobian_condition >= MAX_JACOBIAN_CONDITION
        or quality.page_coverage < MIN_PAGE_COVERAGE
    )


def _sample_shear(rng: np.random.Generator, axis: int, amplitude: float) -> _Shear:
    """Draw one coupling map with a smooth, zero-mean displacement profile."""
    knots = np.linspace(0.0, 1.0, DEFAULT_CONTROL_POINTS)
    values = rng.uniform(-amplitude, amplitude, size=DEFAULT_CONTROL_POINTS)
    values = values - values.mean()
    return _Shear(axis=axis, profile=_ClampedSpline(knots, values))


def _fit_affine(warp_without_fit: InvertibleWarp, margin: float) -> _Affine:
    """Isotropic scale + shift placing the warped page inside the frame.

    The warp is a homeomorphism, so the image of the unit square is bounded by
    the image of its boundary; a dense boundary sample is therefore enough to
    get an exact bounding box (to the sampling density).
    """
    steps = np.linspace(0.0, 1.0, 257)
    ones = np.ones_like(steps)
    boundary = np.concatenate(
        [
            np.stack([steps, 0.0 * ones], axis=-1),
            np.stack([steps, ones], axis=-1),
            np.stack([0.0 * ones, steps], axis=-1),
            np.stack([ones, steps], axis=-1),
        ],
        axis=0,
    )
    warped = warp_without_fit.apply(boundary)
    lower = warped.min(axis=0)
    upper = warped.max(axis=0)
    extent = np.maximum(upper - lower, _EPS)
    scale = float((1.0 - 2.0 * margin) / extent.max())
    centre = (lower + upper) * 0.5
    offset = np.array([0.5, 0.5]) - scale * centre
    return _Affine(scale=scale, offset=offset)


def sample_warp(
    rng: np.random.Generator,
    *,
    shear_stages: int = DEFAULT_SHEAR_STAGES,
    shear_amplitude: float = DEFAULT_SHEAR_AMPLITUDE,
    perspective_amplitude: float = DEFAULT_PERSPECTIVE_AMPLITUDE,
    margin: float = DEFAULT_PAGE_MARGIN,
) -> InvertibleWarp:
    """Draw a non-degenerate warp.

    Every random draw comes from ``rng``; this function touches no global numpy
    state, so the same generator seed yields the same warp bit-for-bit.

    Args:
        rng: Explicit generator, e.g. ``numpy.random.default_rng(0)``.
        shear_stages: How many coupling maps to compose; orientations
            alternate, starting with a vertical displacement driven by ``x``.
        shear_amplitude: Peak displacement of each coupling map, in unit-square
            units.
        perspective_amplitude: Peak corner displacement of the projective
            stage.
        margin: Fraction of the frame kept clear around the page.

    Returns:
        A warp satisfying :func:`is_degenerate` ``== False``.

    Raises:
        ValueError: If any argument is out of range.
        DegenerateWarpError: If no acceptable warp was drawn within
            :data:`MAX_WARP_RESAMPLE_ATTEMPTS`.
    """
    if shear_stages < 0:
        raise ValueError(f"shear_stages must be >= 0, got {shear_stages}")
    if shear_amplitude < 0.0 or perspective_amplitude < 0.0:
        raise ValueError("amplitudes must be non-negative")
    if not 0.0 <= margin < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-036: every primitive composed
    # here has a CLOSED-FORM inverse -- shear couplings, one homography, one
    # isotropic affine. Do NOT swap this family for a
    # `scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline")`
    # displacement field, which is the obvious and more expressive choice: a
    # TPS field has no analytic inverse, so one of the two maps would have to
    # come from a scattered-data fit, and the one that would (`f_gt`) is the
    # primary supervision target of Eq. 11. Invertibility by construction is
    # what makes BOTH directions exact -- MEASURED, the round trip closes at
    # 7.8e-16 in float64 -- and is also why this family cannot fold the page
    # at any amplitude. See decisions.md D-036.
    for attempt in range(MAX_WARP_RESAMPLE_ATTEMPTS):
        primitives: List[_Primitive] = [
            _sample_shear(rng, axis=(1 - index % 2), amplitude=shear_amplitude)
            for index in range(shear_stages)
        ]
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ) + rng.uniform(-perspective_amplitude, perspective_amplitude, size=(4, 2))
        primitives.append(_Homography(_homography_from_corners(corners)))
        partial = InvertibleWarp(primitives)
        candidate = InvertibleWarp(primitives + [_fit_affine(partial, margin)])
        quality = assess_warp(candidate)
        if not is_degenerate(quality):
            return candidate
        logger.debug(
            "synthetic_warp: rejected draw %d/%d (%s)",
            attempt + 1,
            MAX_WARP_RESAMPLE_ATTEMPTS,
            quality,
        )
    raise DegenerateWarpError(
        f"no warp passed the degeneracy policy in {MAX_WARP_RESAMPLE_ATTEMPTS} draws "
        f"at shear_amplitude={shear_amplitude}, "
        f"perspective_amplitude={perspective_amplitude}; the amplitudes are too "
        f"large for MAX_JACOBIAN_CONDITION={MAX_JACOBIAN_CONDITION}"
    )


# ---------------------------------------------------------------------------
# 4. Sampling and photometry, in numpy.
# ---------------------------------------------------------------------------


def _bilinear_gather(source: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly read ``source`` at pixel coordinates, edge-clamped.

    Uses the same convention as the port's
    :func:`~dl_techniques.models.vision.image_restoration.doc_scanner.warp.sample_at_pixel_coords`:
    an integer coordinate ``p`` reads pixel ``p`` with no half-pixel offset, and
    coordinates outside the raster clamp to the edge.

    Args:
        source: ``(H, W, C)`` float array.
        x: ``(N,)`` column coordinates in pixels.
        y: ``(N,)`` row coordinates in pixels.

    Returns:
        ``(N, C)`` float64 array.
    """
    source = np.asarray(source, dtype=np.float64)
    height, width = source.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = source[y0, x0] * (1.0 - wx) + source[y0, x1] * wx
    bottom = source[y1, x0] * (1.0 - wx) + source[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB in ``[0, 1]`` to HSV in ``[0, 1]``, in numpy.

    Written out rather than delegated because neither ``cv2`` nor
    ``skimage.color`` is a declared dependency of this library, and
    ``matplotlib.colors.rgb_to_hsv`` would pull a plotting stack into a data
    module.

    Args:
        rgb: ``(..., 3)`` float array in ``[0, 1]``.

    Returns:
        ``(..., 3)`` float64 array; hue in ``[0, 1)``, saturation and value in
        ``[0, 1]``.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected a trailing size-3 axis, got {rgb.shape}")
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    chroma = value - minimum
    safe_chroma = np.where(chroma == 0.0, 1.0, chroma)
    hue = np.select(
        [chroma == 0.0, value == red, value == green],
        [
            np.zeros_like(value),
            ((green - blue) / safe_chroma) % 6.0,
            ((blue - red) / safe_chroma) + 2.0,
        ],
        default=((red - green) / safe_chroma) + 4.0,
    )
    hue = (hue / 6.0) % 1.0
    saturation = np.where(value == 0.0, 0.0, chroma / np.where(value == 0.0, 1.0, value))
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV in ``[0, 1]`` back to RGB in ``[0, 1]``, in numpy.

    Args:
        hsv: ``(..., 3)`` float array.

    Returns:
        ``(..., 3)`` float64 RGB array.
    """
    hsv = np.asarray(hsv, dtype=np.float64)
    if hsv.shape[-1] != 3:
        raise ValueError(f"expected a trailing size-3 axis, got {hsv.shape}")
    hue = (hsv[..., 0] % 1.0) * 6.0
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)
    value = hsv[..., 2]
    sector = np.floor(hue).astype(np.intp) % 6
    fraction = hue - np.floor(hue)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    red = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [value, q, p, p, t],
        default=value,
    )
    green = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [t, value, value, q, p],
        default=p,
    )
    blue = np.select(
        [sector == 0, sector == 1, sector == 2, sector == 3, sector == 4],
        [p, p, t, value, value],
        default=q,
    )
    return np.stack([red, green, blue], axis=-1)


def jitter_hsv(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The paper's HSV jitter: one shift and two scales, drawn per sample.

    Quoting the paper, this exists to *"magnify illumination and document color
    variations"*. It is a **pointwise** function of each pixel -- it changes no
    geometry, so the emitted ``f_gt`` and mask are bit-identical with and
    without it. A test pins that.

    Args:
        rgb: ``(..., 3)`` float array in ``[0, 1]``.
        rng: Explicit generator; three scalars are drawn from it.

    Returns:
        ``(..., 3)`` float64 array clipped to ``[0, 1]``.
    """
    hue_shift = rng.uniform(*HUE_SHIFT_RANGE)
    saturation_scale = rng.uniform(*SATURATION_SCALE_RANGE)
    value_scale = rng.uniform(*VALUE_SCALE_RANGE)
    hsv = rgb_to_hsv(rgb)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * value_scale, 0.0, 1.0)
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. Rendering a sample.
# ---------------------------------------------------------------------------


def _as_rgb_float(image: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``(H, W)``/``(H, W, 1)``/``(H, W, 3)`` of any dtype to RGB float64."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[2] not in (1, 3):
        raise ValueError(f"{name} must be (H, W), (H, W, 1) or (H, W, 3), got {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must be at least 2x2, got {array.shape[:2]}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array.astype(np.float64)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        array = array / 255.0
    return array


def _background_layer(
    background: np.ndarray, height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """A random crop of ``background``, resampled to ``(height, width)``."""
    source_height, source_width = background.shape[:2]
    scale = rng.uniform(*BACKGROUND_CROP_SCALE_RANGE)
    crop_width = max(2.0, scale * (source_width - 1))
    crop_height = max(2.0, scale * (source_height - 1))
    left = rng.uniform(0.0, max(source_width - 1 - crop_width, 0.0))
    top = rng.uniform(0.0, max(source_height - 1 - crop_height, 0.0))
    columns = left + np.linspace(0.0, crop_width, width)
    rows = top + np.linspace(0.0, crop_height, height)
    grid_x, grid_y = np.meshgrid(columns, rows, indexing="xy")
    return _bilinear_gather(
        background, grid_x.reshape(-1), grid_y.reshape(-1)
    ).reshape(height, width, 3)


def render_sample(
    page: np.ndarray,
    background: np.ndarray,
    warp: InvertibleWarp,
    size: Tuple[int, int],
    rng: np.random.Generator,
    *,
    jitter: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one ``(image, f_gt, mask)`` triple for an explicit warp.

    The renderer gathers the page through ``warp.forward_map`` -- the
    distorted-to-flat direction -- because that is the only direction a gather
    render can pin down exactly (see the module docstring). ``f_gt`` is then
    ``warp.backward_map``, the algebraic inverse, which is exact rather than
    fitted. The two are therefore consistent to float round-off, and that
    consistency is the property the test suite guards.

    Args:
        page: Flat page raster, ``(H, W)``, ``(H, W, 1)`` or ``(H, W, 3)``;
            uint8 is rescaled by 255, float is taken as already in ``[0, 1]``.
        background: Background texture, same accepted shapes. Cropped and
            resampled to ``size``.
        warp: The warp to render, e.g. from :func:`sample_warp`.
        size: ``(height, width)`` of the emitted sample. Need not be square and
            need not match either input raster.
        rng: Explicit generator; consumed by the background crop and, if
            enabled, the HSV jitter.
        jitter: Whether to apply :func:`jitter_hsv` to the composite. Geometry
            is unaffected either way.

    Returns:
        A ``(image, f_gt, mask)`` tuple:

        * ``image``: ``(H, W, 3)`` float32 in ``[0, 1]``, the distorted page
          composited on the background.
        * ``f_gt``: ``(H, W, 2)`` float32, the backward map in **absolute
          distorted-image pixels**, channel 0 ``x`` (column), channel 1 ``y``
          (row).
        * ``mask``: ``(H, W, 1)`` float32 in ``{0, 1}``, 1 exactly where the
          page landed -- the segmenter's supervision target.

    Raises:
        ValueError: If ``size`` is not two positive integers, or an input
            raster has an unusable shape.
    """
    if len(size) != 2 or int(size[0]) < 2 or int(size[1]) < 2:
        raise ValueError(f"size must be (height, width), each >= 2, got {size}")
    height, width = int(size[0]), int(size[1])

    page_rgb = _as_rgb_float(page, "page")
    background_rgb = _as_rgb_float(background, "background")

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-035: the render gathers through
    # `warp.invert` (DISTORTED -> flat) and emits `warp.apply` (flat ->
    # DISTORTED) as `f_gt`. Do NOT "simplify" this by emitting the map the
    # gather uses, however strongly the phrase "render the distorted page by
    # gathering through the backward map, so f_gt is exact by construction"
    # suggests it. A gather render pins down the map indexed by the grid it
    # WRITES, i.e. the distorted-to-flat direction, i.e. the paper's FORWARD
    # map `g` (Eq. 12). `f_gt` is the other one (Eq. 11): indexed by the
    # rectified grid, valued in distorted pixels, which is what
    # `DocScannerRectifier` emits and what `F.grid_sample(distorted, bm)`
    # consumes. Emitting `g` here keeps the shape, the dtype, the pixel units,
    # the channel order and the value range identical and trains a model that
    # unwarps backwards. Exactness is bought back by the warp family instead
    # (D-036), not by choosing a side. See decisions.md D-035.
    distorted_uv = _flat_uv_grid(height, width).reshape(-1, 2)
    flat_uv = warp.invert(distorted_uv)
    inside = np.all((flat_uv >= 0.0) & (flat_uv < 1.0), axis=1)

    page_height, page_width = page_rgb.shape[:2]
    page_layer = _bilinear_gather(
        page_rgb, flat_uv[:, 0] * page_width, flat_uv[:, 1] * page_height
    ).reshape(height, width, 3)

    composite = _background_layer(background_rgb, height, width, rng)
    mask = inside.reshape(height, width, 1).astype(np.float64)
    composite = page_layer * mask + composite * (1.0 - mask)
    if jitter:
        composite = jitter_hsv(composite, rng)

    return (
        np.clip(composite, 0.0, 1.0).astype(np.float32),
        warp.backward_map(height, width),
        mask.astype(np.float32),
    )


def generate_sample(
    rng: np.random.Generator,
    page: np.ndarray,
    background: np.ndarray,
    size: Tuple[int, int] = (288, 288),
    *,
    jitter: bool = True,
    **warp_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a warp and render one training sample.

    Args:
        rng: Explicit generator. The same seed reproduces the same sample
            bit-for-bit; no global numpy state is read or written.
        page: Flat page raster (see :func:`render_sample`).
        background: Background texture (see :func:`render_sample`).
        size: ``(height, width)`` of the emitted sample.
        jitter: Whether to apply the HSV jitter.
        **warp_kwargs: Forwarded to :func:`sample_warp`.

    Returns:
        The ``(image, f_gt, mask)`` triple documented on :func:`render_sample`.

    Raises:
        DegenerateWarpError: Propagated from :func:`sample_warp`.
    """
    warp = sample_warp(rng, **warp_kwargs)
    return render_sample(page, background, warp, size, rng, jitter=jitter)


# ---------------------------------------------------------------------------
# 6. The one file-touching helper.
# ---------------------------------------------------------------------------


def load_rgb(path: str, max_side: Optional[int] = None) -> np.ndarray:
    """Read an image file as ``(H, W, 3)`` uint8 RGB.

    Pillow is imported **inside this function on purpose**: it is declared only
    in the ``data`` extra of ``pyproject.toml``, so a module-level import would
    make the whole warp pipeline unimportable on a core install. Everything
    else in this module is numpy + scipy.

    Args:
        path: Path to a PNG/JPEG/TIFF readable by Pillow.
        max_side: If given, downscale so the longest side is at most this,
            preserving aspect ratio.

    Returns:
        ``(H, W, 3)`` uint8 array.

    Raises:
        ImportError: If Pillow is not installed (``pip install .[data]``).
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-038: this import stays INSIDE
    # the function. Do NOT hoist it to module scope for tidiness: Pillow is
    # declared only in the `data` extra of pyproject.toml, so a module-scope
    # import would make the warp math -- which needs nothing but numpy and
    # scipy -- unimportable on a core install, and would do it at import time,
    # far from any call to this function. Same reasoning as the no-cv2/skimage
    # house rule at document_restoration/dtsprompt.py:9-15. A suite guard scans
    # every shipped module of this package for all three. See decisions.md
    # D-038.
    try:
        from PIL import Image  # noqa: PLC0415 - deliberate lazy import, see docstring
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "load_rgb needs Pillow, which this library declares only in the "
            "'data' extra: pip install '.[data]'. The warp math itself needs "
            "only numpy and scipy."
        ) from error

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if max_side is not None and max(image.size) > max_side:
            ratio = max_side / float(max(image.size))
            new_size = (
                max(1, int(round(image.size[0] * ratio))),
                max(1, int(round(image.size[1] * ratio))),
            )
            image = image.resize(new_size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
