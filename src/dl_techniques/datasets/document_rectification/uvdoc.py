"""UVDoc ground-truth reader and densifier for the DocScanner rectification port.

UVDoc (Verhoeven et al., 2023) is the one openly downloadable dewarping corpus
with real captured geometry: 20,000 renders over 4,032 pseudo-photorealistic
page deformations, 488x712 px, no registration wall. It is the corpus this port
trains on alongside
:mod:`~dl_techniques.datasets.document_rectification.synthetic_warp`.

What UVDoc actually ships, MEASURED from the archive (not assumed)
------------------------------------------------------------------
Read directly out of ``UVDoc_final.zip`` (29,538,315,521 B, 182,492 members --
half of them ``__MACOSX/`` resource forks that must be skipped)::

    UVDoc_final/grid2d/<geom>.mat            4032   MATLAB v7.3 (HDF5)
    UVDoc_final/grid3d/<geom>.mat            4032   MATLAB v7.3 (HDF5)
    UVDoc_final/uvmap/<geom>.mat             4032   MATLAB v7.3 (HDF5)
    UVDoc_final/seg/<geom>.mat               4032   MATLAB v7.3 (HDF5)
    UVDoc_final/wc/<geom>.exr                4032
    UVDoc_final/img_geom/<geom>.png          4032
    UVDoc_final/metadata_geom/<geom>.json    4032
    UVDoc_final/img/<sample>.png            20000
    UVDoc_final/warped_textures/<sample>.png 20000
    UVDoc_final/metadata_sample/<sample>.json 20000
    UVDoc_final/textures/*.png               3009
    UVDoc_final/split.json                       1

Every ``.mat`` is **MATLAB v7.3, i.e. HDF5** -- ``scipy.io.loadmat`` raises
``NotImplementedError: Please use HDF reader for matlab v7.3 files``, so this
module reads them with ``h5py`` (lazily imported; see below). h5py returns a
MATLAB array **transposed**, because MATLAB writes column-major; the fix is the
same one the repository already applies at
``datasets/graphs/fraud.py:238-246`` (``_read_mat_h5py`` / ``_h5_dense``), and
this module reuses that pattern rather than inventing a second one.

``grid2d`` is the ground truth that matters here. Raw h5py shape ``(2, 61, 89)``
float64, i.e. a MATLAB ``(89, 61, 2)`` array: **89 rows x 61 columns of control
points, channel 0 = x (column), channel 1 = y (row), values in absolute pixels
of the distorted 488x712 image**. The lattice is uniform in the *rectified*
page's normalised coordinates: control point ``(r, c)`` sits at
``(u, v) = (c / 60, r / 88)`` of the flat page.

That was verified, not assumed, against a second UVDoc artefact: ``uvmap``
(dense per-pixel ``(u, v)`` of the distorted image, NaN off-page) read at the
pixel each control point names returns that control point's own ``(c/60,
r/88)`` to ~2e-3. So **``grid2d`` is already the paper's backward map ``f_gt``**
(Eq. 11) -- indexed by the rectified grid, valued in distorted pixels -- merely
sampled at 89x61 instead of per pixel. It needs densifying, not inverting. The
plan's H-11 ("a coarse grid, not a dense backward map") is right about the
resolution and, importantly, right-way-round about the direction.

The two operations, and which one is approximate
------------------------------------------------
:func:`densify_backward_map` is a **regular-grid** interpolation: the control
lattice is a rectangle in the rectified domain, so this is a tensor-product
cubic spline (``scipy.interpolate.RectBivariateSpline``, ``s=0``), not a
scattered-data fit. No TPS, no ``RBFInterpolator``. Because the spline
interpolates, it reproduces the control points it was fitted to at ~5e-13 px --
which makes "the densified map reproduces its own control points" a guard that
**cannot fail**, and this module does not pretend otherwise. The number that
carries information is the **held-out** error: fit on two thirds of the
lattice, predict the dropped third. MEASURED over 300 random geometries:
median 0.44 px, p95 0.99 px, p99 1.68 px, max 2.015 px, with a per-sample mean
of 0.07-0.21 px. Against an *exact* reference (a closed-form
:class:`~dl_techniques.datasets.document_rectification.synthetic_warp.InvertibleWarp`
sampled on the same 89x61 lattice) the full densification error at 288x288 is
max 0.26-1.09 px, mean 0.003-0.010 px.

:func:`invert_backward_map` is **the one genuinely approximate step in this
port**. A coarse correspondence grid has no analytic inverse, so ``g`` (Eq. 12,
read only by the ``L_line`` circle-consistency term) is obtained by
``scipy.interpolate.griddata`` over the densified map's own output points --
linear inside the point cloud's convex hull, nearest-neighbour outside it.
MEASURED at 288x288: round trip ``||g(f(x)) - x||`` mean 0.009 px, interior max
0.09 px (median over 40 geometries; worst 0.48), whole-grid max 1.07 px median
/ 5.98 px worst. Against the exact forward map of a synthetic warp, on the page
with a 2-px erosion: mean 0.004-0.014 px, p99 0.09-0.28 px, max 0.31-1.75 px.

Note the asymmetry deliberately kept: ``synthetic_warp`` has NO approximate
step at all (both directions are closed form, D-036). Only UVDoc pays for its
realism with an inversion, and only in the auxiliary term.

Units and channel order -- the SAME contract as ``synthetic_warp``
------------------------------------------------------------------
``f_gt`` is ``(H, W, 2)`` float32, absolute pixel coordinates of the distorted
image, channel 0 = ``x`` (column), channel 1 = ``y`` (row), domain = the
rectified pixel grid. ``g`` is ``(H, W, 2)`` float32, absolute pixel
coordinates of the *rectified* image, same channel order, domain = the
distorted pixel grid. Both grids come from
:func:`~dl_techniques.datasets.document_restoration.dtsprompt.base_coordinate_grid`
-- the repository's one place where "channel 0 is x" and "normalise each axis
by its own extent" are decided (D-021 of the DocRes plan) -- multiplied by
``(W, H)``. They are not re-derived here, and
``tests/test_datasets/test_document_rectification/backward_map_convention.py``
asserts the two modules' contracts with ONE shared assertion applied to both.

Resampling convention: every resize in this module (image, mask, and the pixel
scaling of ``grid2d``) uses the same ``source = u * source_extent`` rule that
``base_coordinate_grid`` and ``synthetic_warp``'s renderer use, NOT the
half-pixel rule ``PIL.Image.resize`` and ``cv2.resize`` use. The two differ by
``(source_extent / target_extent - 1) / 2`` source pixels -- 0.74 px at
712 -> 288 -- applied identically to the image, the mask and the map, so the
emitted triple stays mutually registered. Mixing the two conventions would
misregister the image against ``f_gt`` by that amount with no shape symptom.

Dependencies
------------
numpy + scipy at module scope, exactly like ``synthetic_warp``. ``h5py`` (the
``.mat`` v7.3 reader) and ``Pillow`` (the PNG decoder) are imported **inside**
the functions that need them: both are declared only in this project's ``data``
extra, so a module-scope import would make the whole warp/densification stack
unimportable on a core install. ``zipfile`` and ``json`` are stdlib.

Public surface:
    * :class:`UVDocSource` -- reads members out of the ``.zip`` *or* an
      extracted directory, without extracting the 27.5 GB archive.
    * :func:`read_grid2d`, :func:`read_segmentation`, :func:`read_uvmap` --
      the three ground-truth artefacts, transposed into repo convention.
    * :func:`control_point_uv` -- the lattice's rectified-domain coordinates.
    * :func:`densify_backward_map` -- coarse ``grid2d`` to per-pixel ``f_gt``.
    * :func:`invert_backward_map` -- ``f_gt`` to ``g`` (the approximate step).
    * :func:`held_out_densification_error`, :func:`roundtrip_error`,
      :func:`assess_densification`, :class:`UVDocQuality`, :func:`is_unusable`,
      :class:`UVDocDensificationError` -- the accuracy bound and the rejection
      path.
    * :func:`load_geometry`, :class:`UVDocGeometry`, :func:`load_image` -- the
      per-sample entry points.

References:
    - Verhoeven, Hoffmann, Kim, 2023. UVDoc: Neural Grid-based Document
      Unwarping. SIGGRAPH Asia. (https://igl.ethz.ch/projects/uvdoc/)
    - Feng et al., 2021. DocScanner (https://arxiv.org/abs/2110.14968), v2,
      Eq. 11 (``f_gt``) and Eq. 12 (``g``).
"""

import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline, griddata
from scipy.spatial import QhullError

from dl_techniques.datasets.document_restoration.dtsprompt import base_coordinate_grid
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# 1. Archive layout. Every string here was read out of the staged archive; none
#    is a guess. The member counts are in the module docstring.
# ---------------------------------------------------------------------------

#: Top-level directory inside ``UVDoc_final.zip``.
UVDOC_ROOT: str = "UVDoc_final"

#: Resource-fork prefix macOS's zip tooling injects. Half of the archive's
#: 182,492 members live under it and every one of them must be skipped: they
#: carry the same names as the real members and would double every listing.
MACOSX_PREFIX: str = "__MACOSX/"

#: Sub-directory names, keyed by what they hold.
GRID2D_DIR: str = "grid2d"
SEG_DIR: str = "seg"
UVMAP_DIR: str = "uvmap"
IMAGE_DIR: str = "img"
SAMPLE_METADATA_DIR: str = "metadata_sample"
GEOMETRY_METADATA_DIR: str = "metadata_geom"

#: HDF5 dataset names inside each ``.mat`` (measured: one dataset per file, and
#: the ``uvmap`` one is called ``uv``, not ``uvmap``).
GRID2D_KEY: str = "grid2d"
SEG_KEY: str = "seg"
UVMAP_KEY: str = "uv"

#: The control lattice UVDoc actually ships, MEASURED on the archive: 89 rows
#: by 61 columns. Read from each file rather than assumed -- this constant
#: exists so a change in the corpus is *reported*, not silently absorbed.
UVDOC_GRID_ROWS: int = 89
UVDOC_GRID_COLS: int = 61

#: Rendered frame size, MEASURED: 488 wide by 712 tall.
UVDOC_IMAGE_WIDTH: int = 488
UVDOC_IMAGE_HEIGHT: int = 712

# ---------------------------------------------------------------------------
# 2. The interpolation and its accuracy budget. Every threshold below is a
#    MEASURED number with stated headroom, not a round figure.
# ---------------------------------------------------------------------------

#: Tensor-product spline degree used to densify the control lattice. Cubic in
#: both axes; the lattice is 89x61, so degree 3 is far from the size limit.
DENSIFY_SPLINE_DEGREE: int = 3

#: Border, in rectified pixels, excluded from the *interior* round-trip metric.
#: The whole-grid maximum is dominated by an inherent artefact -- the rectified
#: frame's outermost ring maps onto the boundary of the inverted point cloud,
#: where the linear interpolant gives way to the nearest-neighbour fill -- so a
#: threshold on it would mostly measure the hull edge. See D-040.
ROUNDTRIP_BORDER_PIXELS: int = 2

#: Held-out densification bound, in target pixels at 288x288. MEASURED over 300
#: random geometries: median 0.44 px, p95 0.99 px, p99 1.68 px, max 2.015 px --
#: a distribution with a real tail, so the bound is set at 4.0, twice the
#: largest of 300, which rejects 0 of them. Note this metric is deliberately
#: PESSIMISTIC about production: it fits on two thirds of the lattice, while
#: `densify_backward_map` uses all of it.
MAX_HELD_OUT_DENSIFICATION_PIXELS: float = 4.0

#: Interior round-trip bound, in pixels. MEASURED at 288x288: 0.48 px worst of
#: 40 real geometries, 0.57 px worst of 6 deliberately violent synthetic warps.
#: 1.5 leaves ~2.6x headroom on the worst measurement of either population.
MAX_INTERIOR_ROUNDTRIP_PIXELS: float = 1.5

#: Fraction of the distorted frame that may fall outside the inverted point
#: cloud's convex hull before the sample is suspect. MEASURED: 0.383-0.489 over
#: 40 geometries -- the page simply does not fill the frame. A value near 1
#: means the densified map collapsed.
MAX_HULL_MISS_FRACTION: float = 0.85


class UVDocError(RuntimeError):
    """Base class for every error this module raises."""


class UVDocDensificationError(UVDocError):
    """A sample whose densified map or its inverse missed the accuracy bound.

    Raised by :func:`load_geometry` in its default strict mode. The message
    names which measured metric failed and by how much, so a rejection is
    reportable rather than merely countable.
    """


# ---------------------------------------------------------------------------
# 3. Reading the archive. Nothing here extracts the 27.5 GB zip.
# ---------------------------------------------------------------------------


class UVDocSource:
    """Member reader for a staged UVDoc corpus, zipped or extracted.

    Interface contract (this class has more than one caller: the densifier
    entry points here, and the staging/pipeline code of the later plan steps):

    * ``path`` is either ``UVDoc_final.zip`` or a directory that *contains*
      ``UVDoc_final/`` or *is* ``UVDoc_final/``. All three are accepted so a
      caller never has to know which form the data landed in.
    * :meth:`read_bytes` takes a path RELATIVE to ``UVDoc_final/`` (e.g.
      ``"grid2d/00_00000_1_0_0.mat"``) and returns raw bytes, or raises
      :class:`UVDocError` naming the member.
    * The archive is opened read-only and is never written, extracted or
      renamed. Reading a 71 KB ``grid2d`` member out of the 27.5 GB zip costs a
      seek, not a decompression pass over the archive.
    * Usable as a context manager; :meth:`close` is idempotent.

    Args:
        path: Archive file or directory root.

    Raises:
        UVDocError: If ``path`` is neither a readable zip nor a directory that
            contains a recognisable UVDoc tree.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._zip: Optional[zipfile.ZipFile] = None
        self._root_dir: Optional[str] = None

        if os.path.isdir(self.path):
            candidate = os.path.join(self.path, UVDOC_ROOT)
            self._root_dir = candidate if os.path.isdir(candidate) else self.path
            if not os.path.isdir(os.path.join(self._root_dir, GRID2D_DIR)):
                raise UVDocError(
                    f"{self.path!r} is a directory but holds no "
                    f"{UVDOC_ROOT}/{GRID2D_DIR}/ -- point UVDocSource at the "
                    f"staged UVDoc root or at UVDoc_final.zip itself"
                )
        elif os.path.isfile(self.path):
            try:
                self._zip = zipfile.ZipFile(self.path)
            except zipfile.BadZipFile as error:
                raise UVDocError(
                    f"{self.path!r} is not a readable zip archive"
                ) from error
        else:
            raise UVDocError(f"no such UVDoc archive or directory: {self.path!r}")

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the archive handle. Idempotent."""
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> "UVDocSource":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- member access -----------------------------------------------------

    def _zip_name(self, relative: str) -> str:
        return f"{UVDOC_ROOT}/{relative}"

    def exists(self, relative: str) -> bool:
        """Whether a member relative to ``UVDoc_final/`` is present."""
        if self._zip is not None:
            try:
                self._zip.getinfo(self._zip_name(relative))
            except KeyError:
                return False
            return True
        assert self._root_dir is not None
        return os.path.isfile(os.path.join(self._root_dir, relative))

    def read_bytes(self, relative: str) -> bytes:
        """Read one member, relative to ``UVDoc_final/``.

        Args:
            relative: e.g. ``"grid2d/00_00000_1_0_0.mat"``.

        Returns:
            The member's raw bytes.

        Raises:
            UVDocError: If the member is absent.
        """
        if self._zip is not None:
            try:
                return self._zip.read(self._zip_name(relative))
            except KeyError as error:
                raise UVDocError(
                    f"member {self._zip_name(relative)!r} is not in {self.path!r}"
                ) from error
        assert self._root_dir is not None
        full = os.path.join(self._root_dir, relative)
        if not os.path.isfile(full):
            raise UVDocError(f"missing UVDoc file: {full!r}")
        with open(full, "rb") as handle:
            return handle.read()

    def _listing(self, directory: str, suffix: str) -> List[str]:
        if self._zip is not None:
            prefix = f"{UVDOC_ROOT}/{directory}/"
            # DECISION plan-2026-09-10T065432-05fcb6dd/D-039: the `__MACOSX/`
            # filter is load-bearing, not tidiness. UVDoc_final.zip holds
            # 182,492 members and EXACTLY HALF are macOS resource forks under
            # `__MACOSX/UVDoc_final/...`, carrying the same basenames as the
            # real members. Do NOT drop this filter or replace it with a plain
            # `name.endswith(suffix)` scan: every listing would double, every
            # id would appear twice, and a resource fork opened as a `.mat`
            # fails inside h5py with an unrelated-looking HDF5 signature error
            # far from the listing that produced it. Anchoring the filter on a
            # startswith of the REAL prefix (rather than a blacklist of the
            # fork prefix) is what makes this robust to a future re-zip. See
            # decisions.md D-039.
            names = [
                name[len(prefix) :]
                for name in self._zip.namelist()
                if name.startswith(prefix) and name.endswith(suffix)
            ]
        else:
            assert self._root_dir is not None
            folder = os.path.join(self._root_dir, directory)
            if not os.path.isdir(folder):
                return []
            names = [
                name for name in os.listdir(folder) if name.endswith(suffix)
            ]
        return sorted(name[: -len(suffix)] for name in names if name != suffix)

    def geometry_names(self) -> List[str]:
        """Sorted geometry ids, i.e. the basenames under ``grid2d/``."""
        return self._listing(GRID2D_DIR, ".mat")

    def sample_ids(self) -> List[str]:
        """Sorted render ids, i.e. the basenames under ``img/``."""
        return self._listing(IMAGE_DIR, ".png")

    def geometry_for_sample(self, sample_id: str) -> str:
        """The geometry id a render was produced from.

        Twenty thousand renders share 4,032 geometries; the mapping lives in
        each render's own ``metadata_sample/<id>.json`` under ``geom_name``.

        Args:
            sample_id: e.g. ``"00000"``.

        Returns:
            The geometry id, e.g. ``"00_00000_1_0_0"``.

        Raises:
            UVDocError: If the metadata is missing or has no ``geom_name``.
        """
        raw = self.read_bytes(f"{SAMPLE_METADATA_DIR}/{sample_id}.json")
        metadata = json.loads(raw)
        if "geom_name" not in metadata:
            raise UVDocError(
                f"metadata for sample {sample_id!r} has no 'geom_name' key; "
                f"keys are {sorted(metadata)}"
            )
        return str(metadata["geom_name"])

    def geometry_metadata(self, geometry_name: str) -> Dict[str, Any]:
        """The per-geometry JSON (deformation flags, camera intrinsics)."""
        return json.loads(
            self.read_bytes(f"{GEOMETRY_METADATA_DIR}/{geometry_name}.json")
        )


def _read_mat_dataset(payload: bytes, key: str) -> np.ndarray:
    """Read one dataset out of MATLAB-v7.3 ``.mat`` bytes.

    ``h5py`` is imported here rather than at module scope: it is declared only
    in this project's ``data`` extra (same footing as Pillow), and the warp and
    densification math itself needs nothing but numpy and scipy. The repository
    already reads v7.3 ``.mat`` files exactly this way at
    ``datasets/graphs/fraud.py:238-246``; this is that pattern applied to bytes
    instead of a path, so a member can be read straight out of the zip.

    The returned array is **not** transposed here -- each caller states its own
    axis mapping, because "h5py gives you the MATLAB array reversed" is exactly
    the kind of fact that is invisible on a square array.

    Args:
        payload: Raw ``.mat`` bytes.
        key: HDF5 dataset name.

    Returns:
        The dataset as a numpy array, in h5py's (reversed) axis order.

    Raises:
        ImportError: If h5py is not installed.
        UVDocError: If ``key`` is absent from the file.
    """
    try:
        import h5py  # noqa: PLC0415 - deliberate lazy import, see docstring
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "reading UVDoc ground truth needs h5py: every UVDoc .mat is "
            "MATLAB v7.3 (HDF5) and scipy.io.loadmat refuses it. h5py is "
            "declared in the 'data' extra: pip install '.[data]'. The "
            "densification math itself needs only numpy and scipy."
        ) from error

    with h5py.File(io.BytesIO(payload), "r") as handle:
        if key not in handle:
            raise UVDocError(
                f"expected dataset {key!r} in this .mat; found {list(handle)}"
            )
        return np.asarray(handle[key][()])


def read_grid2d(source: UVDocSource, geometry_name: str) -> np.ndarray:
    """UVDoc's coarse backward-map control lattice, in repo convention.

    Args:
        source: An open :class:`UVDocSource`.
        geometry_name: e.g. ``"00_00000_1_0_0"``.

    Returns:
        ``(rows, cols, 2)`` float64. ``[r, c]`` is the ``(x, y)`` pixel of the
        distorted image that the rectified page's ``(c / (cols - 1),
        r / (rows - 1))`` point landed on. Channel 0 is ``x``.

    Raises:
        UVDocError: If the member is missing or has an unexpected rank.
    """
    raw = _read_mat_dataset(
        source.read_bytes(f"{GRID2D_DIR}/{geometry_name}.mat"), GRID2D_KEY
    )
    if raw.ndim != 3 or raw.shape[0] != 2:
        raise UVDocError(
            f"grid2d for {geometry_name!r} has h5py shape {raw.shape}; "
            f"expected (2, cols, rows)"
        )
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-039: this transpose is
    # (2, cols, rows) -> (rows, cols, 2), and it is NOT the identity-looking
    # `.transpose(1, 2, 0)` that a glance suggests. h5py hands back the MATLAB
    # array with ALL axes reversed (MATLAB is column-major), so the raw
    # (2, 61, 89) is a MATLAB (89, 61, 2): 89 ROWS of 61 columns. MEASURED
    # against UVDoc's own dense `uvmap`: channel 0 sweeps [58, 433] against a
    # 488-wide frame while sweeping raw axis 1 (length 61), channel 1 sweeps
    # [83, 615] against a 712-tall frame while sweeping raw axis 2 (length 89).
    # Getting this wrong on a NON-square lattice raises a shape error, which is
    # why 89x61 is safe -- but the same mistake on a square corpus, or on the
    # square 288x288 training grid downstream, is completely silent and
    # transposes every page. See decisions.md D-039.
    grid = np.transpose(raw, (2, 1, 0)).astype(np.float64)
    if grid.shape[0] < 4 or grid.shape[1] < 4:
        raise UVDocError(
            f"grid2d for {geometry_name!r} is {grid.shape[:2]}; a cubic "
            f"tensor-product spline needs at least 4x4 control points"
        )
    return grid


def read_segmentation(source: UVDocSource, geometry_name: str) -> np.ndarray:
    """The page mask of the distorted frame.

    Args:
        source: An open :class:`UVDocSource`.
        geometry_name: Geometry id.

    Returns:
        ``(H, W)`` uint8 in ``{0, 1}``, 1 on the page.
    """
    raw = _read_mat_dataset(
        source.read_bytes(f"{SEG_DIR}/{geometry_name}.mat"), SEG_KEY
    )
    if raw.ndim != 2:
        raise UVDocError(
            f"seg for {geometry_name!r} has h5py shape {raw.shape}; expected 2-D"
        )
    return np.ascontiguousarray(raw.T).astype(np.uint8)


def read_uvmap(source: UVDocSource, geometry_name: str) -> np.ndarray:
    """UVDoc's dense per-pixel ``(u, v)`` map of the distorted frame.

    This is UVDoc's own rendering of the FORWARD direction, and this module
    does not use it to produce ``g``: it is NaN off the page (so it cannot be
    composed by ``L_line``, which samples wherever the predicted flow lands)
    and it exists only at the corpus's native 488x712. It is exposed because it
    is an **independent oracle** for :func:`invert_backward_map` -- an artefact
    produced by UVDoc's renderer, not by anything in this file. MEASURED
    agreement on the page interior: mean 0.52 px, p99 0.65-0.69 px.

    Args:
        source: An open :class:`UVDocSource`.
        geometry_name: Geometry id.

    Returns:
        ``(H, W, 2)`` float64 with ``u`` in channel 0, both in ``[0, 1]``, and
        NaN wherever the frame shows no page.
    """
    raw = _read_mat_dataset(
        source.read_bytes(f"{UVMAP_DIR}/{geometry_name}.mat"), UVMAP_KEY
    )
    if raw.ndim != 3 or raw.shape[0] != 2:
        raise UVDocError(
            f"uvmap for {geometry_name!r} has h5py shape {raw.shape}; "
            f"expected (2, W, H)"
        )
    return np.transpose(raw, (2, 1, 0)).astype(np.float64)


# ---------------------------------------------------------------------------
# 4. The two maps.
# ---------------------------------------------------------------------------


def rectified_pixel_grid(height: int, width: int) -> np.ndarray:
    """The rectified frame's own pixel coordinates, ``(H, W, 2)`` float64.

    Delegates to
    :func:`~dl_techniques.datasets.document_restoration.dtsprompt.base_coordinate_grid`
    and multiplies by ``(width, height)`` -- the same two lines
    ``synthetic_warp`` uses to turn its normalised grid into pixels. Re-used,
    not re-derived: the channel-order and per-axis-normalisation decisions
    (D-021 of the DocRes plan) live in that one function.

    Args:
        height: Grid height.
        width: Grid width.

    Returns:
        ``(height, width, 2)`` float64; ``[r, c]`` is ``(c, r)``.
    """
    normalised = base_coordinate_grid(height, width).astype(np.float64)
    return normalised * np.array([width, height], dtype=np.float64)


def control_point_uv(rows: int, cols: int) -> np.ndarray:
    """Rectified-domain coordinates of a ``rows x cols`` control lattice.

    Spans ``[0, 1]`` **inclusive** at both ends -- the lattice's outer ring sits
    on the page's own corners and edges. That is a different span from
    ``base_coordinate_grid``'s half-open ``[0, 1)`` pixel grid, and the
    difference is not a bug in either: one indexes control points, the other
    indexes pixels. Verified against UVDoc's ``uvmap``, which returns
    ``(c / (cols - 1), r / (rows - 1))`` at each control point's pixel.

    Args:
        rows: Lattice rows.
        cols: Lattice columns.

    Returns:
        ``(rows, cols, 2)`` float64, channel 0 ``u`` (along the width).

    Raises:
        ValueError: If either extent is below 2.
    """
    if rows < 2 or cols < 2:
        raise ValueError(f"a control lattice needs at least 2x2, got {(rows, cols)}")
    u_axis = np.linspace(0.0, 1.0, cols)
    v_axis = np.linspace(0.0, 1.0, rows)
    u_plane, v_plane = np.meshgrid(u_axis, v_axis, indexing="xy")
    return np.stack([u_plane, v_plane], axis=-1)


def _spline_pair(
    grid: np.ndarray,
    v_axis: np.ndarray,
    u_axis: np.ndarray,
) -> Tuple[RectBivariateSpline, RectBivariateSpline]:
    """One interpolating tensor-product spline per channel of a lattice."""
    degree = DENSIFY_SPLINE_DEGREE
    kx = min(degree, len(v_axis) - 1)
    ky = min(degree, len(u_axis) - 1)
    return tuple(  # type: ignore[return-value]
        RectBivariateSpline(v_axis, u_axis, grid[..., channel], kx=kx, ky=ky, s=0)
        for channel in range(2)
    )


def densify_backward_map(
    grid2d: np.ndarray,
    height: int,
    width: int,
    *,
    source_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Interpolate a coarse control lattice into a per-pixel ``f_gt``.

    Args:
        grid2d: ``(rows, cols, 2)`` control points from :func:`read_grid2d`,
            valued in pixels of the distorted frame.
        height: Target ``H`` -- the rectified grid this map is indexed by.
        width: Target ``W``.
        source_size: ``(H_src, W_src)`` of the frame ``grid2d`` is valued in.
            When given and different from ``(height, width)``, the values are
            rescaled by ``(width / W_src, height / H_src)`` so the emitted map
            addresses an image resampled to ``(height, width)`` under the same
            ``source = u * source_extent`` rule the rest of this module uses.
            Pass ``None`` when the values are already in target pixels.

    Returns:
        ``(height, width, 2)`` float32 ``f_gt``: absolute distorted-image
        pixels, channel 0 ``x``, indexed by the rectified pixel grid.

    Raises:
        ValueError: On a malformed lattice or a degenerate target size.
    """
    grid = np.asarray(grid2d, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError(f"grid2d must be (rows, cols, 2), got {grid.shape}")
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        raise ValueError(f"grid2d needs at least a 2x2 lattice, got {grid.shape[:2]}")
    if height < 2 or width < 2:
        raise ValueError(f"target size must be at least 2x2, got {(height, width)}")

    if source_size is not None:
        source_height, source_width = int(source_size[0]), int(source_size[1])
        if source_height < 1 or source_width < 1:
            raise ValueError(f"source_size must be positive, got {source_size}")
        grid = grid * np.array(
            [width / source_width, height / source_height], dtype=np.float64
        )

    rows, cols = grid.shape[:2]
    v_axis = np.linspace(0.0, 1.0, rows)
    u_axis = np.linspace(0.0, 1.0, cols)
    query = base_coordinate_grid(height, width).astype(np.float64)
    query_v = query[:, 0, 1]
    query_u = query[0, :, 0]

    splines = _spline_pair(grid, v_axis, u_axis)
    dense = np.stack(
        [spline(query_v, query_u) for spline in splines], axis=-1
    )
    return dense.astype(np.float32)


def invert_backward_map(
    f_gt: np.ndarray,
    *,
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a dense backward map into the forward map ``g``.

    **This is the one genuinely approximate operation in the DocScanner data
    pipeline.** ``synthetic_warp`` needs nothing like it (its warp family is
    closed-form invertible, D-036); UVDoc's correspondence grid has no analytic
    inverse, so ``g`` is fitted here and its error is measured rather than
    assumed -- see :func:`assess_densification`.

    Method: ``f_gt`` is read as a scattered correspondence
    ``distorted_pixel -> rectified_pixel`` and interpolated onto the distorted
    pixel grid with ``scipy.interpolate.griddata``. Linear (Delaunay) inside
    the convex hull of the page's image; nearest-neighbour outside it, where no
    page pixel exists and any value is an extrapolation. The ``outside`` mask
    is returned rather than hidden so a caller can exclude those pixels or
    reject the sample.

    Args:
        f_gt: ``(H, W, 2)`` backward map in absolute distorted-image pixels.
        subsample: Take every ``subsample``-th row and column of ``f_gt`` as
            fit points. 1 (the default) is the accurate setting; 2 is ~4x
            faster and MEASURED ~2x worse on the round trip. Never above the
            point where the page's footprint stops being sampled.

    Returns:
        ``(g, outside)`` where ``g`` is ``(H, W, 2)`` float32 in absolute
        rectified-image pixels (channel 0 ``x``), indexed by the distorted
        pixel grid, and ``outside`` is ``(H, W)`` bool, True where the value
        came from the nearest-neighbour fill.

    Raises:
        ValueError: On a malformed map or a non-positive ``subsample``.
        UVDocDensificationError: If the map is so degenerate that the linear
            interpolant covers nothing at all.
    """
    dense = np.asarray(f_gt, dtype=np.float64)
    if dense.ndim != 3 or dense.shape[2] != 2:
        raise ValueError(f"f_gt must be (H, W, 2), got {dense.shape}")
    if dense.shape[0] < 2 or dense.shape[1] < 2:
        raise ValueError(f"f_gt needs at least 2x2 samples, got {dense.shape[:2]}")
    if not np.isfinite(dense).all():
        raise ValueError("f_gt holds non-finite values; nothing to invert")
    step = int(subsample)
    if step < 1:
        raise ValueError(f"subsample must be >= 1, got {subsample}")

    height, width = dense.shape[:2]
    rectified = rectified_pixel_grid(height, width)
    points = dense[::step, ::step].reshape(-1, 2)
    values = rectified[::step, ::step].reshape(-1, 2)
    target = (rectified[..., 0], rectified[..., 1])

    try:
        linear = griddata(points, values, target, method="linear")
    except QhullError as error:
        # A collapsed map (every control point on one line, or all identical)
        # does not come back as NaN -- Qhull refuses to triangulate it at all,
        # with a message about cocircular input that names neither this module
        # nor the sample. Translated here so the rejection path sees the same
        # exception type it sees for a merely inaccurate sample.
        raise UVDocDensificationError(
            f"the backward map is degenerate -- its image cannot be "
            f"triangulated (collapsed to a line or a point): {error}"
        ) from error
    outside = ~np.isfinite(linear[..., 0])
    if outside.all():
        raise UVDocDensificationError(
            "the linear inversion covered no pixel at all: the backward map's "
            "image is degenerate (collapsed to a line or a point)"
        )
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-040: the NaN region is filled
    # with the NEAREST fit value and reported through `outside`; it is not left
    # NaN and not clipped to the frame. Do NOT "clean this up" either way. The
    # page covers only ~56% of a UVDoc frame (MEASURED hull miss 0.383-0.489),
    # so ~44% of `g` is outside the page and has no true value at all: leaving
    # NaN there propagates into the L_line term the first time a predicted flow
    # lands off-page and poisons the whole loss with no traceable origin, while
    # silently clipping invents a plausible in-frame coordinate that reads as
    # real. Nearest-fill plus an explicit mask is the only one of the three
    # that is both finite and honest. It is ALSO why the shipped accuracy gate
    # is the INTERIOR round trip: the whole-grid maximum is dominated by the
    # ring where linear gives way to nearest (MEASURED 5.98 px worst vs 0.48 px
    # interior over the same 40 geometries), so gating on it would reject good
    # samples for an inherent hull-edge artefact. See decisions.md D-040.
    if outside.any():
        nearest = griddata(points, values, target, method="nearest")
        linear = np.where(np.isfinite(linear), linear, nearest)

    return linear.astype(np.float32), outside


# ---------------------------------------------------------------------------
# 5. The accuracy bound and the rejection path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UVDocQuality:
    """Measured accuracy of one sample's densification and inversion.

    Attributes:
        held_out_densification_pixels: Largest error, in target pixels, when
            half the control lattice is dropped and predicted from the other
            half. This is the metric with power. Reproducing the control points
            the spline was FITTED to is not: an interpolating spline returns
            them at ~5e-13 px by construction, so that check cannot fail.
        held_out_densification_mean: Mean of the same errors.
        interior_roundtrip_pixels: Largest ``||g(f(x)) - x||`` over the
            rectified grid excluding a :data:`ROUNDTRIP_BORDER_PIXELS` border.
            The shipped gate.
        roundtrip_pixels: The same maximum over the whole grid, border
            included. Reported, deliberately NOT gated -- see D-040.
        roundtrip_mean_pixels: Mean over the whole grid.
        hull_miss_fraction: Fraction of the distorted frame that fell outside
            the inverted point cloud's convex hull.
    """

    held_out_densification_pixels: float
    held_out_densification_mean: float
    interior_roundtrip_pixels: float
    roundtrip_pixels: float
    roundtrip_mean_pixels: float
    hull_miss_fraction: float


def held_out_densification_error(
    grid2d: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Cross-validate the densifier on control points it never saw.

    Holds out a random subset of lattice rows and columns, fits the same
    interpolating spline to the rest, and predicts the held-out points. The
    subset is drawn from ``rng``, so the metric is deterministic under an
    explicit generator without being one fixed sub-lattice.

    Two properties of the split are load-bearing:

    * whole rows and columns are held out, not a scatter of individual points,
      because a tensor-product spline is fitted on a rectangular lattice; but
    * the fit axes need only be strictly increasing, NOT uniformly spaced,
      which is what allows a random subset rather than a fixed parity.

    Args:
        grid2d: ``(rows, cols, 2)`` control lattice, in whatever pixel units
            the caller wants the error reported in.
        rng: Explicit generator; no global numpy state is read or written.

    Returns:
        ``(max_error, mean_error)`` in the lattice's own units.

    Raises:
        ValueError: If the lattice is too small to hold anything out.
    """
    grid = np.asarray(grid2d, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[2] != 2:
        raise ValueError(f"grid2d must be (rows, cols, 2), got {grid.shape}")
    rows, cols = grid.shape[:2]
    if rows < 8 or cols < 8:
        raise ValueError(
            f"cross-validating the densifier needs at least an 8x8 lattice, "
            f"got {(rows, cols)}"
        )

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-042: the held-out set is drawn
    # from the INTERIOR only -- indices 0 and n-1 always stay in the fit set.
    # Do NOT "simplify" this to a parity split (`arange(offset, n, 2)`), which
    # is the obvious way to write it: with a random offset that split hands the
    # lattice's outer ring to the held-out set, and predicting it is
    # EXTRAPOLATION past the fit axes' span, not interpolation. MEASURED on the
    # real corpus: the same geometry scores 0.35 px held out from the interior
    # and 2.48 px when the boundary rows land in the held-out set -- a 7x
    # difference that is entirely an artefact of the split, and which pushed a
    # perfectly good sample past the rejection bound on the first run of this
    # function. The densifier is never asked to extrapolate in production
    # either: `densify_backward_map` queries u, v in [0, (n-1)/n], strictly
    # inside the lattice's [0, 1] span. See decisions.md D-042.
    interior_rows = np.arange(1, rows - 1)
    interior_cols = np.arange(1, cols - 1)
    held_rows = np.sort(
        rng.choice(interior_rows, size=max(2, len(interior_rows) // 3), replace=False)
    )
    held_cols = np.sort(
        rng.choice(interior_cols, size=max(2, len(interior_cols) // 3), replace=False)
    )
    fit_rows = np.setdiff1d(np.arange(rows), held_rows)
    fit_cols = np.setdiff1d(np.arange(cols), held_cols)

    v_axis = np.linspace(0.0, 1.0, rows)
    u_axis = np.linspace(0.0, 1.0, cols)
    sub_grid = grid[np.ix_(fit_rows, fit_cols)]
    splines = _spline_pair(sub_grid, v_axis[fit_rows], u_axis[fit_cols])

    query_u, query_v = np.meshgrid(u_axis[held_cols], v_axis[held_rows], indexing="xy")
    predicted = np.stack(
        [spline.ev(query_v, query_u) for spline in splines], axis=-1
    )
    truth = grid[np.ix_(held_rows, held_cols)]
    error = np.linalg.norm(predicted - truth, axis=-1)
    return float(error.max()), float(error.mean())


def roundtrip_error(f_gt: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Per-pixel ``||g(f(x)) - x||`` over the rectified grid.

    ``g`` is sampled at ``f_gt``'s output coordinates with an independent
    bilinear resampler (``scipy.ndimage.map_coordinates``), not with anything
    in this module, so a bug in the inversion cannot cancel itself out.

    Args:
        f_gt: ``(H, W, 2)`` backward map, distorted-image pixels.
        g: ``(H, W, 2)`` forward map, rectified-image pixels.

    Returns:
        ``(H, W)`` float64 distances in rectified pixels.

    Raises:
        ValueError: If the two maps disagree in shape.
    """
    backward = np.asarray(f_gt, dtype=np.float64)
    forward = np.asarray(g, dtype=np.float64)
    if backward.shape != forward.shape or backward.ndim != 3 or backward.shape[2] != 2:
        raise ValueError(
            f"f_gt and g must both be (H, W, 2) and agree; got "
            f"{backward.shape} and {forward.shape}"
        )
    height, width = backward.shape[:2]
    coordinates = [backward[..., 1], backward[..., 0]]
    composed = np.stack(
        [
            ndimage.map_coordinates(
                forward[..., channel], coordinates, order=1, mode="nearest"
            )
            for channel in range(2)
        ],
        axis=-1,
    )
    return np.linalg.norm(composed - rectified_pixel_grid(height, width), axis=-1)


def assess_densification(
    grid2d: np.ndarray,
    f_gt: np.ndarray,
    g: np.ndarray,
    outside: np.ndarray,
    rng: np.random.Generator,
    *,
    source_size: Optional[Tuple[int, int]] = None,
) -> UVDocQuality:
    """Measure everything :func:`is_unusable` decides on.

    Args:
        grid2d: The control lattice the map was densified from.
        f_gt: The densified backward map.
        g: Its inverse from :func:`invert_backward_map`.
        outside: The convex-hull miss mask from :func:`invert_backward_map`.
        rng: Explicit generator, forwarded to
            :func:`held_out_densification_error`.
        source_size: ``(H_src, W_src)`` if ``grid2d`` is still in source
            pixels, so the held-out error is reported in TARGET pixels like
            every other number in :class:`UVDocQuality`.

    Returns:
        The :class:`UVDocQuality` metrics.
    """
    grid = np.asarray(grid2d, dtype=np.float64)
    height, width = np.asarray(f_gt).shape[:2]
    if source_size is not None:
        grid = grid * np.array(
            [width / int(source_size[1]), height / int(source_size[0])],
            dtype=np.float64,
        )
    held_out_max, held_out_mean = held_out_densification_error(grid, rng)

    error = roundtrip_error(f_gt, g)
    border = ROUNDTRIP_BORDER_PIXELS
    if error.shape[0] > 2 * border and error.shape[1] > 2 * border:
        interior = error[border:-border, border:-border]
    else:
        interior = error
    return UVDocQuality(
        held_out_densification_pixels=held_out_max,
        held_out_densification_mean=held_out_mean,
        interior_roundtrip_pixels=float(interior.max()),
        roundtrip_pixels=float(error.max()),
        roundtrip_mean_pixels=float(error.mean()),
        hull_miss_fraction=float(np.asarray(outside).mean()),
    )


def is_unusable(quality: UVDocQuality) -> bool:
    """Whether a sample must be rejected rather than trained on.

    The policy is reject-and-report: a UVDoc geometry cannot be re-drawn the
    way a synthetic warp can, so the caller drops the sample and says so. Two
    of the four measured quantities are gated, and the choice is deliberate
    (D-040): the whole-grid round-trip maximum and the hull-miss fraction are
    dominated by the page's footprint in the frame and by the hull edge, not by
    the quality of the fit.

    Args:
        quality: Metrics from :func:`assess_densification`.

    Returns:
        ``True`` if any gated metric is out of bounds.
    """
    return bool(
        quality.held_out_densification_pixels > MAX_HELD_OUT_DENSIFICATION_PIXELS
        or quality.interior_roundtrip_pixels > MAX_INTERIOR_ROUNDTRIP_PIXELS
        or quality.hull_miss_fraction > MAX_HULL_MISS_FRACTION
    )


def _rejection_reason(quality: UVDocQuality) -> str:
    """The gated metrics that failed, with their measured values and bounds."""
    failures = []
    if quality.held_out_densification_pixels > MAX_HELD_OUT_DENSIFICATION_PIXELS:
        failures.append(
            f"held-out densification {quality.held_out_densification_pixels:.3f} px "
            f"> {MAX_HELD_OUT_DENSIFICATION_PIXELS} px"
        )
    if quality.interior_roundtrip_pixels > MAX_INTERIOR_ROUNDTRIP_PIXELS:
        failures.append(
            f"interior round trip {quality.interior_roundtrip_pixels:.3f} px "
            f"> {MAX_INTERIOR_ROUNDTRIP_PIXELS} px"
        )
    if quality.hull_miss_fraction > MAX_HULL_MISS_FRACTION:
        failures.append(
            f"hull miss {quality.hull_miss_fraction:.3f} "
            f"> {MAX_HULL_MISS_FRACTION}"
        )
    return "; ".join(failures)


# ---------------------------------------------------------------------------
# 6. Per-sample entry points.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UVDocGeometry:
    """One UVDoc geometry, densified onto a target grid.

    Attributes:
        name: The geometry id.
        f_gt: ``(H, W, 2)`` float32 backward map, distorted-image pixels,
            channel 0 ``x``, indexed by the rectified grid. Identical contract
            to ``synthetic_warp.render_sample``'s second return value.
        g: ``(H, W, 2)`` float32 forward map, rectified-image pixels, indexed
            by the distorted grid. The approximate one.
        mask: ``(H, W, 1)`` float32 in ``{0, 1}``, the page's footprint in the
            distorted frame -- the segmenter's supervision target.
        outside: ``(H, W)`` bool, True where ``g`` is a nearest-neighbour fill.
        quality: The measured accuracy of this sample.
    """

    name: str
    f_gt: np.ndarray
    g: np.ndarray
    mask: np.ndarray
    outside: np.ndarray
    quality: UVDocQuality


def _resample_nearest(source: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize under the ``source = u * source_extent`` rule."""
    source_height, source_width = source.shape[:2]
    normalised = base_coordinate_grid(height, width).astype(np.float64)
    rows = np.clip(
        np.floor(normalised[..., 1] * source_height).astype(np.int64),
        0,
        source_height - 1,
    )
    cols = np.clip(
        np.floor(normalised[..., 0] * source_width).astype(np.int64),
        0,
        source_width - 1,
    )
    return source[rows, cols]


def load_geometry(
    source: UVDocSource,
    geometry_name: str,
    rng: np.random.Generator,
    size: Tuple[int, int] = (288, 288),
    *,
    subsample: int = 1,
    strict: bool = True,
) -> UVDocGeometry:
    """Read one geometry and produce the ``(f_gt, g, mask)`` triple.

    Interface contract: this is the function the staging script and the
    training pipeline both call, and it is the ONLY place the four steps
    (read, densify, invert, assess) are composed. The emitted arrays match
    ``synthetic_warp.render_sample`` element for element in shape, dtype, units
    and channel order, so a pipeline can concatenate the two corpora without a
    per-corpus branch. The image is NOT loaded here -- see :func:`load_image` --
    because the geometry path needs only numpy/scipy/h5py while the image path
    needs Pillow, and keeping them apart is what lets the whole densification
    stack run on a core install.

    Args:
        source: An open :class:`UVDocSource`.
        geometry_name: Geometry id.
        rng: Explicit generator, used only by the held-out cross-validation.
        size: ``(height, width)`` of the emitted arrays.
        subsample: Forwarded to :func:`invert_backward_map`.
        strict: Raise on an out-of-bound sample rather than returning it.

    Returns:
        A :class:`UVDocGeometry`.

    Raises:
        UVDocDensificationError: If ``strict`` and :func:`is_unusable`.
        UVDocError: On a missing or malformed member.
    """
    height, width = int(size[0]), int(size[1])
    grid = read_grid2d(source, geometry_name)
    segmentation = read_segmentation(source, geometry_name)
    source_size = (int(segmentation.shape[0]), int(segmentation.shape[1]))

    f_gt = densify_backward_map(grid, height, width, source_size=source_size)
    g, outside = invert_backward_map(f_gt, subsample=subsample)
    quality = assess_densification(
        grid, f_gt, g, outside, rng, source_size=source_size
    )

    if is_unusable(quality):
        message = (
            f"UVDoc geometry {geometry_name!r} misses the densification bound: "
            f"{_rejection_reason(quality)}"
        )
        if strict:
            raise UVDocDensificationError(message)
        logger.warning(message)

    mask = _resample_nearest(segmentation, height, width)
    return UVDocGeometry(
        name=geometry_name,
        f_gt=f_gt,
        g=g,
        mask=mask.astype(np.float32).reshape(height, width, 1),
        outside=outside,
        quality=quality,
    )


def load_image(
    source: UVDocSource,
    sample_id: str,
    size: Tuple[int, int] = (288, 288),
) -> np.ndarray:
    """Read one render and resample it onto the target grid.

    Args:
        source: An open :class:`UVDocSource`.
        sample_id: e.g. ``"00000"``.
        size: ``(height, width)``.

    Returns:
        ``(height, width, 3)`` float32 in ``[0, 1]``.

    Raises:
        ImportError: If Pillow is not installed (``pip install '.[data]'``).
        UVDocError: If the render is missing.
    """
    # DECISION plan-2026-09-10T065432-05fcb6dd/D-041: Pillow DECODES the PNG
    # here, but it does NOT resize it -- the resampling is the same
    # `source = u * source_extent` gather the maps use. Do NOT replace this
    # with the one-liner `Image.resize((W, H), BILINEAR)`. PIL (like cv2)
    # resizes on the HALF-PIXEL convention, which differs from
    # `base_coordinate_grid`'s rule by `(source_extent / target_extent - 1) / 2`
    # source pixels -- 0.74 px at UVDoc's 712 -> 288 -- and applying it to the
    # image while the map keeps the other rule misregisters the pair by that
    # much, with no shape, dtype or range symptom and no test that fails
    # except an end-to-end rectification residual nobody attributes to a
    # resize. Same reason the mask uses `_resample_nearest`. See decisions.md
    # D-041.
    try:
        from PIL import Image  # noqa: PLC0415 - deliberate lazy import, see above
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "reading UVDoc renders needs Pillow, declared in the 'data' extra: "
            "pip install '.[data]'. The densification math needs only numpy "
            "and scipy."
        ) from error

    payload = source.read_bytes(f"{IMAGE_DIR}/{sample_id}.png")
    with Image.open(io.BytesIO(payload)) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0

    height, width = int(size[0]), int(size[1])
    source_height, source_width = rgb.shape[:2]
    normalised = base_coordinate_grid(height, width).astype(np.float64)
    coordinates = [
        normalised[..., 1] * source_height,
        normalised[..., 0] * source_width,
    ]
    resampled = np.stack(
        [
            ndimage.map_coordinates(
                rgb[..., channel], coordinates, order=1, mode="nearest"
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    return np.clip(resampled, 0.0, 1.0).astype(np.float32)
