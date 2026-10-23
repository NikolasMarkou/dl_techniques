"""Spatial autocorrelation of a map laid out on a 2-D or 3-D grid or a cortical mesh.

What this measures
------------------
Moran's I answers one question about a value map: do neighbouring positions hold
similar values more often than distant ones?

.. code-block:: text

              N
        I = ----  sum_i sum_j w_ij (x_i - xbar)(x_j - xbar)
              W    ---------------------------------------------
                             sum_i (x_i - xbar)^2

``w_ij`` is 1 for neighbours and 0 otherwise, and ``W = sum w_ij``. Positive ``I``
means similar values cluster; negative means they alternate; and the value
expected of a map with no spatial structure at all is ``-1 / (N - 1)``, not
zero. A map scoring zero is therefore slightly *clustered* relative to chance.

Two rules that decide what these numbers mean
---------------------------------------------
**Queen contiguity, not rook.** Queen counts the eight cells sharing an edge or
a corner; rook counts only the four sharing an edge. Queen is the l-infinity
neighbourhood, which is the metric the spatial loss trains with, so the
measurement and the objective agree about what "nearby" means. Using rook here
would make the reported clustering a statement about a different neighbourhood
than the one that was optimised.

**Two ranks: 2-D ``(height, width)`` and 3-D ``(depth, height, width)``.** The grid
functions take a map of either rank. In 3-D the same two rules hold with the
cube in place of the square: queen is the 26 cells of the 3x3x3 l-infinity ball
around a cell (the neighbourhood a 3-D spatial loss trains with), rook is the 6
that share a face. The weights stay binary and unnormalised, ``W = sum w_ij``.
Any other rank is refused by name. The 2-D results are bit-identical to what
the 2-D-only code returned: the 2-D offset order is kept, because a different
summation order moves the last bits.

**Unthresholded maps only.** Thresholding a contrast map produces contiguous
patches of zero, and a patch of zeros is a patch of agreement, so the statistic
jumps without any change in the underlying organisation. Every function here
takes the raw map.

When a model yields few significant units, the standard statistic is dominated by
the non-significant remainder. :func:`islands_morans_i` is the alternative: it
scores each connected island of significant same-sign cells on its own mean and
its own internal adjacency, then averages across islands unweighted. It is
reported alongside the standard value, never instead of it.

These are plain NumPy functions, not ``keras.metrics.Metric``: each needs the
whole map at once, so none reduces to a streaming ``update_state``.

References:
    - Moran, 1950. The variance of the mean-autocorrelation product.
      Journal of the Royal Statistical Society B 12(3).
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse
from scipy import ndimage

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: Map ranks the grid functions accept: ``(height, width)`` and
#: ``(depth, height, width)``.
GRID_RANKS = (2, 3)

#: Connectivity names accepted by :func:`grid_weights`, :func:`grid_weights_nd`
#: and :func:`morans_i_grid`.
CONNECTIVITIES = ("queen", "rook")

# DECISION plan-2026-10-23T190557-972a9ac5/D-003: the 2-D tuples keep this exact
# order and are NOT rebuilt from ``itertools.product``, and ``grid_weights``
# is NOT rewritten to delegate to ``grid_weights_nd``. The generated order is
# (0,1),(1,-1),(1,0),(1,1): it sums the same terms in a different order and
# moves ``morans_i_grid`` by 1.2e-18 on a 28x28 map (measured), which breaks the
# bit-identical 2-D contract that ``TestMoransIGoldenPins`` pins as float.hex
# literals. Rank 3 got its own tables for that reason, beside these.

#: The offset tuples each contiguity counts, one per undirected edge (the first
#: non-zero component is positive); ``morans_i_grid`` multiplies by two, so the
#: weight matrix it implies is symmetric. 2-D tuples are
#: ``(row_step, col_step)``; the 3-D tables are ``(depth, row, col)`` steps and
#: hold 13 (queen) and 3 (rook) tuples.
QUEEN_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
ROOK_OFFSETS = ((0, 1), (1, 0))
QUEEN_OFFSETS_3D = (
    (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
)
ROOK_OFFSETS_3D = ((0, 0, 1), (0, 1, 0), (1, 0, 0))

_OFFSET_TABLES = {
    (2, "queen"): QUEEN_OFFSETS,
    (2, "rook"): ROOK_OFFSETS,
    (3, "queen"): QUEEN_OFFSETS_3D,
    (3, "rook"): ROOK_OFFSETS_3D,
}


def _require_connectivity(connectivity: str) -> None:
    """Raise the shared ``connectivity must be one of`` error for an unknown name."""
    if connectivity not in CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be one of {list(CONNECTIVITIES)}, "
            f"got {connectivity!r}"
        )


def connectivity_structure(ndim: int, connectivity: str = "queen") -> np.ndarray:
    """The boolean structuring element of a contiguity, for any rank.

    Queen is the full ``3**ndim`` block (all faces, edges and corners), rook the
    face neighbours only. This is the one place both metrics modules take their
    structure from, so a labelled island and a Moran neighbour pair can never
    disagree about what "adjacent" means. In 2-D it equals
    :data:`QUEEN_STRUCTURE` / :data:`ROOK_STRUCTURE`.

    :param ndim: Number of grid axes (``>= 1``).
    :type ndim: int
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :return: A ``(3,) * ndim`` boolean array, centre included.
    :rtype: numpy.ndarray
    :raises ValueError: If ``connectivity`` is unknown.
    """
    _require_connectivity(connectivity)
    return ndimage.generate_binary_structure(
        ndim, ndim if connectivity == "queen" else 1
    )


#: Queen contiguity is the eight cells sharing an edge or a corner, which for a
#: square grid is exactly a 3x3 structuring element. Derived from
#: :func:`connectivity_structure`, so the 2-D constants and the N-d helper cannot
#: drift apart.
QUEEN_STRUCTURE = connectivity_structure(2, "queen")
ROOK_STRUCTURE = connectivity_structure(2, "rook")


def require_grid_rank(shape: Sequence[int], name: str = "grid") -> None:
    """Raise a named error unless ``shape`` has one of the supported ranks.

    The single home of the rank rule (:data:`GRID_RANKS`) and of its message,
    which every grid function in this module and in ``topographic_selectivity``
    uses, so the allowed ranks are written down once.

    :param shape: Shape of the map being validated.
    :type shape: Sequence[int]
    :param name: Argument name, for the message.
    :type name: str
    :raises ValueError: If ``len(shape)`` is not in :data:`GRID_RANKS`. The
        message names the argument and the offending shape.
    """
    if len(shape) not in GRID_RANKS:
        raise ValueError(
            f"{name} must be a 2-D (height, width) map or a 3-D "
            f"(depth, height, width) map, got shape {tuple(shape)}"
        )


def _offset_set(connectivity: str, ndim: int = 2) -> Tuple[Tuple[int, ...], ...]:
    """Resolve a connectivity name and a rank to its offset tuples."""
    _require_connectivity(connectivity)
    return _OFFSET_TABLES[(ndim, connectivity)]


def _as_grid_map(grid: np.ndarray, name: str = "grid") -> np.ndarray:
    """Validate that a value argument is a 2-D or 3-D map, as ``float64``."""
    grid = np.asarray(grid, dtype="float64")
    require_grid_rank(grid.shape, name)
    return grid


def morans_i(values: np.ndarray, weights: scipy.sparse.spmatrix) -> float:
    """Moran's I of a vector under an explicit weight matrix.

    The general form; :func:`morans_i_grid` is the shape-exploiting version and
    must agree with this one, which is what makes it checkable.

    :param values: ``(num_positions,)`` map values.
    :type values: numpy.ndarray
    :param weights: Sparse weight matrix, symmetric, zero diagonal.
    :type weights: scipy.sparse.spmatrix
    :return: The statistic, or ``nan`` when it is undefined (no weights, or a map
        with no variance).
    :rtype: float
    :raises ValueError: If the two lengths disagree.
    """
    values = np.asarray(values, dtype="float64").reshape(-1)
    num_positions = values.shape[0]
    if weights.shape[0] != num_positions:
        raise ValueError(
            f"weights is {weights.shape[0]}x{weights.shape[1]} but values has "
            f"length {num_positions}"
        )

    centred = values - values.mean()
    denominator = float(np.sum(centred ** 2))
    total_weight = float(weights.sum())
    if denominator == 0.0 or total_weight == 0.0:
        return float("nan")

    numerator = float(centred @ (weights @ centred))
    return (num_positions / total_weight) * numerator / denominator


def grid_weights(
    height: int,
    width: int,
    connectivity: str = "queen",
) -> scipy.sparse.csr_matrix:
    """Build the binary neighbour matrix of a rectangular 2-D grid.

    The 2-D entry point, kept with its ``(height, width, connectivity)``
    signature because ``connectivity`` is its third positional argument; a 3-D
    grid goes through :func:`grid_weights_nd`. The two builders are separate
    bodies, and ``grid_weights_nd((height, width), c)`` is pinned equal to this
    one array for array.

    :param height: Number of grid rows.
    :type height: int
    :param width: Number of grid columns.
    :type width: int
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :return: A symmetric sparse matrix with a zero diagonal.
    :rtype: scipy.sparse.csr_matrix
    :raises ValueError: If either dimension is not positive, or ``connectivity``
        is unknown.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"grid dimensions must be positive, got {height}x{width}"
        )
    offsets = _offset_set(connectivity)

    rows: List[int] = []
    cols: List[int] = []
    for row in range(height):
        for col in range(width):
            index = row * width + col
            for row_step, col_step in offsets:
                neighbour = (row + row_step, col + col_step)
                if 0 <= neighbour[0] < height and 0 <= neighbour[1] < width:
                    rows.append(index)
                    cols.append(neighbour[0] * width + neighbour[1])

    shape = (height * width, height * width)
    forward = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=shape
    )
    return (forward + forward.T).tocsr()


def grid_weights_nd(
    shape: Sequence[int],
    connectivity: str = "queen",
) -> scipy.sparse.csr_matrix:
    """Build the binary neighbour matrix of a 2-D or 3-D rectangular grid.

    Cells are numbered row-major (C order), so cell ``(i, j, k)`` of a
    ``(D, H, W)`` grid is ``(i * H + j) * W + k`` -- the order of
    ``numpy.ravel`` and of ``SpatialLayout.cell_to_unit``. Queen gives 26
    neighbours to an interior 3-D cell (7 to a corner), rook gives 6. For a
    2-D ``shape`` the returned CSR arrays equal those of :func:`grid_weights`.

    :param shape: ``(height, width)`` or ``(depth, height, width)``.
    :type shape: Sequence[int]
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :return: A symmetric sparse matrix with a zero diagonal, ``float64`` data.
    :rtype: scipy.sparse.csr_matrix
    :raises ValueError: If ``shape`` does not have 2 or 3 axes, an extent is not
        positive, or ``connectivity`` is unknown.
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) not in GRID_RANKS:
        raise ValueError(
            f"shape must have 2 or 3 axes, got {shape}"
        )
    if any(n <= 0 for n in shape):
        raise ValueError(
            f"grid dimensions must be positive, got {'x'.join(map(str, shape))}"
        )
    offsets = _offset_set(connectivity, len(shape))

    num_cells = int(np.prod(shape))
    index = np.arange(num_cells).reshape(shape)
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for offset in offsets:
        # The same two overlapping windows as ``morans_i_grid``: the cells that
        # have a neighbour at ``offset``, and those neighbours.
        source = tuple(
            slice(max(0, -step), size - max(0, step))
            for step, size in zip(offset, shape)
        )
        target = tuple(
            slice(max(0, step), size - max(0, -step))
            for step, size in zip(offset, shape)
        )
        rows.append(index[source].reshape(-1))
        cols.append(index[target].reshape(-1))

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    forward = scipy.sparse.coo_matrix(
        (np.ones(row.size), (row, col)), shape=(num_cells, num_cells)
    )
    return (forward + forward.T).tocsr()


def mesh_weights(
    edges: Sequence[Tuple[int, int]],
    num_vertices: int,
) -> scipy.sparse.csr_matrix:
    """Build the binary adjacency of a surface mesh from an edge list.

    A cortical surface is not a rectangle: HCP meshes give each vertex six
    neighbours, and their arrangement differs across the sheet. The grid form
    cannot express that, so the fMRI arm of this analysis takes its adjacency
    from the mesh's own topology.

    :param edges: ``(num_edges, 2)`` pairs of connected vertex indices, in
        either order.
    :type edges: Sequence[Tuple[int, int]]
    :param num_vertices: Number of vertices in the mesh.
    :type num_vertices: int
    :return: A symmetric sparse adjacency matrix.
    :rtype: scipy.sparse.csr_matrix
    :raises ValueError: If ``num_vertices`` is not positive, an edge index is
        out of range, or an edge joins a vertex to itself.
    """
    if num_vertices <= 0:
        raise ValueError(
            f"num_vertices must be positive, got {num_vertices}"
        )

    edges = np.asarray(edges, dtype="int64").reshape(-1, 2)
    if edges.size and (
        edges.min() < 0 or edges.max() >= num_vertices
    ):
        raise ValueError(
            f"edge indices must lie in [0, {num_vertices}), got range "
            f"[{edges.min()}, {edges.max()}]"
        )
    if edges.size and np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("an edge cannot join a vertex to itself")

    shape = (num_vertices, num_vertices)
    forward = scipy.sparse.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=shape
    )
    return (forward + forward.T).tocsr()


def morans_i_grid(grid: np.ndarray, connectivity: str = "queen") -> float:
    """Moran's I of a 2-D or 3-D map, without forming an ``N x N`` matrix.

    The general form allocates a dense ``N x N`` weight matrix; at a 28x28 grid
    that is 614 656 entries, and at a 32 492-vertex mesh it is a gigabyte. This
    walks the offset classes instead (four queen / two rook in 2-D, thirteen /
    three in 3-D) and never materialises it.

    :param grid: ``(height, width)`` or ``(depth, height, width)`` values. Never
        threshold this.
    :type grid: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :return: The statistic, or ``nan`` when undefined (a single cell, or a
        constant map).
    :rtype: float
    :raises ValueError: If ``grid`` is not 2-D or 3-D, or ``connectivity`` is
        unknown.
    """
    grid = _as_grid_map(grid)
    offsets = _offset_set(connectivity, grid.ndim)

    centred = grid - grid.mean()
    denominator = float(np.sum(centred ** 2))
    if denominator == 0.0:
        return float("nan")

    cross = 0.0
    weight = 0
    for offset in offsets:
        # Slice the two overlapping windows of one offset class, one slice per
        # axis. Each offset contributes one neighbour per cell, and the
        # transpose supplies the other direction, hence the factor of two.
        window_a = tuple(
            slice(max(0, -step), size - max(0, step))
            for step, size in zip(offset, grid.shape)
        )
        window_b = tuple(
            slice(max(0, step), size - max(0, -step))
            for step, size in zip(offset, grid.shape)
        )
        block_a = centred[window_a]
        block_b = centred[window_b]
        cross += 2.0 * float(np.sum(block_a * block_b))
        weight += 2 * block_a.size

    if weight == 0:
        return float("nan")
    return (centred.size / weight) * cross / denominator


def morans_i_permutation_test(
    values: np.ndarray,
    weights: scipy.sparse.spmatrix,
    num_permutations: int = 9999,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Test Moran's I against spatial randomness by permutation.

    Parametric significance assumes the values are drawn independently, which a
    contrast map never is -- its units share an upstream layer. Permutation
    breaks the spatial structure without touching the value distribution, which
    is the assumption that actually holds.

    :param values: ``(num_positions,)`` map values.
    :type values: numpy.ndarray
    :param weights: Sparse weight matrix, as for :func:`morans_i`.
    :type weights: scipy.sparse.spmatrix
    :param num_permutations: Number of shuffles. Default 9999, so the smallest
        attainable p-value is ``1 / 10000``.
    :type num_permutations: int
    :param seed: Seed for the shuffles, so the p-value is reproducible.
    :type seed: Optional[int]
    :return: ``(observed_I, p_value)``. ``nan`` observations propagate.
    :rtype: Tuple[float, float]
    :raises ValueError: If ``num_permutations`` is not positive.
    """
    if num_permutations <= 0:
        raise ValueError(
            f"num_permutations must be positive, got {num_permutations}"
        )

    values = np.asarray(values, dtype="float64").reshape(-1)
    observed = morans_i(values, weights)
    if np.isnan(observed):
        return observed, float("nan")

    rng = np.random.default_rng(seed)
    at_least_as_extreme = 0
    for _ in range(num_permutations):
        if morans_i(rng.permutation(values), weights) >= observed:
            at_least_as_extreme += 1

    p_value = (at_least_as_extreme + 1) / (num_permutations + 1)
    logger.debug(
        f"Moran's I permutation test: observed={observed:.4f}, "
        f"p={p_value:.5f} over {num_permutations} permutations"
    )
    return observed, float(p_value)


def label_islands(
    mask: np.ndarray,
    connectivity: str = "queen",
    min_island_size: int = 1,
) -> Tuple[np.ndarray, int]:
    """Label the connected components of a boolean map.

    Queen contiguity here means the same thing it means in
    :func:`morans_i_grid` -- the full ``3**ndim`` structuring element, from
    :func:`connectivity_structure` -- so an island is a set of cells the training
    loss would have scored as neighbours. In 3-D, queen joins cells that touch
    at a corner and rook only those that share a face.

    :param mask: 2-D or 3-D boolean map.
    :type mask: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Islands smaller than this are relabelled to ``0``.
    :type min_island_size: int
    :return: ``(labels, count)`` where ``labels`` is an int array with ``0`` for
        unlabelled and ``1..count`` for the retained islands.
    :rtype: Tuple[numpy.ndarray, int]
    :raises ValueError: If ``mask`` is not 2-D or 3-D, ``connectivity`` is
        unknown, or ``min_island_size`` is below ``1``.
    """
    mask = np.asarray(mask)
    require_grid_rank(mask.shape, "mask")
    if min_island_size < 1:
        raise ValueError(
            f"min_island_size must be >= 1, got {min_island_size}"
        )

    structure = connectivity_structure(mask.ndim, connectivity)

    labels, count = ndimage.label(mask, structure=structure)
    if count and min_island_size > 1:
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
        too_small = {
            i + 1 for i, size in enumerate(sizes) if size < min_island_size
        }
        if too_small:
            labels = np.where(np.isin(labels, list(too_small)), 0, labels)
            # Renumber to 1..count. `ndimage.label` numbers the islands it finds,
            # so dropping one leaves a gap -- and a caller looping
            # `for island in range(1, count + 1)` would then read label `count`
            # correctly only by luck. Relabelling is what makes the documented
            # "1..count" contract true.
            survivors = np.unique(labels[labels > 0])
            if survivors.size == 0:
                return np.zeros_like(labels, dtype="int64"), 0
            remap = np.zeros(int(labels.max()) + 1, dtype="int64")
            remap[survivors] = np.arange(1, survivors.size + 1, dtype="int64")
            labels = remap[labels]
            count = int(survivors.size)
    return labels.astype("int64"), int(count)


def islands_morans_i(
    t_grid: np.ndarray,
    sig_grid: np.ndarray,
    connectivity: str = "queen",
    min_island_size: int = 3,
) -> float:
    """Average Moran's I computed within each island of significant cells.

    The standard statistic mixes two populations: cells that carry a real signal
    and cells that are noise. When a model yields few significant cells, that
    second population dominates and the result says more about how small the
    significant set is than about whether it clusters. Scoring each island on
    its own mean and its own internal adjacency removes the mixture.

    The average is **unweighted by island size**: weighting would let the
    largest island stand in for every other one, which is the same domination by
    a different route.

    :param t_grid: ``(height, width)`` or ``(depth, height, width)`` contrast
        values, unthresholded.
    :type t_grid: numpy.ndarray
    :param sig_grid: Boolean significance map of the same shape.
    :type sig_grid: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Islands smaller than this are ignored entirely.
    :type min_island_size: int
    :return: The mean over islands, or ``nan`` when no island qualifies -- the
        honest answer, and what the paper reports as "not applicable".
    :rtype: float
    :raises ValueError: If ``t_grid`` is not 2-D or 3-D, or the two maps disagree
        in shape.
    """
    t_grid = _as_grid_map(t_grid, "t_grid")
    sig_grid = np.asarray(sig_grid, dtype=bool)
    if sig_grid.shape != t_grid.shape:
        raise ValueError(
            f"sig_grid shape {sig_grid.shape} does not match t_grid shape "
            f"{t_grid.shape}"
        )

    labels, count = label_islands(
        sig_grid, connectivity=connectivity, min_island_size=min_island_size
    )
    if count == 0:
        return float("nan")

    scores = []
    # Built once, not once per island: the restricted sub-matrix is a cheap slice
    # of the same sparse structure.
    full_weights = grid_weights_nd(t_grid.shape, connectivity)
    for island in range(1, count + 1):
        member = labels == island
        values = t_grid[member]
        if values.size < 2:
            continue
        # Only the island's own cells, its own mean, and adjacency restricted to
        # pairs inside it. A 3x3 window straddling the island boundary would
        # score a significant cell against a non-significant one, which is the
        # mixture this statistic exists to remove.
        restricted = _restrict_weights(
            full_weights, member.reshape(-1), values.size
        )
        score = morans_i(values, restricted)
        if not np.isnan(score):
            scores.append(score)

    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _restrict_weights(
    full_weights: scipy.sparse.spmatrix,
    member: np.ndarray,
    num_members: int,
) -> scipy.sparse.csr_matrix:
    """Slice a weight matrix down to the positions flagged by ``member``."""
    indices = np.flatnonzero(member)
    restricted = full_weights.tocsr()[indices][:, indices]
    return scipy.sparse.csr_matrix(restricted)


def morans_i_summary(
    t_grid: np.ndarray,
    sig_grid: Optional[np.ndarray] = None,
    connectivity: str = "queen",
    min_island_size: int = 3,
    num_permutations: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """All three Moran statistics for one contrast map, plus the input's shape.

    Bundled because the paper always reports them together: the standard value
    for a map the reader can see, the islands value for a map where the
    significant set is small enough to need it, and the permutation p-value for
    the question neither of the other two answers -- whether the standard value
    exceeds what the same value multiset would produce with its spatial
    arrangement destroyed.

    The permutation statistic is OPT-IN via ``num_permutations`` because it costs
    that many shuffles of the map, and the weight matrix is rebuilt per tap here
    rather than cached: this function's contract is one map, and a cache keyed on
    grid shape would outlive the caller holding it. Callers scoring 24 taps
    should build the weights once and call :func:`morans_i_permutation_test`
    directly.

    :param t_grid: ``(height, width)`` or ``(depth, height, width)`` contrast
        values, unthresholded.
    :type t_grid: numpy.ndarray
    :param sig_grid: Optional significance map of the same shape. Omitting it
        makes :func:`islands_morans_i` ``nan`` rather than silently scoring the
        whole grid as one island.
    :type sig_grid: Optional[numpy.ndarray]
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Smallest island the islands statistic will score.
    :type min_island_size: int
    :param num_permutations: Shuffles behind the permutation p-value, or ``None``
        to skip it and report ``nan``. The p-value's floor is
        ``1 / (num_permutations + 1)``.
    :type num_permutations: Optional[int]
    :param seed: Seed for the shuffles, so the p-value is reproducible.
    :type seed: Optional[int]
    :return: ``{"standard", "islands", "permutation_p", "num_permutations",
        "num_units", "connectivity"}``.
    :rtype: dict
    :raises ValueError: If ``t_grid`` is not 2-D or 3-D, or ``num_permutations``
        is not positive and not ``None``.
    """
    t_grid = _as_grid_map(t_grid, "t_grid")
    if sig_grid is None:
        islands = float("nan")
    else:
        islands = islands_morans_i(
            t_grid, sig_grid, connectivity=connectivity,
            min_island_size=min_island_size,
        )

    if num_permutations is None:
        permutation_p = float("nan")
    else:
        _, permutation_p = morans_i_permutation_test(
            t_grid,
            grid_weights_nd(t_grid.shape, connectivity),
            num_permutations=num_permutations,
            seed=seed,
        )

    return {
        "standard": morans_i_grid(t_grid, connectivity=connectivity),
        "islands": islands,
        "permutation_p": float(permutation_p),
        "num_permutations": num_permutations,
        "num_units": int(t_grid.size),
        "connectivity": connectivity,
    }