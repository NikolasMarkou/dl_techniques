"""Spatial autocorrelation of a map laid out on a grid or a cortical mesh.

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

#: The offset pairs each contiguity counts, as ``(row_step, col_step)``. Each
#: undirected edge appears once; ``morans_i_grid`` multiplies by two, so the
#: weight matrix it implies is symmetric.
QUEEN_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
ROOK_OFFSETS = ((0, 1), (1, 0))

#: Queen contiguity is the eight cells sharing an edge or a corner, which for a
#: square grid is exactly a 3x3 structuring element.
QUEEN_STRUCTURE = np.ones((3, 3), dtype=bool)
ROOK_STRUCTURE = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

#: Connectivity names accepted by :func:`grid_weights` and
#: :func:`morans_i_grid`.
CONNECTIVITIES = ("queen", "rook")


def _offset_set(connectivity: str) -> Tuple[Tuple[int, int], ...]:
    """Resolve a connectivity name to its offset pairs."""
    if connectivity == "queen":
        return QUEEN_OFFSETS
    if connectivity == "rook":
        return ROOK_OFFSETS
    raise ValueError(
        f"connectivity must be one of {list(CONNECTIVITIES)}, got {connectivity!r}"
    )


def _as_2d_map(grid: np.ndarray, name: str = "grid") -> np.ndarray:
    """Validate that a value argument is a 2-D map."""
    grid = np.asarray(grid, dtype="float64")
    if grid.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D (height, width) map, got shape {grid.shape}"
        )
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
    """Build the binary neighbour matrix of a rectangular grid.

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
    """Moran's I of a 2-D map, without forming an ``N x N`` matrix.

    The general form allocates a dense ``N x N`` weight matrix; at a 28x28 grid
    that is 614 656 entries, and at a 32 492-vertex mesh it is a gigabyte. This
    walks the four (or two) offset classes instead and never materialises it.

    :param grid: ``(height, width)`` values. Never threshold this.
    :type grid: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :return: The statistic, or ``nan`` when undefined (a 1x1 grid, or a constant
        map).
    :rtype: float
    :raises ValueError: If ``grid`` is not 2-D or ``connectivity`` is unknown.
    """
    grid = _as_2d_map(grid)
    offsets = _offset_set(connectivity)
    height, width = grid.shape

    centred = grid - grid.mean()
    denominator = float(np.sum(centred ** 2))
    if denominator == 0.0:
        return float("nan")

    cross = 0.0
    weight = 0
    for row_step, col_step in offsets:
        # Slice the two overlapping windows of one offset class. Each (dy, dx)
        # pair contributes one neighbour per cell, and the transpose supplies
        # the other direction, hence the factor of two.
        rows_a = slice(0, height - row_step)
        cols_a = slice(max(0, -col_step), width - max(0, col_step))
        rows_b = slice(row_step, height)
        cols_b = slice(max(0, col_step), width - max(0, -col_step))
        block_a = centred[rows_a, cols_a]
        block_b = centred[rows_b, cols_b]
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
    :func:`morans_i_grid` -- the 3x3 structuring element -- so an island is a set
    of cells the training loss would have scored as neighbours.

    :param mask: 2-D boolean map.
    :type mask: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Islands smaller than this are relabelled to ``0``.
    :type min_island_size: int
    :return: ``(labels, count)`` where ``labels`` is an int array with ``0`` for
        unlabelled and ``1..count`` for the retained islands.
    :rtype: Tuple[numpy.ndarray, int]
    :raises ValueError: If ``mask`` is not 2-D, ``connectivity`` is unknown, or
        ``min_island_size`` is below ``1``.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be a 2-D map, got shape {mask.shape}"
        )
    if min_island_size < 1:
        raise ValueError(
            f"min_island_size must be >= 1, got {min_island_size}"
        )

    structure = (
        QUEEN_STRUCTURE if connectivity == "queen" else ROOK_STRUCTURE
    )
    if connectivity not in CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be one of {list(CONNECTIVITIES)}, "
            f"got {connectivity!r}"
        )

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

    :param t_grid: ``(height, width)`` contrast values, unthresholded.
    :type t_grid: numpy.ndarray
    :param sig_grid: ``(height, width)`` boolean significance map.
    :type sig_grid: numpy.ndarray
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Islands smaller than this are ignored entirely.
    :type min_island_size: int
    :return: The mean over islands, or ``nan`` when no island qualifies -- the
        honest answer, and what the paper reports as "not applicable".
    :rtype: float
    :raises ValueError: If the two maps disagree in shape.
    """
    t_grid = _as_2d_map(t_grid, "t_grid")
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
    full_weights = grid_weights(*t_grid.shape, connectivity=connectivity)
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
) -> dict:
    """Both Moran statistics for one contrast map, plus the input's shape.

    Bundled because the paper always reports the pair together: the standard
    value for a map the reader can see, and the islands value for a map where
    the significant set is small enough to need it.

    :param t_grid: ``(height, width)`` contrast values, unthresholded.
    :type t_grid: numpy.ndarray
    :param sig_grid: Optional ``(height, width)`` significance map. Omitting it
        makes :func:`islands_morans_i` ``nan`` rather than silently scoring the
        whole grid as one island.
    :type sig_grid: Optional[numpy.ndarray]
    :param connectivity: One of :data:`CONNECTIVITIES`.
    :type connectivity: str
    :param min_island_size: Smallest island the islands statistic will score.
    :type min_island_size: int
    :return: ``{"standard", "islands", "num_units", "connectivity"}``.
    :rtype: dict
    """
    t_grid = _as_2d_map(t_grid)
    if sig_grid is None:
        islands = float("nan")
    else:
        islands = islands_morans_i(
            t_grid, sig_grid, connectivity=connectivity,
            min_island_size=min_island_size,
        )
    return {
        "standard": morans_i_grid(t_grid, connectivity=connectivity),
        "islands": islands,
        "num_units": int(t_grid.size),
        "connectivity": connectivity,
    }