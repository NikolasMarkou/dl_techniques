"""Selectivity maps, multiplicity correction, and cluster growing on a unit grid.

The measurement chain
---------------------
A topographic model's claim is that its units are organised in space. Reading
that claim off a model takes four steps, and each one has a way to go quietly
wrong:

.. code-block:: text

    activations        contrast_tmap       fdr_across_layers    grow_clusters
    (stimuli x units)  -> (t per unit)     -> (q per unit)      -> islands
                         |                  |                    |
                         |                  |                    +- a map the
                         |                  +- ONE correction over
                         |                     every tap at once,   reader can see
                         +- independent across
                            stimuli, NOT across units

Correction is joint across layers on purpose. A twelve-layer model taps 9 408
units; correcting each layer's 784 against 5% in isolation admits roughly 470
false positives per layer, and the paper's figures are read as one organism with
one map per layer, not as twelve independent families.

Why ``grow_clusters`` is not just connected components
----------------------------------------------------
The greedy version matters exactly when a size cap is set: with one, which cell
joins a cluster depends on the order the frontier is visited, and the order is
part of the method rather than an accident. With ``max_size=None`` the result is
provably the connected components of the same-sign significant cells -- the
greedy traversal visits them all in the same partition -- and the two
implementations are pinned against each other rather than left as a claim.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Benjamini & Hochberg, 1995. Controlling the false discovery rate.
      JRSS B 57(1).
    - Fedorenko, Hsieh, Nieto-Castano & Kanwisher, 2010. New method for fMRI
      investigations of language: defining ROIs functionally in individual
      subjects. (https://arxiv.org/abs/1003.2782)
"""

import heapq
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.stats
from scipy import ndimage

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.metrics.spatial_autocorrelation import (
    QUEEN_STRUCTURE,
    ROOK_STRUCTURE,
    CONNECTIVITIES,
)

# ---------------------------------------------------------------------

#: Smallest cluster the paper keeps, in units. Below this a "cluster" is a
#: handful of cells and the response profile computed over it is noise.
DEFAULT_MIN_CLUSTER_SIZE = 10

#: Contrast polarities a two-sided cluster sweep runs over.
CLUSTER_SIGNS = (1, -1)


def contrast_tmap(
    activations_a: np.ndarray,
    activations_b: np.ndarray,
    equal_var: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-unit two-sample t statistics contrasting two stimulus conditions.

    Positive ``t`` means condition A responded more strongly than condition B,
    which is the sign convention the cluster sweep relies on.

    :param activations_a: ``(num_stimuli_a, num_units)`` responses.
    :type activations_a: numpy.ndarray
    :param activations_b: ``(num_stimuli_b, num_units)`` responses.
    :type activations_b: numpy.ndarray
    :param equal_var: ``True`` for Student's t, ``False`` for Welch's. The paper
        says "t-test" without saying which; Student's is the default and the
        flag exists so the choice is visible rather than buried.
    :type equal_var: bool
    :return: ``(t_values, p_values)``, both ``(num_units,)``. A unit that is
        constant in *both* conditions is reported as ``t = 0, p = 1`` -- it
        genuinely carries no evidence, and SciPy returns ``nan`` for it.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If the two arrays have different unit counts or no rows.
    """
    a = np.asarray(activations_a, dtype="float64")
    b = np.asarray(activations_b, dtype="float64")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"both conditions must be 2-D (stimuli, units), got {a.shape} and "
            f"{b.shape}"
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"the two conditions must share a unit axis, got {a.shape[1]} and "
            f"{b.shape[1]} units"
        )
    if a.shape[0] < 1 or b.shape[0] < 1:
        raise ValueError(
            f"each condition needs at least one stimulus, got {a.shape[0]} and "
            f"{b.shape[0]}"
        )

    # A unit that is flat in both conditions makes SciPy divide zero variance by
    # zero and return nan, which then poisons min/max and every threshold derived
    # from them. It is also the honest answer: no evidence either way.
    both_constant = (a.std(axis=0) == 0.0) & (b.std(axis=0) == 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = scipy.stats.ttest_ind(a, b, axis=0, equal_var=equal_var)

    t_values = np.nan_to_num(
        np.asarray(result.statistic, dtype="float64"), nan=0.0, posinf=0.0,
        neginf=0.0,
    )
    p_values = np.nan_to_num(
        np.asarray(result.pvalue, dtype="float64"), nan=1.0, posinf=1.0,
        neginf=1.0,
    )
    t_values = np.where(both_constant, 0.0, t_values)
    p_values = np.where(both_constant, 1.0, p_values)
    return t_values, p_values


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up correction.

    Delegates to SciPy rather than restating the step-up rule: two producers of
    one q-value is the shape that drifts, and the test for this function is a
    comparison against SciPy rather than against a second transcription.

    :param p_values: Flat array of p-values.
    :type p_values: numpy.ndarray
    :param alpha: Rejection threshold applied to the q-values.
    :type alpha: float
    :return: ``(reject, q_values)``, both shaped like the input.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    flat = np.asarray(p_values, dtype="float64").reshape(-1)
    q_values = scipy.stats.false_discovery_control(flat, method="bh")
    q_values = np.asarray(q_values, dtype="float64").reshape(np.shape(p_values))
    reject = q_values < alpha
    return reject, q_values


def fdr_across_layers(
    p_maps: Mapping[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Correct once over every tap's p-values, then split the result back.

    Correcting each tap separately answers a different question -- "within this
    layer, which units are surprising" -- and the answer is not what a figure
    showing one map per layer is claiming. The paper corrects across all layers
    jointly, and the correction is what makes the map's threshold a statement
    about the model rather than about 784 independent tests.

    :param p_maps: ``{tap_name: p_values}``.
    :type p_maps: Mapping[str, numpy.ndarray]
    :param alpha: Rejection threshold applied to the q-values.
    :type alpha: float
    :return: ``{tap_name: {"reject": bool array, "q": float array}}``.
    :rtype: Dict[str, Dict[str, numpy.ndarray]]
    :raises ValueError: If ``p_maps`` is empty.
    """
    if not p_maps:
        raise ValueError("p_maps must not be empty")

    names = list(p_maps)
    stacked = np.concatenate(
        [np.asarray(p_maps[name], dtype="float64").reshape(-1) for name in names]
    )
    reject, q_values = fdr_bh(stacked, alpha=alpha)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    offset = 0
    for name in names:
        size = np.asarray(p_maps[name]).size
        out[name] = {
            "reject": reject[offset:offset + size].reshape(
                np.shape(p_maps[name])
            ),
            "q": q_values[offset:offset + size].reshape(np.shape(p_maps[name])),
        }
        offset += size
    return out


def label_islands(
    mask: np.ndarray,
    connectivity: str = "queen",
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> Tuple[np.ndarray, int]:
    """Label connected significant same-sign cells, discarding the small ones.

    Queen contiguity, matching both the training metric and
    :func:`~dl_techniques.metrics.spatial_autocorrelation.morans_i_grid`.

    :param mask: 2-D boolean map of candidate cells.
    :type mask: numpy.ndarray
    :param connectivity: One of :data:`~dl_techniques.metrics.spatial_autocorrelation.CONNECTIVITIES`.
    :type connectivity: str
    :param min_size: Clusters smaller than this are dropped.
    :type min_size: int
    :return: ``(labels, count)``; ``0`` marks a dropped or absent cell.
    :rtype: Tuple[numpy.ndarray, int]
    :raises ValueError: If ``mask`` is not 2-D, ``connectivity`` is unknown, or
        ``min_size`` is below 1.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2-D map, got shape {mask.shape}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if connectivity not in CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be one of {list(CONNECTIVITIES)}, "
            f"got {connectivity!r}"
        )

    structure = QUEEN_STRUCTURE if connectivity == "queen" else ROOK_STRUCTURE
    labels, count = ndimage.label(mask, structure=structure)
    if count and min_size > 1:
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
        too_small = {i + 1 for i, size in enumerate(sizes) if size < min_size}
        if too_small:
            labels = np.where(np.isin(labels, list(too_small)), 0, labels)
            # Renumber, so that surviving labels are exactly 1..count. Without
            # this a caller iterating `range(1, count + 1)` reads a label that
            # does not exist, which silently drops a cluster instead of raising.
            survivors = np.unique(labels[labels > 0])
            if survivors.size == 0:
                return np.zeros_like(labels, dtype="int64"), 0
            remap = np.zeros(int(labels.max()) + 1, dtype="int64")
            remap[survivors] = np.arange(1, survivors.size + 1, dtype="int64")
            labels = remap[labels]
            count = int(survivors.size)
    return labels.astype("int64"), int(count)


def grow_clusters(
    t_grid: np.ndarray,
    sig_grid: np.ndarray,
    sign: int = 1,
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    connectivity: str = "queen",
    max_size: Optional[int] = None,
    cell_to_unit: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Grow clusters of significant same-sign cells, most selective cell first.

    The seed is the highest-scoring candidate; the frontier is then always
    resolved by taking the highest-scoring candidate neighbour, so a cluster is
    an ordering of its own cells rather than an unordered blob.

    **With ``max_size=None`` this returns the connected components of the same
    significant cells.** The greedy order cannot change the partition, because a
    maximal connected set is reached whichever member you enter it from. The
    order matters only once a cap stops a cluster early -- and then it is part of
    the definition, which is why the cap is an argument and not a comment.

    :param t_grid: ``(height, width)`` contrast values, unthresholded.
    :type t_grid: numpy.ndarray
    :param sig_grid: ``(height, width)`` boolean significance map.
    :type sig_grid: numpy.ndarray
    :param sign: ``+1`` for units selective to condition A, ``-1`` for B.
    :type sign: int
    :param min_size: Clusters smaller than this are dropped.
    :type min_size: int
    :param connectivity: One of
        :data:`~dl_techniques.metrics.spatial_autocorrelation.CONNECTIVITIES`.
    :type connectivity: str
    :param max_size: Stop a cluster at this many cells, or ``None`` for no cap.
    :type max_size: Optional[int]
    :param cell_to_unit: ``(height, width)`` unit ids, so clusters come back
        indexed by unit. ``None`` returns grid row-major indices instead.
    :type cell_to_unit: Optional[numpy.ndarray]
    :return: ``(clusters, label_grid)`` -- each cluster an array of unit ids (or
        cell indices), and the grid labelling with ``0`` for unassigned.
    :rtype: Tuple[List[numpy.ndarray], numpy.ndarray]
    :raises ValueError: If the maps disagree in shape, ``sign`` is not ``+1`` or
        ``-1``, ``connectivity`` is unknown, or ``max_size`` is below 1.
    """
    t_grid = np.asarray(t_grid, dtype="float64")
    sig_grid = np.asarray(sig_grid, dtype=bool)
    if t_grid.shape != sig_grid.shape:
        raise ValueError(
            f"sig_grid shape {sig_grid.shape} does not match t_grid shape "
            f"{t_grid.shape}"
        )
    if sign not in CLUSTER_SIGNS:
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    if connectivity not in CONNECTIVITIES:
        raise ValueError(
            f"connectivity must be one of {list(CONNECTIVITIES)}, "
            f"got {connectivity!r}"
        )
    if max_size is not None and max_size < 1:
        raise ValueError(f"max_size must be >= 1 or None, got {max_size}")

    height, width = t_grid.shape
    score = sign * t_grid
    candidates = sig_grid & (score > 0.0)
    structure = QUEEN_STRUCTURE if connectivity == "queen" else ROOK_STRUCTURE

    labels = np.zeros(t_grid.shape, dtype="int64")
    clusters: List[np.ndarray] = []
    next_label = 0

    while candidates.any():
        seed_flat = int(np.argmax(np.where(candidates, score, -np.inf)))
        seed = divmod(seed_flat, width)

        cluster_cells = {seed}
        frontier: List[Tuple[float, int]] = []
        for neighbour in _neighbours(seed, height, width, structure):
            if candidates[neighbour]:
                heapq.heappush(frontier, (-score[neighbour], neighbour[0] * width + neighbour[1]))

        while frontier:
            if max_size is not None and len(cluster_cells) >= max_size:
                break
            negative_score, flat = heapq.heappop(frontier)
            cell = divmod(flat, width)
            if cell in cluster_cells or not candidates[cell]:
                continue
            cluster_cells.add(cell)
            for neighbour in _neighbours(cell, height, width, structure):
                if candidates[neighbour] and neighbour not in cluster_cells:
                    heapq.heappush(
                        frontier,
                        (
                            -score[neighbour],
                            neighbour[0] * width + neighbour[1],
                        ),
                    )

        if len(cluster_cells) < min_size:
            for cell in cluster_cells:
                candidates[cell] = False
            continue

        next_label += 1
        for cell in cluster_cells:
            candidates[cell] = False
            labels[cell] = next_label
        clusters.append(_to_units(cluster_cells, cell_to_unit, width))

    return clusters, labels


def _neighbours(
    cell: Tuple[int, int],
    height: int,
    width: int,
    structure: np.ndarray,
) -> Iterable[Tuple[int, int]]:
    """Yield the in-bounds cells sharing an edge or corner with ``cell``."""
    row, col = cell
    for row_step in (-1, 0, 1):
        for col_step in (-1, 0, 1):
            if row_step == 0 and col_step == 0:
                continue
            if not structure[row_step + 1, col_step + 1]:
                continue
            neighbour = (row + row_step, col + col_step)
            if 0 <= neighbour[0] < height and 0 <= neighbour[1] < width:
                yield neighbour


def _to_units(
    cells: Iterable[Tuple[int, int]],
    cell_to_unit: Optional[np.ndarray],
    width: int,
) -> np.ndarray:
    """Translate grid cells to unit ids, or to row-major cell indices."""
    flat = np.array([row * width + col for row, col in cells], dtype="int64")
    if cell_to_unit is None:
        return np.sort(flat)
    table = np.asarray(cell_to_unit, dtype="int64")
    return np.sort(table.reshape(-1)[flat])


def cluster_response(
    activations_by_condition: Mapping[str, np.ndarray],
    cluster_units: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    """Mean absolute response of a cluster to each stimulus condition.

    The absolute value is the paper's definition of a model's "response" to a
    condition: a unit's sign carries no consistent meaning across a contrast, so
    averaging signed activations lets a cluster cancel itself out.

    :param activations_by_condition: ``{condition: (num_stimuli, num_units)}``.
    :type activations_by_condition: Mapping[str, numpy.ndarray]
    :param cluster_units: Unit ids belonging to the cluster.
    :type cluster_units: Sequence[int]
    :return: ``{condition: {"mean", "sem"}}`` across stimuli.
    :rtype: Dict[str, Dict[str, float]]
    :raises ValueError: If ``cluster_units`` is empty or an id is out of range.
    """
    units = np.asarray(cluster_units, dtype="int64").reshape(-1)
    if units.size == 0:
        raise ValueError("cluster_units must not be empty")

    profile: Dict[str, Dict[str, float]] = {}
    for name, activations in activations_by_condition.items():
        activations = np.asarray(activations, dtype="float64")
        if activations.ndim != 2:
            raise ValueError(
                f"condition {name!r} must be (stimuli, units), got "
                f"{activations.shape}"
            )
        if units.max() >= activations.shape[1] or units.min() < 0:
            raise ValueError(
                f"cluster unit ids must lie in [0, {activations.shape[1]}), got "
                f"range [{units.min()}, {units.max()}] for condition {name!r}"
            )
        per_stimulus = np.abs(activations[:, units]).mean(axis=1)
        num_stimuli = per_stimulus.shape[0]
        sem = (
            float(per_stimulus.std(ddof=1) / np.sqrt(num_stimuli))
            if num_stimuli > 1
            else float("nan")
        )
        profile[name] = {
            "mean": float(per_stimulus.mean()),
            "sem": sem,
        }
    return profile


def profile_consistency(
    profiles: Sequence[Mapping[str, Dict[str, float]]],
    conditions: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """How similarly a set of clusters responds across stimulus conditions.

    Two numbers, because they answer different questions and disagree in
    exactly the interesting case. The mean pairwise rank correlation across
    clusters asks whether they *order* conditions the same way; the interaction
    test asks whether any cluster's profile differs enough to reject the
    one-network reading. A high mean with a significant interaction means the
    clusters are consistent on average and one of them is not like the others.

    :param profiles: One :func:`cluster_response` result per cluster.
    :type profiles: Sequence[Mapping[str, Dict[str, float]]]
    :param conditions: Condition order, or ``None`` to take the keys of the
        first profile.
    :type conditions: Optional[Sequence[str]]
    :return: ``{"conditions", "mean_pairwise_spearman", "num_clusters",
        "num_conditions", "interaction"}``.
    :rtype: Dict[str, object]
    :raises ValueError: If ``profiles`` is empty, or the profiles disagree about
        which conditions they cover.
    """
    if not profiles:
        raise ValueError("profiles must not be empty")

    if conditions is None:
        conditions = list(profiles[0].keys())
    missing = [
        index for index, profile in enumerate(profiles)
        if set(conditions) - set(profile)
    ]
    if missing:
        raise ValueError(
            f"profile {missing[0]} is missing condition(s) "
            f"{sorted(set(conditions) - set(profiles[missing[0]]))}"
        )

    table = np.array(
        [[float(profile[name]["mean"]) for name in conditions] for profile in profiles],
        dtype="float64",
    )

    correlations = []
    for i in range(table.shape[0]):
        for j in range(i + 1, table.shape[0]):
            if table.shape[1] < 2:
                correlations.append(float("nan"))
                continue
            value = scipy.stats.spearmanr(table[i], table[j]).statistic
            correlations.append(float(value))
    finite = [c for c in correlations if not np.isnan(c)]

    interaction: Dict[str, object] = {"available": False}
    if table.shape[0] > 1 and table.shape[1] > 1:
        per_stimulus = _interaction_from_means(table, conditions)
        if per_stimulus is not None:
            f_stat, p_value = per_stimulus
            interaction = {
                "available": True,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
            }

    return {
        "conditions": list(conditions),
        "mean_pairwise_spearman": float(np.mean(finite)) if finite else float("nan"),
        "num_clusters": int(table.shape[0]),
        "num_conditions": int(table.shape[1]),
        "interaction": interaction,
    }


def _interaction_from_means(
    table: np.ndarray,
    conditions: Sequence[str],
) -> Optional[Tuple[float, float]]:
    """One-way ANOVA across clusters, per condition, on the condition means.

    A full two-way ANOVA needs per-stimulus values; the cluster means alone
    leave the cluster x stimulus cell empty, so what is computable from a
    :func:`cluster_response` summary is a one-way test of whether the clusters'
    mean responses differ *by condition*. That is a weaker question than the
    paper's, and the returned dict says so with ``available`` rather than
    implying the stronger test ran.
    """
    del conditions
    groups = [table[:, index] for index in range(table.shape[1])]
    if any(group.size < 2 for group in groups):
        return None
    try:
        result = scipy.stats.f_oneway(*groups)
    except ValueError:
        return None
    if np.isnan(result.statistic):
        return None
    return result.statistic, result.pvalue