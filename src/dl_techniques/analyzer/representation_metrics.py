"""
Representation-Geometry and Grokking Metrics

Measurement utilities used by Baek, Liu, Tegmark et al., "Harmonic Loss Trains
Interpretable AI Models" (arXiv:2502.01628) to decide whether a learned embedding
is *interpretable* and whether a model *grokked*. They are loss-agnostic: they
take embedding matrices and accuracy curves, so they apply equally to
cross-entropy and harmonic models.

The paper proposes two interpretability indicators and one training-dynamics
measure:

1. **Compression** -- cumulative explained variance (EV) of a PCA of the
   embedding. A low-dimensional, interpretable structure concentrates variance in
   few components (:func:`explained_variance_ratios`,
   :func:`cumulative_explained_variance`, :func:`top_k_explained_variance`).
2. **Geometry** -- the parallelogram loss (Eq. 3): for a quadruple
   ``(i, j, m, n)`` with ``i -> j`` and ``m -> n`` the same relation
   (jump -> jumped, fasten -> fastened), the first two principal components should
   satisfy ``E_j - E_i = E_n - E_m``. The loss is
   ``|| E_i + E_n - E_j - E_m || / sigma`` with
   ``sigma = sqrt(mean_k ||E_k||^2)`` so it is invariant to rescaling the
   embedding (:func:`parallelogram_loss`, :func:`sample_parallelogram_quadruples`).
   The same module scores coset/cluster structure with the silhouette score
   (:func:`partition_silhouette`, :func:`rank_partitions`; paper Appendix B).
3. **Grokking** -- how many epochs test accuracy lags train accuracy when each
   must stay above a threshold for a run of consecutive epochs
   (:func:`epochs_to_threshold`, :func:`grokking_gap`). A run on the ``y = x``
   line of the paper's Figure 3(c) has a gap of zero.

References:
    Baek, Liu, Tegmark et al. (2025). "Harmonic Loss Trains Interpretable AI
    Models." arXiv:2502.01628.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Compression: PCA explained variance
# ---------------------------------------------------------------------


def _as_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (num_points, dim), got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 points")
    return x


def explained_variance_ratios(embeddings: np.ndarray) -> np.ndarray:
    """Fraction of total variance carried by each principal component.

    :param embeddings: ``(num_points, dim)`` matrix; rows are points.
    :return: Non-increasing vector of length ``min(num_points, dim)`` summing to
        1, or all zeros when every point is identical (zero total variance).
    :raises ValueError: If ``embeddings`` is not 2-D or has fewer than 2 rows.
    """
    x = _as_matrix(embeddings)
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    var = s ** 2
    total = var.sum()
    return var / total if total > 0 else np.zeros_like(var)


def cumulative_explained_variance(embeddings: np.ndarray) -> np.ndarray:
    """Cumulative sum of :func:`explained_variance_ratios` (paper Fig. 3a)."""
    return np.cumsum(explained_variance_ratios(embeddings))


def top_k_explained_variance(embeddings: np.ndarray, k: int = 2) -> float:
    """Variance share of the first ``k`` components (the ``EV`` of paper Fig. 2).

    :param k: Number of leading components, at least 1. Values above the
        available rank return the full sum.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return float(explained_variance_ratios(embeddings)[:k].sum())


# ---------------------------------------------------------------------
# Geometry: parallelogram loss
# ---------------------------------------------------------------------


def pca_project(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project centred embeddings on their first ``n_components`` principal axes.

    Axis signs are arbitrary; every metric here is sign invariant.
    """
    x = _as_matrix(embeddings)
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    return (x - x.mean(axis=0)) @ vt[:n_components].T


def parallelogram_loss(
        embeddings: np.ndarray,
        quadruples: np.ndarray,
        n_components: int = 2,
) -> np.ndarray:
    """Scale-invariant parallelogram loss of each quadruple (paper Eq. 3).

    ``l = || E_i + E_n - E_j - E_m || / sigma`` in the first ``n_components`` PCA
    coordinates, ``sigma = sqrt(mean_k ||E_k||^2)`` over all points. A quadruple
    ``(i, j, m, n)`` is a perfect parallelogram when ``E_j - E_i = E_n - E_m``.

    :param embeddings: ``(num_points, dim)`` matrix (for example a token
        embedding table).
    :param quadruples: Integer array ``(num_quadruples, 4)`` of row indices
        ``(i, j, m, n)``.
    :param n_components: PCA dimensions used (the paper uses 2).
    :return: ``(num_quadruples,)`` non-negative losses.
    :raises ValueError: If ``quadruples`` is not ``(N, 4)`` or indexes outside
        the table, or if all projected points coincide (``sigma == 0``).
    """
    quads = np.asarray(quadruples)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError(f"quadruples must have shape (N, 4), got {quads.shape}")
    e = pca_project(embeddings, n_components)
    if quads.size and (quads.min() < 0 or quads.max() >= e.shape[0]):
        raise ValueError("quadruple index out of range")
    sigma = np.sqrt(np.mean(np.sum(e ** 2, axis=1)))
    if sigma == 0:
        raise ValueError("all projected points coincide; the loss is undefined")
    i, j, m, n = quads.T
    return np.linalg.norm(e[i] + e[n] - e[j] - e[m], axis=1) / sigma


def sample_parallelogram_quadruples(
        pairs: Sequence[Tuple[int, int]],
        num_quadruples: int,
        seed: Optional[int] = 0,
) -> np.ndarray:
    """Draw ``(i, j, m, n)`` from two different pairs of one relation.

    Each pair is an ``(input_id, output_id)`` such as ``(jump, jumped)``; two
    distinct pairs are drawn without replacement per quadruple. Pairs are the
    caller's job: tokenize each word and keep the LAST token if it splits into
    several, as the paper does.

    :param pairs: At least two ``(input_id, output_id)`` pairs.
    :param num_quadruples: How many quadruples to draw (the paper uses 10000).
    :param seed: Seed for ``numpy.random.default_rng``.
    :return: ``(num_quadruples, 4)`` int array.
    """
    p = np.asarray(pairs, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
        raise ValueError("pairs must be an (N>=2, 2) sequence of (input, output) ids")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(p), size=num_quadruples)
    second = (first + rng.integers(1, len(p), size=num_quadruples)) % len(p)
    return np.concatenate([p[first], p[second]], axis=1)


# ---------------------------------------------------------------------
# Geometry: cluster / coset structure
# ---------------------------------------------------------------------


def partition_silhouette(embeddings: np.ndarray, labels: Sequence[int]) -> float:
    """Silhouette score of ``labels`` as a clustering of ``embeddings``.

    Mean over points of ``(b - a) / max(a, b)`` (``a`` the mean intra-cluster
    distance, ``b`` the mean distance to the nearest other cluster); in
    ``[-1, 1]``, higher is better separated. Used in the paper's Appendix B to
    find which subgroup's cosets a permutation embedding has organised into.

    :return: The score, or ``nan`` when it is undefined (fewer than 2 or more
        than ``num_points - 1`` distinct labels).
    """
    from sklearn.metrics import silhouette_score

    x = _as_matrix(embeddings)
    y = np.asarray(labels)
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have one entry per embedding row")
    k = len(np.unique(y))
    if k < 2 or k > x.shape[0] - 1:
        return float("nan")
    return float(silhouette_score(x, y))


def rank_partitions(
        embeddings: np.ndarray,
        partitions: Dict[str, Sequence[int]],
) -> Tuple[Tuple[str, float], ...]:
    """Candidate partitions ordered best silhouette first; undefined ones last.

    :param partitions: Name -> per-point label vector (for example one entry per
        subgroup's coset partition).
    """
    scored = [(name, partition_silhouette(embeddings, lab)) for name, lab in partitions.items()]
    return tuple(sorted(scored, key=lambda t: (np.isnan(t[1]), -(t[1] if not np.isnan(t[1]) else 0.0))))


# ---------------------------------------------------------------------
# Grokking
# ---------------------------------------------------------------------


def epochs_to_threshold(
        accuracy: Sequence[float],
        threshold: float = 0.9,
        consecutive: int = 20,
) -> Optional[int]:
    """First epoch starting a run of ``consecutive`` epochs above ``threshold``.

    This is the paper's "epochs to accuracy > 0.9 for 20 consecutive epochs".

    :param accuracy: Per-epoch accuracy.
    :return: 0-based index of the first epoch of the earliest such run, or
        ``None`` if there is none (the paper omits such runs).
    """
    if consecutive < 1:
        raise ValueError(f"consecutive must be >= 1, got {consecutive}")
    above = np.asarray(accuracy, dtype=np.float64) > threshold
    run = 0
    for t, ok in enumerate(above):
        run = run + 1 if ok else 0
        if run >= consecutive:
            return t - consecutive + 1
    return None


def grokking_gap(
        train_accuracy: Sequence[float],
        test_accuracy: Sequence[float],
        threshold: float = 0.9,
        consecutive: int = 20,
) -> Optional[int]:
    """Epochs by which test accuracy lags train accuracy (paper Fig. 3c).

    ``epochs_to_threshold(test) - epochs_to_threshold(train)``: zero (or
    negative) means no grokking, a large positive value is delayed
    generalization.

    :return: The gap, or ``None`` if either curve never reaches the threshold.
    """
    tr = epochs_to_threshold(train_accuracy, threshold, consecutive)
    te = epochs_to_threshold(test_accuracy, threshold, consecutive)
    if tr is None or te is None:
        return None
    return te - tr
