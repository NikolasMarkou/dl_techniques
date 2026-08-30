"""Pool-level quality metrics for text-embedding models.

Two families, and the distinction between them is the point of this module:

**Ranking metrics** answer "does this embedding space put the right document
first?" They need a query-by-candidate similarity matrix and a ground-truth
index per query, and they are the only metrics here that measure *quality*.

**Geometry diagnostics** answer "what shape is this embedding space?" They need
no labels at all. They are *not* quality metrics and must never be given a
better/worse verdict on their own: a random-projection encoder maximizes
`effective_rank`, minimizes `anisotropy` and minimizes `uniformity` while
retrieving nothing. They exist to explain *why* a ranking number moved.

These are plain functions on numpy arrays, not `keras.metrics.Metric`
subclasses. That is deliberate and structural, not a shortcut: every metric here
needs the complete embedding matrix at once -- an SVD, a mean over all pairs, a
ranking against a whole pool -- which a streaming `update_state` cannot express.
The package already ships plain functions for the same reason
(`llm_metrics.self_bleu`, `llm_metrics.distinct_n`, `perplexity_metric.perplexity`,
`time_series_metrics.calculate_comprehensive_metrics`).

Redundancy, stated once so nobody double-counts evidence:

- `recall_at_k`, `mrr_at_k` and `ndcg_at_k` are all positive-weighted linear
  functionals of the SAME recall curve `(R@1 ... R@k)`. None carries information
  that curve lacks; they are three weightings of one measurement.
- Per query, `mrr` and `ndcg` are rank-equivalent (both strictly decreasing in
  rank). Their MEANS are not: the transforms have different curvature, so they
  can disagree about which system is better. MEASURED -- ranks `[1, 100]` versus
  `[2, 2]` at k=100 give MRR 0.5050 vs 0.5000 (prefers the first) and nDCG
  0.5751 vs 0.6309 (prefers the second); over 20,000 random 8-query system pairs
  they disagree 3.64% of the time. Report one, not both.
- `median_rank` is the exception: it is NOT a functional of the truncated recall
  curve, and it is robust to the long tail that makes MRR and nDCG diverge.
- With exactly one relevant document, `recall_at_k` equals precision@k times k,
  hit-rate@k, and top-k accuracy. Do not add those as separate metrics.
- `anisotropy` and `uniformity` are both cone-collapse detectors and are
  strongly correlated; `alignment` and `uniformity` are the genuinely
  complementary pair (the Wang & Isola trade-off).

References:
    - Jarvelin & Kekalainen, 2002. Cumulated gain-based evaluation of IR
      techniques. (nDCG)
    - Ethayarajh, 2019. How Contextual are Contextualized Word Representations?
      (anisotropy) (https://arxiv.org/abs/1909.00512)
    - Roy & Vetterli, 2007. The effective rank: a measure of effective
      dimensionality.
    - Wang & Isola, 2020. Understanding Contrastive Representation Learning
      through Alignment and Uniformity on the Hypersphere.
      (https://arxiv.org/abs/2005.10242)
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

__all__ = [
    "alignment",
    "anisotropy",
    "effective_rank",
    "embedding_norm_stats",
    "l2_normalize",
    "mrr_at_k",
    "ndcg_at_k",
    "rank_of_ground_truth",
    "ranking_metrics",
    "recall_at_k",
    "recall_at_ks",
    "uniformity",
    "zca_whiten",
]


def l2_normalize(embeddings: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows.

    Args:
        embeddings: Array of shape `(n, d)`.
        eps: Floor on the norm, so an all-zero row returns zeros rather than
            NaN.

    Returns:
        Array of shape `(n, d)`, float64, with unit-norm rows.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def zca_whiten(
    fit: np.ndarray,
    *apply_to: np.ndarray,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, ...]:
    """Centre and whiten embeddings using statistics from `fit`.

    Cosine similarity weights every dimension equally, so content that survives
    only in low-variance directions is unrecoverable from it while remaining
    perfectly available to a linear probe. Equalizing the variances fixes that,
    and on this repository's character-level encoders it is worth a large amount:
    SQuAD recall@1 rose from ~0.060 to ~0.21 on 9 of 9 learned-position
    convolutional cells.

    Two properties make it usable rather than a leak:

    - It is fitted on embeddings only -- no queries, no labels.
    - It transfers across corpora. Fitting on 3000 SST-2 sentences and applying
      to Wikipedia-paragraph retrieval retained ~90% of the gain, so `fit` need
      not be the corpus being searched.

    It is **not** universally beneficial, and that is diagnostic rather than a
    caveat: it helps only where low-variance directions carry content. On the
    same encoders trained with sinusoidal positions it made retrieval *worse*
    in 12 of 12 cells, because there the content is absent rather than drowned.

    Args:
        fit: Array of shape `(n, d)` whose mean and covariance define the
            transform. Should have `n` comfortably larger than `d`.
        *apply_to: Further `(m, d)` arrays to transform with the same
            statistics. `fit` itself is always returned first.
        eps: Floor on eigenvalues before the inverse square root, which keeps a
            near-null direction from being amplified without bound.

    Returns:
        Tuple of arrays, float64: the transformed `fit` followed by each of
        `apply_to`, in order.

    Raises:
        ValueError: If any array in `apply_to` has a different width than
            `fit`, or `fit` has fewer than two rows.

    References:
        - Su et al., 2021. Whitening Sentence Representations for Better
          Semantics and Faster Retrieval. (https://arxiv.org/abs/2103.15316)
    """
    reference = np.asarray(fit, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError(
            f"fit must be (n, d) with n >= 2, got {reference.shape}"
        )
    width = reference.shape[1]
    others = [np.asarray(a, dtype=np.float64) for a in apply_to]
    for i, other in enumerate(others):
        if other.ndim != 2 or other.shape[1] != width:
            raise ValueError(
                f"apply_to[{i}] must be (m, {width}), got {other.shape}"
            )

    mean = reference.mean(axis=0, keepdims=True)
    centred = reference - mean
    covariance = (centred.T @ centred) / (len(centred) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scaling = 1.0 / np.sqrt(np.maximum(eigenvalues, eps))
    transform = eigenvectors @ np.diag(scaling) @ eigenvectors.T

    return tuple(
        [centred @ transform] + [(a - mean) @ transform for a in others]
    )


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------


def rank_of_ground_truth(
    similarity: np.ndarray,
    ground_truth: np.ndarray,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Rank of the correct candidate for each query, 1-based and pessimistic.

    `rank_i = 1 + #{j : s_ij > s_i,g} + #{j != g : s_ij == s_i,g}`

    **Ties are broken pessimistically, and that is load-bearing rather than a
    detail.** A collapsed encoder -- exactly the failure `anisotropy` exists to
    detect -- produces near-identical rows and therefore mass ties. Under
    optimistic or midpoint tie-breaking such a model scores a perfect
    `recall@1`. Under pessimistic ties it scores 0 for every `k` below the pool
    size, which is the truthful answer.

    Args:
        similarity: Array of shape `(n_queries, n_candidates)`. Higher is more
            similar.
        ground_truth: Integer array of shape `(n_queries,)`, indexing the
            candidate axis.
        chunk_size: Number of query rows processed at once. Affects memory
            only; the result is exact for any value.

    Returns:
        Integer array of shape `(n_queries,)` with values in
        `[1, n_candidates]`.

    Raises:
        ValueError: If shapes disagree, the matrix is not 2-D, or a ground-truth
            index is out of range.
    """
    sims = np.asarray(similarity)
    truth = np.asarray(ground_truth).ravel()

    if sims.ndim != 2:
        raise ValueError(f"similarity must be 2-D; got shape {sims.shape}")
    if truth.shape[0] != sims.shape[0]:
        raise ValueError(
            f"ground_truth has {truth.shape[0]} entries but similarity has "
            f"{sims.shape[0]} query rows"
        )
    n_candidates = sims.shape[1]
    if truth.size and (truth.min() < 0 or truth.max() >= n_candidates):
        raise ValueError(
            f"ground_truth indices must lie in [0, {n_candidates}); got "
            f"[{truth.min()}, {truth.max()}]"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")

    ranks = np.empty(sims.shape[0], dtype=np.int64)
    for start in range(0, sims.shape[0], chunk_size):
        stop = min(start + chunk_size, sims.shape[0])
        block = sims[start:stop]
        gold = block[np.arange(block.shape[0]), truth[start:stop]][:, None]
        strictly_better = np.sum(block > gold, axis=1)
        tied = np.sum(block == gold, axis=1) - 1  # exclude the gold cell itself
        ranks[start:stop] = 1 + strictly_better + tied
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Fraction of queries whose correct candidate is ranked at or above `k`.

    With one relevant document per query this equals hit-rate@k and top-k
    accuracy.

    Args:
        ranks: 1-based ranks, as returned by `rank_of_ground_truth`.
        k: Cutoff.

    Returns:
        Value in `[0, 1]`; `nan` for an empty input.

    Raises:
        ValueError: If `k` is below 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    r = np.asarray(ranks)
    if r.size == 0:
        return float("nan")
    return float(np.mean(r <= k))


def recall_at_ks(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Recall at several cutoffs from one rank vector.

    Args:
        ranks: 1-based ranks.
        ks: Cutoffs.

    Returns:
        Mapping `"recall_at_<k>" -> value`.
    """
    return {f"recall_at_{k}": recall_at_k(ranks, k) for k in ks}


def mrr_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Mean reciprocal rank, truncated at `k`.

    Args:
        ranks: 1-based ranks.
        k: Cutoff; queries ranked beyond it contribute 0.

    Returns:
        Value in `[0, 1]`; `nan` for an empty input.

    Raises:
        ValueError: If `k` is below 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    r = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(np.mean(np.where(r <= k, 1.0 / np.maximum(r, 1.0), 0.0)))


def ndcg_at_k(ranks: np.ndarray, k: int = 10) -> float:
    """Normalized discounted cumulative gain, single-relevant-document case.

    With exactly one relevant document and binary gain, `IDCG@k == 1`, so this
    reduces to the mean of `1 / log2(rank + 1)` over queries inside the cutoff.

    **Not independent evidence.** This is a reweighting of the same recall curve
    that `recall_at_k` and `mrr_at_k` reweight -- see the module docstring. It is
    provided for future graded-relevance use and is deliberately kept out of the
    study's headline metrics. Note also that although nDCG and MRR are
    rank-equivalent per query, their means are not, and they disagree about
    which system is better on roughly 3.6% of random system pairs.

    Args:
        ranks: 1-based ranks.
        k: Cutoff.

    Returns:
        Value in `[0, 1]`; `nan` for an empty input.

    Raises:
        ValueError: If `k` is below 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    r = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    gains = np.where(r <= k, 1.0 / np.log2(np.maximum(r, 1.0) + 1.0), 0.0)
    return float(np.mean(gains))


def ranking_metrics(
    similarity: np.ndarray,
    ground_truth: np.ndarray,
    *,
    ks: Sequence[int] = (1, 5, 10),
    mrr_k: int = 10,
    chunk_size: int = 512,
) -> Dict[str, float]:
    """Compute every ranking metric from a single pass over the similarities.

    Args:
        similarity: Array of shape `(n_queries, n_candidates)`.
        ground_truth: Integer array of shape `(n_queries,)`.
        ks: Recall cutoffs.
        mrr_k: Cutoff for MRR.
        chunk_size: Query-row chunk for the rank computation.

    Returns:
        Mapping with the recall cutoffs, `mrr_at_<mrr_k>`, `median_rank`,
        `mean_rank`, `n_queries`, `n_candidates` and `chance_recall_at_1`
        (`1 / n_candidates`, the value a random ranker attains).
    """
    ranks = rank_of_ground_truth(
        similarity, ground_truth, chunk_size=chunk_size
    )
    n_candidates = int(np.asarray(similarity).shape[1])

    out: Dict[str, float] = dict(recall_at_ks(ranks, ks))
    out[f"mrr_at_{mrr_k}"] = mrr_at_k(ranks, mrr_k)
    out["median_rank"] = float(np.median(ranks)) if ranks.size else float("nan")
    out["mean_rank"] = float(np.mean(ranks)) if ranks.size else float("nan")
    out["n_queries"] = float(ranks.size)
    out["n_candidates"] = float(n_candidates)
    out["chance_recall_at_1"] = (
        1.0 / n_candidates if n_candidates else float("nan")
    )
    return out


# ---------------------------------------------------------------------
# Geometry diagnostics
# ---------------------------------------------------------------------


def anisotropy(
    embeddings: np.ndarray,
    *,
    already_normalized: bool = False,
) -> float:
    """Mean cosine similarity over all distinct ordered pairs.

    Near 0 for a well-spread space; approaching 1 when every embedding points
    the same way, which is the classic collapse pathology of an undertrained or
    over-regularized encoder.

    Computed with the exact identity

        `mean_pairwise_cosine = (||sum_i u_i||^2 - n) / (n * (n - 1))`

    on unit-normalized rows, which is `O(n*d)` rather than the naive `O(n^2*d)`
    and gives the same value to floating-point precision.

    **A diagnostic, not a quality metric.** A random projection scores near 0
    here while retrieving nothing.

    Args:
        embeddings: Array of shape `(n, d)` with `n >= 2`.
        already_normalized: Skip normalization if the rows are already unit-norm.

    Returns:
        Value in `[-1, 1]`.

    Raises:
        ValueError: If fewer than two rows are given, or any row has zero norm
            (its cosine is undefined and silently returning NaN would hide it).
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {x.shape}")
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"anisotropy needs at least 2 embeddings; got {n}")

    if not already_normalized:
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms == 0.0):
            raise ValueError(
                f"{int(np.sum(norms == 0.0))} embedding(s) have zero norm; "
                "their cosine is undefined"
            )
        x = x / norms[:, None]

    total = np.sum(x, axis=0)
    sum_sq = float(np.dot(total, total))
    return float((sum_sq - n) / (n * (n - 1)))


def effective_rank(
    embeddings: np.ndarray,
    *,
    center: bool = True,
) -> float:
    """Entropy-based effective rank of the embedding matrix.

    `exp(H(p))` where `p` is the singular-value spectrum normalized to sum to 1.
    Equals the true rank for a matrix with equal singular values, and falls
    toward 1 as the spectrum concentrates -- catching *dimensional* collapse,
    where a space uses a handful of its directions, which `anisotropy` can miss.

    **A diagnostic, not a quality metric.** A random projection maximizes it.

    Args:
        embeddings: Array of shape `(n, d)`.
        center: Subtract the column mean first. `True` matches the existing
            inline computation in `analyzer/analyzers/information_flow_analyzer.py`;
            `False` matches the Roy & Vetterli definition. Exposed rather than
            silently chosen because the two differ whenever the embeddings have
            a large mean offset -- which a collapsed space does.

    Returns:
        Value in `[1, min(n, d)]`, or `0.0` for an all-zero matrix.

    Raises:
        ValueError: If the input is not 2-D.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {x.shape}")
    if x.size == 0:
        return 0.0

    if center:
        x = x - x.mean(axis=0, keepdims=True)

    singular = np.linalg.svd(x, compute_uv=False)
    total = float(np.sum(singular))
    if total <= 1e-12:
        return 0.0

    p = singular / total
    # Mask rather than adding an epsilon: `p * log(p + eps)` biases the entropy
    # upward for near-zero singular values, which inflates the reported rank.
    p = p[p > 0.0]
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def alignment(
    anchors: np.ndarray,
    positives: np.ndarray,
    *,
    alpha: float = 2.0,
    already_normalized: bool = False,
) -> float:
    """Wang & Isola alignment: mean distance between matched positive pairs.

    `mean(||u_i - v_i||_2 ** alpha)` over unit-normalized pairs. **Lower is
    better** -- a good space puts a query near its answer.

    At `alpha=2` this is an affine function of the mean positive cosine, since
    `||u - v||^2 = 2 - 2*cos(u, v)` for unit vectors.

    Half of a pair: read it with `uniformity`, never alone. A fully collapsed
    encoder achieves perfect alignment (0.0) and useless uniformity.

    Args:
        anchors: Array of shape `(n, d)`.
        positives: Array of shape `(n, d)`, row-aligned with `anchors`.
        alpha: Exponent on the distance; 2 is the paper's default.
        already_normalized: Skip normalization if rows are already unit-norm.

    Returns:
        Non-negative value; 0.0 when every pair coincides.

    Raises:
        ValueError: If the two arrays disagree in shape.
    """
    a = np.asarray(anchors, dtype=np.float64)
    b = np.asarray(positives, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return float("nan")

    if not already_normalized:
        a = l2_normalize(a)
        b = l2_normalize(b)
    return float(np.mean(np.linalg.norm(a - b, axis=1) ** alpha))


def uniformity(
    embeddings: np.ndarray,
    *,
    t: float = 2.0,
    max_samples: int = 4096,
    rng: Optional[np.random.Generator] = None,
    already_normalized: bool = False,
) -> float:
    """Wang & Isola uniformity: log mean Gaussian potential over all pairs.

    `log(mean_{i<j} exp(-t * ||u_i - u_j||^2))`. **Lower (more negative) is
    better** -- it means the embeddings spread over the hypersphere rather than
    piling into a cone. A collapsed space scores 0.0, the worst attainable
    value.

    Computed through `logsumexp` so a large `t` cannot underflow to `-inf`.

    Args:
        embeddings: Array of shape `(n, d)` with `n >= 2`.
        t: Temperature of the Gaussian potential.
        max_samples: Subsample above this many rows, since the computation is
            `O(n^2)` in memory.
        rng: Required when subsampling actually happens, so the result is
            reproducible. Passing `None` with a larger input is an error rather
            than a silent irreproducible number.
        already_normalized: Skip normalization if rows are already unit-norm.

    Returns:
        Value in `(-inf, 0]`.

    Raises:
        ValueError: If fewer than two rows are given, or subsampling is needed
            and `rng` is `None`.
    """
    from scipy.special import logsumexp

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {x.shape}")
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"uniformity needs at least 2 embeddings; got {n}")

    if n > max_samples:
        if rng is None:
            raise ValueError(
                f"{n} embeddings exceeds max_samples={max_samples}, so a "
                "subsample is required; pass an explicit rng so the result is "
                "reproducible"
            )
        x = x[rng.permutation(n)[:max_samples]]

    if not already_normalized:
        x = l2_normalize(x)

    sq_dist = np.maximum(
        np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1), 0.0
    )
    iu = np.triu_indices(x.shape[0], k=1)
    return float(logsumexp(-t * sq_dist[iu]) - np.log(iu[0].size))


def embedding_norm_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Descriptive statistics of the embedding norms and centroid alignment.

    Purely descriptive -- no direction, no verdict. `norm_std` far from 0 means
    cosine and dot-product retrieval would rank differently, and
    `cos_to_centroid_mean` near 1 is the one-dominant-direction degeneracy that
    `anisotropy` reports only in aggregate.

    Args:
        embeddings: Array of shape `(n, d)`.

    Returns:
        Mapping with `norm_mean`, `norm_std`, `norm_min`, `norm_max`,
        `n_zero_norm` and `cos_to_centroid_mean`.

    Raises:
        ValueError: If the input is not 2-D.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D; got shape {x.shape}")
    if x.size == 0:
        return {
            "norm_mean": float("nan"), "norm_std": float("nan"),
            "norm_min": float("nan"), "norm_max": float("nan"),
            "n_zero_norm": 0.0, "cos_to_centroid_mean": float("nan"),
        }

    norms = np.linalg.norm(x, axis=1)
    unit = l2_normalize(x)
    centroid = unit.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    cos_to_centroid = (
        float(np.mean(unit @ (centroid / centroid_norm)))
        if centroid_norm > 1e-12
        else 0.0
    )
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "n_zero_norm": float(np.sum(norms == 0.0)),
        "cos_to_centroid_mean": cos_to_centroid,
    }
