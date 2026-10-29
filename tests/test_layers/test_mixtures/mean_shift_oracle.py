"""Independent float64 NumPy oracle for the mean-shift operations.

Shared by ``test_mean_shift_ops.py`` and, in later steps, by the tests of the
layers built on ``layers/mixtures/mean_shift_ops.py`` (several call sites, hence a
module rather than copies: this is the instrument that validates correctness).

**Why this is independent of the implementation.** The implementation computes
``softmax(q.k / h^2 - ||k||^2 / (2 h^2)) @ V`` with a matmul-form squared
distance, a sentinel ``where`` for masked keys and a ``sort`` for the median. This
oracle uses none of that. It never forms the softmax identity and never expands
the square: for every query it subtracts the key vector, squares and sums the
difference, evaluates the textbook weight ``exp(-d^2 / (2 h^2))`` key by key for
the KEPT keys only (shifted by the row minimum before ``exp``, which cancels in
the ratio), and divides by their sum. A dropped key-norm term, a wrong bias sign
or a mishandled mask in the implementation therefore cannot be mirrored here. It
imports neither Keras nor TensorFlow.

**Interface contract.** Every function takes and returns float64 ``np.ndarray``.
``q``/``x`` are ``(*lead, N, D)``, ``k`` is ``(*lead, M, D)``, ``keep`` is a
boolean ``(*lead, M)`` keep-mask or ``None``, and the bandwidth ``h`` is a scalar
or any array that broadcasts to ``(*lead, N, 1)``. A row with no kept key gets
uniform weights over all keys (and a keep-all density).

``oracle_pairwise_sq_dist(a, b) -> (*lead, N, M)``

``oracle_step(q, k, v, h, keep=None, kernel="gaussian") -> (m, weights)``

``oracle_kde_log_density(q, k, h, keep=None) -> (*lead, N)``

``oracle_iterate(x, h, num_steps, step_size=1.0, blurring=True, keep=None,
kernel="gaussian") -> (*lead, N, D)``

``oracle_median_bandwidth(x, keep=None) -> (B, 1, 1)``

:raises ValueError: for an unknown ``kernel``.
"""

import math
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "oracle_pairwise_sq_dist",
    "oracle_step",
    "oracle_kde_log_density",
    "oracle_iterate",
    "oracle_median_bandwidth",
]

_MEDIAN_FLOOR = 1e-6


def _unit_rows(x: np.ndarray) -> np.ndarray:
    """Return ``x`` with every last-axis row scaled to unit norm (zero stays zero)."""
    norms = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return np.where(norms > 0.0, x / np.where(norms > 0.0, norms, 1.0), 0.0)


def oracle_pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distances by direct subtraction, one pair at a time."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lead = a.shape[:-2]
    out = np.zeros(lead + (a.shape[-2], b.shape[-2]), dtype=np.float64)
    for idx in np.ndindex(*lead):
        for i in range(a.shape[-2]):
            for j in range(b.shape[-2]):
                diff = a[idx + (i,)] - b[idx + (j,)]
                out[idx + (i, j)] = float(np.sum(diff * diff))
    return out


def _bandwidth(h, lead: Tuple[int, ...], n: int) -> np.ndarray:
    """Broadcast a bandwidth to ``(*lead, N, 1)`` as float64."""
    return np.broadcast_to(np.asarray(h, dtype=np.float64), lead + (n, 1))


def oracle_step(
    q: np.ndarray,
    k: np.ndarray,
    v: Optional[np.ndarray],
    h,
    keep: Optional[np.ndarray] = None,
    kernel: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """One mean-shift step as the textbook kernel-weighted mean, key by key."""
    if kernel not in ("gaussian", "vmf"):
        raise ValueError(f"kernel must be 'gaussian' or 'vmf', got {kernel!r}")
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = k if v is None else np.asarray(v, dtype=np.float64)
    lead, n, m = q.shape[:-2], q.shape[-2], k.shape[-2]
    bw = _bandwidth(h, lead, n)
    if kernel == "vmf":
        q, k = _unit_rows(q), _unit_rows(k)
    weights = np.zeros(lead + (n, m), dtype=np.float64)
    out = np.zeros(lead + (n, v.shape[-1]), dtype=np.float64)
    for idx in np.ndindex(*lead):
        kept = (
            list(range(m))
            if keep is None
            else [j for j in range(m) if bool(np.asarray(keep)[idx + (j,)])]
        )
        for i in range(n):
            h2 = float(bw[idx + (i, 0)]) ** 2
            if not kept:
                row = np.full(m, 1.0 / m)
            else:
                if kernel == "gaussian":
                    score = np.array(
                        [
                            -float(np.sum((q[idx + (i,)] - k[idx + (j,)]) ** 2)) / (2.0 * h2)
                            for j in kept
                        ]
                    )
                else:
                    score = np.array(
                        [float(np.dot(q[idx + (i,)], k[idx + (j,)])) / h2 for j in kept]
                    )
                kern = np.exp(score - np.max(score))
                row = np.zeros(m, dtype=np.float64)
                row[kept] = kern / np.sum(kern)
            weights[idx + (i,)] = row
            acc = np.zeros(v.shape[-1], dtype=np.float64)
            for j in range(m):
                acc += row[j] * v[idx + (j,)]
            out[idx + (i,)] = acc
    return out, weights


def oracle_kde_log_density(
    q: np.ndarray,
    k: np.ndarray,
    h,
    keep: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``log`` of the Gaussian kernel density estimate at every query."""
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    lead, n, m, d = q.shape[:-2], q.shape[-2], k.shape[-2], q.shape[-1]
    bw = _bandwidth(h, lead, n)
    out = np.zeros(lead + (n,), dtype=np.float64)
    for idx in np.ndindex(*lead):
        kept = list(range(m))
        if keep is not None:
            chosen = [j for j in range(m) if bool(np.asarray(keep)[idx + (j,)])]
            kept = chosen if chosen else kept
        for i in range(n):
            h2 = float(bw[idx + (i, 0)]) ** 2
            expo = np.array(
                [-float(np.sum((q[idx + (i,)] - k[idx + (j,)]) ** 2)) / (2.0 * h2) for j in kept]
            )
            top = float(np.max(expo))
            log_sum = top + math.log(float(np.sum(np.exp(expo - top))))
            out[idx + (i,)] = (
                log_sum - math.log(len(kept)) - 0.5 * d * math.log(2.0 * math.pi * h2)
            )
    return out


def oracle_iterate(
    x: np.ndarray,
    h,
    num_steps: int,
    step_size: float = 1.0,
    blurring: bool = True,
    keep: Optional[np.ndarray] = None,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Damped mean-shift iteration built from :func:`oracle_step`."""
    x = np.asarray(x, dtype=np.float64)
    if num_steps == 0:
        return x
    if kernel == "vmf":
        x = _unit_rows(x)
    z = x
    for _ in range(num_steps):
        keys = z if blurring else x
        moved, _ = oracle_step(z, keys, keys, h, keep, kernel)
        z = (1.0 - step_size) * z + step_size * moved
        if kernel == "vmf":
            z = _unit_rows(z)
    return z


def oracle_median_bandwidth(x: np.ndarray, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Median distance over unordered pairs of valid points, per sample."""
    x = np.asarray(x, dtype=np.float64)
    batch, n = x.shape[0], x.shape[1]
    out = np.ones((batch, 1, 1), dtype=np.float64)
    for b in range(batch):
        valid = [i for i in range(n) if keep is None or bool(np.asarray(keep)[b, i])]
        dists = [
            math.sqrt(float(np.sum((x[b, i] - x[b, j]) ** 2)))
            for pos, i in enumerate(valid)
            for j in valid[pos + 1:]
        ]
        if dists:
            out[b, 0, 0] = max(float(np.median(dists)), _MEDIAN_FLOOR)
    return out
