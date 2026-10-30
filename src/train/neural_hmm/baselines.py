"""A classical baseline for the neural HMM: Baum-Welch in NumPy, on the same data format.

:func:`fit_baum_welch` fits a tabular HMM with diagonal-Gaussian or categorical emissions
by EM, over right-padded sequences. Padded steps are never read, so they may hold anything.
``init="kmeans"`` is the usual start, means from k-means on the valid observations;
``init="random"`` takes the means from random valid observations, or draws random
categorical rows, which separates what the initialization buys from what EM does.

:func:`potentials` gives a fitted model's ``log_pi``, ``log_A`` and ``log_emit`` in the
shapes of :func:`train.neural_hmm.data.true_log_potentials`, for the package's Viterbi.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2

INITS = ("kmeans", "random")
VARIANCE_FLOOR = 1e-3
COUNT_SMOOTHING = 1e-6


@dataclass
class BaumWelchHMM:
    """A fitted tabular HMM.

    :param pi: ``(K,)`` initial distribution.
    :param transition: ``(K, K)`` transition matrix, rows the source.
    :param means: ``(K, D)`` Gaussian means, or ``None``.
    :param variances: ``(K, D)`` Gaussian variances, or ``None``.
    :param emission_probs: ``(K, V)`` categorical probabilities, or ``None``.
    :param history: The training log-likelihood per valid step before each M-step.
    """

    pi: np.ndarray
    transition: np.ndarray
    means: Optional[np.ndarray] = None
    variances: Optional[np.ndarray] = None
    emission_probs: Optional[np.ndarray] = None
    history: List[float] = field(default_factory=list)


def log_emissions(hmm: BaumWelchHMM, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``log p(x_t | z_t = k)``, 0 at padded steps.

    :param hmm: The model.
    :param obs: ``(N, T, D)`` floats or ``(N, T)`` symbols.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``(N, T, K)`` float64.
    """
    valid = np.asarray(mask) > 0
    if hmm.emission_probs is not None:
        symbols = np.where(valid, np.asarray(obs), 0).astype(np.int64)
        log_b = np.moveaxis(np.log(hmm.emission_probs)[:, symbols], 0, -1)
    else:
        x = np.where(valid[:, :, None], np.asarray(obs, dtype=np.float64), 0.0)
        squared = np.square(x[:, :, None, :] - hmm.means[None, None]) / hmm.variances[None, None]
        log_b = -0.5 * np.sum(squared + np.log(2.0 * np.pi * hmm.variances[None, None]), axis=-1)
    return np.where(valid[:, :, None], log_b, 0.0)


def forward_backward(
        pi: np.ndarray, transition: np.ndarray, log_b: np.ndarray, mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the scaled forward-backward pass over a padded batch.

    :param pi: ``(K,)`` initial distribution.
    :param transition: ``(K, K)`` transition matrix.
    :param log_b: ``(N, T, K)`` emission log-probabilities.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``(gamma (N, T, K), xi_sum (K, K), log_lik (N,))``. ``gamma`` is 0 at padded
        steps, ``xi_sum`` the expected transition counts over the batch, and ``log_lik``
        0 for a length-0 sequence.
    """
    n, steps, k = log_b.shape
    valid = np.asarray(mask) > 0
    peak = log_b.max(axis=-1, keepdims=True)
    b = np.exp(log_b - peak)
    alpha = np.zeros((n, steps, k))
    scale = np.ones((n, steps))
    log_lik = np.where(valid, peak[:, :, 0], 0.0).sum(axis=1)
    for t in range(steps):
        rows = valid[:, t]
        step = (pi[None] if t == 0 else alpha[rows, t - 1] @ transition) * b[rows, t]
        scale[rows, t] = step.sum(axis=1)
        alpha[rows, t] = step / scale[rows, t, None]
    log_lik += np.log(scale).sum(axis=1)

    gamma = np.zeros_like(alpha)
    xi_sum = np.zeros((k, k))
    beta = np.ones((n, k))
    for t in range(steps - 1, -1, -1):
        rows = valid[:, t]
        gamma[rows, t] = alpha[rows, t] * beta[rows]
        if t > 0:
            weighted = b[rows, t] * beta[rows]
            xi_sum += transition * (
                alpha[rows, t - 1].T @ (weighted / scale[rows, t, None]))
            beta[rows] = (weighted @ transition.T) / scale[rows, t, None]
    return gamma, xi_sum, log_lik


def log_likelihood(hmm: BaumWelchHMM, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``log p(x)`` of every sequence.

    :param hmm: The model.
    :param obs: Observations.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``(N,)`` float64.
    """
    return forward_backward(hmm.pi, hmm.transition, log_emissions(hmm, obs, mask), mask)[2]


def potentials(hmm: BaumWelchHMM, obs: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the model's potentials, for the package's inference functions.

    :param hmm: The model.
    :param obs: Observations.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``{"log_pi": (N, K), "log_A": (N, T, K, K), "log_emit": (N, T, K)}``.
    """
    n, steps = np.shape(mask)
    k = len(hmm.pi)
    return {
        "log_pi": np.broadcast_to(np.log(hmm.pi), (n, k)).copy(),
        "log_A": np.broadcast_to(np.log(hmm.transition), (n, steps, k, k)).copy(),
        "log_emit": log_emissions(hmm, obs, mask),
    }


def _initial_means(x: np.ndarray, n_states: int, init: str, rng: np.random.Generator) -> np.ndarray:
    """Return the ``(K, D)`` starting means from ``(M, D)`` valid observations."""
    if init == "random":
        return x[rng.choice(len(x), n_states, replace=False)]
    with warnings.catch_warnings():
        # An emptied cluster keeps a random point, which is what a start needs.
        warnings.simplefilter("ignore", UserWarning)
        return kmeans2(x, n_states, iter=20, minit="++", missing="warn", seed=int(rng.integers(2 ** 31)))[0]


def fit_baum_welch(
        obs: np.ndarray,
        mask: np.ndarray,
        n_states: int,
        vocab_size: Optional[int] = None,
        init: str = "kmeans",
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int = 0,
) -> BaumWelchHMM:
    """Fit a tabular HMM by EM.

    :param obs: ``(N, T, D)`` floats (Gaussian), or ``(N, T)`` symbols with ``vocab_size``.
    :param mask: ``(N, T)`` right-padding mask.
    :param n_states: ``K``.
    :param vocab_size: ``V`` of categorical observations, else ``None``.
    :param init: One of :data:`INITS`. Categorical rows are always drawn at random, since
        symbols have no metric for k-means.
    :param max_iter: Most EM iterations.
    :param tol: Stop when the log-likelihood per valid step gains less than this.
    :param seed: Seed of the initialization.
    :return: The fitted model; ``history`` has the log-likelihood per step at each iteration.
    :raises ValueError: If ``init`` is unknown, or the mask has no valid step.
    """
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}.")
    valid = np.asarray(mask) > 0
    steps = float(valid.sum())
    if steps == 0:
        raise ValueError("fit_baum_welch needs at least one valid step.")
    rng = np.random.default_rng(seed)
    stay = 0.5
    transition = np.full((n_states, n_states), (1.0 - stay) / max(n_states - 1, 1))
    np.fill_diagonal(transition, stay)
    hmm = BaumWelchHMM(np.full(n_states, 1.0 / n_states), transition)
    if vocab_size is not None:
        hmm.emission_probs = rng.dirichlet(np.ones(vocab_size), n_states)
    else:
        x = np.asarray(obs, dtype=np.float64)[valid]
        hmm.means = _initial_means(x, n_states, init, rng)
        hmm.variances = np.broadcast_to(np.maximum(x.var(axis=0), VARIANCE_FLOOR), hmm.means.shape).copy()

    for _ in range(max_iter):
        gamma, xi_sum, log_lik = forward_backward(hmm.pi, hmm.transition, log_emissions(hmm, obs, mask), mask)
        hmm.history.append(float(log_lik.sum() / steps))
        if len(hmm.history) > 1 and hmm.history[-1] - hmm.history[-2] < tol:
            break
        hmm.pi = np.maximum(gamma[valid[:, 0], 0].mean(axis=0), COUNT_SMOOTHING)
        hmm.pi /= hmm.pi.sum()
        counts = xi_sum + COUNT_SMOOTHING
        hmm.transition = counts / counts.sum(axis=1, keepdims=True)
        occupancy = gamma[valid]
        totals = occupancy.sum(axis=0) + COUNT_SMOOTHING
        if vocab_size is not None:
            table = np.full((n_states, vocab_size), COUNT_SMOOTHING)
            np.add.at(table.T, np.asarray(obs)[valid].astype(np.int64), occupancy)
            hmm.emission_probs = table / table.sum(axis=1, keepdims=True)
        else:
            hmm.means = occupancy.T @ x / totals[:, None]
            spread = np.square(x[:, None, :] - hmm.means[None])
            hmm.variances = np.maximum(
                np.einsum("mk,mkd->kd", occupancy, spread) / totals[:, None], VARIANCE_FLOOR)
    return hmm
