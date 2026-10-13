"""
Synthetic HMM data and ground-truth references for the Neural HMM trainer
====================================================================

Sequences are sampled from a known hidden Markov model, so that a trained model can be
scored against the truth on both of its jobs: the likelihood of the observations and
the recovery of the hidden states.

The ground truth (:class:`TrueHMM`, built by :func:`make_true_hmm`) has no free random
parameter; the seed only drives the sampling:

- ``pi`` is uniform over the ``K`` states.
- The transition matrix is sticky: ``stay_prob`` on the diagonal and
  ``(1 - stay_prob) / (K - 1)`` everywhere else.
- **Gaussian** observations: state ``k`` emits ``N(mu_k, I)`` in ``obs_dim`` dimensions.
  The means lie on one line along the all-ones direction, consecutive states
  ``separation`` apart (in units of the standard deviation) and centred on 0, so the
  closest pair of states is exactly ``separation`` apart.
- **Categorical** observations: symbol ``v`` belongs to state ``v mod K``. Each state
  puts :data:`OWN_SYMBOL_MASS` of its probability uniformly on its own symbols and the
  rest uniformly on the others. ``separation`` is not read.

:func:`sample_hmm_dataset` draws ``N`` right-padded sequences whose valid lengths are
uniform on ``[min_seq_len, seq_len]``. :func:`sample_splits` draws the train,
validation and test splits from three disjoint random streams of one seed.

The references are plain numpy and share no code with the model package:

- :func:`true_log_likelihood` / :func:`true_nll_per_step`: ``log p(x)`` under the true
  HMM by the scaled forward algorithm (Rabiner 1989, Section V-A).
- :func:`true_path_log_prob` / :func:`true_conditional_nll_per_step`: ``log p(x, z)``
  and ``-log p(z | x)`` under the true HMM, the reference of a supervised model.
- :func:`true_log_potentials`: the true parameters in the model's output shapes, for
  decoding with the model package's own ``inference.viterbi_decode``.
- :func:`majority_rate`, :func:`accuracy`, :func:`hungarian_mapping` and
  :func:`matched_accuracy`: state accuracy, where an unsupervised model's state labels
  are first matched to the true labels by the assignment that maximizes agreement
  (``scipy.optimize.linear_sum_assignment`` on the confusion matrix).

Padding convention, shared with the model: ``mask`` is ``(N, T)`` float32, 1.0 on a
contiguous valid prefix; padded ``states`` are ``-1`` and padded observations 0.

References:
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected Applications in
      Speech Recognition. Proceedings of the IEEE 77(2), pp. 257-286.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Sequence, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------

#: Observation types the generator supports.
OBS_TYPES = ("gaussian", "categorical")

#: Probability a categorical state puts on the symbols it owns.
OWN_SYMBOL_MASS = 0.8

#: Last entry of each split's seed list (the ASCII bytes of TRN, VAL, TST): one
#: independent random stream per split from one user seed.
SPLIT_TAGS: Dict[str, int] = {"train": 0x54524E, "val": 0x56414C, "test": 0x545354}

# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TrueHMM:
    """The parameters of the ground-truth HMM, as float64 numpy arrays.

    :param pi: ``(K,)`` initial state distribution.
    :param transition: ``(K, K)`` transition matrix, row = source, column = destination.
    :param obs_type: ``gaussian`` or ``categorical``.
    :param means: ``(K, D)`` Gaussian means (unit standard deviation), or ``None``.
    :param emission_probs: ``(K, V)`` categorical emission probabilities, or ``None``.
    """

    pi: np.ndarray
    transition: np.ndarray
    obs_type: str
    means: Optional[np.ndarray] = None
    emission_probs: Optional[np.ndarray] = None

    @property
    def n_states(self) -> int:
        """Number of hidden states ``K``."""
        return int(self.pi.shape[0])


class HMMDataset(NamedTuple):
    """One split of sampled sequences and the HMM it came from.

    :param obs: ``(N, T, D)`` float32 (gaussian) or ``(N, T)`` int32 (categorical).
    :param mask: ``(N, T)`` float32, 1.0 on the valid prefix.
    :param states: ``(N, T)`` int32 true states, ``-1`` at padded steps.
    :param true_hmm: The generating :class:`TrueHMM`.
    """

    obs: np.ndarray
    mask: np.ndarray
    states: np.ndarray
    true_hmm: TrueHMM


# ---------------------------------------------------------------------


def make_true_hmm(
        n_states: int,
        obs_type: str,
        obs_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        separation: float = 4.0,
        stay_prob: float = 0.9,
) -> TrueHMM:
    """Build the ground-truth HMM described in the module docstring.

    :param n_states: ``K >= 1``.
    :param obs_type: ``gaussian`` or ``categorical``.
    :param obs_dim: ``D >= 1``; required for ``gaussian``.
    :param vocab_size: ``V >= K``; required for ``categorical``.
    :param separation: Distance between consecutive Gaussian means, in standard
        deviations; finite and ``>= 0``. Not read for ``categorical``.
    :param stay_prob: Diagonal of the transition matrix, in ``(0, 1)``. Not read when
        ``K = 1``.
    :return: The parameters.
    :raises ValueError: If an argument is out of range or missing for ``obs_type``.
    """
    if obs_type not in OBS_TYPES:
        raise ValueError(f"Unknown obs_type '{obs_type}'. Available: {list(OBS_TYPES)}.")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}.")
    if n_states > 1 and not 0.0 < stay_prob < 1.0:
        raise ValueError(f"stay_prob must lie in (0, 1), got {stay_prob}.")
    k = n_states
    pi = np.full(k, 1.0 / k)
    if k == 1:
        transition = np.ones((1, 1))
    else:
        transition = np.full((k, k), (1.0 - stay_prob) / (k - 1))
        np.fill_diagonal(transition, stay_prob)

    if obs_type == "gaussian":
        if obs_dim is None or obs_dim < 1:
            raise ValueError(f"obs_type='gaussian' needs obs_dim >= 1, got {obs_dim}.")
        if not (np.isfinite(separation) and separation >= 0):
            raise ValueError(f"separation must be finite and >= 0, got {separation}.")
        direction = np.ones(obs_dim) / np.sqrt(obs_dim)
        offsets = (np.arange(k) - (k - 1) / 2.0) * separation
        return TrueHMM(pi, transition, obs_type, means=offsets[:, None] * direction[None, :])

    if vocab_size is None or vocab_size < k:
        raise ValueError(
            f"obs_type='categorical' needs vocab_size >= n_states ({k}) so that every state "
            f"owns a symbol, got {vocab_size}."
        )
    owner = np.arange(vocab_size) % k
    probs = np.zeros((k, vocab_size))
    for state in range(k):
        own = owner == state
        n_own, n_other = int(own.sum()), int((~own).sum())
        if n_other == 0:
            probs[state] = 1.0 / vocab_size
        else:
            probs[state, own] = OWN_SYMBOL_MASS / n_own
            probs[state, ~own] = (1.0 - OWN_SYMBOL_MASS) / n_other
    return TrueHMM(pi, transition, obs_type, emission_probs=probs)


def _draw(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one index per row of ``probs`` (``(N, K)`` rows summing to 1) by inverse CDF."""
    cdf = np.cumsum(probs, axis=1)
    draws = (rng.random(probs.shape[0])[:, None] > cdf).sum(axis=1)
    return np.minimum(draws, probs.shape[1] - 1)


def sample_hmm_dataset(
        n_sequences: int,
        seq_len: int,
        min_seq_len: int,
        n_states: int,
        obs_type: str,
        obs_dim: Optional[int] = None,
        vocab_size: Optional[int] = None,
        separation: float = 4.0,
        stay_prob: float = 0.9,
        seed: Union[int, Sequence[int]] = 0,
) -> HMMDataset:
    """Sample ``n_sequences`` right-padded sequences from :func:`make_true_hmm`'s HMM.

    The draws come in a fixed order from one ``numpy.random.default_rng(seed)``: the
    valid lengths, the state paths, then the observations.

    :param n_sequences: ``N >= 1``.
    :param seq_len: Padded length ``T >= 1``.
    :param min_seq_len: Shortest valid length, ``1 <= min_seq_len <= seq_len``.
    :param n_states: ``K``.
    :param obs_type: ``gaussian`` or ``categorical``.
    :param obs_dim: ``D`` of a ``gaussian`` observation.
    :param vocab_size: ``V`` of a ``categorical`` observation.
    :param separation: See :func:`make_true_hmm`.
    :param stay_prob: See :func:`make_true_hmm`.
    :param seed: Seed (or seed list) of the generator.
    :return: The split and its true HMM.
    :raises ValueError: If a size is out of range, or :func:`make_true_hmm` refuses.
    """
    if n_sequences < 1 or seq_len < 1 or not 1 <= min_seq_len <= seq_len:
        raise ValueError(
            f"Need n_sequences >= 1 and 1 <= min_seq_len <= seq_len, got n_sequences="
            f"{n_sequences}, min_seq_len={min_seq_len}, seq_len={seq_len}."
        )
    hmm = make_true_hmm(n_states, obs_type, obs_dim, vocab_size, separation, stay_prob)
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_seq_len, seq_len + 1, size=n_sequences)
    valid = np.arange(seq_len)[None, :] < lengths[:, None]

    states = np.empty((n_sequences, seq_len), dtype=np.int64)
    states[:, 0] = _draw(rng, np.broadcast_to(hmm.pi, (n_sequences, n_states)))
    for t in range(1, seq_len):
        states[:, t] = _draw(rng, hmm.transition[states[:, t - 1]])

    if obs_type == "gaussian":
        obs = hmm.means[states] + rng.standard_normal((n_sequences, seq_len, obs_dim))
        obs = np.where(valid[:, :, None], obs, 0.0).astype(np.float32)
    else:
        flat = _draw(rng, hmm.emission_probs[states.reshape(-1)])
        obs = np.where(valid, flat.reshape(n_sequences, seq_len), 0).astype(np.int32)
    return HMMDataset(
        obs=obs,
        mask=valid.astype(np.float32),
        states=np.where(valid, states, -1).astype(np.int32),
        true_hmm=hmm,
    )


def split_seed(seed: int, split: str) -> list:
    """Return the seed list of one split: ``[seed, SPLIT_TAGS[split]]``.

    :param seed: The user seed.
    :param split: A key of :data:`SPLIT_TAGS`.
    :return: A seed list for ``numpy.random.default_rng``.
    """
    return [int(seed), SPLIT_TAGS[split]]


def sample_splits(sizes: Dict[str, int], seed: int, **kwargs) -> Dict[str, HMMDataset]:
    """Sample every split of ``sizes`` from its own stream of ``seed``.

    :param sizes: Split name (a key of :data:`SPLIT_TAGS`) to sequence count.
    :param seed: The user seed.
    :param kwargs: Every other argument of :func:`sample_hmm_dataset`.
    :return: Split name to :class:`HMMDataset`; the splits share one :class:`TrueHMM`.
    """
    return {
        name: sample_hmm_dataset(n_sequences=count, seed=split_seed(seed, name), **kwargs)
        for name, count in sizes.items()
    }

# ---------------------------------------------------------------------


def true_log_emissions(hmm: TrueHMM, obs: np.ndarray) -> np.ndarray:
    """Return ``log p(x_t | z_t = k)`` under the true HMM, ``(N, T, K)`` float64.

    :param hmm: The true HMM.
    :param obs: Observations as :func:`sample_hmm_dataset` returns them.
    :return: Emission log-probabilities; padded steps hold the value of their padded
        observation and must be masked by the caller.
    """
    if hmm.obs_type == "gaussian":
        x = np.asarray(obs, dtype=np.float64)
        squared = np.sum(np.square(x[:, :, None, :] - hmm.means[None, None]), axis=-1)
        return -0.5 * squared - 0.5 * x.shape[-1] * np.log(2.0 * np.pi)
    log_probs = np.log(hmm.emission_probs)
    return np.moveaxis(log_probs[:, np.asarray(obs, dtype=np.int64)], 0, -1)


def true_log_likelihood(hmm: TrueHMM, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``log p(x)`` of every sequence under the true HMM, by the scaled forward.

    Each step's forward vector is normalized to sum to 1 and the log of the normalizer
    accumulated; emissions are shifted by their per-step maximum before ``exp``. A
    length-0 sequence has ``log p(x) = 0``.

    :param hmm: The true HMM.
    :param obs: Observations.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``(N,)`` float64 log-likelihoods.
    """
    log_b = true_log_emissions(hmm, obs)
    shift = log_b.max(axis=-1)
    b = np.exp(log_b - shift[:, :, None])
    lengths = (np.asarray(mask) > 0).sum(axis=1)
    log_lik = np.zeros(len(lengths))
    alpha = np.zeros((len(lengths), hmm.n_states))
    for t in range(log_b.shape[1]):
        rows = lengths > t
        if not rows.any():
            break
        if t == 0:
            step = hmm.pi[None, :] * b[rows, 0]
        else:
            step = (alpha[rows] @ hmm.transition) * b[rows, t]
        norm = step.sum(axis=1)
        alpha[rows] = step / norm[:, None]
        log_lik[rows] += np.log(norm) + shift[rows, t]
    return log_lik


def true_path_log_prob(
        hmm: TrueHMM, obs: np.ndarray, states: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return ``log p(x, z)`` of every sequence and its state path under the true HMM.

    :param hmm: The true HMM.
    :param obs: Observations.
    :param states: ``(N, T)`` states; padded entries are ignored.
    :param mask: ``(N, T)`` right-padding mask.
    :return: ``(N,)`` float64 joint log-probabilities (0 for a length-0 sequence).
    """
    valid = np.asarray(mask) > 0
    z = np.where(valid, states, 0).astype(np.int64)
    log_b = np.take_along_axis(true_log_emissions(hmm, obs), z[:, :, None], axis=-1)[:, :, 0]
    log_pi, log_a = np.log(hmm.pi), np.log(hmm.transition)
    terms = log_b.copy()
    terms[:, 0] += log_pi[z[:, 0]]
    terms[:, 1:] += log_a[z[:, :-1], z[:, 1:]]
    return np.sum(np.where(valid, terms, 0.0), axis=1)


def _per_step(total_log_prob: np.ndarray, mask: np.ndarray) -> float:
    """``-sum(total_log_prob) / number of valid steps``; raises on an all-padding split."""
    steps = float((np.asarray(mask) > 0).sum())
    if steps == 0:
        raise ValueError("The split has no valid step.")
    return float(-np.sum(total_log_prob) / steps)


def true_nll_per_step(hmm: TrueHMM, obs: np.ndarray, mask: np.ndarray) -> float:
    """Return ``-log p(x)`` per valid step under the true HMM, summed over the split.

    The same reduction as the model's training loss (total NLL over total valid
    steps), so the two are directly comparable.

    :param hmm: The true HMM.
    :param obs: Observations.
    :param mask: ``(N, T)`` right-padding mask.
    :return: Nats per valid step.
    :raises ValueError: If the split has no valid step.
    """
    return _per_step(true_log_likelihood(hmm, obs, mask), mask)


def true_conditional_nll_per_step(
        hmm: TrueHMM, obs: np.ndarray, states: np.ndarray, mask: np.ndarray
) -> float:
    """Return ``-log p(z | x)`` per valid step under the true HMM.

    The reference of a supervised (discriminative) model, whose training loss is the
    same quantity under its own parameters.

    :param hmm: The true HMM.
    :param obs: Observations.
    :param states: ``(N, T)`` true states.
    :param mask: ``(N, T)`` right-padding mask.
    :return: Nats per valid step.
    :raises ValueError: If the split has no valid step.
    """
    conditional = true_path_log_prob(hmm, obs, states, mask) - true_log_likelihood(hmm, obs, mask)
    return _per_step(conditional, mask)


def true_log_potentials(hmm: TrueHMM, obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the true parameters in the model's output shapes, float64.

    :param hmm: The true HMM.
    :param obs: Observations, ``N`` sequences of length ``T``.
    :return: ``{"log_pi": (N, K), "log_A": (N, T, K, K), "log_emit": (N, T, K)}``, the
        input of the model package's ``inference`` functions.
    """
    log_emit = true_log_emissions(hmm, obs)
    n, steps, k = log_emit.shape
    log_pi, log_a = np.log(hmm.pi), np.log(hmm.transition)
    return {
        "log_pi": np.broadcast_to(log_pi, (n, k)).copy(),
        "log_A": np.broadcast_to(log_a, (n, steps, k, k)).copy(),
        "log_emit": log_emit,
    }

# ---------------------------------------------------------------------


def majority_rate(states: np.ndarray) -> float:
    """Return the share of valid steps that hold the most frequent state.

    The accuracy of the best constant predictor; ``states < 0`` are padding.

    :param states: ``(N, T)`` states.
    :return: A rate in ``(0, 1]``.
    :raises ValueError: If no step is valid.
    """
    valid = np.asarray(states)[np.asarray(states) >= 0]
    if valid.size == 0:
        raise ValueError("No valid step.")
    return float(np.bincount(valid).max() / valid.size)


def accuracy(predicted: np.ndarray, states: np.ndarray) -> float:
    """Return the share of valid steps (``states >= 0``) where ``predicted == states``.

    :param predicted: ``(N, T)`` predicted states.
    :param states: ``(N, T)`` true states, ``-1`` at padding.
    :return: A rate in ``[0, 1]``.
    :raises ValueError: If no step is valid.
    """
    valid = np.asarray(states) >= 0
    if not valid.any():
        raise ValueError("No valid step.")
    return float(np.mean(np.asarray(predicted)[valid] == np.asarray(states)[valid]))


def hungarian_mapping(predicted: np.ndarray, states: np.ndarray, n_states: int) -> np.ndarray:
    """Return the label map that makes ``predicted`` agree most with ``states``.

    ``mapping[p]`` is the true state assigned to predicted label ``p``: the one-to-one
    assignment maximizing the number of valid steps where they agree
    (``scipy.optimize.linear_sum_assignment`` on the confusion matrix).

    :param predicted: ``(N, T)`` predicted labels in ``[0, n_states)`` at valid steps.
    :param states: ``(N, T)`` true states, ``-1`` at padding.
    :param n_states: ``K``.
    :return: ``(K,)`` int64 permutation.
    :raises ValueError: If a predicted label at a valid step is outside ``[0, K)``.
    """
    valid = np.asarray(states) >= 0
    pred, true = np.asarray(predicted)[valid], np.asarray(states)[valid]
    if pred.size and (pred.min() < 0 or pred.max() >= n_states):
        raise ValueError(f"Predicted labels must lie in [0, {n_states}) at valid steps.")
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(confusion, (pred, true), 1)
    rows, cols = linear_sum_assignment(confusion, maximize=True)
    mapping = np.empty(n_states, dtype=np.int64)
    mapping[rows] = cols
    return mapping


def matched_accuracy(predicted: np.ndarray, states: np.ndarray, n_states: int) -> float:
    """Return the state accuracy after the :func:`hungarian_mapping` of the labels.

    The score of an unsupervised model, whose state labels are an arbitrary
    permutation of the true ones. A constant predictor scores :func:`majority_rate`.

    :param predicted: ``(N, T)`` predicted labels.
    :param states: ``(N, T)`` true states, ``-1`` at padding.
    :param n_states: ``K``.
    :return: A rate in ``[0, 1]``.
    """
    mapping = hungarian_mapping(predicted, states, n_states)
    valid = np.asarray(states) >= 0
    relabelled = np.where(valid, mapping[np.where(valid, predicted, 0)], -1)
    return accuracy(relabelled, states)

# ---------------------------------------------------------------------
