"""Tests of ``src/train/neural_hmm/data.py``: the sampler and the ground-truth references.

What is pinned here, and the mutation each group is meant to catch:

- shapes, dtypes and the right-padding layout (``-1`` states, zero observations and a
  contiguous valid prefix), and the valid lengths covering ``[min_seq_len, seq_len]``;
- one seed reproduces a split, another changes it, and the three splits of
  :func:`sample_splits` come from disjoint streams;
- the sampled statistics are the documented HMM's: the self-transition frequency
  within 3 sigma of ``stay_prob``, a uniform first state, the Gaussian means and unit
  standard deviation, the categorical own-symbol mass;
- ``log p(x)`` (scaled forward), ``log p(x, z)`` and ``log p(z | x)`` equal a brute-force
  enumeration of every state path, computed here with scipy densities rather than the
  module's own emission code (a dropped first emission, a transposed transition matrix
  or a wrong normalizer all fail);
- the per-step NLL equals the model package's ``inference.forward_log_likelihood`` run
  on :func:`true_log_potentials`;
- :func:`matched_accuracy` matches labels by the optimal assignment, not greedily and
  not at all; a constant predictor scores :func:`majority_rate`.

Nothing here touches a model, a GPU or the file system.
"""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from dl_techniques.models.time_series.neural_hmm import inference
from train.neural_hmm.data import (
    OWN_SYMBOL_MASS, SPLIT_TAGS, accuracy, hungarian_mapping, make_true_hmm, majority_rate,
    matched_accuracy, sample_hmm_dataset, sample_splits, split_seed,
    true_conditional_nll_per_step, true_log_likelihood, true_log_potentials,
    true_nll_per_step, true_path_log_prob,
)

GAUSSIAN = {"obs_type": "gaussian", "obs_dim": 2, "separation": 3.0}
CATEGORICAL = {"obs_type": "categorical", "vocab_size": 7}
OBS_KINDS = {"gaussian": GAUSSIAN, "categorical": CATEGORICAL}


def _sample(kind="gaussian", n=40, seq_len=9, min_seq_len=3, n_states=3, stay_prob=0.8, seed=0):
    return sample_hmm_dataset(n, seq_len, min_seq_len, n_states, stay_prob=stay_prob, seed=seed,
                              **OBS_KINDS[kind])


# ---------------------------------------------------------------------
# layout, seeds, splits
# ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(OBS_KINDS))
def test_shapes_dtypes_and_right_padding(kind) -> None:
    data = _sample(kind)
    n, steps = 40, 9
    if kind == "gaussian":
        assert data.obs.shape == (n, steps, 2) and data.obs.dtype == np.float32
    else:
        assert data.obs.shape == (n, steps) and data.obs.dtype == np.int32
        assert data.obs.min() >= 0 and data.obs.max() < 7
    assert data.mask.shape == (n, steps) and data.mask.dtype == np.float32
    assert data.states.shape == (n, steps) and data.states.dtype == np.int32

    lengths = data.mask.sum(axis=1).astype(int)
    prefix = (np.arange(steps)[None, :] < lengths[:, None]).astype(np.float32)
    np.testing.assert_array_equal(data.mask, prefix)
    assert np.all((data.states == -1) == (data.mask == 0))
    assert data.states[data.mask > 0].min() >= 0 and data.states.max() < 3
    padded = data.obs[data.mask == 0]
    assert padded.size and np.all(padded == 0)


def test_valid_lengths_cover_the_whole_range() -> None:
    lengths = _sample(n=600, seq_len=8, min_seq_len=3).mask.sum(axis=1).astype(int)
    assert set(lengths.tolist()) == set(range(3, 9))


def test_the_same_seed_reproduces_and_another_seed_changes_the_data() -> None:
    first, again, other = _sample(seed=5), _sample(seed=5), _sample(seed=6)
    for name in ("obs", "mask", "states"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
    assert not np.array_equal(first.states, other.states)
    assert not np.array_equal(first.obs, other.obs)


def test_the_splits_come_from_disjoint_streams_of_one_seed() -> None:
    kwargs = dict(seq_len=9, min_seq_len=3, n_states=3, stay_prob=0.8, **GAUSSIAN)
    splits = sample_splits({"train": 30, "val": 30, "test": 30}, seed=4, **kwargs)
    assert set(splits) == set(SPLIT_TAGS)
    for name, split in splits.items():
        alone = sample_hmm_dataset(30, seed=split_seed(4, name), **kwargs)
        np.testing.assert_array_equal(split.obs, alone.obs)
    for a, b in itertools.combinations(sorted(splits), 2):
        assert not np.array_equal(splits[a].obs, splits[b].obs), (a, b)
    assert len(set(SPLIT_TAGS.values())) == 3
    for name in ("val", "test"):
        np.testing.assert_array_equal(splits[name].true_hmm.transition, splits["train"].true_hmm.transition)
        np.testing.assert_array_equal(splits[name].true_hmm.means, splits["train"].true_hmm.means)


# ---------------------------------------------------------------------
# sampled statistics
# ---------------------------------------------------------------------


@pytest.mark.parametrize("stay_prob", [0.9, 0.6])
def test_the_self_transition_frequency_is_stay_prob_within_three_sigma(stay_prob) -> None:
    data = _sample(n=2000, seq_len=40, min_seq_len=40, n_states=4, stay_prob=stay_prob, seed=1)
    stays = data.states[:, 1:] == data.states[:, :-1]
    count = stays.size
    sigma = np.sqrt(stay_prob * (1 - stay_prob) / count)
    assert abs(stays.mean() - stay_prob) < 3 * sigma, (stays.mean(), stay_prob, sigma)
    # every move is equally likely to each of the other states
    moved_to = data.states[:, 1:][~stays]
    moved_from = data.states[:, :-1][~stays]
    offset = (moved_to - moved_from) % 4
    counts = np.bincount(offset, minlength=4)[1:]
    expected = len(offset) / 3
    assert np.all(np.abs(counts - expected) < 3 * np.sqrt(expected * (2 / 3))), counts


def test_the_first_state_is_uniform() -> None:
    first = _sample(n=6000, n_states=3, seed=2).states[:, 0]
    counts = np.bincount(first, minlength=3)
    sigma = np.sqrt(6000 * (1 / 3) * (2 / 3))
    assert np.all(np.abs(counts - 2000) < 3 * sigma), counts


def test_gaussian_means_are_separation_apart_with_unit_noise() -> None:
    hmm = make_true_hmm(4, "gaussian", obs_dim=3, separation=2.5)
    gaps = np.linalg.norm(np.diff(hmm.means, axis=0), axis=1)
    np.testing.assert_allclose(gaps, 2.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(hmm.means.mean(axis=0), 0.0, rtol=0, atol=1e-12)
    pairwise = [np.linalg.norm(a - b) for a, b in itertools.combinations(hmm.means, 2)]
    assert min(pairwise) == pytest.approx(2.5, abs=1e-12)

    data = sample_hmm_dataset(3000, 20, 20, 4, obs_type="gaussian", obs_dim=3, separation=2.5,
                              stay_prob=0.7, seed=3)
    for state in range(4):
        x = data.obs[data.states == state].astype(np.float64)
        np.testing.assert_allclose(x.mean(axis=0), hmm.means[state], rtol=0,
                                   atol=4 / np.sqrt(len(x)))
        np.testing.assert_allclose(x.std(axis=0), 1.0, rtol=0, atol=0.05)


def test_categorical_states_put_the_own_symbol_mass_on_their_symbols() -> None:
    data = sample_hmm_dataset(3000, 20, 20, 3, obs_type="categorical", vocab_size=7,
                              stay_prob=0.8, seed=4)
    valid = data.states >= 0
    own = (data.obs[valid] % 3) == data.states[valid]
    sigma = np.sqrt(OWN_SYMBOL_MASS * (1 - OWN_SYMBOL_MASS) / own.size)
    assert abs(own.mean() - OWN_SYMBOL_MASS) < 3 * sigma
    probs = make_true_hmm(3, "categorical", vocab_size=7).emission_probs
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [
    dict(n_states=0), dict(stay_prob=1.0), dict(stay_prob=0.0), dict(obs_type="poisson"),
    dict(obs_dim=0), dict(separation=float("nan")), dict(separation=-1.0),
    dict(obs_type="categorical", obs_dim=None, vocab_size=2),
])
def test_make_true_hmm_refuses_bad_arguments(bad) -> None:
    kwargs = dict(n_states=3, obs_type="gaussian", obs_dim=2, vocab_size=None,
                  separation=1.0, stay_prob=0.9)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        make_true_hmm(**kwargs)


@pytest.mark.parametrize("sizes", [(0, 5, 3), (4, 5, 6), (4, 5, 0)])
def test_sample_hmm_dataset_refuses_bad_sizes(sizes) -> None:
    n, seq_len, min_seq_len = sizes
    with pytest.raises(ValueError):
        sample_hmm_dataset(n, seq_len, min_seq_len, 2, **GAUSSIAN)


# ---------------------------------------------------------------------
# references against brute force and against the model's inference
# ---------------------------------------------------------------------


def _brute_force(hmm, obs, states, mask):
    """``log p(x)`` and ``log p(x, z)`` by enumerating every path, with scipy densities."""
    k = hmm.n_states
    marginal, joint = [], []
    for row in range(len(mask)):
        length = int((mask[row] > 0).sum())
        if hmm.obs_type == "gaussian":
            log_b = np.stack([norm.logpdf(obs[row, :length].astype(np.float64),
                                          loc=hmm.means[s]).sum(axis=-1) for s in range(k)], axis=-1)
        else:
            log_b = np.log(hmm.emission_probs[:, obs[row, :length]]).T

        def score(path):
            total = np.log(hmm.pi[path[0]]) + log_b[0, path[0]]
            for t in range(1, length):
                total += np.log(hmm.transition[path[t - 1], path[t]]) + log_b[t, path[t]]
            return total

        if length == 0:
            marginal.append(0.0)
            joint.append(0.0)
            continue
        marginal.append(logsumexp([score(p) for p in itertools.product(range(k), repeat=length)]))
        joint.append(score(states[row, :length]))
    return np.array(marginal), np.array(joint)


@pytest.mark.parametrize("kind", sorted(OBS_KINDS))
@pytest.mark.parametrize("n_states", [1, 2, 3])
@pytest.mark.parametrize("seq_len", [1, 2, 5])
def test_the_references_equal_brute_force_enumeration(kind, n_states, seq_len) -> None:
    if kind == "categorical":
        kwargs = dict(obs_type="categorical", vocab_size=max(n_states, 4))
    else:
        kwargs = dict(obs_type="gaussian", obs_dim=2, separation=1.5)
    data = sample_hmm_dataset(6, seq_len, 1, n_states, stay_prob=0.7, seed=seq_len, **kwargs)
    mask = data.mask.copy()
    mask[0] = 0.0  # a length-0 sequence
    states = np.where(mask > 0, data.states, -1)
    marginal, joint = _brute_force(data.true_hmm, data.obs, states, mask)

    np.testing.assert_allclose(true_log_likelihood(data.true_hmm, data.obs, mask), marginal,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(true_path_log_prob(data.true_hmm, data.obs, states, mask), joint,
                               rtol=0, atol=1e-10)
    steps = mask.sum()
    assert true_nll_per_step(data.true_hmm, data.obs, mask) == pytest.approx(
        -marginal.sum() / steps, abs=1e-10)
    assert true_conditional_nll_per_step(data.true_hmm, data.obs, states, mask) == pytest.approx(
        -(joint - marginal).sum() / steps, abs=1e-10)


@pytest.mark.parametrize("kind", sorted(OBS_KINDS))
def test_the_true_nll_equals_the_model_inference_on_the_true_parameters(kind) -> None:
    data = _sample(kind, n=25, seq_len=30, min_seq_len=1, n_states=4, seed=9)
    potentials = true_log_potentials(data.true_hmm, data.obs)
    assert potentials["log_A"].shape == (25, 30, 4, 4)
    log_z = np.asarray(inference.forward_log_likelihood(
        potentials["log_pi"], potentials["log_A"], potentials["log_emit"], data.mask))
    np.testing.assert_allclose(true_log_likelihood(data.true_hmm, data.obs, data.mask), log_z,
                               rtol=0, atol=1e-9)
    assert true_nll_per_step(data.true_hmm, data.obs, data.mask) == pytest.approx(
        -log_z.sum() / data.mask.sum(), abs=1e-10)


def test_a_split_without_a_valid_step_is_refused() -> None:
    data = _sample()
    with pytest.raises(ValueError, match="no valid step"):
        true_nll_per_step(data.true_hmm, data.obs, np.zeros_like(data.mask))


# ---------------------------------------------------------------------
# accuracy, matching, base rate
# ---------------------------------------------------------------------


def test_a_permuted_perfect_labelling_scores_one_after_matching() -> None:
    data = _sample(n_states=3, n=50, seed=11)
    permutation = np.array([2, 0, 1])
    predicted = np.where(data.states >= 0, permutation[np.maximum(data.states, 0)], -1)
    assert accuracy(predicted, data.states) < 0.5
    assert matched_accuracy(predicted, data.states, 3) == 1.0
    mapping = hungarian_mapping(predicted, data.states, 3)
    np.testing.assert_array_equal(mapping[permutation], np.arange(3))


def test_matching_is_the_optimal_assignment_not_the_greedy_one() -> None:
    # confusion[predicted, true]: greedy takes (0 -> 0, 5 steps) and is left with
    # (1 -> 1, 0 steps) = 5; the optimum is (0 -> 1, 4) + (1 -> 0, 4) = 8.
    pairs = [(0, 0)] * 5 + [(0, 1)] * 4 + [(1, 0)] * 4
    predicted = np.array([[p for p, _ in pairs]])
    states = np.array([[s for _, s in pairs]])
    assert matched_accuracy(predicted, states, 2) == pytest.approx(8 / 13)
    np.testing.assert_array_equal(hungarian_mapping(predicted, states, 2), [1, 0])


def test_a_constant_predictor_scores_the_majority_rate() -> None:
    data = _sample(n_states=4, n=80, seed=12)
    for label in range(4):
        constant = np.where(data.states >= 0, label, -1)
        assert matched_accuracy(constant, data.states, 4) == pytest.approx(majority_rate(data.states))


def test_majority_rate_and_accuracy_ignore_padding() -> None:
    states = np.array([[0, 0, 1, -1, -1], [1, 1, 1, 1, -1]])
    assert majority_rate(states) == pytest.approx(5 / 7)
    predicted = np.array([[0, 1, 1, 0, 0], [1, 1, 0, 1, 0]])
    assert accuracy(predicted, states) == pytest.approx(5 / 7)


def test_an_out_of_range_predicted_label_is_refused() -> None:
    states = np.array([[0, 1, 1]])
    with pytest.raises(ValueError, match="Predicted labels"):
        hungarian_mapping(np.array([[0, -1, 1]]), states, 2)
