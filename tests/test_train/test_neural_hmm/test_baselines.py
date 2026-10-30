"""``baselines.py``: Baum-Welch, against the true model and against brute force.

Guards:

* the forward-backward pass equals enumeration over every path, on a padded batch;
* with the true parameters the likelihood equals ``data.true_log_likelihood``;
* EM never lowers the training likelihood, in either observation type;
* EM recovers the true parameters and a held-out NLL gap near zero, from both inits, and
  padded observations, even NaN, change nothing;
* ``compare_baselines`` runs end to end at a tiny size and appends every row.
"""

import itertools
import json

import numpy as np
import pytest

from train.neural_hmm import baselines, compare_baselines, data


def _dataset(obs_type="gaussian", n=300, seed=0, n_states=3):
    shape = {"obs_dim": 2} if obs_type == "gaussian" else {"vocab_size": 6}
    return data.sample_hmm_dataset(n, 40, 20, n_states, obs_type, seed=seed, **shape)


def test_forward_backward_matches_enumeration_on_a_padded_batch():
    rng = np.random.default_rng(0)
    k, steps = 3, 5
    pi = rng.dirichlet(np.ones(k))
    transition = rng.dirichlet(np.ones(k), k)
    log_b = rng.normal(size=(3, steps, k)) * 3.0
    lengths = np.array([5, 3, 0])
    mask = (np.arange(steps)[None] < lengths[:, None]).astype("float32")
    gamma, xi_sum, log_lik = baselines.forward_backward(pi, transition, log_b, mask)
    want_xi = np.zeros((k, k))
    for b, length in enumerate(lengths):
        if length == 0:
            assert log_lik[b] == 0.0 and (gamma[b] == 0.0).all()
            continue
        total, want_gamma, xi = 0.0, np.zeros((length, k)), np.zeros((k, k))
        for path in itertools.product(range(k), repeat=int(length)):
            p = pi[path[0]] * np.exp(log_b[b, 0, path[0]])
            for t in range(1, length):
                p *= transition[path[t - 1], path[t]] * np.exp(log_b[b, t, path[t]])
            total += p
            want_gamma[np.arange(length), list(path)] += p
            for t in range(1, length):
                xi[path[t - 1], path[t]] += p
        want_xi += xi / total
        np.testing.assert_allclose(log_lik[b], np.log(total), rtol=0, atol=1e-10)
        np.testing.assert_allclose(gamma[b, :length], want_gamma / total, rtol=0, atol=1e-10)
        assert (gamma[b, length:] == 0.0).all()
    np.testing.assert_allclose(xi_sum, want_xi, rtol=0, atol=1e-10)


@pytest.mark.parametrize("obs_type", ["gaussian", "categorical"])
def test_the_true_parameters_score_the_true_likelihood(obs_type):
    split = _dataset(obs_type, n=20)
    hmm = split.true_hmm
    model = baselines.BaumWelchHMM(hmm.pi, hmm.transition)
    if obs_type == "gaussian":
        model.means, model.variances = hmm.means, np.ones_like(hmm.means)
    else:
        model.emission_probs = hmm.emission_probs
    np.testing.assert_allclose(
        baselines.log_likelihood(model, split.obs, split.mask),
        data.true_log_likelihood(hmm, split.obs, split.mask), rtol=0, atol=1e-9)


@pytest.mark.parametrize("obs_type,vocab", [("gaussian", None), ("categorical", 6)])
def test_em_never_lowers_the_training_likelihood(obs_type, vocab):
    split = _dataset(obs_type, n=100)
    history = baselines.fit_baum_welch(split.obs, split.mask, 3, vocab_size=vocab, max_iter=40, tol=-1.0).history
    assert len(history) == 40
    assert (np.diff(history) >= -1e-8).all(), np.diff(history).min()


@pytest.mark.parametrize("init", baselines.INITS)
def test_em_recovers_the_true_model(init):
    train, test = _dataset(seed=0), _dataset(seed=1, n=100)
    hmm = train.true_hmm
    fitted = baselines.fit_baum_welch(train.obs, train.mask, 3, init=init, seed=1)
    order = np.argsort(fitted.means[:, 0])
    np.testing.assert_allclose(fitted.means[order], hmm.means, rtol=0, atol=0.1)
    np.testing.assert_allclose(fitted.variances, 1.0, rtol=0, atol=0.1)
    np.testing.assert_allclose(fitted.transition[np.ix_(order, order)], hmm.transition, rtol=0, atol=0.03)
    nll = -baselines.log_likelihood(fitted, test.obs, test.mask).sum() / test.mask.sum()
    assert abs(nll - data.true_nll_per_step(hmm, test.obs, test.mask)) < 0.01


def test_a_categorical_fit_beats_chance_and_nears_the_true_nll():
    train, test = _dataset("categorical", n=500, seed=0), _dataset("categorical", n=100, seed=1)
    fitted = baselines.fit_baum_welch(train.obs, train.mask, 3, vocab_size=6, seed=2)
    nll = -baselines.log_likelihood(fitted, test.obs, test.mask).sum() / test.mask.sum()
    assert nll - data.true_nll_per_step(train.true_hmm, test.obs, test.mask) < 0.02
    assert nll < np.log(6)


def test_padded_observations_change_nothing_even_when_nan():
    split = _dataset(n=50)
    poisoned = np.where(split.mask[:, :, None] > 0, split.obs, np.nan)
    clean = baselines.fit_baum_welch(split.obs, split.mask, 3, max_iter=5)
    dirty = baselines.fit_baum_welch(poisoned, split.mask, 3, max_iter=5)
    assert clean.history == dirty.history
    np.testing.assert_array_equal(clean.means, dirty.means)


def test_a_bad_init_or_an_empty_mask_is_refused():
    split = _dataset(n=5)
    with pytest.raises(ValueError, match="init must be"):
        baselines.fit_baum_welch(split.obs, split.mask, 3, init="nope")
    with pytest.raises(ValueError, match="at least one valid step"):
        baselines.fit_baum_welch(split.obs, np.zeros_like(split.mask), 3)


def test_the_bio_sampler_obeys_its_constraints_and_the_rate_detects_a_break():
    _, labels = compare_baselines.sample_bio(500, 30, 0)
    assert compare_baselines.invalid_bio_rate(labels) == 0.0
    assert set(np.unique(labels)) == {0, 1, 2}
    assert (labels[:, 1:][labels[:, :-1] == 1] == 2).all()
    broken = np.zeros((4, 3), "int32")
    broken[1, 1] = 2
    assert compare_baselines.invalid_bio_rate(broken) == 0.25


def test_the_autoregressive_sampler_is_strongly_predicted_by_its_history():
    obs, states, log_emit = compare_baselines.sample_autoregressive(300, 40, 3, 0)
    assert np.corrcoef(obs[:, :-1, 0].ravel(), obs[:, 1:, 0].ravel())[0, 1] > 0.8
    picked = np.take_along_axis(log_emit, states[:, :, None], axis=-1)
    assert picked.mean() > log_emit.mean()


def test_the_comparison_runs_end_to_end_and_appends_every_row(tmp_path):
    output = tmp_path / "rows.jsonl"
    rows = compare_baselines.main([
        "--output", str(output), "--states", "2", "--seeds", "1", "--crf-seeds", "1", "--num-train", "16",
        "--num-val", "4", "--num-test", "4", "--seq-len", "6", "--hidden-dim", "8", "--epochs", "1",
        "--batch-size", "8",
    ])
    assert [json.loads(line) for line in output.read_text().splitlines()] == json.loads(json.dumps(rows))
    methods = {(r["experiment"], r["method"]) for r in rows}
    assert methods == {
        ("baum-welch", "neural-hmm"), ("baum-welch", "neural-hmm-kmeans"), ("baum-welch", "baum-welch-kmeans"),
        ("baum-welch", "baum-welch-random"),
        ("crf", "softmax"), ("crf", "crf"), ("crf", "crf-bio-mask"),
        ("autoregressive", "autoregressive-neural-hmm"), ("autoregressive", "neural-hmm"),
        ("autoregressive", "baum-welch-kmeans"),
    }
    assert all(r["invalid_path_rate"] == 0.0 for r in rows if r["method"] == "crf-bio-mask")
