"""``diagnostics.py``: compiled batching, the state diagnostic and the input checks.

Guards:

* :func:`decode` and :func:`score` equal the eager helpers, across batch boundaries, for
  every obs type, in both modes, with exogenous inputs and with end constraints;
* :func:`state_diagnostics` agrees with the posteriors by hand, ignores padding, and
  DISCRIMINATES: a trained model on sticky separated data has a large shuffled gap, and the
  same model with its emissions made identical across states has none;
* :func:`validate_inputs` names each bad input, ignores whatever padding holds, and passes
  clean data.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm.diagnostics import (
    decode, score, state_diagnostics, validate_inputs,
)

K, D, V, U = 3, 2, 5, 2
N, T = 7, 9
SMALL = {"hidden_dim": 8, "num_layers": 1}


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _batch(obs_type="gaussian", seed=0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(2, T + 1, N)
    mask = (np.arange(T)[None] < lengths[:, None]).astype("float32")
    if obs_type == "categorical":
        obs = rng.integers(0, V, (N, T)).astype("int32")
    elif obs_type == "poisson":
        obs = rng.poisson(2.0, (N, T, D)).astype("float32")
    else:
        obs = rng.normal(size=(N, T, D)).astype("float32")
    return {"obs": obs, "mask": mask}


def _shape(obs_type):
    return {"vocab_size": V} if obs_type == "categorical" else {"obs_dim": D}


CASES = {
    "gaussian": lambda: (NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D), _batch(), None),
    "poisson": lambda: (NeuralHMM(n_states=K, obs_type="poisson", obs_dim=D), _batch("poisson"), None),
    "categorical": lambda: (
        create_neural_hmm("context-transition-hmm", n_states=K, obs_type="categorical", vocab_size=V, **SMALL),
        _batch("categorical"), None),
    "exogenous": lambda: (
        create_neural_hmm("io-hmm", n_states=K, obs_type="gaussian", obs_dim=D, input_dim=U, **SMALL),
        {**_batch(), "inputs": np.random.default_rng(1).normal(size=(N, T, U)).astype("float32")}, None),
    "ended": lambda: (
        NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, allowed_ends=[False, True, True]), _batch(), None),
    "crf": lambda: (
        create_neural_hmm("neural-crf-tagger", n_states=K, obs_type="gaussian", obs_dim=D, encoder_type="gru", **SMALL),
        _batch(), np.random.default_rng(2).integers(0, K, (N, T)).astype("int32")),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("batch_size", [3, 32])
def test_the_compiled_helpers_equal_the_eager_ones(name, batch_size):
    keras.utils.set_random_seed(5)
    model, batch, states = CASES[name]()
    path, best = decode(model, batch, batch_size=batch_size)
    want_path, want_best = model.viterbi(batch)
    np.testing.assert_array_equal(path, _np(want_path))
    np.testing.assert_allclose(best, _np(want_best), rtol=0, atol=2e-3)
    np.testing.assert_allclose(
        score(model, batch, states, batch_size=batch_size), _np(model.log_likelihood(batch, states)),
        rtol=0, atol=2e-3)
    assert len(model.weights) > 0


def test_score_refuses_labels_on_a_generative_model():
    with pytest.raises(ValueError):
        score(NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D), _batch(), np.zeros((N, T), "int32"))


def test_the_occupancy_and_entropy_are_the_posteriors_by_hand():
    keras.utils.set_random_seed(0)
    model = NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D)
    batch = _batch()
    report = state_diagnostics(model, batch, batch_size=3)
    gamma = _np(model.posteriors(batch)).astype("float64")
    steps = batch["mask"].sum()
    occupancy = gamma.sum(axis=(0, 1)) / steps
    np.testing.assert_allclose(report["occupancy"], occupancy, rtol=0, atol=1e-5)
    np.testing.assert_allclose(report["occupancy"].sum(), 1.0, rtol=0, atol=1e-5)
    assert report["occupancy_entropy"] == pytest.approx(-(occupancy * np.log(occupancy)).sum(), abs=1e-5)
    terms = np.where(gamma > 0, gamma * np.log(np.where(gamma > 0, gamma, 1.0)), 0.0)
    assert report["mean_posterior_entropy"] == pytest.approx(-terms.sum() / steps, abs=1e-4)


def test_the_diagnostic_ignores_padded_content():
    keras.utils.set_random_seed(0)
    model = NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D)
    batch = _batch()
    poisoned = dict(batch, obs=np.where(batch["mask"][:, :, None] > 0, batch["obs"], np.nan))
    clean, dirty = state_diagnostics(model, batch), state_diagnostics(model, poisoned)
    np.testing.assert_array_equal(clean["occupancy"], dirty["occupancy"])
    assert clean["state_shuffled_gap"] == dirty["state_shuffled_gap"]


def test_the_diagnostic_needs_a_valid_step():
    model = NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D)
    batch = _batch()
    with pytest.raises(ValueError, match="at least one valid step"):
        state_diagnostics(model, dict(batch, mask=np.zeros_like(batch["mask"])))


def test_a_discriminative_model_has_no_gap():
    model = create_neural_hmm(
        "neural-crf-tagger", n_states=K, obs_type="gaussian", obs_dim=D, encoder_type="gru", **SMALL)
    report = state_diagnostics(model, _batch())
    assert report["state_shuffled_gap"] is None and report["occupancy"].shape == (K,)


def _sticky_data(n=200, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, steps), "int64")
    z[:, 0] = rng.integers(0, 2, n)
    for t in range(1, steps):
        z[:, t] = np.where(rng.random(n) < 0.95, z[:, t - 1], 1 - z[:, t - 1])
    means = np.array([[-3.0, 0.0], [3.0, 0.0]])
    obs = (means[z] + rng.normal(size=(n, steps, 2))).astype("float32")
    return {"obs": obs, "mask": np.ones((n, steps), "float32")}


def test_the_diagnostic_separates_a_healthy_model_from_a_collapsed_one():
    keras.utils.set_random_seed(0)
    data = _sticky_data()
    model = NeuralHMM(n_states=2, obs_type="gaussian", obs_dim=2, hidden_dim=16, state_emb_dim=8)
    model.compile(optimizer=keras.optimizers.Adam(1e-2))
    model.fit(data, None, batch_size=64, epochs=25, verbose=0)
    healthy = state_diagnostics(model, data)
    assert healthy["state_shuffled_gap"] > 0.3, healthy
    assert healthy["mean_posterior_entropy"] < 0.1, healthy
    assert healthy["occupancy"].min() > 0.3, healthy

    embedding = model.emission.state_embedding
    table = _np(embedding)
    embedding.assign(np.broadcast_to(table[:1], table.shape))
    collapsed = state_diagnostics(model, data)
    assert abs(collapsed["state_shuffled_gap"]) < 1e-3, collapsed
    assert collapsed["mean_posterior_entropy"] > 0.5, collapsed


@pytest.mark.parametrize("obs_type,field,index,value,match", [
    ("categorical", "obs", (0, 0), V, "symbols outside"),
    ("categorical", "obs", (1, 1), -1, "symbols outside"),
    ("gaussian", "obs", (0, 1, 0), np.nan, "non-finite observations"),
    ("gaussian", "obs", (0, 1, 1), np.inf, "non-finite observations"),
    ("poisson", "obs", (2, 1, 0), -1.0, "negative counts"),
    ("poisson", "obs", (2, 1, 0), np.nan, "non-finite observations"),
])
def test_a_bad_value_at_a_valid_step_is_named(obs_type, field, index, value, match):
    model = NeuralHMM(n_states=K, obs_type=obs_type, **_shape(obs_type))
    batch = _batch(obs_type)
    validate_inputs(model, batch)
    batch[field] = batch[field].copy()
    batch[field][index] = value
    with pytest.raises(ValueError, match=match):
        validate_inputs(model, batch)


def test_padded_content_may_hold_anything():
    model = NeuralHMM(n_states=K, obs_type="categorical", vocab_size=V)
    batch = _batch("categorical")
    batch["obs"] = np.where(batch["mask"] > 0, batch["obs"], -1)
    labels = np.where(batch["mask"] > 0, 0, -1)
    validate_inputs(model, batch, labels)
    gaussian = _batch()
    gaussian["obs"] = np.where(gaussian["mask"][:, :, None] > 0, gaussian["obs"], np.nan)
    validate_inputs(NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D), gaussian)


def test_a_non_prefix_mask_a_wrong_mask_shape_and_a_bad_label_are_named():
    model = NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D)
    batch = _batch()
    holed = batch["mask"].copy()
    holed[3] = 0.0
    holed[3, 1] = 1.0
    with pytest.raises(ValueError, match=r"not a valid prefix in 1 sequences \(first: 3\)"):
        validate_inputs(model, dict(batch, mask=holed))
    with pytest.raises(ValueError, match="mask has shape"):
        validate_inputs(model, dict(batch, mask=batch["mask"][:, :-1]))
    labels = np.zeros((N, T), "int32")
    labels[0, 0] = K
    with pytest.raises(ValueError, match="labels outside"):
        validate_inputs(model, batch, labels)


def test_a_non_finite_exogenous_input_is_named():
    model = create_neural_hmm("io-hmm", n_states=K, obs_type="gaussian", obs_dim=D, input_dim=U, **SMALL)
    batch = {**_batch(), "inputs": np.zeros((N, T, U), "float32")}
    validate_inputs(model, batch)
    batch["inputs"][0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite exogenous inputs"):
        validate_inputs(model, batch)


def test_every_problem_is_reported_at_once():
    model = NeuralHMM(n_states=K, obs_type="poisson", obs_dim=D)
    batch = _batch("poisson")
    batch["obs"] = batch["obs"].copy()
    batch["obs"][0, 0, 0] = -1.0
    labels = np.full((N, T), K, "int32")
    with pytest.raises(ValueError, match="negative counts.*labels outside"):
        validate_inputs(model, batch, labels)
