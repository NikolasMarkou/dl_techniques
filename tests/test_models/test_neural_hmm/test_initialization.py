"""``initialization.py``: k-means emissions.

Guards: each state's emission starts at one k-means cluster with its spread, the first
log-likelihood on data far from the origin is then near the true model's where the default
start is far below it, padding is never read, a seed reproduces, and a model the fit does
not apply to is refused rather than half-initialized.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm.initialization import initialize_emissions_from_kmeans

K, D = 3, 2
MEANS = np.array([[96.0, 96.0], [100.0, 100.0], [104.0, 104.0]])


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _data(n=100, steps=30, seed=0):
    rng = np.random.default_rng(seed)
    z = np.zeros((n, steps), "int64")
    z[:, 0] = rng.integers(0, K, n)
    for t in range(1, steps):
        z[:, t] = np.where(rng.random(n) < 0.9, z[:, t - 1], rng.integers(0, K, n))
    obs = MEANS[z] + 0.5 * rng.normal(size=(n, steps, D))
    return {"obs": obs.astype("float32"), "mask": np.ones((n, steps), "float32")}, z


def _model(**kwargs):
    keras.utils.set_random_seed(0)
    return NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, hidden_dim=16, **kwargs)


def _head_output(model):
    return _np(model.emission.head(keras.ops.convert_to_tensor(model.emission.state_embedding)))


def test_each_state_starts_at_a_cluster_with_its_spread():
    model = _model()
    data, _ = _data()
    initialize_emissions_from_kmeans(model, data)
    params = _head_output(model)
    means, raw = params[:, :D], params[:, D:]
    np.testing.assert_allclose(means[np.argsort(means[:, 0])], MEANS, rtol=0, atol=0.1)
    scale = _np(keras.ops.softplus(raw)) + model.emission.min_scale
    np.testing.assert_allclose(scale, 0.5, rtol=0, atol=0.05)


def test_the_first_likelihood_far_from_the_origin_is_good_only_with_the_fit():
    data, _ = _data()
    default = _model()
    fitted = _model()
    initialize_emissions_from_kmeans(fitted, data)
    steps = data["mask"].sum()
    before = float(_np(default.log_likelihood(data)).sum() / steps)
    after = float(_np(fitted.log_likelihood(data)).sum() / steps)
    assert after > -3.0, after
    assert before < -100.0, before


def test_training_continues_from_the_fit():
    data, _ = _data()
    model = _model()
    initialize_emissions_from_kmeans(model, data)
    start = float(_np(model.log_likelihood(data)).mean())
    model.compile(optimizer=keras.optimizers.Adam(1e-3))
    model.fit(data, None, batch_size=32, epochs=5, verbose=0)
    assert float(_np(model.log_likelihood(data)).mean()) >= start - 1.0


def test_padding_is_never_read_and_a_seed_reproduces():
    data, _ = _data()
    data["mask"][:, 20:] = 0.0
    first, second = _model(), _model()
    initialize_emissions_from_kmeans(first, data, seed=3)
    poisoned = dict(data, obs=np.where(data["mask"][:, :, None] > 0, data["obs"], np.nan))
    initialize_emissions_from_kmeans(second, poisoned, seed=3)
    # lstsq is reproducible to about one float32 ulp at 100, not bit for bit.
    np.testing.assert_allclose(_head_output(first), _head_output(second), rtol=0, atol=1e-4)


@pytest.mark.parametrize("build", [
    lambda: NeuralHMM(n_states=K, obs_type="categorical", vocab_size=5),
    lambda: NeuralHMM(n_states=K, obs_type="poisson", obs_dim=D),
    lambda: create_neural_hmm("neural-crf-tagger", n_states=K, obs_type="gaussian", obs_dim=D),
    lambda: create_neural_hmm("autoregressive-neural-hmm", n_states=K, obs_type="gaussian", obs_dim=D),
], ids=["categorical", "poisson", "discriminative", "autoregressive"])
def test_a_model_the_fit_does_not_apply_to_is_refused(build):
    data, _ = _data(n=2, steps=3)
    with pytest.raises(ValueError, match="initialize_emissions_from_kmeans needs"):
        initialize_emissions_from_kmeans(build(), data)


def test_invalid_inputs_are_refused_before_any_fit():
    data, _ = _data(n=2, steps=3)
    data["obs"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite observations"):
        initialize_emissions_from_kmeans(_model(), data)
