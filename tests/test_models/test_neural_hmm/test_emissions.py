"""``neural_hmm.emissions``: ``GenerativeEmission`` and ``DiscriminativeEmission``.

Guards (SC3):

* no weights exist before ``build()``; ``compute_output_shape`` answers on an unbuilt layer
  and equals the real output shape, with and without a context;
* the log-densities equal scipy on the head's own parameters: ``norm.logpdf`` (gaussian)
  and ``poisson.logpmf`` including ``x = 0`` (poisson, so the ``-log x!`` term is present),
  summed over features, in float64 to 1e-9; float32 within a derived rounding bound;
  categorical rows normalize over the vocabulary and equal a numpy ``log_softmax``; a
  symbol outside the vocabulary scores exactly 0 in every state, an uninformative step
  (D-027: the model adds its penalty outside the forward pass);
* ``min_scale`` changes the output (gaussian, poisson);
* the context path is live, with a rank-4 ``(B, T, K, E + H)`` head input (A3);
* under ``mixed_float16`` an observation 1e3 away from the mean at ``scale = min_scale``
  stays finite and float32 (the float16 square would overflow), and the symbolic output
  dtype is the float32 ``call()`` returns;
* the discriminative potentials equal ``h @ W + b``;
* gradients reach every weight;
* ``get_config``/``from_config`` + ``set_weights`` and ``.keras`` round trips are exact;
  registration keys are literal; bad arguments raise ``ValueError`` naming the value.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from scipy import stats

from dl_techniques.models.time_series.neural_hmm.emissions import (
    DiscriminativeEmission,
    GenerativeEmission,
)
from dl_techniques.utils.dtype_policy import mask_sentinel
from tests.numerics import reassociation_atol

from ..precision_arm_oracle import precision_policy

K, D, V, H = 3, 2, 7, 4
B, T = 2, 6

#: Float64 tolerance: SC3 asks for 1e-5; the measured float64 error is ~1e-13.
ATOL_F64 = 1e-9

#: Constructor arguments per observation type.
OBS_KWARGS = {"gaussian": {"obs_dim": D}, "poisson": {"obs_dim": D}, "categorical": {"vocab_size": V}}


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _data_dtype(policy):
    """Observation and context dtype under a policy: float64 for float64, else float32."""
    return "float64" if policy == "float64" else "float32"


def _obs(obs_type, dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    if obs_type == "categorical":
        return rng.integers(0, V, size=(B, T)).astype("int32")
    if obs_type == "poisson":
        counts = rng.poisson(3.0, size=(B, T, D))
        counts[0, 0] = 0  # x = 0 must be scored too
        return counts.astype(dtype)
    return (3.0 * rng.normal(size=(B, T, D))).astype(dtype)


def _context(dtype="float32", seed=1):
    return np.random.default_rng(seed).normal(size=(B, T, H)).astype(dtype)


def _emission(obs_type, context_dim=0, policy="float32", **kwargs):
    """Build under the global ``policy``: the factory-built head inherits only the global one."""
    dtype = _data_dtype(policy)
    with precision_policy(policy):
        layer = GenerativeEmission(K, obs_type, context_dim=context_dim, **OBS_KWARGS[obs_type], **kwargs)
        layer(*((_obs(obs_type, dtype), _context(dtype)) if context_dim else (_obs(obs_type, dtype),)))
    assert layer.head.compute_dtype == layer.compute_dtype
    return layer


def _call(layer, obs_type, dtype="float32"):
    if layer.context_dim:
        return _np(layer(_obs(obs_type, dtype), _context(dtype)))
    return _np(layer(_obs(obs_type, dtype)))


def _head_params(layer, dtype):
    """The head's parameters, per step and state: ``(B, T, K, P)`` (or ``(1, 1, K, P)``)."""
    embedding = _np(layer.state_embedding)
    if not layer.context_dim:
        return _np(layer.head(embedding)).astype("float64")[None, None]
    context = _context(dtype)
    features = np.concatenate(
        [np.broadcast_to(embedding, (B, T, K, embedding.shape[-1])),
         np.broadcast_to(context[:, :, None, :], (B, T, K, H))], axis=-1,
    )
    return _np(layer.head(features.astype(dtype))).astype("float64")


def _softplus(x):
    return np.logaddexp(0.0, x)


def _scipy_reference(layer, obs_type, dtype):
    params = _head_params(layer, dtype)
    obs = _obs(obs_type, dtype).astype("float64")
    if obs_type == "categorical":
        log_probs = params - params.max(-1, keepdims=True)
        log_probs -= np.log(np.exp(log_probs).sum(-1, keepdims=True))
        log_probs = np.broadcast_to(log_probs, (B, T, K, V))
        return np.take_along_axis(log_probs, obs.astype(int)[:, :, None, None].repeat(K, 2), axis=-1)[..., 0]
    x = obs[:, :, None, :]
    if obs_type == "gaussian":
        mean, raw = params[..., :D], params[..., D:]
        return stats.norm.logpdf(x, mean, _softplus(raw) + layer.min_scale).sum(-1)
    return stats.poisson.logpmf(x, _softplus(params) + layer.min_scale).sum(-1)


# ---------------------------------------------------------------------
# Build contract and shapes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_no_generative_weights_exist_before_build(obs_type):
    layer = GenerativeEmission(K, obs_type, context_dim=H, **OBS_KWARGS[obs_type])
    assert not layer.built and layer.weights == []


def test_no_discriminative_weights_exist_before_build():
    layer = DiscriminativeEmission(K)
    assert not layer.built and layer.weights == []


@pytest.mark.parametrize("context_dim", [0, H])
@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_compute_output_shape_answers_unbuilt_and_matches_the_forward(obs_type, context_dim):
    layer = GenerativeEmission(K, obs_type, context_dim=context_dim, **OBS_KWARGS[obs_type])
    obs = _obs(obs_type)
    declared = layer.compute_output_shape(obs.shape, (B, T, H) if context_dim else None)
    assert not layer.built
    out = layer(obs, _context()) if context_dim else layer(obs)
    assert declared == (B, T, K) == tuple(out.shape)

    disc = DiscriminativeEmission(K)
    assert disc.compute_output_shape((B, T, H)) == (B, T, K) == tuple(disc(_context()).shape)


# ---------------------------------------------------------------------
# Values against scipy / numpy
# ---------------------------------------------------------------------


@pytest.mark.parametrize("context_dim", [0, H])
@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_float64_log_densities_match_scipy(obs_type, context_dim):
    layer = _emission(obs_type, context_dim, policy="float64")
    got = _call(layer, obs_type, "float64")
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, _scipy_reference(layer, obs_type, "float64"), rtol=0, atol=ATOL_F64)


@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_float32_log_densities_match_scipy_within_the_rounding_bound(obs_type):
    """The reference runs in float64 on the float32 head's parameters, so only the float32
    density arithmetic is measured. Per feature that is at most the 9-term Lanczos series
    of ``log_gamma`` (two roundings per term) plus about 8 elementwise operations, summed
    over ``D`` features: ``2 * 9 + 8 = 26`` roundings per feature."""
    layer = _emission(obs_type)
    got = _call(layer, obs_type)
    expected = _scipy_reference(layer, obs_type, "float32")
    assert got.dtype == np.float32
    atol = reassociation_atol([26 * D], 1, np.abs(expected).max())
    np.testing.assert_allclose(got, expected, rtol=0, atol=atol)


def test_categorical_rows_normalize_over_the_vocabulary():
    layer = _emission("categorical", policy="float64")
    every_symbol = np.arange(V, dtype="int32")[None]  # one sequence that visits each symbol
    log_probs = _np(layer(every_symbol))[0]  # (V, K)
    total = np.log(np.exp(log_probs).sum(axis=0))
    np.testing.assert_allclose(total, 0.0, rtol=0, atol=ATOL_F64)


def test_a_categorical_symbol_outside_the_vocabulary_is_uninformative():
    """D-027: every out-of-range symbol, ``-1`` included, scores exactly 0 in EVERY state.

    The same value in every state is an uninformative step, so the forward recursion,
    Viterbi and the posteriors treat that step as a missing observation. The emission must
    NOT carry ``mask_sentinel``: -1e9 in every state of one step absorbed every float32
    forward/backward value of the row (D-022's first fix). The model makes such a symbol
    loud by a penalty outside the forward pass; test_model.py pins that.
    """
    layer = _emission("categorical")
    out = _np(layer(np.array([[0, -1, V, V + 5, -7, V - 1]], "int32")))
    assert np.all(out[0, 1:5] == 0.0), out[0]
    in_range = out[0, [0, 5]]
    assert np.all(in_range < 0.0) and np.all(in_range > -50.0), in_range
    assert not np.any(out <= 0.5 * mask_sentinel("float32")), out[0]


def test_the_discriminative_potentials_are_an_affine_map_of_the_context():
    context = _context("float64")
    with precision_policy("float64"):
        layer = DiscriminativeEmission(K)
        out = _np(layer(context))
    kernel, bias = layer.get_weights()
    np.testing.assert_allclose(out, context @ kernel + bias, rtol=0, atol=ATOL_F64)


# ---------------------------------------------------------------------
# Knobs and liveness
# ---------------------------------------------------------------------


@pytest.mark.parametrize("obs_type", ["gaussian", "poisson"])
def test_min_scale_changes_the_log_density(obs_type):
    small = _emission(obs_type, min_scale=1e-3)
    large = _emission(obs_type, min_scale=0.5)
    large.set_weights(small.get_weights())
    assert np.abs(_call(small, obs_type) - _call(large, obs_type)).max() > 1e-2


@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_the_context_changes_the_log_density_per_step(obs_type):
    layer = _emission(obs_type, context_dim=H)
    obs = _obs(obs_type)
    context = _context()
    moved = context.copy()
    moved[:, 2] += 1.0
    base, after = _np(layer(obs, context)), _np(layer(obs, moved))
    assert np.abs(after[:, 2] - base[:, 2]).max() > 1e-3
    np.testing.assert_array_equal(np.delete(after, 2, axis=1), np.delete(base, 2, axis=1))


def test_a_missing_context_is_refused_when_context_dim_is_set():
    layer = _emission("gaussian", context_dim=H)
    with pytest.raises(ValueError, match="context"):
        layer(_obs("gaussian"))


# ---------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------


def _pin_head_output(layer, params):
    """Zero every head weight, then set the output bias, so the head emits ``params``."""
    weights = [np.zeros_like(w) for w in layer.head.get_weights()]
    assert weights[-1].shape == params.shape, [w.shape for w in weights]
    weights[-1] = params
    layer.head.set_weights(weights)


@pytest.mark.parametrize("context_dim", [0, H])
def test_a_far_observation_at_the_minimum_scale_stays_finite_under_mixed_float16(context_dim):
    """``|x - mu| / sd = 1e3 / 1e-3 = 1e6``: its square overflows float16 (max 65504)."""
    layer = _emission("gaussian", context_dim, policy="mixed_float16", min_scale=1e-3)
    assert layer.head.compute_dtype == "float16"
    _pin_head_output(layer, np.array([0.0] * D + [-30.0] * D, "float32"))  # mean 0, sd -> min_scale
    obs = np.full((B, T, D), 1e3, "float32")
    out = layer(obs, _context()) if context_dim else layer(obs)
    values = _np(out)
    assert np.all(np.isfinite(values)), values  # checked before the dtype: this is the overflow
    assert out.dtype == "float32"
    out = values
    expected = D * (-0.5 * (1e3 / 1e-3) ** 2 - np.log(1e-3) - 0.5 * np.log(2 * np.pi))
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_the_symbolic_output_dtype_is_the_float32_that_call_returns_under_mixed_float16(obs_type):
    with precision_policy("mixed_float16"):
        layer = GenerativeEmission(K, obs_type, context_dim=H, **OBS_KWARGS[obs_type])
        obs_input = keras.Input((None,), dtype="int32") if obs_type == "categorical" else keras.Input((None, D))
        symbolic = layer(obs_input, keras.Input((None, H)))
        eager = layer(_obs(obs_type), _context())
        disc = DiscriminativeEmission(K)
        disc_symbolic, disc_eager = disc(keras.Input((None, H))), disc(_context())
    assert layer.head.compute_dtype == "float16" and disc.state_scores.compute_dtype == "float16"
    assert symbolic.dtype == "float32" and symbolic.shape == (None, None, K)
    assert eager.dtype == "float32"
    assert disc_symbolic.dtype == "float32" and disc_eager.dtype == "float32"


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------


def _assert_every_weight_gets_a_gradient(layer, call):
    weights = np.random.default_rng(5).normal(size=(B, T, K)).astype("float32")
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.cast(call(), tf.float32) * weights)
    grads = tape.gradient(loss, layer.trainable_weights)
    assert layer.trainable_weights
    for weight, grad in zip(layer.trainable_weights, grads):
        assert grad is not None, weight.path
        assert np.abs(_np(grad)).max() > 0.0, weight.path


@pytest.mark.parametrize("context_dim", [0, H])
@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_gradients_reach_every_generative_weight(obs_type, context_dim):
    layer = _emission(obs_type, context_dim)
    names = {w.path.split("/")[-1] for w in layer.trainable_weights}
    assert "state_embedding" in names and len(layer.trainable_weights) >= 3
    obs, context = tf.constant(_obs(obs_type)), tf.constant(_context())
    _assert_every_weight_gets_a_gradient(
        layer, lambda: layer(obs, context, training=True) if context_dim else layer(obs, training=True)
    )


def test_gradients_reach_every_discriminative_weight():
    layer = DiscriminativeEmission(K)
    context = tf.constant(_context())
    layer(context)
    _assert_every_weight_gets_a_gradient(layer, lambda: layer(context, training=True))


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


@pytest.mark.parametrize("context_dim", [0, H])
@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_get_config_from_config_and_set_weights_reproduce_the_output_exactly(obs_type, context_dim):
    layer = _emission(obs_type, context_dim, state_emb_dim=5, hidden_dim=9, min_scale=0.01)
    config = layer.get_config()
    expected = {
        "n_states": K, "obs_type": obs_type, "obs_dim": OBS_KWARGS[obs_type].get("obs_dim"),
        "vocab_size": OBS_KWARGS[obs_type].get("vocab_size"), "state_emb_dim": 5, "hidden_dim": 9,
        "context_dim": context_dim, "min_scale": 0.01,
    }
    assert {k: config[k] for k in expected} == expected
    clone = GenerativeEmission.from_config(config)
    args = (_obs(obs_type), _context()) if context_dim else (_obs(obs_type),)
    clone(*args)
    clone.set_weights(layer.get_weights())
    np.testing.assert_array_equal(_np(clone(*args)), _np(layer(*args)))


def test_the_discriminative_config_round_trip_is_exact():
    layer = DiscriminativeEmission(K)
    layer(_context())
    assert layer.get_config()["n_states"] == K
    clone = DiscriminativeEmission.from_config(layer.get_config())
    clone(_context())
    clone.set_weights(layer.get_weights())
    np.testing.assert_array_equal(_np(clone(_context())), _np(layer(_context())))


def _roundtrip(model, layer_name, inputs, tmp_path):
    path = tmp_path / "emission.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_array_equal(loaded.predict(inputs, verbose=0), model.predict(inputs, verbose=0))
    for got, want in zip(loaded.get_layer(layer_name).get_weights(), model.get_layer(layer_name).get_weights()):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("obs_type", list(OBS_KWARGS))
def test_a_keras_archive_round_trip_reproduces_the_generative_output_and_weights(obs_type, tmp_path):
    obs_input = keras.Input((T,), dtype="int32") if obs_type == "categorical" else keras.Input((T, D))
    context_input = keras.Input((T, H))
    layer = GenerativeEmission(K, obs_type, context_dim=H, name="emission", **OBS_KWARGS[obs_type])
    model = keras.Model([obs_input, context_input], layer(obs_input, context_input))
    _roundtrip(model, "emission", [_obs(obs_type), _context()], tmp_path)


def test_a_keras_archive_round_trip_reproduces_the_discriminative_output_and_weights(tmp_path):
    context_input = keras.Input((T, H))
    model = keras.Model(context_input, DiscriminativeEmission(K, name="emission")(context_input))
    _roundtrip(model, "emission", _context(), tmp_path)


def test_the_registration_keys_are_literal():
    assert keras.saving.get_registered_name(GenerativeEmission) == (
        "dl_techniques.models.neural_hmm.emissions>GenerativeEmission"
    )
    assert keras.saving.get_registered_name(DiscriminativeEmission) == (
        "dl_techniques.models.neural_hmm.emissions>DiscriminativeEmission"
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"obs_type": "beta", "obs_dim": D}, "beta"),
        ({"obs_type": "gaussian"}, "obs_dim"),
        ({"obs_type": "poisson", "obs_dim": D, "vocab_size": V}, "vocab_size"),
        ({"obs_type": "categorical"}, "vocab_size"),
        ({"obs_type": "categorical", "vocab_size": V, "obs_dim": D}, "obs_dim"),
        ({"obs_type": "gaussian", "obs_dim": D, "min_scale": 0.0}, "min_scale"),
        ({"obs_type": "gaussian", "obs_dim": D, "context_dim": -1}, "context_dim"),
        ({"obs_type": "gaussian", "obs_dim": D, "state_emb_dim": 0}, "state_emb_dim"),
    ],
)
def test_bad_generative_arguments_raise_naming_the_value(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GenerativeEmission(K, **kwargs)


def test_observations_of_the_wrong_width_are_refused_at_build():
    with pytest.raises(ValueError, match="features"):
        GenerativeEmission(K, "gaussian", obs_dim=D)(np.zeros((B, T, D + 1), "float32"))
