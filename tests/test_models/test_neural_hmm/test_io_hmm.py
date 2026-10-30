"""``io-hmm``: a true input-output HMM, ``p(x | u)``, with an exogenous input.

Guards:

* the preset refuses to build without ``input_dim``, and the model refuses a mismatched or
  missing ``inputs`` entry, and an ``inputs`` entry nothing would read;
* ``sum_x p(x | u) = 1`` by exact enumeration for a fixed ``u``, for every encoder type: the
  input may reach the potentials at its own step without breaking the likelihood;
* the transition into step ``t`` reads ``u_{<=t}``, in particular ``u_t`` (D-002: no shift),
  never a later input, and never the observations;
* padded inputs change nothing, also when they hold NaN;
* the config round-trips, a saved archive restores the outputs, and one optimizer step
  moves the weights it should; ``input_dim`` is a live structural knob.
"""

import itertools

import keras
import numpy as np
import pytest

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm.encoders import ENCODER_TYPES

from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights

K, D, U, V, STEPS = 3, 2, 4, 3, 6
SMALL = {"hidden_dim": 8, "num_layers": 1}


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _model(encoder_type="conv", obs_type="gaussian", seed=1, **overrides):
    keras.utils.set_random_seed(seed)
    shape = {"vocab_size": V} if obs_type == "categorical" else {"obs_dim": D}
    config = {"n_states": K, "obs_type": obs_type, "input_dim": U, "encoder_type": encoder_type, **shape, **SMALL}
    if encoder_type == "transformer":
        config.update(num_heads=2, intermediate_size=12, max_seq_len=16)
    elif encoder_type == "mlp":
        config.update(intermediate_size=12)
    return create_neural_hmm("io-hmm", **{**config, **overrides})


def _batch(batch=2, steps=STEPS, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.normal(size=(batch, steps, D)).astype("float32"),
        "inputs": rng.normal(size=(batch, steps, U)).astype("float32"),
        "mask": np.ones((batch, steps), "float32"),
    }


def _randomize(model, seed=3):
    """Redraw every weight: the transition's projection starts at zero and reads nothing."""
    rng = np.random.default_rng(seed)
    for weight in model.trainable_weights:
        weight.assign(rng.normal(0.0, 0.7, size=weight.shape).astype(weight.dtype))
    return model


def test_the_preset_refuses_to_build_without_an_input_dim():
    with pytest.raises(ValueError, match="context-transition-hmm"):
        create_neural_hmm("io-hmm", n_states=K, obs_type="gaussian", obs_dim=D)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_input_dim_is_refused(bad):
    with pytest.raises(ValueError, match="input_dim must be positive"):
        _model(input_dim=bad)


def test_an_input_dim_nothing_would_read_is_refused():
    with pytest.raises(ValueError, match="input_dim=4"):
        NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, input_dim=U)
    with pytest.raises(ValueError, match="input_dim=4"):
        NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, input_dim=U, mode="discriminative")


def test_a_missing_or_mismatched_inputs_entry_is_refused():
    with pytest.raises(ValueError, match="no such entry"):
        _model().build({"obs": (None, None, D), "mask": (None, None)})
    with pytest.raises(ValueError, match="input_dim=4"):
        _model().build({"obs": (None, None, D), "inputs": (None, None, U + 1), "mask": (None, None)})


def test_an_inputs_entry_nothing_would_read_is_refused():
    model = create_neural_hmm("context-transition-hmm", n_states=K, obs_type="gaussian", obs_dim=D, **SMALL)
    with pytest.raises(ValueError, match="nothing would read it"):
        model.build({"obs": (None, None, D), "inputs": (None, None, U), "mask": (None, None)})


@pytest.mark.parametrize("emission_context", ["none", "autoregressive"])
@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_the_model_is_a_normalized_distribution_given_the_input(encoder_type, emission_context):
    """Conditioning on u_t is legal, so x_t's distribution still sums to 1 for every u."""
    model = _model(encoder_type, "categorical", emission_context=emission_context)
    length = 5
    symbols = np.array(list(itertools.product(range(V), repeat=length)), dtype="int32")
    u = np.random.default_rng(2).normal(size=(1, length, U)).astype("float32")
    inputs = {"obs": symbols, "inputs": np.repeat(u, len(symbols), axis=0), "mask": np.ones(symbols.shape, "float32")}
    model(inputs, training=False)
    _randomize(model)
    log_p = _np(model.log_likelihood(inputs)).astype("float64")
    assert abs(np.exp(log_p).sum() - 1.0) <= 1e-5


@pytest.mark.parametrize("encoder_type", ["conv", "lstm", "gru", "transformer", "mlp"])
def test_the_transition_reads_the_input_at_its_own_step_and_never_a_later_one(encoder_type):
    model = _model(encoder_type)
    batch = _batch()
    model(batch, training=False)
    _randomize(model)
    clean = _np(model(batch, training=False)["log_A"])
    for s in range(1, STEPS):
        moved = dict(batch, inputs=batch["inputs"].copy())
        moved["inputs"][:, s] += 3.0
        out = _np(model(moved, training=False)["log_A"])
        assert np.array_equal(out[:, :s], clean[:, :s]), f"log_A before step {s} read a later input"
        assert np.abs(out[:, s] - clean[:, s]).max() > 1e-4, f"log_A[:, {s}] did not read u_{s}"


def test_the_transition_never_reads_the_observations():
    model = _model()
    batch = _batch()
    model(batch, training=False)
    _randomize(model)
    moved = dict(batch, obs=batch["obs"] + 5.0)
    assert np.array_equal(_np(model(moved, training=False)["log_A"]), _np(model(batch, training=False)["log_A"]))


def test_a_categorical_io_hmm_builds_no_symbol_embedding():
    """The encoder reads u, so a symbol embedding would be a dead component (D-005)."""
    model = _model(obs_type="categorical")
    assert model.obs_embedding is None


@pytest.mark.parametrize("encoder_type", ["conv", "lstm", "transformer"])
def test_padded_inputs_change_nothing_even_when_nan(encoder_type):
    model = _model(encoder_type)
    batch = _batch()
    batch["mask"][:, 4:] = 0.0
    model(batch, training=False)
    _randomize(model)
    reference = _np(model.log_likelihood(batch))
    poisoned = dict(batch, inputs=batch["inputs"].copy())
    poisoned["inputs"][:, 4:] = np.nan
    np.testing.assert_array_equal(_np(model.log_likelihood(poisoned)), reference)
    truncated = {key: value[:, :4] for key, value in batch.items()}
    np.testing.assert_allclose(_np(model.log_likelihood(truncated)), reference, rtol=0, atol=1e-4)


def test_the_archive_restores_the_outputs(tmp_path):
    model = _model()
    batch = _batch()
    model(batch, training=False)
    _randomize(model)
    path = tmp_path / "io_hmm.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert restored.input_dim == U
    np.testing.assert_array_equal(_np(restored.log_likelihood(batch)), _np(model.log_likelihood(batch)))


def test_training_moves_the_encoder_and_the_transition():
    model = _model()
    batch = _batch(batch=8)
    model.compile(optimizer=keras.optimizers.Adam(1e-2))
    model(batch, training=False)
    before = {w.path: _np(w).copy() for w in model.trainable_weights}
    model.fit(batch, None, batch_size=8, epochs=2, verbose=0)
    frozen = [w.path for w in model.trainable_weights if np.array_equal(_np(w), before[w.path])]
    assert not frozen, f"weights that never moved: {frozen}"


def test_input_dim_is_a_live_structural_knob():
    def builder(value):
        def build():
            model = _model(input_dim=value)
            model.build({"obs": (None, None, D), "inputs": (None, None, value), "mask": (None, None)})
            return model
        return build
    assert_structural_knob_changes_weights({value: builder(value) for value in (2, 5)}, knob="input_dim")
