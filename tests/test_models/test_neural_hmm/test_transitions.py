"""``neural_hmm.transitions``: ``StaticTransition`` and ``InputDependentTransition``.

Guards (SC3):

* no weights exist before ``build()``; ``compute_output_shape`` answers on an unbuilt layer
  and equals the real output shapes;
* ``log_A`` equals an independent numpy log-softmax over the topology's allowed entries;
  every row of ``exp(log_A)`` sums to 1 and every forbidden transition and start state is
  exactly 0.0, with finite float32 outputs, under float32 AND ``mixed_float16``, for every
  topology (the additive ``-1e9`` mask is the RED proof: it NaNs under float16);
* the static output is one matrix at every step and batch row; the input-dependent one
  starts there (zero-initialized projection) and moves with the context once the
  projection is nonzero, without touching ``log_pi``;
* knobs: ``topology``, ``band_width`` (under ``banded``) and ``self_loop_init`` each change
  ``log_A``;
* gradients reach every weight;
* a ``get_config``/``from_config`` + ``set_weights`` round trip and a ``.keras`` round trip
  reproduce every output exactly (``rtol=0, atol=0``); registration keys are literal;
* the symbolic output dtype under ``mixed_float16`` is the float32 that ``call()`` returns;
* unknown batch and time axes trace; bad arguments raise ``ValueError`` naming the value.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.neural_hmm.transitions import (
    InputDependentTransition,
    StaticTransition,
)

from ..precision_arm_oracle import precision_policy

K = 4
CLASSES = [StaticTransition, InputDependentTransition]
TOPOLOGIES = ["ergodic", "left_to_right", "banded"]
DTYPES = ["float32", "mixed_float16"]


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _context(batch=2, steps=5, width=6, seed=0):
    return np.random.default_rng(seed).normal(size=(batch, steps, width)).astype("float32")


def _allowed(n_states, topology, band_width):
    """Independent restatement of the topologies: ``(start_keep, transition_keep)``."""
    source, destination = np.meshgrid(np.arange(n_states), np.arange(n_states), indexing="ij")
    if topology == "ergodic":
        return np.ones(n_states, bool), np.ones((n_states, n_states), bool)
    forward = destination >= source
    if topology == "banded":
        forward &= destination - source <= band_width
    return np.arange(n_states) == 0, forward


def _built(cls, seed=0, context=None, randomize=True, policy="float32", **kwargs):
    """Build a layer under the global ``policy`` (sublayers inherit only the global one)
    on ``context``, and give every weight a random nonzero value."""
    with precision_policy(policy):
        layer = cls(K, **kwargs)
        layer(_context() if context is None else context)
    if randomize:
        rng = np.random.default_rng(seed)
        layer.set_weights([rng.normal(size=w.shape).astype("float32") for w in layer.get_weights()])
    return layer


def _log_softmax_masked(logits, keep):
    """Numpy reference: log-softmax over the allowed entries; ``-inf`` elsewhere."""
    masked = np.where(keep, logits.astype("float64"), -np.inf)
    peak = masked.max(axis=-1, keepdims=True)
    return masked - peak - np.log(np.exp(masked - peak).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------
# Build contract and shapes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
def test_no_weights_exist_before_build(cls):
    layer = cls(K)
    assert not layer.built
    assert layer.weights == []


@pytest.mark.parametrize("cls", CLASSES)
def test_compute_output_shape_answers_unbuilt_and_matches_the_forward(cls):
    layer = cls(K)
    declared = layer.compute_output_shape((2, 5, 6))
    assert not layer.built
    log_pi, log_A = layer(_context())
    assert declared == ((2, K), (2, 5, K, K))
    assert (tuple(log_pi.shape), tuple(log_A.shape)) == declared


# ---------------------------------------------------------------------
# Values: masked log-softmax, row sums, exact zeros, float32 and mixed_float16
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("cls", CLASSES)
def test_rows_sum_to_one_and_forbidden_entries_are_exactly_zero(cls, topology, dtype):
    layer = _built(cls, topology=topology, band_width=1, policy=dtype)
    log_pi, log_A = layer(_context())
    assert log_pi.dtype == "float32" and log_A.dtype == "float32"
    log_pi, log_A = _np(log_pi), _np(log_A)
    assert np.all(np.isfinite(log_pi)) and np.all(np.isfinite(log_A))

    start_keep, transition_keep = _allowed(K, topology, 1)
    pi, A = np.exp(log_pi), np.exp(log_A)
    eps = np.finfo(np.float32).eps
    np.testing.assert_allclose(pi.sum(-1), 1.0, rtol=0, atol=4 * K * eps)
    np.testing.assert_allclose(A.sum(-1), 1.0, rtol=0, atol=4 * K * eps)
    assert np.all(pi[:, ~start_keep] == 0.0)
    assert np.all(A[:, :, ~transition_keep] == 0.0)
    assert np.all(A[:, :, transition_keep] > 0.0)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_static_log_A_is_the_masked_log_softmax_at_every_step_and_row(topology):
    layer = _built(StaticTransition, topology=topology, band_width=1)
    pi_logits, transition_logits = layer.get_weights()
    start_keep, transition_keep = _allowed(K, topology, 1)
    log_pi, log_A = (_np(t) for t in layer(_context()))

    expected_A = _log_softmax_masked(transition_logits, transition_keep)
    expected_pi = _log_softmax_masked(pi_logits, start_keep)
    allowed = np.broadcast_to(transition_keep, log_A.shape)
    np.testing.assert_allclose(
        log_A[allowed], np.broadcast_to(expected_A, log_A.shape)[allowed], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(log_pi[:, start_keep], np.broadcast_to(expected_pi, log_pi.shape)[:, start_keep],
                               rtol=0, atol=1e-5)
    # One matrix everywhere: bit-identical across batch rows and steps.
    assert np.all(log_A == log_A[:1, :1])
    assert np.all(log_pi == log_pi[:1])


def test_the_input_dependent_transition_starts_as_the_static_one_and_then_follows_the_context():
    context = _context(steps=6)
    dynamic = _built(InputDependentTransition, context=context, randomize=False)
    static = _built(StaticTransition, context=context, randomize=False)
    rng = np.random.default_rng(3)
    pi_logits = rng.normal(size=(K,)).astype("float32")
    transition_logits = rng.normal(size=(K, K)).astype("float32")
    static.set_weights([pi_logits, transition_logits])
    zero_kernel = dynamic.context_projection.get_weights()[0]
    assert np.all(zero_kernel == 0.0)
    dynamic.set_weights([pi_logits, transition_logits, zero_kernel])

    # Zero-initialized projection: exactly the static solution.
    for got, want in zip(dynamic(context), static(context)):
        np.testing.assert_array_equal(_np(got), _np(want))

    # A nonzero projection: each step follows its own context; log_pi does not move.
    dynamic.set_weights([pi_logits, transition_logits, rng.normal(size=zero_kernel.shape).astype("float32")])
    log_pi, log_A = (_np(t) for t in dynamic(context))
    assert np.abs(log_A[:, 1:] - log_A[:, :1]).max() > 1e-2
    np.testing.assert_array_equal(log_pi, _np(static(context)[0]))

    moved = context.copy()
    moved[:, 3] += 1.0
    _, log_A_moved = (_np(t) for t in dynamic(moved))
    assert np.abs(log_A_moved[:, 3] - log_A[:, 3]).max() > 1e-2
    np.testing.assert_array_equal(np.delete(log_A_moved, 3, axis=1), np.delete(log_A, 3, axis=1))


# ---------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------


def test_each_topology_changes_log_A():
    outputs = {t: _np(_built(StaticTransition, topology=t, band_width=1)(_context())[1]) for t in TOPOLOGIES}
    assert np.abs(outputs["ergodic"] - outputs["left_to_right"]).max() > 1e-3
    assert np.abs(outputs["left_to_right"] - outputs["banded"]).max() > 1e-3
    assert np.abs(outputs["ergodic"] - outputs["banded"]).max() > 1e-3


def test_band_width_changes_log_A_under_banded():
    narrow = _np(_built(StaticTransition, topology="banded", band_width=1)(_context())[1])
    wide = _np(_built(StaticTransition, topology="banded", band_width=2)(_context())[1])
    assert np.abs(narrow - wide).max() > 1e-3


def test_self_loop_init_changes_the_initial_log_A_on_the_diagonal():
    low = _np(_built(StaticTransition, randomize=False, self_loop_init=0.0)(_context())[1])
    high = _np(_built(StaticTransition, randomize=False, self_loop_init=3.0)(_context())[1])
    np.testing.assert_allclose(np.exp(low[0, 0]), 1.0 / K, rtol=0, atol=1e-6)
    stay = np.exp(np.diagonal(high[0, 0]))
    np.testing.assert_allclose(stay, np.exp(3.0) / (np.exp(3.0) + K - 1), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------


@pytest.mark.parametrize("topology", ["ergodic", "left_to_right"])
@pytest.mark.parametrize("cls", CLASSES)
def test_gradients_reach_every_weight(cls, topology):
    layer = _built(cls, topology=topology)
    context = tf.constant(_context())
    rng = np.random.default_rng(7)
    w_pi = tf.constant(rng.normal(size=(2, K)).astype("float32"))
    w_A = tf.constant(rng.normal(size=(2, 5, K, K)).astype("float32"))
    with tf.GradientTape() as tape:
        log_pi, log_A = layer(context, training=True)
        loss = tf.reduce_sum(w_pi * log_pi) + tf.reduce_sum(w_A * log_A)
    grads = tape.gradient(loss, layer.trainable_weights)
    assert len(grads) == len(layer.trainable_weights) == (2 if cls is StaticTransition else 3)
    for weight, grad in zip(layer.trainable_weights, grads):
        assert grad is not None, weight.path
        assert np.abs(_np(grad)).max() > 0.0, weight.path


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
def test_get_config_from_config_and_set_weights_reproduce_every_output_exactly(cls):
    layer = _built(cls, topology="banded", band_width=2, self_loop_init=1.5)
    config = layer.get_config()
    assert {k: config[k] for k in ("n_states", "topology", "band_width", "self_loop_init")} == {
        "n_states": K, "topology": "banded", "band_width": 2, "self_loop_init": 1.5,
    }
    clone = cls.from_config(config)
    clone(_context())
    clone.set_weights(layer.get_weights())
    for got, want in zip(clone(_context()), layer(_context())):
        np.testing.assert_array_equal(_np(got), _np(want))


@pytest.mark.parametrize("cls", CLASSES)
def test_a_keras_archive_round_trip_reproduces_every_output_and_weight(cls, tmp_path):
    inputs = keras.Input((5, 6))
    layer = cls(K, topology="left_to_right", name="transition")
    model = keras.Model(inputs, layer(inputs))
    rng = np.random.default_rng(11)
    layer.set_weights([rng.normal(size=w.shape).astype("float32") for w in layer.get_weights()])

    path = tmp_path / "transition.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    for got, want in zip(loaded.predict(_context(), verbose=0), model.predict(_context(), verbose=0)):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(loaded.get_layer("transition").get_weights(), layer.get_weights()):
        np.testing.assert_array_equal(got, want)


def test_the_registration_keys_are_literal():
    assert keras.saving.get_registered_name(StaticTransition) == (
        "dl_techniques.models.neural_hmm.transitions>StaticTransition"
    )
    assert keras.saving.get_registered_name(InputDependentTransition) == (
        "dl_techniques.models.neural_hmm.transitions>InputDependentTransition"
    )


# ---------------------------------------------------------------------
# Graph safety and dtypes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSES)
def test_the_symbolic_output_dtype_is_the_float32_that_call_returns_under_mixed_float16(cls):
    with precision_policy("mixed_float16"):
        layer = cls(K)
        symbolic = layer(keras.Input((None, 6)))
        eager = layer(_context())
    sublayers = [layer.context_projection] if cls is InputDependentTransition else []
    assert all(sub.compute_dtype == "float16" for sub in [layer] + sublayers)
    assert [s.dtype for s in symbolic] == ["float32", "float32"]
    assert [s.shape for s in symbolic] == [(None, K), (None, None, K, K)]
    assert [str(e.dtype.name) for e in eager] == ["float32", "float32"]


@pytest.mark.parametrize("cls", CLASSES)
def test_the_layer_traces_with_unknown_batch_and_time(cls):
    layer = _built(cls)

    @tf.function(input_signature=[tf.TensorSpec((None, None, 6), tf.float32)])
    def run(context):
        return layer(context)

    for steps in (3, 7):
        context = _context(batch=3, steps=steps)
        for got, want in zip(run(context), layer(context)):
            np.testing.assert_array_equal(_np(got), _np(want))
            assert got.shape[0] == 3


def test_the_static_transition_accepts_any_batch_time_tensor():
    """The default variant has no encoder: the model calls it on the (B, T) mask."""
    layer = StaticTransition(K)
    log_pi, log_A = layer(np.ones((3, 7), "float32"))
    assert tuple(log_pi.shape) == (3, K) and tuple(log_A.shape) == (3, 7, K, K)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_states": 0}, "n_states"),
        ({"n_states": K, "band_width": -1}, "band_width"),
        ({"n_states": K, "topology": "circular"}, "circular"),
    ],
)
@pytest.mark.parametrize("cls", CLASSES)
def test_bad_arguments_raise_naming_the_value(cls, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(**kwargs)


def test_the_input_dependent_transition_refuses_a_non_sequence_context():
    with pytest.raises(ValueError, match="batch, time, features"):
        InputDependentTransition(K)(np.ones((2, 5), "float32"))
