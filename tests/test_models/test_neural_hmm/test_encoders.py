"""``neural_hmm.encoders``: ``NeuralHMMEncoder``, for every ``encoder_type`` x ``causal``.

Guards (SC4), with ``dropout_rate = 0`` and non-constant inputs whose ``T``, ``F`` and
``hidden_dim`` all differ:

* **(a) future leak**: perturbing ``x[:, s:]`` moves ``out[:, :s]`` by exactly 0.0 when
  causal, and by more than 0 when not (``mlp`` is per-step, so exactly 0.0 under both);
* **(b) past-to-future liveness**: perturbing ``x[:, :s]`` moves ``out[:, s:]`` (``mlp``:
  exactly 0.0, it is per-step);
* **(c) batch isolation**: perturbing sample 0 leaves every other sample bit-identical;
* **(d) padding isolation**: padded content, large or NaN, leaves every valid output
  bit-identical and finite, with lengths ``T``, a middle length, 1 and 0 in one batch;
* **(e) truncation**: the valid outputs of a padded sequence equal those of the truncated
  sequence (float64, so the only difference is reassociation);
* no weights before ``build()``; ``compute_output_shape`` answers unbuilt and matches;
  unknown batch and time axes trace; under ``mixed_float16`` the output is finite float16
  in eager and symbolic calls;
* knobs: ``num_layers``, ``hidden_dim``, ``intermediate_size``, ``kernel_size``,
  ``max_seq_len``, ``normalization_type`` and ``ffn_type`` change the weight shapes where
  they are read; ``num_heads`` and ``causal`` (where it adds no weights) change the output
  with identical weights; ``dropout_rate`` and ``attention_dropout_rate`` move the output
  under ``training=True`` only, which also proves ``training`` is forwarded;
* every weight has a nonzero gradient after one optimizer step;
* ``get_config``/``from_config`` + ``set_weights`` and ``.keras`` round trips are exact;
  the registration key is literal; bad arguments raise ``ValueError`` naming the value.

Exact-zero arms compare two calls of ONE layer on inputs of one shape, so the kernels and
their reduction order are the same on both sides.

Two guards bite only on a GPU, where Keras 3.8 runs ``lstm``/``gru`` on cuDNN (D-016):
the dropout test catches a recurrent ``dropout=`` that the cuDNN path applies at
inference, and the graph-mode trace (and ``.keras`` ``predict``) with a length-0 row
catches an empty mask row reaching cuDNN's right-padding assertion. Run this file with
``CUDA_VISIBLE_DEVICES=1`` as well as on the CPU.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.neural_hmm.encoders import (
    ARGUMENT_READERS,
    ENCODER_TYPES,
    NeuralHMMEncoder,
)
from tests.numerics import reassociation_atol

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import precision_policy

B, T, F, H = 4, 9, 5, 8
#: The split step of arms (a) and (b).
S = 4
#: Valid lengths of the padded batch: full, middle, one step, empty.
LENGTHS = (T, 5, 1, 0)
#: Float64 bound of arm (e): two evaluations of one formula at different shapes.
ATOL_F64 = 1e-10

CASES = [(encoder_type, causal) for encoder_type in ENCODER_TYPES for causal in (True, False)]
CASE_IDS = [f"{encoder_type}-{'causal' if causal else 'noncausal'}" for encoder_type, causal in CASES]

#: Small, distinct widths: T, F, hidden_dim and intermediate_size all differ.
BASE_KWARGS = {"hidden_dim": H, "num_layers": 2, "num_heads": 2, "intermediate_size": 12, "max_seq_len": 16}


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _x(dtype="float32", seed=0):
    return np.random.default_rng(seed).normal(size=(B, T, F)).astype(dtype)


def _mask():
    return np.arange(T)[None, :] < np.asarray(LENGTHS)[:, None]


def _encoder(encoder_type, causal, seed=1, **kwargs):
    keras.utils.set_random_seed(seed)
    layer = NeuralHMMEncoder(encoder_type, causal=causal, **{**BASE_KWARGS, **kwargs})
    layer(_x())
    return layer


def _perturbation(shape, seed):
    return 3.0 * np.random.default_rng(seed).normal(size=shape).astype("float32")


# ---------------------------------------------------------------------
# The four isolation arms, and truncation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_a_future_perturbation_reaches_the_past_only_when_not_causal(encoder_type, causal):
    layer = _encoder(encoder_type, causal)
    x = _x()
    moved = x.copy()
    moved[:, S:] += _perturbation((B, T - S, F), 2)
    delta = np.abs(_np(layer(moved)) - _np(layer(x)))
    assert delta[:, S:].max() > 0.0  # the perturbation is live where it lands
    if causal or encoder_type == "mlp":
        assert delta[:, :S].max() == 0.0, delta[:, :S].max()
    else:
        assert delta[:, :S].max(axis=(1, 2)).min() > 0.0, delta[:, :S].max(axis=(1, 2))


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_a_past_perturbation_reaches_the_future_except_per_step(encoder_type, causal):
    layer = _encoder(encoder_type, causal)
    x = _x()
    moved = x.copy()
    moved[:, :S] += _perturbation((B, S, F), 3)
    delta = np.abs(_np(layer(moved)) - _np(layer(x)))
    if encoder_type == "mlp":
        assert delta[:, S:].max() == 0.0, delta[:, S:].max()
    else:
        # Every sample, at the first step past the split.
        assert delta[:, S].max(axis=-1).min() > 0.0, delta[:, S].max(axis=-1)


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_perturbing_one_sample_leaves_every_other_sample_bit_identical(encoder_type, causal):
    layer = _encoder(encoder_type, causal)
    x, mask = _x(), _mask()
    moved = x.copy()
    moved[0] += _perturbation((T, F), 4)
    base, out = _np(layer(x, mask=mask)), _np(layer(moved, mask=mask))
    assert np.abs(out[0] - base[0]).max() > 0.0
    np.testing.assert_array_equal(out[1:], base[1:])


@pytest.mark.parametrize("content", ["large", "nan"])
@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_padded_content_leaves_every_valid_output_bit_identical(encoder_type, causal, content):
    layer = _encoder(encoder_type, causal)
    x, mask = _x(), _mask()
    padded = x.copy()
    padded[~mask] = np.nan if content == "nan" else 1e3 + _perturbation((int((~mask).sum()), F), 5)
    base, out = _np(layer(x, mask=mask)), _np(layer(padded, mask=mask))
    assert np.all(np.isfinite(out)), "padded content reached an output as a non-finite value"
    np.testing.assert_array_equal(out[mask], base[mask])


# A truncated length-1 sequence is a softmax over one key, which Keras warns about.
@pytest.mark.filterwarnings("ignore:.*softmax over axis -1 of a tensor.*:UserWarning")
@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_padded_valid_outputs_equal_the_truncated_sequence(encoder_type, causal):
    with precision_policy("float64"):
        layer = _encoder(encoder_type, causal)
        x, mask = _x("float64"), _mask()
        padded = _np(layer(x, mask=mask))
        for row, length in enumerate(LENGTHS):
            if length == 0:
                continue
            truncated = _np(layer(x[row:row + 1, :length]))[0]
            np.testing.assert_allclose(padded[row, :length], truncated, rtol=0, atol=ATOL_F64)


# ---------------------------------------------------------------------
# Build contract, shapes, dtypes
# ---------------------------------------------------------------------


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_no_weights_exist_before_build_and_the_output_shape_is_declared(encoder_type, causal):
    layer = NeuralHMMEncoder(encoder_type, causal=causal, **BASE_KWARGS)
    assert not layer.built and layer.weights == []
    declared = layer.compute_output_shape((B, T, F))
    assert not layer.built
    assert declared == (B, T, H) == tuple(_np(layer(_x())).shape)


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_unknown_batch_and_time_axes_trace(encoder_type, causal):
    layer = NeuralHMMEncoder(encoder_type, causal=causal, **BASE_KWARGS)
    out = layer(keras.Input((None, F)), mask=keras.Input((None,), dtype="bool"))
    assert tuple(out.shape) == (None, None, H)

    traced = tf.function(
        lambda x, m: layer(x, mask=m),
        input_signature=[tf.TensorSpec((None, None, F), tf.float32), tf.TensorSpec((None, None), tf.bool)],
    )
    # Graph and eager kernels may fuse and order reductions differently, so this compares
    # within a reassociation bound over every contraction one path chains, not bit-for-bit.
    eager = _np(layer(_x(), mask=_mask()))
    # input projection, attention/recurrent widths, ffn width, attention over T, conv taps
    # (the default kernel_size 3 times H)
    reductions = [F, H, BASE_KWARGS["intermediate_size"], T, 3 * H]
    atol = reassociation_atol(reductions, 2 * BASE_KWARGS["num_layers"] + 2, np.abs(eager).max())
    np.testing.assert_allclose(_np(traced(_x(), _mask())), eager, rtol=0, atol=atol)


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_the_output_follows_the_compute_dtype_under_mixed_float16(encoder_type, causal):
    with precision_policy("mixed_float16"):
        layer = NeuralHMMEncoder(encoder_type, causal=causal, **BASE_KWARGS)
        symbolic = layer(keras.Input((None, F)), mask=keras.Input((None,), dtype="bool"))
        eager = layer(_x(), mask=_mask())
    assert all(block.compute_dtype == "float16" for block in layer.blocks)
    assert symbolic.dtype == eager.dtype == "float16"
    assert np.all(np.isfinite(_np(eager)))


def test_a_static_length_beyond_max_seq_len_is_refused_at_build():
    with pytest.raises(ValueError, match="max_seq_len"):
        NeuralHMMEncoder("transformer", **{**BASE_KWARGS, "max_seq_len": T - 1})(_x())


@pytest.mark.parametrize("encoder_type", ["lstm", "conv"])
def test_paths_without_a_position_table_have_no_length_limit(encoder_type):
    layer = NeuralHMMEncoder(encoder_type, **{**BASE_KWARGS, "max_seq_len": 2})
    assert tuple(_np(layer(_x())).shape) == (B, T, H)


# ---------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------


def _builders(encoder_type, knob, values, causal=True, **kwargs):
    return {value: (lambda value=value: _encoder(encoder_type, causal, **kwargs, **{knob: value})) for value in values}


STRUCTURAL_KNOBS = [
    (encoder_type, "num_layers", (1, 2)) for encoder_type in ENCODER_TYPES
] + [
    (encoder_type, "hidden_dim", (8, 12)) for encoder_type in ENCODER_TYPES
] + [
    (encoder_type, "normalization_type", ("layer_norm", "rms_norm")) for encoder_type in ENCODER_TYPES
] + [
    ("mlp", "intermediate_size", (12, 20)),
    ("transformer", "intermediate_size", (12, 20)),
    ("transformer", "max_seq_len", (16, 24)),
    ("transformer", "ffn_type", ("mlp", "glu")),
    ("conv", "kernel_size", (3, 5)),
    ("lstm", "causal", (True, False)),
    ("gru", "causal", (True, False)),
]


@pytest.mark.parametrize(
    "encoder_type, knob, values", STRUCTURAL_KNOBS, ids=[f"{t}-{k}" for t, k, _ in STRUCTURAL_KNOBS]
)
def test_a_structural_knob_changes_the_weight_shapes(encoder_type, knob, values):
    if knob == "causal":
        builders = {value: (lambda value=value: _encoder(encoder_type, value)) for value in values}
    else:
        builders = _builders(encoder_type, knob, values)
    assert_structural_knob_changes_weights(builders, knob=knob)


VALUE_KNOBS = [("transformer", "num_heads", (2, 4)), ("transformer", "causal", (True, False)), ("conv", "causal", (True, False))]


@pytest.mark.parametrize("encoder_type, knob, values", VALUE_KNOBS, ids=[f"{t}-{k}" for t, k, _ in VALUE_KNOBS])
def test_a_value_knob_changes_the_output_with_identical_weights(encoder_type, knob, values):
    if knob == "causal":
        builders = {value: (lambda value=value: _encoder(encoder_type, value)) for value in values}
    else:
        builders = _builders(encoder_type, knob, values)
    assert_value_knob_changes_output(builders, _x(), knob=knob)


DROPOUT_KNOBS = [(encoder_type, "dropout_rate") for encoder_type in ENCODER_TYPES] + [
    ("transformer", "attention_dropout_rate")
]


@pytest.mark.parametrize("encoder_type, knob", DROPOUT_KNOBS, ids=[f"{t}-{k}" for t, k in DROPOUT_KNOBS])
def test_dropout_moves_the_output_in_training_only(encoder_type, knob):
    off = _encoder(encoder_type, True)
    on = _encoder(encoder_type, True, **{knob: 0.5})
    on.set_weights(off.get_weights())
    x = _x()
    np.testing.assert_array_equal(_np(on(x, training=False)), _np(off(x, training=False)))
    np.testing.assert_array_equal(_np(off(x, training=True)), _np(off(x, training=False)))
    assert np.abs(_np(on(x, training=True)) - _np(on(x, training=False))).max() > 0.0


# ---------------------------------------------------------------------
# ARGUMENT_READERS is the wiring (D-025: the model refuses what it says is unread)
# ---------------------------------------------------------------------

#: A non-default probe value per listed argument, valid for every encoder type.
UNREAD_PROBES = {
    "num_layers": 3, "normalization_type": "rms_norm", "dropout_rate": 0.5, "intermediate_size": 20,
    "num_heads": 4, "max_seq_len": 24, "ffn_type": "glu", "attention_dropout_rate": 0.5, "kernel_size": 5,
}
UNREAD_PAIRS = [
    (encoder_type, name) for name, readers in ARGUMENT_READERS.items()
    for encoder_type in ENCODER_TYPES if encoder_type not in readers
]


def test_the_reader_table_lists_every_knob_and_matches_the_read_side_tests():
    """Every argument but the always-read ones is listed, and the (type, argument) pairs
    it calls read are exactly the pairs the knob tests above prove move something."""
    import inspect

    params = set(inspect.signature(NeuralHMMEncoder.__init__).parameters)
    assert set(ARGUMENT_READERS) == params - {"self", "kwargs", "encoder_type", "causal", "hidden_dim"}
    assert set(UNREAD_PROBES) == set(ARGUMENT_READERS)
    tested = {(t, k) for t, k, _ in STRUCTURAL_KNOBS + VALUE_KNOBS} | set(DROPOUT_KNOBS)
    listed = {(t, k) for k, readers in ARGUMENT_READERS.items() for t in readers}
    assert {pair for pair in tested if pair[1] in ARGUMENT_READERS} == listed


@pytest.mark.parametrize("encoder_type, knob", UNREAD_PAIRS, ids=[f"{t}-{k}" for t, k in UNREAD_PAIRS])
def test_an_argument_the_table_calls_unread_changes_nothing(encoder_type, knob):
    """Same seed, the argument off its default: identical weights and identical outputs in
    BOTH modes (training=True would expose a dropout rate that is read)."""
    plain = _encoder(encoder_type, True)
    probed = _encoder(encoder_type, True, **{knob: UNREAD_PROBES[knob]})
    assert [w.shape for w in plain.weights] == [w.shape for w in probed.weights]
    for a, b in zip(plain.get_weights(), probed.get_weights()):
        np.testing.assert_array_equal(a, b)
    x = _x()
    for training in (False, True):
        keras.utils.set_random_seed(7)
        expected = _np(plain(x, training=training))
        keras.utils.set_random_seed(7)
        np.testing.assert_allclose(_np(probed(x, training=training)), expected, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_every_weight_has_a_nonzero_gradient_after_one_optimizer_step(encoder_type, causal):
    layer = _encoder(encoder_type, causal)
    x, mask = tf.constant(_x()), tf.constant(_mask())
    # A fixed random projection, not a mean square: the output norm makes the mean
    # square of its output nearly constant, which would starve every weight before it.
    projection = tf.constant(np.random.default_rng(6).normal(size=(B, T, H)).astype("float32"))

    def loss_fn(out):
        return tf.reduce_sum(out * projection)

    optimizer = keras.optimizers.SGD(learning_rate=0.05)
    with tf.GradientTape() as tape:
        loss = loss_fn(layer(x, mask=mask, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, layer.trainable_weights), layer.trainable_weights))

    class _Bound:
        trainable_weights = layer.trainable_weights

        def __call__(self, inputs, training=None):
            return layer(inputs, mask=mask, training=training)

    assert_gradients_reach_every_trainable_weight(_Bound(), x, loss_fn=loss_fn)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_get_config_from_config_and_set_weights_reproduce_the_output_exactly(encoder_type, causal):
    kwargs = {**BASE_KWARGS, "kernel_size": 5, "ffn_type": "glu", "normalization_type": "rms_norm",
              "dropout_rate": 0.1, "attention_dropout_rate": 0.2}
    layer = NeuralHMMEncoder(encoder_type, causal=causal, **kwargs)
    layer(_x())
    config = layer.get_config()
    assert {k: config[k] for k in ["encoder_type", "causal", *kwargs]} == {
        "encoder_type": encoder_type, "causal": causal, **kwargs,
    }
    clone = NeuralHMMEncoder.from_config(config)
    clone(_x())
    clone.set_weights(layer.get_weights())
    np.testing.assert_array_equal(_np(clone(_x(), mask=_mask())), _np(layer(_x(), mask=_mask())))


@pytest.mark.parametrize("encoder_type, causal", CASES, ids=CASE_IDS)
def test_a_keras_archive_round_trip_reproduces_the_output_and_weights(encoder_type, causal, tmp_path):
    x_input, mask_input = keras.Input((T, F)), keras.Input((T,), dtype="bool")
    encoder = NeuralHMMEncoder(encoder_type, causal=causal, name="encoder", **BASE_KWARGS)
    model = keras.Model([x_input, mask_input], encoder(x_input, mask=mask_input))
    path = tmp_path / "encoder.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    inputs = [_x(), _mask()]
    np.testing.assert_array_equal(loaded.predict(inputs, verbose=0), model.predict(inputs, verbose=0))
    want = model.get_layer("encoder").get_weights()
    got = loaded.get_layer("encoder").get_weights()
    assert len(got) == len(want) > 0
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_the_registration_key_is_literal():
    assert keras.saving.get_registered_name(NeuralHMMEncoder) == (
        "dl_techniques.models.neural_hmm.encoders>NeuralHMMEncoder"
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder_type, kwargs, match",
    [
        ("cnn", {}, "cnn"),
        ("mlp", {"hidden_dim": 0}, "hidden_dim"),
        ("mlp", {"num_layers": 0}, "num_layers"),
        ("conv", {"kernel_size": 0}, "kernel_size"),
        ("transformer", {"max_seq_len": -1}, "max_seq_len"),
        ("mlp", {"dropout_rate": 1.0}, "dropout_rate"),
        ("transformer", {"attention_dropout_rate": -0.1}, "attention_dropout_rate"),
        ("transformer", {"hidden_dim": 10, "num_heads": 4}, "divisible"),
    ],
)
def test_bad_arguments_raise_naming_the_value(encoder_type, kwargs, match):
    with pytest.raises(ValueError, match=match):
        NeuralHMMEncoder(encoder_type, **kwargs)


def test_inputs_that_are_not_rank_3_are_refused_at_build():
    with pytest.raises(ValueError, match="features"):
        NeuralHMMEncoder("mlp", hidden_dim=H)(np.zeros((B, T), "float32"))
