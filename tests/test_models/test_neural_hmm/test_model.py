"""``neural_hmm.model``: ``NeuralHMM``, its five presets, its loss and its factories.

Guards (SC5), each through the shared oracle with its real signature:

* **smoke contract** (``smoke_contract_oracle``): per preset, the D-003 dict with exact
  shapes, float32 dtypes, finite values and normalized ``exp(log_pi)``/``exp(log_A)``
  rows, and the contract is shown to reject a broken forward;
* **knobs** (``knob_sensitivity_oracle``): every constructor argument in a configuration
  where it is read; ``sample_weight`` through ``compute_loss``; dropout in training only;
* **gradients** (``gradient_flow_oracle``) on the real loss, and **dead components**
  (``dead_component_oracle``) after one real ``fit()`` step, per preset, with each waiver
  named, measured and two-sided;
* **lazy build** (``lazy_build_contract_oracle``) and the **``.keras`` round trip**
  (``roundtrip_instrument_oracle``, ``rtol=0``/``atol=0`` on outputs and every weight,
  perturbed first), per preset; the ``get_config`` round trip on a config with every
  argument off its default; the literal registration keys;
* shapes: ``compute_output_shape`` on an unbuilt model, a ``keras.Input`` trace, batched
  ``predict`` and ``predict`` at a second sequence length;
* training: ``fit`` per preset, the discriminative preset LEARNS on separable data (its
  gold-path accuracy rises), ``jit_compile=True`` per encoder type with the resolved
  ``model.jit_compile`` asserted;
* numerics: padded likelihood equals the truncated sequence's per preset (float64),
  NaN in padded content changes neither the loss nor the gradients, length-0/length-1
  sequences, ``mixed_float16`` (float32 outputs, D-010) and float64 arms;
* factories: ``from_variant`` overrides, ``pretrained=True``, unknown variants,
  ``create_neural_hmm`` == ``from_variant``.

Run on the CPU and with ``CUDA_VISIBLE_DEVICES=1``: the ``lstm`` paths take cuDNN on a GPU.
"""

import inspect
import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.time_series.neural_hmm import inference
from dl_techniques.models.time_series.neural_hmm.emissions import (
    DiscriminativeEmission,
    GenerativeEmission,
)
from dl_techniques.models.time_series.neural_hmm.encoders import ENCODER_TYPES, NeuralHMMEncoder
from dl_techniques.models.time_series.neural_hmm.transitions import (
    InputDependentTransition,
    StaticTransition,
)

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..lazy_build_contract_oracle import assert_lazy_build_costs_nothing
from ..precision_arm_oracle import (
    assert_float64_arm,
    assert_precision_arm,
    default_call,
    float64_scope,
    precision_policy,
)
from ..precision_arm_subjects import SUBJECTS
from ..roundtrip_instrument_oracle import (
    assert_roundtrip_output_values,
    assert_weights_restored_before_first_call,
    measure_roundtrip,
)
from ..smoke_contract_oracle import assert_contract_rejects_a_broken_forward
from ..test_sam.dead_component_oracle import fit_one_step_moved_variables

B, T, D, K, V = 4, 7, 2, 3, 5
#: Valid lengths of the padded batch: full, middle, one step, empty.
LENGTHS = (T, 4, 1, 0)
VARIANTS = sorted(NeuralHMM.MODEL_VARIANTS)
OUTPUT_KEYS = {"log_pi", "log_A", "log_emit"}
#: Small widths, all different from T, D, K and V.
SMALL = {"hidden_dim": 8, "state_emb_dim": 6, "num_layers": 1, "num_heads": 2, "intermediate_size": 12,
         "max_seq_len": 16}

#: Named dead-component waivers, per preset: the weights that MUST NOT move after one
#: optimizer step, and why. Two-sided: every listed weight must be exactly unmoved and
#: every other weight must move.
# DECISION plan-2026-10-13T192257-1bb3bf1f/D-019: do NOT "fix" these by widening the
# oracle or deleting the rows; both are measured exact zeros with a follow-up test each.
ONE_STEP_UNMOVED = {
    # The transition's context projection is zero-initialized (training starts at the
    # static solution), so at step 1 the gradient reaching the encoder through it is
    # W^T g = 0 exactly. A second step moves them (test_..._move_after_a_second_step).
    "io-hmm": "encoder/",
    # The banded topology forces the chain to start in state 0: log_pi = log_softmax of
    # [x_0, sentinel, ...] is exactly [0, -inf-ish, ...] for every x_0, so pi_logits gets
    # an exactly-zero gradient forever. Structural, not a training state.
    "left-to-right-segmenter": "transition/pi_logits",
}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _mask(lengths=LENGTHS, steps=T):
    return (np.arange(steps)[None, :] < np.asarray(lengths)[:, None]).astype("float32")


def _obs(obs_type="gaussian", steps=T, batch=B, seed=0):
    rng = np.random.default_rng(seed)
    if obs_type == "categorical":
        return rng.integers(0, V, (batch, steps)).astype("int32")
    if obs_type == "poisson":
        return rng.poisson(2.0, (batch, steps, D)).astype("float32")
    return rng.normal(size=(batch, steps, D)).astype("float32")


def _inputs(obs_type="gaussian", seed=0, lengths=LENGTHS, steps=T):
    return {"obs": _obs(obs_type, steps, len(lengths), seed), "mask": _mask(lengths, steps)}


def _labels(seed=0, lengths=LENGTHS, steps=T):
    states = np.random.default_rng(seed + 100).integers(0, K, (len(lengths), steps)).astype("int32")
    return np.where(_mask(lengths, steps) > 0, states, -1).astype("int32")


def _kwargs(obs_type="gaussian", **overrides):
    shape = {"vocab_size": V} if obs_type == "categorical" else {"obs_dim": D}
    return {"n_states": K, "obs_type": obs_type, **shape, **SMALL, **overrides}


def _model(variant="neural-hmm", obs_type="gaussian", seed=1, **overrides):
    keras.utils.set_random_seed(seed)
    return create_neural_hmm(variant, **_kwargs(obs_type, **overrides))


def _y(model, seed=0, lengths=LENGTHS, steps=T):
    return _labels(seed, lengths, steps) if model.mode == "discriminative" else None


def _flat(out):
    """Every output, flattened into one vector (the knob oracle compares one array)."""
    return np.concatenate([_np(out[key]).ravel() for key in sorted(out)])


def _assert_contract(out):
    """The forward contract the smoke test enforces (and the breakers must violate)."""
    assert isinstance(out, dict), f"expected a dict, got {type(out).__name__}"
    assert set(out) == OUTPUT_KEYS, f"keys {sorted(out)}"
    expected = {"log_pi": (B, K), "log_A": (B, T, K, K), "log_emit": (B, T, K)}
    for key, shape in expected.items():
        assert tuple(out[key].shape) == shape, f"{key}: {tuple(out[key].shape)} != {shape}"
        assert _np(out[key]).dtype == np.float32, f"{key}: dtype {out[key].dtype}"
        assert np.isfinite(_np(out[key])).all(), f"{key} is not finite"
    np.testing.assert_allclose(np.exp(_np(out["log_pi"])).sum(-1), 1.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(np.exp(_np(out["log_A"])).sum(-1), 1.0, rtol=0, atol=1e-5)


def _loss(model, inputs, y=None, training=False, sample_weight=None):
    return float(_np(model.compute_loss(inputs, y, model(inputs, training=training), sample_weight)))


# ---------------------------------------------------------------------
# the variant table
# ---------------------------------------------------------------------


def test_the_variant_table_is_the_d005_table():
    assert VARIANTS == sorted(
        ["neural-hmm", "autoregressive-neural-hmm", "io-hmm", "left-to-right-segmenter", "neural-crf-tagger"]
    )
    params = set(inspect.signature(NeuralHMM.__init__).parameters) - {"self", "kwargs"}
    for name, preset in NeuralHMM.MODEL_VARIANTS.items():
        assert set(preset) <= params, f"{name}: {sorted(set(preset) - params)} are not constructor arguments"


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_preset_builds_only_what_it_reads(variant):
    """D-005: an encoder exists exactly when a context reader exists."""
    model = _model(variant)
    reads = (
        model.emission_context == "autoregressive" or model.transition_type == "input_dependent"
        or model.mode == "discriminative"
    )
    assert (model.encoder is not None) == reads
    assert (variant == "neural-hmm") == (model.encoder is None)


# ---------------------------------------------------------------------
# smoke contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_smoke_build_and_forward(variant):
    _assert_contract(_model(variant)(_inputs(), training=False))


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_contract_rejects_a_broken_forward(variant):
    rejections = assert_contract_rejects_a_broken_forward(_model(variant), _inputs(), _assert_contract)
    assert len(rejections) == 3


# ---------------------------------------------------------------------
# knob sensitivity
# ---------------------------------------------------------------------

#: (knob, values, preset, obs_type, extra overrides): weight shapes must differ.
STRUCTURAL_KNOBS = [
    ("n_states", (2, 3), "neural-hmm", "gaussian", {}),
    ("obs_dim", (2, 3), "neural-hmm", "gaussian", {}),
    ("vocab_size", (5, 6), "neural-hmm", "categorical", {}),
    ("obs_type", ("gaussian", "poisson"), "neural-hmm", "gaussian", {}),
    ("hidden_dim", (8, 12), "autoregressive-neural-hmm", "gaussian", {}),
    ("state_emb_dim", (6, 9), "neural-hmm", "gaussian", {}),
    # ffn_type reaches only the transformer encoder's blocks; the emission head is a fixed
    # mlp (D-020), so the knob is exercised under a transformer preset.
    ("ffn_type", ("mlp", "glu"), "left-to-right-segmenter", "gaussian", {}),
    ("encoder_type", ("lstm", "gru"), "autoregressive-neural-hmm", "gaussian", {}),
    ("num_layers", (1, 2), "autoregressive-neural-hmm", "gaussian", {}),
    ("intermediate_size", (12, 16), "left-to-right-segmenter", "gaussian", {}),
    ("max_seq_len", (16, 20), "left-to-right-segmenter", "gaussian", {}),
    ("kernel_size", (2, 3), "io-hmm", "gaussian", {}),
    ("normalization_type", ("layer_norm", "rms_norm"), "neural-crf-tagger", "gaussian", {}),
    ("mode", ("generative", "discriminative"), "neural-hmm", "gaussian", {}),
    ("emission_context", ("none", "autoregressive"), "neural-hmm", "gaussian", {}),
    ("transition", ("static", "input_dependent"), "neural-hmm", "gaussian", {}),
]

#: (knob, values, preset, obs_type, extra overrides): same weights, different output.
VALUE_KNOBS = [
    ("topology", ("ergodic", "left_to_right", "banded"), "neural-hmm", "gaussian", {}),
    ("band_width", (1, 2), "neural-hmm", "gaussian", {"topology": "banded", "n_states": 4}),
    ("self_loop_init", (1.0, 3.0), "neural-hmm", "gaussian", {}),
    ("min_scale", (1e-3, 0.5), "neural-hmm", "gaussian", {}),
    ("min_scale", (1e-3, 0.5), "neural-hmm", "poisson", {}),
    ("num_heads", (1, 2), "left-to-right-segmenter", "gaussian", {}),
]


def _knob_builders(knob, values, variant, obs_type, extra):
    def build(value):
        overrides = {**extra, knob: value}
        kind = value if knob == "obs_type" else obs_type
        if knob == "obs_type" and value != obs_type:
            overrides.pop("vocab_size", None)
        return lambda: create_neural_hmm(variant, **{**_kwargs(kind), **overrides})

    return {value: build(value) for value in values}


@pytest.mark.parametrize("knob,values,variant,obs_type,extra", STRUCTURAL_KNOBS,
                         ids=[f"{row[0]}-{row[2]}" for row in STRUCTURAL_KNOBS])
def test_a_structural_knob_changes_the_weights(knob, values, variant, obs_type, extra):
    builders = _knob_builders(knob, values, variant, obs_type, extra)

    def built(value, builder):
        def run():
            model = builder()
            x = _inputs(value if knob == "obs_type" else obs_type)
            if knob == "obs_dim":
                x["obs"] = np.random.default_rng(0).normal(size=(B, T, value)).astype("float32")
            model(x, training=False)
            return model
        return run

    assert_structural_knob_changes_weights({k: built(k, b) for k, b in builders.items()}, knob=knob)


@pytest.mark.parametrize("knob,values,variant,obs_type,extra", VALUE_KNOBS,
                         ids=[f"{row[0]}-{row[3]}-{row[2]}" for row in VALUE_KNOBS])
def test_a_value_knob_changes_the_output(knob, values, variant, obs_type, extra):
    builders = _knob_builders(knob, values, variant, obs_type, extra)
    assert_value_knob_changes_output(builders, _inputs(obs_type), knob=knob, extract=_flat)


def test_sample_weight_reaches_the_loss():
    """The reduction is sum_b w_b nll_b / sum_b w_b len_b, by hand."""
    model = _model("neural-hmm")
    x = _inputs()
    out = model(x, training=False)
    nll = -_np(inference.forward_log_likelihood(out["log_pi"], out["log_A"], out["log_emit"], x["mask"]))
    lengths = np.asarray(LENGTHS, dtype="float64")
    weights = np.array([0.5, 2.0, 1.0, 3.0], dtype="float32")

    unweighted = _loss(model, x)
    weighted = _loss(model, x, sample_weight=weights)
    np.testing.assert_allclose(unweighted, nll.sum() / lengths.sum(), rtol=1e-6)
    np.testing.assert_allclose(weighted, (weights * nll).sum() / (weights * lengths).sum(), rtol=1e-6)
    assert abs(weighted - unweighted) > 1e-3


@pytest.mark.parametrize("knob", ["dropout_rate", "attention_dropout_rate"])
def test_dropout_moves_the_output_in_training_only(knob):
    x = _inputs()
    plain = _model("left-to-right-segmenter")
    noisy = _model("left-to-right-segmenter", **{knob: 0.5})
    plain(x)
    noisy(x)
    noisy.set_weights(plain.get_weights())
    assert np.array_equal(_flat(plain(x, training=False)), _flat(noisy(x, training=False)))
    assert np.max(np.abs(_flat(plain(x, training=True)) - _flat(noisy(x, training=True)))) > 1e-4


# ---------------------------------------------------------------------
# gradients and dead components
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_gradients_reach_every_trainable_weight(variant):
    model = _model(variant)
    x = _inputs()
    y = _y(model)
    model(x)
    waiver = ONE_STEP_UNMOVED.get(variant)
    assert_gradients_reach_every_trainable_weight(
        model, x, loss_fn=lambda out: model.compute_loss(x, y, out), expect_zero=(waiver,) if waiver else ()
    )


def _compiled(variant, **overrides):
    model = _model(variant, **overrides)
    x = _inputs()
    model(x)
    model.compile(optimizer=keras.optimizers.Adam(1e-2))
    return model, x, _y(model)


def _waived(model, variant):
    pattern = ONE_STEP_UNMOVED.get(variant)
    return {
        getattr(v, "path", v.name) for v in model.trainable_variables if pattern and pattern in getattr(v, "path", "")
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_built_weight_moves_after_one_fit_step(variant):
    model, x, y = _compiled(variant)
    report = fit_one_step_moved_variables(model, x, y, batch_size=B)
    assert np.isfinite(report.final_loss)
    expected_unmoved = _waived(model, variant)
    assert set(report.unmoved) == expected_unmoved, report.summary()
    assert report.n_moved == report.total - len(expected_unmoved) > 0


def test_the_io_hmm_encoder_moves_after_a_second_step():
    """The io-hmm waiver is a step-1 fact only: once W leaves zero, the encoder learns."""
    model, x, y = _compiled("io-hmm")
    fit_one_step_moved_variables(model, x, y, batch_size=B)
    report = fit_one_step_moved_variables(model, x, y, batch_size=B)
    assert report.unmoved == (), report.summary()


def test_the_segmenter_start_logits_never_move():
    """The segmenter waiver is structural: pi_logits stays put however long it trains."""
    model, x, y = _compiled("left-to-right-segmenter")
    for _ in range(3):
        report = fit_one_step_moved_variables(model, x, y, batch_size=B)
        assert set(report.unmoved) == _waived(model, "left-to-right-segmenter"), report.summary()


# ---------------------------------------------------------------------
# build, serialization, registration
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_lazy_build_costs_nothing(variant):
    input_shape = {"obs": (None, T, D), "mask": (None, T)}
    report = assert_lazy_build_costs_nothing(lambda: _model(variant), _inputs, input_shape=input_shape)
    assert report["materialization"]["ratio"] == 1.0, report["materialization"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_keras_round_trip_is_exact(variant):
    """Outputs at rtol=0/atol=0 and every weight read BEFORE the loaded model's first call."""
    report = measure_roundtrip(lambda: _model(variant), _inputs)
    assert report["self_max_delta"] == 0.0
    assert_roundtrip_output_values(report, atol=0.0)
    assert_weights_restored_before_first_call(report, atol=0.0)


@pytest.mark.parametrize("variant", ["neural-hmm", "neural-crf-tagger"])
def test_a_trained_model_reloads_exactly(variant, tmp_path):
    """After a real fit: the archive in tmp_path reproduces outputs, every weight and the loss."""
    model, x, y = _compiled(variant)
    model.fit(x, y, batch_size=2, epochs=1, verbose=0)
    path = os.path.join(str(tmp_path), "neural_hmm.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    for a, b in zip(model.weights, loaded.weights):
        assert np.array_equal(_np(a), _np(b)), a.path
    assert np.array_equal(_flat(model(x, training=False)), _flat(loaded(x, training=False)))
    assert _loss(model, x, y) == _loss(loaded, x, y)


def test_get_config_round_trips_every_argument():
    """Every argument set OFF its default, so a dropped key changes the rebuilt config."""
    overrides = {
        "n_states": 5, "mode": "generative", "obs_type": "poisson", "obs_dim": 3, "vocab_size": None,
        "emission_context": "autoregressive", "transition": "input_dependent", "topology": "banded",
        "band_width": 2, "self_loop_init": 1.5, "min_scale": 0.01, "hidden_dim": 12, "state_emb_dim": 5,
        "encoder_type": "gru", "num_layers": 3, "num_heads": 3, "intermediate_size": 20, "max_seq_len": 40,
        "kernel_size": 5, "ffn_type": "glu", "normalization_type": "rms_norm", "dropout_rate": 0.1,
        "attention_dropout_rate": 0.2,
    }
    params = set(inspect.signature(NeuralHMM.__init__).parameters) - {"self", "kwargs"}
    assert set(overrides) == params
    model = NeuralHMM(**overrides)
    config = model.get_config()
    assert {k: config[k] for k in params} == overrides
    assert NeuralHMM.from_config(config).get_config() == config


def test_the_registration_keys_are_literal():
    names = {cls: keras.saving.get_registered_name(cls) for cls in (
        NeuralHMM, NeuralHMMEncoder, StaticTransition, InputDependentTransition, GenerativeEmission,
        DiscriminativeEmission,
    )}
    assert names == {
        NeuralHMM: "dl_techniques.models.neural_hmm.model>NeuralHMM",
        NeuralHMMEncoder: "dl_techniques.models.neural_hmm.encoders>NeuralHMMEncoder",
        StaticTransition: "dl_techniques.models.neural_hmm.transitions>StaticTransition",
        InputDependentTransition: "dl_techniques.models.neural_hmm.transitions>InputDependentTransition",
        GenerativeEmission: "dl_techniques.models.neural_hmm.emissions>GenerativeEmission",
        DiscriminativeEmission: "dl_techniques.models.neural_hmm.emissions>DiscriminativeEmission",
    }


# ---------------------------------------------------------------------
# shapes and tracing
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_compute_output_shape_answers_unbuilt_and_matches(variant):
    model = _model(variant)
    shapes = model.compute_output_shape({"obs": (None, T, D), "mask": (None, T)})
    assert not model.built
    out = model(_inputs(), training=False)
    assert shapes == {"log_pi": (None, K), "log_A": (None, T, K, K), "log_emit": (None, T, K)}
    assert {k: (B,) + tuple(v[1:]) for k, v in shapes.items()} == {k: tuple(v.shape) for k, v in out.items()}


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_symbolic_trace_has_an_unknown_batch(variant):
    inputs = {"obs": keras.Input((T, D)), "mask": keras.Input((T,))}
    out = _model(variant)(inputs)
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "log_pi": (None, K), "log_A": (None, T, K, K), "log_emit": (None, T, K)
    }
    assert all(v.dtype == "float32" for v in out.values())


@pytest.mark.parametrize("variant", VARIANTS)
def test_predict_batches_and_runs_at_a_second_length(variant):
    """predict retraces with T = None at a second length (D-014); both lengths must work."""
    model = _model(variant)
    first = model.predict(_inputs(lengths=(T, 4, 1, 0, 2)), batch_size=2, verbose=0)
    assert first["log_pi"].shape == (5, K) and first["log_A"].shape == (5, T, K, K)
    second = model.predict(_inputs(lengths=(11, 3, 0), steps=11), batch_size=2, verbose=0)
    assert second["log_emit"].shape == (3, 11, K)
    third = model.predict(_inputs(lengths=(5, 5, 2), steps=5), batch_size=2, verbose=0)
    assert all(np.isfinite(v).all() for out in (first, second, third) for v in out.values())


# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_fit_two_steps_gives_a_finite_loss(variant):
    model, x, y = _compiled(variant)
    history = model.fit(x, y, batch_size=2, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1 and np.isfinite(history.history["loss"][0])
    assert np.isfinite(model.evaluate(x, y, batch_size=2, verbose=0))


def test_the_labels_must_match_the_mode():
    generative = _model("neural-hmm")
    discriminative = _model("neural-crf-tagger")
    x = _inputs()
    with pytest.raises(ValueError, match="pass y=None"):
        generative.compute_loss(x, _labels(), generative(x))
    with pytest.raises(ValueError, match="needs the state labels"):
        discriminative.compute_loss(x, None, discriminative(x))


def _separable(n=64, steps=8, seed=0):
    """Gaussian observations whose mean is the state index times 3: decodable per step."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n, steps), dtype="int32")
    states[:, 0] = rng.integers(0, K, n)
    for t in range(1, steps):
        stay = rng.random(n) < 0.8
        states[:, t] = np.where(stay, states[:, t - 1], rng.integers(0, K, n))
    obs = (3.0 * states[..., None] + rng.normal(size=(n, steps, D))).astype("float32")
    return {"obs": obs, "mask": np.ones((n, steps), dtype="float32")}, states


def test_the_discriminative_preset_learns_the_gold_path():
    """32 steps on separable data: the CRF NLL falls and the decoded accuracy rises."""
    x, states = _separable()
    model = _model("neural-crf-tagger")
    model.compile(optimizer=keras.optimizers.Adam(1e-2))
    nll_before = -float(np.mean(_np(model.log_likelihood(x, states))))
    acc_before = float(np.mean(_np(model.viterbi(x)[0]) == states))
    model.fit(x, states, batch_size=16, epochs=8, verbose=0, shuffle=False)  # 32 optimizer steps
    nll_after = -float(np.mean(_np(model.log_likelihood(x, states))))
    acc_after = float(np.mean(_np(model.viterbi(x)[0]) == states))
    # Chance is 1/3; 0.9 is far above it and below the measured 0.947 (CPU, seed 1).
    assert nll_after < 0.5 * nll_before, (nll_before, nll_after)
    assert acc_after >= 0.9 and acc_after > acc_before + 0.3, (acc_before, acc_after)


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES)
def test_jit_compile_true_trains_per_encoder_type(encoder_type):
    """The resolved jit flag is asserted, not assumed. On a GPU Keras turns XLA off for a
    model holding a cuDNN-capable LSTM/GRU (supports_jit=False), with a warning; on a CPU
    every encoder compiles."""
    model = _model("autoregressive-neural-hmm", encoder_type=encoder_type)
    x = _inputs()
    model(x)
    cudnn_rnn = encoder_type in ("lstm", "gru") and bool(tf.config.list_physical_devices("GPU"))
    if cudnn_rnn:
        with pytest.warns(UserWarning, match="doesn't support `jit_compile=True`"):
            model.compile(optimizer=keras.optimizers.Adam(1e-2), jit_compile=True)
    else:
        model.compile(optimizer=keras.optimizers.Adam(1e-2), jit_compile=True)
    assert model.jit_compile is (not cudnn_rnn)
    history = model.fit(x, None, batch_size=2, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])


# ---------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_padding_equals_truncation(variant):
    """D-009 at the model level, in float64: each padded sequence scores as its truncation.

    The shortest truncation is 2 steps, not 1: a transformer called on a length-1
    sequence runs a softmax over one key, and Keras emits a UserWarning for that (an
    error under this suite's ``error::UserWarning``). The value is still right; length 1
    inside a padded batch is covered by every other test here.
    """
    lengths = (T, 4, 2)
    with float64_scope():
        model = _model(variant)
        x = _inputs(lengths=lengths)
        x = {"obs": x["obs"].astype("float64"), "mask": x["mask"].astype("float64")}
        y = _y(model, lengths=lengths)
        padded = _np(model.log_likelihood(x, y))
        for b, length in enumerate(lengths):
            single = {"obs": x["obs"][b:b + 1, :length], "mask": x["mask"][b:b + 1, :length]}
            states = None if y is None else y[b:b + 1, :length]
            alone = _np(model.log_likelihood(single, states))[0]
            np.testing.assert_allclose(padded[b], alone, rtol=0, atol=1e-9)


@pytest.mark.parametrize("variant", VARIANTS)
def test_padded_nan_changes_neither_the_loss_nor_the_gradients(variant):
    model = _model(variant)
    x = _inputs()
    y = _y(model)
    dirty = {"obs": x["obs"].copy(), "mask": x["mask"]}
    dirty["obs"][x["mask"] == 0] = np.nan
    model(x)

    def loss_and_grads(inputs):
        with tf.GradientTape() as tape:
            loss = model.compute_loss(inputs, y, model(inputs, training=False))
        return _np(loss), [_np(g) for g in tape.gradient(loss, model.trainable_variables)]

    clean_loss, clean_grads = loss_and_grads(x)
    dirty_loss, dirty_grads = loss_and_grads(dirty)
    assert np.isfinite(dirty_loss) and dirty_loss == clean_loss
    # A leaked NaN makes a gradient NaN, which the isfinite arm convicts. The value arm is
    # not bit-exact because on GPU 1 two CLEAN backward passes of one model already differ
    # (unordered GPU reductions; measured max|delta| 4.8e-07 crf, 9.5e-07 autoregressive,
    # 1.9e-06 segmenter, 2026-10-14); it is bit-exact on the CPU.
    for g_clean, g_dirty in zip(clean_grads, dirty_grads):
        assert np.isfinite(g_dirty).all()
        np.testing.assert_allclose(g_dirty, g_clean, rtol=1e-5, atol=1e-5)


def test_empty_and_single_step_sequences_are_finite():
    model = _model("autoregressive-neural-hmm")
    single = _inputs(lengths=(1, 1, 0))
    assert np.isfinite(_loss(model, single))
    empty = _inputs(lengths=(0, 0))
    assert _loss(model, empty) == 0.0
    path, score = model.viterbi(single)
    assert _np(path)[2].tolist() == [-1] * T and _np(score)[2] == 0.0


def test_the_helpers_agree_with_the_inference_module():
    model = _model("io-hmm")
    x = _inputs()
    out = model(x, training=False)
    args = (out["log_pi"], out["log_A"], out["log_emit"], x["mask"])
    np.testing.assert_array_equal(_np(model.log_likelihood(x)), _np(inference.forward_log_likelihood(*args)))
    path, score = model.viterbi(x)
    ref_path, ref_score = inference.viterbi_decode(*args)
    np.testing.assert_array_equal(_np(path), _np(ref_path))
    np.testing.assert_array_equal(_np(score), _np(ref_score))
    posteriors = _np(model.posteriors(x))
    np.testing.assert_array_equal(posteriors, _np(inference.state_posteriors(*args)))
    valid = x["mask"] > 0
    np.testing.assert_allclose(posteriors.sum(-1)[valid], 1.0, rtol=0, atol=1e-5)
    assert (posteriors[~valid] == 0.0).all() and (_np(path)[~valid] == -1).all()
    # The loss is the log-likelihood the helper reports, per valid step.
    np.testing.assert_allclose(_loss(model, x), -_np(model.log_likelihood(x)).sum() / sum(LENGTHS), rtol=1e-6)


def test_a_categorical_padding_symbol_is_never_embedded():
    """Padded symbols may be -1 or out of range; the embedding and the emission skip them."""
    model = _model("io-hmm", obs_type="categorical")
    x = _inputs("categorical")
    dirty = {"obs": x["obs"].copy(), "mask": x["mask"]}
    dirty["obs"][x["mask"] == 0] = -1
    dirty["obs"][-1, :] = V + 7  # the empty sequence holds an out-of-range symbol
    clean, noisy = model(x), model(dirty)
    valid = x["mask"] > 0
    assert np.array_equal(_np(clean["log_emit"])[valid], _np(noisy["log_emit"])[valid])
    assert np.array_equal(_np(clean["log_A"])[valid], _np(noisy["log_A"])[valid])
    assert _loss(model, x) == _loss(model, dirty)


def test_n_states_one_is_finite():
    model = _model("neural-hmm", n_states=1)
    out = model(_inputs())
    assert np.isfinite(_loss(model, _inputs()))
    assert (_np(out["log_A"]) == 0.0).all()


# ---------------------------------------------------------------------
# precision arms
# ---------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_mixed_float16_forward_is_finite_float32(variant):
    with precision_policy("mixed_float16"):
        model = _model(variant)
        x = _inputs()
        out = model(x, training=False)
        assert all(_np(v).dtype == np.float32 for v in out.values())
        assert model.encoder is None or model.encoder.compute_dtype == "float16"
        assert np.isfinite(_loss(model, x, _y(model)))


def test_the_float32_output_contract_convicts_a_float16_output():
    """The shared subject's expected_compute_dtype='float32' is a live assertion (D-010).

    Forcing ONE output to float16 under mixed_float16 must fail the arm, so the subject's
    declaration is not an exemption in disguise.
    """
    build, make_inputs, kwargs = SUBJECTS["neural_hmm"]
    assert kwargs == {"expected_compute_dtype": "float32"}
    assert_precision_arm(build=build, make_inputs=make_inputs, **kwargs)

    def float16_log_emit(model, inputs, training):
        out = dict(default_call(model, inputs, training))
        out["log_emit"] = keras.ops.cast(out["log_emit"], "float16")
        return out

    with pytest.raises(AssertionError, match="expected every float one to be 'float32'"):
        assert_precision_arm(build=build, make_inputs=make_inputs, call_fn=float16_log_emit, **kwargs)


def test_the_float64_arm():
    """Float64 policy in, float64 log-potentials out: no float32 island survives (D-010)."""
    build, make_inputs, _ = SUBJECTS["neural_hmm"]
    report = assert_float64_arm(build=build, make_inputs=make_inputs)
    assert report["dtypes"] == ["float64"] * 3


# ---------------------------------------------------------------------
# factories and validation
# ---------------------------------------------------------------------


def test_from_variant_honours_an_override_and_leaves_the_table_alone():
    before = {k: dict(v) for k, v in NeuralHMM.MODEL_VARIANTS.items()}
    model = NeuralHMM.from_variant("io-hmm", **_kwargs(encoder_type="gru", topology="banded"))
    assert model.encoder_type == "gru" and model.topology == "banded"
    assert model.encoder.encoder_type == "gru"
    assert NeuralHMM.MODEL_VARIANTS == before


def _config(model):
    """The config without the auto-generated model name, which differs per instance."""
    return {k: v for k, v in model.get_config().items() if k != "name"}


def test_create_neural_hmm_equals_from_variant():
    for variant in VARIANTS:
        assert _config(create_neural_hmm(variant, **_kwargs())) == _config(NeuralHMM.from_variant(variant, **_kwargs()))
    assert _config(create_neural_hmm(**_kwargs())) == _config(NeuralHMM.from_variant("neural-hmm", **_kwargs()))


def test_pretrained_raises_naming_the_variant():
    with pytest.raises(NotImplementedError, match="io-hmm"):
        create_neural_hmm("io-hmm", pretrained=True, **_kwargs())
    with pytest.raises(NotImplementedError, match="neural-hmm"):
        NeuralHMM.from_variant("neural-hmm", pretrained=True)


def test_an_unknown_variant_lists_the_keys():
    with pytest.raises(ValueError, match="neural-crf-tagger"):
        create_neural_hmm("neural-emission-hmm", **_kwargs())


@pytest.mark.parametrize("overrides,match", [
    ({"obs_type": "gaussian", "obs_dim": None}, "positive obs_dim"),
    ({"obs_type": "poisson", "obs_dim": 0}, "positive obs_dim"),
    ({"obs_type": "categorical", "obs_dim": None, "vocab_size": None}, "positive vocab_size"),
    ({"obs_type": "categorical", "obs_dim": 2, "vocab_size": 5}, "obs_dim=None"),
    ({"mode": "discriminative", "emission_context": "autoregressive"}, "emission_context must be 'none'"),
    ({"mode": "supervised"}, "Unknown mode 'supervised'"),
    ({"obs_type": "beta"}, "Unknown obs_type 'beta'"),
    ({"emission_context": "full"}, "Unknown emission_context 'full'"),
    ({"transition": "dynamic"}, "Unknown transition 'dynamic'"),
    ({"encoder_type": "rnn"}, "Unknown encoder_type 'rnn'"),
    ({"topology": "circular"}, "Unknown topology 'circular'"),
])
def test_bad_arguments_raise_naming_the_value(overrides, match):
    kwargs = {"n_states": K, "obs_type": "gaussian", "obs_dim": D, **overrides}
    with pytest.raises(ValueError, match=match):
        NeuralHMM(**kwargs)


def test_build_refuses_a_non_dict_or_wrong_rank_shape():
    with pytest.raises(ValueError, match="expects inputs"):
        _model().build((None, T, D))
    with pytest.raises(ValueError, match="rank 3"):
        _model().build({"obs": (None, T), "mask": (None, T)})
