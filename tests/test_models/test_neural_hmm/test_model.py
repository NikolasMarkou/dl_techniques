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
* the generative right shift (D-022), per generative preset and encoder type with
  randomized weights: ``sum_x p(x) = 1`` by exact enumeration up to length 6, step
  ``t``'s potentials read ``x_{<t}`` only, at length 6 at every step and at length 16 at
  late steps (RED-proven against a no-shift mutant and leaks that start at step 4 and,
  D-029, step 7), and they DO read ``x_{t-1}``, at the same two lengths (RED-proven
  against a shift by two, from the start and, D-029, from step 7; D-027);
* out-of-range inputs (D-022, D-027): a categorical symbol or a gold label outside its
  range at a VALID step drives the log-likelihood to the sentinel's scale and the loss
  to a huge finite value with finite gradients; at a padded step it stays ignored; ``k``
  bad valid steps add exactly ``k`` sentinels and bad padded steps none (RED-proven
  against a doubled and a once-per-sequence penalty, D-029); and a bad symbol leaves
  ``posteriors`` and ``viterbi`` exact for the sequence with that step's emission
  uninformative and token 0 in the encoder's input (brute-force reference);
* factories: ``from_variant`` overrides, ``pretrained=True``, unknown variants,
  ``create_neural_hmm`` == ``from_variant``;
* ignored arguments (D-025): a non-default value for an argument the configuration does
  not read is refused, naming it, its value and the reason, one row per rule branch; the
  same argument at its default is accepted; every preset sets only what it reads;
  ``n_states < 2`` is refused for every preset through every construction path, where the
  one-state transition is measurably constant (D-029); ``from_config`` resets a stale
  unread value in an old config (D-026).

Run on the CPU and with ``CUDA_VISIBLE_DEVICES=1``: the ``lstm`` paths take cuDNN on a GPU.
"""

import inspect
import itertools
import os
import re

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, create_neural_hmm
from dl_techniques.models.probabilistic.neural_hmm import inference
from dl_techniques.models.probabilistic.neural_hmm.emissions import (
    DiscriminativeEmission,
    GenerativeEmission,
)
from dl_techniques.models.probabilistic.neural_hmm.encoders import ENCODER_TYPES, NeuralHMMEncoder
from dl_techniques.models.probabilistic.neural_hmm.transitions import (
    InputDependentTransition,
    StaticTransition,
)
from dl_techniques.utils.dtype_policy import mask_sentinel

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
from .test_inference import brute_force

B, T, D, K, V = 4, 7, 2, 3, 5
#: Valid lengths of the padded batch: full, middle, one step, empty.
LENGTHS = (T, 4, 1, 0)
#: Every preset that takes the shared {"obs", "mask"} inputs. io-hmm also needs an exogenous
#: input, and test_io_hmm.py holds it to the same contracts.
VARIANTS = sorted(set(NeuralHMM.MODEL_VARIANTS) - {"io-hmm"})
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
    "context-transition-hmm": "encoder/",
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


def _kwargs(obs_type="gaussian", variant="neural-hmm", **overrides):
    """Constructor arguments for ``variant``: the data shape, the SMALL widths the
    configuration reads (the model refuses a non-default unread one, D-025) and
    ``overrides``, which are passed as given."""
    shape = {"vocab_size": V} if obs_type == "categorical" else {"obs_dim": D}
    config = {"n_states": K, "obs_type": obs_type, **shape, **overrides}
    unread = NeuralHMM.unread_arguments({**NeuralHMM.MODEL_VARIANTS[variant], **config})
    return {**{k: v for k, v in SMALL.items() if k not in unread}, **config}


def _model(variant="neural-hmm", obs_type="gaussian", seed=1, **overrides):
    keras.utils.set_random_seed(seed)
    return create_neural_hmm(variant, **_kwargs(obs_type, variant, **overrides))


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
    assert sorted(NeuralHMM.MODEL_VARIANTS) == sorted(
        ["neural-hmm", "autoregressive-neural-hmm", "context-transition-hmm", "io-hmm", "left-to-right-segmenter",
         "neural-crf-tagger"]
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
    ("kernel_size", (2, 3), "context-transition-hmm", "gaussian", {}),
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
        kind = overrides.pop("obs_type", obs_type)
        return lambda: create_neural_hmm(variant, **_kwargs(kind, variant, **overrides))

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
    """The context-transition-hmm waiver is a step-1 fact only: once W leaves zero, the encoder learns."""
    model, x, y = _compiled("context-transition-hmm")
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


#: Three complete argument sets that together put EVERY argument off its default, each
#: holding only arguments its configuration reads (no one configuration reads them all:
#: kernel_size is conv-only, num_heads transformer-only, D-025).
ROUND_TRIP_CONFIGS = [
    {
        "n_states": 5, "mode": "generative", "obs_type": "poisson", "obs_dim": 3, "vocab_size": None,
        "emission_context": "autoregressive", "transition": "input_dependent", "topology": "banded",
        "band_width": 2, "self_loop_init": 1.5, "min_scale": 0.01, "hidden_dim": 12, "state_emb_dim": 5,
        "encoder_type": "transformer", "num_layers": 3, "num_heads": 3, "intermediate_size": 20,
        "max_seq_len": 40, "kernel_size": 3, "ffn_type": "glu", "normalization_type": "rms_norm",
        "dropout_rate": 0.1, "attention_dropout_rate": 0.2,
    },
    {
        "n_states": 4, "mode": "generative", "obs_type": "categorical", "obs_dim": None, "vocab_size": 7,
        "emission_context": "autoregressive", "transition": "static", "topology": "left_to_right",
        "band_width": 1, "self_loop_init": 0.5, "min_scale": 1e-3, "hidden_dim": 10, "state_emb_dim": 4,
        "encoder_type": "conv", "num_layers": 3, "num_heads": 4, "intermediate_size": 128,
        "max_seq_len": 512, "kernel_size": 5, "ffn_type": "mlp", "normalization_type": "rms_norm",
        "dropout_rate": 0.2, "attention_dropout_rate": 0.0,
    },
    {
        "n_states": 3, "mode": "discriminative", "obs_type": "gaussian", "obs_dim": 2, "vocab_size": None,
        "emission_context": "none", "transition": "static", "topology": "ergodic",
        "band_width": 1, "self_loop_init": 2.5, "min_scale": 1e-3, "hidden_dim": 6, "state_emb_dim": 16,
        "encoder_type": "gru", "num_layers": 1, "num_heads": 4, "intermediate_size": 128,
        "max_seq_len": 512, "kernel_size": 3, "ffn_type": "mlp", "normalization_type": "layer_norm",
        "dropout_rate": 0.0, "attention_dropout_rate": 0.0,
    },
]


def _defaults():
    return {
        name: p.default for name, p in inspect.signature(NeuralHMM.__init__).parameters.items()
        if p.default is not p.empty
    }


def test_the_round_trip_configs_put_every_argument_off_its_default():
    # input_dim changes the input dict, so test_io_hmm.py round-trips it instead; the
    # structural constraints depend on K, so test_structured_decoding.py does.
    params = set(inspect.signature(NeuralHMM.__init__).parameters) - {
        "self", "kwargs", "input_dim", "allowed_transitions", "allowed_starts", "allowed_ends", "end_potential",
    }
    defaults = _defaults()
    off = set()
    for config in ROUND_TRIP_CONFIGS:
        assert set(config) == params
        off |= {k for k, v in config.items() if k not in defaults or v != defaults[k]}
    assert off == params


@pytest.mark.parametrize("arguments", ROUND_TRIP_CONFIGS, ids=["transformer", "conv", "gru-crf"])
def test_get_config_round_trips_every_argument(arguments):
    """Arguments OFF their defaults, so a dropped key changes the rebuilt config."""
    params = set(arguments)
    model = NeuralHMM(**arguments)
    config = model.get_config()
    assert {k: config[k] for k in params} == arguments
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
    model = _model("context-transition-hmm")
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


#: One preset per way a categorical symbol is read: the emission alone (neural-hmm), the
#: emission and an encoder (context-transition-hmm, autoregressive), the emission and an encoder under the
#: banded topology, whose transition holds forbidden entries (the segmenter, D-029), the
#: embedding alone (the CRF). Every preset, so a new one fails the coverage check below.
CATEGORICAL_READERS = ["neural-hmm", "context-transition-hmm", "autoregressive-neural-hmm", "left-to-right-segmenter",
                       "neural-crf-tagger"]


def test_the_categorical_readers_are_every_preset():
    assert sorted(CATEGORICAL_READERS) == VARIANTS


def _loss_and_grads(model, inputs, y):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(inputs, y, model(inputs, training=False))
    grads = tape.gradient(loss, model.trainable_variables)
    return float(_np(loss)), [None if g is None else _np(g) for g in grads]


@pytest.mark.parametrize("variant", CATEGORICAL_READERS)
def test_a_padded_categorical_symbol_is_ignored(variant):
    """Padded symbols may be -1 or out of range: neither the outputs at valid steps, the
    loss nor the gradients see them (the mask skips them; their emission is a neutral 0)."""
    model = _model(variant, obs_type="categorical")
    x = _inputs("categorical")
    y = _y(model)
    dirty = {"obs": x["obs"].copy(), "mask": x["mask"]}
    dirty["obs"][x["mask"] == 0] = -1
    dirty["obs"][-1, :] = V + 7  # the empty sequence holds an out-of-range symbol
    clean, noisy = model(x), model(dirty)
    valid = x["mask"] > 0
    assert np.array_equal(_np(clean["log_emit"])[valid], _np(noisy["log_emit"])[valid])
    assert np.array_equal(_np(clean["log_A"])[valid], _np(noisy["log_A"])[valid])
    clean_loss, clean_grads = _loss_and_grads(model, x, y)
    dirty_loss, dirty_grads = _loss_and_grads(model, dirty, y)
    assert np.isfinite(dirty_loss) and dirty_loss == clean_loss
    for g_clean, g_dirty in zip(clean_grads, dirty_grads):
        if g_dirty is not None:
            assert np.isfinite(g_dirty).all()
            np.testing.assert_allclose(g_dirty, g_clean, rtol=1e-5, atol=1e-5)


def _assert_loud(model, clean_x, bad_x, clean_y, bad_y):
    """Row 0 (fully valid) holds the bad value: its log-likelihood is at the sentinel's
    scale, every other row is untouched, and the loss is huge with finite gradients."""
    sentinel = mask_sentinel("float32")
    clean = _np(model.log_likelihood(clean_x, clean_y))
    bad = _np(model.log_likelihood(bad_x, bad_y))
    assert bad[0] <= 0.5 * sentinel, f"log-likelihood with the bad value: {bad[0]} (clean {clean[0]})"
    np.testing.assert_array_equal(bad[1:], clean[1:])
    loss, grads = _loss_and_grads(model, bad_x, bad_y)
    assert np.isfinite(loss) and loss > 1e6, loss
    assert all(g is None or np.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("bad_symbol", [-1, V, V + 7])
@pytest.mark.parametrize("variant", CATEGORICAL_READERS)
def test_an_out_of_range_symbol_at_a_valid_step_is_loud(variant, bad_symbol):
    """D-022/D-027: it scores a neutral 0 in the generative emission and embeds as token 0
    in the CRF; XLA drops in-graph asserts, so the model adds the sentinel per bad step to
    the sequence log-likelihood, outside the forward pass."""
    model = _model(variant, obs_type="categorical")
    x = _inputs("categorical")
    y = _y(model)
    bad = {"obs": x["obs"].copy(), "mask": x["mask"]}
    bad["obs"][0, 3] = bad_symbol
    _assert_loud(model, x, bad, y, y)


@pytest.mark.parametrize("bad_label", [-1, K, K + 4])
def test_an_out_of_range_gold_label_at_a_valid_step_is_loud(bad_label):
    """D-022: its start, emission and transition terms used to drop out of the gold score."""
    model = _model("neural-crf-tagger")
    x = _inputs()
    y = _labels()
    bad = y.copy()
    bad[0, 2] = bad_label
    _assert_loud(model, x, x, y, bad)
    padded = y.copy()
    padded[x["mask"] == 0] = K + 4  # at padded steps any label stays ignored
    assert _loss(model, x, padded) == _loss(model, x, y)


def _out_of_range_reference(model, bad, bad_steps):
    """Brute-force posteriors and Viterbi of ``bad`` with each bad step UNINFORMATIVE.

    The potentials come from the same sequence with each bad symbol replaced by token 0,
    which is what the encoder reads for it (the embedding maps a bad symbol to 0). In
    generative mode that step's emission row is then set to 0 in every state, i.e. the
    observation is missing; in the CRF the step stays the token-0 potentials it is.
    """
    tokens = {"obs": bad["obs"].copy(), "mask": bad["mask"]}
    for row, step in bad_steps:
        tokens["obs"][row, step] = 0
    out = {k: _np(v).astype("float64") for k, v in model(tokens, training=False).items()}
    if model.mode == "generative":
        for row, step in bad_steps:
            out["log_emit"][row, step] = 0.0
    lengths = bad["mask"].sum(axis=1).astype(int)
    return brute_force(out["log_pi"], out["log_A"], out["log_emit"], lengths)


@pytest.mark.parametrize("bad_symbol", [-1, V, V + 7])
@pytest.mark.parametrize("variant", CATEGORICAL_READERS)
def test_an_out_of_range_symbol_leaves_viterbi_and_posteriors_exact(variant, bad_symbol):
    """D-027: the bad symbol's penalty lives outside the potentials, so the decoding
    helpers stay exact distributions over the in-range evidence.

    With the sentinel inside log_emit (D-022's first fix), one bad step made every
    posterior of its row 1.0 (rows summed to K) and Viterbi decode garbage.
    """
    model = _model(variant, obs_type="categorical")
    x = _inputs("categorical")
    bad_steps = [(0, 3), (1, 1)]  # a full-length row and a padded one, both at valid steps
    bad = {"obs": x["obs"].copy(), "mask": x["mask"]}
    for row, step in bad_steps:
        bad["obs"][row, step] = bad_symbol
    _, best_path, best_score, reference = _out_of_range_reference(model, bad, bad_steps)

    posteriors = _np(model.posteriors(bad)).astype("float64")
    valid = x["mask"] > 0
    np.testing.assert_allclose(posteriors.sum(axis=-1)[valid], 1.0, rtol=0, atol=1e-5)
    assert (posteriors[~valid] == 0.0).all()
    np.testing.assert_allclose(posteriors, reference, rtol=0, atol=1e-5)

    path, score = model.viterbi(bad)
    path, score = _np(path), _np(score).astype("float64")
    np.testing.assert_array_equal(path, best_path)
    assert np.isfinite(score).all()
    np.testing.assert_allclose(score, best_score, rtol=0, atol=1e-4)


#: Valid lengths of the penalty-count batch, the bad VALID steps of each row (the count the
#: log-likelihood must carry) and the bad PADDED steps (which must add nothing).
PENALTY_LENGTHS = (T, T, 5, 3)
PENALTY_BAD_VALID = ((0, 3, 6), (2,), (1, 4), ())
PENALTY_BAD_PADDED = ((), (), (5, 6), (3, 4, 5, 6))


@pytest.mark.parametrize("variant", CATEGORICAL_READERS)
def test_the_out_of_range_penalty_counts_the_bad_steps(variant):
    """D-029: k bad symbols at valid steps add exactly k sentinels, bad padded steps none.

    The reference is the same potentials' likelihood without any penalty, through the
    inference module: ``forward_log_likelihood`` (generative) or ``path_log_score`` minus
    it (the CRF). Rows hold 3, 1, 2 and 0 bad valid steps, so a doubled penalty (6, 2, 4)
    and one sentinel per sequence (1, 1, 1) both miss; rows 2 and 3 also hold bad padded
    steps. The bad valid steps' generative emission is 0 in every state (the neutral
    potentials the reference reads).
    """
    model = _model(variant, obs_type="categorical")
    x = _inputs("categorical", lengths=PENALTY_LENGTHS)
    y = _y(model, lengths=PENALTY_LENGTHS)
    bad = {"obs": x["obs"].copy(), "mask": x["mask"]}
    symbols = itertools.cycle([-1, V, V + 7])
    for steps_per_row in (PENALTY_BAD_VALID, PENALTY_BAD_PADDED):
        for row, steps in enumerate(steps_per_row):
            for step in steps:
                bad["obs"][row, step] = next(symbols)
    assert all(bad["mask"][row, step] > 0 for row, steps in enumerate(PENALTY_BAD_VALID) for step in steps)
    assert all(bad["mask"][row, step] == 0 for row, steps in enumerate(PENALTY_BAD_PADDED) for step in steps)

    out = model(bad, training=False)
    args = (out["log_pi"], out["log_A"], out["log_emit"], bad["mask"])
    neutral = _np(inference.forward_log_likelihood(*args)).astype("float64")
    if y is not None:
        neutral = _np(inference.path_log_score(*args[:3], y, bad["mask"])).astype("float64") - neutral
    else:
        log_emit = _np(out["log_emit"])
        assert all((log_emit[row, step] == 0.0).all()
                   for row, steps in enumerate(PENALTY_BAD_VALID) for step in steps)
    got = _np(model.log_likelihood(bad, y)).astype("float64")
    sentinel = float(mask_sentinel("float32"))
    expected = np.array([len(steps) for steps in PENALTY_BAD_VALID], dtype="float64")
    counts = np.rint((got - neutral) / sentinel)
    assert counts.tolist() == expected.tolist(), f"sentinels per row: {counts} (expected {expected})"
    # float32 resolution at 3e9 is 256; the in-range part must be untouched to that grain.
    np.testing.assert_allclose(got - neutral, expected * sentinel, rtol=0, atol=1024.0)
    np.testing.assert_allclose(got[expected == 0], neutral[expected == 0], rtol=0, atol=1e-4)


@pytest.mark.parametrize("n_states", [1, 0])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fewer_than_two_states_is_refused_for_every_preset(variant, n_states):
    """D-029: an HMM needs two states. The message names n_states and why; every
    construction path refuses (the constructor, ``from_variant``, ``create_neural_hmm``),
    and K = 2 builds and round-trips."""
    kwargs = _shape_kwargs({"n_states": n_states})
    for build in (lambda: NeuralHMM(**{**NeuralHMM.MODEL_VARIANTS[variant], **kwargs}),
                  lambda: NeuralHMM.from_variant(variant, **kwargs),
                  lambda: create_neural_hmm(variant, **kwargs)):
        with pytest.raises(ValueError, match=re.escape(f"n_states={n_states} is refused")) as info:
            build()
        message = str(info.value)
        assert "at least two states" in message and "never train" in message and "n_states >= 2" in message
    model = create_neural_hmm(variant, **_shape_kwargs({"n_states": 2}))
    assert model.n_states == 2
    assert NeuralHMM.from_config(model.get_config()).get_config() == model.get_config()


@pytest.mark.parametrize("transition_class", [StaticTransition, InputDependentTransition])
def test_at_one_state_the_transition_is_constant(transition_class):
    """D-029: why ``n_states < 2`` is refused in every configuration. At one state every
    topology mask is all-true and the 1x1 transition normalizes to log 1 = 0: neither the
    logits nor any knob (topology, band_width, self_loop_init) changes an output, and
    ``pi_logits`` and ``transition_logits`` get exactly-zero gradients; at two states each
    knob changes the output. The layer itself still accepts one state; the model refuses."""
    context = np.random.default_rng(0).normal(size=(2, 5, 3)).astype("float32")

    def outputs(n_states, **kwargs):
        layer = transition_class(n_states, **kwargs)
        layer.build(context.shape)
        if transition_class is InputDependentTransition:  # a live context projection
            kernel = layer.context_projection.kernel
            kernel.assign(np.random.default_rng(1).normal(size=kernel.shape).astype("float32"))
        with tf.GradientTape() as tape:
            log_pi, log_A = layer(context)
            total = keras.ops.sum(log_pi) + keras.ops.sum(log_A * np.arange(1.0, 6.0)[None, :, None, None])
        grad_a, grad_pi = tape.gradient(total, [layer.transition_logits, layer.pi_logits])
        return _np(log_pi), _np(log_A), _np(grad_a), _np(grad_pi)

    variants = [{"topology": "banded", "band_width": 0}, {"topology": "banded", "band_width": 3},
                {"topology": "left_to_right"}, {"self_loop_init": 7.0}]
    single = outputs(1)
    assert all((part == 0.0).all() for part in single), single
    for kwargs in variants:
        for got, want in zip(outputs(1, **kwargs), single):
            np.testing.assert_array_equal(got, want, err_msg=str(kwargs))
    two = outputs(2)
    for kwargs in variants:
        assert not np.array_equal(outputs(2, **kwargs)[1], two[1]), kwargs


# ---------------------------------------------------------------------
# the generative right-shift (D-022)
# ---------------------------------------------------------------------

#: Every generative preset with every encoder type it can take. The one-step right shift
#: of the encoder input is what makes each of them a valid likelihood; removing it once
#: left all 739 package and trainer tests green while sum_x p(x) reached 1.174
#: (review-iter-1, concern 1). neural-hmm builds no encoder and is the control.
GENERATIVE_CASES = (
    [("neural-hmm", None)]
    + [("autoregressive-neural-hmm", encoder) for encoder in ENCODER_TYPES]
    + [("context-transition-hmm", encoder) for encoder in ENCODER_TYPES]
    + [("left-to-right-segmenter", "transformer")]
)
GENERATIVE_IDS = [f"{variant}-{encoder or 'none'}" for variant, encoder in GENERATIVE_CASES]
#: Vocabulary and longest length of the exact enumeration: 3 + 9 + ... + 729 sequences.
#: Length 6, not 4 (D-027): at 4, a leak starting at step 4 passed every autoregressive
#: and segmenter case, since neither the enumeration nor the per-step arm reached it.
ENUM_V, ENUM_T = 3, 6
#: |sum_x p(x) - 1| bound. The measured float32 error at lengths up to 6 is at most
#: 4.0e-7 (GPU 1); the no-shift mutant misses by 7.5e-3 (context-transition-hmm/conv) to 0.17
#: (autoregressive/lstm) at lengths up to 4, the step-4 leak by 5e-4 to 0.99 at 5 and 6.
NORMALIZATION_ATOL = 1e-5
#: The long arm of the perturbation guards (D-029): a sequence of LONG_T steps, perturbed
#: at late split points only. The enumeration stops at 6 steps and the lstm/gru encoders
#: have an unbounded receptive field, so a leak that starts at step 6 or 7 passed every
#: guard (review pass 3, concern 3). LONG_T is SMALL's max_seq_len (the transformer's
#: position table).
LONG_T = 16
LONG_SPLITS = (6, 9, 11, 15)


def _every_sequence(length, steps=ENUM_T, seed=0):
    """All ENUM_V**length sequences of one valid length, right-padded with random symbols."""
    prefix = np.array(list(itertools.product(range(ENUM_V), repeat=length)), dtype="int32")
    pad = np.random.default_rng(seed).integers(0, ENUM_V, (len(prefix), steps - length)).astype("int32")
    return {"obs": np.concatenate([prefix, pad], axis=1), "mask": _mask([length] * len(prefix), steps)}


def _randomized(variant, encoder_type, obs_type, seed=3):
    """A built preset with every trainable weight redrawn from N(0, 0.7).

    The default initialization hides an input dependence: context-transition-hmm's context projection
    starts at zero, so its transitions would not read the encoder at all.
    """
    overrides = {} if encoder_type is None else {"encoder_type": encoder_type}
    if obs_type == "categorical":
        overrides["vocab_size"] = ENUM_V
        model = _model(variant, obs_type, **overrides)
        model(_every_sequence(ENUM_T), training=False)
    else:
        model = _model(variant, obs_type, **overrides)
        model(_inputs(obs_type), training=False)
    rng = np.random.default_rng(seed)
    for weight in model.trainable_weights:
        weight.assign(rng.normal(0.0, 0.7, size=weight.shape).astype(weight.dtype))
    return model


def _logsumexp(values, axis):
    peak = values.max(axis=axis, keepdims=True)
    return np.squeeze(peak, axis) + np.log(np.exp(values - peak).sum(axis=axis))


@pytest.mark.parametrize("variant,encoder_type", GENERATIVE_CASES, ids=GENERATIVE_IDS)
def test_the_generative_model_is_a_normalized_distribution(variant, encoder_type):
    """sum over every sequence x of each valid length L of p(x) is 1 (D-022).

    Any leak of x_t into its own emission or into the transition into step t breaks this,
    whichever encoder carries it. Padded steps hold random symbols, which must not count.
    """
    model = _randomized(variant, encoder_type, "categorical")
    misses = {}
    for length in range(1, ENUM_T + 1):
        log_p = _np(model.log_likelihood(_every_sequence(length))).astype("float64")
        misses[length] = float(np.exp(log_p).sum() - 1.0)
    assert all(abs(miss) <= NORMALIZATION_ATOL for miss in misses.values()), f"sum_x p(x) - 1 per L: {misses}"


@pytest.mark.parametrize("variant,encoder_type", GENERATIVE_CASES, ids=GENERATIVE_IDS)
def test_the_generative_potentials_read_only_the_past(variant, encoder_type):
    """Step t's transition and emission parameters read x_{<t} only (D-022).

    Two arms. (1) Perturbing obs[:, s:] leaves log_emit[:, :s] and log_A[:, :s + 1]
    bit-identical: log_A[:, s] is the transition INTO s, built before x_s is seen. (2) An
    emission's parameters must not read the symbol they score. Arm 1 cannot see that leak:
    log_emit[:, s] depends on x_s legitimately. So for sequences that share x_{<s} and
    differ only at x_s, every state's emission must sum to 1 over the symbols at step s.
    Each arm runs at 6 steps, every split point, and at LONG_T steps, the late split
    points LONG_SPLITS (D-029).
    """
    model = _randomized(variant, encoder_type, "gaussian")
    rng = np.random.default_rng(5)
    for steps, splits in ((6, range(6)), (LONG_T, LONG_SPLITS)):
        obs = rng.normal(size=(2, steps, D)).astype("float32")
        mask = np.ones((2, steps), dtype="float32")
        clean = model({"obs": obs, "mask": mask}, training=False)
        for s in splits:
            moved = obs.copy()
            moved[:, s:] += 5.0 * rng.normal(size=moved[:, s:].shape).astype("float32")
            out = model({"obs": moved, "mask": mask}, training=False)
            where = f"T={steps}, s={s}"
            assert np.array_equal(_np(out["log_emit"])[:, :s], _np(clean["log_emit"])[:, :s]), f"log_emit, {where}"
            assert np.array_equal(_np(out["log_A"])[:, :s + 1], _np(clean["log_A"])[:, :s + 1]), f"log_A, {where}"
            assert not np.array_equal(_np(out["log_emit"])[:, s], _np(clean["log_emit"])[:, s]), where

    model = _randomized(variant, encoder_type, "categorical")
    # Arm 2 runs over all ENUM_T = 6 steps, so a leak that starts late is seen (D-027),
    # and at the late steps of a LONG_T sequence (D-029).
    for steps, splits in ((ENUM_T, tuple(range(ENUM_T))), (LONG_T, LONG_SPLITS)):
        base = np.random.default_rng(6).integers(0, ENUM_V, steps).astype("int32")
        symbols = np.repeat(base[None], len(splits) * ENUM_V, axis=0)
        for i, s in enumerate(splits):
            symbols[i * ENUM_V:(i + 1) * ENUM_V, s] = np.arange(ENUM_V)
        out = model({"obs": symbols, "mask": np.ones(symbols.shape, "float32")}, training=False)
        log_emit = _np(out["log_emit"]).astype("float64").reshape(len(splits), ENUM_V, steps, K)
        per_step = np.stack([_logsumexp(log_emit[i, :, s, :], axis=0) for i, s in enumerate(splits)])
        assert np.abs(per_step).max() <= NORMALIZATION_ATOL, (
            f"T={steps}, splits {splits}: log sum_v p(x_s = v | z_s = k): {per_step}"
        )


#: The generative cases with an input-dependent potential, and which potential reads
#: x_{t-1}: the emission (autoregressive, segmenter) or the transition INTO t (context-transition-hmm).
LIVE_CASES = [(variant, encoder) for variant, encoder in GENERATIVE_CASES if variant != "neural-hmm"]
LIVE_IDS = [f"{variant}-{encoder}" for variant, encoder in LIVE_CASES]


@pytest.mark.parametrize("variant,encoder_type", LIVE_CASES, ids=LIVE_IDS)
def test_the_generative_potentials_read_the_previous_step(variant, encoder_type):
    """The shift is exactly ONE step (D-027): perturbing x_{s-1} alone must change
    log_emit[:, s] (an autoregressive emission) or log_A[:, s] (context-transition-hmm), in every row.

    The normalization and read-only-the-past arms cannot see a shift by two: h_t built
    from x_{<t-1} is still a valid likelihood, but the model no longer conditions on the
    previous observation (an all-green suite measured it, review pass 2, concern 3).
    """
    model = _randomized(variant, encoder_type, "gaussian")
    key = "log_A" if variant == "context-transition-hmm" else "log_emit"
    rng = np.random.default_rng(7)
    # Every step of a 6-step sequence, and the late steps of a LONG_T one (D-029).
    for steps, splits in ((6, range(1, 6)), (LONG_T, LONG_SPLITS)):
        obs = rng.normal(size=(2, steps, D)).astype("float32")
        mask = np.ones((2, steps), dtype="float32")
        clean = _np(model({"obs": obs, "mask": mask}, training=False)[key])
        for s in splits:
            moved = obs.copy()
            moved[:, s - 1] += 3.0 * rng.normal(size=moved[:, s - 1].shape).astype("float32")
            out = _np(model({"obs": moved, "mask": mask}, training=False)[key])
            change = np.abs(out[:, s] - clean[:, s]).reshape(2, -1).max(axis=1)
            assert (change > 1e-4).all(), (
                f"T={steps}: {key}[:, {s}] did not read x_{s - 1}: max change per row {change}"
            )


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
    model = NeuralHMM.from_variant("context-transition-hmm", **_kwargs(variant="context-transition-hmm", encoder_type="gru", topology="banded"))
    assert model.encoder_type == "gru" and model.topology == "banded"
    assert model.encoder.encoder_type == "gru"
    assert NeuralHMM.MODEL_VARIANTS == before


def _config(model):
    """The config without the auto-generated model name, which differs per instance."""
    return {k: v for k, v in model.get_config().items() if k != "name"}


def test_create_neural_hmm_equals_from_variant():
    for variant in VARIANTS:
        kwargs = _kwargs(variant=variant)
        assert _config(create_neural_hmm(variant, **kwargs)) == _config(NeuralHMM.from_variant(variant, **kwargs))
    assert _config(create_neural_hmm(**_kwargs())) == _config(NeuralHMM.from_variant("neural-hmm", **_kwargs()))


def test_pretrained_raises_naming_the_variant():
    with pytest.raises(NotImplementedError, match="context-transition-hmm"):
        create_neural_hmm("context-transition-hmm", pretrained=True, **_kwargs(variant="context-transition-hmm"))
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


#: (preset, extra overrides, knob, non-default value, words of the reason): one row per
#: rule branch of D-025, written out by hand from the wiring in model.py, encoders.py,
#: transitions.py and emissions.py (not generated from the rule under test).
_NO_ENCODER = "no encoder is built"
INAPPLICABLE = [
    # no encoder at all: every encoder argument
    ("neural-hmm", {}, "encoder_type", "gru", _NO_ENCODER),
    ("neural-hmm", {}, "num_layers", 3, _NO_ENCODER),
    ("neural-hmm", {}, "num_heads", 2, _NO_ENCODER),
    ("neural-hmm", {}, "intermediate_size", 12, _NO_ENCODER),
    ("neural-hmm", {}, "max_seq_len", 16, _NO_ENCODER),
    ("neural-hmm", {}, "kernel_size", 5, _NO_ENCODER),
    ("neural-hmm", {}, "ffn_type", "glu", _NO_ENCODER),
    ("neural-hmm", {}, "normalization_type", "rms_norm", _NO_ENCODER),
    ("neural-hmm", {}, "dropout_rate", 0.1, _NO_ENCODER),
    ("neural-hmm", {}, "attention_dropout_rate", 0.1, _NO_ENCODER),
    # an lstm / gru encoder: the transformer-, mlp- and conv-only arguments
    ("autoregressive-neural-hmm", {}, "num_heads", 2, "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {}, "max_seq_len", 16, "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {}, "ffn_type", "glu", "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {}, "attention_dropout_rate", 0.1, "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {}, "intermediate_size", 12, "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {}, "kernel_size", 5, "encoder_type='lstm'"),
    ("autoregressive-neural-hmm", {"encoder_type": "gru"}, "num_heads", 2, "encoder_type='gru'"),
    ("autoregressive-neural-hmm", {"encoder_type": "gru"}, "kernel_size", 5, "encoder_type='gru'"),
    # an mlp encoder reads intermediate_size, nothing transformer- or conv-only
    ("autoregressive-neural-hmm", {"encoder_type": "mlp"}, "num_heads", 2, "encoder_type='mlp'"),
    ("autoregressive-neural-hmm", {"encoder_type": "mlp"}, "max_seq_len", 16, "encoder_type='mlp'"),
    ("autoregressive-neural-hmm", {"encoder_type": "mlp"}, "ffn_type", "glu", "encoder_type='mlp'"),
    ("autoregressive-neural-hmm", {"encoder_type": "mlp"}, "attention_dropout_rate", 0.1, "encoder_type='mlp'"),
    ("autoregressive-neural-hmm", {"encoder_type": "mlp"}, "kernel_size", 5, "encoder_type='mlp'"),
    # a conv encoder (context-transition-hmm)
    ("context-transition-hmm", {}, "num_heads", 2, "encoder_type='conv'"),
    ("context-transition-hmm", {}, "intermediate_size", 12, "encoder_type='conv'"),
    ("context-transition-hmm", {}, "max_seq_len", 16, "encoder_type='conv'"),
    ("context-transition-hmm", {}, "ffn_type", "glu", "encoder_type='conv'"),
    ("context-transition-hmm", {}, "attention_dropout_rate", 0.1, "encoder_type='conv'"),
    # a transformer encoder, generative and discriminative
    ("left-to-right-segmenter", {}, "kernel_size", 5, "encoder_type='transformer'"),
    ("neural-crf-tagger", {}, "kernel_size", 5, "encoder_type='transformer'"),
    # band_width off the banded topology
    ("neural-hmm", {}, "band_width", 2, "topology='ergodic'"),
    ("neural-hmm", {"topology": "left_to_right"}, "band_width", 2, "topology='left_to_right'"),
    # the emission: no density floor for symbols or CRF potentials, no state embeddings in a CRF
    ("neural-hmm", {"obs_type": "categorical"}, "min_scale", 0.5, "categorical emission"),
    ("neural-crf-tagger", {}, "min_scale", 0.5, "mode='discriminative'"),
    ("neural-crf-tagger", {"obs_type": "categorical"}, "min_scale", 0.5, "mode='discriminative'"),
    ("neural-crf-tagger", {}, "state_emb_dim", 9, "mode='discriminative'"),
]
INAPPLICABLE_IDS = [
    f"{variant}-{'-'.join(f'{k}={v}' for k, v in extra.items()) or 'preset'}-{knob}"
    for variant, extra, knob, _, _ in INAPPLICABLE
]


def _shape_kwargs(extra):
    obs_type = extra.get("obs_type", "gaussian")
    shape = {"vocab_size": V} if obs_type == "categorical" else {"obs_dim": D}
    return {"n_states": K, "obs_type": obs_type, **shape, **extra}


@pytest.mark.parametrize("variant,extra,knob,value,reason", INAPPLICABLE, ids=INAPPLICABLE_IDS)
def test_an_inapplicable_knob_with_a_non_default_value_is_refused(variant, extra, knob, value, reason):
    """D-025: the error names the knob, its value and why nothing reads it."""
    assert value != _defaults()[knob]
    with pytest.raises(ValueError, match=re.escape(f"{knob}={value!r}")) as info:
        create_neural_hmm(variant, **_shape_kwargs(extra), **{knob: value})
    assert reason in str(info.value) and "would be ignored" in str(info.value), str(info.value)
    assert knob in NeuralHMM.unread_arguments({**NeuralHMM.MODEL_VARIANTS[variant], **_shape_kwargs(extra)})


@pytest.mark.parametrize("variant,extra,knob,value,reason", INAPPLICABLE, ids=INAPPLICABLE_IDS)
def test_the_same_inapplicable_knob_at_its_default_is_accepted(variant, extra, knob, value, reason):
    """from_config passes every argument back, so a default-valued unread knob must pass."""
    default = _defaults()[knob]
    model = create_neural_hmm(variant, **_shape_kwargs(extra), **{knob: default})
    assert model.get_config()[knob] == default


@pytest.mark.parametrize("obs_type", ["gaussian", "poisson", "categorical"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_preset_constructs_and_sets_only_what_it_reads(variant, obs_type):
    """No preset carries a value for an argument its own configuration ignores."""
    preset = NeuralHMM.MODEL_VARIANTS[variant]
    unread = NeuralHMM.unread_arguments({**preset, **_shape_kwargs({"obs_type": obs_type})})
    assert not set(preset) & set(unread), sorted(set(preset) & set(unread))
    model = create_neural_hmm(variant, **_shape_kwargs({"obs_type": obs_type}))
    assert NeuralHMM.from_config(model.get_config()).get_config() == model.get_config()


def test_from_config_resets_a_stale_unread_value_to_its_default():
    """An archive written before D-025 (the trainer passed max_seq_len=50 to every model)
    still loads, into the model it was: the stale value never reached a layer."""
    reference = create_neural_hmm("context-transition-hmm", **_shape_kwargs({}))
    stale = {**reference.get_config(), "max_seq_len": 50, "num_heads": 2}
    with pytest.raises(ValueError, match="would be ignored"):
        NeuralHMM(**{k: v for k, v in stale.items() if k != "name"})
    restored = NeuralHMM.from_config(stale)
    assert restored.max_seq_len == 512 and restored.num_heads == 4
    assert _config(restored) == _config(reference)


def test_build_refuses_a_non_dict_or_wrong_rank_shape():
    with pytest.raises(ValueError, match="expects inputs"):
        _model().build((None, T, D))
    with pytest.raises(ValueError, match="rank 3"):
        _model().build({"obs": (None, T), "mask": (None, T)})
