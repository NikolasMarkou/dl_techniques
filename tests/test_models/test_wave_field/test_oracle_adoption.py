"""
Oracle adoption for ``models/wave_field`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

WHAT THIS FILE DELIBERATELY DOES **NOT** DO
---------------------------------------------
It ships **no causality test**. ``test_model.py`` already owns that claim, and
owns it better than an audit sweep would: ``TestWaveFieldLLMCausalityRatioSweep``
pins six ``field_size / max_seq_len`` ratios as MEASURED (0.50 / 0.75 / 1.50
leak; 1.00 / 2.00 / 4.00 clean), with a device-dependence note, a bound derived
from the measurement gap rather than from taste, and -- the part a second copy
would lose -- the fact that **ratio 1.50 leaks while ratio 1.00 does not**, which
falsifies the monotone-in-ratio assumption that an audit-written causality guard
would almost certainly encode. A weaker second copy that asserted "the default
config is causal" would pass on every one of those six rows and quietly become
the guard people read. Not shipped.

WHAT IT ADDS OVER THE EXISTING GRADIENT TEST
----------------------------------------------
``TestWaveFieldLLMGradient::test_gradient_flow`` asserts only ``g is not None``
per variable. That accepts a weight whose gradient is identically ``0.0`` --
built, saved, on the graph, and not learning. The oracle's assertion is the
strictly stronger one (``None`` OR exactly-zero convicts), and it is taken AFTER
a real Adam step rather than at initialisation.

Measured 2026-08-21, one Adam step, ramp loss, at
``vocab=64 / embed_dim=32 / depth=2 / heads=2 / max_seq_len=32 / field_size=64``:

=========================  ==========  ======
arm                        weights     dead
=========================  ==========  ======
WaveFieldLLM (tied)        42          0
=========================  ==========  ======

``field_size=64`` is ratio **2.0** against ``max_seq_len=32`` -- the shipped
default and a row ``test_model.py`` measured CLEAN. That is deliberate: a
gradient reading taken on a leaky config would be measuring a different model
from the one the package ships.

Both dropout rates are pinned to ``0.0`` and every build is seeded: batch A had
an arm flaky 1 run in 4 and batch B one flaky 2 in 5, both from an unpinned
stochastic rate or an unseeded build.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.wave_field.model import WaveFieldLLM

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

VOCAB = 64
EMBED_DIM = 32
MAX_SEQ_LEN = 32
#: Ratio 2.0 -- the shipped default, and a row ``test_model.py`` measured CLEAN.
FIELD_SIZE = 64
SEQ_LEN = 16
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 42


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed: this is the ramp
    D-059 installed, and a second copy is a second thing that can drift back
    into a symmetric loss."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _ids(batch: int = 2, seq_len: int = SEQ_LEN, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, VOCAB, (batch, seq_len)).astype("int32")


def _inputs(**o) -> Dict[str, np.ndarray]:
    return {"input_ids": _ids(**o)}


def _wave_field(**o) -> WaveFieldLLM:
    kwargs: Dict[str, Any] = dict(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, depth=2, num_heads=2,
        max_seq_len=MAX_SEQ_LEN, field_size=FIELD_SIZE,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    kwargs.update(o)
    return WaveFieldLLM(**kwargs)


def _built(build_fn=_wave_field, seed: int = BUILD_SEED) -> WaveFieldLLM:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_inputs(batch=1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = ramp_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestWaveFieldGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero stochastic rate is live: {stochastic}"

    def test_the_measured_config_sits_on_a_clean_causality_ratio(self):
        """The premise of every number in the module docstring.

        ``test_model.py``'s sweep measured ratio 2.0 CLEAN and ratio 1.5 LEAKY.
        If this file's config drifted onto a leaky ratio its gradient readings
        would describe a different model from the shipped one, so the ratio is
        asserted rather than assumed. The LEAK itself is not re-measured here
        -- that claim belongs to ``test_model.py`` and is not duplicated.
        """
        model = _built()
        assert model.field_size / model.max_seq_len == 2.0

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_untied_head_is_also_fully_live(self):
        """``tie_word_embeddings=False`` adds an independent output matrix.

        A tied model cannot tell you whether the untied head is on the graph:
        with tying, the "head" IS the embedding table and is reached through
        the input path regardless.
        """
        model = _built(lambda: _wave_field(tie_word_embeddings=False))
        x = _inputs()
        _one_adam_step(model, x)
        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)
        assert len(report) > GF_WEIGHTS, (
            f"untying added no weight tensor ({len(report)} vs {GF_WEIGHTS}); "
            f"the kwarg is not reaching the head"
        )

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _inputs(), loss_fn=ramp_loss)


class TestWaveFieldKnobSensitivity:

    def test_depth_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _wave_field(depth=d)))
            for d in (1, 2, 4)
        }
        assert_structural_knob_changes_weights(builders, knob="depth")

    def test_embed_dim_changes_the_parameterisation(self):
        builders = {
            e: (lambda e=e: _built(
                lambda: _wave_field(embed_dim=e, num_heads=2)))
            for e in (16, 32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="embed_dim")

    def test_field_size_reaches_the_forward_pass(self):
        """A VALUE knob, and the one this architecture is ABOUT.

        ``field_size`` is the wave-field grid resolution. It changes no weight
        shape at all -- both settings hold bit-identical weights under a fixed
        seed, which the oracle asserts before comparing anything -- so a
        shape-only sweep is blind to it and a build that never routed it into
        ``WaveFieldAttention`` would pass one. Both values swept here are ratios
        ``test_model.py`` measured CLEAN (2.0 and 4.0), so this test says
        nothing about causality and cannot be mistaken for a causality guard.
        """
        x = _inputs()
        builders = {
            f: (lambda f=f: _wave_field(field_size=f)) for f in (64, 128)
        }
        deltas = assert_value_knob_changes_output(
            builders, x, knob="field_size", extract=lambda o: o["logits"])
        assert all(d > 1e-4 for d in deltas.values()), deltas

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depth")

    def test_the_value_knob_assertion_can_fail(self):
        x = _inputs()
        builders = {k: (lambda: _wave_field(field_size=64)) for k in ("a", "b")}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, x, knob="field_size", extract=lambda o: o["logits"])


class TestWaveFieldSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _inputs()

        def contract(out):
            assert isinstance(out, dict), (
                f"WaveFieldLLM.call returns a dict, got {type(out)}")
            assert set(out) == {"logits", "last_hidden_state"}, (
                f"unexpected key set {sorted(out)}")
            logits, hidden = out["logits"], out["last_hidden_state"]
            assert tuple(logits.shape) == (2, SEQ_LEN, VOCAB), (
                f"expected {(2, SEQ_LEN, VOCAB)}, got {tuple(logits.shape)}")
            assert tuple(hidden.shape) == (2, SEQ_LEN, EMBED_DIM), (
                f"expected {(2, SEQ_LEN, EMBED_DIM)}, got {tuple(hidden.shape)}")
            assert_finite(logits)
            assert_finite(hidden)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_logits_are_logits_not_probabilities(self):
        """A softmax slipped onto the head would leave every shape identical
        and every finiteness check green."""
        model = _built()
        logits = np.asarray(keras.ops.convert_to_numpy(
            model(_inputs(), training=False)["logits"]))
        sums = logits.sum(axis=-1)
        assert not np.allclose(sums, 1.0, atol=1e-3), (
            "the head output sums to 1 over the vocabulary -- it is a "
            "distribution, but the contract (and every CLM loss in this repo, "
            "from_logits=True) says logits")

    def test_pretrained_true_is_refused_rather_than_silently_ignored(self):
        with pytest.raises(NotImplementedError):
            WaveFieldLLM.from_variant("tiny", pretrained=True)
