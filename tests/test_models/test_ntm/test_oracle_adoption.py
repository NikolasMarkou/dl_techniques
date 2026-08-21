"""
Oracle adoption for ``models/ntm`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), ``NTMModel(input_shape=(6, 4), output_dim=3,
config=NTMConfig(memory_size=16, memory_dim=8, controller_dim=32,
num_read_heads=1))`` after one real Adam step: **30** trainable weights, **0**
dead, **0** disconnected.

What this package has already taught, and what this file does NOT re-litigate
-----------------------------------------------------------------------------
A single ``ops.roll`` SIGN in the addressing shift survived **249 green tests**,
because nothing in the suite tested addressing DIRECTION. The lesson is that a
magnitude probe against an NTM proves very little: this file therefore claims
only what a gradient/knob/contract adoption can honestly claim -- that every
weight of the controller AND the memory heads is on the backward graph, that the
head-count and memory-geometry knobs reach the parameterisation, and that the
forward contract can fail. Direction is pinned by
``test_determinism_and_training_flag.py`` and the addressing tests, and is
deliberately not restated here where it would become a second, weaker copy.

The read/write head weights are named explicitly in the gradient assertion.
Without that, the 30-weight count could be met by the controller alone -- an NTM
whose memory addressing never received a gradient would look identical.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.ntm.model import NTMConfig, NTMModel

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

TIME_STEPS = 6
FEATURES = 4
OUTPUT_DIM = 3

#: Measured 2026-08-21 at the config below.
GF_N_WEIGHTS = 30

#: Weight-path fragments that must be PRESENT -- the memory machinery, not the
#: controller. See the module docstring.
NTM_PATH_FRAGMENTS = ("read", "write")


def _config(**overrides) -> NTMConfig:
    kwargs = dict(
        memory_size=16, memory_dim=8, controller_dim=32, num_read_heads=1,
    )
    kwargs.update(overrides)
    return NTMConfig(**kwargs)


def _sequences(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).normal(
        size=(batch, TIME_STEPS, FEATURES)).astype("float32")


def _model(**config_overrides) -> NTMModel:
    model = NTMModel(
        input_shape=(TIME_STEPS, FEATURES),
        output_dim=OUTPUT_DIM,
        config=_config(**config_overrides),
    )
    model(_sequences(1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestNTMGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _model()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}"
        )

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _sequences()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        for fragment in NTM_PATH_FRAGMENTS:
            assert any(fragment in path for path in report), (
                f"no weight whose path contains {fragment!r} -- the memory "
                f"machinery the count above rests on is not in the trainable set"
            )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _sequences())


class TestNTMKnobSensitivity:

    def test_controller_dim_changes_the_parameterisation(self):
        builders = {
            c: (lambda c=c: _model(controller_dim=c)) for c in (16, 32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="controller_dim")

    def test_num_read_heads_changes_the_addressing_parameterisation(self):
        """A knob that reaches ONLY the memory heads.

        ``controller_dim`` above would still pass if the memory were removed
        entirely. This one would not: the read-head count sets the addressing
        projections and the controller's read-vector input width.
        """
        builders = {
            h: (lambda h=h: _model(num_read_heads=h)) for h in (1, 2, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="num_read_heads")

    def test_memory_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _model(memory_dim=d)) for d in (8, 16, 32)
        }
        assert_structural_knob_changes_weights(builders, knob="memory_dim")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_read_heads")


class TestNTMSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _sequences()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"NTMModel(return_sequences=True) returns one tensor, got "
                f"{type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], TIME_STEPS, OUTPUT_DIM), (
                f"expected {(x.shape[0], TIME_STEPS, OUTPUT_DIM)}, got "
                f"{tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
