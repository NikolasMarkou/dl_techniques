"""
Oracle adoption for ``models/time_series/mdn`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), ``MDNModel(hidden_layers=[32, 16],
output_dimension=2, num_mixtures=3)`` after one real Adam step: **22** trainable
weights, **0** dead, **0** disconnected.

Why this matters for a mixture-density head specifically. An MDN's output packs
THREE parameter groups into one flat vector -- ``num_mixtures *
output_dimension`` means, the same count of (log-)sigmas, and ``num_mixtures``
mixing logits. A head that emitted, say, constant mixing weights would still
produce the right SHAPE and finite values, so a shape/finiteness smoke test
cannot see it; a per-weight gradient assertion can, and the knob arm below
separates the two structural knobs (``num_mixtures`` and ``output_dimension``)
that both change that flat width.

The default ``dropout_rate`` is ``None`` and ``use_batch_norm`` is off, so this
model has no stochastic training-mode forward at all; that premise is asserted
rather than assumed, because a report taken under an unpinned stochastic rate
reports the DRAW, not the model.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.time_series.mdn.model import MDNModel

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

INPUT_DIM = 5
OUTPUT_DIM = 2
NUM_MIXTURES = 3

#: Measured 2026-08-21 at ``hidden_layers=[32, 16]``.
GF_N_WEIGHTS = 22

#: ``2 * num_mixtures * output_dimension + num_mixtures`` -- means, sigmas, pi.
MDN_WIDTH = 2 * NUM_MIXTURES * OUTPUT_DIM + NUM_MIXTURES


def _inputs(batch: int = 8) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(batch, INPUT_DIM)).astype("float32")


def _model(**overrides) -> MDNModel:
    kwargs = dict(
        hidden_layers=[32, 16],
        output_dimension=OUTPUT_DIM,
        num_mixtures=NUM_MIXTURES,
    )
    kwargs.update(overrides)
    model = MDNModel(**kwargs)
    model(_inputs(1), training=False)
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


class TestMDNGradientFlow:

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
        x = _inputs()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        assert any("mdn" in path.lower() for path in report), (
            "no weight under the MDN head is in the trainable set -- the count "
            "above could then be met by the backbone MLP alone"
        )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _inputs())


class TestMDNKnobSensitivity:

    def test_hidden_layers_changes_the_parameterisation(self):
        builders = {
            tuple(h): (lambda h=h: _model(hidden_layers=list(h)))
            for h in ([16], [32, 16], [32, 16, 8])
        }
        assert_structural_knob_changes_weights(builders, knob="hidden_layers")

    def test_num_mixtures_changes_the_head_width(self):
        """A knob that reaches ONLY the mixture head.

        ``hidden_layers`` above would still pass if the MDN head were replaced
        by a fixed-width Dense. This one would not: ``num_mixtures`` sets the
        packed output width and touches nothing in the backbone.
        """
        builders = {
            m: (lambda m=m: _model(num_mixtures=m)) for m in (2, 3, 5)
        }
        signatures = assert_structural_knob_changes_weights(
            builders, knob="num_mixtures")
        widths = [signatures[m][-1][-1] for m in (2, 3, 5)]
        assert widths == sorted(widths), (
            f"more mixtures must not shrink the packed output: {widths}"
        )

    def test_output_dimension_changes_the_head_width(self):
        builders = {
            d: (lambda d=d: _model(output_dimension=d)) for d in (1, 2, 4)
        }
        assert_structural_knob_changes_weights(builders, knob="output_dimension")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_mixtures")


class TestMDNSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _inputs()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"MDNModel returns one packed tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], MDN_WIDTH), (
                f"expected {(x.shape[0], MDN_WIDTH)} "
                f"(2 * {NUM_MIXTURES} * {OUTPUT_DIM} + {NUM_MIXTURES}), got "
                f"{tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
