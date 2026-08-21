"""
Oracle adoption for ``models/resnet`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), ``ResNet(num_classes=10, blocks_per_stage=[1, 1],
filters_per_stage=[16, 32], block_type='basic', input_shape=(32, 32, 3))`` after
one real Adam step: **20** trainable weights, **0** dead, **0** disconnected.

Why the ``basic`` arm specifically. ``block_type='basic'`` was repaired at
iteration-2 step 17 to work at any stage-0 width, and every shipped variant uses
64, so the repair's own regime -- a stage-0 width that is NOT 64 -- is the one no
shipped configuration exercises. Both block types are therefore probed here, and
the ``basic`` arm runs at ``filters_per_stage[0] = 16``.

ResNet has no stochastic-depth and no dropout, so the reading is not a draw:
there is nothing in this model whose training-mode forward is random. That is
asserted rather than assumed (``test_no_layer_is_stochastic``), because a
gradient report taken under an unpinned stochastic rate reports the DRAW, not
the model -- the hazard that made a BeiT arm in batch A flaky 1 run in 4.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.resnet.model import ResNet

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

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

#: Measured 2026-08-21 at ``blocks_per_stage=[1, 1]``, ``block_type='basic'``.
GF_N_WEIGHTS_BASIC = 20


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + INPUT_SHAPE).astype("float32")


def _model(**overrides) -> ResNet:
    kwargs = dict(
        num_classes=NUM_CLASSES,
        blocks_per_stage=[1, 1],
        filters_per_stage=[16, 32],
        block_type="basic",
        input_shape=INPUT_SHAPE,
    )
    kwargs.update(overrides)
    model = ResNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    """One REAL optimizer step, so the report is not an init-time artifact."""
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestResNetGradientFlow:

    def test_no_layer_is_stochastic(self):
        """The premise of every measurement below: nothing here draws."""
        model = _model()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}. A gradient "
            f"report taken under one reports the DRAW, not the model"
        )

    def test_gradients_reach_every_trainable_weight_after_one_step_basic(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS_BASIC == len(model.trainable_weights)

    def test_gradients_reach_every_trainable_weight_bottleneck(self):
        """The other shipped block type, at the same geometry."""
        model = _model(block_type="bottleneck")
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights) > GF_N_WEIGHTS_BASIC

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestResNetKnobSensitivity:

    def test_blocks_per_stage_changes_the_parameterisation(self):
        builders = {
            tuple(b): (lambda b=b: _model(blocks_per_stage=list(b),
                                          filters_per_stage=[16] * len(b)))
            for b in ([1, 1], [2, 2], [2, 2, 2])
        }
        assert_structural_knob_changes_weights(builders, knob="blocks_per_stage")

    def test_block_type_changes_the_parameterisation(self):
        """The knob step 17 repaired, asserted at a stage-0 width of 16.

        Every shipped ResNet variant uses 64 at stage 0, so this is the width
        at which the repair actually lives.
        """
        builders = {
            t: (lambda t=t: _model(block_type=t))
            for t in ("basic", "bottleneck")
        }
        assert_structural_knob_changes_weights(builders, knob="block_type")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="blocks_per_stage")


class TestResNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"ResNet with include_top=True returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
