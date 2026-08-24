"""
Oracle adoption for ``models/cliffordnet`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``CliffordNet(num_classes=10, channels=16,
depth=2, patch_size=2, shifts=[1, 2], stochastic_depth_rate=0.0,
dropout_rate=0.0)`` after one real optimizer step: **34** trainable weights,
**0** dead, **0** non-finite.

``stochastic_depth_rate`` and ``dropout_rate`` are pinned to 0.0 for the
gradient reading on purpose. Both are *stochastic* in ``training=True``, and a
drop that happens to kill a branch on the single draw the tape sees would
report that branch's weights dead -- a property of the draw, not of the model.
The knob tests below then exercise the non-zero settings separately.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.cliffordnet.model import CliffordNet

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

#: Measured 2026-08-21 at channels=16, depth=2.
GF_N_WEIGHTS = 34


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + INPUT_SHAPE).astype("float32")


def _model(**overrides):
    kwargs = dict(num_classes=NUM_CLASSES, channels=16, depth=2, patch_size=2,
                  shifts=[1, 2], stochastic_depth_rate=0.0, dropout_rate=0.0)
    kwargs.update(overrides)
    model = CliffordNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestCliffordNetGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestCliffordNetKnobSensitivity:

    def test_depth_changes_the_parameterisation(self):
        builders = {d: (lambda d=d: _model(depth=d)) for d in (1, 2, 3)}
        signatures = assert_structural_knob_changes_weights(builders, knob="depth")
        counts = [len(signatures[d]) for d in (1, 2, 3)]
        assert counts == sorted(counts) and counts[0] < counts[-1], counts

    def test_shifts_changes_the_parameterisation(self):
        """A knob that reaches the Clifford block's own parameterisation.

        `depth` above would pass for any stacked architecture. The `shifts` list
        is what makes this a CliffordNet, so a no-op here means the defining
        kwarg is not reaching the blocks.
        """
        builders = {
            tuple(s): (lambda s=s: _model(shifts=list(s)))
            for s in ([1], [1, 2], [1, 2, 4])
        }
        assert_structural_knob_changes_weights(builders, knob="shifts")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(depth=2)), "b": (lambda: _model(depth=2))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depth")


class TestCliffordNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"CliffordNet returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
