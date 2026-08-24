"""
Oracle adoption for ``models/cbam`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU), ``CBAMNet(num_classes=10, input_shape=(32,32,3),
dims=[32,64])`` after one real optimizer step: **18** trainable weights, **0**
dead. Both CBAM sub-modules are represented in that set and both are live --
``stage_N_cbam/channel_attention/...`` (2 tensors per stage, the bottleneck MLP,
which the reference architecture leaves bias-free) and
``stage_N_cbam/spatial_attention/spatial_attention_conv/{kernel,bias}``. That
matters for this package specifically: an attention module whose output is
computed and then multiplied by nothing -- or added to a branch that is
discarded -- is exactly the shape a per-weight gradient assertion convicts and a
shape/finiteness smoke test cannot see.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.cbam.model import CBAMNet

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

#: Measured 2026-08-21 at `dims=[32, 64]`.
GF_N_WEIGHTS = 18

#: The CBAM sub-module weight-path fragments that must be PRESENT in the report.
#: Without this the count above could be met by 18 backbone weights and no
#: attention at all.
CBAM_PATH_FRAGMENTS = (
    "stage_0_cbam/channel_attention/",
    "stage_0_cbam/spatial_attention/",
    "stage_1_cbam/channel_attention/",
    "stage_1_cbam/spatial_attention/",
)


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + INPUT_SHAPE).astype("float32")


def _model(**overrides):
    kwargs = dict(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE, dims=[32, 64])
    kwargs.update(overrides)
    model = CBAMNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestCBAMGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        for fragment in CBAM_PATH_FRAGMENTS:
            assert any(fragment in path for path in report), (
                f"no weight under {fragment!r} -- the attention module the count "
                f"above rests on is not in the trainable set"
            )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestCBAMKnobSensitivity:

    def test_dims_changes_the_parameterisation(self):
        builders = {
            tuple(d): (lambda d=d: _model(dims=list(d)))
            for d in ([16, 32], [32, 64], [32, 64, 128])
        }
        assert_structural_knob_changes_weights(builders, knob="dims")

    def test_attention_ratio_changes_the_channel_attention_bottleneck(self):
        """A knob that reaches ONLY the attention module.

        `dims` above would still pass if CBAM were removed entirely. This one
        would not: `attention_ratio` sets the channel-attention bottleneck width
        and touches nothing else, so a no-op here means the attention module is
        not being parameterised by its own kwarg.
        """
        builders = {
            r: (lambda r=r: _model(attention_ratio=r)) for r in (2, 4, 8)
        }
        assert_structural_knob_changes_weights(builders, knob="attention_ratio")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model(dims=[32, 64])), "b": (lambda: _model(dims=[32, 64]))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="dims")


class TestCBAMSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"CBAMNet returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)
            rows = keras.ops.convert_to_numpy(keras.ops.sum(out, axis=-1))
            np.testing.assert_allclose(rows, np.ones_like(rows), atol=1e-5)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
