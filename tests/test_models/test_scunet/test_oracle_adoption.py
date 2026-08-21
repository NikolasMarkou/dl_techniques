"""
Oracle adoption for ``models/scunet`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

ENTERING STATE, MEASURED BEFORE THIS FILE EXISTED
---------------------------------------------------
``pytest tests/test_models/test_scunet -q -p no:randomly`` at HEAD, 2026-08-21::

    2 failed, 71 passed, 1 warning in 721.92s (0:12:01)

    test_dynamic_spatial_dims.py::TestDynamicTrace
        ::test_one_dynamic_trace_serves_two_different_sizes
    test_dynamic_spatial_dims.py::TestDynamicTrace
        ::test_dynamic_and_static_builds_agree

Those two are on the plan's KNOWN PRE-EXISTING RED list -- float32
reassociation noise between a static and a dynamic trace -- and this directory
owns two of the six. Nothing in this file touches them, and a failure of either
after this commit is the SAME pre-existing red, not a regression.

THE INPUT FLOOR IS 33, NOT A ROUND NUMBER
-------------------------------------------
SCUNet reflect-pads its input up to the next multiple of 64, and a reflect pad
must be strictly smaller than the extent it pads. At ``height=32`` the required
pad is exactly 32, which is not less than 32, so the model REFUSES with a named
message rather than emitting garbage. This file therefore runs at 64 px and
asserts the refusal at 32 px rather than quietly avoiding it -- the refusal is
the model's contract, and a test suite that only ever passes it legal sizes
would not notice if it were removed.

Measured 2026-08-21, one Adam step, ramp loss, at
``config=[1]*7 / dim=16 / head_dim=8 / window_size=4`` on a ``(1, 64, 64, 3)``
input:

===============================  ==========  ======
arm                              weights     dead
===============================  ==========  ======
SCUNet                           148         0
===============================  ==========  ======

``stochastic_depth_rate`` is pinned to ``0.0`` and every build is seeded. That
is not decoration on THIS package in particular: SCUNet schedules stochastic
depth linearly across all ``sum(config)`` blocks, so a non-zero rate makes a
gradient report describe which blocks the single tape draw happened to keep.
Batch A had a BEiT arm flaky 1 run in 4 for exactly that reason.
"""

from typing import Any, Dict, List

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.scunet.model import SCUNet, create_scunet

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

SHAPE = (64, 64, 3)
#: The smallest height/width SCUNet accepts -- see the module docstring.
MIN_EXTENT = 33
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 148


def ramp_loss(outputs: Any) -> Any:
    """IMPORTED from ``precision_arm_oracle``, never re-typed (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _images(batch: int = 2, shape=SHAPE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + shape).astype("float32")


def _scunet(**o) -> SCUNet:
    kwargs: Dict[str, Any] = dict(
        in_nc=3, config=[1] * 7, dim=16, head_dim=8, window_size=4,
        # Pinned, not defaulted: see the module docstring.
        stochastic_depth_rate=0.0, input_resolution=64,
    )
    kwargs.update(o)
    return create_scunet(**kwargs)


def _built(build_fn=_scunet, shape=SHAPE, seed: int = BUILD_SEED) -> SCUNet:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_images(1, shape), training=False)
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


class TestSCUNetGradientFlow:

    def test_no_layer_is_stochastic(self):
        """The premise of every measurement below, and of the module table."""
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero stochastic rate is live: {stochastic}"

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _images(batch=1)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(batch=1), loss_fn=ramp_loss)


class TestSCUNetKnobSensitivity:

    def test_config_changes_the_parameterisation(self):
        """The per-stage block counts. ``[1]*7`` vs ``[2]*7`` vs an asymmetric
        layout -- three configurations, two independent adjacent claims."""
        builders = {
            tuple(c): (lambda c=c: _built(lambda: _scunet(config=list(c))))
            for c in ([1] * 7, [2] * 7, [1, 2, 1, 2, 1, 2, 1])
        }
        assert_structural_knob_changes_weights(builders, knob="config")

    def test_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _scunet(dim=d)))
            for d in (16, 32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="dim")

    def test_in_nc_changes_the_parameterisation(self):
        builders = {
            c: (lambda c=c: _built(lambda: _scunet(in_nc=c),
                                   SHAPE[:2] + (c,)))
            for c in (1, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="in_nc")

    def test_window_size_changes_the_parameterisation(self):
        """MEASURED: this knob is STRUCTURAL here, and the reading is a textbook
        case of the trap the oracle's own docstring warns about.

        It was written as a value knob first -- the Swin window decides WHICH
        pixels attend to which, which sounds like pure routing. The value
        instrument rejected it, and the numbers are the point: ``window_size=4``
        and ``window_size=8`` both hold **148 weight tensors** and differ at
        **348,918 vs 352,790 parameters**, because the relative-position-bias
        table is sized by the window. *"The counts may match while the shapes do
        not"* is the oracle's own error text, and a hand-written sweep asserting
        ``len(model.weights)`` would have seen nothing at all.
        """
        builders = {
            w: (lambda w=w: _built(lambda: _scunet(window_size=w)))
            for w in (4, 8)
        }
        signatures = assert_structural_knob_changes_weights(
            builders, knob="window_size")
        counts = {k: len(v) for k, v in signatures.items()}
        params = {
            k: sum(int(np.prod(shape)) for shape in v)
            for k, v in signatures.items()
        }
        assert counts[4] == counts[8] == 148, counts
        assert params[4] == 348918 and params[8] == 352790, params

    def test_the_value_instrument_convicts_input_resolution_as_inert(self):
        """The VALUE instrument's adoption, and its RED proof in one.

        ``input_resolution`` is DOCUMENTED advisory -- ``create_scunet`` says it
        "changes no attention geometry". So the value instrument must call it a
        no-op, and it does. This is a two-for-one: the instrument is exercised
        on a real knob of this package, and the fact that it convicts rather
        than passes is the proof it is not vacuous here.

        A reader must not "fix" this by widening a tolerance. The knob really is
        inert; the next test pins that as an EXACT output equality.
        """
        x = _images(batch=1)
        builders = {r: (lambda r=r: _scunet(input_resolution=r))
                    for r in (64, 256)}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(builders, x, knob="input_resolution")

    def test_input_resolution_is_advisory_and_that_is_asserted(self):
        """``create_scunet``'s docstring says ``input_resolution`` "changes no
        attention geometry" -- it is a hint forwarded to every ``SwinConvBlock``.

        A claim like that is exactly the kind that rots. Pinned as an EXACT
        output equality: two models differing only in this argument must agree
        bit-for-bit. If it ever starts mattering, this test says so before a
        user discovers it as a silent accuracy change.
        """
        outs = []
        for res in (64, 256):
            model = _built(lambda res=res: _scunet(input_resolution=res))
            outs.append(np.asarray(keras.ops.convert_to_numpy(
                model(_images(batch=1), training=False))))
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="dim")

    def test_a_config_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError):
            _scunet(config=[1] * 6)


class TestSCUNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _images(batch=2)

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"SCUNet.call returns ONE restored image, got {type(out)}")
            assert tuple(out.shape) == tuple(x.shape), (
                f"a restoration model's output must match its input shape "
                f"exactly; got {tuple(out.shape)} for {tuple(x.shape)}")
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_an_input_below_the_reflect_pad_floor_is_refused_with_a_reason(self):
        """The contract, asserted at the boundary from BOTH sides.

        32 px must raise and 33 px must not. A one-sided test would pass on a
        model that refused everything.
        """
        model = _built()
        with pytest.raises(ValueError, match="reflect"):
            model(_images(batch=1, shape=(32, 32, 3)), training=False)
        out = model(_images(batch=1, shape=(MIN_EXTENT, MIN_EXTENT, 3)),
                    training=False)
        assert tuple(out.shape) == (1, MIN_EXTENT, MIN_EXTENT, 3)

    def test_the_output_is_not_the_input(self):
        """A restoration network whose residual path was severed would return
        its input unchanged -- with the right shape, finite values, and every
        assertion above green."""
        model = _built()
        x = _images(batch=1)
        out = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)))
        assert float(np.max(np.abs(out - x))) > 1e-4, (
            "the model returned its input; the network contributes nothing")
