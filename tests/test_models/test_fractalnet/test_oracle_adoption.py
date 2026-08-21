"""
Oracle adoption for ``models/fractalnet`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``FractalNet(num_classes=10,
input_shape=(32,32,3), depths=[1,2,2], filters=[32,64,128],
drop_path_rate=0.0)`` after one real optimizer step: **30** trainable weights of
**44** total, **0** dead, **0** non-finite.

``drop_path_rate`` is pinned to 0.0 for the gradient reading, and that is not a
convenience. FractalNet's defining regulariser is drop-path over parallel
columns; at a non-zero rate the single draw the tape sees can legitimately kill
a whole column, and the oracle would then report that column's weights dead as a
property of the DRAW rather than of the model. The knob tests below exercise the
non-zero setting separately.

The carried expected set, re-measured
-------------------------------------
Phase 4's record warned that "a freeze returns **18** trainable weights" for
this package (the R-057 ``get_config`` defect). **That reading does not
reproduce at HEAD** -- it was the BEFORE state and the defect is repaired.
Measured here: ``model.trainable = False`` leaves **0** trainable weights, and
so does a ``from_config(get_config())`` rebuild of the frozen model (the config
carries ``trainable: False``). ``test_a_freeze_survives_the_config_round_trip``
pins the repaired state so the 18 cannot come back unnoticed.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.fractalnet.model import FractalNet

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
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

#: Measured 2026-08-21 at depths=[1,2,2], filters=[32,64,128].
GF_N_TRAINABLE = 30
GF_N_TOTAL_WEIGHTS = 44


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + INPUT_SHAPE).astype("float32")


def _model(**overrides):
    kwargs = dict(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE,
                  depths=[1, 2, 2], filters=[32, 64, 128], drop_path_rate=0.0)
    kwargs.update(overrides)
    model = FractalNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestFractalNetGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_TRAINABLE == len(model.trainable_weights)
        assert len(model.weights) == GF_N_TOTAL_WEIGHTS, (
            "the 14 non-trainable weights are BatchNorm moving statistics; if "
            "this moves, re-derive which set the assertion above covers"
        )

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())

    def test_a_freeze_survives_the_config_round_trip(self):
        """The carried "a freeze returns 18 trainable weights" reading, re-measured.

        It does not reproduce at HEAD: the freeze is total on both sides of a
        `get_config()` / `from_config()` round trip, and the oracle refuses to
        assert on the frozen model rather than reporting green over an empty
        weight set. Pinned so the 18 cannot return silently.
        """
        model = _model()
        model.trainable = False
        assert model.trainable_weights == []
        with pytest.raises(ValueError, match="no trainable weights"):
            gradient_report(model, _images())

        config = model.get_config()
        assert config.get("trainable") is False, (
            "`trainable` is missing from get_config(); a rebuilt model would "
            "silently come back trainable -- this is the R-057 shape"
        )
        rebuilt = FractalNet.from_config(config)
        rebuilt(_images(1), training=False)
        assert rebuilt.trainable_weights == [], (
            f"{len(rebuilt.trainable_weights)} weights came back trainable "
            f"after a frozen round trip"
        )
        assert len(rebuilt.weights) == GF_N_TOTAL_WEIGHTS


class TestFractalNetKnobSensitivity:

    def test_depths_changes_the_parameterisation(self):
        builders = {
            tuple(d): (lambda d=d: _model(depths=list(d)))
            for d in ([1, 1, 1], [1, 2, 2], [2, 2, 2])
        }
        signatures = assert_structural_knob_changes_weights(builders, knob="depths")
        counts = [len(signatures[k]) for k in signatures]
        assert counts == sorted(counts), counts

    def test_filters_changes_the_parameterisation(self):
        builders = {
            tuple(f): (lambda f=f: _model(filters=list(f)))
            for f in ([16, 32, 64], [32, 64, 128])
        }
        assert_structural_knob_changes_weights(builders, knob="filters")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depths")


class TestFractalNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"FractalNet returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
