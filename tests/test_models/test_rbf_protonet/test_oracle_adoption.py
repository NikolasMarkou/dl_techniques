"""
Oracle adoption for ``models/vision/rbf_protonet``.

Three shared instruments are adopted here, following the
``tests/test_models/test_resnet/test_oracle_adoption.py`` precedent: no new
oracle is authored.

All gradient-liveness assertions run AFTER one real optimizer step, never at
init -- per ``plans/LESSONS.md``: a zero-init-gate suite must not assert
gradient liveness at init. This matters even more here than for a plain conv
net, because the RBF head's own center-repulsion auxiliary loss
(``repulsion_strength``, D-002/D-006/D-007) only participates in the backward
graph once at least one optimizer step has actually been taken through
``model.losses``.

The knob exercised for ``assert_structural_knob_changes_weights`` is
``num_classes``: it is the one parameter this model has that changes a
weight's SHAPE (the RBF head's ``centers``/``gamma_raw`` are shaped by
``units=num_classes``) rather than merely a numeric hyperparameter value --
``filters_per_stage``/``stem_filters`` would work equally well, but
``num_classes`` is the knob most specific to this model's whole reason for
existing (one prototype per class).
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.rbf_protonet.model import RBFProtoNet

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


def _images(batch: int = 4) -> np.ndarray:
    return np.random.default_rng(0).random((batch,) + INPUT_SHAPE).astype("float32")


def _model(**overrides) -> RBFProtoNet:
    kwargs = dict(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
    )
    kwargs.update(overrides)
    model = RBFProtoNet(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    """One REAL optimizer step, so the gradient report is not an init-time
    artifact -- required both by `plans/LESSONS.md` and because the RBF
    head's repulsion loss only enters `model.losses` once weights have moved
    off their initial (already center-repelled) draw."""
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
        if model.losses:
            loss = loss + tf.cast(tf.add_n(model.losses), loss.dtype)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestRBFProtoNetGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestRBFProtoNetKnobSensitivity:

    def test_num_classes_changes_the_parameterisation(self):
        """`num_classes` reshapes the RBF head's `centers`/`gamma_raw` weights
        (one prototype per class) -- the structural knob most specific to
        this model's design."""
        builders = {
            n: (lambda n=n: _model(num_classes=n))
            for n in (5, 10, 20)
        }
        assert_structural_knob_changes_weights(builders, knob="num_classes")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_classes")


class TestRBFProtoNetSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"RBFProtoNet returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-vvv"]))
