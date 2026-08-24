"""
Oracle adoption for ``models/fastvit`` -- Phase 5 batch A.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (CPU) on ``FastVitImageEncoder(variant='mci0',
layers=(1,1,1,1), embed_dims=(32,64,128,256), input_shape=(64,64,3),
projection_dim=48)`` after one real optimizer step: **142** trainable weights,
**0** dead, **0** non-finite.

Why that number is worth pinning for THIS package: FastViT is a
reparameterisable architecture -- its blocks carry parallel branches that a
deployment path is meant to fold together. A branch that is constructed but
never summed into the residual is architecturally invisible (the output shape,
the parameter count and the finiteness are all unchanged) and shows up only as a
subtree of weights with an identically-zero gradient.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.fastvit.model import FastVitImageEncoder

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

INPUT_SHAPE = (64, 64, 3)
PROJECTION_DIM = 48

#: Measured 2026-08-21 at the tiny mci0 configuration in `_model`.
GF_N_WEIGHTS = 142


def _images(batch: int = 2) -> np.ndarray:
    return np.random.default_rng(17).standard_normal(
        (batch,) + INPUT_SHAPE
    ).astype("float32")


def _model(**overrides):
    kwargs = dict(variant="mci0", layers=(1, 1, 1, 1),
                  embed_dims=(32, 64, 128, 256), input_shape=INPUT_SHAPE,
                  projection_dim=PROJECTION_DIM)
    kwargs.update(overrides)
    model = FastVitImageEncoder(**kwargs)
    model(_images(1), training=False)
    return model


def _one_adam_step(model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))


class TestFastVitGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _model()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_every_stage_is_represented_in_the_live_set(self):
        """A four-stage tower whose last stage is detached still returns (B, 48).

        The count above cannot see that: the weights would still exist. This
        pins that each stage index appears among the weights that receive a
        gradient, not merely among the weights that exist.
        """
        model = _model()
        x = _images()
        _one_adam_step(model, x)
        report = assert_gradients_reach_every_trainable_weight(model, x)

        live = [p for p, v in report.items() if v is not None and v > 0.0]
        assert len(live) == GF_N_WEIGHTS
        prefixes = {p.split("/")[1] for p in live if "/" in p}
        assert {f"stage_{n}" for n in range(4)} <= prefixes, (
            f"a stage has no live weight; prefixes seen: {sorted(prefixes)}"
        )

    def test_the_gradient_assertion_can_fail(self):
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _images())


class TestFastVitKnobSensitivity:

    def test_layers_changes_the_parameterisation(self):
        builders = {
            tuple(l): (lambda l=l: _model(layers=tuple(l)))
            for l in ((1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1))
        }
        signatures = assert_structural_knob_changes_weights(builders, knob="layers")
        counts = [len(signatures[k]) for k in signatures]
        assert counts == sorted(counts), counts

    def test_projection_dim_changes_the_parameterisation(self):
        """A head-only knob: `layers` above would pass with the head removed."""
        builders = {d: (lambda d=d: _model(projection_dim=d)) for d in (32, 48, 64)}
        assert_structural_knob_changes_weights(builders, knob="projection_dim")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="layers")


class TestFastVitSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _images()

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"the encoder returns one tensor, got {type(out)}"
            )
            assert tuple(out.shape) == (x.shape[0], PROJECTION_DIM), (
                f"expected {(x.shape[0], PROJECTION_DIM)}, got {tuple(out.shape)}"
            )
            assert_finite(out)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
