"""
Oracle adoption for ``models/superpoint`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE TWO CARRIED HAZARDS ARE BOTH *REPAIRED* STATES, AND A RED HERE IS A
REGRESSION -- SAY SO LOUDLY
-------------------------------------------------------------------------
The batch-C brief carried "``superpoint``: 2 ``None`` gradients of 183 under
fp16 only" (D-060) and "its ULP-void ``1e-12`` guard was repaired at step 17.1
(NaN 32,768/32,768 -> 0/32,768)". **Both of those numbers are the BEFORE state
of defects this plan already fixed.** ``model.py``'s D-060 anchor runs the
bicubic resize in float32 and casts back, and D-050 floors the L2 normalisation
at ``max(1e-12, finfo(dtype).tiny)``.

So this file asserts the REPAIRED state on both axes:

* :class:`TestSuperPointMixedPrecisionIsRepaired` requires **0** ``None``
  gradients under ``mixed_float16`` -- if the count is 2 again, the D-060
  float32 round-trip has been removed and the entire descriptor head is
  silently untrained under mixed precision.
* the same class requires a finite, non-zero descriptor field under
  ``mixed_float16`` -- if it NaNs, the D-050 floor has been reverted to a bare
  ``1e-12``, which is EXACTLY ``0.0`` in float16.

Neither is a new instrument; both read through ``gradient_report`` and the
existing ``precision_arm_oracle.precision_policy`` context manager.

Measured 2026-08-21, one Adam step, ramp loss, on a
``depths=(1,1,1) / dims=(16,32,64)`` SuperPoint at ``(64, 64, 1)``:

=========================  ==========  ======
arm                        weights     dead
=========================  ==========  ======
float32                    51          0
mixed_float16              51          0
=========================  ==========  ======

(The 183 in D-060 is the shipped ``"tiny"`` variant's weight count; this file
runs a smaller encoder so the suite stays affordable. The *kind* of reading is
the same and the descriptor-head weights -- the ones D-060 was about -- are
named explicitly rather than counted.)

``drop_path_rate`` is pinned to ``0.0`` and every build is seeded: batch A had
an arm flaky 1 run in 4 and batch B one flaky 2 in 5, both from an unpinned
stochastic rate or an unseeded build.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.superpoint.model import SuperPoint

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors, precision_policy
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

SHAPE = (64, 64, 1)
DESCRIPTOR_DIM = 32
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 51

#: The two weights D-060 was about, as path SUFFIXES.
#:
#: Never absolute ``Variable.path`` strings: Keras uniquifies a model name per
#: process, so the SECOND ``SuperPoint`` built in one pytest session is
#: ``super_point_1/...``. An absolute pin is green alone and red behind any
#: other test that builds the same class -- it bit batch B twice.
DESCRIPTOR_HEAD = ("descriptor_head/kernel", "descriptor_head/bias")


def ramp_loss(outputs: Any) -> Any:
    """``default_loss``'s asymmetric twin, IMPORTED from the precision oracle
    rather than re-typed -- a second copy is a second thing that can drift back
    into a symmetric loss (D-059)."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _images(batch: int = 2, shape=SHAPE, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + shape).astype("float32")


def _superpoint(**o) -> SuperPoint:
    kwargs: Dict[str, Any] = dict(
        depths=(1, 1, 1), dims=(16, 32, 64), input_shape=SHAPE,
        descriptor_dim=DESCRIPTOR_DIM,
        # Pinned, not defaulted: a gradient report taken through live stochastic
        # depth reports the draw, not the model.
        drop_path_rate=0.0,
    )
    kwargs.update(o)
    return SuperPoint(**kwargs)


def _built(build_fn=_superpoint, shape=SHAPE, seed: int = BUILD_SEED) -> SuperPoint:
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


class TestSuperPointGradientFlow:

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

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _images()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(), loss_fn=ramp_loss)


class TestSuperPointMixedPrecisionIsRepaired:
    """D-060 and D-050, asserted in their REPAIRED state.

    A failure in this class is a REGRESSION of a fix this plan already shipped,
    not a newly-discovered defect. Read the failure message before writing a
    report.
    """

    def test_the_descriptor_head_still_receives_a_gradient_under_mixed_float16(self):
        """D-060's exact subject. Two ``None`` gradients here means the float32
        round-trip in ``SuperPoint.call`` has been removed and the whole
        descriptor head is untrained under mixed precision -- silently, because
        TensorFlow returns ``None`` for a float16 ``ResizeBicubic`` gradient
        rather than raising."""
        with precision_policy("mixed_float16"):
            model = _built()
            x = _images()
            report = gradient_report(model, x, loss_fn=ramp_loss)

        disconnected = sorted(p for p, v in report.items() if v is None)
        assert disconnected == [], (
            f"REGRESSION of D-060: {len(disconnected)} weight(s) receive NO "
            f"gradient under mixed_float16: {disconnected}"
        )
        for suffix in DESCRIPTOR_HEAD:
            path = next(p for p in report if p.endswith(suffix))
            assert report[path] is not None and report[path] > 0.0, (
                f"{suffix} is dead under mixed_float16 (max|grad|="
                f"{report[path]}) -- this is D-060 returning"
            )

    def test_the_descriptor_field_is_finite_and_normalised_under_mixed_float16(self):
        """D-050's subject: ``np.float16(1e-12)`` is EXACTLY 0.0, so a bare
        ``1e-12`` L2 floor divides by zero at every all-zero location. The
        repaired floor is ``max(1e-12, finfo(dtype).tiny)``."""
        with precision_policy("mixed_float16"):
            model = _built()
            out = model(_images(), training=False)
            desc = np.asarray(keras.ops.convert_to_numpy(out["descriptors"]))

        assert np.isfinite(desc).all(), (
            f"REGRESSION of D-050: {int((~np.isfinite(desc)).sum())} of "
            f"{desc.size} descriptor entries are non-finite under "
            f"mixed_float16"
        )
        norms = np.sqrt((desc.astype("float32") ** 2).sum(axis=-1))
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=2e-2)

    def test_the_float32_arm_is_the_control(self):
        """Both readings above are about the DTYPE. Without this the two could
        be failing for a reason that has nothing to do with mixed precision."""
        model = _built()
        report = gradient_report(model, _images(), loss_fn=ramp_loss)
        assert [p for p, v in report.items() if v is None] == []


class TestSuperPointKnobSensitivity:

    def test_dims_change_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _superpoint(dims=d)))
            for d in ((16, 32, 64), (24, 48, 96), (32, 64, 128))
        }
        assert_structural_knob_changes_weights(builders, knob="dims")

    def test_descriptor_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _superpoint(descriptor_dim=d)))
            for d in (16, 32, 64)
        }
        assert_structural_knob_changes_weights(builders, knob="descriptor_dim")

    def test_activation_reaches_the_forward_pass(self):
        """A VALUE knob: same weight shapes, different arithmetic.

        A shape-only sweep is blind to it, and a build that dropped the kwarg
        on the way into the encoder blocks would pass one.
        """
        x = _images()
        builders = {
            a: (lambda a=a: _superpoint(activation=a)) for a in ("gelu", "relu")
        }
        deltas = assert_value_knob_changes_output(
            builders, x, knob="activation", extract=lambda o: o["keypoints"])
        assert all(d > 1e-4 for d in deltas.values()), deltas

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="dims")

    def test_the_value_knob_assertion_can_fail(self):
        x = _images()
        builders = {k: (lambda: _superpoint(activation="gelu"))
                    for k in ("a", "b")}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, x, knob="activation",
                extract=lambda o: o["keypoints"])


class TestSuperPointSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _images()

        def contract(out):
            assert isinstance(out, dict), (
                f"SuperPoint.call returns a dict, got {type(out)}")
            assert set(out) == {"keypoints", "descriptors"}, (
                f"unexpected key set {sorted(out)}")
            kp, desc = out["keypoints"], out["descriptors"]
            assert tuple(kp.shape) == (x.shape[0], SHAPE[0] // 8, SHAPE[1] // 8, 65), (
                f"detector grid is an 8x8 cell + 1 dustbin; got {tuple(kp.shape)}")
            assert tuple(desc.shape) == (
                x.shape[0], SHAPE[0], SHAPE[1], DESCRIPTOR_DIM), (
                f"descriptors are at FULL resolution; got {tuple(desc.shape)}")
            assert_finite(kp)
            assert_finite(desc)
            norms = np.asarray(keras.ops.convert_to_numpy(
                keras.ops.sqrt(keras.ops.sum(keras.ops.square(desc), axis=-1))))
            assert np.allclose(norms, 1.0, atol=1e-4), (
                f"descriptors must be unit-L2 along channels; norms range "
                f"{norms.min()} .. {norms.max()}")

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_detector_output_is_logits_not_probabilities(self):
        """The docstring says ``"keypoints"`` are RAW LOGITS. A softmax slipped
        onto that head would leave every shape identical and every finiteness
        check green, so it is asserted directly."""
        model = _built()
        kp = np.asarray(keras.ops.convert_to_numpy(
            model(_images(), training=False)["keypoints"]))
        sums = kp.sum(axis=-1)
        assert not np.allclose(sums, 1.0, atol=1e-3), (
            "the detector output sums to 1 along the 65-way axis -- it has been "
            "turned into a probability distribution, but the contract is logits"
        )
