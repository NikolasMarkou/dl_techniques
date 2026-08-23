"""
Oracle adoption for ``models/squeezenet`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE D-059 SYMMETRY TRAP LIVES HERE TOO, AND IT IS PINNED
---------------------------------------------------------
``gradient_flow_oracle.default_loss`` is ``mean(square(t))``, which is SYMMETRIC
under any permutation of the output. Both SqueezeNet generations end in
``Activation('softmax')`` (V1 ``predictions``; V2 ``activation='softmax'``, see
that file's D-063), and the gradient of a symmetric function of an EXACTLY
uniform softmax is EXACTLY zero -- so the whole backward pass reads ``0.0`` and
the instrument convicts a perfectly healthy model. D-059 measured that on
``mobilenet``, and this package was named alongside it.

**Measured here 2026-08-21: the trap does NOT fire on a randomly-initialised
SqueezeNet.** With the shipped initializers the head's softmax is only NEARLY
uniform, and the symmetric loss reports **0 dead of 52** (V1) and **0 of 36**
(V2). That is the same reading batch B took on ``mobilenet``.

The shipped initializers CHANGED under this reading on 2026-08-23
(plan-2026-08-23T091307-9a110062/D-481 replaced ``glorot_uniform`` with the
prototxt's ``xavier``/gaussian-0.01 fillers, which narrows ``conv10`` and so
pushes the softmax CLOSER to uniform -- toward the trap). Re-measured on the new
initializers: still **0 dead** in both arms, and the weight counts are unmoved at
52 / 36. The assertion at ``test_a_random_squeezenet_is_not_on_the_saddle``
re-derives the count at runtime rather than trusting this paragraph. It is pinned
anyway, on a zeroed head where the output is EXACTLY ``1/C`` and both halves are
asserted, so a reader who measures ``0.0`` on a converged model has a test
naming the reason instead of a defect report to file. Every gradient assertion
below uses the RAMP loss (``precision_arm_oracle._asymmetric_loss``, IMPORTED,
not re-typed) for the same reason.

Measured 2026-08-21, one Adam step, ramp loss:

===============================  ==============  =========  ======
model                            input           weights    dead
===============================  ==============  =========  ======
SqueezeNetV1 "1.1"               (32, 32, 3)     52         0
SqueezeNoduleNetV2 "v2"          (40, 40, 1)     36         0
===============================  ==============  =========  ======

THE SPATIAL FLOOR IS A REAL CONSTRAINT, NOT A TEST-SIZE PREFERENCE
-------------------------------------------------------------------
Every downsampling stage uses ``padding='valid'``, so an input below the
variant's floor collapses an axis to length zero and yields an ALL-NaN output of
the correct shape. The floor is 35 for the ``"1.0"``/``"1.0_bypass"`` stem family
and 31 for ``"1.1"``; ``spatial_guard`` raises instead. The V1 arm therefore runs
``"1.1"`` at 32 px (which clears 31 and not 35), and the refusal is asserted
rather than assumed.

Every dropout rate is pinned to ``0.0`` (the shipped default is ``0.5``) and
every build is seeded. Batch A had an arm flaky 1 run in 4 and batch B one flaky
2 in 5, both from an unpinned rate or an unseeded build.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.squeezenet.squeezenet_v1 import (
    SqueezeNetV1,
    create_squeezenet_v1,
)
from dl_techniques.models.squeezenet.squeezenet_v2 import (
    SqueezeNoduleNetV2,
    create_squeezenodule_net_v2,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

NUM_CLASSES = 10
V1_SHAPE = (32, 32, 3)
V2_SHAPE = (40, 40, 1)
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = {"v1": 52, "v2": 36}


def ramp_loss(outputs: Any) -> Any:
    """``default_loss``'s asymmetric twin -- see the module docstring."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _images(shape=V1_SHAPE, batch: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + shape).astype("float32")


def _v1(**o) -> SqueezeNetV1:
    kwargs: Dict[str, Any] = dict(
        variant="1.1", num_classes=NUM_CLASSES, input_shape=V1_SHAPE,
        dropout_rate=0.0,
    )
    kwargs.update(o)
    variant = kwargs.pop("variant")
    return SqueezeNetV1.from_variant(variant, **kwargs)


def _v2(**o) -> SqueezeNoduleNetV2:
    kwargs: Dict[str, Any] = dict(
        variant="v2", num_classes=NUM_CLASSES, input_shape=V2_SHAPE,
        dropout_rate=0.0,
    )
    kwargs.update(o)
    variant = kwargs.pop("variant")
    return SqueezeNoduleNetV2.from_variant(variant, **kwargs)


def _built(build_fn, shape=V1_SHAPE, seed: int = BUILD_SEED):
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_images(shape, batch=1), training=False)
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


class TestTheUniformSoftmaxSaddleIsNotADeadModel:
    """D-059, pinned on this package's own head shape.

    Both halves are asserted, so restoring the symmetric loss anywhere in this
    file fails a test rather than quietly re-creating the false finding.
    """

    @staticmethod
    def _uniform_softmax_head() -> keras.Model:
        """A classifier whose softmax output is EXACTLY uniform.

        Not "nearly": the final ``Dense`` is zero-initialised in both kernel and
        bias, so every logit is exactly 0.0 and every class probability is
        exactly ``1/NUM_CLASSES`` -- the state a converged-to-chance SqueezeNet
        head approaches, and the state a ``predictions`` softmax over a dead
        feature stack sits at exactly.
        """
        return keras.Sequential([
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(
                NUM_CLASSES, activation="softmax",
                kernel_initializer="zeros", bias_initializer="zeros",
            ),
        ])

    def test_the_output_is_exactly_uniform(self):
        model = self._uniform_softmax_head()
        out = keras.ops.convert_to_numpy(
            model(np.ones((2, 8), "float32"), training=False))
        np.testing.assert_array_equal(out, np.full_like(out, 1.0 / NUM_CLASSES))

    def test_the_symmetric_loss_reports_exactly_zero(self):
        """The false CRITICAL: ``default_loss`` convicts a healthy model."""
        model = self._uniform_softmax_head()
        report = gradient_report(model, np.ones((2, 8), "float32"),
                                 loss_fn=default_loss)
        assert report and all(v == 0.0 for v in report.values()), (
            f"expected an all-zero report from the symmetric loss on a uniform "
            f"softmax (D-059); got {report}"
        )

    def test_the_ramp_loss_reports_nonzero_on_the_same_weights(self):
        """The discriminating half: same model, same weights, ramp loss."""
        model = self._uniform_softmax_head()
        report = gradient_report(model, np.ones((2, 8), "float32"),
                                 loss_fn=ramp_loss)
        assert report and all(v is not None and v > 0.0
                              for v in report.values()), (
            f"the ramp loss must break the symmetry; got {report}"
        )

    def test_the_shipped_heads_are_softmax(self):
        """The premise. If either head stops being a softmax this file's whole
        explanation stops applying, and this test says so."""
        for build, shape in ((_v1, V1_SHAPE), (_v2, V2_SHAPE)):
            model = _built(build, shape)
            out = keras.ops.convert_to_numpy(model(_images(shape), training=False))
            rows = out.sum(axis=-1)
            np.testing.assert_allclose(rows, np.ones_like(rows), atol=1e-5)

    def test_a_random_squeezenet_is_not_on_the_saddle(self):
        """MEASURED, and the reason the trap did not fire unseeded.

        A randomly-initialised head is only NEARLY uniform, so the symmetric
        loss reads non-zero on every weight. Recorded as a test because the
        opposite was the batch's stated expectation.
        """
        model = _built(_v1, V1_SHAPE)
        report = gradient_report(model, _images(V1_SHAPE), loss_fn=default_loss)
        dead = {p for p, v in report.items() if v is None or v == 0.0}
        assert dead == set(), (
            f"the symmetric loss DOES convict a random SqueezeNet here: {dead}"
        )


class TestSqueezeNetGradientFlow:

    def test_no_layer_is_stochastic(self):
        """The premise of every measurement below. The shipped default is 0.5."""
        for name, build, shape in (("v1", _v1, V1_SHAPE), ("v2", _v2, V2_SHAPE)):
            model = _built(build, shape)
            stochastic = [
                (layer.name, attr, getattr(layer, attr))
                for layer in model._flatten_layers(include_self=False)
                for attr in ("rate", "drop_path_rate", "dropout_rate")
                if isinstance(getattr(layer, attr, None), float)
                and getattr(layer, attr) > 0.0
            ]
            assert stochastic == [], (
                f"{name}: a non-zero stochastic rate is live: {stochastic}")

    @pytest.mark.parametrize(
        "name,build,shape", [("v1", _v1, V1_SHAPE), ("v2", _v2, V2_SHAPE)])
    def test_gradients_reach_every_trainable_weight_after_one_step(
            self, name, build, shape):
        model = _built(build, shape)
        x = _images(shape)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS[name] == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built(_v1, V1_SHAPE)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(V1_SHAPE), loss_fn=ramp_loss)


class TestSqueezeNetKnobSensitivity:

    def test_v1_variant_changes_the_parameterisation(self):
        """``"1.1"`` vs ``"1.0_bypass"``: different stem AND different pooling
        indices. Both clear their own spatial floor at 64 px."""
        builders = {
            v: (lambda v=v: _built(
                lambda: _v1(variant=v, input_shape=(64, 64, 3)), (64, 64, 3)))
            for v in ("1.0", "1.1", "1.0_bypass")
        }
        assert_structural_knob_changes_weights(builders, knob="variant")

    def test_num_classes_changes_the_parameterisation(self):
        builders = {
            c: (lambda c=c: _built(lambda: _v1(num_classes=c), V1_SHAPE))
            for c in (2, 10, 100)
        }
        assert_structural_knob_changes_weights(builders, knob="num_classes")

    def test_v2_variant_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _built(lambda: _v2(variant=v), V2_SHAPE))
            for v in ("v1", "v2")
        }
        assert_structural_knob_changes_weights(builders, knob="variant")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built(_v1, V1_SHAPE)),
                    "b": (lambda: _built(_v1, V1_SHAPE))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="variant")


class TestSqueezeNetSmokeContract:

    @pytest.mark.parametrize(
        "build,shape", [(_v1, V1_SHAPE), (_v2, V2_SHAPE)])
    def test_the_forward_contract_rejects_a_broken_forward(self, build, shape):
        model = _built(build, shape)
        x = _images(shape)

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"a SqueezeNet with include_top=True returns one tensor, got "
                f"{type(out)}")
            assert tuple(out.shape) == (x.shape[0], NUM_CLASSES), (
                f"expected {(x.shape[0], NUM_CLASSES)}, got {tuple(out.shape)}")
            assert_finite(out)
            rows = keras.ops.convert_to_numpy(keras.ops.sum(out, axis=-1))
            np.testing.assert_allclose(rows, np.ones_like(rows), atol=1e-5)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_an_input_below_the_spatial_floor_is_refused_at_construction(self):
        """The all-NaN forward that ``spatial_guard`` exists to prevent.

        32 px clears ``"1.1"``'s floor of 31 and does NOT clear ``"1.0"``'s
        floor of 35, so the same size is accepted by one variant and refused by
        the other -- which is what makes this a guard rather than a size
        preference.
        """
        with pytest.raises(ValueError):
            _v1(variant="1.0", input_shape=(32, 32, 3))
        _v1(variant="1.1", input_shape=(32, 32, 3))  # must NOT raise

    def test_pretrained_weights_are_refused_rather_than_silently_ignored(self):
        """The refusal lives in the module FACTORIES, not in ``from_variant``.

        Measured: ``SqueezeNetV1.from_variant(..., weights="imagenet")`` does
        NOT raise -- ``weights`` is swallowed by ``**kwargs`` and handed to
        ``keras.Model``. Only ``create_squeezenet_v1`` /
        ``create_squeezenodule_net_v2`` guard it. This test therefore calls the
        factories, and names the asymmetry so a reader does not conclude the
        guard is universal.
        """
        with pytest.raises(NotImplementedError):
            create_squeezenet_v1("1.1", num_classes=NUM_CLASSES,
                                 input_shape=V1_SHAPE, weights="imagenet")
        with pytest.raises(NotImplementedError):
            create_squeezenodule_net_v2("v2", num_classes=NUM_CLASSES,
                                        input_shape=V2_SHAPE,
                                        weights="imagenet")
