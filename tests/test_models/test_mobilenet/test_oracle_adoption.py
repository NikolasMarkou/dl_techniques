"""
Oracle adoption for ``models/mobilenet`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

THE SYMMETRY TRAP LIVES HERE, and this file is where it is pinned
--------------------------------------------------------------------
``gradient_flow_oracle.default_loss`` is ``mean(square(t))``, which is SYMMETRIC
under any permutation of the output. An untrained classifier head emits a
(near-)uniform softmax, and the gradient of a symmetric function of an EXACTLY
uniform softmax is EXACTLY zero -- so the whole backward pass reads ``0.0`` and
the instrument convicts a perfectly healthy model. That reading was measured on
this very package during iteration-2 step 18.1 (``grad_norm_sum`` exactly
``0.000000e+00`` under BOTH ``mixed_float16`` and ``float32``) and ruled on in
decisions.md D-059.

Every gradient assertion in this file therefore uses the RAMP loss --
``precision_arm_oracle._asymmetric_loss``, the SAME function D-059 installed, not
a re-typed copy -- and ``TestTheUniformSoftmaxSaddleIsNotADeadModel`` pins the
fact itself two-sided on a zeroed head: EXACTLY ``0.0`` under the symmetric loss
and ``> 0.0`` under the ramp, in one process on one set of weights. A later
reader who measures ``0.0`` here and starts writing a defect report has a test
naming the reason, so it cannot be re-filed as a dead model.

Measured 2026-08-21 (GPU 1) after one real Adam step under the ramp loss, all
four shipped generations at ``num_classes=10, dropout_rate=0.0``:

===============  ==============  =========  =====================
model            input           weights    dead
===============  ==============  =========  =====================
MobileNetV1      (32, 32, 3)     83         0
MobileNetV2      (64, 64, 3)     161        0
MobileNetV3      (32, 32, 3)     127        0
MobileNetV4      (32, 32, 3)     37         0
===============  ==============  =========  =====================

THE SECOND TRAP: MobileNetV2 AT 32x32 IS NOT A DEFECT EITHER
------------------------------------------------------------
At ``(32, 32, 3)`` MobileNetV2's last stage collapses to a **1x1** feature map
(measured: blocks 13/14/15 are ``(1, 1, 1, 40)`` and block 16 ``(1, 1, 1, 80)``),
and at 1x1 a ``project_norm`` ``beta`` shift is spatially UNIFORM, so the next
BatchNorm removes it exactly. Exactly four weights read ``0.0``, and they are
exactly the four blocks at 1x1. At 64/128/224 the dead set is EMPTY. The V2 arm
above therefore runs at 64x64, and the 32x32 reading is pinned as a geometry
artifact rather than silently avoided.
THE THIRD TRAP: MobileNetV3's SQUEEZE-EXCITE ReLU CAN DIE ON A DRAW
--------------------------------------------------------------------
At ``width_multiplier=0.25`` MobileNetV3's ``block_0`` squeeze-excite
bottleneck is **2 channels wide**. On roughly one initialisation in eight its
pre-ReLU is negative in every channel and every batch row, the ReLU output is
EXACTLY zero, and both SE kernels report an identically-zero gradient.
Unseeded, this file measured ``1 failed / 18 passed`` in **2 of 5** consecutive
runs. Every build here is therefore seeded (:data:`BUILD_SEED`), and
``TestTheSqueezeExciteReLUCanDieOnADraw`` pins both sides -- dead at seed 6 with
the ReLU output measured all-zero, live at the build seed -- so the reading is a
draw rather than a disconnected module, and a later reader cannot re-file it.

"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
# `_asymmetric_loss` is imported rather than re-typed on purpose: it is the
# ramp D-059 installed, and a second copy is a second thing that can drift back
# into a symmetric loss. `test_precision_arm_oracle.py` imports it the same way.
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

NUM_CLASSES = 10
SMALL = (32, 32, 3)
#: MobileNetV2 runs here instead -- see the module docstring's second trap.
V2_SHAPE = (64, 64, 3)

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = {"v1": 83, "v2": 161, "v3": 127, "v4": 37}

#: The four MobileNetV2 weights that read EXACTLY 0.0 at a 32x32 input, and the
#: only ones. Each names a block whose output is 1x1 at that input size.
#:
#: Stored as path SUFFIXES, never as absolute ``Variable.path`` strings: Keras
#: uniquifies a model's name per process, so the SECOND ``MobileNetV2`` built in
#: one pytest session is ``mobile_net_v2_1/...``. A pin written against the
#: absolute path passes when its file is run alone and fails behind any other
#: test that builds the same class -- an order-dependent guard, which is the
#: same class of defect batch A found inside the gradient oracle's own control.
V2_32PX_UNIFORM_SHIFT_BETAS = frozenset(
    f"block_{i}/project_norm/beta" for i in (13, 14, 15, 16)
)


def ramp_loss(outputs: Any) -> Any:
    """``default_loss``'s asymmetric twin -- see the module docstring."""
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _images(shape=SMALL, batch: int = 2, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch,) + shape).astype("float32")


def _v1(**o) -> MobileNetV1:
    kwargs: Dict[str, Any] = dict(
        num_classes=NUM_CLASSES, width_multiplier=0.25, dropout_rate=0.0,
        input_shape=SMALL,
    )
    kwargs.update(o)
    return MobileNetV1(**kwargs)


def _v2(**o) -> MobileNetV2:
    kwargs: Dict[str, Any] = dict(
        num_classes=NUM_CLASSES, width_multiplier=0.25, dropout_rate=0.0,
        input_shape=V2_SHAPE,
    )
    kwargs.update(o)
    return MobileNetV2(**kwargs)


def _v3(**o) -> MobileNetV3:
    kwargs: Dict[str, Any] = dict(
        num_classes=NUM_CLASSES, variant="small", width_multiplier=0.25,
        dropout_rate=0.0, input_shape=SMALL,
    )
    kwargs.update(o)
    return MobileNetV3(**kwargs)


def _v4(**o) -> MobileNetV4:
    kwargs: Dict[str, Any] = dict(
        num_classes=NUM_CLASSES, depths=(1, 1, 1), dims=(16, 24, 32),
        block_types=("IB", "IB", "ExtraDW"), strides=(1, 2, 2),
        dropout_rate=0.0, input_shape=SMALL,
    )
    kwargs.update(o)
    return MobileNetV4(**kwargs)


#: Build seed. NOT decoration: MobileNetV3's squeeze-excite bottleneck is 2
#: channels wide at ``width_multiplier=0.25``, and on roughly one draw in eight
#: its pre-ReLU is negative in EVERY channel and EVERY batch row, so the ReLU
#: output is EXACTLY zero and both SE kernels report an identically-zero
#: gradient. Measured unseeded: 1 failed / 18 passed in 2 of 5 consecutive runs
#: of this file. An unseeded gradient report on this package reports the DRAW,
#: not the model -- exactly the hazard that made a BeiT arm in batch A flaky 1
#: in 4. ``BUILD_SEED`` is a seed at which the SE path is live;
#: ``TestTheSqueezeExciteReLUCanDieOnADraw`` pins the other side at ``6``.
BUILD_SEED = 0


def _built(build_fn, shape=SMALL, seed: int = BUILD_SEED):
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


# ---------------------------------------------------------------------------
# The trap, pinned first, because everything below depends on it
# ---------------------------------------------------------------------------
class TestTheUniformSoftmaxSaddleIsNotADeadModel:
    """D-059, pinned on this package's own head.

    Both halves are asserted, so restoring the symmetric loss anywhere in this
    file fails a test rather than quietly re-creating the false finding.
    """

    @staticmethod
    def _uniform_softmax_head() -> keras.Model:
        """A classifier whose softmax output is EXACTLY uniform.

        Not "nearly": the final ``Dense`` is zero-initialised in both kernel
        and bias, so every logit is exactly 0.0 and every class probability is
        exactly ``1/NUM_CLASSES``, which is the state an untrained MobileNet
        head approaches.
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
        """The false CRITICAL. ``default_loss`` convicts a healthy model."""
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


class TestMobileNetV2GeometryArtifact:
    """The second false-CRITICAL candidate, pinned two-sided.

    A future reader measuring four dead ``project_norm/beta`` weights on
    MobileNetV2 has this test to tell them it is the 32x32 input, not the model.
    """

    def test_at_32px_exactly_the_1x1_blocks_beta_reads_zero(self):
        model = _built(lambda: _v2(input_shape=SMALL), SMALL)
        x = _images(SMALL)
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=ramp_loss)
        dead = {p for p, v in report.items() if v is None or v == 0.0}
        assert {p.split("/", 1)[1] for p in dead} == set(
            V2_32PX_UNIFORM_SHIFT_BETAS), (
            f"expected exactly the four 1x1-stage project_norm betas, got {dead}"
        )

    def test_the_same_weights_are_live_at_64px(self):
        model = _built(_v2, V2_SHAPE)
        x = _images(V2_SHAPE)
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=ramp_loss)
        for suffix in V2_32PX_UNIFORM_SHIFT_BETAS:
            path = next(p for p in report if p.endswith(suffix))
            value = report[path]
            assert value is not None and value > 0.0, (
                f"{path} is still dead at 64x64; the 32x32 reading is then NOT "
                f"a geometry artifact and this whole explanation is wrong"
            )


class TestTheSqueezeExciteReLUCanDieOnADraw:
    """The third false-CRITICAL candidate, pinned two-sided.

    A future reader who measures ``block_0/se_reduce/kernel`` and
    ``block_0/se_expand/kernel`` dead on MobileNetV3 has this test to tell them
    it is a dying ReLU at a 2-channel bottleneck on an unlucky INITIALISATION,
    not a disconnected squeeze-excite module. It is also why every build in this
    file is seeded.
    """

    #: A seed at which the SE ReLU is fully dead. Measured 2026-08-21: of seeds
    #: 0..7 exactly this one, matching the 2-in-5 unseeded failure rate.
    DEAD_SEED = 6

    @staticmethod
    def _se_block(model):
        return {l.name: l for l in model._flatten_layers(include_self=False)}[
            "block_0"]

    def test_the_se_bottleneck_is_only_two_channels_wide(self):
        """The premise: a 2-wide ReLU is a plausible thing to lose entirely."""
        model = _built(_v3)
        assert self._se_block(model).se_reduce.filters == 2

    def test_at_the_unlucky_seed_the_se_relu_output_is_exactly_zero(self):
        model = _built(_v3, seed=self.DEAD_SEED)
        block = self._se_block(model)
        seen = {}
        original = block.se_activation1.call

        def spy(inputs, *args, **kwargs):
            result = original(inputs, *args, **kwargs)
            seen["post"] = keras.ops.convert_to_numpy(result)
            seen["pre"] = keras.ops.convert_to_numpy(inputs)
            return result

        block.se_activation1.call = spy
        try:
            model(_images(), training=True)
        finally:
            block.se_activation1.call = original

        assert seen["pre"].max() < 0.0, (
            f"the SE pre-activation is not entirely negative "
            f"({seen['pre'].min()} .. {seen['pre'].max()}); the explanation "
            f"below does not apply at this seed"
        )
        assert np.all(seen["post"] == 0.0), "the SE ReLU output is not all zero"

    def test_at_that_seed_exactly_the_two_se_kernels_read_zero(self):
        model = _built(_v3, seed=self.DEAD_SEED)
        x = _images()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=ramp_loss)
        dead = {p for p, v in report.items() if v is None or v == 0.0}
        assert {p.split("/", 1)[1] for p in dead} == {
            "block_0/se_reduce/kernel", "block_0/se_expand/kernel"}, (
            f"expected exactly the two SE kernels, got {sorted(dead)}"
        )

    def test_at_the_build_seed_they_are_live(self):
        """The discriminating half: the SE module is NOT disconnected."""
        model = _built(_v3, seed=BUILD_SEED)
        x = _images()
        _one_adam_step(model, x)
        report = gradient_report(model, x, loss_fn=ramp_loss)
        for suffix in ("block_0/se_reduce/kernel", "block_0/se_expand/kernel"):
            path = next(p for p in report if p.endswith(suffix))
            assert report[path] is not None and report[path] > 0.0, (
                f"{suffix} is dead at the build seed too; the draw explanation "
                f"is then wrong and this IS a disconnected module"
            )


class TestMobileNetGradientFlow:

    def test_no_layer_is_stochastic(self):
        """Every dropout rate is pinned to 0.0 across all four generations."""
        for name, build, shape in (
            ("v1", _v1, SMALL), ("v2", _v2, V2_SHAPE),
            ("v3", _v3, SMALL), ("v4", _v4, SMALL),
        ):
            model = _built(build, shape)
            stochastic = [
                (layer.name, attr, getattr(layer, attr))
                for layer in model._flatten_layers(include_self=False)
                for attr in ("rate", "drop_path_rate", "dropout_rate")
                if isinstance(getattr(layer, attr, None), float)
                and getattr(layer, attr) > 0.0
            ]
            assert stochastic == [], (
                f"{name}: a non-zero stochastic rate is live: {stochastic}"
            )

    @pytest.mark.parametrize(
        "name,build,shape",
        [("v1", _v1, SMALL), ("v2", _v2, V2_SHAPE),
         ("v3", _v3, SMALL), ("v4", _v4, SMALL)],
    )
    def test_gradients_reach_every_trainable_weight_after_one_step(
            self, name, build, shape):
        model = _built(build, shape)
        x = _images(shape)
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS[name] == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        model = _built(_v1)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _images(), loss_fn=ramp_loss)


class TestMobileNetKnobSensitivity:

    def test_width_multiplier_changes_the_parameterisation(self):
        builders = {
            w: (lambda w=w: _built(lambda: _v1(width_multiplier=w)))
            for w in (0.25, 0.5, 1.0)
        }
        assert_structural_knob_changes_weights(builders, knob="width_multiplier")

    def test_v3_variant_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _built(
                lambda: MobileNetV3(num_classes=NUM_CLASSES, variant=v,
                                    width_multiplier=0.25, dropout_rate=0.0,
                                    input_shape=SMALL)))
            for v in ("small", "large")
        }
        assert_structural_knob_changes_weights(builders, knob="variant")

    def test_v4_block_types_change_the_parameterisation(self):
        """A knob that reaches ONLY the block construction.

        ``depths`` would still pass if every block type built the same layers.
        This one would not.
        """
        builders = {
            t: (lambda t=t: _built(lambda: _v4(block_types=t)))
            for t in (("IB", "IB", "IB"), ("IB", "IB", "ExtraDW"),
                      ("ExtraDW", "ExtraDW", "ExtraDW"))
        }
        assert_structural_knob_changes_weights(builders, knob="block_types")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built(_v1)), "b": (lambda: _built(_v1))}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(
                builders, knob="width_multiplier")


class TestMobileNetSmokeContract:

    @pytest.mark.parametrize(
        "build,shape",
        [(_v1, SMALL), (_v2, V2_SHAPE), (_v3, SMALL), (_v4, SMALL)],
    )
    def test_the_forward_contract_rejects_a_broken_forward(self, build, shape):
        model = _built(build, shape)
        x = _images(shape)

        def contract(out):
            assert not isinstance(out, (dict, list, tuple)), (
                f"a MobileNet with include_top=True returns one tensor, got "
                f"{type(out)}"
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
