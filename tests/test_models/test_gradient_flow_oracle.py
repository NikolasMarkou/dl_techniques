"""
RED proofs for ``tests/test_models/gradient_flow_oracle.py``.

An instrument that has never been shown to fail is the same defect one layer up
as the ``assert all(g is not None ...)`` test it replaces. Every failure mode the
oracle claims to convict is built here as a deliberate dead-component fixture,
paired with a HEALTHY TWIN that differs only in the defect, and the oracle is
required to raise on the first and stay silent on the second.

The three fixtures mirror three defect shapes actually measured in this tree:

``_DisconnectedModel``
    a sublayer constructed and BUILT in ``build()`` but never called -- the
    F-17 ``Sam3ViTDetBackbone`` shape. Its weights are in ``trainable_weights``,
    they are saved to and loaded from the checkpoint, and they receive ``None``.

``_ZeroGatedModel``
    a branch multiplied by a zero-initialized gate -- the darkir ``beta``/
    ``gamma`` shape (F-62). Nothing is disconnected: every gradient is a real
    tensor of the right shape, and the branch's is exactly 0.0. This is the case
    ``assert all(g is not None)`` cannot see, and it is why this file exists.

``_nan_kernel_model``
    a NaN weight in one of two parallel branches, so the report has to
    discriminate rather than fail wholesale.
"""

import keras
import numpy as np
import pytest

from .gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
)

X = np.random.default_rng(0).random((2, 4)).astype("float32")


def _healthy_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(5, activation="relu", name="hidden"),
            keras.layers.Dense(3, name="head"),
        ],
        name="healthy",
    )
    model.build((None, 4))
    return model


class _DisconnectedModel(keras.Model):
    """``orphan`` is built, trainable, serialized -- and never executed."""

    def __init__(self, disconnect: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.disconnect = disconnect
        self.main = keras.layers.Dense(3, name="main")
        self.orphan = keras.layers.Dense(3, name="orphan")

    def build(self, input_shape):
        self.main.build(input_shape)
        self.orphan.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        out = self.main(inputs)
        if not self.disconnect:
            out = out + self.orphan(inputs)
        return out


class _ZeroGatedModel(keras.Model):
    """``branch`` is gated by a scalar whose initializer is zeros."""

    def __init__(self, gate_init: str = "zeros", **kwargs):
        super().__init__(**kwargs)
        self.gate_init = gate_init
        self.trunk = keras.layers.Dense(3, name="trunk")
        self.branch = keras.layers.Dense(3, name="branch")

    def build(self, input_shape):
        self.trunk.build(input_shape)
        self.branch.build(input_shape)
        self.gate = self.add_weight(
            name="gate", shape=(1,), initializer=self.gate_init, trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        return self.trunk(inputs) + self.gate * self.branch(inputs)


def _nan_kernel_model() -> keras.Model:
    """Two parallel heads; the first one's kernel is set to NaN after build."""
    inp = keras.layers.Input(shape=(4,))
    poisoned = keras.layers.Dense(3, name="poisoned")(inp)
    clean = keras.layers.Dense(3, name="clean")(inp)
    model = keras.Model(inp, keras.layers.Concatenate()([poisoned, clean]), name="nan")
    kernel = model.get_layer("poisoned").kernel
    kernel.assign(np.full(kernel.shape, np.nan, dtype="float32"))
    return model


class TestHealthyModelIsSilent:
    """Liveness arm: the oracle must not fire on a model that is fine."""

    def test_sequential_model_passes(self):
        report = assert_gradients_reach_every_trainable_weight(_healthy_model(), X)
        assert len(report) == 4
        assert all(v is not None and v > 0.0 for v in report.values())

    def test_connected_twin_of_the_disconnected_fixture_passes(self):
        model = _DisconnectedModel(disconnect=False)
        model.build((None, 4))
        report = assert_gradients_reach_every_trainable_weight(model, X)
        assert any("orphan" in p for p in report)

    def test_nonzero_gated_twin_passes(self):
        model = _ZeroGatedModel(gate_init="ones")
        model.build((None, 4))
        assert_gradients_reach_every_trainable_weight(model, X)


class TestDisconnectedSubgraphIsCaught:
    """Failure mode 1: a weight that receives no gradient at all."""

    def test_oracle_raises_naming_the_orphan(self):
        model = _DisconnectedModel(disconnect=True)
        model.build((None, 4))
        with pytest.raises(AssertionError) as excinfo:
            assert_gradients_reach_every_trainable_weight(model, X)
        message = str(excinfo.value)
        assert "NO gradient" in message
        assert "orphan" in message
        # Discriminating, not wholesale: the live branch must not be named.
        assert "/main/" not in message

    def test_the_weak_assertion_this_replaces_would_also_have_caught_it(self):
        """``g is not None`` DOES catch this one -- which is why fixture 2 exists."""
        model = _DisconnectedModel(disconnect=True)
        model.build((None, 4))
        report = gradient_report(model, X)
        assert [p for p, v in report.items() if v is None]


class TestExactlyZeroGradientIsCaught:
    """Failure mode 2: the dead-component case ``g is not None`` cannot see."""

    def test_oracle_raises_naming_the_gated_branch(self):
        model = _ZeroGatedModel(gate_init="zeros")
        model.build((None, 4))
        with pytest.raises(AssertionError) as excinfo:
            assert_gradients_reach_every_trainable_weight(model, X)
        message = str(excinfo.value)
        assert "identically-zero" in message
        assert "branch" in message
        assert "trunk" not in message

    def test_the_assertion_this_replaces_reports_green_on_the_same_model(self):
        """The measured reason the oracle exists.

        Every gradient here is a real tensor of the correct shape, so both
        legacy shapes -- ``all(g is not None)`` and ``all(norm >= 0.0)`` -- pass
        on a model whose whole ``branch`` sublayer is untrainable.
        """
        model = _ZeroGatedModel(gate_init="zeros")
        model.build((None, 4))
        report = gradient_report(model, X)
        assert all(v is not None for v in report.values())      # legacy shape 1
        assert all(v >= 0.0 for v in report.values())            # legacy shape 2
        zero = [p for p, v in report.items() if v == 0.0]
        assert any("branch" in p for p in zero), zero


class TestNonFiniteGradientIsCaught:
    """Failure mode 3: NaN/Inf."""

    def test_oracle_raises_naming_the_poisoned_head(self):
        model = _nan_kernel_model()
        with pytest.raises(AssertionError) as excinfo:
            assert_gradients_reach_every_trainable_weight(model, X)
        message = str(excinfo.value)
        assert "non-finite" in message
        assert "poisoned" in message

    def test_clean_head_is_not_convicted(self):
        report = gradient_report(_nan_kernel_model(), X)
        clean = {p: v for p, v in report.items() if "clean" in p}
        assert clean and all(v is not None and not np.isnan(v) for v in clean.values())


class TestExpectZeroIsTwoSided:
    """The waiver must be a pinned claim, not a skip list."""

    def test_waiving_the_gated_branch_makes_the_oracle_silent(self):
        model = _ZeroGatedModel(gate_init="zeros")
        model.build((None, 4))
        report = assert_gradients_reach_every_trainable_weight(
            model, X, expect_zero=("branch/",)
        )
        assert all(report[p] == 0.0 for p in report if "branch/" in p)

    def test_waiving_a_weight_that_does_learn_is_an_error(self):
        """An obsolete waiver is a lie about the model and must fail loudly."""
        model = _ZeroGatedModel(gate_init="ones")
        model.build((None, 4))
        with pytest.raises(AssertionError, match="waiver is obsolete"):
            assert_gradients_reach_every_trainable_weight(
                model, X, expect_zero=("branch/",)
            )

    def test_a_pattern_matching_no_weight_is_an_error(self):
        with pytest.raises(AssertionError, match="stale waiver"):
            assert_gradients_reach_every_trainable_weight(
                _healthy_model(), X, expect_zero=("renamed_away/",)
            )

    def test_expect_zero_also_covers_the_disconnected_case(self):
        """``None`` and all-zero are the same statement to the optimizer."""
        model = _DisconnectedModel(disconnect=True)
        model.build((None, 4))
        assert_gradients_reach_every_trainable_weight(
            model, X, expect_zero=("orphan/",)
        )


class TestVacuityGuards:
    def test_a_model_with_no_trainable_weights_raises(self):
        model = keras.Sequential([keras.layers.Input(shape=(4,)), keras.layers.ReLU()])
        with pytest.raises(ValueError, match="no trainable weights"):
            gradient_report(model, X)

    def test_default_loss_skips_integer_outputs(self):
        outputs = {
            "logits": keras.ops.ones((2, 3)),
            "token_ids": keras.ops.cast(keras.ops.ones((2, 3)), "int32"),
        }
        assert float(keras.ops.convert_to_numpy(default_loss(outputs))) == 1.0

    def test_default_loss_raises_when_nothing_is_differentiable(self):
        outputs = keras.ops.cast(keras.ops.ones((2, 3)), "int32")
        with pytest.raises(ValueError, match="no floating-point tensor"):
            default_loss(outputs)

    def test_a_loss_that_never_reaches_the_model_is_convicted_not_green(self):
        """The general shape of the from-logits trap.

        The measured ``test_bert`` defect was a
        ``categorical_crossentropy(from_logits=False)`` on logits: Keras clips
        to ``[eps, 1-eps]``, the loss is constant in that region, and 61 of 61
        gradients came back identically zero while the suite reported green.
        The detachment is modelled here with ``stop_gradient`` because the
        clipped region cannot be entered deterministically from a random-init
        head; the claim under test is the same one -- when the loss does not
        reach the parameters, this oracle must convict the WHOLE model rather
        than report success.
        """
        model = _healthy_model()
        targets = keras.ops.one_hot(keras.ops.array([0, 1]), 3)
        detached = lambda out: keras.ops.mean(  # noqa: E731
            keras.losses.categorical_crossentropy(
                targets, keras.ops.stop_gradient(out), from_logits=True
            )
        )
        with pytest.raises(AssertionError) as excinfo:
            assert_gradients_reach_every_trainable_weight(model, X, loss_fn=detached)
        assert "0/4 trainable weights receive a live gradient" in str(excinfo.value)
