"""RED proof for ``v2_compliance_oracle``: every helper has been seen to REJECT.

Guide §13.7.1 places a shared instrument's RED proof in a mirrored ``test_<name>.py``, and
§16.3's last-but-two item requires "every guard proven RED by an injection, **in the
committed record**". An instrument that has never rejected anything is indistinguishable
from one that cannot reject anything; three assertions in this repo's history were green
for their whole lifetime while the thing they watched was broken.

Each test below feeds one helper a deliberately broken subject built around a REAL defect
mechanism, not a hand-thrown exception:

============================================ ================================================
Helper                                       Injected defect
============================================ ================================================
``assert_value_round_trip``                  ``get_config`` omits a value knob
``assert_weights_restored_before_first_call``  ``load_own_variables`` silently skips restore
``assert_weights_restored_before_first_call``  no weights at all (anti-vacuity)
``assert_build_parity``                      a weight created lazily in ``call``
``assert_gradients_reach_...``               a trainable weight on a dead branch
``assert_gradients_reach_...``               no trainable variables (anti-vacuity)
``assert_forward_is_finite``                 a NaN in the output
``assert_eager_matches_jit``                 a value that differs between eager and XLA
``assert_value_knob_moves_output_not_shapes``  a knob that is never read
``DtypePolicyScope``                         the ``float64`` arm really moves ``floatx``
============================================ ================================================
"""

from typing import Any, Optional

import keras
import numpy as np
import pytest
from keras import ops

from .v2_compliance_oracle import (
    DtypePolicyScope,
    assert_build_parity,
    assert_eager_matches_jit,
    assert_forward_is_finite,
    assert_gradients_reach_every_trainable_weight,
    assert_value_knob_moves_output_not_shapes,
    assert_value_round_trip,
    assert_weights_restored_before_first_call,
    relative_weight_paths,
)

_PACKAGE = "dl_techniques.tests.layers.statistics.v2_oracle_red_proof"


# ---------------------------------------------------------------------
# broken subjects, each modelling one real defect class
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package=_PACKAGE)
class IncompleteConfigLayer(keras.layers.Layer):
    """``get_config`` omits ``scale``, so a reload silently falls back to the default.

    Guide Pitfall 4 / §6.1. The reloaded layer has the same shapes and the same weight
    count; only the VALUES differ, which is exactly the failure a shape-only round trip
    cannot see.
    """

    def __init__(self, scale: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs * self.scale

    def get_config(self) -> dict:
        config = super().get_config()
        # The injected defect. Note it must be an explicit POP: Keras 3.8 auto-captures
        # `__init__` keyword arguments into `get_config`, so simply not writing a
        # `get_config` no longer reproduces the classic incomplete-config defect.
        config.pop("scale", None)
        return config


@keras.saving.register_keras_serializable(package=_PACKAGE)
class SkipsWeightRestoreLayer(keras.layers.Layer):
    """Overrides ``load_own_variables`` to a no-op, so saved values are never restored.

    The reloaded layer keeps its freshly initialized weights. The weight COUNT and every
    shape match, so only the ``atol=0.0`` VALUE comparison of §8.4 can see it.
    """

    def build(self, input_shape: Any) -> None:
        self.w = self.add_weight(
            name="w",
            shape=(int(input_shape[-1]),),
            # UNSEEDED on purpose: a seeded initializer would re-create the exact
            # saved values on reload, and the skipped restore would be invisible.
            initializer=keras.initializers.RandomNormal(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs * self.w

    def load_own_variables(self, store: Any) -> None:
        return  # the injected defect


@keras.saving.register_keras_serializable(package=_PACKAGE)
class NoWeightsLayer(keras.layers.Layer):
    """A layer with no weights at all, for the ``assert saved`` anti-vacuity arm."""

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs * 2.0


@keras.saving.register_keras_serializable(package=_PACKAGE)
class UnderBuildingParent(keras.layers.Layer):
    """``build()`` does not materialize the sub-layer tree ``call()`` runs.

    The §8.1 under-build defect verbatim: the child is created in ``__init__`` (correct),
    but ``build()`` never builds it, so an explicitly built instance holds ZERO weights
    while a lazily built one holds the child's kernel and bias. Nothing raises; a reloaded
    model would restore into nothing and fill the gap with fresh random weights on its
    first forward pass.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = keras.layers.Dense(3, name="child")

    def build(self, input_shape: Any) -> None:
        super().build(input_shape)  # the injected defect: self.child is never built

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return self.child(inputs)


@keras.saving.register_keras_serializable(package=_PACKAGE)
class WellBuiltParent(UnderBuildingParent):
    """The positive control: identical, except ``build()`` builds the child."""

    def build(self, input_shape: Any) -> None:
        self.child.build(input_shape)
        super(UnderBuildingParent, self).build(input_shape)


@keras.saving.register_keras_serializable(package=_PACKAGE)
class DeadBranchLayer(keras.layers.Layer):
    """Two trainable weights; ``call`` reads only one. The other gets a ``None`` gradient."""

    def build(self, input_shape: Any) -> None:
        units = int(input_shape[-1])
        self.live = self.add_weight(
            name="live", shape=(units,), initializer="ones", trainable=True
        )
        self.dead = self.add_weight(
            name="dead", shape=(units,), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs * self.live


@keras.saving.register_keras_serializable(package=_PACKAGE)
class FrozenOnlyLayer(keras.layers.Layer):
    """Its single weight is ``trainable=False``, so ``trainable_variables`` is empty."""

    def build(self, input_shape: Any) -> None:
        self.w = self.add_weight(
            name="w",
            shape=(int(input_shape[-1]),),
            initializer="ones",
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs * self.w


@keras.saving.register_keras_serializable(package=_PACKAGE)
class NanLayer(keras.layers.Layer):
    """Divides by zero on one column, producing a NaN a shape assertion cannot see."""

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        zero = ops.zeros_like(inputs)
        return ops.divide(zero, zero)


@keras.saving.register_keras_serializable(package=_PACKAGE)
class IgnoresItsKnobLayer(keras.layers.Layer):
    """Stores ``gain`` and never reads it -- the §12.5 dead knob."""

    def __init__(self, gain: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gain = gain

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        return inputs + 1.0


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_input() -> np.ndarray:
    return np.random.default_rng(1234).normal(size=(4, 6)).astype("float32")


# ---------------------------------------------------------------------
# the proofs
# ---------------------------------------------------------------------


class TestTheRoundTripOracleCanReject:
    def test_the_round_trip_oracle_rejects_a_layer_whose_config_drops_a_knob(
        self, sample_input
    ):
        """A reload that silently restores ``scale=1.0`` instead of ``3.0`` must FAIL."""
        with pytest.raises(AssertionError, match="differs after a .keras round trip"):
            assert_value_round_trip(
                lambda: IncompleteConfigLayer(scale=3.0), sample_input, atol=1e-6
            )

    def test_the_round_trip_oracle_accepts_the_same_layer_when_the_knob_is_neutral(
        self, sample_input
    ):
        """The positive arm: at ``scale=1.0`` the dropped knob is invisible and it PASSES.

        Without this arm the rejection above is also satisfied by an oracle that raises on
        everything (§13.2.4: every "nothing moved" assertion is half of a pair).
        """
        assert_value_round_trip(
            lambda: IncompleteConfigLayer(scale=1.0), sample_input, atol=1e-6
        )


class TestTheWeightRestorationOracleCanReject:
    def test_it_rejects_a_layer_that_skips_its_own_weight_restore(
        self, sample_input, tmp_path
    ):
        """Same count, same shapes, different values -- only ``atol=0.0`` sees it."""

        def factory():
            inputs = keras.Input(shape=(6,))
            return keras.Model(inputs, SkipsWeightRestoreLayer()(inputs))

        with pytest.raises(AssertionError, match="was not restored"):
            assert_weights_restored_before_first_call(factory, sample_input, tmp_path)

    def test_its_anti_vacuity_check_fires_on_a_model_with_no_weights(
        self, sample_input, tmp_path
    ):
        """With no weights the value loop iterates zero times and would report green."""

        def factory():
            inputs = keras.Input(shape=(6,))
            return keras.Model(inputs, NoWeightsLayer()(inputs))

        with pytest.raises(AssertionError, match="no weights to compare"):
            assert_weights_restored_before_first_call(factory, sample_input, tmp_path)

    def test_it_accepts_an_honest_layer(self, sample_input, tmp_path):
        """Positive arm: a Dense really does restore, at ``atol=0.0``."""

        def factory():
            inputs = keras.Input(shape=(6,))
            return keras.Model(inputs, keras.layers.Dense(3, name="d")(inputs))

        assert_weights_restored_before_first_call(factory, sample_input, tmp_path)


class TestTheBuildParityOracleCanReject:
    def test_it_rejects_a_layer_that_creates_a_weight_in_call(self, sample_input):
        with pytest.raises(AssertionError, match="different weight trees"):
            assert_build_parity(UnderBuildingParent, (None, 6), sample_input)

    def test_it_accepts_a_parent_that_builds_the_tree_call_runs(self, sample_input):
        """The positive control differs from the rejected subject by ONE line of build()."""
        assert_build_parity(WellBuiltParent, (None, 6), sample_input)

    def test_relative_weight_paths_strips_the_instance_root(self, sample_input):
        """Two instances of the same builder must compare equal after stripping."""
        a, b = keras.layers.Dense(3), keras.layers.Dense(3)
        a(sample_input)
        b(sample_input)
        assert [w.path for w in a.weights] != [w.path for w in b.weights]
        assert relative_weight_paths(a) == relative_weight_paths(b)


class TestTheGradientOracleCanReject:
    def test_it_rejects_a_layer_with_a_weight_on_a_dead_branch(self, sample_input):
        """The failure must NAME the variable, not just say 'a gradient was None'."""
        layer = DeadBranchLayer()
        with pytest.raises(AssertionError, match=r"no gradient for .*dead"):
            assert_gradients_reach_every_trainable_weight(layer, sample_input)

    def test_its_anti_vacuity_check_fires_when_nothing_is_trainable(self, sample_input):
        layer = FrozenOnlyLayer()
        layer(sample_input)
        with pytest.raises(AssertionError, match="no trainable variables"):
            assert_gradients_reach_every_trainable_weight(layer, sample_input)

    def test_it_rejects_an_all_zero_gradient_even_though_the_gradient_exists(self):
        """Non-``None`` is not enough: §13.2.2 measured 61 of 61 weights at exactly zero.

        ``inputs * 0.0 * w`` keeps ``w`` in the graph, so the gradient is a real tensor of
        zeros rather than ``None`` -- the case a ``grad is not None`` guard cannot see.
        """

        @keras.saving.register_keras_serializable(package=_PACKAGE)
        class ZeroGradientLayer(keras.layers.Layer):
            def build(self, input_shape):
                self.w = self.add_weight(
                    name="w", shape=(int(input_shape[-1]),), initializer="ones"
                )
                super().build(input_shape)

            def call(self, inputs, training=None):
                return inputs + 0.0 * self.w

        x = np.zeros((4, 6), dtype="float32")
        with pytest.raises(AssertionError, match=r"all-zero gradient for .*w"):
            assert_gradients_reach_every_trainable_weight(ZeroGradientLayer(), x)

    def test_it_accepts_a_healthy_layer(self, sample_input):
        assert_gradients_reach_every_trainable_weight(
            keras.layers.Dense(3), sample_input
        )


class TestTheFinitenessOracleCanReject:
    def test_it_rejects_a_nan_output(self, sample_input):
        y = NanLayer()(sample_input)
        with pytest.raises(AssertionError, match="is not finite"):
            assert_forward_is_finite(y)

    def test_it_rejects_an_infinity_hidden_inside_a_tuple(self, sample_input):
        """A subject returning ``(z, log_det)`` must be checked on BOTH members."""
        good = ops.convert_to_tensor(sample_input)
        bad = ops.convert_to_tensor(np.full((4,), np.inf, dtype="float32"))
        with pytest.raises(AssertionError, match=r"output 1 is not finite"):
            assert_forward_is_finite((good, bad))

    def test_it_accepts_a_finite_output(self, sample_input):
        assert_forward_is_finite(ops.convert_to_tensor(sample_input))


class TestTheXlaEquivalenceOracleCanReject:
    def test_it_rejects_a_layer_whose_traced_value_differs(self, sample_input):
        """``tf.function`` folds Python state at trace time -- the §11.2 defect.

        The counter advances on the eager call and then FREEZES inside the trace, so the
        two paths return different values while both stay finite and correctly shaped.
        """

        class TraceDivergentLayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.calls = 0.0

            def call(self, inputs, training=None):
                self.calls += 1.0
                return inputs + self.calls

        with pytest.raises(AssertionError, match="eager and jit_compile=True disagree"):
            assert_eager_matches_jit(TraceDivergentLayer(), sample_input)

    def test_it_accepts_an_xla_safe_layer(self, sample_input):
        assert_eager_matches_jit(keras.layers.Dense(3), sample_input, atol=1e-5)


class TestTheValueKnobOracleCanReject:
    def test_it_rejects_a_knob_that_is_never_read(self, sample_input):
        builders = {g: (lambda g=g: IgnoresItsKnobLayer(gain=g)) for g in (0.5, 4.0)}
        with pytest.raises(AssertionError, match="changed nothing in the output"):
            assert_value_knob_moves_output_not_shapes(builders, sample_input)

    def test_it_rejects_a_structural_knob_wearing_a_value_knob_test(self, sample_input):
        """Sweeping ``units`` changes the weight SHAPES, which a value knob must not."""
        builders = {u: (lambda u=u: keras.layers.Dense(u)) for u in (3, 5)}
        with pytest.raises(AssertionError, match="must not change the weight shapes"):
            assert_value_knob_moves_output_not_shapes(builders, sample_input)

    def test_it_accepts_a_live_value_knob(self, sample_input):
        builders = {
            a: (lambda a=a: keras.layers.Dense(3, activation=a))
            for a in ("relu", "sigmoid")
        }
        assert_value_knob_moves_output_not_shapes(builders, sample_input)


class TestTheDtypePolicyScopeRestoresAndReallyMovesFloatx:
    def test_the_float64_arm_moves_floatx_not_just_the_policy(self):
        """§13.2.6: the policy alone leaves ``keras.Input`` on float32.

        Without the ``set_floatx`` half this arm is "a fake reading that agrees with
        float32 to eight digits", so the scope is asserted to move BOTH.
        """
        before = keras.backend.floatx()
        with DtypePolicyScope("float64"):
            assert keras.mixed_precision.global_policy().name == "float64"
            assert keras.backend.floatx() == "float64"
            assert keras.Input(shape=(3,)).dtype == "float64"
        assert keras.backend.floatx() == before

    def test_it_restores_the_policy_even_when_the_body_raises(self):
        before = keras.mixed_precision.global_policy().name
        with pytest.raises(RuntimeError):
            with DtypePolicyScope("mixed_float16"):
                raise RuntimeError("boom")
        assert keras.mixed_precision.global_policy().name == before
