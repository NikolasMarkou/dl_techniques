"""
Tests for SoftValueRangeConstraint

This module contains comprehensive tests for the SoftValueRangeConstraint class:
- Initialization with default and custom parameters
- Direct application to a weight tensor, two-sided and one-sided
- Parameter validation, cross-checked against the shared core function
- The `enforce_hard_bounds` flag: that it is observable, and what it costs
- Serialization and deserialization of all five parameters
- Model integration: a `.keras` save/load round trip and a WGAN-style critic

The class is a thin role adapter over
`dl_techniques.layers.activations.soft_value_range`; the numerics of the map itself
are tested in `tests/test_layers/test_activations/test_soft_value_range.py`. What is
tested here is the CONSTRAINT behaviour: the projection a Keras optimizer actually
applies, and the exact-bound guard that exists only in this wrapper.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.logger import logger

from dl_techniques.constraints.soft_value_range_constraint import (
    SoftValueRangeConstraint,
)
from dl_techniques.layers.activations.soft_value_range import soft_value_range
from tests.optimizer_state import build_optimizer_state


# ---------------------------------------------------------------------
# Parameter sets that every validating entry point must reject.
# Shared by the constraint's own validation test and by the cross-check that pins
# it against the core function, so the two can never be edited apart.
# ---------------------------------------------------------------------

INVALID_PARAMS = [
    # (kwargs, substrings that must appear in the message)
    ({"min_value": 1.0, "max_value": -1.0}, ["1.0", "-1.0"]),
    ({"min_value": 0.0, "max_value": 1.0, "sharpness": 0.0}, ["sharpness", "0.0"]),
    ({"min_value": 0.0, "max_value": 1.0, "sharpness": -3.5}, ["sharpness", "-3.5"]),
    ({"min_value": 0.0, "sharpness": -1.0}, ["sharpness", "-1.0"]),
]


class TestSoftValueRangeConstraint:
    """Test suite for SoftValueRangeConstraint implementation."""

    @pytest.fixture
    def sample_weights(self) -> keras.KerasTensor:
        """Create a sample weight tensor spanning well outside any test range.

        Returns:
            keras.KerasTensor: Deterministic weights in [-2, 2], shape (10, 5).
        """
        rng = np.random.default_rng(1234)
        weights = rng.uniform(-2.0, 2.0, size=(10, 5))
        return keras.ops.cast(weights, dtype="float32")

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        constraint = SoftValueRangeConstraint(min_value=0.0)

        assert constraint.min_value == 0.0
        assert constraint.max_value is None
        assert constraint.sharpness == 50.0
        assert constraint.relative_sharpness is True
        assert constraint.enforce_hard_bounds is True

    def test_initialization_custom(self) -> None:
        """Test initialization with every parameter set away from its default."""
        constraint = SoftValueRangeConstraint(
            min_value=-2.5,
            max_value=3.5,
            sharpness=7.25,
            relative_sharpness=False,
            enforce_hard_bounds=False,
        )

        assert constraint.min_value == -2.5
        assert constraint.max_value == 3.5
        assert constraint.sharpness == 7.25
        assert constraint.relative_sharpness is False
        assert constraint.enforce_hard_bounds is False

    def test_no_clip_gradients_parameter(self) -> None:
        """The inert `clip_gradients` wart of ValueRangeConstraint is NOT propagated.

        `ValueRangeConstraint.clip_gradients` is stored, serialized and repr'd but
        never read; its own suite asserts both values give identical output. This
        class must reject the name rather than accept it as a no-op.
        """
        with pytest.raises(TypeError):
            SoftValueRangeConstraint(min_value=0.0, clip_gradients=False)

    def test_direct_call_projects_into_the_range(
            self, sample_weights: keras.KerasTensor
    ) -> None:
        """Direct `__call__` maps every weight into [min_value, max_value].

        With the default `enforce_hard_bounds=True` the bounds are exact, so this is
        an equality-grade assertion, not a tolerance one.
        """
        constraint = SoftValueRangeConstraint(min_value=-0.5, max_value=0.5)
        constrained = np.asarray(constraint(sample_weights))

        assert constrained.shape == (10, 5)
        assert np.all(np.isfinite(constrained))
        assert np.all(constrained >= np.float32(-0.5))
        assert np.all(constrained <= np.float32(0.5))

    def test_direct_call_is_not_the_identity(
            self, sample_weights: keras.KerasTensor
    ) -> None:
        """The projection actually moves the out-of-range weights.

        Anti-vacuity twin of the feasibility test above: a `__call__` that returned
        its input unchanged would satisfy nothing here. The fixture spans [-2, 2] and
        the box is [-0.5, 0.5], so most entries must move, and by a lot.
        """
        constraint = SoftValueRangeConstraint(min_value=-0.5, max_value=0.5)
        original = np.asarray(sample_weights)
        constrained = np.asarray(constraint(sample_weights))

        outside = np.abs(original) > 0.5
        assert outside.sum() > 20, "fixture no longer exercises the out-of-range path"
        assert np.max(np.abs(constrained - original)) > 1.0

    def test_direct_call_is_not_a_hard_clip(self) -> None:
        """The whole point: outside the box this is not `keras.ops.clip`.

        Two weights at different distances beyond the same bound must land on
        different values. A hard clip collapses them onto the bound; the smooth map
        preserves their order. Measured in-regime -- `sharpness=5` on [-1, 1] gives
        `beta = 2.5`, so the gap outside the bound stays far above float32 spacing at
        the probe distances used here.
        """
        constraint = SoftValueRangeConstraint(
            min_value=-1.0, max_value=1.0, sharpness=5.0
        )
        weights = keras.ops.convert_to_tensor(
            np.array([[1.2, 1.5, 2.0, -1.2, -1.5, -2.0]], dtype="float32")
        )
        y = np.asarray(constraint(weights))
        clipped = np.asarray(keras.ops.clip(weights, -1.0, 1.0))

        # The hard clip maps all three positive probes onto exactly 1.0.
        assert np.all(clipped[0, :3] == np.float32(1.0))
        # The smooth map keeps them apart and strictly ordered.
        assert y[0, 0] < y[0, 1] < y[0, 2]
        assert y[0, 3] < y[0, 4] < y[0, 5] or y[0, 3] > y[0, 4] > y[0, 5]
        assert not np.allclose(y, clipped)

    def test_direct_call_one_sided(self, sample_weights: keras.KerasTensor) -> None:
        """One-sided mode applies a smooth floor and no ceiling."""
        constraint = SoftValueRangeConstraint(
            min_value=0.0, sharpness=10.0, relative_sharpness=False
        )
        constrained = np.asarray(constraint(sample_weights))
        original = np.asarray(sample_weights)

        assert np.all(np.isfinite(constrained))
        assert np.all(constrained >= np.float32(0.0))
        # No ceiling: the largest positive weight is essentially untouched.
        assert constrained.max() > 1.0
        assert np.isclose(constrained.max(), original.max(), atol=1e-3)

    def test_direct_call_preserves_dtype_and_shape(self) -> None:
        """`__call__` returns the same shape and dtype it was handed."""
        constraint = SoftValueRangeConstraint(min_value=-1.0, max_value=1.0)
        weights = keras.ops.cast(np.zeros((3, 4, 5)), dtype="float32")
        out = constraint(weights)

        assert keras.ops.shape(out) == (3, 4, 5)
        assert keras.backend.standardize_dtype(out.dtype) == "float32"


class TestValidation:
    """Parameter validation, and its agreement with the shared core function."""

    @pytest.mark.parametrize("kwargs,expected_substrings", INVALID_PARAMS)
    def test_invalid_parameters_raise_naming_the_value(
            self, kwargs: dict, expected_substrings: list
    ) -> None:
        """Every invalid parameter set raises ValueError NAMING the offending value."""
        with pytest.raises(ValueError) as excinfo:
            SoftValueRangeConstraint(**kwargs)

        message = str(excinfo.value)
        for substring in expected_substrings:
            assert substring in message, (
                f"message {message!r} does not name {substring!r}"
            )

    @pytest.mark.parametrize("kwargs,expected_substrings", INVALID_PARAMS)
    def test_validation_agrees_with_the_core_function(
            self, kwargs: dict, expected_substrings: list
    ) -> None:
        """The constraint rejects exactly what `soft_value_range` rejects.

        The constraint validates in `__init__` rather than importing the activations
        module's private `_validated_bounds` (decisions.md D-003), so the two checks
        are a deliberate copy. This test is the mechanical lockstep guard that makes
        the copy safe: editing one side's predicate without the other reddens here,
        rather than leaving two entry points quietly disagreeing about what is legal.
        """
        x = keras.ops.convert_to_tensor(np.zeros((2, 2), dtype="float32"))

        with pytest.raises(ValueError):
            SoftValueRangeConstraint(**kwargs)

        with pytest.raises(ValueError):
            soft_value_range(x, **kwargs)

    def test_equal_bounds_are_accepted(self) -> None:
        """`min_value == max_value` is degenerate but legal: a constant projection."""
        constraint = SoftValueRangeConstraint(min_value=0.25, max_value=0.25)
        weights = keras.ops.convert_to_tensor(
            np.array([[-3.0, 0.0, 3.0]], dtype="float32")
        )
        out = np.asarray(constraint(weights))

        assert np.all(out == np.float32(0.25))


class TestEnforceHardBounds:
    """The one knob this class adds over the plain map, and its measured effect."""

    # `[-1, 1]` at relative sharpness 1.0 -- the regime in which the composition's
    # lower-bound undershoot is largest (measured maximum 6.265e-01, against a
    # predicted log(1 + exp(-beta*(hi-lo)))/beta with beta = 0.5).
    LO = -1.0
    HI = 1.0
    SHARPNESS = 1.0

    @pytest.fixture
    def far_weights(self) -> keras.KerasTensor:
        """Weights far enough below `lo` for the undershoot to be fully developed."""
        return keras.ops.convert_to_tensor(
            np.array([[-50.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 50.0]],
                     dtype="float32")
        )

    def test_the_flag_changes_the_output(
            self, far_weights: keras.KerasTensor
    ) -> None:
        """`enforce_hard_bounds` is observable: it changes output bits.

        This test is the whole justification for the flag's existence (plan
        pre-mortem #2 pre-registered dropping it if no such pair could be found).
        The composed upper branch reads the ALREADY-lifted value and pulls it back
        below `lo` by up to `log(1 + exp(-beta*(hi - lo)))/beta`. At
        `(hi - lo, sharpness) = (2.0, 1.0)` that undershoot is ~6.3e-01, far above
        any float32 resolution question. At the DEFAULT `sharpness=50` the undershoot
        is exactly 0.0 and the flag would be inert -- which is why this test does not
        use the default.
        """
        hard = SoftValueRangeConstraint(
            min_value=self.LO, max_value=self.HI, sharpness=self.SHARPNESS,
            enforce_hard_bounds=True,
        )
        soft = SoftValueRangeConstraint(
            min_value=self.LO, max_value=self.HI, sharpness=self.SHARPNESS,
            enforce_hard_bounds=False,
        )

        y_hard = np.asarray(hard(far_weights))
        y_soft = np.asarray(soft(far_weights))

        assert not np.array_equal(y_hard, y_soft), (
            "the guard changed no output bit; per pre-mortem #2 the flag would then "
            "have to be dropped rather than kept as an unobservable knob"
        )

        # The guard makes the lower bound EXACT ...
        assert np.all(y_hard >= np.float32(self.LO))
        # ... and without it the smooth map really does undershoot it.
        assert np.any(y_soft < np.float32(self.LO))
        undershoot = float(np.float32(self.LO) - y_soft.min())
        assert undershoot > 0.5, f"undershoot {undershoot} is smaller than measured"

    def test_upper_bound_holds_without_the_guard(
            self, far_weights: keras.KerasTensor
    ) -> None:
        """Only the LOWER bound needs the guard; `y <= hi` is structural.

        `sp` is non-negative, so `hi - sp(...)` can never exceed `hi` regardless of
        sharpness. Pinning this asymmetry stops a future reader from "fixing" the
        upper branch too.
        """
        soft = SoftValueRangeConstraint(
            min_value=self.LO, max_value=self.HI, sharpness=self.SHARPNESS,
            enforce_hard_bounds=False,
        )
        y_soft = np.asarray(soft(far_weights))

        assert np.all(y_soft <= np.float32(self.HI))

    def test_the_flag_is_inert_at_the_default_sharpness(
            self, far_weights: keras.KerasTensor
    ) -> None:
        """At `sharpness=50` the guard is measurably a no-op, and that is documented.

        The docstring claims the undershoot is exactly 0.0 from sharpness 20 upward.
        A claim like that goes stale silently unless something re-derives it.
        """
        hard = SoftValueRangeConstraint(
            min_value=self.LO, max_value=self.HI, enforce_hard_bounds=True
        )
        soft = SoftValueRangeConstraint(
            min_value=self.LO, max_value=self.HI, enforce_hard_bounds=False
        )

        assert np.array_equal(
            np.asarray(hard(far_weights)), np.asarray(soft(far_weights))
        )

    def test_hard_bounds_zero_the_gradient_in_a_HYPOTHETICAL_forward_pass(
            self, far_weights: keras.KerasTensor
    ) -> None:
        """An exact clamp is flat outside the box -- but this never happens in role.

        HYPOTHETICAL FRAMING, deliberately. Keras applies a constraint via
        `variable.assign(variable.constraint(variable))` after
        `_backend_apply_gradients` and OUTSIDE any gradient tape
        (`keras/src/optimizers/base_optimizer.py:447-452`), so no gradient is ever
        taken through `__call__` in the projection role and this measurement costs
        that role nothing. It is recorded because it is exactly the reason
        `enforce_hard_bounds` does NOT exist on the `SoftValueRange` layer: in a
        forward pass the same clamp would destroy the nonzero outside gradient the
        map exists to provide. Anyone tempted to reuse this object inside a model's
        forward pass should read this test and use `soft_value_range` instead.
        """
        x = tf.constant(np.asarray(far_weights))

        grads = {}
        outputs = {}
        for enforce in (True, False):
            constraint = SoftValueRangeConstraint(
                min_value=self.LO, max_value=self.HI, sharpness=self.SHARPNESS,
                enforce_hard_bounds=enforce,
            )
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = constraint(x)
                total = keras.ops.sum(y)
            outputs[enforce] = np.asarray(y)
            grads[enforce] = np.asarray(tape.gradient(total, x))

        # The clamp binds exactly where the smooth map undershot `lo` -- DERIVED
        # from the unguarded output, not from a hand-picked input threshold. At
        # sharpness 1.0 on [-1, 1] the undershoot only develops a couple of interval
        # widths out, so `x = -2.0` is NOT clamped while `x <= -5.0` is.
        clamped = outputs[False][0] < np.float32(self.LO)
        assert clamped.sum() >= 3, "the clamp binds nowhere; nothing is being tested"

        # Under the clamp, every entry pinned to the bound is flat.
        assert np.all(grads[True][0][clamped] == 0.0)
        # Without it, the same entries still carry gradient -- the defining
        # difference against a hard clip.
        assert np.all(grads[False][0][clamped] > 0.0)


class TestSerialization:
    """`get_config` / `from_config` must reproduce all five parameters."""

    def test_serialization_round_trip(self) -> None:
        """Round trip with every parameter set away from its default."""
        original = SoftValueRangeConstraint(
            min_value=-2.5,
            max_value=3.75,
            sharpness=12.5,
            relative_sharpness=False,
            enforce_hard_bounds=False,
        )

        config = original.get_config()
        for key in ("min_value", "max_value", "sharpness", "relative_sharpness",
                    "enforce_hard_bounds"):
            assert key in config, f"{key} is missing from get_config()"

        recreated = SoftValueRangeConstraint.from_config(config)

        assert recreated.min_value == original.min_value
        assert recreated.max_value == original.max_value
        assert recreated.sharpness == original.sharpness
        assert recreated.relative_sharpness == original.relative_sharpness
        assert recreated.enforce_hard_bounds == original.enforce_hard_bounds

    def test_serialization_round_trip_behaviour(self) -> None:
        """The recreated constraint produces bit-identical output.

        Comparing attributes alone cannot see a parameter that round trips but is
        then ignored by `__call__`.
        """
        original = SoftValueRangeConstraint(
            min_value=-1.0, max_value=1.0, sharpness=2.0,
            relative_sharpness=False, enforce_hard_bounds=False,
        )
        recreated = SoftValueRangeConstraint.from_config(original.get_config())

        weights = keras.ops.convert_to_tensor(
            np.linspace(-5.0, 5.0, 101).astype("float32").reshape(1, -1)
        )
        np.testing.assert_array_equal(
            np.asarray(original(weights)), np.asarray(recreated(weights))
        )

    def test_serialization_min_only(self) -> None:
        """Round trip in one-sided mode keeps `max_value` as None."""
        original = SoftValueRangeConstraint(min_value=0.1, sharpness=3.0)
        recreated = SoftValueRangeConstraint.from_config(original.get_config())

        assert recreated.min_value == 0.1
        assert recreated.max_value is None
        assert recreated.sharpness == 3.0

    def test_string_representation_with_max(self) -> None:
        """`__repr__` names every parameter when a maximum is set."""
        constraint = SoftValueRangeConstraint(
            min_value=-1.0, max_value=1.0, sharpness=10.0,
            relative_sharpness=False, enforce_hard_bounds=False,
        )
        text = repr(constraint)

        assert "SoftValueRangeConstraint" in text
        assert "min_value=-1.0" in text
        assert "max_value=1.0" in text
        assert "sharpness=10.0" in text
        assert "relative_sharpness=False" in text
        assert "enforce_hard_bounds=False" in text

    def test_string_representation_min_only(self) -> None:
        """`__repr__` OMITS `max_value` in one-sided mode, matching the house class."""
        constraint = SoftValueRangeConstraint(min_value=0.0)
        text = repr(constraint)

        assert "SoftValueRangeConstraint" in text
        assert "min_value=0.0" in text
        assert "sharpness=50.0" in text
        assert "enforce_hard_bounds=True" in text
        assert "max_value" not in text


class TestModelIntegration:
    """The constraint inside a real Keras layer, model and training loop."""

    def test_layer_integration(self) -> None:
        """A Dense layer accepts the constraint and applies it to its kernel."""
        constraint = SoftValueRangeConstraint(min_value=0.0, max_value=1.0)
        layer = keras.layers.Dense(units=4, kernel_constraint=constraint)
        layer.build((None, 8))

        projected = np.asarray(constraint(layer.kernel))
        assert np.all(projected >= np.float32(0.0))
        assert np.all(projected <= np.float32(1.0))

        y = layer(keras.ops.cast(np.random.random((2, 8)), "float32"))
        assert np.all(np.isfinite(np.asarray(y)))

    def test_model_save_load_with_constraint(self) -> None:
        """Full `.keras` save/load round trip with the constraint on two slots."""
        model = keras.Sequential([
            keras.Input(shape=(8,)),
            keras.layers.Dense(
                units=16,
                activation="relu",
                kernel_constraint=SoftValueRangeConstraint(
                    min_value=-0.5, max_value=0.5
                ),
                bias_constraint=SoftValueRangeConstraint(min_value=0.0),
            ),
            keras.layers.Dense(units=1, activation="sigmoid"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

        x_test = np.random.random((10, 8))
        original_predictions = model.predict(x_test, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "soft_constrained_model.keras")

            # Optimizer slot variables are allocated lazily, so a compiled-but-
            # unfitted model would save an optimizer the reload cannot match.
            # See tests/optimizer_state.py (D-016).
            build_optimizer_state(model)
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "SoftValueRangeConstraint": SoftValueRangeConstraint
                },
            )

            loaded_predictions = loaded_model.predict(x_test, verbose=0)
            np.testing.assert_allclose(
                original_predictions, loaded_predictions, rtol=1e-5
            )

            loaded_constraint = loaded_model.layers[0].kernel_constraint
            assert isinstance(loaded_constraint, SoftValueRangeConstraint)
            assert loaded_constraint.min_value == -0.5
            assert loaded_constraint.max_value == 0.5
            assert loaded_constraint.sharpness == 50.0
            assert loaded_constraint.relative_sharpness is True
            assert loaded_constraint.enforce_hard_bounds is True

            # A predictions-only comparison compares the model with itself and so
            # cannot see a constraint that reloaded as an identity map. Exercise the
            # RELOADED object directly.
            probe = keras.ops.convert_to_tensor(
                np.array([[-3.0, 0.0, 3.0]], dtype="float32")
            )
            projected = np.asarray(loaded_constraint(probe))
            assert np.all(projected >= np.float32(-0.5))
            assert np.all(projected <= np.float32(0.5))

            logger.info(
                "Model save/load test with SoftValueRangeConstraint passed"
            )

    def test_wgan_style_critic_stays_in_its_weight_box(self) -> None:
        """A WGAN critic's weight box survives real optimizer steps.

        This is the motivating use case: `[-0.01, 0.01]` weight clipping to enforce a
        Lipschitz critic (Arjovsky et al., 2017). Glorot initialization on 8 inputs
        starts the kernel around +-0.7, i.e. ~70x outside the box, so the assertion
        below is only satisfiable if the constraint is actually applied after each
        step -- an identity `__call__` leaves the kernel at its initial scale.

        The bound is compared CAST TO THE KERNEL'S DTYPE: a narrow dtype represents
        `0.01` only to its own resolution, so a float64-framed comparison would fail
        a perfectly correct implementation.
        """
        constraint = SoftValueRangeConstraint(
            min_value=-0.01, max_value=0.01, sharpness=50.0
        )
        model = keras.Sequential([
            keras.Input(shape=(8,)),
            keras.layers.Dense(units=16, kernel_constraint=constraint),
            keras.layers.Dense(units=1, kernel_constraint=constraint),
        ])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5), loss="mse")

        rng = np.random.default_rng(7)
        x = rng.normal(size=(64, 8)).astype("float32")
        y = rng.normal(size=(64, 1)).astype("float32")
        history = model.fit(x, y, epochs=3, batch_size=16, verbose=0)

        assert np.all(np.isfinite(history.history["loss"]))

        for layer in model.layers:
            kernel = np.asarray(layer.kernel)
            dtype = kernel.dtype
            lo = dtype.type(-0.01)
            hi = dtype.type(0.01)
            assert np.all(kernel >= lo), (
                f"{layer.name} kernel min {kernel.min()} below {lo}"
            )
            assert np.all(kernel <= hi), (
                f"{layer.name} kernel max {kernel.max()} above {hi}"
            )

    def test_the_critic_box_is_not_satisfied_by_accident(self) -> None:
        """Anti-vacuity twin: the initial kernel is far OUTSIDE the box.

        Without this, `test_wgan_style_critic_stays_in_its_weight_box` could pass on
        an initialization that happened to be small enough already.
        """
        layer = keras.layers.Dense(units=16)
        layer.build((None, 8))
        kernel = np.asarray(layer.kernel)

        assert np.max(np.abs(kernel)) > 0.01 * 10

# ---------------------------------------------------------------------
