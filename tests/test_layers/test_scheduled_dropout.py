"""Tests for the ScheduledDropout layer.

Two tiers:
  1. The 6-item baseline bar every layer in this repo must clear (construction +
     validation, `compute_output_shape` unbuilt, training-vs-inference
     divergence, config round-trip, `.keras` round-trip, `fit()` smoke).
  2. The schedule-specific guards that justify the design at all (counter
     arithmetic, counter persistence, inference inertness, rate decay, clamp).
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.scheduled_dropout import ScheduledDropout

B, D = 8, 16

# The two rate kinds every rate-kind-agnostic assertion is parametrized over.
RATE_KINDS = ["schedule", "float"]
FLOAT_RATE = 0.3

# The layer's single clamp ceiling (a rate of exactly 1.0 divides by zero).
MAX_RATE = 1.0 - 1e-6


def make_rate(kind: str):
    """Build a fresh rate object of the requested kind."""
    if kind == "schedule":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.5, decay_steps=100, alpha=0.1
        )
    return FLOAT_RATE


def make_model(rate, seed=1234, jit_compile="auto"):
    """Small compiled Dense/ScheduledDropout/Dense model."""
    inputs = keras.Input(shape=(D,))
    x = keras.layers.Dense(D, activation="relu")(inputs)
    x = ScheduledDropout(rate, seed=seed, name="sd")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", jit_compile=jit_compile)
    return model


def counter_value(layer) -> int:
    """Read the layer's step counter as a Python int."""
    return int(keras.ops.convert_to_numpy(layer.step_counter))


def rate_value(layer) -> float:
    """Read the layer's current (clipped) rate as a Python float."""
    return float(keras.ops.convert_to_numpy(layer.current_rate()))


def schedule_value(schedule, step) -> float:
    """Evaluate a schedule directly, bypassing the layer."""
    return float(keras.ops.convert_to_numpy(schedule(step)))


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, D)).astype("float32")


@pytest.fixture(params=RATE_KINDS)
def rate_kind(request):
    return request.param


@pytest.fixture
def mixed_float16_policy():
    """Set the global mixed_float16 policy, always restoring the previous one."""
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


class TestScheduledDropout:
    """Baseline bar (step 3)."""

    # ------------------------------------------------------------------
    # 1. Construction + input validation
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.5, 0.9999])
    def test_construction_valid_float(self, value):
        layer = ScheduledDropout(value)
        assert layer.rate == value
        assert layer.built is False

    def test_construction_valid_int(self):
        layer = ScheduledDropout(0)
        assert layer.rate == 0.0

    def test_construction_valid_schedule(self):
        schedule = make_rate("schedule")
        layer = ScheduledDropout(schedule)
        assert layer.rate is schedule

    def test_construction_stores_optional_args(self):
        layer = ScheduledDropout(0.2, noise_shape=(None, 1), seed=7)
        assert layer.noise_shape == (None, 1)
        assert layer.seed == 7

    @pytest.mark.parametrize("bad", [1.0, -0.1, 1.5, 2.0])
    def test_invalid_float_rate_raises_value_error(self, bad):
        with pytest.raises(ValueError):
            ScheduledDropout(bad)

    @pytest.mark.parametrize("bad", ["0.5", None, True, False, [0.5], {"rate": 0.5}])
    def test_invalid_rate_type_raises_type_error(self, bad):
        with pytest.raises(TypeError):
            ScheduledDropout(bad)

    # ------------------------------------------------------------------
    # 2. compute_output_shape on an unbuilt instance
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [(B, D), (2, 8, 8, 3), (None, 16), (1,), (4, 5, 6, 7, 8)],
    )
    def test_compute_output_shape_unbuilt(self, rate_kind, shape):
        layer = ScheduledDropout(make_rate(rate_kind))
        assert layer.built is False
        assert layer.compute_output_shape(shape) == tuple(shape)
        # Still unbuilt: computing a shape must not create state.
        assert layer.built is False

    def test_output_shape_matches_input(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        out = layer(sample, training=True)
        assert tuple(out.shape) == sample.shape

    # ------------------------------------------------------------------
    # 3. Training-vs-inference behavioural divergence
    # ------------------------------------------------------------------

    def test_training_differs_from_input(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        out = keras.ops.convert_to_numpy(layer(sample, training=True))
        assert not np.array_equal(out, sample)

    def test_inference_is_identity(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        layer.build(sample.shape)
        out = keras.ops.convert_to_numpy(layer(sample, training=False))
        assert np.array_equal(out, sample)

    def test_inference_is_repeatable_training_is_not(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        inf = [keras.ops.convert_to_numpy(layer(sample, training=False)) for _ in range(4)]
        assert all(np.array_equal(inf[0], o) for o in inf[1:])

        train = [keras.ops.convert_to_numpy(layer(sample, training=True)) for _ in range(4)]
        assert any(not np.array_equal(train[0], o) for o in train[1:])

    # ------------------------------------------------------------------
    # 4. get_config / from_config round-trip
    # ------------------------------------------------------------------

    def test_config_round_trip_float(self):
        original = ScheduledDropout(0.42, noise_shape=None, seed=11)
        restored = ScheduledDropout.from_config(original.get_config())
        assert restored.rate == 0.42
        assert restored.seed == 11
        assert restored.noise_shape is None

    def test_config_round_trip_schedule(self):
        schedule = make_rate("schedule")
        original = ScheduledDropout(schedule, seed=11)
        config = original.get_config()
        # The nested schedule must be serialized, not stored as a live object.
        assert isinstance(config["rate"], dict)

        restored = ScheduledDropout.from_config(config)
        assert type(restored.rate) is type(schedule)
        assert restored.rate.get_config() == schedule.get_config()
        assert restored.seed == 11

    def test_config_is_json_serializable(self, rate_kind):
        import json

        config = ScheduledDropout(make_rate(rate_kind), seed=3).get_config()
        json.dumps(config)  # raises if any value is not JSON-encodable

    # ------------------------------------------------------------------
    # 5. Full .keras save/load round-trip
    # ------------------------------------------------------------------

    def test_keras_round_trip(self, rate_kind, sample, tmp_path):
        rate = make_rate(rate_kind)
        model = make_model(rate)
        y0 = keras.ops.convert_to_numpy(model(sample, training=False))

        path = os.path.join(tmp_path, "sd.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        y1 = keras.ops.convert_to_numpy(loaded(sample, training=False))
        np.testing.assert_allclose(y0, y1, atol=1e-6)

        restored_layer = loaded.get_layer("sd")
        assert isinstance(restored_layer, ScheduledDropout)
        if rate_kind == "schedule":
            assert type(restored_layer.rate) is type(rate)
            assert restored_layer.rate.get_config() == rate.get_config()
        else:
            assert restored_layer.rate == FLOAT_RATE

    # ------------------------------------------------------------------
    # 6. model.fit() smoke test
    # ------------------------------------------------------------------

    def test_fit_smoke(self, rate_kind):
        model = make_model(make_rate(rate_kind))
        rng = np.random.default_rng(1)
        x = rng.standard_normal((32, D)).astype("float32")
        y = rng.standard_normal((32, 1)).astype("float32")

        history = model.fit(x, y, epochs=2, batch_size=8, verbose=0)
        assert "loss" in history.history
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_gradients_flow(self, rate_kind, sample):
        import tensorflow as tf

        model = make_model(make_rate(rate_kind))
        y = np.zeros((B, 1), dtype="float32")
        with tf.GradientTape() as tape:
            loss = keras.ops.mean((model(sample, training=True) - y) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)

    # ==================================================================
    # Schedule-specific guards (step 4). These are the assertions the
    # design lives or dies on: the counter, its persistence, and the
    # rate the schedule actually produces from it.
    # ==================================================================

    # ------------------------------------------------------------------
    # SC-3 — counter arithmetic
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("jit_compile", [False, True])
    def test_counter_after_fit_equals_epochs_times_steps(self, rate_kind, jit_compile):
        epochs, steps_per_epoch, batch = 3, 8, 4
        model = make_model(make_rate(rate_kind), jit_compile=jit_compile)
        layer = model.get_layer("sd")

        rng = np.random.default_rng(2)
        n = steps_per_epoch * batch
        x = rng.standard_normal((n, D)).astype("float32")
        y = rng.standard_normal((n, 1)).astype("float32")

        assert counter_value(layer) == 0
        model.fit(x, y, epochs=epochs, batch_size=batch, shuffle=False, verbose=0)
        assert counter_value(layer) == epochs * steps_per_epoch

    def test_counter_increments_once_per_training_call(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        layer.build(sample.shape)
        assert counter_value(layer) == 0
        for expected in range(1, 6):
            layer(sample, training=True)
            assert counter_value(layer) == expected

    # ------------------------------------------------------------------
    # SC-4 — counter persistence across .keras save/load
    # ------------------------------------------------------------------

    def test_counter_survives_save_load(self, rate_kind, tmp_path):
        model = make_model(make_rate(rate_kind))
        layer = model.get_layer("sd")

        rng = np.random.default_rng(3)
        x = rng.standard_normal((32, D)).astype("float32")
        y = rng.standard_normal((32, 1)).astype("float32")
        model.fit(x, y, epochs=2, batch_size=4, shuffle=False, verbose=0)

        saved = counter_value(layer)
        assert saved == 16  # 2 epochs x 8 steps; guards against a silent zero

        path = os.path.join(tmp_path, "counter.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = counter_value(loaded.get_layer("sd"))

        assert restored == saved
        assert restored != 0

    def test_reloaded_layer_resumes_the_schedule(self, tmp_path):
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.5, decay_steps=64, end_learning_rate=0.05
        )
        model = make_model(schedule)
        layer = model.get_layer("sd")

        rng = np.random.default_rng(4)
        x = rng.standard_normal((32, D)).astype("float32")
        y = rng.standard_normal((32, 1)).astype("float32")
        model.fit(x, y, epochs=2, batch_size=4, shuffle=False, verbose=0)

        before = rate_value(layer)
        assert before < schedule_value(schedule, 0)  # decay really moved

        path = os.path.join(tmp_path, "resume.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert rate_value(loaded.get_layer("sd")) == pytest.approx(before, abs=1e-7)

    # ------------------------------------------------------------------
    # SC-6 — inference is inert (bit-exact identity, no counter movement)
    # ------------------------------------------------------------------

    def test_inference_leaves_counter_unchanged(self, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        layer.build(sample.shape)

        for _ in range(3):
            layer(sample, training=True)
        assert counter_value(layer) == 3

        for _ in range(5):
            out = keras.ops.convert_to_numpy(layer(sample, training=False))
            assert np.array_equal(out, sample)
        assert counter_value(layer) == 3

        # Default `training` (bare call) must also be inert.
        out = keras.ops.convert_to_numpy(layer(sample))
        assert np.array_equal(out, sample)
        assert counter_value(layer) == 3

    def test_predict_and_evaluate_leave_counter_unchanged(self, rate_kind):
        model = make_model(make_rate(rate_kind))
        layer = model.get_layer("sd")

        rng = np.random.default_rng(5)
        x = rng.standard_normal((32, D)).astype("float32")
        y = rng.standard_normal((32, 1)).astype("float32")
        model.fit(x, y, epochs=1, batch_size=4, shuffle=False, verbose=0)

        after_fit = counter_value(layer)
        assert after_fit == 8

        model.predict(x, batch_size=4, verbose=0)
        assert counter_value(layer) == after_fit

        model.evaluate(x, y, batch_size=4, verbose=0)
        assert counter_value(layer) == after_fit

    # ------------------------------------------------------------------
    # SC-5 — the rate follows the schedule
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("schedule_name", ["cosine", "polynomial"])
    def test_rate_decays_with_counter(self, schedule_name, sample):
        if schedule_name == "cosine":
            schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.5, decay_steps=100, alpha=0.1
            )
        else:
            schedule = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=0.5, decay_steps=100, end_learning_rate=0.05
            )

        layer = ScheduledDropout(schedule, seed=0)
        layer.build(sample.shape)

        steps = [0, 10, 25, 50, 75, 99]
        rates = []
        for step in steps:
            layer.step_counter.assign(step)
            rate = rate_value(layer)
            # D-007: current_rate() reads the schedule at the CURRENT counter.
            assert rate == pytest.approx(schedule_value(schedule, step), abs=1e-7)
            rates.append(rate)

        assert len(rates) == len(steps)
        assert rates[0] > rates[-1]
        for previous, nxt in zip(rates, rates[1:]):
            assert nxt < previous

    def test_float_rate_is_exactly_constant(self, sample):
        layer = ScheduledDropout(0.37, seed=0)
        layer.build(sample.shape)

        rates = []
        for step in [0, 1, 7, 100, 10_000]:
            layer.step_counter.assign(step)
            rates.append(rate_value(layer))

        assert len(rates) == 5
        assert all(r == pytest.approx(0.37, abs=1e-7) for r in rates)
        assert len(set(rates)) == 1

    def test_first_training_call_uses_schedule_at_step_zero(self, sample):
        # Probe (g): read-then-increment, so call #1 sees schedule(0) and
        # current_rate() afterwards reports the NEXT call's rate.
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.5, decay_steps=20, end_learning_rate=0.05
        )
        layer = ScheduledDropout(schedule, seed=0)
        layer.build(sample.shape)

        assert counter_value(layer) == 0
        assert rate_value(layer) == pytest.approx(schedule_value(schedule, 0), abs=1e-7)

        layer(sample, training=True)
        assert counter_value(layer) == 1
        assert rate_value(layer) == pytest.approx(schedule_value(schedule, 1), abs=1e-7)
        # The two are genuinely different, so the assertion above is not vacuous.
        assert schedule_value(schedule, 1) < schedule_value(schedule, 0)

    # ------------------------------------------------------------------
    # SC-5 (forward path) — the MASK must track the schedule.
    #
    # Every other SC-5 assertion above reads `current_rate()`, which is a
    # PARALLEL read path (`scheduled_dropout.py:210`), never the masking path
    # (`:234`). A defect confined to `call()` — computing `rate =
    # self._rate_at(0)` forever while leaving `_rate_at`, `current_rate()` and
    # the counter perfectly intact — passed 75/75 tests. This test is the only
    # one that observes the scheduled rate THROUGH the mask, so it is the only
    # one that goes red under that injection. Do not delete it in favour of a
    # `current_rate()` assertion; that is precisely the gap it exists to close.
    # ------------------------------------------------------------------

    def test_forward_mask_tracks_the_schedule(self):
        # 0.0 -> 0.99 is an unambiguously large swing: no single frozen rate
        # can sit within tolerance of every probed step.
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0, decay_steps=100, end_learning_rate=0.99
        )
        layer = ScheduledDropout(schedule, seed=20260722)
        # All-ones input, so every zero in the output comes from the mask and
        # from nothing else. 64x256 = 16384 draws per probe.
        shape = (64, 256)
        x = keras.ops.ones(shape, dtype="float32")
        layer.build(shape)

        steps = [0, 25, 50, 75, 100]
        observed = []
        for step in steps:
            layer.step_counter.assign(step)
            out = keras.ops.convert_to_numpy(layer(x, training=True))
            zero_fraction = float(np.mean(out == 0.0))
            observed.append(zero_fraction)

            # Expected value comes from the schedule DIRECTLY, bypassing the
            # layer entirely — comparing the mask against `current_rate()`
            # would just re-test the parallel path.
            expected = schedule_value(schedule, step)

            # Tolerance: 16384 Bernoulli draws give a binomial sd <= 0.0039.
            # Measured over 201 seeds the worst error on these interior probes
            # is 0.0138 (seed 119, step 75), so abs=0.02 has a real margin of
            # 1.45x -- NOT the ~5x an earlier four-seed sample suggested. It is
            # still 12x tighter than the smallest deviation this test must
            # catch (a rate frozen at step 0 reads 0.000 where step 25 expects
            # 0.2475), but it is too loose to see a ONE-step shift: that is
            # what the zero-variance step-0 probe below is for.
            assert zero_fraction == pytest.approx(expected, abs=0.02), (
                f"mask at step {step}: observed zero-fraction {zero_fraction}, "
                f"schedule says {expected}"
            )

        # Non-vacuous: the mask really did move across (almost) the full range.
        assert observed[-1] - observed[0] > 0.9
        for previous, nxt in zip(observed, observed[1:]):
            assert nxt > previous

        # ------------------------------------------------------------------
        # ZERO-VARIANCE STEP-0 PROBE (D-015) — pins the read-then-increment
        # ordering that `scheduled_dropout.py`'s `# DECISION .../D-010` anchor
        # forbids reordering ("swapping these lines shifts every rate by one
        # step").
        #
        # WHY THIS ONE PROBE MAY CARRY A TIGHT TOLERANCE WHILE THE OTHERS MAY
        # NOT: this schedule's value at step 0 is EXACTLY 0.0, so every one of
        # the 16384 Bernoulli trials has p=0. A binomial with p=0 has variance
        # exactly 0 — there is no randomness left to vary, and the observed
        # zero-fraction is not "0.0 within sampling noise", it is 0.0. Every
        # other probe sits at 0 < p < 1 where the binomial sd is ~0.0039, which
        # is what forces those to abs=0.02.
        #
        # Under the D-010 reorder (`assign_add` before the read) the step-0
        # probe reads schedule(1) = 0.0099 instead of schedule(0) = 0.0 and
        # observes ~0.0116 with sd 0.00077 — ~15 sd above this bound, whereas
        # the abs=0.02 loop above clears it by 18% (max error 0.01635) and
        # stays green. This assertion is the only thing in the suite that sees
        # a one-step shift.
        assert schedule_value(schedule, 0) == 0.0, (
            "this probe's tight tolerance is only valid because schedule(0) is "
            "exactly 0.0; if the schedule changes, the tolerance must too"
        )
        assert observed[0] < 0.005, (
            f"step-0 mask must be a zero-variance identity (schedule(0) is "
            f"exactly 0.0, so p=0 for every draw), but the observed "
            f"zero-fraction was {observed[0]}. A non-zero value here means "
            f"call() did not evaluate the schedule at the counter's PRE-"
            f"increment value -- see the D-010 anchor in scheduled_dropout.py"
        )

    # ------------------------------------------------------------------
    # SC-8 — the clamp holds
    # ------------------------------------------------------------------

    def test_clamp_holds_at_both_ends(self, sample):
        # Deliberately out of range at BOTH ends.
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.5, decay_steps=10, end_learning_rate=-0.2
        )
        # Prove the clip is load-bearing: the raw schedule really is out of range.
        assert schedule_value(schedule, 0) > MAX_RATE
        assert schedule_value(schedule, 1000) < 0.0

        layer = ScheduledDropout(schedule, seed=0)
        layer.build(sample.shape)

        layer.step_counter.assign(0)
        assert rate_value(layer) == pytest.approx(MAX_RATE, abs=1e-7)

        layer.step_counter.assign(1000)
        assert rate_value(layer) == pytest.approx(0.0, abs=1e-7)

        for step in range(0, 21):
            layer.step_counter.assign(step)
            assert 0.0 <= rate_value(layer) <= MAX_RATE

    def test_clamp_ceiling_forward_pass_amplifies_survivors(self):
        # CHARACTERIZATION TEST (D-013), not a defence. It pins the DOCUMENTED
        # behaviour at the clamp ceiling: the clamp bounds the RATE, it does not
        # bound the activation MAGNITUDE. At rate 1 - 1e-6 the few survivors are
        # rescaled by 1/(1 - rate) ~ 1e6, which is a real and expected spike.
        # Do not "fix" this by lowering the ceiling or by adding a magnitude
        # clip -- that would silently change dropout's rescaling contract. The
        # sharp edge is documented in the layer's class docstring instead.
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=1.5, decay_steps=10, end_learning_rate=1.2
        )
        # Out of range at BOTH ends, so the clip pins the rate to the ceiling
        # at every step and the counter needs no manual placement.
        assert schedule_value(schedule, 0) > 1.0
        assert schedule_value(schedule, 10_000) > 1.0

        layer = ScheduledDropout(schedule, seed=99)
        layer.build((256, 16384))
        assert rate_value(layer) == pytest.approx(MAX_RATE, abs=1e-7)

        # All-ones input, so every non-zero output equals the rescale factor
        # exactly. 8 chunks of 4194304 draws => ~33 expected survivors, i.e.
        # P(zero survivors) ~ 3e-15. A single chunk would expect only ~4 and
        # measured as few as 1 across seeds -- too thin to assert on.
        x = keras.ops.ones((256, 16384), dtype="float32")
        expected_scale = 1.0 / (1.0 - MAX_RATE)
        survivors = []
        for _ in range(8):
            out = keras.ops.convert_to_numpy(layer(x, training=True))
            assert not np.any(np.isnan(out))
            assert not np.any(np.isinf(out))
            survivors.extend(out[out != 0.0].tolist())

        assert len(survivors) > 0
        # The spike is ~1e6, not exactly 1e6: float32 rounds `1 - (1 - 1e-6)`
        # to 1.0132e-6, so the measured factor is 986895.06, 1.3% low. rel=0.05
        # accepts that rounding while still pinning the order of magnitude.
        for value in survivors:
            assert value == pytest.approx(expected_scale, rel=0.05)

    @pytest.mark.parametrize(
        "schedule_name", ["cosine", "polynomial", "exponential", "piecewise"]
    )
    def test_rate_in_range_beyond_decay_steps(self, schedule_name, sample):
        schedules = {
            "cosine": keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=0.5, decay_steps=100, alpha=0.1
            ),
            "polynomial": keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=0.5, decay_steps=100, end_learning_rate=0.05
            ),
            "exponential": keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.5, decay_steps=100, decay_rate=0.5
            ),
            "piecewise": keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[10, 50], values=[0.5, 0.3, 0.1]
            ),
        }
        layer = ScheduledDropout(schedules[schedule_name], seed=0)
        layer.build(sample.shape)

        for step in [0, 100, 101, 1_000, 100_000]:
            layer.step_counter.assign(step)
            rate = rate_value(layer)
            assert 0.0 <= rate <= MAX_RATE
            assert np.isfinite(rate)

        # A forward pass at the far-out step must still run.
        out = layer(sample, training=True)
        assert tuple(out.shape) == sample.shape

    # ------------------------------------------------------------------
    # D-006 — rate 0.0 is a bit-exact identity on the TRAINING path
    # ------------------------------------------------------------------

    def test_zero_rate_is_identity_on_training_path(self, sample):
        layer = ScheduledDropout(0.0, seed=0)
        layer.build(sample.shape)
        out = keras.ops.convert_to_numpy(layer(sample, training=True))
        assert np.array_equal(out, sample)
        # It went down the training path, so it still counted.
        assert counter_value(layer) == 1

    def test_zero_rate_schedule_is_identity_on_training_path(self, sample):
        schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0, decay_steps=10, end_learning_rate=0.0
        )
        layer = ScheduledDropout(schedule, seed=0)
        layer.build(sample.shape)
        out = keras.ops.convert_to_numpy(layer(sample, training=True))
        assert np.array_equal(out, sample)
        assert counter_value(layer) == 1

    # ------------------------------------------------------------------
    # Mixed precision — the ops.cast in call() is load-bearing (D-004)
    # ------------------------------------------------------------------

    def test_mixed_float16_forward_pass(self, mixed_float16_policy, rate_kind, sample):
        layer = ScheduledDropout(make_rate(rate_kind), seed=0)
        out = layer(sample, training=True)
        assert keras.backend.standardize_dtype(out.dtype) == "float16"
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))
        assert counter_value(layer) == 1

    def test_mixed_float16_model_fit(self, mixed_float16_policy):
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.5, decay_steps=100, alpha=0.1
        )
        model = make_model(schedule)
        rng = np.random.default_rng(6)
        x = rng.standard_normal((16, D)).astype("float32")
        y = rng.standard_normal((16, 1)).astype("float32")
        model.fit(x, y, epochs=1, batch_size=4, shuffle=False, verbose=0)
        assert counter_value(model.get_layer("sd")) == 4

    # ------------------------------------------------------------------
    # seed= reproducibility — CONSTRAINED BY D-011: fresh layers only.
    # Mask reproducibility across a save/load boundary is deliberately NOT
    # asserted: Keras excludes seed-generator state from `Layer.weights` by
    # design and stock keras.layers.Dropout behaves identically, so such a
    # test would pin a Keras limitation rather than this layer's contract.
    # ------------------------------------------------------------------

    def test_same_seed_gives_identical_masks(self, sample):
        a = ScheduledDropout(0.5, seed=1234)
        b = ScheduledDropout(0.5, seed=1234)
        for _ in range(3):
            out_a = keras.ops.convert_to_numpy(a(sample, training=True))
            out_b = keras.ops.convert_to_numpy(b(sample, training=True))
            assert np.array_equal(out_a, out_b)
            # Non-vacuous: dropout actually zeroed something.
            assert np.any(out_a == 0.0)

    def test_different_seed_gives_different_mask(self, sample):
        a = ScheduledDropout(0.5, seed=1234)
        b = ScheduledDropout(0.5, seed=4321)
        out_a = keras.ops.convert_to_numpy(a(sample, training=True))
        out_b = keras.ops.convert_to_numpy(b(sample, training=True))
        assert not np.array_equal(out_a, out_b)

    def test_current_rate_before_build_raises(self):
        layer = ScheduledDropout(0.5)
        with pytest.raises(ValueError):
            layer.current_rate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
