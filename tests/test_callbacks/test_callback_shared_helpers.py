"""The shared schedule and layer-walk helpers equal the copies they replaced.

``interpolate_schedule`` replaced three private copies (temperature annealing,
harmonic-exponent annealing, noise-sigma curriculum) and ``walk_layers`` replaced
three layer-tree walks. Both moves are refactors, so the proof is equivalence, not
a behaviour change:

* the schedule helper against FROZEN copies of the old formulas, compared on the
  float32 value each callback actually assigns into its variable, with
  ``np.array_equal`` (atol=0), over linear / cosine / exp, endpoints, out-of-range
  epochs, ``total_epochs == 1`` and randomised parameters;
* the three callbacks driven through their real ``on_epoch_begin`` against the same
  frozen formulas;
* the walk against an explicit expected name set on nested lists, dicts, tuples and a
  Functional submodel, and again with ``_flatten_layers`` forced to fail.
"""

import math

import keras
import numpy as np
import pytest

from dl_techniques.callbacks._shared import interpolate_schedule, walk_layers
from dl_techniques.callbacks.anneal_harmonic_exponent import (
    HarmonicExponentAnnealingCallback,
)
from dl_techniques.callbacks.noise_sigma_curriculum import (
    NoiseSigmaCurriculumCallback,
)
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
from dl_techniques.callbacks.temperature_annealing import (
    TemperatureAnnealingCallback,
)
from dl_techniques.layers.regularization.spatial_smoothness import SpatialSmoothness

SCHEDULES = ("linear", "cosine", "exp")


# ---------------------------------------------------------------------
# Frozen copies of the pre-refactor formulas. Do NOT edit these to follow the helper.
# ---------------------------------------------------------------------


def _frozen_temperature(schedule, init, final, total, epoch):
    """temperature_annealing.py / anneal_harmonic_exponent.py before the move."""
    if total == 1:
        return final
    frac = min(max(epoch, 0), total - 1) / (total - 1)
    if schedule == "linear":
        return init + (final - init) * frac
    if schedule == "cosine":
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        return final + (init - final) * cos
    log_init = math.log(init)
    log_final = math.log(final)
    return math.exp(log_init + (log_final - log_init) * frac)


def _frozen_noise(schedule, start, end, total, epoch):
    """noise_sigma_curriculum.py before the move (the cosine is written as 1 - cos)."""
    if total == 1:
        return end
    frac = min(max(epoch, 0), total - 1) / (total - 1)
    if schedule == "linear":
        return start + (end - start) * frac
    if schedule == "cosine":
        cos = 0.5 * (1.0 - math.cos(math.pi * frac))
        return start + (end - start) * cos
    log_start = math.log(start)
    log_end = math.log(end)
    return math.exp(log_start + (log_end - log_start) * frac)


def _grid(total):
    """Every epoch from well below 0 to well above the schedule length."""
    return range(-5, total + 6)


def _random_sets(count, seed):
    rng = np.random.default_rng(seed)
    return [
        (
            float(10 ** rng.uniform(-3, 1.5)),
            float(10 ** rng.uniform(-3, 1.5)),
            int(rng.integers(1, 120)),
        )
        for _ in range(count)
    ]


def _f32(value):
    return np.asarray(value, dtype=np.float32)


def _softplus_raw_f32(value):
    return _f32(math.log(math.expm1(value)))


# ---------------------------------------------------------------------
# interpolate_schedule against the frozen formulas
# ---------------------------------------------------------------------


class TestInterpolateScheduleEqualsTheFrozenCopies:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_temperature_and_harmonic_forms_are_bit_identical_in_float64(self, schedule):
        # The helper keeps this cosine form, so the two annealers are exact, not close.
        sets = [(5.0, 0.1, 50), (1.0, 8.0, 50), (4.0, 0.25, 3)] + _random_sets(150, 1)
        for start, end, total in sets:
            for epoch in _grid(total):
                assert interpolate_schedule(
                    schedule, start, end, epoch, total
                ) == _frozen_temperature(schedule, start, end, total, epoch)

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_the_float32_a_noise_curriculum_assigns_is_unchanged(self, schedule):
        # DECISION plan-2026-11-11T131349-13b28b37/D-032 is pinned here: the noise
        # curriculum wrote its cosine as ``1 - cos`` and now shares the other
        # callbacks' form. The float64 values differ by a few ulp (see
        # ``_shared.py``); the value assigned into the float32 variable must not.
        sets = (
            [(0.05, 0.5, 50), (0.05, 0.5, 100), (0.1, 1.0, 30), (0.3, 0.7, 2)]
            + _random_sets(150, 2)
        )
        if schedule != "exp":
            sets += [(0.0, 0.0, 50), (0.0, 0.3, 40), (0.0, 1.0, 7)]
        for start, end, total in sets:
            for epoch in _grid(total):
                new = interpolate_schedule(schedule, start, end, epoch, total)
                old = _frozen_noise(schedule, start, end, total, epoch)
                assert np.array_equal(_f32(new), _f32(old)), (
                    schedule, start, end, total, epoch, new, old,
                )

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_the_softplus_raw_value_a_temperature_layer_stores_is_unchanged(self, schedule):
        sets = [(5.0, 0.1, 50), (1.0, 8.0, 10), (0.5, 0.5, 5)] + _random_sets(100, 3)
        for start, end, total in sets:
            for epoch in _grid(total):
                new = interpolate_schedule(schedule, start, end, epoch, total)
                old = _frozen_temperature(schedule, start, end, total, epoch)
                assert np.array_equal(
                    _softplus_raw_f32(new), _softplus_raw_f32(old)
                )

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_endpoints_and_the_clamp(self, schedule):
        assert interpolate_schedule(schedule, 2.0, 0.5, 0, 10) == pytest.approx(2.0)
        assert interpolate_schedule(schedule, 2.0, 0.5, 9, 10) == pytest.approx(0.5)
        assert interpolate_schedule(schedule, 2.0, 0.5, -7, 10) == interpolate_schedule(
            schedule, 2.0, 0.5, 0, 10
        )
        assert interpolate_schedule(schedule, 2.0, 0.5, 99, 10) == interpolate_schedule(
            schedule, 2.0, 0.5, 9, 10
        )

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_a_one_epoch_schedule_returns_the_end_value(self, schedule):
        assert interpolate_schedule(schedule, 2.0, 0.5, 0, 1) == 0.5
        assert interpolate_schedule(schedule, 2.0, 0.5, 4, 1) == 0.5

    def test_an_unknown_schedule_raises_rather_than_falling_to_exp(self):
        with pytest.raises(ValueError, match="schedule"):
            interpolate_schedule("bogus", 1.0, 2.0, 0, 10)


# ---------------------------------------------------------------------
# The three callbacks, driven through their real hooks
# ---------------------------------------------------------------------


class _Probe(keras.layers.Layer):
    """A layer with the attributes the annealing callbacks look for."""

    def __init__(self, softplus=False, **kwargs):
        super().__init__(**kwargs)
        self.softplus_temperature = softplus
        self.temperature = self.add_weight(
            name="temperature", shape=(), initializer="ones", trainable=True
        )
        self.effective_n = None

    def set_n_per_level(self, values):
        self.effective_n = [float(v) for v in values]

    def call(self, x):
        return x


def _model_with(layer):
    inp = keras.Input((2,))
    return keras.Model(inp, layer(inp))


class TestTheCallbacksAssignTheFrozenValues:
    @pytest.mark.parametrize("softplus", [False, True])
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_temperature_annealing(self, schedule, softplus):
        layer = _Probe(softplus=softplus, name="probe")
        cb = TemperatureAnnealingCallback(
            schedule=schedule, t_init=5.0, t_final=0.1, total_epochs=50
        )
        cb.set_model(_model_with(layer))
        for epoch in _grid(50):
            cb.on_epoch_begin(epoch)
            expected = _frozen_temperature(schedule, 5.0, 0.1, 50, epoch)
            if softplus:
                assert np.array_equal(
                    _f32(layer.temperature.numpy()), _softplus_raw_f32(expected)
                )
            else:
                assert np.array_equal(_f32(layer.temperature.numpy()), _f32(expected))

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_harmonic_exponent_annealing_per_level(self, schedule):
        layer = _Probe(name="probe")
        layer.num_levels = 3
        init, final = [1.0, 2.0, 0.5], [8.0, 4.0, 6.0]
        cb = HarmonicExponentAnnealingCallback(
            schedule=schedule, n_init=init, n_final=final, total_epochs=40
        )
        cb.set_model(_model_with(layer))
        for epoch in _grid(40):
            cb.on_epoch_begin(epoch)
            expected = [
                _frozen_temperature(schedule, a, b, 40, epoch)
                for a, b in zip(init, final)
            ]
            assert layer.effective_n == expected  # float64, bit-identical

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_noise_sigma_curriculum_both_bounds(self, schedule):
        smax = keras.Variable(0.0, dtype="float32", trainable=False)
        smin = keras.Variable(0.0, dtype="float32", trainable=False)
        low = (0.01, 0.2) if schedule == "exp" else (0.0, 0.2)
        cb = NoiseSigmaCurriculumCallback(
            sigma_max_var=smax, sigma_max_start=0.05, sigma_max_end=0.5,
            total_epochs=50, schedule=schedule, sigma_min_var=smin,
            sigma_min_start=low[0], sigma_min_end=low[1],
        )
        for epoch in _grid(50):
            cb.on_epoch_begin(epoch)
            assert np.array_equal(
                _f32(smax.numpy()),
                _f32(_frozen_noise(schedule, 0.05, 0.5, 50, epoch)),
            )
            assert np.array_equal(
                _f32(smin.numpy()),
                _f32(_frozen_noise(schedule, low[0], low[1], 50, epoch)),
            )


# ---------------------------------------------------------------------
# walk_layers
# ---------------------------------------------------------------------


class _Holder(keras.layers.Layer):
    """Sub-layers held in a nested list, a nested dict, a tuple and a plain attribute."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_list = [
            keras.layers.Dense(3, name="l0"),
            [keras.layers.Dense(3, name="l1"), keras.layers.Dense(3, name="l2")],
        ]
        self.in_dict = {
            "a": keras.layers.Dense(3, name="d0"),
            "b": {"c": keras.layers.Dense(3, name="d1")},
        }
        self.in_tuple = (keras.layers.Dense(3, name="t0"),)
        self.single = keras.layers.Dense(3, name="s0")

    def call(self, x):
        for layer in (
            self.in_list[0], *self.in_list[1], self.in_dict["a"],
            self.in_dict["b"]["c"], self.in_tuple[0], self.single,
        ):
            x = layer(x)
        return x


_HOLDER_NAMES = {"holder", "l0", "l1", "l2", "d0", "d1", "t0", "s0"}


def _nested_functional_model():
    inp = keras.Input((3,))
    sub_in = keras.Input((3,))
    inner = keras.layers.Dense(3, name="fsub")(
        keras.layers.Dense(3, name="fsub0")(sub_in)
    )
    sub = keras.Model(sub_in, inner, name="subfn")
    out = keras.layers.Dense(3, name="top")(_Holder(name="holder")(sub(inp)))
    return keras.Model(inp, out, name="outer")


def _names(layers):
    return {layer.name for layer in layers}


class TestWalkLayers:
    def test_finds_layers_in_lists_dicts_tuples_and_a_functional_submodel(self):
        model = _nested_functional_model()
        found = _names(walk_layers(model))
        assert _HOLDER_NAMES | {"fsub", "fsub0", "subfn", "top"} <= found

    def test_the_root_model_is_not_in_the_result(self):
        model = _nested_functional_model()
        assert all(layer is not model for layer in walk_layers(model))

    def test_each_layer_appears_once(self):
        found = walk_layers(_nested_functional_model())
        assert len({id(layer) for layer in found}) == len(found)

    def test_the_old_annealer_stack_walk_found_the_same_target_layers(self):
        # The loop both annealing callbacks used to carry, frozen. The new walk may
        # also return an InputLayer; no target predicate accepts one, so the layers
        # that are NOT InputLayers must agree exactly.
        model = _nested_functional_model()
        seen, stack, old = set(), list(model.layers), []
        while stack:
            layer = stack.pop()
            if id(layer) in seen:
                continue
            seen.add(id(layer))
            old.append(layer)
            if hasattr(layer, "_layers"):
                stack.extend(getattr(layer, "_layers"))
            elif hasattr(layer, "layers"):
                stack.extend(getattr(layer, "layers"))
        keep = lambda ls: {id(l) for l in ls if not isinstance(l, keras.layers.InputLayer)}
        assert keep(walk_layers(model)) == keep(old)

    def test_the_manual_fallback_returns_the_same_layers(self, monkeypatch):
        model = _nested_functional_model()
        expected = {id(layer) for layer in walk_layers(model)}

        def boom(*args, **kwargs):
            raise RuntimeError("private name moved")

        monkeypatch.setattr(type(model), "_flatten_layers", boom)
        assert {id(layer) for layer in walk_layers(model)} == expected

    def test_the_spatial_loss_logger_finds_a_tap_inside_a_dict(self):
        # The tap sits in a dict held by the model; the logger discovers it through
        # the shared walk.
        class _TapModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.bag = {"deep": [SpatialSmoothness(radius=2, seed=0, name="tap")]}

        model = _TapModel()
        logger_cb = SpatialLossLogger(log_every=1)
        logger_cb.set_model(model)
        assert [tap.name for tap in logger_cb._taps] == ["tap"]
