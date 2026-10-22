"""``HoistedAdamW`` is stock AdamW with the decay's schedule read once per step.

Guard for plan-2026-10-14T062616-81938e81 iteration 3 step 4 (D-280). Keras
3.8's AdamW calls a ``LearningRateSchedule`` once per decayed variable inside
its decoupled weight decay; CliffordCLIP's 691 variables trace one copy of
``WarmupSchedule`` per decayed variable in the first training step.
``HoistedAdamW`` reads the schedule once for the whole decay.

Two properties are held here. Equality: after 5 steps under a warmup plus
cosine schedule, with one variable named ``logit_scale`` excluded from decay,
every parameter and every optimizer slot equals stock AdamW's within 1e-6
(the excluded variable is the one a lost exclusion moves, and the warmup
makes the learning rate change every step, so a shifted schedule moves every
decayed variable). Hoisting: the schedule is called twice per step (decay and
update) where stock AdamW calls it ``1 + n_decayed`` times.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.optimization import learning_rate_schedule_builder
from dl_techniques.optimization.hoisted_adamw import HoistedAdamW

STEPS = 5
TOL = 1e-6
WEIGHT_DECAY = 0.1


def _schedule():
    """The CliffordCLIP trainer's schedule shape: warmup, then cosine decay."""
    return learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": 1e-2,
        "decay_steps": 8,
        "alpha": 0.01,
        "warmup_steps": 2,
        "warmup_start_lr": 1e-4,
    })


class _CountingSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Counts how often the optimizer calls its schedule."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, step):
        self.calls += 1
        return self.inner(step)

    def get_config(self):
        return {}


def _variables(n_dense=3, seed=0):
    rng = np.random.default_rng(seed)
    dense = [keras.Variable(rng.normal(size=(4, 3)).astype("float32"), name=f"kernel_{i}")
             for i in range(n_dense)]
    logit_scale = keras.Variable(np.float32(np.log(1 / 0.07)), name="logit_scale")
    return dense + [logit_scale]


def _gradients(variables, steps=STEPS, seed=1):
    rng = np.random.default_rng(seed)
    return [[rng.normal(size=v.shape).astype("float32") for v in variables]
            for _ in range(steps)]


def _make(cls, learning_rate, exclude=True):
    optimizer = cls(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98,
                    epsilon=1e-6, weight_decay=WEIGHT_DECAY)
    if exclude:
        optimizer.exclude_from_weight_decay(var_names=["logit_scale"])
    return optimizer


def _train(cls, exclude=True, traced=False):
    variables = _variables()
    optimizer = _make(cls, _schedule(), exclude=exclude)
    optimizer.build(variables)

    def step(grads):
        optimizer.apply(grads, variables)

    if traced:
        step = tf.function(step)
    for grads in _gradients(variables):
        step([tf.constant(g) for g in grads])
    params = [keras.ops.convert_to_numpy(v) for v in variables]
    slots = [keras.ops.convert_to_numpy(v) for v in optimizer.variables]
    return params, slots, optimizer


def _max_difference(a, b):
    assert len(a) == len(b)
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
def test_five_steps_match_stock_adamw_within_1e_6(traced):
    stock_params, stock_slots, stock = _train(keras.optimizers.AdamW, traced=traced)
    params, slots, hoisted = _train(HoistedAdamW, traced=traced)
    # Same layout: the same slots, in the same order, with the same shapes.
    assert [(v.name, tuple(v.shape)) for v in hoisted.variables] == [
        (v.name, tuple(v.shape)) for v in stock.variables]
    assert _max_difference(params, stock_params) <= TOL
    assert _max_difference(slots, stock_slots) <= TOL


def test_the_comparison_can_see_a_lost_exclusion():
    """Control: decaying ``logit_scale`` too moves it far beyond the tolerance."""
    stock_params, _, _ = _train(keras.optimizers.AdamW)
    decayed_params, _, _ = _train(keras.optimizers.AdamW, exclude=False)
    assert abs(float(decayed_params[-1] - stock_params[-1])) > 100 * TOL


@pytest.mark.parametrize("n_dense", [3, 8])
def test_the_schedule_is_called_twice_per_step_not_once_per_variable(n_dense):
    variables = _variables(n_dense)
    grads = [tf.constant(g) for g in _gradients(variables, steps=1)[0]]
    counts = {}
    for cls in (keras.optimizers.AdamW, HoistedAdamW):
        schedule = _CountingSchedule(_schedule())
        optimizer = _make(cls, schedule)
        optimizer.build(variables)
        schedule.calls = 0
        optimizer.apply(grads, variables)
        counts[cls.__name__] = schedule.calls
    # Stock: one call per decayed variable (logit_scale is excluded), one for the update.
    assert counts["AdamW"] == 1 + n_dense
    assert counts["HoistedAdamW"] == 2


def test_a_step_through_the_loss_scale_optimizer_matches_stock():
    results = []
    for cls in (keras.optimizers.AdamW, HoistedAdamW):
        variables = _variables()
        optimizer = keras.optimizers.LossScaleOptimizer(_make(cls, _schedule()))
        optimizer.build(variables)
        scale = float(keras.ops.convert_to_numpy(optimizer.dynamic_scale))
        for grads in _gradients(variables, steps=2):
            optimizer.apply([tf.constant(g * scale) for g in grads], variables)
        results.append([keras.ops.convert_to_numpy(v) for v in variables])
    assert _max_difference(results[0], results[1]) <= TOL
    assert _max_difference(results[1], [keras.ops.convert_to_numpy(v) for v in _variables()]) > 0


def test_the_config_round_trips():
    optimizer = _make(HoistedAdamW, _schedule())
    restored = keras.optimizers.deserialize(keras.optimizers.serialize(optimizer))
    assert type(restored) is HoistedAdamW
    original, back = optimizer.get_config(), restored.get_config()
    assert original.keys() == back.keys()
    for key in ("weight_decay", "beta_1", "beta_2", "epsilon", "amsgrad"):
        assert back[key] == original[key]
    for step in (0, 1, 5):
        assert float(restored._learning_rate(step)) == float(optimizer._learning_rate(step))
