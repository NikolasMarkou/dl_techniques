"""The CliffordCLIP contrastive stage trains with ``HoistedAdamW``.

Guard for plan-2026-10-14T062616-81938e81 iteration 3 step 4 (D-280). Stock
``keras.optimizers.AdamW`` reads its learning-rate schedule once per decayed
variable; on CliffordCLIP ``nano`` that cost about 42 s of a 246 s first step
on the RTX 4090 (README Gotcha 5). ``_build_optimizer`` builds
``HoistedAdamW``, which equals AdamW within 1e-6
(``tests/test_optimization/test_hoisted_adamw.py``) and reads the schedule
once per step for the decay. Checked here: the class, the Penguin-VL
hyperparameters the stock builder used to set, the schedule, and the count
of schedule calls in one step.
"""

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.optimization import HoistedAdamW, learning_rate_schedule_builder
from train.cliffordnet import train_clip as trainer

STEPS_PER_EPOCH = 10


def _optimizer():
    config = trainer.CliffordCLIPTrainConfig(image_size=64, epochs=2, batch_size=2, peak_lr=1e-3)
    return config, trainer._build_optimizer(config, steps_per_epoch=STEPS_PER_EPOCH)


def test_the_builder_returns_hoisted_adamw_with_the_recipe_values():
    config, optimizer = _optimizer()
    assert type(optimizer) is HoistedAdamW
    assert optimizer.name == "AdamW"
    assert (optimizer.beta_1, optimizer.beta_2, optimizer.epsilon) == (0.9, 0.98, 1e-6)
    assert optimizer.weight_decay == config.weight_decay
    assert not optimizer.amsgrad


def test_the_schedule_is_the_warmup_cosine_the_trainer_states():
    config, optimizer = _optimizer()
    total = config.epochs * STEPS_PER_EPOCH
    warmup = max(1, int(round(config.warmup_ratio * total)))
    expected = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": config.peak_lr,
        "decay_steps": max(1, total - warmup),
        "alpha": config.lr_alpha,
        "warmup_steps": warmup,
        "warmup_start_lr": config.min_warmup_lr,
    })
    for step in (0, 1, warmup, total // 2, total):
        assert float(optimizer._learning_rate(step)) == float(expected(step))


def test_one_step_calls_the_schedule_twice_not_once_per_variable():
    _, optimizer = _optimizer()
    calls = []

    class Counting(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, step):
            calls.append(step)
            return self.inner(step)

    optimizer._learning_rate = Counting(optimizer._learning_rate)
    rng = np.random.default_rng(0)
    variables = [keras.Variable(rng.normal(size=(3, 3)).astype("float32"), name=f"w{i}")
                 for i in range(6)]
    optimizer.build(variables)
    calls.clear()
    optimizer.apply([tf.ones((3, 3))] * len(variables), variables)
    assert len(calls) == 2, len(calls)
