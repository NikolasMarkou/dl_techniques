"""`PTRMTrainer.train_step` runs, and it trains.

Rationale
---------
`PTRMTrainer.train_step` is a custom GradientTape step: the ACT carry loop
(up to ``halt_max_steps`` supervision steps per batch, each with its own loss,
exiting early once every item halts) is the training algorithm, as it is for its
frozen siblings `HRMTrainer.train_step` and `TRMTrainer.train_step`. It is
therefore a frozen key in ``_FROZEN_STEP_OVERRIDES``
(tests/test_models/test_package_api_contract.py), and the freeze rule asks for an
execution test. Neither sibling has one to follow (`tests/test_train/test_hrm/`
replaces `HRMTrainer.train_step` with a lambda), so this file is it.

Why eager and not ``jit_compile=True``: the step cannot be traced at all, and the
last test here pins that measured fact. Its loop condition starts as the Python
int ``step_count < halt_max_steps`` and its ``break`` depends on the tensor
``all_finished``; AutoGraph refuses that ("The condition of while loop started as
non-Tensor, then changed to Tensor"), under plain ``tf.function`` and under
``jit_compile=True`` alike. The step also calls ``float()`` on tensors for its
metrics. The trainer itself calls the step eagerly (``train_epoch``), so eager is
the mode it ships in. Making it traceable is a trainer rewrite outside the freeze.

See decisions.md D-018 (plan-2026-10-13T035756-5dd4d94d).
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.ptrm.train_ptrm import PTRMTrainer, get_quick_test_config

# ---------------------------------------------------------------------

SEQ_LEN = 8
VOCAB_SIZE = 16
HALT_MAX_STEPS = 2


def _tiny_config():
    config = get_quick_test_config()
    config["model"].update(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_heads=4,
        halt_max_steps=HALT_MAX_STEPS,
    )
    return config


def _batch(seed):
    rng = np.random.default_rng(seed)
    return {
        "inputs": rng.integers(2, VOCAB_SIZE, size=(2, SEQ_LEN)).astype("int32"),
        "labels": rng.integers(2, VOCAB_SIZE, size=(2, SEQ_LEN)).astype("int32"),
    }


@pytest.fixture
def trainer() -> PTRMTrainer:
    """The REAL constructor, with no dataset (the step does not need one)."""
    keras.utils.set_random_seed(0)
    return PTRMTrainer(config=_tiny_config(), train_dataset=None)


def test_two_steps_have_a_finite_loss_and_move_every_weight(trainer):
    """Two steps, because the warmup schedule starts at ``warmup_start_lr=1e-8``.

    One step at 1e-8 moves the small weights but not an RMSNorm scale of 1.0
    (1.0 + 1e-8 rounds to 1.0 in float32), although its gradient is non-zero.
    The second step runs at about 1e-4 and moves every variable.
    """
    variables = trainer.model.trainable_variables
    before = [keras.ops.convert_to_numpy(v).copy() for v in variables]

    for seed in (1, 2):
        metrics = trainer.train_step(_batch(seed))
        assert np.isfinite(metrics["loss"]), metrics
        assert metrics["loss"] > 0.0, metrics
        assert 1.0 <= metrics["act_steps"] <= HALT_MAX_STEPS, metrics

    assert trainer.current_step == 2
    unmoved = [
        v.path
        for b, v in zip(before, variables)
        if np.array_equal(b, keras.ops.convert_to_numpy(v))
    ]
    assert not unmoved, f"train_step left these weights untouched: {unmoved}"


def test_the_step_cannot_be_traced_so_it_runs_eagerly(trainer):
    """The measured reason this file tests the step eagerly (module docstring).

    If this starts passing the trace, the step became traceable: replace this
    test with a ``jit_compile=True`` execution arm.
    """
    step = tf.function(trainer.train_step, jit_compile=True)
    with pytest.raises(NotImplementedError, match="condition of while loop"):
        step(_batch(1))
