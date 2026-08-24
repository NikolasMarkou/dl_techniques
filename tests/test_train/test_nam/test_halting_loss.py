"""RED proof for C-35(b) (plan-2026-08-14T233721-d4f9beb2, D-034).

`src/train/nam/train_nam.py` appended `q_halt_logits` / `q_continue_logits` to
`all_q_halt` / `all_q_cont` and no loss in the file ever read them — there was no
`L_halt` term at all. The ACT head was therefore never trained, so tuning
`--halt-exploration-prob` or `halt_max_steps` changed nothing measurable.

This drives the REAL compiled train step, not a re-implementation of it: if the
term is removed from `train_nam.py`, or `--w-halt` stops reaching it, the
assertions below fire.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.nam import NAM, NAMConfig
from train.nam.train_nam import _make_compiled_train_fn

B, L, ACT_STEPS = 2, 16, 2


@pytest.fixture
def model():
    keras.utils.set_random_seed(0)
    return NAM(
        config=NAMConfig(
            hidden_size=32,
            num_heads=4,
            num_tree_layers=1,
            intermediate_size=64,
            memory_size=8,
            num_read_heads=2,
            max_expression_len=L,
            halt_max_steps=ACT_STEPS,
            hidden_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )
    )


def _inputs():
    ids = np.zeros((B, L), dtype="int32")
    ids[:, :5] = np.array([[1, 5, 12, 7, 2], [1, 9, 13, 3, 2]], dtype="int32")
    return dict(
        input_ids=tf.constant(ids),
        targets=tf.constant([[12.0], [12.0]]),
        target_validity=tf.constant([[1.0], [1.0]]),
        true_left=tf.constant([[5.0], [9.0]]),
        true_right=tf.constant([[7.0], [3.0]]),
        true_op_index=tf.constant([0, 0], dtype=tf.int32),
        true_op_position=tf.constant([2, 2], dtype=tf.int32),
        per_step_token_ids=tf.constant(np.repeat(ids[:, None, :], ACT_STEPS, axis=1)),
        per_step_op_index=tf.zeros((B, ACT_STEPS), dtype=tf.int32),
        per_step_op_position=tf.fill((B, ACT_STEPS), 2),
        per_step_valid_mask=tf.ones((B, ACT_STEPS)),
    )


def _run(model, w_halt):
    model.build()
    optimizer = keras.optimizers.SGD(learning_rate=1.0)
    fn = _make_compiled_train_fn(
        model=model,
        optimizer=optimizer,
        max_act_steps=ACT_STEPS,
        ponder_cost=0.0,
        result_loss_weight=0.0,
        valid_loss_weight=0.0,
        w_number=0.0,
        w_operator=0.0,
        w_reduction=0.0,
        w_halt=w_halt,
    )
    kernel = model.cell.halt_head.kernel
    before = np.asarray(keras.ops.convert_to_numpy(kernel)).copy()
    metrics = fn(**_inputs())
    after = np.asarray(keras.ops.convert_to_numpy(kernel))
    return metrics, float(np.max(np.abs(after - before)))


class TestTheTrainerActuallyTrainsTheHaltHead:
    def test_l_halt_is_reported_and_moves_the_halt_head(self, model):
        """THE defect: with every other loss weight zeroed, only L_halt can move it."""
        metrics, delta = _run(model, w_halt=1.0)

        assert "L_halt" in metrics, (
            "the compiled train step reports no L_halt; the ACT loss is missing "
            "from train_nam.py again (decisions.md D-034)"
        )
        assert float(metrics["L_halt"]) > 0.0, (
            f"L_halt is {float(metrics['L_halt'])}, so the term contributes "
            "nothing to the total loss"
        )
        assert delta > 1e-8, (
            f"halt_head.kernel moved by {delta:.3e} under a step where L_halt is "
            "the ONLY weighted term — the halting head is not being trained"
        )

    def test_w_halt_zero_leaves_the_halt_head_untouched(self, model):
        """ANTI-VACUITY control: the movement above comes from L_halt, not drift.

        Reproduces the pre-D-034 behaviour exactly. If this fails, some other
        term is moving `halt_head` and the assertion above proves nothing about
        the halting loss.
        """
        _, delta = _run(model, w_halt=0.0)
        assert delta == 0.0, (
            f"with w_halt=0 and every other weight zeroed, halt_head.kernel "
            f"still moved by {delta:.3e}; the probe is not isolating L_halt"
        )
