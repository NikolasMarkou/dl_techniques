"""NAM's validity head emits logits and the trainer reads them so (D-151).

Until 2026-10-17 ``NAM.validity_head`` was ``Dense(1, activation='sigmoid')``,
its output key was ``"valid"``, and ``train_nam.py`` clipped it to
``[1e-7, 1 - 1e-7]`` before a probability BCE. The head is now linear, the key
is ``"valid_logits"``, and the BCE reads it with ``from_logits=True``.

Driven through the REAL compiled train step with every other loss weight zero:
the validity head alone memorises two expressions with opposite validity
targets, so ``L_valid`` falls far below ``ln 2`` (the loss of the logit 0) and
the logit's sign matches each target. A sigmoid head under ``from_logits=True``
cannot get below ``log(1 + e^0) = ln 2`` for a target of 0.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.neural_computer.nam import NAM, NAMConfig
from train.nam.train_nam import _make_compiled_train_fn

from .test_halting_loss import ACT_STEPS, _inputs


@pytest.fixture
def model():
    keras.utils.set_random_seed(0)
    nam = NAM(config=NAMConfig(
        hidden_size=32, num_heads=4, num_tree_layers=1, intermediate_size=64,
        memory_size=8, num_read_heads=2, max_expression_len=16,
        halt_max_steps=ACT_STEPS, hidden_dropout_rate=0.0, attention_dropout_rate=0.0))
    nam.build()
    return nam


def test_a_validity_bias_shift_moves_the_output_by_exactly_the_shift(model):
    batch = {"input_ids": _inputs()["input_ids"]}
    _, out = model(model.initial_carry(batch), batch, training=False)
    assert "valid" not in out
    base = keras.ops.convert_to_numpy(out["valid_logits"])
    shift = -50.0
    model.validity_head.bias.assign(model.validity_head.bias + shift)
    _, moved = model(model.initial_carry(batch), batch, training=False)
    moved = keras.ops.convert_to_numpy(moved["valid_logits"])
    bound = 8 * np.finfo(np.float32).eps * (np.abs(base).max() + abs(shift))
    np.testing.assert_allclose(moved - base, shift, rtol=0, atol=bound)


def test_the_validity_head_alone_memorises_opposite_targets(model):
    inputs = _inputs()
    inputs["target_validity"] = tf.constant([[1.0], [0.0]])
    fn = _make_compiled_train_fn(
        model=model, optimizer=keras.optimizers.Adam(5e-2), max_act_steps=ACT_STEPS,
        ponder_cost=0.0, result_loss_weight=0.0, valid_loss_weight=1.0,
        w_number=0.0, w_operator=0.0, w_reduction=0.0, w_halt=0.0)
    for _ in range(150):
        metrics = fn(**inputs)
    final_loss = float(metrics["L_valid"])
    assert final_loss < 0.1 * np.log(2.0), final_loss
    logits = keras.ops.convert_to_numpy(metrics["last_valid"])[:, 0]
    assert logits[0] > 0.0 > logits[1], logits
