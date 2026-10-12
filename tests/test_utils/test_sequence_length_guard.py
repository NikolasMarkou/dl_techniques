"""Unit contract for ``dl_techniques.utils.sequence_length_guard``.

The behaviour on real models (a refused first call leaves the model usable,
XLA refuses an over-long retrace) is pinned in
``tests/test_models/test_a_position_limit_is_refused_cleanly.py``. This file
pins the module's own promises, which no model test isolates:

* ``refuse_static_sequence_length`` is a no-op for ``None`` and for the limit
  itself, and does NOT touch the cached step functions then; above the limit it
  nulls all three BEFORE raising, with the caller's message verbatim.
* ``gate_dynamic_sequence_length`` returns a symbolic ``KerasTensor`` unchanged,
  returns the VALUES of a legal input bit for bit, and names the XLA guard node
  after the caller's limit name.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.sequence_length_guard import (
    gate_dynamic_sequence_length,
    refuse_static_sequence_length,
)


class _Holder:
    """Stands in for a model: only the three cached step functions matter."""

    def __init__(self):
        self.predict_function = object()
        self.train_function = object()
        self.test_function = object()

    def cached(self):
        return [self.predict_function, self.train_function, self.test_function]


@pytest.mark.parametrize("seq_len", [None, 1, 16])
def test_a_legal_or_unknown_length_is_a_no_op_and_keeps_the_cached_functions(seq_len):
    holder = _Holder()
    before = holder.cached()
    refuse_static_sequence_length(holder, seq_len, 16, "msg")
    assert holder.cached() == before


def test_an_over_long_length_drops_all_three_functions_and_raises_the_message():
    holder = _Holder()
    with pytest.raises(ValueError) as excinfo:
        refuse_static_sequence_length(holder, 17, 16, "the exact message")
    assert str(excinfo.value) == "the exact message"
    assert holder.cached() == [None, None, None]


def test_a_symbolic_tensor_is_returned_unchanged():
    symbolic = keras.Input(shape=(None,), dtype="int32")
    assert gate_dynamic_sequence_length(symbolic, 16, "msg", "max_len") is symbolic


def test_a_legal_dynamic_length_keeps_the_values_bit_for_bit():
    ids = np.arange(2 * 7, dtype="int32").reshape(2, 7)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def run(x):
        return gate_dynamic_sequence_length(x, 16, "msg", "max_len")

    assert np.array_equal(run(ids).numpy(), ids)


def test_an_over_long_dynamic_length_raises_with_the_message_on_the_graph_path():
    ids = np.zeros((2, 17), dtype="int32")

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def run(x):
        return gate_dynamic_sequence_length(x, 16, "the exact message", "max_len")

    with pytest.raises(tf.errors.InvalidArgumentError, match="the exact message"):
        run(ids)


def test_the_xla_guard_node_is_named_after_the_limit_name():
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def run(x):
        return gate_dynamic_sequence_length(x, 16, "msg", "max_len")

    names = [op.name for op in run.get_concrete_function().graph.get_operations()]
    assert any(n.endswith("seq_len_exceeds_max_len_16") for n in names), names
