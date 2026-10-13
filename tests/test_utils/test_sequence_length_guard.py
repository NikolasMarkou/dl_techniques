"""Unit contract for ``dl_techniques.utils.sequence_length_guard``.

The behaviour on real models (a refused first call leaves the model usable,
XLA refuses an over-long retrace) is pinned in
``tests/test_models/test_a_position_limit_is_refused_cleanly.py``. This file
pins the module's own promises, which no model test isolates:

* ``refuse_static_sequence_length`` is a no-op for ``None`` and for the limit
  itself, and does NOT touch the cached step functions then; above the limit it
  raises with the caller's message verbatim, nulling all three first on an
  UNBUILT model only. A built model keeps them: a refusal caught inside a
  running ``fit`` (a callback probing an over-long call) would otherwise leave
  the fit loop calling a ``None`` ``train_function``. The error is a
  ``SequenceLengthError``, a ``ValueError`` subclass: a wrapper model catches
  exactly that type from its backbone's build, drops its OWN cached functions
  with ``drop_cached_step_functions`` and re-raises.
* ``gate_dynamic_sequence_length`` returns a symbolic ``KerasTensor`` unchanged,
  returns the VALUES of a legal input bit for bit, and names the XLA guard node
  after the caller's limit name.
* ``check_sequence_length`` routes by the sequence axis: a static length above
  the limit raises the formatted message and drops the cached functions, the
  limit itself and anything of rank below 2 pass through untouched, and a
  ``None`` axis goes to the dynamic gate with ``{seq_len}`` rendered as
  ``<run-time length>``.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.utils.sequence_length_guard import (
    SequenceLengthError,
    check_sequence_length,
    drop_cached_step_functions,
    gate_dynamic_sequence_length,
    refuse_static_sequence_length,
)


class _Holder:
    """Stands in for a model: its ``built`` flag and three cached step functions."""

    def __init__(self, built: bool = False):
        self.built = built
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


def test_an_over_long_length_on_an_unbuilt_model_drops_all_three_functions_and_raises():
    holder = _Holder(built=False)
    with pytest.raises(ValueError) as excinfo:
        refuse_static_sequence_length(holder, 17, 16, "the exact message")
    assert str(excinfo.value) == "the exact message"
    assert holder.cached() == [None, None, None]


def test_an_over_long_length_on_a_built_model_keeps_all_three_functions_and_raises():
    holder = _Holder(built=True)
    before = holder.cached()
    with pytest.raises(ValueError) as excinfo:
        refuse_static_sequence_length(holder, 17, 16, "the exact message")
    assert str(excinfo.value) == "the exact message"
    assert holder.cached() == before


@pytest.mark.parametrize("built", [False, True])
def test_the_refusal_is_a_sequence_length_error_and_still_a_value_error(built):
    with pytest.raises(SequenceLengthError) as excinfo:
        refuse_static_sequence_length(_Holder(built=built), 17, 16, "msg")
    assert isinstance(excinfo.value, ValueError)


def test_drop_cached_step_functions_nulls_all_three_whatever_the_built_flag():
    holder = _Holder(built=True)
    drop_cached_step_functions(holder)
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


_TEMPLATE = "len {seq_len} over {limit}"


def test_check_a_static_over_long_length_raises_the_formatted_message():
    holder = _Holder()
    ids = keras.KerasTensor((2, 17), dtype="int32")
    with pytest.raises(ValueError) as excinfo:
        check_sequence_length(holder, ids, 16, "max_len", _TEMPLATE)
    assert str(excinfo.value) == "len 17 over 16"
    assert holder.cached() == [None, None, None]


@pytest.mark.parametrize("shape", [(2, 16), (2, 3), (7,), ()])
def test_check_the_limit_itself_and_low_rank_tensors_pass_through(shape):
    holder = _Holder()
    before = holder.cached()
    ids = keras.KerasTensor(shape, dtype="int32")
    assert check_sequence_length(holder, ids, 16, "max_len", _TEMPLATE) is ids
    assert holder.cached() == before


def test_check_a_symbolic_dynamic_axis_is_returned_unchanged():
    ids = keras.Input(shape=(None,), dtype="int32")
    assert check_sequence_length(_Holder(), ids, 16, "max_len", _TEMPLATE) is ids


def test_check_a_dynamic_axis_goes_to_the_gate_with_the_run_time_placeholder():
    ids = np.zeros((2, 17), dtype="int32")

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def run(x):
        return check_sequence_length(_Holder(), x, 16, "max_len", _TEMPLATE)

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="len <run-time length> over 16"
    ):
        run(ids)
