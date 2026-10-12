"""Refuse a sequence longer than a model's position limit, on every path.

Why this exists
---------------
A model with a real position limit (a learned position table, a precomputed
RoPE table) fails on an over-long sequence in ways that depend on the DEVICE
and on HOW the model is called. Measured on twelve language models
(``plans/plan-2026-10-11T170958-dc4d2202/findings/carried-technical-items.md``
item 1):

* **Finite garbage under XLA.** Keras 3.8 resolves ``jit_compile="auto"`` to
  XLA whenever a GPU is visible. A position table read by a GATHER then returns
  zeros for an out-of-range index and the model returns a finite, wrong
  answer with no error (DistilBERT, FNet, GPT2, TopoLM). On CPU the same call
  raises ``InvalidArgumentError``.
* **A model left unusable by its own refusal.** A ``ValueError`` raised during
  the FIRST trace of Keras's cached ``predict``/``train``/``test`` step function
  leaves that cached function broken for every later call ("Structures don't
  have the same nested structure", "Creating variables on a non-first call"),
  so the next legal ``predict`` fails too.

Two functions answer the two defects. A model calls the first from ``build``
(before any variable exists) and again from ``call`` for a statically known
length, and calls the second from ``call`` when the sequence axis is ``None``.

.. code-block:: python

    def build(self, input_shape):
        refuse_static_sequence_length(
            self, seq_len, self.max_position_embeddings, message
        )
        ...

    def call(self, input_ids, ...):
        if input_ids.shape[-1] is None:
            input_ids = gate_dynamic_sequence_length(
                input_ids, self.max_position_embeddings, message,
                "max_position_embeddings",
            )

The model keeps its own message (it names its own attribute and what to do)
and its own limit attribute; this module owns only the mechanism.
"""

from typing import Optional

import keras

# ---------------------------------------------------------------------

__all__ = ["gate_dynamic_sequence_length", "refuse_static_sequence_length"]

# ---------------------------------------------------------------------


def refuse_static_sequence_length(
        model: keras.Model,
        seq_len: Optional[int],
        limit: int,
        message: str,
) -> None:
    """Raise for a statically known ``seq_len`` above ``limit``.

    Interface contract (call sites: the ``build`` and ``call`` of every model
    with a position limit, starting with ``BERT``). A no-op when ``seq_len`` is
    ``None`` (length unknown, see :func:`gate_dynamic_sequence_length`) or
    ``seq_len <= limit`` (the limit itself is legal). Otherwise it first sets
    ``model.predict_function``, ``model.train_function`` and
    ``model.test_function`` to ``None`` so Keras builds fresh ones on the next
    call, then raises. Call it BEFORE any variable is created: variables made
    inside a step-function trace that then fails are left unusable.

    :param model: The model whose cached step functions are dropped.
    :param seq_len: The static sequence length, or ``None`` when unknown.
    :param limit: The largest legal sequence length.
    :param message: The text of the raised error; the caller owns the wording.
    :raises ValueError: If ``seq_len`` exceeds ``limit``.
    """
    if seq_len is None or seq_len <= limit:
        return
    # A refusal during the FIRST trace of a cached Keras step function leaves
    # that function unusable for every later call ("Structures don't have the
    # same nested structure", or "Creating variables on a non-first call");
    # drop them so the next call builds fresh ones.
    model.predict_function = None
    model.train_function = None
    model.test_function = None
    raise ValueError(message)

# ---------------------------------------------------------------------


def gate_dynamic_sequence_length(
        input_ids: keras.KerasTensor,
        limit: int,
        message: str,
        limit_name: str,
) -> keras.KerasTensor:
    """Gate ``input_ids`` on a run-time length check inside a TF graph.

    Interface contract (call sites: ``call`` of every model with a position
    limit, for an input whose sequence axis is ``None``). Returns
    ``input_ids`` itself, unchanged, off the TensorFlow backend or for a
    symbolic :class:`keras.KerasTensor` (there is nothing to check). Inside a
    TensorFlow graph it returns ``input_ids`` plus an all-zero term sized
    ``limit - seq_len``, both gated on a run-time assertion; the value is
    ``input_ids`` bit for bit when the length is legal. An over-long batch
    raises ``tf.errors.InvalidArgumentError`` (a Python ``ValueError`` cannot
    cross a graph boundary): carrying ``message`` on the graph path, naming the
    node ``seq_len_exceeds_<limit_name>_<limit>`` under XLA. The model must
    stay usable afterwards.

    XLA drops the assertion, and Keras's default ``jit_compile="auto"`` selects
    XLA whenever a GPU is visible. XLA compiles each length with static shapes,
    so the zero term, whose size goes negative for an over-long length, fails
    that compile instead.

    :param input_ids: Token ids whose sequence axis is the LAST axis and is not
        statically known.
    :param limit: The largest legal sequence length.
    :param message: The text of the run-time assertion; the caller owns the
        wording.
    :param limit_name: Name of the limit attribute, used only to name the XLA
        guard node (``seq_len_exceeds_<limit_name>_<limit>``) so a failure
        points at the right knob.
    :return: ``input_ids``, or ``input_ids`` gated on the length check.
    """
    # DECISION plan-2026-10-11T004932-b83935e1/D-021: raw TF on purpose; do NOT replace with a
    # keras.ops expression -- keras.ops has no assertion, and without the control dependency the
    # out-of-range gather can fail first with an opaque message (or, on GPU, return zeros). See decisions.md.
    # DECISION plan-2026-10-11T170958-dc4d2202/D-006: one shared mechanism for every class with a
    # position limit; do NOT copy this back into a model file or gate only the shared embedding
    # layers (that leaves a refused first call unusable). See decisions.md.
    if keras.backend.backend() != "tensorflow" or isinstance(
        input_ids, keras.KerasTensor
    ):
        return input_ids
    import tensorflow as tf

    seq_len = tf.shape(input_ids)[-1]
    check = tf.debugging.assert_less_equal(seq_len, limit, message=message)
    with tf.control_dependencies([check]):
        # DECISION plan-2026-10-11T004932-b83935e1/D-030: XLA drops `check`, which Keras enables
        # by default when a GPU is visible; do NOT remove this zero-sum term or opt the model out of
        # XLA instead. XLA compiles each length with static shapes, so a negative size here
        # fails that compile, naming this node. See decisions.md.
        xla_guard = tf.zeros(
            tf.reshape(limit - seq_len, [1]),
            dtype=input_ids.dtype,
            name=f"seq_len_exceeds_{limit_name}_{limit}",
        )
        return input_ids + tf.reduce_sum(xla_guard)

# ---------------------------------------------------------------------
