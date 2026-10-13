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

Two functions answer the two defects, and a third routes between them.
:func:`refuse_static_sequence_length` raises for a statically known length and
:func:`gate_dynamic_sequence_length` guards a length known only at run time.
A model calls :func:`check_sequence_length` from ``call`` right after it
resolves ``input_ids``; that is the whole wiring for a model whose ``build``
traces ``call`` (``materialize_sublayers``), because the trace reaches the
refusal before any sub-layer is built. A model whose ``build`` builds its
sub-layers by hand calls :func:`refuse_static_sequence_length` first in
``build`` too, so no variable exists when it refuses.

.. code-block:: python

    class MyModel(keras.Model):
        _SEQUENCE_LENGTH_MESSAGE = (
            "seq_len={seq_len} exceeds max_position_embeddings={limit}. ..."
        )

        def call(self, input_ids, ...):
            input_ids = check_sequence_length(
                self, input_ids, self.max_position_embeddings,
                "max_position_embeddings", self._SEQUENCE_LENGTH_MESSAGE,
            )

The model keeps its own message (it names its own attribute and what to do)
and its own limit attribute; this module owns only the mechanism.

A refusal has one exception: rebuilding from a shape recorded in a stored
``.keras`` archive. Keras 3.8 runs that rebuild (``Model.build_from_config``)
inside a bare ``try/except`` and then marks the model built anyway, so a
refusal there loads the archive with no weights at all. A guarded class lists
:class:`RecordedBuildShapeMixin` first among its bases; during that rebuild the
refusal logs a warning and returns instead.
"""

from typing import Any, Dict, Optional

import keras

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

__all__ = [
    "RecordedBuildShapeMixin",
    "SequenceLengthError",
    "check_sequence_length",
    "drop_cached_step_functions",
    "gate_dynamic_sequence_length",
    "refuse_static_sequence_length",
]

# ---------------------------------------------------------------------


class SequenceLengthError(ValueError):
    """A statically known sequence length above a model's position limit.

    Raised by :func:`refuse_static_sequence_length`. A ``ValueError``, so every
    caller that catches or expects ``ValueError`` is unchanged; the subclass
    exists so a WRAPPER whose ``build`` builds a guarded backbone can tell a
    length refusal from any other build error, drop its own cached step
    functions (:func:`drop_cached_step_functions`) and re-raise, staying
    unbuilt. ``CausalLanguageModel`` and ``MaskedLanguageModel`` do.
    """

# ---------------------------------------------------------------------


class RecordedBuildShapeMixin:
    """Let a guarded model rebuild from an over-long shape a stored archive recorded.

    Interface contract (13 guarded models list it FIRST among their bases,
    ``class BERT(RecordedBuildShapeMixin, keras.Model)``; BLT keeps its own
    override). Overrides only ``build_from_config``: it sets
    ``_loading_recorded_build_shape = True`` on the model and on every layer
    of ``model._flatten_layers()`` that declares that attribute
    (``TextDecoder``, ``PositionalEmbedding``), calls the next
    ``build_from_config`` in the MRO (Keras's), and sets them all back to
    ``False`` in ``finally``. While the flag is set,
    :func:`refuse_static_sequence_length` logs its message and returns. It adds
    no constructor argument, no config key and no registration, so
    ``get_config`` and the serialized form are unchanged; returns nothing, and
    anything Keras's ``build_from_config`` raises propagates.

    Keras 3.8 calls ``build_from_config`` on every ``.keras`` load, with the
    shape the model recorded when it was saved, inside a bare ``try/except``,
    and then sets ``built = True`` whatever happened. An archive written
    before the guard, at a length above the limit, therefore loaded with
    every weight left at its random initialization and only a ``UserWarning``.
    """

    _loading_recorded_build_shape = False

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild from ``config`` with the length refusal turned into a warning.

        :param config: The build config recorded in the archive.
        """
        layers = [
            layer for layer in self._flatten_layers()
            if hasattr(layer, "_loading_recorded_build_shape")
        ]
        for layer in layers:
            layer._loading_recorded_build_shape = True
        try:
            super().build_from_config(config)
        finally:
            for layer in layers:
                layer._loading_recorded_build_shape = False

# ---------------------------------------------------------------------


def drop_cached_step_functions(model: keras.Model) -> None:
    """Set the model's three cached Keras step functions to ``None``.

    Interface contract (call sites: :func:`refuse_static_sequence_length` for
    the guarded model, and the ``build`` of a wrapper that catches
    :class:`SequenceLengthError` from its backbone). Sets
    ``model.predict_function``, ``model.train_function`` and
    ``model.test_function`` to ``None`` so Keras builds fresh ones on the next
    call; returns nothing and never raises. A refusal during the FIRST trace
    of a cached step function leaves that function unusable for every later
    call, and each model, a wrapper included, caches its own.

    :param model: The model whose step functions are dropped.
    """
    model.predict_function = None
    model.train_function = None
    model.test_function = None

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
    ``seq_len <= limit`` (the limit itself is legal). Otherwise it raises,
    and on a model that is not yet built it first sets
    ``model.predict_function``, ``model.train_function`` and
    ``model.test_function`` to ``None`` so Keras builds fresh ones on the next
    call. A built model keeps its step functions: a refusal there fires
    outside any first trace that creates variables, and ``fit`` makes
    ``train_function`` once before its epoch loop, so a refusal caught by a
    callback mid-fit that dropped it would kill the running fit (``None`` is
    not callable). Call it BEFORE any variable is created: variables made
    inside a step-function trace that then fails are left unusable. While
    ``model._loading_recorded_build_shape`` is true (set only by
    :meth:`RecordedBuildShapeMixin.build_from_config`) it logs ``message`` as
    a warning and returns instead.

    :param model: The model whose cached step functions are dropped when it
        is not yet built.
    :param seq_len: The static sequence length, or ``None`` when unknown.
    :param limit: The largest legal sequence length.
    :param message: The text of the raised error; the caller owns the wording.
    :raises SequenceLengthError: If ``seq_len`` exceeds ``limit`` (a
        ``ValueError``).
    """
    if seq_len is None or seq_len <= limit:
        return
    # DECISION plan-2026-10-13T035756-5dd4d94d/D-004: warn and build while a stored archive's
    # recorded shape is rebuilt; do NOT raise here -- Keras swallows a build_from_config raise and
    # marks the model built, so the archive loads with no weights. See decisions.md.
    if getattr(model, "_loading_recorded_build_shape", False):
        logger.warning(
            f"{message} Building anyway because this shape came from a stored "
            f"config; the model refuses an input this long at call time."
        )
        return
    # A refusal during the FIRST trace of a cached Keras step function leaves
    # that function unusable for every later call ("Structures don't have the
    # same nested structure", or "Creating variables on a non-first call");
    # drop them so the next call builds fresh ones. Only while unbuilt: the
    # refusal inside `build` sees built=False, and a built model's functions
    # may belong to a running `fit` loop.
    if not getattr(model, "built", False):
        drop_cached_step_functions(model)
    raise SequenceLengthError(message)

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


def check_sequence_length(
        model: keras.Model,
        input_ids: keras.KerasTensor,
        limit: int,
        limit_name: str,
        message: str,
) -> keras.KerasTensor:
    """Apply the right half of the guard to the ids a model's ``call`` resolved.

    Interface contract (call site: one line in the ``call`` of every model
    with a position limit, right after it resolves ``input_ids``). A statically
    known sequence axis goes to :func:`refuse_static_sequence_length`; an
    unknown one (``None``) goes to :func:`gate_dynamic_sequence_length`. A
    tensor of rank below 2, or one with no ``shape``, is returned untouched.
    Returns ``input_ids`` (gated when the axis is dynamic); the value is
    unchanged bit for bit for every legal length.

    :param model: The model whose limit is enforced.
    :param input_ids: Token ids whose LAST axis is the sequence.
    :param limit: The largest legal sequence length.
    :param limit_name: Name of the limit attribute on the model, shown in the
        XLA guard node name.
    :param message: Error text, a ``str.format`` template that may use
        ``{seq_len}`` (the offending length, or ``<run-time length>`` when it
        is only known inside a graph) and ``{limit}``.
    :return: ``input_ids``, or ``input_ids`` gated on the length check.
    :raises SequenceLengthError: For a statically known length above
        ``limit`` (a ``ValueError``).
    """
    shape = getattr(input_ids, "shape", None)
    if shape is None or len(shape) < 2:
        return input_ids
    seq_len = shape[-1]
    if seq_len is None:
        return gate_dynamic_sequence_length(
            input_ids,
            limit,
            message.format(seq_len="<run-time length>", limit=limit),
            limit_name,
        )
    refuse_static_sequence_length(
        model,
        int(seq_len),
        limit,
        message.format(seq_len=int(seq_len), limit=limit),
    )
    return input_ids

# ---------------------------------------------------------------------
