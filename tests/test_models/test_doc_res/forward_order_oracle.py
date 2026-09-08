"""Shared instrument: the ASSEMBLED model's real leaf-op execution order.

Named without a ``test_`` prefix so pytest does not collect it, following the
``*_oracle.py`` convention of ``tests/test_models/``. Two modules in this
directory consume it and it is written once:

* ``test_model.py`` re-runs ``test_components.py``'s two bracketing assertions
  over the assembled model. That is the step-5 obligation: within a single
  component both resamplers END at their pixel op, so the "and is followed by"
  half of the D-007 adjacency is unanswerable there and can only be closed
  across the component boundary that ``doc_res/model.py`` draws.
* ``test_architecture_facts.py`` reads the channel width of the tensors that
  actually flowed, rather than a config attribute, for the
  ``decoder_level1``/``refinement``-run-at-96 guard.

Why a recorded forward and not a source grep or a restated op list
------------------------------------------------------------------
``DocRes.call`` is imperative: there is no functional graph to walk and no
``op_sequence`` property spanning the whole model. The two rejected
alternatives are both worse:

* **Re-typing the op order in the test.** That is a second copy of ``call``,
  and it agrees with itself no matter what ``call`` does.
* **Grepping ``model.py``.** Source text is not the forward path; this repo has
  a recorded case (LESSONS: "a byte-identity guard measures text, not
  behaviour") of exactly that confusion.

:func:`record_forward_ops` instead patches ``keras.layers.Layer.__call__`` for
the duration of ONE forward pass and appends a record when a LEAF layer
returns. Leaf completion order is execution order for a model with no parallel
branches, and the patch is removed in a ``finally`` so a raising forward cannot
leak it into the rest of the session.

The two things that would make it lie, and the guards against them
------------------------------------------------------------------
1. **A recorded op that is not a leaf**, or a leaf that is not recorded, would
   silently reorder the sequence. :func:`assert_every_leaf_is_recorded` pins the
   recorded set against the model's own recursive layer walk, so a
   sub-layer that ``call`` never runs (over-build) and one that runs but was
   not tracked both fire.
2. **A non-layer op between two records.** ``keras.ops.concatenate``,
   ``keras.ops.add`` and ``keras.activations.gelu`` are FUNCTIONS, not layers,
   so no patch can see them and "temporal adjacency" is not automatically
   "data-flow adjacency". :func:`assert_consumes` closes that gap for the sites
   that matter by comparing VALUES: the successor's input must either BE the
   predecessor's output, or contain it verbatim as the leading slice of a
   channel-axis concatenation. A blind function in between would change the
   values and fail.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, List, NamedTuple, Sequence

import keras
import numpy as np

__all__ = [
    "OpRecord",
    "ForwardTrace",
    "leaf_layers",
    "record_forward",
    "record_forward_ops",
    "assert_every_leaf_is_recorded",
    "assert_consumes",
    "RecordedForward",
]


class OpRecord(NamedTuple):
    """One leaf layer's invocation during a recorded forward pass.

    :ivar layer: The leaf layer that ran.
    :ivar inputs: Its first positional input, as a numpy array.
    :ivar outputs: Its return value, as a numpy array.
    """

    layer: keras.layers.Layer
    inputs: np.ndarray
    outputs: np.ndarray


def leaf_layers(model: keras.Model) -> List[keras.layers.Layer]:
    """Every sub-layer of ``model`` that owns no sub-layer of its own.

    Interface contract (3 call sites, all in this directory): recursive, the
    model itself excluded, order unspecified (the caller compares SETS by
    ``id``). A composite such as ``RestormerTransformerBlock`` is deliberately
    absent -- it contributes no op of its own, only the ops of its children,
    and recording both would double-count the sequence.

    :param model: A BUILT model. An unbuilt one has an empty sub-layer tree and
        every downstream set comparison would be vacuously true.
    :return: The leaf layers.
    :rtype: List[keras.layers.Layer]
    :raises ValueError: If the walk finds no leaf at all.
    """
    everything = list(model._flatten_layers(include_self=False, recursive=True))
    leaves = [
        layer for layer in everything
        if not list(layer._flatten_layers(include_self=False, recursive=True))
    ]
    if not leaves:
        raise ValueError(
            f"{type(model).__name__} exposes no leaf sub-layer; build the "
            "model before recording, or every order assertion below is "
            "vacuously true"
        )
    return leaves


# DECISION plan-2026-09-08T111844-de235227/D-017
# A recorded forward, NOT a re-typed op list and NOT a grep over `model.py`.
# Do not "simplify" this monkeypatch away: `DocRes.call` is imperative, so
# re-typing its op order in the test produces a second copy that agrees with
# itself no matter what `call` does, and a source grep measures text rather
# than behaviour. The patch is scoped to one forward and restored in a
# `finally`; a leaked one would turn every later forward in the session into an
# array copy. See D-017 in the plan's decisions.md.
@contextlib.contextmanager
def _patched_call(tracked: dict, records: List[OpRecord]) -> Iterator[None]:
    """Record every tracked layer's invocation for the duration of the block."""
    original = keras.layers.Layer.__call__

    def patched(self, *args, **kwargs):
        outputs = original(self, *args, **kwargs)
        if id(self) in tracked:
            first = args[0] if args else kwargs.get("inputs")
            records.append(OpRecord(
                layer=self,
                inputs=np.asarray(keras.ops.convert_to_numpy(first)),
                outputs=np.asarray(keras.ops.convert_to_numpy(outputs)),
            ))
        return outputs

    keras.layers.Layer.__call__ = patched
    try:
        yield
    finally:
        # A leaked patch would follow every later test in the session and
        # convert every forward pass in the process into an array copy.
        keras.layers.Layer.__call__ = original


class ForwardTrace(NamedTuple):
    """One recorded forward pass.

    :ivar records: Leaf invocations in execution order.
    :ivar outputs: What the MODEL returned, as a numpy array. Kept beside the
        records because the interesting question about the last op is whether
        anything happened AFTER it, and only these two together can answer it.
    """

    records: List[OpRecord]
    outputs: np.ndarray


def record_forward(model: keras.Model, inputs: Any) -> ForwardTrace:
    """Run ONE forward pass and return its leaf ops plus the model's output.

    :param model: A BUILT model.
    :param inputs: Whatever ``model(inputs, training=False)`` accepts.
    :return: The trace. Record order is leaf COMPLETION order, which is
        execution order for a model with no parallel branches.
    :rtype: ForwardTrace
    """
    tracked = {id(layer): layer for layer in leaf_layers(model)}
    records: List[OpRecord] = []
    with _patched_call(tracked, records):
        outputs = model(inputs, training=False)
    return ForwardTrace(
        records=records,
        outputs=np.asarray(keras.ops.convert_to_numpy(outputs)),
    )


def record_forward_ops(model: keras.Model, inputs: Any) -> List[OpRecord]:
    """:func:`record_forward` for the callers that need only the op order."""
    return record_forward(model, inputs).records


def assert_every_leaf_is_recorded(
        model: keras.Model,
        records: Sequence[OpRecord],
) -> None:
    """Assert the recorded ops are EXACTLY the model's leaf layers, once each.

    This is the build-parity claim stated over the forward path rather than
    over a weight list, and it is two-sided on purpose: a leaf that never ran
    is an over-built tree (a sub-layer ``call`` does not use, whose weights are
    dead), and a leaf that ran twice would make "the op after this one" an
    ambiguous phrase.

    :param model: The model that was recorded.
    :param records: The output of :func:`record_forward_ops`.
    :raises AssertionError: naming the offending layers.
    """
    expected = {id(layer): layer for layer in leaf_layers(model)}
    counts: dict = {}
    for record in records:
        counts[id(record.layer)] = counts.get(id(record.layer), 0) + 1

    never_ran = sorted(layer.name for key, layer in expected.items()
                       if key not in counts)
    assert not never_ran, (
        f"{len(never_ran)} leaf sub-layers exist but `call` never ran them, so "
        f"they are built weights on no forward path: {never_ran}"
    )
    repeated = sorted(f"{expected[key].name} x{n}"
                      for key, n in counts.items() if n != 1)
    assert not repeated, (
        f"leaf sub-layers ran more than once in a single forward: {repeated}"
    )


def assert_consumes(producer: OpRecord, consumer: OpRecord) -> str:
    """Assert ``consumer`` really reads ``producer``'s output, and say how.

    Temporal adjacency in the recorded sequence is not by itself data-flow
    adjacency: ``keras.ops.concatenate`` is a function and no layer patch can
    observe it. This compares VALUES instead, at ``atol=0`` -- floats are
    copied, not recomputed, across a concatenation.

    :param producer: The earlier record.
    :param consumer: The record that must consume it.
    :return: ``"direct"`` or ``"concat(axis=-1)"``, so the caller can assert
        which mechanism it expected rather than accepting either silently.
    :rtype: str
    :raises AssertionError: If the consumer's input contains no trace of the
        producer's output.
    """
    made, taken = producer.outputs, consumer.inputs
    if made.shape == taken.shape and np.array_equal(made, taken):
        return "direct"
    channels = made.shape[-1]
    if (
            made.shape[:-1] == taken.shape[:-1]
            and taken.shape[-1] > channels
            and np.array_equal(taken[..., :channels], made)
    ):
        return "concat(axis=-1)"
    raise AssertionError(
        f"{consumer.layer.name} does not read {producer.layer.name}'s output: "
        f"produced {made.shape}, consumed {taken.shape}; the consumed tensor "
        "neither equals it nor carries it as the leading channel slice of a "
        "concatenation, so something per-channel blind runs in between"
    )


class RecordedForward:
    """Adapter presenting a recorded forward as an ``op_sequence`` holder.

    Exists so ``test_components.py``'s two bracketing assertions -- written
    against a component's ``op_sequence`` -- can be re-run verbatim over the
    ASSEMBLED model instead of being re-implemented for it. Reimplementing them
    is what would let the model-level copy drift from the component-level one.

    :ivar records: The recorded ops, in execution order.
    """

    def __init__(self, records: Sequence[OpRecord]) -> None:
        self.records = list(records)

    @property
    def op_sequence(self) -> List[keras.layers.Layer]:
        """The recorded leaf layers, in execution order."""
        return [record.layer for record in self.records]
