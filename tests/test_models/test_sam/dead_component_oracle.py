"""
Dead-component instrument (shared oracle for the SAM trainer guards)
====================================================================

This module is an *instrument*, not a test suite. It is deliberately named
without a ``test_`` prefix so pytest does not collect it, mirroring
``tests/test_layers/test_mixtures/cluster_axis_oracle.py``. Its RED proofs live
in ``test_training_model.py`` (``TestDeadComponentInstrument``).

Why it exists
-------------
The dominant failure mode of a new guard in this repository is a probe that
passes *both ways*: green when the component under test is live AND green when
it is dead. Iteration 2's plan makes ``>=1 dead-component probe per loss path
and per training path`` a non-negotiable budget line (invariant I-5). This
module supplies the three primitives those probes need, and nothing else:

1. :func:`fit_one_step_moved_variables` -- run exactly one ``fit()`` step and
   report which trainable variables MOVED, **by name**. ``moved > 0`` is not an
   acceptable assertion (iteration 1 shipped ``118/137`` and ``111/137`` figures
   whose residual was never identified); this returns both name sets so a caller
   can pin an exact set and justify every non-mover.
2. :func:`outputs_stop_gradient` -- inject ``ops.stop_gradient`` on **every**
   output of a model, so a training path that is genuinely live raises
   ``ValueError: No gradients provided for any variable``. A training path that
   does NOT raise under this injection was never carrying gradient in the first
   place.
3. :func:`component_response` (+ the killers :func:`zeroed_variables`,
   :func:`destroy_negatives`, :func:`destroy_positives`) -- measure a metric
   before and after a component is destroyed, and report whether it actually
   moved. This is the instrument that catches the measured
   ``SegmentationLosses.focal_loss`` negative blindness (base ``0.041383``,
   bit-identical after every negative pixel is set to maximally wrong).

Every function here reports a NUMBER, never a bare boolean verdict alone: a
probe with no number is not a probe.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from unittest import mock

import keras
import numpy as np
from keras import ops

#: The exact message Keras 3.8.0 raises when every gradient is ``None``.
#: Asserted verbatim by the dead-component guards -- never ``raises(Exception)``.
NO_GRADIENTS_MESSAGE = "No gradients provided for any variable"


# ---------------------------------------------------------------------------
# Variable identity
# ---------------------------------------------------------------------------
def variable_labels(model: keras.Model) -> Tuple[str, ...]:
    """
    Stable, unique, human-readable labels for a model's trainable variables.

    Args:
        model: A BUILT model. An unbuilt model has an empty variable list and
            every downstream comparison would be vacuously green.

    Returns:
        One label per entry of ``model.trainable_variables``, in that order.
        The label is the variable's Keras ``path``; if two variables share a
        path (legal in Keras 3) the positional index is appended as ``#i`` so
        the labels remain a usable dict key.

    Raises:
        ValueError: if the model has no trainable variables, because every
            "moved / did not move" statement built on an empty list is true by
            vacuity.
    """
    variables = list(model.trainable_variables)
    if not variables:
        raise ValueError(
            "variable_labels() received a model with ZERO trainable variables. "
            "Build the model (call it once, or run one fit step) before "
            "measuring; an empty variable list makes every moved/unmoved claim "
            "vacuously true."
        )
    seen: Dict[str, int] = {}
    labels: List[str] = []
    for index, variable in enumerate(variables):
        path = getattr(variable, "path", None) or variable.name
        if path in seen:
            labels.append(f"{path}#{index}")
        else:
            seen[path] = index
            labels.append(path)
    return tuple(labels)


def _snapshot(model: keras.Model) -> Dict[str, np.ndarray]:
    """Copy every trainable variable's value, keyed by :func:`variable_labels`."""
    labels = variable_labels(model)
    return {
        label: np.array(ops.convert_to_numpy(variable), copy=True)
        for label, variable in zip(labels, model.trainable_variables)
    }


# ---------------------------------------------------------------------------
# (a) moved / unmoved variables after exactly one fit() step
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MovedVariablesReport:
    """
    Result of :func:`fit_one_step_moved_variables`.

    Attributes:
        moved: Labels of trainable variables whose value changed.
        unmoved: Labels of trainable variables whose value did NOT change.
            Every entry here is a defect until it is named and justified.
        max_abs_delta: Per-label maximum absolute change, so a caller can report
            a magnitude rather than a boolean.
        total: ``len(moved) + len(unmoved)``.
        final_loss: The loss reported by the single ``fit()`` step.
    """

    moved: Tuple[str, ...]
    unmoved: Tuple[str, ...]
    max_abs_delta: Dict[str, float]
    total: int
    final_loss: float

    @property
    def n_moved(self) -> int:
        """Number of variables that moved."""
        return len(self.moved)

    def summary(self) -> str:
        """One-line ``moved/total`` summary suitable for an assertion message."""
        return (
            f"{self.n_moved}/{self.total} trainable variables moved "
            f"(loss={self.final_loss:.6f}); unmoved={list(self.unmoved)}"
        )


def fit_one_step_moved_variables(
    model: keras.Model,
    x: Any,
    y: Any = None,
    *,
    epochs: int = 1,
    verbose: int = 0,
    **fit_kwargs: Any,
) -> MovedVariablesReport:
    """
    Run one ``fit()`` and report which trainable variables moved, by name.

    The model must already be COMPILED and BUILT. Building is the caller's job
    on purpose: a model built by ``fit()`` itself has no pre-step snapshot to
    compare against, and the resulting report would list every variable as
    "moved" merely because it did not exist before.

    Args:
        model: A compiled, built ``keras.Model``.
        x: Inputs, in any form ``fit()`` accepts (array, dict of arrays, or a
            ``tf.data.Dataset``).
        y: Targets, or ``None`` when ``x`` is a dataset that yields them.
        epochs: Passed through; defaults to a single epoch.
        verbose: Passed through; defaults to silent.
        **fit_kwargs: Forwarded verbatim to ``model.fit``.

    Returns:
        A :class:`MovedVariablesReport`.

    Raises:
        ValueError: if the model has no trainable variables (see
            :func:`variable_labels`), or whatever ``fit()`` itself raises --
            notably ``ValueError: No gradients provided for any variable`` when
            the training path is dead. That raise is the point of the
            instrument and is deliberately NOT swallowed.
    """
    # DECISION plan-2026-08-03T191222-1d751f81/D-034: the "before" snapshot is
    # taken HERE, before `fit()`, and the model is required to be built already.
    # Do NOT let `fit()` build the model and then compare two post-fit reads, and
    # do NOT "simplify" this by snapshotting once and re-reading the same
    # objects -- Keras variables are mutated IN PLACE, so a snapshot that is not
    # an explicit `np.array(..., copy=True)` taken before the step reports every
    # variable as unmoved. Iteration 1's D-018 lost an entire measurement to
    # exactly this timing error (a post-forward weight count reported 202/202
    # for both branches of a decision, hiding 64 re-initialized weights); the
    # re-break that pins this is `test_moved_report_names_exactly_the_live_branch`
    # plus `test_the_same_model_trains_without_the_injection`.
    before = _snapshot(model)
    history = model.fit(x, y, epochs=epochs, verbose=verbose, **fit_kwargs)
    after = _snapshot(model)

    moved: List[str] = []
    unmoved: List[str] = []
    deltas: Dict[str, float] = {}
    for label in before:
        delta = float(np.max(np.abs(after[label] - before[label])))
        deltas[label] = delta
        (moved if delta > 0.0 else unmoved).append(label)

    return MovedVariablesReport(
        moved=tuple(moved),
        unmoved=tuple(unmoved),
        max_abs_delta=deltas,
        total=len(before),
        final_loss=float(history.history["loss"][-1]),
    )


# ---------------------------------------------------------------------------
# (b) the killers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def outputs_stop_gradient(model: keras.Model) -> Iterator[None]:
    """
    Wrap every output of ``model.call`` in ``ops.stop_gradient`` for the block.

    This is the training-path dead-component injection. Inside the block a
    ``fit()`` step on a genuinely live model must raise
    ``ValueError: No gradients provided for any variable``
    (:data:`NO_GRADIENTS_MESSAGE`); if it does not, the "green" result outside
    the block proved nothing.

    ``mock.patch.object`` is used rather than a plain attribute assignment so
    the instance attribute is REMOVED on exit (restoring the class-level
    ``call``) instead of shadowing it permanently.

    Args:
        model: The model to sabotage. Mutated only for the duration of the
            ``with`` block.

    Yields:
        ``None``.
    """
    original_call = model.call

    def _stop_gradient_call(*args: Any, **kwargs: Any) -> Any:
        return keras.tree.map_structure(ops.stop_gradient, original_call(*args, **kwargs))

    with mock.patch.object(model, "call", _stop_gradient_call):
        yield


@contextlib.contextmanager
def zeroed_variables(variables: Iterable[Any]) -> Iterator[None]:
    """
    Set the given variables to all-zero for the block, then restore them exactly.

    Args:
        variables: Any iterable of Keras variables (e.g. ``layer.weights``).

    Yields:
        ``None``.

    Raises:
        ValueError: if ``variables`` is empty -- zeroing nothing and observing
            no change is the archetypal probe that passes both ways.
    """
    variables = list(variables)
    if not variables:
        raise ValueError(
            "zeroed_variables() received an EMPTY variable list. Killing "
            "nothing and observing no response is not a probe."
        )
    saved = [np.array(ops.convert_to_numpy(v), copy=True) for v in variables]
    try:
        for variable in variables:
            variable.assign(ops.zeros_like(variable))
        yield
    finally:
        for variable, value in zip(variables, saved):
            variable.assign(value)


@contextlib.contextmanager
def layer_returns_its_input(layer: Any, *, name: Optional[str] = None) -> Iterator[None]:
    """
    Replace ``layer.call`` with the identity for the block -- the layer is DEAD.

    This is the killer the "central claim" guards need: an attention block, a
    routing loop, a spline basis or a Fourier mixer that is replaced by
    ``lambda x: x`` keeps every shape, every weight and every serialized config
    intact, so a test suite that only checks those reports green on a model with
    the mechanism removed. Measured examples in this tree: ``CBAM.call ->
    return inputs`` passed all 16 of its tests; substituting
    ``keras.layers.Dense`` for the KAN spline layer passed everything.

    ``mock.patch.object`` is used rather than attribute assignment so the
    instance attribute is REMOVED on exit, restoring the class-level ``call``
    (same reason as :func:`outputs_stop_gradient`).

    Args:
        layer: The layer to kill. Only its ``call`` is replaced; weights,
            ``build`` state and config are untouched.
        name: Optional label used in the "never invoked" error.

    Yields:
        ``None``.

    Raises:
        AssertionError: on exit, if the identity was never actually invoked --
            i.e. the block ran without the layer being on the path at all, so
            "nothing changed" would have been an artefact of the injection
            rather than a property of the model.
    """
    # DECISION plan-2026-08-17T183311-79c63e38/D-042: the invocation counter and
    # its exit assertion are NOT ceremony -- do not drop them to "simplify" this
    # to a bare mock.patch. A killer applied to a layer that is not on the
    # executed path produces "nothing changed", which reads as EXACTLY the same
    # verdict as a model that ignores the component. Nor should this be replaced
    # by a per-package local injection: it is shared precisely so all ten
    # central-claim guards use one killer with one control. See decisions.md
    # D-042.
    calls = {"n": 0}

    def _identity(inputs: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return inputs

    with mock.patch.object(layer, "call", _identity):
        yield
    assert calls["n"] > 0, (
        f"the identity injection on {name or getattr(layer, 'name', layer)!r} "
        "was NEVER invoked: the layer is not on the executed path, so any "
        "'nothing changed' verdict measured under it is meaningless"
    )


def destroy_negatives(prediction: np.ndarray, ground_truth: np.ndarray, wrong: float = 0.99) -> np.ndarray:
    """
    Return ``prediction`` with every NEGATIVE pixel's value set maximally wrong.

    A loss that does not move under this transformation is blind to the
    background -- the measured failure of ``SegmentationLosses.focal_loss`` on a
    1-channel binary map (``0.041383`` before and after, identical to six
    decimals).

    Args:
        prediction: Predicted probabilities/logits, same shape as
            ``ground_truth``.
        ground_truth: Binary ground truth (``0`` = negative, ``1`` = positive).
        wrong: The value written into every negative pixel. Defaults to
            ``0.99`` (confidently predicting foreground where there is none).

    Returns:
        A NEW array; the input is not mutated.

    Raises:
        ValueError: if the shapes disagree, or if ``ground_truth`` has no
            negative pixel at all (the probe would then be a no-op).
    """
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"destroy_negatives shape mismatch: prediction {prediction.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )
    negatives = ground_truth <= 0.5
    if not bool(np.any(negatives)):
        raise ValueError(
            "destroy_negatives: ground truth contains NO negative pixel, so "
            "this probe would be a no-op and would pass regardless of the loss."
        )
    out = np.array(prediction, copy=True)
    out[negatives] = wrong
    return out


def destroy_positives(prediction: np.ndarray, ground_truth: np.ndarray, wrong: float = 0.01) -> np.ndarray:
    """
    Return ``prediction`` with every POSITIVE pixel's value set maximally wrong.

    Args:
        prediction: Predicted probabilities/logits, same shape as
            ``ground_truth``.
        ground_truth: Binary ground truth (``0`` = negative, ``1`` = positive).
        wrong: The value written into every positive pixel. Defaults to
            ``0.01`` (confidently predicting background on the object).

    Returns:
        A NEW array; the input is not mutated.

    Raises:
        ValueError: if the shapes disagree, or if ``ground_truth`` has no
            positive pixel at all.
    """
    prediction = np.asarray(prediction)
    ground_truth = np.asarray(ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"destroy_positives shape mismatch: prediction {prediction.shape} "
            f"vs ground_truth {ground_truth.shape}"
        )
    positives = ground_truth > 0.5
    if not bool(np.any(positives)):
        raise ValueError(
            "destroy_positives: ground truth contains NO positive pixel, so "
            "this probe would be a no-op and would pass regardless of the loss."
        )
    out = np.array(prediction, copy=True)
    out[positives] = wrong
    return out


# ---------------------------------------------------------------------------
# (c) did the metric actually move?
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ComponentResponse:
    """
    Result of :func:`component_response`.

    Attributes:
        name: Human-readable name of the component that was killed.
        before: Metric value with the component intact.
        after: Metric value with the component destroyed.
        delta: ``abs(after - before)``.
        moved: ``delta > atol``.
        atol: The threshold used.
    """

    name: str
    before: float
    after: float
    delta: float
    moved: bool
    atol: float

    def summary(self) -> str:
        """One-line summary carrying the two measured numbers."""
        verdict = "MOVED" if self.moved else "DID NOT MOVE"
        return (
            f"{self.name}: {verdict} -- before={self.before:.6f} "
            f"after={self.after:.6f} delta={self.delta:.6f} (atol={self.atol})"
        )


def component_response(
    metric_fn: Callable[[], float],
    kill: Callable[[], ContextManager[Any]],
    *,
    name: str,
    atol: float = 0.0,
) -> ComponentResponse:
    """
    Measure a metric with a component intact and with it destroyed.

    Args:
        metric_fn: Zero-argument callable returning the metric as a float. It
            is invoked twice and must be deterministic, or the measured delta is
            noise rather than a response.
        kill: Zero-argument callable returning a context manager that destroys
            the component for the duration of its block. A *callable* rather
            than a context manager, so the object is constructed fresh and
            cannot be accidentally re-entered.
        name: What is being killed, for the report.
        atol: Deltas at or below this are reported as "did not move". Defaults
            to ``0.0`` (any change at all counts), which is what makes a
            bit-identical blindness visible.

    Returns:
        A :class:`ComponentResponse`.
    """
    before = float(metric_fn())
    with kill():
        after = float(metric_fn())
    delta = float(abs(after - before))
    return ComponentResponse(
        name=name,
        before=before,
        after=after,
        delta=delta,
        moved=delta > atol,
        atol=atol,
    )


@contextlib.contextmanager
def no_op_kill() -> Iterator[None]:
    """
    A killer that destroys nothing.

    Used as the instrument's own negative control: :func:`component_response`
    must report ``moved=False`` with ``delta == 0.0`` under this killer, or the
    measurement is picking up nondeterminism rather than a response.

    Yields:
        ``None``.
    """
    yield
