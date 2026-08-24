"""
Smoke-contract oracle -- proves a smoke test's assertion can actually fail
=========================================================================

This module is an *instrument*, not a test suite. It is deliberately named
without a ``test_`` prefix so pytest does not collect it, mirroring
``tests/test_models/test_sam/dead_component_oracle.py`` and
``tests/test_layers/test_mixtures/cluster_axis_oracle.py``. Its own RED proofs
live in ``tests/test_models/test_smoke_contract_oracle.py``.

Why it exists
-------------
Every ``tests/test_models/*/test_smoke.py`` file used to wrap construction AND
the forward pass in ``except Exception: pytest.xfail(...)``. A total build break
therefore reported ``xfail`` -- green -- and the assertions after the block were
unreachable whenever anything threw. Ten sites carried that shape.

Removing the wrapper fixes half the problem. The other half is that most of
these smoke tests assert only ``isinstance(out, dict)`` and finiteness, so a
model whose forward returns a scalar ``0.0`` still passes. A smoke test that
cannot distinguish a working forward from a destroyed one is not an instrument
either, and "I added a shape assertion" is a claim that needs proving rather
than asserting.

The precedent this replaces
---------------------------
``test_yolo12``'s original ``test_the_smoke_test_fails_on_a_build_break`` proved
the wrong thing: it passed ``scale="not_a_scale"``, which fails at the variant
lookup *inside the factory*, before the model is ever built. Delete every
assertion in that file's smoke body and the meta-test still passed. A meta-test
must break the MODEL, not its argument validation.

How to use it
-------------
Factor the smoke test's assertion into a module-level ``_assert_contract(out)``
function, then::

    def test_smoke_build_and_forward(model):
        _assert_contract(model(_inputs(), training=False))

    def test_the_contract_rejects_a_broken_forward(model):
        assert_contract_rejects_a_broken_forward(model, _inputs(), _assert_contract)

:func:`assert_contract_rejects_a_broken_forward` runs the contract on the REAL
output first (it must pass -- otherwise a later "it raised" proves nothing but
that the contract is broken), then re-runs it under each breaker in
:data:`DEFAULT_BREAKERS` and requires an ``AssertionError`` from every one.

Why ``AssertionError`` specifically, and not ``Exception``
----------------------------------------------------------
``pytest.raises(Exception)`` is the shape that let the yolo12 meta-test pass
vacuously. A ``TypeError`` from a contract that indexed a scalar is the contract
*crashing*, not the contract *judging*; accepting it would let a contract of
``assert out["k"].shape == ...`` count as a real guard even though it never
checks that ``out`` is a dict at all. Requiring ``AssertionError`` forces every
contract to assert its container and key set before it reaches for a ``.shape``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence, Tuple
from unittest import mock

import keras
import numpy as np
from keras import ops


# ---------------------------------------------------------------------------
# Structure walking
# ---------------------------------------------------------------------------
def _map_tensors(structure: Any, fn: Callable[[Any], Any]) -> Any:
    """
    Apply ``fn`` to every tensor leaf of a dict/list/tuple structure.

    ``keras.tree.map_structure`` is deliberately NOT used: several models in
    this tree return dicts with a ``None`` value (``DistilBERT`` returns
    ``{"last_hidden_state": ..., "attention_mask": None}`` when no mask is
    passed -- measured), and the tree libraries disagree about whether ``None``
    is a leaf or an empty subtree. Here ``None`` is passed through untouched.

    Args:
        structure: A tensor, or an arbitrarily nested dict/list/tuple of them.
        fn: Applied to each non-``None`` leaf.

    Returns:
        A new structure of the same container types.
    """
    # DECISION plan-2026-08-17T183311-79c63e38/D-035: do NOT "simplify" this to
    # `keras.tree.map_structure`. DistilBERT's forward returns
    # `{"last_hidden_state": ..., "attention_mask": None}` (MEASURED), and the
    # tree libraries disagree about whether `None` is a leaf or an empty
    # subtree -- under map_structure the injection silently no-ops on that
    # entry, which makes the breaker weaker than it reports. `None` is passed
    # through untouched here on purpose. See decisions.md D-035.
    if structure is None:
        return None
    if isinstance(structure, Mapping):
        return {key: _map_tensors(value, fn) for key, value in structure.items()}
    if isinstance(structure, (list, tuple)):
        mapped = [_map_tensors(value, fn) for value in structure]
        return type(structure)(mapped) if isinstance(structure, tuple) else mapped
    return fn(structure)


# ---------------------------------------------------------------------------
# The breakers
# ---------------------------------------------------------------------------
def collapse_to_scalar(output: Any) -> Any:
    """
    Replace the ENTIRE return value with the scalar ``0.0``.

    Kills every structural guard: ``isinstance(out, dict)``, ``len(feats) == 3``,
    a key-set assertion, and any shape assertion. This is the exact degenerate
    output the finiteness-only smoke tests accepted -- ``np.isnan(0.0)`` is
    ``False`` and ``np.isinf(0.0)`` is ``False``, so a contract of "every value
    is finite" reports green on it.

    Args:
        output: The real forward output (ignored).

    Returns:
        A rank-0 tensor of value ``0.0``.
    """
    return ops.convert_to_tensor(0.0)


def slice_leading_axis(output: Any) -> Any:
    """
    Keep only the first element along axis 0 of every tensor leaf.

    Kills leading-dimension guards (batch size, node count, sequence length)
    while preserving the container type, the key set, the rank, and finiteness.
    A contract that checks ``isinstance(out, dict)`` and the key set but not the
    shapes survives this one -- which is the point: it discriminates a real
    shape assertion from a structural one.

    Args:
        output: The real forward output.

    Returns:
        The same structure with each leaf sliced to ``leaf[:1]``.

    Raises:
        ValueError: if any leaf is rank-0, in which case this breaker is a no-op
            on that leaf and the resulting "the contract did not raise" verdict
            would be an artefact of the breaker rather than of the contract.
    """

    def _slice(tensor: Any) -> Any:
        if len(tensor.shape) == 0:
            raise ValueError(
                "slice_leading_axis received a rank-0 leaf; it cannot break a "
                "scalar, so requiring the contract to reject it would be a "
                "vacuous demand. Drop this breaker for that model."
            )
        return tensor[:1]

    return _map_tensors(output, _slice)


def append_trailing_axis(output: Any) -> Any:
    """
    Append a size-1 trailing axis to every tensor leaf.

    Kills full-shape guards while surviving partial ones. This breaker exists
    because a contract written as ``tuple(feat.shape[:3]) == (2, 8, 8)`` -- the
    original yolo12 smoke assertion -- silently ignores the channel dimension;
    under this breaker it still passes, and the guard is then correctly reported
    as not rejecting a broken forward.

    Args:
        output: The real forward output.

    Returns:
        The same structure with each leaf gaining a trailing axis.
    """
    return _map_tensors(output, lambda tensor: ops.expand_dims(tensor, axis=-1))


#: The breakers every smoke contract must reject. Each targets a different
#: class of under-assertion; see each function's docstring.
DEFAULT_BREAKERS: Tuple[Callable[[Any], Any], ...] = (
    collapse_to_scalar,
    slice_leading_axis,
    append_trailing_axis,
)


# ---------------------------------------------------------------------------
# The injection
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def broken_forward(model: keras.Model, breaker: Callable[[Any], Any]) -> Iterator[None]:
    """
    Route ``model``'s forward output through ``breaker`` for the block.

    ``mock.patch.object`` is used rather than a plain attribute assignment so
    the instance attribute is REMOVED on exit (restoring the class-level
    ``call``) instead of shadowing it permanently -- the same reason
    ``dead_component_oracle.outputs_stop_gradient`` does it. Verified to work on
    both subclassed models and models produced by the functional API, because
    Keras 3's ``Layer.__call__`` dispatches through ``self.call`` in both cases.

    Args:
        model: The model to sabotage. Mutated only for the duration of the block.
        breaker: Receives the real output, returns the degenerate one.

    Yields:
        ``None``.
    """
    original_call = model.call

    def _broken_call(*args: Any, **kwargs: Any) -> Any:
        return breaker(original_call(*args, **kwargs))

    with mock.patch.object(model, "call", _broken_call):
        yield


# ---------------------------------------------------------------------------
# The assertion
# ---------------------------------------------------------------------------
def assert_contract_rejects_a_broken_forward(
    model: keras.Model,
    inputs: Any,
    contract: Callable[[Any], None],
    *,
    breakers: Sequence[Callable[[Any], Any]] = DEFAULT_BREAKERS,
) -> Dict[str, str]:
    """
    Prove a smoke test's assertion can fail, by breaking the model's forward.

    Args:
        model: A model that can be called as ``model(inputs, training=False)``.
        inputs: Whatever that call takes -- an array, a dict, a list, a tuple.
        contract: The smoke test's own assertion function. Called with the
            forward output; must raise ``AssertionError`` when the output is
            wrong and return normally when it is right.
        breakers: The degenerate-output transforms to test against. Defaults to
            :data:`DEFAULT_BREAKERS`. Pass a narrower tuple only with a written
            reason -- each omitted breaker is a class of under-assertion this
            call stops checking for.

    Returns:
        A ``{breaker name: the AssertionError's message}`` map, so a caller can
        report what the guard actually said rather than a bare boolean.

    Raises:
        AssertionError: if the contract rejects the REAL output (the guard is
            broken, and any later rejection would prove nothing), or if any
            breaker fails to make it raise ``AssertionError``.
    """
    if not breakers:
        raise ValueError(
            "assert_contract_rejects_a_broken_forward() was given an EMPTY "
            "breaker list. Breaking nothing and observing no rejection is not a "
            "proof."
        )

    # Anti-vacuity control: the contract must PASS on the real output. Without
    # this, a contract that raises unconditionally would satisfy every breaker
    # below and be reported as a strong guard.
    contract(model(inputs, training=False))

    rejections: Dict[str, str] = {}
    for breaker in breakers:
        name = breaker.__name__
        with broken_forward(model, breaker):
            broken_output = model(inputs, training=False)
        # DECISION plan-2026-08-17T183311-79c63e38/D-035: catch `AssertionError`, NOT `Exception`. The
        # broader form is what made yolo12's original meta-test vacuous. A
        # `TypeError` from a contract that indexed a scalar is the contract
        # CRASHING, not judging; accepting it would let `assert out["k"].shape
        # == ...` -- which never checks that `out` is a dict -- count as a real
        # guard. Do not widen this. See decisions.md D-035.
        try:
            contract(broken_output)
        except AssertionError as exc:
            rejections[name] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"breaker {name!r} made the contract raise "
                f"{type(exc).__name__}: {exc} -- that is the contract CRASHING, "
                f"not judging. Assert the container type and key set before "
                f"indexing into the output."
            ) from exc
        raise AssertionError(
            f"breaker {name!r} did NOT make the contract fail: the smoke "
            f"assertion accepts a forward whose output was transformed by "
            f"{name}. See {name}'s docstring for the class of under-assertion "
            f"this indicates."
        )
    return rejections


def assert_finite(value: Any) -> None:
    """
    Assert every leaf of ``value`` is non-NaN and non-inf.

    Shared by the smoke tests so the identical eight-line helper stops being
    copied into every one of them. ``None`` leaves are skipped.

    Args:
        value: A tensor, or a nested dict/list/tuple of them.

    Raises:
        AssertionError: on the first NaN or inf found.
    """

    def _check(tensor: Any) -> Any:
        array = np.asarray(ops.convert_to_numpy(tensor))
        assert not np.any(np.isnan(array)), f"NaN in output of shape {array.shape}"
        assert not np.any(np.isinf(array)), f"inf in output of shape {array.shape}"
        return tensor

    _map_tensors(value, _check)
