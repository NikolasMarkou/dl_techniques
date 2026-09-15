"""
Shared helpers for this package: activation-argument handling, and the
axis-vs-rank arithmetic that several layers need in more than one method.

Three helpers for handling a layer's ``activation`` argument.

A layer that takes an ``activation`` argument has to cope with four input
forms: a string name, ``None``, a plain callable, and a serialized dict
(which is what a saved config yields). These three functions cover the round
trip:

- :func:`activation_spec` canonicalises the constructor argument into the
  value the layer stores.
- :func:`resolve_activation` turns that stored value into a callable that
  ``call()`` can apply.
- :func:`serialize_activation` turns it back into something JSON can hold,
  for ``get_config``.

Use all three or none. Storing the raw constructor argument instead of
``activation_spec``'s output means a layer rebuilt from a config holds a
different kind of value than one built from scratch, and
:func:`serialize_activation` can no longer round-trip it.

There is a name clash inside this package. ``gelu_tanh.py`` also defines a
``resolve_activation``, and that is the one ``__init__.py`` exports. The two
are different functions: this one rejects ``keras.layers.Layer`` instances,
the other extends ``keras.activations.get`` with the tanh-GELU spellings.
Import this one explicitly, ``from .common import resolve_activation``.

Two further helpers, :func:`axis_is_in_range` and :func:`normalize_axis`,
carry the ``axis``/rank arithmetic that ``build``, ``call`` and
``compute_output_shape`` each have to redo. They are pure functions of their
two arguments -- they read no layer state -- so a
``compute_output_shape`` on an UNBUILT layer can call them, and a method can
resolve an axis against the rank of the shape it was HANDED rather than
against a rank cached at build time.
"""

import keras
from typing import (
    Any,
    Callable,
)

from dl_techniques.utils.activation_serialization import (
    deserialize_activation as _deserialize_activation,
    serialize_activation as _serialize_activation,
)

# ---------------------------------------------------------------------------

def activation_spec(activation: Any) -> Any:
    """
    Canonicalise an activation spec for storage on the layer.

    ``None`` and strings pass through untouched. A dict goes through
    ``keras.activations.deserialize`` and comes back as a callable. Anything
    else is returned unchanged.

    Store the result on the layer, not the raw constructor argument.

    :param activation: String name, ``None``, serialized dict, or callable.
    :type activation: Any
    :return: Canonical activation spec: ``None``, a string, or a callable.
    :rtype: Any
    """
    # DECISION plan-2026-09-15T034909-a7edc8da/D-002
    # Thin wrapper, not a deletion -- see decisions.md D-002. This module's
    # own historical behavior differs from `activation_serialization.py`'s
    # in the dict branch (`keras.activations.deserialize`, not
    # `keras.saving.deserialize_keras_object`), which is why this one line
    # stays inline rather than delegating: the unified helper's dict path
    # additionally dispatches Layer construction, a form this function's
    # contract has never accepted (`resolve_activation` below rejects a
    # resolved Layer, so a config-carried Layer must never reach that far).
    # DECISION plan-2026-09-15T034909-a7edc8da/D-003
    # MEASURED, not assumed: a full delegation to `deserialize_activation`
    # would raise `TypeError` for an unregistered custom activation dict that
    # `keras.activations.deserialize` (used here) resolves successfully. See
    # decisions.md D-003.
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


def resolve_activation(activation: Any) -> Callable[[Any], Any]:
    """
    Resolve an activation spec to a callable.

    Checks are applied in the order shown below, and the first match wins.
    ``None`` maps to ``keras.activations.linear``, which is the identity.
    Strings go through ``keras.activations.get``. Serialized dicts go through
    ``keras.activations.deserialize``. Anything left over must be callable;
    a non-callable value raises ``ValueError``, matching
    ``keras.activations.get``'s contract.

    A ``keras.layers.Layer`` is rejected. A layer can own weights, and those
    weights would be created during ``call()`` rather than ``build()``, which
    breaks ``.keras`` weight loading. Use ``'leaky_relu'`` or
    ``keras.activations.silu`` instead.

    **Architecture Overview:**

    .. code-block:: text

                 activation
                      │
                      ▼
        ┌───────────────────────────┐
        │ is a keras Layer?         │──── yes ──► ValueError
        └─────────────┬─────────────┘
                      │ no
                      ▼
        ┌───────────────────────────┐
        │ is None?                  │──── yes ──► activations.linear
        └─────────────┬─────────────┘
                      │ no
                      ▼
        ┌───────────────────────────┐
        │ is a str?                 │──── yes ──► activations.get
        └─────────────┬─────────────┘
                      │ no
                      ▼
        ┌───────────────────────────┐
        │ is a dict?                │──── yes ──► deserialize
        └─────────────┬─────────────┘
                      │ no
                      ▼
        ┌───────────────────────────┐
        │ is callable?              │──── no ───► ValueError
        └─────────────┬─────────────┘
                      │ yes
                      ▼
             returned unchanged

    The last branch checks ``callable()`` before returning. A non-callable
    value (an int, float, bool, list, ...) raises ``ValueError`` immediately,
    here, rather than deferring to a confusing ``TypeError`` at the call
    site inside ``call()``.

    :param activation: String name, ``None``, serialized dict, or callable.
    :type activation: Any
    :return: A callable applying the activation.
    :rtype: Callable[[Any], Any]
    :raises ValueError: If ``activation`` is a ``keras.layers.Layer``, or if
        it is a non-callable value that is not ``None``, a string, or a dict.
    """
    # DECISION plan-2026-09-15T034909-a7edc8da/D-002
    # Partial delegation, not a full one -- see decisions.md D-002. The
    # str/dict resolution stays inline: the unified helper's `str` arm is a
    # passthrough (it returns the STRING unchanged, because its ~60 existing
    # callers store the string itself), whereas this function's contract is
    # to return a CALLABLE -- delegating the str branch would silently hand
    # back a non-callable string. Its `dict` arm goes through
    # `keras.saving.deserialize_keras_object`, which raises for an
    # unregistered custom activation function that `keras.activations.deserialize`
    # (used here) resolves successfully -- MEASURED, not assumed: delegating
    # that branch too would regress a currently-working round-trip. Only the
    # final Layer-rejection raise is unified, by handing the fully-resolved
    # value to `_deserialize_activation`, which is a no-op resolution step
    # for anything that already isn't a dict and applies the one shared raise
    # message if what's left is a `keras.layers.Layer`.
    # DECISION plan-2026-09-15T034909-a7edc8da/D-003
    # MEASURED, not assumed: delegating the str branch would return the bare
    # STRING unchanged instead of a callable, breaking this function's
    # documented "returns a callable" contract. See decisions.md D-003.
    if activation is None:
        resolved = keras.activations.linear
    elif isinstance(activation, str):
        resolved = keras.activations.get(activation)
    elif isinstance(activation, dict):
        resolved = keras.activations.deserialize(activation)
    else:
        # DECISION plan-2026-09-15T034909-a7edc8da/D-014
        # Restore the callable() validation `keras.activations.get` provided
        # pre-migration -- see decisions.md D-014. Without this check, a
        # non-callable garbage value (an int, float, bool, list) fell through
        # to `_deserialize_activation(resolved, allow_layer=False)` below,
        # which only rejects a resolved `keras.layers.Layer` and otherwise
        # returns its input UNCHANGED -- so `resolve_activation(7)` returned
        # `7` instead of raising, deferring the failure to a confusing
        # `TypeError: 'int' object is not callable` inside the layer's
        # `call()` instead of an immediate `ValueError` at construction time.
        # This check must run BEFORE the `_deserialize_activation` call so a
        # `keras.layers.Layer` (which IS callable) still reaches that
        # function's own, more specific Layer-rejection message rather than
        # this generic one.
        if not callable(activation):
            raise ValueError(
                f"Could not interpret activation function identifier: {activation!r}"
            )
        resolved = activation
    return _deserialize_activation(resolved, allow_layer=False)


def serialize_activation(activation: Any) -> Any:
    """
    Serialize an activation spec for ``get_config``.

    ``None`` and strings pass through unchanged. Anything else goes through
    ``keras.saving.serialize_keras_object``, so a config holding a raw
    function object is still JSON-serialisable.

    Pass the value :func:`activation_spec` produced, not the raw constructor
    argument.

    :param activation: Canonical activation spec.
    :type activation: Any
    :return: JSON-serialisable representation.
    :rtype: Any
    """
    # DECISION plan-2026-09-15T034909-a7edc8da/D-002
    # Thin wrapper -- see decisions.md D-002. Behaviorally identical to the
    # old inline body: `activation_serialization.serialize_activation`
    # already passes `None`/`str` through unchanged and routes everything
    # else through `keras.saving.serialize_keras_object` for a plain
    # callable, and through the same function for a `Layer` (this module's
    # `resolve_activation` never lets a `Layer` reach storage, so that arm is
    # unreachable here, not a behavior change).
    # DECISION plan-2026-09-15T034909-a7edc8da/D-003
    # This is the one function of the three verified safe for FULL delegation
    # -- see decisions.md D-003.
    return _serialize_activation(activation)

# ---------------------------------------------------------------------------

def axis_is_in_range(axis: int, rank: int) -> bool:
    """
    Report whether ``axis`` addresses a real dimension of a rank-``rank`` tensor.

    The legal range is ``[-rank, rank - 1]``: ``rank`` distinct dimensions,
    each reachable by one non-negative and one negative index.

    This is the single predicate behind every axis range check in the package
    that has to run in more than one method. Both call sites in a layer must
    use THIS function rather than re-typing the comparison, so that the two
    cannot drift apart; that drift is the defect the helper exists to prevent.

    Pure function of its arguments. It reads no layer state, so it is safe to
    call from ``compute_output_shape`` on an unbuilt layer.

    Failure mode: none. It raises nothing and never returns anything but a
    ``bool``. Callers own the error message, because the two current callers
    raise deliberately different texts and both texts are asserted by tests.

    :param axis: The configured axis, negative or non-negative.
    :type axis: int
    :param rank: Number of dimensions of the tensor or shape in hand. Must be
        the rank of the shape the CALLER was given, not one cached earlier.
    :type rank: int
    :return: ``True`` if ``-rank <= axis < rank``, else ``False``.
    :rtype: bool
    """
    return -rank <= axis < rank


def normalize_axis(axis: int, rank: int) -> int:
    """
    Convert a possibly-negative ``axis`` to its non-negative equivalent.

    ``-1`` becomes ``rank - 1``, ``-rank`` becomes ``0``, and a non-negative
    axis passes through unchanged.

    Pure function of its arguments, for the same reason as
    :func:`axis_is_in_range`: resolving against the rank actually in hand is
    what makes it correct to call from a method that may see a different rank
    than ``build`` did.

    Failure mode: none, and **no range check**. Out of range in gives out of
    range out (``normalize_axis(5, 3) == 5``). Gate it with
    :func:`axis_is_in_range` first if the value is not already trusted.

    :param axis: The configured axis, negative or non-negative.
    :type axis: int
    :param rank: Number of dimensions of the tensor or shape in hand.
    :type rank: int
    :return: ``axis + rank`` when ``axis < 0``, otherwise ``axis``.
    :rtype: int
    """
    return rank + axis if axis < 0 else axis

# ---------------------------------------------------------------------------

