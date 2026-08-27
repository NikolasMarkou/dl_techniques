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
    if activation is None or isinstance(activation, str):
        return activation
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


def resolve_activation(activation: Any) -> Callable[[Any], Any]:
    """
    Resolve an activation spec to a callable.

    Checks are applied in the order shown below, and the first match wins.
    ``None`` maps to ``keras.activations.linear``, which is the identity.
    Strings go through ``keras.activations.get``. Serialized dicts go through
    ``keras.activations.deserialize``. Anything left over is assumed to be a
    callable and returned unchanged.

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
             returned unchanged

    The last branch does no checking. A non-callable that reaches it fails
    later, at the call site, not here.

    :param activation: String name, ``None``, serialized dict, or callable.
    :type activation: Any
    :return: A callable applying the activation.
    :rtype: Callable[[Any], Any]
    :raises ValueError: If ``activation`` is a ``keras.layers.Layer``.
    """
    if isinstance(activation, keras.layers.Layer):
        raise ValueError(
            "Activation must be a string name or a plain callable, not a "
            f"keras Layer instance ({type(activation).__name__}). Layer "
            "activations may own weights, which would be created during "
            "call() rather than build() and would not survive a .keras "
            "round-trip. Use e.g. 'leaky_relu' or keras.activations.silu."
        )
    if activation is None:
        return keras.activations.linear
    if isinstance(activation, str):
        return keras.activations.get(activation)
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


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
    if activation is None or isinstance(activation, str):
        return activation
    return keras.saving.serialize_keras_object(activation)

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

