import keras
from typing import (
    Any,
    Callable,
)

# ---------------------------------------------------------------------------

def activation_spec(activation: Any) -> Any:
    """
    Canonicalise an activation spec for storage on the layer.

    Returns the spec in a form that :func:`_serialize_activation` can round-trip:
    ``None`` and strings pass through; a serialized dict (as produced by
    deserialization of a saved config) is turned back into a callable; anything
    else is returned unchanged.

    :param activation: String name, ``None``, serialized dict, or callable.
    :return: Canonical activation spec.
    """
    if activation is None or isinstance(activation, str):
        return activation
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


def resolve_activation(activation: Any) -> Callable[[Any], Any]:
    """
    Resolve an activation spec to a callable.

    Strings are resolved via ``keras.activations.get``; ``None`` maps to
    identity (linear); callables are returned as-is.  Stateful activation
    *layers* are rejected: they would create their weights during ``call()``
    instead of ``build()``, which breaks ``.keras`` weight loading.

    :param activation: String name, ``None``, serialized dict, or callable.
    :return: A callable applying the activation.
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

    ``None`` and strings pass through unchanged; callables are serialized via
    ``keras.saving.serialize_keras_object`` so that a config containing a raw
    function object is still JSON-serialisable.

    :param activation: Canonical activation spec.
    :return: JSON-serialisable representation.
    """
    if activation is None or isinstance(activation, str):
        return activation
    return keras.saving.serialize_keras_object(activation)

# ---------------------------------------------------------------------------

