"""Symmetric (de)serialization for ``activation``-valued constructor arguments.

Why this module exists
======================

A class that accepts ``activation: Union[str, Callable]`` and stores that value
**raw** in ``get_config()`` is broken for every non-string value. MEASURED at
HEAD 2026-08-23 on a minimal reproduction, and independently on ``vit_siglip``
/ ``vit`` / ``vit_hmlp``:

============================  =========  =======================  ==============
value passed as ``activation``  ``save()``  ``load_model()``         ``.activation``
                                            after load
============================  =========  =======================  ==============
``"gelu"`` (a string)          ok         ok, ``max|delta|``=0.0   ``str`` -- fine
a **registered** callable      ok         ok, ``max|delta|``=0.0   a raw **dict**
an **unregistered** callable   ok         **ValueError**: "Could
                                          not interpret activation
                                          function identifier: {...}"  --
unregistered + ``custom_objects``  ok     ok                       a raw **dict**
============================  =========  =======================  ==============

Two consequences drive the shape of this module.

1. A guard written with a *registered* callable is **vacuous** on forward
   output: ``max|delta|`` is 0.0 with and without the repair. The observable
   that actually discriminates is (a) whether ``get_config()`` is
   JSON-serializable and (b) whether ``.activation`` is still callable after a
   round-trip. Only an **unregistered** callable exercises the load-time raise.
2. The repair is a **pair**. ``serialize_activation`` in ``get_config`` alone
   leaves the loaded attribute a dict that the next ``get_config`` propagates;
   ``deserialize_activation`` alone leaves ``get_config`` non-JSON-safe. The two
   halves fail *different* assertions, which is how they are RED-proven
   separately in
   ``tests/test_utils/test_activation_serialization.py``.

Interface contract
==================

The pair is exactly inverse on the values this repository stores, and is a
**no-op on every shipped config**, because every shipped config passes a string.

``serialize_activation(activation) -> Any``
    - ``keras.layers.Layer``  -> ``keras.saving.serialize_keras_object`` dict
    - any other callable      -> ``keras.activations.serialize`` (a plain name
      string for a Keras builtin, a config dict for a user function)
    - **anything else, including ``str``, ``None`` and ``bool``, is returned
      unchanged.** This is required, not incidental:
      ``keras.activations.serialize`` REJECTS a bare string ("Unknown
      activation function 'gelu' cannot be serialized"), and many callers in
      this tree store a **dl_techniques activation-factory key** such as
      ``'mish'`` or ``'sparsemax'`` which is not a Keras activation at all and
      must survive verbatim.
    - Never raises for the value types this repository stores.

``deserialize_activation(activation, custom_objects=None) -> Any``
    - ``dict`` -> ``keras.saving.deserialize_keras_object`` (this dispatches
      both the function form and the Layer form)
    - **anything else -- ``str``, ``None``, ``bool``, an already-live callable
      -- is returned unchanged.**
    - Never raises for the value types this repository stores.

Where to call them
==================

``serialize_activation`` goes in ``get_config``. ``deserialize_activation``
goes in ``__init__``, on the way into the attribute -- **not** in
``from_config``. MEASURED: ``keras.models.load_model(..., custom_objects=...)``
runs sub-object construction inside a ``custom_object_scope``, so the
``custom_objects=None`` default resolves an unregistered function correctly
from ``__init__``; and the ``__init__`` site additionally covers ``Cls(**cfg)``
by hand and nested layers whose parent never calls their ``from_config``.
"""

from typing import Any, Dict, Optional

import keras

# ---------------------------------------------------------------------


def serialize_activation(activation: Any) -> Any:
    """
    Make an ``activation`` value safe to place in a ``get_config()`` dict.

    :param activation: A string factory/Keras key, ``None``, a callable, a
        ``keras.layers.Layer``, or any other value a caller stored verbatim.
    :type activation: Any
    :return: A JSON-serializable stand-in. Strings, ``None`` and other
        non-callables are returned **unchanged** -- see the module docstring
        for why that passthrough is mandatory.
    :rtype: Any
    """
    if isinstance(activation, keras.layers.Layer):
        return keras.saving.serialize_keras_object(activation)
    if callable(activation):
        return keras.activations.serialize(activation)
    return activation


# ---------------------------------------------------------------------


def deserialize_activation(
    activation: Any,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Invert :func:`serialize_activation`.

    :param activation: The value read back out of a config dict.
    :type activation: Any
    :param custom_objects: Optional name -> object mapping used to resolve a
        callable that is not registered with
        ``keras.saving.register_keras_serializable``. Usually left ``None``:
        ``load_model`` installs a ``custom_object_scope`` that this call sees.
    :type custom_objects: Optional[Dict[str, Any]]
    :return: The live activation. Non-dict inputs are returned **unchanged**.
    :rtype: Any
    """
    if isinstance(activation, dict):
        return keras.saving.deserialize_keras_object(
            activation, custom_objects=custom_objects
        )
    return activation


# ---------------------------------------------------------------------
