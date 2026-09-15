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
    - Never raises for the value types this repository stores. Behavior is
      identical for both ``allow_layer`` policies below -- the reject/allow
      split only matters on the deserialize (input) side, see D-002.

Two ``deserialize_activation`` contracts, selected by ``allow_layer``
======================================================================

``deserialize_activation(activation, custom_objects=None, allow_layer=True) -> Any``
    - ``allow_layer=True`` (the default -- preserves this module's original,
      Layer-permissive behavior for its ~60 existing callers):

      - ``dict`` -> ``keras.saving.deserialize_keras_object`` (this dispatches
        both the function form and the Layer form)
      - **anything else -- ``str``, ``None``, ``bool``, an already-live
        callable -- is returned unchanged.**
      - Never raises for the value types this repository stores.

    - ``allow_layer=False`` (the ``layers/activations/common.py``-style
      Layer-rejecting policy, unified here per D-002):

      - Resolution proceeds exactly as above, and THEN the resolved value is
        checked: if it is a ``keras.layers.Layer`` instance, raise
        ``ValueError`` (same message shape as
        ``layers.activations.common.resolve_activation``'s existing raise --
        a layer can own weights, and those weights would be created during
        ``call()`` rather than ``build()``, which breaks ``.keras`` weight
        loading).
      - Every other input type behaves identically to the ``allow_layer=True``
        path -- str/None/dict/callable all pass through the same resolution
        logic; only a resolved ``Layer`` diverges.

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
    # DECISION plan-2026-09-15T034909-a7edc8da/D-002
    # Do NOT change this default to False. `allow_layer=True` preserves the
    # exact pre-existing behavior of this function for all ~60 callers that
    # predate this parameter -- none of them opt in, so none of them may
    # observe a new ValueError. `layers/activations/common.py::resolve_activation`
    # is the ONE caller that wants the reject policy; it passes
    # `allow_layer=False` explicitly. See decisions.md D-002 for the full
    # reasoning (also rejects defaulting the ~37 Tier-2 migration files to
    # False, for the same "unproven behavior change" argument).
    allow_layer: bool = True,
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
    :param allow_layer: When ``True`` (the default), a resolved
        ``keras.layers.Layer`` is returned like any other value -- this is
        the original, backward-compatible contract every existing caller
        relies on. When ``False``, a resolved ``keras.layers.Layer`` raises
        ``ValueError`` instead, matching
        ``layers.activations.common.resolve_activation``'s policy. See the
        module docstring "Two ``deserialize_activation`` contracts" section.
    :type allow_layer: bool
    :return: The live activation. Non-dict inputs are returned **unchanged**.
    :rtype: Any
    :raises ValueError: If ``allow_layer`` is ``False`` and the resolved value
        is a ``keras.layers.Layer`` instance.
    """
    if isinstance(activation, dict):
        resolved = keras.saving.deserialize_keras_object(
            activation, custom_objects=custom_objects
        )
    else:
        resolved = activation
    if not allow_layer and isinstance(resolved, keras.layers.Layer):
        raise ValueError(
            "Activation must be a string name or a plain callable, not a "
            f"keras Layer instance ({type(resolved).__name__}). Layer "
            "activations may own weights, which would be created during "
            "call() rather than build() and would not survive a .keras "
            "round-trip. Use e.g. 'leaky_relu' or keras.activations.silu."
        )
    return resolved


# ---------------------------------------------------------------------
