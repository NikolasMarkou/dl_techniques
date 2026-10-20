"""A default layer name that never starts with an underscore, for test-private layers.

Keras names a layer it was not given a name for after its class:
``_Parent`` becomes ``_parent``. TensorFlow refuses a name scope that STARTS
with ``_`` at the root of a graph, and it does so after Keras has already
pushed the scope, so the push raises and the PARENT's scope stays on Keras'
process-global name-scope stack. Every layer built afterwards in the process
gets that prefix in its weight paths (measured: a parent called eagerly with an
underscore-named child leaves ``['outer']`` on the stack with no error at all,
and a later ``Dense`` kernel's path is ``outer/dense/kernel``; D-219 and D-220
of plan-2026-10-14T062616-81938e81). In a test suite that is a leak from one
test into every later one.

``tests/test_no_underscore_named_layer_takes_its_auto_name.py`` holds every
Layer or Model subclass whose class name starts with ``_`` to one of three
forms; inheriting from :class:`PublicAutoName` is the shared one.
"""

import re

import keras


def _snake_case(class_name: str) -> str:
    """Keras' own class-name-to-layer-name rule, without the leading underscores."""
    name = class_name.lstrip("_")
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class PublicAutoName:
    """Mixin: an unnamed instance is named after its class without the leading ``_``.

    Interface contract:

    * Use: list it FIRST among the bases, ``class _Probe(PublicAutoName,
      keras.layers.Layer)``. It works for layers and models, with or without an
      ``__init__`` of their own, and for Functional-style ``keras.Model``
      subclasses.
    * Effect: when ``name`` is missing or ``None``, the instance gets the class
      name in snake case without its leading underscores, uniquified per process
      as Keras does it (``parent``, ``parent_1``, ...). An explicit name is kept
      unchanged.
    * Failure mode: none of its own; every other argument reaches the next class
      in the MRO untouched.
    """

    def __init__(self, *args, **kwargs):
        if kwargs.get("name") is None:
            base = _snake_case(type(self).__name__)
            uid = keras.backend.get_uid(base)
            kwargs["name"] = base if uid == 1 else f"{base}_{uid - 1}"
        super().__init__(*args, **kwargs)
