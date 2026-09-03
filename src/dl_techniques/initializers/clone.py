"""Per-site initializer cloning.

Provides :func:`clone_initializer`, which returns an independent copy of an
initializer so that two weights do not start bit-identical.

A single ``keras.initializers.Initializer`` instance reused across several
weights produces the same tensor at every site whose shape matches. That is
Keras 3 behaviour: a seedless initializer instance self-assigns a fixed seed at
construction and every later draw replays it. Measured after
``keras.utils.set_random_seed(1234)``, ``keras.initializers.get("glorot_uniform")``
reports ``.seed == 835549144`` reproducibly (the value is process-specific).

Measured on two ``Dense(4)`` layers built from ``(None, 6)``:

===================================  =========================
how the initializer is passed        kernels identical?
===================================  =========================
the string ``"glorot_uniform"``      no (a fresh instance per layer)
one shared seedless instance         yes, bit-for-bit
===================================  =========================

The common repo idiom ``self.kernel_initializer = keras.initializers.get(arg)``
in ``__init__``, then handing ``self.kernel_initializer`` to several sub-layers,
takes the second row.

Whether that is a defect depends on the site, not on the shape. Symmetry
between two weights that play the same role is usually harmless. Symmetry
between two weights whose difference is the architecture, such as a main branch
and a basis branch, or a query and a key projection, is a training pathology.
Probe the site before cloning it. See ``plan-2026-08-19T163559-499b6f0e/D-057``.
"""

import keras
import copy
from typing import Any, Optional, Union

# ---------------------------------------------------------------------

__all__ = ["clone_initializer"]

# ---------------------------------------------------------------------

def clone_initializer(
        initializer: Optional[Union[str, keras.initializers.Initializer]],
) -> Any:
    """Return an independent initializer equivalent to ``initializer``.

    **Dispatch:**

    .. code-block:: text

        initializer
             │
             ├── None or str ──────────────► keras.initializers.get(arg)
             │                               (no per-instance state to clone)
             │
             └── anything else
                     │
                     ▼
              keras.initializers.get
                     │
                     ├── not an Initializer ──► returned unchanged
                     │
                     └── Initializer
                             │
                             ▼
                   from_config(get_config())
                             │
                             ├── ok ────────► fresh instance
                             │                (seedless: new seed;
                             │                 seeded: same seed)
                             └── raises ────► copy.deepcopy(instance)

    Cloning a seeded initializer does not break symmetry: two clones of
    ``GlorotUniform(seed=7)`` still produce identical tensors. That is the
    caller's stated intent and this helper does not override it.

    :param initializer: An ``Initializer`` instance, an initializer name, a
        serialized config dict, or ``None``.
    :type initializer: str or keras.initializers.Initializer or dict or None
    :return: A new initializer that draws independently of the argument, or the
        argument itself when it carries no per-instance state.
    :rtype: keras.initializers.Initializer or None
    :raises ValueError: Only from ``keras.initializers.get``, when the argument
        is malformed. A well-formed initializer never raises here.

    Example::

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.main_dense = keras.layers.Dense(
            units, kernel_initializer=self.kernel_initializer)
        self.basis_dense = keras.layers.Dense(
            units, kernel_initializer=clone_initializer(self.kernel_initializer))
    """
    if initializer is None or isinstance(initializer, str):
        return keras.initializers.get(initializer)

    resolved = keras.initializers.get(initializer)
    if not isinstance(resolved, keras.initializers.Initializer):
        return resolved

    try:
        return resolved.__class__.from_config(resolved.get_config())
    except Exception:  # noqa: BLE001 -- a custom initializer may not round trip
        return copy.deepcopy(resolved)

# ---------------------------------------------------------------------
