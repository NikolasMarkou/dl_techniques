"""Per-site initializer cloning -- breaking shared-instance init symmetry.

A single ``keras.initializers.Initializer`` INSTANCE reused across several
weights produces the SAME tensor at every site whose shape matches. This is
Keras 3 behaviour, not a bug: a seedless initializer instance self-assigns a
fixed seed at construction (measured: ``keras.initializers.get("glorot_uniform")``
reports ``.seed == 835549144`` reproducibly after
``keras.utils.set_random_seed(1234)``; the value itself is process-specific), and
every later draw from that instance replays it.

Measured, on two ``Dense(4)`` layers built from ``(None, 6)``:

===================================  =========================
how the initializer is passed        kernels identical?
===================================  =========================
the STRING ``"glorot_uniform"``      **no** (a fresh instance per layer)
one shared seedless INSTANCE         **yes**, bit-for-bit
===================================  =========================

The common repo idiom ``self.kernel_initializer = keras.initializers.get(arg)``
in ``__init__``, then handing ``self.kernel_initializer`` to several sub-layers,
therefore takes the second row.

**Whether that is a defect is a per-site judgement, not a property of the shape**
-- see ``plan-2026-08-19T163559-499b6f0e/D-057``. Symmetry between two weights
that play the SAME role is usually harmless and is sometimes wanted; symmetry
between two weights whose DIFFERENCE is the architecture (a main branch and a
basis branch; a query and a key projection) is a training pathology. Probe the
site before cloning it.
"""

import copy
from typing import Any, Optional, Union

import keras

__all__ = ["clone_initializer"]


def clone_initializer(
        initializer: Optional[Union[str, keras.initializers.Initializer]],
) -> Any:
    """Return an INDEPENDENT initializer equivalent to ``initializer``.

    **Contract.**

    :param initializer: An ``Initializer`` instance, an initializer name, a
        serialized config dict, or ``None``.
    :returns: For an ``Initializer`` instance, a new instance rebuilt from its
        own ``get_config()`` -- so a seedless instance draws a FRESH seed and a
        SEEDED one keeps its explicit seed (reproducibility is preserved by
        construction: an author who asked for a seed still gets it). For ``None``
        or a string, the argument is returned by ``keras.initializers.get``
        unchanged, since neither carries per-instance state to clone. Any object
        whose ``get_config``/``from_config`` round trip raises is returned via a
        ``copy.deepcopy`` fallback.
    :rtype: keras.initializers.Initializer or None
    :raises: nothing. This helper never raises on a well-formed initializer; a
        malformed one is passed to ``keras.initializers.get``, which raises.

    **Failure mode to know about.** Cloning a *seeded* initializer does NOT break
    symmetry -- two clones of ``GlorotUniform(seed=7)`` still produce identical
    tensors. That is the caller's stated intent and this helper does not
    override it.

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
