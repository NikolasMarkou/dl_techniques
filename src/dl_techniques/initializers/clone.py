"""Per-site initializer cloning.

Provides :func:`clone_initializer`, which returns an independent copy of an
initializer so that two weights do not start out as the same random numbers.

A single ``keras.initializers.Initializer`` instance reused across several
weights **replays the same underlying sample at every later site**. That is
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

What is NOT the criterion: matching shapes
------------------------------------------

An earlier revision of this docstring made shape agreement the criterion --
"the same tensor at every site whose shape agrees". **Measurement refutes it**,
and the false rule was used downstream to CLEAR real aliasing sites -- both
``mps_layer.py`` and ``kanvolution.py`` in ``layers/structured_linear/`` -- on
the grounds that their ranks differ. Two weights can be the same random numbers
**without matching shapes**: on this backend the shorter draw comes out as the
longer draw's PREFIX, because one replayed sample is consumed from the front.

* One shared seedless ``glorot_uniform`` drawing ``(8, 4, 3, 3, 7)`` and then
  ``(8, 4, 3, 3)`` gives Pearson ``r = 0.9999999999999964`` between the second
  draw and the first's flattened prefix. The two differ only by the fan-based
  scale ``sqrt(5) = 2.236068`` (ratio std 3.6e-06) -- same sample, different
  scaling.
* With a non-scaling ``RandomUniform(-1, 1)`` the scale factor disappears and
  the smaller draw is **bit-identical** to the larger one's prefix,
  ``max|diff| == 0.0``, verified across 7 shape pairs including ``(13,) ->
  (4, 2)`` and ``(3, 3, 3) -> (9, 2)``.

Bit-identity of the *whole* tensor needs matching shapes; sharing the *sample*
does not. Only the second one matters for symmetry.

Scope of the claim, exactly
---------------------------

Independence holds for a RANDOM SEEDLESS initializer. Three exemptions, all
correct behaviour, none a defect:

1. a caller-supplied **SEEDED** instance (e.g. ``GlorotUniform(seed=7)``):
   :func:`clone_initializer` reproduces an explicit seed deliberately and by
   contract, so every cloned site draws exactly what the shared attribute would
   have drawn -- **across differing shapes too** (measured: ``seed=7`` at
   ``(8, 8, 8)`` and then ``(8, 8)`` gives ``r = 0.9999999999999957`` against
   the prefix, constant ratio 2.828427);
2. a **DETERMINISTIC** initializer (``'zeros'``, ``'ones'``, ``Constant``, and
   ``Identity`` where the weight is 2-D -- it raises on rank 3+): it holds no
   random state, so every site is bit-identical and that is what it is meant to
   do; cloning it is a no-op;
3. a **CUSTOM** initializer whose ``get_config()``/``from_config()`` round trip
   raises: :func:`clone_initializer` falls back to ``copy.deepcopy``, which
   copies the already-resolved seed rather than drawing a new one (measured: a
   deepcopy of a seedless ``glorot_uniform`` keeps ``.seed == 835549144`` and
   draws bit-identically, while a real clone moves it to ``473233032``), so
   such a site can silently stay tied, with no diagnostic.

Callers wanting reproducibility WITHOUT the tie should use
``keras.utils.set_random_seed()`` and leave the initializer seedless.

When to clone
-------------

Whether the sharing is a defect depends on the site, not on the shape. Symmetry
between two weights that play the same role is usually harmless. Symmetry
between two weights whose difference is the architecture, such as a main branch
and a basis branch, or a query and a key projection, is a training pathology.
Probe the site before cloning it -- and probe it by drawing from the shared
instance at each weight's own shape and comparing, not by assuming that unequal
shapes rule the site out. See ``plan-2026-08-19T163559-499b6f0e/D-057`` and, for
the refutation above, ``plan-2026-09-07T161712-985e4d31/D-009`` and ``D-010``.
Every statement in this docstring is pinned by
``tests/test_initializers/test_clone_initializer.py``.
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

    "Independently" means for a RANDOM SEEDLESS initializer, which is the
    common case. The module docstring states the three exemptions in full; in
    short, a **SEEDED** initializer replays by contract (across differing shapes
    too), a **DETERMINISTIC** one (``'zeros'``/``'ones'``/``Constant``, and
    ``Identity`` at 2-D) is identical at every site and correctly so, and a
    **CUSTOM** one that fails the ``get_config()``/``from_config()`` round trip
    takes the ``copy.deepcopy`` branch, which copies the already-resolved seed
    rather than drawing a fresh one and therefore stays tied.

    Cloning a seeded initializer does not break symmetry: two clones of
    ``GlorotUniform(seed=7)`` still produce identical tensors, and they share
    the same underlying sample even when the two weights have DIFFERENT shapes.
    That is the caller's stated intent and this helper does not override it.

    :param initializer: An ``Initializer`` instance, an initializer name, a
        serialized config dict, or ``None``.
    :type initializer: str or keras.initializers.Initializer or dict or None
    :return: A new initializer that draws independently of the argument -- for a
        random seedless argument, and subject to the three exemptions above --
        or the argument itself when it carries no per-instance state. Note that
        "independently" is not conditioned on shape: an un-cloned site replays
        the shared sample whatever shape it draws at.
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
