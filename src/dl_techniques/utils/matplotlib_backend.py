"""One headless-safe way to acquire ``matplotlib.pyplot``.

This module owns a single process-global decision that four visualization
callbacks were each making independently -- and disagreeing about: whether to
FORCE the non-interactive ``Agg`` backend before ``matplotlib.pyplot`` is
imported.

Why it is one decision and not four
-----------------------------------
``matplotlib``'s backend is process-global and is resolved the first time
``pyplot`` is imported. On a genuinely headless host (no ``$DISPLAY``) a module
that leaves that resolution to the environment is exposed to the X11-crash class
the repository's root ``CLAUDE.md`` warns about under "Running Training
Scripts", and its only protection is that every caller remembers the
``MPLBACKEND=Agg`` prefix. Two of the four callbacks called
``matplotlib.use("Agg")``; two did not, and one of those two
(``callbacks/depth_visualization.py``) is driven by a trainer,
``src/train/depth_anything/``, that does not set ``MPLBACKEND`` for itself
either. Consolidating the decision here is what makes it uniform, and
``tests/test_callbacks/test_the_matplotlib_backend_is_headless.py`` asserts the
resulting backend in a fresh subprocess -- an in-process assertion would only
read the suite's own ``MPLBACKEND=Agg`` and could never fail.

**This is a deliberate behaviour change, not a refactor.** Callers that
previously inherited whatever backend the environment resolved now get ``Agg``
unconditionally. That is the fix. A caller that genuinely wants an interactive
backend must import ``pyplot`` itself rather than call this function.

Scope, deliberately narrow. This module is a pure importer. It defines no Keras
object, registers nothing, holds no state of its own, and imports neither
``dl_techniques.layers`` nor ``dl_techniques.models`` -- at module level or
inside a function -- so ``dl_techniques.utils`` stays free of an import cycle.
That is asserted mechanically over the whole ``utils/`` tree by
``tests/test_utils/test_dtype_policy.py``.

What this module does NOT own: any plotting, any figure lifecycle, any output
path. Figure creation, saving and ``plt.close`` remain each callback's own
business, and ``dl_techniques.utils.visualization_manager`` owns the
where-do-plots-go policy. This module owns only the backend and the import.
"""

from typing import Any, Tuple, Union


def import_pyplot(with_cm: bool = False) -> Union[Any, Tuple[Any, Any]]:
    """Import ``matplotlib.pyplot`` with the headless ``Agg`` backend forced.

    ``matplotlib.use("Agg")`` is called BEFORE ``matplotlib.pyplot`` is imported,
    which is the ordering that makes the choice effective on a first import; on a
    later call, when ``pyplot`` is already loaded, ``use`` switches the backend
    of the running process instead, so repeated calls are safe and converge on
    ``Agg`` either way.

    The import is deliberately function-local: ``matplotlib`` is an optional
    cost at library-import time, so an absent installation surfaces as an
    ``ImportError`` at plot time inside a callback rather than at
    ``dl_techniques`` import time.

    :param with_cm: also return ``matplotlib.cm``, for callers that need a
        colormap module as well as the pyplot namespace.
    :return: the ``matplotlib.pyplot`` module, or the pair ``(pyplot, cm)`` when
        ``with_cm`` is true.
    :raises ImportError: if ``matplotlib`` is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if with_cm:
        from matplotlib import cm

        return plt, cm
    return plt
