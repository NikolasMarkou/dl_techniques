"""One way for library code to acquire ``matplotlib.pyplot``.

This module owns a single process-global decision that five visualization
callbacks were each making independently -- and disagreeing about: which
``matplotlib`` backend is in force when ``pyplot`` is first imported.

What this module is for
-----------------------
Consolidation with a documented, predictable default. Before it, three of the
five callbacks (``jepa_visualization``, ``depth_visualization``,
``convunext_bottleneck_monitor``) imported ``pyplot`` bare and two
(``training_curves``, ``coco_multitask_visualization``) called
``matplotlib.use("Agg")`` first, so the resulting backend depended on which
callback happened to plot first in a process. That divergence is the defect
this module removes.

What this module is NOT for (a corrected claim)
-----------------------------------------------
An earlier revision of this docstring said the bare importers "crashed on a
headless host". **That is false on the shipped matplotlib and has been
retracted.** MEASURED on matplotlib 3.10.0, this host:

* ``MPLBACKEND`` unset and ``DISPLAY`` unset -> ``import matplotlib.pyplot``
  resolves to ``agg`` by itself;
* ``MPLBACKEND`` unset and a bogus ``DISPLAY=:99`` -> still ``agg``, and
  ``savefig`` works.

matplotlib's own headless fallback already covers the X11 case the repository
root ``CLAUDE.md`` warns about under "Running Training Scripts". So the value
here is uniformity and one documented default, not a crash fix.

The default, and how an explicit request is respected
-----------------------------------------------------
:func:`import_pyplot` uses **setdefault semantics**, matching the house pattern
already used by ``src/train/video_jepa/train_video_jepa.py:47``
(``os.environ.setdefault("MPLBACKEND", "Agg")``):

* ``MPLBACKEND`` set to anything non-empty -> that request is RESPECTED and this
  module touches nothing. A caller who wants ``svg`` for vector output, or
  ``pdf``, gets it.
* ``MPLBACKEND`` unset -> ``Agg`` is selected, both in-process
  (``matplotlib.use``) and in the environment (so child processes inherit it).

The environment variable is the detection signal, deliberately. A prior
in-process ``matplotlib.use(...)`` is NOT reliably distinguishable from
matplotlib's own default resolution -- ``matplotlib.get_backend()`` returns a
concrete backend name either way -- so keying off it would guess. ``MPLBACKEND``
is the one unambiguous "the caller asked for this" signal, and it is the signal
the repository's own convention already uses.

Scope, deliberately narrow. This module is a pure importer. It defines no Keras
object, registers nothing, holds no state of its own, and imports neither
``dl_techniques.layers`` nor ``dl_techniques.models`` -- at module level or
inside a function -- so ``dl_techniques.utils`` stays free of an import cycle.
That is asserted mechanically over the whole ``utils/`` tree by
``tests/test_utils/test_dtype_policy.py``.

What this module does NOT own: any plotting, any figure lifecycle, any output
path. Figure creation, saving and ``plt.close`` remain each callback's own
business, and ``dl_techniques.utils.visualization_manager`` owns the
where-do-plots-go policy. This module owns only the backend default and the
import.
"""

import os
from typing import Any, Tuple, Union

#: The backend selected when the caller expressed no preference.
DEFAULT_BACKEND = "Agg"


def import_pyplot(with_cm: bool = False) -> Union[Any, Tuple[Any, Any]]:
    """Import ``matplotlib.pyplot``, defaulting to ``Agg`` if nothing was asked for.

    Setdefault semantics, never an override:

    * if the ``MPLBACKEND`` environment variable is set and non-empty, that
      backend is left in force and this function only performs the import;
    * otherwise ``Agg`` is selected before ``pyplot`` is imported, and
      ``MPLBACKEND`` is set so subprocesses inherit the same default.

    ``matplotlib.use`` is called in addition to setting the environment variable
    because the variable alone is read only while matplotlib resolves its
    backend; if ``pyplot`` was already imported earlier in the process, ``use``
    is what makes the default effective there too.

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

    if not os.environ.get("MPLBACKEND"):
        os.environ["MPLBACKEND"] = DEFAULT_BACKEND
        matplotlib.use(DEFAULT_BACKEND)

    import matplotlib.pyplot as plt

    if with_cm:
        from matplotlib import cm

        return plt, cm
    return plt
