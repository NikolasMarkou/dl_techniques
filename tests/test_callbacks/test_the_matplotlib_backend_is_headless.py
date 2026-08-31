"""The four visualization callbacks must force the headless ``Agg`` backend.

Why this file runs everything in a FRESH SUBPROCESS
---------------------------------------------------
Backend selection is a process-global, effectively once-only decision, and this
repository runs its whole test suite under ``MPLBACKEND=Agg`` (repo ``CLAUDE.md``,
"Running Training Scripts"). An in-process ``assert matplotlib.get_backend() ==
"agg"`` therefore reads the ENVIRONMENT rather than the code and passes
identically whether or not the callback sets the backend -- a guard that cannot
fail. Every arm below spawns a fresh interpreter instead.

Why the child env sets ``MPLBACKEND=pdf`` rather than deleting it
----------------------------------------------------------------
MEASURED on this host: with ``MPLBACKEND`` deleted from the environment AND
``DISPLAY`` unset, ``import matplotlib.pyplot`` resolves to ``agg`` all by
itself (matplotlib's own headless fallback). So the "delete MPLBACKEND"
variant is ALSO vacuous here -- it would pass against a callback that never
calls ``matplotlib.use`` at all. What discriminates is a child whose ambient
backend is a valid, always-importable, NON-Agg one: ``pdf`` (chosen because it
needs no display and no optional GUI toolkit, unlike ``TkAgg``). Under
``MPLBACKEND=pdf`` a bare ``import matplotlib.pyplot`` reports ``pdf``, so an
arm reporting ``agg`` can only have gotten there by the module calling
``matplotlib.use("Agg")`` itself.

:func:`test_a_bare_pyplot_import_does_not_report_agg` is the anti-vacuity
control that pins exactly that: it asserts the harness CAN observe a non-Agg
backend in this same subprocess shape. If it ever starts reporting ``agg``,
the other four arms are measuring the environment and certify nothing.

``DISPLAY`` is additionally unset in every child so no arm can be rescued by a
live X server, and ``CUDA_VISIBLE_DEVICES=""`` keeps these keras imports on CPU
(the backend question is GPU-independent, and the repo forbids concurrent
GPU-touching test processes).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"

# A valid, always-importable, NON-Agg backend: needs neither a display nor an
# optional GUI toolkit. This is what makes every arm discriminating.
_NON_AGG_AMBIENT_BACKEND = "pdf"


def _child_env() -> dict:
    """Environment for a fresh interpreter with a NON-Agg ambient backend."""
    env = dict(os.environ)
    env.pop("MPLBACKEND", None)
    env.pop("DISPLAY", None)
    env["MPLBACKEND"] = _NON_AGG_AMBIENT_BACKEND
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_child(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return the reported backend."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env(),
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    reported = [
        line.split("=", 1)[1].strip()
        for line in proc.stdout.splitlines()
        if line.startswith("BACKEND=")
    ]
    assert len(reported) == 1, (
        f"expected exactly one BACKEND= line, got {reported}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return reported[0].lower()


_DRIVE_TEMPLATE = """
import importlib
mod = importlib.import_module({module!r})
# Drive whichever name this module uses to acquire pyplot. Before the
# consolidation these were the module-local ``_import_matplotlib`` /
# ``_import_mpl``; after it, the shared ``import_pyplot``. The arm asserts the
# BACKEND that path leaves behind, not which name it is spelled with.
for _name in ("_import_matplotlib", "_import_mpl", "import_pyplot"):
    _fn = getattr(mod, _name, None)
    if _fn is not None:
        break
else:
    raise AssertionError(
        "no pyplot-acquiring entry point found on " + {module!r}
    )
_result = _fn()
assert _result is not None, _name + "() returned None"
import matplotlib
print("BACKEND=" + matplotlib.get_backend())
"""


@pytest.mark.parametrize(
    "module",
    [
        "dl_techniques.callbacks.jepa_visualization",
        "dl_techniques.callbacks.depth_visualization",
        "dl_techniques.callbacks.training_curves",
        "dl_techniques.callbacks.coco_multitask_visualization",
    ],
)
def test_the_callback_forces_agg(module: str) -> None:
    """Driving the module's pyplot-acquiring path must leave the backend Agg."""
    backend = _run_child(_DRIVE_TEMPLATE.format(module=module))
    assert backend == "agg", (
        f"{module} left the matplotlib backend at {backend!r}; a headless run "
        f"without MPLBACKEND=Agg is exposed to the X11-crash class. The module "
        f"must acquire pyplot through "
        f"dl_techniques.utils.matplotlib_backend.import_pyplot, which calls "
        f'matplotlib.use("Agg") BEFORE importing pyplot.'
    )


def test_a_bare_pyplot_import_does_not_report_agg() -> None:
    """ANTI-VACUITY CONTROL for the four arms above.

    In the exact same subprocess shape, a plain ``import matplotlib.pyplot``
    with no helper at all must report the ambient non-Agg backend. If this ever
    reports ``agg``, the harness is measuring the environment rather than the
    code and the four arms above certify nothing.
    """
    backend = _run_child(
        "import matplotlib.pyplot\n"
        "import matplotlib\n"
        'print("BACKEND=" + matplotlib.get_backend())\n'
    )
    assert backend != "agg", (
        "the anti-vacuity control reported 'agg' for a bare pyplot import, so "
        "this subprocess shape cannot distinguish a module that forces Agg "
        "from one that does not -- the whole file is vacuous. Re-derive the "
        "probe (pick a different ambient backend) before trusting any arm."
    )
    assert backend == _NON_AGG_AMBIENT_BACKEND, (
        f"ambient backend was {backend!r}, expected "
        f"{_NON_AGG_AMBIENT_BACKEND!r}"
    )
