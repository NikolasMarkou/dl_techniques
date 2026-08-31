"""Every matplotlib-using callback acquires ``pyplot`` through the one helper.

What this file pins (the CORRECTED contract)
--------------------------------------------
``dl_techniques.utils.matplotlib_backend.import_pyplot`` uses **setdefault**
semantics, so there are two claims and they need two different instruments:

1. **Behaviour** -- with ``MPLBACKEND`` set to a non-Agg backend, driving a
   callback's pyplot path leaves that backend IN FORCE; with ``MPLBACKEND``
   unset, the resulting backend is ``agg``. Measured in fresh subprocesses.
2. **Consolidation** -- every callback module that touches matplotlib imports
   ``import_pyplot`` and contains no bare ``import matplotlib.pyplot``.
   Measured statically over the module's AST.

The first shipped revision of this file asserted the OPPOSITE of claim 1: that
every callback forces ``Agg`` even when the caller explicitly asked for
something else. That was an override, justified by a headless-crash premise
that does not reproduce -- MEASURED on matplotlib 3.10.0, ``MPLBACKEND`` unset
plus ``DISPLAY`` unset already resolves to ``agg``, and so does a bogus
``DISPLAY=:99`` (with ``savefig`` working). The premise was retracted; this
guard now pins respect-the-caller instead.

Why the behavioural arms need a FRESH SUBPROCESS, with an explicit child env
---------------------------------------------------------------------------
Backend selection is process-global and effectively once-only, and this
repository runs its whole suite under ``MPLBACKEND=Agg`` (repo ``CLAUDE.md``).
An in-process assertion would therefore read the ENVIRONMENT rather than the
code and pass no matter what the callback does -- a guard that cannot fail.
:func:`_child_env` builds each child's environment explicitly: it POPS
``MPLBACKEND`` and ``DISPLAY`` from the inherited copy and then sets only what
the arm wants, so the parent's ``MPLBACKEND=Agg`` cannot make any arm vacuous.
This file is verified to pass both WITH and WITHOUT ``MPLBACKEND=Agg`` in the
parent environment.

Why the subject list is DERIVED and not a literal
-------------------------------------------------
The first revision parametrized over a hardcoded 4-module list while a FIFTH
callback (``convunext_bottleneck_monitor``) had the identical bare-import
shape; the guard could not see the population it exists to police.
:data:`MATPLOTLIB_CALLBACK_MODULES` is now an AST census of
``src/dl_techniques/callbacks/*.py``, and
:func:`test_the_census_is_not_empty_or_shrinking` is the anti-vacuity floor: a
broken walk that finds nothing fails loudly instead of parametrizing over an
empty list.

Scope: ``callbacks/`` only, which is this guard's remit. MEASURED on the same
day: ``grep -rln "import matplotlib" src/dl_techniques --include=*.py`` returns
**29** files tree-wide -- 3 callbacks (``training_curves`` reaches matplotlib
only through the helper, and ``convunext_bottleneck_monitor`` now does too),
the helper itself, and **25** others (all 10 of ``analyzer/``, all 6 of
``visualization/``, ``datasets/{arc/arc_utilities,simple_2d}.py``,
``losses/clustering_loss.py``, ``models/memory/som/model.py``,
``utils/{visualization,visualization_manager,inference,alignment/alignment,
masking/factory}.py``). None of the 25 is a Keras callback.

ONE of them is nevertheless a callback-driven consumer and is reported rather
than silently swept in: ``callbacks/analyzer_callback.py`` imports
``dl_techniques.analyzer``, whose 10 modules import matplotlib directly, so an
epoch hook DOES reach a bare matplotlib import transitively. The census
deliberately does not select ``analyzer_callback.py`` -- it acquires no pyplot
of its own -- and widening this guard to ``analyzer/`` is a separate decision
that has NOT been taken here.

``CUDA_VISIBLE_DEVICES=""`` keeps these keras imports on CPU (the backend
question is GPU-independent, and the repo forbids concurrent GPU-touching test
processes).
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_CALLBACKS_DIR = _SRC / "dl_techniques" / "callbacks"

_HELPER_MODULE = "dl_techniques.utils.matplotlib_backend"
_HELPER_NAME = "import_pyplot"

# A valid, always-importable, NON-Agg backend: needs neither a display nor an
# optional GUI toolkit. This is the ambient backend the "respect the caller"
# arms use, and what makes them discriminating.
_NON_AGG_AMBIENT_BACKEND = "pdf"

# Anti-vacuity floor for the derived census. Five callbacks touch matplotlib
# today (jepa_visualization, depth_visualization, training_curves,
# coco_multitask_visualization, convunext_bottleneck_monitor). A census that
# finds fewer than this is broken, not a shrinking population -- raise the
# floor deliberately if a callback is genuinely deleted.
_CENSUS_FLOOR = 5


# =====================================================================
# Derived census
# =====================================================================
def _module_uses_matplotlib(tree: ast.AST) -> bool:
    """True if this module imports any ``matplotlib`` module or the helper."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib" or alias.name.startswith(
                    "matplotlib."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "matplotlib" or mod.startswith("matplotlib."):
                return True
            if mod == _HELPER_MODULE:
                return True
    return False


def _census() -> List[str]:
    """AST census of ``callbacks/*.py`` modules that touch matplotlib."""
    found = []
    for path in sorted(_CALLBACKS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _module_uses_matplotlib(tree):
            found.append(f"dl_techniques.callbacks.{path.stem}")
    return found


#: The subjects of every parametrized arm below. Derived, never hand-listed.
MATPLOTLIB_CALLBACK_MODULES: List[str] = _census()


def _module_path(module: str) -> Path:
    return _CALLBACKS_DIR / (module.rsplit(".", 1)[1] + ".py")


def test_the_census_is_not_empty_or_shrinking() -> None:
    """ANTI-VACUITY FLOOR for every parametrized arm in this file.

    A broken AST walk (renamed directory, parse failure swallowed upstream,
    predicate that matches nothing) would silently parametrize over an empty
    list and turn the whole file green while measuring nothing.
    """
    assert _CALLBACKS_DIR.is_dir(), (
        f"the callbacks package is not at {_CALLBACKS_DIR}; the census walked "
        f"nothing and every parametrized arm below is vacuous"
    )
    assert len(MATPLOTLIB_CALLBACK_MODULES) >= _CENSUS_FLOOR, (
        f"the matplotlib census found only {len(MATPLOTLIB_CALLBACK_MODULES)} "
        f"callback module(s) ({MATPLOTLIB_CALLBACK_MODULES}), below the floor "
        f"of {_CENSUS_FLOOR}. Either the walk is broken or a callback was "
        f"deleted -- lower the floor deliberately, never silently."
    )


# =====================================================================
# Claim 2: consolidation (static, over the derived census)
# =====================================================================
@pytest.mark.parametrize("module", MATPLOTLIB_CALLBACK_MODULES)
def test_the_callback_acquires_pyplot_through_the_shared_helper(
    module: str,
) -> None:
    """No callback may import ``matplotlib.pyplot`` itself.

    This is the arm that actually pins the consolidation. The behavioural arms
    below cannot: after the merge every module binds the SAME function object,
    so observing a backend proves the name is importable, not that the module
    routes through it.
    """
    path = _module_path(module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    bare_pyplot: List[int] = []
    imports_helper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matplotlib.pyplot":
                    bare_pyplot.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") == "matplotlib.pyplot":
                bare_pyplot.append(node.lineno)
            if (node.module or "") == _HELPER_MODULE and any(
                a.name == _HELPER_NAME for a in node.names
            ):
                imports_helper = True

    assert not bare_pyplot, (
        f"{module} imports matplotlib.pyplot directly at line(s) "
        f"{bare_pyplot}. Inside dl_techniques/callbacks/ the backend decision "
        f"is owned by {_HELPER_MODULE}.{_HELPER_NAME}; a bare import makes the "
        f"process-global backend depend on which callback plots first."
    )
    assert imports_helper, (
        f"{module} was selected by the matplotlib census but does not import "
        f"{_HELPER_NAME} from {_HELPER_MODULE}."
    )


# =====================================================================
# Claim 1: behaviour (subprocess, explicit child env)
# =====================================================================
def _child_env(backend: Optional[str]) -> dict:
    """Environment for a fresh interpreter with an EXPLICIT ambient backend.

    ``MPLBACKEND`` and ``DISPLAY`` are popped from the inherited copy first, so
    the parent's ``MPLBACKEND=Agg`` (how this suite is invoked) can never leak
    in and make an arm vacuous. ``backend=None`` means "unset".
    """
    env = dict(os.environ)
    env.pop("MPLBACKEND", None)
    env.pop("DISPLAY", None)
    if backend is not None:
        env["MPLBACKEND"] = backend
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_child(code: str, backend: Optional[str]) -> str:
    """Run ``code`` in a fresh interpreter and return the reported backend."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=_child_env(backend),
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


_BARE_IMPORT = (
    "import matplotlib.pyplot\n"
    "import matplotlib\n"
    'print("BACKEND=" + matplotlib.get_backend())\n'
)

_HELPER_TEMPLATE = """
from dl_techniques.utils.matplotlib_backend import import_pyplot
_plt = import_pyplot()
assert _plt is not None, "import_pyplot() returned None"
import matplotlib
print("BACKEND=" + matplotlib.get_backend())
"""

_DRIVE_TEMPLATE = """
import importlib
mod = importlib.import_module({module!r})
_fn = getattr(mod, "import_pyplot", None)
if _fn is None:
    raise AssertionError("no import_pyplot bound on " + {module!r})
_result = _fn()
assert _result is not None, "import_pyplot() returned None"
import matplotlib
print("BACKEND=" + matplotlib.get_backend())
"""


# --- anti-vacuity controls for the subprocess shape ------------------
def test_the_harness_controls_the_child_ambient_backend() -> None:
    """ANTI-VACUITY CONTROL for every subprocess arm.

    A bare ``import matplotlib.pyplot`` -- no helper at all -- must report
    ``pdf`` when the harness sets ``MPLBACKEND=pdf`` and ``agg`` when the
    harness unsets it. If these two readings ever agree, the child env is not
    under the harness's control (e.g. the parent's ``MPLBACKEND=Agg`` leaked)
    and every arm below certifies nothing.
    """
    explicit = _run_child(_BARE_IMPORT, _NON_AGG_AMBIENT_BACKEND)
    unset = _run_child(_BARE_IMPORT, None)
    assert explicit == _NON_AGG_AMBIENT_BACKEND, (
        f"a bare pyplot import under MPLBACKEND={_NON_AGG_AMBIENT_BACKEND!r} "
        f"reported {explicit!r}; the harness is not controlling the child "
        f"environment, so the 'explicit request survives' arms are vacuous"
    )
    assert unset == "agg", (
        f"a bare pyplot import with MPLBACKEND unset reported {unset!r}, not "
        f"'agg'. matplotlib's headless fallback is what this file's premise "
        f"correction rests on -- re-measure before trusting the default arms"
    )
    assert explicit != unset, (
        "the two ambient regimes are indistinguishable in this subprocess "
        "shape; the whole file is vacuous"
    )


# --- the helper itself ------------------------------------------------
def test_the_helper_respects_an_explicitly_requested_backend() -> None:
    """``MPLBACKEND=pdf`` -> ``import_pyplot`` leaves ``pdf`` in force.

    This is the arm the previous revision asserted the OPPOSITE of. Forcing
    ``Agg`` here would discard a caller's deliberate choice (e.g.
    ``MPLBACKEND=svg`` for vector output) for a crash that does not reproduce.
    """
    backend = _run_child(_HELPER_TEMPLATE, _NON_AGG_AMBIENT_BACKEND)
    assert backend == _NON_AGG_AMBIENT_BACKEND, (
        f"import_pyplot() overrode an explicit MPLBACKEND="
        f"{_NON_AGG_AMBIENT_BACKEND!r} and left the backend at {backend!r}. "
        f"The helper uses setdefault semantics: a non-empty MPLBACKEND is the "
        f"caller's explicit request and must survive."
    )


def test_the_helper_defaults_to_agg_when_nothing_was_requested() -> None:
    """``MPLBACKEND`` unset -> ``import_pyplot`` selects ``Agg``."""
    backend = _run_child(_HELPER_TEMPLATE, None)
    assert backend == "agg", (
        f"import_pyplot() with MPLBACKEND unset left the backend at "
        f"{backend!r}, not 'agg'; DEFAULT_BACKEND is the documented default"
    )


def test_the_helper_exports_the_default_to_child_processes() -> None:
    """The unset branch also sets ``MPLBACKEND``, so subprocesses inherit it."""
    backend = _run_child(
        "import os\n"
        "from dl_techniques.utils.matplotlib_backend import import_pyplot\n"
        "import_pyplot()\n"
        'print("BACKEND=" + os.environ.get("MPLBACKEND", "<unset>"))\n',
        None,
    )
    assert backend == "agg", (
        f"MPLBACKEND was {backend!r} after import_pyplot() on the unset "
        f"branch; the default must be exported so child processes inherit it"
    )


# --- each callback, over the derived census --------------------------
@pytest.mark.parametrize("module", MATPLOTLIB_CALLBACK_MODULES)
def test_the_callback_does_not_override_an_explicit_backend(
    module: str,
) -> None:
    """Driving the module's pyplot path must not discard ``MPLBACKEND=pdf``.

    This is the behavioural half of the consolidation claim: whatever path the
    module takes to ``pyplot``, it must end at the shared setdefault semantics.
    A re-inlined ``matplotlib.use("Agg")`` anywhere in that path fails here.
    """
    backend = _run_child(
        _DRIVE_TEMPLATE.format(module=module), _NON_AGG_AMBIENT_BACKEND
    )
    assert backend == _NON_AGG_AMBIENT_BACKEND, (
        f"{module} left the matplotlib backend at {backend!r} despite an "
        f"explicit MPLBACKEND={_NON_AGG_AMBIENT_BACKEND!r}. The module must "
        f"acquire pyplot through {_HELPER_MODULE}.{_HELPER_NAME} and must not "
        f'call matplotlib.use("Agg") on its own.'
    )
