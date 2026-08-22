"""R-038 / RD-3..RD-9 root cause **RD-7**: ``plt.show()`` inside LIBRARY code.

Plan ``plan-2026-08-22T035419-a11304c8``, ruling **D-051**.

`src/dl_techniques/models/som/model.py` called ``plt.show()`` unconditionally at
the end of all five ``visualize_*`` methods. A library must not decide to block
on a GUI, and on the headless hosts this repo runs on (``MPLBACKEND=Agg`` is
mandated repo-wide) the call cannot render at all.

**Measured before the repair**, four ``visualize_*`` calls in one process:
`UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown` x10,
and **four figures left open** -- ``len(plt.get_fignums()) == 4``. The figure
leak is the part a warning filter would have hidden: any loop over epochs or
over datasets grows the pyplot figure registry without bound.

Two arms:
1. behavioural, on the real ``SOMModel`` -- no warning, no leaked figure, the
   figure is returned, and ``show=True`` still reaches ``plt.show()``;
2. structural, an AST scan asserting that **every** ``plt.show()`` call under
   ``src/`` is inside an ``if``, so the population cannot regrow elsewhere.
"""

import ast
import pathlib
from typing import List
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import keras
import pytest

from dl_techniques.models.som.model import SOMModel

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture
def trained_som() -> SOMModel:
    keras.utils.set_random_seed(0)
    model = SOMModel(map_size=(4, 4), input_dim=16)
    model.build((None, 16))
    return model


@pytest.fixture
def data() -> tuple:
    rng = np.random.default_rng(0)
    return (
        rng.random((20, 16)).astype("float32"),
        rng.integers(0, 3, 20),
    )


def test_the_visualizers_emit_no_warning_and_leak_no_figure(trained_som, data):
    """The exact pair of consequences measured before the repair."""
    x, y = data
    plt.close("all")
    assert plt.get_fignums() == [], "a previous test leaked figures"

    with _no_user_warning() as caught:
        trained_som.visualize_grid(figsize=(3, 3))
        trained_som.visualize_class_distribution(x, y, figsize=(3, 3))
        trained_som.visualize_u_matrix(figsize=(3, 3))
        trained_som.visualize_hit_histogram(x, figsize=(3, 3))
        trained_som.visualize_memory_recall(x[0], n_similar=2, figsize=(6, 2))

    assert caught == [], (
        "a visualize_* method emitted a UserWarning; before D-051 these were "
        f"'FigureCanvasAgg is non-interactive': {caught}"
    )
    assert plt.get_fignums() == [], (
        "five visualize_* calls leaked "
        f"{len(plt.get_fignums())} open figure(s); the pre-D-051 code leaked one "
        "per call"
    )


def _no_user_warning():
    import warnings

    class _Ctx:
        def __enter__(self):
            self._cm = warnings.catch_warnings(record=True)
            self._log = self._cm.__enter__()
            warnings.simplefilter("always")
            return self._out

        _out: List[str] = []

        def __exit__(self, *a):
            self._out.extend(
                str(w.message) for w in self._log
                if issubclass(w.category, UserWarning)
            )
            self._cm.__exit__(*a)
            return False

    ctx = _Ctx()
    ctx._out = []
    return ctx


def test_every_visualizer_returns_its_figure(trained_som, data):
    x, y = data
    plt.close("all")
    figures = [
        trained_som.visualize_grid(figsize=(3, 3)),
        trained_som.visualize_class_distribution(x, y, figsize=(3, 3)),
        trained_som.visualize_u_matrix(figsize=(3, 3)),
        trained_som.visualize_hit_histogram(x, figsize=(3, 3))[1],
        trained_som.visualize_memory_recall(x[0], n_similar=2, figsize=(6, 2)),
    ]
    for fig in figures:
        assert isinstance(fig, plt.Figure), f"expected a Figure, got {type(fig)}"
    hits, _fig = trained_som.visualize_hit_histogram(x, figsize=(3, 3))
    assert hits.shape == (4, 4), "the hit histogram itself must still come back"
    plt.close("all")


def test_show_true_still_reaches_plt_show(trained_som):
    """The knob is not decorative: ``show=True`` must actually call through."""
    plt.close("all")
    with mock.patch.object(plt, "show") as spy:
        trained_som.visualize_grid(figsize=(3, 3), show=True)
    assert spy.call_count == 1, (
        f"show=True called plt.show() {spy.call_count} times, expected 1"
    )
    plt.close("all")


def _unconditional_show_calls(root: pathlib.Path) -> List[str]:
    """Every ``plt.show()`` under ``root`` that is NOT inside an ``if``."""
    offenders: List[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                for stmt in node.body + node.orelse:
                    for inner in ast.walk(stmt):
                        if isinstance(inner, ast.Call):
                            guarded.add(id(inner))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "show"):
                continue
            base = fn.value
            if not (isinstance(base, ast.Name) and base.id in ("plt", "pyplot")):
                continue
            if id(node) in guarded:
                continue
            try:
                shown = path.relative_to(_REPO_ROOT)
            except ValueError:  # the positive control writes to a tmp_path
                shown = path
            offenders.append(f"{shown}:{node.lineno}")
    return offenders


def test_the_ast_scanner_detects_an_unconditional_show(tmp_path):
    """Positive control: without it, a scanner that finds nothing would pass."""
    (tmp_path / "probe.py").write_text(
        "import matplotlib.pyplot as plt\n"
        "def bad():\n"
        "    plt.show()\n"
        "def ok(show=False):\n"
        "    if show:\n"
        "        plt.show()\n",
        encoding="utf-8",
    )
    found = _unconditional_show_calls(tmp_path)
    assert len(found) == 1 and found[0].endswith("probe.py:3"), (
        f"the scanner failed its own positive control: {found}"
    )


def test_no_library_module_calls_plt_show_unconditionally():
    offenders = _unconditional_show_calls(_REPO_ROOT / "src")
    assert offenders == [], (
        "library code calls plt.show() outside any `if`. A library must not "
        "decide to block on a GUI: take a `show: bool = False` argument, close "
        "the figure otherwise, and return it. See decisions.md D-051.\n  "
        + "\n  ".join(offenders)
    )
