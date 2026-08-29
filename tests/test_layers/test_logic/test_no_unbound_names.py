"""
Package-wide AST undefined-name guard for ``dl_techniques.layers.logic``.

Regression guard for the defect fixed by
``[plan-2026-08-29-22eb6866/iter-1/step-1]``: commit ``527919265`` deleted
``from keras import ops`` from ``arithmetic_operators.py`` but only rewrote the
outermost call of each expression to ``keras.ops.X``, leaving 20 bare ``ops.``
references on 16 lines. Every one raised ``NameError`` at forward-pass time
while the module still imported cleanly, so nothing but the (then red) test
suite could see it.

The guard walks every ``.py`` file in the package, collects every name bound
anywhere in the module by any binding form, and asserts that every name loaded
by the module resolves to a binding or to a builtin. It is deliberately
scope-insensitive: it answers "is this name bound *anywhere* in this module",
which is exactly the question the defect got wrong, and it cannot produce the
false positives a scope-sensitive walk would on closures or late definitions.

Known limitation, stated so it is not mistaken for coverage: a wildcard import
would make every module-level name unresolvable to static analysis. The guard
therefore fails loudly if it finds one rather than silently degrading.
"""

import ast
import builtins
import pathlib
from typing import List, Set, Tuple

import pytest

LOGIC_PKG = (
    pathlib.Path(__file__).resolve().parents[3]
    / "src"
    / "dl_techniques"
    / "layers"
    / "logic"
)

# Names Python injects into every module namespace without an explicit binding.
IMPLICIT_MODULE_NAMES = {
    "__name__",
    "__file__",
    "__doc__",
    "__package__",
    "__spec__",
    "__loader__",
    "__builtins__",
    "__path__",
    "__debug__",
    "__class__",  # implicit closure cell inside any method body
}

BUILTIN_NAMES = set(dir(builtins)) | IMPLICIT_MODULE_NAMES


def _logic_files() -> List[pathlib.Path]:
    return sorted(p for p in LOGIC_PKG.glob("*.py"))


def _collect_bound_names(tree: ast.AST) -> Set[str]:
    """Collect every name bound anywhere in ``tree`` by any binding form.

    Covers: imports (plain and ``as``), function/class definitions, every
    argument kind, every ``Store``/``Del`` context ``ast.Name`` (which subsumes
    assignments, augmented assignments, annotated assignments, walrus, ``for``
    targets, comprehension targets and ``with ... as``), ``except ... as`` (a
    plain ``str`` attribute, not a ``Name`` node), ``global`` and ``nonlocal``,
    and ``match`` capture patterns.
    """
    bound: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.MatchStar) or isinstance(node, ast.MatchAs):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                bound.add(node.rest)
    return bound


def _unbound_loads(path: pathlib.Path) -> List[Tuple[int, str]]:
    """Return ``(lineno, name)`` for every loaded name with no binding."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = _collect_bound_names(tree)
    hits = {
        (node.lineno, node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in bound
        and node.id not in BUILTIN_NAMES
    }
    return sorted(hits)


def test_logic_package_has_source_files() -> None:
    """The guard is worthless if it silently scans nothing."""
    files = _logic_files()
    assert len(files) >= 5, f"expected >=5 modules under {LOGIC_PKG}, found {files}"


def test_no_wildcard_imports() -> None:
    """A wildcard import would blind the undefined-name scan below."""
    offenders = [
        f"{path.name}:{node.lineno}"
        for path in _logic_files()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
    ]
    assert not offenders, f"wildcard imports defeat this guard: {offenders}"


@pytest.mark.parametrize("path", _logic_files(), ids=lambda p: p.name)
def test_no_unbound_names(path: pathlib.Path) -> None:
    """Every loaded name in the module resolves to a binding or a builtin."""
    unbound = _unbound_loads(path)
    listing = "\n".join(f"  {path.name}:{line} -> {name}" for line, name in unbound)
    assert not unbound, (
        f"{len(unbound)} unbound name(s) in {path.name}:\n{listing}\n"
        "Each will raise NameError at runtime. If this is `ops`, the "
        "`keras.` prefix was dropped from a nested call (see 527919265)."
    )


def test_package_wide_unbound_total_is_zero() -> None:
    """Package-wide count, so the failure message names every site at once."""
    per_file = {p.name: _unbound_loads(p) for p in _logic_files()}
    total = sum(len(v) for v in per_file.values())
    listing = "\n".join(
        f"  {name}:{line} -> {ident}"
        for name, hits in per_file.items()
        for line, ident in hits
    )
    assert total == 0, f"{total} unbound name(s) package-wide:\n{listing}"
