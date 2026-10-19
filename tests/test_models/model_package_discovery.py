"""Leaf-model-package discovery for the tests under ``tests/test_models``.

Two test modules used to build their own view of "every model package" with a
one-level directory listing of ``src/dl_techniques/models``.  That was correct
while the package was flat.  After the family restructure the same listing
returns the eleven *family* directories (``vision``, ``language``, ...) instead
of the model packages nested inside them, which silently shrinks the tested
population from seventy-nine packages to eleven non-packages.

This module is the single place that answers the question, so the two call
sites cannot drift apart again.

A **leaf model package** is a directory that

* carries an ``__init__.py``, and
* contains no subdirectory that itself carries an ``__init__.py``.

The second clause is what excludes the containers.  It removes the eleven
families and the four subfamilies (``vision/image_restoration``,
``vision/keypoints``, ``vision/super_resolution``, ``vision_language/sam``)
while keeping the packages nested beneath them, and it does so without naming
any of them -- a new family or a new nesting level needs no change here.

``time_series`` is the one container that also owns a module of its own
(``forecast.py``).  It is still a container, not a leaf, because it has package
children; ``forecast.py`` is shared forecasting machinery rather than a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

__all__ = [
    "MODELS_DIR",
    "DOC_ONLY_PACKAGES",
    "doc_only_violations",
    "leaf_packages",
    "package_names",
    "package_module",
]

MODELS_DIR = Path(__file__).resolve().parents[2] / "src" / "dl_techniques" / "models"

_ROOT_MODULE = "dl_techniques.models"


def _is_package(path: Path) -> bool:
    return (path / "__init__.py").is_file()


def _child_packages(path: Path) -> List[Path]:
    return [
        child
        for child in sorted(path.iterdir())
        if child.is_dir() and child.name != "__pycache__" and _is_package(child)
    ]


def leaf_packages(models_dir: Path | None = None) -> Dict[str, str]:
    """Map every leaf model package's bare name to its dotted module path.

    The key is the bare directory name (``"beit"``, ``"sam1"``) rather than the
    family-qualified path, because the existing coverage tables in
    ``test_roundtrip_instrument_family`` are keyed that way and there is no
    reason to churn them.

    Raises:
        RuntimeError: if two leaf packages share a bare name.  Nothing in the
            tree collides today, and a silent overwrite would drop a package
            from every caller at once, so this fails loudly instead.
    """
    root = MODELS_DIR if models_dir is None else models_dir
    found: Dict[str, str] = {}
    origin: Dict[str, Path] = {}

    def walk(directory: Path, dotted: str) -> None:
        children = _child_packages(directory)
        if children:
            for child in children:
                walk(child, f"{dotted}.{child.name}")
            return
        name = directory.name
        if name in found:
            raise RuntimeError(
                f"two leaf model packages share the bare name {name!r}: "
                f"{origin[name]} and {directory}. Keys must stay unique."
            )
        found[name] = dotted
        origin[name] = directory

    for family in _child_packages(root):
        walk(family, f"{_ROOT_MODULE}.{family.name}")

    return found


def package_names(models_dir: Path | None = None) -> List[str]:
    """Sorted bare names of every leaf model package."""
    return sorted(leaf_packages(models_dir))


def package_module(name: str, models_dir: Path | None = None) -> str:
    """Dotted module path for one leaf package's bare name."""
    return leaf_packages(models_dir)[name]


#: Leaf packages that hold documentation and no model code, by bare name, with
#: the reason. The package-API contract and the round-trip family exempt exactly
#: these names; :func:`doc_only_violations` keeps each entry true.
DOC_ONLY_PACKAGES: Dict[str, str] = {
    "metaformer": (
        "doc-only by the user's choice: model code removed, README and "
        "architecture figures kept (plan-2026-10-14T062616-81938e81 D-161, D-163)"
    ),
}


def doc_only_violations(
    names: Dict[str, str] | None = None, models_dir: Path | None = None
) -> Dict[str, List[str]]:
    """Doc-only entries that no longer match the directory on disk.

    Interface contract: ``names`` defaults to :data:`DOC_ONLY_PACKAGES` and
    ``models_dir`` to the real tree (both are parameters so a fixture tree can
    prove the check fires). Returns ``{name: [problem, ...]}`` for every entry
    that is not a leaf package, holds a ``.py`` file other than its
    ``__init__.py`` (searched recursively, ``__pycache__`` skipped), or whose
    ``__init__.py`` imports or defines anything. An empty dict means every
    exemption still describes reality. Never raises for a missing package; it
    reports it.
    """
    import ast

    names = DOC_ONLY_PACKAGES if names is None else names
    root = MODELS_DIR if models_dir is None else models_dir
    leaves = leaf_packages(root)
    problems: Dict[str, List[str]] = {}
    for name in sorted(names):
        if name not in leaves:
            problems[name] = ["not a leaf model package"]
            continue
        directory = root.joinpath(*leaves[name].split(".")[2:])
        found = []
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path == directory / "__init__.py":
                body = ast.parse(path.read_text(encoding="utf-8")).body
                code = [node for node in body
                        if not (isinstance(node, ast.Expr)
                                and isinstance(node.value, ast.Constant))]
                if code:
                    found.append(f"__init__.py has code at line {code[0].lineno}")
                continue
            found.append(f"model code: {path.relative_to(directory).as_posix()}")
        if found:
            problems[name] = found
    return problems
