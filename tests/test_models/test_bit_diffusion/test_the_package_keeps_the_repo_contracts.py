"""Scoped re-statements of the repo-wide contracts this package has already broken.

**Why this file exists.** `tests/test_models/test_package_api_contract.py` is a
repo-wide suite; `tests/test_models/test_bit_diffusion/` is a scoped one. At step 6
of this port the scoped suite was green at 256 passed while the repo-wide one was
red on two failures THIS PACKAGE had introduced three steps earlier:

1. `sde.py` called `keras.backend.standardize_dtype` -- a Keras-2 residue banned
   under `models/` by `TestNoKeras2Residues::test_no_keras_backend_calls`;
2. `create_bridge_sde(variant=...)` matched the model-factory shape that
   `TestCreateFactoriesDelegateToFromVariant` requires to carry a `from_variant`
   classmethod, and so silently joined that suite's scope-exclusion pin.

Both went unseen for three steps because no gate the package ran could see them.

**What this file is NOT.** It is not a substitute for running
`tests/test_models/test_package_api_contract.py`. Defect (2) above is asserted
here only in its *local* form (no `create_*(variant=...)` in this package); the
repo-wide pin is a SET EQUALITY over the whole tree, and no scoped arm can
reproduce that. The repo-wide suite must still be run before this port closes.
This file buys early detection, not coverage.
"""

import ast
import re
from pathlib import Path
from typing import Set

PACKAGE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "dl_techniques"
    / "models"
    / "vision_language"
    / "bit_diffusion"
)


def _package_modules():
    """Every ``.py`` in this package, as ``(relative_name, source_text)``."""
    paths = sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    return [(p.name, p.read_text(), p) for p in paths]


def _docstring_lines(source: str) -> Set[int]:
    """1-based line numbers covered by any module/class/function docstring.

    Same exclusion the repo-wide guard applies: prose explaining *why*
    ``keras.backend.`` is banned must not itself trip the ban.
    """
    lines: Set[int] = set()
    tree = ast.parse(source)
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def _keras_backend_offenders(source: str, name: str):
    """Executable lines in ``source`` that call ``keras.backend.*``."""
    docstrings = _docstring_lines(source)
    offenders = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if i in docstrings:
            continue
        if "keras.backend." in line:
            offenders.append(f"{name}:{i} {stripped}")
    return offenders


def test_the_package_has_no_keras_backend_calls():
    """The banned Keras-2 residue, asserted where this package can see it.

    Reproduces the predicate of
    ``test_package_api_contract.py::TestNoKeras2Residues::test_no_keras_backend_calls``
    over this package's own files only, so it costs milliseconds and no
    tree-wide import. RED-proved by reinstating the
    ``keras.backend.standardize_dtype`` call in ``bridge_math_dtype``.

    The sanctioned replacement is ``getattr(dtype, "name", None) or str(dtype)``,
    the same two-step ``colbert/components.py``, ``sd3_mmdit/text_encoders.py``
    and ``utils/geometry/poincare_math.py`` already use.
    """
    offenders = []
    for name, source, _path in _package_modules():
        offenders.extend(_keras_backend_offenders(source, name))
    assert not offenders, (
        "keras.backend.* is a Keras-2 residue banned under models/ by "
        "tests/test_models/test_package_api_contract.py::TestNoKeras2Residues. "
        'Use `getattr(dtype, "name", None) or str(dtype)`, keras.config.floatx() '
        f"or keras.config.epsilon(). Found: {offenders}"
    )


def test_the_scan_can_actually_see_a_keras_backend_call():
    """Anti-vacuity: the arm above is worthless if the scanner reads nothing.

    A scanner pointed at an empty file list, or one whose docstring exclusion
    swallowed the whole module, would report zero offenders forever. This pins
    that the scanner (a) reads a non-trivial number of modules and (b) DOES flag
    the exact line shape it is meant to flag.
    """
    modules = _package_modules()
    assert len(modules) >= 5, (
        f"expected the bit_diffusion package to hold at least 5 modules, the "
        f"scanner found {len(modules)}: it is pointed at the wrong directory"
    )
    assert sum(len(s.splitlines()) for _n, s, _p in modules) > 1000

    injected = (
        '"""A docstring naming keras.backend. must NOT count."""\n'
        "# a comment naming keras.backend. must NOT count\n"
        'x = keras.backend.standardize_dtype(d) == "float64"\n'
    )
    hits = _keras_backend_offenders(injected, "injected.py")
    assert len(hits) == 1 and hits[0].startswith("injected.py:3"), hits


def test_no_create_factory_in_this_package_takes_a_variant_parameter():
    """``variant`` is a reserved parameter name for module-level ``create_*``.

    The repo-wide sweep ``_sweep_create_delegation`` classifies any module-level
    ``create_*`` with a parameter literally named ``variant`` as a MODEL factory
    and demands a ``from_variant`` classmethod in the same module; a module with
    none lands in the ``_CREATE_WITHOUT_FROM_VARIANT`` set-equality pin and turns
    it red. Nothing in this package is a ``keras.Model`` factory with variants,
    so the correct answer is to not use the word. See decisions.md D-015.

    **Known blind spot**: this arm is LOCAL. It cannot see the repo-wide set
    equality, so it would not have caught the original defect in the form the
    contract suite reported it -- only in the form that causes it.
    """
    offenders = []
    for name, source, _path in _package_modules():
        tree = ast.parse(source)
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.name.startswith("create_"):
                continue
            params = [a.arg for a in fn.args.args] + [
                a.arg for a in fn.args.kwonlyargs
            ]
            if "variant" in params:
                offenders.append(f"{name}:{fn.lineno} {fn.name}(variant=...)")
    assert not offenders, (
        "a module-level create_*(variant=...) is read tree-wide as a model "
        "factory that must delegate to from_variant. Rename the parameter "
        f"(e.g. `sde_type`). Found: {offenders}"
    )


def test_no_module_level_variant_table_uses_the_reserved_spelling():
    """``[A-Z0-9]+_VARIANTS`` / ``VARIANT_CONFIGS`` are reserved table names too.

    ``_LEGACY_VARIANT_TABLE_RE`` in the contract suite treats a table so named as
    a legacy spelling of ``MODEL_VARIANTS`` and requires it to be reachable as
    such from a ``from_variant``. This package's ``SDE_TYPES`` deliberately does
    not match. Today the rule only fires for classes that define
    ``from_variant``, so this arm guards a LATENT trip-wire, not a live one.
    """
    reserved = re.compile(r"^(?:[A-Z0-9]+_VARIANTS|VARIANT_CONFIGS)$")
    offenders = []
    for name, source, _path in _package_modules():
        tree = ast.parse(source)
        for node in tree.body:
            targets = []
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
            offenders.extend(
                f"{name}:{node.lineno} {t}" for t in targets if reserved.match(t)
            )
    assert not offenders, (
        "this spelling is reserved for MODEL_VARIANTS-equivalent tables; use a "
        f"name like SDE_TYPES. Found: {offenders}"
    )
