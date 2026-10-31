"""Every field of the API's configs is READ by production code (review iter 2, concern 9).

``tests/test_train/test_config_fields_are_live.py`` counts a field consumed when a
test file, a dict key or ANY keyword argument of that name exists. Measured on the
API's own classes (``rev2/p_guard.py``), 51 of 58 fields stayed green with zero
production reads, and six direct mutants of the other seven all stayed green
(``normalize="standardize"`` passes, because the keyword name itself counts). That
guard cannot tell a wired field from a declared one. This file can:

* it walks every ``.py`` file under ``src/train/api`` and
  ``src/train/common/classification_run.py``;
* a field is READ when some ``ast.Attribute`` with ``ctx=Load`` carries its name and
  the receiver is not a module (``keras.ops``, ``recipes.TASK``, ``os.path`` are module
  attributes, not reads of a config);
* a dict key (``given["format"]``), a keyword argument (``f(seed=1)``), a string
  (``getattr(cfg, "top_k")``), a ``Store`` and every test file do NOT count;
* a read INSIDE the declaring class does not count either: ``problems`` and
  ``resolve_auto`` read their own fields to validate and to fill them, and a field that
  is validated and never used is the dead field this file exists to find. (This is
  stricter than the brief of concern 9, which allows any receiver; the declaring-class
  rule is measured below to cost no exemption.)

A field leaves this rule only through a row of :data:`EXEMPT`, which names the field
and says why. A row whose field is in fact read is stale and fails, so the table cannot
only grow. Removing the one read of a field fails ``test_every_field_is_read_...``;
``test_the_scan_*`` plant the shapes that must and must not count.
"""

import ast
import dataclasses
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from train.api import args as args_module
from train.api.data_args import BuiltinImageData, CsvImageData, FolderImageData
from train.api.tasks.image_classification import ImageClassificationConfig

from ._groups import args_groups

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"

#: Where production reads are searched: all of ``train.api`` and the shared run sequence.
SCANNED = sorted((SRC / "train" / "api").rglob("*.py")) + [
    SRC / "train" / "common" / "classification_run.py"
]

#: ``(class name, field)`` -> why the field has no attribute read. One row per field, each
#: with a reason; a row for a field that IS read fails ``test_no_exemption_is_stale``.
EXEMPT: Dict[Tuple[str, str], str] = {}


# ---------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------


def module_names(tree: ast.Module, package: str) -> Set[str]:
    """Return the names a module file binds to MODULES (``import x``, ``from p import module``).

    :param tree: the parsed file.
    :param package: the dotted package the file lives in (``train.api.tasks``), to resolve
        relative imports.
    :return: names bound by ``import a.b`` (``a``), ``import a.b as c`` (``c``) and
        ``from p import n`` where ``p.n`` is a module file or package under ``src/``.
    """
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(parts + ([base] if base else []))
            for alias in node.names:
                path = SRC.joinpath(*base.split("."), alias.name)
                if path.with_suffix(".py").is_file() or (path / "__init__.py").is_file():
                    names.add(alias.asname or alias.name)
    return names


def root_name(node: ast.AST):
    """Return the ``Name`` at the root of an attribute chain, or ``None`` for any other root."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def attribute_reads(source: str, package: str = "train.api") -> List[Tuple[str, int, str]]:
    """Return ``(attribute, line, enclosing class)`` for every real attribute read in ``source``.

    A read is an ``ast.Attribute`` with ``ctx=Load`` whose receiver chain does not start at a
    module name. ``enclosing class`` is the innermost class whose body holds the node, or
    ``""``.
    """
    tree = ast.parse(source)
    modules = module_names(tree, package)
    reads: List[Tuple[str, int, str]] = []

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, ast.ClassDef):
            owner = node.name
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and root_name(node.value) not in modules
        ):
            reads.append((node.attr, node.lineno, owner))
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "")
    return reads


def package_of(path: Path) -> str:
    return ".".join(path.relative_to(SRC).parent.parts)


def production_reads() -> Dict[str, List[str]]:
    """Return ``attribute -> ["file:line (class)", ...]`` over every scanned file."""
    found: Dict[str, List[str]] = {}
    for path in SCANNED:
        for attribute, line, owner in attribute_reads(path.read_text(), package_of(path)):
            found.setdefault(attribute, []).append(
                f"{path.relative_to(SRC).as_posix()}:{line}|{owner}"
            )
    return found


def classes_checked() -> List[type]:
    """The seven groups (found by looking), the task config and the three data configs."""
    return [*args_groups(), ImageClassificationConfig, BuiltinImageData, FolderImageData, CsvImageData]


def fields_of(cls: type) -> List[str]:
    return [field.name for field in dataclasses.fields(cls)]


def outside_read_sites(cls: type, field: str, reads: Dict[str, List[str]]) -> List[str]:
    """The read sites of ``field`` that are not inside ``cls``'s own body."""
    return [site for site in reads.get(field, []) if site.split("|")[1] != cls.__name__]


CASES = [(cls, field) for cls in classes_checked() for field in fields_of(cls)]
IDS = [f"{cls.__name__}.{field}" for cls, field in CASES]


@pytest.fixture(scope="module")
def reads():
    return production_reads()


# ---------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cls, field", CASES, ids=IDS)
def test_every_field_is_read_in_production_outside_its_own_class(cls, field, reads):
    if (cls.__name__, field) in EXEMPT:
        pytest.skip(f"exempt: {EXEMPT[(cls.__name__, field)]}")
    sites = outside_read_sites(cls, field, reads)
    assert sites, (
        f"{cls.__name__}.{field} is declared and no production code under src/train/api or "
        f"classification_run.py reads it as an attribute (reads inside {cls.__name__} itself, "
        "dict keys, keyword names and tests do not count). Wire it or delete it."
    )


def test_no_exemption_is_stale(reads):
    known = {(cls.__name__, field) for cls, field in CASES}
    for (class_name, field), reason in EXEMPT.items():
        assert (class_name, field) in known, f"{class_name}.{field} is not a field any more"
        assert reason.strip(), f"{class_name}.{field} is exempt without a reason"
        cls = next(c for c in classes_checked() if c.__name__ == class_name)
        assert not outside_read_sites(cls, field, reads), (
            f"{class_name}.{field} IS read in production now: delete its row from EXEMPT"
        )


def test_the_scan_covers_the_eleven_classes_and_the_files_it_names():
    assert len(args_groups()) == 7, "the brief names seven groups; a new one is a new row to review"
    assert len(classes_checked()) == 11
    assert all(fields_of(cls) for cls in classes_checked())
    assert all(path.is_file() for path in SCANNED)
    assert SRC / "train" / "api" / "trainer.py" in SCANNED
    assert SRC / "train" / "api" / "tasks" / "image_classification.py" in SCANNED
    assert SRC / "train" / "common" / "classification_run.py" in SCANNED
    assert args_module.__file__ and Path(args_module.__file__) in SCANNED


# ---------------------------------------------------------------------
# What counts, planted (the controls of the walk)
# ---------------------------------------------------------------------

PLANTED = '''
import os
import keras
from . import recipes

TASK = recipes.format
PATH = os.path.top_k


class Group:
    def problems(self):
        return self.inside_only


def consume(cfg, other):
    f(top_k=3)
    d = {"key_only": 1}
    d["subscript_only"]
    getattr(cfg, "string_only")
    cfg.stored = 1
    keras.ops.cast
    return cfg.real_read, other.inner.deep_read, cfg().call_read, [cfg][0].index_read
'''


def planted_reads():
    return {(attribute, owner) for attribute, _, owner in attribute_reads(PLANTED)}


@pytest.mark.parametrize(
    "attribute, owner",
    [
        ("real_read", ""),
        ("inner", ""),
        ("deep_read", ""),
        ("call_read", ""),
        ("index_read", ""),
        ("inside_only", "Group"),
    ],
)
def test_the_scan_counts_a_read_through_any_non_module_receiver(attribute, owner):
    assert (attribute, owner) in planted_reads()


@pytest.mark.parametrize(
    "attribute",
    ["top_k", "key_only", "subscript_only", "string_only", "stored", "format", "cast", "ops"],
)
def test_the_scan_does_not_count_a_keyword_a_key_a_string_a_store_or_a_module_attribute(attribute):
    assert attribute not in {name for name, _ in planted_reads()}


def test_a_read_inside_the_declaring_class_is_not_an_outside_read():
    @dataclasses.dataclass
    class Group:
        inside_only: int = 0

    sites = {"inside_only": [f"x.py:1|{Group.__name__}"], "real_read": ["x.py:2|"]}
    assert outside_read_sites(Group, "inside_only", sites) == []
    assert outside_read_sites(Group, "real_read", sites) == ["x.py:2|"]
