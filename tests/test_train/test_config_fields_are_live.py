r"""Guard: a trainer config must not declare a field that nothing consumes.

The defect class this exists to catch
------------------------------------
A dataclass field is declared, documented, sometimes even written into
`config.json` -- and NOTHING ever reads it. A user who sets it sees nothing
happen. It is the same class of bug as a CLI arg that silently no-ops because
`main()` forgets to forward it, and the same class as
`FinetuneConfig.save_dir` (commit 8973b2668), which existed only to make an
empty directory.

Two traps are baked into the design of this guard, both MEASURED rather than
assumed:

1. SERIALIZATION IS NOT CONSUMPTION.
   `save_config_json` / `asdict` / `prepare_run_dir` dump the WHOLE config, so
   every field of every config appears in the JSON. That records the field; it
   does not consume it. A field that only ever reaches a JSON dump still
   changes no behaviour, and a reader of that JSON is actively misled
   (`"save_model_checkpoints": false` while checkpoints keep being written).
   This guard therefore never counts a whole-config dump as a read.

2. THE READ-SET MUST BE SCOPED TO THE CONFIG OBJECT.
   The first draft of the tree_transformer guard collected every
   `ast.Attribute` name in the module. An unrelated
   `_PRETRAIN_SAVE_DIR = _PretrainConfig.save_dir` line put "save_dir" into the
   read-set, and the guard PASSED against the exact mutation it existed to
   catch. So a read only counts when the receiver plausibly IS the config:
   `self.X`, `config.X`, `cfg.X`, `<anything>_config.X`, `<ClassName>.X`,
   or through one `.config` / `.cfg` hop. Never a bare `<anything>.X`.

   THE FIRST DRAFT OF *THIS* GUARD REPEATED THAT MISTAKE THROUGH A SIDE DOOR.
   Its consumption test was a regex over raw lines, and one of its four
   alternatives -- `\b<field>\s*=[^=]`, meant to catch a kwarg at a
   construction site -- carried no receiver scoping at all. So
   `unrelated_object.save_best_only = 5`, or a bare local `save_best_only = 5`,
   counted as consumption of the config field while the docstring above
   promised it could not. The consumption test is therefore AST-based, and the
   four routes are expressed as AST shapes rather than as text:

     - a SCOPED attribute access (`ast.Attribute` under the rule above), in
       either Load or Store context -- `config.total_steps = args.total_steps`
       counts, because a CLI override wiring a value onto the config is how
       several trainers consume a field;
     - a KEYWORD ARGUMENT named for the field at a call site
       (`ast.keyword`) -- `ExperimentConfig(csv_filename=...)`, not any line
       containing `<field> =`;
     - a string CONSTANT equal to the field name -- dict key / `getattr`;
     - a string CONSTANT equal to the field's CLI flag (`--field-name`) --
       an exact literal, not a substring of some longer line.

   `test_an_unscoped_assignment_is_not_consumption` pins the closed hole.

Scope of the search
-------------------
Module-local search would produce false positives: the NTM task configs are
declared in `train/ntm/config.py` and consumed by `train/ntm/harness.py` and by
`tests/`. So the search covers the declaring module PLUS every file under
`src/` and `tests/` that actually IMPORTS the class from that module.

Importer-scoping rather than a repo-wide name grep is deliberate, and it is what
makes this guard stronger than the scratch detector that motivated it. That
detector matched field names across the whole repo, so
`CIFARSOMConfig.perceptual_weight` looked alive because
`losses/image_restoration_loss.py` has an unrelated `self.perceptual_weight`,
and `CIFARSOMConfig.checkpoint_frequency` looked alive because
`blt/train_blt.py` has an unrelated `self.config.checkpoint_frequency`. Both
CIFARSOM fields are in fact dead. They are pinned in KNOWN_DEAD below rather
than silently tolerated.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"

# (module path relative to src/, class name). Every config class touched by the
# dead-field sweep. Add a row when you fix a config; never delete a row.
REGISTERED: List[Tuple[str, str]] = [
    ("train/time_series/tirex/train_tirex.py", "TiRexTrainingConfig"),
    ("train/bert/wikipedia/pretrain.py", "PretrainConfig"),
    ("train/ntm/config.py", "CopyTaskConfig"),
    ("train/ntm/config.py", "AssociativeRecallConfig"),
    ("train/som_nd_soft/train_cifar.py", "CIFARSOMConfig"),
    ("train/rms_variants_train/config.py", "ExperimentConfig"),
    ("train/resnet/train_resnet.py", "TrainingConfig"),
    ("train/vit/train_vit.py", "TrainingConfig"),
    ("train/rms_variants_train/sweep.py", "RunSpec"),
]

# Fields known to be dead but OUT OF SCOPE of the sweep that introduced this
# guard. Pinned, not ignored: `test_known_dead_fields_are_still_dead` fails if
# one of them becomes live, which is the signal to delete the exemption (or to
# delete the field). An empty exemption list is the goal state.
#
# IT IS NOW EMPTY, which is that goal state reached, not an exemption list
# quietly discarded. All seven fields it used to pin were DELETED:
# `CIFARSOMConfig.perceptual_weight`, `CIFARSOMConfig.checkpoint_frequency`,
# `CopyTaskConfig.max_sequence_length` (with its `:param:` line),
# `ExperimentConfig.csv_filename`, `RunSpec.csv_filename`, and
# `save_best_only` on both resnet's and vit's `TrainingConfig`. Every one of
# those six classes is in REGISTERED, so `test_config_declares_no_field_it_
# never_consumes` -- not an exemption -- is what covers them now. Do not add a
# row here to make a newly-dead field go green; wire it or delete it.
KNOWN_DEAD: Dict[Tuple[str, str], Set[str]] = {}

# Receivers that plausibly ARE the config object. See trap 2 in the module
# docstring: a bare `<anything>.field` is NOT accepted. `<anything>_config` /
# `<anything>_cfg` / the class name itself are accepted too, via
# `_is_config_receiver`.
_BARE_RECEIVERS = frozenset({"self", "config", "cfg", "args"})

_DECL_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z_0-9]*\s*:\s")


@lru_cache(maxsize=None)
def _parse(path: Path) -> Optional[ast.Module]:
    """Parsed AST for `path`, or None if it will not parse.

    Cached: the importer scan walks every file under `src/` and `tests/` once
    per registered class, and re-parsing thousands of files eight times over
    dominated the runtime.
    """
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


@lru_cache(maxsize=None)
def _read_lines(path: Path) -> Tuple[str, ...]:
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _declared_fields(path: Path, class_name: str) -> List[str]:
    """Annotated field names declared directly on `class_name` in `path`."""
    tree = _parse(path)
    assert tree is not None, f"{path} does not parse"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    raise AssertionError(f"class {class_name} not found in {path}")


def _module_dotted(rel: str) -> str:
    return rel[: -len(".py")].replace("/", ".")


def _imports_class(path: Path, dotted: str, class_name: str) -> bool:
    """True if `path` imports `class_name` from the module `dotted`.

    Handles absolute (`from train.ntm.config import CopyTaskConfig`), relative
    (`from .config import CopyTaskConfig`) and whole-module
    (`import train.ntm.config`) forms. Matching the MODULE and not just the
    class name is what keeps the two distinct `TrainingConfig` classes
    (resnet's and vit's) from being conflated.
    """
    tree = _parse(path)
    if tree is None:
        return False
    tail = dotted.split(".")[-1]
    parent = dotted.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            mod = node.module or ""
            if class_name in names and (
                mod == dotted or (node.level and mod.split(".")[-1] == tail)
            ):
                return True
            # `from train.ntm import config` then `config.CopyTaskConfig`
            if tail in names and mod in (parent, ""):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name == dotted for a in node.names):
                return True
    return False


def _search_files(rel: str, class_name: str) -> List[Path]:
    decl = SRC / rel
    dotted = _module_dotted(rel)
    files = [decl]
    for root in (SRC, TESTS):
        for path in root.rglob("*.py"):
            if path == decl or "__pycache__" in path.parts:
                continue
            if _imports_class(path, dotted, class_name):
                files.append(path)
    return files


def _is_config_receiver(node: ast.expr, class_name: str) -> bool:
    """True if `node` plausibly evaluates to the config object.

    `self` / `config` / `cfg` / `args` / `<anything>_config` / `<anything>_cfg`
    / `<ClassName>`, or anything reached through a `.config` / `.cfg`
    attribute (`self.config.X`, `trainer.cfg.X`). Never a bare
    `<anything>.X` -- that is trap 2.
    """
    if isinstance(node, ast.Name):
        name = node.id
        return (
            name in _BARE_RECEIVERS
            or name.endswith(("_config", "_cfg"))
            or name == class_name
        )
    if isinstance(node, ast.Attribute):
        # `<anything>.config.X` / `<anything>.cfg.X`: the receiver is the
        # attribute literally named `config`/`cfg`.
        return node.attr in ("config", "cfg")
    return False


class _Consumption:
    """AST shapes that plausibly CONSUME `field`.

    Four routes -- scoped attribute access, keyword argument at a call site,
    string constant (dict key / getattr), CLI flag literal -- so a field wired
    by any of them counts as live. See the module docstring for why this is an
    AST walk and not a regex over lines: the regex's kwarg alternative had no
    receiver scoping, so an unrelated `x.field = 5` counted.

    Contract: `nodes(tree)` yields every consuming node of a parsed module;
    `search(snippet)` parses a source SNIPPET and returns True if it consumes
    the field (the probe used by the scoping tests below). A snippet that does
    not parse yields no consumption rather than raising.
    """

    def __init__(self, field: str, class_name: str) -> None:
        self.field = field
        self.class_name = class_name
        self.flag = "--" + field.replace("_", "-")

    def _consumes(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == self.field and _is_config_receiver(
                node.value, self.class_name
            )
        if isinstance(node, ast.keyword):
            return node.arg == self.field
        if isinstance(node, ast.Constant):
            return isinstance(node.value, str) and node.value in (self.field, self.flag)
        return False

    def nodes(self, tree: ast.AST) -> List[ast.AST]:
        return [n for n in ast.walk(tree) if self._consumes(n)]

    def search(self, snippet: str) -> bool:
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            return False
        return bool(self.nodes(tree))


def _consumption_pattern(field: str, class_name: str) -> _Consumption:
    """Kept as the name every scoping test probes through."""
    return _Consumption(field, class_name)


def _consumption_sites(field: str, class_name: str, files: List[Path]) -> List[str]:
    consumption = _consumption_pattern(field, class_name)
    sites: List[str] = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        lines = _read_lines(path)
        for node in consumption.nodes(tree):
            lineno = getattr(node, "lineno", None)
            if lineno is None:  # `ast.keyword` has one from 3.9; be safe anyway
                continue
            text = lines[lineno - 1] if lineno <= len(lines) else ""
            # `field: type = default` is a DECLARATION, not consumption.
            if _DECL_RE.match(text) and text.strip().startswith(field + ":"):
                continue
            sites.append(f"{path.relative_to(REPO_ROOT)}:{lineno}  {text.strip()[:100]}")
    return sorted(set(sites))


def _dead_fields(rel: str, class_name: str) -> Dict[str, List[str]]:
    files = _search_files(rel, class_name)
    fields = _declared_fields(SRC / rel, class_name)
    return {f: _consumption_sites(f, class_name, files) for f in fields}


@pytest.mark.parametrize("rel,class_name", REGISTERED, ids=lambda v: v.replace("/", "."))
def test_config_declares_no_field_it_never_consumes(rel: str, class_name: str) -> None:
    """Every annotated field of a registered config is consumed somewhere."""
    exempt = KNOWN_DEAD.get((rel, class_name), set())
    sites = _dead_fields(rel, class_name)
    dead = sorted(f for f, s in sites.items() if not s and f not in exempt)
    assert dead == [], (
        f"{rel}::{class_name} declares field(s) nothing consumes: {dead}. "
        "A config field nothing reads is a knob that silently does nothing -- "
        "either wire it or delete it. Note that a whole-config dump "
        "(save_config_json / asdict / prepare_run_dir) is NOT consumption."
    )


@pytest.mark.parametrize(
    "rel,class_name",
    sorted(KNOWN_DEAD),
    # KNOWN_DEAD is empty (the goal state), so pytest hands the id-maker its
    # own placeholder rather than a string -- guard for it.
    ids=lambda v: v.replace("/", ".") if isinstance(v, str) else "empty",
)
def test_known_dead_fields_are_still_dead(rel: str, class_name: str) -> None:
    """Pinned exemptions must stay dead, so they cannot rot unnoticed.

    If one becomes live, delete it from KNOWN_DEAD -- the general guard above
    then covers it. If it is still dead, it is still a bug worth fixing.
    """
    sites = _dead_fields(rel, class_name)
    revived = {
        f: sites[f] for f in KNOWN_DEAD[(rel, class_name)] if sites.get(f)
    }
    assert revived == {}, (
        f"{rel}::{class_name}: exempted field(s) now have consumption sites "
        f"{revived}. Remove them from KNOWN_DEAD."
    )


def test_the_read_set_rejects_an_unscoped_receiver() -> None:
    """The scoping from trap 2 is real, not decorative.

    `some_unrelated_object.save_dir` must NOT count as a read of a config
    field. This is the mutation the first tree_transformer draft failed to
    catch, asserted here directly so the scoping cannot be loosened silently.
    """
    pattern = _consumption_pattern("save_dir", "FinetuneConfig")
    assert not pattern.search("x = some_unrelated_object.save_dir")
    assert not pattern.search("_P = _PretrainConfig.save_dir")
    assert pattern.search("os.makedirs(config.save_dir)")
    assert pattern.search("self.save_dir")
    assert pattern.search("path = FinetuneConfig.save_dir")


def test_an_unscoped_assignment_is_not_consumption() -> None:
    """The scoping applies to WRITES too, not just reads.

    The first draft of this guard tested consumption with a regex whose kwarg
    alternative was `\\b<field>\\s*=[^=]` -- no receiver at all. So every line
    below counted as consumption of the config field, and the scoping the
    module docstring advertises was defeated through a side door for exactly
    the same reason trap 2 describes. The three lines below were each MEASURED
    green (counted as a read) under that regex; they are asserted dead here.
    The third is the same hole in the CLI-flag alternative, which matched
    `--save-best-only` as a SUBSTRING of any line, prose included.

    A write ONTO the config still counts: a CLI override is a real consumer.
    """
    pattern = _consumption_pattern("save_best_only", "TrainingConfig")
    assert not pattern.search("unrelated_object.save_best_only = 5")
    assert not pattern.search("save_best_only = 5")
    assert not pattern.search("logger.info('nothing to do with --save-best-only here')")
    # ...while the routes that ARE consumption still fire.
    assert pattern.search("config.save_best_only = args.save_best_only")
    assert pattern.search("ModelCheckpoint(save_best_only=True)")
    assert pattern.search("parser.add_argument('--save-best-only')")
    assert pattern.search("getattr(cfg, 'save_best_only')")
