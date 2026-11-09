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
   routes are expressed as AST shapes rather than as text:

     - a SCOPED attribute access (`ast.Attribute` under the rule above), in
       either Load or Store context -- `config.total_steps = args.total_steps`
       counts, because a CLI override wiring a value onto the config is how
       several trainers consume a field;
     - a KEYWORD ARGUMENT named for the field at a call site
       (`ast.keyword`) -- `Trainer(save_best_only=config.save_best_only)`, not
       any line containing `<field> =`;
     - a string CONSTANT equal to the field name -- dict key / `getattr`;
     - a read `<param>.<field>` inside a `src/` function the config is HANDED
       to (`prepare_run_dir(config)` reads `config.output_dir` in a module
       that never imports the class), one hop;
     - a key lookup of the field in a loader of the saved `config.json`,
       declared per field in `READ_BACK_FROM_THE_SAVED_CONFIG` with its reader.

3. PLUMBING IS NOT CONSUMPTION (step 13.10 of plan 98b866fb, measured by
   mutation in step 10e). The field's own CLI flag literal (`'--label-smoothing'`)
   and the keyword of the config's OWN constructor call, value included
   (`TrainingConfig(label_smoothing=args.label_smoothing)`), only fill the
   field. Both used to count, which is how the ResNet `label_smoothing`, wired
   to its flag and read by nothing, stayed green.

   `test_an_unscoped_assignment_is_not_consumption` pins the closed holes.

Scope of the search
-------------------
Module-local search would produce false positives: the NTM task configs are
declared in `train/ntm/config.py` and consumed by `train/ntm/harness.py`. So the
search covers the declaring module, the module of each base class (a subclass is
read wherever its base is), PLUS every file under `src/` that actually IMPORTS
the class from that module. `tests/` is NOT searched: a test that types the name
of a deleted field (`config.label_smoothing`) made the field read as live,
measured by mutation; a field only a test reads is one the program never reads.

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
    ("train/time_series/timestral/train_timestral.py", "TimestralTrainingConfig"),
    ("train/bert/wikipedia/pretrain.py", "PretrainConfig"),
    ("train/ntm/config.py", "CopyTaskConfig"),
    ("train/ntm/config.py", "AssociativeRecallConfig"),
    ("train/som_nd_soft/train_cifar.py", "CIFARSOMConfig"),
    ("train/rms_variants_train/config.py", "ExperimentConfig"),
    ("train/resnet/train_resnet.py", "TrainingConfig"),
    ("train/vit/train_vit.py", "TrainingConfig"),
    ("train/rms_variants_train/sweep.py", "RunSpec"),
    ("train/common/clm_pretrain.py", "ClmPretrainConfig"),
    ("train/gpt2/pretrain.py", "TrainingConfig"),
    ("train/gpt2/pretrain_so.py", "SOTrainingConfig"),
    ("train/gpt2/pretrain_harmonic.py", "HarmonicTrainingConfig"),
    ("train/gpt2/finetune.py", "FinetuneConfig"),
    ("train/wave_field/pretrain.py", "TrainingConfig"),
    ("train/language/colbert/common.py", "TrainingConfig"),
    ("train/bit_diffusion/train_bit_diffusion.py", "TrainingConfig"),
    ("train/dit/train_dit.py", "TrainingConfig"),
    ("train/doc_res/common.py", "DocResTrainingConfig"),
    ("train/hnet/common.py", "HNetTrainingConfig"),
    ("train/doc_scanner/common.py", "DocScannerTrainingConfig"),
    ("train/mamba/common.py", "Mamba2TrainingConfig"),
    ("train/mamba/common_v1.py", "MambaV1TrainingConfig"),
    ("train/zamba2/common.py", "Zamba2TrainingConfig"),
    ("train/gemma/common.py", "Gemma3TrainingConfig"),
    ("train/qwen/common.py", "Qwen3TrainingConfig"),
    ("train/qwen3_next/common.py", "Qwen3NextTrainingConfig"),
    ("train/qwen3_embeddings/common.py", "TrainingConfig"),
    ("train/topographic_vae/train_topographic_vae.py", "TopographicVAEConfig"),
    ("train/distilbert/pretrain.py", "DistilBertTrainingConfig"),
    ("train/modern_bert/pretrain.py", "ModernBertTrainingConfig"),
    ("train/mini_vec2vec/train_mini_vec2vec.py", "MiniVec2VecTrainingConfig"),
    ("train/power_mlp/train_power_mlp.py", "TrainingConfig"),
    ("train/convnext/common.py", "TrainingConfig"),
    ("train/convunext/common.py", "SegTrainingConfig"),
    ("train/bfunet/common.py", "BFUnetTrainingConfig"),
    ("train/bfunet/train_convunext_denoiser.py", "TrainingConfig"),
    ("train/lightglue/train_lightglue.py", "LightGlueTrainConfig"),
    ("train/siamfc/train_siamfc.py", "SiamFCTrainingConfig"),
    ("train/dasiamrpn/train_dasiamrpn.py", "DaSiamRPNTrainingConfig"),
    ("train/deepsort/train_deepsort.py", "DeepSortTrainingConfig"),
    ("train/mambalct/train_mambalct.py", "MambaLCTTrainingConfig"),
    # The nine rows below (the seven argument groups, the task config and the data config
    # of train.api) prove LITTLE here. This guard counts a field consumed when a dict key or
    # ANY keyword argument of that name exists in a searched file (test files stopped counting
    # in step 13.10 of plan 98b866fb), and a mutant such as `normalize="standardize"` keeps
    # a field green (the keyword name itself counts). A green row is NOT evidence that the
    # field is wired. What proves it is
    # tests/test_train/test_api/test_the_api_fields_are_read_in_production.py (a real
    # attribute read in src/train/api, outside the declaring class) and the behaviour tests
    # of test_the_run_reaches_what_it_was_told.py. Review iter 2, concern 9.
    ("train/api/args.py", "OptimArgs"),
    ("train/api/args.py", "TransformArgs"),
    ("train/api/args.py", "MetricArgs"),
    ("train/api/args.py", "CheckpointArgs"),
    ("train/api/args.py", "LoggerArgs"),
    ("train/api/args.py", "AnalysisArgs"),
    ("train/api/args.py", "DebugArgs"),
    ("train/api/tasks/image_classification.py", "ImageClassificationConfig"),
    ("train/api/data_args.py", "BuiltinImageData"),
    ("train/api/data_args.py", "FolderImageData"),
    ("train/api/data_args.py", "CsvImageData"),
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

# A field the trainer writes into the run's ``config.json`` (the whole-config dump) and that a LOADER reads back BY NAME is
# consumed: the loader is the consumer, it only does not import the class, so no route above can see it. This is not an
# exemption: each row names its reader (a path under ``src/``), and the field counts only while that file still holds the
# field's name as a string constant (the key lookup), so a row cannot outlive its reader.
READ_BACK_FROM_THE_SAVED_CONFIG: Dict[Tuple[str, str], Dict[str, str]] = {
    # The pixel-domain provenance stamp (D-005 of plan e56909cd): nothing branches on it in training; every checkpoint's
    # config.json carries it, and ``require_unit_domain_checkpoint`` (``raw.get("data_range", ...)``) refuses a checkpoint
    # whose config.json does not record "[0,1]", for ``--init-from``, ``DenoiserPrior`` and the bfunet eval scripts.
    ("train/bfunet/common.py", "BFUnetTrainingConfig"): {"data_range": "dl_techniques/utils/denoiser_provenance.py"},
}

# Receivers that plausibly ARE the config object. See trap 2 in the module
# docstring: a bare `<anything>.field` is NOT accepted. `<anything>_config` /
# `<anything>_cfg` / the class name itself are accepted too, via
# `_is_config_receiver`.
_BARE_RECEIVERS = frozenset({"self", "config", "cfg", "args"})

_DECL_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z_0-9]*\s*:\s")


@lru_cache(maxsize=None)
def _parse(path: Path) -> Optional[ast.Module]:
    """Parsed AST for `path`, or None if it will not parse.

    Cached: the importer scan walks every file under `src/` once
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
    # PRODUCTION FILES ONLY (step 13.10 of plan 98b866fb, review of iteration 2): a test that types the name of a deleted field
    # (``TrainingConfig(label_smoothing=0.1)``, ``config.label_smoothing``) made it read as live, measured by mutation. A field
    # only a test reads is a field the program never reads.
    for root in (SRC,):
        for path in root.rglob("*.py"):
            if path == decl or "__pycache__" in path.parts:
                continue
            if _imports_class(path, dotted, class_name):
                files.append(path)
    # A SUBCLASS is read wherever its base is read: `common.train(config)` hands the ConvUNeXt `TrainingConfig` to the code of
    # its base `BFUnetTrainingConfig`, which reads the re-declared `model_analysis`. So the module of each base is searched.
    tree = _parse(decl)
    imported = _imported_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for base in node.bases:
                origin = imported.get(base.id) if isinstance(base, ast.Name) else None
                path = _module_file(origin[0]) if origin else None
                if path is not None and path not in files:
                    files.append(path)
    return files


def _module_file(dotted: str) -> Optional[Path]:
    """The file under `src/` that defines module `dotted`, or None (third-party, or not a module)."""
    base = SRC / dotted.replace(".", "/")
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imported_names(tree: Optional[ast.Module]) -> Dict[str, Tuple[str, str]]:
    """Local name -> (dotted module, name in that module) for every absolute `from <module> import <name>`."""
    if tree is None:
        return {}
    return {
        alias.asname or alias.name: (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
        for alias in node.names
    }


@lru_cache(maxsize=None)
def _function_def(module: str, name: str, hops: int = 1) -> Optional[Tuple[Path, ast.FunctionDef]]:
    """The top-level `def <name>` of `module` under `src/`, following at most `hops` re-exports
    (`from train.common.run_io import prepare_run_dir` in `train/common/__init__.py`). None when it is not found."""
    path = _module_file(module)
    tree = _parse(path) if path is not None else None
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return path, node
    origin = _imported_names(tree).get(name)
    return _function_def(*origin, hops=hops - 1) if origin and hops > 0 else None


def _is_the_config(node: ast.expr, class_name: str) -> bool:
    """True if a call argument is the config object itself: a config receiver name other than `self` and the argparse `args`."""
    return isinstance(node, ast.Name) and node.id not in ("self", "args") and _is_config_receiver(node, class_name)


def _handed_config_reads(class_name: str, files: Tuple[Path, ...]) -> Dict[str, List[str]]:
    """Field name -> sites where a function the config is HANDED to reads it (one hop).

    `prepare_run_dir(config)` reads `config.output_dir` in `train/common/run_io.py` and `create_clm_loss_fn(config)` reads
    `config.focal_gamma` in `train/common/clm_pretrain.py`; neither module imports the class, so the importer scan alone
    misses both. For each call, in a searched file, of a function imported from a `src/` module with the config itself as an
    argument (`config`, `cfg`, `<x>_config`; not `self`, not the argparse `args`), the parameter it binds is the receiver,
    and only `<param>.<field>` inside that function's body counts. One hop: a read two calls down is not seen.
    """
    reads: Dict[str, List[str]] = {}
    for path in files:
        tree = _parse(path)
        imported = _imported_names(tree)
        for call in ast.walk(tree) if tree is not None else ():
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in imported):
                continue
            found = _function_def(*imported[call.func.id])
            if found is None:
                continue
            callee_path, func = found
            params = [a.arg for a in func.args.posonlyargs + func.args.args]
            bound = {params[i] for i, v in enumerate(call.args) if i < len(params) and _is_the_config(v, class_name)}
            bound |= {kw.arg for kw in call.keywords if kw.arg and _is_the_config(kw.value, class_name)}
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound:
                    reads.setdefault(node.attr, []).append(
                        f"{callee_path.relative_to(REPO_ROOT)}:{node.lineno} (handed the config at "
                        f"{path.relative_to(REPO_ROOT)}:{call.lineno})"
                    )
    return reads


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

    Three in-file routes -- scoped attribute access, keyword argument at a call
    site other than the config's own constructor, string constant (dict key /
    getattr) -- so a field wired by any of them counts as live (the two
    cross-file routes are `_handed_config_reads` and `_read_back_sites`). See
    the module docstring for why this is an
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

    def _consumes(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == self.field and _is_config_receiver(
                node.value, self.class_name
            )
        if isinstance(node, ast.keyword):
            return node.arg == self.field
        if isinstance(node, ast.Constant):
            # The field's own CLI flag (``--label-smoothing``) is PLUMBING, not consumption: it only puts a value on the config
            # (step 13.10 of plan 98b866fb: how the ResNet ``label_smoothing`` stayed green while nothing read it).
            return isinstance(node.value, str) and node.value == self.field
        return False

    def nodes(self, tree: ast.AST) -> List[ast.AST]:
        # DECISION plan-2026-11-03T221936-98b866fb/D-065: do NOT count the constructor keyword or its value again, and do NOT
        # put `tests/` back in the search: either one alone kept a field wired only to its own flag green.
        # A keyword of the config's OWN constructor call (``TrainingConfig(label_smoothing=args.label_smoothing)``) is plumbing
        # as well: it fills the field, and fills it from the same flag. The WHOLE keyword node is exempt, its value included,
        # since ``args.label_smoothing`` is a scoped read named like the field. The keyword of any OTHER call still counts.
        plumbing = set()
        for call in ast.walk(tree):
            if isinstance(call, ast.Call):
                func = call.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                if name == self.class_name:
                    plumbing.update(id(n) for kw in call.keywords for n in ast.walk(kw))
        return [n for n in ast.walk(tree) if id(n) not in plumbing and self._consumes(n)]

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


def _read_back_sites(rel: str, class_name: str, field: str) -> List[str]:
    """Sites in the declared reader of `field` (READ_BACK_FROM_THE_SAVED_CONFIG) that look its name up as a string key."""
    reader = READ_BACK_FROM_THE_SAVED_CONFIG.get((rel, class_name), {}).get(field)
    tree = _parse(SRC / reader) if reader else None
    if tree is None:
        return []
    return [
        f"src/{reader}:{node.lineno}  (reads the saved config.json by name)"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == field
    ]


def _dead_fields(rel: str, class_name: str) -> Dict[str, List[str]]:
    files = _search_files(rel, class_name)
    fields = _declared_fields(SRC / rel, class_name)
    handed = _handed_config_reads(class_name, tuple(files))
    return {
        f: _consumption_sites(f, class_name, files) + handed.get(f, []) + _read_back_sites(rel, class_name, f)
        for f in fields
    }


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
    assert pattern.search("getattr(cfg, 'save_best_only')")
    # ...and the plumbing that only fills the field does not (step 13.10): its own CLI flag, and its keyword in the config's
    # own constructor call.
    assert not pattern.search("parser.add_argument('--save-best-only')")
    assert not pattern.search("config = TrainingConfig(save_best_only=args.save_best_only)")
    assert pattern.search("trainer = Trainer(save_best_only=config.save_best_only)")
