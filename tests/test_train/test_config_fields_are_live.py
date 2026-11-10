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
   or through one `.config` / `.cfg` hop. Never a bare `<anything>.X`, and
   never `args.X` (step 13.10.1): the argparse namespace is not the config,
   and `args.output_dir` in lightglue's eval script kept the trainer's
   `output_dir` green with the trainer's own read removed.

   THE FIRST DRAFT OF *THIS* GUARD REPEATED THAT MISTAKE THROUGH A SIDE DOOR.
   Its consumption test was a regex over raw lines, and one of its four
   alternatives -- `\b<field>\s*=[^=]`, meant to catch a kwarg at a
   construction site -- carried no receiver scoping at all. So
   `unrelated_object.save_best_only = 5`, or a bare local `save_best_only = 5`,
   counted as consumption of the config field while the docstring above
   promised it could not. The consumption test is therefore AST-based, and the
   routes are expressed as AST shapes rather than as text:

     - a SCOPED attribute READ (`ast.Attribute`, Load context, under the rule
       above). `self.X` counts only inside a method of the config class or of
       one of its bases: `self` in any other class is that class's instance;
     - NOT a keyword argument merely NAMED like the field (step 13.10.1): it
       used to count whatever its value, so `dashboard_draws=DASHBOARD_TARGET_DRAWS`
       and lightglue's `prepare_run_dir(config, output_dir=run_dir)` kept a
       field nothing read green. `Trainer(x=config.x)` still counts, through
       the read in its value;
     - a string CONSTANT equal to the field name in a LOOKUP position only:
       `getattr(<config>, "X")`, `<mapping>["X"]`, `<mapping>.get("X")`;
     - a read `<param>.<field>` inside a `src/` function the config is HANDED
       to (`prepare_run_dir(config)` reads `config.output_dir` in a module
       that never imports the class), one hop, and only on a branch that
       call reaches (see `_parameter_states`);
     - a key lookup of the field inside the named function of a loader of the
       saved `config.json`, declared per field in
       `READ_BACK_FROM_THE_SAVED_CONFIG` with its reader (file and function).

3. PLUMBING IS NOT CONSUMPTION (step 13.10 of plan 98b866fb, measured by
   mutation in step 10e; widened in step 13.10.1, measured by the mutants of
   review-iter-2-pass2). None of these changes what the program does with the
   field, so none of them counts:

     - the field's own CLI flag literal (`'--label-smoothing'`) and the keyword
       of the config's OWN constructor call, value included
       (`TrainingConfig(label_smoothing=args.label_smoothing)`);
     - a WRITE onto the config, the whole statement included:
       `config.X = getattr(args, "X", 0.0)`, `config.X += 1`,
       `setattr(config, "X", value)`. Re-adding the dead ResNet
       `label_smoothing` filled this way stayed green under 13.10;
     - the field's name as a dict key or value or a tuple/list/set element:
       the CLI-to-config maps of dit, bit_diffusion, colbert and
       qwen3_embeddings (`{"ema_decay": "ema_decay"}`) name every field, so
       they made every field of those configs green;
     - a range check in the config's own `__post_init__` (an `if` whose body
       only raises, a `raise`, an `assert`): dit's `ema_decay` stayed green
       through its check while the EMA ran at a constant decay;
     - a REPORT: anything inside `logger.<level>(...)`, `logging.<level>(...)`,
       `warnings.warn(...)`, `print(...)` or `json.dump(s)(...)`, the f-string,
       `str.format` and `%` forms inside them included. A log line kept
       `focal_gamma` green while the loss ran at a literal 2.0.

   `test_an_unscoped_assignment_is_not_consumption` and
   `test_plumbing_mutants_are_not_consumption` pin the closed holes.

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
    # of train.api) prove LITTLE here. Until step 13.10.1 of plan 98b866fb this guard counted any
    # keyword argument of the field's name and any dict key, so a mutant such as
    # `normalize="standardize"` kept a field green; both routes are gone, but the readers of
    # these groups are task-agnostic code that sees the config only as `StrictConfig`, so a green
    # row is still NOT evidence that the field is wired. What proves it is
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
    # The denoising task (step 8 of plan d483ffab): its config, its data config and its four own groups.
    ("train/api/tasks/image_denoising.py", "ImageDenoisingConfig"),
    ("train/api/denoising_args.py", "DenoisingData"),
    ("train/api/denoising_args.py", "DenoiseTransformArgs"),
    ("train/api/denoising_args.py", "DenoiseMetricArgs"),
    ("train/api/denoising_args.py", "DenoiseLossArgs"),
    ("train/api/denoising_args.py", "DenoiseOptimArgs"),
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
# exemption: each row names its reader as (a path under ``src/``, a function in it), and the field counts only while THAT
# function still looks the field's name up as a key (``raw["X"]`` / ``raw.get("X")``), so a row cannot outlive its reader;
# ``test_every_read_back_reader_still_reads_its_field`` fails on a row whose reader is gone or reads something else.
READ_BACK_FROM_THE_SAVED_CONFIG: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]] = {
    # The pixel-domain provenance stamp (D-005 of plan e56909cd): nothing branches on it in training; every checkpoint's
    # config.json carries it, and ``require_unit_domain_checkpoint`` (``raw.get("data_range", ...)``) refuses a checkpoint
    # whose config.json does not record "[0,1]", for ``--init-from``, ``DenoiserPrior`` and the bfunet eval scripts.
    ("train/bfunet/common.py", "BFUnetTrainingConfig"): {
        "data_range": ("dl_techniques/utils/denoiser_provenance.py", "require_unit_domain_checkpoint"),
    },
}

# A field read by a TASK-AGNOSTIC reader that is handed the config through more than one hop and never imports the class,
# so neither the importer scan nor the one-hop handed route reaches it. Same shape and same rule as the table above: each
# row names its reader as (a path under ``src/``, ``function`` or ``Class.method``), and the field counts only while THAT
# function still reads ``<config>.<field>`` (a scoped read, not a write, not a report).
READ_BY_A_READER_THAT_DOES_NOT_IMPORT_THE_CLASS: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]] = {
    # ``ImageClassificationTrainer(config).run()`` hands the config to the shared ``TaskTrainer`` (``train/api/trainer.py``),
    # which reads ``debug_args`` (``.smoke``: one epoch, no success threshold) and passes it on to
    # ``fit_callbacks.build_callbacks``, which reads ``logger_args`` (the CSV and TensorBoard callbacks). Both are typed
    # ``StrictConfig`` there, for every task.
    ("train/api/tasks/image_classification.py", "ImageClassificationConfig"): {
        "debug_args": ("train/api/trainer.py", "TaskTrainer.run"),
        "logger_args": ("train/api/fit_callbacks.py", "build_callbacks"),
        # ``TaskTrainer.run`` opens the run directory from ``out`` and resumes on ``resume_interrupted``; the run sequence
        # (``train/api/run_sequence.py``) fits for ``epochs``.
        "out": ("train/api/trainer.py", "TaskTrainer.run"),
        "resume_interrupted": ("train/api/trainer.py", "TaskTrainer.run"),
        "epochs": ("train/api/run_sequence.py", "train_or_restore"),
    },
    # The same five task-agnostic reads for the denoising task, plus ``model`` (``TaskTrainer.run`` parses it into the
    # family and the variant; the denoising trainer takes both from the context and reads no ``config.model`` itself
    # until its ``task_record`` is written).
    ("train/api/tasks/image_denoising.py", "ImageDenoisingConfig"): {
        "debug_args": ("train/api/trainer.py", "TaskTrainer.run"),
        "logger_args": ("train/api/fit_callbacks.py", "build_callbacks"),
        "out": ("train/api/trainer.py", "TaskTrainer.run"),
        "resume_interrupted": ("train/api/trainer.py", "TaskTrainer.run"),
        "model": ("train/api/trainer.py", "TaskTrainer.run"),
    },
}

# Receivers that plausibly ARE the config object. See trap 2 in the module
# docstring: a bare `<anything>.field` is NOT accepted. `<anything>_config` /
# `<anything>_cfg` / the class name itself are accepted too, via
# `_is_config_receiver`. Not `args`: the argparse namespace is not the config (step 13.10.1).
_BARE_RECEIVERS = frozenset({"self", "config", "cfg"})

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
    and only a read `<param>.<field>` inside that function's body counts (`_reached_reads`: not a write, not a report, and
    on a branch that call reaches). One hop: a read two calls down is not seen.
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
            for node in _reached_reads(func, call, bound):
                reads.setdefault(node.attr, []).append(
                    f"{callee_path.relative_to(REPO_ROOT)}:{node.lineno} (handed the config at "
                    f"{path.relative_to(REPO_ROOT)}:{call.lineno})"
                )
    return reads


def _parameter_states(func: ast.FunctionDef, call: ast.Call) -> Dict[str, str]:
    """Parameter -> "none" or "set" for what `call` binds it to, where that is known from the call alone.

    "none": passed the constant `None`, or not passed and defaulting to `None`. "set": passed anything else, or not passed
    and defaulting to a non-`None` constant. A parameter whose value is not known that way is absent, and a call with
    `*args` or `**kwargs` knows nothing (empty dict), so every branch of the callee stays reachable.
    """
    if any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords):
        return {}
    params = [a.arg for a in func.args.posonlyargs + func.args.args]
    passed = {params[i]: v for i, v in enumerate(call.args) if i < len(params)}
    passed.update({kw.arg: kw.value for kw in call.keywords})
    defaults = dict(zip(params[len(params) - len(func.args.defaults):], func.args.defaults))
    defaults.update({a.arg: d for a, d in zip(func.args.kwonlyargs, func.args.kw_defaults) if d is not None})
    states: Dict[str, str] = {}
    for name in params + [a.arg for a in func.args.kwonlyargs]:
        value = passed.get(name, defaults.get(name))
        if isinstance(value, ast.Constant) and value.value is None:
            states[name] = "none"
        elif name in passed or isinstance(value, ast.Constant):
            states[name] = "set"
    return states


def _test_value(test: ast.expr, states: Dict[str, str]) -> Optional[bool]:
    """The value of an `if` test over a parameter's None-ness (`p is None`, `p is not None`, `p`, `not p`), or None when
    the call does not decide it. A "set" parameter's truthiness is unknown, so a bare `if p:` is decided only for "none"."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _test_value(test.operand, states)
        return None if inner is None else not inner
    if isinstance(test, ast.Name):
        return False if states.get(test.id) == "none" else None
    if (
        isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Name)
        and test.left.id in states and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None and isinstance(test.ops[0], (ast.Is, ast.IsNot))
    ):
        is_none = states[test.left.id] == "none"
        return is_none if isinstance(test.ops[0], ast.Is) else not is_none
    return None


def _reached_reads(func: ast.FunctionDef, call: ast.Call, bound: Set[str]) -> List[ast.Attribute]:
    """Reads `<param>.<field>` (`param` in `bound`) in `func` that `call` reaches, minus the plumbing of trap 3.

    Do NOT credit a read on a branch the call skips (D-065): `prepare_run_dir(config, output_dir=run_dir)` never
    evaluates the `else Path(config.output_dir)` arm, and crediting it helped keep lightglue's `output_dir` green with its
    own read removed (review-iter-2-pass2, NOTE 4). Decided here: an `if` or conditional expression whose test is a
    parameter's None-ness (`_test_value`) given what the call passes (`_parameter_states`). NOT decided, so still
    credited: a test over anything else, and an early `return` above the read.
    """
    states = _parameter_states(func, call)
    parents = {id(child): node for node in ast.walk(func) for child in ast.iter_child_nodes(node)}

    def skipped(node: ast.AST) -> bool:
        child, parent = node, parents.get(id(node))
        while parent is not None:
            if isinstance(parent, (ast.If, ast.IfExp)) and child is not parent.test:
                value = _test_value(parent.test, states)
                body = parent.body if isinstance(parent.body, list) else [parent.body]
                in_body = any(child is stmt for stmt in body)
                if (value is False and in_body) or (value is True and not in_body):
                    return True
            child, parent = parent, parents.get(id(parent))
        return False

    def is_bound(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in bound

    candidates = [
        n for n in ast.walk(func)
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and is_bound(n.value) and not skipped(n)
    ]
    exempt: Dict[str, Set[int]] = {}
    for node in candidates:
        if node.attr not in exempt:
            exempt[node.attr] = _plumbing_ids(func, node.attr, is_bound, frozenset())
    return [n for n in candidates if id(n) not in exempt[n.attr]]


def _is_config_receiver(node: ast.expr, class_name: str) -> bool:
    """True if `node` plausibly evaluates to the config object.

    `self` / `config` / `cfg` / `<anything>_config` / `<anything>_cfg`
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


# Calls whose arguments are only REPORTED, never acted on (trap 3): a read inside one is not use.
_REPORT_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})
_REPORT_OWNERS = frozenset({"logger", "logging", "log", "warnings"})


def _is_report_call(call: ast.Call) -> bool:
    """True for `print(...)`, `json.dump(s)(...)` and `<logger>.<level>(...)` (`logger`, `logging`, `log`, `warnings`,
    `<x>_logger`, `self.logger`)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "print"
    if not isinstance(func, ast.Attribute):
        return False
    value = func.value
    owner = value.id if isinstance(value, ast.Name) else value.attr if isinstance(value, ast.Attribute) else ""
    if owner == "json":
        return func.attr in ("dump", "dumps")
    return func.attr in _REPORT_METHODS and (owner.lower().lstrip("_") in _REPORT_OWNERS or owner.lower().endswith("logger"))


def _only_raises(body: List[ast.stmt]) -> bool:
    """True if a block only raises (or reports and raises): the body of a range check."""
    return all(
        isinstance(s, ast.Raise) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _is_report_call(s.value))
        for s in body
    )


def _plumbing_ids(tree: ast.AST, field: str, is_receiver, own_classes: frozenset) -> Set[int]:
    """Ids of every node of `tree` that only FILLS, CHECKS or REPORTS `field` (trap 3), so `_consumes` never sees it.

    `is_receiver(node)` says whether an expression is the config; `own_classes` names the config class and its bases,
    whose `__post_init__` range checks are plumbing. Contract shared by the in-file route (`_Consumption.nodes`) and the
    handed route (`_reached_reads`): a node in the returned set is never counted as a read of `field`.
    """
    exempt: Set[int] = set()

    def cover(node: ast.AST) -> None:
        exempt.update(id(n) for n in ast.walk(node))

    def writes_field(target: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Attribute) and n.attr == field and isinstance(n.ctx, ast.Store) and is_receiver(n.value)
            for n in ast.walk(target)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(writes_field(t) for t in node.targets):
            cover(node)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and writes_field(node.target):
            cover(node)
        elif isinstance(node, ast.Call):
            if _is_report_call(node):
                cover(node)
            elif (
                isinstance(node.func, ast.Name) and node.func.id == "setattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant) and node.args[1].value == field and is_receiver(node.args[0])
            ):
                cover(node)
        elif isinstance(node, ast.ClassDef) and node.name in own_classes:
            for method in node.body:
                if not (isinstance(method, ast.FunctionDef) and method.name == "__post_init__"):
                    continue
                for inner in ast.walk(method):
                    if isinstance(inner, ast.Raise) or (isinstance(inner, ast.Assert) and _checks_only(inner.test, field)):
                        cover(inner)
                    elif isinstance(inner, ast.If) and _only_raises(inner.body) and _checks_only(inner.test, field):
                        cover(inner.test)
    return exempt


def _checks_only(test: ast.expr, field: str) -> bool:
    """True if every `self.<attr>` the test reads is `field`: a check OF the field. A test that also reads another field
    (mambalct's `not self.allow_custom_geometry and (self.template_size, ...) != expected`) is a GATE: the field decides
    whether another field's value is refused, which is use."""
    return all(
        n.attr == field for n in ast.walk(test)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    )


def _enclosing_classes(tree: ast.AST) -> Dict[int, Optional[str]]:
    """Node id -> name of the nearest enclosing `class`, or None at module level (iterative: deep expressions)."""
    out: Dict[int, Optional[str]] = {}
    stack: List[Tuple[ast.AST, Optional[str]]] = [(tree, None)]
    while stack:
        node, cls = stack.pop()
        inner = node.name if isinstance(node, ast.ClassDef) else cls
        for child in ast.iter_child_nodes(node):
            out[id(child)] = inner
            stack.append((child, inner))
    return out


def _typed_receiver_ids(tree: ast.AST, class_name: str) -> Set[int]:
    """Ids of the `Name` uses, inside a function, of a parameter ANNOTATED with the config class (`data: Union[
    FolderImageData, CsvImageData]` makes `data.train_dir` a read of `FolderImageData.train_dir`). The annotation is the
    type, so this receiver is the config whatever it is called; a string annotation counts when it names the class."""
    pattern = re.compile(rf"\b{re.escape(class_name)}\b")

    def names_class(annotation: Optional[ast.expr]) -> bool:
        return annotation is not None and any(
            (isinstance(n, ast.Name) and n.id == class_name)
            or (isinstance(n, ast.Attribute) and n.attr == class_name)
            or (isinstance(n, ast.Constant) and isinstance(n.value, str) and bool(pattern.search(n.value)))
            for n in ast.walk(annotation)
        )

    ids: Set[int] = set()
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
            typed = {a.arg for a in args if a.arg != "self" and names_class(a.annotation)}
            if typed:
                ids.update(id(n) for n in ast.walk(func) if isinstance(n, ast.Name) and n.id in typed)
    return ids


def _is_lookup_key(node: ast.Constant, parent: Optional[ast.AST], is_receiver) -> bool:
    """True if the string constant `node` is looked UP: `getattr(<config>, "X")`, `<m>["X"]` (Load), `<m>.get("X")`."""
    if isinstance(parent, ast.Call):
        func = parent.func
        if isinstance(func, ast.Name) and func.id == "getattr":
            return len(parent.args) >= 2 and parent.args[1] is node and is_receiver(parent.args[0])
        return isinstance(func, ast.Attribute) and func.attr == "get" and bool(parent.args) and parent.args[0] is node
    return isinstance(parent, ast.Subscript) and parent.slice is node and isinstance(parent.ctx, ast.Load)


class _Consumption:
    """AST shapes that plausibly CONSUME `field`.

    Two in-file routes -- scoped attribute READ (a parameter annotated with the
    config class is a receiver too), string constant in a lookup position
    (`getattr(<config>, "X")`, `<m>["X"]`, `<m>.get("X")`) -- so a field wired
    by either counts as live (the cross-file routes are
    `_handed_config_reads`, `_name_map_sites` and `_reader_sites`). Plumbing (trap 3,
    `_plumbing_ids`) never counts. See the module docstring for why this is an
    AST walk and not a regex over lines: the regex's kwarg alternative had no
    receiver scoping, so an unrelated `x.field = 5` counted.

    `own_classes` names the classes whose `self` IS the config: the config
    class and its bases (default: the class alone). `self.X` anywhere else is
    another object (review-iter-2-pass2, NOTE 4: `self.dashboard_draws` of a
    visualization callback in bfunet's base module kept a field nothing set
    green).

    Contract: `nodes(tree)` yields every consuming node of a parsed module;
    `search(snippet)` parses a source SNIPPET and returns True if it consumes
    the field (the probe used by the scoping tests below). A snippet that does
    not parse yields no consumption rather than raising.
    """

    def __init__(self, field: str, class_name: str, own_classes: Optional[frozenset] = None) -> None:
        self.field = field
        self.class_name = class_name
        self.own_classes = own_classes if own_classes is not None else frozenset({class_name})

    def nodes(self, tree: ast.AST) -> List[ast.AST]:
        enclosing = _enclosing_classes(tree)
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        typed = _typed_receiver_ids(tree, self.class_name)

        def is_receiver(node: ast.expr) -> bool:
            if isinstance(node, ast.Name) and node.id == "self":
                return enclosing.get(id(node)) in self.own_classes
            return id(node) in typed or _is_config_receiver(node, self.class_name)

        # DECISION plan-2026-11-03T221936-98b866fb/D-065: do NOT count the constructor keyword or its value again, and do NOT
        # put `tests/` back in the search: either one alone kept a field wired only to its own flag green.
        # A keyword of the config's OWN constructor call (``TrainingConfig(label_smoothing=args.label_smoothing)``) is plumbing
        # as well: it fills the field, and fills it from the same flag. The WHOLE keyword node is exempt, its value included,
        # so a value copied off another config (``TrainingConfig(x=base_config.x)``) does not count either. A keyword NAME
        # never counts (step 13.10.1); the read in the value of any other call's keyword does.
        # Writes, name maps, `__post_init__` checks and reports are plumbing too (step 13.10.1, `_plumbing_ids`).
        plumbing = _plumbing_ids(tree, self.field, is_receiver, self.own_classes)
        for call in ast.walk(tree):
            if isinstance(call, ast.Call):
                func = call.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                if name == self.class_name:
                    plumbing.update(id(n) for kw in call.keywords for n in ast.walk(kw))
        found: List[ast.AST] = []
        for node in ast.walk(tree):
            if id(node) in plumbing:
                continue
            if isinstance(node, ast.Attribute):
                hit = node.attr == self.field and isinstance(node.ctx, ast.Load) and is_receiver(node.value)
            elif isinstance(node, ast.Constant):
                # The field's own CLI flag (``--label-smoothing``) never equals the name; a dict key or value, a tuple or list
                # element (the CLI-to-config maps) is not a lookup, so neither counts.
                hit = node.value == self.field and _is_lookup_key(node, parents.get(id(node)), is_receiver)
            else:
                hit = False
            if hit:
                found.append(node)
        return found

    def search(self, snippet: str) -> bool:
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            return False
        return bool(self.nodes(tree))


def _consumption_pattern(field: str, class_name: str, own_classes: Optional[frozenset] = None) -> _Consumption:
    """Kept as the name every scoping test probes through."""
    return _Consumption(field, class_name, own_classes)


def _own_classes(rel: str, class_name: str) -> frozenset:
    """The config class and the names of its direct bases: the classes whose `self` IS the config."""
    for node in ast.walk(_parse(SRC / rel)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            bases = {
                b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))
            }
            return frozenset({class_name} | bases)
    return frozenset({class_name})


def _consumption_sites(field: str, class_name: str, files: List[Path], own_classes: frozenset) -> List[str]:
    consumption = _consumption_pattern(field, class_name, own_classes)
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


def _reader_function(reader: Tuple[str, str]) -> Optional[ast.FunctionDef]:
    """The function a reader row names: (path under ``src/``, ``name`` or ``Class.name``). None if the file or it is gone."""
    path = SRC / reader[0]
    tree = _parse(path) if path.is_file() else None
    owner, _, name = reader[1].rpartition(".")
    for node in ast.walk(tree) if tree is not None else ():
        if isinstance(node, ast.ClassDef) and owner and node.name == owner:
            return next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == name), None)
        if not owner and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _reader_sites(
    table: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]], rel: str, class_name: str, field: str
) -> List[str]:
    """Sites in the reader a table row names for `field` that still read it: a key lookup of its name for
    `READ_BACK_FROM_THE_SAVED_CONFIG` (``raw.get("X")``), a scoped read ``<config>.X`` for the other table. Empty when the
    row is absent, the reader is gone, or it no longer reads the field."""
    reader = table.get((rel, class_name), {}).get(field)
    func = _reader_function(reader) if reader else None
    if func is None:
        return []
    if table is READ_BACK_FROM_THE_SAVED_CONFIG:
        parents = {id(c): n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
        hits = [
            n for n in ast.walk(func)
            if isinstance(n, ast.Constant) and n.value == field and _is_lookup_key(n, parents.get(id(n)), lambda _: False)
        ]
    else:
        hits = _Consumption(field, class_name).nodes(func)
        hits = [n for n in hits if isinstance(n, ast.Attribute)]
    return [f"src/{reader[0]}:{n.lineno}  ({reader[1]} reads it)" for n in hits]


def _name_map_sites(field: str, files: List[Path]) -> List[str]:
    """Sites where a searched file HANDS its CLI-to-config map to a `src/` function that looks the field's value up BY NAME.

    ``config_values_from_args(parser, argv, CLI_TO_CONFIG, SMOKE_PRESET)`` (``train/common/args.py``) builds the config's
    keyword values from the map and reads ``values.get("smoke")`` to decide whether the preset applies, before the config
    exists: the ``smoke`` value is consumed there. Narrow on purpose: the map must be a module-level dict literal whose
    VALUES name the field (in the searched file, or imported by it from ``src/``, as colbert's trainers import
    ``CLI_TO_CONFIG`` from ``common.py``), it must be an argument of a call to a function imported from ``src/``, and that
    function must look the name up as a key (``_is_lookup_key``). Naming the field in the map alone is plumbing (trap 3).
    """

    def dict_names(tree: Optional[ast.Module]) -> Set[str]:
        return {
            target.id
            for node in (tree.body if tree is not None else [])
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict)
            and any(isinstance(v, ast.Constant) and v.value == field for v in node.value.values)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }

    sites: List[str] = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        imported = _imported_names(tree)
        maps = dict_names(tree) | {
            local for local, (module, name) in imported.items()
            if (origin := _module_file(module)) is not None and name in dict_names(_parse(origin))
        }
        for call in ast.walk(tree) if maps else ():
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in imported):
                continue
            if not any(isinstance(a, ast.Name) and a.id in maps for a in [*call.args, *(k.value for k in call.keywords)]):
                continue
            found = _function_def(*imported[call.func.id])
            if found is None:
                continue
            callee_path, func = found
            parents = {id(c): n for n in ast.walk(func) for c in ast.iter_child_nodes(n)}
            sites.extend(
                f"{callee_path.relative_to(REPO_ROOT)}:{n.lineno} (looks it up in the map handed at "
                f"{path.relative_to(REPO_ROOT)}:{call.lineno})"
                for n in ast.walk(func)
                if isinstance(n, ast.Constant) and n.value == field and _is_lookup_key(n, parents.get(id(n)), lambda _: False)
            )
    return sites


def _dead_fields(rel: str, class_name: str) -> Dict[str, List[str]]:
    files = _search_files(rel, class_name)
    fields = _declared_fields(SRC / rel, class_name)
    handed = _handed_config_reads(class_name, tuple(files))
    own = _own_classes(rel, class_name)
    sites: Dict[str, List[str]] = {}
    for f in fields:
        sites[f] = (
            _consumption_sites(f, class_name, files, own) + handed.get(f, [])
            + _reader_sites(READ_BACK_FROM_THE_SAVED_CONFIG, rel, class_name, f)
            + _reader_sites(READ_BY_A_READER_THAT_DOES_NOT_IMPORT_THE_CLASS, rel, class_name, f)
        )
        if not sites[f]:  # the costliest route, and only needed when nothing else reads the field
            sites[f] = _name_map_sites(f, files)
    return sites


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
    assert pattern.search("class FinetuneConfig:\n    def run_dir(self):\n        return self.save_dir")
    assert pattern.search("path = FinetuneConfig.save_dir")
    # `self` is the config only inside the config class (or a base): elsewhere it is another object (step 13.10.1).
    assert not pattern.search("class Callback:\n    def run_dir(self):\n        return self.save_dir")
    assert not pattern.search("self.save_dir")


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

    A write ONTO the config does NOT count either (step 13.10.1): it fills the
    field; something must still read it.
    """
    pattern = _consumption_pattern("save_best_only", "TrainingConfig")
    assert not pattern.search("unrelated_object.save_best_only = 5")
    assert not pattern.search("save_best_only = 5")
    assert not pattern.search("logger.info('nothing to do with --save-best-only here')")
    assert not pattern.search("config.save_best_only = args.save_best_only")
    # ...while the routes that ARE consumption still fire.
    # A keyword merely NAMED like the field is not a read of it (step 13.10.1); a read in its value is.
    assert not pattern.search("ModelCheckpoint(save_best_only=True)")
    assert pattern.search("getattr(cfg, 'save_best_only')")
    # ...and the plumbing that only fills the field does not (step 13.10): its own CLI flag, and its keyword in the config's
    # own constructor call.
    assert not pattern.search("parser.add_argument('--save-best-only')")
    assert not pattern.search("config = TrainingConfig(save_best_only=args.save_best_only)")
    assert pattern.search("trainer = Trainer(save_best_only=config.save_best_only)")


# The planted sources below are the mutants of review-iter-2-pass2 (WARNING 2, NOTE 4) and of step 13.10.1, each of which
# the guard of eff3da8ab counted as consumption. Each is asserted dead here, beside a control that keeps the route alive.

_PLANTED_STORE = """
config = TrainingConfig(epochs=args.epochs)
config.label_smoothing = getattr(args, "label_smoothing", 0.0)
logger.info(f"Config: {config.model_variant}")
"""

_PLANTED_DIT = """
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    ema_decay: float = DEFAULT_EMA_DECAY

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError(f"ema_decay must lie in [0, 1], got {self.ema_decay}")

CLI_TO_CONFIG = {"ema_decay": "ema_decay", "seed": "seed"}
NAME_LIST = ("ema_decay", "seed")

def build_parser():
    defaults = TrainingConfig()
    optim.add_argument("--ema-decay", type=float, default=defaults.ema_decay)

def train(config):
    ema = WeightEMACallback(decay=DEFAULT_EMA_DECAY)
"""


def test_plumbing_mutants_are_not_consumption() -> None:
    """Trap 3 as widened in step 13.10.1: a write, a name map, a `__post_init__` check and a report are not use.

    The first two planted sources are the two mutants the review named: ResNet's `label_smoothing` re-added and filled by
    `config.label_smoothing = getattr(args, ...)`, and dit with `WeightEMACallback(decay=DEFAULT_EMA_DECAY)` where it reads
    `config.ema_decay` (its `__post_init__` check and its CLI-to-config map kept it green).
    """
    smoothing = _consumption_pattern("label_smoothing", "TrainingConfig")
    assert not smoothing.search(_PLANTED_STORE)
    assert not smoothing.search("config.label_smoothing += 0.1")
    assert not smoothing.search("setattr(config, 'label_smoothing', 0.1)")
    assert not smoothing.search("config.label_smoothing: float = config.label_smoothing * 2")
    assert smoothing.search(_PLANTED_STORE + "loss = make_loss(config.label_smoothing)\n")

    ema = _consumption_pattern("ema_decay", "TrainingConfig")
    assert not ema.search(_PLANTED_DIT)
    assert ema.search(_PLANTED_DIT.replace("decay=DEFAULT_EMA_DECAY", "decay=config.ema_decay"))
    # The name map and the name list alone, and the lookups that ARE reads by name.
    assert not ema.search('CLI_TO_CONFIG = {"ema_decay": "ema_decay"}\nFIELDS = ["ema_decay"]\nNAMES = {"ema_decay"}')
    assert ema.search("decay = getattr(config, 'ema_decay')")
    assert ema.search("decay = saved['ema_decay']")
    assert ema.search("decay = saved.get('ema_decay', 0.9999)")
    assert not ema.search("decay = getattr(unrelated_object, 'ema_decay')")
    # A `__post_init__` check OF the field is plumbing; a derivation from it, and a GATE on another field's check, are not.
    derive = _PLANTED_DIT.replace("        if not 0.0", "        self.half_life = 1 / (1 - self.ema_decay)\n        if not 0.0")
    assert ema.search(derive)
    gate = _PLANTED_DIT.replace("not 0.0 <= self.ema_decay <= 1.0", "self.ema_decay and self.seed < 0")
    assert ema.search(gate)


def test_a_read_inside_a_report_is_not_consumption() -> None:
    """A log line, a print, a warning or a JSON dump only REPORTS the value (step 13.10.1).

    `clm_pretrain.create_clm_loss_fn` with `gamma=2.0` kept `focal_gamma` green through its log line.
    """
    gamma = _consumption_pattern("focal_gamma", "ClmPretrainConfig")
    for snippet in (
        'logger.info(f"Focal loss: gamma={config.focal_gamma}")',
        'logger.info("gamma=%s", config.focal_gamma)',
        'logger.warning("gamma=%s" % config.focal_gamma)',
        'logger.info("gamma={}".format(config.focal_gamma))',
        'self.logger.debug(config.focal_gamma)',
        'logging.info(config.focal_gamma)',
        'print(config.focal_gamma)',
        'warnings.warn(f"{config.focal_gamma}")',
        'json.dump({"gamma": config.focal_gamma}, handle)',
        'json.dump(asdict(config), handle)',
        'text = json.dumps(vars(config))',
    ):
        assert not gamma.search(snippet), snippet
    assert gamma.search('loss = FocalLoss(gamma=config.focal_gamma)\nlogger.info(f"{config.focal_gamma}")')
    assert gamma.search('name = "gamma_{}".format(config.focal_gamma)')
    assert gamma.search('path = f"{config.focal_gamma}/run"')


def test_self_is_the_config_only_inside_the_config_or_its_base() -> None:
    """The base-class route credits `self.X` only in a method of the config class or of one of its bases (NOTE 4).

    Planted from the review's mutant: `dashboard_draws` added to bfunet's ConvUNeXt `TrainingConfig` stayed green because
    `DenoisingVisualizationCallback`, in the base module `train/bfunet/common.py`, assigns and reads `self.dashboard_draws`.
    """
    own = frozenset({"TrainingConfig", "BFUnetTrainingConfig"})
    draws = _consumption_pattern("dashboard_draws", "TrainingConfig", own)
    callback = (
        "class DenoisingVisualizationCallback(keras.callbacks.Callback):\n"
        "    def __init__(self, draws):\n"
        "        self.dashboard_draws = draws\n"
        "    def on_epoch_end(self, epoch, logs=None):\n"
        "        for _ in range(self.dashboard_draws):\n"
        "            pass\n"
    )
    assert not draws.search(callback)
    assert draws.search("class BFUnetTrainingConfig:\n    def draws(self):\n        return self.dashboard_draws\n")
    assert draws.search(callback + "def train(config):\n    return config.dashboard_draws\n")
    # ...and the keyword that hands the callback its value is not a read of the field either: the review's mutant also
    # stayed green through `dashboard_draws=DASHBOARD_TARGET_DRAWS` and `prepare_run_dir(config, output_dir=run_dir)`.
    assert not draws.search(callback + "cb = DenoisingVisualizationCallback(dashboard_draws=DASHBOARD_TARGET_DRAWS)\n")
    out = _consumption_pattern("output_dir", "LightGlueTrainConfig")
    assert not out.search("prepare_run_dir(config, output_dir=run_dir)")
    # ...nor is the argparse namespace of another script (lightglue's eval_homography imports the class).
    assert not out.search("base_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / 'results'")
    assert out.search("base_dir = Path(config.output_dir) if config.output_dir else REPO_ROOT / 'results'")
    # A parameter ANNOTATED with the config class is the config, whatever its name.
    folder = _consumption_pattern("train_dir", "FolderImageData")
    assert folder.search("def _list_files(data: Union[FolderImageData, CsvImageData]):\n    return scan(data.train_dir)")
    assert not folder.search("def _list_files(data):\n    return scan(data.train_dir)")


_PLANTED_CALLEE = """
def prepare_run_dir(config, output_dir=None, config_filename="config.json"):
    run_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir) / config.experiment_name
    if output_dir is None:
        note = config.seed
    if not output_dir:
        other = config.dataset
    logger.info("run %s", config.model)
    config.written = 1
    save_config_json(config, str(run_dir), config_filename)
    return run_dir
"""


def test_a_handed_read_on_a_branch_the_call_skips_is_not_credited() -> None:
    """The handed route credits a read only on a branch the caller reaches (NOTE 4).

    Planted from `train/common/run_io.py`: `prepare_run_dir(config, output_dir=run_dir)` (lightglue) never evaluates
    `Path(config.output_dir)`, and the review's mutant that dropped lightglue's own read of `output_dir` stayed green.
    """
    func = ast.parse(_PLANTED_CALLEE).body[0]

    def reads(call_src: str) -> Set[str]:
        call = ast.parse(call_src).body[0].value
        return {n.attr for n in _reached_reads(func, call, {"config"})}

    every = {"output_dir", "experiment_name", "seed", "dataset"}
    assert reads("prepare_run_dir(config)") == every
    assert reads("prepare_run_dir(config, output_dir=None)") == every
    assert reads("prepare_run_dir(config, output_dir=run_dir)") == {"dataset"}
    assert reads("prepare_run_dir(config, run_dir)") == {"dataset"}
    # Nothing is known about a `**kwargs` call, so every branch stays reachable.
    assert reads("prepare_run_dir(config, **options)") == every
    # The log line and the write onto the config are never credited.
    assert "model" not in reads("prepare_run_dir(config)") and "written" not in reads("prepare_run_dir(config)")


@pytest.mark.parametrize(
    "table", [READ_BACK_FROM_THE_SAVED_CONFIG, READ_BY_A_READER_THAT_DOES_NOT_IMPORT_THE_CLASS],
    ids=["read_back", "task_agnostic_reader"],
)
def test_every_named_reader_still_reads_its_field(table: Dict[Tuple[str, str], Dict[str, Tuple[str, str]]]) -> None:
    """Each row of a reader table names a file AND a function, and that function still reads the field.

    A row whose reader was renamed, moved or stopped reading the field would otherwise keep the field green on the say-so
    of the table alone.
    """
    stale = {
        (rel, cls, field): reader
        for (rel, cls), rows in table.items()
        for field, reader in rows.items()
        if not _reader_sites(table, rel, cls, field)
    }
    assert stale == {}, f"reader rows whose function is gone or no longer reads the field: {stale}"


def test_a_reader_row_counts_only_while_its_function_reads_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader check is real: the right file with the wrong function, or a missing function, credits nothing."""
    key = ("train/bfunet/common.py", "BFUnetTrainingConfig")
    assert _reader_sites(READ_BACK_FROM_THE_SAVED_CONFIG, *key, "data_range")
    monkeypatch.setitem(
        READ_BACK_FROM_THE_SAVED_CONFIG, key, {"data_range": ("dl_techniques/utils/denoiser_provenance.py", "_read_config")}
    )
    assert _reader_sites(READ_BACK_FROM_THE_SAVED_CONFIG, *key, "data_range") == []
    monkeypatch.setitem(
        READ_BACK_FROM_THE_SAVED_CONFIG, key, {"data_range": ("dl_techniques/utils/denoiser_provenance.py", "gone")}
    )
    assert _reader_sites(READ_BACK_FROM_THE_SAVED_CONFIG, *key, "data_range") == []
