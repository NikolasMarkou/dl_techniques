"""Every ``src/train`` entry point that trains takes ``--seed`` and seeds through ``set_seeds``.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 12.6(a) (D-247). Step 12
found AlexNet and DarkIR seeding nothing (D-246's table: "none: the trainer seeds
nothing"), so two runs of the same command drew different weights, data order and
augmentation. The shared helper ``train.common.seed.set_seeds`` seeds Python ``random``,
NumPy and Keras (``keras.utils.set_random_seed``, which seeds TensorFlow too).

The census is an AST call graph, built per entry point:

- An entry point is a ``src/train`` module with a top-level ``if __name__ == "__main__":``.
- From that block, every function, class or module-level value it names is followed,
  through same-module names and imports of other ``train.*`` modules (a package
  ``__init__`` re-export is followed to its definition; a class brings all its methods).
  The library under ``dl_techniques`` is not followed: a model's own ``train_step`` is
  not the trainer's seeding.
- The entry point TRAINS when a reached function calls ``fit``, ``train_on_batch`` or
  ``apply_gradients``, or it is one of ``TRAINS_WITHOUT_A_TRAIN_CALL``.
- A trainer must reach a call of the shared ``set_seeds`` whose argument is not a
  literal (a literal ignores the flag), and must declare ``--seed`` whose default
  resolves to the shared ``train.common.seed.DEFAULT_SEED`` (a literal of equal value,
  the constant itself, or a ``seed`` field of the config class the default reads that
  is either), unless it runs several seeds (``MULTI_SEED``, ``--seeds``, one
  ``set_seeds`` per seed). The constant is read from ``seed.py`` by AST, so the census
  holds no second copy of its value.
- Every entry point that does not train is named in ``NOT_TRAINING`` with its reason,
  and the classification is exact both ways, so a new entry point fails until it is
  classified.

At 2026-10-21 (ed86f8c05) the census found 173 entry points, 131 of them trainers and 42
not, and 37 trainers broke the rule: 26 reached no ``set_seeds``, 8 seeded a literal 42
with no flag, BLT had no flag, and the two YOLOv12 trainers named it ``--random-seed``.
It then kept 17 older trainers' default 0 in a ``KEPT_DEFAULTS`` list; iteration 3 step 7
(plan-2026-10-14T062616-81938e81 D-289) moved them to ``DEFAULT_SEED`` and deleted the list.
"""

import ast
import functools
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
TRAIN_CALLS = ("fit", "train_on_batch", "apply_gradients")
SHARED_HELPER = ("train.common.seed", "set_seeds")
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
SEED_MODULE = ("train.common.seed", "DEFAULT_SEED")

#: Trainers whose update is not a Keras training call inside ``src/train``.
TRAINS_WITHOUT_A_TRAIN_CALL: Dict[str, str] = {
    "train.mothnet.train_mothnet": (
        "Hebbian readout learned by train_hebbian's own loop, never model.fit"),
    "train.mini_vec2vec.train_mini_vec2vec": (
        "learns its map through MiniVec2VecAligner.align (library: KMeans + Procrustes)"),
}

#: Entry points that do not train, each with its reason.
NOT_TRAINING: Dict[str, str] = {
    "train.bert.deploy": "loads a saved model and serves predictions",
    "train.bfunet.eval_empirical_mean_shift": (
        "scores the exact empirical Tweedie denoiser and an optional saved denoiser on "
        "seeded draws; trains nothing"),
    "train.bfunet.eval_per_pixel_uncertainty": "evaluates saved denoisers",
    "train.bfunet.eval_psnr_vs_noise": "evaluates saved denoisers",
    "train.common.compare_runs": "compares two finished runs' histories",
    "train.convnext.run_stochastic_comparison": (
        "launches the seeded ConvNeXt trainer per arm and passes its --seed"),
    "train.doc_res.infer_doc_res": "inference with a saved model",
    "train.doc_res.prepare_doc_res_data": "data preparation",
    "train.doc_scanner.prepare_doc_scanner_data": "data preparation",
    "train.doc_scanner.stage_uvdoc_samples": "data staging",
    "train.head_comparison.analyze_heads": "analyses finished runs",
    "train.hnet.infer_hnet": "generates from a saved model",
    "train.hnet.prepare_hnet_data": "data preparation",
    "train.lightglue.eval_homography": "evaluates a saved matcher",
    "train.logic.multiseed_sweep": "launches seeded trainers as subprocesses",
    "train.mothnet.multiseed_sweep": "launches the seeded MothNet trainer per seed",
    "train.nam.create_dataset": "data generation",
    "train.nam.eval_dfsa": "evaluates a saved DFSA",
    "train.nam.test_extreme": "evaluates a saved DFSA",
    "train.ntm.run_benchmark_suite": "evaluates a saved model",
    "train.omnipoint.generate_synthetic_camera_eval_set": "data generation",
    "train.rms_variants_train.norm_overhead_bench": (
        "times forward and backward passes; applies no update"),
    "train.rms_variants_train.report": "aggregates finished cells",
    "train.rms_variants_train.sweep": "launches seeded cells as subprocesses",
    "train.superpoint.homographic_adaptation": "labels images with a saved detector",
    "train.thera.eval_thera": "evaluates a saved model",
    "train.time_series.adaptive_ema.export": "exports a saved model",
    "train.time_series.deepar.export": "exports a saved model",
    "train.time_series.ets.export": "exports a saved model",
    "train.time_series.mdn.export": "exports a saved model",
    "train.time_series.nbeats.export": "exports a saved model",
    "train.time_series.prism.export": "exports a saved model",
    "train.time_series.tirex.export": "exports a saved model",
    "train.time_series.xlstm.export": "exports a saved model",
    "train.topolm.controls": (
        "scores finished TopoLM runs against the framing, length and label-shuffle "
        "controls; its random draws are seeded by its own --seed"),
    "train.topolm.figure_facts": (
        "regenerates figure_facts.json from finished TopoLM run directories (read "
        "only); it rebuilds the seeded initial weights of step 0 through "
        "train.topolm.common.set_seeds and takes the controls' draws from --seed"),
    "train.vae.run_sampler_comparison": (
        "launches the seeded VAE trainer per arm and passes its --seed"),
}

#: Entry points with a train call that are nonetheless not training (also in NOT_TRAINING).
NOT_TRAINING_DESPITE_A_TRAIN_CALL: Tuple[str, ...] = ()

#: Trainers that loop over ``--seeds`` and call ``set_seeds`` once per seed.
MULTI_SEED: Tuple[str, ...] = (
    "train.bfunet.variance_probe",
    "train.logic.train_e4_lowdata_mux",
    "train.logic.train_e4_monks",
    "train.mean_shift.exp1_mode_pooling",
    "train.sam3.baselines",
)


class _Module:
    """One parsed module: its top-level definitions and its import table."""

    def __init__(self, name: str, source: str, is_package: bool = False):
        self.name = name
        self.is_package = is_package
        self.tree = ast.parse(source)
        self.defs: Dict[str, List[ast.AST]] = {}
        self.imports: Dict[str, Tuple[str, Optional[str]]] = {}
        for node in self.tree.body:
            self._index(node)
        # A function-scope import (`from train.common import setup_gpu, set_seeds`
        # inside `main`, the bert/wikipedia and tabm pattern) binds the same name;
        # a module-level binding of that name wins.
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                base = self._absolute(node)
                for alias in node.names:
                    self.imports.setdefault(alias.asname or alias.name, (base, alias.name))

    def _index(self, node: ast.AST) -> None:
        if isinstance(node, DEFS):
            self.defs.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    self.defs.setdefault(target.id, []).append(node)
        elif isinstance(node, ast.ImportFrom):
            base = self._absolute(node)
            for alias in node.names:
                self.imports[alias.asname or alias.name] = (base, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    self.imports[alias.asname] = (alias.name, None)
        elif isinstance(node, (ast.If, ast.Try)):
            for field in ("body", "orelse", "finalbody"):
                for child in getattr(node, field, []):
                    self._index(child)
            for handler in getattr(node, "handlers", []):
                for child in handler.body:
                    self._index(child)

    def _absolute(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        package = self.name.split(".")
        if not self.is_package:
            package = package[:-1]
        package = package[: len(package) - (node.level - 1)]
        return ".".join(package + ([node.module] if node.module else []))

    def main_guards(self) -> List[ast.If]:
        return [n for n in self.tree.body if isinstance(n, ast.If)
                and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__"]


class _Census:
    """The call graph over ``train.*`` modules; ``extra`` adds in-memory modules."""

    def __init__(self, extra: Optional[Dict[str, str]] = None):
        self.extra = dict(extra or {})
        self._load = functools.lru_cache(maxsize=None)(self._load_uncached)

    def _load_uncached(self, name: str) -> Optional[_Module]:
        if name in self.extra:
            return _Module(name, self.extra[name])
        if not name or name.split(".")[0] != "train":
            return None
        base = SRC.joinpath(*name.split("."))
        if base.with_suffix(".py").is_file():
            return _Module(name, base.with_suffix(".py").read_text())
        if (base / "__init__.py").is_file():
            return _Module(name, (base / "__init__.py").read_text(), is_package=True)
        return None

    def module(self, name: str) -> Optional[_Module]:
        return self._load(name)

    def resolve(self, module_name: str, name: str, depth: int = 0):
        """``("defs", module, nodes)``, ``("module", name)`` or ``None``."""
        module = self.module(module_name)
        if module is None or depth > 8:
            return None
        if name in module.defs:
            return ("defs", module_name, module.defs[name])
        if name in module.imports:
            base, attr = module.imports[name]
            if attr is None:
                return ("module", base) if self.module(base) else None
            if self.module(f"{base}.{attr}"):
                return ("module", f"{base}.{attr}")
            return self.resolve(base, attr, depth + 1)
        return None

    def reached(self, entry: str) -> List[Tuple[str, ast.AST]]:
        """Every (module, definition) the ``__main__`` block of ``entry`` can reach."""
        work = [(entry, guard) for guard in self.module(entry).main_guards()]
        seen, out = set(), []
        while work:
            module_name, node = work.pop()
            if (module_name, id(node)) in seen:
                continue
            seen.add((module_name, id(node)))
            out.append((module_name, node))
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.Assign, ast.AnnAssign)):
                        work.append((module_name, item))
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        target = self.resolve(module_name, base.id)
                        if target and target[0] == "defs":
                            work.extend((target[1], d) for d in target[2])
                continue
            for child in _own_nodes(node):
                target = self._target(module_name, child)
                if target and target[0] == "defs":
                    work.extend((target[1], d) for d in target[2])
        return out

    def _target(self, module_name: str, node: ast.AST):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            return self.resolve(module_name, node.id)
        if isinstance(node, ast.Attribute):
            parts, current = [], node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                head = self.resolve(module_name, current.id)
                if head and head[0] == "module":
                    dotted = ".".join([head[1]] + parts[::-1])
                    owner, _, attr = dotted.rpartition(".")
                    return self.resolve(owner, attr)
        return None

    def is_shared_helper(self, module_name: str, call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Name):
            target = self.resolve(module_name, func.id)
            return bool(target and target[0] == "defs"
                        and (target[1], func.id) == SHARED_HELPER)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            head = self.resolve(module_name, func.value.id)
            if head and head[0] == "module":
                target = self.resolve(head[1], func.attr)
                return bool(target and target[0] == "defs"
                            and (target[1], func.attr) == SHARED_HELPER)
        return False

    def value_of(self, module_name: str, node: Optional[ast.AST], depth: int = 0):
        """The constant an expression in ``module_name`` names, or ``None``.

        A literal is its value; a bare name is resolved through ``resolve`` (same-module
        assignments first, then imports, so ``DEFAULT_SEED`` reached through the
        ``train.common`` re-export is ``seed.py``'s value and a local ``DEFAULT_SEED = 0``
        shadows it). Anything else (``DEFAULT_SEED + 1``, a call) is ``None``.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if not isinstance(node, ast.Name) or depth > 8:
            return None
        target = self.resolve(module_name, node.id)
        if not (target and target[0] == "defs"):
            return None
        assigned = target[2][-1]
        if not isinstance(assigned, (ast.Assign, ast.AnnAssign)):
            return None
        return self.value_of(target[1], assigned.value, depth + 1)

    def field_default(self, module_name: str, class_name: str, field: str):
        target = self.resolve(module_name, class_name)
        if not (target and target[0] == "defs"):
            return None
        for node in target[2]:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                        and item.target.id == field):
                    return self.value_of(target[1], item.value)
                if (isinstance(item, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == field
                                for t in item.targets)):
                    return self.value_of(target[1], item.value)
        return None

    def default_value(self, module_name: str, function: ast.AST, default: ast.AST):
        """The value a ``--seed`` default resolves to, or ``None`` if the census cannot."""
        if isinstance(default, (ast.Constant, ast.Name)):
            return self.value_of(module_name, default)
        if isinstance(default, ast.Attribute) and isinstance(default.value, ast.Name):
            owner = default.value.id
            value = self.field_default(module_name, owner, default.attr)
            if value is not None:
                return value
            for node in _own_nodes(function):
                if not (isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == owner
                                for t in node.targets)):
                    continue
                value = node.value
                if isinstance(value, ast.Name):  # `default = TrainingConfig`
                    return self.field_default(module_name, value.id, default.attr)
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    # `defaults = Config()`, or `defaults = stage_defaults(stage)`
                    # whose return annotation names the config class (a `replace`
                    # of it that sets no seed keeps the field's default).
                    target = self.resolve(module_name, value.func.id)
                    if not (target and target[0] == "defs"):
                        return None
                    made = target[2][0]
                    if isinstance(made, ast.ClassDef):
                        return self.field_default(target[1], made.name, default.attr)
                    if (isinstance(made, ast.FunctionDef)
                            and isinstance(made.returns, ast.Name)
                            and not any(isinstance(n, ast.keyword) and n.arg == default.attr
                                        for n in ast.walk(made))):
                        return self.field_default(target[1], made.returns.id, default.attr)
        return None

    def analyse(self, entry: str) -> "_Report":
        report = _Report(entry)
        for module_name, node in self.reached(entry):
            if isinstance(node, ast.ClassDef):
                continue
            for child in _own_nodes(node):
                if not isinstance(child, ast.Call):
                    continue
                name = _call_name(child)
                where = f"{module_name}:{child.lineno}"
                if name in TRAIN_CALLS:
                    report.train_calls.append(f"{where} {name}")
                if name == "set_seeds" and self.is_shared_helper(module_name, child):
                    literal = bool(child.args) and isinstance(child.args[0], ast.Constant)
                    report.seed_calls.append((where, literal))
                if name == "add_argument":
                    options = [a.value for a in child.args
                               if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                    for option in ("--seed", "--seeds"):
                        if option in options:
                            default = next((k.value for k in child.keywords
                                            if k.arg == "default"), None)
                            value = (None if default is None else
                                     self.default_value(module_name, node, default))
                            report.options.setdefault(option, []).append((where, value))
        return report


class _Report:
    def __init__(self, entry: str):
        self.entry = entry
        self.train_calls: List[str] = []
        self.seed_calls: List[Tuple[str, bool]] = []
        self.options: Dict[str, List[Tuple[str, object]]] = {}


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """``node``'s subtree without nested def/class bodies (they are reached by name)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        stack = list(node.body) + [d for d in node.args.defaults + node.args.kw_defaults if d]
    else:
        stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, DEFS):
                stack.append(child)


def _call_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _entry_points() -> List[str]:
    census = _Census()
    entries = []
    for path in sorted((SRC / "train").rglob("*.py")):
        name = ".".join(path.relative_to(SRC).with_suffix("").parts)
        if name.endswith(".__init__"):
            continue
        if census.module(name).main_guards():
            entries.append(name)
    return entries


def _violations(report: _Report) -> List[str]:
    """What a trainer's report lacks under the rule; empty when it complies."""
    problems = []
    if not report.seed_calls:
        problems.append("reaches no train.common.seed.set_seeds call")
    elif all(literal for _, literal in report.seed_calls):
        problems.append(f"seeds only a literal {report.seed_calls} (the flag is ignored)")
    if report.entry in MULTI_SEED:
        if "--seeds" not in report.options:
            problems.append("multi-seed trainer declares no --seeds")
        return problems
    declared = report.options.get("--seed")
    if not declared:
        problems.append("declares no --seed")
        return problems
    expected = SEED_DEFAULT
    values = {value for _, value in declared}
    if values != {expected}:
        problems.append(f"--seed default {sorted(map(str, values))} at {declared}, "
                        f"expected {expected}")
    return problems


#: The house default, read from ``train.common.seed`` by AST (``None`` if it is missing).
SEED_DEFAULT = _Census().value_of(
    SEED_MODULE[0], ast.Name(id=SEED_MODULE[1], ctx=ast.Load()))


@pytest.fixture(scope="module")
def reports() -> Dict[str, _Report]:
    census = _Census()
    return {entry: census.analyse(entry) for entry in _entry_points()}


def _trainers(reports: Dict[str, _Report]) -> List[str]:
    return sorted(e for e, r in reports.items()
                  if (r.train_calls or e in TRAINS_WITHOUT_A_TRAIN_CALL)
                  and e not in NOT_TRAINING)


def test_every_entry_point_is_classified(reports):
    entries = set(reports)
    detected = {e for e, r in reports.items() if r.train_calls}
    unclassified = sorted(entries - detected - set(TRAINS_WITHOUT_A_TRAIN_CALL)
                          - set(NOT_TRAINING))
    assert not unclassified, (
        f"{len(unclassified)} entry points neither call a training step nor are listed: "
        f"{unclassified}. Seed a trainer, or add a non-trainer to NOT_TRAINING with its reason.")
    stale = sorted((set(NOT_TRAINING) | set(TRAINS_WITHOUT_A_TRAIN_CALL)
                    | set(MULTI_SEED)) - entries)
    assert not stale, f"listed modules that are no longer entry points: {stale}"
    assert sorted(detected & set(NOT_TRAINING)) == sorted(NOT_TRAINING_DESPITE_A_TRAIN_CALL), (
        "an entry point listed as not training calls a training step: "
        f"{sorted(detected & set(NOT_TRAINING))}")
    assert not (detected & set(TRAINS_WITHOUT_A_TRAIN_CALL)), (
        "TRAINS_WITHOUT_A_TRAIN_CALL lists a module the census now detects by itself: "
        f"{sorted(detected & set(TRAINS_WITHOUT_A_TRAIN_CALL))}")


def test_every_trainer_seeds_through_the_shared_helper_from_its_seed_flag(reports):
    offenders = {e: _violations(reports[e]) for e in _trainers(reports)}
    offenders = {e: p for e, p in offenders.items() if p}
    assert not offenders, (
        f"{len(offenders)} trainers break the seed rule (D-247):\n"
        + "\n".join(f"  {e}: {'; '.join(p)}" for e, p in sorted(offenders.items())))


def test_the_census_sees_trainers_seeds_and_defaults(reports):
    """Positive controls on the real tree: each detector fires where it must."""
    trainers = _trainers(reports)
    assert len(trainers) >= 120, len(trainers)
    alexnet = reports["train.alexnet.train_alexnet"]
    assert alexnet.train_calls
    # Reached through an import: the ConvNeXt entry seeds in train.convnext.common, and
    # its --seed default is a dataclass field read through `defaults = Config()`.
    convnext = reports["train.convnext.train_convnext_v1"]
    assert any(where.startswith("train.convnext.common:") for where, _ in convnext.seed_calls)
    assert {value for _, value in convnext.options["--seed"]} == {42}
    # A flag default spelled `DEFAULT_SEED` (hkan's) resolves to the shared constant.
    assert {value for _, value in reports["train.hkan.train_hkan"].options["--seed"]} == {
        SEED_DEFAULT}
    # A literal-only seed is caught.
    synthetic = _Census(extra={"train._seed_probe": (
        "from train.common import set_seeds\n"
        "def main():\n    model.fit()\n    set_seeds(42)\n"
        "if __name__ == '__main__':\n    main()\n")}).analyse("train._seed_probe")
    assert synthetic.train_calls and synthetic.seed_calls == [("train._seed_probe:4", True)]
    assert _violations(synthetic)


def test_a_trainer_without_a_seed_fails_the_rule():
    """Negative control: an unseeded trainer and one seeding through a private helper fail;
    the compliant shape passes."""
    sources = {
        "train._unseeded": (
            "import argparse\n"
            "def main():\n    p = argparse.ArgumentParser()\n    model.fit()\n"
            "if __name__ == '__main__':\n    main()\n"),
        "train._private": (
            "import argparse\nimport keras\n"
            "def set_seeds(s):\n    keras.utils.set_random_seed(s)\n"
            "def main():\n    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--seed', type=int, default=42)\n"
            "    a = p.parse_args()\n    set_seeds(a.seed)\n    model.fit()\n"
            "if __name__ == '__main__':\n    main()\n"),
        "train._compliant": (
            "import argparse\nfrom train.common.seed import set_seeds\n"
            "def main():\n    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--seed', type=int, default=42)\n"
            "    a = p.parse_args()\n    set_seeds(a.seed)\n    model.fit()\n"
            "if __name__ == '__main__':\n    main()\n"),
    }
    census = _Census(extra=sources)
    assert _violations(census.analyse("train._unseeded"))
    assert _violations(census.analyse("train._private"))
    assert _violations(census.analyse("train._compliant")) == []


def test_the_shared_default_seed_is_read_from_seed_py():
    """``SEED_DEFAULT`` is ``train.common.seed.DEFAULT_SEED`` read by AST, the one home of
    the house value (D-247); a missing constant fails here by name. Its value is not
    restated: the trainers' literal ``default=42`` sites are held equal to it."""
    assert isinstance(SEED_DEFAULT, int) and not isinstance(SEED_DEFAULT, bool), (
        f"train.common.seed.DEFAULT_SEED is missing or not an int literal: {SEED_DEFAULT!r}")


def _seed_flag_probe(prelude: str, default: str) -> str:
    return (f"import argparse\n{prelude}"
            "def main():\n    p = argparse.ArgumentParser()\n"
            f"    p.add_argument('--seed', type=int, default={default})\n"
            "    a = p.parse_args()\n    set_seeds(a.seed)\n    model.fit()\n"
            "if __name__ == '__main__':\n    main()\n")


def test_a_seed_default_named_by_the_constant_resolves_to_it():
    """Controls for the ``ast.Name`` branch: the constant reached through the
    ``train.common`` re-export (in a flag and in a config field) passes; a local
    ``DEFAULT_SEED = 0`` shadow and ``DEFAULT_SEED + 1`` fail."""
    sources = {
        "train._reexport": _seed_flag_probe(
            "from train.common import DEFAULT_SEED, set_seeds\n", "DEFAULT_SEED"),
        "train._field": (
            "import argparse\nfrom dataclasses import dataclass\n"
            "from train.common import DEFAULT_SEED, set_seeds\n"
            "@dataclass\nclass Config:\n    seed: int = DEFAULT_SEED\n"
            "def main():\n    defaults = Config()\n    p = argparse.ArgumentParser()\n"
            "    p.add_argument('--seed', type=int, default=defaults.seed)\n"
            "    a = p.parse_args()\n    set_seeds(a.seed)\n    model.fit()\n"
            "if __name__ == '__main__':\n    main()\n"),
        "train._shadow": _seed_flag_probe(
            "from train.common import DEFAULT_SEED, set_seeds\nDEFAULT_SEED = 0\n",
            "DEFAULT_SEED"),
        "train._offset": _seed_flag_probe(
            "from train.common import DEFAULT_SEED, set_seeds\n", "DEFAULT_SEED + 1"),
    }
    census = _Census(extra=sources)
    assert _violations(census.analyse("train._reexport")) == []
    assert _violations(census.analyse("train._field")) == []
    shadow = census.analyse("train._shadow")
    assert {value for _, value in shadow.options["--seed"]} == {0}
    assert _violations(shadow)
    offset = census.analyse("train._offset")
    assert {value for _, value in offset.options["--seed"]} == {None}
    assert _violations(offset)
