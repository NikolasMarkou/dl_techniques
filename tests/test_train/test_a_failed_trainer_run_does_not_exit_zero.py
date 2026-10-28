"""A trainer whose run failed must not exit 0.

Guard for plan-2026-10-14T062616-81938e81 step 13 (D-095). Measured at
561e02165:

- ``train/darkir/train_darkir.py::main`` caught every ``Exception`` from
  ``train_model``, logged it, and returned, so a crashed run exited 0.
- ``train/cliffordnet/train_downsampling_techniques.py::train_all_variants``
  logged a variant whose build or training failed (``train_one_variant``
  returns ``None``, or raises) and went on to the next, and returned normally
  even when every variant had failed.

The sweep still trains every variant it can and writes the comparison of the
ones that trained; it raises at the end, naming the failed variants.

Step 13.5 (D-102) extends the guard to the end of a run: a failed final save,
history write or summary write raises instead of being logged (seven trainers,
below).
"""

import os
import sys

import numpy as np
import pytest

from tests.classification_fakes import use_raw_from_unit


def _tiny_cifar(*_args, **_kwargs):
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)
    return (x, y), (x, y), (32, 32, 3), 100


def test_darkir_main_propagates_a_training_error(monkeypatch):
    from train.darkir import train_darkir as module

    def crash(_args):
        raise RuntimeError("the run crashed")

    monkeypatch.setattr(module, "train_model", crash)
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(RuntimeError, match="the run crashed"):
        module.main()


def _sweep(monkeypatch, tmp_path, outcome):
    from train.cliffordnet import train_downsampling_techniques as module

    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_train_one_variant(variant_name, args, run_root, *rest):
        calls.append(variant_name)
        return outcome(variant_name, run_root)

    use_raw_from_unit(monkeypatch, _tiny_cifar)
    monkeypatch.setattr(module, "train_one_variant", fake_train_one_variant)
    args = module._parse_args(["--output-dir", str(tmp_path / "out")])
    return module, args, calls


def _row(variant_name, run_root):
    return {
        "variant": variant_name, "parameters": 1,
        "best_val_accuracy": 0.5, "final_val_accuracy": 0.5,
        "best_val_top5_accuracy": 0.5, "wall_seconds": 0.0,
        "epochs_trained": 1, "results_dir": run_root,
    }


def test_a_sweep_where_a_variant_returns_none_raises(monkeypatch, tmp_path):
    def outcome(name, run_root):
        return None if name == "V2_3stage_avgpool" else _row(name, run_root)

    module, args, calls = _sweep(monkeypatch, tmp_path, outcome)
    with pytest.raises(RuntimeError, match="V2_3stage_avgpool"):
        module.train_all_variants(args)
    # Every variant was still attempted.
    assert calls == list(module.VARIANTS)


def test_a_sweep_where_a_variant_raises_raises(monkeypatch, tmp_path):
    def outcome(name, run_root):
        if name == "V3_3stage_patch_merging":
            raise ValueError("boom")
        return _row(name, run_root)

    module, args, calls = _sweep(monkeypatch, tmp_path, outcome)
    with pytest.raises(RuntimeError, match="V3_3stage_patch_merging"):
        module.train_all_variants(args)
    assert calls == list(module.VARIANTS)


def test_a_sweep_where_every_variant_trains_returns(monkeypatch, tmp_path):
    module, args, calls = _sweep(monkeypatch, tmp_path, _row)
    module.train_all_variants(args)
    assert calls == list(module.VARIANTS)


# ---------------------------------------------------------------------------
# A failed final save, history or summary write fails the run (step 13.5, D-102)
# ---------------------------------------------------------------------------
#
# Measured at ffa4c2870: ``bit_diffusion``, ``sam``, ``sam2``, ``sam3`` and
# ``dit`` caught every ``Exception`` from the final ``model.save`` and logged
# it; ``bit_diffusion`` and ``sam3`` did the same for their history JSON;
# ``depth_anything`` and ``omnipoint`` logged a failed save (and Depth
# Anything a failed summary) as a warning. Each run then exited 0 with the
# artefact missing. Each case below runs the trainer's REAL ``train_*`` body
# with data, model and fit stubbed, so the only real code is the trainer's own
# control flow after ``fit``.

import builtins
import collections
import types

import keras


class _DiskFull(OSError):
    """The error every failing write in this section raises."""


class _StubModel:
    """Model stand-in: ``fit`` returns a two-epoch history, ``save`` may fail."""

    def __init__(self, fail_save):
        self.fail_save = fail_save
        self.saved = []

    def compile(self, *args, **kwargs):
        return None

    def summary(self, *args, **kwargs):
        return None

    def count_params(self):
        return 1

    def fit(self, *args, **kwargs):
        history = collections.defaultdict(lambda: [1.0, 0.5])
        history.update({"loss": [1.0, 0.5], "val_loss": [1.0, 0.5]})
        return types.SimpleNamespace(history=history)

    def save(self, path, *args, **kwargs):
        if self.fail_save:
            raise _DiskFull(f"disk full writing {path}")
        self.saved.append(str(path))


def _failing_open(name):
    """``open`` that raises for one file name and is the builtin otherwise."""

    def fake_open(file, *args, **kwargs):
        if str(file).endswith(name):
            raise _DiskFull(f"disk full writing {file}")
        return builtins.open(file, *args, **kwargs)

    return fake_open


class _Sized:
    def __init__(self, n=2):
        self.n = n

    def __len__(self):
        return self.n


def _bit_diffusion(monkeypatch, tmp_path, model):
    from train.bit_diffusion import train_bit_diffusion as module

    monkeypatch.setattr(module, "build_sde", lambda config: None)
    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets", lambda config, sde: (None, None))
    monkeypatch.setattr(module, "build_optimizer", lambda *a, **k: None)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_bit_diffusion(config)


def _sam(monkeypatch, tmp_path, model):
    from train.sam import train_sam as module

    monkeypatch.setattr(module, "create_dataset", lambda config: (None, None))
    monkeypatch.setattr(module, "create_training_model", lambda config: model)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam(config)


def _sam2(monkeypatch, tmp_path, model):
    from train.sam2 import train_sam2 as module

    monkeypatch.setattr(module, "create_dataset", lambda config: (None, None))
    monkeypatch.setattr(module, "create_sam2_training_model", lambda config: model)
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam2(config)


def _sam3(monkeypatch, tmp_path, model):
    from train.sam3 import train_sam3 as module

    monkeypatch.setattr(module, "create_training_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets",
                        lambda config, m, **kwargs: (None, None))
    monkeypatch.setattr(module, "build_callbacks", lambda *a: [])
    monkeypatch.setattr(module, "evaluate_sam3",
                        lambda m, ds: collections.defaultdict(lambda: 1.0))
    config = module.config_from_argv(["--smoke", "--output-dir", str(tmp_path)])
    return module, lambda: module.train_sam3(config)


def _dit(monkeypatch, tmp_path, model):
    from train.dit import train_dit as module

    class _StubEMA(keras.callbacks.Callback):
        updates, decay = 0, 0.9999

        def __init__(self, decay):
            super().__init__()

        def shadow_values(self):
            return []

    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_datasets", lambda config: (None, None))
    monkeypatch.setattr(module, "build_dit_optimizer", lambda *a, **k: None)
    monkeypatch.setattr(module, "DDPMHybridLoss", lambda **kwargs: None)
    monkeypatch.setattr(module, "WeightEMACallback", _StubEMA)
    config = module.config_from_argv(["--output-dir", str(tmp_path)])
    return module, lambda: module.train_dit(config)


def _depth_anything(monkeypatch, tmp_path, model):
    from train.depth_anything import train_depth_anything as module

    paths = [str(tmp_path / f"p{i}") for i in range(20)]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(module, "split_megadepth_pairs",
                        lambda *a, **k: (paths[:18], paths[:18], paths[18:], paths[18:]))
    monkeypatch.setattr(module, "MegaDepthDataset", lambda *a, **k: _Sized())
    monkeypatch.setattr(module, "create_model", lambda config: model)
    monkeypatch.setattr(module, "create_callbacks",
                        lambda *a, **k: ([], str(run_dir)))
    config = module.DepthAnythingTrainingConfig(
        output_dir=str(tmp_path), experiment_name="run", epochs=2,
    )
    return module, lambda: module.train_depth_anything(config)


def _omnipoint(monkeypatch, tmp_path, model):
    from train.omnipoint import train_omnipoint as module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(module, "build_datasets",
                        lambda config: (_Sized(), _Sized()))
    monkeypatch.setattr(module, "create_model", lambda config: (model, model))
    monkeypatch.setattr(module, "create_callbacks",
                        lambda *a, **k: ([], str(run_dir)))
    config = module._config_from_args(module.parse_arguments(
        ["--smoke", "--output-dir", str(tmp_path)]))
    config.show_plots = False
    return module, lambda: module.train_omnipoint(config)


_TRAINERS = {
    "bit_diffusion": _bit_diffusion,
    "sam": _sam,
    "sam2": _sam2,
    "sam3": _sam3,
    "dit": _dit,
    "depth_anything": _depth_anything,
    "omnipoint": _omnipoint,
}


@pytest.mark.parametrize("trainer", sorted(_TRAINERS))
def test_a_run_whose_final_save_fails_raises(monkeypatch, tmp_path, trainer):
    model = _StubModel(fail_save=True)
    _module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    with pytest.raises(_DiskFull, match="disk full"):
        run()


@pytest.mark.parametrize("trainer", sorted(_TRAINERS))
def test_a_run_whose_writes_succeed_returns_and_saves(
        monkeypatch, tmp_path, trainer):
    # The stubs are not what makes the failing case raise.
    model = _StubModel(fail_save=False)
    _module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    run()
    assert len(model.saved) == 1, model.saved


@pytest.mark.parametrize("trainer, artefact", [
    ("bit_diffusion", "training_history.json"),
    ("sam3", "training_history.json"),
    ("depth_anything", "training_summary.txt"),
])
def test_a_run_whose_history_or_summary_write_fails_raises(
        monkeypatch, tmp_path, trainer, artefact):
    model = _StubModel(fail_save=False)
    module, run = _TRAINERS[trainer](monkeypatch, tmp_path, model)
    monkeypatch.setattr(module, "open", _failing_open(artefact), raising=False)
    with pytest.raises(_DiskFull, match=artefact):
        run()
    # The model is saved before the reporting write, so a failed report never
    # costs the weights.
    assert len(model.saved) == 1, model.saved


# ---------------------------------------------------------------------------
# The downsampling sweep's per-variant JSON artefacts (step 13.7 (b), D-117)
# ---------------------------------------------------------------------------
#
# Measured at b50fd4e80: ``train_one_variant`` wrapped its ``config.json`` and
# ``training_history.json`` writes in ``try/except Exception`` and logged a
# WARNING, so the variant was recorded as trained with its record missing and
# the sweep exited 0. The shared ``save_training_history_json`` returned ``None``
# on failure until D-174 and the trainer checked it; it now raises itself, so the
# history case below fails the real writer's ``open``.


class _PredictingStubModel(_StubModel):
    def predict(self, x, *args, **kwargs):
        return np.zeros((len(x), 100), dtype=np.float32)


def _one_variant(monkeypatch, tmp_path, model):
    from train.cliffordnet import train_downsampling_techniques as module

    monkeypatch.setattr(module, "build_variant", lambda *a, **k: model)
    monkeypatch.setattr(module, "_smoke_forward", lambda *a, **k: None)
    monkeypatch.setattr(module, "build_train_dataset", lambda *a, **k: None)
    monkeypatch.setattr(module, "build_eval_dataset", lambda *a, **k: None)
    monkeypatch.setattr(module, "validate_model_loading", lambda *a, **k: None)
    args = module._parse_args(["--output-dir", str(tmp_path)])
    (x, y), (x_test, y_test), input_shape, num_classes = _tiny_cifar()

    def run():
        return module.train_one_variant(
            "V2_3stage_avgpool", args, str(tmp_path), x, y, x_test, y_test,
            num_classes, input_shape)

    return module, run


def test_a_variant_whose_config_json_write_fails_raises(monkeypatch, tmp_path):
    model = _PredictingStubModel(fail_save=False)
    module, run = _one_variant(monkeypatch, tmp_path, model)
    monkeypatch.setattr(module, "open", _failing_open("config.json"), raising=False)
    with pytest.raises(_DiskFull, match="config.json"):
        run()
    # The weights are saved before the record, so a failed record never costs them.
    assert len(model.saved) == 1, model.saved


def test_a_variant_whose_history_write_fails_raises(monkeypatch, tmp_path):
    model = _PredictingStubModel(fail_save=False)
    module, run = _one_variant(monkeypatch, tmp_path, model)
    # The real shared writer, its file write failing (D-174: it raises).
    from train.common import run_io

    monkeypatch.setattr(run_io, "open", _failing_open("training_history.json"), raising=False)
    with pytest.raises(_DiskFull, match="training_history"):
        run()
    assert len(model.saved) == 1, model.saved


def test_a_variant_whose_writes_succeed_returns_its_row(monkeypatch, tmp_path):
    model = _PredictingStubModel(fail_save=False)
    _module, run = _one_variant(monkeypatch, tmp_path, model)
    row = run()
    assert row is not None and row["variant"] == "V2_3stage_avgpool"
    results_dir = row["results_dir"]
    for name in ("config.json", "training_history.json", "training_summary.txt"):
        assert os.path.isfile(os.path.join(results_dir, name)), name
    assert len(model.saved) == 1, model.saved


# ---------------------------------------------------------------------------
# Ctrl-C (step 14 Part 0, D-118)
# ---------------------------------------------------------------------------
#
# Measured at 7bd01f8c5: ``train_downsampling_techniques.main`` caught
# ``KeyboardInterrupt``, logged "Interrupted by user." and returned, so a run
# stopped with Ctrl-C exited 0, the code of a run that finished. Seven more
# entry points in ``src/train`` had the same handler (the census below).

import ast  # noqa: E402
import subprocess  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_an_interrupted_downsampling_run_propagates_the_interrupt(monkeypatch):
    from train.cliffordnet import train_downsampling_techniques as module

    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "setup_gpu", lambda **_kwargs: None)
    monkeypatch.setattr(module, "train_all_variants", interrupted)
    monkeypatch.setattr(sys, "argv", ["train_downsampling_techniques"])
    with pytest.raises(KeyboardInterrupt):
        module.main()


def test_an_interrupted_downsampling_process_exits_non_zero():
    """The process-level claim: the exit code a shell or a launcher sees."""
    script = (
        "import sys\n"
        "import train.cliffordnet.train_downsampling_techniques as m\n"
        "m.setup_gpu = lambda **_kwargs: None\n"
        "def interrupted(_args):\n"
        "    raise KeyboardInterrupt\n"
        "m.train_all_variants = interrupted\n"
        "sys.argv = ['train_downsampling_techniques']\n"
        "m.main()\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    done = subprocess.run([sys.executable, "-c", script], env=env,
                          cwd=str(_SRC), capture_output=True, text=True,
                          timeout=300)
    assert done.returncode != 0, done.stderr[-2000:]


def _swallowing_interrupt_handlers(source):
    """Line numbers of ``except KeyboardInterrupt`` handlers that exit 0.

    A handler passes when its body re-raises (any ``raise``, which includes
    ``raise SystemExit(...)``), calls ``sys.exit``/``exit`` with a non-zero
    integer literal, or returns a non-zero integer literal (an entry point
    whose ``__main__`` exits with ``main()``'s value). Anything else lets the
    process end with code 0.
    """
    def catches_interrupt(handler):
        names = (handler.type.elts if isinstance(handler.type, ast.Tuple)
                 else [handler.type])
        return any(isinstance(name, ast.Name) and name.id == "KeyboardInterrupt"
                   for name in names)

    def non_zero_literal(node):
        return (isinstance(node, ast.Constant) and isinstance(node.value, int)
                and not isinstance(node.value, bool) and node.value != 0)

    def exits_non_zero(handler):
        for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
            if isinstance(node, ast.Raise):
                return True
            if isinstance(node, ast.Return) and non_zero_literal(node.value):
                return True
            if isinstance(node, ast.Call):
                func = node.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else getattr(func, "id", None))
                if name == "exit" and node.args and non_zero_literal(node.args[0]):
                    return True
        return False

    with warnings.catch_warnings():  # invalid escapes in scanned sources
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        tree = ast.parse(source)
    return sorted(
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.type is not None
        and catches_interrupt(node) and not exits_non_zero(node))


_HANDLER_FORMS = '''
try:
    run()
except KeyboardInterrupt:
    logger.info("interrupted")
try:
    run()
except KeyboardInterrupt:
    return None
try:
    run()
except (KeyboardInterrupt, SystemExit):
    sys.exit(0)
try:
    run()
except KeyboardInterrupt:
    raise
try:
    run()
except KeyboardInterrupt:
    raise SystemExit(130)
try:
    run()
except KeyboardInterrupt:
    return 130
try:
    run()
except KeyboardInterrupt:
    sys.exit(1)
try:
    run()
except Exception:
    pass
'''


def test_the_interactive_prompt_exemption_is_still_a_prompt():
    """The one exemption must still be what justifies it: Ctrl-C ends an
    ``input()`` loop. If the file stops being a prompt, it loses the waiver."""
    for relative in _INTERACTIVE_PROMPTS:
        source = (_SRC / relative).read_text(encoding="utf-8")
        assert "input(" in source and _swallowing_interrupt_handlers(source), relative


def test_the_interrupt_census_tells_a_swallow_from_an_exit():
    """Positive control: the three exit-0 handlers are found, nothing else."""
    lines = _HANDLER_FORMS.splitlines()
    swallowing = [i for i, line in enumerate(lines, start=1)
                  if line.startswith("except") and "Exception" not in line][:3]
    assert _swallowing_interrupt_handlers(_HANDLER_FORMS) == swallowing


# Not a run: an interactive prompt whose documented way out is Ctrl-C (or
# "quit"); leaving it is the expected end, so exit code 0 is right there.
_INTERACTIVE_PROMPTS = {"train/bert/deploy.py"}


def test_no_entry_point_turns_ctrl_c_into_exit_0():
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        relative = path.relative_to(_SRC).as_posix()
        if relative in _INTERACTIVE_PROMPTS:
            continue
        for line in _swallowing_interrupt_handlers(path.read_text(encoding="utf-8")):
            offenders.append(f"{relative}:{line}")
    assert offenders == [], (
        "these handlers turn Ctrl-C into exit code 0; let the interrupt "
        "propagate (or exit non-zero):\n" + "\n".join(offenders))


# ---------------------------------------------------------------------------
# Broad handlers in ``main`` (iteration 2 step 4, D-176)
# ---------------------------------------------------------------------------
#
# Measured at 3da9b5075 (AST): 80 broad handlers (bare ``except``,
# ``Exception``, ``BaseException``) sit lexically in a ``main`` under
# ``src/train``. 21 neither raise nor exit non-zero. 9 of those let a failed
# run exit 0: ``nbeats`` (``os._exit(0)`` after the handler), ``tabm`` (a
# failed example is logged), the three logic sweeps (a failed cell becomes a
# NaN row and the run returns 0), ``train_e5_clevr_hans`` (a failed dataset
# build leaves ``None``, every model is skipped, ``DONE``) and ``mothnet``
# (a failed final-model save is logged; ``main()``'s value was discarded).
# The other 12 are fallbacks by design and are named below.
#
# Every broad handler in a ``main`` must now do one of:
#   - re-raise (any ``raise`` in its body);
#   - exit non-zero: ``sys.exit``/``exit``/``os._exit`` with a non-zero
#     literal, or ``return`` a non-zero literal from a ``main`` whose value
#     the module's ``__main__`` block passes to ``sys.exit``/``SystemExit``;
#   - record the failure (assign a name, or append/add to it) that ``main``
#     reads after the ``try`` in an ``if`` that exits non-zero, or passes to an
#     exit call (a flag the handler sets to a non-zero literal). This is the
#     D-095 sweep shape: finish every cell, keep the partial rows, then fail;
#   - be a named exemption below, whose shape is checked.

_EXIT_CALLS = {"sys.exit", "exit", "quit", "os._exit"}
_RECORDING_METHODS = {"append", "add", "extend", "update"}


def _is_broad(handler):
    if handler.type is None:
        return True
    names = (handler.type.elts if isinstance(handler.type, ast.Tuple)
             else [handler.type])
    return any(isinstance(name, ast.Name)
               and name.id in ("Exception", "BaseException") for name in names)


def _walk_no_defs(nodes):
    """Every node under ``nodes``, not entering nested defs, classes or lambdas."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _non_zero_int(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool) and node.value != 0)


def _exit_call_argument(node):
    """The first argument of a ``sys.exit``-like call, else ``None``."""
    if (isinstance(node, ast.Call) and ast.unparse(node.func) in _EXIT_CALLS
            and node.args):
        return node.args[0]
    return None


def _main_value_is_the_exit_code(tree):
    """``True`` when ``__main__`` runs ``sys.exit(main())`` or ``SystemExit(main())``."""
    for statement in tree.body:
        if not (isinstance(statement, ast.If)
                and "__main__" in ast.unparse(statement.test)):
            continue
        for node in ast.walk(statement):
            if (isinstance(node, ast.Call)
                    and ast.unparse(node.func) in _EXIT_CALLS | {"SystemExit"}
                    and node.args and isinstance(node.args[0], ast.Call)
                    and ast.unparse(node.args[0].func) == "main"):
                return True
    return False


def _exits_non_zero(statements, return_is_exit_code):
    for node in _walk_no_defs(statements):
        if isinstance(node, ast.Raise):
            return True
        if (return_is_exit_code and isinstance(node, ast.Return)
                and _non_zero_int(node.value)):
            return True
        if _non_zero_int(_exit_call_argument(node)):
            return True
    return False


def _recorded_names(handler):
    """(names the handler records into, names it sets to a non-zero literal)."""
    recorded, flags = set(), set()
    for node in _walk_no_defs(handler.body):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    recorded.add(target.id)
                    if isinstance(node, ast.Assign) and _non_zero_int(node.value):
                        flags.add(target.id)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RECORDING_METHODS
                and isinstance(node.func.value, ast.Name)):
            recorded.add(node.func.value.id)
    return recorded, flags


def _reads(node, names):
    return any(isinstance(sub, ast.Name) and sub.id in names
               for sub in ast.walk(node))


def _fails_later(func, try_node, handler, return_is_exit_code):
    recorded, flags = _recorded_names(handler)
    if not recorded:
        return False
    for node in _walk_no_defs(func.body):
        if getattr(node, "lineno", 0) <= try_node.end_lineno:
            continue
        if (isinstance(node, ast.If) and not isinstance(node.test, ast.UnaryOp)
                and _reads(node.test, recorded)
                and _exits_non_zero(node.body, return_is_exit_code)):
            return True
        argument = _exit_call_argument(node)
        if argument is not None and _reads(argument, flags):
            return True
    return False


def _main_broad_handlers(tree):
    """(main, try, handler) for every broad handler lexically in a ``main``."""
    for func in ast.walk(tree):
        if not (isinstance(func, ast.FunctionDef) and func.name == "main"):
            continue
        for node in _walk_no_defs(func.body):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if _is_broad(handler):
                        yield func, node, handler


# path -> [(text in the handler, text in the try body, fallback names the
# handler may assign, reason)]. The handler text picks the handler; the try
# text is what makes the fallback optional; the shape check holds the handler
# to logging (plus the named fallback assignment) and nothing else.
_EXEMPT_MAIN_HANDLERS = {
    "train/bert/deploy.py": [
        ("An unexpected error occurred", "input(", (),
         "an interactive prompt: a failed prediction is reported and the "
         "prompt goes on; the user leaves with quit or Ctrl-C"),
    ],
    "train/bert/wikipedia/pretrain.py": [
        ("Training on default strategy", "MirroredStrategy", ("strategy",),
         "no multi-GPU strategy on this host; the default strategy trains"),
    ],
    "train/bert/wikipedia/pretrain_english.py": [
        ("Using default strategy", "MirroredStrategy", ("strategy",),
         "no multi-GPU strategy on this host; the default strategy trains"),
    ],
    "train/hnet/prepare_hnet_data.py": [
        ("remote metadata lookup failed", "remote_size_bytes", (),
         "an optional remote size for the plan printout; the manifest budget "
         "is used instead"),
    ],
    "train/lightglue/train_lightglue.py": [
        ("viz batch capture failed", "LightGlueVizCallback", (),
         "optional match panels; training and the saved model do not need them"),
    ],
    "train/logic/train_e3_faithfulness.py": [
        ("LIME failed", "lime_attributions", (),
         "an optional comparison attribution (third-party LIME); the circuit's "
         "own attributions are not in the try"),
        ("SHAP failed", "shap_attributions", (),
         "an optional comparison attribution (third-party SHAP); the circuit's "
         "own attributions are not in the try"),
    ],
    "train/mothnet/train_mothnet.py": [
        ("render_training_dashboard failed", "render_training_dashboard", (),
         "a figure; a visualization failure never aborts training (the "
         "trainer's documented invariant 9)"),
        ("render_mb_sparsity failed", "render_mb_sparsity", (),
         "a figure; invariant 9"),
        ("AL/MB activation visualization", "extract_al_features", (),
         "a figure; invariant 9"),
        ("confusion_matrix visualization", "ClassificationResults", (),
         "a figure; invariant 9"),
        ("{_plugin_name} visualization", "_visualize_or_warn", (),
         "a figure (end-of-run per-class plots); invariant 9"),
    ],
}


def _exemption_shape_holds(try_node, handler, try_text, fallbacks):
    if try_text not in "\n".join(ast.unparse(s) for s in try_node.body):
        return False
    for statement in handler.body:
        if isinstance(statement, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func).startswith("logger.")):
            continue
        if (isinstance(statement, ast.Assign)
                and all(isinstance(t, ast.Name) and t.id in fallbacks
                        for t in statement.targets)):
            continue
        return False
    return True


def _classify_main_handlers(source, exemptions=()):
    """Census rows ``(line, verdict)``; verdict is ``None`` for an offender.

    ``exemptions`` is this file's list for the source. An exemption whose
    handler text matches but whose shape no longer holds is an offender.
    """
    with warnings.catch_warnings():  # invalid escapes in scanned sources
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        tree = ast.parse(source)
    return_is_exit_code = _main_value_is_the_exit_code(tree)
    rows = []
    for func, try_node, handler in _main_broad_handlers(tree):
        handler_text = ast.unparse(handler)
        if any(isinstance(n, ast.Raise) for n in _walk_no_defs(handler.body)):
            verdict = "raises"
        elif _exits_non_zero(handler.body, return_is_exit_code):
            verdict = "exits"
        elif _fails_later(func, try_node, handler, return_is_exit_code):
            verdict = "records"
        else:
            verdict = None
            for marker, try_text, fallbacks, _reason in exemptions:
                if (marker in handler_text and _exemption_shape_holds(
                        try_node, handler, try_text, fallbacks)):
                    verdict = f"exempt: {marker}"
        rows.append((handler.lineno, verdict))
    return sorted(rows)


def _main_handler_census():
    census = {}
    for path in sorted((_SRC / "train").rglob("*.py")):
        relative = path.relative_to(_SRC).as_posix()
        rows = _classify_main_handlers(
            path.read_text(encoding="utf-8"),
            _EXEMPT_MAIN_HANDLERS.get(relative, ()))
        if rows:
            census[relative] = rows
    return census


def test_every_broad_handler_in_a_main_fails_the_run_or_is_exempt():
    census = _main_handler_census()
    offenders = [f"{path}:{line}" for path, rows in census.items()
                 for line, verdict in rows if verdict is None]
    assert offenders == [], (
        "these broad handlers in main() let a failed run exit 0; re-raise, "
        "exit non-zero, or record the failure and fail after the sweep "
        "(D-095):\n" + "\n".join(offenders))


def test_each_main_handler_exemption_names_exactly_one_handler():
    """No stale or ambiguous exemption: each matches one handler, by shape."""
    census = _main_handler_census()
    for relative, exemptions in _EXEMPT_MAIN_HANDLERS.items():
        verdicts = [verdict for _line, verdict in census.get(relative, [])]
        for marker, _try_text, _fallbacks, reason in exemptions:
            assert reason
            assert verdicts.count(f"exempt: {marker}") == 1, (relative, marker)
    exempt = sum(1 for rows in census.values() for _line, verdict in rows
                 if verdict and verdict.startswith("exempt"))
    assert exempt == 12


_MAIN_HANDLER_FORMS = '''
import os
import sys


def main():
    try:
        run()
    except Exception as e:
        logger.error(f"failed: {e}")
    try:
        run()
    except Exception:
        return 1
    try:
        run()
    except Exception:
        raise
    try:
        run()
    except Exception:
        sys.exit(2)
    failed = []
    for cell in cells:
        try:
            run(cell)
        except Exception:
            failed.append(cell)
    if failed:
        raise RuntimeError(failed)
    seen = []
    for cell in cells:
        try:
            run(cell)
        except Exception:
            seen.append(cell)
    if not seen:
        raise RuntimeError("nothing ran")
    code = 0
    try:
        run()
    except Exception:
        code = 1
    os._exit(code)
    try:
        run()
    except BaseException:
        logger.warning("optional plot failed")
    try:
        input("> ")
    except Exception:
        logger.info("prompt error")
'''

_PLANTED_EXEMPTION = [
    ("prompt error", "input(", (), "a prompt"),
    # Matches by text, but its try holds no plot: the shape check refuses it.
    ("optional plot failed", "render_plot", (), "a figure"),
]


def test_the_main_handler_census_tells_a_swallow_from_a_failure():
    """Positive control over planted forms, with and without ``sys.exit(main())``."""
    lines = _MAIN_HANDLER_FORMS.splitlines()
    excepts = [i for i, line in enumerate(lines, start=1)
               if line.strip().startswith("except")]
    verdicts = dict(_classify_main_handlers(_MAIN_HANDLER_FORMS, _PLANTED_EXEMPTION))
    # The logged swallow, ``return 1`` whose value nobody exits with, the
    # ``if not seen`` that fails only when nothing failed, and the exemption
    # whose shape does not hold.
    offenders = sorted(line for line, verdict in verdicts.items() if verdict is None)
    assert offenders == [excepts[0], excepts[1], excepts[5], excepts[7]]
    assert verdicts[excepts[2]] == "raises"
    assert verdicts[excepts[3]] == "exits"
    assert verdicts[excepts[4]] == "records"
    assert verdicts[excepts[6]] == "records"
    assert verdicts[excepts[8]] == "exempt: prompt error"

    exiting = _MAIN_HANDLER_FORMS + '\n\nif __name__ == "__main__":\n    sys.exit(main())\n'
    assert dict(_classify_main_handlers(exiting, _PLANTED_EXEMPTION))[excepts[1]] == "exits"


# Process exit codes for 3 of the 9: the failing step is stubbed in a child
# interpreter, which runs the trainer the way a shell does. Each writes only
# under ``tmp_path``.

def _run_child(script, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "-c", script], env=env,
                          cwd=str(tmp_path), capture_output=True, text=True,
                          timeout=600)


_NBEATS_CHILD = '''
import sys
import train.time_series.nbeats.train_nbeats as m

class Trainer:
    def __init__(self, config, generator_config):
        if {fail!r}:
            raise RuntimeError("the run crashed (test)")

    def run_experiment(self):
        return {{"results_dir": "unused"}}

m.NBeatsTrainer = Trainer
sys.argv = ["train_nbeats"]
m.main()
'''


@pytest.mark.parametrize("fail, expected", [(True, 1), (False, 0)])
def test_a_failed_nbeats_process_exits_non_zero(tmp_path, fail, expected):
    """N-BEATS ends in ``os._exit`` (its D-008 teardown); the code is the run's."""
    done = _run_child(_NBEATS_CHILD.format(fail=fail), tmp_path)
    assert done.returncode == expected, done.stderr[-3000:]


_MONKS_CHILD = '''
import runpy
import sys
import train.logic.monks_data as data

def unavailable(problem_id):
    raise RuntimeError(f"monks-{{problem_id}} data unavailable (test)")

data.load_monks = unavailable
sys.argv = ["train_e4_monks", "--problems", "1,2", "--models", "circuit",
            "--seeds", "0", "--out-dir", {out!r}]
runpy.run_module("train.logic.train_e4_monks", run_name="__main__")
'''


def test_a_logic_sweep_with_failed_cells_exits_non_zero(tmp_path):
    """The sweep finishes and writes its rows, then the process fails."""
    done = _run_child(_MONKS_CHILD.format(out=str(tmp_path / "monks")), tmp_path)
    assert done.returncode != 0, done.stderr[-3000:]
    (out_dir,) = tmp_path.glob("monks_*")
    rows = (out_dir / "results.csv").read_text().splitlines()
    assert len(rows) == 3, rows  # header + both cells
    assert "problem=1" in done.stderr and "problem=2" in done.stderr


_MOTHNET_CHILD = '''
import runpy
import sys
import train.common
from dl_techniques.models.general_purpose.mothnet.model import MothNet

# --gpu defaults to 0; the child sees no device (CUDA_VISIBLE_DEVICES="").
train.common.setup_gpu = lambda *args, **kwargs: None

real_save = MothNet.save

def save(self, path, *args, **kwargs):
    if str(path).endswith("final_model.keras"):
        raise OSError(f"disk full writing {{path}} (test)")
    return real_save(self, path, *args, **kwargs)

MothNet.save = save
sys.argv = ["train_mothnet", "--mb-units", "200", "--al-units", "64",
            "--epochs", "1", "--num-train-samples", "100",
            "--num-val-samples", "50", "--batch-size", "32", "--viz-freq", "0",
            "--output-dir", {out!r}, "--experiment-name", "run"]
runpy.run_module("train.mothnet.train_mothnet", run_name="__main__")
'''


def test_a_mothnet_process_whose_final_save_fails_exits_non_zero(tmp_path):
    done = _run_child(_MOTHNET_CHILD.format(out=str(tmp_path)), tmp_path)
    assert done.returncode != 0, done.stderr[-3000:]
    assert "disk full writing" in done.stderr
    # The epoch checkpoint was written before the failed final save.
    checkpoints = [p for p in (tmp_path / "run").rglob("*.keras")
                   if p.name != "final_model.keras"]
    assert checkpoints, sorted(str(p) for p in (tmp_path / "run").rglob("*"))


# In-process checks for the other sweeps: every cell is attempted, the rows are
# written, then the run fails naming the failed cells.

def _crash(*_args, **_kwargs):
    raise RuntimeError("the cell crashed (test)")


def test_a_tabm_run_with_a_failed_example_raises_after_the_rest(monkeypatch):
    from train.tabm import train_tabm as module

    ran = []
    examples = {name: (lambda name=name: ran.append(name)) for name in module.EXAMPLES}
    examples["regression"] = _crash
    monkeypatch.setattr(module, "EXAMPLES", examples)
    monkeypatch.setattr("train.common.setup_gpu", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="regression"):
        module.main(["--examples", "regression", "real-dataset"])
    assert ran == ["real-dataset"]


def test_the_logic_benchmark_with_a_failed_cell_raises_after_the_rest(
        monkeypatch, tmp_path):
    from train.logic import train_benchmark as module

    monkeypatch.chdir(tmp_path)  # it writes under ./results
    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(module, "train_one", _crash)
    monkeypatch.setattr(sys, "argv", ["train_benchmark", "--only-task", "parity_k6"])
    with pytest.raises(RuntimeError, match="parity_k6/mlp_large"):
        module.main()
    (csv_path,) = tmp_path.glob("results/*/results.csv")
    assert len(csv_path.read_text().splitlines()) == 4  # header + 3 cells


def test_the_lowdata_mux_sweep_with_a_failed_cell_raises_after_the_rest(
        monkeypatch, tmp_path):
    from train.logic import train_e4_lowdata_mux as module

    def run_cell(n, model_name, seed):
        if seed == 1:
            _crash()
        return {"N": n, "model": model_name, "seed": seed, "params": 1,
                "epochs_used": 1, "wall_s": 0.0, "test_acc": 0.9}

    monkeypatch.setattr(module, "run_cell", run_cell)
    monkeypatch.setattr(sys, "argv", [
        "train_e4_lowdata_mux", "--seeds", "0,1,2", "--n-values", "32",
        "--models", "circuit", "--out-dir", str(tmp_path / "mux")])
    with pytest.raises(RuntimeError, match="1 cells failed: N=32 model=circuit seed=1"):
        module.main()
    (out_dir,) = tmp_path.glob("mux_*")
    assert len((out_dir / "mux_results.csv").read_text().splitlines()) == 4
    # The failed seed's NaN row no longer breaks the aggregate (statistics.mean
    # raised on it): the report and the plot are written.
    assert "circuit=0.900" in (out_dir / "report.md").read_text()
    assert (out_dir / "learning_curve.png").is_file()


def test_a_lowdata_mux_sweep_where_every_cell_fails_writes_its_report_then_raises(
        monkeypatch, tmp_path):
    """Every cell failed: the CSV, the report and the plot are still written,
    and the run raises naming every cell (the plot crashed on all-NaN data)."""
    from train.logic import train_e4_lowdata_mux as module

    monkeypatch.setattr(module, "run_cell", _crash)
    monkeypatch.setattr(sys, "argv", [
        "train_e4_lowdata_mux", "--seeds", "0,1", "--n-values", "32,64",
        "--models", "circuit", "--out-dir", str(tmp_path / "mux")])
    with pytest.raises(RuntimeError, match="4 cells failed: N=32 model=circuit seed=0"):
        module.main()
    (out_dir,) = tmp_path.glob("mux_*")
    assert len((out_dir / "mux_results.csv").read_text().splitlines()) == 5
    report = (out_dir / "report.md").read_text()
    assert "circuit=nan" in report and "Verdict" in report
    assert (out_dir / "learning_curve.png").is_file()


def _clevr(monkeypatch, tmp_path, argv):
    from train.logic import train_e5_clevr_hans as module

    monkeypatch.setattr(module, "setup_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["train_e5_clevr_hans", "--skip-download",
                                      "--out-dir", str(tmp_path / "out")] + argv)
    return module


def test_a_clevr_run_without_data_raises_after_its_report(monkeypatch, tmp_path):
    module = _clevr(monkeypatch, tmp_path, ["--data-dir", str(tmp_path / "none")])
    with pytest.raises(RuntimeError, match="no model trained"):
        module.main()
    assert (tmp_path / "out" / "report.md").is_file()


def test_a_clevr_run_whose_dataset_build_fails_raises(monkeypatch, tmp_path):
    (tmp_path / "data" / "images").mkdir(parents=True)
    module = _clevr(monkeypatch, tmp_path, ["--data-dir", str(tmp_path / "data")])
    monkeypatch.setattr(module, "find_splits",
                        lambda _d: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(module, "infer_num_classes", lambda _s: 3)
    monkeypatch.setattr(module, "build_image_dataset", _crash)
    with pytest.raises(RuntimeError, match="the cell crashed"):
        module.main()


def test_a_clevr_run_with_a_failed_fit_raises_after_its_report(monkeypatch, tmp_path):
    (tmp_path / "data" / "images").mkdir(parents=True)
    module = _clevr(monkeypatch, tmp_path, ["--data-dir", str(tmp_path / "data")])
    monkeypatch.setattr(module, "find_splits",
                        lambda _d: {"train": [], "val": [], "test": []})
    monkeypatch.setattr(module, "infer_num_classes", lambda _s: 3)
    for name in ("build_image_dataset", "build_symbolic_dataset"):
        monkeypatch.setattr(module, name, lambda *a, **k: (object(), 1))
    for name in ("build_resnet50_circuit", "build_resnet50_mlp", "build_symbolic_circuit"):
        monkeypatch.setattr(module, name, lambda *a, **k: object())

    def train_and_eval(model, *args, name, **kwargs):
        if name == "resnet50_mlp":
            return {"model": name, "status": "fit_error", "error": "boom", "wall_s": 0.0}
        return {"model": name, "status": "complete", "params": 1, "epochs_used": 1,
                "wall_s": 0.0, "val_acc": 0.5, "test_acc": 0.5, "shortcut_gap": 0.0}

    monkeypatch.setattr(module, "train_and_eval", train_and_eval)
    with pytest.raises(RuntimeError, match="resnet50_mlp"):
        module.main()
    rows = (tmp_path / "out" / "results.csv").read_text().splitlines()
    assert len(rows) == 4  # header + all three models
