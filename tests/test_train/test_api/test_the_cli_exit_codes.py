"""``python -m train`` exits 0, 1 or 2, and says why.

Exit code contract (D-009):

* 0: help, ``list_*``, or a task whose result has ``success=True``.
* 2: a :class:`~train.api.errors.DLTrainError` (unknown command, repeated key,
  bad value, unknown key, unreadable config file). The message goes to stderr
  with no traceback.
* 1: a task that returns ``success=False``, or raises anything else. An
  unexpected exception reaches the user as a traceback.
* Ctrl-C: non-zero. The exact value is the interpreter's business and is not
  pinned.

Every arm is a real child process with ``cwd=tmp_path``. A task is injected by
running ``runpy.run_module("train", run_name="__main__")`` after registering a
fake :class:`~train.api.registry.TaskSpec`, so ``train/__main__.py`` and
``sys.exit(main(...))`` really run. The fake task prints a marker before it
returns or raises, so each arm proves the task was reached and an early exit
cannot pass.

The fake task functions have the keyword-only signature a real task has
(``def run(*, value=1, rate=0.5)``) and do NOT validate. A function that
accepted ``**kwargs`` and validated by itself hid a defect once: a bad config
file reached a real signature as a ``TypeError`` and exited 1 (D-034).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Sequence

import pytest

MODULE = "fake_train_exit_code_tasks"
MARKER = "FAKE-RAN"

SOURCE = '''
import dataclasses
import sys
import typing
from types import SimpleNamespace

from train.api.config import StrictConfig
from train.api.errors import DLTrainError, UnknownModelError, UnresolvedAutoError


@dataclasses.dataclass(frozen=True)
class FakeConfig(StrictConfig):
    value: int = dataclasses.field(default=1, metadata={"help": "a count"})
    rate: float = dataclasses.field(default=0.5, metadata={"help": "a rate"})
    tags: typing.List[int] = dataclasses.field(default_factory=list, metadata={"help": "tags"})


@dataclasses.dataclass(frozen=True)
class RequiredConfig(StrictConfig):
    out: str = dataclasses.field(metadata={"help": "where to write"})
    value: int = dataclasses.field(default=1, metadata={"help": "a count"})


def _announce(name, **fields):
    shown = " ".join(f"{key}={val!r}" for key, val in fields.items())
    sys.stdout.write(f"FAKE-RAN {name} {shown}\\n")
    sys.stdout.flush()


def run_ok(*, value=1, rate=0.5, tags=()):
    _announce("ok", value=value, rate=rate)
    return SimpleNamespace(success=True)


def run_unmet(*, value=1, rate=0.5, tags=()):
    _announce("unmet", value=value, rate=rate)
    return SimpleNamespace(success=False)


def run_raises(*, value=1, rate=0.5, tags=()):
    _announce("raises", value=value, rate=rate)
    raise RuntimeError("the fake task blew up")


def run_interrupted(*, value=1, rate=0.5, tags=()):
    _announce("interrupted", value=value, rate=rate)
    raise KeyboardInterrupt


def run_none(*, value=1, rate=0.5, tags=()):
    _announce("none", value=value, rate=rate)
    return SimpleNamespace(success=None)


def run_unresolved(*, value=1, rate=0.5, tags=()):
    _announce("unresolved", value=value, rate=rate)
    raise UnresolvedAutoError("these fields are still 'auto': optim_args.lr")


def run_unknown_model(*, value=1, rate=0.5, tags=()):
    _announce("unknown_model", value=value, rate=rate)
    raise UnknownModelError("no model 'vit/huge'")


def run_dltrain(*, value=1, rate=0.5, tags=()):
    _announce("dltrain", value=value, rate=rate)
    raise DLTrainError("a plain base-class failure")


def run_required(*, out, value=1):
    _announce("required", out=out, value=value)
    return SimpleNamespace(success=True)
'''

# The driver registers the fake tasks, then runs the real ``train.__main__``.
# It reads sys.argv[1:] the way ``python -m train`` does, so the arguments
# after ``-c`` are the command line under test.
FAKE_NAMES = (
    "ok", "unmet", "raises", "interrupted", "required",
    "none", "unresolved", "unknown_model", "dltrain",
)

DRIVER = f'''
import runpy

from train.api.registry import TaskSpec, register_task

for name in {FAKE_NAMES!r}:
    register_task(
        TaskSpec(
            name=f"fake_{{name}}",
            config_class="{MODULE}:" + ("RequiredConfig" if name == "required" else "FakeConfig"),
            function=f"{MODULE}:run_{{name}}",
            summary="a fake task",
        )
    )
runpy.run_module("train", run_name="__main__")
'''


@pytest.fixture
def source_dir(tmp_path) -> Path:
    """Write the fake task module into its own directory, apart from ``cwd``."""
    directory = tmp_path / "fake_source"
    directory.mkdir()
    (directory / f"{MODULE}.py").write_text(textwrap.dedent(SOURCE))
    return directory


@pytest.fixture
def cwd(tmp_path) -> Path:
    directory = tmp_path / "cwd"
    directory.mkdir()
    return directory


def run_injected(
    args: Sequence[str], source_dir: Path, cwd: Path, python_args: Sequence[str] = ()
):
    """Run ``python -m train <args>`` with the fake tasks registered first.

    :param python_args: interpreter options placed before ``-c``, e.g. ``-X importtime``.
    """
    existing = os.environ.get("PYTHONPATH")
    path = os.pathsep.join([str(source_dir)] + ([existing] if existing else []))
    env = {**os.environ, "PYTHONPATH": path}
    return subprocess.run(
        [sys.executable, *python_args, "-c", DRIVER, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def run_real(args: Sequence[str], cwd: Path):
    """Run the unmodified ``python -m train <args>``."""
    return subprocess.run(
        [sys.executable, "-m", "train", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---------------------------------------------------------------------
# 0
# ---------------------------------------------------------------------


def test_help_exits_zero(cwd):
    result = run_real(["help"], cwd)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


def test_a_successful_task_exits_zero_and_receives_the_typed_values(source_dir, cwd):
    result = run_injected(["fake_ok", "value=7", "rate=3e-4"], source_dir, cwd)
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"{MARKER} ok value=7 rate=0.0003" in result.stdout
    assert "Traceback" not in result.stderr


def test_the_injected_tasks_are_listed(source_dir, cwd):
    # Liveness for every injected arm: the driver really registered the fakes.
    result = run_injected(["list_tasks"], source_dir, cwd)
    assert result.returncode == 0, result.stderr[-2000:]
    for name in FAKE_NAMES:
        assert f"fake_{name}" in result.stdout


# ---------------------------------------------------------------------
# 2
# ---------------------------------------------------------------------

CONFIG_ERROR_ARMS = [
    pytest.param(["fake_ok", "value=1", "value=2"], "Repeated key 'value'", id="repeated-key"),
    pytest.param(["fake_ok", "value=abc"], "Invalid value for 'value': expected int, got 'abc' (str)", id="bad-value"),
    pytest.param(["fake_ok", "valu=3"], "Did you mean 'value'?", id="unknown-key"),
    pytest.param(["fake_ok", "value"], "is not key=value", id="token-without-equals"),
    pytest.param(["fake_ok", "config=/nonexistent_dir/none.yaml"], "Cannot read config file '/nonexistent_dir/none.yaml'", id="unreadable-config-file"),
]


@pytest.mark.parametrize("args, message", CONFIG_ERROR_ARMS)
def test_a_config_error_exits_two_with_the_message_and_no_traceback(
    args: List[str], message: str, source_dir, cwd
):
    result = run_injected(args, source_dir, cwd)
    assert result.returncode == 2, (result.stdout[-1000:], result.stderr[-2000:])
    assert result.stderr.startswith("error: "), result.stderr
    assert message in result.stderr, result.stderr
    assert "Traceback" not in result.stderr
    # The task is never reached for a bad command line.
    assert MARKER not in result.stdout


# A config file is not coerced by the command line, so each of these reaches
# ``run_task`` as a mapping the keyword-only function could not take.
CONFIG_FILE_ARMS = [
    pytest.param("fake_ok", b"valu: 3\n", "Unknown key 'valu'. Did you mean 'value'?", id="unknown-top-key"),
    pytest.param("fake_ok", b"1: 2\n", "Invalid key '1': keys must be strings", id="non-string-key"),
    pytest.param("fake_ok", b"value: 010x\n", "Invalid value for 'value': expected int, got '010x' (str)", id="bad-value"),
    pytest.param("fake_ok", b"value: 1_000\n", "Invalid value for 'value': expected int, got '1_000' (str)", id="underscore-int"),
    pytest.param("fake_ok", b"value: 1:30\n", "Invalid value for 'value': expected int, got '1:30' (str)", id="sexagesimal-int"),
    pytest.param("fake_ok", b"value: 0x10\n", "Invalid value for 'value': expected int, got '0x10' (str)", id="hex-int"),
    pytest.param("fake_ok", b"- 1\n- 2\n", "must have a mapping at the top level", id="list-top-level"),
    pytest.param("fake_ok", b"value: [1\n", "is not valid YAML", id="yaml-syntax"),
    pytest.param("fake_ok", b"value: 1\nvalue: 2\n", "duplicate key 'value'", id="repeated-key"),
    pytest.param("fake_ok", b"value: \xff\xfe\n", "Cannot read config file", id="not-utf-8"),
    pytest.param("fake_required", b"value: 3\n", "Missing required key 'out'.", id="missing-required-key"),
    pytest.param("fake_required", b"", "mapping at the top level", id="empty-file"),
]


@pytest.mark.parametrize("task, content, message", CONFIG_FILE_ARMS)
def test_a_bad_config_file_exits_two_before_the_keyword_only_function(
    task, content, message, source_dir, cwd, tmp_path
):
    path = tmp_path / "config.yaml"
    path.write_bytes(content)
    result = run_injected([task, f"config={path}"], source_dir, cwd)
    assert result.returncode == 2, (result.stdout[-1000:], result.stderr[-2000:])
    assert result.stderr.startswith("error: "), result.stderr
    assert message in result.stderr, result.stderr
    assert "Traceback" not in result.stderr
    assert MARKER not in result.stdout


def test_a_missing_required_key_exits_two_and_the_control_with_the_key_runs(source_dir, cwd):
    refused = run_injected(["fake_required", "value=3"], source_dir, cwd)
    assert refused.returncode == 2, refused.stderr[-2000:]
    assert "Missing required key 'out'." in refused.stderr
    assert "Traceback" not in refused.stderr and MARKER not in refused.stdout
    accepted = run_injected(["fake_required", "out=somewhere", "value=3"], source_dir, cwd)
    assert accepted.returncode == 0, accepted.stderr[-2000:]
    assert f"{MARKER} required out='somewhere' value=3" in accepted.stdout


def test_a_good_config_file_runs_and_a_yaml_octal_means_ten(source_dir, cwd, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("value: 010\nrate: 3e-4\n")
    result = run_injected(["fake_ok", f"config={path}"], source_dir, cwd)
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"{MARKER} ok value=10 rate=0.0003" in result.stdout


@pytest.mark.parametrize(
    "token",
    ["value=" + "9" * 5000, "tags=" + "[" * 100000],
    ids=["huge-digit-string", "deeply-nested-json"],
)
def test_a_token_that_would_crash_the_parser_is_a_refusal_with_exit_two(token, source_dir, cwd):
    result = run_injected(["fake_ok", token], source_dir, cwd)
    assert result.returncode == 2, (result.stdout[-500:], result.stderr[-1500:])
    assert "Traceback" not in result.stderr
    assert result.stderr.startswith("error: ")
    assert len(result.stderr) < 1000
    assert MARKER not in result.stdout


@pytest.mark.parametrize("command", ["list_tasks", "list_models"])
@pytest.mark.parametrize("word", ["-h", "--help", "help"])
def test_a_list_command_accepts_every_help_word(command, word, cwd):
    result = run_real([command, word], cwd)
    assert result.returncode == 0, result.stderr[-2000:]
    if command == "list_tasks":
        assert result.stderr == ""
    else:
        # It lists the real families, which imports TensorFlow; that prints
        # its own start-up lines on stderr. Ours would start with ``error:``.
        assert "Traceback" not in result.stderr
        assert not any(
            line.startswith("error:") for line in result.stderr.splitlines()
        )
        assert len(result.stdout.split()) == 36


# A task that raises a DLTrainError of its own is a user-fixable failure, not a
# crash: every subclass of the base class exits 2, whichever branch it sits on.
DLTRAIN_ARMS = [
    pytest.param("fake_unresolved", "these fields are still 'auto': optim_args.lr", id="unresolved-auto"),
    pytest.param("fake_unknown_model", "no model 'vit/huge'", id="unknown-model"),
    pytest.param("fake_dltrain", "a plain base-class failure", id="base-class"),
]


@pytest.mark.parametrize("task, message", DLTRAIN_ARMS)
def test_every_dltrain_error_a_task_raises_exits_two_without_a_traceback(
    task, message, source_dir, cwd
):
    result = run_injected([task, "value=1"], source_dir, cwd)
    assert f"{MARKER} {task[len('fake_'):]}" in result.stdout, "the task was not reached"
    assert result.returncode == 2, (result.stdout[-500:], result.stderr[-1500:])
    assert result.stderr == f"error: {message}\n", result.stderr
    assert "Traceback" not in result.stderr


def test_two_bad_keys_print_two_lines_at_once(source_dir, cwd):
    result = run_injected(["fake_ok", "value=abc", "rate=zzz"], source_dir, cwd)
    assert result.returncode == 2
    assert "'value'" in result.stderr and "'rate'" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_unknown_command_exits_two_in_the_injected_process(source_dir, cwd):
    result = run_injected(["fake_okk"], source_dir, cwd)
    assert result.returncode == 2
    assert "unknown command 'fake_okk'" in result.stderr
    assert "Did you mean 'fake_ok'?" in result.stderr
    assert "Traceback" not in result.stderr


def test_the_real_command_rejects_an_unknown_command(cwd):
    result = run_real(["bogus"], cwd)
    assert result.returncode == 2, result.stderr[-2000:]
    assert "unknown command 'bogus'" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------
# 1
# ---------------------------------------------------------------------


def test_an_unmet_result_exits_one_without_a_traceback(source_dir, cwd):
    result = run_injected(["fake_unmet", "value=1"], source_dir, cwd)
    assert result.returncode == 1, result.stderr[-2000:]
    assert f"{MARKER} unmet" in result.stdout
    assert "Traceback" not in result.stderr


def test_a_result_with_no_success_value_exits_one(source_dir, cwd):
    result = run_injected(["fake_none", "value=1"], source_dir, cwd)
    assert f"{MARKER} none" in result.stdout, "the task was not reached"
    assert result.returncode == 1, (result.stdout[-500:], result.stderr[-1500:])
    assert "Traceback" not in result.stderr


def test_an_unexpected_exception_exits_one_with_a_traceback(source_dir, cwd):
    result = run_injected(["fake_raises", "value=1"], source_dir, cwd)
    assert result.returncode == 1, result.stderr[-2000:]
    assert f"{MARKER} raises" in result.stdout
    assert "Traceback" in result.stderr
    assert "RuntimeError: the fake task blew up" in result.stderr
    # It is not reported as a configuration error.
    assert not result.stderr.startswith("error: ")


# ---------------------------------------------------------------------
# Ctrl-C
# ---------------------------------------------------------------------


def test_an_interrupted_task_exits_non_zero(source_dir, cwd):
    result = run_injected(["fake_interrupted", "value=1"], source_dir, cwd)
    assert f"{MARKER} interrupted" in result.stdout
    # The value is the interpreter's (130 or -SIGINT); only "not 0" is the contract.
    assert result.returncode != 0
    assert result.returncode != 2, "Ctrl-C must not look like a configuration error"


# ---------------------------------------------------------------------
# A real training run: the image-classification task, on fake arrays
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

# The fake loader is installed in the child before ``train.__main__`` runs, so
# the real task, the real trainer and ``sys.exit(main(...))`` all run.
TRAIN_DRIVER = f'''
import runpy
import sys

sys.path.insert(0, {str(REPO_ROOT)!r})
import train.common.classification_data as classification_data
from tests.classification_fakes import tiny_raw

classification_data._load_raw = tiny_raw(n_train=40, n_test=8)
runpy.run_module("train", run_name="__main__")
'''

TRAIN_TOKENS = [
    "image_classification",
    "model=vit/vit_pico",
    "data.dataset=cifar10",
    "epochs=1",
    "batch_size=4",
    "analysis_args.model_analysis=false",
]


def run_training(extra: Sequence[str], cwd: Path):
    return subprocess.run(
        [sys.executable, "-c", TRAIN_DRIVER, *TRAIN_TOKENS, *extra],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
    )


def error_lines(stderr: str) -> List[str]:
    return [line for line in stderr.splitlines() if line.startswith("error:")]


def test_a_real_training_run_exits_zero_and_writes_its_run_directory(cwd, tmp_path):
    out = tmp_path / "run"
    result = run_training([f"out={out}", "metric_args.success_threshold=0.0"], cwd)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (out / "results_summary.json").is_file() and (out / "final_model.keras").is_file()
    assert "Traceback" not in result.stderr
    # Nothing was written next to the process: the run lives under `out` only.
    assert list(cwd.iterdir()) == []


def test_an_unmet_threshold_of_a_real_run_exits_one_without_a_traceback(cwd, tmp_path):
    out = tmp_path / "run"
    result = run_training([f"out={out}", "metric_args.success_threshold=1.0"], cwd)
    assert result.returncode == 1, result.stderr[-3000:]
    assert (out / "results_summary.json").is_file()
    assert "Traceback" not in result.stderr
    assert error_lines(result.stderr) == []


def test_an_erasing_probability_with_another_preset_exits_two_before_the_run_is_claimed(
    cwd, tmp_path
):
    out = tmp_path / "run"
    result = run_training(
        [f"out={out}", "transform_args.augment=flip", "transform_args.random_erasing_prob=0.3"],
        cwd,
    )
    assert result.returncode == 2, (result.stdout[-1000:], result.stderr[-3000:])
    lines = error_lines(result.stderr)
    assert len(lines) == 1 and "'transform_args.random_erasing_prob'" in lines[0], lines
    assert "Traceback" not in result.stderr
    assert not out.exists()


# ---------------------------------------------------------------------
# `export` and `predict`: exit 0 on a trained run, exit 2 for anything the user can fix
# ---------------------------------------------------------------------
#
# One tiny real run (fake CIFAR-10 arrays, one epoch) is trained once. Its `task.json` is the
# only source of the input shape the export command traces with: the ViT is a subclassed
# model with no `.inputs`, so a command that asked the model for it would fail here.

import hashlib
import json
import math

import numpy as np
from PIL import Image

CLASSES = 10


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    import train.api.recipes as recipes
    from tests.classification_fakes import use_tiny_raw
    from train.api import registry
    from train.api.tasks.image_classification import train_image_classification

    out = tmp_path_factory.mktemp("trained") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        use_tiny_raw(patch, n_train=40, n_test=12)
        recipes.register_builtin_models()
        result = train_image_classification(
            out=str(out),
            model="vit/vit_pico",
            data={"dataset": "cifar10"},
            epochs=1,
            batch_size=4,
            analysis_args={"model_analysis": False},
            metric_args={"success_threshold": 0.0},
        )
    assert result.success
    return out


@pytest.fixture(scope="module")
def pictures(tmp_path_factory):
    """Five non-square images: three at the top of a directory, two under a sub-directory."""
    root = tmp_path_factory.mktemp("pictures")
    rng = np.random.default_rng(1)
    for name, size in (("a.png", (40, 30)), ("b.jpg", (30, 45)), ("c.PNG", (36, 36)),
                       ("nested/d.png", (50, 20)), ("nested/e.bmp", (24, 24))):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        pixels = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
    (root / "notes.txt").write_text("not an image")
    (root / ".hidden.png").write_bytes(b"not read")
    return root


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_command(argv, capsys):
    code = cli_main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def cli_main(argv):
    from train.api import cli

    return cli.main(argv)


def one_error(err: str) -> str:
    lines = [line for line in err.splitlines() if line.startswith("error:")]
    assert "Traceback" not in err, err[-1500:]
    assert len(lines) >= 1, err[-1500:]
    return "\n".join(err.splitlines())


@pytest.mark.parametrize("format, name", [("keras", "m.keras"), ("saved_model", "m_dir"), ("tflite", "m.tflite")])
def test_export_writes_each_format_from_a_run_directory(format, name, trained_run, tmp_path, capsys):
    import keras

    from train.api.predictor import load_model

    out = tmp_path / name
    code, stdout, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", f"format={format}"], capsys
    )
    assert code == 0, stderr[-1500:]
    assert stdout == f"{out}\n"  # ONLY the path, for every format (concern 17)
    assert out.exists()
    # The run's record travels with the export (concern 12): same content, a new file.
    assert json.loads((tmp_path / (name + ".task.json")).read_text()) == json.loads(
        (trained_run / "task.json").read_text()
    )
    if format == "keras":
        # Liveness: the written file is the run's model, not any model.
        batch = np.random.default_rng(0).random((2, 32, 32, 3)).astype(np.float32)
        expected = load_model(trained_run).model(batch, training=False)
        got = keras.models.load_model(str(out), compile=False)(batch, training=False)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6, rtol=0)
    if format == "tflite":
        assert (tmp_path / "m.tflite.saved_model").is_dir()  # kept, never deleted


def test_export_reads_the_input_shape_from_task_json_for_a_model_with_no_inputs(trained_run, tmp_path, capsys):
    """The ViT reports no `inputs`, so the traced size can only have come from `task.json`."""
    from train.api.predictor import load_model

    assert not getattr(load_model(trained_run).model, "inputs", None), "the shape must come from task.json"
    out = tmp_path / "m_dir"
    code, _, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", "format=saved_model"], capsys
    )
    assert code == 0, stderr[-1500:]
    import tensorflow as tf

    loaded = tf.saved_model.load(str(out))
    spec = loaded.signatures["serving_default"].structured_input_signature[1]
    assert [dim for dim in list(spec.values())[0].shape][1:] == [32, 32, 3]


def test_export_to_onnx_runs_or_is_refused_with_exit_two(trained_run, tmp_path, capsys, monkeypatch):
    out = tmp_path / "m.onnx"
    monkeypatch.setitem(sys.modules, "tf2onnx", None)  # `import tf2onnx` raises ImportError
    code, _, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", "format=onnx"], capsys
    )
    assert code == 2 and not out.exists()
    assert one_error(stderr).startswith("error: Invalid value for 'format': 'onnx' needs the tf2onnx")
    monkeypatch.undo()
    pytest.importorskip("tf2onnx")
    pytest.importorskip("onnxruntime")
    code, _, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", "format=onnx"], capsys
    )
    assert code == 0, stderr[-1500:]
    assert out.is_file() and out.stat().st_size > 0


def test_export_accepts_a_keras_file_beside_its_task_json(trained_run, tmp_path, capsys):
    out = tmp_path / "again.keras"
    code, _, stderr = run_command(
        ["export", f"model={trained_run / 'final_model.keras'}", f"out={out}", "format=keras"], capsys
    )
    assert code == 0, stderr[-1500:]
    assert out.is_file()


def lone_model(run, tmp):
    """A copy of the run's model file in a directory that holds no ``task.json``."""
    import shutil

    target = tmp / "lone" / "model.keras"
    target.parent.mkdir()
    shutil.copy(run / "final_model.keras", target)
    return target


def a_file(tmp):
    path = tmp / "afile"
    path.write_text("a file, not a directory")
    return path


def task_json_taken(tmp, name):
    """Create ``<name>.task.json`` in ``tmp`` and return the free ``tmp / name`` it sits beside."""
    (tmp / (name + ".task.json")).write_text("{}")
    return tmp / name


def a_dangling_link(tmp, name):
    link = tmp / ("link_" + name)
    link.symlink_to(tmp / "points_nowhere")
    return link


EXPORT_REFUSALS = [
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model={tmp / 'gone'}"],
                 "Invalid value for 'model': ", id="model-path-missing"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model={tmp}"],
                 "holds no best_model.keras", id="directory-without-a-run"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model={lone_model(run, tmp)}"],
                 "task.json", id="model-file-without-a-task-json"),
    pytest.param(lambda run, tmp: ["format=zip", f"out={tmp / 'o.keras'}", f"model={run}"],
                 "Invalid value for 'format': expected one of 'keras', 'saved_model', 'tflite', 'onnx', got 'zip'",
                 id="bad-format"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.h5'}", f"model={run}"],
                 "Invalid value for 'out': format 'keras' writes a file named *.keras", id="wrong-suffix"),
    pytest.param(lambda run, tmp: ["format=keras", f"model={run}"],
                 "Missing required key 'out'.", id="missing-key"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model={run}", "data=x"],
                 "Unknown key 'data'.", id="unknown-key"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model.x={run}"],
                 "'model' takes one value, not sub-keys.", id="nested-key"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", f"model={run}", "config=c.yaml"],
                 "export takes no config=", id="config-file"),
    pytest.param(lambda run, tmp: ["format=keras", "out=" + str(tmp), f"model={run}"],
                 "Invalid value for 'out': ", id="out-is-an-existing-directory-with-wrong-suffix"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={tmp / 'o.keras'}", "model="],
                 "Invalid value for 'model': it is empty.", id="empty-model"),
    pytest.param(lambda run, tmp: ["format=keras", "out=", f"model={run}"],
                 "Invalid value for 'out': it is empty.", id="empty-out"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={a_file(tmp) / 'o.keras'}", f"model={run}"],
                 "is not a directory, so", id="out-under-a-file"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={a_dangling_link(tmp, 'o.keras')}", f"model={run}"],
                 "already exists", id="out-is-a-dangling-symlink"),
    pytest.param(lambda run, tmp: ["format=keras", f"out={task_json_taken(tmp, 'o.keras')}", f"model={run}"],
                 "o.keras.task.json already exists", id="the-task-json-beside-out-exists"),
]


@pytest.mark.parametrize("build, message", EXPORT_REFUSALS)
def test_a_bad_export_exits_two_with_one_message_and_no_traceback(build, message, trained_run, tmp_path, capsys):
    arguments = build(trained_run, tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    code, stdout, stderr = run_command(["export", *arguments], capsys)
    assert code == 2, (stdout, stderr[-1500:])
    assert message in one_error(stderr), stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "a refused export wrote something"


def test_export_stdout_is_only_the_path_even_when_the_export_prints_and_when_it_fails(
    trained_run, tmp_path, capsys, monkeypatch
):
    """Keras prints `Saved artifact at ...` to stdout; the command sends that to stderr (concern 17)."""
    from train.api import exporting

    def chatty(*, model, out, format, input_shape, record):
        print("Saved artifact at 'x'. The following endpoints are available:")
        print("* Endpoint 'serve'")
        return Path(out)

    monkeypatch.setattr(exporting, "export", chatty)
    out = tmp_path / "m.keras"
    code, stdout, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", "format=keras"], capsys
    )
    assert code == 0 and stdout == f"{out}\n", (stdout, stderr[-500:])
    assert "Saved artifact at" in stderr and "Endpoint 'serve'" in stderr  # moved, not lost

    def chatty_and_failing(*, model, out, format, input_shape, record):
        print("* Endpoint 'serve'")
        raise exporting.ConfigError("Invalid value for 'format': the converter said no")

    monkeypatch.setattr(exporting, "export", chatty_and_failing)
    code, stdout, stderr = run_command(
        ["export", f"model={trained_run}", f"out={tmp_path / 'n.keras'}", "format=keras"], capsys
    )
    assert code == 2 and stdout == "" and "the converter said no" in stderr


def test_export_writes_the_record_beside_it_and_predict_reads_a_keras_export(
    trained_run, pictures, tmp_path, capsys
):
    """Concern 12: an exported .keras file predicts on its own, via `<out>.task.json`."""
    out = tmp_path / "moved" / "m.keras"
    code, _, stderr = run_command(
        ["export", f"model={trained_run / 'final_model.keras'}", f"out={out}", "format=keras"], capsys
    )
    assert code == 0, stderr[-1500:]
    record = out.parent / "m.keras.task.json"
    assert json.loads(record.read_text()) == json.loads((trained_run / "task.json").read_text())
    predicted = tmp_path / "p.json"
    code, _, stderr = run_command(
        ["predict", f"model={out}", f"data={pictures / 'a.png'}", f"out={predicted}"], capsys
    )
    assert code == 0, stderr[-1500:]
    from train.api.predictor import load_model

    expected = load_model(trained_run).predict(str(pictures / "a.png"))
    assert read_predictions(predicted)[0]["labels"] == expected["labels"]


def test_export_help_states_the_input_contract(capsys):
    code, stdout, _ = run_command(["export", "help"], capsys)
    text = " ".join(stdout.split())
    assert code == 0
    for phrase in ("standardized", "mean", "std", "raw logits", "<out>.task.json", "mixed_float16", "1e-4"):
        assert phrase in text, phrase


def test_export_to_an_existing_out_is_refused_and_leaves_it_byte_identical(trained_run, tmp_path, capsys):
    out = tmp_path / "taken.keras"
    out.write_bytes(b"precious")
    before = sha256_of(out)
    code, _, stderr = run_command(
        ["export", f"model={trained_run}", f"out={out}", "format=keras"], capsys
    )
    assert code == 2 and "already exists" in one_error(stderr)
    assert sha256_of(out) == before


def test_a_tflite_export_whose_saved_model_sibling_exists_is_refused_before_a_write(trained_run, tmp_path, capsys):
    (tmp_path / "m.tflite.saved_model").mkdir()
    code, _, stderr = run_command(
        ["export", f"model={trained_run}", f"out={tmp_path / 'm.tflite'}", "format=tflite"], capsys
    )
    assert code == 2 and "already exists" in one_error(stderr)
    assert not (tmp_path / "m.tflite").exists()


def read_predictions(path):
    entries = json.loads(Path(path).read_text())
    assert isinstance(entries, list)
    return entries


def test_predict_labels_a_directory_and_matches_the_predictor(trained_run, pictures, tmp_path, capsys):
    from train.api.predictor import load_model

    out = tmp_path / "pred.json"
    code, stdout, stderr = run_command(
        ["predict", f"model={trained_run}", f"data={pictures}", f"out={out}"], capsys
    )
    assert code == 0, stderr[-1500:]
    assert stdout.strip().splitlines()[-1] == str(out)
    entries = read_predictions(out)
    paths = [entry["path"] for entry in entries]
    # Recursive, sorted, by suffix (case-folded), hidden entries and other suffixes skipped.
    assert paths == sorted(
        str(pictures / name) for name in ("a.png", "b.jpg", "c.PNG", "nested/d.png", "nested/e.bmp")
    )
    predictor = load_model(trained_run)
    expected = predictor.predict(paths)
    for entry, labels, scores in zip(entries, expected["labels"], expected["scores"]):
        assert set(entry) == {"path", "labels", "scores"}
        assert entry["labels"] == labels and entry["labels"][0] in predictor.class_names
        assert sorted(entry["labels"]) == sorted(predictor.class_names) and len(entry["labels"]) == CLASSES
        np.testing.assert_allclose(entry["scores"], scores, atol=1e-9, rtol=0)
        assert all(math.isfinite(s) and 0 <= s <= 1 for s in entry["scores"])
        assert sum(entry["scores"]) == pytest.approx(1.0, abs=1e-6)
        assert entry["scores"] == sorted(entry["scores"], reverse=True)


def test_predict_takes_one_image_file(trained_run, pictures, tmp_path, capsys):
    out = tmp_path / "one.json"
    code, _, stderr = run_command(
        ["predict", f"model={trained_run}", f"data={pictures / 'b.jpg'}", f"out={out}"], capsys
    )
    assert code == 0, stderr[-1500:]
    entries = read_predictions(out)
    assert [e["path"] for e in entries] == [str(pictures / "b.jpg")]


def test_predict_creates_the_parent_of_out(trained_run, pictures, tmp_path, capsys):
    out = tmp_path / "deeper" / "still" / "pred.json"
    code, _, stderr = run_command(
        ["predict", f"model={trained_run}", f"data={pictures / 'a.png'}", f"out={out}"], capsys
    )
    assert code == 0, stderr[-1500:]
    assert out.is_file()


def test_predict_takes_more_images_than_one_chunk(trained_run, tmp_path, capsys, monkeypatch):
    """The command predicts in slices of `CHUNK`; the answer is the same for every slice size."""
    from train.api import predictor as predictor_module

    root = tmp_path / "many"
    root.mkdir()
    rng = np.random.default_rng(5)
    for index in range(7):
        Image.fromarray(rng.integers(0, 255, size=(20 + index, 30, 3), dtype=np.uint8)).save(root / f"i{index}.png")
    whole = tmp_path / "whole.json"
    sliced = tmp_path / "sliced.json"
    assert run_command(["predict", f"model={trained_run}", f"data={root}", f"out={whole}"], capsys)[0] == 0
    monkeypatch.setattr(predictor_module, "CHUNK", 3)
    assert run_command(["predict", f"model={trained_run}", f"data={root}", f"out={sliced}"], capsys)[0] == 0
    a, b = read_predictions(whole), read_predictions(sliced)
    assert len(a) == len(b) == 7
    for left, right in zip(a, b):
        assert left["path"] == right["path"] and left["labels"] == right["labels"]
        np.testing.assert_allclose(left["scores"], right["scores"], atol=1e-6, rtol=0)


def empty_dir(tmp):
    path = tmp / "empty"
    path.mkdir()
    return path


def broken_image(tmp):
    path = tmp / "broken.png"
    path.write_bytes(b"this is not a png")
    return path


PREDICT_REFUSALS = [
    pytest.param(lambda run, pics, tmp: [f"model={tmp / 'gone'}", f"data={pics}", f"out={tmp / 'p.json'}"],
                 "Invalid value for 'model': ", id="model-path-missing"),
    pytest.param(lambda run, pics, tmp: [f"model={tmp}", f"data={pics}", f"out={tmp / 'p.json'}"],
                 "holds no best_model.keras", id="directory-without-a-run"),
    pytest.param(lambda run, pics, tmp: [f"model={lone_model(run, tmp)}", f"data={pics}", f"out={tmp / 'p.json'}"],
                 "task.json", id="model-file-without-a-task-json"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={tmp / 'gone'}", f"out={tmp / 'p.json'}"],
                 "Invalid value for 'data': ", id="data-missing"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={empty_dir(tmp)}", f"out={tmp / 'p.json'}"],
                 "holds no image", id="no-image-in-the-directory"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={broken_image(tmp)}", f"out={tmp / 'p.json'}"],
                 "Invalid image", id="unreadable-image"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={pics}"],
                 "Missing required key 'out'.", id="missing-key"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={pics}", f"out={tmp / 'p.json'}", "format=keras"],
                 "Unknown key 'format'.", id="unknown-key"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={pics}", f"out={tmp / 'p.json'}", "config=c.yaml"],
                 "predict takes no config=", id="config-file"),
    pytest.param(lambda run, pics, tmp: ["model=", f"data={pics}", f"out={tmp / 'p.json'}"],
                 "Invalid value for 'model': it is empty.", id="empty-model"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", "data=", f"out={tmp / 'p.json'}"],
                 "Invalid value for 'data': it is empty.", id="empty-data"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={pics}", f"out={a_file(tmp) / 'p.json'}"],
                 "is not a directory, so", id="out-under-a-file"),
    pytest.param(lambda run, pics, tmp: [f"model={run}", f"data={pics}", f"out={a_dangling_link(tmp, 'p.json')}"],
                 "already exists", id="out-is-a-dangling-symlink"),
]


@pytest.mark.parametrize("build, message", PREDICT_REFUSALS)
def test_a_bad_predict_exits_two_with_a_message_and_no_traceback(
    build, message, trained_run, pictures, tmp_path, capsys
):
    arguments = build(trained_run, pictures, tmp_path)
    existing = sorted(p.name for p in tmp_path.iterdir())
    code, stdout, stderr = run_command(["predict", *arguments], capsys)
    assert code == 2, (stdout, stderr[-1500:])
    assert message in one_error(stderr), stderr
    assert not (tmp_path / "p.json").exists(), "a refused predict wrote its out"
    assert sorted(p.name for p in tmp_path.iterdir()) == existing


def test_predict_refuses_an_unwritable_out_before_the_model_is_loaded(trained_run, pictures, tmp_path, capsys, monkeypatch):
    """Concern 16: `out` under a file or a dangling link was found after inference, as a traceback."""
    from train.api import predictor as predictor_module

    def boom(path):
        raise AssertionError("the model was loaded before `out` was checked")

    monkeypatch.setattr(predictor_module, "load_model", boom)
    for out in (a_file(tmp_path) / "p.json", a_dangling_link(tmp_path, "p.json")):
        code, _, stderr = run_command(
            ["predict", f"model={trained_run}", f"data={pictures}", f"out={out}"], capsys
        )
        assert code == 2 and one_error(stderr), out


def test_predict_to_an_existing_out_is_refused_and_leaves_it_byte_identical(
    trained_run, pictures, tmp_path, capsys
):
    out = tmp_path / "taken.json"
    out.write_text("[]")
    before = sha256_of(out)
    code, _, stderr = run_command(
        ["predict", f"model={trained_run}", f"data={pictures}", f"out={out}"], capsys
    )
    assert code == 2 and "already exists" in one_error(stderr)
    assert sha256_of(out) == before


def test_the_real_command_exports_and_predicts_with_exit_zero(trained_run, pictures, tmp_path):
    """`python -m train` in a child process, from a directory that is not the repository."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    exported = run_real(
        ["export", f"model={trained_run}", f"out={tmp_path / 'm.keras'}", "format=keras"], cwd
    )
    assert exported.returncode == 0, exported.stderr[-2000:]
    assert exported.stdout == f"{tmp_path / 'm.keras'}\n"
    assert (tmp_path / "m.keras.task.json").is_file()
    assert f"Wrote {tmp_path / 'm.keras.task.json'}" in exported.stderr  # the log says so
    assert "mean and std" in exported.stderr and "raw logits" in exported.stderr
    assert not any(line.startswith("error:") for line in exported.stderr.splitlines())
    predicted = run_real(
        ["predict", f"model={trained_run}", f"data={pictures / 'a.png'}", f"out={tmp_path / 'p.json'}"], cwd
    )
    assert predicted.returncode == 0, predicted.stderr[-2000:]
    assert len(read_predictions(tmp_path / "p.json")) == 1
    assert list(cwd.iterdir()) == []


def test_the_real_command_refuses_a_bad_path_format_and_argument_with_exit_two(trained_run, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    arms = [
        ["export", f"model={tmp_path / 'gone'}", f"out={tmp_path / 'm.keras'}", "format=keras"],
        ["export", f"model={trained_run}", f"out={tmp_path / 'm.keras'}", "format=zip"],
        ["predict", f"model={trained_run}", f"data={tmp_path / 'gone'}", f"out={tmp_path / 'p.json'}"],
        ["predict", "model"],
    ]
    for args in arms:
        result = run_real(args, cwd)
        assert result.returncode == 2, (args, result.stderr[-1500:])
        assert "Traceback" not in result.stderr and error_lines(result.stderr), args


def test_train_api_export_is_the_function_after_a_command_ran_and_after_the_module_was_imported(
    trained_run, tmp_path
):
    """Concern 11: the submodule is `exporting`, so the lazy name `export` is never shadowed by it."""
    out = tmp_path / "m.keras"
    code = (
        "import sys\n"
        "from train.api import main\n"
        f"assert main(['export', 'model={trained_run}', 'out={out}', 'format=keras']) == 0\n"
        "import train.api\n"
        "function = train.api.export\n"
        "assert callable(function) and not hasattr(function, '__path__'), type(function)\n"
        "assert function.__module__ == 'train.api.exporting' and function.__name__ == 'export'\n"
        "import train.api.exporting\n"
        "assert train.api.export is function, 'importing the module changed train.api.export'\n"
        "assert train.api.exporting.FORMATS == ('keras', 'saved_model', 'tflite', 'onnx')\n"
        "from train.api import export\n"
        "assert export is function\n"
        "print('EXPORT-IS-THE-FUNCTION')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2500:]
    assert result.stdout.endswith("EXPORT-IS-THE-FUNCTION\n") and out.is_file()
