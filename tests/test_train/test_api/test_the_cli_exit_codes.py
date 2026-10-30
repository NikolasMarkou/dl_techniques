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
    assert result.stderr == ""


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
