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
from types import SimpleNamespace

from train.api.config import StrictConfig, validate


@dataclasses.dataclass(frozen=True)
class FakeConfig(StrictConfig):
    value: int = dataclasses.field(default=1, metadata={"help": "a count"})
    rate: float = dataclasses.field(default=0.5, metadata={"help": "a rate"})


def _announce(name, kwargs):
    config = validate(FakeConfig, kwargs)
    sys.stdout.write(f"FAKE-RAN {name} value={config.value!r} rate={config.rate!r}\\n")
    sys.stdout.flush()


def run_ok(**kwargs):
    _announce("ok", kwargs)
    return SimpleNamespace(success=True)


def run_unmet(**kwargs):
    _announce("unmet", kwargs)
    return SimpleNamespace(success=False)


def run_raises(**kwargs):
    _announce("raises", kwargs)
    raise RuntimeError("the fake task blew up")


def run_interrupted(**kwargs):
    _announce("interrupted", kwargs)
    raise KeyboardInterrupt
'''

# The driver registers the fake tasks, then runs the real ``train.__main__``.
# It reads sys.argv[1:] the way ``python -m train`` does, so the arguments
# after ``-c`` are the command line under test.
DRIVER = f'''
import runpy

from train.api.registry import TaskSpec, register_task

for name in ("ok", "unmet", "raises", "interrupted"):
    register_task(
        TaskSpec(
            name=f"fake_{{name}}",
            config_class="{MODULE}:FakeConfig",
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


def run_injected(args: Sequence[str], source_dir: Path, cwd: Path):
    """Run ``python -m train <args>`` with the fake tasks registered first."""
    existing = os.environ.get("PYTHONPATH")
    path = os.pathsep.join([str(source_dir)] + ([existing] if existing else []))
    env = {**os.environ, "PYTHONPATH": path}
    return subprocess.run(
        [sys.executable, "-c", DRIVER, *args],
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
    for name in ("fake_ok", "fake_unmet", "fake_raises", "fake_interrupted"):
        assert name in result.stdout


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
