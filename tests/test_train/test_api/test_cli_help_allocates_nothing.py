"""``python -m train help`` is the one property of the CLI with a hard limit.

The help path, ``<task> help`` included, imports no ``tensorflow``, ``keras``,
``train.common``, ``dl_techniques.models`` or ``yaml``, and
``python -m train help`` finishes in under one second. Every ``train.common``
import costs about 4.2 s and loads TensorFlow (D-012), so one stray
module-scope import is enough to break it.

The test runs the real command in a child process under
``-X importtime`` and reads the module names from stderr. A stand-in driver
that inspects ``sys.modules`` would not run ``train/__main__.py``.
"""

import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import pytest

from train.api import cli

FORBIDDEN = ("tensorflow", "keras", "train.common", "dl_techniques.models", "yaml")
HELP_ARMS = [["help"], ["-h"], ["--help"], []]
RUN_ARMS = [["list_tasks"], ["list_tasks", "help"]]
TIME_LIMIT_SECONDS = 1.0

_IMPORT_LINE = re.compile(r"^import time:\s+\d+ \|\s+\d+ \|\s+(\S+)\s*$")


def imported_modules(stderr: str) -> List[str]:
    """Return the module names listed by ``-X importtime`` on ``stderr``."""
    found = []
    for line in stderr.splitlines():
        match = _IMPORT_LINE.match(line)
        if match:
            found.append(match.group(1))
    return found


def forbidden_in(modules: Sequence[str]) -> List[str]:
    """Return the modules that are, or live under, a forbidden package."""
    return sorted(
        name
        for name in set(modules)
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)
    )


def run_train(args: Sequence[str], cwd: Path, importtime: bool):
    command = [sys.executable]
    if importtime:
        command += ["-X", "importtime"]
    command += ["-m", "train", *args]
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("args", HELP_ARMS + RUN_ARMS, ids=lambda a: " ".join(a) or "no-args")
def test_the_help_path_imports_nothing_heavy(args, tmp_path):
    result = run_train(args, tmp_path, importtime=True)
    assert result.returncode == 0, result.stderr[-2000:]
    modules = imported_modules(result.stderr)
    # Liveness: the listing really covers the CLI, so an empty parse cannot pass.
    assert "train.api.cli" in modules, modules[-20:]
    assert forbidden_in(modules) == []


@pytest.mark.parametrize("args", HELP_ARMS, ids=lambda a: " ".join(a) or "no-args")
def test_the_help_forms_print_usage_on_stdout(args, tmp_path):
    result = run_train(args, tmp_path, importtime=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


@pytest.mark.parametrize("args", RUN_ARMS, ids=lambda a: " ".join(a))
def test_a_list_command_runs_instead_of_printing_help(args, tmp_path):
    result = run_train(args, tmp_path, importtime=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" not in result.stdout
    assert result.stdout.strip()


def test_the_listing_helper_flags_a_forbidden_import(tmp_path):
    """Control: the same helper flags ``yaml`` when it really is imported."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import yaml"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "yaml" in forbidden_in(imported_modules(result.stderr))


def test_help_finishes_in_under_a_second(tmp_path):
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = run_train(["help"], tmp_path, importtime=False)
        times.append(time.perf_counter() - start)
        assert result.returncode == 0, result.stderr[-2000:]
    assert min(times) < TIME_LIMIT_SECONDS, times


def test_help_lists_every_command_in_the_table_and_nothing_else(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    listed = re.findall(r"^  (\w+)\s{2,}", out, flags=re.MULTILINE)
    assert listed == list(cli.COMMANDS)


def test_verbose_flag_anywhere_sets_the_dl_logger_to_debug(capsys):
    logger = logging.getLogger("dl")
    before = logger.level
    try:
        assert cli.main(["list_tasks", "-v"]) == 0
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(before)
    capsys.readouterr()
