"""``python -m train help`` is the one property of the CLI with a hard limit.

The help path, ``<task> help`` included, imports no ``tensorflow``, ``keras``,
``tf_keras``, ``tensorboard``, ``train.common``, ``dl_techniques.models`` or
``yaml``, and
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

# ``cwd`` and ``source_dir`` are fixtures; importing them makes them visible here.
from .test_the_cli_exit_codes import cwd, run_injected, source_dir  # noqa: F401

FORBIDDEN = (
    "tensorflow",
    "keras",
    "tf_keras",
    "tensorboard",
    "train.common",
    "dl_techniques.models",
    "yaml",
)
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


@pytest.mark.parametrize(
    "args",
    [["fake_ok"], ["fake_ok", "help"], ["fake_ok", "-h"], ["fake_ok", "--help"]],
    ids=lambda a: " ".join(a),
)
def test_a_task_help_imports_nothing_heavy_either(args, source_dir, cwd):
    result = run_injected(args, source_dir, cwd, python_args=("-X", "importtime"))
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage: python -m train fake_ok" in result.stdout
    modules = imported_modules(result.stderr)
    # Liveness: the listing covers the CLI, and the task's config class really
    # was imported by the help path (importlib.import_module is not an
    # importtime line, so the keys it supplied are the evidence).
    assert "train.api.cli" in modules, modules[-20:]
    assert "  value  (int)  a count [default: 1]" in result.stdout, result.stdout
    assert forbidden_in(modules) == []


@pytest.mark.parametrize(
    "args",
    [["image_classification"], ["image_classification", "help"], ["image_classification", "-h"]],
    ids=lambda a: " ".join(a),
)
def test_the_real_task_help_imports_nothing_heavy_either(args, tmp_path):
    result = run_train(args, tmp_path, importtime=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage: python -m train image_classification" in result.stdout
    modules = imported_modules(result.stderr)
    # Liveness: the CLI ran, and the task's config class really was imported by the
    # help path (importlib.import_module is not an importtime line, so the keys it
    # supplied are the evidence).
    assert "train.api.cli" in modules, modules[-20:]
    assert "  data.dataset  (" in result.stdout and "  optim_args.lr  (" in result.stdout
    assert forbidden_in(modules) == []


MODEL_COMMANDS = {
    "export": ["model", "out", "format"],
    "predict": ["model", "data", "out"],
}
MODEL_COMMAND_ARMS = [
    [command, *word] for command in MODEL_COMMANDS for word in ([], ["help"], ["-h"], ["--help"])
]


@pytest.mark.parametrize("args", MODEL_COMMAND_ARMS, ids=lambda a: " ".join(a))
def test_the_export_and_predict_help_imports_nothing_heavy_and_names_its_keys(args, tmp_path):
    started = time.perf_counter()
    result = run_train(args, tmp_path, importtime=True)
    elapsed = time.perf_counter() - started
    assert result.returncode == 0, result.stderr[-2000:]
    command = args[0]
    assert f"usage: python -m train {command} " in result.stdout
    for key in MODEL_COMMANDS[command]:
        assert f"  {key}" in result.stdout and f"{key}=<{key}>" in result.stdout, key
    modules = imported_modules(result.stderr)
    assert "train.api.cli" in modules, modules[-20:]
    assert forbidden_in(modules) == []
    # `-X importtime` slows the child, so this is a loose bound: a TensorFlow import is 4 s.
    assert elapsed < 3 * TIME_LIMIT_SECONDS, elapsed


@pytest.mark.parametrize("command", sorted(MODEL_COMMANDS))
def test_the_export_and_predict_help_finishes_in_under_a_second(command, tmp_path):
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = run_train([command, "help"], tmp_path, importtime=False)
        times.append(time.perf_counter() - start)
        assert result.returncode == 0, result.stderr[-2000:]
    assert min(times) < TIME_LIMIT_SECONDS, times


@pytest.mark.parametrize("command", sorted(MODEL_COMMANDS))
def test_a_cheap_refusal_of_export_and_predict_imports_nothing_heavy_either(command, tmp_path):
    """A missing key and a bad path are refused before TensorFlow is imported (D-064)."""
    missing = "model=" + str(tmp_path / "no_such_run")
    if command == "export":
        good = [missing, "out=" + str(tmp_path / "o.keras"), "format=keras"]
    else:
        good = [missing, "data=" + str(tmp_path), "out=" + str(tmp_path / "o.json")]
    arms = [[command, "nope=1"], [command, *good]]
    for args in arms:
        result = run_train(args, tmp_path, importtime=True)
        assert result.returncode == 2, (args, result.stderr[-1500:])
        assert "Traceback" not in result.stderr
        assert forbidden_in(imported_modules(result.stderr)) == [], args


def test_load_model_and_export_are_on_the_lazy_surface_and_import_nothing_until_read(tmp_path):
    """`train.api.load_model` and `train.api.export` resolve on first read, to functions (D-064)."""
    code = (
        "import sys\n"
        "import train.api\n"
        "assert 'load_model' in train.api.__all__ and 'export' in train.api.__all__\n"
        "assert not [n for n in sys.modules if n.startswith('train.api.')], 'import loaded a submodule'\n"
        "export = train.api.export\n"
        "assert callable(export) and not hasattr(export, '__path__'), type(export)\n"
        "assert export.__module__ == 'train.api.export' and export.__name__ == 'export'\n"
        "assert 'tensorflow' not in sys.modules, 'export pulled in TensorFlow'\n"
        "load_model = train.api.load_model\n"
        "assert callable(load_model) and load_model.__module__ == 'train.api.predictor'\n"
        "assert 'tensorflow' not in sys.modules, 'load_model pulled in TensorFlow'\n"
        "print('SURFACE-OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "SURFACE-OK" in result.stdout


def test_the_command_help_lists_the_real_task(tmp_path):
    result = run_train(["help"], tmp_path, importtime=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "tasks:\n  image_classification  " in result.stdout


def test_the_forbidden_list_names_the_two_keras_backends_and_their_children():
    assert forbidden_in(
        ["tf_keras", "tf_keras.src.layers", "tensorboard", "tensorboard.compat.v1"]
    ) == ["tensorboard", "tensorboard.compat.v1", "tf_keras", "tf_keras.src.layers"]
    # A longer name that merely starts with a forbidden one is not under it.
    assert forbidden_in(["tensorboardx", "kerastuner", "yamlish"]) == []


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
    # Only the "commands:" section: registered tasks list under "tasks:".
    section = out.split("commands:\n", 1)[1].split("\n\n", 1)[0]
    listed = re.findall(r"^  (\w+)\s{2,}", section, flags=re.MULTILINE)
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


def test_the_logger_level_is_left_alone_without_the_verbose_flag(capsys):
    logger = logging.getLogger("dl")
    before = logger.level
    logger.setLevel(logging.WARNING)
    try:
        assert cli.main(["list_tasks"]) == 0
        assert logger.level == logging.WARNING
        assert cli.main(["help"]) == 0
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(before)
    capsys.readouterr()


@pytest.mark.parametrize(
    "argv",
    [["-v", "list_tasks"], ["list_tasks", "--verbose"], ["--verbose", "help", "-v"]],
    ids=str,
)
def test_the_verbose_flag_in_any_position_sets_debug_and_is_not_an_argument(argv, capsys):
    logger = logging.getLogger("dl")
    before = logger.level
    logger.setLevel(logging.WARNING)
    try:
        # A leftover flag would be refused by list_tasks and by help ("takes no arguments").
        assert cli.main(argv) == 0
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(before)
    assert capsys.readouterr().err == ""
