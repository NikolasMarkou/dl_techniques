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

import ast
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import pytest

from train.api import cli
from train.api.tasks import BUILTIN_TASKS

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


# The fresh-interpreter census of what a help command leaves in ``sys.modules``.
#
# ``-X importtime`` above lists imports; this reads ``sys.modules`` itself, so a heavy library that arrives
# through any path (a module-scope import, a lazy import run too early) is found by name. It does not use the
# importtime FORBIDDEN list: that list pre-dates the forecasting task, whose csv reader needs numpy and pandas,
# and the list let a module-scope ``import numpy`` through for seven steps (the rule is test_convert's and
# AGENTS.md's: help and ``import train.api`` load no heavy library).
HEAVY_LIBRARIES = ("tensorflow", "keras", "tf_keras", "tensorboard", "numpy", "pandas", "PIL", "yaml",
                   "scipy", "sklearn", "matplotlib")
#: ``train.api`` logs through ``dl_techniques.utils.logger``; those three module names are the whole of the
#: ``dl_techniques`` package a help command may load (the library's layers, models and losses are TensorFlow).
LOGGER_ONLY = ("dl_techniques", "dl_techniques.utils", "dl_techniques.utils.logger")

_CENSUS_DRIVER = """
import sys

HEAVY = {heavy!r}
LOGGER_ONLY = {logger_only!r}


def census():
    # One name per heavy library (``numpy``, not its 90 submodules), the full name for the rest.
    found = set()
    for name in sys.modules:
        top = name.split(".")[0]
        if top in HEAVY:
            found.add(top)
        elif top == "dl_techniques" and name not in LOGGER_ONLY:
            found.add(name)
        elif name == "train.common" or name.startswith("train.common."):
            found.add(name)
    print("CENSUS", sorted(found))


{body}
"""


def heavy_after(body: str, cwd: Path, argv: Sequence[str] = ()):
    """Run ``body`` in a fresh interpreter; return ``(returncode, stdout, stderr, modules found)``."""
    code = _CENSUS_DRIVER.format(heavy=HEAVY_LIBRARIES, logger_only=LOGGER_ONLY, body=body)
    result = subprocess.run(
        [sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True, timeout=120
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("CENSUS ")]
    found = ast.literal_eval(lines[-1][len("CENSUS "):]) if lines else None
    return result.returncode, result.stdout, result.stderr, found


def every_command_word() -> List[str]:
    """The command words of the CLI: the rows of ``COMMANDS`` and the rows of the built-in task table."""
    return [*cli.COMMANDS, *(spec.name for spec in BUILTIN_TASKS)]


_HELP_BODY = """
from train.api import cli

code = cli.main(sys.argv[1:])
census()
sys.exit(code)
"""

#: The one command whose listing imports the model families (``cli._list_models``: "not on the help path").
IMPORTS_THE_FAMILIES = ("list_models",)


def help_argv(word: str) -> List[str]:
    """The command line that prints ``word``'s help: ``help`` and the listings take no ``help`` word."""
    return [word] if word == "help" else [word, "help"]


def test_the_command_words_cover_the_commands_and_the_task_table():
    words = every_command_word()
    assert words[: len(cli.COMMANDS)] == list(cli.COMMANDS)
    # The forecasting task is the one whose module imports once pulled numpy in; it must be a row here.
    assert "time_series_forecasting" in words and "image_classification" in words
    assert len(words) == len(set(words)) == len(cli.COMMANDS) + len(BUILTIN_TASKS)
    assert set(IMPORTS_THE_FAMILIES) <= set(cli.COMMANDS)


@pytest.mark.parametrize("word", [w for w in every_command_word() if w not in IMPORTS_THE_FAMILIES])
def test_the_help_of_every_command_and_task_leaves_no_heavy_library_in_sys_modules(word, tmp_path):
    code, out, err, found = heavy_after(_HELP_BODY, tmp_path, help_argv(word))
    assert code == 0, err[-2000:]
    # Liveness: the command really printed (usage text, or the task rows of the listing).
    assert ("usage:" in out) or (word == "list_tasks" and "time_series_forecasting" in out), out[-500:]
    assert found is not None, out[-500:]
    assert found == [], f"`python -m train {' '.join(help_argv(word))}` loaded {found}"
    if word == "time_series_forecasting":
        # What the task's own help test used to assert, kept with the census that replaced it: the help names the knobs.
        assert "jit_compile" in out and "data.lookback" in out, out[-800:]


def test_the_listing_of_the_models_is_the_one_command_that_may_import_the_families(tmp_path):
    """The exemption is real and exact: ``list_models`` imports keras, so it is not on the help path."""
    code, out, err, found = heavy_after(_HELP_BODY, tmp_path, ["list_models", "help"])
    assert code == 0, err[-2000:]
    assert "keras" in found, found


def test_importing_train_api_leaves_no_heavy_library_and_no_dl_techniques_module(tmp_path):
    code, out, err, found = heavy_after("import train.api\ncensus()\nassert 'dl_techniques' not in sys.modules\n", tmp_path)
    assert code == 0, err[-2000:]
    assert found == [], found


def test_the_census_flags_numpy_pandas_and_a_dl_techniques_module_when_they_really_load(tmp_path):
    """Control: the same driver names numpy, pandas and a dl_techniques layer module that is really imported."""
    body = "import numpy, pandas\nimport dl_techniques.layers\ncensus()\n"
    code, out, err, found = heavy_after(body, tmp_path)
    assert code == 0, err[-2000:]
    assert found == ["dl_techniques.layers", "numpy", "pandas"], found
    # The logger trio alone is not flagged.
    code, out, err, found = heavy_after("import dl_techniques.utils.logger\ncensus()\n", tmp_path)
    assert code == 0, err[-2000:]
    assert found == [], found


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
    arms = [[command, "nope=1"], [command, *good], [command, "model=", *good[1:]]]
    if command == "export":
        # The model path is checked before `tf2onnx` (and so TensorFlow) is imported (concern 16).
        arms.append([command, missing, "out=" + str(tmp_path / "o.onnx"), "format=onnx"])
    else:
        (tmp_path / "afile").write_text("x")
        arms += [
            [command, "model=" + str(tmp_path), "data=" + str(tmp_path), "out=" + str(tmp_path / "afile" / "o.json")],
            [command, "model=" + str(tmp_path), "data=", "out=" + str(tmp_path / "o.json")],
        ]
    for args in arms:
        result = run_train(args, tmp_path, importtime=True)
        assert result.returncode == 2, (args, result.stderr[-1500:])
        assert "Traceback" not in result.stderr
        assert forbidden_in(imported_modules(result.stderr)) == [], args


def test_load_model_and_export_are_on_the_lazy_surface_and_import_nothing_until_read(tmp_path):
    """`train.api.load_model` and `train.api.export` resolve on first read, to functions (D-064, D-068)."""
    code = (
        "import sys\n"
        "import train.api\n"
        "assert 'load_model' in train.api.__all__ and 'export' in train.api.__all__\n"
        "assert not [n for n in sys.modules if n.startswith('train.api.')], 'import loaded a submodule'\n"
        "export = train.api.export\n"
        "assert callable(export) and not hasattr(export, '__path__'), type(export)\n"
        "assert export.__module__ == 'train.api.exporting' and export.__name__ == 'export'\n"
        "assert 'tensorflow' not in sys.modules, 'export pulled in TensorFlow'\n"
        "import train.api.exporting\n"
        "assert train.api.export is export, 'importing the module replaced the function (D-068)'\n"
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
    first = min(spec.name for spec in BUILTIN_TASKS)
    assert f"tasks:\n  {first}  " in result.stdout and "\n  image_classification  " in result.stdout


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
