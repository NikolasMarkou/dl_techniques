"""The Python function and ``python -m train`` are one entry point (D-008), observed three ways.

* **Same config** (D-018). ``out`` is part of ``config.json`` and a second run into
  the same ``out`` is refused, so the function runs into ``runs/run``, that
  directory is renamed inside ``tmp_path``, and the CLI then runs into the
  original path. ``config.json`` and ``config_requested.json`` must be equal byte
  for byte. The tokens spell the same values in every shape the coercion handles
  (an int for a float field, a bool, a nested dict, a float in scientific form).
  The vacuity arm runs a third time with ``seed=8``: the files must then differ,
  and in the ``seed`` key only, so an equal pair cannot come from a comparison
  that compares nothing.
* **Same seed.** ``train.common.seed.set_seeds`` is replaced by a recorder that
  stops the run; both entry points hand it ``config.seed``, and the default is 42.
* **Same place.** The runs of the first arm write under ``out`` and nowhere
  else: not into the working directory, not beside ``out``, and not into the
  repo-root ``results/`` (watched by the names it holds; this module never
  creates, deletes or cleans anything there).

The fits are real ones of ``vit/vit_pico`` on fake CIFAR-10 arrays, on the CPU,
for one epoch with the figures and the model analysis off.
"""

import json
import os
from pathlib import Path

import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.tasks import register_builtin_tasks
from train.api.tasks.image_classification import train_image_classification

from tests.classification_fakes import use_tiny_raw
from tests.results_session_guard import results_entries, results_root

#: The run, as a Python caller writes it.
KEYWORDS = {
    "model": "vit/vit_pico",
    "data": {"dataset": "cifar10"},
    "epochs": 1,
    "batch_size": 4,
    "seed": 7,
    "model_args": {"dropout_rate": 0.2},
    "optim_args": {"lr": 1e-3, "warmup_start_lr": 0},
    "metric_args": {"success_threshold": 0},
    "analysis_args": {"model_analysis": False, "figures": False},
    "debug_args": {"terminate_on_nan": True},
}

#: The same run, as a command line writes it. `warmup_start_lr=0` and
#: `success_threshold=0` are ints for float fields (converted to 0.0 on both sides).
TOKENS = (
    "model=vit/vit_pico",
    "data.dataset=cifar10",
    "epochs=1",
    "batch_size=4",
    "seed=7",
    "model_args.dropout_rate=0.2",
    "optim_args.lr=1e-3",
    "optim_args.warmup_start_lr=0",
    "metric_args.success_threshold=0",
    "analysis_args.model_analysis=false",
    "analysis_args.figures=false",
    "debug_args.terminate_on_nan=true",
)


@pytest.fixture(scope="module")
def world():
    """Fake arrays; an empty registry holding the six families and the real task row."""
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def tree(root: Path):
    """Every entry under ``root`` as ``(relative path, kind, size, mtime_ns)``; ``()`` when absent."""
    if not root.exists():
        return ()
    found = []
    for path in sorted(root.rglob("*")):
        status = path.stat()
        kind = "dir" if path.is_dir() else "file"
        found.append((path.relative_to(root).as_posix(), kind, status.st_size, status.st_mtime_ns))
    return tuple(found)


def outside(base: Path, *inside: str):
    """The tree of ``base`` without the subtrees named in ``inside`` (the run directories)."""
    return tuple(row for row in tree(base) if row[0].split("/")[0] not in inside)


class Runs:
    """The directories and exit codes of the three runs of :func:`runs`."""


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    """Run the function, then the CLI, then the CLI with another seed, all into ``runs/run``."""
    base = tmp_path_factory.mktemp("agree")
    (base / "cwd").mkdir()
    (base / "neighbour.txt").write_text("untouched")
    (base / "runs").mkdir()
    out = base / "runs" / "run"

    found = Runs()
    found.base, found.out = base, out
    found.outside_before = outside(base, "runs")
    found.cwd_before = tree(base / "cwd")
    found.results_before = results_entries(results_root())

    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(base / "cwd")
        train_image_classification(out=str(out), **KEYWORDS)
        out.rename(base / "runs" / "function")

        found.cli_code = cli.main(["image_classification", f"out={out}", *TOKENS])
        out.rename(base / "runs" / "cli")

        tokens = ["seed=8" if token == "seed=7" else token for token in TOKENS]
        found.other_code = cli.main(["image_classification", f"out={out}", *tokens])
        out.rename(base / "runs" / "other_seed")

    found.outside_after = outside(base, "runs")
    found.cwd_after = tree(base / "cwd")
    found.results_after = results_entries(results_root())
    return found


def read(base: Path, name: str, file: str) -> bytes:
    return (base / "runs" / name / file).read_bytes()


# ---------------------------------------------------------------------
# The same config
# ---------------------------------------------------------------------


def test_the_runs_were_real_and_the_cli_exited_zero(runs):
    assert runs.cli_code == 0 and runs.other_code == 0
    for name in ("function", "cli", "other_seed"):
        assert (runs.base / "runs" / name / "final_model.keras").is_file(), name


def test_the_config_is_the_same_bytes_whichever_entry_point_ran(runs):
    resolved = json.loads(read(runs.base, "function", "config.json"))
    # The renamed run kept the `out` it was started with: the premise of the compare.
    assert resolved["out"] == str(runs.out)
    assert read(runs.base, "function", "config.json") == read(runs.base, "cli", "config.json")


def test_the_request_is_the_same_bytes_whichever_entry_point_ran(runs):
    assert read(runs.base, "function", "config_requested.json") == read(
        runs.base, "cli", "config_requested.json"
    )
    requested = json.loads(read(runs.base, "cli", "config_requested.json"))
    assert requested["epochs"] == 1 and requested["optim_args"]["lr"] == 1e-3
    # An int for a float field is written as the float on both sides.
    assert requested["metric_args"]["success_threshold"] == 0.0
    assert isinstance(requested["optim_args"]["warmup_start_lr"], float)


def test_vacuity_one_changed_value_changes_the_bytes_in_that_key_only(runs):
    first = json.loads(read(runs.base, "cli", "config.json"))
    other = json.loads(read(runs.base, "other_seed", "config.json"))
    assert read(runs.base, "cli", "config.json") != read(runs.base, "other_seed", "config.json")
    changed = {key for key in first if first[key] != other[key]}
    assert changed == {"seed"} and first["out"] == other["out"]


# ---------------------------------------------------------------------
# The same seed
# ---------------------------------------------------------------------


class Stop(Exception):
    """Raised by the recorder, so a seed arm ends right after seeding."""


@pytest.fixture
def seeded(world, monkeypatch):
    """Replace ``set_seeds`` with a recorder that stops the run; return its list of calls."""
    import train.common.seed as seed_module

    calls = []

    def record(seed):
        calls.append(seed)
        raise Stop

    monkeypatch.setattr(seed_module, "set_seeds", record)
    return calls


def call_function(out, seed):
    given = {key: value for key, value in KEYWORDS.items() if key != "seed"}
    if seed is not None:
        given["seed"] = seed
    train_image_classification(out=str(out), **given)


def call_cli(out, seed):
    tokens = [token for token in TOKENS if not token.startswith("seed=")]
    if seed is not None:
        tokens.append(f"seed={seed}")
    cli.main(["image_classification", f"out={out}", *tokens])


@pytest.mark.parametrize("entry", [call_function, call_cli], ids=["function", "cli"])
@pytest.mark.parametrize("given, expected", [(7, 7), (None, 42)], ids=["seed-7", "default"])
def test_set_seeds_receives_the_seed_of_the_config(entry, given, expected, seeded, tmp_path):
    with pytest.raises(Stop):
        entry(tmp_path / "run", given)
    assert seeded == [expected]


# ---------------------------------------------------------------------
# The same place
# ---------------------------------------------------------------------


def test_the_snapshot_sees_a_file_written_to_the_working_directory(tmp_path):
    # Liveness of the instrument the next arms rely on.
    (tmp_path / "cwd").mkdir()
    before = tree(tmp_path / "cwd")
    (tmp_path / "cwd" / "stray.txt").write_text("x")
    assert tree(tmp_path / "cwd") != before
    assert outside(tmp_path, "runs") != ()


def test_a_run_writes_nothing_into_the_working_directory(runs):
    assert runs.cwd_before == ()
    assert runs.cwd_after == ()


def test_a_run_writes_nothing_beside_its_out(runs):
    assert runs.outside_after == runs.outside_before
    assert "neighbour.txt" in {row[0] for row in runs.outside_after}


def test_a_run_adds_no_entry_to_the_repo_results_directory(runs):
    assert runs.results_after - runs.results_before == frozenset()
    assert os.path.isabs(str(runs.out)) and results_root() not in runs.out.parents
