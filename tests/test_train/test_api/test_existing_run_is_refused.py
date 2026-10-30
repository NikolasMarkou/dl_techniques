"""A run directory that already holds a run is refused, and nothing is overwritten.

``refuse_existing_run`` alone does not fire on a directory holding only
``config_requested.json`` or only ``checkpoints/backup/`` (measured), so the API
carries its own artifact tuple. Every arm works under ``tmp_path``; ``resolve_out``
creates nothing, so the ``results/`` listing is compared before and after.
"""

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pytest

import train
from train.api.errors import ConfigError
from train.api.run_dir import (
    open_run,
    resolve_out,
    run_artifact_names,
    write_requested,
    write_resolved,
)

REPO = Path(train.__file__).resolve().parents[2]

SHARED_NAMES = (
    "results_summary.json",
    "config.json",
    "best_model.keras",
    "run.log",
    "training_log.csv",
)
API_NAMES = ("config_requested.json", "final_model.keras", "task.json", "checkpoints/backup")


def snapshot(root: Path) -> Dict[str, str]:
    """Map every path under ``root`` to a sha256 (files) or ``dir`` (directories)."""
    seen: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        seen[rel] = "dir" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return seen


def plant(run_dir: Path, name: str) -> None:
    """Create ``name`` inside ``run_dir``; ``checkpoints/backup`` is a directory."""
    target = run_dir / name
    if name == "checkpoints/backup":
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"content of {name}")


# ---------------------------------------------------------------------
# The artifact tuple
# ---------------------------------------------------------------------


def test_the_artifact_tuple_is_the_shared_five_plus_the_api_files():
    names = run_artifact_names()
    assert names == SHARED_NAMES + API_NAMES
    assert len(names) >= 9  # floor: the tuple is not silently emptied


def test_the_shared_default_tuple_is_not_widened():
    from train.common.run_artifacts import RUN_ARTIFACT_NAMES

    assert tuple(RUN_ARTIFACT_NAMES) == SHARED_NAMES


@pytest.mark.parametrize("name", SHARED_NAMES + API_NAMES)
def test_each_artifact_alone_is_refused_and_nothing_changes(tmp_path, name):
    assert name in run_artifact_names()  # the table and the module agree
    run_dir = tmp_path / "run"
    plant(run_dir, name)
    before = snapshot(tmp_path)

    with pytest.raises(ConfigError) as caught:
        open_run(run_dir, resume=False)

    assert name.split("/")[0] in str(caught.value)
    assert "'out'" in str(caught.value)
    assert snapshot(tmp_path) == before


def test_every_name_in_the_modules_tuple_is_in_the_parametrized_list():
    assert set(run_artifact_names()) == set(SHARED_NAMES + API_NAMES)


# ---------------------------------------------------------------------
# Accepted directories
# ---------------------------------------------------------------------


def test_a_missing_directory_is_created_with_its_parents(tmp_path):
    run_dir = tmp_path / "a" / "b" / "run"
    assert open_run(run_dir, resume=False) == run_dir
    assert run_dir.is_dir()


def test_an_empty_directory_is_accepted(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert open_run(run_dir, resume=False) == run_dir
    assert snapshot(tmp_path) == {"run": "dir"}


def test_unrelated_files_and_a_bare_checkpoints_directory_are_accepted(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "notes.txt").write_text("mine")
    before = snapshot(tmp_path)
    assert open_run(run_dir, resume=False) == run_dir
    assert snapshot(tmp_path) == before


# ---------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------


def test_resume_without_a_backup_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    with pytest.raises(ConfigError, match="checkpoints/backup"):
        open_run(run_dir, resume=True)


def test_resume_of_a_directory_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="checkpoints/backup"):
        open_run(tmp_path / "missing", resume=True)
    assert not (tmp_path / "missing").exists()


def test_resume_with_a_backup_returns_the_path_and_touches_nothing(tmp_path):
    run_dir = tmp_path / "run"
    plant(run_dir, "checkpoints/backup")
    plant(run_dir, "config_requested.json")
    plant(run_dir, "training_log.csv")
    before = snapshot(tmp_path)
    assert open_run(run_dir, resume=True) == run_dir
    assert snapshot(tmp_path) == before


def test_resume_of_a_finished_run_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    plant(run_dir, "checkpoints/backup")
    plant(run_dir, "results_summary.json")
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="results_summary.json"):
        open_run(run_dir, resume=True)
    assert snapshot(tmp_path) == before


# ---------------------------------------------------------------------
# Writing the two configs
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Resolved:
    epochs: int = 3
    lr: float = 0.001


def test_a_new_run_writes_only_inside_its_run_directory(tmp_path):
    outside = tmp_path / "bystander"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    before_outside = snapshot(outside)

    run_dir = tmp_path / "work" / "run"
    open_run(run_dir, resume=False)
    write_requested(run_dir, {"epochs": 3, "lr": "auto"})
    write_resolved(run_dir, Resolved())

    assert snapshot(outside) == before_outside
    assert set(snapshot(tmp_path / "work")) == {
        "run",
        "run/config_requested.json",
        "run/config.json",
    }
    assert json.loads((run_dir / "config_requested.json").read_text()) == {
        "epochs": 3,
        "lr": "auto",
    }
    assert json.loads((run_dir / "config.json").read_text()) == {"epochs": 3, "lr": 0.001}


def test_the_request_alone_makes_the_same_out_refused(tmp_path):
    """A failure during data load leaves ``config_requested.json``; that is deliberate."""
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    write_requested(run_dir, {"epochs": 3})
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="config_requested.json"):
        open_run(run_dir, resume=False)
    assert snapshot(tmp_path) == before


# ---------------------------------------------------------------------
# resolve_out
# ---------------------------------------------------------------------


def test_auto_goes_under_the_repo_results_directory_and_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = REPO / "results"
    listing_before = sorted(p.name for p in results.iterdir()) if results.is_dir() else None

    run_dir = resolve_out("auto", "image_classification", "vit", "vit_pico")

    assert run_dir.parent == results
    assert re.fullmatch(r"image_classification_vit_vit_pico_\d{8}_\d{6}", run_dir.name)
    assert not run_dir.exists()
    assert not (tmp_path / "results").exists()
    listing_after = sorted(p.name for p in results.iterdir()) if results.is_dir() else None
    assert listing_after == listing_before


def test_a_relative_path_is_anchored_at_the_repo_root_not_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    run_dir = resolve_out("a_relative_probe/run_one", "t", "f", "v")
    assert run_dir == REPO / "a_relative_probe" / "run_one"
    assert not run_dir.exists()
    assert snapshot(tmp_path) == {}


def test_an_absolute_path_is_kept(tmp_path):
    target = tmp_path / "somewhere" / "run"
    assert resolve_out(str(target), "t", "f", "v") == target
    assert resolve_out(target, "t", "f", "v") == target
    assert not target.exists()


@pytest.mark.parametrize("out", ["", "   ", ".", "..", "a/..", "./..", "/"])
def test_a_path_that_does_not_name_a_run_directory_is_refused(out):
    with pytest.raises(ConfigError, match="'out'"):
        resolve_out(out, "t", "f", "v")


def test_a_path_naming_an_existing_file_is_refused(tmp_path):
    victim = tmp_path / "a_file"
    victim.write_text("data")
    with pytest.raises(ConfigError, match="existing file"):
        resolve_out(str(victim), "t", "f", "v")
    assert victim.read_text() == "data"


def test_a_value_that_is_not_a_path_is_refused():
    with pytest.raises(ConfigError, match="expected a path"):
        resolve_out(5, "t", "f", "v")
    with pytest.raises(ConfigError, match="expected a path"):
        resolve_out(None, "t", "f", "v")


def test_two_auto_runs_in_one_second_share_a_name_and_the_second_is_refused(tmp_path, monkeypatch):
    """The 1 s timestamp collides; the shared refusal is what stops the second run."""
    import train.common.run_io as run_io

    monkeypatch.setattr(run_io, "run_timestamp", lambda: "20260101_000000")
    first = resolve_out("auto", "t", "f", "v")
    second = resolve_out("auto", "t", "f", "v")
    assert first == second
    # The same collision under tmp_path, where it is safe to create the directory.
    run_dir = tmp_path / first.name
    open_run(run_dir, resume=False)
    write_requested(run_dir, {"epochs": 1})
    with pytest.raises(ConfigError):
        open_run(run_dir, resume=False)


# ---------------------------------------------------------------------
# The module stays off the help path
# ---------------------------------------------------------------------


def test_importing_run_dir_loads_no_heavy_module():
    code = (
        "import sys; import train.api.run_dir; "
        "bad = [m for m in ('tensorflow', 'keras', 'train.common', 'dl_techniques.models', 'yaml') "
        "if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
