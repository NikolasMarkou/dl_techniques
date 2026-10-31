"""A run directory that already holds a run is refused, and nothing is overwritten.

``refuse_existing_run`` alone does not fire on a directory holding only
``config_requested.json`` or only ``checkpoints/backup/`` (measured), so the API
carries its own artifact tuple. Every arm works under ``tmp_path``; ``resolve_out``
creates nothing, so the ``results/`` listing is compared before and after.
"""

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

import train
from train.api.args import OptimArgs
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
        if path.is_symlink():
            seen[rel] = "link->" + os.readlink(path)
        elif path.is_dir():
            seen[rel] = "dir"
        else:
            seen[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
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
    assert "--experiment-name" not in str(caught.value)
    assert "out=auto" in str(caught.value)
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


def test_an_empty_directory_is_accepted_and_claimed(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert open_run(run_dir, resume=False) == run_dir
    assert set(snapshot(tmp_path)) == {"run", "run/config_requested.json"}
    assert (run_dir / "config_requested.json").read_bytes() == b""


@pytest.mark.parametrize("held", ["notes.txt", ".hidden", "checkpoints/", "tensorboard/x"])
def test_a_non_empty_directory_is_refused_and_nothing_changes(tmp_path, held):
    """Any content refuses a new run: the run layout has more files than the tuple names."""
    run_dir = tmp_path / "run"
    if held.endswith("/"):
        (run_dir / held).mkdir(parents=True)
    else:
        (run_dir / held).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / held).write_text("mine")
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="not empty") as caught:
        open_run(run_dir, resume=False)
    assert "'out'" in str(caught.value)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("name", run_artifact_names())
def test_a_run_nested_inside_a_used_directory_is_refused(tmp_path, name):
    parent = tmp_path / "finished"
    plant(parent, name)
    before = snapshot(tmp_path)
    for nested in ("nested", "checkpoints/backup/x", "a/b/c"):
        with pytest.raises(ConfigError, match="lies inside") as caught:
            open_run(parent / nested, resume=False)
        assert name.split("/")[0] in str(caught.value)
        assert "'out'" in str(caught.value)
    assert snapshot(tmp_path) == before


def test_a_directory_beside_a_finished_run_is_accepted(tmp_path):
    plant(tmp_path / "finished", "results_summary.json")
    assert open_run(tmp_path / "sibling", resume=False) == tmp_path / "sibling"


#: Files of a run's layout that are not all in the artifact tuple; none may
#: be deleted by a call that accepts the directory.
RUN_TREE = (
    "config.json",
    "training_history.json",
    "model_summary.txt",
    "best_model.keras",
    "tensorboard/events.out",
    "visualizations/curves.png",
    "model_analysis/a.json",
    "epoch_analysis/e1/x.png",
    "checkpoints/fit_state.json",
    "conformal.json",
    "notes.json",
)


def test_accepting_a_directory_deletes_nothing_beside_a_finished_run(tmp_path):
    finished = tmp_path / "finished"
    for name in (*RUN_TREE, "results_summary.json"):
        plant(finished, name)
    (tmp_path / "empty_sibling").mkdir()  # an empty directory is not ours to remove
    before = snapshot(tmp_path)
    assert open_run(tmp_path / "sibling", resume=False) == tmp_path / "sibling"
    assert open_run(tmp_path / "new_parent" / "run", resume=False)
    after = snapshot(tmp_path)
    assert {key: after.get(key) for key in before} == before, "something under the finished run changed"
    assert set(after) - set(before) == {
        "sibling", "sibling/config_requested.json",
        "new_parent", "new_parent/run", "new_parent/run/config_requested.json",
    }


def test_the_writers_delete_nothing_that_is_already_in_the_run_directory(tmp_path):
    # Between open_run and the writers a trainer puts files in the directory
    # (a log, a stale config.json from a crashed attempt, an empty directory).
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    for name in ("config.json", "run.log", "best_model.keras", "notes.json", "tensorboard/x"):
        plant(run_dir, name)
    (run_dir / "empty_dir").mkdir()
    before = snapshot(tmp_path)
    write_requested(run_dir, {"epochs": 1})
    after = snapshot(tmp_path)
    assert {key: after.get(key) for key in before if key != "run/config_requested.json"} == {
        key: value for key, value in before.items() if key != "run/config_requested.json"
    }
    before = after
    write_resolved(run_dir, {"epochs": 1})
    after = snapshot(tmp_path)
    assert {key: after.get(key) for key in before if key != "run/config.json"} == {
        key: value for key, value in before.items() if key != "run/config.json"
    }


def test_the_two_config_writers_add_files_and_delete_none(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    write_requested(run_dir, {"epochs": 1})
    after_request = snapshot(tmp_path)
    write_resolved(run_dir, {"epochs": 1})
    after_resolved = snapshot(tmp_path)
    assert set(after_request) == {"run", "run/config_requested.json"}
    assert set(after_resolved) == {"run", "run/config_requested.json", "run/config.json"}
    assert after_resolved["run/config_requested.json"] == after_request["run/config_requested.json"]


# ---------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------


def test_a_second_open_with_no_write_between_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="config_requested.json"):
        open_run(run_dir, resume=False)
    assert snapshot(tmp_path) == before


def test_write_requested_replaces_the_claim(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    write_requested(run_dir, {"epochs": 2})
    assert json.loads((run_dir / "config_requested.json").read_text()) == {"epochs": 2}


_RACER = """
import json, sys, time
from pathlib import Path
from train.api.errors import ConfigError
from train.api.run_dir import open_run

root, tag, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
open_run(root / ("warm_" + tag), False)  # the heavy imports happen before the race
(root / ("ready_" + tag)).write_text("")
while not (root / "go").exists():
    pass
results = []
for i in range(count):
    try:
        open_run(root / f"run_{i}", False)
        results.append("ok")
    except ConfigError:
        results.append("refused")
print(json.dumps(results))
"""


def test_two_processes_opening_the_same_directories_never_both_win(tmp_path):
    """Two real processes, released together, open the same 150 new directories."""
    count = 150
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg"}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RACER, str(tmp_path), tag, str(count)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for tag in ("a", "b")
    ]
    try:
        deadline = time.time() + 180
        while not all((tmp_path / f"ready_{t}").exists() for t in ("a", "b")):
            assert time.time() < deadline, "the racers never got ready"
            assert all(p.poll() is None for p in procs), [p.communicate() for p in procs]
            time.sleep(0.05)
        (tmp_path / "go").write_text("")
        outputs = [p.communicate(timeout=180) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert [p.returncode for p in procs] == [0, 0], outputs
    first, second = (json.loads(out) for out, _ in outputs)
    assert len(first) == len(second) == count
    for i, pair in enumerate(zip(first, second)):
        assert sorted(pair) == ["ok", "refused"], (i, pair)
    for i in range(count):
        assert (tmp_path / f"run_{i}" / "config_requested.json").is_file()


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


def test_resume_where_the_backup_is_a_plain_file_is_refused(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "backup").write_text("not a directory")
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="checkpoints/backup"):
        open_run(run_dir, resume=True)
    assert snapshot(tmp_path) == before


def test_resume_leaves_a_whole_unfinished_run_tree_byte_identical(tmp_path):
    run_dir = tmp_path / "run"
    for name in RUN_TREE:
        plant(run_dir, name)
    plant(run_dir, "checkpoints/backup")
    (run_dir / "checkpoints" / "backup" / "state.json").write_text("{}")
    # The sidecar of an unfinished run (`plant` writes text, which is not JSON).
    (run_dir / "checkpoints" / "fit_state.json").write_text('{"epochs": {}, "fit_finished": false}')
    before = snapshot(tmp_path)
    assert len(before) >= len(RUN_TREE), "the tree was not planted"
    assert open_run(run_dir, resume=True) == run_dir
    assert snapshot(tmp_path) == before


def test_resume_of_a_run_whose_fit_returned_is_refused_though_it_has_no_summary(tmp_path):
    run_dir = tmp_path / "run"
    plant(run_dir, "checkpoints/backup")
    (run_dir / "checkpoints" / "fit_state.json").write_text('{"epochs": {}, "fit_finished": true}')
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="fit_finished"):
        open_run(run_dir, resume=True)
    assert snapshot(tmp_path) == before


def test_resume_with_an_unreadable_sidecar_is_refused_and_names_the_file(tmp_path):
    run_dir = tmp_path / "run"
    plant(run_dir, "checkpoints/backup")
    plant(run_dir, "checkpoints/fit_state.json")  # text, not JSON
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="fit_state.json is unreadable"):
        open_run(run_dir, resume=True)
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


@dataclasses.dataclass(frozen=True)
class Plain:
    lr: float = 0.5
    pair: tuple = (1, 2)


def test_write_requested_accepts_the_group_instances_validate_accepts(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    mapping = {
        "epochs": 3,
        "optim_args": OptimArgs(lr=1.0),
        "nested": {"inner": Plain(), "values": (1, 2.5, "auto")},
    }
    write_requested(run_dir, mapping)
    written = json.loads((run_dir / "config_requested.json").read_text())
    assert written == {
        "epochs": 3,
        "optim_args": dataclasses.asdict(OptimArgs(lr=1.0)),
        "nested": {"inner": {"lr": 0.5, "pair": [1, 2]}, "values": [1, 2.5, "auto"]},
    }
    assert written["optim_args"]["lr"] == 1.0
    assert written["optim_args"]["optimizer"] == "auto"


def test_write_requested_accepts_a_top_level_dataclass(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    write_requested(run_dir, Plain())
    assert json.loads((run_dir / "config_requested.json").read_text()) == {
        "lr": 0.5,
        "pair": [1, 2],
    }


@pytest.mark.parametrize(
    "mapping, where",
    [
        ({"lr": float("nan")}, "'lr'"),
        ({"lr": float("inf")}, "'lr'"),
        ({"a": {"b": [1, float("-inf")]}}, "'a.b[1]'"),
        ({"model_args": {"x": float("nan")}}, "'model_args.x'"),
        ({"opt": Plain(lr=float("nan"))}, "'opt.lr'"),
        ({"v": np.float32("nan")}, "'v'"),
        ({"v": (np.float64("inf"),)}, "'v[0]'"),
    ],
)
def test_write_requested_refuses_a_non_finite_number_and_writes_no_nan(tmp_path, mapping, where):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    before = snapshot(tmp_path)
    with pytest.raises(ConfigError, match="NaN or infinity") as caught:
        write_requested(run_dir, mapping)
    assert where in str(caught.value)
    assert snapshot(tmp_path) == before
    assert (run_dir / "config_requested.json").read_bytes() == b""


def test_write_requested_refuses_every_non_finite_number_at_once(tmp_path):
    run_dir = tmp_path / "run"
    open_run(run_dir, resume=False)
    with pytest.raises(ConfigError) as caught:
        write_requested(run_dir, {"a": float("nan"), "b": [float("inf")]})
    assert "'a'" in str(caught.value) and "'b[0]'" in str(caught.value)


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


@pytest.mark.parametrize("word", ["AUTO", "Auto", "auto/", "auto2"])
def test_only_the_exact_lowercase_word_asks_for_a_timestamped_directory(word):
    run_dir = resolve_out(word, "image_classification", "vit", "vit_pico")
    assert run_dir == REPO / word.rstrip("/"), run_dir
    assert run_dir.parent == REPO


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


@pytest.mark.parametrize(
    "out",
    [
        "a/../b",
        "missing/../finished_run",
        "../outside",
        "../../x",
        "results/../../x",
        "x/..",
        "/tmp/a/../b",
        "~",
        "~/x",
        "~someone/x",
    ],
)
def test_a_dotdot_segment_and_a_leading_tilde_are_refused(out):
    with pytest.raises(ConfigError, match="'out'"):
        resolve_out(out, "t", "f", "v")


def test_the_missing_parent_bypass_overwrites_nothing_and_creates_nothing(tmp_path):
    """The review's reproduction: ``typo/../finished`` used to reach a finished run."""
    finished = tmp_path / "finished_run"
    plant(finished, "results_summary.json")
    plant(finished, "config.json")
    before = snapshot(tmp_path)
    for out in (f"{tmp_path}/typo_dir/../finished_run", f"{tmp_path}/typo_dir/../fresh"):
        with pytest.raises(ConfigError, match="'out'"):
            open_run(resolve_out(out, "t", "f", "v"), resume=False)
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "typo_dir").exists()


def test_the_repo_root_and_the_results_root_are_refused_by_identity(tmp_path):
    results = REPO / "results"
    listing_before = sorted(p.name for p in results.iterdir()) if results.is_dir() else None
    link = tmp_path / "link_to_repo"
    link.symlink_to(REPO)
    spellings = [
        str(REPO),
        str(REPO) + "/",
        str(REPO / "."),
        str(REPO / "results"),
        str(REPO / "results") + "/",
        "results",
        "results/",
        "./results",
        str(link),
        str(link / "results"),
    ]
    for out in spellings:
        with pytest.raises(ConfigError, match="repo root|results root"):
            resolve_out(out, "t", "f", "v")
    listing_after = sorted(p.name for p in results.iterdir()) if results.is_dir() else None
    assert listing_after == listing_before
    assert snapshot(tmp_path) == {"link_to_repo": "link->" + str(REPO)}


def test_a_directory_inside_the_results_root_is_still_a_run_directory():
    assert resolve_out("results/one_run", "t", "f", "v") == REPO / "results" / "one_run"


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
# An OSError never reaches the user as a traceback
# ---------------------------------------------------------------------


def _under_a_file(tmp_path):
    (tmp_path / "a_file").write_text("data")
    return str(tmp_path / "a_file" / "sub" / "run")


def _dangling(tmp_path):
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    return str(tmp_path / "dangling")


def _under_a_dangling_link(tmp_path):
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    return str(tmp_path / "dangling" / "run")


def _loop(tmp_path):
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    return str(tmp_path / "loop")


def _too_long(tmp_path):
    return str(tmp_path / ("x" * 300))


def _unwritable_parent(tmp_path):
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked").chmod(0o500)
    return str(tmp_path / "locked" / "run")


def _unsearchable_parent(tmp_path):
    (tmp_path / "shut" / "inner").mkdir(parents=True)
    (tmp_path / "shut").chmod(0o000)
    return str(tmp_path / "shut" / "inner" / "run")


OS_ERROR_CASES = {
    "NotADirectoryError": _under_a_file,
    "FileExistsError, dangling symlink": _dangling,
    "dangling symlink as a parent": _under_a_dangling_link,
    "symlink loop": _loop,
    "name too long": _too_long,
    "PermissionError, unwritable parent": _unwritable_parent,
    "PermissionError, unsearchable parent": _unsearchable_parent,
}


@pytest.mark.parametrize("case", list(OS_ERROR_CASES))
def test_an_os_error_becomes_a_config_error_naming_out_and_creates_nothing(tmp_path, case):
    if case.startswith("PermissionError") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    out = OS_ERROR_CASES[case](tmp_path)
    try:
        with pytest.raises(ConfigError, match="'out'"):
            open_run(resolve_out(out, "t", "f", "v"), resume=False)
    finally:
        for name in ("locked", "shut"):
            if (tmp_path / name).exists():
                (tmp_path / name).chmod(0o700)
    after = snapshot(tmp_path)
    expected_names = {"a_file", "dangling", "loop", "locked", "shut", "shut/inner"}
    assert set(after) <= expected_names, after  # no run directory and no `nowhere`
    assert not (tmp_path / "nowhere").exists()


def test_open_run_alone_translates_when_given_a_path_resolve_out_did_not_see(tmp_path):
    (tmp_path / "a_file").write_text("data")
    with pytest.raises(ConfigError, match="'out'"):
        open_run(tmp_path / "a_file" / "sub", resume=False)
    with pytest.raises(ConfigError, match="'out'"):
        open_run(tmp_path / "a_file", resume=False)
    assert snapshot(tmp_path) == {"a_file": hashlib.sha256(b"data").hexdigest()}


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
