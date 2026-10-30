"""The Phase B cell runner: idle gate, manifest-first, fsynced rows, resume, retries.

Plan plan-2026-10-30T025213-f75ab4e9 step 2. Every long GPU run depends on this module, so
each rule is pinned by a test that a plausible mutant turns red: accepting a failed
``nvidia-smi`` or two idle samples out of three, re-running a finished cell, skipping a
failed one, or keying the resume on the seed alone. The real subprocess path is driven with
a fake worker (``fake_worker.py``) so nothing here needs TensorFlow, a GPU, or ``results/``.
Everything is written under ``tmp_path``.
"""

import argparse
import ast
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from train.mean_shift import common

REPO = Path(__file__).resolve().parents[3]
WORKER = "tests.test_train.test_mean_shift.fake_worker"
VAR = "CUDA_VISIBLE_DEVICES"


def cell(arm="a", seed=0, config="default", behavior="ok", experiment="exp"):
    return common.CellSpec(experiment, arm, config, seed, ("--behavior", behavior))


def recipe():
    return common.Recipe.for_epochs(10)


def sweep(tmp_path, cells, **overrides):
    options = dict(root=tmp_path / "sweep", recipe=recipe(), devices="", cell_timeout_s=60.0)
    options.update(overrides)
    return common.run_sweep(cells, WORKER, **options)


def rows_of(root):
    lines = (Path(root) / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class Sleeps:
    """An injectable sleeper that only records."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def scripted(samples):
    """A sampler that returns the scripted readings in order, then fails the test."""
    iterator = iter(samples)

    def sampler():
        try:
            return next(iterator)
        except StopIteration:
            raise AssertionError("the gate sampled more often than the script allows")

    return sampler


# ---------------------------------------------------------------------------
# the idle gate
# ---------------------------------------------------------------------------

def test_the_gate_passes_on_three_consecutive_idle_samples():
    sleeps = Sleeps()
    result = common.wait_for_idle(scripted([(527, 0)] * 3), sleeps)
    assert result.samples == [(527, 0)] * 3
    assert sleeps.calls == [common.GATE_GAP_S, common.GATE_GAP_S]
    assert result.refusals == 0


def test_two_idle_samples_out_of_three_do_not_pass():
    sleeps = Sleeps()
    with pytest.raises(common.GateTimeout):
        common.wait_for_idle(scripted([(527, 0), (527, 0), (9000, 0)] * 20), sleeps,
                             max_wait_s=300.0)


def test_a_busy_sample_resets_the_streak_and_a_refusal_sleeps_sixty_seconds():
    sleeps = Sleeps()
    readings = [(527, 0), (527, 0), (9000, 0), (527, 0), (527, 0), (527, 0)]
    result = common.wait_for_idle(scripted(readings), sleeps)
    assert sleeps.calls == [5.0, 5.0, 60.0, 5.0, 5.0]
    assert result.refusals == 1
    assert result.samples == [(527, 0)] * 3
    assert result.waited_s == sum(sleeps.calls)


@pytest.mark.parametrize("reading, idle", [
    ((2048, 20), True), ((2049, 20), False), ((2048, 21), False),
    ((0, 0), True), ((527, 100), False), ((40000, 0), False),
])
def test_the_thresholds_are_inclusive(reading, idle):
    sleeps = Sleeps()
    if idle:
        assert common.wait_for_idle(scripted([reading] * 3), sleeps).samples == [reading] * 3
    else:
        with pytest.raises(common.GateTimeout):
            common.wait_for_idle(lambda: reading, sleeps, max_wait_s=120.0)


def test_a_failed_reading_counts_as_busy_never_idle():
    sleeps = Sleeps()
    result = common.wait_for_idle(scripted([None, None, None, (527, 0), (527, 0), (527, 0)]), sleeps)
    assert result.refusals == 3
    assert sleeps.calls[:3] == [common.GATE_RETRY_S] * 3
    with pytest.raises(common.GateTimeout):
        common.wait_for_idle(lambda: None, Sleeps(), max_wait_s=600.0)


def test_the_wait_cap_raises_after_the_stated_time():
    sleeps = Sleeps()
    with pytest.raises(common.GateTimeout, match="600"):
        common.wait_for_idle(lambda: (9000, 99), sleeps, max_wait_s=600.0)
    assert sum(sleeps.calls) == 600.0


def test_the_default_wait_cap_is_six_hours():
    assert common.GATE_MAX_WAIT_S == 6 * 3600


@pytest.mark.parametrize("text, expected", [
    ("527, 0\n", (527, 0)), ("527,0", (527, 0)), (" 2048 , 20 \n", (2048, 20)),
    ("", None), ("\n", None), ("527 MiB, 0 %", None), ("[N/A], 0", None), ("527, [N/A]", None),
    ("-5, 0", None), ("527, 0\n12, 3\n", None),
    ("Unable to determine the device handle for GPU1: 0000:0A:00.0: Unknown Error", None),
])
def test_the_smi_output_is_parsed_strictly(text, expected):
    assert common.parse_smi_output(text) == expected


class _Done:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


def test_the_sampler_asks_for_gpu_zero_only_and_drops_stderr():
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = list(cmd), kwargs
        return _Done(0, "527, 0\n")

    assert common.sample_gpu0(run=run) == (527, 0)
    assert seen["cmd"] == ["nvidia-smi", "-i", "0",
                           "--query-gpu=memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"]
    assert seen["kwargs"]["stderr"] == subprocess.DEVNULL
    assert seen["kwargs"]["timeout"] > 0


def test_a_nonzero_return_code_a_hang_and_a_missing_binary_are_busy():
    assert common.sample_gpu0(run=lambda cmd, **k: _Done(9, "527, 0\n")) is None

    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    def missing(cmd, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert common.sample_gpu0(run=hang) is None
    assert common.sample_gpu0(run=missing) is None


# ---------------------------------------------------------------------------
# cells and the recipe
# ---------------------------------------------------------------------------

def test_a_cell_is_keyed_by_the_full_tuple():
    a = common.CellSpec("e", "mode", "default", 1)
    assert common.cell_key(a) == ("e", "mode", "default", 1)
    assert common.cell_key({"experiment": "e", "arm": "mode", "config": "default", "seed": 1,
                            "status": "ok"}) == common.cell_key(a)
    assert a.cell_id == "e__mode__default__s1"
    with pytest.raises(Exception):
        a.seed = 2


@pytest.mark.parametrize("field, value", [
    ("experiment", ""), ("arm", "a/b"), ("config", "x__y"), ("arm", "a b"), ("config", ".."),
])
def test_a_cell_name_that_could_break_a_path_or_the_id_is_refused(field, value):
    fields = dict(experiment="e", arm="a", config="c", seed=0)
    fields[field] = value
    with pytest.raises(ValueError):
        common.CellSpec(**fields)


def test_duplicate_cells_are_refused(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        sweep(tmp_path, [cell("a", 0), cell("a", 0)])
    assert not (tmp_path / "sweep").exists()


def test_cells_run_seed_major():
    cells = [cell(arm, seed) for arm in ("a", "b") for seed in (0, 1, 2)]
    order = [(c.arm, c.seed) for c in common.seed_major(cells)]
    assert order == [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)]


def test_the_recipe_hash_is_stable_and_sensitive_to_every_field():
    base = common.Recipe.for_epochs(30)
    assert base.recipe_hash == common.Recipe.for_epochs(30).recipe_hash
    import dataclasses
    changes = {"epochs": 31, "warmup_epochs": 4, "learning_rate": 1e-3, "weight_decay": 0.01,
               "batch_size": 64, "gradient_clipping": 0.5, "lr_schedule_type": "exponential_decay",
               "optimizer_type": "adam"}
    assert set(changes) == {f.name for f in dataclasses.fields(base)}
    for name, value in changes.items():
        assert dataclasses.replace(base, **{name: value}).recipe_hash != base.recipe_hash, name


@pytest.mark.parametrize("epochs, warmup", [(50, 5), (30, 3), (10, 1), (5, 1), (100, 5)])
def test_warmup_is_a_tenth_of_the_epochs_between_one_and_five(epochs, warmup):
    assert common.Recipe.for_epochs(epochs).warmup_epochs == warmup


def test_the_recipe_is_the_stated_one():
    r = common.Recipe.for_epochs(50)
    assert (r.learning_rate, r.weight_decay, r.batch_size, r.gradient_clipping) == (3e-4, 0.05, 128, 1.0)
    assert (r.optimizer_type, r.lr_schedule_type) == ("adamw", "cosine_decay")


def test_the_optimizer_is_the_shared_adamw_with_the_recipe_values():
    optimizer = common.build_optimizer(common.Recipe.for_epochs(10), steps_per_epoch=7)
    config = optimizer.get_config()
    assert type(optimizer).__name__ == "AdamW"
    assert config["weight_decay"] == 0.05 and config["global_clipnorm"] == 1.0
    schedule = config["learning_rate"]
    assert schedule["class_name"] == "WarmupSchedule"
    assert schedule["config"]["warmup_steps"] == 7
    primary = schedule["config"]["primary_schedule"]
    assert primary["class_name"] == "CosineDecay"
    assert primary["config"]["initial_learning_rate"] == 3e-4
    assert primary["config"]["decay_steps"] == 70


def test_the_epoch_timer_writes_seconds_into_the_history():
    import numpy as np
    import keras
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(1)])
    model.compile("sgd", "mse")
    history = model.fit(np.zeros((8, 3), "float32"), np.zeros((8, 1), "float32"), epochs=3,
                        batch_size=4, verbose=0, callbacks=[common.epoch_timer()])
    seconds = history.history["epoch_seconds"]
    assert len(seconds) == 3 and all(s > 0 for s in seconds)


def test_build_result_makes_strict_json_and_flags_non_finite_loss():
    import numpy as np
    good = common.build_result(
        {"loss": [np.float32(2.0), np.float32(1.0), 0.5], "epoch_seconds": [9.0, 2.0, 4.0]},
        params=np.int64(10), metrics={"acc": np.float32(0.5)})
    assert good["finite"] is True and good["epochs_run"] == 3 and good["params"] == 10
    assert good["seconds_per_epoch"] == 3.0
    json.dumps(good, allow_nan=False)
    bad = common.build_result({"loss": [1.0, float("nan")], "epoch_seconds": [1.0, 1.0]}, params=1)
    assert bad["finite"] is False
    assert bad["history"]["loss"] == [1.0, None]
    json.dumps(bad, allow_nan=False)
    one = common.build_result({"loss": [1.0], "epoch_seconds": [7.0]}, params=1)
    assert one["seconds_per_epoch"] == 7.0


# ---------------------------------------------------------------------------
# a sweep: manifest first, rows, device, order
# ---------------------------------------------------------------------------

def test_a_sweep_writes_the_manifest_first_then_one_row_per_cell(tmp_path):
    cells = [cell("a", 0), cell("b", 0), cell("a", 1)]
    result = sweep(tmp_path, cells)
    root = tmp_path / "sweep"
    assert result.exit_code == 0 and result.ok == 3 and result.failed == []
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert [c["arm"] for c in manifest["cells"]] == ["a", "b", "a"]
    assert manifest["recipe_hash"] == recipe().recipe_hash
    assert manifest["recipe"]["epochs"] == 10
    assert manifest["devices"] == "" and manifest["git_sha"]
    rows = rows_of(root)
    assert [(r["arm"], r["seed"], r["status"], r["attempt"]) for r in rows] == [
        ("a", 0, "ok", 1), ("b", 0, "ok", 1), ("a", 1, "ok", 1)]
    for row in rows:
        assert row["metrics"]["metrics"]["manifest_present"] is True
        assert row["seconds"] > 0 and row["device"] == "CUDA_VISIBLE_DEVICES="
        assert row["cell_dir"].startswith("cells/exp__") and row["cell_dir"].endswith("__try1")
    assert rows[0]["metrics"]["metrics"]["rows_present"] is False
    assert rows[1]["metrics"]["metrics"]["rows_present"] is True


def test_the_manifest_exists_before_the_first_cell_is_gated(tmp_path):
    seen = []

    def sampler():
        seen.append((tmp_path / "sweep" / "manifest.json").is_file())
        return (527, 0)

    sweep(tmp_path, [cell()], devices="0", sampler=sampler, sleeper=Sleeps())
    assert seen and all(seen)


def test_the_manifest_and_every_row_are_fsynced(tmp_path, monkeypatch):
    real = os.fsync
    kinds = []

    def spy(fd):
        kinds.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    sweep(tmp_path, [cell("a", 0)])
    after_manifest = len(kinds)
    assert "file" in kinds[:after_manifest] and "dir" in kinds[:after_manifest]
    kinds.clear()
    sweep(tmp_path, [cell("b", 0), cell("c", 0)], root=tmp_path / "second")
    assert kinds.count("file") >= 1 + 2 and "dir" in kinds


def test_the_child_gets_the_resolved_device_and_the_agg_backend(tmp_path, monkeypatch):
    _inherit(monkeypatch, "1")
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    sweep(tmp_path, [cell("a", 0)], devices="0", sampler=lambda: (527, 0), sleeper=Sleeps())
    metrics = rows_of(tmp_path / "sweep")[0]["metrics"]["metrics"]
    assert metrics["seen_cuda_visible_devices"] == "0"
    assert metrics["seen_mplbackend"] == "Agg"


def test_a_cpu_sweep_hides_the_gpus_and_skips_the_gate(tmp_path, monkeypatch):
    _inherit(monkeypatch, "0")

    def forbidden():
        raise AssertionError("the cpu mode must not sample the GPU")

    sweep(tmp_path, [cell("a", 0)], devices="", sampler=forbidden, sleeper=Sleeps())
    row = rows_of(tmp_path / "sweep")[0]
    assert row["metrics"]["metrics"]["seen_cuda_visible_devices"] == ""
    assert row["gate"] is None


def test_the_child_runs_from_the_repository_root(tmp_path):
    sweep(tmp_path, [cell()])
    assert rows_of(tmp_path / "sweep")[0]["metrics"]["metrics"]["cwd"] == str(REPO)


def test_a_gpu_row_records_the_gate_samples_and_the_device(tmp_path):
    sleeps = Sleeps()
    sweep(tmp_path, [cell()], devices="0", sleeper=sleeps,
          sampler=scripted([(9000, 90), (527, 0), (530, 1), (528, 0)]))
    row = rows_of(tmp_path / "sweep")[0]
    assert row["device"] == "CUDA_VISIBLE_DEVICES=0"
    assert row["gate"]["samples"] == [[527, 0], [530, 1], [528, 0]]
    assert row["gate"]["refusals"] == 1 and row["gate"]["waited_s"] == 70.0
    log = (tmp_path / "sweep" / row["cell_dir"] / "cell.log").read_text(encoding="utf-8")
    assert log.startswith("# cmd:") and "# CUDA_VISIBLE_DEVICES=0" in log
    assert "# MPLBACKEND=Agg" in log and "# gate samples:" in log


def test_every_cell_is_gated_before_it_starts(tmp_path):
    calls = []

    def sampler():
        calls.append(len(list((tmp_path / "sweep" / "cells").glob("*"))) if (tmp_path / "sweep" / "cells").exists() else 0)
        return (527, 0)

    sweep(tmp_path, [cell("a", 0), cell("b", 0)], devices="0", sampler=sampler, sleeper=Sleeps())
    assert calls == [0, 0, 0, 1, 1, 1]


def test_the_wait_cap_is_shared_by_the_whole_sweep(tmp_path):
    with pytest.raises(common.GateTimeout):
        sweep(tmp_path, [cell("a", 0), cell("b", 0)], devices="0", sleeper=Sleeps(),
              sampler=lambda: (9000, 99), max_wait_s=120.0)
    assert not (tmp_path / "sweep" / "rows.jsonl").exists()
    assert (tmp_path / "sweep" / "manifest.json").is_file()


def test_a_device_other_than_zero_is_refused_before_anything_is_written(tmp_path):
    for devices in ("1", "0,1", "GPU-abc"):
        with pytest.raises(ValueError, match="GPU 0"):
            sweep(tmp_path, [cell()], devices=devices, sampler=lambda: (0, 0), sleeper=Sleeps())
    assert not (tmp_path / "sweep").exists()


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------

def test_a_failed_cell_appends_a_failed_row_with_the_stderr_tail_and_the_sweep_goes_on(tmp_path):
    result = sweep(tmp_path, [cell("a", 0, behavior="fail"), cell("b", 0)])
    rows = rows_of(tmp_path / "sweep")
    assert [(r["arm"], r["status"]) for r in rows] == [("a", "failed"), ("b", "ok")]
    assert "boom: the fake training diverged" in rows[0]["error_tail"]
    assert rows[0]["returncode"] == 3
    assert result.exit_code == 1 and result.failed == [("exp", "a", "default", 0)]


def test_a_cell_that_exits_zero_without_a_result_is_failed(tmp_path):
    result = sweep(tmp_path, [cell("a", 0, behavior="noresult")])
    row = rows_of(tmp_path / "sweep")[0]
    assert row["status"] == "failed" and "result.json" in row["error_tail"]
    assert result.exit_code == 1


def test_a_non_finite_loss_makes_the_cell_failed_but_keeps_its_metrics(tmp_path):
    result = sweep(tmp_path, [cell("a", 0, behavior="nan")])
    row = rows_of(tmp_path / "sweep")[0]
    assert row["status"] == "failed" and "non-finite" in row["error_tail"]
    assert row["metrics"]["finite"] is False and row["metrics"]["history"]["loss"] == [None, None]
    assert result.exit_code == 1


def test_a_hanging_cell_is_killed_at_the_timeout(tmp_path):
    started = time.time()
    result = sweep(tmp_path, [cell("a", 0, behavior="hang")], cell_timeout_s=3.0)
    assert time.time() - started < 60
    row = rows_of(tmp_path / "sweep")[0]
    assert row["status"] == "failed" and "timeout after 3" in row["error_tail"]
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# resume and retries
# ---------------------------------------------------------------------------

def test_resume_skips_ok_cells_and_retries_a_failed_one_in_a_new_directory(tmp_path):
    cells = [cell("a", 0), cell("b", 0, behavior="fail_first_try")]
    first = sweep(tmp_path, cells)
    root = tmp_path / "sweep"
    assert first.exit_code == 1
    old_log = root / rows_of(root)[1]["cell_dir"] / "cell.log"
    old_text = old_log.read_text(encoding="utf-8")
    ok_dir = root / rows_of(root)[0]["cell_dir"]
    before = sorted(p.name for p in (root / "cells").iterdir())

    second = sweep(tmp_path, cells, resume=True)
    rows = rows_of(root)
    assert second.exit_code == 0 and second.skipped == 1 and second.ok == 1
    assert [(r["arm"], r["status"], r["attempt"]) for r in rows] == [
        ("a", "ok", 1), ("b", "failed", 1), ("b", "ok", 2)]
    after = sorted(p.name for p in (root / "cells").iterdir())
    assert after == sorted(before + ["exp__b__default__s0__try2"])
    assert old_log.read_text(encoding="utf-8") == old_text and ok_dir.is_dir()


def test_resume_of_a_finished_sweep_runs_nothing(tmp_path):
    cells = [cell("a", 0), cell("b", 0)]
    sweep(tmp_path, cells)
    root = tmp_path / "sweep"
    before = (root / "rows.jsonl").read_text(encoding="utf-8")
    again = sweep(tmp_path, cells, resume=True, sampler=lambda: pytest.fail("gated"), devices="")
    assert again.skipped == 2 and again.ok == 0 and again.exit_code == 0
    assert (root / "rows.jsonl").read_text(encoding="utf-8") == before


def test_resume_is_keyed_by_the_full_tuple_not_the_seed_or_the_arm(tmp_path):
    cells = [
        cell("a", 0, "c1"), cell("b", 0, "c1", "fail_first_try"),
        cell("a", 1, "c1", "fail_first_try"), cell("a", 0, "c2", "fail_first_try"),
        cell("b", 1, "c2", "fail_first_try"),
    ]
    sweep(tmp_path, cells)
    again = sweep(tmp_path, cells, resume=True)
    assert again.skipped == 1 and again.ok == 4 and again.exit_code == 0
    reran = [(r["arm"], r["config"], r["seed"], r["attempt"]) for r in rows_of(tmp_path / "sweep")[5:]]
    assert sorted(reran) == sorted([
        ("b", "c1", 0, 2), ("a", "c1", 1, 2), ("a", "c2", 0, 2), ("b", "c2", 1, 2)])


def test_a_cell_with_a_different_experiment_is_not_skipped(tmp_path):
    first = [cell("a", 0, experiment="e1"), cell("a", 0, experiment="e2", behavior="fail_first_try")]
    sweep(tmp_path, first)
    again = sweep(tmp_path, first, resume=True)
    assert again.skipped == 1 and again.ok == 1


@pytest.mark.parametrize("change", ["fewer", "more", "other_seed", "other_arg", "other_experiment"])
def test_resume_refuses_a_manifest_whose_cell_list_differs(tmp_path, change):
    cells = [cell("a", 0), cell("b", 0)]
    sweep(tmp_path, cells)
    changed = {
        "fewer": cells[:1],
        "more": cells + [cell("c", 0)],
        "other_seed": [cell("a", 5), cell("b", 0)],
        "other_arg": [cell("a", 0), cell("b", 0, behavior="fail")],
        "other_experiment": [cell("a", 0, experiment="zzz"), cell("b", 0)],
    }[change]
    rows_before = (tmp_path / "sweep" / "rows.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        sweep(tmp_path, changed, resume=True)
    assert (tmp_path / "sweep" / "rows.jsonl").read_text(encoding="utf-8") == rows_before


def test_resume_refuses_a_changed_recipe(tmp_path):
    cells = [cell("a", 0)]
    sweep(tmp_path, cells)
    with pytest.raises(ValueError, match="recipe"):
        sweep(tmp_path, cells, resume=True, recipe=common.Recipe.for_epochs(11))


def test_resume_accepts_the_same_cells_in_another_order(tmp_path):
    cells = [cell("a", 0), cell("b", 0)]
    sweep(tmp_path, cells)
    again = sweep(tmp_path, list(reversed(cells)), resume=True)
    assert again.skipped == 2


def test_resume_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        sweep(tmp_path, [cell()], resume=True)
    assert not (tmp_path / "sweep" / "rows.jsonl").exists()


def test_a_fresh_sweep_never_reuses_a_root(tmp_path):
    sweep(tmp_path, [cell()])
    root = tmp_path / "sweep"
    before = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        sweep(tmp_path, [cell("zzz", 9)])
    assert (root / "manifest.json").read_bytes() == before
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        sweep(tmp_path, [cell()], root=stray)
    assert (stray / "keep.txt").read_text(encoding="utf-8") == "x"


def test_a_leftover_attempt_directory_is_never_reused_or_touched(tmp_path):
    root = tmp_path / "sweep"
    cells = [cell("a", 0)]
    sweep(tmp_path, cells, resume=False)
    rows_path = root / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    leftover = root / "cells" / "exp__a__default__s0__try2"
    leftover.mkdir()
    (leftover / "half.txt").write_text("killed mid-fit", encoding="utf-8")
    sweep(tmp_path, cells, resume=True)
    rows = rows_of(root)
    assert [r["attempt"] for r in rows] == [3]
    assert (leftover / "half.txt").read_text(encoding="utf-8") == "killed mid-fit"
    assert sorted(p.name for p in (root / "cells").iterdir()) == [
        "exp__a__default__s0__try1", "exp__a__default__s0__try2", "exp__a__default__s0__try3"]


def test_a_torn_last_row_is_ignored_and_the_cell_reruns(tmp_path):
    cells = [cell("a", 0), cell("b", 0)]
    sweep(tmp_path, cells)
    rows_path = tmp_path / "sweep" / "rows.jsonl"
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    rows_path.write_text(lines[0] + "\n" + lines[1][: len(lines[1]) // 2], encoding="utf-8")
    again = sweep(tmp_path, cells, resume=True)
    assert again.skipped == 1 and again.ok == 1
    parsed = common.read_rows(tmp_path / "sweep")
    assert [(r["arm"], r["attempt"]) for r in parsed] == [("a", 1), ("b", 2)]


def test_read_rows_returns_nothing_for_a_missing_file(tmp_path):
    assert common.read_rows(tmp_path) == []


# ---------------------------------------------------------------------------
# the driver front end
# ---------------------------------------------------------------------------

def _parser():
    parser = argparse.ArgumentParser()
    common.add_sweep_arguments(parser)
    return parser


def _inherit(monkeypatch, value):
    """Set the launcher's export; the placeholder first makes monkeypatch restore the original."""
    monkeypatch.setenv(VAR, "placeholder")
    if value is None:
        monkeypatch.delenv(VAR)
    else:
        monkeypatch.setenv(VAR, value)


def _drive(tmp_path, argv, cells, monkeypatch, **extra):
    _inherit(monkeypatch, "")
    args = _parser().parse_args(argv)
    return common.run_driver(args, cells, worker_module=WORKER, recipe=recipe(),
                             cell_timeout_s=60.0, **extra)


def test_the_driver_returns_nonzero_after_a_sweep_with_a_failed_cell(tmp_path, monkeypatch):
    cells = [cell("a", 0, behavior="fail"), cell("b", 0)]
    code = _drive(tmp_path, ["--cpu", "--out-dir", str(tmp_path / "d1")], cells, monkeypatch)
    assert code == 1
    assert [r["status"] for r in rows_of(tmp_path / "d1")] == ["failed", "ok"]
    ok = _drive(tmp_path, ["--cpu", "--out-dir", str(tmp_path / "d2")], [cell("a", 0)], monkeypatch)
    assert ok == 0


def test_the_driver_resumes_from_the_given_root(tmp_path, monkeypatch):
    cells = [cell("a", 0), cell("b", 0, behavior="fail_first_try")]
    _drive(tmp_path, ["--cpu", "--out-dir", str(tmp_path / "d1")], cells, monkeypatch)
    code = _drive(tmp_path, ["--cpu", "--resume", str(tmp_path / "d1")], cells, monkeypatch)
    assert code == 0 and len(rows_of(tmp_path / "d1")) == 3


@pytest.mark.parametrize("argv", [
    ["--gpu", "1", "--out-dir", "X"],
    ["--gpu", "2", "--out-dir", "X"],
    ["--gpu", "-1", "--out-dir", "X"],
    ["--cpu", "--gpu", "0", "--out-dir", "X"],
    [],
    ["--resume", "X", "--out-dir", "Y"],
])
def test_the_driver_accepts_only_gpu_zero_and_one_way_to_name_the_root(
        tmp_path, monkeypatch, argv):
    argv = [str(tmp_path / a) if a in ("X", "Y") else a for a in argv]
    with pytest.raises(ValueError):
        _drive(tmp_path, argv, [cell()], monkeypatch, sampler=lambda: (0, 0), sleeper=Sleeps())
    assert not (tmp_path / "X").exists() and not (tmp_path / "Y").exists()


@pytest.mark.parametrize("inherited", ["1", "0,1", "", "GPU-abc"])
def test_a_parent_that_exports_another_device_is_refused_before_any_cell(
        tmp_path, monkeypatch, inherited):
    _inherit(monkeypatch, inherited)
    args = _parser().parse_args(["--out-dir", str(tmp_path / "d")])
    with pytest.raises(ValueError, match=VAR):
        common.run_driver(args, [cell()], worker_module=WORKER, recipe=recipe(),
                          cell_timeout_s=60.0, sampler=lambda: (0, 0), sleeper=Sleeps())
    assert not (tmp_path / "d").exists()


@pytest.mark.parametrize("inherited, gpu_flag", [(None, []), ("0", []), ("0", ["--gpu", "0"]),
                                                  (None, ["--gpu", "0"])])
def test_gpu_zero_mode_runs_the_children_on_zero_and_hides_the_driver(
        tmp_path, monkeypatch, inherited, gpu_flag):
    _inherit(monkeypatch, inherited)
    args = _parser().parse_args(["--out-dir", str(tmp_path / "d"), *gpu_flag])
    code = common.run_driver(args, [cell()], worker_module=WORKER, recipe=recipe(),
                             cell_timeout_s=60.0, sampler=lambda: (527, 0), sleeper=Sleeps())
    assert code == 0
    assert rows_of(tmp_path / "d")[0]["metrics"]["metrics"]["seen_cuda_visible_devices"] == "0"
    assert os.environ[VAR] == ""


def test_the_worker_arguments_are_all_required():
    parser = argparse.ArgumentParser()
    common.add_worker_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--worker"])
    namespace = parser.parse_args(["--worker", "--cell-dir", "d", "--arm", "a", "--config", "c",
                                   "--seed", "3"])
    assert (namespace.arm, namespace.config, namespace.seed) == ("a", "c", 3)


# ---------------------------------------------------------------------------
# import hygiene and the no-deletion rule
# ---------------------------------------------------------------------------

def test_the_runner_imports_without_tensorflow():
    code = ("import sys, train.mean_shift.common as c; "
            "c.Recipe.for_epochs(10).recipe_hash; c.parse_smi_output('1, 2'); "
            "raise SystemExit(int('tensorflow' in sys.modules or 'keras' in sys.modules))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert done.returncode == 0, done.stderr[-800:]


def test_the_runner_source_has_no_deleting_call():
    """Results are unrecoverable: the module may create and rename, never delete."""
    tree = ast.parse(Path(common.__file__).read_text(encoding="utf-8"))
    names = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
             for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert names.isdisjoint({"rmtree", "remove", "unlink", "rmdir", "removedirs", "truncate"}), names
