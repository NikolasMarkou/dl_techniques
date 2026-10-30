"""The two Phase B experiment workers and drivers, end to end on CPU.

Plan plan-2026-10-30T025213-f75ab4e9 step 3. Exp1 compares four pooling heads on vit_pico.
The claims pinned here are the ones a quiet mistake would break without any run failing:

* the four exp1 arms differ only in the pooling head, with equal parameter counts, and the
  mean and mode heads ignore the CLS position;
* every arm scores the very same corrupted images for the same seed and condition, the
  selection score reads the held-out slice of train and never the test split;
* a cell that runs through the real subprocess driver writes one complete row.

Workers run in this process on a tiny fake CIFAR (``tests.classification_fakes``); one test
drives the real subprocess path with ``--cpu`` and the Keras cache. Everything is written
under ``tmp_path``; nothing here touches ``results/`` or a GPU.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.classification_fakes import use_tiny_raw
from train.common.classification_data import load_split_arrays
from train.mean_shift import common, corruption
from train.mean_shift import exp1_mode_pooling as exp1

REPO = Path(__file__).resolve().parents[3]
CIFAR_CACHE = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py-target"
MAX_SAMPLES = 256


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def expected_condition(x, name, seed):
    """The images a worker must score for ``name``, built from the two primitives.

    Deliberately not ``corruption.apply_condition``: a worker that bypassed or re-seeded
    that dispatcher must not make this reference follow it.
    """
    kind, parameter = corruption.CONDITIONS[name]
    if kind == "occlude":
        return corruption.occlude(x, parameter, seed, name)
    if kind == "noise":
        return corruption.add_noise(x, parameter, seed, name)
    return np.array(x, copy=True)


def run_worker(module, root, arm, config, *, seed=0, dataset="cifar10", epochs=1):
    cell_dir = Path(root) / f"{arm}__{config}__s{seed}"
    cell_dir.mkdir(parents=True)
    argv = ["--worker", "--cell-dir", str(cell_dir), "--arm", arm, "--config", config,
            "--seed", str(seed), "--dataset", dataset, "--epochs", str(epochs),
            "--max-samples", str(MAX_SAMPLES)]
    assert module.main(argv) == 0
    return json.loads((cell_dir / "result.json").read_text(encoding="utf-8"))


class Recorder:
    """Wraps ``common.top1`` and keeps a hash of every array a worker scored."""

    def __init__(self, monkeypatch):
        self.hashes = []
        real = common.top1

        def recording(model, x, y, batch_size=256):
            self.hashes.append(sha(x))
            return real(model, x, y, batch_size)

        monkeypatch.setattr(common, "top1", recording)


@pytest.fixture(scope="module")
def fake_cifar():
    """A tiny offline CIFAR for the in-process workers (module scoped)."""
    patch = pytest.MonkeyPatch()
    use_tiny_raw(patch, n_train=400, n_test=300)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def split_arrays(fake_cifar):
    return load_split_arrays("cifar10", seed=0, max_samples=MAX_SAMPLES)


# ---------------------------------------------------------------------------
# exp1: arms, heads, cells
# ---------------------------------------------------------------------------

ARM_NAMES = ("mean", "cls", "mode", "mode_s4")


def test_the_four_arms_are_the_planned_ones():
    assert tuple(exp1.ARMS) == ARM_NAMES


def test_the_four_arms_have_equal_parameter_counts():
    counts = {arm: exp1.build_model(arm, 10).count_params() for arm in ARM_NAMES}
    assert len(set(counts.values())) == 1, counts
    # vit_pico at patch 4 (sequence 65) plus Dense(10): 2,710,948 - Dense(100) + Dense(10)
    assert counts["mean"] == 2_693_578
    assert exp1.build_model("mean", 100).count_params() == 2_710_948


def test_the_backbone_sees_64_patches_not_4():
    model = exp1.build_model("mean", 10)
    backbone = next(layer for layer in model.layers if layer.name == "backbone")
    assert backbone.compute_output_shape((None, 32, 32, 3))[1] == 65


def test_each_arm_pools_with_its_planned_head():
    heads = {arm: exp1.pooling_layer(arm) for arm in ARM_NAMES}
    assert heads["mean"].strategy == ["mean"] and heads["mean"].exclude_positions == [0]
    assert heads["cls"].strategy == ["cls"]
    for arm, starts in (("mode", 1), ("mode_s4", 4)):
        mode = heads[arm]
        assert mode.num_steps == 5 and mode.bandwidth == "auto"
        assert mode.exclude_positions == [0] and mode.num_starts == starts


@pytest.mark.parametrize("arm", ["mean", "mode", "mode_s4"])
def test_the_mean_and_mode_heads_ignore_the_cls_position(arm):
    sequence = np.random.default_rng(0).normal(size=(3, 65, 16)).astype("float32")
    changed = sequence.copy()
    changed[:, 0, :] += 50.0
    head = exp1.pooling_layer(arm)
    assert np.array_equal(np.asarray(head(sequence)), np.asarray(head(changed)))


def test_the_cls_head_reads_the_cls_position():
    sequence = np.random.default_rng(1).normal(size=(3, 65, 16)).astype("float32")
    changed = sequence.copy()
    changed[:, 0, :] += 50.0
    head = exp1.pooling_layer("cls")
    assert np.array_equal(np.asarray(head(sequence))[:, :], sequence[:, 0, :])
    assert not np.array_equal(np.asarray(head(changed)), np.asarray(head(sequence)))


def test_the_exp1_cells_are_four_arms_by_three_seeds_seed_major():
    args = exp1.parse_driver_arguments(["--out-dir", "x", "--epochs", "7"])
    cells = exp1.build_cells(args)
    assert len(cells) == 12 and len({c.key for c in cells}) == 12
    assert [c.seed for c in cells] == [0] * 4 + [1] * 4 + [2] * 4
    assert [c.arm for c in cells[:4]] == list(ARM_NAMES)
    assert {c.experiment for c in cells} == {"exp1"} and {c.config for c in cells} == {"cifar10"}
    assert all("--epochs" in c.extra_args and "7" in c.extra_args for c in cells)


def test_the_exp1_driver_takes_a_dataset_arm_and_seed_subset():
    args = exp1.parse_driver_arguments(
        ["--out-dir", "x", "--epochs", "3", "--dataset", "cifar100", "--arms", "mode", "cls",
         "--seeds", "5", "--max-samples", "256"])
    cells = exp1.build_cells(args)
    assert [(c.arm, c.config, c.seed) for c in cells] == [
        ("mode", "cifar100", 5), ("cls", "cifar100", 5)]
    assert "--max-samples" in cells[0].extra_args


def test_an_unknown_exp1_arm_is_refused():
    with pytest.raises(SystemExit):
        exp1.parse_driver_arguments(["--out-dir", "x", "--epochs", "3", "--arms", "median"])


@pytest.fixture(scope="module")
def exp1_runs(fake_cifar, tmp_path_factory):
    """One in-process worker per arm, with the scored arrays recorded."""
    root = tmp_path_factory.mktemp("exp1_workers")
    patch = pytest.MonkeyPatch()
    recorders = {}
    results = {}
    for arm in ARM_NAMES:
        recorders[arm] = Recorder(patch)
        results[arm] = run_worker(exp1, root, arm, "cifar10")
        patch.undo()
    return results, recorders


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_an_exp1_worker_writes_one_complete_result(arm, exp1_runs):
    result = exp1_runs[0][arm]
    metrics = result["metrics"]
    assert result["finite"] is True and result["epochs_run"] == 1
    assert result["seconds_per_epoch"] is not None and result["seconds_per_epoch"] > 0
    assert set(metrics["accuracy"]) == set(corruption.CONDITIONS)
    assert all(0.0 <= v <= 1.0 for v in metrics["accuracy"].values())
    assert 0.0 <= metrics["select_accuracy"] <= 1.0
    assert metrics["dtype_policy"] == "float32" and metrics["n_test"] == MAX_SAMPLES
    assert metrics["dataset"] == "cifar10" and metrics["arm"] == arm
    assert result["devices"] and all("GPU" not in d for d in result["devices"])


def test_the_exp1_arms_report_the_same_parameter_count(exp1_runs):
    counts = {arm: r["params"] for arm, r in exp1_runs[0].items()}
    assert len(set(counts.values())) == 1 and counts["mean"] == 2_693_578


def test_every_arm_scores_the_same_corrupted_images(exp1_runs, split_arrays):
    _, recorders = exp1_runs
    expected = {c: sha(expected_condition(split_arrays.x_test, c, 0))
                for c in corruption.CONDITIONS}
    selection = sha(split_arrays.x_select)
    for arm, recorder in recorders.items():
        assert sorted(recorder.hashes) == sorted([*expected.values(), selection]), arm


def test_the_scored_conditions_really_differ_from_the_clean_images(split_arrays):
    clean = sha(split_arrays.x_test)
    others = [sha(expected_condition(split_arrays.x_test, c, 0))
              for c in corruption.CONDITIONS if c != "clean"]
    assert clean not in others and len(set(others)) == len(others)


def test_the_exp1_selection_score_reads_the_selection_split_not_the_test_split(
        exp1_runs, split_arrays):
    _, recorders = exp1_runs
    for recorder in recorders.values():
        assert sha(split_arrays.x_select) in recorder.hashes
    assert len(split_arrays.x_select) != len(split_arrays.x_test)
    assert not np.array_equal(split_arrays.x_select[:5], split_arrays.x_test[:5])


# ---------------------------------------------------------------------------
# the real subprocess path
# ---------------------------------------------------------------------------

needs_cache = pytest.mark.skipif(not CIFAR_CACHE.exists(), reason="no CIFAR-10 in the Keras cache")


def run_module(module, *argv, cwd=REPO, timeout=900):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="2")
    return subprocess.run([sys.executable, "-m", module, *argv], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("module", ["train.mean_shift.exp1_mode_pooling"])
def test_help_prints_usage_and_nothing_else_is_written(module, tmp_path):
    done = run_module(module, "--help", cwd=tmp_path)
    assert done.returncode == 0 and done.stdout.lstrip().startswith("usage:")
    assert not any(tmp_path.iterdir())


@needs_cache
def test_the_exp1_driver_runs_a_cell_in_a_subprocess_on_cpu(tmp_path):
    root = tmp_path / "sweep"
    done = run_module("train.mean_shift.exp1_mode_pooling", "--cpu", "--out-dir", str(root),
                      "--epochs", "1", "--max-samples", str(MAX_SAMPLES), "--arms", "mode_s4",
                      "--seeds", "0")
    assert done.returncode == 0, done.stderr[-2000:]
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert [(r["arm"], r["status"], r["seed"]) for r in rows] == [("mode_s4", "ok", 0)]
    assert rows[0]["device"] == "CUDA_VISIBLE_DEVICES="
    accuracy = rows[0]["metrics"]["metrics"]["accuracy"]
    assert set(accuracy) == set(corruption.CONDITIONS)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["resolved"]["epochs"] == 1 and manifest["devices"] == ""
