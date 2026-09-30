"""End-to-end and unit tests of ``src/train/hkan/train_hkan.py``.

Every run goes through the real ``main(argv)`` into a pytest temporary directory
(``--output-dir <tmp> --experiment-name <name>``) with tiny settings: 300 train
rows, 120 test rows, one hidden layer of 8 units, 6 basis functions, 2 repeats and
3 epochs. One module-scoped run per training mode serves several assertions.

Mechanisms guarded, each shown to fail under its mutation (``decisions.md`` of
plan-2026-09-30T082355-4d999dbc, the step-5 entry):

- the training mode decides which phases run and which files are written;
- the validation rows index the TRAIN split, never the test split;
- a reused experiment name is refused BEFORE anything is written;
- the closed-form phase of the two-phase mode is scored before the fine-tune;
- every repeat gets its own seed;
- a failing figure does not cost the run its results.

Added by the completion fix (D-041 to D-044 of the same file):

- a diverged run finishes and says it has no best checkpoint;
- in the two-phase mode ``best_model.keras`` exists only when an epoch beat the
  closed-form start, which is saved as ``closed_form_model.keras``;
- a bad argument is refused at parse time, before the experiment name is used;
- the validation split's generator is no stream of the model package;
- ``fit_rows`` is what the fit received;
- every flag's NON-default value reaches the constructor, the closed-form fit,
  the optimizer, ``fit`` or the prediction pass (``wired``);
- the best checkpoint is the argmin epoch and reproduces the monitor's minimum;
- ``--val-fraction 0`` end to end; median and IQR on three repeats.

Nothing may be written under repo-root ``results/``: the autouse fixture in
``tests/conftest.py`` fails any test that adds an entry there.
"""

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple

import keras
import numpy as np
import pytest

import train.hkan.train_hkan as train_hkan
from dl_techniques.models.general_purpose.hkan import hkan_layer as hkan_layer_module
from dl_techniques.models.general_purpose.hkan import model as hkan_model_module
from train.hkan.data import make_dataset

TRAIN_ROWS, TEST_ROWS, REPEATS, EPOCHS = 300, 120, 2, 3
TINY = (
    "--num-train-samples", str(TRAIN_ROWS), "--num-test-samples", str(TEST_ROWS),
    "--hidden-units", "8", "--num-basis", "6", "--predict-batch-size", "128",
)

BASE_FILES = {
    "config.json", "run.log", "results_summary.json", "final_model.keras",
    "visualizations/test_predictions.png",
}
CLOSED_FORM_FILES = {"visualizations/input_importance.png"}
REPEAT_FILES = {"visualizations/rmse_over_repeats.png"}
BACKPROP_FILES = {
    "training_log.csv", "training_history.json", "visualizations/loss_curve.png",
}
BEST_FILE = "best_model.keras"
CLOSED_FORM_MODEL = "closed_form_model.keras"
#: The three per-mode runs of ``runs``. In both backprop runs an epoch is the best
#: model (measured: the two-phase run's epoch 2 has validation MSE 0.0181 against
#: 0.0232 at the closed-form start), so ``best_model.keras`` is in both sets.
EXPECTED_FILES = {
    "closed_form": BASE_FILES | CLOSED_FORM_FILES | REPEAT_FILES,
    "backprop": BASE_FILES | REPEAT_FILES | BACKPROP_FILES | {BEST_FILE},
    "closed_form_then_backprop": (
        BASE_FILES | CLOSED_FORM_FILES | REPEAT_FILES | BACKPROP_FILES
        | {BEST_FILE, CLOSED_FORM_MODEL}),
}
MODES = sorted(EXPECTED_FILES)
BEST_KEYS = {
    "monitor", "source", "file", "epoch", "monitor_value", "train_rmse", "test_rmse", "reason",
}

SUMMARY_KEYS = {
    "model", "training_mode", "dataset", "num_inputs", "train_rows", "fit_rows",
    "test_rows", "params", "seed", "repeats", "train_rmse", "test_rmse", "fit_seconds",
    "closed_form_phase", "best_checkpoint", "input_importance", "repeat_results",
    "visualizations",
}
SPREAD_KEYS = {"median", "iqr", "min", "max"}


class Run(NamedTuple):
    """One finished trainer run."""

    run_dir: Path
    argv: List[str]
    summary: Dict


def _no_constants(name: str):
    raise AssertionError(f"results_summary.json holds the non-JSON constant {name}")


def _strict_summary(run_dir: Path) -> Dict:
    text = (run_dir / "results_summary.json").read_text()
    return json.loads(text, parse_constant=_no_constants)


def _files(run_dir: Path) -> set:
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}


def _hashes(run_dir: Path) -> Dict[str, str]:
    return {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in sorted(_files(run_dir))
    }


def _argv(base: Path, name: str, *extra: str) -> List[str]:
    return [*TINY, "--output-dir", str(base), "--experiment-name", name, *extra]


def _run(base: Path, name: str, *extra: str) -> Run:
    argv = _argv(base, name, *extra)
    assert train_hkan.main(argv) == 0
    run_dir = Path(base) / name
    return Run(run_dir, argv, _strict_summary(run_dir))


MODE_ARGS = {
    "closed_form": ("--dataset", "tf5", "--seed", "0"),
    "backprop": ("--dataset", "tf2", "--seed", "0", "--epochs", str(EPOCHS)),
    "closed_form_then_backprop": ("--dataset", "tf5", "--seed", "0", "--epochs", str(EPOCHS)),
}


#: Further named runs, made on first use (``extra_runs``). All single-repeat
#: unless they say otherwise.
EXTRA_ARGS = {
    # learning rate 0.3: validation MSE 1.21, 1.33, 0.52, 0.076, 0.30, 0.18 (measured),
    # so the best epoch (4) is neither the first nor the last
    "best_in_the_middle": ("--training-mode", "backprop", "--dataset", "tf2", "--seed", "0",
                           "--epochs", "6", "--learning-rate", "0.3"),
    # learning rate 0.05 throws the closed-form weights off: validation MSE 9.1, 7.2,
    # 1.9 against 0.0232 at the start (measured)
    "fine_tune_hurts": ("--training-mode", "closed_form_then_backprop", "--dataset", "tf5",
                        "--seed", "0", "--epochs", "3", "--learning-rate", "0.05"),
    "diverged": ("--training-mode", "backprop", "--dataset", "tf2", "--seed", "0",
                 "--epochs", "3", "--learning-rate", "1e12"),
    "three_repeats": ("--training-mode", "closed_form_then_backprop", "--dataset", "tf5",
                      "--seed", "0", "--epochs", "2", "--repeats", "3"),
    "backprop_again": ("--training-mode", "backprop", "--repeats", str(REPEATS),
                       "--dataset", "tf2", "--seed", "0", "--epochs", str(EPOCHS)),
}


class _LazyRuns:
    """One real run per name, made on first use and kept for the module.

    Lazy and per mode on purpose: a trainer defect that crashes one mode must not
    turn the tests of the other two into fixture errors, where no assertion runs.
    A failed run is remembered and re-raised rather than retried into the
    directory it already started.
    """

    def __init__(self, base: Path, table: Dict[str, tuple]) -> None:
        self._base = base
        self._table = table
        self._done: Dict[str, object] = {}

    def __getitem__(self, mode: str) -> Run:
        if mode not in self._done:
            try:
                self._done[mode] = _run(self._base, mode, *self._table[mode])
            except Exception as error:  # re-raised below, for every test of this mode
                self._done[mode] = error
        if isinstance(self._done[mode], Exception):
            raise self._done[mode]
        return self._done[mode]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> _LazyRuns:
    """The three per-mode runs, 2 repeats each."""
    return _LazyRuns(tmp_path_factory.mktemp("hkan_runs"), {
        mode: ("--training-mode", mode, "--repeats", str(REPEATS), *MODE_ARGS[mode])
        for mode in MODE_ARGS
    })


@pytest.fixture(scope="module")
def extra_runs(tmp_path_factory) -> _LazyRuns:
    """The runs of ``EXTRA_ARGS``."""
    return _LazyRuns(tmp_path_factory.mktemp("hkan_extra"), EXTRA_ARGS)


def _monitor_mse(run: Run, model_file: str) -> float:
    """Repeat 0's monitored MSE of a saved model, computed as Keras computes it.

    Float32 predictions against float32 targets on the validation rows of repeat
    0, or on its fit rows when nothing is held out.
    """
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    seed = run.summary["repeat_results"][0]["seed"]
    fit_rows, val_rows = train_hkan.split_validation(
        len(data["x_train"]), args.val_fraction, seed)
    rows = val_rows if len(val_rows) else fit_rows
    model = keras.models.load_model(run.run_dir / model_file)
    predicted = keras.ops.convert_to_numpy(model(data["x_train"][rows], training=False))[:, 0]
    return float(np.mean(np.square(predicted - data["y_train"][rows].astype(np.float32))))


# ---------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------


def test_help_prints_usage_and_builds_nothing(monkeypatch, capsys) -> None:
    touched = []

    def _fail(name):
        def sentinel(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"--help reached {name}")
        return sentinel

    for name in ("HKAN", "load_data", "setup_gpu", "prepare_run_dir", "make_dataset"):
        monkeypatch.setattr(train_hkan, name, _fail(name))

    with pytest.raises(SystemExit) as exit_info:
        train_hkan.main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "--training-mode" in out and "closed_form_then_backprop" in out
    assert touched == []


# ---------------------------------------------------------------------
# artifact set and summary, per mode
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_the_run_directory_holds_exactly_the_files_of_its_mode(runs, mode) -> None:
    found = _files(runs[mode].run_dir)
    assert found == EXPECTED_FILES[mode], (
        f"{mode}: unexpected files {sorted(found - EXPECTED_FILES[mode])}, "
        f"missing files {sorted(EXPECTED_FILES[mode] - found)}"
    )


def test_closed_form_writes_none_of_the_backprop_files(runs) -> None:
    assert not (
        _files(runs["closed_form"].run_dir) & (BACKPROP_FILES | {BEST_FILE, CLOSED_FORM_MODEL}))


@pytest.mark.parametrize("mode", MODES)
def test_summary_keys_and_nullness_rules(runs, mode) -> None:
    summary = runs[mode].summary
    assert set(summary) == SUMMARY_KEYS
    assert summary["model"] == "hkan"
    assert summary["training_mode"] == mode
    assert summary["dataset"] == MODE_ARGS[mode][1]
    assert summary["num_inputs"] == 2
    assert summary["train_rows"] == TRAIN_ROWS
    assert summary["test_rows"] == TEST_ROWS
    assert summary["seed"] == 0 and summary["repeats"] == REPEATS
    assert isinstance(summary["params"], int) and summary["params"] > 0
    for key in ("train_rmse", "test_rmse", "fit_seconds"):
        assert set(summary[key]) == SPREAD_KEYS, key
        assert all(isinstance(v, float) and v >= 0 for v in summary[key].values()), key

    two_phase = mode == "closed_form_then_backprop"
    assert (summary["closed_form_phase"] is not None) == two_phase
    assert (summary["best_checkpoint"] is not None) == (mode != "closed_form")
    assert (summary["input_importance"] is not None) == (mode != "backprop")
    if two_phase:
        assert set(summary["closed_form_phase"]) == {"train_rmse", "test_rmse"}
        for spread in summary["closed_form_phase"].values():
            assert set(spread) == SPREAD_KEYS
    if summary["best_checkpoint"] is not None:
        assert set(summary["best_checkpoint"]) == BEST_KEYS
        assert summary["best_checkpoint"]["monitor"] == "val_loss"
        assert summary["best_checkpoint"]["source"] == "epoch"
        assert summary["best_checkpoint"]["file"] == BEST_FILE
        assert summary["best_checkpoint"]["reason"] is None
        assert 1 <= summary["best_checkpoint"]["epoch"] <= EPOCHS
    if summary["input_importance"] is not None:
        assert len(summary["input_importance"]) == summary["num_inputs"]

    expected_png = sorted(
        Path(name).name for name in EXPECTED_FILES[mode] if name.startswith("visualizations/"))
    assert summary["visualizations"] == {"files": expected_png, "failed": []}


@pytest.mark.parametrize("mode", MODES)
def test_repeat_results_carry_the_phases_of_their_mode_and_nothing_private(runs, mode) -> None:
    """The per-repeat keys show which phases actually ran (mode dispatch)."""
    results = runs[mode].summary["repeat_results"]
    assert [r["repeat"] for r in results] == list(range(REPEATS))
    ran_closed_form = mode != "backprop"
    ran_backprop = mode != "closed_form"
    for result in results:
        assert not [key for key in result if key.startswith("_")]
        assert ("layer_rmse" in result) == ran_closed_form
        assert ("forward_deviation_rms" in result) == ran_closed_form
        assert isinstance(result["fit_rows"], int)
        assert ("closed_form_seconds" in result) == ran_closed_form
        assert ("final_loss" in result) == ran_backprop, (
            f"{mode}: the backprop phase {'did not run' if ran_backprop else 'ran'}"
        )
        assert ("closed_form_phase" in result) == (mode == "closed_form_then_backprop")
    assert ("best_checkpoint" in results[0]) == ran_backprop
    assert all("best_checkpoint" not in r for r in results[1:])


@pytest.mark.parametrize("mode", MODES)
def test_summary_spreads_are_the_statistics_of_the_repeats(runs, mode) -> None:
    summary = runs[mode].summary
    for key in ("train_rmse", "test_rmse"):
        values = [r[key] for r in summary["repeat_results"]]
        q25, q75 = np.percentile(values, [25, 75])
        assert summary[key]["median"] == pytest.approx(float(np.median(values)), rel=1e-12)
        assert summary[key]["iqr"] == pytest.approx(float(q75 - q25), rel=1e-9, abs=1e-15)
        assert summary[key]["min"] == min(values) and summary[key]["max"] == max(values)


@pytest.mark.parametrize("mode", ["backprop", "closed_form_then_backprop"])
def test_backprop_logs_one_row_per_epoch(runs, mode) -> None:
    run_dir = runs[mode].run_dir
    lines = (run_dir / "training_log.csv").read_text().strip().splitlines()
    assert lines[0].split(",") == ["epoch", "loss", "val_loss"]
    assert len(lines) == 1 + EPOCHS
    history = json.loads((run_dir / "training_history.json").read_text())
    assert len(history["loss"]) == EPOCHS and len(history["val_loss"]) == EPOCHS


@pytest.mark.parametrize("mode", MODES)
def test_config_json_records_the_parsed_arguments(runs, mode) -> None:
    config = json.loads((runs[mode].run_dir / "config.json").read_text())
    assert config["training_mode"] == mode
    assert config["hidden_units"] == [8] and config["num_basis"] == [6]
    assert config["num_train_samples"] == TRAIN_ROWS


# ---------------------------------------------------------------------
# the saved model
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_final_model_reloads_and_reproduces_the_recorded_test_rmse(runs, mode) -> None:
    """The reload restores the weights exactly, so only the prediction pass can differ.

    The trainer scored in batches of 128; this scores all 120 test rows in one
    call, in float32, and takes the RMSE in float64. Measured on CPU for the three
    modes: relative difference 0.0. The bound allows 1e-6, about eight float32
    ulps of a prediction, for a backend that orders the sums differently.
    """
    run = runs[mode]
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    model = keras.models.load_model(run.run_dir / "final_model.keras")
    predicted = keras.ops.convert_to_numpy(model(data["x_test"], training=False))
    assert predicted.shape == (TEST_ROWS, 1)
    rmse = float(np.sqrt(np.mean(np.square(predicted[:, 0].astype(np.float64) - data["y_test"]))))
    recorded = run.summary["repeat_results"][0]["test_rmse"]
    assert rmse == pytest.approx(recorded, rel=1e-6), (
        f"{mode}: reloaded final_model.keras scores {rmse!r}, the summary recorded {recorded!r}"
    )


@pytest.mark.parametrize("mode", ["backprop", "closed_form_then_backprop"])
def test_best_checkpoint_reloads_and_reproduces_its_recorded_test_rmse(runs, mode) -> None:
    run = runs[mode]
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    best = keras.models.load_model(run.run_dir / run.summary["best_checkpoint"]["file"])
    rmse = train_hkan._rmse(best, data["x_test"], data["y_test"], 128)
    assert rmse == pytest.approx(run.summary["best_checkpoint"]["test_rmse"], rel=1e-6)


def test_load_data_keeps_the_inputs_unrounded_in_float64() -> None:
    """The trainer must fit on the inputs the targets were computed from.

    An earlier version rounded the inputs to float32 in ``load_data``. The targets
    are a function of the unrounded inputs, so the rounding made them inconsistent
    at about 5e-8; on TF5 at width 912 the closed-form fit chased that with weights
    near 70 and the train RMSE was 1.4e-4 instead of 1.4e-7 (decisions.md D-031 of
    plan-2026-09-30T082355-4d999dbc). The guard is on the values, not only the
    dtype: a round trip through float32 must change them.
    """
    args = train_hkan.parse_arguments([*TINY, "--dataset", "tf4"])
    data = train_hkan.load_data(args)
    assert data["x_train"].dtype == np.float64 and data["x_test"].dtype == np.float64
    for key in ("x_train", "x_test"):
        rounded = data[key].astype(np.float32).astype(np.float64)
        assert np.any(rounded != data[key]), f"{key} is already float32-rounded"
    assert data["y_train"].dtype == np.float64 and data["y_test"].dtype == np.float64
    assert data["x_train"].shape == (TRAIN_ROWS, 10) and data["x_test"].shape == (TEST_ROWS, 10)


# ---------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def seed_runs(tmp_path_factory) -> Dict[str, Run]:
    base = tmp_path_factory.mktemp("hkan_seeds")
    common = ("--training-mode", "closed_form", "--dataset", "tf5")
    return {
        "zero_again": _run(base, "zero_again", *common, "--seed", "0", "--repeats", str(REPEATS)),
        "one": _run(base, "one", *common, "--seed", "1", "--repeats", "1"),
        "one_again": _run(base, "one_again", *common, "--seed", "1", "--repeats", "1"),
    }


def _scores(run: Run) -> List[tuple]:
    return [(r["seed"], r["train_rmse"], r["test_rmse"]) for r in run.summary["repeat_results"]]


def test_the_same_seed_gives_identical_closed_form_scores_including_seed_zero(
        runs, seed_runs) -> None:
    assert _scores(runs["closed_form"]) == _scores(seed_runs["zero_again"])
    assert _scores(seed_runs["one"]) == _scores(seed_runs["one_again"])


def test_a_different_seed_gives_a_different_closed_form_score(runs, seed_runs) -> None:
    zero = runs["closed_form"].summary["repeat_results"][0]
    one = seed_runs["one"].summary["repeat_results"][0]
    assert zero["seed"] != one["seed"]
    assert zero["test_rmse"] != one["test_rmse"]
    assert zero["train_rmse"] != one["train_rmse"]


@pytest.mark.parametrize("mode", MODES)
def test_every_repeat_has_its_own_seed_and_its_own_score(runs, mode) -> None:
    results = runs[mode].summary["repeat_results"]
    seeds = [r["seed"] for r in results]
    assert len(set(seeds)) == REPEATS, f"{mode}: the repeats share a seed: {seeds}"
    assert len({r["test_rmse"] for r in results}) == REPEATS, (
        f"{mode}: the repeats have the same test RMSE, they are the same fit"
    )


@pytest.mark.parametrize("train, test, scale", [
    ([1.0e-3, 2.0e-3, 8.0e-3], [1.1e-3, 2.1e-3, 8.1e-3], "linear"),
    ([1.0e-7, 1.2e-7, 1.5e-7], [0.11, 0.12, 0.12], "log"),
])
def test_the_rmse_box_uses_a_log_axis_only_across_a_decade(
        monkeypatch, tmp_path, train, test, scale) -> None:
    """Inside one decade a log axis carries at most one labelled tick (seen in the
    TF5 backprop run of batch 20260930c), so the box plot switches to linear."""
    captured = []
    monkeypatch.setattr(train_hkan.plt, "close", lambda fig: captured.append(fig))
    train_hkan.render_rmse_box(train, test, tmp_path / "box.png")
    assert captured[0].axes[0].get_yscale() == scale


def test_one_repeat_draws_no_box_plot(seed_runs) -> None:
    assert _files(seed_runs["one"].run_dir) == BASE_FILES | CLOSED_FORM_FILES


# ---------------------------------------------------------------------
# a reused experiment name
# ---------------------------------------------------------------------


def test_a_reused_experiment_name_is_refused_and_the_first_run_is_untouched(tmp_path) -> None:
    first = _run(tmp_path, "taken", "--training-mode", "closed_form", "--seed", "0")
    before = _hashes(first.run_dir)
    mtimes = {name: (first.run_dir / name).stat().st_mtime_ns for name in before}

    with pytest.raises(FileExistsError, match="Nothing was written"):
        train_hkan.main(_argv(
            tmp_path, "taken", "--training-mode", "backprop", "--seed", "5", "--epochs", "2"))

    assert _hashes(first.run_dir) == before
    assert {name: (first.run_dir / name).stat().st_mtime_ns for name in before} == mtimes


@pytest.mark.parametrize("artifact", train_hkan.RUN_ARTIFACTS)
def test_the_refusal_comes_before_any_write(tmp_path, artifact) -> None:
    """A directory holding ONE file of an earlier run: refused, and nothing added.

    The earlier run is fabricated so that the check does not depend on a first
    run succeeding: a refusal placed after ``prepare_run_dir`` would already have
    written (or rewritten) ``config.json`` by the time it raises.
    """
    run_dir = tmp_path / "earlier"
    run_dir.mkdir()
    (run_dir / artifact).write_bytes(b"an earlier run's bytes\n")
    before = _hashes(run_dir)

    raised = None
    try:
        train_hkan.main(_argv(tmp_path, "earlier", "--training-mode", "closed_form"))
    except FileExistsError as error:
        raised = error

    after = _hashes(run_dir)
    assert after == before, (
        f"the trainer wrote into a directory that already held {artifact}: "
        f"before {sorted(before)}, after {sorted(after)}, "
        f"changed {sorted(n for n in before if after.get(n) != before[n])}"
    )
    assert raised is not None, f"a directory holding {artifact} was not refused"
    assert artifact in str(raised)


# ---------------------------------------------------------------------
# figures are fail-soft
# ---------------------------------------------------------------------


class _InjectedFigureError(RuntimeError):
    pass


@pytest.mark.parametrize("renderer,figure", [
    ("render_predictions", "test_predictions"),
    ("render_importance", "input_importance"),
])
def test_a_failing_figure_does_not_abort_the_run(monkeypatch, tmp_path, renderer, figure) -> None:
    def _raise(*args, **kwargs):
        raise _InjectedFigureError(f"injected into {renderer}")

    monkeypatch.setattr(train_hkan, renderer, _raise)
    argv = _argv(tmp_path, "figfail", "--training-mode", "closed_form", "--repeats", "2")
    try:
        code = train_hkan.main(argv)
    except _InjectedFigureError as error:
        pytest.fail(f"a failing figure aborted the run before the summary was written: {error}")

    assert code == 0
    run_dir = tmp_path / "figfail"
    summary = _strict_summary(run_dir)
    others = sorted({"test_predictions", "input_importance", "rmse_over_repeats"} - {figure})
    assert summary["visualizations"]["failed"] == [figure]
    assert summary["visualizations"]["files"] == [f"{name}.png" for name in others]
    assert _files(run_dir) == (
        BASE_FILES | CLOSED_FORM_FILES | REPEAT_FILES) - {f"visualizations/{figure}.png"}
    assert f"Figure '{figure}' was not written" in (run_dir / "run.log").read_text()


# ---------------------------------------------------------------------
# validation split
# ---------------------------------------------------------------------


@pytest.mark.parametrize("num_rows,fraction,seed", [
    (300, 0.1, 0), (300, 0.25, 7), (11, 0.5, 3), (1000, 0.001, 0), (50, 0.0, 0), (50, 0.0, 9),
])
def test_split_validation_partitions_the_train_rows(num_rows, fraction, seed) -> None:
    fit, val = train_hkan.split_validation(num_rows, fraction, seed)
    expected_val = 0 if fraction == 0 else max(1, int(round(fraction * num_rows)))
    assert len(val) == expected_val
    assert len(fit) == num_rows - expected_val
    assert np.array_equal(fit, np.sort(fit)) and np.array_equal(val, np.sort(val))
    assert not set(fit.tolist()) & set(val.tolist())
    assert np.array_equal(np.sort(np.concatenate([fit, val])), np.arange(num_rows))
    if fraction == 0:
        assert val.size == 0 and np.array_equal(fit, np.arange(num_rows))


def test_split_validation_is_seeded_and_not_a_tail_slice() -> None:
    fit_a, val_a = train_hkan.split_validation(200, 0.2, 0)
    fit_b, val_b = train_hkan.split_validation(200, 0.2, 0)
    _, val_c = train_hkan.split_validation(200, 0.2, 1)
    assert np.array_equal(val_a, val_b) and np.array_equal(fit_a, fit_b)
    assert not np.array_equal(val_a, val_c)
    assert not np.array_equal(val_a, np.arange(160, 200))
    assert not np.array_equal(val_a, np.arange(40))


def test_split_validation_refuses_to_leave_fewer_than_two_fit_rows() -> None:
    with pytest.raises(ValueError, match="at least 2 are needed"):
        train_hkan.split_validation(3, 0.9, 0)


@pytest.mark.parametrize("mode", MODES)
def test_recorded_fit_rows_is_train_rows_minus_validation_rows(runs, mode) -> None:
    # default --val-fraction 0.1 of 300 rows is 30; closed_form holds nothing out
    held_out = 0 if mode == "closed_form" else 30
    assert runs[mode].summary["fit_rows"] == TRAIN_ROWS - held_out


@pytest.fixture
def disjoint_csv(tmp_path):
    """Train rows with every value in [0, 1]; test rows with every value in [10, 11].

    The test file has MORE rows than the train file, so train row indices are also
    valid test row indices: a validation part carved from the test split would not
    crash, it would hand ``fit`` rows from the [10, 11] range.
    """
    rng = np.random.default_rng(5)
    train = rng.uniform(0.0, 1.0, (80, 3))
    test = rng.uniform(10.0, 11.0, (120, 3))
    np.savetxt(tmp_path / "train.csv", train, delimiter=",", fmt="%.17g")
    np.savetxt(tmp_path / "test.csv", test, delimiter=",", fmt="%.17g")
    return tmp_path / "train.csv", tmp_path / "test.csv", train, test


@pytest.mark.parametrize("mode", ["backprop", "closed_form_then_backprop"])
def test_validation_rows_come_from_the_train_split_never_the_test_split(
        monkeypatch, tmp_path, disjoint_csv, mode) -> None:
    train_csv, test_csv, train, _ = disjoint_csv
    captured = []
    real_fit = keras.Model.fit

    def _recording_fit(self, x=None, y=None, **kwargs):
        captured.append({
            "x": np.array(x), "y": np.array(y),
            "validation_data": kwargs.get("validation_data"),
        })
        return real_fit(self, x, y, **kwargs)

    monkeypatch.setattr(train_hkan.HKAN, "fit", _recording_fit)
    argv = [
        "--dataset", "csv", "--train-csv", str(train_csv), "--test-csv", str(test_csv),
        "--training-mode", mode, "--hidden-units", "6", "--num-basis", "5",
        "--epochs", "2", "--val-fraction", "0.25", "--repeats", "2",
        "--output-dir", str(tmp_path), "--experiment-name", "split",
    ]
    assert train_hkan.main(argv) == 0

    assert len(captured) == 2, "fit() was not called once per repeat"
    train_rows = np.hstack([train[:, :-1].astype(np.float32),
                            train[:, -1:].astype(np.float32)])
    train_sorted = train_rows[np.lexsort(train_rows.T)]
    for call in captured:
        assert call["validation_data"] is not None
        x_val, y_val = (np.array(part) for part in call["validation_data"])
        assert x_val.shape == (20, 2) and call["x"].shape == (60, 2)
        assert x_val.max() <= 1.0 and y_val.max() <= 1.0, (
            f"validation rows reach {x_val.max():.3f} / {y_val.max():.3f}: they come from "
            "the test split (every test value is in [10, 11], every train value in [0, 1])"
        )
        assert call["x"].max() <= 1.0 and call["y"].max() <= 1.0
        # fit rows and validation rows together are exactly the train rows, once each
        seen = np.vstack([
            np.hstack([call["x"], call["y"].reshape(-1, 1)]),
            np.hstack([x_val, y_val.reshape(-1, 1)]),
        ]).astype(np.float32)
        np.testing.assert_array_equal(seen[np.lexsort(seen.T)], train_sorted)

    summary = _strict_summary(tmp_path / "split")
    assert summary["train_rows"] == 80 and summary["test_rows"] == 120
    assert summary["fit_rows"] == 60 == summary["train_rows"] - 20
    # different repeats hold out different rows
    assert not np.array_equal(captured[0]["validation_data"][0], captured[1]["validation_data"][0])


# ---------------------------------------------------------------------
# the two-phase mode scores the closed-form fit BEFORE the fine-tune
# ---------------------------------------------------------------------


def test_the_closed_form_phase_is_the_score_before_the_fine_tune(runs) -> None:
    """Refit the closed form alone on the same rows and compare.

    ``fit_closed_form`` is deterministic given the model seed and the fit rows, so
    the recorded phase must equal an independent closed-form fit (measured
    relative difference on CPU: 0.0; bound 1e-6 as in the reload test), and must
    differ from the score after three epochs of Adam.
    """
    run = runs["closed_form_then_backprop"]
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    for result in run.summary["repeat_results"]:
        fit_rows, _ = train_hkan.split_validation(TRAIN_ROWS, args.val_fraction, result["seed"])
        model = train_hkan.build_model(args, result["seed"])
        model.fit_closed_form(data["x_train"][fit_rows], data["y_train"][fit_rows])
        reference = {
            "train_rmse": train_hkan._rmse(
                model, data["x_train"][fit_rows], data["y_train"][fit_rows], 128),
            "test_rmse": train_hkan._rmse(model, data["x_test"], data["y_test"], 128),
        }
        for key, value in reference.items():
            assert result["closed_form_phase"][key] == pytest.approx(value, rel=1e-6), (
                f"repeat {result['repeat']}: recorded closed-form {key} "
                f"{result['closed_form_phase'][key]!r} is not the closed-form fit's "
                f"{value!r}; the score after the fine-tune is {result[key]!r}"
            )
            assert result[key] != pytest.approx(value, rel=1e-6), (
                f"repeat {result['repeat']}: the fine-tune left {key} unchanged"
            )


def test_the_recorded_importance_is_the_first_repeats(runs) -> None:
    """``input_importance`` belongs to repeat 0, the repeat whose model and figures
    are saved: it must equal a closed-form refit at repeat 0's seed on the same
    rows and differ from the refit at repeat 1's seed (repeats draw new centers).
    """
    run = runs["closed_form"]
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    refits = []
    for result in run.summary["repeat_results"][:2]:
        model = train_hkan.build_model(args, result["seed"])
        refits.append(model.fit_closed_form(data["x_train"], data["y_train"])["importance"])
    recorded = np.asarray(run.summary["input_importance"])
    np.testing.assert_allclose(recorded, refits[0], rtol=1e-12, atol=0)
    assert not np.allclose(recorded, refits[1], rtol=1e-12, atol=0), (
        "repeat 0 and repeat 1 have the same importance: the guard cannot tell them apart"
    )


# ---------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("argv,message", [
    (["--dataset", "csv"], "--dataset csv needs both --train-csv and --test-csv"),
    (["--dataset", "csv", "--train-csv", "a.csv"],
     "--dataset csv needs both --train-csv and --test-csv"),
    (["--dataset", "csv", "--test-csv", "b.csv"],
     "--dataset csv needs both --train-csv and --test-csv"),
    (["--repeats", "0"], "--repeats must be >= 1, got 0"),
    (["--epochs", "0"], "--epochs must be >= 1, got 0"),
    (["--batch-size", "0"], "--batch-size must be >= 1, got 0"),
    (["--predict-batch-size", "0"], "--predict-batch-size must be >= 1, got 0"),
    (["--val-fraction", "1"], "--val-fraction must be in [0, 1), got 1.0"),
    (["--val-fraction", "-0.1"], "--val-fraction must be in [0, 1), got -0.1"),
    (["--learning-rate", "0"], "--learning-rate must be > 0, got 0.0"),
    (["--hidden-units", "8,4", "--num-basis", "5,6"],
     "--num-basis takes 1 value or 3 (one per layer), got 2"),
    (["--slope", "1,2,3"], "--slope takes 1 value or 2 (one per layer), got 3"),
    (["--l2-block", ""], "--l2-block takes 1 value or 2 (one per layer), got 0"),
    (["--hidden-units", "eight"], "argument --hidden-units: invalid literal for int()"),
    (["--training-mode", "sgd"], "argument --training-mode: invalid choice: 'sgd'"),
    (["--dataset", "tf6"], "argument --dataset: invalid choice: 'tf6'"),
    *[(argv, message) for argv, message in (
        (["--basis", "foo"], "--basis items must be among sigmoid, gaussian, relu, tanh, "
                             "softplus, identity; got ['foo']"),
        (["--hidden-units", "4", "--basis", "tanh,Tanh"], "--basis items must be among"),
        (["--centers", "grid"], "--centers items must be among random, equally_spaced, data; "
                                "got ['grid']"),
        (["--chunk-size", "0"], "--chunk-size must be >= 1, got 0"),
        (["--seed", "-1"], "--seed must be >= 0, got -1"),
        (["--num-train-samples", "1"], "--num-train-samples must be >= 2, got 1"),
        (["--num-test-samples", "1"], "--num-test-samples must be >= 2, got 1"),
        (["--num-basis", "0"], "--num-basis items must be >= 1, got [0]"),
        (["--hidden-units", "8,0", "--num-basis", "5"], "--hidden-units items must be >= 1"),
        (["--slope", "nan"], "--slope items must be finite, got [nan]"),
        (["--slope", "inf"], "--slope items must be finite, got [inf]"),
        (["--l2-block", "-0.1"], "--l2-block items must be finite and >= 0, got [-0.1]"),
        (["--l2-block", "nan"], "--l2-block items must be finite and >= 0, got [nan]"),
        (["--l2-mix", "-1"], "--l2-mix items must be finite and >= 0, got [-1.0]"),
        (["--l2-mix", "inf"], "--l2-mix items must be finite and >= 0, got [inf]"),
    )],
])
def test_parse_arguments_rejects_bad_input_on_stderr(capsys, argv, message) -> None:
    with pytest.raises(SystemExit) as exit_info:
        train_hkan.parse_arguments(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_parse_arguments_accepts_the_edges_of_the_valid_ranges() -> None:
    args = train_hkan.parse_arguments(
        ["--val-fraction", "0", "--repeats", "1", "--hidden-units", "", "--num-basis", "7"])
    assert args.val_fraction == 0.0 and args.hidden_units == [] and args.num_basis == [7]
    args = train_hkan.parse_arguments(["--hidden-units", "8, 4", "--basis", "tanh,relu,identity"])
    assert args.hidden_units == [8, 4] and args.basis == ["tanh", "relu", "identity"]


def test_build_model_passes_one_value_or_a_per_layer_list() -> None:
    args = train_hkan.parse_arguments(
        ["--hidden-units", "8,4", "--num-basis", "5,6,7", "--slope", "2.5", "--no-bias"])
    config = train_hkan.build_model(args, seed=42).get_config()
    assert list(config["hidden_units"]) == [8, 4]
    assert list(config["num_basis"]) == [5, 6, 7]
    assert config["slope"] == 2.5 or list(config["slope"]) == [2.5, 2.5, 2.5]
    assert config["use_bias"] is False and config["use_block_bias"] is True
    assert config["seed"] == 42


# ---------------------------------------------------------------------
# a bad argument must not use up the experiment name
# ---------------------------------------------------------------------

BAD_ARGUMENTS = [
    ("--basis", "foo"), ("--centers", "grid"), ("--chunk-size", "0"), ("--seed", "-1"),
    ("--num-train-samples", "1"), ("--num-test-samples", "1"), ("--num-basis", "0"),
    ("--hidden-units", "0"), ("--slope", "nan"), ("--l2-block", "-0.1"), ("--l2-mix", "inf"),
]


@pytest.mark.parametrize("bad", BAD_ARGUMENTS, ids=[" ".join(bad) for bad in BAD_ARGUMENTS])
def test_a_bad_argument_is_refused_before_anything_is_written(tmp_path, bad) -> None:
    """Later flags win in argparse, so the bad value replaces the tiny one."""
    raised = None
    try:
        train_hkan.main([*_argv(tmp_path, "burned", "--training-mode", "closed_form"), *bad])
    except BaseException as error:  # SystemExit is the expected one
        raised = error
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert written == [], f"{' '.join(bad)} was refused only after writing {written}"
    assert isinstance(raised, SystemExit) and raised.code == 2, (
        f"{' '.join(bad)}: expected the parser to exit with code 2, got {raised!r}")


def test_basis_and_center_names_are_the_model_packages() -> None:
    """The parser imports the allowed names; this pins what they are."""
    assert train_hkan.BASIS_NAMES is hkan_layer_module.BASIS_NAMES
    assert train_hkan.CENTER_MODES is hkan_layer_module.CENTER_MODES
    assert set(train_hkan.BASIS_NAMES) == set(hkan_layer_module.KERAS_BASIS) == {
        "sigmoid", "gaussian", "relu", "tanh", "softplus", "identity"}
    assert set(train_hkan.CENTER_MODES) == {"random", "equally_spaced", "data"}
    for name in train_hkan.BASIS_NAMES:
        assert train_hkan.parse_arguments(["--basis", name]).basis == [name]
    for mode in train_hkan.CENTER_MODES:
        assert train_hkan.parse_arguments(["--centers", mode]).centers == [mode]


def test_slope_and_ridge_defaults_are_the_model_constructors() -> None:
    args = train_hkan.parse_arguments([])
    signature = inspect.signature(hkan_model_module.HKAN.__init__).parameters
    for flag in ("slope", "l2_block", "l2_mix"):
        assert getattr(args, flag) == [signature[flag].default], (
            f"--{flag.replace('_', '-')} defaults to {getattr(args, flag)}, the model's "
            f"constructor to {signature[flag].default}")
    assert args.slope == [hkan_layer_module.DEFAULT_SLOPE]
    assert args.l2_block == [hkan_model_module.DEFAULT_L2_BLOCK]
    assert args.l2_mix == [hkan_model_module.DEFAULT_L2_MIX]


# ---------------------------------------------------------------------
# the validation split has its own generator
# ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_the_validation_split_is_no_other_stream(seed) -> None:
    """Not the caller's ``default_rng(seed)``, not the old ``[seed, 1]``, no model stream."""
    num_rows, num_val = 200, 40
    _, val = train_hkan.split_validation(num_rows, 0.2, seed)

    def held_out(seed_list) -> np.ndarray:
        return np.sort(np.random.default_rng(seed_list).permutation(num_rows)[:num_val])

    tag = train_hkan.VALIDATION_SPLIT_TAG
    assert tag != 0 and tag != hkan_layer_module.SEED_DOMAIN_TAG
    np.testing.assert_array_equal(val, held_out([seed, tag]))  # the tag is LAST
    others = {"default_rng(seed)": seed, "[seed, 1]": [seed, 1], "[seed, 0, tag]": [seed, 0, tag]}
    for layer_index in range(3):
        for stream in range(3):
            others[f"model stream layer {layer_index}, stream {stream}"] = [
                seed, layer_index, stream, hkan_layer_module.SEED_DOMAIN_TAG]
    for name, seed_list in others.items():
        assert not np.array_equal(val, held_out(seed_list)), (
            f"the validation rows are the first {num_val} of {name}'s permutation")


# ---------------------------------------------------------------------
# a diverged run still finishes
# ---------------------------------------------------------------------


def test_a_diverged_run_finishes_and_reports_no_best_checkpoint(extra_runs) -> None:
    """Learning rate 1e12: every epoch's loss is NaN and no checkpoint is written."""
    run = extra_runs["diverged"]
    history = json.loads((run.run_dir / "training_history.json").read_text())
    assert len(history["val_loss"]) == EPOCHS
    assert not np.any(np.isfinite(np.array(history["val_loss"], dtype=float))), (
        "the run did not diverge, the test does not reach the path it is for")
    files = _files(run.run_dir)
    assert BEST_FILE not in files
    assert {"results_summary.json", "final_model.keras", "training_log.csv"} <= files
    best = run.summary["best_checkpoint"]
    assert set(best) == BEST_KEYS
    assert best["monitor"] == "val_loss" and best["source"] == "none"
    assert all(best[key] is None for key in
               ("file", "epoch", "monitor_value", "train_rmse", "test_rmse"))
    assert "never finite" in best["reason"]
    # non-finite scores are written as null, never as a NaN token (strict JSON)
    assert run.summary["repeat_results"][0]["test_rmse"] is None
    assert run.summary["test_rmse"]["median"] is None
    assert "No best checkpoint" in (run.run_dir / "run.log").read_text()


# ---------------------------------------------------------------------
# two-phase mode: which model is the best
# ---------------------------------------------------------------------


def test_a_fine_tune_that_hurts_leaves_the_closed_form_model_as_the_best(extra_runs) -> None:
    run = extra_runs["fine_tune_hurts"]
    start = _monitor_mse(run, CLOSED_FORM_MODEL)
    history = json.loads((run.run_dir / "training_history.json").read_text())
    assert min(history["val_loss"]) > start, (
        "an epoch beat the closed-form start, the test does not reach the path it is for")

    files = _files(run.run_dir)
    assert BEST_FILE not in files, (
        f"best_model.keras was written although every epoch ({history['val_loss']}) is "
        f"worse than the closed-form start ({start})")
    assert CLOSED_FORM_MODEL in files
    best = run.summary["best_checkpoint"]
    assert set(best) == BEST_KEYS
    assert best["source"] == "closed_form" and best["file"] == CLOSED_FORM_MODEL
    assert best["epoch"] == 0 and best["monitor"] == "val_loss"
    assert best["monitor_value"] == pytest.approx(start, rel=1e-6)
    phase = run.summary["repeat_results"][0]["closed_form_phase"]
    assert (best["train_rmse"], best["test_rmse"]) == (phase["train_rmse"], phase["test_rmse"])
    assert "closed-form start" in best["reason"]

    # the file it points at is that model
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    model = keras.models.load_model(run.run_dir / CLOSED_FORM_MODEL)
    assert train_hkan._rmse(model, data["x_test"], data["y_test"], 128) == pytest.approx(
        phase["test_rmse"], rel=1e-6)


def test_a_fine_tune_that_helps_writes_a_best_model_below_the_closed_form_start(runs) -> None:
    run = runs["closed_form_then_backprop"]
    start = _monitor_mse(run, CLOSED_FORM_MODEL)
    best = run.summary["best_checkpoint"]
    assert best["source"] == "epoch" and best["file"] == BEST_FILE
    assert (run.run_dir / BEST_FILE).is_file() and (run.run_dir / CLOSED_FORM_MODEL).is_file()
    assert best["monitor_value"] < start
    assert _monitor_mse(run, BEST_FILE) < start
    assert _monitor_mse(run, BEST_FILE) == pytest.approx(best["monitor_value"], rel=1e-5)


# ---------------------------------------------------------------------
# the best checkpoint is the best epoch
# ---------------------------------------------------------------------


def test_the_best_checkpoint_is_the_argmin_epoch_and_reproduces_its_value(extra_runs) -> None:
    """Measured relative difference between the reloaded file's validation MSE and
    the epoch's logged ``val_loss``: below 1e-7 on CPU; the bound allows 1e-5."""
    run = extra_runs["best_in_the_middle"]
    history = json.loads((run.run_dir / "training_history.json").read_text())
    values = history["val_loss"]
    expected_epoch = int(np.argmin(values)) + 1
    assert 1 < expected_epoch < len(values), (
        f"the best epoch of {values} is the first or the last: an argmax or a "
        "last-epoch checkpoint could pass")
    best = run.summary["best_checkpoint"]
    assert best["epoch"] == expected_epoch, f"recorded epoch {best['epoch']}, monitor {values}"
    assert best["monitor_value"] == min(values)
    reloaded = _monitor_mse(run, BEST_FILE)
    assert reloaded == pytest.approx(min(values), rel=1e-5), (
        f"best_model.keras has validation MSE {reloaded}; the monitor's minimum is "
        f"{min(values)} and its last value {values[-1]}")


# ---------------------------------------------------------------------
# --val-fraction 0, end to end
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["backprop", "closed_form_then_backprop"])
def test_val_fraction_zero_monitors_the_train_loss(monkeypatch, tmp_path, mode) -> None:
    seen = {"checkpoint": [], "fit": []}
    real_checkpoint, real_fit = keras.callbacks.ModelCheckpoint, keras.Model.fit

    def _checkpoint(*args, **kwargs):
        seen["checkpoint"].append(kwargs)
        return real_checkpoint(*args, **kwargs)

    def _fit(self, x=None, y=None, **kwargs):
        seen["fit"].append({"rows": len(x), "validation_data": kwargs.get("validation_data")})
        return real_fit(self, x, y, **kwargs)

    monkeypatch.setattr(keras.callbacks, "ModelCheckpoint", _checkpoint)
    monkeypatch.setattr(train_hkan.HKAN, "fit", _fit)
    crashed = None
    try:
        train_hkan.main(_argv(
            tmp_path, "vf0", "--training-mode", mode, "--dataset", "tf2", "--seed", "0",
            "--val-fraction", "0", "--epochs", str(EPOCHS)))
    except Exception as error:  # reported after the assertions on what was captured
        crashed = error

    assert [call["validation_data"] for call in seen["fit"]] == [None], (
        "fit() got validation data although nothing is held out")
    assert [call["rows"] for call in seen["fit"]] == [TRAIN_ROWS]
    assert [kwargs["monitor"] for kwargs in seen["checkpoint"]] == ["loss"], (
        "the checkpoint monitors a validation loss that does not exist")
    assert crashed is None, f"the run crashed: {crashed!r}"

    run_dir = tmp_path / "vf0"
    summary = _strict_summary(run_dir)
    assert summary["fit_rows"] == TRAIN_ROWS == summary["train_rows"]
    assert summary["best_checkpoint"]["monitor"] == "loss"
    assert summary["best_checkpoint"]["source"] in ("epoch", "closed_form")
    assert (run_dir / summary["best_checkpoint"]["file"]).is_file()
    assert ((run_dir / BEST_FILE).is_file()) == (summary["best_checkpoint"]["source"] == "epoch")
    history = json.loads((run_dir / "training_history.json").read_text())
    assert set(history) == {"loss"} and len(history["loss"]) == EPOCHS
    # Keras' CSVLogger always declares a `val_loss` column and fills it with NA
    log = [line.split(",") for line in (run_dir / "training_log.csv").read_text().splitlines()]
    assert log[0][:2] == ["epoch", "loss"] and len(log) == 1 + EPOCHS
    assert all(cell == "NA" for row in log[1:] for cell in row[2:])
    if mode == "backprop":
        assert summary["best_checkpoint"]["epoch"] == int(np.argmin(history["loss"])) + 1


# ---------------------------------------------------------------------
# median and IQR on three repeats; the aggregated closed-form phase
# ---------------------------------------------------------------------


def _quartile_spread(values) -> Dict[str, float]:
    q25, q75 = np.percentile(values, [25, 75])
    return {"median": float(np.median(values)), "iqr": float(q75 - q25),
            "min": float(np.min(values)), "max": float(np.max(values))}


def test_three_repeats_give_a_median_and_an_interquartile_range(extra_runs) -> None:
    """With two repeats the IQR equals the standard deviation; with three it does not."""
    summary = extra_runs["three_repeats"].summary
    assert summary["repeats"] == 3 and len(summary["repeat_results"]) == 3
    for key in ("train_rmse", "test_rmse", "fit_seconds"):
        values = [r[key] for r in summary["repeat_results"]]
        expected = _quartile_spread(values)
        assert abs(expected["iqr"] - float(np.std(values))) > 0.01 * expected["iqr"], (
            f"{key}: IQR and standard deviation coincide, the test cannot tell them apart")
        assert summary[key] == pytest.approx(expected, rel=1e-9), key


def test_the_aggregated_closed_form_phase_is_the_spread_of_the_closed_form_scores(
        extra_runs) -> None:
    summary = extra_runs["three_repeats"].summary
    for key in ("train_rmse", "test_rmse"):
        phase = [r["closed_form_phase"][key] for r in summary["repeat_results"]]
        final = [r[key] for r in summary["repeat_results"]]
        assert _quartile_spread(phase)["median"] != pytest.approx(
            _quartile_spread(final)["median"], rel=1e-6), (
            f"{key}: the fine-tune left the median unchanged, the test cannot tell them apart")
        assert summary["closed_form_phase"][key] == pytest.approx(
            _quartile_spread(phase), rel=1e-9), (
            f"{key}: the aggregated closed-form phase is not the spread of the repeats' "
            f"closed-form scores {phase}; their final scores are {final}")


# ---------------------------------------------------------------------
# seeds in the backprop mode
# ---------------------------------------------------------------------


def test_the_same_seed_reproduces_a_backprop_run(runs, extra_runs) -> None:
    """Measured on CPU: bit-identical scores and final losses (difference 0.0).

    The bound allows 1e-6 relative for a device whose kernels are not
    deterministic; an unseeded run differs in the first digit.
    """
    first = runs["backprop"].summary["repeat_results"]
    again = extra_runs["backprop_again"].summary["repeat_results"]
    for a, b in zip(first, again):
        assert a["seed"] == b["seed"]
        for key in ("train_rmse", "test_rmse", "final_loss"):
            assert a[key] == pytest.approx(b[key], rel=1e-6), (
                f"repeat {a['repeat']}: {key} is {a[key]!r} in one run and {b[key]!r} in a "
                "second run with the same arguments")


# ---------------------------------------------------------------------
# every flag's non-default value reaches the thing it controls
# ---------------------------------------------------------------------

WIRED_ARGV = (
    "--training-mode", "closed_form_then_backprop", "--dataset", "tf2", "--seed", "3",
    "--num-train-samples", "120", "--num-test-samples", "60", "--repeats", "2",
    "--hidden-units", "5,4", "--num-basis", "4,5,3", "--basis", "tanh,gaussian,identity",
    "--slope", "2.5,3.5,1.5", "--centers", "equally_spaced,random,data",
    "--l2-block", "0.125,0.25,0.5", "--l2-mix", "0.375,0.0625,0.03125",
    "--no-block-bias", "--no-bias", "--chunk-size", "3",
    "--epochs", "2", "--batch-size", "19", "--learning-rate", "0.03125",
    "--val-fraction", "0.35", "--predict-batch-size", "37",
)
WIRED_FIT_ROWS, WIRED_VAL_ROWS = 78, 42  # round(0.35 * 120) = 42 held out


@pytest.fixture(scope="module")
def wired(tmp_path_factory) -> Dict:
    """One two-phase run with every flag off its default, and what each callee saw.

    The constructor is recorded by replacing the name the trainer calls; the
    methods by wrapping them on the class. Every wrapper calls the real thing.
    """
    base = tmp_path_factory.mktemp("hkan_wired")
    real = train_hkan.HKAN
    seen: Dict[str, list] = {key: [] for key in (
        "init", "fit_closed_form", "fit", "optimizer", "predict", "set_seeds")}

    def _construct(**kwargs):
        seen["init"].append(kwargs)
        return real(**kwargs)

    def _fit_closed_form(self, x, y, **kwargs):
        diagnostics = real_fit_closed_form(self, x, y, **kwargs)
        seen["fit_closed_form"].append({
            "rows": len(x), "kwargs": kwargs,
            "importance": np.array(diagnostics["importance"]),
            "forward_deviation_rms": float(diagnostics["forward_deviation_rms"])})
        return diagnostics

    def _fit(self, x=None, y=None, **kwargs):
        seen["fit"].append({
            "rows": len(x), "batch_size": kwargs.get("batch_size"),
            "epochs": kwargs.get("epochs"),
            "val_rows": len(kwargs["validation_data"][0]) if kwargs.get("validation_data") else 0,
            "learning_rate": float(keras.ops.convert_to_numpy(self.optimizer.learning_rate))})
        return real_fit(self, x, y, **kwargs)

    def _optimizer_builder(*args, **kwargs):
        seen["optimizer"].append((args, kwargs))
        return real_optimizer_builder(*args, **kwargs)

    def _predict(model, x, batch_size):
        seen["predict"].append(batch_size)
        return real_predict(model, x, batch_size)

    def _set_seeds(seed):
        seen["set_seeds"].append(seed)
        return real_set_seeds(seed)

    real_fit_closed_form, real_fit = real.fit_closed_form, keras.Model.fit
    real_optimizer_builder, real_predict = train_hkan.optimizer_builder, train_hkan._predict
    real_set_seeds = train_hkan.set_seeds
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_hkan, "HKAN", _construct)
        patch.setattr(real, "fit_closed_form", _fit_closed_form)
        patch.setattr(real, "fit", _fit)
        patch.setattr(train_hkan, "optimizer_builder", _optimizer_builder)
        patch.setattr(train_hkan, "_predict", _predict)
        patch.setattr(train_hkan, "set_seeds", _set_seeds)
        argv = [*WIRED_ARGV, "--output-dir", str(base), "--experiment-name", "wired"]
        assert train_hkan.main(argv) == 0
    seen["summary"] = _strict_summary(base / "wired")
    return seen


def _constructor(seen: Dict, key: str) -> list:
    assert len(seen["init"]) == 2, "the model was not constructed once per repeat"
    return [kwargs[key] for kwargs in seen["init"]]


#: flag -> (what the callee received, what the CLI asked for); one value per repeat.
WIRED_CHECKS = {
    "--hidden-units": lambda s: (_constructor(s, "hidden_units"), [[5, 4]] * 2),
    "--num-basis": lambda s: (_constructor(s, "num_basis"), [[4, 5, 3]] * 2),
    "--basis": lambda s: (_constructor(s, "basis"), [["tanh", "gaussian", "identity"]] * 2),
    "--slope": lambda s: (_constructor(s, "slope"), [[2.5, 3.5, 1.5]] * 2),
    "--centers": lambda s: (_constructor(s, "centers"), [["equally_spaced", "random", "data"]] * 2),
    "--l2-block": lambda s: (_constructor(s, "l2_block"), [[0.125, 0.25, 0.5]] * 2),
    "--l2-mix": lambda s: (_constructor(s, "l2_mix"), [[0.375, 0.0625, 0.03125]] * 2),
    "--no-block-bias": lambda s: (_constructor(s, "use_block_bias"), [False] * 2),
    "--no-bias": lambda s: (_constructor(s, "use_bias"), [False] * 2),
    "--seed (model)": lambda s: (
        _constructor(s, "seed"), [r["seed"] for r in s["summary"]["repeat_results"]]),
    "--seed (set_seeds)": lambda s: (
        s["set_seeds"], [r["seed"] for r in s["summary"]["repeat_results"]]),
    "--chunk-size": lambda s: (
        [call["kwargs"] for call in s["fit_closed_form"]], [{"chunk_size": 3}] * 2),
    "--learning-rate (optimizer_builder)": lambda s: (
        [args[1] for args, _ in s["optimizer"]], [0.03125] * 2),
    "--learning-rate (the optimizer fit uses)": lambda s: (
        [call["learning_rate"] for call in s["fit"]], [0.03125] * 2),
    "--batch-size": lambda s: ([call["batch_size"] for call in s["fit"]], [19] * 2),
    "--epochs": lambda s: ([call["epochs"] for call in s["fit"]], [2] * 2),
    "--val-fraction (fit rows)": lambda s: (
        [call["rows"] for call in s["fit"]], [WIRED_FIT_ROWS] * 2),
    "--val-fraction (validation rows)": lambda s: (
        [call["val_rows"] for call in s["fit"]], [WIRED_VAL_ROWS] * 2),
    "--val-fraction (closed-form rows)": lambda s: (
        [call["rows"] for call in s["fit_closed_form"]], [WIRED_FIT_ROWS] * 2),
    "--predict-batch-size": lambda s: (sorted(set(s["predict"])), [37]),
}


@pytest.mark.parametrize("flag", sorted(WIRED_CHECKS))
def test_a_non_default_flag_reaches_what_it_controls(wired, flag) -> None:
    received, asked = WIRED_CHECKS[flag](wired)
    assert received == asked, f"{flag}: the callee received {received!r}, the CLI asked {asked!r}"


def test_the_wired_probe_values_all_differ_from_the_defaults() -> None:
    """Otherwise a flag that is never forwarded would still 'arrive'."""
    default, probe = train_hkan.parse_arguments([]), train_hkan.parse_arguments(list(WIRED_ARGV))
    for field in ("hidden_units", "num_basis", "basis", "slope", "centers", "l2_block", "l2_mix",
                  "no_block_bias", "no_bias", "chunk_size", "epochs", "batch_size",
                  "learning_rate", "val_fraction", "predict_batch_size", "seed"):
        assert getattr(default, field) != getattr(probe, field), field
    # per-layer probes differ from the default in EVERY layer, not only the first
    for field in ("slope", "l2_block", "l2_mix"):
        assert getattr(default, field)[0] not in getattr(probe, field), field
    assert "sigmoid" not in probe.basis


def test_recorded_fit_rows_are_the_rows_the_fit_received(wired) -> None:
    summary = wired["summary"]
    received = [call["rows"] for call in wired["fit_closed_form"]]
    assert [r["fit_rows"] for r in summary["repeat_results"]] == received
    assert summary["fit_rows"] == received[0] == WIRED_FIT_ROWS


def test_the_summary_carries_repeat_zeros_importance_and_every_forward_deviation(wired) -> None:
    summary = wired["summary"]
    calls = wired["fit_closed_form"]
    np.testing.assert_array_equal(summary["input_importance"], calls[0]["importance"])
    assert not np.array_equal(calls[0]["importance"], calls[1]["importance"]), (
        "both repeats have the same importance, the test cannot tell them apart")
    assert [r["forward_deviation_rms"] for r in summary["repeat_results"]] == [
        call["forward_deviation_rms"] for call in calls]
    assert all(value > 0 for value in (r["forward_deviation_rms"]
                                       for r in summary["repeat_results"]))


def test_closed_form_mode_fits_on_every_train_row_whatever_the_val_fraction(
        monkeypatch, tmp_path) -> None:
    """``--val-fraction`` is a backprop setting; closed_form must not hold rows out."""
    received = []
    real_fit_closed_form = train_hkan.HKAN.fit_closed_form

    def _fit_closed_form(self, x, y, **kwargs):
        received.append(len(x))
        return real_fit_closed_form(self, x, y, **kwargs)

    monkeypatch.setattr(train_hkan.HKAN, "fit_closed_form", _fit_closed_form)
    run = _run(tmp_path, "all_rows", "--training-mode", "closed_form", "--val-fraction", "0.35")
    assert run.summary["fit_rows"] == received[0], (
        f"the summary says {run.summary['fit_rows']} fit rows, the fit received {received[0]}")
    assert received == [TRAIN_ROWS], (
        f"closed_form held rows out: fitted on {received[0]} of {TRAIN_ROWS} train rows")
    assert run.summary["repeat_results"][0]["fit_rows"] == TRAIN_ROWS


@pytest.mark.parametrize("centers,expected_calls", [("data", REPEATS), ("random", 0)])
def test_backprop_mode_draws_data_centers_from_the_fit_rows(
        monkeypatch, tmp_path, centers, expected_calls) -> None:
    received = []
    real_initialize = train_hkan.HKAN.initialize_centers

    def _initialize(self, x):
        received.append(np.array(x))
        return real_initialize(self, x)

    monkeypatch.setattr(train_hkan.HKAN, "initialize_centers", _initialize)
    run = _run(tmp_path, "centers", "--training-mode", "backprop", "--dataset", "tf2",
               "--epochs", "1", "--repeats", str(REPEATS), "--centers", centers)
    assert len(received) == expected_calls, (
        f"initialize_centers was called {len(received)} times with --centers {centers}")
    args = train_hkan.parse_arguments(run.argv)
    data = train_hkan.load_data(args)
    for x, result in zip(received, run.summary["repeat_results"]):
        fit_rows, _ = train_hkan.split_validation(TRAIN_ROWS, args.val_fraction, result["seed"])
        np.testing.assert_array_equal(x, data["x_train"][fit_rows])


# ---------------------------------------------------------------------
# load_data: the data half of --seed, --scale-csv, the CSV range warning
# ---------------------------------------------------------------------


def test_the_cli_seed_is_the_seed_of_the_data() -> None:
    args = train_hkan.parse_arguments([*TINY, "--dataset", "tf5", "--seed", "3"])
    data = train_hkan.load_data(args)
    expected = make_dataset("tf5", seed=3, num_train=TRAIN_ROWS, num_test=TEST_ROWS)
    other = make_dataset("tf5", seed=0, num_train=TRAIN_ROWS, num_test=TEST_ROWS)
    for key in expected:
        np.testing.assert_array_equal(data[key], expected[key], err_msg=key)
        assert not np.array_equal(data[key], other[key]), key


@pytest.fixture
def wide_csv(tmp_path):
    """Inputs in [-2, 3], target in [50, 90]; the last input column stays in [0, 1]."""
    rng = np.random.default_rng(21)

    def table(rows):
        return np.column_stack([
            rng.uniform(-2.0, 3.0, rows), rng.uniform(0.0, 1.0, rows),
            rng.uniform(50.0, 90.0, rows)])

    train, test = table(40), table(30)
    np.savetxt(tmp_path / "train.csv", train, delimiter=",", fmt="%.17g")
    np.savetxt(tmp_path / "test.csv", test, delimiter=",", fmt="%.17g")
    return ["--dataset", "csv", "--train-csv", str(tmp_path / "train.csv"),
            "--test-csv", str(tmp_path / "test.csv")], train


def test_scale_csv_reaches_the_loader(wide_csv) -> None:
    argv, train = wide_csv
    raw = train_hkan.load_data(train_hkan.parse_arguments(argv))
    np.testing.assert_array_equal(raw["x_train"], train[:, :-1])
    scaled = train_hkan.load_data(train_hkan.parse_arguments([*argv, "--scale-csv"]))
    np.testing.assert_array_equal(scaled["x_train"].min(axis=0), 0.0)
    np.testing.assert_array_equal(scaled["x_train"].max(axis=0), 1.0)
    assert scaled["y_train"].min() == 0.0 and scaled["y_train"].max() == 1.0


def _range_warnings(caplog) -> List[str]:
    return [record.getMessage() for record in caplog.records
            if record.levelno == logging.WARNING and "leave [0, 1]" in record.getMessage()]


@pytest.mark.parametrize("extra,warns", [
    ((), True),
    (("--centers", "equally_spaced"), True),
    (("--hidden-units", "4", "--centers", "random,data"), True),
    (("--scale-csv",), False),
    (("--centers", "data"), False),
    (("--hidden-units", "4", "--centers", "data,random"), False),
])
def test_an_unscaled_csv_outside_the_unit_range_warns_unless_handled(
        caplog, wide_csv, extra, warns) -> None:
    argv, _ = wide_csv
    with caplog.at_level(logging.WARNING, logger="dl"):
        train_hkan.load_data(train_hkan.parse_arguments([*argv, *extra]))
    messages = _range_warnings(caplog)
    assert len(messages) == (1 if warns else 0), messages
    if warns:
        assert "column(s) [0]" in messages[0], "only column 0 leaves [0, 1]"
        assert "--scale-csv" in messages[0] and "--centers" in messages[0]


def test_a_csv_inside_the_unit_range_does_not_warn(caplog, disjoint_csv) -> None:
    train_csv, test_csv, _, _ = disjoint_csv
    argv = ["--dataset", "csv", "--train-csv", str(train_csv), "--test-csv", str(test_csv)]
    with caplog.at_level(logging.WARNING, logger="dl"):
        train_hkan.load_data(train_hkan.parse_arguments(argv))
    assert _range_warnings(caplog) == []
