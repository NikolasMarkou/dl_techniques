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

Nothing may be written under repo-root ``results/``: the autouse fixture in
``tests/conftest.py`` fails any test that adds an entry there.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, NamedTuple

import keras
import numpy as np
import pytest

import train.hkan.train_hkan as train_hkan

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
    "best_model.keras", "training_log.csv", "training_history.json",
    "visualizations/loss_curve.png",
}
EXPECTED_FILES = {
    "closed_form": BASE_FILES | CLOSED_FORM_FILES | REPEAT_FILES,
    "backprop": BASE_FILES | REPEAT_FILES | BACKPROP_FILES,
    "closed_form_then_backprop": BASE_FILES | CLOSED_FORM_FILES | REPEAT_FILES | BACKPROP_FILES,
}
MODES = sorted(EXPECTED_FILES)

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


class _LazyRuns:
    """One real run per training mode, made on first use and kept for the module.

    Lazy and per mode on purpose: a trainer defect that crashes one mode must not
    turn the tests of the other two into fixture errors, where no assertion runs.
    A failed run is remembered and re-raised rather than retried into the
    directory it already started.
    """

    def __init__(self, base: Path) -> None:
        self._base = base
        self._done: Dict[str, object] = {}

    def __getitem__(self, mode: str) -> Run:
        if mode not in self._done:
            try:
                self._done[mode] = _run(
                    self._base, mode, "--training-mode", mode, "--repeats", str(REPEATS),
                    *MODE_ARGS[mode])
            except Exception as error:  # re-raised below, for every test of this mode
                self._done[mode] = error
        if isinstance(self._done[mode], Exception):
            raise self._done[mode]
        return self._done[mode]


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> _LazyRuns:
    """The three per-mode runs, 2 repeats each."""
    return _LazyRuns(tmp_path_factory.mktemp("hkan_runs"))


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
    assert not (_files(runs["closed_form"].run_dir) & BACKPROP_FILES)


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
        assert set(summary["best_checkpoint"]) == {"monitor", "epoch", "train_rmse", "test_rmse"}
        assert summary["best_checkpoint"]["monitor"] == "val_loss"
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
    best = keras.models.load_model(run.run_dir / "best_model.keras")
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
