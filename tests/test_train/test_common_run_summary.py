"""Unit tests for ``train.common.run_summary``.

These helpers were trainer-local copies in PowerMLP and ConvNeXt (D-015) and are now
defined once. Each guard below was proven RED by injecting the defect it names
(``findings/red-proofs-iter2.md``, step 11):

- a NaN epoch must not be reported as the best one (``np.argmin`` returns its index)
- an empty or non-finite history is not a healthy run
- ``lr`` is the rate used DURING an epoch, so a drop at 0-based index i is epoch i + 1
- one figure failing must not stop the others, and a failed report writes no report file
- the analyzer status is read back from disk, because ``run_model_analysis`` logs
  "completed successfully" whatever happened
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from train.common import run_summary  # noqa: E402
from train.common.callbacks import best_checkpoint_path  # noqa: E402

FIGURES = {"confusion_matrix.png", "per_class_metrics.png", "confidence_calibration.png",
           "misclassifications.png", "classification_report.json"}


# ---------------------------------------------------------------------
# best_epoch / non_finite_metrics / lr_reduction_epochs
# ---------------------------------------------------------------------


def test_best_epoch_picks_the_lowest_value_one_based() -> None:
    assert run_summary.best_epoch({"val_loss": [0.9, 0.5, 0.7]}, "val_loss") == 2
    assert run_summary.best_epoch({"val_loss": [0.4]}, "val_loss") == 1
    assert run_summary.best_epoch({"val_loss": [0.5, 0.3, 0.3]}, "val_loss") == 2, "a tie is the first epoch"


def test_best_epoch_follows_the_monitor_argument_and_its_direction() -> None:
    """The monitor is a parameter, not a module constant: a rising metric picks its maximum."""
    history = {"val_loss": [0.9, 0.5, 0.7], "val_accuracy": [0.6, 0.7, 0.9]}
    assert run_summary.best_epoch(history, "val_loss") == 2
    assert run_summary.best_epoch(history, "val_accuracy") == 3


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_best_epoch_refuses_a_non_finite_history(bad) -> None:
    """``np.argmin([nan, 1.0, 0.5]) == 0``: the NaN epoch would be reported as the best."""
    assert int(np.argmin([float("nan"), 1.0, 0.5])) == 0
    with pytest.raises(ValueError, match="non-finite"):
        run_summary.best_epoch({"val_loss": [bad, 1.0, 0.5]}, "val_loss")
    with pytest.raises(ValueError, match="non-finite"):
        run_summary.best_epoch({"val_loss": [1.0, 0.5, bad]}, "val_loss")


def test_non_finite_metrics_names_the_offending_keys() -> None:
    finite = {"loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}
    assert run_summary.non_finite_metrics(finite, "val_loss") == []
    assert run_summary.non_finite_metrics(
        {"loss": [float("nan")], "val_loss": [1.0]}, "val_loss") == ["loss"]
    assert run_summary.non_finite_metrics(
        {"loss": [1.0], "val_loss": [float("inf")]}, "val_loss") == ["val_loss"]
    assert run_summary.non_finite_metrics({}, "val_loss") == ["loss", "val_loss"], (
        "no epoch at all is not a finite run")
    assert run_summary.non_finite_metrics({"loss": [1.0]}, "val_loss") == ["val_loss"], "absent monitor"


def test_non_finite_metrics_checks_the_named_monitor_and_lists_loss_once() -> None:
    history = {"loss": [1.0], "val_accuracy": [float("nan")], "val_loss": [float("nan")]}
    assert run_summary.non_finite_metrics(history, "val_accuracy") == ["val_accuracy"]
    assert run_summary.non_finite_metrics({"loss": [float("nan")]}, "loss") == ["loss"]


def test_lr_reduction_epochs_are_the_epochs_trained_at_a_lower_rate() -> None:
    """1-based; ``lr`` is the rate USED in each epoch, so a drop at index i (0-based)
    is reported as epoch i + 1."""
    reduce = run_summary.lr_reduction_epochs
    assert reduce([3e-4, 3e-4, 1.5e-4, 1.5e-4, 7.5e-5]) == [3, 5]
    assert reduce([3e-4] * 6) == []
    assert reduce([]) == [] and reduce([3e-4]) == []
    assert reduce([1e-3, 5e-4]) == [2]
    assert reduce([1e-3, 2e-3, 1e-3]) == [3], "a rise is not a reduction"


# ---------------------------------------------------------------------
# write_classification_figures
# ---------------------------------------------------------------------


def _predictions(n: int, c: int, accuracy: float = 0.7):
    rng = np.random.default_rng(0)
    y = rng.integers(0, c, n)
    pred = np.where(rng.random(n) < accuracy, y, (y + 1) % c)
    probs = np.full((n, c), 0.02)
    probs[np.arange(n), pred] = 1.0 - 0.02 * (c - 1)
    return rng, y, probs


def _boom(*args, **kwargs):
    raise RuntimeError("figure exploded")


def test_a_failing_figure_is_recorded_and_the_others_still_render(monkeypatch, tmp_path) -> None:
    rng, y, probs = _predictions(80, 10)
    x = rng.normal(size=(80, 784)).astype("float32")
    monkeypatch.setattr(run_summary, "plot_confusion_matrix", _boom)

    out = run_summary.write_classification_figures(
        tmp_path, x, y, probs, [str(i) for i in range(10)],
        (28, 28, 1), np.array([0.1307], "float32"), np.array([0.3081], "float32"))

    assert out["failed"] == ["confusion_matrix.png"]
    assert set(out["files"]) == FIGURES - {"confusion_matrix.png"}
    assert not (tmp_path / "confusion_matrix.png").exists()
    for name in FIGURES - {"confusion_matrix.png"}:
        assert (tmp_path / name).stat().st_size > 0, name
    assert isinstance(out["ece"], float)


@pytest.mark.parametrize("name,patched", [
    ("per_class_metrics.png", "plot_per_class_metrics"),
    ("confidence_calibration.png", "plot_calibration"),
    ("misclassifications.png", "plot_confident_errors"),
])
def test_each_figure_is_isolated_from_every_other(monkeypatch, tmp_path, name, patched) -> None:
    rng, y, probs = _predictions(80, 10)
    x = rng.normal(size=(80, 784)).astype("float32")
    monkeypatch.setattr(run_summary, patched, _boom)

    out = run_summary.write_classification_figures(
        tmp_path, x, y, probs, [str(i) for i in range(10)], (28, 28, 1))

    assert name in out["failed"]
    written = FIGURES - {name}
    if name == "per_class_metrics.png":
        # The report comes from the per-class figure: no figure, no report file.
        written -= {"classification_report.json"}
        assert not (tmp_path / "classification_report.json").exists()
    assert out["failed"] == [name]
    assert set(out["files"]) == written
    if name == "confidence_calibration.png":
        assert out["ece"] is None


def test_the_misclassification_grid_takes_images_without_a_flat_shape(tmp_path) -> None:
    """ConvNeXt hands NHWC images plus mean and std; PowerMLP hands flat rows plus a shape."""
    rng, y, probs = _predictions(60, 10)
    images = rng.normal(size=(60, 8, 8, 3)).astype("float32")
    nhwc_dir = tmp_path / "nhwc"
    nhwc_dir.mkdir()
    out = run_summary.write_classification_figures(
        nhwc_dir, images, y, probs, [str(i) for i in range(10)],
        mean=np.zeros(3, "float32"), std=np.ones(3, "float32"))
    assert out["failed"] == [] and set(out["files"]) == FIGURES
    assert (nhwc_dir / "misclassifications.png").stat().st_size > 0

    flat = images.reshape(60, -1)
    out = run_summary.write_classification_figures(
        tmp_path, flat, y, probs, [str(i) for i in range(10)], (8, 8, 3),
        np.zeros(3, "float32"), np.ones(3, "float32"))
    assert out["failed"] == [] and set(out["files"]) == FIGURES


def test_a_perfect_classifier_has_no_misclassification_grid_and_that_is_not_a_failure(tmp_path) -> None:
    rng, y, _ = _predictions(40, 4)
    perfect = np.eye(4)[y]
    x = rng.normal(size=(40, 4, 4, 1)).astype("float32")
    out = run_summary.write_classification_figures(tmp_path, x, y, perfect, list("abcd"))
    assert out["failed"] == []
    assert not (tmp_path / "misclassifications.png").exists()


# ---------------------------------------------------------------------
# read_analysis_status / skipped_analysis_status
# ---------------------------------------------------------------------


def _write_analysis(run_dir: Path, payload) -> None:
    (run_dir / "model_analysis").mkdir()
    (run_dir / "model_analysis" / "analysis_results.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload))


def test_analyzer_status_success_is_read_back_from_the_file(tmp_path) -> None:
    _write_analysis(tmp_path, {"model_metrics": {"m": {"status": "success", "loss": 0.5, "accuracy": 0.9}}})
    out = run_summary.read_analysis_status(tmp_path, "m")
    assert out == {"status": "success", "loss": 0.5, "accuracy": 0.9, "error": None,
                   "path": str(tmp_path / "model_analysis" / "analysis_results.json")}


def test_analyzer_status_failed_carries_the_error_text(tmp_path) -> None:
    _write_analysis(tmp_path, {"model_metrics": {"m": {"status": "failed", "error": "OOM"}}})
    out = run_summary.read_analysis_status(tmp_path, "m")
    assert out["status"] == "failed" and out["error"] == "OOM"
    assert out["loss"] is None and out["accuracy"] is None


def test_analyzer_status_a_recorded_skipped_passes_through(tmp_path) -> None:
    _write_analysis(tmp_path, {"model_metrics": {"m": {"status": "skipped"}}})
    assert run_summary.read_analysis_status(tmp_path, "m")["status"] == "skipped"


def test_analyzer_status_without_a_status_key_is_missing_not_success(tmp_path) -> None:
    _write_analysis(tmp_path, {"model_metrics": {"m": {"loss": 0.1}}})
    assert run_summary.read_analysis_status(tmp_path, "m")["status"] == "missing"


@pytest.mark.parametrize("payload", [
    None,                                              # no file at all
    "{not json",                                       # unparseable
    {"model_metrics": {}},                             # model absent
    {"other": 1},                                      # section absent
])
def test_analyzer_status_missing_when_the_file_key_or_json_is_unusable(tmp_path, payload) -> None:
    if payload is not None:
        _write_analysis(tmp_path, payload)
    out = run_summary.read_analysis_status(tmp_path, "m")
    assert out["status"] == "missing"
    assert out["error"], "the reason must be recorded"
    assert out["loss"] is None and out["accuracy"] is None


def test_the_skipped_status_has_the_read_back_schema(tmp_path) -> None:
    skipped = run_summary.skipped_analysis_status()
    assert skipped == {"status": "skipped", "loss": None, "accuracy": None, "error": None, "path": None}
    assert set(skipped) == set(run_summary.read_analysis_status(tmp_path, "m"))


# ---------------------------------------------------------------------
# load_best_metrics
# ---------------------------------------------------------------------


def test_load_best_metrics_evaluates_the_reloaded_checkpoint(tmp_path) -> None:
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
    model.save(best_checkpoint_path(str(tmp_path)))
    seen = {}

    def evaluate(reloaded):
        seen["model"] = reloaded
        return {"loss": 0.25}

    metrics, error = run_summary.load_best_metrics(tmp_path, evaluate)
    assert metrics == {"loss": 0.25} and error is None
    assert isinstance(seen["model"], keras.Model) and seen["model"] is not model


def test_load_best_metrics_reports_a_missing_checkpoint_instead_of_raising(tmp_path) -> None:
    metrics, error = run_summary.load_best_metrics(tmp_path, lambda m: {"loss": 0.0})
    assert metrics is None and error and "Error" in error


def test_load_best_metrics_reports_an_evaluation_error_instead_of_raising(tmp_path) -> None:
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(2)])
    model.save(best_checkpoint_path(str(tmp_path)))

    def evaluate(reloaded):
        raise RuntimeError("bad eval")

    assert run_summary.load_best_metrics(tmp_path, evaluate) == (None, "RuntimeError: bad eval")
