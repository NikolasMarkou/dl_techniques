"""``TaskTrainer.run_analysis`` and the named failure of an analysis (plan 2faf890a step 12, D-020).

The hook is the one place a task decides how the model analyzer is called. These tests run the
REAL analyzer (a fake one would not see the budget, the pixel path or the swallowed exception that
the step is about) on tiny models, and one-epoch fits of ``vit/vit_pico`` on fake CIFAR-10 arrays:

* the default hook gives a classifier the stock call (``config=None``, so the stock 2048 MB
  information-flow budget and no per-pixel field in the artifact) and a per-pixel model a 256 MB
  budget, read back from the ``config`` block of the analyzer's own ``analysis_results.json``;
* ``analysis_config_for`` passes ``ignore_label`` through for a per-pixel model only;
* a failed analysis is named: a seam that returns ``None`` gives ``failed_analyses:
  ["model_analysis"]``, and a mask run whose void label is not configured gives
  ``["calibration"]`` beside a model status that still reads ``success`` (the swallow at
  ``calibration_analyzer.py``);
* the key is absent when nothing failed, and absent for a caller that did not ask
  (``name_failed_analyses`` off: the legacy trainers).

The tests write only under ``tmp_path``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.config import validate
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
)
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import classification_run, evaluation, run_summary

from tests.classification_fakes import use_tiny_raw

VOID = 255
SIZE = 16
CLASSES = 3


def make_config(out, **changes):
    """Return a validated ``vit/vit_pico`` config on cifar10."""
    mapping = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        "epochs": 1,
        "analysis_args": {"figures": False},
    }
    mapping.update(changes)
    return validate(ImageClassificationConfig, mapping)


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def mask_model(ignore_class=VOID):
    """A small fully convolutional net: ``(N, 16, 16, 3)`` images to ``(N, 16, 16, 3)`` logits."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input((SIZE, SIZE, 3))
    h = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    model = keras.Model(inputs, keras.layers.Conv2D(CLASSES, 1)(h))
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=ignore_class),
    )
    return model


def mask_data(void=False, n=12):
    rng = np.random.default_rng(0)
    x = rng.random((n, SIZE, SIZE, 3), dtype=np.float32)
    y = np.digitize(x[..., 0], [1 / 3, 2 / 3]).astype(np.int32)
    if void:
        y[:, :4, :4] = VOID
    return x, y


def classifier():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    h = keras.layers.GlobalAveragePooling2D()(keras.layers.Conv2D(4, 3, activation="relu")(inputs))
    model = keras.Model(inputs, keras.layers.Dense(CLASSES)(h))
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def classifier_data(n=30):
    rng = np.random.default_rng(1)
    return rng.random((n, 8, 8, 3), dtype=np.float32), rng.integers(0, CLASSES, n).astype(np.int32)


def history(model, x, y):
    return model.fit(x, y, epochs=2, batch_size=8, verbose=0, validation_split=0.25)


def analysis_file(run_dir):
    return json.loads((Path(run_dir) / "model_analysis" / "analysis_results.json").read_text())


# ---------------------------------------------------------------------
# The hook and its configuration
# ---------------------------------------------------------------------


def test_the_hook_is_optional_and_classification_inherits_it(world, tmp_path):
    assert "run_analysis" not in REQUIRED_HOOKS
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    assert type(task).run_analysis is TaskTrainer.run_analysis


def test_the_configuration_of_a_classifier_is_the_stock_one():
    assert evaluation.analysis_config_for(classifier()) is None
    assert evaluation.analysis_config_for(classifier(), ignore_label=VOID) is None


def test_the_configuration_of_a_per_pixel_model_carries_the_budget_and_the_void_label():
    config = evaluation.analysis_config_for(mask_model(), ignore_label=VOID)
    assert config.memory_limit_mb == evaluation.PER_PIXEL_MEMORY_LIMIT_MB == 256
    assert config.ignore_label == VOID
    assert (config.analyze_weights, config.analyze_calibration, config.analyze_information_flow,
            config.analyze_training_dynamics, config.analyze_spectral) == (True,) * 5
    assert evaluation.analysis_config_for(mask_model()).ignore_label is None


def test_the_default_hook_passes_a_256_mb_budget_to_the_real_analyzer_for_a_mask_model(world, tmp_path):
    """Read from the artifact the real ModelAnalyzer wrote, not from a spy on the call."""
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    model = mask_model()
    x, y = mask_data()
    results = task.run_analysis(task.config, model, (x, y), history(model, x, y), "seg",
                                str(tmp_path / "out"))
    assert results is not None
    payload = analysis_file(tmp_path / "out")
    assert payload["config"]["memory_limit_mb"] == 256
    assert payload["config"]["ignore_label"] is None and payload["config"]["max_pixels"] == 100_000
    # the per-pixel path ran: counts are recorded
    assert payload["calibration_metrics"]["seg"]["pixels_used"] == 12 * SIZE * SIZE


def test_the_default_hook_leaves_a_classifier_on_the_stock_budget_and_artifact(world, tmp_path):
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    model = classifier()
    x, y = classifier_data()
    results = task.run_analysis(task.config, model, (x, y), history(model, x, y), "clf",
                                str(tmp_path / "out"))
    assert results is not None
    payload = analysis_file(tmp_path / "out")
    assert payload["config"]["memory_limit_mb"] == 2048
    assert not {"ignore_label", "max_pixels"} & set(payload["config"])
    assert "pixels_used" not in payload["calibration_metrics"]["clf"]


# ---------------------------------------------------------------------
# A failed analysis is named
# ---------------------------------------------------------------------


def report(task, model, x, y, hist, run_dir, **kwargs):
    """``report_best_weights`` as the classification task calls it, figures off."""
    data = SimpleNamespace(x_fig=x, y_fig=y)
    _, analysis = classification_run.report_best_weights(
        model, data, hist, Path(run_dir), Path(run_dir) / "visualizations", "seg", True,
        figures=False, **kwargs,
    )
    return analysis


def test_a_seam_that_returns_none_is_named_model_analysis(world, tmp_path):
    (tmp_path / "run").mkdir()
    analysis = report(
        None, classifier(), *classifier_data(), keras.callbacks.History(), tmp_path / "run",
        run_analysis=lambda *args: None, name_failed_analyses=True,
    )
    assert analysis["failed_analyses"] == [run_summary.MODEL_ANALYSIS_FAILED] == ["model_analysis"]
    assert analysis["status"] == "missing"


def test_calibration_that_raised_is_named_beside_a_model_status_that_still_reads_success(world, tmp_path):
    """The swallow: void labels, no ``ignore_label``. ``model.evaluate`` is fine (``ignore_class``),
    calibration raises on 255 inside the analyzer and is skipped; nothing else shows it."""
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    model = mask_model()
    x, y = mask_data(void=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    analysis = report(
        task, model, x, y, keras.callbacks.History(), run_dir,
        run_analysis=lambda *args: task.run_analysis(task.config, *args),
        name_failed_analyses=True,
    )
    assert analysis["status"] == "success"
    assert analysis["failed_analyses"] == ["calibration"]
    assert "seg" not in analysis_file(run_dir)["calibration_metrics"]


def test_nothing_failed_writes_no_key(world, tmp_path):
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    model = mask_model()
    x, y = mask_data(void=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    analysis = report(
        task, model, x, y, keras.callbacks.History(), run_dir,
        run_analysis=lambda *args: task.run_analysis(task.config, *args),
        name_failed_analyses=True,
    )
    assert analysis["status"] == "success" and "failed_analyses" not in analysis


def test_a_caller_that_does_not_ask_keeps_the_legacy_analyzer_block(world, tmp_path):
    (tmp_path / "run").mkdir()
    analysis = report(
        None, classifier(), *classifier_data(), keras.callbacks.History(), tmp_path / "run",
        run_analysis=lambda *args: None,
    )
    assert "failed_analyses" not in analysis


def test_failed_analyses_reads_the_requested_analyses_from_the_results():
    from dl_techniques.analyzer import AnalysisConfig
    from dl_techniques.analyzer.data_types import AnalysisResults

    off = AnalysisResults(config=AnalysisConfig(analyze_calibration=False, verbose=False))
    assert run_summary.failed_analyses(off, "m") == []
    on = AnalysisResults(config=AnalysisConfig(verbose=False))
    assert run_summary.failed_analyses(on, "m") == ["calibration"]
    on.calibration_metrics["m"] = {"ece": 0.1}
    assert run_summary.failed_analyses(on, "m") == []
    assert run_summary.failed_analyses(on, "other") == ["calibration"]


# ---------------------------------------------------------------------
# End to end: the task calls its hook, and the summary carries the name
# ---------------------------------------------------------------------


def recording_task(calls, result):
    class Recording(ImageClassificationTrainer):
        def run_analysis(self, config, model, test_data, history, experiment_name, run_dir):
            calls.append(
                {"config": config, "x_shape": tuple(test_data[0].shape),
                 "name": experiment_name, "run_dir": run_dir}
            )
            return result

    return Recording


def test_the_classification_task_reports_through_its_hook_and_the_summary_names_the_failure(world, tmp_path):
    calls = []
    run_dir = tmp_path / "run"
    config = make_config(run_dir, analysis_args={"figures": False, "model_analysis": True})
    result = recording_task(calls, None)(config).run()
    (call,) = calls
    assert call["config"].out == config.out and call["run_dir"] == str(run_dir)
    assert call["x_shape"][0] > 0
    summary = json.loads(result.summary_path.read_text())
    assert summary["analyzer"]["failed_analyses"] == ["model_analysis"]


def test_a_successful_classification_analysis_writes_the_legacy_analyzer_block(world, tmp_path):
    run_dir = tmp_path / "run"
    config = make_config(run_dir, analysis_args={"figures": False, "model_analysis": True})
    result = ImageClassificationTrainer(config).run()
    summary = json.loads(result.summary_path.read_text())
    assert set(summary["analyzer"]) == {"status", "loss", "accuracy", "error", "path"}
    assert summary["analyzer"]["status"] == "success"
