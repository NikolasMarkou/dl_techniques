"""``TaskTrainer.figure_callbacks`` and the one switch, ``analysis_args.figures`` (plan 2faf890a step 11).

A recording task (a subclass of the real classification task that records what the hook and the
reporter are given and writes one file from each) proves, on real one- and two-epoch fits of
``vit/vit_pico`` on fake CIFAR-10 arrays:

* the hook is called once, with the task's config, the loaded data and the run's existing
  ``visualizations`` directory, and its callbacks run after ``EpochLogLine`` and before the
  dashboard;
* with ``figures=False`` the hook is not called, the reporter is told so, and nothing is written
  under ``visualizations`` (the directory is not even created);
* a task whose figure record has no ``ece`` finishes: the summary writes no ``ece`` key instead of
  raising ``KeyError``, while a classification summary keeps the key where it was;
* the dashboard needs no accuracy key (a segmentation history has ``miou`` and no ``accuracy``),
  and importing its module initialises no TensorFlow device (D-055).

The tests write only under ``tmp_path``.
"""

import dataclasses
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import keras
import pytest

import train.api.recipes as recipes
from train.api import registry
from train.api.config import validate
from train.api.tasks.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainer,
)
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import classification_run
from train.common.callbacks import EpochLogLine
from train.common.classification_viz import (
    TrainingDashboardCallback,
    render_training_dashboard,
)

from tests.classification_fakes import use_tiny_raw

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_config(out, **changes):
    """Return a validated ``vit/vit_pico`` config on cifar10 with the analyzer off."""
    mapping = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        "epochs": 2,
        "analysis_args": {"model_analysis": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping[head] = {**mapping.get(head, {}), tail: value}
        else:
            mapping[head] = value
    return validate(ImageClassificationConfig, mapping)


@pytest.fixture(scope="module")
def world():
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=8)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


class Notes:
    """What a :func:`recording_task` saw, in order."""

    def __init__(self):
        self.hook_calls = []
        self.reporter_calls = []
        self.order = []


def recording_task(notes, *, no_ece=False):
    """A classification task that records the figure hook and the reporter, and writes a file in each.

    :param notes: the :class:`Notes` the task fills.
    :param no_ece: when true the reporter returns a figure record without ``ece`` (what a task
        that is not a probability classifier would return).
    """

    class Figure(keras.callbacks.Callback):
        def __init__(self, vis_dir):
            super().__init__()
            self.vis_dir = vis_dir

        def on_epoch_end(self, epoch, logs=None):
            notes.order.append("figure")
            (self.vis_dir / f"recorded_epoch_{epoch + 1}.txt").write_text("x")

    class Recording(ImageClassificationTrainer):
        def figure_callbacks(self, config, data, vis_dir):
            notes.hook_calls.append(
                {
                    "config": config,
                    "data": data,
                    "vis_dir": vis_dir,
                    "existed": Path(vis_dir).is_dir(),
                }
            )
            return [Figure(Path(vis_dir))]

        def evaluate_and_report(self, config, data, fit):
            notes.reporter_calls.append(
                {"figures": config.analysis_args.figures, "vis_dir": fit.vis_dir}
            )
            evaluated = super().evaluate_and_report(config, data, fit)
            if config.analysis_args.figures:
                (fit.vis_dir / "recorded_final.txt").write_text("x")
            if no_ece:
                return dataclasses.replace(
                    evaluated, visualizations={"files": ["recorded_final.txt"], "failed": []}
                )
            return evaluated

    return Recording


@pytest.fixture(scope="module")
def figures_on(world, tmp_path_factory):
    notes = Notes()
    run_dir = tmp_path_factory.mktemp("on") / "run"
    with pytest.MonkeyPatch.context() as patch:
        for cls, name in ((EpochLogLine, "epoch_line"), (TrainingDashboardCallback, "dashboard")):
            original = cls.on_epoch_end

            def wrapped(self, epoch, logs=None, _original=original, _name=name):
                notes.order.append(_name)
                return _original(self, epoch, logs)

            patch.setattr(cls, "on_epoch_end", wrapped)
        config = make_config(run_dir)
        result = recording_task(notes)(config).run()
    return result, run_dir, config, notes


@pytest.fixture(scope="module")
def figures_off(world, tmp_path_factory):
    notes = Notes()
    run_dir = tmp_path_factory.mktemp("off") / "run"
    config = make_config(run_dir, epochs=1, **{"analysis_args.figures": False})
    result = recording_task(notes)(config).run()
    return result, run_dir, config, notes


# ---------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------


def test_the_hook_is_optional_and_its_default_returns_nothing(world, tmp_path):
    assert "figure_callbacks" not in REQUIRED_HOOKS
    task = ImageClassificationTrainer(make_config(tmp_path / "run"))
    assert task.figure_callbacks(task.config, object(), tmp_path) == []
    assert TaskTrainer.figure_callbacks(task, task.config, object(), tmp_path) == []


def test_the_hook_is_called_once_with_the_config_the_data_and_the_existing_vis_dir(figures_on):
    _, run_dir, config, notes = figures_on
    (call,) = notes.hook_calls
    # The resolved config (every "auto" replaced), not the requested one.
    assert call["config"].out == config.out and call["config"].model == "vit/vit_pico"
    assert call["config"].analysis_args.figures is True
    assert call["config"].optim_args.lr is not None and call["config"].optim_args.lr != "auto"
    assert call["data"].n_train > 0
    assert call["vis_dir"] == run_dir / "visualizations"
    assert call["existed"] is True


def test_the_files_the_hook_and_the_reporter_write_land_under_vis_dir(figures_on):
    _, run_dir, _, notes = figures_on
    vis = run_dir / "visualizations"
    assert (vis / "recorded_epoch_1.txt").is_file() and (vis / "recorded_epoch_2.txt").is_file()
    assert (vis / "recorded_final.txt").is_file()
    assert notes.reporter_calls == [{"figures": True, "vis_dir": vis}]
    # The classification figures the reporter writes are unchanged.
    for name in ("training_dashboard.png", "confusion_matrix.png", "classification_report.json"):
        assert (vis / name).is_file(), name


def test_the_task_figure_callbacks_run_after_the_epoch_line_and_before_the_dashboard(figures_on):
    _, _, _, notes = figures_on
    assert notes.order == ["epoch_line", "figure", "dashboard"] * 2


# ---------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------


def test_figures_off_does_not_call_the_hook_and_writes_nothing_under_vis_dir(figures_off):
    _, run_dir, _, notes = figures_off
    assert notes.hook_calls == []
    assert notes.reporter_calls == [{"figures": False, "vis_dir": run_dir / "visualizations"}]
    assert not (run_dir / "visualizations").exists()
    assert not [p for p in run_dir.rglob("recorded_*")]
    summary = json.loads((run_dir / "results_summary.json").read_text())
    assert summary["visualizations"]["files"] == [] and summary["ece"] is None


# ---------------------------------------------------------------------
# A task without ece
# ---------------------------------------------------------------------


def test_a_task_whose_figure_record_has_no_ece_finishes_and_writes_no_ece_key(world, tmp_path):
    notes = Notes()
    run_dir = tmp_path / "run"
    result = recording_task(notes, no_ece=True)(make_config(run_dir, epochs=1)).run()
    assert result.summary_path == run_dir / "results_summary.json"
    summary = json.loads(result.summary_path.read_text())
    assert "ece" not in summary
    assert summary["visualizations"] == {"files": ["recorded_final.txt"], "failed": []}


def _summary(visualizations):
    return classification_run.base_summary(
        {"model": "m"}, {"lr": [0.1], "val_loss": [1.0]}, 1, 1, 1, [], None, None, None, None,
        [0.5], 1.0, visualizations, None, None, {"status": "ok"}, [],
    )


def test_a_classification_summary_keeps_ece_beside_the_figure_record():
    summary = _summary({"files": [], "ece": 0.25, "failed": []})
    keys = list(summary)
    assert summary["ece"] == 0.25
    assert keys.index("ece") + 1 == keys.index("visualizations")
    assert keys.index("fit_wall_seconds") + 1 == keys.index("ece")


def test_a_skipped_classification_record_keeps_ece_as_null():
    summary = _summary({"files": [], "ece": None, "failed": [], "skipped": True})
    assert "ece" in summary and summary["ece"] is None


# ---------------------------------------------------------------------
# The dashboard needs no accuracy
# ---------------------------------------------------------------------

SEGMENTATION_HISTORY = {
    "loss": [1.0, 0.8, 0.7],
    "val_loss": [1.1, 0.9, 0.8],
    "miou": [0.1, 0.2, 0.3],
    "val_miou": [0.1, 0.15, 0.25],
    "lr": [1e-3, 1e-3, 5e-4],
}


def test_the_dashboard_draws_a_history_with_miou_and_no_accuracy(tmp_path):
    out = tmp_path / "dashboard.png"
    panels = render_training_dashboard(
        SEGMENTATION_HISTORY, out, "seg", epoch_times=[1.0, 1.0, 1.0],
        baseline={"loss": 2.0, "miou": 0.0},
    )
    assert panels == ["Loss", "mIoU", "Learning rate", "Generalization gap", "Per-epoch time"]
    assert out.stat().st_size > 0


def test_the_dashboard_draws_a_history_with_nothing_but_a_loss(tmp_path):
    out = tmp_path / "dashboard.png"
    assert render_training_dashboard({"loss": [1.0, 0.5]}, out, "x") == ["Loss"]
    assert out.is_file()


def test_the_dashboard_callback_reads_only_the_keys_the_logs_hold(tmp_path):
    out = tmp_path / "dashboard.png"
    callback = TrainingDashboardCallback(out_path=out, baseline_fn=None)
    callback.on_train_begin()
    for epoch in range(3):
        callback.on_epoch_begin(epoch)
        callback.on_epoch_end(
            epoch,
            {
                "loss": 1.0 - 0.1 * epoch,
                "val_loss": 1.1 - 0.1 * epoch,
                "miou": 0.1 * epoch,
                "val_miou": 0.1 * epoch,
                "lr": 1e-3,
            },
        )
    callback.on_train_end()
    assert set(callback.history) == {"loss", "val_loss", "miou", "val_miou", "lr"}
    assert out.is_file()


def test_importing_the_dashboard_module_initialises_no_tensorflow_device():
    driver = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from tensorflow.python.eager import context as eager

        def initialised():
            return eager.context()._context_handle is not None

        before = initialised()
        import train.common.classification_viz
        after = initialised()
        import keras
        keras.ops.zeros(1)
        print(before, after, initialised())
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True, text=True, check=True, timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg"},
    )
    # Before the import, after it, and after a real op (the control: the probe can see an
    # initialised context, so the middle `False` is a measurement and not a blind spot).
    assert done.stdout.split()[-3:] == ["False", "False", "True"], done.stdout + done.stderr
