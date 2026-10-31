"""``train_image_classification`` and ``python -m train image_classification``, end to end.

Every fit is a real one on fake CIFAR-10 arrays (``tests/classification_fakes``)
with ``vit/vit_pico``, on the CPU, under ``tmp_path``. The fits are shared by the
arms that read their outputs:

* **main**: the Python function, three epochs; the literal file list, the
  request against the resolved config, the log order, the CSV;
* **scripted**: a trainer whose monitor is scripted, so the best epoch is not
  the last; what ``best_model.keras`` and ``final_model.keras`` hold is read
  against the weights recorded at each epoch end;
* **full**: every optional output on (epoch analyzer, model analysis,
  conformal sets);
* one fit per remaining knob, with the effect observed on disk or in the history.

``test_task_trainer.py`` tests the shared sequence with this same task; this file
tests the task's surface (the keyword-only function, the CLI row) and the knobs
no other file observes.
"""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

import train.api.recipes as recipes
from train.api import cli, registry
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import BUILTIN_TASKS, register_builtin_tasks
from train.api.tasks.image_classification import (
    ImageClassificationTrainer,
    train_image_classification,
)
from train.api.trainer import TrainResult

from tests.classification_fakes import use_tiny_raw

from .test_task_trainer import files_of, make_config, read_json

EPOCHS = 3

#: The files of a default run. ``checkpoints/`` is the resume state (Step 24);
#: ``task.json`` is what the predictor reads (Step 26).
MAIN_FILES = {
    "config.json",
    "checkpoints/backup/latest.weights.h5",
    "checkpoints/backup/latest.weights.h5.bkp",
    "checkpoints/backup/training_metadata.json",
    "checkpoints/backup/training_metadata.json.bkp",
    "checkpoints/fit_state.json",
    "run.lock",
    "config_requested.json",
    "results_summary.json",
    "run.log",
    "task.json",
    "training_history.json",
    "training_log.csv",
    "best_model.keras",
    "final_model.keras",
    "tensorboard/",
    "visualizations/classification_report.json",
    "visualizations/confidence_calibration.png",
    "visualizations/confusion_matrix.png",
    "visualizations/misclassifications.png",
    "visualizations/per_class_metrics.png",
    "visualizations/training_dashboard.png",
}


@pytest.fixture(scope="module")
def world():
    """Fake arrays, and a registry holding the six families (a task function registers them itself)."""
    with pytest.MonkeyPatch.context() as patch:
        use_tiny_raw(patch, n_train=40, n_test=64)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        yield


def run_function(out, **changes):
    """Call the Python function the way a caller would, with the test defaults."""
    arguments = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "epochs": 1,
        "batch_size": 4,
        "analysis_args": {"model_analysis": False},
    }
    arguments.update(changes)
    return train_image_classification(**arguments)


def weights_equal(left, right):
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


def saved_weights(path):
    return keras.models.load_model(path).get_weights()


# ---------------------------------------------------------------------
# The main run
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def main(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("main") / "run"
    result = run_function(
        run_dir,
        epochs=EPOCHS,
        metric_args={"success_threshold": 0.0},
        optim_args={"warmup_start_lr": 0},
    )
    return result, run_dir


def test_the_function_returns_the_result(main):
    result, run_dir = main
    assert isinstance(result, TrainResult)
    assert result.run_dir == run_dir
    assert result.success is True


def test_the_run_directory_holds_exactly_these_files(main):
    _, run_dir = main
    assert files_of(run_dir) == MAIN_FILES


def test_the_request_holds_auto_and_the_resolved_config_holds_none(main):
    _, run_dir = main
    requested = read_json(run_dir, "config_requested.json")
    resolved = read_json(run_dir, "config.json")
    assert requested["optim_args"]["lr"] == "auto" and requested["batch_size"] == 4
    assert requested["optim_args"]["warmup_start_lr"] == 0.0
    assert "auto" not in json.dumps(resolved)
    assert resolved["optim_args"]["lr"] == 3e-4
    assert resolved["epochs"] == EPOCHS
    assert resolved["model"] == "vit/vit_pico"


def test_args_are_logged_before_the_resolved_args(main):
    _, run_dir = main
    log = (run_dir / "run.log").read_text()
    assert 0 <= log.index("Args:") < log.index("Resolved args:")


def test_the_history_has_every_epoch_and_the_top_k_metric(main):
    _, run_dir = main
    history = read_json(run_dir, "training_history.json")
    assert len(history["val_accuracy"]) == EPOCHS
    assert "top5_accuracy" in history and "val_top5_accuracy" in history


def test_the_best_checkpoint_reloads_and_is_scored(main):
    result, run_dir = main
    reloaded = keras.models.load_model(run_dir / "best_model.keras")
    assert reloaded(np.zeros((1, 32, 32, 3), "float32")).shape == (1, 10)
    assert set(result.test_metrics_best) >= {"loss", "accuracy", "top5_accuracy"}


# ---------------------------------------------------------------------
# Best is not last
# ---------------------------------------------------------------------

#: The scripted ``val_accuracy`` of each epoch: the best one is the second.
SCRIPT = (0.2, 0.9, 0.3)


class Recorder(keras.callbacks.Callback):
    """Scripts ``val_accuracy`` and records the weights and optimizer at each epoch end."""

    def __init__(self, script):
        super().__init__()
        self.script = script
        self.weights = []
        self.optimizer = None

    def on_epoch_end(self, epoch, logs=None):
        logs["val_accuracy"] = self.script[epoch]
        self.weights.append([np.array(w) for w in self.model.get_weights()])
        self.optimizer = type(self.model.optimizer).__name__


class Scripted(ImageClassificationTrainer):
    """A trainer whose monitor is scripted: ``Recorder`` runs before the checkpoint callbacks."""

    script = SCRIPT
    recorder = None

    def _callbacks(self, *arguments):
        epoch_line, callbacks = super()._callbacks(*arguments)
        type(self).recorder = Recorder(self.script)
        # Index 1: after the learning-rate logger, before EarlyStopping and ModelCheckpoint,
        # which read `logs` in list order.
        callbacks.insert(1, type(self).recorder)
        return epoch_line, callbacks


@pytest.fixture(scope="module")
def scripted(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("scripted") / "run"
    result = Scripted(
        make_config(run_dir, epochs=EPOCHS, **{"metric_args.success_threshold": 0.5})
    ).run()
    return result, run_dir, Scripted.recorder


def test_the_scripted_monitor_made_an_earlier_epoch_the_best(scripted):
    result, run_dir, recorder = scripted
    summary = read_json(run_dir, "results_summary.json")
    assert summary["best_epoch"] == 2 and summary["epochs_run"] == EPOCHS
    assert result.best_metric == 0.9
    assert result.success is True  # 0.9 >= 0.5, though the last epoch scored 0.3
    # Liveness: the epochs really ended in different weights.
    assert not weights_equal(recorder.weights[1], recorder.weights[2])


def test_best_model_holds_the_best_epochs_weights_and_final_model_the_last(scripted):
    _, run_dir, recorder = scripted
    best = saved_weights(run_dir / "best_model.keras")
    final = saved_weights(run_dir / "final_model.keras")
    assert weights_equal(best, recorder.weights[1])
    assert weights_equal(final, recorder.weights[2])
    assert not weights_equal(best, final)


# ---------------------------------------------------------------------
# Knobs: one observed effect each
# ---------------------------------------------------------------------


def test_early_stopping_patience_stops_the_run_and_the_best_weights_come_back(world, tmp_path):
    run_dir = tmp_path / "run"

    class Stopping(Scripted):
        script = (0.9, 0.1, 0.1, 0.1, 0.1)

    result = Stopping(
        make_config(run_dir, epochs=5, **{"checkpoint_args.early_stopping_patience": 1})
    ).run()
    summary = read_json(run_dir, "results_summary.json")
    assert summary["epochs_run"] == 2 and summary["best_epoch"] == 1
    assert summary["stopped_early"] is True
    assert result.best_metric == 0.9
    # The in-memory model was the best weights when the best checkpoint was compared with it.
    assert summary["best_checkpoint_load_error"] is None


@pytest.fixture(scope="module")
def full(world, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("full") / "run"
    result = run_function(
        run_dir,
        epochs=2,
        analysis_args={
            "model_analysis": True,
            "epoch_analysis_every": 2,
            "conformal_alpha": 0.2,
        },
    )
    return result, run_dir


def test_epoch_analysis_every_writes_the_snapshots_the_standard_schedule_names(full):
    _, run_dir = full
    snapshots = sorted(path.name for path in (run_dir / "epoch_analysis").iterdir())
    assert snapshots == ["epoch_000", "epoch_001", "epoch_002"]


def test_model_analysis_true_runs_the_analyzer_and_records_it(full):
    _, run_dir = full
    summary = read_json(run_dir, "results_summary.json")
    assert summary["analyzer"]["status"] == "success"
    assert (run_dir / "model_analysis").is_dir()


def test_conformal_alpha_writes_the_sets_and_records_them(full):
    _, run_dir = full
    summary = read_json(run_dir, "results_summary.json")
    assert (run_dir / "conformal.json").is_file()
    assert summary["conformal"] is not None
    assert summary["conformal"]["alpha"] == 0.2


@pytest.mark.parametrize("terminate, epochs_run", [(True, 1), (False, 3)])
def test_terminate_on_nan_stops_a_diverging_run_at_the_first_bad_batch(
    terminate, epochs_run, world, tmp_path
):
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Training diverged"):
        run_function(
            run_dir,
            epochs=3,
            optim_args={"lr": 1e30, "warmup_epochs": 0},
            debug_args={"terminate_on_nan": terminate},
        )
    assert read_json(run_dir, "results_summary.json")["epochs_run"] == epochs_run


def test_a_mixed_float16_epoch_wraps_the_optimizer_scores_top_k_and_restores_the_policy(
    world, tmp_path
):
    class Mixed(Scripted):
        script = (0.5,)

    before = keras.mixed_precision.global_policy().name
    run_dir = tmp_path / "run"
    result = Mixed(make_config(run_dir, epochs=1, precision="mixed_float16")).run()
    assert Mixed.recorder.optimizer == "LossScaleOptimizer"
    assert keras.mixed_precision.global_policy().name == before == "float32"
    history = read_json(run_dir, "training_history.json")
    # The stock top-k metric cannot take half-precision logits (D-041); ours can (D-051).
    assert np.isfinite(history["top5_accuracy"]).all()
    assert read_json(run_dir, "config.json")["precision"] == "mixed_float16"
    assert result.test_metrics_final["top5_accuracy"] >= 0.0


def test_checkpoint_loads_its_weights_into_the_new_model(main, world, tmp_path):
    _, main_dir = main
    source = main_dir / "final_model.keras"
    trained = saved_weights(source)

    class Probe(Scripted):
        script = (0.5,)
        at_start = None

        def extra_callbacks(self, config, data):
            outer = type(self)

            class AtStart(keras.callbacks.Callback):
                def on_train_begin(self, logs=None):
                    outer.at_start = [np.array(w) for w in self.model.get_weights()]

            return [AtStart()]

    Probe(make_config(tmp_path / "with", epochs=1, checkpoint=str(source))).run()
    loaded = Probe.at_start
    Probe(make_config(tmp_path / "without", epochs=1)).run()
    fresh = Probe.at_start
    assert weights_equal(loaded, trained)
    assert not weights_equal(fresh, trained)


def test_a_checkpoint_of_another_architecture_is_a_config_error_naming_the_key(
    main, world, tmp_path
):
    _, main_dir = main
    with pytest.raises(ConfigError, match="'checkpoint'"):
        run_function(
            tmp_path / "run",
            model="resnet/resnet18",
            checkpoint=str(main_dir / "final_model.keras"),
        )


# ---------------------------------------------------------------------
# The command line, in this process
# ---------------------------------------------------------------------

TOKENS = [
    "model=vit/vit_pico",
    "data.dataset=cifar10",
    "epochs=1",
    "batch_size=4",
    "analysis_args.model_analysis=false",
]


@pytest.fixture
def cli_world(world):
    register_builtin_tasks()
    yield


def run_cli(*tokens):
    return cli.main(["image_classification", *tokens])


def test_the_cli_trains_and_exits_zero(cli_world, tmp_path, capsys):
    out = tmp_path / "run"
    code = run_cli(f"out={out}", "metric_args.success_threshold=0.0", *TOKENS)
    assert code == 0, capsys.readouterr().err
    # One epoch: the backup copies its previous pair before the SECOND save only.
    assert files_of(out) == MAIN_FILES - {
        "checkpoints/backup/latest.weights.h5.bkp", "checkpoints/backup/training_metadata.json.bkp"}


def test_an_unmet_success_threshold_exits_one(cli_world, tmp_path, capsys):
    out = tmp_path / "run"
    assert run_cli(f"out={out}", "metric_args.success_threshold=1.0", *TOKENS) == 1
    assert read_json(out, "results_summary.json")["success"] is False
    assert "Traceback" not in capsys.readouterr().err


def test_an_explicit_erasing_probability_under_another_preset_exits_two_and_claims_nothing(
    cli_world, tmp_path, capsys
):
    out = tmp_path / "run"
    code = run_cli(
        f"out={out}",
        "transform_args.augment=flip",
        "transform_args.random_erasing_prob=0.3",
        *TOKENS,
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err.startswith("error: Invalid value for 'transform_args.random_erasing_prob'")
    assert "cifar_autoaugment" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists(), "the refusal comes before the run directory is claimed"


def test_the_same_probability_under_cifar_autoaugment_is_accepted(cli_world, tmp_path):
    out = tmp_path / "run"
    code = run_cli(
        f"out={out}",
        "transform_args.augment=cifar_autoaugment",
        "transform_args.random_erasing_prob=0.3",
        "metric_args.success_threshold=0.0",
        *TOKENS,
    )
    assert code == 0
    assert read_json(out, "config.json")["transform_args"]["random_erasing_prob"] == 0.3


def test_the_function_registers_the_families_before_it_validates(world, tmp_path, monkeypatch):
    # A Python caller never went through `python -m train`, so the registry is empty (D-040).
    monkeypatch.setattr(registry, "MODELS", {})
    with pytest.raises(ConfigValidationError):
        run_function(tmp_path / "run", epochs="5")
    assert sorted(registry.MODELS) == [
        "cliffordnet", "cliffordnet_hier", "convnext_v1", "convnext_v2", "resnet", "vit",
    ]


def test_the_function_is_keyword_only_and_strict(world, tmp_path):
    with pytest.raises(TypeError):
        train_image_classification(str(tmp_path / "run"), "vit/vit_pico", {"dataset": "cifar10"})
    with pytest.raises(ConfigValidationError, match="expected int, got '5'"):
        run_function(tmp_path / "run", epochs="5")
    assert not (tmp_path / "run").exists()


def test_the_built_in_task_rows_are_what_the_package_registers(world):
    assert [spec.name for spec in BUILTIN_TASKS] == ["image_classification"]
    register_builtin_tasks()
    register_builtin_tasks()
    assert sorted(registry.TASKS) == ["image_classification"]
    assert recipes.TASK == "image_classification"
