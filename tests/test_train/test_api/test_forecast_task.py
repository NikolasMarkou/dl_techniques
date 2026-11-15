"""The ``time_series_forecasting`` task: the hook table, the bar, the seams, the refusals and the registration.

``train.api.tasks.time_series_forecasting.TimeSeriesForecastingTrainer`` is the
task of seven families (``nbeats``, ``prism``, ``tirex``, ``xlstm``, ``ets``, ``timestral``, ``mdn``). It supplies the ten
required hooks, the seams of the shared sequence and the bar, and nothing else (a task overrides hooks, never stages).
This first part pins, on the CPU at tiny sizes (a bank of four synthetic patterns, lookback 32, horizon 8, batch 8):

* **The hook table.** Every hook of ``REQUIRED_HOOKS`` is overridden, and so is each optional one the task relies on;
  ``task_record`` answers the record of ``train.api.predictor.RECORD_KEYS`` (its content is held by
  ``test_forecast_predictor_and_export.py``), no per-epoch figure is attached.
* **The bar** (D-008, D-037). ``val_mae`` is an error, so the bar is STRICTLY below, in float32, and a model that scores
  within a few float32 steps of the threshold is a miss: the compiled metric accumulates in float32 and the baselines are
  float64 numpy, so a model that outputs the naive-last forecast scores one step BELOW its own baseline on some seeds and a
  bare float32 comparison would call it a success (measured, see ``test_a_naive_last_model_never_meets_the_bar``).
  ``success_value`` is the MINIMUM of ``val_mae`` over the selection window; the threshold is the minimum of three terms
  (naive-last, seasonal-naive, epoch-zero) on the SAME fixed select windows, the epoch-zero one read from the fit sidecar so
  that a resume reads the number the first process wrote.
* **The seams.** ``fit_kwargs`` gives ``steps_per_epoch`` and ``validation_steps``; ``resume_stream`` is called by a resume
  that trains (never by a fresh run or a reporting-only resume) and gives another train stream with the split unchanged;
  ``compile_options`` is ``jit_compile=False`` unless the field asks, and ets refuses it by name.
* **Refusals.** Every request the task can refuse before the claim leaves ``out`` absent; the conformal sentence is the
  forecasting one and the two older sentences are untouched.
* **Registration.** One ``TaskSpec`` row, seven family rows that train this task only, the lazy names of ``train.api``,
  ``list_models task=`` and a help path that loads no TensorFlow.

Tests write only under ``tmp_path``.
"""

import dataclasses
import importlib
import inspect
import json
import math
import shutil
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pandas as pd
import pytest

from train.api import cli, registry
from train.api import forecast_models as fm
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.forecast_series import split_bounds
from train.api.tasks import BUILTIN_TASKS
from train.api.tasks.time_series_forecasting import (
    TimeSeriesForecastingConfig,
    TimeSeriesForecastingTrainer,
    train_time_series_forecasting,
)
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer, config_problems
from train.common import forecast_data
from train.common import forecast_run as fr
from train.common.pretrain_data import derive_stream_seed
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

from ._forecast_fixtures import CSV_ROWS, csv_file, world, write_series_csv  # noqa: F401  (``world`` and ``csv_file`` are fixtures)

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "src"

TASK = "time_series_forecasting"
BATCH, LOOKBACK, HORIZON, STEPS = 8, 32, 8, 6
PATTERNS = ["ar_process", "complex_seasonal", "cubic_trend", "arma_process"]
MODEL = "nbeats/tiny"
DATA = {"lookback": LOOKBACK, "horizon": HORIZON, "realizations": 1, "series_length": 600, "patterns": PATTERNS,
        "n_select": 16, "n_test": 16, "steps_per_epoch": STEPS}


def make_config(out: Path, **changes: Any) -> TimeSeriesForecastingConfig:
    """A validated nbeats config at the test sizes; a dotted key (``"debug_args.smoke"``) sets a field of a group."""
    mapping: Dict[str, Any] = {
        "out": str(out),
        "model": MODEL,
        "data": dict(DATA),
        "epochs": 2,
        "batch_size": BATCH,
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping[head] = {**mapping.get(head, {}), tail: value}
        else:
            mapping[head] = value
    return validate(TimeSeriesForecastingConfig, mapping)


def prepared(out: Path, trainer_class=TimeSeriesForecastingTrainer, **changes: Any):
    """``(trainer, resolved config, context)``: what the sequence hands the hooks, resolved the way ``run`` resolves."""
    config = make_config(out, **changes)
    trainer = trainer_class(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def windows(ds):
    """``(x, y)`` numpy of every batch of a finite dataset."""
    pairs = [(x.numpy(), y.numpy()) for x, y in ds]
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def bare_trainer(trainer_class=TimeSeriesForecastingTrainer):
    """A trainer with no config, for the pure hooks (``object.__new__``: the constructor needs a config)."""
    trainer = object.__new__(trainer_class)
    trainer.config = None
    return trainer


# ---------------------------------------------------------------------
# The hook table
# ---------------------------------------------------------------------

#: The optional hooks the task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",          # the train stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",       # a resumed fit starts a fresh iterator: another stream, not a replay
    "bar_is_met",          # an error: strictly below, in float32 (the default is >=)
    "compile_options",     # jit_compile is False unless the field asks
    "run_analysis",        # data-free (the output is a forecast, not class probabilities)
    "summary_fields",      # the window, the stream rule, the bank
    "recipe_notices",      # what the recipe doubts about a family
    "used_arguments",      # the model keywords the run was built with
    "check_model_args",    # the refusals before the claim
    "epoch_line_metrics",  # the epoch line prints loss and mae
    "unjudged_note",       # why init_scale_warning is null
    "task_notes",          # the lines of the summary's notes
)


def test_the_class_names_its_task_and_its_metric():
    assert TimeSeriesForecastingTrainer.task == TASK
    assert TimeSeriesForecastingTrainer.success_metric == "val_mae"
    assert issubclass(TimeSeriesForecastingTrainer, TaskTrainer)
    assert TimeSeriesForecastingTrainer.resume_order, "a resume trains on another stream and the summary says so"


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_every_required_hook_is_overridden(hook):
    assert getattr(TimeSeriesForecastingTrainer, hook) is not getattr(TaskTrainer, hook), hook


@pytest.mark.parametrize("hook", OPTIONAL_HOOKS)
def test_every_optional_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(TimeSeriesForecastingTrainer, hook) is not getattr(TaskTrainer, hook), hook


@pytest.mark.parametrize("module, name", [("image_denoising", "ImageDenoisingTrainer"), ("pretrain", "PretrainTrainer"),
                                           ("time_series_forecasting", "TimeSeriesForecastingTrainer")])
def test_the_tasks_without_classes_refuse_a_class_count_from_either_side_with_one_sentence(module, name):
    """The three tasks without classes share ``trainer.check_holds_no_classes``; each override calls it (nothing tested the refusal before)."""
    from train.api.data_args import DataFacts
    from train.api.trainer import check_holds_no_classes

    cls = getattr(importlib.import_module(f"train.api.tasks.{module}"), name)
    trainer = object.__new__(cls)
    none = DataFacts(dataset="synthetic", num_classes=None, image_size=None)
    cls.input_check(trainer, none, SimpleNamespace(num_classes=None))
    for facts, data, said, loaded in ((DataFacts(dataset="synthetic", num_classes=10, image_size=None), SimpleNamespace(num_classes=None), 10, None),
                                      (none, SimpleNamespace(num_classes=3), None, 3)):
        with pytest.raises(RuntimeError, match=rf"{name} holds no classes, but describe_data said {said} and build_data loaded {loaded}"):
            cls.input_check(trainer, facts, data)
        with pytest.raises(RuntimeError, match="holds no classes"):
            check_holds_no_classes(trainer, facts, data)


def test_the_task_attaches_no_per_epoch_figure_and_writes_the_record_of_its_table(world, tmp_path):
    from train.api.predictor import RECORD_KEYS

    trainer, resolved, context = prepared(tmp_path / "run")
    assert trainer.figure_callbacks(resolved, None, tmp_path) == []
    data = trainer.build_data(resolved, context)
    assert list(trainer.task_record(resolved, context, data)) == list(RECORD_KEYS[TASK])
    assert trainer.config_stamps(resolved, registry.MODELS["nbeats"]) == {}


def test_the_task_does_not_name_a_stage():
    own = {name for name in vars(TimeSeriesForecastingTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim"}


# ---------------------------------------------------------------------
# The bar: strictly below, in float32, with a tie band
# ---------------------------------------------------------------------


def test_a_value_equal_to_the_threshold_is_a_miss():
    trainer = bare_trainer()
    assert trainer.bar_is_met(None, 0.5, 0.5) is False
    assert trainer.bar_is_met(None, 0.4999, 0.5) is True
    assert trainer.bar_is_met(None, 0.5001, 0.5) is False


def test_a_float64_value_that_rounds_to_the_float32_threshold_is_a_miss():
    trainer = bare_trainer()
    threshold = float(np.float32(0.3))
    reached = threshold - 1e-10          # below the threshold in float64, equal to it in float32
    assert reached < threshold
    assert np.float32(reached) == np.float32(threshold)
    assert trainer.bar_is_met(None, reached, threshold) is False


def test_the_tie_band_is_a_few_float32_steps_wide_and_not_wider():
    trainer = bare_trainer()
    threshold = np.float32(0.8)
    step = np.spacing(threshold)
    for ulps in range(0, fr.TIE_ULPS + 1):
        below = float(threshold - np.float32(ulps) * step)
        assert trainer.bar_is_met(None, below, float(threshold)) is False, f"{ulps} float32 steps below is inside the band"
    assert trainer.bar_is_met(None, float(threshold - np.float32(fr.TIE_ULPS + 1) * step), float(threshold)) is True
    assert fr.TIE_ULPS == 16, "the same width as the segmentation task's, measured here at 1 step (D-008)"


@pytest.mark.parametrize("batches, steps", [(1, 16), (4, 16), (16, 16), (17, 32), (64, 32), (256, 64), (2048, 192)])
def test_the_tie_band_grows_with_the_square_root_of_the_select_batches_above_sixteen(batches, steps):
    """D-043: ``TIE_ULPS * max(1, ceil(sqrt(n_batches / 16)))``: 16 float32 steps up to 16 batches, 192 at 2,048 (the measured +20
    steps of the float32 accumulation at 262,144 windows is inside it, the 16 of a 2,176-window run is not widened)."""
    assert fr.tie_band(batches) == steps


@pytest.mark.parametrize("batches", [16, 2048])
def test_a_value_inside_the_band_of_its_own_batch_count_is_a_miss_and_one_step_outside_is_a_hit(batches):
    threshold = np.float32(0.8)
    step = np.spacing(threshold)
    band = fr.tie_band(batches)
    assert fr.beats_bar(float(threshold - np.float32(band) * step), float(threshold), batches) is False
    assert fr.beats_bar(float(threshold - np.float32(band + 1) * step), float(threshold), batches) is True
    assert fr.beats_bar(float(threshold - np.float32(17) * step), float(threshold), batches) is (band < 17)


def test_the_trainer_hands_the_bar_the_select_batch_count_of_its_data(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    assert trainer._select_batches == data.n_select // BATCH == 2
    threshold = np.float32(0.8)
    step = np.spacing(threshold)
    trainer._select_batches = 2048
    assert trainer.bar_is_met(None, float(threshold - np.float32(190) * step), float(threshold)) is False
    assert trainer.bar_is_met(None, float(threshold - np.float32(193) * step), float(threshold)) is True
    trainer._select_batches = 2
    assert trainer.bar_is_met(None, float(threshold - np.float32(190) * step), float(threshold)) is True


def test_the_tie_ulps_is_the_one_definition_of_the_segmentation_module():
    from train.common import segmentation_run

    assert fr.TIE_ULPS is segmentation_run.TIE_ULPS
    source = Path(fr.__file__).read_text()
    assert "TIE_ULPS = " not in source, "forecast_run imports the constant, it does not define it again"


def test_the_value_is_the_minimum_of_val_mae_and_not_the_maximum_or_the_last():
    trainer = bare_trainer()
    assert trainer.success_value({"mae": [9.0, 8.0, 7.0], "val_mae": [3.0, 1.0, 2.0]}) == 1.0
    assert trainer.success_value({"val_mae": [0.5, 1.0, 2.0]}) == 0.5
    assert trainer.success_value({"val_mae": [2.0, 1.0, 0.25]}) == 0.25


def test_the_value_ignores_the_epochs_before_the_selection_start():
    class Windowed(TimeSeriesForecastingTrainer):
        def selection_start_epoch(self, config):
            return 2

    trainer = bare_trainer(Windowed)
    # The best value of the whole run (0.1) is at epoch 0, before the window: it was never saved, so it cannot count.
    assert trainer.success_value({"val_mae": [0.1, 0.2, 0.9, 0.8]}) == 0.8
    assert bare_trainer().success_value({"val_mae": [0.1, 0.2, 0.9, 0.8]}) == 0.1


def trainer_with(naive: float, seasonal: float, epoch_zero: float):
    trainer = bare_trainer()
    trainer._baselines = {"naive_last": naive, "seasonal_naive": seasonal, "context_mean": 9.0, "n_windows": 16}
    trainer._epoch_zero = epoch_zero
    return trainer


@pytest.mark.parametrize(
    "terms, expected",
    [((1.0, 2.0, 3.0), 1.0), ((2.0, 1.0, 3.0), 1.0), ((2.0, 3.0, 1.0), 1.0), ((1.0, 1.0, 1.0), 1.0)],
    ids=["naive-last binds", "seasonal-naive binds", "epoch-zero binds", "all three equal"],
)
def test_the_threshold_is_the_smallest_of_the_three_terms(terms, expected):
    assert trainer_with(*terms).success_threshold(None) == expected


def test_a_threshold_asked_before_the_data_or_the_sidecar_is_refused_by_name():
    trainer = bare_trainer()
    trainer._baselines, trainer._epoch_zero = None, 0.8
    with pytest.raises(RuntimeError, match="baselines"):
        trainer.success_threshold(None)
    trainer._baselines, trainer._epoch_zero = {"naive_last": 1.0, "seasonal_naive": 1.0}, None
    with pytest.raises(RuntimeError, match="epoch-zero"):
        trainer.success_threshold(None)


def test_the_epoch_zero_value_is_read_from_the_sidecar(tmp_path):
    path = tmp_path / FIT_STATE_NAME
    update_fit_state(path, initial={"metrics": {"loss": 0.9, "mae": 0.7}, "ratio": 1.0, "warned": None})
    assert fr.epoch_zero_mae(path) == 0.7
    with pytest.raises(ValueError, match="epoch-zero"):
        fr.epoch_zero_mae(tmp_path / "no_such_sidecar.json")
    update_fit_state(path, initial={"metrics": {"loss": 0.9}, "ratio": 1.0, "warned": None})
    with pytest.raises(ValueError, match="initial.metrics.mae"):
        fr.epoch_zero_mae(path)


def test_the_baselines_are_the_ones_of_the_select_windows_and_not_the_test_windows(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    x, y = windows(data.select_ds)
    assert trainer._baselines == fr.baseline_maes(x, y)
    tx, ty = windows(data.test_ds)
    on_test = fr.baseline_maes(tx, ty)
    assert on_test["naive_last"] != trainer._baselines["naive_last"], "the control: the two splits give other numbers"
    assert trainer._baselines["n_windows"] == data.n_select == 16


def find_a_low_seed(tmp_path):
    """The first seed at which a naive-last model's float32 ``mae`` is BELOW its float64 numpy baseline."""
    for seed in range(1, 40):
        trainer, resolved, context = prepared(tmp_path / f"seed{seed}", seed=seed)
        data = trainer.build_data(resolved, context)
        reached = naive_last_score(data)
        if np.float32(reached) < np.float32(trainer._baselines["naive_last"]):
            return trainer, resolved, data, reached
    raise AssertionError("no seed in 1..39 gives a naive-last float32 score below its float64 baseline")


def naive_last_score(data) -> float:
    """The compiled ``mae`` metric of a model that outputs the naive-last forecast, on the select windows."""
    horizon = data.horizon

    class NaiveLast(keras.Model):
        def call(self, inputs):
            return keras.ops.repeat(inputs[:, -1:, :], horizon, axis=1)

    model = NaiveLast()
    decoder = fr.Decoder("ets", lambda out: (out, None), None, None)
    model.compile(loss="mae", metrics=[fr.DecodedMAE(decoder)])
    return float(model.evaluate(data.select_ds, return_dict=True, verbose=0)["mae"])


def test_a_naive_last_model_never_meets_the_bar(world, tmp_path):
    # The instrument's zero: a model that outputs the naive-last forecast IS the baseline, so it must never meet a bar cut
    # from it. The compiled metric adds in float32 and the baseline is float64 numpy, so at some seeds the model scores
    # one float32 step BELOW the baseline: the bare strict comparison passes it (the control below), the tie band does not.
    trainer, resolved, data, reached = find_a_low_seed(tmp_path)
    naive = trainer._baselines["naive_last"]
    assert np.float32(reached) < np.float32(naive), "the control: a bare float32 comparison is fooled"
    assert abs(float(np.float32(reached)) - float(np.float32(naive))) <= 2 * float(np.spacing(np.float32(naive)))
    assert trainer.bar_is_met(resolved, reached, naive) is False
    assert trainer.bar_is_met(resolved, reached, trainer_with(naive, 1e9, 1e9).success_threshold(None)) is False


# ---------------------------------------------------------------------
# The seams
# ---------------------------------------------------------------------


def test_fit_kwargs_gives_the_epoch_and_the_validation_length(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    kwargs = trainer.fit_kwargs(resolved, data)
    assert kwargs == {"steps_per_epoch": STEPS, "validation_steps": int(data.select_ds.cardinality())}
    assert data.steps_per_epoch == STEPS == data.n_train // BATCH
    assert kwargs["validation_steps"] == 16 // BATCH


def test_compile_options_is_jit_off_unless_the_field_asks(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    spec = registry.MODELS["nbeats"]
    assert trainer.compile_options(resolved, spec) == {"jit_compile": False}
    asked = dataclasses.replace(resolved, jit_compile=True)
    assert trainer.compile_options(asked, spec) == {"jit_compile": True}
    for options in (trainer.compile_options(resolved, spec), trainer.compile_options(asked, spec)):
        assert not set(options) & {"optimizer", "loss", "metrics"}


def test_jit_compile_is_refused_for_ets_by_name_before_the_claim(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        TimeSeriesForecastingTrainer(make_config(out, model="ets/ANN", jit_compile=True)).run()
    assert "jit_compile" in str(caught.value) and "'ets'" in str(caught.value) and "XLA" in str(caught.value)
    assert not out.exists()
    asked = dataclasses.replace(make_config(out, model="ets/ANN"), jit_compile=True)
    with pytest.raises(ConfigError, match="jit_compile"):
        TimeSeriesForecastingTrainer(asked).compile_options(asked, registry.MODELS["ets"])


class KillBefore(keras.callbacks.Callback):
    """Raise before batch ``batch`` of epoch ``epoch``: a stand-in for a kill between two batches."""

    def __init__(self, epoch, batch):
        super().__init__()
        self.epoch, self.batch, self.at = epoch, batch, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.at = epoch

    def on_train_batch_begin(self, batch, logs=None):
        if self.at == self.epoch and batch == self.batch:
            raise Killed(f"killed before batch {batch} of epoch {self.epoch}")


class Killed(RuntimeError):
    pass


class Watching(TimeSeriesForecastingTrainer):
    """The real task with ``resume_stream`` recording its arguments, a ``fit`` spy and an optional kill.

    :cvar kill: ``(epoch, batch)`` of a :class:`KillBefore`, or ``None``.
    :ivar calls: ``(completed_steps, data)`` of every ``resume_stream`` call; ``fit_calls`` the ``(x, kwargs)`` of every ``fit``.
    """

    kill = None

    def __init__(self, config):
        super().__init__(config)
        self.calls, self.fit_calls, self.returned = [], [], None

    def build_model(self, config, context, data):
        model = super().build_model(config, context, data)
        real_fit = model.fit

        def spy_fit(*args, **kwargs):
            self.fit_calls.append((args[0], kwargs))
            return real_fit(*args, **kwargs)

        model.fit = spy_fit
        return model

    def extra_callbacks(self, config, data):
        return [] if self.kill is None else [KillBefore(*self.kill)]

    def resume_stream(self, config, data, completed_steps):
        self.calls.append((completed_steps, data))
        self.returned = super().resume_stream(config, data, completed_steps)
        return self.returned


def watching(**attributes):
    return type("Watching", (Watching,), attributes)


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    """A whole run, the same run killed before epoch 1, and ``ran`` the whole run's trainer (its ``fit`` spy)."""
    root = tmp_path_factory.mktemp("forecast_runs")
    whole = root / "whole"
    trainer = watching()(make_config(whole, epochs=3))
    trainer.run()
    killed = root / "killed"
    with pytest.raises(Killed):
        watching(kill=(1, 0))(make_config(killed, epochs=3)).run()
    return {"whole": whole, "killed": killed, "trainer": trainer}


def test_a_fresh_run_never_calls_resume_stream_and_fit_sees_both_lengths(runs):
    trainer = runs["trainer"]
    assert trainer.calls == []
    (x, kwargs), = trainer.fit_calls
    assert kwargs["steps_per_epoch"] == STEPS and kwargs["validation_steps"] == 16 // BATCH
    assert kwargs["epochs"] == 3


def test_a_reporting_only_resume_never_calls_the_hook(runs, tmp_path):
    run_dir = tmp_path / "crashed_after_fit"
    shutil.copytree(runs["whole"], run_dir)
    (run_dir / "results_summary.json").unlink()
    trainer = watching()(make_config(run_dir, epochs=3, resume_interrupted=True))
    trainer.run()
    assert (run_dir / "results_summary.json").is_file(), "vacuity guard: the run was resumed and reported"
    assert trainer.calls == [] and trainer.fit_calls == []


def test_a_resume_that_trains_gets_the_settled_count_and_trains_on_another_stream(runs, tmp_path):
    run_dir = tmp_path / "resumed"
    shutil.copytree(runs["killed"], run_dir)
    trainer = watching()(make_config(run_dir, epochs=3, resume_interrupted=True))
    result = trainer.run()
    (completed, before), = trainer.calls
    assert completed == STEPS, "one epoch was finished: the count is the settled copy's"
    after = trainer.returned
    assert after is not None, "resume_stream returns the replacement batches, not None (a None replays the first epoch's stream)"
    assert after is not before and after.train_ds is not before.train_ds
    assert after.stream_seed == derive_stream_seed(before.seed, STEPS) != before.stream_seed
    assert after.first_batch_digest != before.first_batch_digest, "a resume that replays epoch 0 starts on its first batch"
    assert (after.seed, after.fingerprint, after.n_select, after.n_test, after.steps_per_epoch) == (
        before.seed, before.fingerprint, before.n_select, before.n_test, before.steps_per_epoch)
    (x, kwargs), = trainer.fit_calls
    assert x is after.train_ds, "fit trained on the replacement stream"
    assert kwargs["steps_per_epoch"] == STEPS
    assert result.summary_path.is_file()
    assert any(TimeSeriesForecastingTrainer.resume_order in note for note in read_json(result.summary_path)["notes"]), (
        "the summary's resume note says what the resume did to the stream")


# ---------------------------------------------------------------------
# A whole run: the epoch-zero term, the verdict, the resume reads one epoch-zero value
# ---------------------------------------------------------------------


def test_the_summary_holds_the_metric_the_baselines_the_epoch_zero_value_the_threshold_and_the_verdict(runs):
    summary = read_json(runs["whole"] / "results_summary.json")
    sidecar = read_json(runs["whole"] / "checkpoints" / FIT_STATE_NAME)
    epoch_zero = sidecar["initial"]["metrics"]["mae"]
    assert summary["success_metric"] == "val_mae" and summary["monitor"] == "val_mae" and summary["monitor_mode"] == "min"
    assert summary["epoch_zero"]["val_mae"] == epoch_zero
    baselines = summary["baselines"]
    assert {"naive_last", "seasonal_naive", "context_mean", "n_windows"} <= set(baselines)
    assert summary["success_threshold"] == min(baselines["naive_last"], baselines["seasonal_naive"], epoch_zero)
    column = [float(v) for v in read_json(runs["whole"] / "training_history.json")["val_mae"]]
    assert len(column) == 3 and summary["success_value"] == pytest.approx(min(column), rel=1e-6)
    threshold = np.float32(summary["success_threshold"])
    cleared = np.float32(summary["success_value"]) < threshold - np.float32(fr.TIE_ULPS) * np.spacing(threshold)
    assert summary["success"] is bool(cleared)
    assert summary["steps_per_epoch"] == STEPS and summary["n_val"] == 16 and summary["n_test"] == 16


def test_the_reload_gap_of_the_best_checkpoint_is_exactly_zero(runs):
    summary = read_json(runs["whole"] / "results_summary.json")
    assert summary["best_checkpoint_max_abs_diff"] == 0.0 and summary["best_checkpoint_load_error"] is None
    assert summary["model_loading_validated"] is True


def test_a_resume_reads_the_epoch_zero_value_the_fresh_run_wrote(runs, tmp_path):
    run_dir = tmp_path / "reported_again"
    shutil.copytree(runs["whole"], run_dir)
    first = read_json(run_dir / "results_summary.json")
    (run_dir / "results_summary.json").unlink()
    watching()(make_config(run_dir, epochs=3, resume_interrupted=True)).run()
    again = read_json(run_dir / "results_summary.json")
    assert again["reporting_resumed"] is True
    # The resumed process never evaluated the untrained model: the sidecar the FIRST process wrote is the only place the
    # number is, and the threshold of the second report is cut from it.
    assert again["epoch_zero"] == first["epoch_zero"]
    assert again["success_threshold"] == first["success_threshold"]
    assert again["baselines"] == first["baselines"]


# ---------------------------------------------------------------------
# Refusals before the claim
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"model": "deepar/base"}, "teacher-forced"),
        ({"model": "adaptive_ema/base"}, "no horizon axis"),
        ({"model_args": {"use_normalization": True}}, "model_args.use_normalization"),
        ({"model_args": {"kernel_regularizer": "l2"}}, "model_args.kernel_regularizer"),
        ({"model_args": {"scale": "tiny"}}, "model_args.scale"),
        ({"optim_args": {"optimizer": "sgd", "lr": 0.01, "weight_decay": 0.01}}, "optim_args.weight_decay"),
        ({"model": "prism/tiny", "model_args": {"quantile_levels": [0.1, 0.5, 0.9]}, "loss_args": {"name": "mae"}},
         "loss_args.name"),
        ({"model": "tirex/tiny", "loss_args": {"name": "mse"}}, "loss_args.name"),
        ({"model": "timestral/5M", "data": {"lookback": 64, "horizon": 40}}, "data.horizon"),
        ({"metric_args": {"watch_metric": "loss"}}, "metric_args.watch_metric"),
        ({"metric_args": {"watch_mode": "max"}}, "metric_args.watch_mode"),
        ({"data": {**DATA, "lookback": 400, "horizon": 400}}, "data"),
    ],
    ids=["deepar", "adaptive_ema", "double-normalization", "regularizer", "unknown-key", "sgd-decay", "quantile-with-loss",
         "tirex-with-loss", "timestral-horizon", "watch-the-loss", "watch-mode-max", "segment-too-short"],
)
def test_a_refused_request_leaves_out_absent(world, tmp_path, changes, fragment):
    out = tmp_path / "run"
    if "data" in changes:
        changes = {**changes, "data": {**DATA, **changes["data"]}}
    with pytest.raises((ConfigError, ConfigValidationError)) as caught:
        TimeSeriesForecastingTrainer(make_config(out, **changes)).run()
    assert fragment in str(caught.value)
    assert not out.exists(), "the refusal came after the claim"


@pytest.mark.parametrize("model", ["tirex/tiny", "ets/ANN"])
def test_a_second_channel_is_refused_before_the_claim_for_a_one_channel_family(world, tmp_path, model):
    out, csv = tmp_path / "run", write_series_csv(tmp_path / "two.csv", rows=900, columns=2, time_col=None)
    changes = {"model": model, "data": {"format": "csv", "path": str(csv), "lookback": 32, "horizon": 8}}
    with pytest.raises(ConfigError) as caught:
        TimeSeriesForecastingTrainer(make_config(out, **changes)).run()
    assert "data.value_cols" in str(caught.value) and "channel" in str(caught.value)
    assert not out.exists()


# ---------------------------------------------------------------------
# The conformal sentence
# ---------------------------------------------------------------------

THE_IMAGE_SENTENCE = (
    "a conformal prediction set is a set of classes for a whole image, which a {kind} run does not produce; leave it null, got 0.1"
)
THE_FORECAST_SENTENCE = (
    "a conformal prediction set is a set of classes, which a time_series_forecasting run does not produce; the interval "
    "forecast of this task is a quantile head (model_args.quantile_levels); leave it null, got 0.1"
)


def test_a_conformal_level_is_refused_by_key_with_the_forecasting_sentence():
    with pytest.raises(ConfigValidationError) as caught:
        make_config(Path("/nonexistent/run"), **{"analysis_args.conformal_alpha": 0.1})
    text = str(caught.value)
    assert "analysis_args.conformal_alpha" in text and THE_FORECAST_SENTENCE in text


def test_the_older_sentence_is_byte_for_byte_what_it_was():
    config = make_config(Path("/nonexistent/run"))
    config = dataclasses.replace(config, analysis_args=dataclasses.replace(config.analysis_args, conformal_alpha=0.1))
    (line,) = config_problems(config, "", no_conformal_for="denoising")
    assert line == "Invalid value for 'analysis_args.conformal_alpha': " + THE_IMAGE_SENTENCE.format(kind="denoising")
    (line,) = config_problems(config, "", no_conformal_for="segmentation", because=None)
    assert line == "Invalid value for 'analysis_args.conformal_alpha': " + THE_IMAGE_SENTENCE.format(kind="segmentation")
    (line,) = config_problems(config, "", no_conformal_for="x", because="a reason of its own")
    assert line == "Invalid value for 'analysis_args.conformal_alpha': a reason of its own; leave it null, got 0.1"
    assert config_problems(make_config(Path("/nonexistent/run")), "", no_conformal_for="denoising") == []


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------


def variant_names(path: str):
    module, _, attr = path.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return sorted(obj)


def test_the_task_has_one_row_and_the_seven_families_train_it_alone(world):
    assert [spec.name for spec in BUILTIN_TASKS].count(TASK) == 1
    assert TASK in registry.TASKS
    spec = registry.TASKS[TASK]
    assert spec.config_class == "train.api.tasks.time_series_forecasting:TimeSeriesForecastingConfig"
    assert spec.function == "train.api.tasks.time_series_forecasting:train_time_series_forecasting"
    assert sorted(f for f, row in registry.MODELS.items() if TASK in row.tasks) == sorted(fm.FAMILIES)
    for family in fm.FAMILIES:
        assert registry.MODELS[family].tasks == (TASK,), family
    for other in ("vit/vit_pico", "convunext/tiny"):
        with pytest.raises(ConfigError):
            registry.parse_model_name(other, task=TASK)
    with pytest.raises(ConfigError):
        registry.parse_model_name("nbeats/base", task="image_classification")


def test_list_models_names_the_variants_the_tables_hold(world):
    expected = sorted(f"{family}/{name}" for family, row in fm.FAMILIES.items() for name in variant_names(row.variants))
    assert registry.list_models(task=TASK) == expected
    assert not [name for name in registry.list_models(task="image_classification") if name.split("/")[0] in fm.FAMILIES]


def test_the_lazy_names_of_the_package_resolve_to_the_task_module():
    import train.api as api

    assert api.train_time_series_forecasting is importlib.import_module(
        "train.api.tasks.time_series_forecasting").train_time_series_forecasting
    assert api.TimeSeriesForecastingConfig is TimeSeriesForecastingConfig
    assert api.TimeSeriesData is importlib.import_module("train.api.forecast_args").TimeSeriesData
    assert api.load_forecast_data_config is importlib.import_module("train.api.forecast_args").load_forecast_data_config
    assert {"train_time_series_forecasting", "TimeSeriesForecastingConfig", "TimeSeriesData",
            "load_forecast_data_config"} <= set(api.__all__)


# ---------------------------------------------------------------------
# What the task added to shared code, and what stays what it was
# ---------------------------------------------------------------------


def test_the_load_check_predicts_at_the_batch_it_is_given_and_at_the_cap_when_it_is_not(tmp_path):
    # A model built at a STATIC batch (timestral) can predict only at that batch, which `predict_batch_cap` (Keras' 32 unless
    # a smaller one is asked for) cannot say; the keyword is the one door, and `None` (every older caller) is the cap.
    from types import SimpleNamespace

    from train.common import classification_run

    seen = []

    def run(**given):
        model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(2)])
        real = model.predict

        def spy(x, batch_size=None, **kwargs):
            seen.append(batch_size)
            return real(x, batch_size=batch_size, **kwargs)

        model.predict = spy
        history = keras.callbacks.History()
        history.history = {"loss": [1.0]}
        run_dir = tmp_path / f"run{len(seen)}"
        run_dir.mkdir()
        classification_run.finalize_last_epoch(
            model, SimpleNamespace(weights=model.get_weights()), SimpleNamespace(test_ds=None), run_dir, history, 1,
            evaluate=classification_run.evaluate_dict, atomic=True, exact_reload=True,
            sample=np.zeros((5, 4), "float32"), **given)

    run()
    run(eval_batch_size=3)
    run(eval_batch_size=3, predict_batch_size=64)
    run(predict_batch_size=5)
    assert seen[:2] == [None, 3], "the older callers' arguments give the cap they always gave"
    assert seen[2] == 64 and seen[3] == 5, "a given predict batch wins over the cap"
    from train.common import run_summary

    for function in (classification_run.finalize_last_epoch, run_summary.report_data_free_run):
        assert inspect.signature(function).parameters["predict_batch_size"].default is None


def test_the_quantile_losses_flat_target_follows_the_family_and_its_head(world):
    # tirex and a quantile xlstm train on (batch, horizon); every other head on the window layout.
    assert fm.flat_target("tirex", {}) is True
    assert fm.flat_target("xlstm", {"quantile_levels": [0.1, 0.5, 0.9]}) is True
    assert fm.flat_target("xlstm", {}) is False
    for family in ("nbeats", "prism", "ets", "timestral", "mdn"):
        assert fm.flat_target(family, {"quantile_levels": [0.1, 0.5, 0.9]}) is False, family


def test_the_loss_name_refusal_before_the_claim_and_the_one_at_build_time_hold_the_same_heads():
    rows = {
        "nbeats": ("nbeats", {}), "prism_point": ("prism", {}), "prism_quantile": ("prism", {"quantile_levels": [0.1, 0.9]}),
        "tirex": ("tirex", {}), "xlstm_point": ("xlstm", {}), "xlstm_quantile": ("xlstm", {"quantile_levels": [0.1, 0.9]}),
        "ets": ("ets", {}), "timestral": ("timestral", {}), "mdn": ("mdn", {}),
    }
    assert set(rows) == set(fr.DECODE)
    for row, (family, model_args) in rows.items():
        accepts_a_name = fm.loss_problem(family, model_args, "mae") is None
        assert accepts_a_name == (fr.DECODE[row].objective == "point"), row
        assert fm.loss_problem(family, model_args, None) is None, row


def test_a_quantile_xlstm_is_refused_for_a_second_channel_before_the_claim(world, tmp_path):
    out, csv = tmp_path / "run", write_series_csv(tmp_path / "two.csv", rows=900, columns=2, time_col=None)
    data = {"format": "csv", "path": str(csv), "lookback": 32, "horizon": 8}
    with pytest.raises(ConfigError) as caught:
        TimeSeriesForecastingTrainer(make_config(
            out, model="xlstm/tiny", data=data, model_args={"quantile_levels": [0.1, 0.5, 0.9]})).run()
    assert "data.value_cols" in str(caught.value) and "no channel axis" in str(caught.value)
    assert not out.exists()
    # The control: the point head takes two channels, so the refusal is about the head and not the family.
    trainer, resolved, context = prepared(tmp_path / "ok", model="xlstm/tiny", data=data)
    trainer.check_model_args(resolved, context)


def test_build_data_gives_a_flat_target_to_tirex_and_the_window_layout_to_nbeats_with_one_set_of_baselines(world, tmp_path):
    flat, plain = {}, {}
    for model, store in (("tirex/tiny", flat), ("nbeats/tiny", plain)):
        trainer, resolved, context = prepared(tmp_path / model.replace("/", "_"), model=model)
        data = trainer.build_data(resolved, context)
        store.update(
            data=data, baselines=trainer._baselines,
            targets={name: next(iter(getattr(data, name)))[1].shape for name in ("train_ds", "select_ds", "test_ds")})
    assert {name: shape for name, shape in flat["targets"].items()} == {n: (BATCH, HORIZON) for n in flat["targets"]}
    assert {name: shape for name, shape in plain["targets"].items()} == {n: (BATCH, HORIZON, 1) for n in plain["targets"]}
    # The baselines are the window layout's whichever the loss wants: one bar for every family.
    assert flat["baselines"] == plain["baselines"]
    assert flat["data"].fingerprint == plain["data"].fingerprint
    assert int(flat["data"].select_ds.cardinality()) == int(plain["data"].select_ds.cardinality()) == 2


# ---------------------------------------------------------------------
# Two families whose reporting is not the common one: a static batch and one parameter
# ---------------------------------------------------------------------

WIDE = {**DATA, "lookback": 64, "horizon": 16, "n_select": 64, "n_test": 64, "steps_per_epoch": 4}


def test_a_static_batch_family_trains_reloads_and_reports_at_its_batch(world, tmp_path):
    # timestral is built at the run's batch and predicts at no other. Keras predicts at 32 unless told otherwise, so a batch
    # ABOVE 32 (here 40, which divides neither 32 nor the windows) reaches the reload checks as pieces of the wrong size unless
    # they are told the batch: the checks of the best checkpoint and of the last epoch both read the full select batch at it.
    result = TimeSeriesForecastingTrainer(make_config(
        tmp_path / "timestral", model="timestral/5M", data=WIDE, batch_size=40, epochs=1)).run()
    summary = read_json(result.summary_path)
    assert summary["best_checkpoint_max_abs_diff"] == 0.0 and summary["best_checkpoint_load_error"] is None
    assert summary["model_loading_validated"] is True and summary["model_loading_check"]["ok"] is True
    assert summary["canonical_metrics"]["test"]["MAE"] == pytest.approx(summary["test_metrics_best"]["mae"], rel=1e-4)
    assert summary["batch_size"] == 40 and summary["n_val"] == 40 and summary["n_test"] == 40


def test_an_ets_run_reports_its_analysis_as_skipped_and_still_judges(world, tmp_path):
    # ets holds one parameter: the data-free analyzer has nothing to read and says `missing`; the summary says `skipped`
    # (D-025) and the run is judged all the same, with an epoch-zero term that can bind (an untrained ets can beat naive-last).
    result = TimeSeriesForecastingTrainer(make_config(
        tmp_path / "ets", model="ets/ANN", epochs=2, **{"analysis_args.model_analysis": True})).run()
    summary = read_json(result.summary_path)
    assert summary["analyzer"]["status"] == "skipped", summary["analyzer"]
    assert isinstance(summary["success"], bool) and summary["success_value"] == summary["best_val_metrics"]["val_mae"]
    assert summary["epoch_zero"]["binds"] is bool(
        summary["epoch_zero"]["val_mae"] < min(summary["baselines"]["naive_last"], summary["baselines"]["seasonal_naive"]))


# ---------------------------------------------------------------------
# The epoch-0 check refuses only a non-finite value, and the notices
# ---------------------------------------------------------------------


def test_the_epoch_zero_check_refuses_only_a_non_finite_value_and_reads_no_ratio(world, tmp_path, monkeypatch):
    from train.common import classification_run

    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    model = trainer.build_model(resolved, context, data)
    loss, metrics = trainer.build_loss_and_metrics(resolved, data)
    model.compile(optimizer="adam", loss=loss, metrics=metrics, jit_compile=False)
    baseline, ratio, warned = trainer.initial_loss_check(resolved, model, data)
    assert warned is None, "no verdict on the initialization is given (D-029)"
    assert ratio == pytest.approx(baseline["mae"] / trainer._baselines["naive_last"], rel=1e-12)
    # A huge ratio is a number, not a refusal: an untrained nbeats sits at 9 times naive-last (measured).
    monkeypatch.setattr(classification_run, "evaluate_dict", lambda m, ds: {"loss": 1e9, "mae": 1e9})
    assert trainer.initial_loss_check(resolved, model, data)[1] > 1e6
    for broken in ({"loss": float("nan"), "mae": 1.0}, {"loss": 1.0, "mae": float("inf")}, {"loss": 1.0}):
        monkeypatch.setattr(classification_run, "evaluate_dict", lambda m, ds, broken=broken: broken)
        with pytest.raises(RuntimeError, match="diverges before training"):
            trainer.initial_loss_check(resolved, model, data)


def test_only_nbeats_and_ets_carry_a_recipe_notice_and_each_says_why():
    trainer = bare_trainer()
    notices = {family: trainer.recipe_notices(None, family) for family in fm.FAMILIES}
    assert {family for family, lines in notices.items() if lines} == {"nbeats", "ets"}
    (level, text), = notices["nbeats"]
    assert level == "info" and "no gradient" in text and "D-007" in text
    (level, text), = notices["ets"]
    assert level == "info" and "untrained" in text and "D-029" in text


# ---------------------------------------------------------------------
# Second part: a stubbed end to end per family, the run directory, csv, the bar that can fail, the refusals
# ---------------------------------------------------------------------
#
# Every family trains two epochs of ``E2E_STEPS`` steps on the CPU at tiny sizes, once through ``train_time_series_forecasting``
# and once through ``cli.main`` (the same request, the run renamed between the two). What each run is held to is derived from
# the code: the listing is what the sequence, the callbacks and the reporting write (``run_sequence``, ``fit_callbacks``,
# ``TimeSeriesForecastingTrainer.evaluate_and_report`` and ``task_record``), the summary keys are the ones
# ``evaluate_and_report`` puts in ``Evaluation.extras`` and ``summary_fields`` names.

E2E_EPOCHS, E2E_BATCH, E2E_STEPS = 2, 16, 20
E2E_DATA = {**DATA, "n_select": 32, "n_test": 32, "steps_per_epoch": E2E_STEPS}
QUANTILES = [0.1, 0.5, 0.9]
#: One row per decode row of ``forecast_run.DECODE`` (a held test): the model and its ``model_args``.
CASES: Dict[str, Any] = {
    "nbeats": ("nbeats/tiny", {}),
    "prism_point": ("prism/tiny", {}),
    "prism_quantile": ("prism/tiny", {"quantile_levels": QUANTILES}),
    "tirex": ("tirex/tiny", {}),
    "xlstm_point": ("xlstm/tiny", {}),
    "xlstm_quantile": ("xlstm/tiny", {"quantile_levels": QUANTILES}),
    "ets": ("ets/ANN", {}),
    "timestral": ("timestral/5M", {}),
    "mdn": ("mdn/tiny", {}),
}
#: A fit that runs more batches than ``EPOCHS * STEPS`` has an epoch with no length (an infinite stream, no steps_per_epoch).
RUNAWAY_BATCHES = 4 * E2E_EPOCHS * E2E_STEPS


class Spy(keras.callbacks.Callback):
    """Counts the batches of each epoch of one ``fit`` and stops a fit whose epoch never ends (reported, not hung)."""

    def __init__(self):
        super().__init__()
        self.last_batch, self.total, self.epoch = {}, 0, 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        self.total += 1
        self.last_batch[self.epoch] = batch
        if self.total > RUNAWAY_BATCHES:
            self.model.stop_training = True   # the epoch ends here and the run goes on: the test reports it


def flatten_request(asked: Dict[str, Any]) -> List[str]:
    """``key=value`` tokens as the command line takes them: groups are dotted, ``model_args`` and lists are JSON."""
    def text(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else str(value)

    tokens = []
    for key, value in asked.items():
        if isinstance(value, dict) and key != "model_args":
            tokens += [f"{key}.{inner}={text(v)}" for inner, v in value.items()]
        elif key != "model_args" or value:
            tokens.append(f"{key}={text(value)}")
    return tokens


def spying_fit(fits: List[Dict[str, Any]]):
    """A replacement of ``keras.Model.fit`` that records its keywords and adds a :class:`Spy` to the callbacks."""
    real_fit = keras.Model.fit

    def fit(self, *args, **keywords):
        spy = Spy()
        fits.append({"keywords": {k: keywords.get(k) for k in ("steps_per_epoch", "validation_steps")}, "spy": spy})
        return real_fit(self, *args, **{**keywords, "callbacks": [*keywords.get("callbacks", []), spy]})

    return fit


def twice(root: Path, asked: Dict[str, Any], fits: List[Dict[str, Any]]):
    """Run ``asked`` through the function and then through the command line; the run is renamed after each.

    :return: ``(result, code)`` of the function and of ``cli.main``; the runs are ``root/function`` and ``root/cli``.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(keras.Model, "fit", spying_fit(fits))
        patch.chdir(root)
        result = train_time_series_forecasting(**asked)
        (root / "run").rename(root / "function")
        code = cli.main([TASK, *flatten_request(asked)])
        (root / "run").rename(root / "cli")
    return result, code


@pytest.fixture(scope="module", params=list(CASES))
def ended(request, world, tmp_path_factory):
    """One 2-epoch run per family through the function and the same request through ``cli.main``, in this process."""
    case = request.param
    model, model_args = CASES[case]
    root = tmp_path_factory.mktemp("end_to_end_" + case)
    asked = {
        "out": str(root / "run"), "model": model, "data": dict(E2E_DATA), "epochs": E2E_EPOCHS, "batch_size": E2E_BATCH,
        "model_args": model_args, "analysis_args": {"model_analysis": True, "figures": True},
    }
    fits: List[Dict[str, Any]] = []
    result, code = twice(root, asked, fits)
    return SimpleNamespace(case=case, root=root, asked=asked, result=result, code=code, fits=fits,
                           function=root / "function", cli=root / "cli")


def test_the_cases_are_the_decode_rows_and_the_registered_families():
    assert set(CASES) == set(fr.DECODE), "a decode row without an end to end run, or the other way round"
    assert {model.partition("/")[0] for model, _ in CASES.values()} == set(fm.FAMILIES)


#: The files of a finished run, from the code: the sequence writes the config pair, the log and the lock, the callbacks the
#: csv and the two checkpoints, the reporting the history and the summary, and ``task.json`` (``task_record``).
RUN_FILES = {
    "best_model.keras", "config.json", "config_requested.json", "final_model.keras", "results_summary.json",
    "run.lock", "run.log", "task.json", "training_history.json", "training_log.csv",
}
RUN_DIRS = {"checkpoints", "model_analysis", "tensorboard", "visualizations"}
#: The figures of every run: the dashboard of the callbacks and the forecast figure of ``evaluate_and_report``.
FIGURES = {"forecast_test.png", "training_dashboard.png"}
#: ``model_analysis/`` holds what the data-free analysis drew. The json and the weight figure are always there; the spectral
#: figures need a layer with at least ``spectral_min_evals`` = 10 eigenvalues, which the tiny prism quantile head (a 3-level
#: head on a horizon of 8) has not (status ``partial``, the failed analysis named in the summary), and the one-parameter ets
#: has nothing to analyse at all (``missing`` in the analyzer, ``skipped`` in the summary, D-025). Measured by running every
#: family at these sizes (function and command line agree), pinned here so a silent loss of a figure is a change of the contract.
ANALYSIS_WEIGHTS = {"analysis_results.json", "weight_learning_journey.png"}
ANALYSIS_SPECTRAL = {"spectral_funnel_diagram.png", "spectral_summary.png"}
ANALYSIS = {
    case: ("ok", ANALYSIS_WEIGHTS | ANALYSIS_SPECTRAL) for case in CASES if case not in ("ets", "prism_quantile")
}
ANALYSIS["prism_quantile"] = ("partial", ANALYSIS_WEIGHTS)
ANALYSIS["ets"] = ("skipped", ANALYSIS_WEIGHTS)


def names(path: Path) -> set:
    return {child.name for child in Path(path).iterdir()}


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_run_directory_holds_the_listing_the_code_writes_task_json_included(ended, which):
    run_dir = getattr(ended, which)
    assert names(run_dir) == RUN_FILES | RUN_DIRS, "an added or a missing file is a change of the contract"
    assert names(run_dir / "checkpoints") == {"backup", "fit_state.json"}
    assert names(run_dir / "tensorboard") == {"train", "validation"}
    assert names(run_dir / "visualizations") == FIGURES
    summary = read_json(run_dir / "results_summary.json")
    status, files = ANALYSIS[ended.case]
    assert summary["analyzer"]["status"] == status
    assert names(run_dir / "model_analysis") == files
    assert ("failed_analyses" in summary["analyzer"]) is (status != "ok"), "an analysis that did not complete names what failed"
    assert summary["visualizations"]["files"] == sorted(FIGURES)


def test_the_function_and_the_cli_write_the_same_config_bytes_and_the_cli_exits_by_the_verdict(ended):
    for name in ("config.json", "config_requested.json"):
        assert (ended.function / name).read_bytes() == (ended.cli / name).read_bytes(), name
    resolved = read_json(ended.cli / "config.json")
    model, model_args = CASES[ended.case]
    assert resolved["model"] == model and resolved["epochs"] == E2E_EPOCHS and resolved["batch_size"] == E2E_BATCH
    assert resolved["data"]["lookback"] == LOOKBACK and resolved["data"]["horizon"] == HORIZON
    assert resolved["jit_compile"] is False
    success = read_json(ended.cli / "results_summary.json")["success"]
    assert ended.code == (0 if success else 1) and ended.result.success is read_json(ended.function / "results_summary.json")["success"]


#: What ``canonical_metrics`` returns for a point head; a quantile head adds the interval block.
CANONICAL_POINT = {"MAE", "RMSE", "sMAPE", "rMAE", "MASE", "MASE_excluded"}
CANONICAL_QUANTILE = {"coverage", "sharpness", "coverage_nominal"}


@pytest.fixture(scope="module")
def reference_baselines(world, tmp_path_factory):
    """The numpy baselines of the select windows of ``E2E_DATA``, worked out here from the data module alone."""
    root = tmp_path_factory.mktemp("reference_baselines")
    config = make_config(root / "run", data=dict(E2E_DATA), batch_size=E2E_BATCH)
    batches = forecast_data.load_forecast_batches(
        format="builtin", dataset="synthetic", path=None, value_cols=None, time_col=None, lookback=LOOKBACK, horizon=HORIZON,
        realizations=1, series_length=600, noise_level=0.1, patterns=PATTERNS, select_fraction=0.15, test_fraction=0.15,
        n_select=32, n_test=32, steps_per_epoch=E2E_STEPS, max_samples=None, batch_size=E2E_BATCH, seed=config.seed,
        stream_seed=None, scale_floor=config.transform_args.scale_floor)
    return fr.baseline_maes(*windows(batches.select_ds))


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_summary_holds_the_metric_the_baselines_the_epoch_zero_value_the_threshold_the_verdict_and_the_post_hoc_block(
        ended, which, reference_baselines):
    run_dir = getattr(ended, which)
    summary = read_json(run_dir / "results_summary.json")
    # every block this test reads is in the summary: a summary that drops one is a failed claim, named here
    # rather than a KeyError further down
    wanted = {"status", "success_metric", "monitor", "monitor_mode", "task", "baselines", "epoch_zero", "success_threshold",
              "success_value", "success", "canonical_metrics", "test_metrics_best", "best_val_metrics", "analyzer", "flat_windows"}
    assert wanted <= set(summary), f"the summary lacks {sorted(wanted - set(summary))}"
    # D-045: the windows scored in the series' raw units (a flat context) are counted, never silent; the bank of this test has none
    flat = summary["flat_windows"]
    assert set(flat) == {"select", "test", "train_fraction"} and flat["select"] == flat["test"] == 0, flat
    assert 0.0 <= flat["train_fraction"] < 0.1
    assert summary["status"] == "ok" and summary["success_metric"] == "val_mae" and summary["monitor"] == "val_mae"
    assert summary["monitor_mode"] == "min" and summary["task"] == TASK
    # the baselines: the keys, finite and positive, the same select windows for every family (one reference), no model in them
    baselines = summary["baselines"]
    assert set(baselines) == {"naive_last", "seasonal_naive", "context_mean", "n_windows"}
    assert baselines == reference_baselines and baselines["n_windows"] == E2E_DATA["n_select"]
    assert all(math.isfinite(v) and v > 0 for v in baselines.values())
    # the epoch-zero value is the sidecar's, and the threshold is the smallest of the three
    epoch_zero = read_json(run_dir / "checkpoints" / FIT_STATE_NAME)["initial"]["metrics"]["mae"]
    assert summary["epoch_zero"]["val_mae"] == epoch_zero > 0
    assert summary["success_threshold"] == min(baselines["naive_last"], baselines["seasonal_naive"], epoch_zero)
    column = [float(v) for v in read_json(run_dir / "training_history.json")["val_mae"]]
    assert len(column) == E2E_EPOCHS and summary["success_value"] == pytest.approx(min(column), rel=1e-6)
    assert summary["success"] is fr.beats_bar(summary["success_value"], summary["success_threshold"])
    # the canonical post-hoc block, on the select and on the test windows
    quantile = fr.DECODE[ended.case].quantile
    for split in ("select", "test"):
        block = summary["canonical_metrics"][split]
        assert set(block) == CANONICAL_POINT | (CANONICAL_QUANTILE if quantile else set()), (split, sorted(block))
        assert all(math.isfinite(v) for key, v in block.items() if key != "MASE") and math.isfinite(block["MASE"])
        if quantile:
            assert 0.0 <= block["coverage"] <= 1.0 and block["sharpness"] > 0 and 0.0 < block["coverage_nominal"] <= 1.0
    # the same numbers two ways: the compiled `mae` and the post-hoc MAE decode the same windows through one row
    assert summary["canonical_metrics"]["test"]["MAE"] == pytest.approx(summary["test_metrics_best"]["mae"], rel=1e-4)
    assert summary["canonical_metrics"]["select"]["MAE"] == pytest.approx(summary["best_val_metrics"]["val_mae"], rel=1e-4)
    assert summary["analyzer"]["status"] == ANALYSIS[ended.case][0]


@pytest.mark.parametrize("which", ["function", "cli"])
def test_no_summary_note_says_the_run_cannot_be_predicted_or_exported_and_a_note_naming_task_json_is_true(ended, which):
    # The run writes `task.json` (the listing test above), so `predict` and `export` read it; a note that said otherwise
    # was once written, before the run wrote a record, and stayed in the summary of every run.
    run_dir = getattr(ended, which)
    notes = read_json(run_dir / "results_summary.json")["notes"]
    assert notes and all(isinstance(note, str) for note in notes)
    assert (run_dir / "task.json").is_file()
    for note in notes:
        assert "no `task.json` is written" not in note, note
        assert not ("`predict`" in note and "refused" in note), note
        assert not ("`export`" in note and "refused" in note), note
        if "task.json" in note:
            assert (run_dir / "task.json").is_file(), note


def test_an_epoch_is_steps_per_epoch_steps_and_fit_sees_both_lengths(ended):
    summary = read_json(ended.function / "results_summary.json")
    assert summary["steps_per_epoch"] == E2E_STEPS == summary["n_train"] // E2E_BATCH and summary["n_val"] // E2E_BATCH == 2
    assert len(ended.fits) == 2, "one fit per run: the function and the command line"
    for fit in ended.fits:
        assert fit["keywords"] == {"steps_per_epoch": E2E_STEPS, "validation_steps": summary["n_val"] // E2E_BATCH}
        # Observed, not read back: each epoch ran batches 0..STEPS-1, and a stream with no length would have run away.
        assert fit["spy"].last_batch == {epoch: E2E_STEPS - 1 for epoch in range(E2E_EPOCHS)}, "an epoch has the wrong length"


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_best_checkpoint_reloads_with_a_gap_of_exactly_zero(ended, which):
    summary = read_json(getattr(ended, which) / "results_summary.json")
    # The select and test windows are fixed by (seed, split, index): the reloaded file scores the in-memory weights bit for bit.
    assert summary["best_checkpoint_load_error"] is None and summary["best_checkpoint_max_abs_diff"] == 0.0
    assert summary["model_loading_validated"] is True and summary["model_loading_check"]["max_abs_diff"] == 0.0
    assert summary["test_metrics_best"] is not None and summary["test_metrics_final"] is not None


# ---------------------------------------------------------------------
# The bar can fail: a model that outputs the naive-last forecast, trained end to end, is a miss and exits 1
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="test_forecast_task")
class NaiveLastModel(keras.Model):
    """The naive-last forecast as a real, savable model with the output layout of nbeats ``(forecast, residual)``.

    One trainable scalar multiplies zero, so the optimizer has a variable and a gradient and the weights never change the
    forecast: the model IS the baseline at every epoch.
    """

    def __init__(self, horizon: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.idle = None

    def build(self, input_shape):
        self.idle = self.add_weight(name="idle", shape=(), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        forecast = keras.ops.repeat(inputs[:, -1:, :], self.horizon, axis=1) + 0.0 * self.idle
        return forecast, keras.ops.zeros_like(inputs[:, :, 0])

    def get_config(self):
        return {**super().get_config(), "horizon": self.horizon}


def create_naive_last(*, variant, num_classes=None, input_shape, horizon, **model_args):
    """The nbeats adapter's keywords, a :class:`NaiveLastModel` built at ``(None, lookback, channels)``."""
    model = NaiveLastModel(horizon)
    model.build((None, *input_shape))
    return model


@pytest.fixture(scope="module")
def naive_ended(world, tmp_path_factory):
    root = tmp_path_factory.mktemp("naive_last_end_to_end")
    asked = {"out": str(root / "run"), "model": MODEL, "data": dict(E2E_DATA), "epochs": E2E_EPOCHS,
             "batch_size": E2E_BATCH, "analysis_args": {"model_analysis": False, "figures": False}}
    fits: List[Dict[str, Any]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fm, "create_nbeats", create_naive_last)
        result, code = twice(root, asked, fits)
    return SimpleNamespace(root=root, result=result, code=code, function=root / "function", cli=root / "cli")


@pytest.mark.parametrize("which", ["function", "cli"])
def test_a_naive_last_model_run_end_to_end_is_a_miss(naive_ended, which):
    summary = read_json(getattr(naive_ended, which) / "results_summary.json")
    assert summary["status"] == "ok", "vacuity guard: the run finished and was judged"
    baseline = summary["baselines"]["naive_last"]
    # the instrument's zero: the model scores its own baseline to float32 rounding, and the run is a miss
    assert abs(summary["success_value"] - baseline) <= 8 * float(np.spacing(np.float32(baseline)))
    assert summary["success"] is False
    assert summary["canonical_metrics"]["test"]["rMAE"] == pytest.approx(1.0, rel=1e-5), "rMAE against naive-last is 1"


def test_a_naive_last_model_run_exits_one_through_the_command_line_and_reports_false_through_the_function(naive_ended):
    assert naive_ended.code == 1 and naive_ended.result.success is False


# ---------------------------------------------------------------------
# A csv, end to end: trains, reports, and no window of one segment holds a row of another
# ---------------------------------------------------------------------

def csv_data(path: Path, **changes: Any) -> Dict[str, Any]:
    return {"format": "csv", "path": str(path), "value_cols": ["v0"], "time_col": "t", "lookback": LOOKBACK,
            "horizon": HORIZON, "steps_per_epoch": E2E_STEPS, **changes}


@pytest.fixture(scope="module")
def csv_ended(world, tmp_path_factory, csv_file):
    root = tmp_path_factory.mktemp("csv_end_to_end")
    asked = {"out": str(root / "run"), "model": MODEL, "data": csv_data(csv_file), "epochs": E2E_EPOCHS,
             "batch_size": E2E_BATCH, "analysis_args": {"model_analysis": False, "figures": True}}
    fits: List[Dict[str, Any]] = []
    result, code = twice(root, asked, fits)
    return SimpleNamespace(root=root, csv=csv_file, asked=asked, result=result, code=code, fits=fits,
                           function=root / "function", cli=root / "cli")


@pytest.mark.parametrize("which", ["function", "cli"])
def test_a_csv_run_trains_and_reports_the_same_summary_blocks_as_the_synthetic_one(csv_ended, which):
    run_dir = getattr(csv_ended, which)
    summary = read_json(run_dir / "results_summary.json")
    assert summary["status"] == "ok" and summary["success_metric"] == "val_mae"
    assert names(run_dir) == RUN_FILES | (RUN_DIRS - {"model_analysis"}), "analysis_args.model_analysis=false writes no model_analysis/"
    assert summary["channels"] == 1 and summary["n_series"] == 1 and summary["series_dropped"] == []
    assert summary["flat_windows"] == {"select": 0, "test": 0, "train_fraction": 0.0}
    bounds = split_bounds(CSV_ROWS, 0.15, 0.15)
    assert summary["segments"] == {name: list(b) for name, b in zip(("train", "select", "test"), bounds)}
    baselines = summary["baselines"]
    assert summary["success_threshold"] == min(baselines["naive_last"], baselines["seasonal_naive"], summary["epoch_zero"]["val_mae"])
    assert summary["best_checkpoint_max_abs_diff"] == 0.0 and summary["n_val"] == 32 and summary["n_test"] == 32
    assert set(summary["canonical_metrics"]["test"]) == CANONICAL_POINT
    assert all(len(fit["spy"].last_batch) == E2E_EPOCHS for fit in csv_ended.fits), "the fit ran its epochs"


def test_the_csv_function_and_cli_write_the_same_config_bytes_and_the_cli_exits_by_the_verdict(csv_ended):
    assert (csv_ended.function / "config.json").read_bytes() == (csv_ended.cli / "config.json").read_bytes()
    resolved = read_json(csv_ended.cli / "config.json")["data"]
    assert resolved["format"] == "csv" and resolved["path"] == str(csv_ended.csv) and resolved["time_col"] == "t"
    success = read_json(csv_ended.cli / "results_summary.json")["success"]
    assert csv_ended.code == (0 if success else 1) and csv_ended.result.success is success


def locate(window: np.ndarray, raw: np.ndarray) -> int:
    """The row at which the standardized ``window`` ``(width, 1)`` starts in ``raw`` ``(rows, 1)``, or ``-1``.

    The standardization is affine with a positive scale, so the vector of one-step differences of a window points the same
    way as the one of the rows it was cut from: the candidate whose unit vector matches is where the window was cut.
    """
    width = len(window)
    unit = lambda seq: np.diff(seq, axis=-1) / np.linalg.norm(np.diff(seq, axis=-1), axis=-1, keepdims=True)
    starts = np.lib.stride_tricks.sliding_window_view(raw[:, 0], width)
    found = np.nonzero(np.all(np.isclose(unit(starts), unit(window[:, 0]), atol=1e-4), axis=1))[0]
    return int(found[0]) if len(found) else -1


def test_no_csv_window_of_one_segment_holds_a_row_of_another(world, csv_file, tmp_path):
    # No window crosses its segment, read off the data module's own windows: every window (context and horizon) of the three streams is cut from
    # inside its segment of the file. The windows are standardized, so each is located in the csv by its shape (`locate`).
    config = make_config(tmp_path / "probe", data=csv_data(csv_file), batch_size=E2E_BATCH)
    trainer = TimeSeriesForecastingTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    data = trainer.build_data(resolved, context)
    raw = pd.read_csv(csv_file)[["v0"]].to_numpy(np.float64)
    width = LOOKBACK + HORIZON
    streams = {"train": data.train_ds.take(4), "select": data.select_ds, "test": data.test_ds}
    for name, ds in streams.items():
        lo, hi = data.segments[name]
        x, y = windows(ds)
        assert len(x) >= 32, f"vacuity guard: {name} holds windows"
        for window in np.concatenate([x, y.reshape(len(y), HORIZON, 1)], axis=1):
            start = locate(window, raw)
            assert start >= 0, f"a {name} window is not a slice of the csv"
            assert lo <= start and start + width <= hi, f"a {name} window at rows [{start}, {start + width}) leaves [{lo}, {hi})"


# ---------------------------------------------------------------------
# Every refusal that can be made before the claim, through the command line: each is exit 2 with its named sentence, before the run directory is claimed
# ---------------------------------------------------------------------


@pytest.fixture
def within_30_seconds():
    """Fail a test that does not return in 30 s: a request that cannot be honoured must be refused, never hang."""
    def hang(signum, frame):
        raise TimeoutError("no answer within 30 seconds: the request hangs")

    previous = signal.signal(signal.SIGALRM, hang)
    signal.setitimer(signal.ITIMER_REAL, 30)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def level_csv(path: Path, offset: float = 1e12, factor: float = 1.0) -> Path:
    """A csv of one moving column times ``factor`` sitting ``offset`` above zero: its spread over its level is 1e-12 at the default."""
    write_series_csv(path, rows=1200)
    frame = pd.read_csv(path)
    frame["v0"] = frame["v0"] * factor + offset
    frame.to_csv(path, index=False)
    return path


PRECISION_SENTENCE = (
    "a checkpoint trained under a mixed precision policy cannot be reloaded in a float32 process (float16 against float32 in the "
    "layers), so predict and export of the run would be refused"
)


#: One refusal per (family, variant) whose measured minimum lookback is above the floor of every family, derived from the tables.
FLOOR_PREFIX = "lookback-below-minimum-"
FLOOR_KINDS = [f"{FLOOR_PREFIX}{family}-{variant}" for family, row in fm.FAMILIES.items() for variant in row.min_lookback]


def without(mapping: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def constant_select_csv(path: Path) -> Path:
    """A csv whose select segment (rows 1904 to 2311 of 2720 at the default fractions) holds ONE value in its only column."""
    write_series_csv(path)
    frame = pd.read_csv(path)
    lo, hi = split_bounds(len(frame), 0.15, 0.15)[1]
    frame.loc[lo:hi - 1, "v0"] = 5.0
    frame.to_csv(path, index=False)
    return path


def broken_csv(path: Path, kind: str) -> Path:
    """A csv with ONE defect: ``unsorted`` time, a ``non_finite`` cell, or ``two`` value columns."""
    write_series_csv(path, rows=1200, columns=2 if kind == "two" else 1)
    frame = pd.read_csv(path)
    if kind == "unsorted":
        frame.loc[[10, 11], "t"] = frame.loc[[11, 10], "t"].to_numpy()
    elif kind == "non_finite":
        frame.loc[5, "v0"] = np.inf
    frame.to_csv(path, index=False)
    return path


def refusal(tmp_path: Path, kind: str):
    """``(argv changes, fragment)`` of one refusal: what the request changes and the sentence the user reads."""
    if kind.startswith(FLOOR_PREFIX):
        family, variant = kind.removeprefix(FLOOR_PREFIX).split("-")
        floor = fm.min_lookback(family, variant)
        return ({"model": f"{family}/{variant}", "data": {**E2E_DATA, "lookback": floor - 1}},
                f"Invalid value for 'data.lookback': {family}/{variant} needs a lookback of at least {floor} (")
    csv = lambda name: broken_csv(tmp_path / f"{name}.csv", name)
    two = lambda model, fragment, **extra: (
        {"model": model, "data": csv_data(csv("two"), value_cols=None, time_col="t"), **extra}, fragment)
    return {
        "series-too-short": ({"data": {**E2E_DATA, "series_length": 100}},
                             "a synthetic series of 100 rows is too short: its select segment has 15 rows and one window "
                             "needs lookback + horizon = 40"),
        "csv-too-short": ({"data": csv_data(write_series_csv(tmp_path / "short.csv", rows=100))},
                          "the csv of 100 rows is too short: its select segment has 15 rows and one window needs lookback + "
                          "horizon = 40"),
        "unsorted-time": ({"data": csv_data(csv("unsorted"))},
                          "data.time_col 't' must be strictly increasing with no empty cell: sort the file by time"),
        "non-finite-cell": ({"data": csv_data(csv("non_finite"))},
                            "column 'v0' holds 1 non-finite cell(s), the first at row 5"),
        "missing-column": ({"data": csv_data(write_series_csv(tmp_path / "plain.csv"), value_cols=["nope"])},
                           "data.value_cols: column 'nope' is missing from plain.csv, which has ['t', 'v0']"),
        "timestral-horizon": ({"model": "timestral/5M", "data": {**E2E_DATA, "lookback": 64, "horizon": 40, "series_length": 1200}},
                              "timestral forecasts one patch of 32 points at a time"),
        "tirex-two-channels": two("tirex/tiny", "'tirex' takes 1 channel (TiRexCore reads one series, (B, L, 1)), got 2"),
        "ets-two-channels": two("ets/ANN", "'ets' takes 1 channel (ETSModel is a recursion over one scalar series, (B, L, 1)), got 2"),
        "mdn-two-channels": two(
            "mdn/tiny", "'mdn' takes 1 channel (the mixture is over the horizon of one series, (B, L, 1)), got 2"),
        "xlstm-quantile-two-channels": two(
            "xlstm/tiny", "an xlstm with a quantile head forecasts one series (its output is (batch, horizon, quantiles), with "
            "no channel axis, measured at 2 input channels), got 2 channels", model_args={"quantile_levels": QUANTILES}),
        "deepar": ({"model": "deepar/base"}, "the family 'deepar' is not trained by time_series_forecasting"),
        "adaptive-ema": ({"model": "adaptive_ema/base"}, "the family 'adaptive_ema' is not trained by time_series_forecasting"),
        "ets-jit": ({"model": "ets/ANN", "jit_compile": True}, "Invalid value for 'jit_compile': 'ets' cannot be compiled with XLA"),
        "lookback-one": ({"data": {**E2E_DATA, "lookback": 1}},
                         "Invalid value for 'data.lookback': expected an int of at least 2 (one value has no spread to standardize "
                         "by) or auto, got 1"),
        "csv-level": ({"data": csv_data(level_csv(tmp_path / "level.csv"))}, "have a flat context"),
        # D-045: the flat-window share (the review's csv shifted by 1e7) and the float32 range, both before the claim
        "csv-flat-1e7": ({"data": csv_data(level_csv(tmp_path / "flat7.csv", 1e7))},
                         "windows of the train segment have a flat context (std not above transform_args.scale_floor 1e-06"),
        "csv-float32-range-1e40": ({"data": csv_data(level_csv(tmp_path / "big40.csv", 0.0, 1e40))},
                                   "column 'v0' has max |x| "),
        "csv-float32-range-1e200": ({"data": csv_data(level_csv(tmp_path / "big200.csv", 0.0, 1e200))},
                                    "outside [1e-12, 1e15], the float32 range"),
        # The refusals the review listed as coming AFTER the claim (W4, W8 and the NOTE on late refusals, D-041 and D-042):
        "precision-mixed-float16": ({"precision": "mixed_float16"}, PRECISION_SENTENCE),
        "precision-mixed-bfloat16": ({"precision": "mixed_bfloat16"}, PRECISION_SENTENCE),
        "n-select-below-batch": ({"data": {**E2E_DATA, "n_select": E2E_BATCH - 1}},
                                 f"data.n_select = {E2E_BATCH - 1} is fewer than one batch ({E2E_BATCH}): a split needs a whole batch"),
        "n-test-below-batch": ({"data": {**E2E_DATA, "n_test": 3}},
                               f"data.n_test = 3 is fewer than one batch ({E2E_BATCH}): a split needs a whole batch"),
        "patterns-subset-below-batch": ({"data": {**without(E2E_DATA, "n_select"), "patterns": ["ar_process"]}},
                                        f"data.n_select = 8 is fewer than one batch ({E2E_BATCH}): a split needs a whole batch"),
        "csv-select-below-batch": ({"data": csv_data(write_series_csv(tmp_path / "small.csv", rows=800))},
                                   f"the select segment (120 rows) holds 11 windows with non-overlapping targets, fewer than one "
                                   f"batch of {E2E_BATCH}"),
        "csv-n-select-above-available": ({"data": csv_data(write_series_csv(tmp_path / "plain.csv"), n_select=64)},
                                         "data.n_select = 64 asks for more windows than the select segment holds (47)"),
        "csv-select-constant": ({"data": csv_data(constant_select_csv(tmp_path / "constant.csv"))},
                                "the select segment of the csv (rows [1904, 2312)) is constant in every value column"),
        "horizon-below-nbeats-minimum": ({"data": {**E2E_DATA, "horizon": 1}},
                                         "Invalid value for 'data.horizon': nbeats needs a horizon of at least 2"),
    }[kind]


REFUSALS = ["series-too-short", "csv-too-short", "unsorted-time", "non-finite-cell", "missing-column", "timestral-horizon",
            "tirex-two-channels", "ets-two-channels", "mdn-two-channels", "xlstm-quantile-two-channels", "deepar",
            "adaptive-ema", "ets-jit", "lookback-one", "csv-level", "csv-flat-1e7", "csv-float32-range-1e40",
            "csv-float32-range-1e200", "precision-mixed-float16", "precision-mixed-bfloat16",
            "n-select-below-batch", "n-test-below-batch", "patterns-subset-below-batch", "csv-select-below-batch",
            "csv-n-select-above-available", "csv-select-constant", "horizon-below-nbeats-minimum", *FLOOR_KINDS]


@pytest.mark.parametrize("kind", REFUSALS)
def test_a_refusal_of_invariant_9_is_exit_2_with_its_named_sentence_and_no_run_directory(
        world, tmp_path, capsys, within_30_seconds, kind):
    changes, fragment = refusal(tmp_path, kind)
    out = tmp_path / "run"
    asked = {"out": str(out), "model": MODEL, "data": dict(E2E_DATA), "epochs": 1, "batch_size": E2E_BATCH, **changes}
    code = cli.main([TASK, *flatten_request(asked)])
    error = capsys.readouterr().err
    assert code == 2, error
    assert fragment in error, error
    assert "Traceback" not in error and error.startswith("error:")
    assert not out.exists(), "the refusal came after the claim"


def test_the_measured_reason_of_the_precision_refusal_still_holds(tmp_path):
    """Review W4, D-041: a forecaster built and saved under ``mixed_float16`` does not run when it is loaded in a float32 process (a
    half tensor meets a float tensor in the layers), while it runs under the mixed policy. When Keras or the model stops doing this,
    the refusal's sentence is wrong and this test fails. The global policy is restored whatever happens."""
    import tensorflow as tf

    path, window = tmp_path / "mixed.keras", np.zeros((1, LOOKBACK, 1), np.float32)
    before = keras.config.dtype_policy()
    try:
        keras.config.set_dtype_policy("mixed_float16")
        model = fm.create_nbeats(variant="tiny", input_shape=(LOOKBACK, 1), horizon=HORIZON)
        model.save(str(path))
        keras.saving.load_model(str(path), compile=False)(window, training=False)  # the same process, the same policy: it runs
        keras.config.set_dtype_policy("float32")
        loaded = keras.saving.load_model(str(path), compile=False)
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="half|float16|float32"):
            loaded(window, training=False)
    finally:
        keras.config.set_dtype_policy(before)


def test_the_refusal_table_names_every_case_of_invariant_9():
    kinds = " ".join(REFUSALS)
    for needed in ("too-short", "unsorted", "non-finite", "missing-column", "timestral-horizon", "tirex-two", "ets-two", "mdn-two",
                   "xlstm-quantile-two", "deepar", "adaptive-ema", "ets-jit", "lookback-one", "csv-level", "precision-mixed-float16",
                   "precision-mixed-bfloat16", "n-select-below-batch", "n-test-below-batch", "patterns-subset-below-batch",
                   "csv-select-below-batch", "csv-n-select-above-available", "csv-select-constant", "horizon-below-nbeats-minimum",
                   "csv-flat-1e7", "csv-float32-range-1e40", "csv-float32-range-1e200"):
        assert needed in kinds, needed
    assert FLOOR_KINDS, "vacuity guard: some family has a minimum lookback above the floor of every family"
    assert len(REFUSALS) == len(set(REFUSALS)) == 27 + len(FLOOR_KINDS), "27 written here, one per (family, variant) minimum derived"


# ---------------------------------------------------------------------
# Guards for the rows of the mutant table that only a fixture error caught, and the edge of the bar through the command
# ---------------------------------------------------------------------

#: One model per decode row but timestral (its levels are the library's and building it is the slow part).
RECORD_CASES = [name for name in CASES if name != "timestral"]


@pytest.mark.parametrize("case", RECORD_CASES)
def test_the_record_of_a_built_model_carries_the_levels_and_the_median_the_model_has_and_passes_the_model_check(
        world, tmp_path, case):
    from train.api.predictor import check_model_against_record, check_record

    name, model_args = CASES[case]
    trainer, resolved, context = prepared(tmp_path / "run", model=name, model_args=model_args)
    data = trainer.build_data(resolved, context)
    model = trainer.build_model(resolved, context, data)
    record = trainer.task_record(resolved, context, data)
    decoder = fr.decoder_for(context.family, model)
    assert record["decode"] == case == decoder.row
    if fr.DECODE[case].quantile:
        assert record["head"] == "quantile" and len(record["quantile_levels"]) > 1
        assert record["quantile_levels"] == [float(q) for q in decoder.levels]
        assert record["point_index"] == decoder.median_index and record["point_index"] is not None
        if "quantile_levels" in model_args:
            assert record["quantile_levels"] == model_args["quantile_levels"] and record["point_index"] == QUANTILES.index(0.5)
    else:
        assert record["head"] == "point" and record["quantile_levels"] is None and record["point_index"] is None
    try:
        check_record(record, "<built>")
        # the model the record names is the model that was built, for a family whose variants are a table and for ets (a tuple
        # of names)
        check_model_against_record(model, record, registry.MODELS[context.family], Path("best_model.keras"), Path("task.json"))
    except Exception as error:
        raise AssertionError(f"the record of a built {case} model is refused or does not match its model: {error!r}") from error


def test_the_best_checkpoint_reload_check_is_fed_the_sample_the_expected_outputs_came_from(world, tmp_path, monkeypatch):
    from train.common import evaluation

    seen: List[Any] = []
    original = evaluation.check_model_loading

    def spy(path, sample, expected, **kwargs):
        seen.append((np.array(sample), expected))
        return original(path, sample, expected, **kwargs)

    monkeypatch.setattr(evaluation, "check_model_loading", spy)
    config = make_config(tmp_path / "run")
    batches = forecast_data.load_forecast_batches(
        format="builtin", dataset="synthetic", path=None, value_cols=None, time_col=None, lookback=LOOKBACK, horizon=HORIZON,
        realizations=1, series_length=600, noise_level=0.1, patterns=PATTERNS, select_fraction=0.15, test_fraction=0.15,
        n_select=16, n_test=16, steps_per_epoch=STEPS, max_samples=None, batch_size=BATCH, seed=config.seed,
        stream_seed=None, scale_floor=config.transform_args.scale_floor)
    first_batch = next(iter(batches.select_ds))[0].numpy()
    failure = None
    try:
        train_time_series_forecasting(out=str(tmp_path / "run"), model=MODEL, data=dict(DATA), epochs=1, batch_size=BATCH,
                                      analysis_args={"model_analysis": False, "figures": False})
    except Exception as error:  # the assertions below say what went wrong first
        failure = error
    assert seen, "vacuity guard: the best checkpoint was checked"
    sample, expected = seen[0]
    assert np.array_equal(sample, first_batch), "the reload check is fed the first selection batch the expected outputs came from"
    assert expected[0].shape == (BATCH, HORIZON, 1) and sample.shape == (BATCH, LOOKBACK, 1)
    assert failure is None, failure


def threshold_whose_band_edge_is(value: float, steps: int = 0) -> float:
    """The float32 threshold ``t`` with ``t - TIE_ULPS * spacing(t) == value`` (``steps`` float32 steps above that)."""
    value32 = np.float32(value)
    ulps = np.float32(fr.TIE_ULPS)
    t = value32 + ulps * np.spacing(value32)
    for _ in range(8 * fr.TIE_ULPS):
        low = t - ulps * np.spacing(t)
        if low == value32:
            break
        t = np.nextafter(t, np.float32(np.inf if low < value32 else -np.inf))
    else:
        raise AssertionError(f"no float32 threshold has {value} on the edge of its band")
    for _ in range(steps):
        t = np.nextafter(t, np.float32(np.inf))
    return float(t)


@pytest.fixture(scope="module", params=[0, 1], ids=["on_the_edge", "one_step_inside"])
def bar_edge_ended(request, world, tmp_path_factory, naive_ended):
    """The naive-last run again with its threshold set so that its ``success_value`` sits exactly on the edge of the band."""
    reached = read_json(naive_ended.function / "results_summary.json")["success_value"]
    threshold = threshold_whose_band_edge_is(reached, request.param)
    root = tmp_path_factory.mktemp(f"bar_edge_{request.param}")
    asked = {"out": str(root / "run"), "model": MODEL, "data": dict(E2E_DATA), "epochs": E2E_EPOCHS,
             "batch_size": E2E_BATCH, "analysis_args": {"model_analysis": False, "figures": False}}
    baselines = fr.baseline_maes
    fits: List[Dict[str, Any]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fm, "create_nbeats", create_naive_last)
        patch.setattr(fr, "baseline_maes", lambda x, y: {**baselines(x, y), "naive_last": threshold, "seasonal_naive": 1e9})
        patch.setattr(fr, "epoch_zero_mae", lambda path: 1e9)
        result, code = twice(root, asked, fits)
    return SimpleNamespace(offset=request.param, reached=reached, threshold=threshold, result=result, code=code,
                           function=root / "function", cli=root / "cli")


def test_the_planted_threshold_puts_the_reached_value_on_the_edge_of_the_band(bar_edge_ended):
    ended = bar_edge_ended
    assert np.float32(ended.threshold) - np.float32(fr.TIE_ULPS) * np.spacing(np.float32(ended.threshold)) == (
        np.float32(ended.reached) + np.float32(ended.offset) * np.spacing(np.float32(ended.reached)))
    assert fr.beats_bar(ended.reached, ended.threshold) is bool(ended.offset)


def test_the_bar_is_strict_at_the_edge_of_its_band_through_the_command_line_and_the_function(bar_edge_ended):
    ended = bar_edge_ended
    for which in ("function", "cli"):
        summary = read_json(getattr(ended, which) / "results_summary.json")
        assert summary["status"] == "ok" and summary["success_threshold"] == ended.threshold, "the threshold is the planted one"
        assert summary["success_value"] == ended.reached, "the model scores what the first run scored"
        assert summary["success"] is bool(ended.offset)
    # the edge itself is a MISS (strictly below is required), one float32 step inside it is a hit: exit 1 against exit 0
    assert ended.code == (0 if ended.offset else 1) and ended.result.success is bool(ended.offset)
