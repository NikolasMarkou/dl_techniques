"""The ``time_series_forecasting`` task: the hook table, the bar, the seams, the refusals and the registration.

``train.api.tasks.time_series_forecasting.TimeSeriesForecastingTrainer`` (plan-2026-11-13T105600-3a8319ef step 12) is the
task of seven families (``nbeats``, ``prism``, ``tirex``, ``xlstm``, ``ets``, ``timestral``, ``mdn``). It supplies the ten
required hooks, the seams of the shared sequence and the bar, and nothing else (a task overrides hooks, never stages).
This first part pins, on the CPU at tiny sizes (a bank of four synthetic patterns, lookback 32, horizon 8, batch 8):

* **The hook table.** Every hook of ``REQUIRED_HOOKS`` is overridden, and so is each optional one the task relies on;
  ``task_record`` answers ``None`` for now (step 15 gives the task a record), no per-epoch figure is attached.
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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import keras
import numpy as np
import pytest

from train.api import forecast_models as fm
from train.api import recipes, registry
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import BUILTIN_TASKS, register_builtin_tasks
from train.api.tasks.time_series_forecasting import (
    TimeSeriesForecastingConfig,
    TimeSeriesForecastingTrainer,
)
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer, config_problems
from train.common import forecast_run as fr
from train.common.pretrain_data import derive_stream_seed
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

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


@pytest.fixture(scope="module")
def world():
    """An empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


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


def test_the_task_attaches_no_per_epoch_figure_and_writes_no_record_yet(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    assert trainer.figure_callbacks(resolved, None, tmp_path) == []
    assert trainer.task_record(resolved, context, None) is None
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
    decoder = fr.Decoder("ets", lambda out: (out, None), None, None, False)
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


def write_csv(path: Path, columns: int, rows: int = 900) -> Path:
    rng = np.random.default_rng(0)
    values = np.cumsum(rng.normal(size=(rows, columns)), axis=0)
    header = ",".join(f"v{i}" for i in range(columns))
    np.savetxt(path, values, delimiter=",", header=header, comments="")
    return path


@pytest.mark.parametrize("model", ["tirex/tiny", "ets/ANN"])
def test_a_second_channel_is_refused_before_the_claim_for_a_one_channel_family(world, tmp_path, model):
    out, csv = tmp_path / "run", write_csv(tmp_path / "two.csv", 2)
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


_HELP_CHILD = """
import json, sys
from train.api import cli
codes = [cli.main(["help"]), cli.main(["time_series_forecasting", "help"])]
heavy = sorted(m for m in ("tensorflow", "keras", "train.common", "yaml") if m in sys.modules)
print("@@" + json.dumps([codes, heavy]))
"""


def test_the_help_of_the_command_and_of_the_task_load_no_tensorflow(tmp_path):
    env = {**__import__("os").environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"}
    done = subprocess.run([sys.executable, "-c", _HELP_CHILD], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                          timeout=60)
    assert done.returncode == 0, done.stderr[-1500:]
    codes, heavy = json.loads(done.stdout.rsplit("@@", 1)[1])
    assert codes == [0, 0] and heavy == [], heavy
    assert TASK in done.stdout and "jit_compile" in done.stdout and "data.lookback" in done.stdout


# ---------------------------------------------------------------------
# What the step added to shared code, and what stays what it was
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
    out, csv = tmp_path / "run", write_csv(tmp_path / "two.csv", 2)
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
