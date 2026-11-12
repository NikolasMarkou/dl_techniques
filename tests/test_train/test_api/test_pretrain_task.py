"""The ``pretrain`` task: the hook table, the bar, the epoch-zero term and the refusals before the claim (first part).

``train.api.tasks.pretrain.PretrainTrainer`` (plan-2026-11-12T015747-46e5318f step 10) is the task of two families:
``energy_transformer`` (masked-patch completion) and ``beit`` (masked image modelling against a frozen tokenizer's code
ids). It supplies the ten required hooks, the two seams of the shared sequence (``fit_kwargs``, ``resume_stream``) and
the bar, and nothing else (a task overrides hooks, never stages). This part pins:

* **The hook table.** Every hook of ``REQUIRED_HOOKS`` is overridden, and so is each optional one the task relies on;
  ``task_record`` answers ``None`` (the task writes no ``task.json``, D-013).
* **The bar** (D-009). ``val_loss`` is a loss, so the bar is STRICTLY below, in float32: equal is a miss, and a float64
  value that rounds to the float32 threshold is judged in float32. ``success_value`` is the MINIMUM of ``val_loss`` over
  the selection window (and ignores the epochs before it). The threshold is ``min(baseline, epoch-zero)``: the baseline is
  the ``val_loss`` of a constant model on the fixed select stream, the epoch-zero value is read from the fit sidecar in
  ``evaluate_and_report``, so a resume (which never evaluates the untrained model) reads the SAME number the fresh run
  wrote. The resume here is real: a run whose reporting died, resumed to report.
* **The seams.** ``fit_kwargs`` gives ``steps_per_epoch`` and ``validation_steps``; ``resume_stream`` gives another
  stream (the control: two fresh builds agree), with the split, the counts and the select stream unchanged.
* **Refusals.** A model argument the family refuses, a missing tokenizer and a conformal level leave ``out`` absent;
  ``debug_args.smoke`` applies no bar.

The runs are real ``energy_transformer/et_tiny`` runs (``model_args.num_steps=2``) on the fake CIFAR-10 of
``tests/classification_fakes.py``: 54 fit rows, batch 4, so 13 steps of an epoch. Tests write only under ``tmp_path``.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tests.classification_fakes import use_identity_cifar10
from train.api import recipes, registry
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import pretrain_run
from train.common.pretrain_run import Baseline
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

from train.api.tasks.pretrain import PretrainConfig, PretrainTrainer

BATCH = 4
EPOCHS = 2
N_FIT = 54                         # the fake cifar10's 60 train rows less the 10 percent selected
STEPS = N_FIT // BATCH             # 13
#: A baseline far above any random-init ET loss (2 to 3), so that the epoch-zero term of the threshold binds.
HUGE = 1000.0

ET_MODEL = "energy_transformer/et_tiny"
BEIT_MODEL = "beit/beit_tiny"


def make_config(out: Path, **changes: Any) -> PretrainConfig:
    """A validated ET config at the test sizes; a dotted key (``"debug_args.smoke"``) sets a field of a group."""
    mapping: Dict[str, Any] = {
        "out": str(out),
        "model": ET_MODEL,
        "data": {"dataset": "cifar10"},
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "model_args": {"num_steps": 2},
        "analysis_args": {"model_analysis": False, "figures": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping[head] = {**mapping.get(head, {}), tail: value}
        else:
            mapping[head] = value
    return validate(PretrainConfig, mapping)


@pytest.fixture(scope="module")
def world():
    """Fake cifar10 and an empty registry holding every built-in family and task."""
    with pytest.MonkeyPatch.context() as patch:
        use_identity_cifar10(patch)
        patch.setattr(registry, "MODELS", {})
        patch.setattr(registry, "TASKS", {})
        recipes.register_builtin_models()
        register_builtin_tasks()
        yield


def prepared(out: Path, **changes: Any):
    """``(trainer, resolved config, context)``: what the sequence hands the hooks, resolved the way ``run`` resolves."""
    config = make_config(out, **changes)
    trainer = PretrainTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    return trainer, resolved, context


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def csv_column(run_dir: Path, name: str) -> List[float]:
    with open(run_dir / "training_log.csv", newline="") as handle:
        return [float(row[name]) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------
# The hook table
# ---------------------------------------------------------------------

#: The optional hooks the pretrain task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",            # the stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",         # a resumed fit starts a fresh iterator: another stream, not a replay
    "bar_is_met",            # a loss: strictly below, in float32 (the default is >=)
    "figure_callbacks",      # the energy trace of ET
    "run_analysis",          # data-free (a two-input model whose output is not class probabilities)
    "summary_fields",        # the mask counts, the stream seed rule, the method
    "recipe_notices",        # the learning-rate regime the recipe doubts (D-012)
    "used_arguments",        # the model keywords the run was built with
    "check_model_args",      # the refusals before the claim
    "epoch_line_metrics",    # the epoch line prints loss and val_loss only
    "unjudged_note",         # why init_scale_warning is null
    "task_record",           # the task writes none
)


def test_the_class_names_its_task_and_its_metric():
    assert PretrainTrainer.task == "pretrain"
    assert PretrainTrainer.success_metric == "val_loss"
    assert issubclass(PretrainTrainer, TaskTrainer)


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_every_required_hook_is_overridden(hook):
    assert getattr(PretrainTrainer, hook) is not getattr(TaskTrainer, hook), hook
    assert callable(getattr(PretrainTrainer, hook))


@pytest.mark.parametrize("hook", OPTIONAL_HOOKS)
def test_every_optional_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(PretrainTrainer, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_writes_no_record_and_no_stamps(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    assert trainer.task_record(resolved, context, None) is None
    assert trainer.config_stamps(resolved, registry.MODELS["energy_transformer"]) == {}


def test_the_task_does_not_name_a_stage():
    # A task overrides hooks, never the stages of the shared sequence.
    own = {name for name in vars(PretrainTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim"}


# ---------------------------------------------------------------------
# The bar: strictly below, in float32
# ---------------------------------------------------------------------


def bare_trainer() -> PretrainTrainer:
    """A trainer with no config, for the pure hooks (``object.__new__``: the constructor needs a config)."""
    trainer = object.__new__(PretrainTrainer)
    trainer.config = None
    return trainer


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
    # Control: one float32 step below is a hit, so the float32 judgement is not "never".
    assert trainer.bar_is_met(None, float(np.nextafter(np.float32(threshold), np.float32(0))), threshold) is True


def test_the_value_is_the_minimum_of_val_loss_and_not_the_maximum_or_the_last():
    trainer = bare_trainer()
    history = {"loss": [9.0, 8.0, 7.0], "val_loss": [3.0, 1.0, 2.0]}
    assert trainer.success_value(history) == 1.0
    # The minimum, wherever it falls.
    assert trainer.success_value({"val_loss": [0.5, 1.0, 2.0]}) == 0.5
    assert trainer.success_value({"val_loss": [2.0, 1.0, 0.25]}) == 0.25


def test_the_value_ignores_the_epochs_before_the_selection_start():
    class Windowed(PretrainTrainer):
        def selection_start_epoch(self, config):
            return 2

    trainer = object.__new__(Windowed)
    trainer.config = None
    # The best loss of the whole run (0.1) is at epoch 0, before the window: it was never saved, so it cannot count.
    assert trainer.success_value({"val_loss": [0.1, 0.2, 0.9, 0.8]}) == 0.8
    assert bare_trainer().success_value({"val_loss": [0.1, 0.2, 0.9, 0.8]}) == 0.1


def test_the_threshold_is_the_smaller_of_the_baseline_and_the_epoch_zero_loss():
    trainer = bare_trainer()
    trainer._baseline = Baseline("zero_patches", 1.0)
    trainer._epoch_zero = 0.8
    assert trainer.success_threshold(None) == 0.8
    trainer._epoch_zero = 2.5
    assert trainer.success_threshold(None) == 1.0


def test_a_threshold_asked_before_the_data_or_the_sidecar_is_refused_by_name():
    trainer = bare_trainer()
    trainer._baseline, trainer._epoch_zero = None, 0.8
    with pytest.raises(RuntimeError, match="baseline"):
        trainer.success_threshold(None)
    trainer._baseline, trainer._epoch_zero = Baseline("zero_patches", 1.0), None
    with pytest.raises(RuntimeError, match="epoch-zero"):
        trainer.success_threshold(None)


def test_the_epoch_zero_loss_is_read_from_the_sidecar(tmp_path):
    path = tmp_path / FIT_STATE_NAME
    update_fit_state(path, initial={"metrics": {"loss": 0.7}, "ratio": 1.0, "warned": None})
    assert pretrain_run.epoch_zero_val_loss(path) == 0.7
    with pytest.raises(ValueError, match="epoch-zero"):
        pretrain_run.epoch_zero_val_loss(tmp_path / "no_such_sidecar.json")


# ---------------------------------------------------------------------
# A real run, its reporting killed, and the resume that reports: one epoch-zero value
# ---------------------------------------------------------------------


class ReportingCrash(PretrainTrainer):
    """The real task whose reporting dies after ``fit``: a run that a resume can only report."""

    def evaluate_and_report(self, config, data, fit):
        raise RuntimeError("the reporting died")


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):
    """A whole run (A), a run killed in its reporting (B) and B resumed, each with the epoch-zero term binding."""
    root = tmp_path_factory.mktemp("pretrain_runs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "constant_baseline", lambda *a, **k: Baseline("zero_patches", HUGE))
        whole = root / "whole"
        PretrainTrainer(make_config(whole)).run()
        crashed = root / "crashed"
        with pytest.raises(RuntimeError, match="the reporting died"):
            ReportingCrash(make_config(crashed, **{"analysis_args.figures": True})).run()
        before = read_json(crashed / "checkpoints" / FIT_STATE_NAME)
        trace_before = (crashed / "visualizations" / "energy_trace.csv").read_text()
        PretrainTrainer(make_config(crashed, resume_interrupted=True, **{"analysis_args.figures": True})).run()
    return {"whole": whole, "crashed": crashed, "sidecar_before": before, "trace_before": trace_before}


def sidecar_epoch_zero(run_dir: Path) -> float:
    return read_json(run_dir / "checkpoints" / FIT_STATE_NAME)["initial"]["metrics"]["loss"]


def test_a_whole_run_judges_against_the_smaller_of_the_baseline_and_its_epoch_zero_loss(runs):
    summary = read_json(runs["whole"] / "results_summary.json")
    epoch_zero = sidecar_epoch_zero(runs["whole"])
    assert epoch_zero < HUGE, "the control: the epoch-zero term must bind for this test to see the threshold"
    assert summary["success_metric"] == "val_loss"
    assert summary["baseline"] == {"kind": "zero_patches", "val_loss": HUGE}
    assert summary["epoch_zero"]["val_loss"] == epoch_zero
    assert summary["success_threshold"] == min(HUGE, epoch_zero)
    # The value is the minimum of the val_loss column and the verdict is the strict float32 comparison of it.
    column = csv_column(runs["whole"], "val_loss")
    assert len(column) == EPOCHS and summary["success_value"] == pytest.approx(min(column), rel=1e-6)
    assert summary["success"] is bool(np.float32(summary["success_value"]) < np.float32(summary["success_threshold"]))
    assert summary["steps_per_epoch"] == STEPS


def test_a_resume_reads_the_epoch_zero_value_the_fresh_run_wrote(runs):
    crashed = runs["crashed"]
    summary = read_json(crashed / "results_summary.json")
    assert summary["reporting_resumed"] is True
    # The resumed process never evaluated the untrained model (`initial_loss_check` is not called on a resume): the
    # sidecar the FIRST process wrote is the only place the number is.
    first = runs["sidecar_before"]["initial"]["metrics"]["loss"]
    assert sidecar_epoch_zero(crashed) == first
    assert summary["epoch_zero"]["val_loss"] == first
    assert summary["success_threshold"] == min(HUGE, first)
    assert summary["initial_loss_sanity_eval"]["loss"] == first


def test_the_resumed_run_and_the_whole_run_start_from_one_epoch_zero_loss(runs):
    # Same seed, same data: the two fresh epoch-zero evaluations agree (CPU), so the verdicts are comparable.
    assert sidecar_epoch_zero(runs["crashed"]) == pytest.approx(sidecar_epoch_zero(runs["whole"]), rel=1e-5)


def test_the_energy_trace_has_one_header_and_one_row_per_finished_epoch(runs):
    trace = (runs["crashed"] / "visualizations" / "energy_trace.csv").read_text().splitlines()
    assert trace[0].split(",")[0] == "epoch" and [line.split(",")[0] for line in trace[1:]] == [str(e) for e in range(EPOCHS)]
    # A reporting-only resume trains nothing, so the file is what the first process wrote.
    assert "\n".join(trace) == runs["trace_before"].rstrip("\n")


def test_the_run_directory_holds_no_task_record(runs):
    for name in ("whole", "crashed"):
        assert not (runs[name] / "task.json").exists(), name


# ---------------------------------------------------------------------
# The seams
# ---------------------------------------------------------------------


def first_batch_digest(ds) -> str:
    import hashlib

    (image, mask), _, _ = next(iter(ds))
    return hashlib.sha256(np.asarray(image).tobytes() + np.asarray(mask).tobytes()).hexdigest()


def test_fit_kwargs_gives_the_epoch_and_the_validation_length(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    assert trainer.fit_kwargs(resolved, data) == {"steps_per_epoch": STEPS, "validation_steps": data.val_steps}
    assert data.steps_per_epoch == STEPS == data.n_train // BATCH


def test_a_resumed_stream_is_not_a_replay_and_keeps_the_split(world, tmp_path):
    # Un-augmented: the augmentation seeds are re-randomised on every iterator (`rerandomize_each_iteration`), so two
    # iterators of one stream differ in their pixels. And every digest below is the FIRST iterator of its dataset: the
    # shuffle reshuffles on each new iterator, so a second pass over one dataset is another order (a Keras `fit` after a
    # resume starts the first iterator of the rebuilt dataset, which is what is compared).
    trainer, resolved, context = prepared(tmp_path / "run", **{"transform_args.augment": False})

    def fresh():
        return PretrainTrainer(resolved).build_data(resolved, context)

    # The control: two fresh builds start on the same first batch, so a difference below is the stream, not noise.
    reference = first_batch_digest(fresh().train_ds)
    assert first_batch_digest(fresh().train_ds) == reference
    data = fresh()
    resumed = trainer.resume_stream(resolved, data, STEPS)
    assert resumed is not data and resumed.train_ds is not data.train_ds
    here = first_batch_digest(resumed.train_ds)
    assert here != reference, "the resumed stream replays the fresh run's first batch"
    # Another count, another stream; the same count, the same stream.
    assert first_batch_digest(trainer.resume_stream(resolved, data, 2 * STEPS).train_ds) != here
    assert first_batch_digest(trainer.resume_stream(resolved, data, STEPS).train_ds) == here
    # What the sequence checks of a replacement: the same epoch, the same counts, the same select and test streams.
    assert (resumed.steps_per_epoch, resumed.n_train, resumed.n_select, resumed.n_test) == (
        data.steps_per_epoch, data.n_train, data.n_select, data.n_test)
    assert first_batch_digest(resumed.select_ds) == first_batch_digest(data.select_ds)
    assert resumed.stream_seed != data.stream_seed and resumed.seed == data.seed


def test_the_energy_trace_callback_skips_its_header_only_on_a_resume(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run", **{"analysis_args.figures": True})
    data = trainer.build_data(resolved, context)
    vis = tmp_path / "vis"
    vis.mkdir()
    (fresh,) = trainer.figure_callbacks(resolved, data, vis)
    assert fresh.skip_header_if_file_exists is False and Path(fresh.csv_path).parent == vis
    again, resolved_again, context_again = prepared(tmp_path / "run2", resume_interrupted=True, **{"analysis_args.figures": True})
    (resumed,) = again.figure_callbacks(resolved_again, data, vis)
    assert resumed.skip_header_if_file_exists is True


# ---------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"model_args": {"tokenizer_checkpoint": "t.keras"}}, "model_args.tokenizer_checkpoint"),
        ({"model_args": {"vocab_size": 16}}, "model_args.vocab_size"),
        ({"model_args": {"scale": "tiny"}}, "model_args.scale"),
        ({"optim_args": {"global_clipnorm": None}}, "optim_args.global_clipnorm"),
        ({"data": {"dataset": "cifar10", "num_mask_patches": 10}}, "data.num_mask_patches"),
    ],
    ids=["tokenizer-on-et", "vocab-size", "scale", "no-clip", "beit-mask-key"],
)
def test_a_refused_request_leaves_out_absent(world, tmp_path, changes, fragment):
    out = tmp_path / "run"
    with pytest.raises((ConfigError, ConfigValidationError)) as caught:
        PretrainTrainer(make_config(out, **changes)).run()
    assert fragment in str(caught.value)
    assert not out.exists(), "the refusal came after the claim"


def test_a_beit_run_without_a_tokenizer_is_refused_before_the_claim(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        PretrainTrainer(make_config(out, model=BEIT_MODEL, model_args={})).run()
    assert "model_args.tokenizer_checkpoint" in str(caught.value)
    assert not out.exists()


def test_a_conformal_level_is_refused_by_key():
    with pytest.raises(ConfigValidationError) as caught:
        make_config(Path("/nonexistent/run"), **{"analysis_args.conformal_alpha": 0.1})
    assert "analysis_args.conformal_alpha" in str(caught.value)


def test_a_smoke_run_applies_no_bar(world, tmp_path):
    out = tmp_path / "smoke"
    result = PretrainTrainer(make_config(out, epochs="auto", **{"debug_args.smoke": True})).run()
    summary = read_json(out / "results_summary.json")
    assert summary["success"] is None and summary["success_threshold"] is None
    assert result.success is True, "a smoke run that finished is a success"
    assert any("not applicable" in note for note in summary["notes"])
    # The measurements the bar would have used are still reported.
    assert summary["baseline"]["kind"] == "zero_patches" and summary["epoch_zero"]["val_loss"] > 0
    assert summary["epochs_requested"] == 1
