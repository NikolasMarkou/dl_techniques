"""The ``dino`` task's own wiring, held against the legacy trainer's functions: the callbacks it builds, the bar's two terms, what it refuses.

``DinoTrainer._callbacks`` hands four values to library code and the task tests that follow never saw one of them move (plan-2026-11-26T224507-e3381378,
fix round 11.1, review-iter-1 concern 1): a refactor that dropped the EMA warmup argument trained the legacy BASELINE arm while the README said the
teacher was warmed up for an epoch, and ``model_args.teacher_temp_final=0.07`` (an offered key) was silently ignored. Each is read here from the callbacks
the task really builds, over a real small config, and compared with the legacy trainer's own function:

* **the EMA warmup in steps** against ``train.dino.train_dino.resolve_ema_warmup_steps`` (and the value the data gives);
* **the cosine decay horizon** (the whole run's steps) against ``cosine_ema_schedule`` over ``steps_per_epoch * epochs``;
* **the teacher temperature**, ``teacher_temp`` to ``teacher_temp_final`` through ``set_teacher_temp`` at each epoch, against
  ``create_teacher_temp_callback``, with the two ends different and across a resume;
* **the k-NN column** the bar's two terms are read from (``dino_knn_top1_k20``, never ``k10``), through ``evaluate_and_report`` with a sidecar whose two
  columns differ.

Also here: the bar refuses a term that is not a number, the teacher warmup that is as long as the run (before the claim, fresh and resumed), a NaN
student ends the run as diverged through the loss alone (the probe stays finite), and a resume that only reports scores the test split against the
centre of the last epoch. The runs are tiny and offline (the identity fake of CIFAR-10, a 2-block backbone); tests write only under ``tmp_path``.
"""

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import cli, registry
from train.api.config import validate
from train.api.tasks import dino as task
from train.api.tasks.dino import CENTRE_FILE, DinoConfig, DinoTrainer, judge
from train.common.resume_state import FIT_STATE_NAME
from train.common.seed import set_seeds

from ._dino_world import (  # noqa: F401  (``identity_world`` is a fixture)
    IMAGE, OUT_DIM, PATCH, flatten_request, identity_world, shallow_backbone, task_request)

WATCH = "dino_knn_top1_k20"
#: A request whose epoch has 6 steps (54 fit rows over 8), over 4 epochs: the numbers below are derived from the data, never typed.
BATCH, EPOCHS = 8, 4


def resume_point(start: int) -> Any:
    """The point a run killed after ``start`` finished epochs resumes from (a real ``ResumePoint``: rows for every finished epoch)."""
    from train.common.resume_state import ResumePoint

    rows = tuple((epoch, {"loss": 1.0, WATCH: 0.2}) for epoch in range(start))
    return ResumePoint(start=start, best=0.2, wait=0, best_epoch=0, rows=rows, state={})


def with_data(tmp_path: Path, **changes: Any) -> Any:
    """``(trainer, resolved config, context, data)``: a request of ``BATCH`` and ``EPOCHS`` (``changes`` override) resolved and its data built."""
    config = validate(DinoConfig, task_request(tmp_path / "out", **{"batch_size": BATCH, "epochs": EPOCHS, **changes}))
    trainer = DinoTrainer(config)
    family, variant = registry.parse_model_name(config.model, task=trainer.task)
    resolved, context = trainer._resolve(trainer.describe_data(config), family, variant)
    set_seeds(resolved.seed)
    return trainer, resolved, context, trainer.build_data(resolved, context)


def prepare(tmp_path: Path, resume: Any = None, **changes: Any) -> SimpleNamespace:
    """A trainer taken as far as ``_callbacks``: resolved, its data built, its model and loss compiled for, its callbacks listed."""
    trainer, resolved, context, data = with_data(tmp_path, **changes)
    with pytest.MonkeyPatch.context() as patch:
        shallow_backbone(patch)
        trainer.build_model(resolved, context, data)
    trainer.build_loss_and_metrics(resolved, data)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    from train.common.classification_run import LastEpochWeights

    _, callbacks = trainer._callbacks(
        resolved, data, run_dir, run_dir / "visualizations", {"loss": 1.0}, "wiring", LastEpochWeights(), resume, None if resume is None else [])
    return SimpleNamespace(trainer=trainer, config=resolved, data=data, loss=trainer._loss, callbacks=callbacks, steps=data.steps_per_epoch)


def one(callbacks: List[Any], kind: type) -> Any:
    found = [callback for callback in callbacks if type(callback) is kind]
    assert len(found) == 1, f"{len(found)} {kind.__name__} in {[type(callback).__name__ for callback in callbacks]}"
    return found[0]


def legacy_config(**changes: Any) -> Any:
    from train.dino.train_dino import TrainingConfig

    return TrainingConfig(experiment_name="wiring", **changes)


# ---------------------------------------------------------------------
# the EMA: warmup in steps and the decay's horizon
# ---------------------------------------------------------------------


@pytest.mark.parametrize("warmup_epochs", [0.0, 0.5, 1.0, 2.5])
def test_the_ema_warmup_is_the_legacy_trainers_steps_of_the_epochs_the_request_names(identity_world, tmp_path, warmup_epochs):
    from train.api.dino_models import ema_warmup_steps
    from train.common.dino_run import CompiledTeacherEMA
    from train.dino.train_dino import resolve_ema_warmup_steps

    run = prepare(tmp_path, model_args={"dino_out_dim": OUT_DIM, "ema_warmup_epochs": warmup_epochs})
    ema = one(run.callbacks, CompiledTeacherEMA)
    assert run.steps == 6, "the data's epoch: 54 fit rows over a batch of 8, the number every figure below is derived from"
    assert ema.warmup_steps == resolve_ema_warmup_steps(legacy_config(ema_warmup_epochs=warmup_epochs), run.steps)
    assert ema.warmup_steps == ema_warmup_steps(warmup_epochs, run.steps) == int(round(warmup_epochs * run.steps))
    assert ema.log_every == run.steps


def test_the_ema_decays_along_the_cosine_of_the_whole_run_to_its_last_step_and_not_of_one_epoch(identity_world, tmp_path):
    from dl_techniques.models.vision.depth_anything.teacher_ema import cosine_ema_schedule
    from train.common.dino_run import CompiledTeacherEMA

    run = prepare(tmp_path)
    ema = one(run.callbacks, CompiledTeacherEMA)
    total = run.steps * EPOCHS
    legacy = cosine_ema_schedule(decay_start=0.996, decay_end=0.9999, total_steps=total)
    for step in (0, 1, run.steps, total // 2, total - 1, total, total + 5):
        assert ema.schedule(step) == legacy(step), step
    assert ema.schedule(total) == pytest.approx(0.9999) and ema.schedule(0) == pytest.approx(0.996)
    # A horizon of one epoch (the planted break) would be at its end after `steps` steps; the whole run's is far from it.
    assert ema.schedule(run.steps) < 0.9975, "the decay reaches its end after one epoch: the horizon is not the run's"


# ---------------------------------------------------------------------
# the teacher temperature
# ---------------------------------------------------------------------

#: Epoch -> temperature of the warmup 0.04 -> 0.07 over the task's 30 epochs (a linear ramp, clamped at its end).
TEMPERATURES = {0: 0.04, 15: 0.055, 30: 0.07, 31: 0.07}


def temperature_after(callbacks: List[Any], loss: Any, epoch: int) -> float:
    one(callbacks, keras.callbacks.LambdaCallback).on_epoch_begin(epoch)
    return float(keras.ops.convert_to_numpy(loss.teacher_temp))


@pytest.mark.parametrize("resumed_at", [None, 15])
def test_the_teacher_temperature_climbs_from_its_start_to_its_final_value_as_the_legacy_trainer_climbs_it(identity_world, tmp_path, resumed_at):
    from dl_techniques.losses.dino_loss import DINOLoss
    from train.dino.train_dino import create_teacher_temp_callback

    resume = None if resumed_at is None else resume_point(resumed_at)
    run = prepare(tmp_path, resume=resume, model_args={"dino_out_dim": OUT_DIM, "teacher_temp": 0.04, "teacher_temp_final": 0.07})
    assert run.loss.out_dim == OUT_DIM
    oracle_loss = DINOLoss(out_dim=OUT_DIM, teacher_temp=0.04)
    oracle = create_teacher_temp_callback(legacy_config(teacher_temp=0.04, teacher_temp_final=0.07, teacher_temp_warmup_epochs=30), oracle_loss)
    for epoch, expected in TEMPERATURES.items():
        oracle.on_epoch_begin(epoch)
        assert temperature_after(run.callbacks, run.loss, epoch) == pytest.approx(float(keras.ops.convert_to_numpy(oracle_loss.teacher_temp)), abs=0)
        assert temperature_after(run.callbacks, run.loss, epoch) == pytest.approx(expected, abs=1e-6), epoch


def test_a_constant_schedule_stays_where_it_is_and_the_two_ends_are_what_the_request_gave(identity_world, tmp_path):
    run = prepare(tmp_path)    # the recipe: 0.04 to 0.04
    assert [temperature_after(run.callbacks, run.loss, epoch) for epoch in (0, 15, 30)] == pytest.approx([0.04] * 3, abs=1e-7)
    run = prepare(tmp_path / "other", model_args={"dino_out_dim": OUT_DIM, "teacher_temp": 0.05, "teacher_temp_final": 0.09})
    assert temperature_after(run.callbacks, run.loss, 0) == pytest.approx(0.05, abs=1e-6)
    assert temperature_after(run.callbacks, run.loss, 30) == pytest.approx(0.09, abs=1e-6)


# ---------------------------------------------------------------------
# the k-NN column the bar's two terms come from
# ---------------------------------------------------------------------


def reporting_stub(tmp_path: Path, initial: Dict[str, float], baseline: Dict[str, float], monkeypatch: pytest.MonkeyPatch) -> Any:
    """``evaluate_and_report`` of a bare trainer, stopped right after the bar's two terms are read (the stream of the loss batch raises)."""
    from train.common import dino_run
    from train.common.resume_state import FIT_STATE_NAME

    class Stop(Exception):
        pass

    sidecar = tmp_path / "checkpoints" / FIT_STATE_NAME
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"initial": {"metrics": initial}}))
    monkeypatch.setattr(dino_run, "constant_feature_baseline", lambda data: baseline)
    trainer = object.__new__(DinoTrainer)
    trainer.config = SimpleNamespace(data=SimpleNamespace(dataset="cifar10"))
    trainer._loss, trainer._baseline, trainer._epoch_zero, trainer._flags, trainer._first_epoch, trainer._trained_here = object(), None, None, None, 1, True
    config = SimpleNamespace(
        checkpoint_args=SimpleNamespace(save_best=False), analysis_args=SimpleNamespace(), metric_args=SimpleNamespace(watch_metric=WATCH, watch_mode="max"))

    class Data:
        @property
        def loss_batch_ds(self):
            raise Stop

    fit = SimpleNamespace(run_dir=tmp_path, model=None, epochs_run=3)
    return SimpleNamespace(trainer=trainer, run=lambda: trainer.evaluate_and_report(config, Data(), fit), stop=Stop)


def test_the_bars_two_terms_are_read_from_the_k20_column_and_not_the_k10_one(tmp_path, monkeypatch):
    stub = reporting_stub(
        tmp_path, {"dino_knn_top1_k10": 0.10, WATCH: 0.30, "loss": 1.0}, {"dino_knn_top1_k10": 0.50, WATCH: 0.20}, monkeypatch)
    with pytest.raises(stub.stop):
        stub.run()
    assert stub.trainer._epoch_zero == 0.30, "the epoch-zero term is the sidecar's k20, not its k10 (0.10)"
    assert stub.trainer._baseline == 0.20, "the baseline term is the labels' k20, not their k10 (0.50)"
    assert stub.trainer.success_threshold(stub.trainer.config) == pytest.approx(0.30 + task.MARGIN["cifar10"])


@pytest.mark.parametrize("initial_value, baseline_value, named", [(math.nan, 0.2, "epoch zero"), (0.3, math.nan, "baseline"), (math.inf, 0.2, "epoch zero")])
def test_a_term_that_is_not_a_number_refuses_the_report_naming_it_before_any_scoring(tmp_path, monkeypatch, initial_value, baseline_value, named):
    stub = reporting_stub(tmp_path, {WATCH: initial_value, "loss": 1.0}, {WATCH: baseline_value}, monkeypatch)
    with pytest.raises((RuntimeError, stub.stop)) as caught:
        stub.run()
    assert not isinstance(caught.value, stub.stop), "the report went on to score the splits with a bar term that is not a number"
    assert re.search(rf"the bar cannot be judged: the {named} \S+ is not a finite number", str(caught.value)), str(caught.value)
    with pytest.raises(RuntimeError, match=f"the bar has no threshold: the {named}"):
        stub.trainer.success_threshold(stub.trainer.config)


@pytest.mark.parametrize("baseline, zero, named", [(0.1, math.nan, "epoch zero"), (math.nan, 0.1, "baseline"), (0.1, -math.inf, "epoch zero")])
def test_the_verdict_has_no_threshold_for_a_term_that_is_not_a_number_and_misses_part_one_saying_which(baseline, zero, named):
    verdict = judge(best=0.9, baseline=baseline, epoch_zero=zero, margin=0.05, collapse_flags=[0.0], centred_flags=[1.0])
    assert not verdict.part1_met and not verdict.met and verdict.part2_met and math.isnan(verdict.threshold)
    assert "part 1 missed" in verdict.text and named in verdict.text and "not a finite number" in verdict.text
    assert judge(best=0.9, baseline=0.1, epoch_zero=0.1, margin=0.05, collapse_flags=[0.0], centred_flags=[1.0]).met, "the control: finite terms pass"


# ---------------------------------------------------------------------
# a teacher warmup as long as the run, before the claim
# ---------------------------------------------------------------------


def run_cli(out: Path, **changes: Any) -> int:
    return cli.main(["dino", *flatten_request(task_request(out, **changes))])


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("warmup", [2, 2.0, 3])
def test_a_warmup_as_long_as_the_run_or_longer_is_refused_by_name_before_the_claim_fresh_and_resumed(identity_world, tmp_path, capfd, warmup, resume):
    out = tmp_path / "frozen"
    code = run_cli(out, epochs=2, resume_interrupted="true" if resume else "false", model_args={"dino_out_dim": OUT_DIM, "ema_warmup_epochs": warmup})
    error = capfd.readouterr().err
    assert code == 2 and not out.exists(), "a refusal is made BEFORE the run directory is claimed"
    assert "model_args.ema_warmup_epochs" in error and f"{warmup!r} is not below the run's 2" in error and "never start" in error


def test_the_recipes_own_warmup_is_refused_for_a_one_epoch_run_and_a_smaller_one_or_a_smoke_run_is_not(identity_world, tmp_path, capfd):
    assert run_cli(tmp_path / "one", epochs=1) == 2 and "is not below the run's 1" in capfd.readouterr().err and not (tmp_path / "one").exists()
    from train.api.dino_models import warmup_problems

    assert warmup_problems(0.5, 1) == [] and warmup_problems(1.0, 2) == [] and warmup_problems(0, 1) == []
    assert warmup_problems(1.0, 1, smoke=True) == [], "a smoke run is one epoch with no bar: it keeps the recipe's warmup"
    assert warmup_problems(1.0, 1) and warmup_problems(1.5, 1.5) and warmup_problems(math.nan, 2) == [], "NaN is the resolver's refusal, not this one's"


@pytest.mark.parametrize("warmup, epochs, steps, refused", [
    (59.999, 60, 266, True), (59.99, 60, 266, False), (1.999, 2, 28, True), (1.98, 2, 28, False), (0.92, 1, 6, True), (0.91, 1, 6, False), (0.0, 1, 1, False)])
def test_a_warmup_that_rounds_to_every_step_of_the_run_is_refused_though_its_epochs_are_below_the_run(warmup, epochs, steps, refused):
    from train.api.dino_models import ema_warmup_steps, warmup_problems, warmup_steps_problems

    assert warmup_problems(warmup, epochs) == [], "the epochs compare below the run's: the epoch rule lets every row through"
    lines = warmup_steps_problems(warmup, epochs, steps)
    assert bool(lines) is refused and (ema_warmup_steps(warmup, steps) >= steps * epochs) is refused
    if refused:
        assert "model_args.ema_warmup_epochs" in lines[0] and f"warmup of {steps * epochs} steps, which is all {steps * epochs} steps" in lines[0]
    assert warmup_steps_problems(warmup, epochs, steps, smoke=True) == []


@pytest.mark.parametrize("epochs, warmup, refused", [(1, 0.92, True), (1, 0.91, False), (4, 3.99, True), (4, 3.9, False)])
def test_the_data_stage_refuses_that_warmup_before_the_model_is_built_and_accepts_the_one_below_it(identity_world, tmp_path, epochs, warmup, refused):
    trainer, resolved, _, data = with_data(tmp_path, epochs=epochs, model_args={"dino_out_dim": OUT_DIM, "ema_warmup_epochs": warmup})
    assert data.steps_per_epoch == 6
    if refused:
        with pytest.raises(ValueError, match=rf"rounds to a warmup of {6 * epochs} steps, which is all {6 * epochs} steps of the run"):
            trainer.fit_kwargs(resolved, data)
    else:
        assert trainer.fit_kwargs(resolved, data) == {"steps_per_epoch": 6}


def test_that_refusal_exits_2_through_the_command_line_after_the_claim_naming_the_value_and_trains_nothing(identity_world, tmp_path, capfd):
    out = tmp_path / "rounded"
    code = run_cli(out, epochs=1, batch_size=BATCH, model_args={"dino_out_dim": OUT_DIM, "ema_warmup_epochs": 0.92})
    error = capfd.readouterr().err
    assert code == 2 and "Invalid argument for the fit options" in error and "rounds to a warmup of 6 steps" in error and "'out' is now used" in error
    assert not (out / "training_log.csv").exists() and not (out / "best_model.keras").exists(), "the refusal came before any training"


# ---------------------------------------------------------------------
# real tiny runs: a NaN student and a resume that only reports
# ---------------------------------------------------------------------


def run_with(out: Path, extra: Any = None, **changes: Any) -> int:
    """One ``cli.main`` of a tiny run whose callbacks list ends with ``extra(trainer)`` (when given); the 2-block backbone."""
    original = DinoTrainer._callbacks

    def callbacks(self, *arguments, **keywords):
        epoch_line, built = original(self, *arguments, **keywords)
        if extra is not None:
            built.append(extra(self))
        return epoch_line, built

    with pytest.MonkeyPatch.context() as patch:
        shallow_backbone(patch)
        patch.setattr(DinoTrainer, "_callbacks", callbacks)
        patch.setitem(task.MARGIN, "cifar10", -0.5)
        return run_cli(out, **changes)


def test_a_student_of_nan_weights_ends_the_run_as_diverged_through_the_loss_alone_because_the_probe_stays_finite(identity_world, tmp_path):
    class Poison(keras.callbacks.Callback):
        def __init__(self, trainer):
            super().__init__()
            self.trainer = trainer
            self.probe_values: List[float] = []

        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 1:
                for weight in self.model.student.weights:
                    weight.assign(np.full(tuple(weight.shape), np.nan, np.float32))

    out = tmp_path / "nan"
    # The command line lets this exception propagate (cli.py: "any other exception propagates, so Python exits 1"): a diverged run exits nonzero.
    with pytest.raises(RuntimeError, match=r"Training diverged: \['loss'\] hold a non-finite or missing value after 2 epoch"):
        run_with(out, Poison, epochs=3, **{"checkpoint_args.early_stopping_patience": 10})
    summary = json.loads((out / "results_summary.json").read_text())
    assert summary["status"] == "diverged", summary.get("status")
    assert "loss" in summary["non_finite_metrics"], "the loss is what marks the run diverged"
    assert WATCH not in summary["non_finite_metrics"], "the probe of a NaN student is FINITE (a chance accuracy), so the monitor alone would pass it"
    assert summary["history"]["loss"][-1] is None, "non-finite values are written as null (strict JSON)"
    assert np.isfinite(summary["history"][WATCH]).all()
    assert summary.get("success") is not True and "bar" not in summary, "a diverged run applies no bar and is never a success"
    assert not (out / "final_model.keras").exists(), "no final model of a diverged run"


class Ended(Exception):
    """Stands in for the reporting dying after ``fit`` finished."""


def test_a_resume_that_only_reports_scores_the_test_split_against_the_centre_of_the_last_epoch(identity_world, tmp_path, monkeypatch):
    from train.common import dino_run

    out = tmp_path / "reported"
    original = DinoTrainer.evaluate_and_report

    def dies(self, config, data, fit):
        raise Ended("the reporting died")

    monkeypatch.setattr(DinoTrainer, "evaluate_and_report", dies)
    with pytest.raises(Ended, match="the reporting died"):
        run_with(out, epochs=2)  # the first process trained and its reporting died
    monkeypatch.setattr(DinoTrainer, "evaluate_and_report", original)
    saved = np.load(out / "checkpoints" / CENTRE_FILE)
    assert np.abs(saved).max() > 0.0, "the control: the last epoch's centre is not zero"
    # The k10 term of the sidecar is moved far from its k20, so a reporting resume that read the wrong column would move the bar's threshold.
    sidecar = out / "checkpoints" / FIT_STATE_NAME
    state = json.loads(sidecar.read_text())
    zero = state["initial"]["metrics"][WATCH]
    state["initial"]["metrics"]["dino_knn_top1_k10"] = zero + 0.5
    sidecar.write_text(json.dumps(state))
    # A centre file that is gone after two finished epochs is REFUSED (a zero centre would be the silent wrong answer), not warned about.
    kept = (out / "checkpoints" / CENTRE_FILE).read_bytes()
    (out / "checkpoints" / CENTRE_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="is missing, but this run has finished 2 epochs"):
        run_with(out, epochs=2, resume_interrupted="true")
    (out / "checkpoints" / CENTRE_FILE).write_bytes(kept)
    seen: List[np.ndarray] = []
    real = dino_run.knn_of

    def spy(model, ds, *, data, loss):
        seen.append(np.array(keras.ops.convert_to_numpy(loss.center)))
        return real(model, ds, data=data, loss=loss)

    monkeypatch.setattr(dino_run, "knn_of", spy)
    assert run_with(out, epochs=2, resume_interrupted="true") == 0
    summary = json.loads((out / "results_summary.json").read_text())
    assert summary["reporting_resumed"] is True and seen, "the resume reported and scored the test split"
    for centre in seen:
        np.testing.assert_array_equal(centre, saved, err_msg="the test split was scored against another centre than the last epoch's")
    assert summary["bar"]["part2"]["epochs_checked"] == 2, "the bar's own columns come from the history, never from the centre"
    # The bar's part 1 of the reporting resume: the threshold is the sidecar's k20 (or the baseline) plus the margin the run patched, never the k10 column.
    assert summary["bar"]["part1"]["success_threshold"] == pytest.approx(max(summary["baseline"][WATCH], zero) - 0.5)
    assert summary["epoch_zero"][WATCH] == zero and summary["epoch_zero"]["dino_knn_top1_k10"] == zero + 0.5, "the summary reports the sidecar as it is"
