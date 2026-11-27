"""The ``dino`` task: the hooks, the two-part bar and the controls that can fail it, real tiny runs, the refusals before the claim.

``train.api.tasks.dino.DinoTrainer`` (plan-2026-11-26T224507-e3381378 step 6) trains the library ``DINOTrainingModel`` on stock ``compile`` and
``fit`` with the library ``DINOLoss`` (no ``train_step``, no validation data) and judges the run twice: its best k-NN accuracy against what a
constant and an untrained model score plus a margin, AND the collapse diagnostics at every epoch of the window. A bar that a model of zero
features, a collapsed teacher, a bank with the wrong labels or an untrained model could meet is vacuous, so each is a real probe of a real model
judged by the real hooks:

* **zero features** (the final norm zeroed) score chance, the collapse flag stays CLEAR (the flag cannot see it), so only part 1 misses;
* **a collapsed teacher** (the loss centre set so the teacher distribution is one-hot) fires the flag with the k-NN number unchanged, so a bar on
  part 1 alone would pass it and only part 2 misses;
* **a bank with the wrong labels** (every label moved by one class) misses part 1; **an untrained model** scores its own epoch-zero value and
  misses part 1; **a run that meets one part** misses.

Also pinned: the margin and its arithmetic, the hook table, ``success_value`` as the nanmax of the window with ``best_model.keras`` the epoch it
names (a planted best-not-last), the loss centre advancing exactly ``steps_per_epoch`` times per epoch with no ``evaluate`` or ``test_on_batch`` on
the DINO model, the CSV with every probe key each epoch and no ``val_`` column, the callback order, the files, the refusals before the claim
(exit 2, ``out`` absent), a smoke run (no bar) and the import rule. The runs are the REAL CIFAR-10 rows capped by ``data.max_samples`` (a loud skip
without the Keras cache) through a 2-block backbone; tests write only under ``tmp_path``. The resume is ``test_dino_resume.py``.
"""

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import cli, registry
from train.api.tasks import BUILTIN_TASKS
from train.api.tasks.dino import (
    CENTRE_FILE, CONTROL_SD, MARGIN, NULL_P99, PROBE_KEYS, DinoTrainer, judge, margin_of,
)
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer

from ._dino_world import (
    IMAGE, N_LOCAL, OUT_DIM, PATCH, TASK_BATCH, TASK_EPOCHS, TASK_MAX_SAMPLES, csv_column, flatten_request, shallow_backbone, task_request,
    world,  # noqa: F401  (a fixture)
)
from ._tokenizer_world import cifar, read_json  # noqa: F401  (``cifar`` is a fixture)

REPO_ROOT = Path(__file__).resolve().parents[3]
WATCH = "dino_knn_top1_k20"

# ---------------------------------------------------------------------
# The margin and the arithmetic of the bar (no model)
# ---------------------------------------------------------------------


def test_the_margin_is_a_function_of_the_two_measured_spreads_and_its_two_outputs_are_pinned():
    assert MARGIN == {"imagenette": 0.05, "cifar10": 0.085}
    assert CONTROL_SD == {"imagenette": 0.0103, "cifar10": 0.0198} and NULL_P99 == {"imagenette": 0.0487, "cifar10": 0.0416}
    assert {name: margin_of(CONTROL_SD[name], NULL_P99[name]) for name in CONTROL_SD} == MARGIN
    # The arithmetic: imagenette 3 sqrt(2) x 0.0103 = 0.0437 is under the null's 99th percentile 0.0487, which binds (ceil to 0.005 is 0.050);
    # cifar10 3 sqrt(2) x 0.0198 = 0.0840 binds over 0.0416 (ceil is 0.085).
    assert 3 * math.sqrt(2) * CONTROL_SD["imagenette"] < NULL_P99["imagenette"] < 0.05
    assert NULL_P99["cifar10"] < 3 * math.sqrt(2) * CONTROL_SD["cifar10"] <= 0.085 < 0.09
    # Each term binds in its own case, and the ceiling is to the step of 0.005 (an exact multiple stays).
    assert margin_of(0.01, 0.0) == 0.045 and margin_of(0.0, 0.0412) == 0.045 and margin_of(0.0, 0.045) == 0.045 and margin_of(0.0, 0.0451) == 0.05


@pytest.mark.parametrize(
    "best, baseline, zero, margin, met",
    [
        (0.5, 0.1, 0.3, 0.05, True),                                   # clearly above
        (0.3499, 0.1, 0.3, 0.05, False),                               # below the threshold (M28: the margin ignored would call it a pass)
        (0.2, 0.1, 0.3, 0.05, False),
        (0.35, 0.1, 0.3, 0.05, False),                                 # exactly the threshold: strict (M29)
        (float(np.nextafter(0.3 + 0.05, 1.0)), 0.1, 0.3, 0.05, True),  # the next float above passes
        (0.33, 0.4, 0.3, 0.05, False),                                 # the baseline binds over epoch zero (M34): 0.4 + 0.05 > 0.33 > 0.3 + 0.05
        (0.40, 0.4, 0.3, 0.05, False),                                 # above epoch zero + margin (0.35) and below baseline + margin (0.45): only the baseline term refuses it
        (0.46, 0.4, 0.3, 0.05, True),
    ],
)
def test_part_one_is_strictly_above_the_larger_of_baseline_and_epoch_zero_plus_the_margin(best, baseline, zero, margin, met):
    verdict = judge(best=best, baseline=baseline, epoch_zero=zero, margin=margin, collapse_flags=[0.0], centred_flags=[1.0])
    assert verdict.part1_met is met and verdict.part2_met and verdict.met is met
    assert verdict.threshold == max(baseline, zero) + margin
    assert ("part 1 missed" in verdict.text) is (not met) and "part 2" not in verdict.text


@pytest.mark.parametrize(
    "flags, centred, offending",
    [([0.0, 1.0], [1.0, 1.0], 2), ([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], 2), ([0.0, float("nan")], [1.0, 1.0], 2), ([0.0], [float("nan")], 1),
     ([], [], None)],
)
def test_part_two_names_the_first_epoch_where_a_flag_or_the_centred_marker_is_not_clean(flags, centred, offending):
    verdict = judge(best=0.9, baseline=0.1, epoch_zero=0.1, margin=0.05, collapse_flags=flags, centred_flags=centred, first_epoch=1)
    assert verdict.part1_met and not verdict.part2_met and not verdict.met  # M30, M31: either marker ignored passes one of these
    assert verdict.first_offending_epoch == offending
    assert "part 2 missed" in verdict.text and "part 1" not in verdict.text
    if offending is not None:
        assert f"epoch {offending}" in verdict.text


def test_a_window_that_starts_later_counts_its_epochs_from_its_start():
    verdict = judge(best=0.9, baseline=0.1, epoch_zero=0.1, margin=0.05, collapse_flags=[0.0, 1.0], centred_flags=[1.0, 1.0], first_epoch=4)
    assert verdict.first_offending_epoch == 5


def test_a_run_that_meets_one_part_misses_and_the_text_names_the_other():
    both = judge(best=0.0, baseline=0.1, epoch_zero=0.3, margin=0.05, collapse_flags=[1.0], centred_flags=[1.0])
    assert not both.part1_met and not both.part2_met and "part 1 missed" in both.text and "part 2 missed" in both.text


def bare_trainer(dataset="cifar10", **state) -> DinoTrainer:
    """A trainer with no config and the state ``evaluate_and_report`` leaves (``object.__new__``: the constructor needs a config)."""
    trainer = object.__new__(DinoTrainer)
    trainer.config = types_namespace(dataset)
    trainer._baseline, trainer._epoch_zero, trainer._flags, trainer._first_epoch = 0.1, 0.3, ([0.0, 0.0], [1.0, 1.0]), 1
    for key, value in state.items():
        setattr(trainer, key, value)
    return trainer


def types_namespace(dataset: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(data=SimpleNamespace(dataset=dataset), metric_args=SimpleNamespace(watch_metric=WATCH, watch_mode="max"))


def test_the_threshold_is_cut_from_the_larger_term_and_the_dataset_margin_and_the_bar_reads_it():
    for dataset in ("cifar10", "imagenette"):
        trainer = bare_trainer(dataset)
        threshold = trainer.success_threshold(trainer.config)
        assert threshold == 0.3 + MARGIN[dataset]
        assert trainer.bar_is_met(trainer.config, threshold + 1e-6, threshold) is True      # M32: another dataset's margin moves this
        assert trainer.bar_is_met(trainer.config, threshold, threshold) is False
    with pytest.raises(RuntimeError, match="threshold"):
        bare_trainer().bar_is_met(bare_trainer().config, 0.9, 0.5)                          # a threshold that is not the verdict's is refused
    with pytest.raises(RuntimeError, match="evaluate_and_report"):
        bare_trainer(_baseline=None).success_threshold(None)


def test_the_task_validates_on_nothing_and_a_run_has_no_validation_steps():
    from types import SimpleNamespace

    trainer = bare_trainer(_warmup_epochs=1.0)
    assert trainer.validation_stream(None, object()) is None, "a validation stream would run the loss, which advances the centre (M35)"
    config = SimpleNamespace(epochs=60, debug_args=SimpleNamespace(smoke=False))
    assert trainer.fit_kwargs(config, types_namespace_data(28)) == {"steps_per_epoch": 28}


def types_namespace_data(steps: int) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(steps_per_epoch=steps)


def test_success_value_is_the_nanmax_of_the_window_not_its_last_epoch():
    trainer = bare_trainer()                                                                  # M33: reading the last epoch gives 0.31
    assert trainer.success_value({WATCH: [0.2, 0.5, float("nan"), 0.31]}) == 0.5
    assert trainer.success_value({WATCH: [0.7, 0.5, 0.31]}) == 0.7


# ---------------------------------------------------------------------
# The hook table and the module
# ---------------------------------------------------------------------

#: The optional hooks the task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs", "validation_stream", "resume_stream", "bar_is_met", "run_analysis", "summary_fields", "recipe_notices", "used_arguments",
    "check_model_args", "epoch_line_metrics", "unjudged_note", "task_notes", "task_record", "init_identity", "_callbacks",
)


def test_the_class_names_its_task_and_its_metric_and_a_resume_is_not_exact():
    assert DinoTrainer.task == "dino" and issubclass(DinoTrainer, TaskTrainer) and DinoTrainer.success_metric == WATCH
    assert DinoTrainer.resume_stream_exact is False and "fresh stream" in DinoTrainer.resume_order


@pytest.mark.parametrize("hook", [*REQUIRED_HOOKS, *OPTIONAL_HOOKS])
def test_every_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(DinoTrainer, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_adds_no_train_step_and_no_stage():
    own = {name for name in vars(DinoTrainer) if not name.startswith("__")}
    assert not own & {"train_step", "test_step", "run", "_resolve", "_summary_head", "_notes", "_refuse_before_the_claim", "extra_callbacks"}
    assert not hasattr(DinoTrainer, "train_step")


def test_the_probe_keys_the_monitor_check_names_are_the_probes_own(tmp_path):
    from train.common.dino_run import build_probe

    class Data:  # the probe reads only these two names at construction
        probe_bank_ds = probe_query_ds = None
        bank_batches = query_batches = 1

    assert PROBE_KEYS == build_probe(Data(), object()).log_keys


def test_the_task_module_imports_no_tensorflow_and_the_help_lists_the_keys(tmp_path):
    code = (
        "import sys; import train.api.tasks.dino; "
        "heavy = sorted(m for m in ('tensorflow', 'keras', 'dl_techniques.models', 'train.common') if m in sys.modules); print('HEAVY', heavy)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0 and "HEAVY []" in result.stdout, result.stderr[-1500:] + result.stdout
    helped = subprocess.run([sys.executable, "-m", "train", "dino", "help"], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})
    assert helped.returncode == 0, helped.stderr[-1500:]
    for key in ("data.dataset", "data.patch_size", "data.n_local_crops", "transform_args.image_size", "metric_args.watch_metric", "model_args", "init"):
        assert key in helped.stdout, key
    assert "tensorflow" not in helped.stderr.lower().split("warning")[0]


def test_the_task_is_the_tenth_row_of_the_table_and_the_registry_counts_say_so(world):
    assert [spec.name for spec in BUILTIN_TASKS][-1] == "dino"
    assert registry.TASKS["dino"].config_class.endswith(":DinoConfig") and "dino_v1" in registry.MODELS
    assert len(registry.TASKS) == len(BUILTIN_TASKS) == 10


# ---------------------------------------------------------------------
# The refusals before the claim
# ---------------------------------------------------------------------


def run_cli(out: Path, **changes: Any) -> int:
    return cli.main(["dino", *flatten_request(task_request(out, **changes))])


@pytest.mark.parametrize(
    "changes, words",
    [
        ({"optim_args.schedule": "constant"}, "ReduceLROnPlateau"),
        ({"checkpoint_args.backup_every_steps": 5}, "centre"),
        ({"data.patch_size": 12}, "patch_size"),
        ({"data.patch_size": 32}, "tokens"),
        ({"precision": "mixed_float16"}, "float32"),
        ({"model_args": {"dino_out_dim": 64, "student_temp": 0.2}}, "student_temp"),
        ({"model_args": {"dino_out_dim": 64, "not_a_key": 1}}, "not_a_key"),
        ({"metric_args.watch_metric": "val_loss"}, "watch_metric"),
        ({"analysis_args.conformal_alpha": 0.1}, "conformal"),
        ({"model": "vit/vit_tiny"}, "vit"),
        ({"model": "dino_v1/base"}, "base"),
    ],
)
def test_a_request_the_task_cannot_honour_exits_2_and_leaves_out_unclaimed(world, tmp_path, capfd, changes, words):
    out = tmp_path / "refused"
    assert run_cli(out, **changes) == 2
    assert not out.exists(), "a refusal is made BEFORE the run directory is claimed"
    assert words.lower() in (capfd.readouterr().err + "").lower()


def test_init_naming_any_run_is_refused_before_the_claim_with_the_reason(world, tmp_path, capfd):
    source = tmp_path / "source_run"
    source.mkdir()
    out = tmp_path / "refused"
    assert run_cli(out, init=str(source)) == 2 and not out.exists()                          # M37: a dino target accepted
    error = capfd.readouterr().err
    assert "student and a teacher" in error and "evaluate" in error and "no `init` target" in error


def test_a_family_of_another_task_is_refused_under_dino_and_dino_v1_under_every_other_task(world, tmp_path):
    assert registry.MODELS["dino_v1"].tasks == ("dino",)
    for task in sorted(registry.TASKS):
        if task != "dino":
            with pytest.raises(Exception, match="dino_v1"):
                registry.parse_model_name("dino_v1/tiny", task=task)


# ---------------------------------------------------------------------
# Real tiny runs
# ---------------------------------------------------------------------


class Spy:
    """What happened to ``keras.Model.evaluate``, ``test_on_batch`` and ``predict`` on a ``DINOTrainingModel``, in order, with ``train_end`` between."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def install(self, patch: pytest.MonkeyPatch) -> None:
        for name in ("evaluate", "test_on_batch", "predict"):
            original = getattr(keras.Model, name)

            def wrapper(model, *args, _name=name, _original=original, **kwargs):
                if type(model).__name__ == "DINOTrainingModel":
                    self.events.append(_name)
                return _original(model, *args, **kwargs)

            patch.setattr(keras.Model, name, wrapper)


@pytest.fixture(scope="module")
def runs(world, cifar, tmp_path_factory):
    """Real tiny runs: bar met (the margin patched to the toy's reach), the real margin (a miss, exit 1), a smoke run."""
    from dl_techniques.losses.dino_loss import DINOLoss
    from train.api.tasks import dino as task

    root = tmp_path_factory.mktemp("dino_runs")
    done: Dict[str, Any] = {"root": root, "spy": Spy(), "order": [], "epoch_counts": [], "centres": {}}

    @keras.saving.register_keras_serializable(package="dino_task_tests")
    class CountingLoss(DINOLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.executions = keras.Variable(0, dtype="int32", trainable=False, name="executions")

        def call(self, y_true, y_pred):
            self.executions.assign_add(1)
            return super().call(y_true, y_pred)

    class Recorder(keras.callbacks.Callback):
        def __init__(self, trainer):
            super().__init__()
            self.trainer = trainer

        def on_epoch_end(self, epoch, logs=None):
            done["epoch_counts"].append(int(self.trainer._loss.executions))

        def on_train_end(self, logs=None):
            done["spy"].events.append("train_end")
            done["centres"]["train_end"] = keras.ops.convert_to_numpy(self.trainer._loss.center).copy()
            done["trainer"] = self.trainer

    original_callbacks = DinoTrainer._callbacks

    def callbacks(self, *arguments, **keywords):
        epoch_line, built = original_callbacks(self, *arguments, **keywords)
        done["order"] = [type(callback).__name__ for callback in built]
        done.setdefault("ema", next(callback for callback in built if type(callback).__name__ == "CompiledTeacherEMA"))   # the first run's: "met"
        built.append(Recorder(self))
        return epoch_line, built

    with pytest.MonkeyPatch.context() as patch:
        shallow_backbone(patch)
        done["spy"].install(patch)
        patch.setattr("dl_techniques.losses.dino_loss.DINOLoss", CountingLoss)
        patch.setattr(DinoTrainer, "_callbacks", callbacks)
        patch.setitem(task.MARGIN, "cifar10", -0.5)            # the margin patched to a value the toy reaches: ONLY in this run
        done["met_code"] = cli.main(["dino", *flatten_request(task_request(root / "met", **{"analysis_args.model_analysis": True}))])
        done["centres"]["end"] = keras.ops.convert_to_numpy(done["trainer"]._loss.center).copy()
    with pytest.MonkeyPatch.context() as patch:
        shallow_backbone(patch)
        done["miss_code"] = cli.main(["dino", *flatten_request(task_request(root / "miss"))])   # the real margin, 0.085
        done["smoke_code"] = cli.main(["dino", *flatten_request(task_request(root / "smoke", **{"debug_args.smoke": True, "epochs": "auto"})).copy()])
    return done


def summary_of(runs, name):
    return read_json(runs["root"] / name / "results_summary.json")


def test_a_tiny_run_through_the_command_line_meets_both_parts_and_exits_zero(runs):
    summary = summary_of(runs, "met")
    assert runs["met_code"] == 0 and summary["success"] is True and summary["status"] == "ok"
    bar = summary["bar"]
    assert bar["part1"]["met"] is True and bar["part2"]["met"] is True and bar["met"] is True and bar["applied"] is True
    assert bar["part2"]["epochs_checked"] == TASK_EPOCHS and bar["part2"]["first_offending_epoch"] is None
    assert "both parts met" in bar["verdict"]
    assert summary["success_metric"] == WATCH and summary["success_threshold"] == pytest.approx(
        max(summary["baseline"][WATCH], summary["epoch_zero"][WATCH]) - 0.5)
    for key in ("baseline", "epoch_zero", "margin", "best", "last3_mean", "collapse_max", "selection"):
        assert key in summary, key
    assert summary["margin"]["value"] == MARGIN["cifar10"] or summary["margin"]["value"] == -0.5
    assert summary["margin"]["control_sd"] == CONTROL_SD["cifar10"] and summary["margin"]["null_p99"] == NULL_P99["cifar10"]
    assert summary["collapse_max"] == 0.0 and summary["best"] == max(csv_column(runs["root"] / "met", WATCH))
    assert any("both are required" in note for note in summary["notes"])


def test_the_real_margin_is_not_reachable_by_a_toy_and_the_run_misses_with_exit_1_and_the_files_kept(runs):
    summary = summary_of(runs, "miss")
    assert runs["miss_code"] == 1 and summary["success"] is False and summary["status"] == "ok"
    assert summary["bar"]["part1"]["met"] is False and "part 1 missed" in summary["bar"]["verdict"]
    assert summary["bar"]["part1"]["success_threshold"] == pytest.approx(max(summary["baseline"][WATCH], summary["epoch_zero"][WATCH]) + 0.085)
    assert (runs["root"] / "miss" / "best_model.keras").is_file() and (runs["root"] / "miss" / "final_model.keras").is_file()


def test_a_smoke_run_applies_no_bar_and_still_reports_the_numbers(runs):
    assert runs["smoke_code"] == 0, "a smoke run keeps the recipe's one-epoch warmup: both warmup refusals (D-100, D-109) exempt it"
    summary = summary_of(runs, "smoke")
    assert summary["success"] is None and summary["success_threshold"] is None
    assert summary["epochs_run"] == 1 and summary["success_value"] == summary["best"]
    assert summary["bar"]["applied"] is False, "the numbers are reported and the summary says no bar was applied"


def test_the_csv_has_every_probe_key_each_epoch_and_no_val_column_and_the_probe_sits_before_the_csv_logger(runs):
    with open(runs["root"] / "met" / "training_log.csv") as handle:
        header = handle.readline().strip().split(",")
    assert not [name for name in header if name.startswith("val_")], header
    assert set(PROBE_KEYS) <= set(header) and "loss" in header
    for key in PROBE_KEYS:
        assert len(csv_column(runs["root"] / "met", key)) == TASK_EPOCHS and all(math.isfinite(v) for v in csv_column(runs["root"] / "met", key))
    order = runs["order"]
    assert order.index("KNNEvalCallback") < order.index("EpochCSVLogger") and order.index("KNNEvalCallback") < order.index("EarlyStopping")  # M39
    assert order.index("KNNEvalCallback") < order.index("AtomicModelCheckpoint")
    assert order.index("CentreFile") < order.index("BackupAndRestore") < order.index("CompiledTeacherEMA")
    assert "LambdaCallback" in order and "FinishKilledEpoch" not in order


def test_the_centre_advances_exactly_steps_per_epoch_times_per_epoch_and_no_evaluate_touches_the_dino_model(runs):
    summary = summary_of(runs, "met")
    steps = summary["steps_per_epoch"]
    assert steps == 28 and runs["epoch_counts"] == [steps * (e + 1) for e in range(TASK_EPOCHS)]                      # M35: validation adds batches
    events = runs["spy"].events
    assert "train_end" in events, "the spy saw the run end: a spy that records nothing would pass every line below"
    assert "evaluate" not in events and "test_on_batch" not in events, events                                         # M40: the test split through evaluate
    after = events[events.index("train_end"):]
    assert "predict" in after, "the positive control: the final load check calls predict after training, and the spy saw it"
    assert "predict" not in events[:events.index("train_end")], "no predict while the loss is running"
    assert set(after) <= {"train_end", "predict"}
    centre = runs["centres"]
    assert np.abs(centre["train_end"]).max() > 0.0 and np.array_equal(centre["train_end"], centre["end"]), "the reporting moved the centre"
    saved = np.load(runs["root"] / "met" / "checkpoints" / CENTRE_FILE)
    assert saved.shape == (1, OUT_DIM) and np.array_equal(saved, centre["train_end"]), "the centre file is the last epoch end's"


def test_the_run_moved_the_teacher_after_the_one_epoch_warmup_and_not_before(runs):
    """The fresh run's EMA count: ``steps * epochs`` batches, the first ``ema_warmup_epochs * steps`` of them frozen (the recipe's warmup is 1 epoch)."""
    ema = runs["ema"]
    steps = summary_of(runs, "met")["steps_per_epoch"]
    assert ema.warmup_steps == steps and ema.updates == steps * TASK_EPOCHS - steps == steps, (ema.warmup_steps, ema.updates, steps)


def test_the_files_the_run_leaves_and_the_best_checkpoint_is_the_epoch_the_summary_names(runs):
    run = runs["root"] / "met"
    summary = summary_of(runs, "met")
    for name in ("best_model.keras", "final_model.keras", "task.json", "config.json", "training_log.csv", "results_summary.json"):
        assert (run / name).is_file(), name
    values = csv_column(run, WATCH)
    assert summary["selection"]["best_epoch"] == int(np.argmax(values)) + 1 and summary["best"] == max(values)
    assert summary["epochs_run"] == TASK_EPOCHS and summary["task"] == "dino" and "val_loss" not in summary.get("best_val_metrics", {})
    assert summary["best_checkpoint_max_abs_diff"] is not None and summary["best_checkpoint_max_abs_diff"] < 1e-4 and summary["model_loading_validated"] is True
    record = read_json(run / "task.json")
    assert record["task"] == "dino" and record["output"] == "cls_embeddings" and record["n_local_crops"] == N_LOCAL
    assert record["dino_out_dim"] == OUT_DIM and record["embedding_dim"] == 192 and record["patch_size"] == PATCH and record["input_size"] == [IMAGE, IMAGE]
    assert summary["test_metrics_best"][WATCH] == pytest.approx(summary["test_metrics_final"][WATCH], abs=0.2) and len(summary["test_metrics_best"]) == len(PROBE_KEYS)
    assert summary["analyzer"]["status"] in ("ok", "partial"), summary["analyzer"]


def test_the_best_checkpoint_reloads_and_its_probe_value_is_the_in_memory_best_within_one_query_vote(runs):
    from dl_techniques.losses.dino_loss import DINOLoss
    from train.common import dino_data, dino_run

    run = runs["root"] / "met"
    summary = summary_of(runs, "met")
    # No `set_seeds` here: the bank is a function of `seed` alone (D-095), so it is the run's bank whatever global seed the process was left with.
    data = dino_data.load_dino_batches(
        dataset="cifar10", image_size=IMAGE, n_local_crops=N_LOCAL, batch_size=TASK_BATCH, seed=42, stream_seed=None, validation_split=None,
        max_samples=TASK_MAX_SAMPLES, knn_bank_batches=64, knn_query_batches=32)
    model = keras.saving.load_model(run / "best_model.keras", compile=False)
    scored = dino_run.knn_of(model, data.select_ds, data=data, loss=DINOLoss(out_dim=OUT_DIM, teacher_temp=0.04))
    assert abs(scored[WATCH] - summary["best"]) <= 1.0 / data.n_select + 1e-9, (scored[WATCH], summary["best"])


# ---------------------------------------------------------------------
# The controls: real probes of real models, judged by the real hooks
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def controls(world, cifar):
    """Probe a depth-2 model on the real rows under the four controls; ``numbers[name] = (probe metrics, constant baseline, epoch zero)``."""
    import dataclasses

    from train.common import dino_data, dino_run

    from ._dino_world import make_loss, make_model

    # No `set_seeds` here either (D-095): the bank is a function of `seed` alone, so the rows are the run's whatever global seed was left behind.
    data = dino_data.load_dino_batches(
        dataset="cifar10", image_size=IMAGE, n_local_crops=N_LOCAL, batch_size=TASK_BATCH, seed=42, stream_seed=None, validation_split=None,
        max_samples=TASK_MAX_SAMPLES, knn_bank_batches=64, knn_query_batches=32)

    def probe(model, loss, bank=None):
        used = data if bank is None else dataclasses.replace(data, probe_bank_ds=bank)
        callback = dino_run.build_probe(used, loss)
        callback.set_model(model)
        return {k: float(v) for k, v in callback.evaluate_split(used.probe_query_ds, used.query_batches).items()}

    out: Dict[str, Any] = {"data": data}
    model, loss = make_model(), make_loss(teacher_temp=0.04)
    zero = dino_run.epoch_zero(model, data, loss)
    out["baseline"], out["zero"] = zero.baseline[WATCH], zero.probe[WATCH]
    out["untrained"] = zero.probe
    out["shuffled"] = probe(model, loss, bank=data.probe_bank_ds.map(lambda x, y: (x, (y + 1) % 10)))
    spiked = make_loss(teacher_temp=0.04)
    # The centre is SUBTRACTED from the teacher's logits: minus a million at one output makes it the only one that survives, a one-hot teacher.
    spiked.center.assign(-np.eye(1, OUT_DIM, 3, dtype=np.float32) * 1e6)
    out["collapsed"] = probe(model, spiked, bank=None)
    final_norm = model.student.get_layer("norm")   # the CLS token the probe reads is this layer's output: gamma and beta zero make every feature 0
    for weight in final_norm.weights:
        weight.assign(np.zeros(weight.shape, np.float32))
    out["zero_features"] = probe(model, loss, bank=None)
    return out


def verdict_of(controls, metrics, *, margin=MARGIN["cifar10"]):
    return judge(best=metrics[WATCH], baseline=controls["baseline"], epoch_zero=controls["zero"], margin=margin,
                 collapse_flags=[metrics["dino_collapse_flag"]], centred_flags=[metrics["dino_teacher_entropy_is_centered"]])


def test_an_untrained_model_scores_its_own_epoch_zero_and_misses_part_one(controls):
    verdict = verdict_of(controls, controls["untrained"])
    assert controls["untrained"][WATCH] == controls["zero"] and not verdict.part1_met and verdict.part2_met
    assert controls["zero"] > controls["baseline"] + 0.05, "the control: random features carry information, a constant carries none"


def test_a_bank_with_the_wrong_labels_misses_part_one(controls):
    verdict = verdict_of(controls, controls["shuffled"])
    assert controls["shuffled"][WATCH] < controls["zero"] - 0.05, "the same features, scored against another bank's labels, lose their signal"
    assert not verdict.part1_met and "part 1 missed" in verdict.text


def test_a_model_of_exactly_zero_features_scores_chance_and_the_flag_cannot_see_it(controls):
    metrics = controls["zero_features"]
    verdict = verdict_of(controls, metrics)
    assert metrics[WATCH] <= controls["baseline"] + 0.03, "chance"
    assert metrics["dino_collapse_flag"] == 0.0, "the flag cannot see exactly-zero features: only the k-NN part can"
    assert not verdict.part1_met and verdict.part2_met


def test_a_teacher_collapsed_by_the_loss_centre_fires_the_flag_with_the_knn_number_unchanged(controls):
    metrics = controls["collapsed"]
    assert metrics[WATCH] == controls["untrained"][WATCH], "the k-NN reads the student features, which a centre does not touch"
    assert metrics["dino_collapse_flag"] == 1.0
    patched = verdict_of(controls, metrics, margin=-0.5)                                       # part 1 given a margin the numbers clear
    assert patched.part1_met and not patched.part2_met and "part 2 missed" in patched.text and "part 1" not in patched.text


def test_the_bar_needs_both_parts_over_the_real_numbers(controls):
    good = controls["untrained"]
    assert verdict_of(controls, good, margin=-0.5).met, "the control: both parts can be met by these numbers"
    assert not verdict_of(controls, good).met and not verdict_of(controls, controls["collapsed"], margin=-0.5).met
