"""The ``pretrain`` task: the hook table, the bar, the epoch-zero term, the refusals before the claim and the BEiT tokenizer.

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

Second part (step 11): the BEiT tokenizer. A ``beit`` run needs the path of a stage-0 ``.keras`` (refused before the claim when it
is missing, does not exist or is not a ``.keras``); the grid proof, the file that is not a ``VQVAERotationTrick`` and the usage
floor are found at build time (the run directory is claimed, and the message says so); a healthy tokenizer's usage reaches
``results_summary.json`` and the BEiT baseline is the in-sample unigram ``val_loss``. The tokenizers are tiny
``VQVAERotationTrick`` files (32 px, 16 codes) saved under ``tmp_path``. The usage floor (D-007, D-019) is 64 effective codes, above
what a 16-code toy can reach, so the arms that need a tokenizer to PASS lower it by patching the module constant.

Third part (step 12): a 2-epoch run per family through ``train_pretrain`` and through ``cli.main(["pretrain", ...])`` on the fake
CIFAR-10 (CPU, ``tmp_path``). It pins the run-directory listing (derived by running, no ``task.json``), the equal ``config.json``
bytes, the summary's metric, baseline, epoch-zero loss, threshold and verdict, ``steps_per_epoch == n_train // batch`` as a fit
spy sees it (a callback stops a fit that has no epoch length, so a missing ``steps_per_epoch`` is reported, not hung), the best
checkpoint's reload gap of exactly 0.0, and the ``task.json`` refusals of ``predict`` and ``export``.

The runs are real ``energy_transformer/et_tiny`` runs (``model_args.num_steps=2``) on the fake CIFAR-10 of
``tests/classification_fakes.py``: 54 fit rows, batch 4, so 13 steps of an epoch. Tests write only under ``tmp_path``.
"""

import csv
import dataclasses
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick
from tests.classification_fakes import use_identity_cifar10
from train.api import recipes, registry, resume_decisions
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import register_builtin_tasks
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import pretrain_data, pretrain_run
from train.common.pretrain_run import Baseline
from train.common.resume_state import FIT_STATE_NAME, update_fit_state

from train.api.tasks.image_classification import ImageClassificationTrainer
from train.api.tasks.image_denoising import ImageDenoisingTrainer
from train.api.tasks.image_segmentation import ImageSegmentationTrainer
from train.api.tasks.pretrain import PretrainConfig, PretrainTrainer, selection_block

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


@pytest.mark.parametrize("name", ["whole", "crashed"])
def test_the_summary_names_the_best_epoch_and_whether_it_is_the_last_one_the_csv_holds(runs, name):
    # The oracle is the CSV: the 1-based epoch of the lowest val_loss, and whether that is the last row (the resumed run's
    # rows are the first process's plus its own, so the count is the real epochs run).
    column = csv_column(runs[name], "val_loss")
    best = int(np.argmin(column)) + 1
    selection = read_json(runs[name] / "results_summary.json")["selection"]
    assert selection == {"start_epoch": 0, "window_epochs": len(column), "best_epoch": best, "final_is_best": best == len(column)}


@pytest.mark.parametrize(
    "best, epochs_run, final_is_best",
    [(3, 3, True), (2, 3, False), (1, 3, False), (1, 1, True), (2, 2, True), (1, 2, False)],
    ids=["last of three", "middle of three", "first of three", "only epoch", "last of two", "first of two"],
)
def test_final_is_best_exactly_when_the_selected_epoch_is_the_last_one_run(best, epochs_run, final_is_best):
    # `best_epoch` is 1-based, so `best == epochs_run` (never `epochs_run - 1`) is the last epoch.
    block = selection_block(0, epochs_run, best)
    assert block == {"start_epoch": 0, "window_epochs": epochs_run, "best_epoch": best, "final_is_best": final_is_best}
    assert selection_block(1, epochs_run, best)["window_epochs"] == epochs_run - 1, "the window starts at the 0-based epoch"


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
        ({"metric_args": {"watch_metric": "loss"}}, "metric_args.watch_metric"),
        ({"metric_args": {"watch_mode": "max"}}, "metric_args.watch_mode"),
    ],
    ids=["tokenizer-on-et", "vocab-size", "scale", "no-clip", "beit-mask-key", "watch-the-train-loss", "watch-mode-max"],
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


def test_the_command_the_model_help_names_lists_the_variants(world, capsys):
    from train.api import cli

    (field,) = [f for f in dataclasses.fields(PretrainConfig) if f.name == "model"]
    written = re.search(r"python -m train (list_models \S+)", field.metadata["help"])
    assert written, "the help of `model` names the command that lists the variants"
    assert cli.main(written.group(1).split()) == 0, written.group(1)
    assert "beit/beit_tiny" in capsys.readouterr().out
    # The docstring of the function says the same command.
    from train.api.tasks import pretrain

    assert f"python -m train {written.group(1)}`` lists them" in pretrain.__doc__


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


# ---------------------------------------------------------------------
# The BEiT tokenizer (second part): the required path, the grid proof, the usage report and the floor
# ---------------------------------------------------------------------

TOKENIZER_KEY = "tokenizer_checkpoint"
#: A one-layer BEiT, so that a whole run on the fake cifar10 costs seconds.
BEIT_SMALL = {"num_layers": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64}
#: Below the toy's own perplexity (at most 16 codes), so a run with a healthy toy tokenizer passes the floor.
LOW_FLOOR = 1.5


def make_tokenizer(path: Path, *, image_size: int = 32, downsample: int = 4, collapse: bool = False) -> str:
    """Save a 16-code ``VQVAERotationTrick`` as a ``.keras`` file; ``collapse`` puts every codebook row but the first far away."""
    keras.utils.set_random_seed(0)
    model = VQVAERotationTrick(
        num_embeddings=16, embedding_dim=4, input_shape=(image_size, image_size, 3),
        downsample_factor=downsample, hidden_channels=8, num_res_blocks=1,
    )
    model(np.zeros((1, image_size, image_size, 3), "float32"))
    if collapse:
        book = np.full((1, 16, 4), 1e4, "float32")
        book[:, 0] = 0.0
        model.quantizer.embeddings.assign(book)
    model.save(str(path))
    return str(path)


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    """``name -> path`` of the tokenizer files: a healthy toy, a collapsed one, two of the wrong geometry, a file that is no tokenizer."""
    root = tmp_path_factory.mktemp("tokenizers")
    not_one = root / "classifier.keras"
    keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(2)]).save(str(not_one))
    return {
        "healthy": make_tokenizer(root / "healthy.keras"),
        "collapsed": make_tokenizer(root / "collapsed.keras", collapse=True),
        "other_size": make_tokenizer(root / "other_size.keras", image_size=64, downsample=8),
        "coarse_grid": make_tokenizer(root / "coarse_grid.keras", downsample=8),
        "not_a_tokenizer": str(not_one),
    }


def beit_config(out: Path, path: str, **changes: Any) -> PretrainConfig:
    return make_config(out, model=BEIT_MODEL, model_args={TOKENIZER_KEY: path, **BEIT_SMALL}, **changes)


def run_beit(out: Path, path: str, **changes: Any):
    return PretrainTrainer(beit_config(out, path, **changes)).run()


def test_a_beit_run_without_a_tokenizer_names_the_script_that_makes_one(world, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        PretrainTrainer(make_config(out, model=BEIT_MODEL, model_args={})).run()
    assert "python -m train.beit.train_tokenizer" in str(caught.value)
    assert not out.exists()


def test_a_tokenizer_path_that_does_not_exist_is_refused_before_the_claim(world, tmp_path):
    out, missing = tmp_path / "run", tmp_path / "nowhere" / "tokenizer.keras"
    with pytest.raises(ConfigError) as caught:
        run_beit(out, str(missing))
    assert "model_args.tokenizer_checkpoint" in str(caught.value) and str(missing) in str(caught.value)
    assert "does not exist" in str(caught.value)
    assert not out.exists(), "the refusal came after the claim"


def test_a_tokenizer_path_that_is_not_a_keras_file_is_refused_before_the_claim(world, tmp_path):
    out, wrong = tmp_path / "run", tmp_path / "tokenizer.h5"
    wrong.write_bytes(b"")
    with pytest.raises(ConfigError) as caught:
        run_beit(out, str(wrong))
    assert ".keras" in str(caught.value) and str(wrong) in str(caught.value)
    assert not out.exists(), "the refusal came after the claim"


@pytest.mark.parametrize(
    "name, fragments",
    [
        # Built at 64 px: the encoder pins its input shape, so a 32 px image is refused by the grid proof's probe.
        ("other_size", ("other_size.keras", "(32, 32, 3)", "cannot encode")),
        # Built at 32 px with a down-sample of 8: a 4 x 4 code grid against the trainer's 8 x 8 patch grid.
        ("coarse_grid", ("coarse_grid.keras", "(4, 4)", "(8, 8)", "code-grid mismatch")),
    ],
)
def test_a_tokenizer_of_another_grid_is_refused_by_the_grid_proof_and_named(world, tokenizers, tmp_path, name, fragments):
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        run_beit(out, tokenizers[name])
    for fragment in fragments:
        assert fragment in str(caught.value), (fragment, str(caught.value))
    assert "was claimed" in str(caught.value) and out.exists(), "found at build time: the run directory is claimed and says so"


def test_a_keras_file_that_is_not_a_tokenizer_is_refused_by_name(world, tokenizers, tmp_path):
    with pytest.raises(ConfigError) as caught:
        run_beit(tmp_path / "run", tokenizers["not_a_tokenizer"])
    assert isinstance(caught.value.__cause__, TypeError), "the loader's TypeError, kept as the cause"
    assert "classifier.keras" in str(caught.value) and "not a VQVAERotationTrick" in str(caught.value)


def test_a_collapsed_tokenizer_is_refused_below_the_floor_with_its_measured_perplexity(world, tokenizers, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        run_beit(out, tokenizers["collapsed"])
    message = str(caught.value)
    assert "perplexity 1.00" in message and "1 distinct" in message and "floor" in message and "64" in message
    assert "was claimed" in message and out.exists(), "found at build time: the run directory is claimed and says so"
    assert not (out / "results_summary.json").exists()


def test_the_floor_is_the_one_measured_and_sits_between_the_two_tokenizers_it_was_measured_on():
    # D-019, scratch_measure item (g): over the masked positions of the cifar10 selection split the 1-epoch tokenizer had
    # perplexity 24.7 and the 5-epoch one 354.8; the floor separates them. A toy of 16 codes cannot reach it.
    assert 24.7 < pretrain_run.MIN_TOKENIZER_PERPLEXITY < 354.8
    assert pretrain_run.MIN_TOKENIZER_PERPLEXITY == 64.0


@pytest.mark.parametrize(
    "perplexity, refused",
    [(64.0, False), (float(np.nextafter(64.0, 0.0)), True), (63.99, True), (64.01, False)],
    ids=["exactly the floor", "one ulp below", "just below", "just above"],
)
def test_the_floor_is_strict_so_a_perplexity_of_exactly_the_floor_passes(monkeypatch, perplexity, refused):
    # "Below the floor" is refused and the floor itself is not. The usage is handed in as a record: a perplexity of exactly
    # 64.0 is not reachable through counts (64 codes used equally give 63.99999999999998 in float64).
    usage = pretrain_data.TokenizerUsage(
        n_codes=5000, distinct=200, perplexity=perplexity, top1_share=0.02, entropy=math.log(perplexity))
    monkeypatch.setattr(pretrain_run, "tokenizer_usage", lambda ids: usage)
    assert pretrain_run.MIN_TOKENIZER_PERPLEXITY == 64.0
    if refused:
        with pytest.raises(ValueError, match="below the floor of 64"):
            pretrain_run.checked_tokenizer_usage(np.zeros(3, np.int64), path="t.keras")
    else:
        assert pretrain_run.checked_tokenizer_usage(np.zeros(3, np.int64), path="t.keras") is usage


def test_a_refusal_on_few_positions_says_the_split_size_limits_the_estimate_and_names_what_it_measured():
    # The perplexity of a few positions is limited by how many there are (a tokenizer measured at 354 over 5,000 images
    # reads 59 over 40), so the sentence names the positions and the distinct codes it counted, and does not blame the
    # tokenizer alone.
    ids = np.arange(96) % 60                       # 96 positions, 60 distinct codes: perplexity 57
    with pytest.raises(ValueError) as caught:
        pretrain_run.checked_tokenizer_usage(ids, path="t.keras")
    message = str(caught.value)
    assert "60 distinct codes over the 96 masked positions" in message and "perplexity 57" in message
    assert "limited by the split size" in message and "data.max_samples" in message
    assert "train the tokenizer longer" in message, "the other cause stays named"


def test_a_toy_that_uses_every_code_it_has_is_still_below_the_real_floor(world, tokenizers, tmp_path):
    with pytest.raises(ConfigError, match=r"below the floor of 64"):
        run_beit(tmp_path / "run", tokenizers["healthy"])


def usage_oracle(data) -> Dict[str, Any]:
    """The usage of the masked positions of one pass of the selection stream, and the in-sample unigram ``val_loss`` (float64)."""
    elements = list(data.select_ds.as_numpy_iterator())
    ids = np.concatenate([e[1] for e in elements])
    masks = np.concatenate([e[0][1] for e in elements]).astype(bool)
    counts = np.bincount(ids[masks], minlength=16).astype(np.float64)
    share = counts[counts > 0] / counts.sum()
    entropy = float(-(share * np.log(share)).sum())
    log_p = np.log(np.maximum(counts / counts.sum(), 1e-30))
    unigram = float(np.mean([(-log_p[row[mask]]).mean() for row, mask in zip(ids, masks)]))
    return {"n_codes": int(masks.sum()), "distinct": int(len(share)), "top1": float(share.max()),
            "perplexity": math.exp(entropy), "unigram": unigram}


@pytest.fixture(scope="module")
def healthy_run(world, tokenizers, tmp_path_factory):
    """A whole one-epoch BEiT run on the healthy toy tokenizer (floor lowered), its summary and the oracle of its data."""
    root = tmp_path_factory.mktemp("beit_run")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        out = root / "run"
        run_beit(out, tokenizers["healthy"], epochs=1)
        trainer, resolved, context = prepared(
            root / "again", model=BEIT_MODEL, model_args={TOKENIZER_KEY: tokenizers["healthy"], **BEIT_SMALL})
        oracle = usage_oracle(trainer.build_data(resolved, context))
    return {"summary": read_json(out / "results_summary.json"), "oracle": oracle, "path": tokenizers["healthy"]}


def test_a_healthy_tokenizers_usage_reaches_the_summary(healthy_run):
    block, oracle = healthy_run["summary"]["tokenizer"], healthy_run["oracle"]
    assert oracle["distinct"] > 1, "the control: a toy that uses one code would make every number below trivial"
    assert block["checkpoint"] == healthy_run["path"] and block["num_embeddings"] == 16
    assert (block["n_codes"], block["distinct"]) == (oracle["n_codes"], oracle["distinct"])
    assert block["perplexity"] == pytest.approx(oracle["perplexity"], rel=1e-9)
    assert block["top1_share"] == pytest.approx(oracle["top1"], rel=1e-9)
    assert block["entropy"] == pytest.approx(math.log(oracle["perplexity"]), rel=1e-9)
    assert block["floor"] == LOW_FLOOR


def test_the_beit_baseline_is_the_in_sample_unigram_val_loss_and_not_the_uniform_one(healthy_run):
    baseline, oracle = healthy_run["summary"]["baseline"], healthy_run["oracle"]
    assert baseline["kind"] == "unigram_codes"
    assert abs(oracle["unigram"] - math.log(16)) > 0.01, "the control: the uniform bar must differ from the unigram one"
    assert baseline["val_loss"] == pytest.approx(oracle["unigram"], rel=1e-5)


def test_an_energy_transformer_run_reports_no_tokenizer(runs):
    assert read_json(runs["whole"] / "results_summary.json")["tokenizer"] is None


def test_the_select_stream_is_read_once_for_the_usage_and_the_baseline(world, tokenizers, tmp_path, monkeypatch):
    calls = []
    real = pretrain_data.masked_code_ids

    def counting(ds):
        calls.append(1)
        return real(ds)

    monkeypatch.setattr(pretrain_data, "masked_code_ids", counting)
    monkeypatch.setattr(pretrain_run, "masked_code_ids", counting)
    monkeypatch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
    trainer, resolved, context = prepared(
        tmp_path / "run", model=BEIT_MODEL, model_args={TOKENIZER_KEY: tokenizers["healthy"], **BEIT_SMALL})
    trainer.build_data(resolved, context)
    assert len(calls) == 1, "each extra pass runs the tokenizer over the whole selection split again"


# ---------------------------------------------------------------------
# The resume note of the summary names the order the task's resume really trains on
# ---------------------------------------------------------------------

#: What the three shipped tasks have always written on a resume (the sentence of ``TaskTrainer._notes``, byte for byte).
DEFAULT_RESUME_NOTE = (
    "resumed after epoch 2: epochs 0..1 come from the killed segment's training_log.csv (its torn or surplus rows are in "
    "training_log.pre_resume_<k>.csv), the training data order of the resumed epochs replays the order of epoch 0, and "
    "`epoch_times` and `fit_wall_seconds` cover the resumed epochs only"
)


def resume_note(task_class, monkeypatch) -> str:
    """The resumed-run note of ``task_class`` from the real ``TaskTrainer._notes`` (the task's own lines are left out)."""
    trainer = object.__new__(task_class)
    monkeypatch.setattr(task_class, "task_notes", lambda *args: [])
    monkeypatch.setattr(resume_decisions, "training_complete_reason", lambda resume, config: None)
    config = SimpleNamespace(debug_args=SimpleNamespace(smoke=False))
    return trainer._notes(config, {}, 1.0, False, None, SimpleNamespace(start=2, batch=0))[0]


@pytest.mark.parametrize("task_class", [ImageClassificationTrainer, ImageSegmentationTrainer, ImageDenoisingTrainer])
def test_the_resume_note_of_a_task_that_says_nothing_is_what_it_always_was(task_class, monkeypatch):
    assert TaskTrainer.resume_order is None and task_class.resume_order is None
    assert resume_note(task_class, monkeypatch) == DEFAULT_RESUME_NOTE


def test_the_pretrain_resume_note_names_its_own_order_and_claims_no_replay(monkeypatch):
    note = resume_note(PretrainTrainer, monkeypatch)
    assert PretrainTrainer.resume_order and PretrainTrainer.resume_order in note
    assert "replays" not in note
    # One clause in both places: the help of `resume_interrupted` and the note of the summary.
    help_text = {f.name: f.metadata.get("help") for f in dataclasses.fields(PretrainConfig)}["resume_interrupted"]
    assert PretrainTrainer.resume_order in help_text


# ---------------------------------------------------------------------
# Third part (step 12): a stubbed end to end per family, the run directory, the refusals of predict and export
# ---------------------------------------------------------------------

FAMILIES = ("energy_transformer", "beit")
#: A fit that runs more batches than ``EPOCHS * STEPS`` has an epoch with no length (an infinite stream, no steps_per_epoch).
RUNAWAY_BATCHES = 4 * EPOCHS * STEPS


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
    """``key=value`` tokens as the command line takes them: groups are dotted, ``model_args`` is one JSON value."""
    tokens = []
    for key, value in asked.items():
        if isinstance(value, dict) and key != "model_args":
            tokens += [f"{key}.{inner}={str(v).lower() if isinstance(v, bool) else v}" for inner, v in value.items()]
        else:
            tokens.append(f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}")
    return tokens


@pytest.fixture(scope="module", params=FAMILIES)
def ended(request, world, tokenizers, tmp_path_factory):
    """One 2-epoch run per family through ``train_pretrain`` and the same request through ``cli.main``, in this process."""
    from train.api import cli
    from train.api.tasks.pretrain import train_pretrain

    family = request.param
    root = tmp_path_factory.mktemp("end_to_end_" + family)
    model_args = {"num_steps": 2} if family == "energy_transformer" else {TOKENIZER_KEY: tokenizers["healthy"], **BEIT_SMALL}
    asked = {
        "out": str(root / "run"), "model": ET_MODEL if family == "energy_transformer" else BEIT_MODEL,
        "data": {"dataset": "cifar10"}, "epochs": EPOCHS, "batch_size": BATCH, "model_args": model_args,
        "analysis_args": {"model_analysis": True, "figures": True},
    }
    fits, real_fit = [], keras.Model.fit

    def fit(self, *args, **keywords):
        spy = Spy()
        fits.append({"keywords": {k: keywords.get(k) for k in ("steps_per_epoch", "validation_steps")}, "spy": spy})
        return real_fit(self, *args, **{**keywords, "callbacks": [*keywords.get("callbacks", []), spy]})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(keras.Model, "fit", fit)
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        patch.chdir(root)
        result = train_pretrain(**asked)
        (root / "run").rename(root / "function")
        code = cli.main(["pretrain", *flatten_request(asked)])
        (root / "run").rename(root / "cli")
    return SimpleNamespace(family=family, root=root, asked=asked, result=result, code=code, fits=fits,
                           function=root / "function", cli=root / "cli", tokenizers=tokenizers)


RUN_FILES = {
    "best_model.keras", "config.json", "config_requested.json", "final_model.keras", "results_summary.json",
    "run.lock", "run.log", "training_history.json", "training_log.csv",
}  # and NO task.json (D-013)
RUN_DIRS = {"checkpoints", "model_analysis", "tensorboard", "visualizations"}
#: The contents of the directories whose names do not depend on the run's clock (tensorboard holds event files).
NESTED = {
    "checkpoints": {"backup", "fit_state.json"},
    "model_analysis": {"analysis_results.json", "spectral_funnel_diagram.png", "spectral_summary.png", "weight_learning_journey.png"},
    "visualizations": {"energy_transformer": {"energy_trace.csv", "training_dashboard.png"}, "beit": {"training_dashboard.png"}},
}


def names(path: Path) -> set:
    return {child.name for child in Path(path).iterdir()}


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_run_directory_holds_the_pinned_listing_and_no_task_record(ended, which):
    run_dir = getattr(ended, which)
    assert names(run_dir) == RUN_FILES | RUN_DIRS, "an added or a missing file (task.json included) is a change of the contract"
    assert names(run_dir / "checkpoints") == NESTED["checkpoints"]
    assert names(run_dir / "model_analysis") == NESTED["model_analysis"]
    assert names(run_dir / "visualizations") == NESTED["visualizations"][ended.family]
    assert names(run_dir / "tensorboard") == {"train", "validation"}


def test_the_function_and_the_cli_write_the_same_config_bytes_and_the_cli_exits_by_the_verdict(ended):
    for name in ("config.json", "config_requested.json"):
        assert (ended.function / name).read_bytes() == (ended.cli / name).read_bytes(), name
    resolved = read_json(ended.cli / "config.json")
    assert resolved["model"] == ended.asked["model"] and resolved["epochs"] == EPOCHS and resolved["batch_size"] == BATCH
    success = read_json(ended.cli / "results_summary.json")["success"]
    assert ended.code == (0 if success else 1) and ended.result.success is read_json(ended.function / "results_summary.json")["success"]


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_summary_holds_the_metric_the_baseline_the_epoch_zero_loss_the_threshold_and_the_verdict(ended, which):
    summary = read_json(getattr(ended, which) / "results_summary.json")
    kind = "zero_patches" if ended.family == "energy_transformer" else "unigram_codes"
    assert summary["task"] == "pretrain" and summary["status"] == "ok" and summary["success_metric"] == "val_loss"
    assert summary["baseline"]["kind"] == kind and summary["baseline"]["val_loss"] > 0
    epoch_zero = read_json(getattr(ended, which) / "checkpoints" / FIT_STATE_NAME)["initial"]["metrics"]["loss"]
    assert summary["epoch_zero"]["val_loss"] == epoch_zero > 0
    assert summary["success_threshold"] == min(summary["baseline"]["val_loss"], epoch_zero)
    column = csv_column(getattr(ended, which), "val_loss")
    assert len(column) == EPOCHS and summary["success_value"] == pytest.approx(min(column), rel=1e-6)
    assert summary["success"] is bool(np.float32(summary["success_value"]) < np.float32(summary["success_threshold"]))
    assert (summary["tokenizer"] is None) is (ended.family == "energy_transformer")


def test_an_epoch_is_n_train_over_batch_steps_and_fit_sees_both_lengths(ended):
    summary = read_json(ended.function / "results_summary.json")
    assert summary["steps_per_epoch"] == summary["n_train"] // BATCH == STEPS
    assert len(ended.fits) == 2, "one fit per run: the function and the command line"
    for fit in ended.fits:
        assert fit["keywords"] == {"steps_per_epoch": STEPS, "validation_steps": math.ceil(summary["n_val"] / BATCH)}
        # Observed, not read back: each epoch ran batches 0..STEPS-1 and a stream with no length would have run away.
        assert fit["spy"].last_batch == {epoch: STEPS - 1 for epoch in range(EPOCHS)}, "an epoch has the wrong length"


@pytest.mark.parametrize("which", ["function", "cli"])
def test_the_best_checkpoint_reloads_with_a_gap_of_exactly_zero(ended, which):
    summary = read_json(getattr(ended, which) / "results_summary.json")
    # The masks are fixed by (seed, split, row position): the reloaded file scores the in-memory weights bit for bit.
    assert summary["best_checkpoint_load_error"] is None and summary["best_checkpoint_max_abs_diff"] == 0.0
    assert summary["model_loading_validated"] is True and summary["model_loading_check"]["max_abs_diff"] == 0.0
    assert summary["test_metrics_best"] is not None


SENTENCE_OF_A_DIRECTORY = "task.json does not exist; a model is loaded from a run directory, or from a .keras file whose directory holds task.json"
SENTENCE_OF_A_FILE = (
    "has no record beside it; a .keras file is read with the first of best_model.keras.task.json, best_model.task.json, "
    "task.json that exists in"
)


@pytest.mark.parametrize("command", ["predict", "export"])
@pytest.mark.parametrize("form", ["directory", "keras_file"])
def test_predict_and_export_refuse_a_pretrain_run_with_the_task_json_sentence_of_their_input_form(ended, command, form, capsys, tmp_path):
    from PIL import Image

    from train.api import cli

    model = ended.function if form == "directory" else ended.function / "best_model.keras"
    picture = tmp_path / "a.png"
    Image.new("RGB", (32, 32)).save(picture)
    answer = tmp_path / ("answers.json" if command == "predict" else "exported.keras")
    asked = [f"data={picture}", f"out={answer}"] if command == "predict" else [f"out={answer}", "format=keras"]
    code = cli.main([command, f"model={model}", *asked])
    error = capsys.readouterr().err
    assert code == 2, error
    assert f"Invalid value for 'model': {model}" in error
    assert (SENTENCE_OF_A_DIRECTORY if form == "directory" else SENTENCE_OF_A_FILE) in error
    assert not answer.exists() and not (ended.function / "task.json").exists()
