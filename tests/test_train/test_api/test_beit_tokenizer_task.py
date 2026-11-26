"""The ``beit_tokenizer`` task: the hooks, the two-part bar and the controls that can fail it, real tiny runs, the refusals before the claim.

``train.api.tasks.beit_tokenizer.BeitTokenizerTrainer`` (plan-2026-11-25T124419-fa94f59c step 6) trains the library ``VQVAERotationTrick``
with its OWN ``train_step`` (the task adds none), compiled with no loss and no metric (D-021), and judges the run twice: its
reconstruction against the constant image (strictly below ``min(baseline, epoch zero)`` beyond the repo's float32 tie band) AND the
codebook perplexity of its best weights over every selection position (at least the floor of ``pretrain_run``). A bar that a constant
model, a collapsed codebook or a tokenizer that meets one part could meet is vacuous, so each is a real run through the real hooks:

* **A constant model** (a bring-your-own constant decoder, the mean image of the selection rows) misses on the reconstruction whatever
  the codebook does, and eight compiled constant models are judged without the tie band's help being needed: a bare ``<`` or ``<=`` would
  have called some of them a success, which the test counts.
* **A codebook of one live code** (an encoder that cannot tell two images apart) misses on the perplexity although its reconstruction
  clears the bar it is given.
* **A code that is the cell index alone** (an encoder that ignores the image) has the floor's POOLED perplexity exactly at the default
  geometry (8 x 8 cells, 64 codes) and misses on the PER-CELL one (review pass 1, W3).
* **A tokenizer that reconstructs but whose perplexity is under the real floor** misses, and the SAME training meets the bar under a floor
  patched to what a toy can reach (the patch is made only where a run is meant to pass).
* **The usage is the best weights'**: a run whose best epoch is not its last, the last being sabotaged into one code, reports the usage
  of ``best_model.keras``, recomputed here from the file.

Also pinned: the hook table, the bar's arithmetic, ``success_value`` as the minimum of the selection window of the reconstruction term,
the reload of the best checkpoint within ``WEIGHT_MISMATCH_TOLERANCE``, the files ``beit_mim`` reads, the refusals before the claim (exit 2,
``out`` absent), a smoke run (no bar), and the import rule of the task module (no TensorFlow). The runs are the REAL CIFAR-10 rows capped by
``data.max_samples`` (a loud skip without the Keras cache); tests write only under ``tmp_path``. The resume is ``test_beit_tokenizer_resume.py``.
"""

import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import registry
from train.api.config import validate
from train.api.errors import ConfigError, ConfigValidationError
from train.api.tasks import BUILTIN_TASKS
from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer, train_beit_tokenizer
from train.api.tasks.pretrain import selection_block
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import classification_run, pretrain_run
from train.common import tokenizer_data as td
from train.common.forecast_run import tie_band
from train.common.pretrain_data import TokenizerUsage, tokenizer_usage
from train.common.resume_state import FIT_STATE_NAME
from train.common.tokenizer_data import ConstantImageBaseline

from .test_cli_help_allocates_nothing import FORBIDDEN as HEAVY  # the list the help path is held to: one definition
from ._tokenizer_world import (
    BATCH, CIFAR_CACHE, EPOCHS, LOW_FLOOR, MAX_SAMPLES, MODEL, NO_CIFAR, TINY_MODEL_ARGS, constant_model, csv_column, positional_model,
    prepared, read_json, request, world,  # noqa: F401  (``world`` is a fixture)
)

REPO_ROOT = Path(__file__).resolve().parents[3]
#: A threshold far above any reachable loss: a run judged against it clears the reconstruction part by construction.
HUGE = 1000.0
GRID = 8                              # 32 px over a patch of 4
POSITIONS_PER_IMAGE = GRID * GRID


def usage_of(codes: int) -> TokenizerUsage:
    """The usage of ``codes`` equally used codes, by the real calculation (``exp(entropy)`` of 64 of them is 63.99999999999998)."""
    return tokenizer_usage(np.repeat(np.arange(codes), 4))


def cell_usage_of(perplexity: float) -> "td.CellUsage":
    """A per-cell usage worth ``perplexity`` effective codes per cell (the entropy is its log: the floor is judged on the entropy)."""
    return td.CellUsage(cells=POSITIONS_PER_IMAGE, images=100, entropy=math.log(perplexity), perplexity=perplexity)


def bare_trainer(*, select_batches: int = 5, perplexity: int = 100, per_cell: float = 100.0) -> BeitTokenizerTrainer:
    """A trainer with no config and the state ``evaluate_and_report`` leaves (``object.__new__``: the constructor needs a config)."""
    trainer = object.__new__(BeitTokenizerTrainer)
    trainer.config = None
    trainer._select_batches, trainer._usage, trainer._cell_usage = select_batches, usage_of(perplexity), cell_usage_of(per_cell)
    return trainer


def config_of(out: Any, **changes: Any) -> BeitTokenizerConfig:
    """The validated, unresolved config of a tiny request: what ``train_beit_tokenizer`` hands to the trainer."""
    return validate(BeitTokenizerConfig, request(out, **changes))


# ---------------------------------------------------------------------
# The hook table
# ---------------------------------------------------------------------

#: The optional hooks the task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",            # the stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",         # a resumed fit starts a fresh iterator: another stream, not a replay
    "bar_is_met",            # a loss strictly below beyond the tie band AND a codebook at the floor
    "run_analysis",          # data-free (the output is an image, not class probabilities)
    "summary_fields",        # the patch, the code grid, the stream rule
    "recipe_notices",        # the recipe's unmeasured numbers
    "used_arguments",        # the model keywords the run was built with
    "check_model_args",      # the refusals before the claim
    "epoch_line_metrics",    # the epoch line prints the reconstruction term
    "unjudged_note",         # why init_scale_warning is null
    "task_notes",            # the two-part bar in words
)


def test_the_class_names_its_task_and_its_metric_and_a_resume_is_not_exact():
    assert BeitTokenizerTrainer.task == "beit_tokenizer" and issubclass(BeitTokenizerTrainer, TaskTrainer)
    assert BeitTokenizerTrainer.success_metric == "val_reconstruction_loss"
    # D-014: a resume rebuilds the train stream on a seed derived from the completed steps, so it does NOT read the batches the unkilled
    # run reads next; the flag stays False and the order says what a resume does.
    assert BeitTokenizerTrainer.resume_stream_exact is False and "fresh stream" in BeitTokenizerTrainer.resume_order


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_every_required_hook_is_overridden(hook):
    assert getattr(BeitTokenizerTrainer, hook) is not getattr(TaskTrainer, hook), hook


@pytest.mark.parametrize("hook", OPTIONAL_HOOKS)
def test_every_optional_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(BeitTokenizerTrainer, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_does_not_name_a_stage():
    # A task overrides hooks, never the stages of the shared sequence.
    own = {name for name in vars(BeitTokenizerTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim"}


def test_the_task_adds_no_train_step_no_figure_and_no_callback():
    own = {name for name in vars(BeitTokenizerTrainer) if not name.startswith("__")}
    assert not own & {"train_step", "test_step", "compile_options", "optimizer_options", "figure_callbacks", "extra_callbacks"}
    assert not hasattr(BeitTokenizerTrainer, "train_step"), "the model's own train_step trains it (D-003); the task adds none"


def test_the_task_compiles_no_loss_and_no_metric(world, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    assert trainer.build_loss_and_metrics(resolved, None) == (None, [])
    assert trainer.metric_names(resolved, trainer.describe_data(resolved)) == ["reconstruction_loss", "vq_loss"]


def test_the_task_row_and_the_family_row_are_registered_and_resolve(world):
    spec = registry.TASKS["beit_tokenizer"]
    assert spec.load_config_class() is BeitTokenizerConfig and spec.load_function() is train_beit_tokenizer
    assert [row for row in BUILTIN_TASKS if row.name == "beit_tokenizer"] == [spec]
    assert registry.MODELS["vq_vae_rotation"].tasks == ("beit_tokenizer",), "register_builtin_models registers the family row with its task"
    from train import api

    assert api.train_beit_tokenizer is train_beit_tokenizer and api.BeitTokenizerConfig is BeitTokenizerConfig


# ---------------------------------------------------------------------
# The bar: two parts, strictly below beyond the tie band
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reached, perplexity, per_cell, met",
    [(0.5, 100, 100, True), (0.5, 10, 100, False), (1.5, 100, 100, False), (1.5, 10, 100, False),
     (0.5, 64, 1, False), (0.5, 128, 2, False), (0.5, 512, 7.9, False), (0.5, 64, 8, True)],
    ids=["both parts", "reconstructs, codebook under the floor", "codebook at the floor, no reconstruction", "neither",
         "the cell index alone: pooled 64, per cell 1", "cell index plus a bit: pooled 128, per cell 2", "just under the per-cell floor",
         "exactly both floors"],
)
def test_the_bar_needs_both_parts(reached, perplexity, per_cell, met):  # the codebook part is two floors, pooled and per cell (D-067)
    assert pretrain_run.MIN_TOKENIZER_PERPLEXITY == 64.0, "the real floor: 100 meets it and 10 does not"
    assert td.MIN_PER_CELL_PERPLEXITY == 8.0, "the real per-cell floor: 8 meets it and 7.9 does not"
    assert bare_trainer(perplexity=perplexity, per_cell=per_cell).bar_is_met(None, reached, 1.0) is met


def test_exactly_the_floor_meets_it_through_the_one_definition_of_the_floor():
    # 64 equally used codes: exp(log(64)) is 63.99999999999998, so a comparison of the perplexity would refuse them (D-028).
    assert usage_of(64).perplexity < 64.0, "the control: the real calculation does come out below 64"
    assert bare_trainer(perplexity=64).bar_is_met(None, 0.5, 1.0) is True
    assert bare_trainer(perplexity=63).bar_is_met(None, 0.5, 1.0) is False


@pytest.mark.parametrize("select_batches", [1, 5, 157, 2048])
def test_a_value_inside_the_tie_band_is_a_miss_and_one_step_beyond_it_is_a_hit(select_batches):
    trainer = bare_trainer(select_batches=select_batches)
    threshold = np.float32(0.9874)
    band, step = tie_band(select_batches), float(np.spacing(threshold))
    assert trainer.bar_is_met(None, float(threshold), float(threshold)) is False, "equal is a miss"
    assert trainer.bar_is_met(None, float(threshold) - step, float(threshold)) is False, "one float32 step below is a rounding"
    assert trainer.bar_is_met(None, float(threshold) - band * step, float(threshold)) is False, "the band's whole width is still a miss"
    assert trainer.bar_is_met(None, float(threshold) - (band + 1) * step, float(threshold)) is True
    # The control: a bare `<` and a `<=` of the same float32 numbers call the first three a success or a tie, which is what the band is for.
    assert np.float32(float(threshold) - step) < threshold and np.float32(float(threshold)) <= threshold


def test_a_float64_value_that_rounds_to_the_float32_threshold_is_a_miss():
    threshold = float(np.float32(0.3))
    reached = threshold - 1e-10
    assert reached < threshold and np.float32(reached) == np.float32(threshold)
    assert bare_trainer().bar_is_met(None, reached, threshold) is False


def test_the_value_is_the_minimum_of_the_reconstruction_loss_of_the_window_and_no_other_column():
    trainer = bare_trainer()
    history = {
        "loss": [0.01, 0.01, 0.01], "reconstruction_loss": [0.02, 0.02, 0.02], "vq_loss": [0.03, 0.03, 0.03],
        "val_loss": [0.04, 0.04, 0.04], "val_vq_loss": [0.05, 0.05, 0.05], "val_reconstruction_loss": [0.9, 0.7, 0.8],
    }
    assert trainer.success_value(history) == 0.7, "the lowest val_reconstruction_loss, not the train loss or the total"
    assert trainer.success_value({"val_reconstruction_loss": [0.3, 0.9, 0.8]}) == 0.3
    assert trainer.success_value({"val_reconstruction_loss": [0.9, 0.8, 0.2]}) == 0.2, "the minimum, wherever it falls"


def test_a_nan_epoch_is_never_a_success_and_a_later_nan_does_not_move_the_minimum():
    # Review pass 1 (blind spot): Python's `min` keeps a LEADING NaN, so a first epoch whose val_reconstruction_loss is NaN gives a NaN value and the run misses
    # even if a later epoch is good; that is the safe direction (a comparison with NaN is False, so a NaN is never beaten and never beats the bar). A NaN LATER
    # in the window does not move the minimum. The sequence refuses a non-finite initial loss and names a non-finite monitor in the summary, so this is the
    # verdict's own behaviour and not the only guard.
    trainer = bare_trainer()
    assert math.isnan(trainer.success_value({"val_reconstruction_loss": [float("nan"), 0.3, 0.2]}))
    assert trainer.success_value({"val_reconstruction_loss": [0.3, float("nan"), 0.2]}) == 0.2
    assert trainer.bar_is_met(None, float("nan"), 1.0) is False and trainer.bar_is_met(None, 0.2, float("nan")) is False


def test_the_value_ignores_the_epochs_before_the_selection_start():
    class Windowed(BeitTokenizerTrainer):
        def selection_start_epoch(self, config):
            return 2

    trainer = object.__new__(Windowed)
    trainer.config = None
    assert trainer.success_value({"val_reconstruction_loss": [0.1, 0.2, 0.9, 0.8]}) == 0.8
    assert bare_trainer().success_value({"val_reconstruction_loss": [0.1, 0.2, 0.9, 0.8]}) == 0.1


def test_the_threshold_is_the_smaller_of_the_baseline_and_the_epoch_zero_loss_and_is_refused_before_either_is_known():
    trainer = bare_trainer()
    trainer._baseline, trainer._epoch_zero = ConstantImageBaseline("mean_image", 1.0, 1.1, 1.0), 0.8
    assert trainer.success_threshold(None) == 0.8
    trainer._epoch_zero = 2.5
    assert trainer.success_threshold(None) == 1.0
    trainer._baseline = None
    with pytest.raises(RuntimeError, match="baseline"):
        trainer.success_threshold(None)
    trainer._baseline, trainer._epoch_zero = ConstantImageBaseline("mean_image", 1.0, 1.1, 1.0), None
    with pytest.raises(RuntimeError, match="epoch-zero"):
        trainer.success_threshold(None)


@pytest.mark.parametrize("missing", ["_usage", "_cell_usage"])
def test_the_bar_asked_before_the_codebook_is_judged_is_refused_by_name(missing):
    trainer = bare_trainer()
    setattr(trainer, missing, None)
    with pytest.raises(RuntimeError, match="codebook"):
        trainer.bar_is_met(None, 0.5, 1.0)


# ---------------------------------------------------------------------
# The data, the baseline and the control of eight constant models
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def cifar():
    if not CIFAR_CACHE.exists():
        pytest.skip(NO_CIFAR)


def test_build_data_states_the_epoch_the_validation_and_the_baseline_of_its_own_selection_rows(world, cifar, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "run")
    data = trainer.build_data(resolved, context)
    assert (data.n_train, data.n_select) == (MAX_SAMPLES - MAX_SAMPLES // 10, MAX_SAMPLES // 10) and data.num_classes is None
    assert trainer.fit_kwargs(resolved, data) == {"steps_per_epoch": data.n_train // BATCH, "validation_steps": data.val_steps}
    assert data.steps_per_epoch == data.n_train // BATCH == 45 and data.val_steps == data.n_select // BATCH
    baseline = td.constant_image_baseline(data.select_ds)
    assert trainer._baseline == baseline and 0 < baseline.reconstruction_loss < 2 and trainer._select_batches == data.val_steps
    trainer.check_model_args(resolved, context)
    trainer.input_check(trainer.describe_data(resolved), data)


def test_a_selection_split_too_small_to_show_the_floor_is_refused_by_name_after_the_data_is_built(world, cifar, tmp_path):
    # 20 rows of the train pool select 2 images; at 16 px and a patch of 4 that is 2 x 16 positions, and a perplexity over 32 positions cannot
    # reach 64 (a codebook that uses every position once is worth 32 codes).
    changes = {"data": {"dataset": "cifar10", "max_samples": 20}, "transform_args": {"augment": False, "image_size": 16}, "batch_size": 8}
    data = td.load_tokenizer_batches(
        dataset="cifar10", image_size=16, batch_size=8, seed=42, stream_seed=None, validation_split=None, max_samples=20, augment=False)
    positions = data.n_select * (16 // 4) ** 2
    assert 0 < positions < 64, "the control: the split really is too small"
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        BeitTokenizerTrainer(config_of(out, **changes)).run()
    message = str(caught.value)
    assert f"{positions} positions" in message and "64" in message and "data.max_samples" in message
    assert "was claimed" in message and out.exists()
    # a smoke run applies no bar, so the same small split is not refused
    trainer, resolved, context = prepared(tmp_path / "smoke", epochs="auto", **changes, **{"debug_args.smoke": True})
    assert trainer.build_data(resolved, context).n_select == data.n_select


def test_a_selection_split_with_fewer_images_than_the_per_cell_number_reads_true_from_is_named_in_a_warning_and_the_run_goes_on(world, cifar, tmp_path, caplog):
    # A perplexity counted PER CELL over n images is at most n (7 images cannot reach the floor of 8) and reads low well above that: on the saved codes of two
    # trained arms every draw of 8 images missed the floor and none of 300 draws of 12 did (review pass 2, N3; `tokenizer_data.MIN_PER_CELL_IMAGES` is 16 with the
    # numbers). So the run says so below 16 images, when its data is built: 7 (the bound itself) and 15 (the last size under it) name it, 16 does not, and a smoke
    # run applies no bar and says nothing. A WARNING and not a refusal: the resume tests train on six images and their data may not change.
    caplog.set_level(logging.WARNING, logger="dl")
    said = lambda: [record.getMessage() for record in caplog.records if "per-cell floor" in record.getMessage()]  # noqa: E731
    for rows, images in ((70, 7), (150, 15)):
        caplog.clear()
        trainer, resolved, context = prepared(tmp_path / f"run{rows}", **{"data": {"dataset": "cifar10", "max_samples": rows}})
        assert trainer.build_data(resolved, context).n_select == images
        assert len(said()) == 1 and f"{images} images" in said()[0] and f"at most {images}" in said()[0] and "data.max_samples" in said()[0], said()
        assert f"fewer than the {td.MIN_PER_CELL_IMAGES}" in said()[0] and td.MIN_PER_CELL_IMAGES == 16, said()
    caplog.clear()
    trainer, resolved, context = prepared(tmp_path / "enough", **{"data": {"dataset": "cifar10", "max_samples": 160}})
    assert trainer.build_data(resolved, context).n_select == 16 and not said(), "16 images read true: no warning"
    trainer, resolved, context = prepared(tmp_path / "smoke", **{"data": {"dataset": "cifar10", "max_samples": 70}, "debug_args.smoke": True})
    assert trainer.build_data(resolved, context).n_select == 7 and not said(), "a smoke run applies no bar"


def test_eight_compiled_constant_models_are_not_called_a_success_though_a_bare_compare_would_call_some(world, cifar, capsys):
    trainer = bare_trainer()
    rows, below, equal = [], 0, 0
    for max_samples in (200, 300, 400, 500):
        data = td.load_tokenizer_batches(
            dataset="cifar10", image_size=32, batch_size=BATCH, seed=42, stream_seed=None, validation_split=None,
            max_samples=max_samples, augment=False)
        baseline = td.constant_image_baseline(data.select_ds)
        images = np.concatenate([np.asarray(batch[0], np.float64) for batch in data.select_ds])
        for kind, constant in ((td.CHANNEL_MEAN, images.mean(axis=(0, 1, 2))), (td.MEAN_IMAGE, images.mean(axis=0))):
            reached = classification_run.evaluate_dict(constant_model(constant), data.select_ds)["reconstruction_loss"]
            threshold = {td.CHANNEL_MEAN: baseline.channel_mean, td.MEAN_IMAGE: baseline.mean_image}[kind]
            trainer._select_batches = data.val_steps
            rows.append((max_samples, kind, reached, threshold, data.val_steps))
            below += bool(np.float32(reached) < np.float32(threshold))
            equal += bool(np.float32(reached) == np.float32(threshold))
            assert trainer.bar_is_met(None, reached, threshold) is False, f"{kind} at {max_samples} rows is called a success"
    with capsys.disabled():
        print(f"\nCONSTANT MODELS: {len(rows)} compiled, a bare `<` of float32 passes {below}, a `<=` passes {below + equal}, the bar passes 0")
        for row in rows:
            print(f"  max_samples {row[0]:4d} {row[1]:12s} evaluate {row[2]!r}  baseline {row[3]!r}  batches {row[4]}")
    assert len(rows) == 8
    # The vacuity guard: without it the test could pass because none of the eight ever reached its own baseline.
    assert below + equal >= 1, "no constant model sat at or below its own baseline in float32: the control saw nothing to refuse"


# ---------------------------------------------------------------------
# Real tiny runs: the command line, a constant model, one live code, a best epoch that is not the last
# ---------------------------------------------------------------------


def flatten_request(asked: Dict[str, Any]) -> List[str]:
    """``key=value`` tokens as the command line takes them: groups are dotted, ``model_args`` is one JSON value."""
    tokens = []
    for key, value in asked.items():
        if isinstance(value, dict) and key != "model_args":
            tokens += [f"{key}.{inner}={str(v).lower() if isinstance(v, bool) else v}" for inner, v in value.items()]
        else:
            tokens.append(f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, dict) else value}")
    return tokens


class ConstantTrainer(BeitTokenizerTrainer):
    """The real task whose model reconstructs one constant image, the mean image of the selection rows: the control the bar must refuse."""

    def build_model(self, config, context, data):
        images = np.concatenate([np.asarray(batch[0], np.float64) for batch in data.select_ds])
        return constant_model(images.mean(axis=0), size=data.input_shape[0], patch=config.data.patch_size,
                              dim=config.model_args["embedding_dim"], codes=config.model_args["num_embeddings"])


def collapse(model) -> None:
    """Put every row of the codebook but the first a million times further away than any encoder output: every position has ONE code."""
    book = np.full(model.quantizer.embeddings.shape, 1e4, np.float32)
    book[:, 0] = 0.0
    model.quantizer.embeddings.assign(book)


class OneCodeTrainer(BeitTokenizerTrainer):
    """The real task whose codebook has one live code from the first step: no position can have another, whatever the encoder learns."""

    def build_model(self, config, context, data):
        model = super().build_model(config, context, data)
        collapse(model)
        return model


class PositionalTrainer(BeitTokenizerTrainer):
    """The real task whose code is the cell index alone, the image ignored: at 8 x 8 cells its pooled perplexity is exactly the floor's 64."""

    def build_model(self, config, context, data):
        return positional_model(size=data.input_shape[0], patch=config.data.patch_size, dim=config.model_args["embedding_dim"])


class WreckedTrainer(BeitTokenizerTrainer):
    """The real task whose LAST epoch is wrecked (the codebook collapses to one code at its start): the best epoch is not the last."""

    EPOCH = 2

    def extra_callbacks(self, config, data):
        class Wreck(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                if epoch == WreckedTrainer.EPOCH:
                    collapse(self.model)

        return [Wreck()]


def steps_between(a: float, b: float) -> float:
    """``|a - b|`` in float32 steps of ``b``."""
    return abs(a - b) / float(np.spacing(np.float32(b)))


@pytest.fixture(scope="module")
def runs(world, cifar, tmp_path_factory):
    """Seven real tiny runs: the command line twice (the floor patched, and not), a constant model, one live code, a code that is the cell
    index alone, a wrecked last epoch, a smoke run."""
    from train.api import cli

    root = tmp_path_factory.mktemp("tokenizer_runs")
    done: Dict[str, Any] = {"root": root}

    def command(name: str, **changes: Any) -> int:
        asked = request(root / name, **changes)
        done[name + "_asked"] = asked
        return cli.main(["beit_tokenizer", *flatten_request(asked)])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        done["met_code"] = command("met", **{"analysis_args.model_analysis": True})
    # The SAME training under the real floor of 64: the tokenizer reconstructs and its codebook does not reach it.
    done["short_code"] = command("short")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", 1.0)            # the codebook part is vacuous: only the image can miss
        patch.setattr(pretrain_run, "epoch_zero_val_loss", lambda path, key="loss": HUGE)   # the baseline, not epoch zero, binds
        done["constant"] = ConstantTrainer(config_of(root / "constant")).run()
    with pytest.MonkeyPatch.context() as patch:
        # The reconstruction part is given a threshold it clears by construction, so only the codebook can miss.
        patch.setattr(td, "constant_image_baseline", lambda ds: ConstantImageBaseline(td.MEAN_IMAGE, HUGE, HUGE, HUGE))
        done["one_code"] = OneCodeTrainer(config_of(root / "one_code")).run()
    with pytest.MonkeyPatch.context() as patch:
        # As the one-live-code run: the image part is given a threshold it clears, so only the codebook can miss. NO floor is patched: the real
        # pooled floor of 64 is exactly this code's pooled perplexity (64 cells, 64 codes), and the real per-cell floor is what must refuse it.
        patch.setattr(td, "constant_image_baseline", lambda ds: ConstantImageBaseline(td.MEAN_IMAGE, HUGE, HUGE, HUGE))
        done["positional"] = PositionalTrainer(config_of(root / "positional")).run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pretrain_run, "MIN_TOKENIZER_PERPLEXITY", LOW_FLOOR)
        done["wrecked"] = WreckedTrainer(config_of(root / "wrecked", epochs=3)).run()
    done["smoke"] = train_beit_tokenizer(
        out=root / "smoke", model=MODEL, data={"dataset": "cifar10"}, batch_size=BATCH,
        model_args={**TINY_MODEL_ARGS, "num_embeddings": 16}, optim_args={"lr": 3e-3, "warmup_epochs": 0},
        transform_args={"augment": False}, analysis_args={"model_analysis": False, "figures": False}, debug_args={"smoke": True})
    return SimpleNamespace(**done)


def summary_of(runs, name: str) -> Dict[str, Any]:
    return read_json(runs.root / name / "results_summary.json")


def usage_of_file(path: Path, request_mapping: Dict[str, Any]) -> Dict[str, float]:
    """Usage of a saved tokenizer over the selection rows, recomputed with numpy alone from the file (never through the task)."""
    model = keras.models.load_model(str(path))
    data = request_mapping["data"]
    selection = td.load_tokenizer_batches(
        dataset=data["dataset"], image_size=32, batch_size=request_mapping["batch_size"], seed=request_mapping.get("seed", 42),
        stream_seed=None, validation_split=None, max_samples=data["max_samples"], augment=False)
    grid = np.concatenate([np.asarray(model.encode_to_indices(batch[0])) for batch in selection.select_ds])
    ids = grid.reshape(-1)
    share = np.bincount(ids) / ids.size
    share = share[share > 0]
    cell_entropies = []
    for column in grid.reshape(len(grid), -1).T:  # one cell at a time: the codes it gives to every selection image
        cell_share = np.bincount(column) / len(column)
        cell_share = cell_share[cell_share > 0]
        cell_entropies.append(-(cell_share * np.log(cell_share)).sum())
    return {"positions": int(ids.size), "distinct": int(len(share)), "perplexity": float(np.exp(-(share * np.log(share)).sum())),
            "top1_share": float(share.max()), "per_cell_perplexity": float(np.exp(np.mean(cell_entropies)))}


def test_a_tiny_run_through_the_command_line_meets_both_parts_and_exits_zero(runs):
    summary = summary_of(runs, "met")
    assert runs.met_code == 0 and summary["success"] is True and summary["status"] == "ok" and summary["task"] == "beit_tokenizer"
    assert summary["success_metric"] == "val_reconstruction_loss" and summary["reconstruction"]["beats_bar"] is True
    assert summary["codebook_usage"]["meets_floor"] is True and summary["codebook_usage"]["floor"] == LOW_FLOOR
    assert summary["success_value"] < summary["success_threshold"], "the reconstruction part: strictly below"
    assert summary["codebook_usage"]["perplexity"] >= summary["codebook_usage"]["floor"], "the codebook part"
    # The per-cell floor is the REAL one here (nothing patches it): the toy's codes reach it (15.6 per cell against 8), reported beside the pooled one.
    usage = summary["codebook_usage"]
    assert usage["per_cell_floor"] == td.MIN_PER_CELL_PERPLEXITY == 8.0 and usage["per_cell_meets_floor"] is True
    assert usage["per_cell_perplexity"] >= usage["per_cell_floor"] and usage["per_cell_perplexity"] <= usage["perplexity"] + 1e-9
    assert summary["epochs_run"] == EPOCHS and summary["steps_per_epoch"] == 45 and summary["n_val"] == MAX_SAMPLES // 10


def test_the_same_training_misses_under_the_real_floor_on_the_codebook_part_alone(runs):
    met, short = summary_of(runs, "met"), summary_of(runs, "short")
    assert runs.short_code == 1 and short["success"] is False and short["status"] == "ok", "a miss is exit 1 with the files kept"
    assert short["reconstruction"]["beats_bar"] is True, "it reconstructs: the image part is met"
    assert short["codebook_usage"]["meets_floor"] is False and short["codebook_usage"]["floor"] == pretrain_run.MIN_TOKENIZER_PERPLEXITY == 64.0
    assert short["codebook_usage"]["perplexity"] < 64.0
    # Control: it is the one training. The two runs differ in the floor only, so only the verdict differs.
    assert short["success_value"] == pytest.approx(met["success_value"], rel=1e-3)
    assert short["codebook_usage"]["perplexity"] == pytest.approx(met["codebook_usage"]["perplexity"], rel=1e-2)
    assert (runs.root / "short" / "best_model.keras").is_file() and (runs.root / "short" / "final_model.keras").is_file()


def test_the_summary_names_both_parts_the_baseline_the_epoch_zero_term_and_the_window(runs):
    summary = summary_of(runs, "met")
    sidecar = read_json(runs.root / "met" / "checkpoints" / FIT_STATE_NAME)
    epoch_zero = sidecar["initial"]["metrics"]["reconstruction_loss"]
    baseline = summary["baseline"]
    assert baseline["kind"] == td.MEAN_IMAGE and baseline["reconstruction_loss"] == min(baseline["channel_mean"], baseline["mean_image"]) > 0
    assert summary["epoch_zero"] == {"reconstruction_loss": epoch_zero, "binds": bool(epoch_zero < baseline["reconstruction_loss"])}
    assert summary["success_threshold"] == min(baseline["reconstruction_loss"], epoch_zero)
    column = csv_column(runs.root / "met", "val_reconstruction_loss")
    assert len(column) == EPOCHS and summary["success_value"] == pytest.approx(min(column), rel=1e-6), "the window's minimum"
    best = int(np.argmin(column)) + 1
    assert summary["selection"] == selection_block(0, EPOCHS, best) and summary["best_epoch"] == best
    assert summary["reconstruction"] == {
        "success_value": summary["success_value"], "success_threshold": summary["success_threshold"],
        "tie_band_float32_steps": tie_band(MAX_SAMPLES // 10 // BATCH), "beats_bar": True}
    usage = summary["codebook_usage"]
    assert summary["success"] is bool(summary["reconstruction"]["beats_bar"] and usage["meets_floor"] and usage["per_cell_meets_floor"])
    assert any("both parts" in note for note in summary["notes"]), "the notes say the bar has two parts"


def test_the_codebook_usage_is_the_best_weights_over_every_selection_position_recomputed_from_the_file(runs):
    summary = summary_of(runs, "met")
    block = summary["codebook_usage"]
    independent = usage_of_file(runs.root / "met" / "best_model.keras", runs.met_asked)
    assert block["positions"] == independent["positions"] == (MAX_SAMPLES // 10) * POSITIONS_PER_IMAGE, "every position of every selection image"
    assert block["distinct"] == independent["distinct"] and block["num_embeddings"] == TINY_MODEL_ARGS["num_embeddings"]
    assert block["perplexity"] == pytest.approx(independent["perplexity"], rel=1e-5)
    assert block["top1_share"] == pytest.approx(independent["top1_share"], rel=1e-5) and 1.0 < block["perplexity"] <= block["num_embeddings"]
    assert block["per_cell_perplexity"] == pytest.approx(independent["per_cell_perplexity"], rel=1e-5), "the per-cell number, cell by cell, from the file"
    assert block["per_cell_entropy"] == pytest.approx(math.log(block["per_cell_perplexity"]), rel=1e-9)


def test_the_best_checkpoint_reloads_within_the_tolerance_and_the_files_beit_mim_reads_are_written(runs):
    summary = summary_of(runs, "met")
    assert summary["best_checkpoint_load_error"] is None
    assert summary["best_checkpoint_max_abs_diff"] <= classification_run.WEIGHT_MISMATCH_TOLERANCE
    assert summary["model_loading_validated"] is True and summary["test_metrics_best"] is not None
    assert set(summary["test_metrics_best"]) == {"loss", "reconstruction_loss", "vq_loss"}
    for name in ("best_model.keras", "final_model.keras", "results_summary.json", "training_log.csv", "config.json", "run.log"):
        assert (runs.root / "met" / name).is_file(), name
    # The deliverable is the file `beit_mim` loads: its loader proves the 8 x 8 code grid by a forward pass at 32 px.
    from train.beit.common import load_frozen_tokenizer

    for name in ("best_model.keras", "final_model.keras"):
        tokenizer = load_frozen_tokenizer(str(runs.root / "met" / name), (GRID, GRID), (32, 32, 3))
        assert int(tokenizer.model.num_embeddings) == TINY_MODEL_ARGS["num_embeddings"], name
    assert summary["analyzer"]["status"] == "ok", "the data-free analysis completes for this model"


def test_a_constant_model_through_the_same_hooks_does_not_meet_the_bar(runs):
    summary = summary_of(runs, "constant")
    baseline = summary["baseline"]["reconstruction_loss"]
    assert runs.constant.success is False and summary["success"] is False and summary["status"] == "ok"
    assert summary["epoch_zero"]["binds"] is False and summary["success_threshold"] == baseline, "the baseline, not epoch zero, binds"
    steps = steps_between(summary["success_value"], baseline)
    print(f"\nCONSTANT MODEL RUN: val_reconstruction_loss {summary['success_value']!r} baseline {baseline!r} {steps:.2f} float32 steps")
    assert steps <= tie_band(MAX_SAMPLES // 10 // BATCH), "it reconstructs the constant image: the baseline IS its score, to the band"
    assert summary["reconstruction"]["beats_bar"] is False
    assert summary["codebook_usage"]["meets_floor"] is True, "the codebook part is made vacuous, so the image part alone refused it"


def test_a_codebook_of_one_live_code_misses_on_the_perplexity_though_its_reconstruction_clears_the_bar_it_is_given(runs):
    summary = summary_of(runs, "one_code")
    assert runs.one_code.success is False and summary["success"] is False
    assert summary["reconstruction"]["beats_bar"] is True, "given a threshold it clears (epoch zero, before the decoder learned a constant)"
    usage = summary["codebook_usage"]
    assert (usage["distinct"], usage["perplexity"], usage["top1_share"]) == (1, 1.0, 1.0)
    assert usage["meets_floor"] is False and usage["floor"] == 64.0 and usage["positions"] == (MAX_SAMPLES // 10) * POSITIONS_PER_IMAGE


def test_a_tokenizer_whose_code_is_the_cell_index_alone_misses_though_its_pooled_perplexity_meets_the_floor(runs):
    # Review pass 1, W3: at 32 px and patch 4 there are 8 x 8 = 64 cells, so a code that ignores the image has a pooled perplexity of exactly the
    # floor (64) and the pooled floor called it a tokenizer; part 1 asks only for the constant image, which the threshold below is made to clear.
    summary = summary_of(runs, "positional")
    usage = summary["codebook_usage"]
    assert usage["distinct"] == 64 and usage["perplexity"] == pytest.approx(64.0, abs=1e-9) and usage["floor"] == 64.0
    assert usage["meets_floor"] is True, "the control: the pooled floor of the real recipe is met by this code, to the digit"
    assert summary["reconstruction"]["beats_bar"] is True, "the image part is given a threshold it clears: only the codebook can miss"
    assert runs.positional.success is False and summary["success"] is False, "the bar called a code that is the cell index alone a tokenizer"
    assert usage["per_cell_perplexity"] == pytest.approx(1.0, abs=1e-9) and usage["per_cell_meets_floor"] is False
    assert usage["per_cell_floor"] == td.MIN_PER_CELL_PERPLEXITY == 8.0
    # The control is what it says: the codes of the saved best weights are the cell index at every cell of every selection image.
    model = keras.models.load_model(str(runs.root / "positional" / "best_model.keras"))
    codes = td.code_ids_of(model, td.load_tokenizer_batches(
        dataset="cifar10", image_size=32, batch_size=BATCH, seed=42, stream_seed=None, validation_split=None, max_samples=MAX_SAMPLES,
        augment=False).select_ds)
    assert (codes == np.arange(POSITIONS_PER_IMAGE).reshape(1, GRID, GRID)).all()


def test_a_best_epoch_that_is_not_the_last_reports_the_best_weights_usage_and_not_the_last_ones(runs):
    summary = summary_of(runs, "wrecked")
    assert summary["selection"]["final_is_best"] is False and summary["best_epoch"] < summary["epochs_run"] == 3
    asked = {**request(runs.root / "wrecked", epochs=3)}
    best, last = (usage_of_file(runs.root / "wrecked" / name, asked) for name in ("best_model.keras", "final_model.keras"))
    assert last["distinct"] == 1, "the control: the wrecked last epoch really is one code"
    assert best["perplexity"] > LOW_FLOOR, "the control: the best epoch's codebook meets the patched floor the last one cannot"
    assert summary["codebook_usage"]["perplexity"] == pytest.approx(best["perplexity"], rel=1e-5)
    assert summary["codebook_usage"]["distinct"] == best["distinct"] > 1
    assert summary["codebook_usage"]["meets_floor"] is True and runs.wrecked.success is True


def test_a_smoke_run_applies_no_bar_and_a_small_codebook_is_allowed_in_it(runs):
    summary = read_json(runs.root / "smoke" / "results_summary.json")
    assert runs.smoke.success is True, "a smoke run that finished is a success"
    assert summary["success"] is None and summary["success_threshold"] is None
    assert any("not applicable" in note for note in summary["notes"])
    assert summary["epochs_requested"] == 1 and summary["n_train"] + summary["n_val"] == 512
    # the measurements the bar would have used are still reported
    assert summary["baseline"]["reconstruction_loss"] > 0 and summary["codebook_usage"]["num_embeddings"] == 16
    assert summary["codebook_usage"]["meets_floor"] is False, "16 codes cannot reach 64: reported, not applied"


# ---------------------------------------------------------------------
# Refusals before the claim: exit 2, nothing claimed
# ---------------------------------------------------------------------

#: The settings that keep a run short; a request the task wrongly accepts then fails the assertion within a minute and not after the recipe's epochs.
CHEAP = {"data.max_samples": "64", "epochs": "1", "batch_size": "8", "analysis_args.model_analysis": "false", "analysis_args.figures": "false"}

REFUSALS = [
    (["data.patch_size=5"], "data.patch_size", "power of 2"),
    (["data.patch_size=3"], "data.patch_size", "power of 2"),
    (["data.patch_size=1"], "data.patch_size", "below 2"),
    (["data.patch_size=64"], "data.patch_size", "does not divide"),
    (["transform_args.image_size=34"], "data.patch_size", "does not divide"),
    (['model_args={"bogus":1}'], "model_args.bogus", "Unknown key"),
    (['model_args={"kmeans_init":true}'], "model_args.kmeans_init", "not offered"),
    (['model_args={"num_embeddings":32}'], "model_args.num_embeddings", "64"),
    (['model_args={"hidden_channels":0}'], "model_args.hidden_channels", "positive int"),
    (["model_args.use_ema=False"], "model_args.use_ema", "true or false"),
    (["model_args.num_res_blocks=true"], "model_args.num_res_blocks", "expected an int"),
    (['model_args={"downsample_factor":2}'], "model_args.downsample_factor", "data.patch_size"),
    (["precision=mixed_float16"], "precision", "float32"),
    (["metric_args.watch_metric=val_loss"], "metric_args.watch_metric", "val_reconstruction_loss"),
    (["analysis_args.conformal_alpha=0.1"], "analysis_args.conformal_alpha", "conformal"),
    (["model=vit/vit_pico"], "vit", "trains"),
    (["model=vq_vae_rotation/tiny"], "vq_vae_rotation", "variant"),
]


@pytest.mark.parametrize("tokens, key, fragment", REFUSALS, ids=[" ".join(tokens)[:48] for tokens, _, _ in REFUSALS])
def test_a_request_the_task_cannot_honour_exits_two_before_the_claim(world, tmp_path, capsys, tokens, key, fragment):
    from train.api import cli

    out = tmp_path / "run"
    # A run that a missing refusal let through must stay short (the recipe's 50 epochs of the whole dataset are hours): the exit code is the test.
    asked = {"out": str(out), "model": MODEL, "data.dataset": "cifar10", **CHEAP}
    asked.update(token.split("=", 1) for token in tokens)
    assert cli.main(["beit_tokenizer", *(f"{name}={value}" for name, value in asked.items())]) == 2
    error = capsys.readouterr().err
    assert key in error and fragment in error, error
    assert not out.exists(), "the refusal came after the claim"


def test_the_function_refuses_a_patch_that_does_not_divide_before_the_claim(world, tmp_path):
    out = tmp_path / "run"
    cheap = dict(epochs=1, batch_size=8, analysis_args={"model_analysis": False, "figures": False})
    with pytest.raises(ConfigError, match="data.patch_size"):
        train_beit_tokenizer(out=out, model=MODEL, data={"dataset": "cifar10", "max_samples": 64, "patch_size": 5}, **cheap)
    assert not out.exists()
    with pytest.raises(ConfigValidationError, match="model_args.kmeans_init"):
        train_beit_tokenizer(out=out, model=MODEL, data={"dataset": "cifar10", "max_samples": 64}, model_args={"kmeans_init": True}, **cheap)
    assert not out.exists()


def test_a_family_of_another_task_is_refused_under_this_one_and_this_family_under_every_other_task(world, tmp_path):
    with pytest.raises(ConfigError, match="the family 'vit' trains"):
        prepared(tmp_path / "run", model="vit/vit_pico")
    others = sorted({task for spec in registry.MODELS.values() for task in spec.tasks} - {"beit_tokenizer"})
    assert len(others) >= 8
    for task in others:
        with pytest.raises(ConfigError, match="trains beit_tokenizer"):
            registry.parse_model_name("vq_vae_rotation/small", task=task)


# ---------------------------------------------------------------------
# The import rule and the help
# ---------------------------------------------------------------------

def fresh_process(code: str):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""})


def test_the_task_module_imports_no_tensorflow_no_keras_and_no_model_code():
    done = fresh_process(
        "import sys\n"
        "import train.api.tasks.beit_tokenizer\n"
        f"print('LOADED', sorted(m for m in {HEAVY!r} if m in sys.modules))\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED []"


def test_the_pre_claim_checks_initialise_no_tensorflow_device():
    """``setup_gpu`` comes after the checks that run before the claim, and an eager op before it makes memory growth impossible."""
    done = fresh_process(
        "import keras\n"
        "from tensorflow.python.eager import context as eager\n"
        "from train.api import recipes, registry\n"
        "from train.api.config import validate\n"
        "from train.api.tasks import register_builtin_tasks\n"
        "from train.api.tasks.beit_tokenizer import BeitTokenizerConfig, BeitTokenizerTrainer\n"
        "recipes.register_builtin_models()\n"
        "register_builtin_tasks()\n"
        "config = validate(BeitTokenizerConfig, {'out': 'x', 'model': 'vq_vae_rotation/small', 'data': {'dataset': 'cifar10'}})\n"
        "trainer = BeitTokenizerTrainer(config)\n"
        "family, variant = registry.parse_model_name(config.model, task=trainer.task)\n"
        "trainer._resolve(trainer.describe_data(config), family, variant)\n"
        "before = eager.context()._context_handle is not None\n"
        "keras.ops.zeros((1,))\n"
        "print('INITIALISED', before, 'CONTROL', eager.context()._context_handle is not None)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "INITIALISED False CONTROL True", "the control: an eager op does initialise the context"


def test_the_help_of_the_command_lists_the_keys_and_allocates_nothing():
    done = fresh_process(
        "import sys\n"
        "from train.api import cli\n"
        "code = cli.main(['beit_tokenizer', 'help'])\n"
        f"print('LOADED', sorted(m for m in {HEAVY!r} if m in sys.modules), 'EXIT', code)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "LOADED [] EXIT 0"
    for key in ("data.dataset", "data.patch_size", "data.max_samples", "model_args", "metric_args.watch_metric", "transform_args.image_size",
                "optim_args.lr", "checkpoint_args.early_stopping_patience", "debug_args.smoke", "resume_interrupted"):
        assert key in done.stdout, f"`python -m train beit_tokenizer help` does not list {key}"
    assert "vq_vae_rotation/small" in done.stdout and "val_reconstruction_loss" in done.stdout


def test_the_command_is_listed_with_its_family(world, capsys):
    from train.api import cli

    assert cli.main(["list_tasks"]) == 0 and "beit_tokenizer" in capsys.readouterr().out
    assert cli.main(["list_models", "task=beit_tokenizer"]) == 0
    assert "vq_vae_rotation/small" in capsys.readouterr().out
