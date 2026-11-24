"""The ``language_model_pretraining`` task: one tiny run end to end, the bar, the loss, the vocabulary and the refusals.

``train.api.tasks.language_model_pretraining.LanguageModelPretrainingTrainer`` (plan-2026-11-24T092348-7806a290 step 4) trains the
library GPT-2 (family ``gpt2_lm``) as a causal language model on packed windows of tokenized documents, with the stock ``fit``:
the model is a functional ``keras.Model`` over the library backbone (a float32 id input, an in-graph cast, the logits), and
``compile`` is the stock sparse cross entropy from logits. The data is the offline ``synthetic`` bank under the character
tokenizer ``ascii95`` (101 ids, no file), so every run here is a CPU run of seconds (``_text_world``: 64 wide, 2 blocks).
What is pinned:

* **The run.** A 3-epoch run exits 0 and meets the bar: the best ``val_loss`` of the selection windows is strictly below
  ``min(unigram, epoch zero)``, the unigram being the entropy of the selection targets (counted, never evaluated) and the epoch
  zero the untrained model's value read from the fit sidecar. A COMPILED constant unigram model does not meet that bar, and a
  value that merely equals the threshold does not either. The bank has word-order structure a unigram cannot use, so the bar
  is beatable by learning; it is never loosened to make a run pass.
* **The vocabulary** is the tokenizer's (101 for ``ascii95``), never the model's default of 100,277, and the summary names the
  tokenizer, its digest and the stream rule.
* **The loss** is the library wrapper's: the stock compiled loss of the functional model equals
  ``CausalLanguageModel.compute_loss`` of the same backbone, weights and batch.
* **Refusals.** A tokenizer directory that holds another hash, a missing one, an unknown dataset, a model key the family refuses,
  a ``seq_len`` above the position table, a missing TFDS directory, a family of another task and a conformal level are
  refused before the run directory is claimed (``out`` does not exist afterwards); a corpus too small to hold a selection window
  is refused by name after the data is read, and says that ``out`` is used.
* **The seams.** ``fit_kwargs`` states ``steps_per_epoch`` and ``validation_steps``; ``debug_args.smoke`` reaches the data's
  ``max_samples`` (the key the shared sequence caps); ``python -m train language_model_pretraining help`` lists the keys, and the
  three new modules import neither TensorFlow nor Keras nor ``train.common``.

Tests write only under ``tmp_path``.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

import train
from dl_techniques.models.common.lm_pretraining.clm import CausalLanguageModel
from dl_techniques.models.language.gpt2.gpt2 import GPT2
from dl_techniques.utils.logger import logger
from train.api import cli, registry
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
from train.api.tasks.language_model_pretraining import (
    LanguageModelPretrainingConfig,
    LanguageModelPretrainingTrainer,
    train_language_model_pretraining,
)
from train.api.text_models import create_gpt2_lm, language_loss_and_metrics
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import text_run

from ._text_world import ASCII_VOCAB, BATCH, SEQ_LEN, TINY_MODEL_ARGS, prepared, request, world, wrong_tokenizer_dir  # noqa: F401
from .test_text_baselines import constant_language_model

SRC = Path(train.__file__).resolve().parents[1]
TASK = "language_model_pretraining"


# ---------------------------------------------------------------------
# The hook table
# ---------------------------------------------------------------------

#: The optional hooks this task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",           # the train stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",        # a resumed fit starts a fresh iterator, which must continue the stream, not replay it
    "bar_is_met",           # a loss: strictly below, beyond the repo's tie band (the default is >=)
    "run_analysis",         # data-free (the stock analysis reads class probabilities)
    "summary_fields",       # the tokenizer, the sequence length, the document counts, the stream rule
    "task_notes",           # what val_loss, the bar and the perplexities are
    "recipe_notices",       # the recipe's numbers that no run measured
    "used_arguments",       # the model keywords the run was built with
    "check_model_args",     # the refusals before the claim
    "epoch_line_metrics",   # the epoch line prints the token accuracy too
    "unjudged_note",        # why init_scale_warning is null
)


def test_the_class_names_its_task_and_its_metric():
    assert LanguageModelPretrainingTrainer.task == TASK
    assert LanguageModelPretrainingTrainer.success_metric == "val_loss"
    assert issubclass(LanguageModelPretrainingTrainer, TaskTrainer)


@pytest.mark.parametrize("hook", REQUIRED_HOOKS)
def test_every_required_hook_is_overridden(hook):
    assert getattr(LanguageModelPretrainingTrainer, hook) is not getattr(TaskTrainer, hook), hook


@pytest.mark.parametrize("hook", OPTIONAL_HOOKS)
def test_every_optional_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(LanguageModelPretrainingTrainer, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_does_not_name_a_stage():
    own = {name for name in vars(LanguageModelPretrainingTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim", "train_step"}


# ---------------------------------------------------------------------
# One tiny run
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny(world, tmp_path_factory):  # noqa: F811
    """A 3-epoch run on the synthetic bank with the data-free model analysis and the figures on (a user's defaults)."""
    out = tmp_path_factory.mktemp("language_model") / "run"
    asked = request(out, **{"analysis_args.model_analysis": True, "analysis_args.figures": True})
    result = train_language_model_pretraining(**asked)
    summary = json.loads((out / "results_summary.json").read_text())
    return SimpleNamespace(result=result, out=out, summary=summary, asked=asked)


def test_a_tiny_run_exits_zero_and_meets_the_bar(tiny):
    summary = tiny.summary
    assert tiny.result.success is True and summary["success"] is True and summary["status"] == "ok"
    assert (summary["task"], summary["model_family"], summary["success_metric"]) == (TASK, "gpt2_lm", "val_loss")
    reached, threshold = summary["success_value"], summary["success_threshold"]
    assert np.float32(reached) < np.float32(threshold)
    baseline, epoch_zero = summary["baseline"]["val_loss"], summary["epoch_zero"]["val_loss"]
    assert threshold == min(baseline, epoch_zero)
    logger.info(
        f"LM-TINY best val_loss {reached:.6f} bar {threshold:.6f} (unigram {baseline:.6f}, epoch zero {epoch_zero:.6f}) "
        f"perplexity {math.exp(reached):.4f} against the unigram's {math.exp(baseline):.4f}; "
        f"epochs {summary['epochs_run']} steps/epoch {summary['steps_per_epoch']} vocab {summary['tokenizer']['vocab_size']}")
    assert reached < baseline - 0.2, "the bank has structure a unigram cannot use: a model that learns beats it by far"


def test_the_unigram_is_the_entropy_of_the_selection_targets_counted_from_the_data(tiny, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "again")
    data = trainer.build_data(resolved, context)
    counted = text_run.unigram_val_loss(data.split.select[1], ASCII_VOCAB)
    assert tiny.summary["baseline"] == {"kind": "unigram", "val_loss": counted}
    assert trainer._baseline == counted
    assert tiny.summary["n_val"] == len(data.split.select[0]) == data.n_select


def test_the_summary_names_the_tokenizer_the_sequence_and_the_stream_rule(tiny):
    summary, spec = tiny.summary, tt.tokenizer_spec("ascii95")
    assert summary["tokenizer"] == {
        "encoding": "ascii95", "n_vocab": spec.n_vocab, "vocab_size": spec.vocab_size, "eot_id": spec.eot_id,
        "cls_id": spec.cls_id, "sep_id": spec.sep_id, "pad_id": spec.pad_id, "mask_id": spec.mask_id,
        "files": {}, "digest": spec.digest,
    }
    assert (summary["seq_len"], summary["input_shape"]) == (SEQ_LEN, [SEQ_LEN])
    assert "num_classes" not in summary, "a language model has no classes (never a stand-in count)"
    assert summary["documents"]["train"] + summary["documents"]["select"] == tiny.asked["data"]["max_samples"]
    rule = summary["stream"]["rule"]
    assert "(seed, step)" in rule and "never" in rule
    assert any("not judged" in note for note in summary["notes"]), "the untrained loss against a unigram is no scale test"
    assert summary["init_scale_warning"] is None and summary["initial_loss_ratio"] > 1


def test_the_vocabulary_is_the_tokenizers_never_the_models_default(tiny):
    model = keras.models.load_model(tiny.out / "best_model.keras", compile=False)
    backbone = model.get_layer("gpt2_backbone")
    assert backbone.vocab_size == ASCII_VOCAB != GPT2.DEFAULT_VOCAB_SIZE
    assert model.output_shape == (None, SEQ_LEN, ASCII_VOCAB)
    assert model.inputs[0].dtype == "float32" and model.input_shape == (None, SEQ_LEN)
    assert [weight.path.split("/")[0] for weight in model.weights][0] == "gpt2_backbone"


def test_an_epoch_is_the_streams_steps_and_the_selection_pass_is_stated(tiny, tmp_path):
    summary = tiny.summary
    assert summary["steps_per_epoch"] == summary["n_train"] // BATCH
    rows = (tiny.out / "training_log.csv").read_text().splitlines()
    assert len(rows) == 1 + summary["epochs_run"]
    trainer, resolved, context = prepared(tmp_path / "seams")
    data = trainer.build_data(resolved, context)
    assert trainer.fit_kwargs(resolved, data) == {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}
    assert data.val_steps == -(-data.n_select // BATCH) == int(data.select_ds.cardinality())
    assert int(data.test_ds.cardinality()) == -(-data.n_test // BATCH)
    assert int(data.train_ds.cardinality()) == tf.data.INFINITE_CARDINALITY, "the train stream is infinite"


def test_the_data_free_analysis_ran_on_the_functional_model(tiny):
    analyzer = tiny.summary["analyzer"]
    assert analyzer["status"] == "ok" and analyzer["analyzers"] == ["weights", "spectral"], analyzer
    assert any("data-free" in note for note in tiny.summary["notes"])


def test_the_figures_of_the_default_request_are_drawn_for_a_model_with_no_image(tiny):
    visualizations = tiny.summary["visualizations"]
    assert visualizations["skipped"] is False and visualizations["failed"] == []
    assert visualizations["files"], "the training dashboard reads loss and val_loss, which a language run reports"
    assert tiny.summary["model_loading_validated"] is True and (tiny.out / "final_model.keras").is_file()


# ---------------------------------------------------------------------
# The bar: a constant model and a tie miss it
# ---------------------------------------------------------------------


def test_a_compiled_constant_unigram_model_does_not_meet_the_bar_and_a_trained_run_does(tiny, tmp_path):
    trainer, resolved, context = prepared(tmp_path / "bar")
    data = trainer.build_data(resolved, context)
    targets = data.split.select[1]
    share = np.bincount(targets.reshape(-1), minlength=ASCII_VOCAB) / targets.size
    constant = constant_language_model(np.log(np.maximum(share, 1e-30)), SEQ_LEN)
    val_loss = float(constant.evaluate(data.select_ds, verbose=0, return_dict=True)["loss"])
    unigram = trainer._baseline
    assert abs(val_loss - unigram) < 1e-4, "the constant model IS the unigram (the control: it scores the baseline)"
    assert trainer.bar_is_met(resolved, val_loss, unigram) is False, "a constant model never meets the bar"
    assert trainer.bar_is_met(resolved, unigram, unigram) is False, "equal to the threshold is a miss"
    assert trainer.bar_is_met(resolved, float("nan"), unigram) is False
    assert trainer.bar_is_met(resolved, tiny.summary["success_value"], unigram) is True
    assert trainer.bar_is_met(resolved, unigram - 1.0, unigram) is True


def test_the_threshold_needs_the_epoch_zero_term_the_reporting_reads(tmp_path):
    trainer, resolved, _ = prepared(tmp_path / "early")
    with pytest.raises(RuntimeError, match="baseline"):
        trainer.success_threshold(resolved)
    trainer._baseline = 2.5
    with pytest.raises(RuntimeError, match="epoch-zero"):
        trainer.success_threshold(resolved)
    trainer._epoch_zero = 3.0
    assert trainer.success_threshold(resolved) == 2.5
    trainer._epoch_zero = 2.0
    assert trainer.success_threshold(resolved) == 2.0
    assert trainer.success_value({"val_loss": [3.0, 2.0, 2.5]}) == 2.0


# ---------------------------------------------------------------------
# The loss is the library wrapper's
# ---------------------------------------------------------------------


def test_the_stock_compiled_loss_equals_the_library_causal_language_models(world, tmp_path):  # noqa: F811
    trainer, resolved, context = prepared(tmp_path / "parity")
    data = trainer.build_data(resolved, context)
    x, y = data.split.test[0][:BATCH], data.split.test[1][:BATCH]
    keras.utils.set_random_seed(3)
    model = create_gpt2_lm(variant="tiny", input_shape=(SEQ_LEN,), vocab_size=ASCII_VOCAB, **TINY_MODEL_ARGS)
    loss, metrics = language_loss_and_metrics()
    model.compile(optimizer="sgd", loss=loss, metrics=metrics)
    stock = float(model.evaluate(x, y, batch_size=BATCH, verbose=0, return_dict=True)["loss"])
    wrapper = CausalLanguageModel(
        backbone=model.get_layer("gpt2_backbone"), vocab_size=ASCII_VOCAB, skip_head=True, output_key="logits",
        pre_shifted=True)
    library = float(keras.ops.convert_to_numpy(wrapper.compute_loss(y=y, y_pred=wrapper(x, training=False))))
    logger.info(f"LM-PARITY stock {stock:.9f} library {library:.9f} abs diff {abs(stock - library):.3e}")
    assert abs(stock - library) <= 1e-6, (stock, library)
    assert stock > math.log(ASCII_VOCAB) - 0.5, "an untrained model sits near ln(vocab): the control that the loss is a real loss"
    # the control: a loss that took the logits for probabilities is far from the library's number
    wrong = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="sgd", loss=wrong)
    assert abs(float(model.evaluate(x, y, batch_size=BATCH, verbose=0, return_dict=True)["loss"]) - library) > 0.05


# ---------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------

#: ``(id, changes, text the message must contain)``: each is refused with ``out`` unclaimed.
BEFORE_THE_CLAIM = [
    ("tokenizer_dir_other_hash", {"data.tokenizer": "gpt2", "data.tokenizer_dir": "{wrong}"}, "has sha256"),
    ("tokenizer_dir_missing", {"data.tokenizer": "gpt2", "data.tokenizer_dir": "{missing}"}, "is not a directory"),
    ("tokenizer_dir_for_ascii95", {"data.tokenizer_dir": "{missing}"}, "data.tokenizer_dir"),
    ("unknown_dataset", {"data.dataset": "wikipedia"}, "imdb_reviews"),
    ("unknown_tokenizer", {"data.tokenizer": "cl100k_base"}, "ascii95"),
    ("a_key_the_family_refuses", {"model_args": {**TINY_MODEL_ARGS, "include_top": True}}, "model_args.include_top"),
    ("the_vocabulary_is_the_tokenizers", {"model_args": {**TINY_MODEL_ARGS, "vocab_size": 50}}, "set by the task"),
    ("an_unknown_model_key", {"model_args": {**TINY_MODEL_ARGS, "bogus": 1}}, "model_args.bogus"),
    ("seq_len_above_the_position_table", {"data.seq_len": 513}, "max_seq_len"),
    ("a_smaller_position_table", {"model_args": {**TINY_MODEL_ARGS, "max_seq_len": 16}}, "max_seq_len"),
    ("conformal_level", {"analysis_args.conformal_alpha": 0.1}, "conformal"),
    ("a_metric_that_is_not_the_bar", {"metric_args.watch_metric": "val_accuracy"}, "val_loss"),
    ("a_family_of_another_task", {"model": "vit/vit_pico"}, "language_model_pretraining"),
    ("a_share_of_nothing", {"data.validation_split": 1.0}, "data.validation_split"),
    ("a_zero_length", {"data.seq_len": 0}, "data.seq_len"),
    ("an_sgd_decay_no_regularizer_carries", {"optim_args.optimizer": "sgd", "optim_args.lr": 0.1, "optim_args.weight_decay": 0.01},
     "weight_decay"),
]


@pytest.mark.parametrize("name, changes, text", BEFORE_THE_CLAIM, ids=[row[0] for row in BEFORE_THE_CLAIM])
def test_a_bad_request_is_refused_before_the_run_is_claimed(name, changes, text, world, tmp_path):  # noqa: F811
    fills = {"wrong": str(wrong_tokenizer_dir(tmp_path)), "missing": str(tmp_path / "nowhere")}
    changes = {key: value.format(**fills) if isinstance(value, str) else value for key, value in changes.items()}
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="") as caught:
        train_language_model_pretraining(**request(out, **changes))
    assert text in str(caught.value), f"{name}: {caught.value}"
    assert not out.exists(), f"{name}: the run directory was claimed"


def test_a_missing_tfds_directory_is_refused_before_the_claim_and_names_it(world, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "no_tfds"))
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="imdb_reviews") as caught:
        train_language_model_pretraining(**request(out, **{"data.dataset": "imdb_reviews"}))
    assert "no_tfds" in str(caught.value) and "TFDS_DATA_DIR" in str(caught.value)
    assert not out.exists()


@pytest.mark.skipif(not (Path(os.environ.get("TFDS_DATA_DIR", "")) / "imdb_reviews").is_dir(),
                    reason="imdb_reviews is not under TFDS_DATA_DIR on this machine")
def test_the_tfds_directory_of_this_machine_passes_the_pre_claim_check_and_a_prefix_of_the_reviews_reads(world, tmp_path):  # noqa: F811
    trainer, resolved, _ = prepared(tmp_path / "imdb", **{"data.dataset": "imdb_reviews", "data.max_samples": 40})
    facts = trainer.describe_data(resolved)
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("imdb_reviews", None, None)
    from train.common import text_data

    (texts, labels), (test_texts, _) = text_data.corpus_documents("imdb_reviews", text_data.LM, max_samples=7)
    assert len(texts) == len(labels) == len(test_texts) == 7 and all(isinstance(text, str) and text for text in texts)


def test_a_request_that_is_valid_is_resolved_and_the_neighbouring_values_pass(world, tmp_path):  # noqa: F811
    # the other half of each refusal: seq_len equal to the table is accepted, and the position table is the variant's 512
    trainer, resolved, _ = prepared(tmp_path / "ok", **{"data.seq_len": 512})
    assert resolved.data.seq_len == 512 and resolved.model_args == TINY_MODEL_ARGS
    facts = trainer.describe_data(resolved)
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("synthetic", None, None)
    assert resolved.optim_args.optimizer == "adamw" and resolved.optim_args.lr == 3e-3
    assert resolved.optim_args.global_clipnorm == 1.0 and resolved.optim_args.warmup_epochs == 1
    assert resolved.metric_args.watch_metric == "val_loss" and resolved.metric_args.watch_mode == "min"


def test_a_corpus_too_small_for_a_selection_window_is_refused_by_name_after_the_data_is_read(world, tmp_path):  # noqa: F811
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        train_language_model_pretraining(**request(out, **{"data.max_samples": 1}))
    message = str(caught.value)
    assert "data.max_samples" in message and "'out' is now used" in message
    assert out.exists(), "this refusal comes after the claim, and the message says so"


def test_the_command_line_exits_two_and_claims_nothing_for_a_tokenizer_directory_of_another_hash(world, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "cli_run"
    code = cli.main([
        TASK, f"out={out}", "model=gpt2_lm/tiny", "data.dataset=synthetic", "data.tokenizer=gpt2",
        f"data.tokenizer_dir={wrong_tokenizer_dir(tmp_path)}"])
    assert code == 2 and not out.exists()
    assert "has sha256" in capsys.readouterr().err


def test_the_command_line_exits_two_for_an_unknown_dataset(world, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "cli_run"
    assert cli.main([TASK, f"out={out}", "model=gpt2_lm/tiny", "data.dataset=wikipedia"]) == 2 and not out.exists()
    assert "imdb_reviews" in capsys.readouterr().err


def test_a_family_of_this_task_is_refused_under_another_task(world, tmp_path):  # noqa: F811
    with pytest.raises(ConfigError, match="language_model_pretraining"):
        registry.parse_model_name("gpt2_lm/tiny", task="image_classification")
    registry.parse_model_name("gpt2_lm/tiny", task=TASK)


# ---------------------------------------------------------------------
# Smoke, help and imports
# ---------------------------------------------------------------------


def test_a_smoke_run_caps_the_documents_through_the_key_the_sequence_reads_and_the_command_line_agrees(world, tmp_path):  # noqa: F811
    out, cli_out = tmp_path / "smoke", tmp_path / "smoke_cli"
    data = {"dataset": "synthetic", "tokenizer": "ascii95", "seq_len": SEQ_LEN}
    result = train_language_model_pretraining(**request(out, data=data, epochs="auto", **{"debug_args.smoke": True}))
    summary = json.loads((out / "results_summary.json").read_text())
    config = json.loads((out / "config.json").read_text())
    assert config["data"]["max_samples"] == 512, "the shared sequence caps data.max_samples under smoke"
    assert result.success is True and summary["success"] is None and summary["epochs_run"] == 1
    assert summary["documents"]["train"] + summary["documents"]["select"] == 512
    # the same request typed as a command line: the function and the command cannot disagree
    code = cli.main([
        TASK, f"out={cli_out}", "model=gpt2_lm/tiny", "data.dataset=synthetic", "data.tokenizer=ascii95", f"data.seq_len={SEQ_LEN}",
        f"batch_size={BATCH}", *(f"model_args.{key}={value}" for key, value in TINY_MODEL_ARGS.items()), "optim_args.lr=3e-3",
        "analysis_args.model_analysis=false", "analysis_args.figures=false", "debug_args.smoke=true"])
    assert code == 0
    cli_config = json.loads((cli_out / "config.json").read_text())
    for each in (config, cli_config):
        each.pop("out")
    assert cli_config == config
    assert json.loads((cli_out / "results_summary.json").read_text())["success"] is None


def test_the_help_lists_the_keys(world, capsys):  # noqa: F811
    assert cli.main([TASK, "help"]) == 0
    text = capsys.readouterr().out
    for key in ("data.dataset", "data.tokenizer", "data.tokenizer_dir", "data.seq_len", "data.validation_split",
                "data.max_samples", "model_args", "optim_args.lr", "metric_args.watch_metric", "resume_interrupted"):
        assert key in text, key
    assert "init" in text


def child(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)


def test_the_new_modules_import_no_heavy_module():
    done = child(
        "import sys\n"
        "import train.api.text_args, train.api.text_models, train.api.tasks.language_model_pretraining\n"
        "heavy = sorted(name for name in sys.modules if name.split('.')[0] in ('tensorflow', 'keras', 'tf_keras', 'tensorboard',"
        " 'yaml', 'tiktoken', 'numpy') or name == 'train.common' or name.startswith(('train.common.', 'dl_techniques.models')))\n"
        "print('HEAVY', heavy)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "HEAVY []", done.stdout[-800:]
