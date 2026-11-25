"""The ``text_classification`` task: a tiny run per family, the majority bar, the stream, the refusals and the seams.

``train.api.tasks.text_classification.TextClassificationTrainer`` (plan-2026-11-24T092348-7806a290 step 10) trains a text classifier,
``bert`` or ``gpt2`` (:mod:`train.api.text_models`), a functional model of float32 ids to class logits, with the stock ``fit`` and the
stock sparse cross entropy, on rows ``[CLS]`` + the head of a document + pad. The data is the offline labelled ``synthetic`` bank
(60 percent of the documents are class 0) under the character tokenizer ``ascii95`` (``_text_world``: 64 wide, 2 blocks, rows of
32), so every run here is a CPU run of seconds. What is pinned:

* **The run.** A tiny run of each family exits 0 and meets the bar: the best ``val_accuracy`` of the selection split is strictly above
  the share of its majority class. A constant predictor clears CHANCE (1 / classes) and does not meet that bar, nor does a value that
  equals it (Pre-Mortem 9: a bar a constant model meets is rewritten, the model is never weakened).
* **The bar is the SELECTION split's** majority share, not the train rows' (they differ in this bank), and under ``init`` it must also
  beat the loaded weights' epoch-zero accuracy.
* **The stream.** A train batch is a function of ``(seed, step)``; the labels of a row travel with it through the epoch permutation (a
  row that carries its own label in its ids), the split is the language task's for the same seed (same documents), and a selection
  split that lacks a class is refused by name.
* **Refusals** before the claim (a family of another task, a tokenizer directory, a position table, a metric that is not the bar, a
  conformal level, a row too short for ``[CLS]`` and a token) and the seams: ``fit_kwargs``, ``resume_stream``, smoke, the help.

Tests write only under ``tmp_path``.
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest
import tensorflow as tf

import train
from dl_techniques.utils.logger import logger
from train.api import cli, registry, resume_decisions
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
from train.api.tasks.text_classification import TextClassificationTrainer, train_text_classification
from train.api.trainer import REQUIRED_HOOKS, TaskTrainer
from train.common import text_data as td
from train.common import text_run

from ._text_world import (
    BATCH, CLASSIFIER_DOCUMENTS, CLASSIFIER_SPLIT, SEQ_LEN, classifier_prepared, classifier_request, prepared, world,
    wrong_tokenizer_dir)  # noqa: F401

SRC = Path(train.__file__).resolve().parents[1]
TASK = "text_classification"
FAMILIES = ("bert", "gpt2")
ASCII = tt.tokenizer_spec("ascii95")

# ---------------------------------------------------------------------
# The hook table and the registry
# ---------------------------------------------------------------------

#: The optional hooks this task overrides, and why each one is needed.
OPTIONAL_HOOKS = (
    "fit_kwargs",           # the train stream is infinite: an epoch is steps_per_epoch batches
    "resume_stream",        # a resumed fit starts a fresh iterator, which must continue the stream, not replay it
    "bar_is_met",           # strictly above the majority share (the default is >=), and above the loaded weights under init
    "run_analysis",         # data-free (the stock analysis reads images)
    "summary_fields",       # the tokenizer, the classes, their counts and the truncation
    "task_notes",           # what val_accuracy and the bar are
    "recipe_notices",       # the recipe's numbers that no run measured
    "used_arguments",       # the model keywords the run was built with
    "check_model_args",     # the refusals before the claim
    "epoch_line_metrics",   # the epoch line prints the accuracy
    "init_identity",        # the vocabulary and the tokenizer's digest an init source must share (D-040)
    "init_overlap",         # the source's train documents among this run's selection documents
)


def test_the_class_names_its_task_and_its_metric():
    assert TextClassificationTrainer.task == TASK
    assert TextClassificationTrainer.success_metric == "val_accuracy"
    assert issubclass(TextClassificationTrainer, TaskTrainer)


@pytest.mark.parametrize("hook", [*REQUIRED_HOOKS, *OPTIONAL_HOOKS])
def test_every_hook_the_task_relies_on_is_overridden(hook):
    assert getattr(TextClassificationTrainer, hook) is not getattr(TaskTrainer, hook), hook


def test_the_task_does_not_name_a_stage():
    own = {name for name in vars(TextClassificationTrainer) if not name.startswith("__")}
    assert not own & {"run", "_resolve", "_callbacks", "_summary_head", "_notes", "_refuse_before_the_claim", "train_step"}


def test_the_task_and_both_families_are_registered_and_each_family_trains_this_task_only(world):  # noqa: F811
    assert registry.TASKS[TASK].name == TASK and list(registry.TASKS)[-1] == TASK
    for family in FAMILIES:
        assert registry.MODELS[family].tasks == (TASK,)
        assert registry.parse_model_name(f"{family}/tiny", task=TASK) == (family, "tiny")
        with pytest.raises(ConfigError, match=TASK):
            registry.parse_model_name(f"{family}/tiny", task="language_model_pretraining")
    with pytest.raises(ConfigError, match="text_classification"):
        registry.parse_model_name("gpt2_lm/tiny", task=TASK)


# ---------------------------------------------------------------------
# One tiny run per family
# ---------------------------------------------------------------------


@pytest.fixture(scope="module", params=FAMILIES)
def tiny(request, world, tmp_path_factory):  # noqa: F811
    """An 8-epoch run on the labelled bank with the data-free analysis and the figures on (a user's defaults)."""
    out = tmp_path_factory.mktemp(f"classifier_{request.param}") / "run"
    asked = classifier_request(out, request.param, **{"analysis_args.model_analysis": True, "analysis_args.figures": True})
    result = train_text_classification(**asked)
    summary = json.loads((out / "results_summary.json").read_text())
    return SimpleNamespace(family=request.param, result=result, out=out, summary=summary, asked=asked)


def selection_labels(trainer, resolved, context):
    data = trainer.build_data(resolved, context)
    return data, data.split.select[1]


def test_a_tiny_run_exits_zero_and_meets_the_majority_bar(tiny, tmp_path):
    summary = tiny.summary
    assert tiny.result.success is True and summary["success"] is True and summary["status"] == "ok"
    assert (summary["task"], summary["model_family"], summary["success_metric"]) == (TASK, tiny.family, "val_accuracy")
    trainer, resolved, context = classifier_prepared(tmp_path / "again", tiny.family)
    data, labels = selection_labels(trainer, resolved, context)
    share = float(np.bincount(labels).max() / len(labels))
    reached, threshold, baseline = summary["success_value"], summary["success_threshold"], summary["baseline"]
    margin = text_run.accuracy_margin(share, len(labels))
    assert (baseline["val_accuracy"], baseline["margin"], baseline["standard_errors"], baseline["rows"]) == (share, margin, 3.0, len(labels))
    assert threshold == share + margin and 0.5 < share < threshold < 1.0
    assert np.float32(reached) > np.float32(threshold), "strictly above the majority share of the selection split plus its margin"
    logger.info(f"CLS-TINY {tiny.family}: best val_accuracy {reached:.4f} bar (share plus margin) {threshold:.4f}, "
                f"epochs {summary['epochs_run']} steps/epoch {summary['steps_per_epoch']} selection rows {data.n_select}")
    assert summary["n_val"] == data.n_select == len(labels) and summary["num_classes"] == 2
    assert reached >= 0.9, "the bank's cue words separate the classes: a classifier that learns scores far above 0.6"


def test_the_summary_names_the_tokenizer_the_classes_their_counts_and_the_truncation(tiny):
    summary = tiny.summary
    assert summary["tokenizer"] == ASCII.record() and (summary["seq_len"], summary["input_shape"]) == (SEQ_LEN, [SEQ_LEN])
    assert summary["class_names"] == ["neg", "pos"]
    assert summary["corpus_sha256"] == json.loads((tiny.out / "checkpoints" / "fit_state.json").read_text())["data_fingerprint"]    # N6
    counts = summary["class_counts"]
    assert set(counts) == {"train", "select", "test"} and all(len(row) == 2 for row in counts.values())
    assert sum(counts["train"]) + sum(counts["select"]) == tiny.asked["data"]["max_samples"]
    assert summary["documents"] == {"train": sum(counts["train"]), "select": sum(counts["select"]), "test": sum(counts["test"])}
    # a character is an id and a row keeps 31 of them: the documents cut are those of more than 31 characters (all but one here)
    texts = [text for part in td.corpus_documents("synthetic", td.CLASSIFICATION, max_samples=CLASSIFIER_DOCUMENTS) for text in part[0]]
    assert summary["truncated_documents"] == sum(len(text) > SEQ_LEN - 1 for text in texts) < len(texts) == sum(summary["documents"].values())
    assert summary["baseline"]["class"] == "neg" and "(seed, step)" in summary["stream"]["rule"]
    assert 0.5 < summary["initial_loss_ratio"] < 2 and isinstance(summary["init_scale_warning"], bool)


def test_the_model_is_the_families_classifier_over_the_tokenizers_vocabulary(tiny):
    model = keras.models.load_model(tiny.out / "best_model.keras", compile=False)
    assert model.output_shape == (None, 2) and model.input_shape == (None, SEQ_LEN) and model.inputs[0].dtype == "float32"
    assert model.get_layer(f"{tiny.family}_backbone" if tiny.family == "bert" else "gpt2_backbone") is not None
    assert [weight.path.split("/")[0] for weight in model.weights][0].endswith("_backbone")
    assert any(weight.path.split("/")[-3:-2] == ["task_head_classifier"] or "task_head" in weight.path for weight in model.weights)


def test_the_data_free_analysis_and_the_figures_ran(tiny):
    assert tiny.summary["analyzer"]["status"] == "ok" and tiny.summary["analyzer"]["analyzers"] == ["weights", "spectral"]
    assert any("data-free" in note for note in tiny.summary["notes"])
    assert tiny.summary["visualizations"]["skipped"] is False and tiny.summary["visualizations"]["failed"] == []
    assert tiny.summary["model_loading_validated"] is True and (tiny.out / "final_model.keras").is_file()
    assert tiny.summary["test_metrics_best"]["accuracy"] > 0.5, "scored on the corpus's own test documents"


# ---------------------------------------------------------------------
# The bar: a constant predictor and a tie miss it
# ---------------------------------------------------------------------


def constant_classifier(logits, seq_len=SEQ_LEN):
    """A model that answers the same logits to every row, compiled as the task compiles."""
    inputs = keras.Input((seq_len,), dtype="float32")
    head = keras.layers.Dense(len(logits), kernel_initializer="zeros", bias_initializer=keras.initializers.Constant(0.0))
    model = keras.Model(inputs, head(inputs))
    head.set_weights([np.zeros((seq_len, len(logits)), np.float32), np.asarray(logits, np.float32)])
    loss, metrics = keras.losses.SparseCategoricalCrossentropy(from_logits=True), [keras.metrics.SparseCategoricalAccuracy("accuracy")]
    model.compile(optimizer="sgd", loss=loss, metrics=metrics)
    return model


def test_a_constant_predictor_clears_chance_and_does_not_meet_the_bar_and_a_trained_run_does(tiny, tmp_path):
    trainer, resolved, context = classifier_prepared(tmp_path / "bar", tiny.family)
    data, labels = selection_labels(trainer, resolved, context)
    majority = int(np.bincount(labels).argmax())
    logits = [10.0 if index == majority else 0.0 for index in range(2)]
    scored = float(constant_classifier(logits).evaluate(data.select_ds, verbose=0, return_dict=True)["accuracy"])
    threshold = trainer.success_threshold(resolved)
    assert np.float32(scored) == np.float32(text_run.majority_share(labels)), "the constant predictor scores the majority share to the last bit"
    assert scored > 1 / 2, "the control: it clears CHANCE (a bar at 1 / classes calls it a success), the bank is imbalanced"
    assert trainer.bar_is_met(resolved, scored, threshold) is False, "a constant predictor never meets the bar"
    assert trainer.bar_is_met(resolved, scored + 0.01, threshold) is False, "nor does a model a point above the share (W2)"
    assert trainer.bar_is_met(resolved, threshold, threshold) is False, "equal to the threshold is a miss"
    assert trainer.bar_is_met(resolved, float("nan"), threshold) is False
    assert trainer.bar_is_met(resolved, tiny.summary["success_value"], threshold) is True
    assert trainer.bar_is_met(resolved, 1.0, threshold) is True
    # under init the success value must also beat the loaded weights' epoch zero, in float32, beyond the tie band
    assert trainer.bar_is_met(resolved, 0.95, threshold, loaded=0.95) is False
    assert trainer.bar_is_met(resolved, 0.95, threshold, loaded=0.5) is True


def test_the_threshold_is_the_selection_splits_majority_share_never_the_train_rows_or_chance(world, tmp_path):  # noqa: F811
    trainer, resolved, context = classifier_prepared(tmp_path / "share")
    data, labels = selection_labels(trainer, resolved, context)
    train_share = float(np.bincount(data.split.train_labels).max() / len(data.split.train_labels))
    share = text_run.majority_share(labels)
    assert trainer.success_threshold(resolved) == share + text_run.accuracy_margin(share, len(labels))
    assert share != train_share and share != 1 / 2, "the control: the selection split's share is neither of the wrong bars"
    assert trainer.success_value({"val_accuracy": [0.4, 0.7, 0.6]}) == 0.7


def test_the_threshold_asks_for_the_data_the_baseline_is_cut_from(world, tmp_path):  # noqa: F811
    trainer, resolved, _ = classifier_prepared(tmp_path / "early")
    with pytest.raises(RuntimeError, match="build_data"):
        trainer.success_threshold(resolved)


# ---------------------------------------------------------------------
# The stream: rows, labels, the split
# ---------------------------------------------------------------------


def tagged(n=40):
    """``n`` documents that carry their own label and index in their ids: document ``i`` is ``[5 + label_i, 10 + i]``."""
    labels = np.array([(i * 3) % 5 % 2 for i in range(n)], np.int64)
    return [np.array([5 + labels[i], 10 + i], np.int32) for i in range(n)], labels


def tagged_split(**changes):
    docs, labels = tagged()
    asked = dict(spec=ASCII, seq_len=6, batch_size=4, seed=3, validation_split=0.3, num_classes=2)
    return td.class_split(docs, labels, docs[:6], labels[:6], **{**asked, **changes}), docs, labels


def test_a_row_keeps_its_label_through_the_permutation_of_every_epoch():
    split, docs, labels = tagged_split()
    stream = split.stream
    rows = {}
    for step in range(3 * stream.steps_per_epoch):
        x, y = stream.batch_at(step)
        assert (x[:, 0] == ASCII.cls_id).all() and (x[:, 4:] == ASCII.pad_id).all()
        assert (y == x[:, 1] - 5).all(), f"step {step}: a label moved without its row"
        assert (labels[x[:, 2] - 10] == y).all(), f"step {step}: the label is not the label of the document the row came from"
        rows.setdefault(step // stream.steps_per_epoch, []).extend(x[:, 2].tolist())
    kept = stream.steps_per_epoch * stream.batch_size  # the rows of a ragged tail are not read in that epoch
    assert all(len(set(order)) == len(order) == kept and set(order) <= set(rows[0] + rows[1] + rows[2]) for order in rows.values())
    assert rows[0] != rows[1] != rows[2], "the control: each epoch is another order"
    assert split.train_labels.tolist() == labels[split.train_index].tolist()


def test_the_batch_at_a_step_is_a_pure_function_of_the_seed_and_the_step_and_the_split_never_follows_the_stream_seed():
    first, second, other, restreamed = tagged_split()[0], tagged_split()[0], tagged_split(seed=4)[0], tagged_split(stream_seed=9)[0]
    one, two = first.stream, second.stream
    steps = 2 * one.steps_per_epoch + 1
    for step in (0, 1, one.steps_per_epoch, steps):
        np.testing.assert_array_equal(one.batch_at(step)[0], two.batch_at(step)[0])
    assert not all(np.array_equal(one.batch_at(s)[0], other.stream.batch_at(s)[0]) for s in range(steps)), "another seed, another order"
    resumed = one.with_start(steps)
    for position, (x, y) in zip(range(3), resumed.iterate()):
        np.testing.assert_array_equal(x, one.batch_at(steps + position)[0])
        np.testing.assert_array_equal(y, one.batch_at(steps + position)[1])
    # another stream seed orders the same train rows differently and moves no selection document
    np.testing.assert_array_equal(restreamed.select_index, first.select_index)
    np.testing.assert_array_equal(restreamed.select[0], first.select[0])
    assert not all(np.array_equal(restreamed.stream.batch_at(s)[0], one.batch_at(s)[0]) for s in range(steps))


def test_the_selection_documents_are_the_language_tasks_for_the_same_seed_and_split(world, tmp_path):  # noqa: F811
    trainer, resolved, context = classifier_prepared(tmp_path / "cls")
    same = {"data.max_samples": CLASSIFIER_DOCUMENTS, "data.validation_split": CLASSIFIER_SPLIT}
    lm_trainer, lm_resolved, lm_context = prepared(tmp_path / "lm", **same)
    ours, theirs = trainer.build_data(resolved, context), lm_trainer.build_data(lm_resolved, lm_context)
    oracle = td.split_documents(resolved.data.max_samples, resolved.seed, resolved.data.validation_split)
    for part in range(2):
        np.testing.assert_array_equal([ours.split.train_index, ours.split.select_index][part], oracle[part])
        np.testing.assert_array_equal([ours.split.train_index, ours.split.select_index][part],
                                      [theirs.split.train_index, theirs.split.select_index][part])
    assert ours.n_train + ours.n_select == resolved.data.max_samples and ours.num_classes == 2
    assert ours.input_shape == (SEQ_LEN,) and (ours.n_select, ours.n_train) == (len(oracle[1]), len(oracle[0]))


def test_a_part_that_lacks_a_class_is_refused_by_name_with_its_count_and_a_stream_of_less_than_a_batch_too():
    docs, labels = tagged()
    _, select_index = td.split_documents(len(docs), 3, 0.3)
    lacking = labels.copy()
    lacking[select_index] = 0   # the selection part holds class 0 alone, the train part both
    with pytest.raises(ValueError, match=rf"selection documents \({len(select_index)} of them\) hold no document of class 1"):
        td.class_split(docs, lacking, docs[:6], labels[:6], spec=ASCII, seq_len=6, batch_size=4, seed=3, validation_split=0.3,
                       num_classes=2)
    with pytest.raises(ValueError, match=r"train documents \(\d+ of them\) hold no document of class 1"):
        td.class_split(docs, np.zeros_like(labels), docs[:6], labels[:6], spec=ASCII, seq_len=6, batch_size=4, seed=3,
                       validation_split=0.3, num_classes=2)
    with pytest.raises(ValueError, match="fewer than one batch of 64"):
        tagged_split(batch_size=64)


def test_a_classifier_split_holds_the_digest_of_its_documents_alone_beside_the_one_with_the_labels():
    # review pass 2, concern 1: a language split digests no labels, so the classifier's split must offer the same digest of the same documents
    split, docs, labels = tagged_split()
    assert split.fingerprint == td.corpus_digest((docs, labels), (docs[:6], labels[:6]))
    assert split.documents_fingerprint == td.corpus_digest((docs, None), (docs[:6], None)) != split.fingerprint


@pytest.mark.parametrize("change", ["a character", "a label"])
def test_a_resume_on_other_documents_is_refused_naming_both_digests_and_the_same_corpus_is_not(change, world, tmp_path, monkeypatch):  # noqa: F811
    # review pass 1, N6: nothing identified the documents a run read, so a resume over another corpus (a TFDS build, a bank change) went on
    trainer, resolved, context = classifier_prepared(tmp_path / "run")
    first, real = trainer.build_data(resolved, context), td.corpus_documents

    def another(dataset, kind, *, max_samples):         # one review with one character or one label changed: every count stays
        (texts, labels), test = real(dataset, kind, max_samples=max_samples)
        if change == "a label":
            return (texts, np.where(np.arange(len(labels)) == 0, 1 - labels, labels)), test
        return ([texts[0][:-2] + "x.", *texts[1:]], labels), test

    monkeypatch.setattr(td, "corpus_documents", another)
    second = trainer.build_data(resolved, context)
    state = {"data_counts": resume_decisions.data_counts(first), **resume_decisions.data_fingerprint_field(resolved, first)}
    assert resume_decisions.data_counts(second) == state["data_counts"] and len(first.fingerprint) == 64 and first.fingerprint != second.fingerprint
    resume_decisions.check_data_unchanged(tmp_path, resolved, None, first, SimpleNamespace(state=state))
    with pytest.raises(ConfigError, match=rf"data_fingerprint: {first.fingerprint[:12]} -> {second.fingerprint[:12]}"):
        resume_decisions.check_data_unchanged(tmp_path, resolved, None, second, SimpleNamespace(state=state))


@pytest.mark.parametrize("language", [False, True], ids=["text_classification", "language_model_pretraining"])
def test_the_refusal_of_a_text_resume_speaks_of_documents_and_a_run_with_no_digest_is_refused_by_that_fact(language, world, tmp_path, monkeypatch):  # noqa: F811
    # review pass 2, concern 4: the shared refusal spoke of files, a CSV and standardizing, and said a language run made before the digest
    # existed had "recorded its files by a rule that could not tell a reordered CSV"
    trainer, resolved, context = prepared(tmp_path / "lm") if language else classifier_prepared(tmp_path / "cls")
    first, real = trainer.build_data(resolved, context), td.corpus_documents

    def another(dataset, kind, *, max_samples):
        (texts, labels), test = real(dataset, kind, max_samples=max_samples)
        return ([texts[0][:-2] + "x.", *texts[1:]], labels), test

    monkeypatch.setattr(td, "corpus_documents", another)
    second = trainer.build_data(resolved, context)
    state = {"data_counts": resume_decisions.data_counts(first), **resume_decisions.data_fingerprint_field(resolved, first)}
    assert state["data_fingerprint_scheme"] == td.CORPUS_SCHEME != "listing-order", "the digest names its own rule"
    for earlier, now, said in ((state, second, "a document, its tokenization, its label or the order"),
                               ({"data_counts": state["data_counts"]}, first, "recorded no digest of its documents")):
        with pytest.raises(ConfigError) as caught:
            resume_decisions.check_data_unchanged(tmp_path, resolved, None, now, SimpleNamespace(state=earlier))
        text = str(caught.value)
        assert said in text and "restore the documents and the tokenizer as they were, or choose a new 'out'" in text, text
        assert not any(word in text for word in ("CSV", "standardize", "files", "listing")), text
    assert "made before the corpus digest existed" in text, "the run with no digest is refused by that fact"


@pytest.mark.parametrize("language", [False, True], ids=["text_classification", "language_model_pretraining"])
def test_the_record_a_task_writes_names_the_tokenizer_directory_only_when_the_run_set_one(language, world, tmp_path):  # noqa: F811
    # review pass 1, N2: the key is what lets `predict` open the BPE files where the run read them
    trainer, resolved, context = prepared(tmp_path / "lm") if language else classifier_prepared(tmp_path / "cls")
    for directory in (None, "/somewhere/bpe", "bpe"):       # set past the request's own check, which would read the files there
        config = dataclasses.replace(resolved, data=dataclasses.replace(resolved.data, tokenizer_dir=directory))
        record = trainer.task_record(config, context, SimpleNamespace(input_shape=(SEQ_LEN,), spec=ASCII))
        assert record.get("tokenizer_dir") == (None if directory is None else str(Path(directory).resolve())), record   # absolute: any cwd
        assert ("tokenizer_dir" in record) == (directory is not None)


def test_a_document_longer_than_the_row_is_cut_to_its_head_and_counted(world, tmp_path):  # noqa: F811
    trainer, resolved, context = classifier_prepared(tmp_path / "cut", **{"data.seq_len": 6})
    data = trainer.build_data(resolved, context)
    ids = data.split.select[0]
    assert ids.shape[1] == 6 and (ids[:, 0] == ASCII.cls_id).all() and not (ids[:, 1:] == ASCII.pad_id).all(axis=1).any()
    assert data.split.truncated > 0 and trainer.summary_fields(resolved, data)["truncated_documents"] == data.split.truncated


# ---------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------

#: ``(id, changes, text the message must contain)``: each is refused with ``out`` unclaimed.
BEFORE_THE_CLAIM = [
    ("tokenizer_dir_other_hash", {"data.tokenizer": "gpt2", "data.tokenizer_dir": "{wrong}"}, "has sha256"),
    ("tokenizer_dir_missing", {"data.tokenizer": "gpt2", "data.tokenizer_dir": "{missing}"}, "is not a directory"),
    ("unknown_dataset", {"data.dataset": "wikipedia"}, "imdb_reviews"),
    ("a_key_the_family_refuses", {"model_args": {"embed_dim": 64, "depth": 2, "num_heads": 2, "include_top": True}},
     "model_args.include_top"),
    ("the_vocabulary_is_the_tokenizers", {"model_args": {"embed_dim": 64, "depth": 2, "num_heads": 2, "vocab_size": 50}},
     "set by the task"),
    ("seq_len_above_the_gpt2_position_table", {"data.seq_len": 513}, "max_seq_len"),
    ("a_smaller_position_table", {"model_args": {"embed_dim": 64, "depth": 2, "num_heads": 2, "max_seq_len": 16}}, "max_seq_len"),
    ("conformal_level", {"analysis_args.conformal_alpha": 0.1}, "conformal"),
    ("a_metric_that_is_not_the_bar", {"metric_args.watch_metric": "val_loss"}, "val_accuracy"),
    ("a_direction_that_is_not_the_bar", {"metric_args.watch_mode": "min"}, "max"),
    ("a_family_of_another_task", {"model": "gpt2_lm/tiny"}, "text_classification"),
    ("a_row_of_cls_alone", {"data.seq_len": 1}, "data.seq_len"),
    ("a_share_of_nothing", {"data.validation_split": 1.0}, "data.validation_split"),
]


@pytest.mark.parametrize("name, changes, text", BEFORE_THE_CLAIM, ids=[row[0] for row in BEFORE_THE_CLAIM])
def test_a_bad_request_is_refused_before_the_run_is_claimed(name, changes, text, world, tmp_path):  # noqa: F811
    fills = {"wrong": str(wrong_tokenizer_dir(tmp_path)), "missing": str(tmp_path / "nowhere")}
    changes = {key: value.format(**fills) if isinstance(value, str) else value for key, value in changes.items()}
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="") as caught:
        train_text_classification(**classifier_request(out, **changes))
    assert text in str(caught.value), f"{name}: {caught.value}"
    assert not out.exists(), f"{name}: the run directory was claimed"


def test_bert_refuses_a_row_longer_than_its_position_table_before_the_claim(world, tmp_path):  # noqa: F811
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="max_position_embeddings") as caught:
        train_text_classification(**classifier_request(out, "bert", **{"data.seq_len": 513}))
    assert not out.exists() and "513" in str(caught.value)
    trainer, resolved, _ = classifier_prepared(tmp_path / "ok", "bert", **{"data.seq_len": 512})
    assert resolved.data.seq_len == 512 and resolved.model_args["hidden_size"] == 64


def test_a_request_that_is_valid_is_resolved_with_the_classifier_recipe(world, tmp_path):  # noqa: F811
    trainer, resolved, _ = classifier_prepared(tmp_path / "ok", **{"data.seq_len": 512})
    facts = trainer.describe_data(resolved)
    assert (facts.dataset, facts.num_classes, facts.image_size) == ("synthetic", 2, None)
    assert resolved.optim_args.optimizer == "adamw" and resolved.optim_args.global_clipnorm == 1.0
    assert resolved.metric_args.watch_metric == "val_accuracy" and resolved.metric_args.watch_mode == "max"
    assert resolved.checkpoint_args.early_stopping_patience == 4 and resolved.batch_size == BATCH


def test_a_selection_split_that_lacks_a_class_is_refused_by_name_after_the_data_is_read(world, tmp_path):  # noqa: F811
    out = tmp_path / "run"
    with pytest.raises(ConfigError) as caught:
        train_text_classification(**classifier_request(out, **{"data.max_samples": 8}))
    message = str(caught.value)
    assert re.search(r"selection documents \(\d+ of them\) hold no document of class \d", message) and "data.max_samples" in message
    assert "'out' is now used" in message and out.exists(), "this refusal comes after the claim, and the message says so"


def test_the_command_line_exits_two_and_claims_nothing_for_a_family_of_another_task(world, tmp_path, capsys):  # noqa: F811
    out = tmp_path / "cli_run"
    assert cli.main([TASK, f"out={out}", "model=gpt2_lm/tiny", "data.dataset=synthetic"]) == 2 and not out.exists()
    assert "text_classification" in capsys.readouterr().err


def test_the_text_tasks_data_stands_for_both_corpora_and_a_tfds_directory_is_checked_before_the_claim(world, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "no_tfds"))
    out = tmp_path / "run"
    with pytest.raises(ConfigError, match="imdb_reviews") as caught:
        train_text_classification(**classifier_request(out, **{"data.dataset": "imdb_reviews"}))
    assert "no_tfds" in str(caught.value) and not out.exists()


@pytest.mark.skipif(not (Path(os.environ.get("TFDS_DATA_DIR", "")) / "imdb_reviews").is_dir(),
                    reason="imdb_reviews is not under TFDS_DATA_DIR on this machine")
def test_a_prefix_of_the_imdb_reviews_is_a_two_class_corpus_with_a_majority_share(world, tmp_path):  # noqa: F811
    trainer, resolved, _ = classifier_prepared(tmp_path / "imdb", **{"data.dataset": "imdb_reviews", "data.max_samples": 40})
    assert trainer.describe_data(resolved).num_classes == 2
    (texts, labels), (test_texts, test_labels) = td.corpus_documents("imdb_reviews", td.CLASSIFICATION, max_samples=40)
    assert len(texts) == len(labels) == 40 == len(test_texts) == len(test_labels) and set(labels.tolist()) == {0, 1}


# ---------------------------------------------------------------------
# Seams, smoke, help and imports
# ---------------------------------------------------------------------


def test_an_epoch_is_the_streams_steps_and_the_selection_pass_is_stated(tiny, tmp_path):
    summary = tiny.summary
    assert summary["steps_per_epoch"] == summary["n_train"] // BATCH
    assert len((tiny.out / "training_log.csv").read_text().splitlines()) == 1 + summary["epochs_run"]
    trainer, resolved, context = classifier_prepared(tmp_path / "seams", tiny.family)
    data = trainer.build_data(resolved, context)
    assert trainer.fit_kwargs(resolved, data) == {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}
    assert data.val_steps == -(-data.n_select // BATCH) == int(data.select_ds.cardinality())
    assert int(data.test_ds.cardinality()) == -(-data.n_test // BATCH)
    assert int(data.train_ds.cardinality()) == tf.data.INFINITE_CARDINALITY, "the train stream is infinite"


def test_a_smoke_run_caps_the_documents_through_the_key_the_sequence_reads(world, tmp_path):  # noqa: F811
    out = tmp_path / "smoke"
    data = {"dataset": "synthetic", "tokenizer": "ascii95", "seq_len": SEQ_LEN}
    result = train_text_classification(**classifier_request(out, data=data, epochs="auto", **{"debug_args.smoke": True}))
    summary = json.loads((out / "results_summary.json").read_text())
    assert json.loads((out / "config.json").read_text())["data"]["max_samples"] == 512
    assert result.success is True and summary["success"] is None and summary["epochs_run"] == 1


def test_the_help_lists_the_keys_says_what_a_row_is_and_a_resume_continues_the_stream(world, capsys):  # noqa: F811
    assert cli.main([TASK, "help"]) == 0
    text = capsys.readouterr().out
    for key in ("data.dataset", "data.tokenizer", "data.tokenizer_dir", "data.seq_len", "data.validation_split", "data.max_samples",
                "model_args", "optim_args.lr", "metric_args.watch_metric", "resume_interrupted", "init"):
        assert key in text, key
    assert "[CLS]" in text and "a window holds seq_len + 1 tokens" not in text, "the row of a classifier is not a language window"
    assert text.count("the batches the unkilled run reads next") == 2 and "fresh pass" not in text
    assert "top_k" not in text and "success_threshold" not in text, "the group has no knob it would ignore"


def child(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
           "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)


def test_the_new_task_module_imports_no_heavy_module():
    done = child(
        "import sys\n"
        "import train.api.text_args, train.api.text_models, train.api.tasks.text_classification\n"
        "heavy = sorted(name for name in sys.modules if name.split('.')[0] in ('tensorflow', 'keras', 'tf_keras', 'tensorboard',"
        " 'yaml', 'tiktoken', 'numpy') or name == 'train.common' or name.startswith(('train.common.', 'dl_techniques.models')))\n"
        "print('HEAVY', heavy)\n")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "HEAVY []", done.stdout[-800:]
