"""The pair ``gpt2_lm`` to ``gpt2``: a language-model run seeds a classifier, and ``bert`` takes no earlier run.

plan-2026-11-24T092348-7806a290 Step 9 (D-005, D-040, D-054). A causal decoder trained on windows of text is the trunk of a GPT-2
classifier: the same ``decoder`` weights, the classifier's head (``task_head``, 8 variables) fresh. A bidirectional encoder cannot be
started from a causal decoder, so ``bert`` declares no source and an LM run is refused by the engine's own sentence. Pinned here, on
the real models in the PRODUCTION ORDER (the target built first, then the source's saved archive loaded beside it, as
``init_from.transfer`` loads it; a loaded archive can root its paths differently, and none does here: ``pair_probe.md``):

* the census: copied equals every decoder weight of both sides, reset equals the head's variables, nothing is dropped, missing or
  unused; at the test geometry and at the gate's (``gpt2_lm/tiny`` to ``gpt2/tiny`` at the real vocabulary and 256 positions);
* an UNTIED language model also seeds the classifier (its ``lm_head`` is dropped; with the bare ``CrossTask()`` the engine refuses it);
* no weight depends on ``seq_len``, and another vocabulary is refused at the weight level (behind the refusal before the claim);
* the transfer assigns: after it every decoder weight of the classifier IS the language model's and every head weight is untouched;
* before the claim (``init_from.read_source`` with a trainer that names the task and its identity): an LM run is a declared source
  of ``gpt2``, ``bert`` has none and the refusal says what it does take, and a source of another tokenizer or vocabulary is refused
  naming both sides;
* the ``[CLS]`` layout: a row is ``[CLS]``, the head of the document, pad; every id of it is a row of the source's embedding table.

Step 10 (the ``text_classification`` task exists): the same refusals come from the real task before the claim (exit 2, ``out``
unclaimed), and a real tiny ``gpt2_lm`` run seeds a real ``gpt2`` run (``cross_task``): copied is every weight of the source (derived
by counting the archive), reset is the 8 variables of the head, the run's own epoch zero is the loaded weights' accuracy and the bar
must beat it. Step 11 (the classifier writes ``task.json``): the ``init`` block is in the record and a resume keeps it; a classifier run
as a SOURCE is in ``test_text_classification_predictor_and_export.py``.

Tests write only under ``tmp_path``.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.api import cli, init_from, predictor
from train.api import text_models as tm
from train.api.errors import ConfigError
from train.api.registry import CrossTask
from train.api.tasks.language_model_pretraining import train_language_model_pretraining
from train.api.tasks.text_classification import train_text_classification
from train.api.text_tokenizer import DEFAULT_TOKENIZER_DIR, tokenizer_spec
from train.common import text_data as td
from train.common import text_run

from ._text_world import (
    ASCII_VOCAB, CLASSIFIER_SPLIT, SEQ_LEN, TINY_MODEL_ARGS, classifier_prepared, classifier_request, prepared, request,
    world)  # noqa: F401  (``world``: fixture)

TASK, LM_TASK, LM_FAMILY, FAMILY = "text_classification", "language_model_pretraining", "gpt2_lm", "gpt2"
ASCII, GPT2_TOKENIZER = tokenizer_spec("ascii95"), tokenizer_spec("gpt2")
#: What a placeholder archive holds: nothing before the claim reads it, only its hash.
PLACEHOLDER = b"not a keras archive: nothing before the claim opens it"
HEAD = "task_head"


def row_of(family):
    return {spec.family: spec for spec in tm.builtin_text_models()}[family]


def paths(model, side):
    """The engine's own view of a model's weights: ``{path relative to the model: variable}``."""
    return init_from._by_path(model, side)


def pair(tmp_path, *, vocab=ASCII_VOCAB, seq_len=SEQ_LEN, target_seq_len=None, target_vocab=None, source_args=None,
         target_args=None, shrink=True):
    """``(loaded source, target)`` in the production order: the target built after ``clear_session``, as a run builds its model,
    then the saved archive of a ``gpt2_lm`` loaded beside it (no ``clear_session`` between)."""
    keras.backend.clear_session()
    source = tm.create_gpt2_lm(variant="tiny", input_shape=(seq_len,), vocab_size=vocab,
                               **{**(TINY_MODEL_ARGS if shrink else {}), **(source_args or {})})
    archive = tmp_path / "source.keras"
    source.save(archive)
    keras.backend.clear_session()
    spec = tokenizer_spec("gpt2") if vocab == GPT2_TOKENIZER.vocab_size else ASCII
    target = tm.create_gpt2_classifier(
        variant="tiny", num_classes=2, input_shape=(target_seq_len or seq_len,), vocab_size=target_vocab or vocab,
        pad_id=spec.pad_id, cls_id=spec.cls_id, **{**(TINY_MODEL_ARGS if shrink else {}), **(target_args or {})})
    return keras.saving.load_model(archive, compile=False), target


def plan_of(loaded, target, relation=None, name="gpt2/tiny"):
    spec = row_of(FAMILY).transfer
    return init_from.plan_transfer(loaded, target, spec, relation=spec.sources[(LM_TASK, LM_FAMILY)] if relation is None else relation,
                                   name=name)


def counts_line(label, plan):
    return (f"CENSUS gpt2 from gpt2_lm | {label} | copied {len(plan.copied)} reset {len(plan.reset)} dropped {len(plan.dropped)} "
            f"missing {len(plan.missing)} unused {len(plan.unused)} (source {plan.n_source}, target {plan.n_target})")


def assert_the_decoder_is_copied_and_the_head_is_fresh(loaded, target, plan):
    decoder = sorted(path for path in paths(target, "target") if path.split("/")[0] != HEAD)
    head = sorted(path for path in paths(target, "target") if path.split("/")[0] == HEAD)
    assert head and decoder, "the control: both halves exist"
    assert sorted(plan.copied) == decoder == sorted(paths(loaded, "source")), "copied is every decoder weight of both sides"
    assert sorted(plan.reset) == head, "reset is exactly the head's variables"
    assert not (plan.dropped or plan.missing or plan.unused or plan.resampled or plan.grown)
    assert plan.n_source == len(decoder) and plan.n_target == len(decoder) + len(head)
    return decoder, head


# ---------------------------------------------------------------------
# The census, in the production order
# ---------------------------------------------------------------------


def test_production_order_an_lm_archive_pairs_path_for_path_with_the_gpt2_classifier(tmp_path):
    loaded, target = pair(tmp_path)
    plan = plan_of(loaded, target)
    init_from.check(plan)
    assert_the_decoder_is_copied_and_the_head_is_fresh(loaded, target, plan)
    logger.info(counts_line("test geometry, production order", plan))


def test_production_order_the_gate_geometry_pairs_too(tmp_path):
    """``gpt2_lm/tiny`` to ``gpt2/tiny`` at the real tokenizer's 50,261 ids and 256 positions: the pair the night's gate runs."""
    loaded, target = pair(tmp_path, vocab=GPT2_TOKENIZER.vocab_size, seq_len=256, shrink=False)
    assert loaded.get_layer("gpt2_backbone").depth == 4 and loaded.get_layer("gpt2_backbone").embed_dim == 256
    plan = plan_of(loaded, target)
    init_from.check(plan)
    assert_the_decoder_is_copied_and_the_head_is_fresh(loaded, target, plan)
    logger.info(counts_line("the gate's geometry (vocab 50261, 4 blocks of 256, seq 256), production order", plan))


def test_production_order_an_untied_language_model_seeds_the_classifier_and_its_lm_head_is_dropped(tmp_path):
    """The ``gpt2_lm`` table offers ``tie_word_embeddings``; the classifier has no language-model head to put the projection in."""
    loaded, target = pair(tmp_path, source_args={"tie_word_embeddings": False})
    assert "lm_head/kernel" in paths(loaded, "source")
    plan = plan_of(loaded, target)
    init_from.check(plan)
    assert plan.dropped == ("lm_head/kernel",) and not (plan.missing or plan.unused)
    decoder = sorted(path for path in paths(target, "target") if path.split("/")[0] != HEAD)
    assert sorted(plan.copied) == decoder and len(plan.reset) == plan.n_target - len(decoder)
    logger.info(counts_line("untied source, production order", plan))
    with pytest.raises(ConfigError, match=r"1 of \d+ weights of the source run are neither used nor dropped.*lm_head/kernel"):
        init_from.check(plan_of(loaded, target, relation=CrossTask()))  # the control: the bare relation refuses it


def test_production_order_a_source_trained_at_another_length_copies_every_weight(tmp_path):
    """No weight depends on ``seq_len`` (the position table has ``max_seq_len`` rows), so it is no ``fixed`` key (D-040)."""
    loaded, target = pair(tmp_path, target_seq_len=2 * SEQ_LEN)
    plan = plan_of(loaded, target)
    init_from.check(plan)
    assert_the_decoder_is_copied_and_the_head_is_fresh(loaded, target, plan)


def test_production_order_another_vocabulary_is_refused_at_the_weight_level(tmp_path):
    """Behind the refusal before the claim (``fixed``): the embedding table has another number of rows; nothing is copied."""
    loaded, target = pair(tmp_path, target_vocab=ASCII_VOCAB + 19)
    plan = plan_of(loaded, target)
    with pytest.raises(ConfigError, match="nothing was copied"):
        init_from.check(plan)
    assert len(plan.missing) == 1 and "word_embeddings/embeddings" in plan.missing[0]
    assert f"target ({ASCII_VOCAB + 19}, 64)" in plan.missing[0] and f"source ({ASCII_VOCAB}, 64)" in plan.missing[0]


def test_the_transfer_assigns_the_decoder_and_leaves_the_head_as_built(tmp_path):
    loaded, target = pair(tmp_path)
    fresh = {path: variable.numpy().copy() for path, variable in paths(target, "target").items()}
    plan = plan_of(loaded, target)
    init_from.apply(plan)
    source, now = paths(loaded, "source"), paths(target, "target")
    assert all(np.array_equal(now[path].numpy(), source[path].numpy()) for path in plan.copied)
    assert any(not np.array_equal(fresh[path], now[path].numpy()) for path in plan.copied), "the decoder really changed"
    assert all(np.array_equal(fresh[path], now[path].numpy()) for path in plan.reset), "the head kept its fresh values"


# ---------------------------------------------------------------------
# Before the claim: what the engine reads of a source directory
# ---------------------------------------------------------------------


def lm_record(tokenizer="ascii95", **changes):
    record = predictor.build_language_model_record(
        model="gpt2_lm/tiny", family=LM_FAMILY, variant="tiny", seq_len=SEQ_LEN, tokenizer=tokenizer_spec(tokenizer).record())
    return {**record, **changes}


def write_source(root, record):
    """A fake run directory: ``task.json``, ``config.json`` and a placeholder archive (``read_source`` only hashes it)."""
    root.mkdir(parents=True)
    (root / "task.json").write_text(json.dumps(record))
    (root / "config.json").write_text(json.dumps({"model": record["model"], "used": {"model_args": {}}}))
    (root / "best_model.keras").write_bytes(PLACEHOLDER)
    return root


def refuse(source, family, tmp_path):
    """Run the real task with ``init=source``: the sequence reads the source in ``_resolve``, ahead of ``open_run``."""
    return train_text_classification(**{**classifier_request(tmp_path / "run", family), "init": str(source)})


def test_an_lm_run_is_a_declared_source_of_the_gpt2_classifier_before_the_claim(world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", lm_record())
    trainer, resolved, context = classifier_prepared(tmp_path / "run", FAMILY, init=str(source))
    taken = trainer._init_source
    assert taken.relation == row_of(FAMILY).transfer.sources[(LM_TASK, LM_FAMILY)] and taken.relation is not None
    assert dict(taken.pins) == {} and taken.notice is None and taken.record == lm_record() and context.pinned == {}
    assert init_from.file_sha256(source / "best_model.keras") == taken.sha256 and not (tmp_path / "run").exists()
    assert trainer.init_identity(resolved, context) == {"vocab_size": ASCII.vocab_size, "tokenizer": ASCII.digest}


def test_a_source_trained_at_another_length_is_taken_before_the_claim(world, tmp_path):  # noqa: F811
    """``seq_len`` is no ``fixed`` key (D-040): no weight depends on it, so a language run of 64-token windows seeds a 32-token row."""
    source = write_source(tmp_path / "source", lm_record(seq_len=64))
    trainer, resolved, _ = classifier_prepared(tmp_path / "run", FAMILY, init=str(source))
    assert trainer._init_source.relation is not None and trainer._init_source.record["seq_len"] == 64 != resolved.data.seq_len
    assert not (tmp_path / "run").exists()


def test_the_gpt2_row_declares_the_language_family_as_its_source_and_bert_declares_none(world):  # noqa: F811
    assert sorted(row_of(FAMILY).transfer.sources) == [(LM_TASK, LM_FAMILY)]
    assert not row_of("bert").transfer.sources


def test_bert_takes_no_earlier_run_and_the_refusal_says_what_it_does_take(world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", lm_record())
    with pytest.raises(ConfigError, match=r"Invalid value for 'init': gpt2_lm \(language_model_pretraining\) cannot seed bert "
                                          r"\(text_classification\); bert takes weights from bert \(text_classification\)$"):
        refuse(source, "bert", tmp_path)
    assert not (tmp_path / "run").exists()


def vit_record():
    """An image-classification record: a run of another task, the case "a classifier source" can be before Step 11 has a record."""
    return predictor.build_task_record(
        task="image_classification", model="vit/vit_pico", family="vit", variant="vit_pico", class_names=["a", "b"],
        input_shape=[32, 32, 3], normalize="standardize", mean=[0.5] * 3, std=[0.25] * 3, data_kind="cifar10", source_channels=3)


REFUSALS = [
    pytest.param(lambda: lm_record() | {"tokenizer": "f" * 64},
                 rf"gpt2 sizes its weights by the tokenizer \(data\.tokenizer, as its digest\); the source trained at {'f' * 64}, "
                 rf"this run asks for {ASCII.digest}$", id="another_tokenizer"),
    pytest.param(lambda: lm_record() | {"vocab_size": 120},
                 r"gpt2 sizes its weights by the model's vocabulary size; the source trained at 120, this run asks for 101$",
                 id="another_vocabulary"),
    pytest.param(lambda: lm_record("gpt2"),
                 rf"gpt2 sizes its weights by the model's vocabulary size and the tokenizer \(data\.tokenizer, as its digest\); "
                 rf"the source trained at {GPT2_TOKENIZER.vocab_size} and {GPT2_TOKENIZER.digest}, this run asks for 101 and "
                 rf"{ASCII.digest}$", id="another_encoding"),
]


@pytest.mark.parametrize("record, pattern", REFUSALS)
def test_a_gpt2_classifier_at_another_tokenizer_or_vocabulary_is_refused_naming_both_sides(record, pattern, world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", record())
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + pattern):
        refuse(source, FAMILY, tmp_path)
    assert not (tmp_path / "run").exists(), "a pre-claim refusal claimed the run directory"


def test_a_run_of_another_task_is_refused_by_the_declared_sources_sentence_and_the_command_exits_two(world, tmp_path, capsys):  # noqa: F811
    source = write_source(tmp_path / "source", vit_record())
    sentence = (r"vit \(image_classification\) cannot seed gpt2 \(text_classification\); gpt2 takes weights from gpt2 "
                r"\(text_classification\) or gpt2_lm \(language_model_pretraining\)$")
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + sentence):
        refuse(source, FAMILY, tmp_path)
    out = tmp_path / "cli_run"
    stale = write_source(tmp_path / "stale", lm_record() | {"tokenizer": "f" * 64})
    code = cli.main(["text_classification", f"out={out}", "model=gpt2/tiny", "data.dataset=synthetic", "data.tokenizer=ascii95",
                     f"init={stale}"])
    assert code == 2 and not out.exists()
    assert "gpt2 sizes its weights by the tokenizer (data.tokenizer, as its digest)" in capsys.readouterr().err


def test_every_fixed_key_of_the_classifier_rows_has_a_name_in_the_refusal():
    for family in ("bert", FAMILY):
        assert set(row_of(family).transfer.fixed) <= set(init_from.GEOMETRY_NAMES), family


# ---------------------------------------------------------------------
# The layout the classifier trains on
# ---------------------------------------------------------------------


@pytest.mark.parametrize("spec", [ASCII, GPT2_TOKENIZER], ids=lambda spec: spec.encoding)
def test_a_row_is_cls_then_the_head_of_the_document_then_pad_and_every_id_is_a_row_of_the_sources_table(spec):
    """M23: a row that does not begin with the tokenizer's ``[CLS]`` is a row of another layout (``bert`` reads position 0)."""
    docs = [np.array(ids, np.int32) for ids in ([5, 6, 7], [], list(range(3, 30)), [9] * 5)]
    ids, truncated = td.encode_rows(docs, spec, 8)
    assert ids.shape == (4, 8) and (ids[:, 0] == spec.cls_id).all() and ids[0].tolist() == [spec.cls_id, 5, 6, 7] + [spec.pad_id] * 4
    assert truncated == 1 and (ids[1, 1:] == spec.pad_id).all(), "an empty document is the cls id alone, still no all-pad row"
    # the source's embedding table has one row per id of the vocabulary, the four specials included (D-004)
    assert ids.max() < spec.vocab_size and spec.cls_id != spec.pad_id and spec.cls_id < spec.vocab_size
    # a classifier built from the spec takes these rows as they are
    keras.backend.clear_session()
    model = tm.create_gpt2_classifier(variant="tiny", num_classes=2, input_shape=(8,), vocab_size=spec.vocab_size,
                                      pad_id=spec.pad_id, cls_id=spec.cls_id, embed_dim=16, depth=1, num_heads=2)
    assert model(ids.astype(np.float32)).shape == (4, 2)


# ---------------------------------------------------------------------
# A real pair: a tiny gpt2_lm run seeds a real gpt2 run
# ---------------------------------------------------------------------

#: A rate that moves no float32 weight: the run scores what the loaded weights scored at epoch zero.
STILL = {"lr": 1e-30, "warmup_epochs": 0}


def weights_of(run_dir):
    """The engine's view of the saved best model of a run: ``{path relative to the model: array}``."""
    model = keras.models.load_model(str(run_dir / "best_model.keras"), compile=False)
    return {path: variable.numpy() for path, variable in init_from._by_path(model, "run").items()}


@pytest.fixture(scope="module")
def lm_source(world, tmp_path_factory):  # noqa: F811
    """gpt2_lm on the synthetic bank (64 wide, 2 blocks, windows of 32), 3 epochs."""
    out = tmp_path_factory.mktemp("lm_source") / "run"
    keras.backend.clear_session()
    train_language_model_pretraining(**request(out))
    return out


def classifier_run(root, source, **changes):
    keras.backend.clear_session()
    out = root / "run"
    train_text_classification(**classifier_request(out, FAMILY, init=str(source), **changes))
    return out, json.loads((out / "results_summary.json").read_text())


@pytest.fixture(scope="module")
def still(world, lm_source, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("still")
    source = root / "source"
    shutil.copytree(lm_source, source)
    return SimpleNamespace(source=source, **dict(zip(("out", "summary"), classifier_run(root, source, epochs=1, optim_args=STILL))))


def test_a_gpt2_run_starts_from_a_language_model_run_and_the_counts_are_the_archives(still, lm_source):
    block = still.summary["init"]
    source, target = weights_of(lm_source), weights_of(still.out)
    head = sorted(path for path in target if path.split("/")[0] == HEAD)
    assert (block["task"], block["family"], block["variant"], block["relation"], block["pinned"]) == (
        LM_TASK, LM_FAMILY, "tiny", "cross_task", {})
    assert block["geometry"] == {"seq_len": SEQ_LEN, "vocab_size": ASCII_VOCAB}
    assert (block["transfer"]["copied"], sorted(block["transfer"]["reset"])) == (len(source), head) and len(head) == 8
    assert not (block["transfer"]["dropped"] or block["transfer"]["resampled"] or block["transfer"]["grown"])
    logger.info(f"CLS-INIT copied {block['transfer']['copied']} of {len(source)} weights of the language run, reset "
                f"{len(block['transfer']['reset'])} (the head's {len(head)} variables), dropped 0, split {block['split']['relation']}")
    # the run really holds the source's weights: a still fine-tune moves none of them, and the head is the run's own
    decoder = sorted(path for path in target if path.split("/")[0] != HEAD)
    assert decoder == sorted(source)
    for path in decoder:
        np.testing.assert_array_equal(target[path], source[path], err_msg=path)


def test_the_init_block_is_in_the_record_and_a_resumed_run_reports_it_without_opening_the_source(still, tmp_path):
    record = json.loads((still.out / "task.json").read_text())
    assert predictor.record_problems(record) == [] and record["task"] == TASK and record["init"] == still.summary["init"]
    run_dir = tmp_path / "run"
    shutil.copytree(still.out, run_dir)
    (run_dir / "results_summary.json").unlink()  # a run whose reporting died: resumed to report, its block read from its own record
    moved = still.source.with_name("moved_away")
    still.source.rename(moved)
    try:
        keras.backend.clear_session()
        train_text_classification(**classifier_request(run_dir, FAMILY, init=str(still.source), epochs=1, optim_args=STILL,
                                                      resume_interrupted=True))
    finally:
        moved.rename(still.source)
    resumed = json.loads((run_dir / "results_summary.json").read_text())
    assert resumed["reporting_resumed"] is True and resumed["init"] == still.summary["init"] and json.loads((run_dir / "task.json").read_text()) == record


def test_the_two_synthetic_banks_are_other_texts_so_the_split_is_different_data(still):
    split = still.summary["init"]["split"]
    assert (split["relation"], split["n_select_overlap"], split["n_test_overlap"]) == ("different_data", 0, 0)
    assert "language bank" in split["reason"] and "labelled bank" in split["reason"]
    assert still.summary["epoch_zero_of"] == "init" and not any(init_from.bias_of(still.summary["init"]).values())


def test_under_init_epoch_zero_is_the_loaded_weights_accuracy_and_a_fine_tune_that_only_holds_them_misses(still):
    summary = still.summary
    loaded = json.loads((still.out / "checkpoints" / "fit_state.json").read_text())["initial"]["metrics"]["accuracy"]
    assert summary["loaded_success_value"] == loaded and (summary["initial_loss_ratio"], summary["init_scale_warning"]) == (None, None)
    assert summary["success_value"] == loaded, "the still run scores what the loaded weights scored"
    assert summary["success"] is False and not text_run.strictly_above(summary["success_value"], loaded)
    assert init_from.NOT_JUDGED_UNDER_INIT in summary["notes"]
    assert "Evaluate of the loaded weights before fit: " in (still.out / "run.log").read_text()
    logger.info(f"CLS-INIT-STILL epoch zero (the loaded weights) {loaded:.4f}, bar (share plus margin) "
                f"{summary['success_threshold']:.4f}, best of the still run {summary['success_value']:.4f}, success {summary['success']}")


def test_a_fine_tune_that_learns_meets_the_bar_and_beats_the_loaded_weights(world, lm_source, tmp_path):  # noqa: F811
    out, summary = classifier_run(tmp_path, lm_source)
    loaded = summary["loaded_success_value"]
    assert summary["success"] is True and text_run.strictly_above(summary["success_value"], summary["success_threshold"])
    assert text_run.strictly_above(summary["success_value"], loaded), "and it beats the loaded weights' epoch zero"
    logger.info(f"CLS-INIT-LEARNS epoch zero {loaded:.4f}, bar {summary['success_threshold']:.4f}, best {summary['success_value']:.4f}")


def test_the_overlap_of_a_language_source_with_the_imdb_selection_is_counted_as_the_oracle(world, tmp_path, monkeypatch):  # noqa: F811
    """The same reviews serve both tasks, so by document index the count is exact (the synthetic banks are other texts)."""
    use_reviews(monkeypatch, tmp_path)    # a TFDS directory of its own: no TFDS_DATA_DIR of the machine is read
    import dataclasses
    import hashlib

    def held_out(seed, n, split):
        return {i for i in range(n)
                if int.from_bytes(hashlib.sha256(f"{seed}:select:{i}".encode()).digest()[:8], "big") / 2 ** 64 < split}

    n, split = 120, CLASSIFIER_SPLIT
    trainer, resolved, _ = classifier_prepared(tmp_path / "imdb", **{"data.dataset": "imdb_reviews", "data.max_samples": n})
    theirs = {"seed": 42, "data": {"dataset": "imdb_reviews", "validation_split": split, "max_samples": n}}
    source = SimpleNamespace(config=theirs, record={"task": LM_TASK}, path=tmp_path)    # a source is a directory: no sidecar in it, no digest

    def data_of(seed):
        chosen = held_out(seed, n, split)
        return SimpleNamespace(split=SimpleNamespace(
            train_index=np.array(sorted(set(range(n)) - chosen)), select_index=np.array(sorted(chosen)), n_test_docs=n))

    other = trainer.init_overlap(source, dataclasses.replace(resolved, seed=43), data_of(43))
    expected = len((set(range(n)) - held_out(42, n, split)) & held_out(43, n, split))
    assert expected > 0 and (other["relation"], other["unit"], other["n_select_overlap"], other["n_test_overlap"]) == (
        "overlap", "documents", expected, 0)
    assert predictor._split_problems(other) == []
    same = trainer.init_overlap(source, resolved, data_of(42))
    assert (same["relation"], same["n_select_overlap"], same["n_test_overlap"]) == ("same_split", 0, 0)


# ---------------------------------------------------------------------
# The split of a language source on the same reviews (review pass 2, concern 1)
# ---------------------------------------------------------------------

#: What both runs ask of ``imdb_reviews``: equal keys (dataset, seed, split, cap), so the split is the same when the documents are.
IMDB = {"data.dataset": "imdb_reviews", "data.max_samples": 300, "data.validation_split": CLASSIFIER_SPLIT}
REAL_IMDB = pytest.mark.skipif(
    not ((Path(os.environ.get("TFDS_DATA_DIR", "")) / "imdb_reviews").is_dir()
         and all((Path(DEFAULT_TOKENIZER_DIR) / name).is_file() for name, _ in tokenizer_spec("gpt2").files)),
    reason="imdb_reviews or the pinned tokenizer directory is not on this machine")


def stand_in_reviews(split, data_dir, limit):
    """What TFDS reads, in read order: the labelled bank's texts as the reviews both tasks share."""
    texts, labels = td.synthetic_documents(td.BANK_SEED, 400, td.CLASSIFICATION, split)
    return ((text.encode(), int(label)) for text, label in list(zip(texts, labels))[:limit])


def another_review(split, data_dir, limit):
    """The same reviews with one character of the first train review changed: every count stays."""
    rows = list(stand_in_reviews(split, data_dir, limit))
    return iter([(rows[0][0][:-2] + b"x.", rows[0][1]), *rows[1:]] if split == "train" else rows)


def use_reviews(patch, root, reader=stand_in_reviews):
    """Put ``reader`` where TFDS's reader is and a TFDS directory that passes the check made before the claim."""
    where = root / "tfds" / "imdb_reviews" / "plain_text" / "1.0.0"
    where.mkdir(parents=True, exist_ok=True)
    (where / "dataset_info.json").write_text("{}")
    patch.setenv("TFDS_DATA_DIR", str(root / "tfds"))
    patch.setattr(td, "_tfds_imdb", reader)


@pytest.fixture(scope="module")
def imdb_pair(world, tmp_path_factory):  # noqa: F811
    """A tiny gpt2_lm run, and a tiny gpt2 run seeded from it, both reading imdb_reviews through the stand-in reader with equal keys."""
    root = tmp_path_factory.mktemp("imdb_pair")
    with pytest.MonkeyPatch.context() as patch:
        use_reviews(patch, root)
        keras.backend.clear_session()
        train_language_model_pretraining(**request(root / "lm", epochs=1, **IMDB))
        out, summary = classifier_run(root, root / "lm", epochs=1, optim_args=STILL, **IMDB)
    return SimpleNamespace(root=root, lm=root / "lm", out=out, summary=summary)


def test_a_gpt2_run_seeded_from_a_language_run_on_the_same_reviews_and_keys_has_the_same_split(imdb_pair):
    # the language run digests documents, the classifier documents and labels, so their sidecars never agree: the split must not read that
    block = imdb_pair.summary["init"]
    lm, cls = (json.loads((run / "checkpoints" / "fit_state.json").read_text())["data_fingerprint"] for run in (imdb_pair.lm, imdb_pair.out))
    assert len(lm) == len(cls) == 64 and lm != cls, "the control: the two sidecars hold different digests of the same corpus"
    assert (block["relation"], block["split"]["relation"], block["split"]["unit"]) == ("cross_task", "same_split", "documents"), block["split"]
    assert imdb_pair.summary["overlap_unknown"] is False and imdb_pair.summary["selection_biased"] is False


#: ``case: (the reader, this run's keys that differ, whether the source's sidecar loses its digest, the relation, a phrase of the reason)``.
REVIEWS = {
    "the same reviews": (stand_in_reviews, {}, False, "same_split", "equal the source's"),
    "one character of a review changed": (another_review, {}, False, "unknown", "another data_fingerprint"),
    "a sidecar with no digest": (stand_in_reviews, {}, True, "unknown", "no data_fingerprint"),
    "a lower cap than the source's": (stand_in_reviews, {"data.max_samples": 200}, False, "unknown", "data.max_samples"),
}


@pytest.mark.parametrize("case", sorted(REVIEWS))
def test_the_split_of_a_language_source_is_read_from_its_own_sidecar_and_a_missing_or_other_digest_is_unknown(case, imdb_pair, tmp_path, monkeypatch):
    reader, changes, strip, relation, reason = REVIEWS[case]
    use_reviews(monkeypatch, tmp_path, reader)
    trainer, resolved, context = classifier_prepared(tmp_path / "run", **{**IMDB, **changes})
    data = trainer.build_data(resolved, context)
    path = imdb_pair.lm
    if strip:       # a source made before the digest existed, or whose sidecar lost it: a copy of the sidecar without the key
        path = tmp_path / "stripped"
        (path / "checkpoints").mkdir(parents=True)
        state = json.loads((imdb_pair.lm / "checkpoints" / "fit_state.json").read_text())
        state.pop("data_fingerprint")
        (path / "checkpoints" / "fit_state.json").write_text(json.dumps(state))
    source = SimpleNamespace(config=json.loads((imdb_pair.lm / "config.json").read_text()),
                             record=json.loads((imdb_pair.lm / "task.json").read_text()), path=path)
    split = trainer.init_overlap(source, resolved, data)
    assert predictor._split_problems(split) == [] and split["relation"] == relation and reason in split["reason"], split
    assert init_from.bias_of({"split": split})["overlap_unknown"] == (relation == "unknown")


@REAL_IMDB
def test_on_the_real_reviews_the_two_tasks_digest_the_same_documents_and_their_split_is_the_same(world, tmp_path):  # noqa: F811
    # the real corpus through each task's own data code (a prefix of it: the same code as the whole, whose digests are the gate's claims):
    # the two digests of the corpus differ with labels and agree without them
    options = {"data.dataset": "imdb_reviews", "data.tokenizer": "gpt2", "data.max_samples": 2000, "data.validation_split": CLASSIFIER_SPLIT}
    lm_trainer, lm_config, lm_context = prepared(tmp_path / "lm", **options)
    trainer, resolved, context = classifier_prepared(tmp_path / "cls", **options)
    lm_data, data = lm_trainer.build_data(lm_config, lm_context), trainer.build_data(resolved, context)
    assert len(lm_data.fingerprint) == 64 and lm_data.fingerprint == data.split.documents_fingerprint != data.fingerprint
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "fit_state.json").write_text(json.dumps({"data_fingerprint": lm_data.fingerprint}))
    theirs = {"dataset": "imdb_reviews", "validation_split": CLASSIFIER_SPLIT, "max_samples": 2000}
    source = SimpleNamespace(config={"seed": resolved.seed, "data": theirs}, record={"task": LM_TASK}, path=tmp_path)
    assert trainer.init_overlap(source, resolved, data)["relation"] == "same_split"
