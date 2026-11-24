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

Tests write only under ``tmp_path``.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.utils.logger import logger
from train.api import init_from, predictor
from train.api import text_models as tm
from train.api.args import AutoContext
from train.api.errors import ConfigError
from train.api.registry import CrossTask
from train.api.text_tokenizer import tokenizer_spec
from train.common import text_data as td

from ._text_world import ASCII_VOCAB, SEQ_LEN, TINY_MODEL_ARGS, classifier_world, world  # noqa: F401  (fixtures)

TASK, LM_TASK, LM_FAMILY, FAMILY = "text_classification", "language_model_pretraining", "gpt2_lm", "gpt2"
ASCII, GPT2_TOKENIZER = tokenizer_spec("ascii95"), tokenizer_spec("gpt2")
#: What a placeholder archive holds: nothing before the claim reads it, only its hash.
PLACEHOLDER = b"not a keras archive: nothing before the claim opens it"
HEAD = "task_head"


def row_of(family):
    return {spec.family: spec for spec in tm.builtin_classifier_models()}[family]


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


class Classifier:
    """The two things ``read_source`` asks of a task's trainer: its name and the identity of the run it starts."""

    task = TASK

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def init_identity(self, config, context):
        spec = tokenizer_spec(self.tokenizer)
        return {"vocab_size": spec.vocab_size, "tokenizer": spec.digest}


def read(source, family):
    """What the sequence asks before the claim of a ``text_classification`` run on the ascii tokenizer."""
    context = AutoContext(family=family, variant="tiny", dataset="synthetic", num_classes=2, image_size=None, defaults={})
    return init_from.read_source(source, Classifier("ascii95"), None, context)


def test_an_lm_run_is_a_declared_source_of_the_gpt2_classifier_before_the_claim(classifier_world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", lm_record())
    taken = read(source, FAMILY)
    assert taken.relation == row_of(FAMILY).transfer.sources[(LM_TASK, LM_FAMILY)] and taken.relation is not None
    assert dict(taken.pins) == {} and taken.notice is None and taken.record == lm_record()
    assert init_from.file_sha256(source / "best_model.keras") == taken.sha256


def test_the_gpt2_row_declares_the_language_family_as_its_source_and_bert_declares_none(classifier_world):  # noqa: F811
    assert sorted(row_of(FAMILY).transfer.sources) == [(LM_TASK, LM_FAMILY)]
    assert not row_of("bert").transfer.sources


def test_bert_takes_no_earlier_run_and_the_refusal_says_what_it_does_take(classifier_world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", lm_record())
    with pytest.raises(ConfigError, match=r"Invalid value for 'init': gpt2_lm \(language_model_pretraining\) cannot seed bert "
                                          r"\(text_classification\); bert takes weights from bert \(text_classification\)$"):
        read(source, "bert")


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
def test_a_gpt2_classifier_at_another_tokenizer_or_vocabulary_is_refused_naming_both_sides(
        record, pattern, classifier_world, tmp_path):  # noqa: F811
    source = write_source(tmp_path / "source", record())
    with pytest.raises(ConfigError, match="Invalid value for 'init': " + pattern):
        read(source, FAMILY)


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
