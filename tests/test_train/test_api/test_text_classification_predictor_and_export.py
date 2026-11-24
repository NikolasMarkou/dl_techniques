"""The ``text_classification`` record, ``predict`` and ``export`` (plan-2026-11-24T092348-7806a290 step 11, D-007, D-008).

A text-classification run writes a schema-v2 ``task.json`` (``task: "text_classification"``, the eighth record kind) before ``fit``, and its
directory can be predicted with, exported and used as an ``init`` source. ``predict`` answers per document the class names best first and
their softmax probabilities, as the image classifier does; ``export`` writes the model as a file taking float32 token ids at the trained
length. Pinned here, on one real tiny run per family (the offline labelled bank, ``ascii95``, 64 wide, 2 blocks, rows of 32 ids), CPU,
written under ``tmp_path``:

* **the record**: it passes ``record_problems`` and equals the record the test builds itself; a record missing any key or holding an
  impossible value is refused naming the key; a resume compares ``seq_len``, the encoding, the vocabulary, the tokenizer digest and the
  class names (``DATA_KEYS``) and nothing of how the predictor answers; ``load_model`` refuses a record whose length or class count is
  not the model's;
* **predict**: ``scores`` equal the test's own softmax of the model's logits for the row the test builds itself (``[CLS]``, the head of
  the text, pad), a text longer than the row is cut at its HEAD, a text answers alone as it does in a batch, and the command writes one
  JSON list of a ``.txt`` or a ``.jsonl`` file;
* **export**: every format equals the model on a batch of VALID rows, on the tiny runs and on real gpt2-vocabulary builds (50,261 ids);
  another length is refused before a file is written; the tflite check feeds the rule's valid ids;
* **init**: a classifier run is an ``init`` source of its own family (every weight copied, the head too) and its head is reset when the
  class names differ; another family's run is refused by the declared-sources sentence.
"""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import cli, exporting, init_from
from train.api import text_models as tm
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, SUFFIX, TFLITE_CHECK_RTOL
from train.api.tasks.text_classification import train_text_classification
from train.common import text_data as td

from ._text_world import CLASSIFIER_ARGS, SEQ_LEN, classifier_request, world  # noqa: F401  (``world`` is a fixture)
from .test_export_roundtrip import read_back

TASK = "text_classification"
FAMILIES = ("bert", "gpt2")
CLASSES = ["neg", "pos"]
#: The keys of a text-classification record, in the order they are written: the language model's, then the classifier's own.
KEYS = ("schema_version", "task", "model", "family", "variant", "seq_len", "encoding", "n_vocab", "vocab_size", "eot_id", "pad_id",
        "tokenizer", "tokenizer_files", "cls_id", "class_names", "layout", "output")
ASCII = tt.tokenizer_spec("ascii95")
REAL_SEQ = 64

NEGATIVE = "bad poor awful worst hate dull"
POSITIVE = "good great fine nice best love"
SHORT = "good"
MIDDLE = "the film was bad and dull"
LONG = f"{NEGATIVE} {NEGATIVE} {POSITIVE} {POSITIVE}"          # the first 31 characters read negative, the last 31 positive
DOCUMENTS = [SHORT, MIDDLE, LONG]
#: What a run's head, and the tail the LAST ``SEQ_LEN - 1`` characters, of ``LONG`` say.
HEAD, TAIL = LONG[:SEQ_LEN - 1], LONG[-(SEQ_LEN - 1):]
STILL = {"lr": 1e-30, "warmup_epochs": 0}                       # a rate that moves no float32 weight


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def ascii_tokenizer() -> td.Tokenizer:
    return td.open_tokenizer(ASCII)


def built_record(family: str = "gpt2", **changes: Any) -> Dict[str, Any]:
    """The record of a tiny run as the test builds it, without a run."""
    from train.api.predictor import build_text_classification_record

    record = build_text_classification_record(
        model=f"{family}/tiny", family=family, variant="tiny", seq_len=SEQ_LEN, tokenizer=ASCII.record(), class_names=CLASSES,
        variant_overrides=CLASSIFIER_ARGS[family])
    return {**record, **changes}


@pytest.fixture(scope="module")
def runs(world, tmp_path_factory):  # noqa: F811
    """One finished 8-epoch run per family (the Step 10 shape: both reach an accuracy of 1.0), and what ``load_model`` makes of it."""
    from train.api.predictor import load_model

    found = {}
    for family in FAMILIES:
        out = tmp_path_factory.mktemp(f"classifier_record_{family}") / "run"
        keras.backend.clear_session()
        train_text_classification(**classifier_request(out, family))
        predictor = load_model(out)
        found[family] = SimpleNamespace(family=family, out=out, record=read_json(out / "task.json"), predictor=predictor,
                                        model=predictor.model, summary=read_json(out / "results_summary.json"))
    return found


@pytest.fixture(params=FAMILIES)
def run(request, runs):
    return runs[request.param]


def record_of(run: SimpleNamespace) -> Dict[str, Any]:
    return dict(run.record)


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_the_record_the_task_wrote_passes_the_reader_and_is_the_one_the_test_builds_from_what_it_asked(run):
    from train.api.predictor import OPTIONAL_RECORD_KEYS, RECORD_KEYS, record_problems

    record = record_of(run)
    assert record_problems(record) == []
    assert RECORD_KEYS[TASK] == KEYS and tuple(record)[: len(KEYS)] == KEYS and set(record) - set(KEYS) <= set(OPTIONAL_RECORD_KEYS)
    assert (record["schema_version"], record["task"], record["model"], record["family"], record["variant"]) == (
        2, TASK, f"{run.family}/tiny", run.family, "tiny")
    assert (record["seq_len"], record["encoding"], record["n_vocab"], record["vocab_size"]) == (SEQ_LEN, "ascii95", 101, ASCII.vocab_size)
    assert (record["eot_id"], record["pad_id"], record["cls_id"], record["tokenizer"], record["tokenizer_files"]) == (
        ASCII.eot_id, ASCII.pad_id, ASCII.cls_id, ASCII.digest, {})
    assert (record["class_names"], record["layout"], record["output"]) == (CLASSES, "cls_head_pad", "logits")
    assert record["variant_overrides"] == CLASSIFIER_ARGS[run.family]
    assert record == built_record(run.family)


@pytest.mark.parametrize("key", KEYS)
def test_a_record_missing_any_key_is_refused(key):
    from train.api.predictor import record_problems

    problems = record_problems({k: v for k, v in built_record().items() if k != key})
    assert problems, f"a record without {key!r} was accepted"
    if key not in ("schema_version", "task"):
        assert problems == [f"missing key(s) {key!r}"]


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"seq_len": 0}, "'seq_len'"), ({"seq_len": 32.0}, "'seq_len'"), ({"vocab_size": 0}, "'vocab_size'"), ({"n_vocab": 200}, "'n_vocab'"),
        ({"pad_id": 101}, "'pad_id'"), ({"eot_id": -1}, "'eot_id'"), ({"cls_id": 101}, "'cls_id'"), ({"cls_id": 0}, "'cls_id'"),
        ({"encoding": "cl100k_base"}, "'encoding'"), ({"tokenizer": "abc"}, "'tokenizer'"), ({"tokenizer_files": ["a"]}, "'tokenizer_files'"),
        ({"tokenizer_files": {"abc": "0" * 64}}, "'tokenizer_files'"), ({"class_names": []}, "'class_names'"),
        ({"class_names": ["a", "a"]}, "'class_names'"), ({"layout": "cls_tail_pad"}, "'layout'"),
        ({"output": "next_token_logits"}, "'output'"),
    ],
    ids=["seq_zero", "seq_float", "vocab_zero", "n_vocab_above_vocab", "pad_outside", "eot_negative", "cls_outside", "cls_is_pad",
         "other_encoding", "short_digest", "files_not_object", "ascii_with_files", "no_classes", "repeated_class", "other_layout",
         "the_language_models_output"],
)
def test_a_record_of_impossible_values_is_refused_naming_the_key(change, fragment):
    from train.api.predictor import record_problems

    problems = record_problems(built_record(**change))
    assert any(fragment in line for line in problems), problems


def test_a_gpt2_record_needs_its_files_and_one_built_from_the_pinned_spec_passes():
    from train.api.predictor import build_text_classification_record, record_problems

    spec = tt.tokenizer_spec("gpt2")
    record = build_text_classification_record(
        model="gpt2/tiny", family="gpt2", variant="tiny", seq_len=REAL_SEQ, tokenizer=spec.record(), class_names=CLASSES)
    assert record_problems(record) == [] and record["cls_id"] == 50257 and record["tokenizer_files"] == dict(spec.files)
    assert any("'tokenizer_files'" in line for line in record_problems({**record, "tokenizer_files": {}}))


def test_a_resume_compares_the_length_the_encoding_the_vocabulary_the_tokenizer_and_the_class_names():
    from train.api.predictor import DATA_KEYS, record_differences

    record = built_record()
    assert set(DATA_KEYS[TASK]) == {"seq_len", "encoding", "vocab_size", "tokenizer", "class_names"}
    for key, value in [("seq_len", 64), ("encoding", "gpt2"), ("vocab_size", 50261), ("tokenizer", "0" * 64), ("class_names", ["pos", "neg"])]:
        assert [line.split(":")[0] for line in record_differences(record, {**record, key: value})] == [key]
    for key, value in [("output", "other"), ("layout", "other"), ("cls_id", 7)]:
        assert record_differences(record, {**record, key: value}) == [], f"{key} is how the predictor answers, not what the run saw"


@pytest.mark.parametrize("key, value", [("seq_len", 64), ("tokenizer", "0" * 64), ("class_names", ["neg", "pos", "neutral"])])
def test_a_resume_of_a_run_on_other_class_names_a_length_or_a_tokenizer_is_refused_by_the_record(run, tmp_path, key, value):
    from train.api.resume_decisions import check_data_unchanged

    shutil.copy(run.out / "task.json", tmp_path / "task.json")
    data, resume = SimpleNamespace(n_train=1, n_select=1, n_test=1), SimpleNamespace(state={"data_counts": {"n_train": 1, "n_select": 1, "n_test": 1}})
    check_data_unchanged(tmp_path, None, record_of(run), data, resume)  # the control: the same record is the same data
    with pytest.raises(ConfigError, match=rf"the data is not the data the run started with \({key}: "):
        check_data_unchanged(tmp_path, None, {**record_of(run), key: value}, data, resume)


def doctored(run: SimpleNamespace, tmp_path: Path, name: str = "run", **changes: Any) -> Path:
    """A run directory ``tmp_path / name`` holding the run's archive beside a record with ``changes``."""
    copy = tmp_path / name
    copy.mkdir()
    shutil.copy(run.out / "best_model.keras", copy / "best_model.keras")
    (copy / "task.json").write_text(json.dumps({**record_of(run), **changes}))
    return copy


def test_load_model_loads_the_honest_record_and_refuses_one_whose_length_vocabulary_or_class_count_is_not_the_models(run, tmp_path):
    from train.api.predictor import load_model

    assert type(load_model(doctored(run, tmp_path))).__name__ == "TextClassifierPredictor", "the control: the honest copy loads"
    with pytest.raises(ConfigError, match="does not take the seq_len 64"):
        load_model(doctored(run, tmp_path, "length", seq_len=64))
    with pytest.raises(ConfigError, match=r"names a vocabulary of 120 ids but .*best_model\.keras embeds 101; the file and the model are not from one run"):
        load_model(doctored(run, tmp_path, "vocabulary", vocab_size=120))
    with pytest.raises(ConfigError, match=r"outputs \(2,\) per document but task\.json names 3 classes"):
        load_model(doctored(run, tmp_path, "classes", class_names=["neg", "pos", "neutral"]))


def test_load_model_refuses_a_record_naming_another_variant(runs, tmp_path):
    from train.api.predictor import load_model

    with pytest.raises(ConfigError, match=r"is not that variant \(max_seq_len is 512 in the model, 1024 in 'small'"):
        load_model(doctored(runs["gpt2"], tmp_path, model="gpt2/small", variant="small"))


# ---------------------------------------------------------------------
# predict: the test's own softmax of the test's own row
# ---------------------------------------------------------------------


def own_row(text: str) -> List[int]:
    """The row of a text, written here and not read from the module: ``[CLS]``, the first ``SEQ_LEN - 1`` ids, pad."""
    ids = ascii_tokenizer().encode(text)[: SEQ_LEN - 1]
    return [ASCII.cls_id] + ids + [ASCII.pad_id] * (SEQ_LEN - 1 - len(ids))


def own_answer(model: Any, text: str):
    """``(labels best first, probabilities in that order, logits)`` of one text: one model call, the softmax in float64."""
    logits = np.asarray(model(np.asarray([own_row(text)], np.float32), training=False), np.float64)[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    order = np.argsort(-probabilities, kind="stable")
    return [CLASSES[i] for i in order], [float(probabilities[i]) for i in order], logits


@pytest.mark.parametrize("text", DOCUMENTS, ids=["short", "middle", "longer_than_the_row"])
def test_the_scores_are_the_independent_softmax_of_the_models_logits_for_the_row_the_task_trains_on(run, text):
    labels, probabilities, _ = own_answer(run.model, text)
    answer = run.predictor.predict([text])
    assert answer["labels"] == [labels] and len(answer["scores"]) == 1
    assert answer["scores"][0] == pytest.approx(probabilities, rel=1e-5, abs=1e-6)
    assert sum(answer["scores"][0]) == pytest.approx(1.0, abs=1e-9) and answer["scores"][0] == sorted(answer["scores"][0], reverse=True)


def test_a_single_text_answers_flat_lists_and_a_list_one_list_per_text_and_a_text_does_not_depend_on_its_neighbours(run):
    many = run.predictor.predict(DOCUMENTS)
    assert len(many["labels"]) == len(many["scores"]) == 3 and all(isinstance(entry, list) for entry in many["labels"])
    for number, text in enumerate(DOCUMENTS):
        one = run.predictor.predict(text)
        assert one["labels"] == many["labels"][number] and all(isinstance(label, str) for label in one["labels"])
        assert one["scores"] == pytest.approx(many["scores"][number], rel=1e-5, abs=1e-6)
    back = run.predictor.predict(DOCUMENTS[::-1])
    assert back["labels"][::-1] == many["labels"]
    np.testing.assert_allclose(back["scores"][::-1], many["scores"], rtol=1e-5, atol=1e-6)


def test_a_short_text_pads_to_the_same_answer_in_a_longer_row(run):
    """The same weights in a model built for rows twice as long (no weight depends on ``seq_len``) answer a short text alike."""
    from train.api.text_predictor import TextClassifierPredictor

    factory = {"bert": tm.create_bert_classifier, "gpt2": tm.create_gpt2_classifier}[run.family]
    longer = factory(variant="tiny", num_classes=2, input_shape=(2 * SEQ_LEN,), vocab_size=ASCII.vocab_size, pad_id=ASCII.pad_id,
                     cls_id=ASCII.cls_id, **CLASSIFIER_ARGS[run.family])
    longer.set_weights(run.model.get_weights())
    wide = TextClassifierPredictor(longer, {**record_of(run), "seq_len": 2 * SEQ_LEN})
    for text in (SHORT, MIDDLE):
        there, here = wide.predict(text), run.predictor.predict(text)
        assert there["labels"] == here["labels"]
        np.testing.assert_allclose(there["scores"], here["scores"], rtol=1e-4, atol=1e-6)


def test_the_cue_words_are_read_as_the_class_they_belong_to_with_logits_a_wrong_score_would_not_survive(run):
    answer = run.predictor.predict([NEGATIVE, POSITIVE])
    assert [labels[0] for labels in answer["labels"]] == CLASSES, "the control for the test below: the two ends of LONG say different things"
    assert all(abs(own_answer(run.model, text)[2][0] - own_answer(run.model, text)[2][1]) > 1 for text in (NEGATIVE, POSITIVE)), "trained logits"


def test_a_text_longer_than_the_row_is_cut_at_its_head_like_the_training_rows(run):
    assert len(ascii_tokenizer().encode(LONG)) > SEQ_LEN - 1 and HEAD != TAIL
    answer = run.predictor.predict([LONG, HEAD, TAIL])
    negative = [dict(zip(labels, scores))["neg"] for labels, scores in zip(answer["labels"], answer["scores"])]
    assert negative[0] == pytest.approx(negative[1], rel=1e-5, abs=1e-6), "the head of a long text is the row it trains on"
    assert negative[0] > 0.5 > negative[2], "and not its tail: the two ends of LONG say different things"


@pytest.mark.parametrize("bad, fragment", [([""], r"document 0.*no token"), (["fine", ""], r"document 1.*no token"), ([], "no document"),
                                           (3, "text"), ([3], "document 0"), ([None], "document 0")])
def test_a_document_with_no_token_and_a_call_that_is_not_text_are_refused_by_name(run, bad, fragment):
    with pytest.raises(ConfigError, match=fragment):
        run.predictor.predict(bad)


def test_the_tokenizer_digest_of_the_record_is_checked_before_the_first_prediction_and_a_record_of_another_task_is_refused(run):
    from train.api.predictor import build_language_model_record
    from train.api.text_predictor import TextClassifierPredictor

    with pytest.raises(ConfigError, match=r"names the tokenizer digest 0{64} .* pins"):
        TextClassifierPredictor(run.model, {**record_of(run), "tokenizer": "0" * 64}).predict(["hello"])
    language = build_language_model_record(model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=SEQ_LEN, tokenizer=ASCII.record())
    with pytest.raises(ConfigError, match="serves 'text_classification' only"):
        TextClassifierPredictor(run.model, language)


# ---------------------------------------------------------------------
# predict: the command
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["documents.txt", "documents.jsonl"])
def test_predict_writes_one_json_list_with_one_answer_per_line(run, tmp_path, capsys, name):
    (tmp_path / "documents.txt").write_text("\n".join(DOCUMENTS) + "\n")
    (tmp_path / "documents.jsonl").write_text("".join(json.dumps({"text": text, "id": number}) + "\n" for number, text in enumerate(DOCUMENTS)))
    out = tmp_path / "answers.json"
    assert cli.main(["predict", f"model={run.out}", f"data={tmp_path / name}", f"out={out}"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    written, expected = read_json(out), run.predictor.predict(DOCUMENTS)
    assert [list(entry) for entry in written] == [["index", "labels", "scores"]] * 3 and [entry["index"] for entry in written] == [0, 1, 2]
    assert [entry["labels"] for entry in written] == expected["labels"] and [entry["scores"] for entry in written] == expected["scores"]


def test_a_directory_and_a_suffix_the_reader_does_not_take_are_refused_before_the_model_is_loaded(run, tmp_path, capsys, monkeypatch):
    from train.api import predictor as predictor_module

    monkeypatch.setattr(predictor_module, "load_model", lambda path: (_ for _ in ()).throw(AssertionError("the model was loaded")))
    (tmp_path / "a.csv").write_text("x\n")
    for data, fragment in [(tmp_path, "is a directory, not a file"), (tmp_path / "a.csv", r"suffix '\.csv'")]:
        assert cli.main(["predict", f"model={run.out}", f"data={data}", f"out={tmp_path / 'answers.json'}"]) == 2
        error = capsys.readouterr().err
        assert re.search(fragment, error) and "AssertionError" not in error, error
    assert not (tmp_path / "answers.json").exists()


def test_the_help_of_predict_and_export_say_what_a_text_classifier_takes_and_gives(capsys):
    assert cli.main(["predict", "help"]) == 0
    assert "a text-classification model a .txt file" in capsys.readouterr().out
    assert cli.main(["export", "help"]) == 0
    text = capsys.readouterr().out
    assert "a text classifier takes float32 token ids" in text and "the [CLS] id" in text and "(batch, classes)" in text


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


def valid_rows(spec: tt.TokenizerSpec, length: int, seed: int = 11) -> np.ndarray:
    """Float ids a classifier may be fed: ``[CLS]``, random ids of the vocabulary (the top of it in the last row), a pad tail."""
    rows = np.random.default_rng(seed).integers(0, spec.n_vocab, size=(3, length)).astype(np.float32)
    rows[-1, 1:length // 2] = spec.n_vocab - 2
    rows[:, 0] = spec.cls_id
    rows[1:, length - length // 4:] = spec.pad_id
    return rows


def within_the_shared_check(got: np.ndarray, expected: np.ndarray) -> float:
    assert got.shape == expected.shape
    scale = max(1.0, float(np.abs(expected).max()))
    worst = float(np.abs(got.astype(np.float64) - expected).max())
    assert worst <= TFLITE_CHECK_RTOL * scale, f"{worst} against an allowance of {TFLITE_CHECK_RTOL * scale}"
    return worst


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_equals_the_model_on_valid_rows(run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={run.out}", f"out={out}", f"format={fmt}"]) == 0
    assert read_json(out.with_name(out.name + ".task.json")) == record_of(run)
    batch = valid_rows(ASCII, SEQ_LEN)
    expected = np.asarray(run.model(batch, training=False), np.float64)
    assert expected.shape == (3, 2) and np.abs(expected).max() > 0.5, "trained logits: a wrong file would show"
    within_the_shared_check(np.asarray(read_back(fmt, out, batch)), expected)


@pytest.fixture(scope="module", params=FAMILIES)
def real_vocabulary_run(request, tmp_path_factory):
    """An untrained one-block classifier at the gpt2 vocabulary (50,261 ids) and its record: the run directory ``load_model`` reads."""
    from train.api.predictor import build_text_classification_record

    family, spec = request.param, tt.tokenizer_spec("gpt2")
    args = {"bert": {"num_layers": 1}, "gpt2": {"depth": 1}}[family]
    factory = {"bert": tm.create_bert_classifier, "gpt2": tm.create_gpt2_classifier}[family]
    root = tmp_path_factory.mktemp(f"classifier_real_{family}")
    keras.utils.set_random_seed(0)
    keras.backend.clear_session()
    model = factory(variant="tiny", num_classes=2, input_shape=(REAL_SEQ,), vocab_size=spec.vocab_size, pad_id=spec.pad_id,
                    cls_id=spec.cls_id, **args)
    model.save(str(root / "best_model.keras"))
    record = build_text_classification_record(
        model=f"{family}/tiny", family=family, variant="tiny", seq_len=REAL_SEQ, tokenizer=spec.record(), class_names=CLASSES,
        variant_overrides=args)
    (root / "task.json").write_text(json.dumps(record))
    return SimpleNamespace(out=root, record=record, model=model, spec=spec)


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_of_a_model_at_the_gpt2_vocabulary_equals_the_model_on_valid_rows(real_vocabulary_run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={real_vocabulary_run.out}", f"out={out}", f"format={fmt}"]) == 0
    batch = valid_rows(real_vocabulary_run.spec, REAL_SEQ)
    within_the_shared_check(np.asarray(read_back(fmt, out, batch)), np.asarray(real_vocabulary_run.model(batch, training=False), np.float64))


@pytest.mark.parametrize("fmt", FORMATS)
def test_another_input_shape_is_refused_before_a_file_is_written(run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    with pytest.raises(ConfigError, match=rf"\(64,\) is not the trained length \({SEQ_LEN},\) \(seq_len\) of this text_classification model"):
        exporting.export(model=run.model, out=out, format=fmt, input_shape=(64,), record=record_of(run))
    assert list(tmp_path.iterdir()) == []


def test_a_vocabulary_a_float32_id_cannot_carry_exactly_is_refused_by_name_for_a_classifier_too(run):
    from train.api.text_predictor import check_export

    check_export({**record_of(run), "vocab_size": 2 ** 24 - 1}, "onnx", (SEQ_LEN,))
    with pytest.raises(ConfigError, match=r"vocab_size 16777216 .*float32.*exactly"):
        check_export({**record_of(run), "vocab_size": 2 ** 24}, "onnx", (SEQ_LEN,))


def test_the_rule_feeds_the_check_valid_ids_and_a_file_wrong_only_above_id_zero_is_refused(run, tmp_path, monkeypatch):
    from train.api.predictor import EXPORT_RULES

    rule = EXPORT_RULES[TASK]
    batch = rule.sample(record_of(run), (2, SEQ_LEN))
    assert batch.dtype == np.float32 and batch.shape == (2, SEQ_LEN) and (batch == np.floor(batch)).all()
    assert 0 <= batch.min() and batch.max() < ASCII.vocab_size and batch.max() > 50, "ids across the vocabulary, not one id"
    real = exporting._tflite_logits
    wrong_above_zero = lambda blob, x: real(blob, x) + 0.5 * (x >= 1).any(axis=1, keepdims=True)  # noqa: E731 - right for id 0 only
    monkeypatch.setattr(exporting, "_tflite_logits", wrong_above_zero)
    out = tmp_path / "wrong.tflite"
    with pytest.raises(ConfigError, match=r"the tflite file converted with TensorFlow operators differs from the Keras model by"):
        exporting.export(model=run.model, out=out, format="tflite", input_shape=(SEQ_LEN,), record=record_of(run))
    assert not out.exists() and not (tmp_path / "wrong.tflite.task.json").exists()
    sibling = tmp_path / "control.tflite"  # the control: the check batch of every other task (random floats, all id 0) cannot see that file
    exporting._export_tflite(run.model, sibling, exporting._input_spec(run.model, (SEQ_LEN,), rank=1), lambda x: run.model(x, training=False))
    assert sibling.is_file()


def test_the_export_note_says_what_goes_in_what_a_row_is_and_what_comes_out(run):
    from train.api.predictor import EXPORT_RULES

    note = EXPORT_RULES[TASK].note(record_of(run))
    for fragment in ("float32", f"(batch, {SEQ_LEN})", "token ids", "3.0", f"[CLS] id {ASCII.cls_id}", f"pad id {ASCII.pad_id}", "(batch, 2)",
                     "neg, pos"):
        assert fragment in note, (fragment, note)


# ---------------------------------------------------------------------
# init: a classifier run is a source
# ---------------------------------------------------------------------


def seeded_run(root: Path, family: str, source: Path, **changes: Any):
    keras.backend.clear_session()
    out = root / "run"
    train_text_classification(**classifier_request(out, family, init=str(source), epochs=1, optim_args=STILL, **changes))
    return out, read_json(out / "results_summary.json")


def archive_weights(directory: Path) -> Dict[str, Any]:
    return init_from._by_path(keras.models.load_model(str(directory / "best_model.keras"), compile=False), "run")


def test_a_classifier_run_seeds_a_run_of_its_family_with_every_weight_copied_the_head_included(run, tmp_path):
    out, summary = seeded_run(tmp_path, run.family, run.out)
    block, held = summary["init"], archive_weights(run.out)
    assert (block["task"], block["family"], block["relation"], block["pinned"]) == (TASK, run.family, "same_task", {})
    assert block["geometry"] == {"seq_len": SEQ_LEN, "vocab_size": ASCII.vocab_size}
    assert block["transfer"]["copied"] == len(held) and not (block["transfer"]["reset"] or block["transfer"]["dropped"])
    assert read_json(out / "task.json")["init"] == block, "the block is in the record now, and in the summary"
    assert summary["loaded_success_value"] == pytest.approx(run.summary["success_value"], abs=1e-6) and summary["success"] is False
    for path, value in archive_weights(out).items():
        np.testing.assert_array_equal(value.numpy(), held[path].numpy(), err_msg=path)


def test_another_class_names_in_the_source_reset_the_head_and_copy_the_rest(runs, tmp_path):
    from train.api.predictor import build_text_classification_record, load_model

    source = tmp_path / "source"
    shutil.copytree(runs["gpt2"].out, source)
    (source / "task.json").write_text(json.dumps({**record_of(runs["gpt2"]), "class_names": ["no", "yes"]}))
    assert load_model(source).record["class_names"] == ["no", "yes"]
    block = seeded_run(tmp_path, "gpt2", source)[1]["init"]
    held = archive_weights(runs["gpt2"].out)
    head = sorted(path for path in held if path.split("/")[0] == "task_head")
    assert sorted(block["transfer"]["reset"]) == head and len(head) == 8 and block["transfer"]["copied"] == len(held) - 8
    assert init_from.HEAD_KEYS[TASK] == ("class_names",)
    same = build_text_classification_record(model="gpt2/tiny", family="gpt2", variant="tiny", seq_len=SEQ_LEN, tokenizer=ASCII.record(), class_names=CLASSES)
    assert not init_from.head_differs(TASK, same, same) and init_from.head_differs(TASK, same, {**same, "class_names": ["no", "yes"]})


@pytest.mark.parametrize("target, source", [("gpt2", "bert"), ("bert", "gpt2")])
def test_a_classifier_run_of_the_other_family_is_refused_by_the_declared_sources_sentence_before_the_claim(runs, tmp_path, target, source):
    taken = {"gpt2": "gpt2 (text_classification) or gpt2_lm (language_model_pretraining)", "bert": "bert (text_classification)"}[target]
    with pytest.raises(ConfigError, match=rf"Invalid value for 'init': {source} \(text_classification\) cannot seed {target} "
                                          rf"\(text_classification\); {target} takes weights from {re.escape(taken)}$"):
        train_text_classification(**classifier_request(tmp_path / "run", target, init=str(runs[source].out)))
    assert not (tmp_path / "run").exists(), "a pre-claim refusal claimed the run directory"
