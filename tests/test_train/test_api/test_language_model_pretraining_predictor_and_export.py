"""The ``language_model_pretraining`` record, ``predict`` and ``export`` (plan-2026-11-24T092348-7806a290 step 5, D-007, D-008).

A language-model run writes a schema-v2 ``task.json`` (``task: "language_model_pretraining"``) before ``fit``, and its directory can
be predicted with and exported. ``predict`` answers per document, in nats per token: ``n_tokens``, ``mean_nll`` (the mean negative
log-likelihood of the ids the model predicts, the end-of-text id after the document included), ``perplexity`` and the five most
likely next tokens; ``export`` writes the model as a file taking float32 token ids at the trained length. Pinned here, on one real
tiny run (the offline synthetic bank, ``ascii95``, 64 wide, 2 blocks, 32 ids), CPU, written under ``tmp_path``:

* **the record**: it passes ``record_problems`` and equals the record the test builds itself; a record missing any key, or holding an
  impossible value, is refused naming the key; a resume compares ``seq_len``, the encoding, the vocabulary and the tokenizer digest
  (``DATA_KEYS``) and nothing of how the predictor answers; ``load_model`` refuses a record whose length, vocabulary or variant is
  not the model's;
* **predict**: every number equals the test's own numpy computation from the model's logits (windows of ``seq_len + 1`` ids, the
  last one right-padded and scored on its real positions only, whatever the filler), a one-window document scores the ``val_loss``
  Keras reports for that window, the five next tokens are an independent ``argsort`` of the last real position and the ids above the
  BPE vocabulary are decoded by name, and the command writes one JSON file of a ``.txt`` or a ``.jsonl`` file;
* **export**: every format equals the model on a batch of VALID ids, on the tiny run and on a real gpt2-vocabulary build
  (50,261 wide); another length is refused before a file is written; the tflite check feeds the
  rule's valid ids, so a file wrong only for ids above 0 is refused (random floats in [0, 1) cast to id 0 would not show it).
"""

import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from train.api import cli, exporting
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
from train.api.exporting import FORMATS, SUFFIX, TFLITE_CHECK_RTOL
from train.api.tasks.language_model_pretraining import train_language_model_pretraining
from train.api.text_models import create_gpt2_lm
from train.common import text_data as td

from ._text_world import ASCII_VOCAB, SEQ_LEN, TINY_MODEL_ARGS, request, world  # noqa: F401  (``world`` is a fixture)
from .test_export_roundtrip import read_back

TASK = "language_model_pretraining"
#: The keys of a language-model record, in the order they are written (Problem Statement 3, D-007).
KEYS = ("schema_version", "task", "model", "family", "variant", "seq_len", "encoding", "n_vocab", "vocab_size", "eot_id",
        "pad_id", "tokenizer", "tokenizer_files", "output")
REAL_SEQ, REAL_VOCAB = 64, 50261

SHORT = "the cat sat"
ONE_WINDOW = "abcdefghijklmnopqrstuvwxyz012345"                  # 32 ids and an end-of-text id: exactly one window of 33
LONG = "the quick brown fox jumps over the lazy dog and then it sat down by the river bank all day"
LONE_EOT = ("abcdef " * 10)[:66]                                  # 66 ids and an end-of-text id: 33 + 33 + one id left over
DOCUMENTS = [SHORT, ONE_WINDOW, LONG, LONE_EOT]

NO_PIN = pytest.mark.skipif(
    not all((Path(tt.DEFAULT_TOKENIZER_DIR) / name).is_file() for name, _ in tt.tokenizer_spec("gpt2").files),
    reason=f"the pinned tokenizer directory {tt.DEFAULT_TOKENIZER_DIR} is not on this machine",
)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def ascii_tokenizer() -> td.Tokenizer:
    return td.open_tokenizer(tt.tokenizer_spec("ascii95"))


@pytest.fixture(scope="module")
def run(world, tmp_path_factory):  # noqa: F811
    """One finished 2-epoch run, and what ``load_model`` makes of it."""
    from train.api.predictor import load_model

    out = tmp_path_factory.mktemp("language_record") / "run"
    train_language_model_pretraining(**request(out, epochs=2))
    predictor = load_model(out)
    return SimpleNamespace(out=out, record=read_json(out / "task.json"), predictor=predictor, model=predictor.model)


def record_of(run: SimpleNamespace) -> Dict[str, Any]:
    return dict(run.record)


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------


def test_the_record_passes_the_reader_and_is_the_one_the_test_builds_from_what_it_asked(run):
    from train.api.predictor import OPTIONAL_RECORD_KEYS, RECORD_KEYS, build_language_model_record, record_problems

    record = record_of(run)
    assert record_problems(record) == []
    assert RECORD_KEYS[TASK] == KEYS and tuple(record)[: len(KEYS)] == KEYS and set(record) - set(KEYS) <= set(OPTIONAL_RECORD_KEYS)
    spec = tt.tokenizer_spec("ascii95")
    assert (record["schema_version"], record["task"], record["model"], record["family"], record["variant"]) == (
        2, TASK, "gpt2_lm/tiny", "gpt2_lm", "tiny")
    assert (record["seq_len"], record["encoding"], record["n_vocab"], record["vocab_size"]) == (SEQ_LEN, "ascii95", 101, ASCII_VOCAB)
    assert (record["eot_id"], record["pad_id"], record["tokenizer"], record["tokenizer_files"], record["output"]) == (
        spec.eot_id, spec.pad_id, spec.digest, {}, "next_token_logits")
    # The keys the table does not hold are the run's own overrides of the variant (64 wide, 2 blocks, 2 heads).
    assert record["variant_overrides"] == TINY_MODEL_ARGS
    assert record == build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=SEQ_LEN, tokenizer=spec.record(),
        variant_overrides=TINY_MODEL_ARGS)


@pytest.mark.parametrize("key", KEYS)
def test_a_record_missing_any_key_is_refused(run, key):
    from train.api.predictor import record_problems

    problems = record_problems({k: v for k, v in record_of(run).items() if k != key})
    assert problems, f"a record without {key!r} was accepted"
    if key not in ("schema_version", "task"):
        assert problems == [f"missing key(s) {key!r}"]


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"seq_len": 0}, "'seq_len'"),
        ({"seq_len": 32.0}, "'seq_len'"),
        ({"seq_len": True}, "'seq_len'"),
        ({"vocab_size": 0}, "'vocab_size'"),
        ({"n_vocab": 200}, "'n_vocab'"),
        ({"pad_id": 101}, "'pad_id'"),
        ({"eot_id": -1}, "'eot_id'"),
        ({"encoding": "cl100k_base"}, "'encoding'"),
        ({"tokenizer": "abc"}, "'tokenizer'"),
        ({"tokenizer": "A" * 64}, "'tokenizer'"),
        ({"tokenizer_files": ["a"]}, "'tokenizer_files'"),
        ({"tokenizer_files": {"abc": "short"}}, "'tokenizer_files'"),
        ({"tokenizer_files": {"abc": "0" * 64}}, "'tokenizer_files'"),
        ({"output": "logits"}, "'output'"),
    ],
    ids=["seq_zero", "seq_float", "seq_bool", "vocab_zero", "n_vocab_above_vocab", "pad_outside", "eot_negative", "other_encoding",
         "short_digest", "upper_digest", "files_not_object", "file_hash_short", "ascii_with_files", "other_output"],
)
def test_a_record_of_impossible_values_is_refused_naming_the_key(run, change, fragment):
    from train.api.predictor import record_problems

    problems = record_problems({**record_of(run), **change})
    assert any(fragment in line for line in problems), problems


def test_a_gpt2_record_needs_its_files_and_one_built_from_the_pinned_spec_passes():
    from train.api.predictor import build_language_model_record, record_problems

    spec = tt.tokenizer_spec("gpt2")
    record = build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=REAL_SEQ, tokenizer=spec.record())
    assert record_problems(record) == []
    assert record["tokenizer_files"] == dict(spec.files) and record["vocab_size"] == REAL_VOCAB and record["pad_id"] == 50259
    assert any("'tokenizer_files'" in line for line in record_problems({**record, "tokenizer_files": {}}))


def test_a_resume_compares_the_length_the_encoding_the_vocabulary_and_the_tokenizer_digest(run):
    from train.api.predictor import DATA_KEYS, record_differences

    record = record_of(run)
    assert set(DATA_KEYS[TASK]) == {"seq_len", "encoding", "vocab_size", "tokenizer"}
    for key, value in [("seq_len", 64), ("encoding", "gpt2"), ("vocab_size", 50261), ("tokenizer", "0" * 64)]:
        assert [line.split(":")[0] for line in record_differences(record, {**record, key: value})] == [key]
    assert record_differences(record, {**record, "output": "something_else"}) == [], "how the predictor answers is not what the run saw"


@pytest.mark.parametrize("key, value", [("seq_len", 64), ("tokenizer", "0" * 64)])
def test_a_resume_of_a_run_on_another_length_or_tokenizer_is_refused_by_the_record(run, tmp_path, key, value):
    from train.api.resume_decisions import check_data_unchanged

    shutil.copy(run.out / "task.json", tmp_path / "task.json")
    counts = {"n_train": 1, "n_select": 1, "n_test": 1}
    data = SimpleNamespace(n_train=1, n_select=1, n_test=1)
    resume = SimpleNamespace(state={"data_counts": counts})
    check_data_unchanged(tmp_path, None, record_of(run), data, resume)  # the control: the same record is the same data
    with pytest.raises(ConfigError, match=rf"the data is not the data the run started with \({key}: "):
        check_data_unchanged(tmp_path, None, {**record_of(run), key: value}, data, resume)


@pytest.mark.parametrize(
    "change, fragment",
    [({"seq_len": 64}, "does not take the seq_len 64"), ({"vocab_size": 120}, r"\(32, 101\).*vocabulary of 120")],
    ids=["other_length", "other_vocabulary"],
)
def test_load_model_refuses_a_record_whose_length_or_vocabulary_is_not_the_models(run, tmp_path, change, fragment):
    from train.api.predictor import load_model

    copy = tmp_path / "run"
    copy.mkdir()
    shutil.copy(run.out / "best_model.keras", copy / "best_model.keras")
    (copy / "task.json").write_text(json.dumps({**record_of(run), **change}))
    with pytest.raises(ConfigError, match=fragment):
        load_model(copy)


def test_load_model_refuses_a_record_naming_another_variant_and_loads_the_honest_one(run, tmp_path):
    from train.api.predictor import load_model

    copy = tmp_path / "run"
    copy.mkdir()
    shutil.copy(run.out / "best_model.keras", copy / "best_model.keras")
    (copy / "task.json").write_text(json.dumps(record_of(run)))
    assert type(load_model(copy)).__name__ == "LanguageModelPredictor", "the control: the honest copy loads"
    (copy / "task.json").write_text(json.dumps({**record_of(run), "model": "gpt2_lm/small", "variant": "small"}))
    with pytest.raises(ConfigError, match=r"is not that variant \(max_seq_len is 512 in the model, 1024 in 'small'"):
        load_model(copy)


# ---------------------------------------------------------------------
# predict: the test's own numpy computation
# ---------------------------------------------------------------------


def own_rows(ids: List[int], length: int, eot: int, filler: int):
    """The ``(x, y)`` rows of one document, written here and not read from the module: the ids and an end-of-text id are cut into
    windows of ``length + 1``; each window predicts its last ids from the earlier ones; a window that holds one id has no target."""
    stream = list(ids) + [eot]
    rows = []
    for start in range(0, len(stream), length + 1):
        chunk = stream[start:start + length + 1]
        if len(chunk) >= 2:
            rows.append((chunk[:-1] + [filler] * (length - len(chunk) + 1), chunk[1:]))
    return rows


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    top = z.max(axis=-1, keepdims=True)
    return z - top - np.log(np.exp(z - top).sum(axis=-1, keepdims=True))


def own_scores(model: Any, text: str, filler: int, *, count_filler: bool = False):
    """``(n_tokens, mean_nll, scored)`` of a document: one model call per row, the NLL summed over its real positions."""
    spec = tt.tokenizer_spec("ascii95")
    ids = ascii_tokenizer().encode(text)
    total, scored = 0.0, 0
    for x, y in own_rows(ids, SEQ_LEN, spec.eot_id, filler):
        logp = log_softmax(np.asarray(model(np.asarray([x], np.float32), training=False))[0])
        picked = logp[np.arange(len(y)), y]
        total -= float(picked.sum())
        scored += len(y)
        if count_filler:
            total -= float(logp[np.arange(len(y), SEQ_LEN), [filler] * (SEQ_LEN - len(y))].sum())
            scored += SEQ_LEN - len(y)
    return len(ids), total / scored, scored


def own_next_tokens(model: Any, text: str, filler: int):
    """The logits after the document's last id, given its last ``seq_len`` ids: ``(log probabilities, top five ids)``."""
    ids = ascii_tokenizer().encode(text)[-SEQ_LEN:]
    x = ids + [filler] * (SEQ_LEN - len(ids))
    logp = log_softmax(np.asarray(model(np.asarray([x], np.float32), training=False))[0][len(ids) - 1])
    return logp, [int(i) for i in np.argsort(-logp, kind="stable")[:5]]


@pytest.mark.parametrize("text", DOCUMENTS, ids=["short", "one_window", "three_windows", "a_lone_end_of_text_left_over"])
def test_every_number_of_a_document_is_the_independent_numpy_computation(run, text):
    answer = run.predictor.predict([text])["documents"]
    assert len(answer) == 1
    n_tokens, mean_nll, scored = own_scores(run.model, text, filler=7)
    entry = answer[0]
    assert entry["n_tokens"] == n_tokens == len(ascii_tokenizer().encode(text))
    assert entry["mean_nll"] == pytest.approx(mean_nll, rel=1e-4, abs=1e-5)
    assert entry["perplexity"] == pytest.approx(math.exp(entry["mean_nll"]), rel=1e-9)
    # the first id of every window is context only (as in the training windows): no id is scored twice
    assert scored == n_tokens + 1 - math.ceil((n_tokens + 1) / (SEQ_LEN + 1))
    logp, expected_ids = own_next_tokens(run.model, text, filler=3)
    assert [item["token"] for item in entry["next_tokens"]] == expected_ids
    assert [item["probability"] for item in entry["next_tokens"]] == pytest.approx([math.exp(logp[i]) for i in expected_ids], rel=1e-4)
    assert [item["text"] for item in entry["next_tokens"]] == [ascii_tokenizer().decode([i]) for i in expected_ids]
    assert [item["probability"] for item in entry["next_tokens"]] == sorted((item["probability"] for item in entry["next_tokens"]), reverse=True)


def test_the_answer_of_a_document_does_not_depend_on_its_neighbours_or_on_the_batch(run):
    alone = [run.predictor.predict([text])["documents"][0] for text in DOCUMENTS]
    together = run.predictor.predict(DOCUMENTS)["documents"]
    reversed_ = run.predictor.predict(DOCUMENTS[::-1])["documents"][::-1]
    for one, many, back in zip(alone, together, reversed_):
        assert one["n_tokens"] == many["n_tokens"] == back["n_tokens"]
        assert many["mean_nll"] == pytest.approx(one["mean_nll"], rel=1e-5) and back["mean_nll"] == pytest.approx(one["mean_nll"], rel=1e-5)
        assert [i["token"] for i in many["next_tokens"]] == [i["token"] for i in one["next_tokens"]]
    assert run.predictor.predict(SHORT)["documents"][0] == run.predictor.predict([SHORT])["documents"][0], "one text is a list of one"


def test_a_short_document_is_scored_on_its_real_positions_only_and_the_filler_does_not_matter(run):
    entry = run.predictor.predict([SHORT])["documents"][0]
    real = own_scores(run.model, SHORT, filler=0)
    assert real[2] == len(SHORT), "11 ids, and the id after the last: 11 targets"
    for filler in (0, 7, 50, 100):
        assert own_scores(run.model, SHORT, filler=filler)[1] == pytest.approx(real[1], rel=1e-5), "the model is causal"
    assert entry["mean_nll"] == pytest.approx(real[1], rel=1e-4, abs=1e-5)
    counted = own_scores(run.model, SHORT, filler=0, count_filler=True)[1]
    assert abs(counted - entry["mean_nll"]) > 0.05, "the control: counting the 21 filler positions would be another number"


def test_the_mean_nll_of_a_one_window_document_is_the_val_loss_keras_reports_for_that_window(run):
    tokenizer = ascii_tokenizer()
    ids = tokenizer.encode(ONE_WINDOW) + [tokenizer.spec.eot_id]
    assert len(ids) == SEQ_LEN + 1
    x, y = np.asarray([ids[:-1]], np.float32), np.asarray([ids[1:]])
    model = keras.models.load_model(str(run.out / "best_model.keras"), compile=False)
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    val_loss = float(model.evaluate(x, y, batch_size=1, verbose=0))
    assert run.predictor.predict([ONE_WINDOW])["documents"][0]["mean_nll"] == pytest.approx(val_loss, rel=1e-5)


@pytest.mark.parametrize("bad, fragment", [([""], r"document 0.*no token"), (["fine", ""], r"document 1.*no token"),
                                           ([], "no document"), (3, "text"), ([3], "document 0"), ([None], "document 0")])
def test_a_document_with_no_token_and_a_call_that_is_not_text_are_refused_by_name(run, bad, fragment):
    with pytest.raises(ConfigError, match=fragment):
        run.predictor.predict(bad)


def test_a_score_that_cannot_be_written_as_json_is_refused_by_name(run):
    from train.api.text_predictor import LanguageModelPredictor

    def broken(batch, training=False):
        logits = np.zeros((len(batch), SEQ_LEN, ASCII_VOCAB), np.float32)
        logits[:, :, 5] = 1e4        # every id but 5 costs 1e4 nats: the mean is finite, its exponential is not
        return logits

    with pytest.raises(ConfigError, match=r"score of document 0 is not a finite number \(mean_nll 1\d{4}"):
        LanguageModelPredictor(broken, record_of(run)).predict([SHORT])


@NO_PIN
def test_the_next_tokens_above_the_bpe_vocabulary_are_decoded_by_name_through_the_record():
    from train.api.predictor import build_language_model_record
    from train.api.text_predictor import LanguageModelPredictor

    spec = tt.tokenizer_spec("gpt2")
    record = build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=8, tokenizer=spec.record())
    top = [spec.pad_id, spec.cls_id, spec.mask_id, 11, 13]

    def model(batch, training=False):
        logits = np.zeros((len(batch), 8, spec.vocab_size), np.float32)
        logits[:, :, top] = [9.0, 8.0, 7.0, 6.0, 5.0]
        return logits

    entry = LanguageModelPredictor(model, record).predict(["hello world"])["documents"][0]
    assert [item["token"] for item in entry["next_tokens"]] == top
    assert [item["text"] for item in entry["next_tokens"]] == ["<|pad|>", "<|cls|>", "<|mask|>", ",", "."]


def test_the_tokenizer_is_reopened_from_the_record_and_refused_when_its_files_are_not_the_records(run, tmp_path, monkeypatch):
    from train.api.predictor import build_language_model_record
    from train.api.text_predictor import LanguageModelPredictor

    from ._text_world import wrong_tokenizer_dir

    spec = tt.tokenizer_spec("gpt2")
    record = build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=8, tokenizer=spec.record())
    predictor = LanguageModelPredictor(lambda batch, training=False: None, record)  # opening the files waits for the first call
    monkeypatch.setattr(tt, "DEFAULT_TOKENIZER_DIR", str(wrong_tokenizer_dir(tmp_path)))
    with pytest.raises(ConfigError, match=r"the tokenizer of this run \(gpt2\) cannot be opened.*is not the file"):
        predictor.predict(["hello"])
    doctored = {**record_of(run), "tokenizer": "0" * 64}
    with pytest.raises(ConfigError, match=r"names the tokenizer digest 0{64} .* pins"):
        LanguageModelPredictor(run.model, doctored).predict(["hello"])


# ---------------------------------------------------------------------
# predict: the command
# ---------------------------------------------------------------------


def write_documents(directory: Path):
    texts = ["the cat sat on the mat\n", LONG, SHORT]
    plain = directory / "documents.txt"
    plain.write_text("\n".join(text.rstrip("\n") for text in texts) + "\n")
    lines = directory / "documents.jsonl"
    lines.write_text("".join(json.dumps({"text": text.rstrip("\n"), "id": number}) + "\n" for number, text in enumerate(texts)))
    return plain, lines, [text.rstrip("\n") for text in texts]


@pytest.mark.parametrize("name", ["documents.txt", "documents.jsonl"])
def test_predict_writes_one_json_file_with_one_answer_per_line(run, tmp_path, capsys, name):
    plain, lines, texts = write_documents(tmp_path)
    out = tmp_path / "answers.json"
    assert cli.main(["predict", f"model={run.out}", f"data={tmp_path / name}", f"out={out}"]) == 0
    assert capsys.readouterr().out == f"{out}\n"
    written = read_json(out)
    assert list(written) == ["task", "model", "data", "seq_len", "tokenizer", "scoring", "documents"]
    assert (written["task"], written["model"], written["data"], written["seq_len"]) == (TASK, "gpt2_lm/tiny", str(tmp_path / name), SEQ_LEN)
    assert written["tokenizer"] == tt.tokenizer_spec("ascii95").digest and written["scoring"]
    expected = run.predictor.predict(texts)["documents"]
    assert [entry["index"] for entry in written["documents"]] == [0, 1, 2]
    for entry, wanted in zip(written["documents"], expected):
        assert {key: value for key, value in entry.items() if key != "index"} == wanted, "JSON floats round-trip exactly"


def test_a_byte_order_mark_and_windows_line_ends_are_not_part_of_the_text(tmp_path):
    from train.api.text_predictor import read_documents

    plain = tmp_path / "bom.txt"
    plain.write_bytes(b"\xef\xbb\xbfthe cat sat\r\nthe dog\r\n")
    lines = tmp_path / "bom.jsonl"
    lines.write_bytes(b"\xef\xbb\xbf" + json.dumps({"text": "the cat sat"}).encode() + b"\r\n")
    assert read_documents(plain) == ["the cat sat", "the dog"] and read_documents(lines) == ["the cat sat"]


def test_a_predict_that_cannot_answer_writes_no_file(run, tmp_path, monkeypatch):
    plain, _, _ = write_documents(tmp_path)
    out = tmp_path / "answers.json"
    import train.api.text_predictor as module

    monkeypatch.setattr(module.LanguageModelPredictor, "predict", lambda self, x: (_ for _ in ()).throw(ConfigError("cannot")))
    assert cli.main(["predict", f"model={run.out}", f"data={plain}", f"out={out}"]) == 2
    assert not out.exists(), "the answer is written last, in one piece"


def put(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "make, fragment",
    [
        (lambda d: d, r"is a directory, not a file"),
        (lambda d: put(d, "a.csv", "x\n"), r"suffix '\.csv'"),
        (lambda d: put(d, "e.txt", ""), r"holds no document"),
        (lambda d: put(d, "g.txt", "ok\n\nok\n"), r"line 2 .*empty"),
        (lambda d: put(d, "b.jsonl", '{"text": "ok"}\nnot json\n'), r"line 2 .*not JSON"),
        (lambda d: put(d, "n.jsonl", '{"text": "ok"}\n{"body": "x"}\n'), r'line 2 .*"text"'),
        (lambda d: put(d, "t.jsonl", '{"text": 3}\n'), r'line 1 .*"text"'),
    ],
    ids=["directory", "csv", "empty_file", "empty_line", "bad_json", "no_text_key", "text_not_a_string"],
)
def test_a_data_file_that_cannot_be_read_as_documents_is_refused_by_name_before_the_model_is_loaded(
        run, tmp_path, capsys, monkeypatch, make, fragment):
    from train.api import predictor as predictor_module

    monkeypatch.setattr(predictor_module, "load_model", lambda path: (_ for _ in ()).throw(AssertionError("the model was loaded")))
    data = make(tmp_path)
    out = tmp_path / "answers.json"
    assert cli.main(["predict", f"model={run.out}", f"data={data}", f"out={out}"]) == 2
    error = capsys.readouterr().err
    assert re.search(fragment, error) and "AssertionError" not in error, error
    assert not out.exists()


def test_predict_help_and_export_help_say_what_a_language_model_takes_and_gives(capsys):
    assert cli.main(["predict", "help"]) == 0
    assert ".txt" in capsys.readouterr().out
    assert cli.main(["export", "help"]) == 0
    text = capsys.readouterr().out
    assert "a language model takes float32 token ids" in text and "next-token logits" in text


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


def valid_ids(vocab: int, length: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab, size=(3, length)).astype(np.float32)
    ids[-1, length - length // 4:] = vocab - 2          # the top of the vocabulary: the ids a random float would never reach
    return ids


def within_the_shared_check(got: np.ndarray, expected: np.ndarray) -> float:
    assert got.shape == expected.shape
    scale = max(1.0, float(np.abs(expected).max()))
    worst = float(np.abs(got.astype(np.float64) - expected).max())
    assert worst <= TFLITE_CHECK_RTOL * scale, f"{worst} against an allowance of {TFLITE_CHECK_RTOL * scale}"
    return worst


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_equals_the_model_on_valid_ids(run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={run.out}", f"out={out}", f"format={fmt}"]) == 0
    assert read_json(out.with_name(out.name + ".task.json")) == record_of(run)
    batch = valid_ids(ASCII_VOCAB, SEQ_LEN)
    expected = np.asarray(run.model(batch, training=False), np.float64)
    assert expected.shape == (3, SEQ_LEN, ASCII_VOCAB)
    within_the_shared_check(np.asarray(read_back(fmt, out, batch)), expected)


@pytest.fixture(scope="module")
def real_vocabulary_run(tmp_path_factory):
    """An untrained one-block model at the gpt2 vocabulary (50,261 wide) and its record: the run directory ``load_model`` reads."""
    from train.api.predictor import build_language_model_record

    root = tmp_path_factory.mktemp("language_real_vocabulary")
    keras.utils.set_random_seed(0)
    model = create_gpt2_lm(variant="tiny", input_shape=(REAL_SEQ,), vocab_size=REAL_VOCAB, depth=1)
    model.save(str(root / "best_model.keras"))
    record = build_language_model_record(
        model="gpt2_lm/tiny", family="gpt2_lm", variant="tiny", seq_len=REAL_SEQ, tokenizer=tt.tokenizer_spec("gpt2").record(),
        variant_overrides={"depth": 1})
    (root / "task.json").write_text(json.dumps(record))
    return SimpleNamespace(out=root, record=record, model=model)


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_export_of_a_model_at_the_gpt2_vocabulary_equals_the_model_on_valid_ids(real_vocabulary_run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    assert cli.main(["export", f"model={real_vocabulary_run.out}", f"out={out}", f"format={fmt}"]) == 0
    batch = valid_ids(REAL_VOCAB, REAL_SEQ)
    expected = np.asarray(real_vocabulary_run.model(batch, training=False), np.float64)
    assert expected.shape == (3, REAL_SEQ, REAL_VOCAB)
    within_the_shared_check(np.asarray(read_back(fmt, out, batch)), expected)


@pytest.mark.parametrize("fmt", FORMATS)
def test_another_input_shape_is_refused_before_a_file_is_written(run, fmt, tmp_path):
    out = tmp_path / f"exported{SUFFIX.get(fmt, '')}"
    with pytest.raises(ConfigError, match=rf"\(64,\) is not the trained length \({SEQ_LEN},\)"):
        exporting.export(model=run.model, out=out, format=fmt, input_shape=(64,), record=record_of(run))
    assert list(tmp_path.iterdir()) == []


def test_a_vocabulary_a_float32_id_cannot_carry_exactly_is_refused_by_name(run):
    from train.api.text_predictor import check_export

    check_export({**record_of(run), "vocab_size": 2 ** 24 - 1}, "onnx", (SEQ_LEN,))
    with pytest.raises(ConfigError, match=r"vocab_size 16777216 .*float32.*exactly"):
        check_export({**record_of(run), "vocab_size": 2 ** 24}, "onnx", (SEQ_LEN,))


def test_the_rule_feeds_the_check_valid_ids_and_a_file_wrong_only_above_id_zero_is_refused(run, tmp_path, monkeypatch):
    from train.api.predictor import EXPORT_RULES

    rule = EXPORT_RULES[TASK]
    batch = rule.sample(record_of(run), (2, SEQ_LEN))
    assert batch.dtype == np.float32 and batch.shape == (2, SEQ_LEN) and (batch == np.floor(batch)).all()
    assert 0 <= batch.min() and batch.max() < ASCII_VOCAB and batch.max() > 50, "ids across the vocabulary, not one id"
    assert (batch[-1, -SEQ_LEN // 4:] == run.record["pad_id"]).all(), "a pad tail"
    assert np.array_equal(batch, rule.sample(record_of(run), (2, SEQ_LEN))), "seeded: the same batch every time"

    real = exporting._tflite_logits
    wrong_above_zero = lambda blob, x: real(blob, x) + 0.5 * (x >= 1)[..., None]  # noqa: E731 - right for id 0, wrong for the rest
    monkeypatch.setattr(exporting, "_tflite_logits", wrong_above_zero)
    out = tmp_path / "wrong.tflite"
    with pytest.raises(ConfigError, match=r"the tflite file converted with TensorFlow operators differs from the Keras model by"):
        exporting.export(model=run.model, out=out, format="tflite", input_shape=(SEQ_LEN,), record=record_of(run))
    assert not out.exists() and not (tmp_path / "wrong.tflite.task.json").exists()
    # the control: the batch the check fed before this step (random floats in [0, 1), every one id 0) cannot see that file
    spec = exporting._input_spec(run.model, (SEQ_LEN,), rank=1)
    sibling = tmp_path / "control.tflite"
    exporting._export_tflite(run.model, sibling, spec, lambda x: run.model(x, training=False))
    assert sibling.is_file(), "the file the rule's batch refuses is written when the check feeds random floats"


def test_the_export_note_says_what_goes_in_and_what_comes_out(run):
    from train.api.predictor import EXPORT_RULES

    note = EXPORT_RULES[TASK].note(record_of(run))
    for fragment in ("float32", f"(batch, {SEQ_LEN})", "token ids", "3.0", str(ASCII_VOCAB), "pad id 0", f"(batch, {SEQ_LEN}, {ASCII_VOCAB})"):
        assert fragment in note, (fragment, note)
