"""The text data layer: a tokenizer record, a document-keyed split and a stream that is a pure function of (seed, step).

``train.api.text_tokenizer`` (the TensorFlow-free half: the tokenizer record, its digest and the check of the pinned directory)
and ``train.common.text_data`` (the documents, the split, the windows, the stream and the dataset builders) are what the two
language tasks read. What is pinned here, on tiny hand-made corpora and the offline ``ascii95`` tokenizer:

* the id layout of ``gpt2`` is the legacy trainers' (``ClmPretrainConfig``), the one of ``ascii95`` the tokenizer module's
  own; the digest moves with every field and with every file hash; the file table is the one tiktoken checks its own
  downloads against;
* ``open_tokenizer`` checks the pinned files BEFORE tiktoken's loader runs (that loader deletes a cache file whose hash
  differs, D-015), then sets ``TIKTOKEN_CACHE_DIR`` for the call and gives it back; a refusal leaves the file on disk;
* the split of the documents is ``sha256(seed:select:index)`` and nothing else: stable when documents are appended, a
  function of the split seed only, never of the stream seed;
* the windows: the concatenation of the documents with an end-of-text id after each, cut into windows of ``seq_len + 1``;
  the labels are the inputs shifted by one; the count is the same for every order of the documents;
* the stream: epoch ``e`` is a hash-keyed permutation of the documents and then of the windows, ``batch_at(step)`` is a
  function of ``(seed, step)``, ``iterate`` and ``with_start`` yield the same batches, every epoch has the same number of steps;
* a selection dataset states its length; the train dataset is infinite and yields the stream's batches;
* the TensorFlow-free half loads neither TensorFlow nor Keras nor tiktoken (fresh interpreter), and the two ``train.common``
  modules import none of them at module scope (the package ``__init__`` of ``train.common`` does, so they are loaded here
  under a stand-in parent).
"""

import ast
import dataclasses
import hashlib
import itertools
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import tiktoken

import train
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
from train.common import text_data as td

SRC = Path(train.__file__).resolve().parents[1]
PINNED = Path(tt.DEFAULT_TOKENIZER_DIR)
NO_PIN = pytest.mark.skipif(
    not all((PINNED / name).is_file() for name, _ in tt.tokenizer_spec("gpt2").files),
    reason=f"the pinned tokenizer directory {PINNED} is not on this machine",
)
TFDS_DIR = Path(os.environ.get("TFDS_DATA_DIR", "")) / "imdb_reviews"
NO_TFDS = pytest.mark.skipif(not TFDS_DIR.is_dir(), reason="imdb_reviews is not under TFDS_DATA_DIR on this machine")

EOT = 99
SEQ, BATCH = 8, 4


def sha(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def key(seed, epoch, tag, i) -> int:
    return int.from_bytes(sha(f"{seed}:{epoch}:{tag}:{i}")[:8], "big")


def corpus(n=40, seed=0) -> List[np.ndarray]:
    """``n`` documents of 1 to 30 ids drawn from 1..20 (never the end-of-text id)."""
    rng = np.random.RandomState(seed)
    return [rng.randint(1, 21, size=rng.randint(1, 31)).astype(np.int32) for _ in range(n)]


def oracle_windows(docs, seed, epoch, seq_len=SEQ, eot=EOT) -> np.ndarray:
    """The epoch's windows by plain python: hash-sorted documents, an eot after each, chunks, hash-sorted windows."""
    order = sorted(range(len(docs)), key=lambda i: (key(seed, epoch, "docs", i), i))
    flat: List[int] = []
    for i in order:
        flat.extend(docs[i].tolist())
        flat.append(eot)
    width = seq_len + 1
    n = len(flat) // width
    chunks = [flat[k * width:(k + 1) * width] for k in range(n)]
    shuffled = sorted(range(n), key=lambda k: (key(seed, epoch, "windows", k), k))
    return np.array([chunks[k] for k in shuffled], np.int32).reshape(n, width)


def stream_of(docs=None, seed=3, batch=BATCH, seq_len=SEQ):
    return td.lm_stream(corpus() if docs is None else docs, seq_len=seq_len, batch_size=batch, seed=seed, eot=EOT)


def child(code: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports the same ``train`` as this process."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=180)


# ---------------------------------------------------------------------
# the tokenizer record
# ---------------------------------------------------------------------


def test_the_gpt2_id_layout_is_the_legacy_trainers():
    from train.common.clm_pretrain import ClmPretrainConfig

    legacy, spec = ClmPretrainConfig(), tt.tokenizer_spec("gpt2")
    assert spec.encoding == legacy.encoding_name == "gpt2"
    assert (spec.cls_id, spec.sep_id, spec.pad_id, spec.mask_id) == (
        legacy.cls_token_id, legacy.sep_token_id, legacy.pad_token_id, legacy.mask_token_id)
    assert spec.vocab_size == legacy.vocab_size
    assert (spec.n_vocab, spec.eot_id) == (50257, 50256)
    assert spec.vocab_size == spec.mask_id + 1, "the four specials sit right after the base vocabulary"


def test_the_ascii95_id_layout_is_the_tokenizers_own():
    from dl_techniques.layers.tokenizers import ascii_char as ac

    spec = tt.tokenizer_spec("ascii95")
    assert spec.encoding == ac.ENCODING_NAME
    assert (spec.pad_id, spec.cls_id, spec.sep_id, spec.mask_id) == (ac.PAD_ID, ac.CLS_ID, ac.SEP_ID, ac.MASK_ID)
    assert spec.eot_id == ac.SEP_ID, "the character tokenizer has no end-of-text id: its separator is the document end"
    assert spec.n_vocab == spec.vocab_size == ac.VOCAB_SIZE
    assert spec.files == ()


def test_an_unknown_tokenizer_is_refused_by_name_with_the_offered_ones():
    with pytest.raises(ConfigError, match=r"'cl100k_base'.*ascii95.*gpt2|ascii95.*gpt2.*'cl100k_base'"):
        tt.tokenizer_spec("cl100k_base")


def test_the_digest_is_a_sha256_that_moves_with_every_field():
    base = tt.tokenizer_spec("gpt2")
    assert re.fullmatch(r"[0-9a-f]{64}", base.digest)
    assert dataclasses.replace(base).digest == base.digest
    name, file_hash = base.files[0]
    changed = {
        "encoding": "gpt2x", "n_vocab": base.n_vocab + 1, "vocab_size": base.vocab_size + 1, "eot_id": base.eot_id - 1,
        "cls_id": base.cls_id + 1, "sep_id": base.sep_id + 1, "pad_id": base.pad_id + 1, "mask_id": base.mask_id + 1,
        # one file hash differs, one file is dropped, one is added: each is another tokenizer
        "files": ((name, "0" * 64),) + base.files[1:],
    }
    assert set(changed) == {field.name for field in dataclasses.fields(base)}, "a new field needs its row here"
    digests = {field: dataclasses.replace(base, **{field: value}).digest for field, value in changed.items()}
    assert all(digest != base.digest for digest in digests.values()), digests
    assert len(set(digests.values())) == len(digests), "two different edits gave one digest"
    assert dataclasses.replace(base, files=base.files[:1]).digest != base.digest
    assert dataclasses.replace(base, files=base.files + (("extra", file_hash),)).digest != base.digest
    assert tt.tokenizer_spec("ascii95").digest != base.digest


def test_the_digest_is_the_sha256_of_the_sorted_json_and_each_tokenizers_is_pinned():
    import json

    for name, golden in (
            ("gpt2", "41b7fab30697ec06a812548e729ff5f3dcf7744e5fcebcfeec39a5c21c2cc7c1"),
            ("ascii95", "253e74fe29fc3d972d0846f24c8d0ed730edd595410ea3b918d9ff47f0d292ba")):
        spec = tt.tokenizer_spec(name)
        fields = {field.name: getattr(spec, field.name) for field in dataclasses.fields(spec)}
        fields["files"] = dict(spec.files)
        assert hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == spec.digest
        # A record, a resume and `init` compare this string: changing an id, the file table or the json form above makes every
        # earlier run's tokenizer another tokenizer. The literals were derived by running the code at Step 3 (D-016).
        assert spec.digest == golden, f"the digest of {name} moved: every run made before this change is refused as another tokenizer"


def test_the_file_table_is_the_one_tiktoken_checks_its_own_downloads_against():
    import inspect

    import tiktoken_ext.openai_public as published

    source = inspect.getsource(published.gpt2)
    url = {role: re.search(rf'{role}_file="([^"]+)"', source).group(1) for role in ("vocab_bpe", "encoder_json")}
    digest = {role: re.search(rf'{role}_hash="([0-9a-f]{{64}})"', source).group(1) for role in url}
    expected = {hashlib.sha1(url[role].encode()).hexdigest(): digest[role] for role in url}
    assert dict(tt.tokenizer_spec("gpt2").files) == expected


def test_a_pinned_directory_is_checked_by_name_and_hash(tmp_path):
    good, other = b"a pinned file", b"another file"
    files = (("name_a", hashlib.sha256(good).hexdigest()), ("name_b", hashlib.sha256(other).hexdigest()))
    (tmp_path / "name_a").write_bytes(good)
    (tmp_path / "name_b").write_bytes(other)
    assert tt.check_files(files, tmp_path) == tmp_path
    where = "'data.tokenizer_dir'"
    with pytest.raises(ConfigError, match=where):
        tt.check_files(files, tmp_path / "nowhere")
    (tmp_path / "name_b").unlink()
    with pytest.raises(ConfigError, match=rf"{where}.*name_b.*does not exist.*{files[1][1]}"):
        tt.check_files(files, tmp_path)
    (tmp_path / "name_b").mkdir()
    with pytest.raises(ConfigError, match=rf"{where}.*name_b.*not a file"):
        tt.check_files(files, tmp_path)
    (tmp_path / "name_b").rmdir()
    (tmp_path / "name_b").write_bytes(other + b"!")
    found = hashlib.sha256(other + b"!").hexdigest()
    with pytest.raises(ConfigError, match=rf"{where}.*name_b.*{found}.*{files[1][1]}"):
        tt.check_files(files, tmp_path)
    assert (tmp_path / "name_b").read_bytes() == other + b"!", "a check never changes a file"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_an_unreadable_file_is_refused_by_name(tmp_path):
    (tmp_path / "name_a").write_bytes(b"x")
    (tmp_path / "name_a").chmod(0)
    try:
        with pytest.raises(ConfigError, match=r"'data.tokenizer_dir'.*name_a.*cannot be read"):
            tt.check_files((("name_a", "0" * 64),), tmp_path)
    finally:
        (tmp_path / "name_a").chmod(0o644)


def test_the_character_tokenizer_has_no_files_and_never_looks_at_a_directory(tmp_path):
    assert tt.check_tokenizer_dir("ascii95", tmp_path / "does_not_exist") is None
    with pytest.raises(ConfigError, match=r"'data.tokenizer_dir'.*is not a directory"):
        tt.check_tokenizer_dir("gpt2", tmp_path / "does_not_exist")
    with pytest.raises(ConfigError, match=rf"'data.tokenizer_dir'.*{tt.tokenizer_spec('gpt2').files[0][0]}.*does not exist"):
        tt.check_tokenizer_dir("gpt2", tmp_path)
    with pytest.raises(ConfigError, match="ascii95"):
        tt.check_tokenizer_dir("nope", tmp_path)


@NO_PIN
def test_the_pinned_directory_on_this_machine_passes_its_check():
    assert tt.check_tokenizer_dir("gpt2") == PINNED
    assert tt.check_tokenizer_dir("gpt2", str(PINNED)) == PINNED


# ---------------------------------------------------------------------
# opening a tokenizer
# ---------------------------------------------------------------------


@pytest.fixture
def fake_tiktoken(monkeypatch, tmp_path):
    """A gpt2 spec over two files of ``tmp_path`` whose hashes are right, and a ``tiktoken.get_encoding`` that records."""
    contents = {"file_a": b"first pinned file", "file_b": b"second pinned file"}
    for name, data in contents.items():
        (tmp_path / name).write_bytes(data)
    files = tuple((name, hashlib.sha256(data).hexdigest()) for name, data in contents.items())
    base = tt.tokenizer_spec("gpt2")
    spec = dataclasses.replace(base, files=files)
    calls = []

    def get_encoding(name):
        calls.append((name, os.environ.get(td.CACHE_ENV)))
        return SimpleNamespace(
            n_vocab=base.n_vocab, eot_token=base.eot_id,
            encode_ordinary=lambda text: [ord(c) for c in text], decode=lambda ids: "".join(chr(i) for i in ids))

    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)
    return SimpleNamespace(spec=spec, calls=calls, directory=tmp_path, base=base)


def test_open_sets_the_cache_directory_for_the_call_and_gives_it_back(fake_tiktoken, monkeypatch):
    monkeypatch.delenv(td.CACHE_ENV, raising=False)
    opened = td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)
    assert fake_tiktoken.calls == [("gpt2", str(fake_tiktoken.directory))], "set before get_encoding runs"
    assert td.CACHE_ENV not in os.environ, "an unset variable is unset again"
    assert opened.encode("ab") == [97, 98]
    monkeypatch.setenv(td.CACHE_ENV, "/somewhere/else")
    td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)
    assert fake_tiktoken.calls[-1] == ("gpt2", str(fake_tiktoken.directory))
    assert os.environ[td.CACHE_ENV] == "/somewhere/else", "a set variable is put back"


def test_open_gives_the_variable_back_when_the_encoding_fails(fake_tiktoken, monkeypatch):
    monkeypatch.setenv(td.CACHE_ENV, "/somewhere/else")
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: (_ for _ in ()).throw(RuntimeError("tiktoken failed")))
    with pytest.raises(RuntimeError, match="tiktoken failed"):
        td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)
    assert os.environ[td.CACHE_ENV] == "/somewhere/else"


def test_open_checks_the_hashes_before_tiktoken_can_delete_a_file(fake_tiktoken):
    # D-015: tiktoken's loader removes a cache file whose sha256 differs and then fetches it. The check comes first, so a
    # file of another hash is refused by name and is still on disk, with its bytes, and tiktoken was never asked.
    victim = fake_tiktoken.directory / "file_b"
    victim.write_bytes(b"not the pinned bytes")
    found = hashlib.sha256(b"not the pinned bytes").hexdigest()
    with pytest.raises(ConfigError, match=rf"file_b.*{found}"):
        td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)
    assert fake_tiktoken.calls == []
    assert victim.read_bytes() == b"not the pinned bytes"


def test_open_refuses_an_encoding_whose_numbers_are_not_the_records(fake_tiktoken, monkeypatch):
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: SimpleNamespace(n_vocab=5, eot_token=4))
    with pytest.raises(ValueError, match=r"n_vocab 5.*50257|50257.*n_vocab 5"):
        td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: SimpleNamespace(n_vocab=50257, eot_token=4))
    with pytest.raises(ValueError, match=r"eot.*4.*50256|50256.*eot.*4"):
        td.open_tokenizer(fake_tiktoken.spec, fake_tiktoken.directory)


def test_the_character_tokenizer_opens_without_a_directory_and_round_trips():
    from dl_techniques.layers.tokenizers.ascii_char import decode_ascii, encode_ascii

    opened = td.open_tokenizer(tt.tokenizer_spec("ascii95"))
    probe = "Hello, World!\nSecond line\u00e9"
    assert opened.encode(probe) == encode_ascii(probe)
    assert opened.decode(opened.encode("abc")) == "abc"
    spec = opened.spec
    assert opened.decode([spec.pad_id, spec.cls_id]) == decode_ascii([spec.pad_id, spec.cls_id], skip_special_tokens=False)
    with pytest.raises(ValueError, match=rf"{spec.vocab_size}"):
        opened.decode([spec.vocab_size])


@NO_PIN
def test_the_real_gpt2_opens_from_the_pinned_directory_and_decodes_every_id():
    opened = td.open_tokenizer(tt.tokenizer_spec("gpt2"))
    spec = opened.spec
    assert opened.encode("hello world") == [31373, 995]
    assert opened.decode([31373, 995]) == "hello world"
    # the literal end-of-text string in a review is ordinary text, never an eot id (encode would raise on it)
    ids = opened.encode("a<|endoftext|>b")
    assert spec.eot_id not in ids and len(ids) == 9
    # ids above tiktoken's vocabulary are the record's specials; tiktoken's own decode raises KeyError on them
    for special in (spec.cls_id, spec.sep_id, spec.pad_id, spec.mask_id):
        assert opened.decode([special]).startswith("<|"), special
    assert opened.decode([31373, spec.pad_id, 995]) == "hello<|pad|> world"
    assert opened.decode([spec.eot_id]) == "<|endoftext|>"
    with pytest.raises(ValueError, match=str(spec.vocab_size)):
        opened.decode([spec.vocab_size])
    with pytest.raises(ValueError, match="-1"):
        opened.decode([-1])


# ---------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------


def test_the_synthetic_bank_is_a_pure_function_of_its_arguments():
    for kind in (td.LM, td.CLASSIFICATION):
        texts, labels = td.synthetic_documents(5, 30, kind)
        again, again_labels = td.synthetic_documents(5, 30, kind)
        assert texts == again and (labels is None) == (again_labels is None)
        assert td.synthetic_documents(6, 30, kind)[0] != texts, "another seed, another bank"
        assert td.synthetic_documents(5, 30, kind, split="test")[0] != texts, "the test documents are other documents"
        assert all(isinstance(text, str) and text.strip() for text in texts)
        assert all(text.isascii() for text in texts)
    with pytest.raises(ValueError, match="kind"):
        td.synthetic_documents(5, 3, "nope")
    with pytest.raises(ValueError, match="split"):
        td.synthetic_documents(5, 3, td.LM, split="validation")


def test_the_synthetic_classification_bank_has_a_fixed_60_40_class_split_and_label_correlated_words():
    for n in (100, 37, 5):
        texts, labels = td.synthetic_documents(11, n, td.CLASSIFICATION)
        assert labels.dtype == np.int64 and labels.shape == (n,)
        assert np.bincount(labels).tolist() == [round(0.6 * n), n - round(0.6 * n)]
    texts, labels = td.synthetic_documents(11, 100, td.CLASSIFICATION)
    assert np.bincount(labels).tolist() == [60, 40]
    # a bag-of-words oracle reads the class off the words: the words correlate with the label
    positive, negative = set(td.POSITIVE_WORDS), set(td.NEGATIVE_WORDS)
    guessed = [int(sum(w in positive for w in t.split()) > sum(w in negative for w in t.split())) for t in texts]
    assert np.mean(np.asarray(guessed) == labels) >= 0.95


def test_the_synthetic_language_bank_has_structure_a_unigram_cannot_use():
    tokenizer = td.open_tokenizer(tt.tokenizer_spec("ascii95"))
    texts, labels = td.synthetic_documents(3, 200, td.LM)
    assert labels is None
    ids = np.concatenate(td.tokenize_documents(texts, tokenizer))

    def entropy(counts):
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log(p)).sum())

    vocab = tokenizer.spec.vocab_size
    unigram = entropy(np.bincount(ids, minlength=vocab).astype(np.float64))
    pairs = np.bincount(ids[:-1] * vocab + ids[1:], minlength=vocab * vocab).reshape(vocab, vocab).astype(np.float64)
    conditional = sum(row.sum() * entropy(row) for row in pairs if row.sum()) / pairs.sum()
    assert conditional < unigram - 0.5, (conditional, unigram)


def test_the_imdb_reader_decodes_in_read_order_and_stops_at_the_cap():
    pulled = []

    def loader(split, data_dir, _limit):
        assert (split, data_dir) == ("test", "/some/dir")
        for i, row in enumerate([(b"first \xc3\xa9", 1), (b"bad \xff byte", 0), ("already text", np.int64(1)), (b"never", 0)]):
            pulled.append(i)
            yield row

    texts, labels = td.imdb_documents("test", "/some/dir", 3, loader=loader)
    assert texts == ["first \u00e9", "bad \ufffd byte", "already text"]
    assert labels.tolist() == [1, 0, 1] and labels.dtype == np.int64
    assert pulled == [0, 1, 2], "a reader that ignores the cap is still read no further than the cap"
    assert td.imdb_documents("train", None, None, loader=lambda s, d, n: iter([(b"x", 0)]))[0] == ["x"]
    with pytest.raises(ValueError, match="unsupervised"):
        td.imdb_documents("unsupervised", loader=loader)
    with pytest.raises(ValueError, match="max_documents"):
        td.imdb_documents("train", None, 0, loader=loader)


@NO_TFDS
def test_the_real_imdb_reader_reads_the_same_documents_every_time_and_a_cap_is_a_prefix():
    texts, labels = td.imdb_documents("test", None, 12)
    again, again_labels = td.imdb_documents("test", None, 12)
    assert len(texts) == 12 and texts == again and labels.tolist() == again_labels.tolist()
    assert all(isinstance(text, str) and text for text in texts) and set(labels.tolist()) <= {0, 1}
    longer, longer_labels = td.imdb_documents("test", None, 30)
    assert longer[:12] == texts and longer_labels[:12].tolist() == labels.tolist(), "a cap keeps the first documents of the read order"


def test_tokenize_documents_gives_int32_arrays_and_refuses_ids_outside_the_record():
    opened = td.open_tokenizer(tt.tokenizer_spec("ascii95"))
    docs = td.tokenize_documents(["ab", "", "c d"], opened)
    assert [d.tolist() for d in docs] == [opened.encode("ab"), [], opened.encode("c d")]
    assert all(d.dtype == np.int32 for d in docs)
    spec = opened.spec

    class Bad:
        def __init__(self, ids):
            self.spec, self.ids = spec, ids

        def encode(self, text):
            return self.ids

    with pytest.raises(ValueError, match=rf"document 1.*{spec.vocab_size}"):
        td.tokenize_documents(["x", "y"], SimpleNamespace(spec=spec, encode=lambda t: [5] if t == "x" else [spec.vocab_size]))
    with pytest.raises(ValueError, match=rf"document 0.*end-of-text id {spec.eot_id}"):
        td.tokenize_documents(["x"], Bad([4, spec.eot_id]))
    with pytest.raises(ValueError, match="document 0.*-3"):
        td.tokenize_documents(["x"], Bad([-3]))


# ---------------------------------------------------------------------
# the split of the documents
# ---------------------------------------------------------------------


def unit(seed, index) -> float:
    return int.from_bytes(sha(f"{seed}:select:{index}")[:8], "big") / 2 ** 64


def test_the_split_is_the_hash_of_seed_tag_and_index_and_nothing_else():
    for seed, share in ((7, 0.2), (0, 0.5), (123456, 0.05)):
        train_index, select_index = td.split_documents(300, seed, share)
        expected = [i for i in range(300) if unit(seed, i) < share]
        assert select_index.tolist() == expected
        assert train_index.tolist() == [i for i in range(300) if i not in set(expected)]
        assert train_index.dtype == select_index.dtype == np.int64
    assert td.split_documents(300, 8, 0.2)[1].tolist() != td.split_documents(300, 7, 0.2)[1].tolist()


def test_the_split_is_stable_when_documents_are_appended_and_near_its_share():
    short = td.split_documents(500, 9, 0.1)
    for extra in (1, 17, 2000):
        longer = td.split_documents(500 + extra, 9, 0.1)
        assert longer[1][longer[1] < 500].tolist() == short[1].tolist()
        assert longer[0][longer[0] < 500].tolist() == short[0].tolist()
    big = td.split_documents(20000, 9, 0.1)
    assert abs(len(big[1]) / 20000 - 0.1) < 0.01
    both = np.concatenate(big)
    assert sorted(both.tolist()) == list(range(20000)), "every document is in exactly one part"


@pytest.mark.parametrize("share", [0, 1, -0.1, 1.5, float("nan")])
def test_a_share_outside_zero_and_one_is_refused(share):
    with pytest.raises(ValueError, match="validation_split"):
        td.split_documents(10, 1, share)


def split_of(seed=7, stream_seed=None, share=0.25, seq_len=SEQ, batch=BATCH, train=None, test=None):
    spec = tt.tokenizer_spec("ascii95")
    train = corpus(60, seed=1) if train is None else train
    test = corpus(30, seed=2) if test is None else test
    docs_spec = dataclasses.replace(spec, eot_id=EOT, vocab_size=EOT + 1, n_vocab=EOT + 1)
    return td.lm_split(train, test, spec=docs_spec, seq_len=seq_len, batch_size=batch, seed=seed,
                       stream_seed=stream_seed, validation_split=share)


def test_the_split_follows_the_seed_and_never_the_stream_seed():
    a, b, c = split_of(stream_seed=1), split_of(stream_seed=2), split_of(stream_seed=None)
    for other in (b, c):
        np.testing.assert_array_equal(a.train_index, other.train_index)
        np.testing.assert_array_equal(a.select_index, other.select_index)
        for fixed, same in zip(a.select + a.test, other.select + other.test):
            np.testing.assert_array_equal(fixed, same)
    assert a.select_index.tolist() == td.split_documents(60, 7, 0.25)[1].tolist()
    assert not np.array_equal(a.stream.batch_at(0)[0], b.stream.batch_at(0)[0]), "the stream seed moves the train order"
    np.testing.assert_array_equal(c.stream.batch_at(3)[0], split_of(stream_seed=7).stream.batch_at(3)[0])
    assert not np.array_equal(split_of(seed=8).select_index, a.select_index), "the split seed moves the split"


def test_the_parts_are_the_documents_the_split_names():
    train, test = corpus(60, seed=1), corpus(30, seed=2)
    part = split_of(train=train, test=test)
    chosen = [train[i] for i in part.select_index]
    expected_x, expected_y = td.lm_fixed_windows(chosen, SEQ, EOT)
    np.testing.assert_array_equal(part.select[0], expected_x)
    np.testing.assert_array_equal(part.select[1], expected_y)
    np.testing.assert_array_equal(part.test[0], td.lm_fixed_windows(test, SEQ, EOT)[0])
    assert part.n_test_docs == 30
    assert part.stream.n_rows == td.lm_window_count([train[i] for i in part.train_index], SEQ)
    # no selection document is trained on
    selected = set(part.select_index.tolist())
    assert selected.isdisjoint(part.train_index.tolist())


def test_a_split_that_leaves_no_window_is_refused_by_name():
    with pytest.raises(ValueError, match=r"selection documents.*window"):
        split_of(share=0.0001)
    with pytest.raises(ValueError, match=r"test documents.*window"):
        split_of(test=[np.array([1, 2], np.int32)])
    with pytest.raises(ValueError, match=r"train.*batch"):
        split_of(share=0.9, batch=64)


# ---------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------


def flat_of(docs, order, eot=EOT):
    out: List[int] = []
    for i in order:
        out.extend(docs[i].tolist())
        out.append(eot)
    return out


def test_the_windows_are_the_documents_with_an_eot_after_each_cut_to_seq_len_plus_one():
    docs = corpus(12, seed=4)
    order = np.random.RandomState(0).permutation(12)
    windows = td.lm_epoch_windows(docs, order, SEQ, EOT)
    flat = flat_of(docs, order)
    n = len(flat) // (SEQ + 1)
    assert windows.shape == (n, SEQ + 1) and windows.dtype == np.int32
    np.testing.assert_array_equal(windows.reshape(-1), flat[:n * (SEQ + 1)])
    # the remainder is the tail of the stream, fewer than one window of tokens
    assert 0 <= len(flat) - n * (SEQ + 1) < SEQ + 1


def test_an_eot_follows_every_document_and_no_window_crosses_one_without_it():
    docs = [np.array(d, np.int32) for d in ([1, 2, 3], [4, 5], [6, 7, 8, 1, 2], [3, 4, 5, 6])]
    windows = td.lm_epoch_windows(docs, np.array([0, 1, 2, 3]), 3, EOT)
    flat = windows.reshape(-1).tolist()
    # 18 tokens (14 of the documents, 4 eots) make four windows of four; the last two tokens are the dropped tail
    assert flat == [1, 2, 3, EOT, 4, 5, EOT, 6, 7, 8, 1, 2, EOT, 3, 4, 5]
    # where a document ends, the next token in the stream is its eot: read the boundaries off the lengths
    end = -1
    for d in docs:
        end += len(d) + 1
        if end < len(flat):
            assert flat[end] == EOT and (np.asarray(flat[end - len(d):end]) != EOT).all(), (end, flat)
    assert flat.count(EOT) == 3, "the fourth eot is in the dropped tail"
    # an empty document is just its eot
    gap = [np.array([1, 2], np.int32), np.zeros(0, np.int32), np.array([3, 4, 5], np.int32)]
    assert td.lm_epoch_windows(gap, np.arange(3), 3, EOT).tolist() == [[1, 2, EOT, EOT], [3, 4, 5, EOT]]


def test_the_window_count_is_the_same_for_every_order_of_the_documents():
    docs = corpus(25, seed=6)
    counts = {len(td.lm_epoch_windows(docs, np.random.RandomState(s).permutation(25), SEQ, EOT)) for s in range(8)}
    assert counts == {td.lm_window_count(docs, SEQ)}
    total = sum(len(d) for d in docs) + len(docs)
    assert td.lm_window_count(docs, SEQ) == total // (SEQ + 1)
    a = td.lm_epoch_windows(docs, np.arange(25), SEQ, EOT)
    b = td.lm_epoch_windows(docs, np.arange(25)[::-1].copy(), SEQ, EOT)
    assert not np.array_equal(a, b), "the order of the documents is the order of the windows"
    assert td.lm_epoch_windows(docs[:1], np.array([0]), 400, EOT).shape == (0, 401), "too short for one window"


def test_the_fixed_windows_are_the_documents_in_index_order_with_labels_shifted_by_one():
    docs = corpus(10, seed=2)
    x, y = td.lm_fixed_windows(docs, SEQ, EOT)
    windows = td.lm_epoch_windows(docs, np.arange(10), SEQ, EOT)
    np.testing.assert_array_equal(x, windows[:, :-1])
    np.testing.assert_array_equal(y, windows[:, 1:])
    assert x.flags["C_CONTIGUOUS"] and y.flags["C_CONTIGUOUS"] and x.dtype == y.dtype == np.int32


# ---------------------------------------------------------------------
# the stream
# ---------------------------------------------------------------------


def test_every_epoch_follows_the_hash_rule_of_the_design():
    docs = corpus()
    stream = stream_of(docs, seed=3)
    for epoch in range(3):
        x, y = stream.epoch_rows(epoch)
        expected = oracle_windows(docs, 3, epoch)
        np.testing.assert_array_equal(x, expected[:, :-1])
        np.testing.assert_array_equal(y, expected[:, 1:])
    assert stream.n_rows == len(oracle_windows(docs, 3, 0))


def test_epochs_differ_and_the_same_epoch_is_bit_equal():
    stream = stream_of()
    first, second = stream.epoch_rows(0)[0].copy(), stream.epoch_rows(1)[0].copy()
    assert not np.array_equal(first, second), "an epoch that ignores its number is the same epoch forever"
    np.testing.assert_array_equal(stream_of().epoch_rows(0)[0], first)
    np.testing.assert_array_equal(stream.epoch_rows(0)[0], first)
    assert not np.array_equal(stream_of(seed=4).epoch_rows(0)[0], first), "another seed, another order"
    spe = stream.steps_per_epoch
    assert not np.array_equal(stream.batch_at(0)[0], stream.batch_at(spe)[0])
    np.testing.assert_array_equal(stream.batch_at(spe + 1)[0], stream_of().batch_at(spe + 1)[0])


def test_the_labels_are_the_inputs_shifted_by_one():
    stream = stream_of()
    for step in (0, 1, stream.steps_per_epoch - 1, stream.steps_per_epoch, 3 * stream.steps_per_epoch + 2):
        x, y = stream.batch_at(step)
        assert x.shape == y.shape == (BATCH, SEQ) and x.dtype == y.dtype == np.int32
        np.testing.assert_array_equal(y[:, :-1], x[:, 1:])
    # the last label of a window is the token after its last input: read it off the oracle
    expected = oracle_windows(corpus(), 3, 0)
    np.testing.assert_array_equal(stream.batch_at(0)[1][:, -1], expected[:BATCH, -1])


def test_batch_at_iterate_and_with_start_yield_the_same_batches():
    stream = stream_of()
    run = list(itertools.islice(stream.iterate(0), 2 * stream.steps_per_epoch + 3))
    for step in (0, 1, 7, stream.steps_per_epoch - 1, stream.steps_per_epoch, 2 * stream.steps_per_epoch + 2):
        for got, want in zip(stream.batch_at(step), run[step]):
            np.testing.assert_array_equal(got, want)
    for start in (0, 5, stream.steps_per_epoch, stream.steps_per_epoch + 4):
        resumed = stream.with_start(start)
        assert resumed.start_step == start and stream.start_step == 0
        for offset, (x, y) in enumerate(itertools.islice(resumed.iterate(), 6)):
            np.testing.assert_array_equal(x, run[start + offset][0])
            np.testing.assert_array_equal(y, run[start + offset][1])
        np.testing.assert_array_equal(resumed.batch_at(start)[0], stream.batch_at(start)[0])
    assert resumed.n_rows == stream.n_rows and resumed.steps_per_epoch == stream.steps_per_epoch


def test_every_epoch_has_the_same_steps_and_each_window_is_used_once_in_an_epoch():
    docs = corpus()
    stream = stream_of(docs)
    spe = stream.steps_per_epoch
    assert spe == stream.n_rows // BATCH
    for epoch in (0, 1, 2):
        used = [tuple(np.append(x[i], y[i, -1])) for step in range(epoch * spe, (epoch + 1) * spe)
                for x, y in [stream.batch_at(step)] for i in range(BATCH)]
        assert len(used) == len(set(used)) == spe * BATCH
        assert set(used) <= {tuple(w) for w in oracle_windows(docs, 3, epoch)}


def test_a_returned_batch_is_a_copy_and_a_stream_survives_threads():
    stream = stream_of()
    x, _ = stream.batch_at(2)
    saved = x.copy()
    x[:] = -1
    np.testing.assert_array_equal(stream.batch_at(2)[0], saved)
    spe = stream.steps_per_epoch
    steps = [0, spe, 2 * spe, 1, spe + 1, 2 * spe + 1] * 20
    want = [stream_of().batch_at(s)[0] for s in steps]
    got, errors = [None] * len(steps), []

    def work(lo):
        try:
            for i in range(lo, len(steps), 4):
                got[i] = stream.batch_at(steps[i])[0]
        except Exception as exc:  # collected so the test names it; a thread cannot raise into pytest
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(lo,)) for lo in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert all(np.array_equal(a, b) for a, b in zip(got, want))


def test_a_stream_with_fewer_rows_than_a_batch_or_a_bad_step_is_refused():
    stream = stream_of()
    too_big = stream.n_rows + 1
    with pytest.raises(ValueError, match=rf"fewer than one batch of {too_big}"):
        stream_of(batch=too_big)
    assert stream_of(batch=stream.n_rows).steps_per_epoch == 1
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="step"):
            stream.batch_at(bad)
    with pytest.raises(ValueError, match="start_step"):
        stream.with_start(-2)
    with pytest.raises(ValueError, match="batch_size"):
        stream_of(batch=0)
    with pytest.raises(ValueError, match="seq_len"):
        stream_of(seq_len=0)
    np.testing.assert_array_equal(stream.batch_at(np.int64(2))[0], stream.batch_at(2)[0])


# ---------------------------------------------------------------------
# the rows of a classifier
# ---------------------------------------------------------------------


def test_a_row_is_cls_then_the_head_of_the_document_then_pad():
    spec = tt.tokenizer_spec("gpt2")
    docs = [np.array(d, np.int32) for d in ([5, 6, 7], [], [1, 2, 3, 4, 5, 6, 7, 8, 9], [9] * 5)]
    ids, truncated = td.encode_rows(docs, spec, 6)
    assert ids.shape == (4, 6) and ids.dtype == np.int32
    assert ids.tolist() == [
        [spec.cls_id, 5, 6, 7, spec.pad_id, spec.pad_id],
        [spec.cls_id] + [spec.pad_id] * 5,
        [spec.cls_id, 1, 2, 3, 4, 5],
        [spec.cls_id, 9, 9, 9, 9, 9],
    ]
    assert truncated == 1, "only the nine-token document is cut; a document of exactly seq_len - 1 tokens is not"
    assert (ids != spec.pad_id).any(axis=1).all(), "a row is never all pad: a BERT mask never sums to zero"
    assert (ids[:, 0] == spec.cls_id).all()
    with pytest.raises(ValueError, match="seq_len"):
        td.encode_rows(docs, spec, 1)


# ---------------------------------------------------------------------
# the datasets
# ---------------------------------------------------------------------


def test_a_selection_dataset_states_its_length_and_keeps_every_row_in_order():
    import tensorflow as tf

    x = np.arange(23 * 5, dtype=np.int32).reshape(23, 5)
    y = x + 1
    for batch in (4, 23, 100):
        ds = td.select_dataset((x, y), batch)
        assert int(ds.cardinality()) == -(-23 // batch) != int(tf.data.UNKNOWN_CARDINALITY)
        rows = list(ds.as_numpy_iterator())
        assert len(rows) == int(ds.cardinality())
        np.testing.assert_array_equal(np.concatenate([r[0] for r in rows]), x)
        np.testing.assert_array_equal(np.concatenate([r[1] for r in rows]), y)
        assert rows[0][0].dtype == np.int32
    with pytest.raises(ValueError, match="no rows"):
        td.select_dataset((x[:0], y[:0]), 4)


def test_the_train_dataset_is_infinite_and_yields_the_streams_batches():
    import tensorflow as tf

    stream = stream_of()
    ds = td.train_dataset(stream)
    assert int(ds.cardinality()) == int(tf.data.INFINITE_CARDINALITY)
    assert tuple(ds.element_spec[0].shape) == (BATCH, SEQ) == tuple(ds.element_spec[1].shape)
    assert ds.element_spec[0].dtype == ds.element_spec[1].dtype == tf.int32
    count = 2 * stream.steps_per_epoch + 3
    for step, (x, y) in enumerate(ds.take(count).as_numpy_iterator()):
        np.testing.assert_array_equal(x, stream.batch_at(step)[0])
        np.testing.assert_array_equal(y, stream.batch_at(step)[1])
    start = stream.steps_per_epoch + 2
    resumed = td.train_dataset(stream.with_start(start))
    for offset, (x, _) in enumerate(resumed.take(5).as_numpy_iterator()):
        np.testing.assert_array_equal(x, stream.batch_at(start + offset)[0])
    # two iterators over the same dataset see the same batches: nothing in the pipeline is stateful
    first, second = next(iter(ds.as_numpy_iterator())), next(iter(ds.as_numpy_iterator()))
    np.testing.assert_array_equal(first[0], second[0])


# ---------------------------------------------------------------------
# what the TensorFlow-free half loads
# ---------------------------------------------------------------------

HEAVY = ("tensorflow", "keras", "tf_keras", "tiktoken", "tensorflow_datasets", "numpy", "pandas", "scipy", "PIL")


def test_the_tokenizer_record_loads_nothing_heavy_even_when_it_refuses(tmp_path):
    code = f"""
import sys
from train.api import text_tokenizer as tt
from train.api.errors import ConfigError
spec = tt.tokenizer_spec('gpt2')
assert len(spec.digest) == 64 and tt.tokenizer_spec('ascii95').files == ()
try:
    tt.check_tokenizer_dir('gpt2', {str(tmp_path)!r})
except ConfigError as exc:
    refused = 'data.tokenizer_dir' in str(exc)
assert refused
print('HEAVY', sorted(m for m in {HEAVY!r} if m in sys.modules))
print('TRAIN', sorted(m for m in sys.modules if m.startswith('train.')))
"""
    result = child(code, tmp_path)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "HEAVY []" in result.stdout, result.stdout
    assert "train.common" not in result.stdout, result.stdout


def module_level_imports(path: Path) -> List[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(("." * node.level) + (node.module or ""))
    return found


def test_the_tokenizer_record_and_the_two_common_modules_import_nothing_heavy_at_module_scope():
    forbidden = ("tensorflow", "keras", "tf_keras", "tiktoken", "tensorflow_datasets", "pandas", "scipy")
    for relative in ("api/text_tokenizer.py", "common/text_data.py", "common/text_run.py"):
        imports = module_level_imports(SRC / "train" / relative)
        assert imports, relative
        bad = [name for name in imports if name.split(".")[0] in forbidden]
        assert bad == [], (relative, bad)
    stdlib_only = module_level_imports(SRC / "train" / "api" / "text_tokenizer.py")
    assert not [n for n in stdlib_only if n.split(".")[0] in ("numpy", "train", "dl_techniques")], stdlib_only


def test_the_two_common_modules_load_nothing_heavy_when_their_package_init_is_out_of_the_way(tmp_path):
    # `import train.common.<anything>` runs train/common/__init__.py, which loads gpu, nlp and with them TensorFlow, Keras,
    # tiktoken and tensorflow_datasets. A stand-in parent package shows what the two modules themselves import.
    code = f"""
import importlib, sys, types
from pathlib import Path
import train
parent = types.ModuleType('train.common')
parent.__path__ = [str(Path(train.__file__).parent / 'common')]
sys.modules['train.common'] = parent
importlib.import_module('train.common.text_data')
importlib.import_module('train.common.text_run')
print('HEAVY', sorted(m for m in {HEAVY[:5]!r} if m in sys.modules))
"""
    result = child(code, tmp_path)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "HEAVY []" in result.stdout, result.stdout
    # the control: the real package does load them, which is why the check above needs the stand-in
    real = child("import sys, train.common.seed\nprint('LOADED', 'tensorflow' in sys.modules)", tmp_path)
    assert "LOADED True" in real.stdout, real.stdout + real.stderr[-500:]
