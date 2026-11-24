"""Predict and export for the two text tasks: :class:`LanguageModelPredictor` and :class:`TextClassifierPredictor`, the ``.txt`` and
``.jsonl`` reader and the JSON writers of ``predict``, and the export rows of ``language_model_pretraining`` and ``text_classification``.

``train.api.predictor.load_model`` returns a :class:`LanguageModelPredictor` when the run's ``task.json`` says
``language_model_pretraining`` and a :class:`TextClassifierPredictor` when it says ``text_classification`` (schema in
:mod:`train.api.predictor`). A language model takes ``seq_len`` token ids and returns the
logits of the id after each: ``predict`` answers what the model thinks of a document (how surprised it is, and what it expects
next), the quantities the run's ``val_loss`` is made of, and does not generate text. A text classifier takes a row of ``seq_len`` ids
(``[CLS]``, the head of the document, pad: the row the run trained on) and returns one logit per class: ``predict`` answers the class
names best first with their softmax probabilities, as the image classifier does.

Scoring rule (D-035). A document is its ids followed by the end-of-text id, as the training stream has it. The ids are cut into
consecutive windows of ``seq_len + 1`` (the last one shorter); a window predicts every id after its first from the ones before it
inside the window, exactly as the training windows do, so the first id of a window is context only and no id is scored twice. A
short last window is right-padded with the pad id and scored on its real positions only: the model is causal, so what follows
a position does not change its logits and the padding cannot leak into the score. ``mean_nll`` is the mean over every predicted id
of a document, in nats, and the ``val_loss`` of a document that is exactly one window.

Interface contract
------------------
``LanguageModelPredictor(model, record)``
    :param model: the loaded Keras model; its input is ``(batch, seq_len)`` float32 ids and its output ``(batch, seq_len, vocab_size)``.
    :param record: a language-model record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task. The tokenizer is not opened here (``export`` needs none): the first ``predict``
        opens it, from the record, and refuses a digest this code does not pin for the record's encoding and files that do not hash to it.

``LanguageModelPredictor.predict(x)``
    :param x: one text (``str``) or a list or tuple of texts.
    :return: ``{"documents": [...]}``, ALWAYS a list, one entry per text in the order given: ``{"n_tokens", "mean_nll", "perplexity",
        "next_tokens"}``. ``n_tokens`` is the number of ids the text became; ``mean_nll`` the mean negative log-likelihood, in nats per
        id, of the ids the model predicts (the rule above); ``perplexity`` its exponential; ``next_tokens`` the five most likely ids
        after the text's last one, given its last ``seq_len`` ids, best first (ties by id): ``{"token", "text", "probability"}``, the
        text of an id decoded by the run's tokenizer (the four ids above tiktoken's vocabulary by name). Rows run :data:`CHUNK` at a time.
    :raises ConfigError: for an empty list, an item that is not a text, a text with no id, a score that is not finite (a
        mean_nll whose exponential overflows included) and a tokenizer that cannot be opened.

``TextClassifierPredictor(model, record)``
    :param model: the loaded Keras model; its input is ``(batch, seq_len)`` float32 ids and its output ``(batch, classes)`` logits.
    :param record: a text-classification record that passed :func:`train.api.predictor.check_record`.
    :raises ConfigError: for a record of another task; the tokenizer is opened on the first ``predict``, as the language model's is.

``TextClassifierPredictor.predict(x)``
    :param x: one text (``str``) or a list or tuple of texts.
    :return: ``{"labels", "scores"}``: for one text both are flat lists, the record's class names best first (ties keep class order)
        and their softmax probabilities as plain floats; for a list, one such list per text. A text is made a row as the run made it
        (:func:`train.common.text_data.encode_rows`): ``[CLS]``, its first ``seq_len - 1`` ids (a longer text loses its TAIL) and pad ids.
        Rows run :data:`ROWS` at a time and a row's answer does not depend on the others.
    :raises ConfigError: as the language model's, for an empty list, an item that is not a text, a text with no id and a tokenizer that
        cannot be opened.

``list_data(data)`` / ``read_documents(path)``
    The command's input: ``list_data`` returns ``[data]`` for a ``.txt`` or a ``.jsonl`` file and refuses a directory and any other
    suffix by name; ``read_documents`` returns the documents, one per line (``.txt``: the line; ``.jsonl``: the ``"text"`` of a JSON
    object), a final newline making no extra one. :raises ConfigError: for a file that cannot be read as UTF-8, one with no line,
    an empty line, a line of a ``.jsonl`` that is not JSON or has no ``"text"`` string; each names its line.

``write_labels(predictor, documents, out)``
    Writes ``[{"index", "labels", "scores"}, ...]``, one entry per document and the shape of ``predict``'s image list, as one JSON file
    into the new path ``out`` (nothing is overwritten, and the file is written LAST, in one piece). :raises ConfigError: for any error of
    ``predict``.

``write_predictions(predictor, file, documents, out)``
    Writes ``{"task", "model", "data", "seq_len", "tokenizer", "scoring", "documents"}`` as one JSON file into the new path ``out``
    (nothing is overwritten, and the file is written LAST, in one piece), ``documents`` holding ``{"index", ...}`` per document.
    :raises ConfigError: for any error of ``predict``.

``served(model, record)`` / ``sample_ids(record, shape)`` / ``check_export(record, format, shape)`` / ``export_note(record)`` / ``classifier_export_note(record)``
    The export row of these tasks (:class:`train.api.predictor.ExportRule`; a classifier's differs from the language model's in its note): ``served`` is the function every file runs, the model with
    ``training=False`` (D-032); ``sample_ids`` the seeded batch of valid ids the tflite check is fed (D-008); ``check_export`` raises a
    named :class:`ConfigError`, before anything is written, for a length other than the record's and a vocabulary a float32 cannot
    number; ``export_note`` is the sentence logged beside a file. Every other cell converts and matches the model on valid ids
    (measured on a tiny model and on a gpt2-vocabulary build, ``test_language_model_pretraining_predictor_and_export.py``).

Import rule
-----------
Module scope imports the standard library and ``train.api`` only; ``numpy`` and ``train.common`` are imported inside functions, so
``python -m train help`` allocates nothing.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import ConfigError
from .exporting import check_free
from .predictor import LANGUAGE_MODEL, TEXT_CLASSIFICATION, check_record
from .run_dir import TASK_FILE
from .text_tokenizer import tokenizer_spec

#: Rows (windows) run through the model at a time: a row is ``seq_len x vocab_size`` floats of logits (51 MB at 256 and 50,261).
CHUNK = 4

#: Rows (documents) a classifier runs through the model at a time: a row is ``seq_len`` ids and its answer is a few logits.
ROWS = 32

#: How many next tokens ``predict`` lists.
TOP_K = 5

#: What ``data=`` may name for a language model.
DATA_SUFFIXES = (".txt", ".jsonl")

_DATA_HELP = 'give a .txt file (one document per line) or a .jsonl file (one {"text": ...} object per line)'

#: The ids a float32 holds exactly are the whole numbers below this; a vocabulary from it up cannot be fed as float ids (D-008).
FLOAT_ID_LIMIT = 2 ** 24

#: What the answer of ``predict`` means, in the file the command writes.
SCORING_NOTE = (
    "a document is its ids and then the end-of-text id, cut into consecutive windows of seq_len + 1; a window predicts each id after "
    "its first from the ones before it, a short last window is padded and scored on its real positions only; mean_nll is the mean "
    "negative log-likelihood in nats per predicted id, perplexity its exponential, n_tokens the ids of the text; next_tokens are the "
    "five most likely ids after the last one, given its last seq_len ids"
)


# ---------------------------------------------------------------------
# The command line's input
# ---------------------------------------------------------------------


def list_data(data: Path) -> List[Path]:
    """Return ``[data]`` for a ``.txt`` or ``.jsonl`` file; refuse a directory and any other suffix by name."""
    if data.is_dir():
        raise ConfigError(f"Invalid value for 'data': {data} is a directory, not a file: {_DATA_HELP}")
    if data.suffix.lower() not in DATA_SUFFIXES:
        raise ConfigError(f"Invalid value for 'data': {data} has the suffix {data.suffix!r}: {_DATA_HELP}")
    return [data]


def _jsonl_text(path: Path, number: int, line: str) -> str:
    try:
        item = json.loads(line)
    except ValueError as exc:
        raise ConfigError(f"Invalid value for 'data': line {number} of {path} is not JSON ({exc})") from exc
    if not (isinstance(item, dict) and isinstance(item.get("text"), str)):
        raise ConfigError(f'Invalid value for \'data\': line {number} of {path} has no "text" that is a string: {_DATA_HELP}')
    return item["text"]


def read_documents(path: Path) -> List[str]:
    """Return the documents of ``path``, one per line; see the module contract."""
    try:
        lines = path.read_text(encoding="utf-8-sig").split("\n")  # a byte order mark (an editor's) is not part of the text
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Invalid value for 'data': {path} cannot be read as UTF-8 text ({exc})") from exc
    if lines and lines[-1] == "":
        lines.pop()  # the newline that ends the file starts no document
    lines = [line.rstrip("\r") for line in lines]
    if not lines:
        raise ConfigError(f"Invalid value for 'data': {path} holds no document (it is empty): {_DATA_HELP}")
    documents = lines if path.suffix.lower() == ".txt" else [_jsonl_text(path, n, line) for n, line in enumerate(lines, 1)]
    for number, text in enumerate(documents, 1):
        if not text:
            raise ConfigError(f"Invalid value for 'data': line {number} of {path} is empty; a document needs at least one token")
    return documents


# ---------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------


class _TextPredictor:
    """What the two text predictors share: the checked record, the tokenizer reopened from it and the texts of a call; see the module contract.

    :param model: the Keras model.
    :param record: a record of this class's :attr:`task` that passed :func:`train.api.predictor.check_record`.
    :cvar task: the task word of the records this class serves.
    """

    task = ""

    def __init__(self, model: Any, record: Mapping[str, Any]) -> None:
        check_record(dict(record), "<given>")
        if record["task"] != self.task:
            raise ConfigError(f"Invalid {TASK_FILE} <given>: the task is {record['task']!r} but this predictor serves {self.task!r} only")
        self.model = model
        self.record = dict(record)
        self._tokenizer: Any = None

    def tokenizer(self) -> Any:
        """Return the run's tokenizer, opened on first use from the record; see the module contract."""
        if self._tokenizer is None:
            from train.common import text_data

            record = self.record
            spec = tokenizer_spec(record["encoding"])
            if spec.digest != record["tokenizer"]:
                raise ConfigError(
                    f"Invalid {TASK_FILE} <given>: it names the tokenizer digest {record['tokenizer']} but this code pins "
                    f"{spec.digest} for {spec.encoding}; the ids of that run mean something else here, so it cannot be predicted "
                    "with (its model can still be exported)"
                )
            try:
                self._tokenizer = text_data.open_tokenizer(spec)
            except ConfigError as exc:
                raise ConfigError(f"Invalid value for 'model': the tokenizer of this run ({spec.encoding}) cannot be opened: {exc}") from exc
        return self._tokenizer

    @staticmethod
    def _texts_of(x: Any) -> List[str]:
        if isinstance(x, str):
            return [x]
        if not isinstance(x, (list, tuple)):
            raise ConfigError(f"Invalid document: expected a text or a list of texts, got {type(x).__name__}")
        if not x:
            raise ConfigError("Invalid document: no document to score (the list is empty)")
        for number, text in enumerate(x):
            if not isinstance(text, str):
                raise ConfigError(f"Invalid document {number}: document {number} is a {type(text).__name__}, not a text")
        return list(x)

    def _encode(self, x: Any) -> Tuple[Any, List[List[int]]]:
        """Return ``(tokenizer, one id list per text of x)``; a text with no id is refused by its number."""
        texts = self._texts_of(x)
        tokenizer = self.tokenizer()
        documents = [tokenizer.encode(text) for text in texts]
        for number, ids in enumerate(documents):
            if not ids:
                raise ConfigError(
                    f"Invalid document {number}: the text has no token (it is empty, or {self.record['encoding']} reads none of "
                    "its characters), so there is nothing to score"
                )
        return tokenizer, documents


class LanguageModelPredictor(_TextPredictor):
    """A loaded language model and the tokenizer its run trained with; see the module contract."""

    task = LANGUAGE_MODEL

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"documents": [...]}`` for ``x``; see the module contract."""
        import numpy as np
        from scipy.special import log_softmax

        tokenizer, documents = self._encode(x)
        length, eot, pad = self.record["seq_len"], self.record["eot_id"], self.record["pad_id"]
        # DECISION plan-2026-11-24T092348-7806a290/D-035
        # The rule is the TRAINING windows': consecutive windows of ``length + 1`` ids with the end-of-text id after the document, the
        # first id of each context only, a short last window padded and scored on its real positions. So a document that is exactly one
        # window scores the ``val_loss`` of that window. Do NOT overlap the windows (a sliding score is another number than the
        # metric), do NOT drop the end-of-text target and do NOT score the padding (its targets are not the text).
        # One row per window (inputs, targets), and per document one with no targets: its last ``length`` ids, whose last position
        # gives the next tokens. Rows of every document run together, ``CHUNK`` at a time.
        rows: List[Tuple[int, List[int], List[int]]] = []
        for number, ids in enumerate(documents):
            stream = ids + [eot]
            for start in range(0, len(stream), length + 1):
                window = stream[start:start + length + 1]
                if len(window) > 1:
                    rows.append((number, window[:-1], window[1:]))
            rows.append((number, ids[-length:], []))
        nll = [0.0] * len(documents)
        scored = [0] * len(documents)
        following: List[Any] = [None] * len(documents)
        for begin in range(0, len(rows), CHUNK):
            part = rows[begin:begin + CHUNK]
            batch = np.full((len(part), length), pad, np.float32)
            for row, (_, inputs, _) in enumerate(part):
                batch[row, :len(inputs)] = inputs
            logits = np.asarray(self.model(batch, training=False))
            for row, (number, inputs, targets) in enumerate(part):
                if targets:
                    logp = log_softmax(logits[row, :len(inputs)].astype(np.float64), axis=-1)
                    nll[number] -= float(logp[np.arange(len(targets)), targets].sum())
                    scored[number] += len(targets)
                else:
                    logp = log_softmax(logits[row, len(inputs) - 1].astype(np.float64), axis=-1)
                    following[number] = [
                        {"token": int(token), "text": tokenizer.decode([int(token)]), "probability": float(np.exp(logp[token]))}
                        for token in np.argsort(-logp, kind="stable")[:TOP_K]
                    ]
        answers: List[Dict[str, Any]] = []
        for number, ids in enumerate(documents):
            mean = nll[number] / scored[number]
            perplexity = math.exp(mean) if mean < 700 else math.inf  # exp overflows from 709.78; a NaN is not below 700 either
            if not math.isfinite(perplexity):
                raise ConfigError(f"The model's score of document {number} is not a finite number (mean_nll {mean}): nothing is answered (JSON has no NaN)")
            answers.append({"n_tokens": len(ids), "mean_nll": mean, "perplexity": perplexity, "next_tokens": following[number]})
        return {"documents": answers}


class TextClassifierPredictor(_TextPredictor):
    """A loaded text classifier and the tokenizer its run trained with; see the module contract."""

    task = TEXT_CLASSIFICATION

    def predict(self, x: Any) -> Dict[str, Any]:
        """Return ``{"labels", "scores"}`` for ``x``; see the module contract."""
        import numpy as np
        from scipy.special import softmax

        from train.common import text_data

        tokenizer, documents = self._encode(x)
        # DECISION plan-2026-11-24T092348-7806a290/D-064
        # The row the run trained on, made by the function that made it (``[CLS]``, the head, pad): a text longer than the row loses
        # its TAIL. Do NOT cut or pad here by hand (a second rule for the row would drift from the training rows), do NOT keep the end of
        # a long text (the training rows keep its head), and do NOT answer a list of texts with the language model's per-document
        # dicts: a classifier answers the image predictor's ``{labels, scores}``.
        rows, _ = text_data.encode_rows(documents, tokenizer.spec, self.record["seq_len"])
        logits = np.concatenate([
            np.asarray(self.model(rows[start:start + ROWS].astype(np.float32), training=False), np.float64)
            for start in range(0, len(rows), ROWS)])
        probabilities = softmax(logits, axis=-1)
        order = np.argsort(-probabilities, axis=-1, kind="stable")
        names = self.record["class_names"]
        labels = [[names[index] for index in row] for row in order]
        scores = [[float(chances[index]) for index in row] for chances, row in zip(probabilities, order)]
        return {"labels": labels[0], "scores": scores[0]} if isinstance(x, str) else {"labels": labels, "scores": scores}


def _write_json(out: Path, payload: Any) -> None:
    """Write ``payload`` as one JSON file into the new path ``out`` (``"x"``: nothing is overwritten)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_labels(predictor: TextClassifierPredictor, documents: Sequence[str], out: Path) -> None:
    """Write the class names and scores of ``documents`` as one JSON list into the new path ``out``."""
    check_free(out, "out", "predict")
    answer = predictor.predict(list(documents))
    _write_json(out, [{"index": number, "labels": labels, "scores": scores}
                      for number, (labels, scores) in enumerate(zip(answer["labels"], answer["scores"]))])


def write_predictions(predictor: LanguageModelPredictor, file: Path, documents: Sequence[str], out: Path) -> None:
    """Write the answers for ``documents`` (read from ``file``) as one JSON file into the new path ``out``."""
    check_free(out, "out", "predict")
    answers = predictor.predict(list(documents))["documents"]
    record = predictor.record
    document = {
        "task": LANGUAGE_MODEL,
        "model": record["model"],
        "data": str(file),
        "seq_len": record["seq_len"],
        "tokenizer": record["tokenizer"],
        "scoring": SCORING_NOTE,
        "documents": [{"index": number, **answer} for number, answer in enumerate(answers)],
    }
    _write_json(out, document)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def served(model: Any, record: Mapping[str, Any]) -> Any:
    """Return the function every file of a language model runs: float ids in, the logits out, with ``training=False``."""
    # DECISION plan-2026-11-24T092348-7806a290/D-032
    # The rule HAS a served function although the model already takes float ids: ``export._convert_tflite`` writes a built-in
    # file with NO numeric check when ``served`` is None (measured: every cell of the language model converts with built-in
    # operators, so the check would never run). Do NOT return None here, and do NOT wrap it in a layer (a plain function is all the
    # writers call): a served task's built-in file is checked against this function on the rule's valid ids (``sample_ids``).
    return lambda batch: model(batch, training=False)


def sample_ids(record: Mapping[str, Any], shape: Tuple[int, ...]) -> Any:
    """Return the float32 batch ``shape`` of valid ids the tflite check is fed: seeded draws from the vocabulary, a pad tail in the last row."""
    import numpy as np

    ids = np.random.default_rng(0).integers(0, record["vocab_size"], size=shape).astype(np.float32)
    ids[-1, shape[1] - shape[1] // 4:] = record["pad_id"]
    return ids


def check_export(record: Mapping[str, Any], format: str, shape: Sequence[Optional[int]]) -> None:
    """Raise :class:`ConfigError` for another length or a vocabulary float ids cannot number; see the module contract."""
    wanted = (record["seq_len"],)
    if tuple(shape) != wanted:
        raise ConfigError(
            f"Invalid value for 'input_shape': {tuple(shape)} is not the trained length {wanted} (seq_len) of this {record['task']} "
            "model; its position table and graph are built for that length, so it is exported at that length and a file of another "
            "length would be a new model"
        )
    if record["vocab_size"] >= FLOAT_ID_LIMIT:
        raise ConfigError(
            f"Invalid value for 'format': a vocab_size {record['vocab_size']} cannot be exported: the file takes its ids as float32, "
            f"which holds whole numbers exactly only below {FLOAT_ID_LIMIT} (2^24)"
        )


def export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported language model."""
    length, vocab = record["seq_len"], record["vocab_size"]
    return (
        f"the exported model takes float32 token ids (batch, {length}), the trained length only, each id a whole number written as a "
        f"float (3.0, not 3) from the {record['encoding']} tokenizer (vocab_size {vocab}, pad id {record['pad_id']}: right-pad a shorter "
        "text with it and read only the real positions, which the padding after them cannot change); it returns the raw next-token "
        f"logits (batch, {length}, {vocab}), the row at position i scoring the id after the i-th"
    )


def classifier_export_note(record: Mapping[str, Any]) -> str:
    """Return the sentence logged beside an exported text classifier."""
    length, names = record["seq_len"], record["class_names"]
    return (
        f"the exported model takes float32 token ids (batch, {length}), the trained length only, each id a whole number written as a "
        f"float (3.0, not 3) from the {record['encoding']} tokenizer (vocab_size {record['vocab_size']}); a row is the [CLS] id "
        f"{record['cls_id']}, the first {length - 1} ids of the text and the pad id {record['pad_id']} up to the length (a longer text "
        "is cut at its end, a shorter one right-padded; the model derives the padding from the pad id); it returns the raw logits "
        f"(batch, {len(names)}), one per class in the order {', '.join(names)}"
    )
