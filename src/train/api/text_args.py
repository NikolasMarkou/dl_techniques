"""The text-task arguments of ``python -m train``: the data group the two text tasks share, the transform and metric groups.

The ``language_model_pretraining`` task trains a causal language model on packed windows of tokenized documents: the documents
of a corpus are tokenized once, split into train and selection documents by a hash of ``(seed, document index)``, each epoch
puts the train documents in a seeded order with an end-of-text id after each and cuts the result into windows of ``seq_len + 1``
tokens (``train.common.text_data``). The ``text_classification`` task reads the same documents as rows of ``seq_len`` ids
(``[CLS]``, the head of the document, pad) with their labels. ``TextData`` names the corpus, the tokenizer, the window and how the
documents are split and capped; ``TextTransformArgs``, ``LanguageMetricArgs`` and ``ClassifierMetricArgs`` are the groups of the
task configs that differ from the shared ones: a token id is the data, so there is no transform, and the metric is a loss or a
majority-class bar, so classification's ``MetricArgs`` (a ``top_k``, a ``success_threshold`` that is a multiple of chance) cannot
serve. Every option that cannot apply is REFUSED, never ignored.

Interface contract
------------------
``TASK`` / ``GPT2_LM``
    ``"language_model_pretraining"``, the task these groups belong to, and ``"gpt2_lm"``, the family that trains it
    (:mod:`train.api.text_models` registers its row under that name).

``CLASSIFICATION_TASK`` / ``BERT`` / ``GPT2_CLASSIFIER``
    ``"text_classification"``, the second text task, and the two families that train it, ``"bert"`` and ``"gpt2"`` (the family and the
    tokenizer ``gpt2`` share a name: ``text_tokenizer.GPT2`` is the tokenizer). :mod:`train.api.text_models` builds their rows.

``DATASETS`` / ``SYNTHETIC_TRAIN_DOCUMENTS`` / ``SYNTHETIC_TEST_DOCUMENTS``
    The corpora the text tasks read: the offline ``synthetic`` bank and the local TFDS ``imdb_reviews``; and the default size of
    the bank (``train.common.text_data`` imports both numbers, so the help below and the loader cannot disagree).

``TextData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; ``format`` is always ``"builtin"`` (the shared run
    sequence reads it). ``tokenizer`` is ``ascii95`` (offline, 101 ids) or ``gpt2`` (tiktoken, 50,261 ids read from the pinned
    directory); ``tokenizer_dir`` overrides that directory and is refused for ``ascii95``, which reads no file. ``seq_len`` is the
    model's input length; ``validation_split`` the share of the train DOCUMENTS held out to select the best epoch; ``max_samples``
    the documents read (see the field). ``problems(prefix)`` names every option out of range.

``check_text_sources(data)``
    :raises ConfigError: for ``imdb_reviews`` when no ``imdb_reviews/plain_text/<version>/dataset_info.json`` is under
        ``$TFDS_DATA_DIR`` (or ``~/tensorflow_datasets``), naming the key and the path looked at. ``Path`` tests only: no TFDS
        import, no tensor, nothing opened or written, so it runs before the run directory is claimed. ``synthetic`` returns.

``text_facts(data, num_classes=None)``
    :return: ``DataFacts(dataset, num_classes, None)``: no image size, and a class count for the classifier only (``None``: the
        language model has none). Runs :func:`check_text_sources`.

``CLASS_NAMES``
    ``("neg", "pos")``, the classes of both corpora (TFDS ``imdb_reviews``' own label names; the synthetic bank's class 0 draws its
    cue words from the negative list).

``load_text_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one (a relative ``tokenizer_dir`` is
        anchored to the file).
    :return: the validated config.
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping, or any invalid key (every problem at
        once; a file's lines name the file).

``read_text_data_file(path)``
    :return: the mapping in a YAML file, a relative ``tokenizer_dir`` made absolute against the file.

``TextTransformArgs``
    No field: the ids are the data. The group exists because the shared sequence reads ``config.transform_args`` (D-027).

``LanguageMetricArgs`` / ``ClassifierMetricArgs``
    ``watch_metric`` (the one value accepted: ``val_loss``, ``val_accuracy``; auto is the recipe's) and ``watch_mode`` (``min``,
    ``max``; likewise): the success bar is a statement about the MINIMUM ``val_loss`` or the MAXIMUM ``val_accuracy`` of the selection
    window, so the checkpoint that is saved and reloaded must be that epoch, and ``problems`` refuses any other metric or direction
    by key. The classifier's group has no ``success_threshold``: its bar is the majority-class share of the selection split.

``tokenizer_identity(data)`` / ``documents_overlap(source, config, data, task)``
    The ``init_identity`` and ``init_overlap`` of both text tasks (a language run seeds a classifier, so the two must agree on what a
    source is): ``{"vocab_size", "tokenizer"}`` (the family's two ``fixed`` keys, from the record: no file read), and the ``split``
    of the ``init`` block counted by DOCUMENT (:func:`documents_read`).

Import rule
-----------
Module scope imports only the standard library and ``train.api``; nothing here imports numpy, ``train.common`` or keras, so
``python -m train language_model_pretraining help`` stays free of TensorFlow.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from .args import AutoContext, _is_auto, _watch_values, _with_values
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, _tfds_data_dir, load_data_group, read_data_file
from .errors import ConfigError
from .text_tokenizer import ASCII95, GPT2, TOKENIZERS, tokenizer_spec

# ---------------------------------------------------------------------

TASK = "language_model_pretraining"
GPT2_LM = "gpt2_lm"
CLASSIFICATION_TASK = "text_classification"
BERT = "bert"
GPT2_CLASSIFIER = "gpt2"

SYNTHETIC, IMDB = "synthetic", "imdb_reviews"
DATASETS = (SYNTHETIC, IMDB)

#: The documents of the synthetic bank when ``data.max_samples`` is null: train, and a quarter of that for test.
SYNTHETIC_TRAIN_DOCUMENTS = 600
SYNTHETIC_TEST_DOCUMENTS = SYNTHETIC_TRAIN_DOCUMENTS // 4

#: The one metric and direction each bar can describe; the family recipes carry the same metrics (a test holds them equal).
WATCH_METRIC = "val_loss"
WATCH_MODE = "min"
CLASSIFIER_METRIC = "val_accuracy"
CLASSIFIER_MODE = "max"

#: The classes of both corpora, in label order (TFDS ``imdb_reviews``' own names); the synthetic bank's class 0 is the negative one.
CLASS_NAMES = ("neg", "pos")

_Dataset = Literal[DATASETS]  # type: ignore[valid-type]
_Tokenizer = Literal[TOKENIZERS]  # type: ignore[valid-type]


@dataclasses.dataclass(frozen=True)
class TextData(StrictConfig):
    """The corpus of a text run: the documents, the tokenizer, the window and the split; see the module contract."""

    dataset: _Dataset = dataclasses.field(
        metadata={
            "help": "synthetic (an offline bank of seeded documents: a word chain, so a model that reads its context beats a "
            "unigram) or imdb_reviews (the local TFDS reviews, 25,000 train and 25,000 test documents); the train documents are "
            "split by a hash of (seed, document index) into the train documents and the documents that select the best epoch, "
            "and the test documents are scored once after fit"
        }
    )
    format: Literal["builtin"] = dataclasses.field(
        default="builtin", metadata={"help": "where the data comes from; the only value"}
    )
    tokenizer: _Tokenizer = dataclasses.field(
        default="gpt2",
        metadata={
            "help": f"ascii95 (offline characters, {tokenizer_spec(ASCII95).vocab_size} ids) or gpt2 (tiktoken BPE read from the "
            f"pinned directory, {tokenizer_spec(GPT2).vocab_size:,} ids: its {tokenizer_spec(GPT2).n_vocab:,} tokens and cls, sep, "
            "pad and mask); the model's vocabulary is the tokenizer's, never the model's own default"
        },
    )
    tokenizer_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "gpt2 only: the directory holding the two BPE cache files; null is the pinned copy on the data disk. A "
            "directory whose files hash to other values is refused by name before the run is claimed; refused for ascii95"
        },
    )
    seq_len: int = dataclasses.field(
        default=256,
        metadata={
            "help": "tokens the model reads and predicts (a window holds seq_len + 1 tokens: the inputs and the targets shifted "
            "by one); at most the model's max_seq_len, the size of its position table (refused by name before the run is claimed)"
        },
    )
    validation_split: float = dataclasses.field(
        default=0.1,
        metadata={"help": "share of the train DOCUMENTS held out to select the best epoch, strictly between 0 and 1; the split "
                          "is a hash of (seed, document index), so it never follows the stream's order"},
    )
    # DECISION plan-2026-11-24T092348-7806a290/D-024
    # The cap is `max_samples`, not the plan's `max_documents`: the shared sequence caps a smoke run through `data.max_samples`
    # (trainer.py, `debug_args.smoke`), so a data group without that name makes `smoke` a TypeError. The unit is the DOCUMENT
    # (a sample of a text task is a document, a window is a row of the stream). Do NOT add a second cap beside it.
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "documents read: the first N of the train split and of the test split in read order (imdb_reviews); for the "
            f"synthetic bank the number of train documents, with a quarter of that many test documents; null is every review, "
            f"or {SYNTHETIC_TRAIN_DOCUMENTS} train and {SYNTHETIC_TEST_DOCUMENTS} test synthetic documents; debug_args.smoke "
            "reads at most 512"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range and every option that cannot apply to the tokenizer, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(self, name)!r}")

        if self.seq_len < 1:
            bad("seq_len", "expected an int of at least 1")
        if not 0 < self.validation_split < 1:
            bad("validation_split", "expected a fraction strictly between 0 and 1")
        if self.max_samples is not None and self.max_samples < 1:
            bad("max_samples", "expected a positive int or null")
        if self.tokenizer_dir is not None and self.tokenizer == ASCII95:
            bad("tokenizer_dir", f"the {ASCII95} tokenizer reads no file, so it has no directory; remove it or choose the gpt2 tokenizer")
        if self.tokenizer_dir == "":
            bad("tokenizer_dir", "expected a directory or null (null is the pinned copy)")
        return lines


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------


def check_text_sources(data: TextData) -> None:
    """Refuse an ``imdb_reviews`` that is not under the TFDS directory; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-034
    # Path tests only, and the ones TFDS's own loader makes first: a version directory that holds `dataset_info.json`. Do NOT
    # import TFDS here (the first dataset op creates TensorFlow's devices before `setup_gpu`), do NOT name a version (it is only
    # known by importing TFDS), and do NOT test more than the loader needs (a false refusal of the real directory is the worst
    # failure here). The directory is the one TFDS reads (`data_args._tfds_data_dir`, pinned by its own test).
    if data.dataset != IMDB:
        return
    root = _tfds_data_dir()
    origin = f"TFDS directory {str(root)!r} from {'$TFDS_DATA_DIR' if 'TFDS_DATA_DIR' in os.environ else 'the default ~/tensorflow_datasets'}"
    here = root / IMDB / "plain_text"
    try:
        found = here.is_dir() and any((version / "dataset_info.json").is_file() for version in here.iterdir())
    except OSError as exc:
        raise ConfigError(
            f"Invalid value for 'data.dataset': {IMDB} reads TFDS records, and {exc.filename or str(here)!r} cannot be read "
            f"({exc.strerror or exc}) ({origin}); fix the permission, or set TFDS_DATA_DIR to a directory you can read"
        ) from exc
    if not found:
        raise ConfigError(
            f"Invalid value for 'data.dataset': {IMDB} reads the TFDS records under {str(here)!r}, and none are there "
            f"({origin}); nothing is downloaded. Set TFDS_DATA_DIR to the directory that holds {IMDB}/"
        )


def text_facts(data: TextData, num_classes: Optional[int] = None) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    check_text_sources(data)
    return DataFacts(dataset=data.dataset, num_classes=num_classes, image_size=None)


# ---------------------------------------------------------------------
# The config from a mapping or a file
# ---------------------------------------------------------------------


def read_text_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``, a relative ``tokenizer_dir`` anchored to the file."""
    return read_data_file(path, ("tokenizer_dir",))


def load_text_data_config(data: Any) -> TextData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(TextData, data, read_text_data_file)


# ---------------------------------------------------------------------
# The other groups of the task config
# ---------------------------------------------------------------------


# DECISION plan-2026-11-24T092348-7806a290/D-027
# A group with NO field, not an image group and not a made-up knob: `TaskTrainer` requires `config.transform_args` (CONFIG_FIELDS)
# and a token id has nothing to resize or augment. The group is where a later text transform (a truncation rule) would go. Do NOT
# reuse `TransformArgs` (its `image_size` would be asked of a corpus) and do NOT drop the field (the trainer refuses the config).
@dataclasses.dataclass(frozen=True)
class TextTransformArgs(StrictConfig):
    """The input transform of the text tasks: none, the token ids are the data; read by the language task."""


@dataclasses.dataclass(frozen=True)
class _SelectionMetricArgs(StrictConfig):
    """The metric that selects the checkpoint, of a task whose bar is about ONE metric: the base of the two text metric groups.

    :cvar ONLY: ``(metric, mode, extreme)``: the one metric and direction a subclass accepts and the word its refusal uses for the
        bar's statement ("the minimum val_loss", "the maximum val_accuracy").
    """

    ONLY: ClassVar[Tuple[str, str, str]]

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` or a ``watch_mode`` other than the one the bar describes."""
        metric, mode, extreme = self.ONLY
        lines: List[str] = []
        for name, only in (("watch_metric", metric), ("watch_mode", mode)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success "
                    f"bar is a statement about the {extreme} {metric} of the selection window, so the saved best checkpoint "
                    f"must be that epoch ({metric!r} with mode {mode!r}); another metric or the other direction "
                    "would save weights the verdict does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "_SelectionMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))


@dataclasses.dataclass(frozen=True)
class LanguageMetricArgs(_SelectionMetricArgs):
    """The metric that selects the checkpoint; read by the language-model task."""

    ONLY: ClassVar[Tuple[str, str, str]] = (WATCH_METRIC, WATCH_MODE, "minimum")

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: val_loss on the fixed selection windows, the only value accepted (the "
            "success bar is about the minimum val_loss, so the saved checkpoint must be that epoch); auto is val_loss"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "min, the only value accepted (val_loss is a loss: the best epoch is its minimum); auto is min"},
    )


@dataclasses.dataclass(frozen=True)
class ClassifierMetricArgs(_SelectionMetricArgs):
    """The metric that selects the checkpoint; read by the text-classification task (its bar is the majority-class share)."""

    ONLY: ClassVar[Tuple[str, str, str]] = (CLASSIFIER_METRIC, CLASSIFIER_MODE, "maximum")

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: val_accuracy on the selection rows, the only value accepted (the success "
            "bar is about the maximum val_accuracy, so the saved checkpoint must be that epoch); auto is val_accuracy"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "max, the only value accepted (val_accuracy: the best epoch is its maximum); auto is max"},
    )


# ---------------------------------------------------------------------
# What a source run is to a text task (``init``)
# ---------------------------------------------------------------------


def tokenizer_identity(data: TextData) -> Dict[str, Any]:
    """Return ``{"vocab_size", "tokenizer"}`` this run's record will hold: the two ``fixed`` keys of every text family."""
    spec = tokenizer_spec(data.tokenizer)
    return {"vocab_size": spec.vocab_size, "tokenizer": spec.digest}


def documents_read(dataset: str, cap: Optional[int], ours: Optional[int], read: int) -> Optional[int]:
    """Return how many train documents a run capped at ``cap`` read of ``dataset``, or ``None`` when this run's cap hides it.

    Interface contract: ``ours`` is the cap of THIS run (``data.max_samples``) and ``read`` the number of train documents it read. The
    synthetic bank is as long as it is asked to be (the task's default when ``cap`` is null); a cap above a corpus reads all of it,
    which this run's own read count gives when its cap did not bind; when it did, the corpus holds at least ``ours`` documents and a
    source capped at most ``ours`` read exactly its cap. Pure; never raises.
    """
    if dataset == SYNTHETIC:
        return SYNTHETIC_TRAIN_DOCUMENTS if cap is None else cap
    if ours is None or read < ours:
        return read if cap is None else min(cap, read)
    return cap if cap is not None and cap <= ours else None


def documents_overlap(source: Any, config: Any, data: Any, task: str) -> Dict[str, Any]:
    """The ``init`` block's ``split`` of a text task: the source's train DOCUMENTS against this run's selection documents, counted.

    Interface contract: ``source`` is the :class:`~train.api.init_from.Source` (its ``config`` is the source's ``config.json``, its
    ``record`` its ``task.json``), ``config`` this run's resolved config, ``data`` what ``build_data`` returned (``data.split`` holds
    the train and selection document indices and the test document count) and ``task`` this run's task. The test documents are the
    corpus's own, which no run trains on. :return: ``different_data`` for another dataset and for the two synthetic banks of the two
    tasks (the language bank and the labelled one are other texts under the same document numbers), ``same_split`` when dataset,
    seed, validation split and cap equal the source's, ``unknown`` (with the reason) when this run's cap hides how many documents a
    capped source read, else ``overlap`` with the count.
    """
    import numpy as np

    from train.common.text_data import split_documents

    from . import init_from

    theirs, ours = source.config.get("data") or {}, config.data
    if theirs.get("dataset") != ours.dataset:
        return init_from.split_block("different_data", f"the source trained on {theirs.get('dataset')}, this run reads "
                                                       f"{ours.dataset}", {"data.dataset": theirs.get("dataset")}, None, (0, 0))
    if ours.dataset == SYNTHETIC and (getattr(source, "record", None) or {}).get("task", task) != task:
        return init_from.split_block("different_data", "the source read the synthetic language bank and this run reads the synthetic "
                                                       "labelled bank (or the other way round): other texts under the same document "
                                                       "numbers", {"data.dataset": SYNTHETIC}, None, (0, 0))
    keys = {"data.dataset": ours.dataset, "seed": source.config.get("seed"),
            "data.validation_split": theirs.get("validation_split"), "data.max_samples": theirs.get("max_samples")}
    if keys == {"data.dataset": ours.dataset, "seed": config.seed, "data.validation_split": ours.validation_split,
                "data.max_samples": ours.max_samples}:
        return init_from.same_split("documents", keys)
    # DECISION plan-2026-11-24T092348-7806a290/D-042
    # The source's train documents are rebuilt with the SOURCE's seed, split and cap (its config.json), and this run's selection
    # documents are the ones it holds. Do NOT rebuild both cuts from this run's keys (they would coincide and the count read 0),
    # and do NOT guess how many documents a capped source read: `documents_read` says, or the split is `unknown`.
    select = data.split.select_index
    n_source = documents_read(ours.dataset, keys["data.max_samples"], ours.max_samples, len(data.split.train_index) + len(select))
    if n_source is None:
        return init_from.split_block("unknown", f"this run caps data.max_samples at {ours.max_samples}, so how many "
                                                f"{ours.dataset} documents the source (data.max_samples "
                                                f"{keys['data.max_samples']}) read is not known: its split of them cannot be rebuilt")
    trained = split_documents(n_source, keys["seed"], keys["data.validation_split"])[0]
    return init_from.counted_split("documents", keys, np.intersect1d(trained, select).size, len(select), 0, data.split.n_test_docs)
