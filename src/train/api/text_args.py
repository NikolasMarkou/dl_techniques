"""The language-task arguments of ``python -m train``: the data group the text tasks share, the transform and metric groups.

The ``language_model_pretraining`` task trains a causal language model on packed windows of tokenized documents: the documents
of a corpus are tokenized once, split into train and selection documents by a hash of ``(seed, document index)``, each epoch
puts the train documents in a seeded order with an end-of-text id after each and cuts the result into windows of ``seq_len + 1``
tokens (``train.common.text_data``). ``TextData`` names the corpus, the tokenizer, the window and how the documents are split and
capped; ``TextTransformArgs`` and ``LanguageMetricArgs`` are the groups of the task config
(``train.api.tasks.language_model_pretraining``) that differ from the shared ones: a token id is the data, so there is no
transform, and the metric is a loss, so classification's ``MetricArgs`` (a ``top_k``, a ``success_threshold`` that is a multiple
of chance) cannot serve. Every option that cannot apply is REFUSED, never ignored.

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

``text_facts(data)``
    :return: ``DataFacts(dataset, None, None)``: no class count and no image size. Runs :func:`check_text_sources`.

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

``LanguageMetricArgs``
    ``watch_metric`` (:data:`WATCH_METRIC`, the only value accepted; auto is the recipe's) and ``watch_mode`` (:data:`WATCH_MODE`,
    likewise): the success bar is a statement about the MINIMUM ``val_loss`` of the selection window, so the checkpoint that is
    saved and reloaded must be that epoch, and ``problems`` refuses any other metric or direction by key.

Import rule
-----------
Module scope imports only the standard library and ``train.api``; nothing here imports numpy, ``train.common`` or keras, so
``python -m train language_model_pretraining help`` stays free of TensorFlow.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

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

#: The one metric and direction the bar can describe; the family recipe carries the same metric (a test holds them equal).
WATCH_METRIC = "val_loss"
WATCH_MODE = "min"

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


def text_facts(data: TextData) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    check_text_sources(data)
    return DataFacts(dataset=data.dataset, num_classes=None, image_size=None)


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
class LanguageMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the language-model task."""

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

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``val_loss`` and a ``watch_mode`` other than ``min``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success "
                    f"bar is a statement about the minimum val_loss of the selection window, so the saved best checkpoint "
                    f"must be that epoch ({WATCH_METRIC!r} with mode {WATCH_MODE!r}); another metric or the other direction "
                    "would save weights the verdict does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "LanguageMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))
