"""Shared scaffold for the Qwen3 embedding-tower and reranker training recipes.

**Scope: `Qwen3EmbeddingModel` and `Qwen3RerankerModel` ONLY.**
`src/train/qwen/common.py`'s own D-001 anchor forbids that module from
importing `qwen3_next.py` or `qwen3_embeddings.py`
(`plans/plan-2026-09-12T173329-e20362c4/decisions.md` D-001), so this is a
new sibling package rather than a colocated addition
(`plans/plan-2026-09-13T073704-245ab5d5/decisions.md` D-003, applied here by
the same reasoning).

Neither model is causal-LM shaped: `Qwen3EmbeddingModel.call()` returns a
pooled `(batch, embedding_dim)` vector and `Qwen3RerankerModel.call()`
returns a scalar-per-example `(batch,)` relevance probability
(`findings/qwen3next-embedding-reranker-requirements.md` Thread 2, confirmed
again by direct read here). `CausalLanguageModel` has nothing to shift or
slice against either output, so both trainers use stock
`compile()`/`fit()` directly against the bare model classes -- no custom
`train_step` anywhere in this module.

Data provenance
----------------

**The training data is synthetic and generated in this file**, cloned in
SHAPE (not by import) from `src/train/language/colbert/common.py`'s
query/positive/negatives group generator -- disjoint per-topic vocabularies,
in-process generation. There is no MS MARCO and no other IR dataset anywhere
in this repository (confirmed by the same grep the ColBERT trainers cite:
`msmarco`/`ms_marco` return zero hits and `dl_techniques/datasets/` carries
no retrieval loader). A checkpoint produced by either trainer here is a
WIRING result, never a retrieval-quality claim, for the identical reasons
`train.language.colbert.common`'s module docstring gives.

InfoNCE decision (`plans/plan-2026-09-13T073704-245ab5d5/decisions.md`
D-013)
------------------------------------------------------------------------

`dl_techniques.losses.infonce_loss.SymmetricInfoNCELoss` was read in full
before wiring (the plan's single biggest scope-uncertainty item). Its
ACTUAL contract is: two views of one batch, positional positives on the
diagonal, in-batch negatives (batch size IS the negative count) -- it takes
a stacked `(batch, 2, dim)` tensor, a 2-tuple/2-list, or a
`{"view_a": ..., "view_b": ...}` dict, and needs no explicit negative
examples at all. This is not the "reshape into ColBERT's flattened nway
group" case the plan text anticipated as the preferred path -- that reshape
targets `ColBERTPairwiseSoftmaxLoss`'s LISTWISE contract (explicit
candidates per query), which is a different shape than InfoNCE's own
two-view contract. Once the actual signature was read, the correct
reshape for InfoNCE turned out to be simpler than either path the plan
text named: encode the query and its positive document through the SAME
tower (`Qwen3EmbeddingModel`) as `view_a`/`view_b`, stack them, and let
InfoNCE's own in-batch-negative mechanism supply the negatives from the
OTHER groups already present in the batch -- exactly the shape
`src/train/embeddings_experimental/train_embeddings.py`'s `SimCSEModel`
already uses for its own two-dropout-view case (found by grep: it is the
only OTHER caller of this loss in the repository). The synthetic generator
still produces explicit negatives (cloning ColBERT's group shape as this
plan's Step 7 bullet directs), but they are consumed only by the
synthetic-separation CHECK (:func:`embed_texts` used by that test), never
by the training step itself -- the training step needs no explicit negative
row at all under InfoNCE's real contract.

Reranker's yes/no ids
----------------------

`Qwen3RerankerLayer`'s defaults (`yes_token_id=9891`, `no_token_id=2201`)
are NOT arbitrary Qwen-tokenizer ids that happen to collide with this
tokenizer by luck of documentation -- they are MEASURED here to be exactly
`tiktoken.get_encoding("cl100k_base").encode("yes")` and `...encode("no")`
(each a single token). This module re-derives them from the live encoding
at run time via :func:`_resolve_yes_no_token_ids` rather than trusting that
coincidence, so a caller who ever changes `--encoding-name` still gets
correct ids instead of silently training against the wrong two vocabulary
slots (the HARD constraint from
`findings/qwen3next-embedding-reranker-requirements.md`).

Right-padding
-------------

Both models' last-token pooling (`sequence_lengths = sum(attention_mask) -
1`) assumes right-padding. `dl_techniques.utils.tokenizer.TiktokenPreprocessor`
already right-pads (padding tokens are appended after the real tokens and
the attention mask), confirmed by direct read of
`TiktokenPreprocessor._preprocess_single` -- no additional padding-direction
handling is needed here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import argparse

import keras
import numpy as np
import tensorflow as tf
import tiktoken

from dl_techniques.losses.infonce_loss import SymmetricInfoNCELoss
from dl_techniques.models.language.qwen.qwen3_embeddings import (
    Qwen3EmbeddingModel,
    Qwen3RerankerModel,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tokenizer import TiktokenPreprocessor
from train.common.nlp import create_tokenizer
from train.common.optimizer import build_optimizer

__all__ = [
    "CLI_TO_CONFIG",
    "SMOKE_PRESET",
    "TrainingConfig",
    "Qwen3EmbeddingTowerPair",
    "add_common_arguments",
    "build_embedding_datasets",
    "build_reranker_datasets",
    "build_embedding_model",
    "build_reranker_model",
    "build_optimizer",
    "embed_texts",
    "embedding_steps_per_epoch",
    "reranker_steps_per_epoch",
]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration shared by the embedding-tower and reranker recipes.

    Plain class with class-level defaults, matching
    `src/train/language/colbert/common.py`'s `TrainingConfig` (itself
    matching `src/train/bert/pretrain.py`, Pattern 3). The keyword
    `__init__` lets `train.common.args.config_values_from_args` hand over a
    values dict and rejects unknown field names.

    :param overrides: Field name to value. Every name must already exist as
        a class attribute.
    :raises TypeError: If any override names a field this class does not
        define.
    """

    # -- Tokenizer / vocabulary --------------------------------------
    encoding_name: str = "cl100k_base"

    # -- Model geometry (shared trunk) --------------------------------
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 128
    dropout_rate: float = 0.0
    ffn_type: str = "swiglu"
    normalization_type: str = "rms_norm"
    attention_type: str = "multi_head"

    # -- Embedding tower only ------------------------------------------
    normalize_embeddings: bool = True
    truncate_dim: int = 0  # 0 disables truncation (no MRL); a positive value truncates.
    infonce_temperature: float = 0.05

    # -- Reranker only --------------------------------------------------
    reranker_maxlen: int = 96

    # -- Synthetic data --------------------------------------------------
    query_maxlen: int = 16
    doc_maxlen: int = 48
    #: Candidates per group, positive at index 0 (cloned from ColBERT's
    #: convention -- see the module docstring for why only the positive is
    #: actually consumed by the embedding tower's own training step).
    nway: int = 4
    num_train_groups: int = 64
    num_val_groups: int = 16
    query_words: int = 4
    doc_words: int = 16

    # -- Training ----------------------------------------------------
    #: Groups per batch for both recipes (the embedding tower trains 1 pair
    #: per group; the reranker trains `nway` rows per group).
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    warmup_epochs: int = 0
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    patience: int = 5
    seed: int = 42

    # -- Output ------------------------------------------------------
    output_root: str = "results"
    results_dir_prefix: str = "qwen3_embeddings"

    # -- Smoke -------------------------------------------------------
    smoke: bool = False

    def __init__(self, **overrides: Any) -> None:
        unknown = sorted(k for k in overrides if not hasattr(type(self), k))
        if unknown:
            raise TypeError(
                f"TrainingConfig got unknown field(s) {unknown}; known fields are "
                f"{sorted(self.field_names())}"
            )
        for key, value in overrides.items():
            setattr(self, key, value)
        self._validate()

    def _validate(self) -> None:
        """Validate the resolved configuration.

        :raises ValueError: On a non-positive count, an out-of-range ratio,
            or an `nway` too small/large for the synthetic topic pool.
        """
        if self.nway < 2:
            raise ValueError(
                f"nway must be >= 2 (a one-candidate group has no negative "
                f"to separate from), got {self.nway}"
            )
        if self.nway > len(_TOPIC_WORDS):
            raise ValueError(
                f"nway={self.nway} exceeds the {len(_TOPIC_WORDS)} disjoint "
                f"synthetic topics available."
            )
        for name in (
            "hidden_size", "num_layers", "num_heads", "intermediate_size",
            "query_maxlen", "doc_maxlen", "reranker_maxlen",
            "num_train_groups", "num_val_groups", "query_words", "doc_words",
            "batch_size", "epochs", "patience",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.infonce_temperature <= 0.0:
            raise ValueError(
                f"infonce_temperature must be positive, got {self.infonce_temperature}"
            )
        if self.truncate_dim < 0:
            raise ValueError(
                f"truncate_dim must be >= 0 (0 disables truncation), got "
                f"{self.truncate_dim}"
            )
        if min(self.num_train_groups, self.num_val_groups) < self.batch_size:
            raise ValueError(
                f"num_train_groups/num_val_groups ({self.num_train_groups}/"
                f"{self.num_val_groups}) must each be >= batch_size "
                f"({self.batch_size}), or a split would be EMPTY under "
                f"drop_remainder=True."
            )

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        """Every configurable field name, in declaration order.

        Interface contract (2 callers: `__init__`'s error message and the
        trainer tests' field-by-field argv diff):

        :returns: The public class attributes that are not callables.
        """
        return tuple(
            name
            for name, value in vars(cls).items()
            if not name.startswith("_") and not callable(value)
            and not isinstance(value, (classmethod, staticmethod, property))
        )

    def as_dict(self) -> Dict[str, Any]:
        """Resolve every field to its effective value.

        :returns: `{field: value}` with instance overrides applied over the
            class defaults.
        """
        return {name: getattr(self, name) for name in self.field_names()}

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in sorted(self.as_dict().items()))
        return f"TrainingConfig({body})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainingConfig):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    @property
    def resolved_truncate_dim(self) -> Any:
        """:returns: `None` if `truncate_dim == 0` (no MRL truncation),
        else `truncate_dim`, for `Qwen3EmbeddingModel`'s own `Optional[int]`
        constructor contract."""
        return self.truncate_dim if self.truncate_dim > 0 else None


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

#: argparse `dest` -> `TrainingConfig` field. THE wiring table.
CLI_TO_CONFIG: Dict[str, str] = {
    "encoding_name": "encoding_name",
    "hidden_size": "hidden_size",
    "num_layers": "num_layers",
    "num_heads": "num_heads",
    "intermediate_size": "intermediate_size",
    "dropout_rate": "dropout_rate",
    "ffn_type": "ffn_type",
    "normalization_type": "normalization_type",
    "attention_type": "attention_type",
    "normalize_embeddings": "normalize_embeddings",
    "truncate_dim": "truncate_dim",
    "infonce_temperature": "infonce_temperature",
    "reranker_maxlen": "reranker_maxlen",
    "query_maxlen": "query_maxlen",
    "doc_maxlen": "doc_maxlen",
    "nway": "nway",
    "num_train_groups": "num_train_groups",
    "num_val_groups": "num_val_groups",
    "query_words": "query_words",
    "doc_words": "doc_words",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "learning_rate": "learning_rate",
    "warmup_epochs": "warmup_epochs",
    "weight_decay": "weight_decay",
    "gradient_clipping": "gradient_clipping",
    "optimizer_type": "optimizer_type",
    "lr_schedule_type": "lr_schedule_type",
    "patience": "patience",
    "seed": "seed",
    "output_root": "output_root",
    "results_dir_prefix": "results_dir_prefix",
    "smoke": "smoke",
}

#: Applied only when `--smoke` resolves truthy, and only to fields the
#: caller did not type explicitly. Sized so a full run finishes in seconds.
SMOKE_PRESET: Dict[str, Any] = {
    "hidden_size": 16,
    "num_layers": 1,
    "num_heads": 2,
    "intermediate_size": 32,
    "query_maxlen": 8,
    "doc_maxlen": 16,
    "reranker_maxlen": 24,
    "batch_size": 2,
    "epochs": 1,
    "num_train_groups": 4,
    "num_val_groups": 2,
    "patience": 1,
}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every flag in :data:`CLI_TO_CONFIG` plus `--gpu`.

    Interface contract (2 callers: both `train_qwen3_*.py` `build_parser`
    functions, which may add recipe-specific flags on top):

    :param parser: The trainer's parser, already constructed.
    :returns: The same parser, for chaining.
    """
    default = TrainingConfig

    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index for setup_gpu. Process-level; never reaches the config.",
    )
    parser.add_argument("--encoding-name", type=str, default=default.encoding_name,
                        help="Tiktoken encoding name; also determines vocab_size "
                             "and the reranker's yes/no token ids.")
    parser.add_argument("--hidden-size", type=int, default=default.hidden_size)
    parser.add_argument("--num-layers", type=int, default=default.num_layers)
    parser.add_argument("--num-heads", type=int, default=default.num_heads)
    parser.add_argument("--intermediate-size", type=int, default=default.intermediate_size)
    parser.add_argument("--dropout-rate", type=float, default=default.dropout_rate)
    parser.add_argument("--ffn-type", type=str, default=default.ffn_type)
    parser.add_argument("--normalization-type", type=str, default=default.normalization_type)
    parser.add_argument("--attention-type", type=str, default=default.attention_type)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction,
                        default=default.normalize_embeddings,
                        help="L2-normalize the pooled embedding (disable with "
                             "--no-normalize-embeddings).")
    parser.add_argument("--truncate-dim", type=int, default=default.truncate_dim,
                        help="Matryoshka truncation width; 0 disables it.")
    parser.add_argument("--infonce-temperature", type=float,
                        default=default.infonce_temperature)
    parser.add_argument("--reranker-maxlen", type=int, default=default.reranker_maxlen,
                        help="Max tokens for the reranker's formatted prompt.")
    parser.add_argument("--query-maxlen", type=int, default=default.query_maxlen)
    parser.add_argument("--doc-maxlen", type=int, default=default.doc_maxlen)
    parser.add_argument("--nway", type=int, default=default.nway,
                        help="Candidates per synthetic group, positive at index 0.")
    parser.add_argument("--num-train-groups", type=int, default=default.num_train_groups)
    parser.add_argument("--num-val-groups", type=int, default=default.num_val_groups)
    parser.add_argument("--query-words", type=int, default=default.query_words)
    parser.add_argument("--doc-words", type=int, default=default.doc_words)
    parser.add_argument("--batch-size", type=int, default=default.batch_size,
                        help="Groups per batch (rows per batch differ by recipe).")
    parser.add_argument("--epochs", type=int, default=default.epochs)
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate)
    parser.add_argument("--warmup-epochs", type=int, default=default.warmup_epochs)
    parser.add_argument("--weight-decay", type=float, default=default.weight_decay)
    parser.add_argument("--gradient-clipping", type=float, default=default.gradient_clipping)
    parser.add_argument("--optimizer-type", type=str, default=default.optimizer_type)
    parser.add_argument("--lr-schedule-type", type=str, default=default.lr_schedule_type)
    parser.add_argument("--patience", type=int, default=default.patience)
    parser.add_argument("--seed", type=int, default=default.seed)
    parser.add_argument("--output-root", type=str, default=default.output_root)
    parser.add_argument("--results-dir-prefix", type=str, default=default.results_dir_prefix)
    parser.add_argument("--smoke", action="store_true", default=default.smoke)
    return parser


# ---------------------------------------------------------------------
# Synthetic corpus (cloned in SHAPE from ColBERT's generator, package-local)
# ---------------------------------------------------------------------

_TOPIC_WORDS: Tuple[Tuple[str, ...], ...] = (
    ("nebula", "quasar", "parallax", "supernova", "perihelion", "redshift",
     "exoplanet", "magnetosphere"),
    ("saffron", "reduction", "emulsion", "braise", "confit", "mirepoix",
     "deglaze", "gastrique"),
    ("stamen", "rhizome", "cotyledon", "xylem", "phloem", "corolla",
     "photosynthesis", "chlorophyll"),
    ("stratum", "basalt", "sediment", "tectonic", "igneous", "erosion",
     "fossil", "moraine"),
    ("plumage", "migration", "clutch", "songbird", "wingspan", "roost",
     "fledgling", "talon"),
    ("cipher", "keystream", "nonce", "checksum", "entropy", "ciphertext",
     "salt", "hash"),
)
"""Disjoint topical vocabularies, distinct from ColBERT's own set -- a
package-local clone, not a shared import (`decisions.md` D-004)."""

_DOC_PUNCTUATION: Tuple[str, ...] = tuple(",.;:!?")


def _sentence(
    rng: np.random.Generator,
    topic: Tuple[str, ...],
    n_words: int,
    seed_words: Sequence[str] = (),
    punctuate: bool = False,
) -> str:
    """Compose one synthetic sentence from a topic vocabulary.

    :param rng: Seeded generator; the only randomness source.
    :param topic: The topic's word tuple.
    :param n_words: Total words in the sentence, at least `len(seed_words)`.
    :param seed_words: Words guaranteed to appear.
    :param punctuate: Whether to interleave punctuation symbols.
    :returns: A whitespace-joined sentence.
    """
    filler = max(n_words - len(seed_words), 0)
    words: List[str] = list(seed_words) + [
        str(w) for w in rng.choice(np.array(topic), size=filler, replace=True)
    ]
    rng.shuffle(words)
    if punctuate:
        words = [
            word + str(rng.choice(np.array(_DOC_PUNCTUATION)))
            if rng.random() < 0.25 else word
            for word in words
        ]
    return " ".join(words)


def _make_groups(
    config: TrainingConfig,
    num_groups: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Generate `num_groups` synthetic `<query, positive, negatives...>` tuples.

    :param config: Supplies `nway`, `query_words` and `doc_words`.
    :param num_groups: How many groups to generate.
    :param seed: Split-specific seed.
    :returns: `(queries, documents)` where `queries` has `num_groups`
        entries and `documents` has `num_groups * nway` entries laid out
        group-contiguously with each group's positive first.
    """
    rng = np.random.default_rng(seed)
    n_topics = len(_TOPIC_WORDS)

    queries: List[str] = []
    documents: List[str] = []

    for _ in range(num_groups):
        topic_ids = rng.permutation(n_topics)
        positive_topic = _TOPIC_WORDS[int(topic_ids[0])]

        query_words = [
            str(w) for w in rng.choice(
                np.array(positive_topic), size=config.query_words, replace=False
            )
        ]
        queries.append(" ".join(query_words))

        documents.append(
            _sentence(rng, positive_topic, config.doc_words,
                      seed_words=query_words, punctuate=True)
        )
        for k in range(1, config.nway):
            negative_topic = _TOPIC_WORDS[int(topic_ids[k % n_topics])]
            documents.append(
                _sentence(rng, negative_topic, config.doc_words, punctuate=True)
            )

    return queries, documents


# ---------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------


def _resolve_yes_no_token_ids(encoding_name: str) -> Tuple[int, int]:
    """Derive the reranker's `yes_token_id`/`no_token_id` from the LIVE encoding.

    :param encoding_name: A tiktoken encoding name (e.g. `"cl100k_base"`).
    :returns: `(yes_token_id, no_token_id)`.
    :raises ValueError: If `"yes"` or `"no"` does not encode to exactly one
        token under this encoding -- the restricted-softmax mechanism
        (`Qwen3RerankerLayer.call`) requires a single scalar id for each.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    yes_ids = encoding.encode("yes")
    no_ids = encoding.encode("no")
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            f"encoding {encoding_name!r} does not tokenize 'yes'/'no' as single "
            f"tokens (got {yes_ids}/{no_ids}); Qwen3RerankerLayer's restricted "
            f"softmax needs exactly one id for each."
        )
    return int(yes_ids[0]), int(no_ids[0])


# ---------------------------------------------------------------------
# Embedding-tower dataset + model
# ---------------------------------------------------------------------


def build_embedding_datasets(
    config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, TiktokenPreprocessor, TiktokenPreprocessor]:
    """Build the synthetic (query, positive) pair datasets for InfoNCE training.

    Only the POSITIVE document (index 0 of each group) is used here -- see
    the module docstring's InfoNCE decision: negatives arrive for free as
    other rows in the same batch, so the explicit negatives `_make_groups`
    also generates are unused by training and are read only by
    :func:`embed_texts`-based separation checks.

    :param config: Supplies every geometry, size and seed field.
    :returns: `(train_dataset, val_dataset, query_tokenizer, doc_tokenizer)`.
        Each dataset yields `(inputs, dummy_label)` where `inputs` is a dict
        with keys `query_input_ids`/`query_attention_mask`/
        `doc_input_ids`/`doc_attention_mask`, and `dummy_label` is
        `zeros(batch,)` (`SymmetricInfoNCELoss` ignores `y_true` entirely,
        matching `train.embeddings_experimental.train_embeddings`'s own
        `with_dummy_targets` convention for the same loss).
    """
    query_tokenizer = create_tokenizer(
        encoding_name=config.encoding_name, max_length=config.query_maxlen,
    )
    doc_tokenizer = create_tokenizer(
        encoding_name=config.encoding_name, max_length=config.doc_maxlen,
    )

    splits = []
    for offset, num_groups in (
        (0, config.num_train_groups),
        (1, config.num_val_groups),
    ):
        queries, documents = _make_groups(config, num_groups, seed=config.seed + offset)
        # Positive-only: index 0 of every `nway`-sized group.
        positives = documents[:: config.nway]

        encoded_q = query_tokenizer(list(queries), return_tensors="np")
        encoded_d = doc_tokenizer(list(positives), return_tensors="np")
        inputs = {
            "query_input_ids": encoded_q["input_ids"].astype("int32"),
            "query_attention_mask": encoded_q["attention_mask"].astype("int32"),
            "doc_input_ids": encoded_d["input_ids"].astype("int32"),
            "doc_attention_mask": encoded_d["attention_mask"].astype("int32"),
        }
        labels = np.zeros((num_groups,), dtype="float32")

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        splits.append(
            dataset.batch(config.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        )

    logger.info(
        f"Qwen3 embedding tower: {config.num_train_groups} train / "
        f"{config.num_val_groups} val (query, positive) pairs, "
        f"batch_size={config.batch_size} (in-batch negatives)."
    )
    return splits[0], splits[1], query_tokenizer, doc_tokenizer


# DECISION plan-2026-09-13T073704-245ab5d5/D-013
# WHAT NOT TO DO: do not reshape the synthetic groups into ColBERT's flattened
# nway-group listwise batch layout, and do not fall back to
# ColBERTPairwiseSoftmaxLoss. Both were the plan's two named candidate paths,
# and both target a LISTWISE/grouped loss shape -- but SymmetricInfoNCELoss
# (read in full before this was written) is not listwise at all: it is a
# two-view-of-one-batch, in-batch-negative loss that needs no explicit
# negative examples. The correct shape is this wrapper: encode the query and
# its positive document through the SAME tower as two views, stack them, and
# let InfoNCE's own in-batch mechanism supply negatives from the batch's other
# groups. See decisions.md D-013.
@register_dl_technique("dl_techniques.train.qwen3_embeddings.common")
class Qwen3EmbeddingTowerPair(keras.Model):
    """Encodes a (query, positive-document) pair through ONE shared tower.

    Mirrors `src/train/embeddings_experimental/train_embeddings.py`'s
    `SimCSEModel`: a single tower, two views, stacked `(batch, 2, dim)` so
    `compile(loss=SymmetricInfoNCELoss(...))` + stock `fit()` needs no
    custom `train_step`. The only structural difference from `SimCSEModel`
    is that the two views come from DISTINCT texts (query vs. its positive
    document) rather than two dropout passes of the identical text -- this
    is the standard bi-encoder in-batch-negatives training shape (as used
    e.g. by sentence-transformers' MultipleNegativesRankingLoss), not a
    misuse of a SimCSE-flavored loss.

    :param embedding_model: The shared `Qwen3EmbeddingModel` tower.
    :param kwargs: Forwarded to the base `Model` class.
    """

    def __init__(self, embedding_model: Qwen3EmbeddingModel, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.embedding_model = embedding_model

    def call(
        self, inputs: Dict[str, Any], training: Any = None
    ) -> Any:
        view_a = self.embedding_model(
            {
                "input_ids": inputs["query_input_ids"],
                "attention_mask": inputs["query_attention_mask"],
            },
            training=training,
        )
        view_b = self.embedding_model(
            {
                "input_ids": inputs["doc_input_ids"],
                "attention_mask": inputs["doc_attention_mask"],
            },
            training=training,
        )
        return keras.ops.stack([view_a, view_b], axis=1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["embedding_model"] = keras.saving.serialize_keras_object(
            self.embedding_model
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3EmbeddingTowerPair":
        config = dict(config)
        config["embedding_model"] = keras.saving.deserialize_keras_object(
            config["embedding_model"]
        )
        return cls(**config)


def build_embedding_model(
    config: TrainingConfig,
    vocab_size: int,
    optimizer: keras.optimizers.Optimizer,
) -> Qwen3EmbeddingTowerPair:
    """Construct and compile the embedding-tower training wrapper.

    :param config: Supplies every model geometry field.
    :param vocab_size: The live tokenizer's vocab size.
    :param optimizer: Already built by `build_optimizer`.
    :returns: A compiled `Qwen3EmbeddingTowerPair`.

    .. note::

        `jit_compile=False` is set deliberately, matching
        `train.embeddings_experimental.train_embeddings`'s own measured note:
        two forward passes of one batch feeding a symmetric cross-entropy
        fails to XLA-compile on this TF 2.18 build
        (`FAILED_PRECONDITION: Can not combine dim orders and
        requirements`). Stock `fit()` still applies; only XLA is off.
    """
    embedding_model = Qwen3EmbeddingModel(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=max(config.query_maxlen, config.doc_maxlen),
        normalize=config.normalize_embeddings,
        truncate_dim=config.resolved_truncate_dim,
        dropout_rate=config.dropout_rate,
        ffn_type=config.ffn_type,
        normalization_type=config.normalization_type,
        attention_type=config.attention_type,
        name="qwen3_embedding_model",
    )
    model = Qwen3EmbeddingTowerPair(embedding_model, name="qwen3_embedding_tower_pair")
    model.compile(
        optimizer=optimizer,
        loss=SymmetricInfoNCELoss(temperature=config.infonce_temperature),
        jit_compile=False,
    )
    logger.info(
        f"Qwen3EmbeddingModel: hidden_size={config.hidden_size}, "
        f"num_layers={config.num_layers}, vocab_size={vocab_size}, "
        f"loss=SymmetricInfoNCELoss(temperature={config.infonce_temperature})"
    )
    return model


def embed_texts(
    embedding_model: Qwen3EmbeddingModel,
    tokenizer: TiktokenPreprocessor,
    texts: Sequence[str],
) -> np.ndarray:
    """Encode `texts` to embeddings, deterministically (no dropout).

    :param embedding_model: A (possibly unbuilt) `Qwen3EmbeddingModel`.
    :param tokenizer: The tokenizer the model was trained with.
    :param texts: Raw strings to embed.
    :returns: `(len(texts), embedding_dim)` float32 array.
    """
    encoded = tokenizer(list(texts), return_tensors="np")
    inputs = {
        "input_ids": encoded["input_ids"].astype("int32"),
        "attention_mask": encoded["attention_mask"].astype("int32"),
    }
    embeddings = embedding_model(inputs, training=False)
    return np.asarray(keras.ops.convert_to_numpy(embeddings))


def embedding_steps_per_epoch(config: TrainingConfig) -> int:
    """:returns: `num_train_groups // batch_size`, at least 1."""
    return max(config.num_train_groups // config.batch_size, 1)


# ---------------------------------------------------------------------
# Reranker dataset + model
# ---------------------------------------------------------------------


def _format_reranker_prompt(query: str, document: str) -> str:
    """The wiring-only "yes/no" prompt template the reranker scores.

    :param query: The query text.
    :param document: The candidate document text.
    :returns: A single formatted prompt string.
    """
    return f"Query: {query}\nDocument: {document}\nRelevant:"


def build_reranker_datasets(
    config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, TiktokenPreprocessor, int, int]:
    """Build the synthetic (prompt, 0/1 relevance) datasets for BCE training.

    Every group contributes `nway` rows: the positive (label 1) and
    `nway - 1` negatives (label 0). Unlike ColBERT's grouped batching, row
    order is NOT load-bearing here -- each row is independently labelled, so
    the dataset is shuffled at the row level.

    :param config: Supplies every geometry, size and seed field.
    :returns: `(train_dataset, val_dataset, tokenizer, yes_token_id,
        no_token_id)`.
    """
    tokenizer = create_tokenizer(
        encoding_name=config.encoding_name, max_length=config.reranker_maxlen,
    )
    yes_token_id, no_token_id = _resolve_yes_no_token_ids(config.encoding_name)

    rows_per_batch = config.batch_size * config.nway
    splits = []
    for offset, num_groups in (
        (0, config.num_train_groups),
        (1, config.num_val_groups),
    ):
        queries, documents = _make_groups(config, num_groups, seed=config.seed + 100 + offset)
        prompts: List[str] = []
        labels: List[float] = []
        for group_idx in range(num_groups):
            query = queries[group_idx]
            for k in range(config.nway):
                doc = documents[group_idx * config.nway + k]
                prompts.append(_format_reranker_prompt(query, doc))
                labels.append(1.0 if k == 0 else 0.0)

        encoded = tokenizer(prompts, return_tensors="np")
        inputs = {
            "input_ids": encoded["input_ids"].astype("int32"),
            "attention_mask": encoded["attention_mask"].astype("int32"),
        }
        labels_arr = np.asarray(labels, dtype="float32")

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels_arr))
        if offset == 0:
            dataset = dataset.shuffle(len(prompts), seed=config.seed, reshuffle_each_iteration=True)
        splits.append(
            dataset.batch(rows_per_batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        )

    logger.info(
        f"Qwen3 reranker: {config.num_train_groups} train / {config.num_val_groups} "
        f"val groups at nway={config.nway} ({config.num_train_groups * config.nway} "
        f"train rows); yes_token_id={yes_token_id}, no_token_id={no_token_id}."
    )
    return splits[0], splits[1], tokenizer, yes_token_id, no_token_id


def build_reranker_model(
    config: TrainingConfig,
    vocab_size: int,
    yes_token_id: int,
    no_token_id: int,
    optimizer: keras.optimizers.Optimizer,
) -> Qwen3RerankerModel:
    """Construct and compile the reranker for stock `fit()`.

    :param config: Supplies every model geometry field.
    :param vocab_size: The live tokenizer's vocab size.
    :param yes_token_id: From :func:`_resolve_yes_no_token_ids`.
    :param no_token_id: From :func:`_resolve_yes_no_token_ids`.
    :param optimizer: Already built by `build_optimizer`.
    :returns: A compiled `Qwen3RerankerModel`. The model already emits a
        probability (`softmax([no_logit, yes_logit])[:, 1]`), so
        `keras.losses.BinaryCrossentropy` applies directly -- no new loss
        class is needed.
    """
    model = Qwen3RerankerModel(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.reranker_maxlen,
        dropout_rate=config.dropout_rate,
        ffn_type=config.ffn_type,
        normalization_type=config.normalization_type,
        attention_type=config.attention_type,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        name="qwen3_reranker_model",
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    logger.info(
        f"Qwen3RerankerModel: hidden_size={config.hidden_size}, "
        f"num_layers={config.num_layers}, vocab_size={vocab_size}, "
        f"yes_token_id={yes_token_id}, no_token_id={no_token_id}, "
        f"loss=BinaryCrossentropy"
    )
    return model


def reranker_steps_per_epoch(config: TrainingConfig) -> int:
    """:returns: `(num_train_groups * nway) // (batch_size * nway)`, i.e.
    `num_train_groups // batch_size`, at least 1 -- identical horizon to the
    embedding tower for the same synthetic corpus size."""
    return max(config.num_train_groups // config.batch_size, 1)
