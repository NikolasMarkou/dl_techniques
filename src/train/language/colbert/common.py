"""Shared scaffold for the ColBERT v1 and v2 training recipes.

Holds the three things both entry points need and neither should own:

- :class:`TrainingConfig` -- a plain class with class-level defaults, following
  ``src/train/bert/pretrain.py`` (Pattern 3, "NLP Pretrain/Finetune"), extended
  with a keyword ``__init__`` so the shared ``--smoke`` merge mechanism
  (:func:`train.common.args.config_values_from_args`) can construct it from a
  values dict.
- :func:`build_datasets` -- the **synthetic** triples / ``nway``-tuple pipeline.
- :func:`build_model` -- variant construction plus ``compile``.

Data provenance
---------------

**The training data is synthetic and generated in this file.** There is no MS
MARCO and no other information-retrieval dataset anywhere in this repository
(measured, not assumed: a repo-wide grep for ``msmarco``/``ms_marco`` returns
zero hits, and ``dl_techniques/datasets/`` carries no retrieval loader). The
generator here is therefore not a stand-in for a real corpus; it exists to move
tensors of the right shape, dtype and grouping through the real model, the real
loss and the real optimizer.

The task is nevertheless **learnable in principle**: a query is a bag of words
drawn from one topic, its positive document repeats those words verbatim inside
a longer sentence from the same topic, and every negative is drawn from a
*different* topic and shares no vocabulary with the query. A model that learns
lexical overlap can separate them.

Be precise about how weak that signal is. It says a falling loss is not
*meaningless*; it does not say a falling loss is evidence of retrieval quality.
Distinguishing "documents that literally repeat the query words" from
"documents about a disjoint topic" is the easiest possible discrimination, it is
solvable by term matching with no semantics at all, and it is measured on data
drawn from the same generator it was trained on. Combined with a backbone
initialized from scratch -- ``BERT.from_variant(pretrained=True)`` raises here,
so there are no pretrained weights to start from -- the correct reading of a
successful run is "the pipeline is wired end to end", full stop.

Batching invariant
------------------

A ColBERT training example is a **group**: one query and ``nway`` candidate
documents with the positive at index 0. ``ColBERT.call`` scores exactly one
(query, document) pair per row, and both losses reshape a flat ``(batch *
nway,)`` score vector back into ``(batch, nway)`` rows. The pipeline therefore
emits ``nway`` group-contiguous rows per group and batches in multiples of
``nway`` with ``drop_remainder=True``. Row order inside a batch is load-bearing;
see the ``D-025`` anchor on :func:`_to_dataset`.
"""

from typing import Any, Dict, List, Sequence, Tuple

import argparse
import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.language.colbert import ColBERT, ColBERTTokenizer
from dl_techniques.utils.logger import logger

__all__ = [
    "CLI_TO_CONFIG",
    "SMOKE_PRESET",
    "TrainingConfig",
    "add_common_arguments",
    "build_datasets",
    "build_model",
    "steps_per_epoch_for",
]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


class TrainingConfig:
    """Configuration shared by both ColBERT training recipes.

    Plain class with class-level defaults, matching
    ``src/train/bert/pretrain.py``. The keyword ``__init__`` is the one addition
    over that exemplar: it lets :func:`train.common.args.config_values_from_args`
    hand over a values dict, and it **rejects unknown field names** so a renamed
    CLI destination fails loudly instead of quietly setting an attribute nothing
    reads.

    :param overrides: Field name to value. Every name must already exist as a
        class attribute.
    :raises TypeError: If any override names a field this class does not define.
    """

    # -- Model -------------------------------------------------------
    colbert_variant: str = "tiny"

    # Tiktoken ``cl100k_base`` has 100277 entries, and the ColBERT tokenizer
    # emits ids from that encoding directly -- including its [CLS]/[SEP]/[PAD]/
    # [MASK] specials and the two derived [Q]/[D] markers, all of which sit at
    # the TOP of the range. The backbone's embedding table must therefore cover
    # the full 100277, not BERT's WordPiece 30522: a smaller table makes every
    # marker and special an out-of-range lookup. `src/train/bert/pretrain.py`
    # makes the identical explicit choice for the identical reason.
    vocab_size: int = 100277

    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 220

    # -- Objective ---------------------------------------------------
    #: Candidates per query, positive at index 0. 2 is the v1 triple; the v2
    #: entry point raises its own parser default (the reference uses 64).
    nway: int = 2
    #: v2 only: scales teacher scores before their log-softmax. Ignored by v1.
    distillation_alpha: float = 1.0

    # -- Training ----------------------------------------------------
    #: Groups per batch. Rows per batch are ``batch_size * nway``.
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 3e-5
    warmup_epochs: int = 1
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    patience: int = 5
    seed: int = 42

    # -- Synthetic data ----------------------------------------------
    num_train_groups: int = 256
    num_val_groups: int = 64
    #: Query length in words. Documents are longer by construction.
    query_words: int = 4
    doc_words: int = 24

    # -- Output ------------------------------------------------------
    # Repo-root ``results/`` is the default and the only value a real run should
    # use. It is a FIELD rather than a constant purely so tests can point it at
    # a ``tmp_path``; `tests/conftest.py` errors on any test that writes into
    # repo-root ``results/``.
    #
    # There is deliberately no second static ``save_dir``: the timestamped run
    # directory returned by ``train.common.callbacks.create_callbacks`` is the
    # ONE place artifacts go, and a parallel config path is exactly the
    # divergent-spelling defect `src/train/CLAUDE.md` documents at length.
    output_root: str = "results"
    results_dir_prefix: str = "colbert"

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

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        """Every configurable field name, in declaration order.

        Interface contract (2 callers: :meth:`__init__`'s error message and the
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

        :returns: ``{field: value}`` with instance overrides applied over the
            class defaults.
        """
        return {name: getattr(self, name) for name in self.field_names()}

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in sorted(self.as_dict().items()))
        return f"TrainingConfig({body})"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

#: argparse ``dest`` -> :class:`TrainingConfig` field. THE wiring table.
#:
#: Every entry here is forwarded in a LOOP by
#: :func:`train.common.args.config_values_from_args`; neither trainer's
#: ``main()`` copies fields across by hand. A hand-written
#: ``config.x = args.x`` list is the documented silent-no-op bug class -- omit
#: one line and its flag becomes dead while ``--help`` still advertises it.
#: ``--gpu`` is intentionally absent: it is a process-level concern consumed by
#: ``setup_gpu``, and never reaches the config.
CLI_TO_CONFIG: Dict[str, str] = {
    "variant": "colbert_variant",
    "vocab_size": "vocab_size",
    "dim": "dim",
    "query_maxlen": "query_maxlen",
    "doc_maxlen": "doc_maxlen",
    "nway": "nway",
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
    "num_train_groups": "num_train_groups",
    "num_val_groups": "num_val_groups",
    "query_words": "query_words",
    "doc_words": "doc_words",
    "output_root": "output_root",
    "results_dir_prefix": "results_dir_prefix",
    "smoke": "smoke",
}

#: Applied only when ``--smoke`` resolves truthy, and only to fields the caller
#: did not type explicitly. Sized so a full run finishes in seconds on CPU.
SMOKE_PRESET: Dict[str, Any] = {
    "colbert_variant": "tiny",
    "dim": 16,
    "query_maxlen": 8,
    "doc_maxlen": 24,
    "batch_size": 2,
    "epochs": 1,
    "warmup_epochs": 0,
    "num_train_groups": 8,
    "num_val_groups": 4,
    "patience": 1,
}


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every flag in :data:`CLI_TO_CONFIG` plus ``--gpu``.

    Interface contract (2 callers: both ``train_colbert_v*.py`` ``build_parser``
    functions, which may afterwards add recipe-specific flags and may override a
    default with ``parser.set_defaults``):

    Every default is read from :class:`TrainingConfig` so the class stays the
    single source of truth. That matters more than it looks: because
    ``config_values_from_args`` forwards *every* mapped dest, an argparse default
    that disagreed with the class default would silently win.

    :param parser: The trainer's parser, already constructed.
    :returns: The same parser, for chaining.
    """
    default = TrainingConfig

    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index for setup_gpu. Process-level; never reaches the config.",
    )
    parser.add_argument("--variant", type=str, default=default.colbert_variant,
                        help="ColBERT variant: tiny, small, base or large.")
    parser.add_argument("--vocab-size", type=int, default=default.vocab_size,
                        help="Backbone vocabulary size (must cover the tokenizer).")
    parser.add_argument("--dim", type=int, default=default.dim,
                        help="Retrieval embedding width.")
    parser.add_argument("--query-maxlen", type=int, default=default.query_maxlen,
                        help="Fixed query length, [MASK]-augmented to fill.")
    parser.add_argument("--doc-maxlen", type=int, default=default.doc_maxlen,
                        help="Maximum document length.")
    parser.add_argument("--nway", type=int, default=default.nway,
                        help="Candidates per query, positive at index 0.")
    parser.add_argument("--batch-size", type=int, default=default.batch_size,
                        help="Groups per batch (rows per batch are this times nway).")
    parser.add_argument("--epochs", type=int, default=default.epochs,
                        help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate,
                        help="Peak learning rate.")
    parser.add_argument("--warmup-epochs", type=int, default=default.warmup_epochs,
                        help="Warmup epochs for the LR schedule.")
    parser.add_argument("--weight-decay", type=float, default=default.weight_decay,
                        help="AdamW weight decay (ignored by other optimizers).")
    parser.add_argument("--gradient-clipping", type=float,
                        default=default.gradient_clipping,
                        help="Global gradient clip norm.")
    parser.add_argument("--optimizer-type", type=str, default=default.optimizer_type,
                        help="Optimizer key for dl_techniques.optimization.")
    parser.add_argument("--lr-schedule-type", type=str,
                        default=default.lr_schedule_type,
                        help="Schedule key: cosine_decay, exponential_decay, ...")
    parser.add_argument("--patience", type=int, default=default.patience,
                        help="Early-stopping patience on val_loss.")
    parser.add_argument("--seed", type=int, default=default.seed,
                        help="Seed for weights and for the synthetic data.")
    parser.add_argument("--num-train-groups", type=int,
                        default=default.num_train_groups,
                        help="Synthetic training groups (query + nway candidates).")
    parser.add_argument("--num-val-groups", type=int, default=default.num_val_groups,
                        help="Synthetic validation groups.")
    parser.add_argument("--query-words", type=int, default=default.query_words,
                        help="Words per synthetic query.")
    parser.add_argument("--doc-words", type=int, default=default.doc_words,
                        help="Words per synthetic document.")
    parser.add_argument("--output-root", type=str, default=default.output_root,
                        help="Root for the timestamped run directory. Repo-root "
                             "'results' in every real run; point it at a tmp dir "
                             "in tests.")
    parser.add_argument("--results-dir-prefix", type=str,
                        default=default.results_dir_prefix,
                        help="Prefix of the timestamped run directory name.")
    parser.add_argument("--smoke", action="store_true", default=default.smoke,
                        help="Tiny end-to-end wiring run: applies SMOKE_PRESET to "
                             "every field not typed explicitly.")
    return parser


# ---------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------

# Disjoint topical vocabularies. Disjointness is the whole mechanism: a positive
# repeats the query's own words, and a negative is drawn from a different tuple
# and therefore shares none of them. Keep them disjoint if you edit this.
_TOPIC_WORDS: Tuple[Tuple[str, ...], ...] = (
    ("glacier", "tundra", "permafrost", "moraine", "crevasse", "icefall",
     "snowpack", "meltwater"),
    ("sonata", "cadenza", "arpeggio", "timbre", "counterpoint", "fugue",
     "crescendo", "octave"),
    ("mitochondria", "ribosome", "chloroplast", "cytoplasm", "enzyme",
     "protein", "membrane", "nucleus"),
    ("aqueduct", "amphitheatre", "forum", "basilica", "colonnade", "portico",
     "mosaic", "fresco"),
    ("kernel", "compiler", "register", "pipeline", "scheduler", "allocator",
     "linker", "debugger"),
    ("monsoon", "cyclone", "isobar", "humidity", "troposphere", "updraft",
     "hailstone", "squall"),
)

# Punctuation sprinkled into documents so the tokenizer's skiplist mask has
# something to zero. Queries never receive punctuation: the skiplist is
# documents-only and the trainers must not blur that.
_DOC_PUNCTUATION: Tuple[str, ...] = tuple(",.;:!?")


def _sentence(
    rng: np.random.Generator,
    topic: Tuple[str, ...],
    n_words: int,
    seed_words: Sequence[str] = (),
    punctuate: bool = False,
) -> str:
    """Compose one synthetic sentence from a topic vocabulary.

    Interface contract (called by :func:`_make_groups` for queries, positives
    and negatives alike):

    :param rng: Seeded generator; the only randomness source.
    :param topic: The topic's word tuple.
    :param n_words: Total words in the sentence, at least ``len(seed_words)``.
    :param seed_words: Words guaranteed to appear (the query's words, when
        composing that query's positive document).
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
) -> Tuple[List[str], List[str], np.ndarray]:
    """Generate ``num_groups`` synthetic ``<query, positive, negatives...>`` tuples.

    Interface contract (called by :func:`build_datasets` once per split):

    :param config: Supplies ``nway``, ``query_words`` and ``doc_words``.
    :param num_groups: How many groups to generate.
    :param seed: Split-specific seed. Train and validation must differ.
    :returns: ``(queries, documents, teacher_scores)`` where ``queries`` has
        ``num_groups`` entries, ``documents`` has ``num_groups * nway`` entries
        laid out group-contiguously with each group's positive first, and
        ``teacher_scores`` is ``(num_groups * nway,)`` float32.
    :raises ValueError: If ``nway`` exceeds the number of available topics (each
        negative needs its own distinct topic) or is below 2.
    """
    if config.nway < 2:
        raise ValueError(
            f"nway must be >= 2 (a one-candidate group has an identically-1.0 "
            f"softmax and no gradient), got {config.nway}"
        )
    if config.nway > len(_TOPIC_WORDS):
        raise ValueError(
            f"nway={config.nway} exceeds the {len(_TOPIC_WORDS)} disjoint synthetic "
            f"topics available: every negative is drawn from a topic distinct from "
            f"the query's, so nway-1 negatives need nway-1 other topics. Add topics "
            f"to _TOPIC_WORDS or lower --nway."
        )

    rng = np.random.default_rng(seed)
    n_topics = len(_TOPIC_WORDS)

    queries: List[str] = []
    documents: List[str] = []
    teacher: List[float] = []

    for _ in range(num_groups):
        topic_ids = rng.permutation(n_topics)
        positive_topic = _TOPIC_WORDS[int(topic_ids[0])]

        query_words = [
            str(w) for w in rng.choice(
                np.array(positive_topic), size=config.query_words, replace=False
            )
        ]
        queries.append(" ".join(query_words))

        # Index 0 is the positive, by position. Both losses depend on it.
        documents.append(
            _sentence(rng, positive_topic, config.doc_words,
                      seed_words=query_words, punctuate=True)
        )
        for k in range(1, config.nway):
            negative_topic = _TOPIC_WORDS[int(topic_ids[k % n_topics])]
            documents.append(
                _sentence(rng, negative_topic, config.doc_words, punctuate=True)
            )

        # Synthetic "cross-encoder" teacher. It is NOT a cross-encoder: it is
        # the generator telling the student the answer it already knows, with
        # noise so the target distribution is not a one-hot. Only v2 reads it.
        teacher.append(4.0 + float(rng.normal(0.0, 0.25)))
        teacher.extend(
            float(rng.normal(0.0, 0.5)) for _ in range(config.nway - 1)
        )

    return queries, documents, np.asarray(teacher, dtype="float32")


def _to_dataset(
    tokenizer: ColBERTTokenizer,
    queries: Sequence[str],
    documents: Sequence[str],
    targets: np.ndarray,
    nway: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Tokenize one split and batch it group-contiguously.

    Interface contract (called by :func:`build_datasets` once per split):

    :param tokenizer: Supplies the asymmetric query/document encodings.
    :param queries: ``num_groups`` query strings.
    :param documents: ``num_groups * nway`` document strings, positive first
        within each group.
    :param targets: ``(num_groups * nway,)`` float32 label tensor -- zeros for
        v1 (unused by that loss) or teacher scores for v2.
    :param nway: Candidates per group.
    :param batch_size: Groups per batch.
    :returns: A batched ``tf.data.Dataset`` yielding
        ``(inputs, {"score": targets})``.
    """
    encoded_queries = tokenizer.tokenize_queries(list(queries))
    encoded_docs = tokenizer.tokenize_documents(list(documents))

    # One encode per query, then repeat -- exactly equivalent to encoding the
    # repeated list, at 1/nway the tokenizer work.
    inputs = {
        "query_input_ids": np.repeat(encoded_queries["input_ids"], nway, axis=0),
        "query_attention_mask": np.repeat(
            encoded_queries["attention_mask"], nway, axis=0
        ),
        "doc_input_ids": encoded_docs["input_ids"],
        "doc_attention_mask": encoded_docs["attention_mask"],
        "doc_skiplist_mask": encoded_docs["skiplist_mask"],
    }

    # DECISION plan-2026-08-25T121346-c71fc3ad/D-025
    # ROW ORDER AND BATCH SIZE ARE LOAD-BEARING. Both losses recover the
    # candidate group by RESHAPING a flat (batch*nway,) score vector to
    # (batch, nway) and treating column 0 as the positive. That is only correct
    # while each batch holds whole, contiguous, positive-first groups.
    # WHAT NOT TO DO: (1) do NOT add `.shuffle()` here or anywhere after the
    # flattening -- it interleaves rows from different groups and every
    # "positive" the loss then sees is an arbitrary candidate, which trains
    # confidently on garbage while the loss curve still falls; shuffling is done
    # at GROUP level inside `_make_groups`, before the flattening. (2) do NOT
    # drop `drop_remainder=True` -- a short final batch is not a multiple of
    # nway and `_reshape_to_nway` either raises or, for a statically-unknown
    # length, silently regroups across group boundaries. (3) do NOT batch by
    # rows-per-batch directly; it is `batch_size` GROUPS times nway.
    # See decisions.md D-025.
    dataset = tf.data.Dataset.from_tensor_slices((inputs, {"score": targets}))
    return dataset.batch(batch_size * nway, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )


def build_datasets(
    config: TrainingConfig,
    use_teacher_targets: bool,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, ColBERTTokenizer]:
    """Build the synthetic train and validation datasets.

    Interface contract (2 callers: both ``train_colbert_v*.py`` training
    functions; v1 passes ``use_teacher_targets=False``, v2 ``True``):

    :param config: Supplies every geometry, size and seed field.
    :param use_teacher_targets: ``True`` emits the synthetic teacher scores as
        the label (v2 distillation); ``False`` emits zeros, which
        :class:`~dl_techniques.losses.ColBERTPairwiseSoftmaxLoss` ignores --
        that loss finds the positive by POSITION, never by label.
    :returns: ``(train_dataset, val_dataset, tokenizer)``.
    :raises ValueError: Propagated from :func:`_make_groups`, or if a split has
        fewer groups than one batch (which ``drop_remainder=True`` would render
        empty).
    """
    for name, count in (
        ("num_train_groups", config.num_train_groups),
        ("num_val_groups", config.num_val_groups),
    ):
        if count < config.batch_size:
            raise ValueError(
                f"{name}={count} is smaller than batch_size={config.batch_size}; "
                f"with drop_remainder=True that split would be EMPTY and fit() "
                f"would silently run zero steps."
            )

    tokenizer = ColBERTTokenizer(
        query_maxlen=config.query_maxlen,
        doc_maxlen=config.doc_maxlen,
        mask_punctuation=True,
    )
    logger.info(
        f"Tokenizer: query_maxlen={tokenizer.query_maxlen}, "
        f"doc_maxlen={tokenizer.doc_maxlen}, "
        f"|skiplist|={len(tokenizer.punctuation_token_ids)}, "
        f"[Q]={tokenizer.query_marker_token_id}, [D]={tokenizer.doc_marker_token_id}"
    )

    splits = []
    for offset, num_groups in (
        (0, config.num_train_groups),
        (1, config.num_val_groups),
    ):
        queries, documents, teacher = _make_groups(
            config, num_groups, seed=config.seed + offset
        )
        targets = teacher if use_teacher_targets else np.zeros_like(teacher)
        splits.append(
            _to_dataset(
                tokenizer, queries, documents, targets,
                nway=config.nway, batch_size=config.batch_size,
            )
        )

    logger.info(
        f"Synthetic data: {config.num_train_groups} train / {config.num_val_groups} "
        f"val groups at nway={config.nway} "
        f"({config.num_train_groups * config.nway} train rows); "
        f"targets={'teacher scores' if use_teacher_targets else 'zeros (unused)'}"
    )
    return splits[0], splits[1], tokenizer


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def steps_per_epoch_for(config: TrainingConfig) -> int:
    """Optimizer steps per epoch, matching ``drop_remainder=True`` batching.

    Interface contract (2 callers: both trainers, which pass the result to
    ``train.common.optimizer.build_optimizer`` so the warmup and decay horizons
    match the real step count):

    :param config: Supplies ``num_train_groups`` and ``batch_size``.
    :returns: ``num_train_groups // batch_size``, at least 1.
    """
    return max(config.num_train_groups // config.batch_size, 1)


def build_model(
    config: TrainingConfig,
    loss: keras.losses.Loss,
    optimizer: keras.optimizers.Optimizer,
    factory: Any,
) -> ColBERT:
    """Construct a ColBERT variant and compile it for stock ``fit()``.

    Interface contract (2 callers: both trainers, which differ only in the
    ``factory`` and ``loss`` they hand in):

    :param config: Supplies the variant name and every geometry override.
    :param loss: :class:`~dl_techniques.losses.ColBERTPairwiseSoftmaxLoss` (v1)
        or :class:`~dl_techniques.losses.ColBERTDistillationLoss` (v2).
    :param optimizer: Already built by
        ``train.common.optimizer.build_optimizer``.
    :param factory: ``create_colbert_v1`` or ``create_colbert_v2``. They build
        the same network -- see either factory's docstring for why -- so this is
        provenance, not dispatch.
    :returns: A built, compiled :class:`ColBERT`.
    """
    max_position_embeddings = max(config.query_maxlen, config.doc_maxlen)
    model: ColBERT = factory(
        config.colbert_variant,
        vocab_size=config.vocab_size,
        dim=config.dim,
        query_maxlen=config.query_maxlen,
        doc_maxlen=config.doc_maxlen,
        max_position_embeddings=max_position_embeddings,
    )

    # Materialize the whole sub-layer tree before counting or compiling.
    model.build(
        {
            "query_input_ids": (None, config.query_maxlen),
            "doc_input_ids": (None, config.doc_maxlen),
        }
    )

    # DECISION plan-2026-08-25T121346-c71fc3ad/D-026
    # The loss is attached to the "score" OUTPUT KEY, not to the model as a
    # whole. `ColBERT.call` returns a fixed three-key dict -- `score`,
    # `query_embeddings`, `doc_embeddings` -- and Keras 3 matches a dict `loss`
    # against that structure, leaving the two embedding outputs unsupervised.
    # WHAT NOT TO DO: do NOT pass a bare `loss=...`; Keras then broadcasts it
    # across ALL THREE outputs and applies the score loss to the (batch, len,
    # dim) embedding tensors, which either raises deep inside the loss's reshape
    # or -- worse -- succeeds on a shape that happens to divide by nway. And do
    # NOT write a custom `train_step` to select the key: stock `fit()` is a
    # standing constraint here, and this dict-loss form is what removes the
    # temptation. The dataset's label side must mirror the key: `{"score": y}`.
    # See decisions.md D-026.
    model.compile(optimizer=optimizer, loss={"score": loss})

    logger.info(
        f"ColBERT-{config.colbert_variant.upper()}: {model.count_params():,} params "
        f"(dim={config.dim}, query_maxlen={config.query_maxlen}, "
        f"doc_maxlen={config.doc_maxlen}, vocab_size={config.vocab_size}); "
        f"loss={type(loss).__name__}"
    )
    return model
