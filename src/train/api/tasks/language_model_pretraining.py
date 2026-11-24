"""The ``language_model_pretraining`` task: a config, a trainer and the keyword-only function.

``train_language_model_pretraining(...)`` is the function a Python caller uses and the function ``python -m train
language_model_pretraining key=value ...`` calls, so the two cannot disagree. It validates its keywords into a
:class:`LanguageModelPretrainingConfig` and hands that to :class:`LanguageModelPretrainingTrainer`, a
:class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else (a task overrides hooks, never
stages). One family trains it, ``gpt2_lm`` (:mod:`train.api.text_models`): the library GPT-2 as a causal language model, a
functional model of float32 ids to logits, trained by the stock ``fit`` with the stock sparse cross entropy.

Interface contract
------------------
``LanguageModelPretrainingConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data`` are
    required. The groups are those of classification except ``data`` (:class:`~train.api.text_args.TextData`: the corpus, the
    tokenizer, the window, the split and the cap), ``transform_args`` (:class:`~train.api.text_args.TextTransformArgs`: no field)
    and ``metric_args`` (:class:`~train.api.text_args.LanguageMetricArgs`); there is no loss group (the loss is the stock
    cross entropy of the logits). ``analysis_args.conformal_alpha`` is refused (a conformal set is a set of classes, and a
    language model has none).

``LanguageModelPretrainingTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "language_model_pretraining"`` and ``success_metric = "val_loss"``.

    What a run is. The corpus is tokenized once into memory; the documents are split by a hash of ``(seed, document index)`` into
    train and selection documents and the corpus's own test documents are kept for the end; an epoch is ``n_train // batch_size``
    steps of an INFINITE stream (``fit_kwargs`` gives ``steps_per_epoch`` and ``validation_steps``) whose batch ``i`` is a pure
    function of ``(seed, i)``: epoch ``e`` puts the train documents in an order keyed on ``(seed, e)``, puts an end-of-text id
    after each and cuts windows of ``seq_len + 1`` tokens, inputs and targets shifted by one. The selection and test windows are
    fixed, so ``val_loss`` is a function of the weights alone: equal across passes, epochs and a resume. A resume CONTINUES the
    stream at the settled step (``resume_stream``); it never derives another seed, because the first process consumed the
    documents in THAT order.

    The bar (D-006, D-020). ``val_loss`` is a loss, so the bar is STRICTLY below and beyond the repo's float32 tie band
    (:func:`train.common.text_run.strictly_below`); ``success_value`` is the minimum of ``val_loss`` over the selection window. The
    threshold is ``min(unigram, epoch-zero val_loss)``: the unigram is the entropy of the selection targets (the best a model that
    ignores its context can do; COUNTED from the targets in ``build_data``, never evaluated, because a constant model over a
    50,261-wide vocabulary is 0.4 GB of logits a forward), and the epoch-zero value is read from the fit sidecar in
    ``evaluate_and_report`` (:func:`train.common.pretrain_run.epoch_zero_val_loss`), never from an attribute, because a resume
    never evaluates the untrained model. Under ``debug_args.smoke`` no bar is applied.

    ``init`` (plan-2026-11-24T092348-7806a290 D-005, D-040, D-042). A run starts from an earlier ``gpt2_lm`` run and from nothing else.
    ``init_identity`` gives the vocabulary and the tokenizer's digest this run's record will hold, the family's two ``fixed`` keys, so a
    source of another tokenizer is refused before the claim, naming both values (the window length is not one: no weight depends on
    it). ``init_overlap`` counts the source's seeded train DOCUMENTS among this run's selection documents. Epoch zero is then the
    LOADED weights' ``val_loss``, so the threshold is ``min(unigram, loaded)``, still strictly below; ``Evaluation.loaded`` stays
    ``None`` (the other tasks' higher-is-better comparison with the loaded value would judge a loss backwards).

    ``task_record`` is the schema-v2 ``task.json`` (:func:`train.api.predictor.build_language_model_record`: the length and the tokenizer's
    fields, digest and file hashes), written before ``fit``; a resume compares its ``DATA_KEYS`` and leaves the file as it is.

    The epoch-0 check is the untrained model's loss over the unigram's (``initial_loss_ratio``); no verdict on the initialization
    is given (``unjudged_note`` says why); a non-finite loss is refused.

    Reporting. ``evaluate_and_report`` follows the pretrain order: the best checkpoint reloaded and scored on the TEST windows, the
    data-free analysis (``run_analysis``: the weights and spectral analyses; the stock analysis reads class probabilities, which a
    next-token distribution over a vocabulary is not), the last epoch's weights scored, saved and reloaded through
    ``finalize_last_epoch``. ``Evaluation.extras`` carries ``baseline``, ``epoch_zero``, ``selection`` and ``perplexity``.

``train_language_model_pretraining(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"gpt2_lm/variant"``, e.g. ``"gpt2_lm/tiny"``; ``python -m train list_models task=language_model_pretraining``
        lists them.
    :param data: a mapping (``{"dataset": "synthetic", "tokenizer": "ascii95"}``), a :class:`~train.api.text_args.TextData`, or the
        path of a YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` / ``init`` / ``model_args``
        and the groups: as in :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A family that does
        not train this task, a model key the family refuses, a tokenizer directory that is missing or holds files of another
        hash, a ``seq_len`` above the position table and a TFDS directory without the reviews are refused before the run
        directory is claimed; a corpus too small to hold a selection window is refused by name after the data is read.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train language_model_pretraining help`` imports this
module to print the config's keys and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside
a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union

from .. import recipes, registry
from ..args import (
    BACKUP_EVERY_STEPS_HELP_EXACT, AnalysisArgs, AutoContext, CheckpointArgs, DebugArgs, LoggerArgs, OptimArgs, is_count)
from ..config import (
    AUTO,
    BACKUP_HELP_EXACT,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..errors import ConfigError
from ..predictor import build_language_model_record, overrides_of_variant
from ..run_dir import CHECKPOINTS_DIR
from ..text_args import (
    GPT2_LM,
    SYNTHETIC,
    SYNTHETIC_TRAIN_DOCUMENTS,
    TASK,
    WATCH_METRIC,
    LanguageMetricArgs,
    TextData,
    TextTransformArgs,
    load_text_data_config,
    read_text_data_file,
    text_facts,
)
from ..text_models import METRIC_NAMES, language_loss_and_metrics, recipe_notices_of
from ..text_tokenizer import check_tokenizer_dir, tokenizer_spec
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, check_holds_no_classes, config_problems
from .pretrain import selection_block

# ---------------------------------------------------------------------

#: Rows of the selection windows the reload check of ``final_model.keras`` predicts: each row is ``seq_len x vocab_size`` floats of
#: logits (205 MB for four rows at the 50,261-wide vocabulary and 256 tokens), twice, so the sample is small.
LOAD_CHECK_ROWS = 4

# DECISION plan-2026-11-24T092348-7806a290/D-050
# This task's resume continues the SAME stream at the settled step, so it sets `resume_stream_exact` on the trainer and gives the two
# helps their exact clause (`resume_help(backup=)` and `HELP_OVERRIDES`): the shared sentences say "a fresh pass of the data" and that
# examples repeat or are missed, which is false here. Do NOT drop one of the three (a test holds them equal) and do NOT reword the
# shared sentences for every task.
_RESUME_ORDER = (
    "the training stream continues at the step the killed run stopped at (epoch e is a permutation of the training documents "
    "keyed on (seed, e), so the resumed run reads the batches the uninterrupted run would have read next, never epoch 0's again "
    "and never a selection document), and the selection and test windows stay fixed"
)

_STREAM_RULE = (
    "the document split is keyed on (seed, document index) and never on the stream; train batch i is a function of (seed, step) "
    "alone: epoch e orders the train documents and then the windows by sha256 keys of (seed, e, position); a resume continues "
    "the stream at the settled step and never derives another seed"
)

_CONFORMAL = (
    f"a conformal prediction set is a set of classes, which a {TASK} run does not produce; its output is a distribution over "
    "the next token"
)


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


@dataclasses.dataclass(frozen=True)
class LanguageModelPretrainingConfig(StrictConfig):
    """Everything a language-model run is told; see the module contract.

    :cvar HELP_OVERRIDES: dotted key to the help this task gives it in place of its group's (``cli._field_help_lines`` reads it).
    """

    HELP_OVERRIDES: ClassVar[Dict[str, str]] = {"checkpoint_args.backup_every_steps": BACKUP_EVERY_STEPS_HELP_EXACT}

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": f"family/variant, e.g. {GPT2_LM}/tiny; python -m train list_models task={TASK} lists them"}
    )
    data: TextData = dataclasses.field(
        metadata={
            "help": "the corpus and the windows: data.dataset=synthetic (or imdb_reviews), with data.tokenizer, data.seq_len, "
            "data.validation_split and data.max_samples; or the path of a YAML file holding them (data=data.yaml), over which "
            "dotted keys win (data=data.yaml data.seq_len=128)",
            PATH_FORM: read_text_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(default="float32", metadata={"help": RUN_HELP["precision"]})
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help(
            "an epoch of the imdb_reviews corpus takes minutes", order=_RESUME_ORDER, backup=BACKUP_HELP_EXACT)},
    )
    init: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["init"]})
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"help": RUN_HELP["model_args"]}
    )
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: TextTransformArgs = dataclasses.field(default_factory=TextTransformArgs)
    metric_args: LanguageMetricArgs = dataclasses.field(default_factory=LanguageMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, and a conformal level (a language model has no classes)."""
        return config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL)


class LanguageModelPretrainingTrainer(TaskTrainer):
    """The hooks of a language-model run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training stream, for the summary's resume note (the help of
        ``resume_interrupted`` carries the same clause).
    :cvar resume_stream_exact: true: a resume from a copy taken inside an epoch reads the batches the unkilled run reads next.
    """

    task = TASK
    success_metric = WATCH_METRIC
    resume_order = _RESUME_ORDER
    resume_stream_exact = True
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the corpus."

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The entropy of the selection targets (the unigram's ``val_loss``), set by ``build_data`` (a resume builds the data
        #: again, so it is there before the summary or the verdict).
        self._baseline: Optional[float] = None
        #: The batches of one selection pass, set by ``build_data``: the width of the bar's tie band follows it.
        self._select_batches: Optional[int] = None
        #: The epoch-zero ``val_loss`` (the untrained model's), read from the fit sidecar by ``evaluate_and_report`` and by
        #: nothing else.
        self._epoch_zero: Optional[float] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: LanguageModelPretrainingConfig) -> DataFacts:
        return text_facts(config.data)

    def build_data(self, config: LanguageModelPretrainingConfig, context: AutoContext) -> Any:
        import time

        from dl_techniques.utils.logger import logger
        from train.common import text_data, text_run

        data_cfg: TextData = config.data
        spec = tokenizer_spec(data_cfg.tokenizer)
        started = time.perf_counter()
        tokenizer = text_data.open_tokenizer(spec, data_cfg.tokenizer_dir)
        (train_texts, _), (test_texts, _) = text_data.corpus_documents(
            data_cfg.dataset, text_data.LM, max_samples=data_cfg.max_samples)
        train_docs = text_data.tokenize_documents(train_texts, tokenizer)
        test_docs = text_data.tokenize_documents(test_texts, tokenizer)
        logger.info(
            f"Read and tokenized {len(train_docs)} train and {len(test_docs)} test documents of {data_cfg.dataset} with "
            f"{spec.encoding} ({sum(len(doc) for doc in train_docs):,} and {sum(len(doc) for doc in test_docs):,} tokens, "
            f"{time.perf_counter() - started:.1f} s)")
        split = text_data.lm_split(
            train_docs, test_docs, spec=spec, seq_len=data_cfg.seq_len, batch_size=config.batch_size, seed=config.seed,
            validation_split=data_cfg.validation_split)
        data = text_data.lm_batches(
            split, spec=spec, seq_len=data_cfg.seq_len, batch_size=config.batch_size, seed=config.seed,
            validation_split=data_cfg.validation_split)
        self._baseline = text_run.unigram_val_loss(split.select[1], spec.vocab_size)
        self._select_batches = data.val_steps
        logger.info(
            f"Baseline (the unigram: the entropy of the selection targets, counted) on the {data.n_select} fixed selection "
            f"windows: val_loss {self._baseline:.6g}, perplexity {math.exp(self._baseline):.4g}")
        return data

    def input_check(self, facts: DataFacts, data: Any) -> None:
        check_holds_no_classes(self, facts, data)

    def build_model(self, config: LanguageModelPretrainingConfig, context: AutoContext, data: Any) -> Any:
        # The adapter returns a model BUILT at (None, seq_len), so the sequence's own `build` call is skipped. The vocabulary is
        # the tokenizer's, read off the data: it is a key the resolver refuses from `model_args`.
        kwargs = {**self._model_kwargs(config, context), "vocab_size": data.spec.vocab_size}
        return recipes.build_model(registry.MODELS[context.family], context.variant, None, data.input_shape, kwargs)

    @staticmethod
    def _model_kwargs(config: LanguageModelPretrainingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's (the resolver refuses the rest)."""
        return registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)

    def task_record(self, config: LanguageModelPretrainingConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the ``language_model_pretraining`` ``task.json`` record (schema in :mod:`train.api.predictor`).

        The length is ``data.input_shape`` (``build_model`` builds the model at it), the tokenizer the data's spec. Nothing of the model
        is needed, so a resume asks for it before ``build_model`` and compares it with the file (``DATA_KEYS``).
        """
        table = registry.MODELS[context.family].load_variants()
        return build_language_model_record(
            model=config.model, family=context.family, variant=context.variant, seq_len=data.input_shape[0],
            tokenizer=data.spec.record(),
            variant_overrides=overrides_of_variant(table[context.variant], self._model_kwargs(config, context)),
        )

    def used_arguments(self, config: LanguageModelPretrainingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: LanguageModelPretrainingConfig, context: AutoContext) -> None:
        kwargs = self._model_kwargs(config, context)
        # DECISION plan-2026-11-24T092348-7806a290/D-004
        # The tokenizer's files are checked HERE, before the claim, with paths and hashes only (no tensor, no tokenizer library):
        # a directory that is missing or holds a file of another hash must not cost a claimed `out`, and a run made with such a
        # file would match no record made with the pinned copy. Do NOT leave it to `build_data` (the claim is made by then) and do
        # NOT fall back to /tmp or the network when the pinned copy is gone.
        check_tokenizer_dir(config.data.tokenizer, config.data.tokenizer_dir)
        table = registry.MODELS[context.family].load_variants()[context.variant]
        limit = kwargs.get("max_seq_len", table["max_seq_len"])
        if is_count(limit) and config.data.seq_len > limit:  # a limit that is no count is the library's refusal, at the build
            raise ConfigError(
                f"Invalid value for 'data.seq_len': {config.data.seq_len!r} is above the {limit} rows of the position table of "
                f"{context.family}/{context.variant} (max_seq_len), so the model cannot be built at it; lower data.seq_len, or "
                "raise model_args.max_seq_len"
            )

    def init_identity(self, config: LanguageModelPretrainingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the vocabulary and the tokenizer's digest this run's record will hold, for an ``init`` source."""
        spec = tokenizer_spec(config.data.tokenizer)
        return {"vocab_size": spec.vocab_size, "tokenizer": spec.digest}

    def init_overlap(self, source: Any, config: LanguageModelPretrainingConfig, data: Any) -> Dict[str, Any]:
        """The source's train documents (``split_documents`` of its seed, split and cap, rebuilt) against this run's selection
        documents, counted; the test documents are the corpus's own, which no run trains on."""
        import numpy as np

        from train.common.text_data import split_documents

        from .. import init_from

        theirs, ours = source.config.get("data") or {}, config.data
        if theirs.get("dataset") != ours.dataset:
            return init_from.split_block("different_data", f"the source trained on {theirs.get('dataset')}, this run reads "
                                                           f"{ours.dataset}", {"data.dataset": theirs.get("dataset")}, None, (0, 0))
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
        n_source = documents_read(ours.dataset, keys["data.max_samples"], ours.max_samples,
                                  len(data.split.train_index) + len(select))
        if n_source is None:
            return init_from.split_block("unknown", f"this run caps data.max_samples at {ours.max_samples}, so how many "
                                                    f"{ours.dataset} documents the source (data.max_samples "
                                                    f"{keys['data.max_samples']}) read is not known: its split of them cannot be rebuilt")
        trained = split_documents(n_source, keys["seed"], keys["data.validation_split"])[0]
        return init_from.counted_split("documents", keys, np.intersect1d(trained, select).size, len(select), 0,
                                       data.split.n_test_docs)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: LanguageModelPretrainingConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager op
        # would initialise TensorFlow's devices before setup_gpu.
        return list(METRIC_NAMES)

    def build_loss_and_metrics(self, config: LanguageModelPretrainingConfig, data: Any) -> Tuple[Any, List[Any]]:
        return language_loss_and_metrics()

    def initial_loss_check(
        self, config: LanguageModelPretrainingConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over the unigram's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary

        if self._baseline is None:
            raise RuntimeError(
                "initial_loss_check compares with the unigram's loss, which build_data works out; it was asked for before the "
                "data was built"
            )
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        ratio = run_summary.initial_loss_ratio(metrics, self._baseline, self.initial_loss_hint)
        logger.info(
            f"Untrained evaluate BEFORE fit: val loss {metrics['loss']:.6g} on {data.n_select} selection windows, {ratio:.3g}x "
            f"the unigram's {self._baseline:.6g}"
        )
        # No verdict (`None`): no measurement says what a GPT-2's untrained loss should be against ln(vocab) or the unigram.
        return metrics, ratio, None

    def unjudged_note(self, config: LanguageModelPretrainingConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the loss of the untrained model over the unigram's "
            "(the entropy of the selection targets), kept as the baseline. No measurement says where a GPT-2's untrained loss "
            "should sit against that number or against ln(vocab_size), so no threshold on the ratio is judged; a non-finite "
            "initial loss is refused"
        )

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: LanguageModelPretrainingConfig, data: Any) -> Dict[str, Any]:
        # The train stream is INFINITE (a counter mapped through `batch_at`), so an epoch is `steps_per_epoch` batches by this
        # keyword (the sequence refuses an infinite stream without it); `validation_steps` is the selection pass, stated so that
        # Keras never has to guess a length.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}

    def resume_stream(self, config: LanguageModelPretrainingConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-24T092348-7806a290/D-003
        # A resumed `fit` starts a fresh iterator, which would replay the stream from batch 0. The stream is a function of
        # `(seed, step)`, so the replacement is the SAME stream beginning at `completed_steps`: the first batch of the resume is
        # the batch the uninterrupted run reads there. Do NOT return `data` (that is the replay), do NOT derive a new stream seed
        # (the pretrain rule for masks keyed on a counter: the first process consumed these documents in THIS order, another
        # order would repeat some and skip others) and do NOT reseed the split (it would leak selection documents into train).
        return data.with_start(completed_steps)

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the lowest) ``val_loss`` of the selection window."""
        start = self.selection_start_epoch(self.config)
        return float(min(hist[self.success_metric][start:]))

    def success_threshold(self, config: LanguageModelPretrainingConfig) -> float:
        """Return ``min(unigram, epoch-zero val_loss)``; ``evaluate_and_report`` has read the second term."""
        if self._baseline is None:
            raise RuntimeError(
                "success_threshold is cut from the baseline (the unigram's val_loss on the selection windows), which build_data "
                "works out; it was asked for before the data was built"
            )
        if self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the epoch-zero val_loss, which evaluate_and_report reads from the fit sidecar; it was "
                "asked for before the reporting"
            )
        return float(min(self._baseline, self._epoch_zero))

    def bar_is_met(self, config: LanguageModelPretrainingConfig, reached: float, threshold: float) -> bool:
        from train.common import text_run

        # DECISION plan-2026-11-24T092348-7806a290/D-020
        # STRICTLY below, in float32, beyond the tie band of the selection batch count (`text_run.strictly_below`, the repo's loss
        # bar): the base class's `>=` reads a loss backwards, and `<=` or a bare `<` would call a model that merely EQUALS the
        # unigram a success (a compiled constant model measured 14 float32 steps either side of the analytic number). Do NOT
        # compare the Python floats and do NOT shrink the band.
        return text_run.strictly_below(reached, threshold, self._select_batches or 1)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: LanguageModelPretrainingConfig) -> Tuple[str, ...]:
        return ("loss", "val_loss", "val_accuracy")

    def summary_fields(self, config: LanguageModelPretrainingConfig, data: Any) -> Dict[str, Any]:
        return {
            "seq_len": config.data.seq_len,
            "tokenizer": data.spec.record(),
            "documents": {
                "train": len(data.split.train_index), "select": len(data.split.select_index), "test": data.split.n_test_docs},
            "stream": {"seed": data.seed, "rule": _STREAM_RULE},
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return recipe_notices_of(family)

    def task_notes(
        self, config: LanguageModelPretrainingConfig, baseline: Dict[str, float], ratio: float, data: Any
    ) -> List[str]:
        """Return the language-model lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        zero = "the untrained model's" if config.init is None else "the loaded weights' (under `init`)"
        notes = [
            *([f"initial loss {baseline['loss']:.6g} is {ratio:.3g}x the unigram's val_loss ({self._baseline:.6g}) on the "
               f"{data.n_select} fixed selection windows: a plain evaluate in inference mode; see the `init_scale_warning` "
               "note"] if ratio is not None else []),
            *classification_run.RUN_NOTES[:2],
            "`val_loss` is the mean cross entropy over every position of the fixed selection windows (windows of seq_len + 1 "
            "tokens of the selection documents, an end-of-text id after each); it is a function of the weights alone (equal "
            "across passes, epochs and a resume), and its exponential is the perplexity per token (`perplexity`)",
            "`success_value` is the lowest `val_loss` of the selection window; `success_threshold` is min(`baseline.val_loss`, "
            "`epoch_zero.val_loss`): the unigram's loss (the entropy of the selection targets: the best a model that ignores its "
            f"context can do, counted and never evaluated) and {zero}, read from the fit sidecar; the bar is "
            "STRICTLY below, compared in float32 beyond `train.common.forecast_run.tie_band` float32 steps",
            "`test_metrics_best` is the reloaded best_model.keras and `test_metrics_final` the last epoch's weights, on the test "
            "windows (the corpus's own test documents), which never influenced selection",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock analyses "
            "read class probabilities, which a next-token distribution over the vocabulary is not"
            if analysis_cfg.model_analysis else "analysis_args.model_analysis is false: `analyzer.status` is 'skipped'"
        )
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null"
            )
        return notes

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self,
        config: LanguageModelPretrainingConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # DECISION plan-2026-11-24T092348-7806a290/D-010
        # The stock analysis reads the output as class probabilities (calibration, information flow, training dynamics) and the
        # target as labels; a next-token distribution over a 50,261-wide vocabulary is neither, and a rank-3 output once aborted
        # a whole analysis. Do NOT fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: LanguageModelPretrainingConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the language-model reporting sequence; see the module contract."""
        from types import SimpleNamespace

        import numpy as np

        from train.common import classification_run, pretrain_run, run_summary
        from train.common.resume_state import FIT_STATE_NAME

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: LanguageMetricArgs = config.metric_args
        # The epoch-zero term of the bar, from the sidecar: see `epoch_zero_val_loss` (a resume reads the same number).
        self._epoch_zero = pretrain_run.epoch_zero_val_loss(fit.run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME)
        baseline = self._baseline
        if baseline is None:
            raise RuntimeError("evaluate_and_report needs the baseline build_data works out; the data was not built")
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them). The test windows are fixed, so the
            # reloaded file reproduces them exactly.
            test_best, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model, fit.run_dir, data.test_ds, evaluate=classification_run.evaluate_dict, split_label="Test results")
        # A few fixed selection windows: the sample of the analysis and of the reload check of `final_model.keras`.
        x, y = data.split.select
        sample_x = x[:LOAD_CHECK_ROWS].astype(np.float32)
        # The best weights are analysed, then the last epoch's replace them (the sequence the denoising task shares).
        analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
            fit.model,
            SimpleNamespace(x_fig=sample_x, y_fig=y[:LOAD_CHECK_ROWS]),
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            run_analysis=functools.partial(self.run_analysis, config),
            model_analysis=analysis_cfg.model_analysis,
            eval_batch_size=analysis_cfg.eval_batch_size,
            last_weights=fit.last_weights,
            data=data,
            epochs_run=fit.epochs_run,
            save_last=checkpoint_cfg.save_last,
            load_sample=sample_x,
            predict_batch_size=data.batch_size,
        )
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        start = self.selection_start_epoch(config)
        best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
        extras: Dict[str, Any] = {
            "baseline": {"kind": "unigram", "val_loss": baseline},
            "epoch_zero": {"val_loss": self._epoch_zero, "binds": bool(self._epoch_zero < baseline)},
            "selection": selection_block(start, fit.epochs_run, best),
            "perplexity": {
                "unigram": math.exp(baseline),
                "epoch_zero": math.exp(self._epoch_zero),
                "best_val": math.exp(hist[self.success_metric][best - 1]),
                "test_best": None if test_best is None else math.exp(test_best["loss"]),
                "test_final": None if test_final is None else math.exp(test_final["loss"]),
            },
        }
        # DECISION plan-2026-11-19T081214-5c6cc105/D-012
        # No `loaded=`: `Evaluation.loaded` is higher-is-better (`init_from.beats_loaded`) and would judge this loss backwards.
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail,
            extras=extras,
        )


def train_language_model_pretraining(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[TextData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    init: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[TextTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[LanguageMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a causal language model on a corpus of documents; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_text_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        init=init, model_args=model_args,
        groups={
            "optim_args": optim_args,
            "transform_args": transform_args,
            "metric_args": metric_args,
            "checkpoint_args": checkpoint_args,
            "logger_args": logger_args,
            "analysis_args": analysis_args,
            "debug_args": debug_args,
        },
    )
    config = validate(LanguageModelPretrainingConfig, given)
    return LanguageModelPretrainingTrainer(config).run()
