"""The ``text_classification`` task: a config, a trainer and the keyword-only function.

``train_text_classification(...)`` is the function a Python caller uses and the function ``python -m train text_classification
key=value ...`` calls, so the two cannot disagree. It validates its keywords into a :class:`TextClassificationConfig` and hands that
to :class:`TextClassificationTrainer`, a :class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing
else (a task overrides hooks, never stages). Two families train it, ``bert`` and ``gpt2`` (:mod:`train.api.text_models`): the library
encoder or decoder with the sentiment head, a functional model of float32 ids to class logits, trained by the stock ``fit`` with the
stock sparse cross entropy.

Interface contract
------------------
``TextClassificationConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data`` are
    required. The groups are the language task's (``data`` is :class:`~train.api.text_args.TextData`, ``transform_args`` has no
    field) except ``metric_args`` (:class:`~train.api.text_args.ClassifierMetricArgs`: the checkpoint is selected by ``val_accuracy``
    and the bar has no knob); there is no loss group. ``analysis_args.conformal_alpha`` is refused, and so is a ``data.seq_len`` below
    2 (a row is ``[CLS]`` and a token).

``TextClassificationTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "text_classification"`` and ``success_metric = "val_accuracy"``.

    What a run is. The corpus is tokenized once into memory; the documents are split by a hash of ``(seed, document index)`` into
    train and selection documents (the SAME split the language task makes for the same seed) and the corpus's own test documents are
    kept for the end. A document is a row of ``seq_len`` ids, ``[CLS]``, the head of the document and pad, with its label; an epoch is
    ``n_train // batch_size`` steps of an INFINITE stream whose batch ``i`` is a pure function of ``(seed, i)`` (epoch ``e`` puts the
    train rows, each label with its row, in an order keyed on ``(seed, e)``). The selection and test rows are fixed, so
    ``val_accuracy`` is a function of the weights alone. A resume CONTINUES the stream at the settled step (``resume_stream``) and says
    so (``resume_stream_exact``).

    The bar (D-006, D-057). ``success_value`` is the highest ``val_accuracy`` of the selection split; ``success_threshold`` is the share
    of its majority class, counted from its labels in ``build_data`` (a floor: 0.5 on balanced reviews); the bar is STRICTLY above it,
    in float32 with no band (:func:`train.common.text_run.strictly_above`: a model that answers the majority class scores the share to
    the last bit), and under ``init`` also strictly above the loaded weights' epoch-zero accuracy (``Evaluation.loaded``). A constant
    predictor therefore misses it though it clears chance. Under ``debug_args.smoke`` no bar is applied.

    ``init``. A ``gpt2`` run starts from an earlier ``gpt2_lm`` run (the decoder copied path for path, the head fresh), a ``bert`` run
    from none (an encoder cannot start from a causal decoder); identity and split count are the language task's
    (:func:`~train.api.text_args.tokenizer_identity`, :func:`~train.api.text_args.documents_overlap`). The run writes ``task.json``
    before ``fit`` (:func:`~train.api.predictor.build_text_classification_record`, the eighth record kind), which holds the ``init`` block, so
    a classifier run is itself an ``init`` source of its own family and can be predicted with and exported.

``train_text_classification(*, out, model, data, ...)``
    :param model: ``"bert/variant"`` or ``"gpt2/variant"``, e.g. ``"gpt2/tiny"``; ``python -m train list_models
        task=text_classification`` lists them.
    :param data: a mapping (``{"dataset": "synthetic", "tokenizer": "ascii95"}``), a :class:`~train.api.text_args.TextData`, or the
        path of a YAML file holding one. The other keywords are those of
        :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A family of another
        task, a model key the family refuses, a tokenizer directory that is missing or holds other files, a ``seq_len`` above the
        position table, a TFDS directory without the reviews and an ``init`` source the family cannot take are refused before the run
        directory is claimed; a part that lacks a class, or fewer train rows than a batch, after the data is read.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train text_classification help`` imports this module to
print the config's keys and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple, Union

from .. import init_from, recipes, registry
from ..args import (
    BACKUP_EVERY_STEPS_HELP_EXACT, AnalysisArgs, AutoContext, CheckpointArgs, DebugArgs, LoggerArgs, OptimArgs)
from ..config import (
    AUTO,
    BACKUP_HELP_EXACT,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    join_path,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..predictor import build_text_classification_record, overrides_of_variant
from ..text_args import (
    BERT,
    CLASS_NAMES,
    CLASSIFICATION_TASK as TASK,
    CLASSIFIER_METRIC,
    GPT2_CLASSIFIER,
    ClassifierMetricArgs,
    TextData,
    TextTransformArgs,
    documents_overlap,
    load_text_data_config,
    read_text_data_file,
    text_facts,
    tokenizer_identity,
)
from ..text_models import METRIC_NAMES, check_text_window, language_loss_and_metrics, recipe_notices_of
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, config_problems
from .pretrain import selection_block

# ---------------------------------------------------------------------

# DECISION plan-2026-11-24T092348-7806a290/D-050
# This task's resume continues the SAME stream at the settled step, so it sets `resume_stream_exact` and gives the two helps their exact
# clause (`resume_help(backup=)`, `HELP_OVERRIDES`): the shared sentences say "a fresh pass of the data", which is false here. Do NOT drop
# one of the three (a test holds them equal) and do NOT reword the shared sentences for every task.
_RESUME_ORDER = (
    "the training stream continues at the step the killed run stopped at (epoch e is a permutation of the training rows keyed "
    "on (seed, e), so the resumed run reads the batches the uninterrupted run would have read next, never epoch 0's again and "
    "never a selection row), and the selection and test rows stay fixed"
)

_STREAM_RULE = (
    "the document split is keyed on (seed, document index) and never on the stream; train batch i is a function of (seed, step) "
    "alone: epoch e orders the train rows, each with its label, by sha256 keys of (seed, e, row); a resume continues the stream "
    "at the settled step and never derives another seed"
)

_CONFORMAL = (
    f"a conformal prediction set is the image classifier's calibrated set, which a {TASK} run does not produce; its output is the "
    "logits of the classes"
)

#: The help of the two ``data`` keys whose wording is about the language task's windows: a row of a classifier is another thing.
_HELP_OVERRIDES: Dict[str, str] = {
    "checkpoint_args.backup_every_steps": BACKUP_EVERY_STEPS_HELP_EXACT,
    "data.dataset": (
        "synthetic (an offline labelled bank: 60 percent class neg, each document with cue words of its class, so a model that "
        "reads the text beats the majority share) or imdb_reviews (the local TFDS reviews, 25,000 train and 25,000 test documents, "
        "labelled neg and pos); the train documents are split by a hash of (seed, document index) into the train documents and the "
        "documents that select the best epoch, and the test documents are scored once after fit"
    ),
    "data.seq_len": (
        "ids a row holds: [CLS], the first seq_len - 1 tokens of the document (a longer document is cut, and the summary counts "
        "them) and the pad id up to seq_len; at least 2 and at most the model's position table (refused by name before the run "
        "is claimed)"
    ),
}


@dataclasses.dataclass(frozen=True)
class TextClassificationConfig(StrictConfig):
    """Everything a text-classification run is told; see the module contract.

    :cvar HELP_OVERRIDES: dotted key to the help this task gives it in place of its group's (``cli._field_help_lines`` reads it).
    """

    HELP_OVERRIDES: ClassVar[Dict[str, str]] = _HELP_OVERRIDES

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": f"family/variant, e.g. {GPT2_CLASSIFIER}/tiny or {BERT}/tiny; python -m train list_models task={TASK} lists them"}
    )
    data: TextData = dataclasses.field(
        metadata={
            "help": "the corpus and the rows: data.dataset=synthetic (or imdb_reviews), with data.tokenizer, data.seq_len, "
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
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict, metadata={"help": RUN_HELP["model_args"]})
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: TextTransformArgs = dataclasses.field(default_factory=TextTransformArgs)
    metric_args: ClassifierMetricArgs = dataclasses.field(default_factory=ClassifierMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, a conformal level and a row too short for ``[CLS]`` and a token."""
        lines = config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL)
        if self.data.seq_len < 2:
            lines.append(
                f"Invalid value for {join_path(join_path(prefix, 'data'), 'seq_len')!r}: a row is [CLS] and at least one token, "
                f"expected an int of at least 2, got {self.data.seq_len!r}"
            )
        return lines


class TextClassificationTrainer(TaskTrainer):
    """The hooks of a text-classification run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training stream, for the summary's resume note (the help of
        ``resume_interrupted`` carries the same clause).
    :cvar resume_stream_exact: true: a resume from a copy taken inside an epoch reads the batches the unkilled run reads next.
    """

    task = TASK
    success_metric = CLASSIFIER_METRIC
    resume_order = _RESUME_ORDER
    resume_stream_exact = True
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the corpus."

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The share of the majority class of the selection split (the bar), set by ``build_data`` (a resume builds the data
        #: again, so it is there before the summary or the verdict).
        self._majority: Optional[float] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: TextClassificationConfig) -> DataFacts:
        return text_facts(config.data, len(CLASS_NAMES))

    def build_data(self, config: TextClassificationConfig, context: AutoContext) -> Any:
        from dl_techniques.utils.logger import logger
        from train.common import text_data, text_run

        data_cfg: TextData = config.data
        spec, (train_docs, train_labels), (test_docs, test_labels) = text_data.tokenized_corpus(data_cfg, text_data.CLASSIFICATION)
        split = text_data.class_split(
            train_docs, train_labels, test_docs, test_labels, spec=spec, seq_len=data_cfg.seq_len, batch_size=config.batch_size,
            seed=config.seed, validation_split=data_cfg.validation_split, num_classes=len(CLASS_NAMES))
        data = text_data.class_batches(
            split, spec=spec, seq_len=data_cfg.seq_len, batch_size=config.batch_size, seed=config.seed,
            validation_split=data_cfg.validation_split, num_classes=len(CLASS_NAMES))
        # DECISION plan-2026-11-24T092348-7806a290/D-057
        # The bar is cut from the SELECTION split's labels, the rows the best epoch is chosen on. Do NOT count the train rows' share
        # (another number: the split is by document) and do NOT use 1 / num_classes or twice chance (a constant predictor clears
        # both: it scores the majority share).
        self._majority = text_run.majority_share(split.select[1])
        logger.info(
            f"Baseline (the majority class of the {data.n_select} fixed selection rows, counted): val_accuracy {self._majority:.6g}; "
            f"a model that answers one class to every row scores it, and the bar is strictly above it")
        return data

    def task_record(self, config: TextClassificationConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the ``text_classification`` ``task.json`` record (schema in :mod:`train.api.predictor`).

        The length is ``data.input_shape`` (``build_model`` builds the model at it), the tokenizer the data's spec and the classes the
        corpus's own. Nothing of the model is needed, so a resume asks for it before ``build_model`` and compares it with the file.
        """
        table = registry.MODELS[context.family].load_variants()
        return build_text_classification_record(
            model=config.model, family=context.family, variant=context.variant, seq_len=data.input_shape[0],
            tokenizer=data.spec.record(), class_names=CLASS_NAMES,
            variant_overrides=overrides_of_variant(table[context.variant], self._model_kwargs(config, context)),
        )

    def input_check(self, facts: DataFacts, data: Any) -> None:
        if data.num_classes != facts.num_classes:
            raise RuntimeError(
                f"{type(self).__name__}.describe_data said {facts.num_classes} classes, but build_data loaded {data.num_classes}")

    def build_model(self, config: TextClassificationConfig, context: AutoContext, data: Any) -> Any:
        # The adapter returns a model BUILT at (None, seq_len), so the sequence's own `build` call is skipped. The vocabulary and
        # the two ids are the tokenizer's, read off the data: the resolver refuses them in `model_args`.
        spec = data.spec
        kwargs = {**self._model_kwargs(config, context), "vocab_size": spec.vocab_size, "pad_id": spec.pad_id, "cls_id": spec.cls_id}
        return recipes.build_model(registry.MODELS[context.family], context.variant, data.num_classes, data.input_shape, kwargs)

    @staticmethod
    def _model_kwargs(config: TextClassificationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's (the resolver refuses the rest)."""
        return registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)

    def used_arguments(self, config: TextClassificationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: TextClassificationConfig, context: AutoContext) -> None:
        check_text_window(config.data, context.family, context.variant, self._model_kwargs(config, context))

    def init_identity(self, config: TextClassificationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the vocabulary and the tokenizer's digest this run's record will hold, for an ``init`` source."""
        return tokenizer_identity(config.data)

    def init_overlap(self, source: Any, config: TextClassificationConfig, data: Any) -> Dict[str, Any]:
        """The source's train documents among this run's selection documents, counted (:func:`documents_overlap`)."""
        return documents_overlap(source, config, data, self.task)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: TextClassificationConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager op
        # would initialise TensorFlow's devices before setup_gpu.
        return list(METRIC_NAMES)

    def build_loss_and_metrics(self, config: TextClassificationConfig, data: Any) -> Tuple[Any, List[Any]]:
        return language_loss_and_metrics()

    def initial_loss_check(
        self, config: TextClassificationConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over ``ln(num_classes)``."""
        from train.common import classification_run

        return classification_run.check_initial_loss(
            model, data.select_ds, data.n_select, data.num_classes, evaluate=classification_run.evaluate_dict,
            hint=self.initial_loss_hint)

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: TextClassificationConfig, data: Any) -> Dict[str, Any]:
        # The train stream is INFINITE (a counter mapped through `batch_at`), so an epoch is `steps_per_epoch` batches by this
        # keyword; `validation_steps` is the selection pass, stated so that Keras never has to guess a length.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}

    def resume_stream(self, config: TextClassificationConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-24T092348-7806a290/D-003
        # The SAME stream beginning at `completed_steps`, so the first batch of the resume is the batch the uninterrupted run reads
        # there. Do NOT return `data` (that replays epoch 0's order), do NOT derive a new stream seed (the first process consumed
        # these rows in THIS order) and do NOT reseed the split (it would leak selection documents into train).
        return data.with_start(completed_steps)

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the highest) ``val_accuracy`` of the selection split."""
        return float(max(hist[self.success_metric][self.selection_start_epoch(self.config):]))

    def success_threshold(self, config: TextClassificationConfig) -> float:
        """Return the majority-class share of the selection split, which ``build_data`` counted."""
        if self._majority is None:
            raise RuntimeError(
                "success_threshold is the majority-class share of the selection split, which build_data works out; it was asked "
                "for before the data was built"
            )
        return self._majority

    def bar_is_met(self, config: TextClassificationConfig, reached: float, threshold: float, loaded: Optional[float] = None) -> bool:
        from train.common import text_run

        # DECISION plan-2026-11-24T092348-7806a290/D-057
        # STRICTLY above, in float32 and with no band (`text_run.strictly_above`): a model that answers the majority class scores
        # the share to the last bit, so the base class's `>=` calls a constant predictor a success. Under `init` the success value
        # must also beat the loaded weights' epoch zero (`init_from.beats_loaded`). Do NOT write `>=` and do NOT compare the Python
        # floats.
        return text_run.strictly_above(reached, threshold) and init_from.beats_loaded(reached, loaded, self.success_metric)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: TextClassificationConfig) -> Tuple[str, ...]:
        return ("loss", "accuracy", "val_loss", "val_accuracy")

    def summary_fields(self, config: TextClassificationConfig, data: Any) -> Dict[str, Any]:
        import numpy as np

        split = data.split
        counts = {part: np.bincount(labels, minlength=data.num_classes).tolist()
                  for part, labels in (("train", split.train_labels), ("select", split.select[1]), ("test", split.test[1]))}
        return {
            "seq_len": config.data.seq_len,
            "tokenizer": data.spec.record(),
            "class_names": list(CLASS_NAMES),
            "class_counts": counts,
            "documents": {part: sum(row) for part, row in counts.items()},
            "truncated_documents": split.truncated,
            "stream": {"seed": data.seed, "rule": _STREAM_RULE},
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return recipe_notices_of(family)

    def task_notes(
        self, config: TextClassificationConfig, baseline: Dict[str, float], ratio: float, data: Any
    ) -> List[str]:
        """Return the text-classification lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        notes = [
            *([f"initial loss {baseline['loss']:.4f} vs ln(C)={math.log(data.num_classes):.4f} (ratio {ratio:.2f}), a plain "
               "evaluate on the selection rows"] if ratio is not None else []),
            *classification_run.RUN_NOTES,
            "`val_accuracy` is the accuracy over the fixed selection rows (one row per selection document: [CLS], the head of the "
            "document, pad), a function of the weights alone; `success_value` is its highest value over the epochs and "
            "`success_threshold` is the share of the majority class of those rows (`baseline.val_accuracy`): what a model that "
            "answers one class to every row scores. The bar is STRICTLY above it, compared in float32 with no band, so a constant "
            "predictor misses it, and under `init` also strictly above the loaded weights' epoch-zero accuracy "
            "(`loaded_success_value`)",
            "the bar is a floor, not a target: on balanced reviews it is 0.5, and the accuracies it lets through are the result to read "
            "(`class_counts`, `test_metrics_best`)",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock analyses read "
            "images"
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
        config: TextClassificationConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # DECISION plan-2026-11-24T092348-7806a290/D-010
        # The stock analysis reads image-shaped inputs and calibrates on the test images (a fresh model once aborted a whole
        # analysis); the rows here are token ids. Do NOT fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: TextClassificationConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the text-classification reporting sequence; see the module contract."""
        import numpy as np

        from train.common import text_run

        if self._majority is None:
            raise RuntimeError("evaluate_and_report needs the baseline build_data works out; the data was not built")
        start = self.selection_start_epoch(config)
        fields, _, best = text_run.report_run(config, data, fit, functools.partial(self.run_analysis, config), start)
        extras: Dict[str, Any] = {
            "baseline": {"kind": "majority", "class": CLASS_NAMES[int(np.bincount(data.split.select[1]).argmax())],
                         "val_accuracy": self._majority},
            "selection": selection_block(start, fit.epochs_run, best),
        }
        # Under `init` the bar must also beat the loaded weights' epoch-zero accuracy, read from the sidecar (None without init).
        return Evaluation(**fields, loaded=init_from.loaded_value(config, fit.run_dir, "accuracy"), extras=extras)


def train_text_classification(
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
    metric_args: Union[ClassifierMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a text classifier on a labelled corpus; see the module contract."""
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
    config = validate(TextClassificationConfig, given)
    return TextClassificationTrainer(config).run()
