"""The image-classification task: a config, a trainer and the keyword-only function.

``train_image_classification(...)`` is the function a Python caller uses and the
function ``python -m train image_classification key=value ...`` calls (D-008), so
the two cannot disagree. It validates its keywords into an
:class:`ImageClassificationConfig` and hands that to
:class:`ImageClassificationTrainer`, a :class:`~train.api.trainer.TaskTrainer`
that supplies the hooks of this task and nothing else: the sequence is the
shared one. The hooks are the classification assumptions of the run: its data and model, the
``val_accuracy`` success value and bar, the loss against ``ln(C)`` at epoch 0, the epoch line,
the summary fields and notes, the files a decode failure is searched in, the recipe notices,
and the reporting after ``fit`` (conformal sets, figures, the analyzer).

Interface contract
------------------
``ImageClassificationConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the
    function. ``out``, ``model`` and ``data`` are required. The seven groups of
    :mod:`train.api.args` are fields of the same names.

``ImageClassificationTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "image_classification"``.
    ``describe_data`` reads the dataset table, or LISTS the files of a folder or csv
    (no image is read); ``build_data`` goes through ``build_augment`` and
    :func:`~train.api.data_args.load_image_data`;
    ``build_model`` through the family's resolver and
    :func:`~train.api.recipes.build_model`; the loss is
    ``SparseCategoricalCrossentropy(from_logits=True)`` and the metrics are
    ``accuracy`` and, when there are more classes than ``metric_args.top_k``,
    ``top{k}_accuracy``; ``task_record`` is the ``task.json`` that
    :func:`~train.api.predictor.load_model` reads.

``train_image_classification(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"family/variant"``, e.g. ``"vit/vit_pico"``.
    :param data: a mapping (``{"dataset": "cifar10"}``), a
        :class:`~train.api.data_args.BuiltinImageData`,
        :class:`~train.api.data_args.FolderImageData` or
        :class:`~train.api.data_args.CsvImageData` (a mapping's ``format`` says which;
        ``builtin`` when absent), or the path of a YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices``: ``"auto"`` takes the family's
        recipe (``devices``: ``None``, every visible GPU).
    :param seed: seeds Python, numpy and Keras, the split and the augmentation;
        an int in ``[0, 2**32 - 1]``.
    :param precision: ``"float32"``, ``"mixed_float16"`` or ``"mixed_bfloat16"``.
    :param resume_interrupted: continue a killed run of the same ``out`` with the same
        request (only ``resume_interrupted``, ``out``, ``devices`` and ``analysis_args.eval_batch_size``
        may differ). Rows, the
        best checkpoint, early-stopping and ``ReduceLROnPlateau`` state, the learning rate
        and the history are kept. Limits: the training data order replays the order of
        epoch 0; an epoch killed midway is trained again from its start, or, when
        ``checkpoint_args.backup_every_steps`` is set, continued from the last backup taken
        inside it: only the batches that backup had not taken run, from a fresh pass of the data
        (some examples of that epoch repeat and some are missed), and the optimizer's step count
        and learning rate are the unkilled run's; a kill between the
        best checkpoint's save and the backup can leave ``best_model.keras`` one epoch LATER
        than the resumed state's best (the epoch that is then trained again; never an earlier
        one). A run with no epoch left to train and no ``results_summary.json`` (its
        reporting died, or the process died between the last epoch and ``fit`` returning:
        ``fit_finished``, or every epoch ended, or ``EarlyStopping`` had decided to stop) is not
        trained again: only the reporting runs, from ``best_model.keras``,
        ``checkpoints/backup/latest.weights.h5`` and ``training_log.csv`` (which is not
        written); it is refused if the last epoch's weights cannot be shown to be the backup's
        latest epoch-end copy. ``analysis_args.eval_batch_size`` may also differ from the earlier
        request. A finished run (with a summary), one with ``save_best`` false, one
        whose data (class names, statistics, split sizes) is no longer the data it started
        with, one a live process holds (``run.lock``) and one whose backup has no sound copy
        are refused (exit 2).
    :param checkpoint: a ``.keras`` model (a ``str`` or a ``Path``) whose layers
        of the same names give their weights to the new model.
    :param model_args: keywords of the family factory; they win over the recipe's.
    :param optim_args: / ``transform_args`` / ``metric_args`` /
        ``checkpoint_args`` / ``logger_args`` / ``analysis_args`` /
        ``debug_args``: a mapping or the group's class; ``None`` is the defaults.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see
        :meth:`~train.api.trainer.TaskTrainer.run`.
    :raises RuntimeError: for a diverged or non-finite run.

    The function registers the built-in model families itself: the registry is
    empty at import (D-040), and a Python caller never went through
    ``python -m train``.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012):
``python -m train image_classification help`` imports this module to print the
config's keys and must stay free of TensorFlow. Every ``keras`` and
``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .. import recipes, registry
from ..args import (
    AnalysisArgs,
    AutoContext,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
    MetricArgs,
    OptimArgs,
    TransformArgs,
)
from ..config import (
    AUTO,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Precision,
    Auto,
    StrictConfig,
    resume_help,
    run_request,
    validate,
)
from ..data_args import (
    DataFacts,
    ImageData,
    image_facts,
    load_data_config,
    load_image_data,
    read_data_file,
)
from ..predictor import build_task_record, overrides_of_variant
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, config_problems

#: The side (px) the cliffordnet batches of D-036 were measured at: what ImageNet and Imagenette run at.
LARGE_IMAGE_SIDE = 224

# ---------------------------------------------------------------------

TASK = recipes.TASK

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageClassificationConfig(StrictConfig):
    """Everything a classification run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": "family/variant, e.g. vit/vit_pico; python -m train list_models lists them"}
    )
    data: ImageData = dataclasses.field(
        metadata={
            "help": "the data and its split options: data.dataset=cifar10, or "
            "data.format=folder data.train_dir=..., or data.format=csv data.train_csv=...; "
            "or the path of a YAML file holding them (data=data.yaml), over which dotted keys win "
            "(data=data.yaml data.max_samples=100)",
            PATH_FORM: read_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(
        default=AUTO, metadata={"help": RUN_HELP["devices"]}
    )
    precision: Precision = dataclasses.field(
        default="float32", metadata={"help": RUN_HELP["precision"]}
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an ImageNet epoch is hours")},
    )
    checkpoint: Optional[str] = dataclasses.field(
        default=None, metadata={"help": RUN_HELP["checkpoint"]}
    )
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"help": RUN_HELP["model_args"]}
    )
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: TransformArgs = dataclasses.field(default_factory=TransformArgs)
    metric_args: MetricArgs = dataclasses.field(default_factory=MetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range; see :func:`~train.api.trainer.config_problems`."""
        return config_problems(self, prefix)


class ImageClassificationTrainer(TaskTrainer):
    """The hooks of an image-classification run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    """

    task = TASK
    success_metric = "val_accuracy"
    initial_loss_hint: ClassVar[str] = (
        "Lower the dropout or drop-path rates in model_args, or check the model "
        "and the data settings."
    )

    def describe_data(self, config: ImageClassificationConfig) -> DataFacts:
        transform_cfg: TransformArgs = config.transform_args
        size = transform_cfg.image_size
        return image_facts(config.data, None if size == AUTO else size)

    def build_data(self, config: ImageClassificationConfig, context: AutoContext) -> Any:
        from train.common.image_augment import build_augment

        transform_cfg: TransformArgs = config.transform_args
        data_cfg: ImageData = config.data

        def hook(fill: Any) -> Any:
            # DECISION plan-2026-10-30T132432-a865fa08/D-049: random erasing paints the
            # dataset's per-channel MEAN, as ``train_cliffordnet.py`` does. Do NOT leave
            # ``fill`` at ``build_op``'s default of zero: it erases to black on the [0, 1]
            # image, which is -1.99 after standardize, where the legacy trainer erases to 0.
            return build_augment(
                transform_cfg.augment,
                random_erasing_prob=transform_cfg.random_erasing_prob,
                fill=fill,
            )

        augment, from_mean = None, None
        if data_cfg.format == "builtin":
            from dl_techniques.datasets.vision.classification import get_spec

            augment = hook(get_spec(data_cfg.dataset).mean)
        elif transform_cfg.random_erasing_prob:
            # Folder and csv data have no published statistics: the loader computes the
            # mean from the fit split and builds the hook with it (D-049 for file data).
            from_mean = hook
        else:
            augment = hook((0.0,))
        return load_image_data(
            data_cfg,
            batch_size=config.batch_size,
            seed=config.seed,
            augment=augment,
            augment_from_mean=from_mean,
            normalize=transform_cfg.normalize,
            image_size=transform_cfg.image_size,
            loader=context.defaults.get("loader"),
            eval_batch_size=config.analysis_args.eval_batch_size,
        )

    def build_model(
        self, config: ImageClassificationConfig, context: AutoContext, data: Any
    ) -> Any:
        spec = registry.MODELS[context.family]
        return recipes.build_model(
            spec, context.variant, data.num_classes, data.input_shape,
            self._model_kwargs(config, context),
        )

    @staticmethod
    def _model_kwargs(config: ImageClassificationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family factory's keywords: the recipe's, then the caller's."""
        spec = registry.MODELS[context.family]
        return spec.load_resolver()(context, config.optim_args, config.model_args)

    def used_arguments(
        self, config: ImageClassificationConfig, context: AutoContext
    ) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json`` (D-050).

        :return: ``{"model_args": ..., "loader": ...}``: the factory keywords the
            resolver produced (a regularizer as its serialized config) and the
            recipe's loader mapping.
        """
        return {
            "model_args": recipes.serialize_model_args(self._model_kwargs(config, context)),
            "loader": dict(context.defaults.get("loader") or {}),
        }

    def task_record(
        self, config: ImageClassificationConfig, context: AutoContext, data: Any
    ) -> Dict[str, Any]:
        """Return the ``task.json`` record (schema in :mod:`train.api.predictor`).

        ``mean`` and ``std`` are the loader's own, so the record holds what the training
        inputs were standardized with: a built-in dataset's published constants, or the
        statistics of the fit split of a folder or csv run (D-060). Zeros and ones when
        ``transform_args.normalize`` is ``unit``.
        """
        transform_cfg: TransformArgs = config.transform_args
        data_cfg: ImageData = config.data
        source_channels = 3
        data_kind = data_cfg.format
        if data_cfg.format == "builtin":
            from dl_techniques.datasets.vision.classification import get_spec

            data_kind = data_cfg.dataset
            source_channels = get_spec(data_cfg.dataset).input_shape[-1]
        return build_task_record(
            task=self.task,
            model=config.model,
            family=context.family,
            variant=context.variant,
            class_names=data.class_names,
            input_shape=data.input_shape,
            normalize=transform_cfg.normalize,
            mean=data.mean,
            std=data.std,
            data_kind=data_kind,
            source_channels=source_channels,
            variant_overrides=overrides_of_variant(
                registry.MODELS[context.family].load_variants()[context.variant],
                self._model_kwargs(config, context),
            ),
        )

    def check_model_args(self, config: ImageClassificationConfig, context: AutoContext) -> None:
        # Run the family's resolver before the claim: it refuses a model_args key the library
        # supplies, a double weight decay and a fixed variant key (a ViT's ``scale``), which
        # used to surface only after the run directory was claimed.
        self._model_kwargs(config, context)

    def metric_names(self, config: ImageClassificationConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only. Do NOT build the
        # metrics to read ``.name``: a Keras metric allocates variables, the first eager
        # op initialises TensorFlow's devices, and ``setup_gpu`` (called after this
        # pre-claim check) then fails with "Physical devices cannot be modified".
        from train.common.classification_run import classification_metric_names

        metric_cfg: MetricArgs = config.metric_args
        return classification_metric_names(facts.num_classes, metric_cfg.top_k)

    def build_loss_and_metrics(
        self, config: ImageClassificationConfig, data: Any
    ) -> Tuple[Any, Any]:
        import keras

        from train.common.classification_run import classification_metrics

        metric_cfg: MetricArgs = config.metric_args
        return (
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            classification_metrics(
                data.num_classes, metric_cfg.top_k, half_precision=config.precision != "float32"
            ),
        )

    def input_check(self, facts: DataFacts, data: Any) -> None:
        if (
            data.num_classes != facts.num_classes
            or data.input_shape[0] != facts.image_size
        ):
            raise RuntimeError(
                f"{type(self).__name__}.describe_data said {facts.num_classes} classes "
                f"at {facts.image_size} px, but build_data loaded {data.num_classes} "
                f"classes with input shape {tuple(data.input_shape)}"
            )

    def initial_loss_check(
        self, config: ImageClassificationConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over ``ln(num_classes)``."""
        from train.common import classification_run

        return classification_run.check_initial_loss(
            model,
            data.select_ds,
            data.n_select,
            data.num_classes,
            evaluate=classification_run.evaluate_dict,
            hint=self.initial_loss_hint,
        )

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        return max(hist[self.success_metric])

    def success_threshold(self, config: ImageClassificationConfig) -> float:
        metric_cfg: MetricArgs = config.metric_args
        return metric_cfg.success_threshold

    def epoch_line_metrics(self, config: ImageClassificationConfig) -> Tuple[str, ...]:
        metric_cfg: MetricArgs = config.metric_args
        top_k = metric_cfg.top_k
        return (
            "loss",
            "accuracy",
            f"top{top_k}_accuracy",
            "val_loss",
            "val_accuracy",
            f"val_top{top_k}_accuracy",
        )

    def summary_fields(self, config: ImageClassificationConfig, data: Any) -> Dict[str, Any]:
        return {
            "figure_samples": int(len(data.x_fig)),
            "input_normalization": {"mean": data.mean, "std": data.std},
        }

    def files_to_scan(self, data: Any) -> Optional[Sequence[str]]:
        # A built-in dataset carries no ``image_files``: nothing to scan.
        return getattr(data, "image_files", None)

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        notices: List[Tuple[str, str]] = []
        if facts.dataset in registry.FILE_DATASETS:
            notices.append(
                (
                    "info",
                    f"{facts.dataset} data: no legacy trainer reads files, so the {family} "
                    "recipe is its base recipe with transform_args.augment=flip (D-060) and the "
                    "statistics come from one pass over the fit split",
                )
            )
            # DECISION plan-2026-10-31T180033-2faf890a/D-068
            # A WARNING, not a changed default. Do NOT copy the ImageNet overlay's batch to folder
            # and csv data: the review could not reproduce a death (nano finished at 256 after a
            # 12.27 GiB allocation failure), the values were measured on one card with ImageNet-shaped
            # data, and a recipe keyed by image size is a design change. The numbers are read from
            # the registry, never typed here.
            measured = registry.MODELS[family].defaults.get("per_dataset", {}).get("imagenet", {})
            if "batch_size" in measured and facts.image_size >= LARGE_IMAGE_SIDE:
                from train.common.classification_data import EVAL_BATCH_SIZE

                notices.append(
                    (
                        "warning",
                        f"{family} on {facts.dataset} data of {facts.image_size} px: the batch of "
                        f"{measured['batch_size']} (batch_size) and the evaluation batch "
                        "(analysis_args.eval_batch_size) that fit a 24 GB card at 224 px were "
                        "measured on imagenet and imagenette and are keyed to those datasets, so "
                        "unless you set them this run trains at the base recipe's batch and "
                        f"evaluates at {EVAL_BATCH_SIZE} (cliffordnet/nano at 224 px and "
                        f"{EVAL_BATCH_SIZE}: the allocator logged an out-of-memory request of "
                        "12.27 GiB for the figures and recovered); set both, or use "
                        "transform_args.image_size=128"
                    )
                )
        elif facts.dataset not in registry.MODELS[family].defaults.get("per_dataset", {}):
            notices.append(
                (
                    "warning",
                    f"No recipe overlay for {family} on {facts.dataset}: the family's base "
                    "recipe applies unchanged, with no dataset-specific augmentation, loader "
                    "or model arguments. Check transform_args, data and model_args.",
                )
            )
        return notices

    def task_notes(
        self,
        config: ImageClassificationConfig,
        baseline: Dict[str, float],
        ratio: float,
        data: Any,
    ) -> List[str]:
        """Return the classification lines of ``results_summary.json``'s ``notes``."""
        import numpy as np

        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        notes = [
            f"initial loss {baseline['loss']:.4f} vs ln(C)={np.log(data.num_classes):.4f} "
            f"(ratio {ratio:.2f}), a plain evaluate on the validation split",
            *classification_run.RUN_NOTES,
        ]
        if data.test_ds is None:
            notes.append(
                "the data has no test split: `test_metrics_best`, `test_metrics_final`, "
                "`conformal` and `ece` are null, no figure was written and the model analysis "
                "was skipped; the best_model.keras reload check ran on the selection split "
                "(`best_checkpoint_max_abs_diff`)"
            )
        if not checkpoint_cfg.save_best:
            notes.append(
                "checkpoint_args.save_best is false: no best_model.keras was written, so "
                "`test_metrics_best` is null; figures and analysis use the in-memory best weights"
            )
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and "
                "`model_loading_validated` is null"
            )
        if not analysis_cfg.figures:
            notes.append("analysis_args.figures is false: no figure was written and `ece` is null")
        if not analysis_cfg.model_analysis:
            notes.append("analysis_args.model_analysis is false: `analyzer.status` is 'skipped'")
        if analysis_cfg.conformal_alpha is None:
            notes.append("analysis_args.conformal_alpha is null: `conformal` is null")
        return notes

    def evaluate_and_report(
        self, config: ImageClassificationConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the classification reporting sequence; see the module contract."""
        from dl_techniques.utils.logger import logger
        from train.common import classification_run
        from train.common.conformal import run_conformal

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        # DECISION plan-2026-10-30T132432-a865fa08/D-060: data with no test split
        # (``test_ds`` is ``None``) skips the test metrics, the figures, conformal and the
        # model analysis, which all read test images, and the summary says so. Do NOT
        # substitute the selection split for the test split in any of them: it selected the
        # best epoch, so every number would be optimistic and still look like a test number.
        # The one thing kept is the reload check of ``best_model.keras``, run on the
        # selection split with its metrics discarded.
        has_test = data.test_ds is not None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them).
            reloaded, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model,
                fit.run_dir,
                data.test_ds if has_test else data.select_ds,
                evaluate=classification_run.evaluate_dict,
                split_label="Test results" if has_test else "Selection results (reload check only)",
            )
            test_best = reloaded if has_test else None
        conformal: Optional[Dict[str, Any]] = None
        if analysis_cfg.conformal_alpha is not None and not has_test:
            logger.warning(
                "analysis_args.conformal_alpha is set but the data has no test split to "
                "calibrate on: conformal prediction is skipped"
            )
        elif analysis_cfg.conformal_alpha is not None:
            conformal = run_conformal(
                fit.model,
                data.test_ds,
                alpha=analysis_cfg.conformal_alpha,
                seed=config.seed,
                out_dir=fit.run_dir,
                input_shape=data.input_shape,
                model_name=fit.experiment_name,
                weights=f"best weights, selected on {data.selection_split}",
                selection_overlaps_calibration=False,
            )
        visualizations, analysis = classification_run.report_best_weights(
            fit.model,
            data,
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            analysis_cfg.model_analysis and has_test,
            run_analysis=functools.partial(self.run_analysis, config),
            name_failed_analyses=True,
            figures=analysis_cfg.figures and has_test,
            analysis_off_hint=(
                "the data has no test split"
                if analysis_cfg.model_analysis
                else "analysis_args.model_analysis=false"
            ),
            eval_batch_size=analysis_cfg.eval_batch_size,
        )
        # The last epoch's weights replace the best ones here.
        test_final, load_check = classification_run.finalize_last_epoch(
            fit.model,
            fit.last_weights,
            data,
            fit.run_dir,
            fit.history,
            fit.epochs_run,
            evaluate=classification_run.evaluate_dict,
            save=checkpoint_cfg.save_last,
            atomic=True,
            eval_batch_size=analysis_cfg.eval_batch_size,
        )
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=conformal,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
        )


def train_image_classification(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[ImageData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[TransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[MetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train an image classifier; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        checkpoint=checkpoint, model_args=model_args,
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
    config = validate(ImageClassificationConfig, given)
    return ImageClassificationTrainer(config).run()
