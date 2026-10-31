"""The image-classification task: a config, a trainer and the keyword-only function.

``train_image_classification(...)`` is the function a Python caller uses and the
function ``python -m train image_classification key=value ...`` calls (D-008), so
the two cannot disagree. It validates its keywords into an
:class:`ImageClassificationConfig` and hands that to
:class:`ImageClassificationTrainer`, a :class:`~train.api.trainer.TaskTrainer`
that supplies the four hooks of this task and nothing else: the sequence is the
shared one.

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
        request (only ``resume_interrupted``, ``out`` and ``devices`` may differ). Rows, the
        best checkpoint, early-stopping and ``ReduceLROnPlateau`` state, the learning rate
        and the history are kept. Limits: the training data order replays the order of
        epoch 0; an epoch killed midway is trained again from its start; a kill between the
        best checkpoint's save and the backup can leave ``best_model.keras`` one epoch LATER
        than the resumed state's best (the epoch that is then trained again; never an earlier
        one). A finished run, a run with no epoch left, one with ``save_best`` false, one
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
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

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
from ..config import AUTO, DEFAULT_SEED, PATH_FORM, Auto, StrictConfig, validate
from ..data_args import (
    DataFacts,
    ImageData,
    image_facts,
    load_data_config,
    load_image_data,
    read_data_file,
)
from ..predictor import build_task_record
from ..trainer import TaskTrainer, TrainResult, config_problems

# ---------------------------------------------------------------------

TASK = recipes.TASK

#: The precision policies a run can ask for.
PRECISIONS = ("float32", "mixed_float16", "mixed_bfloat16")

_Precision = Literal[PRECISIONS]  # type: ignore[valid-type]

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageClassificationConfig(StrictConfig):
    """Everything a classification run is told; see the module contract."""

    out: str = dataclasses.field(
        metadata={"help": "run directory (a relative path starts at the repo root); auto names one under results/"}
    )
    model: str = dataclasses.field(
        metadata={"help": "family/variant, e.g. vit/vit_pico; python -m train list_models lists them"}
    )
    data: ImageData = dataclasses.field(
        metadata={
            "help": "the data and its split options: data.dataset=cifar10, or "
            "data.format=folder data.train_dir=..., or data.format=csv data.train_csv=...; "
            "or the path of a YAML file holding them (data=data.yaml)",
            PATH_FORM: read_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(
        default=AUTO, metadata={"help": "epochs to train; auto takes the family's"}
    )
    batch_size: Auto[int] = dataclasses.field(
        default=AUTO, metadata={"help": "batch size; auto takes the family's"}
    )
    seed: int = dataclasses.field(
        default=DEFAULT_SEED, metadata={"help": "seed of the weights, the split and the augmentation"}
    )
    devices: Auto[Optional[int]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "GPU index inside the visible devices; auto or null leaves every visible GPU"},
    )
    precision: _Precision = dataclasses.field(
        default="float32", metadata={"help": "float32, mixed_float16 or mixed_bfloat16"}
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "continue a killed run of the same out with the same request (devices "
            "may differ); the data order replays epoch 0's and a half-done epoch is trained "
            "again; the learning rate, ReduceLROnPlateau and early stopping come back; a kill "
            "between the best checkpoint's save and the backup can leave best_model.keras one "
            "epoch LATER than the resumed state's best (the epoch that is then trained again; "
            "never an earlier one); refused: a finished run, other data than the run's, a live "
            "process holding the run, a backup neither of whose copies is sound"
        },
    )
    checkpoint: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "a .keras model whose same-named layers give their weights to this one"},
    )
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict,
        metadata={"help": "keywords of the model factory; they win over the family's"},
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
    """The four hooks of an image-classification run; see the module contract."""

    task = TASK

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
        )

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
    given = {
        "out": str(out) if isinstance(out, Path) else out,
        "model": model,
        "data": data,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "devices": devices,
        "precision": precision,
        "resume_interrupted": resume_interrupted,
        "checkpoint": str(checkpoint) if isinstance(checkpoint, Path) else checkpoint,
        "model_args": {} if model_args is None else model_args,
    }
    groups = {
        "optim_args": optim_args,
        "transform_args": transform_args,
        "metric_args": metric_args,
        "checkpoint_args": checkpoint_args,
        "logger_args": logger_args,
        "analysis_args": analysis_args,
        "debug_args": debug_args,
    }
    given.update({name: value for name, value in groups.items() if value is not None})
    config = validate(ImageClassificationConfig, given)
    return ImageClassificationTrainer(config).run()
