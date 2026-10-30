"""A task config and a trainer, written the way the image-classification task will be.

``TaskTrainer`` is generic and has no config of its own, so its tests need a
task to run. This one is the smallest honest one: the real ``BuiltinImageData``,
the real loader, the real family registry and resolvers, the real shared
helpers. Only the config class and four hooks are written here.

Interface contract
------------------
``StubConfig``
    Frozen :class:`~train.api.config.StrictConfig` carrying every field of
    :data:`train.api.trainer.CONFIG_FIELDS`. ``out``, ``model`` and ``data`` are
    required.

``StubTrainer(config)``
    A :class:`~train.api.trainer.TaskTrainer`. ``describe_data`` and
    ``build_data`` go through ``data_args``; ``build_model`` through the
    family's resolver and ``recipes.build_model``.

``make_config(out, **changes)``
    :param out: the run directory (an absolute path under ``tmp_path``).
    :param changes: field -> value; a dotted key (``"debug_args.smoke"``) sets a
        field of a group.
    :return: a ``StubConfig`` for ``vit/vit_pico`` on cifar10 with a batch of 4
        and the model analysis off (it costs a minute on the CPU).
"""

import dataclasses
from typing import Any, Dict, Optional

import train.api.recipes as recipes
from train.api import registry
from train.api.args import (
    AnalysisArgs,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
    MetricArgs,
    OptimArgs,
    TransformArgs,
)
from train.api.config import AUTO, DEFAULT_SEED, Auto, StrictConfig, validate
from train.api.data_args import BuiltinImageData, DataFacts, builtin_facts, load_builtin_data
from train.api.trainer import TaskTrainer


@dataclasses.dataclass(frozen=True)
class StubConfig(StrictConfig):
    out: str
    model: str
    data: BuiltinImageData
    epochs: Auto[int] = AUTO
    batch_size: Auto[int] = AUTO
    seed: int = DEFAULT_SEED
    devices: Auto[Optional[int]] = AUTO
    precision: str = "float32"
    resume_interrupted: bool = False
    checkpoint: Optional[str] = None
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: TransformArgs = dataclasses.field(default_factory=TransformArgs)
    metric_args: MetricArgs = dataclasses.field(default_factory=MetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)


class StubTrainer(TaskTrainer):
    task = "image_classification"

    def describe_data(self, config: StubConfig) -> DataFacts:
        size = config.transform_args.image_size
        return builtin_facts(config.data, None if size == AUTO else size)

    def build_data(self, config: StubConfig, context: Any) -> Any:
        from train.common.image_augment import build_augment

        transform = config.transform_args
        augment = build_augment(
            transform.augment, random_erasing_prob=transform.random_erasing_prob
        )
        return load_builtin_data(
            config.data,
            batch_size=config.batch_size,
            seed=config.seed,
            augment=augment,
            normalize=transform.normalize,
            image_size=transform.image_size,
            loader=context.defaults.get("loader"),
        )

    def build_model(self, config: StubConfig, context: Any, data: Any) -> Any:
        spec = registry.MODELS[context.family]
        kwargs = spec.load_resolver()(context, config.optim_args, config.model_args)
        return recipes.build_model(
            spec, context.variant, data.num_classes, data.input_shape, kwargs
        )

    def build_loss_and_metrics(self, config: StubConfig, data: Any) -> Any:
        import keras

        from train.common.classification_run import classification_metrics

        return (
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            classification_metrics(data.num_classes, config.metric_args.top_k),
        )


def make_config(out: Any, **changes: Any) -> StubConfig:
    """Return a validated ``StubConfig``; see the module contract."""
    mapping: Dict[str, Any] = {
        "out": str(out),
        "model": "vit/vit_pico",
        "data": {"dataset": "cifar10"},
        "batch_size": 4,
        "analysis_args": {"model_analysis": False},
    }
    for key, value in changes.items():
        head, _, tail = key.partition(".")
        if tail:
            mapping.setdefault(head, {})
            mapping[head] = {**mapping[head], tail: value}
        else:
            mapping[head] = value
    return validate(StubConfig, mapping)
