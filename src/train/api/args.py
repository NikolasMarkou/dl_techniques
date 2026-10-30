"""The seven argument groups of a training task, and how ``"auto"`` becomes a value.

A task config carries these groups next to its own fields (``out``, ``model``,
``epochs`` ...). Each group is a frozen :class:`~train.api.config.StrictConfig`.
A field that the model family or the data should decide holds ``"auto"`` until
:meth:`resolve_auto` fills it in.

Interface contract
------------------
``AutoContext``
    Frozen record of what is known after the registry lookup and the data load.

    :param family: model family name, e.g. ``"vit"``.
    :param variant: variant name, e.g. ``"vit_pico"``.
    :param dataset: dataset name, e.g. ``"cifar10"``.
    :param num_classes: number of classes the data has.
    :param image_size: side length of the (square) input image.
    :param defaults: the family recipe as a nested mapping,
        ``{"optim_args": {"lr": 3e-4, ...}, "metric_args": {...}, ...}``, one
        entry per group, keyed by the group's field name in the task config.
        Keys a recipe does not carry are simply absent.

``<Group>.resolve_auto(context)``
    :param context: an :class:`AutoContext`.
    :return: a NEW instance of the same group. Only fields still holding
        ``"auto"`` are replaced; every other value, an explicit one that equals
        the recipe's value included, is returned untouched (D-010). A field
        the recipe has no entry for stays ``"auto"``, and
        :func:`~train.api.config.assert_resolved` then reports it by dotted path.
    :raises ConfigValidationError: when a recipe value does not fit the field.
    :raises ConfigError: when ``TransformArgs.augment`` names no preset.

The groups carry only data and the AUTO rules. The code that reads the fields
lands with them: ``optim.py`` (``OptimArgs``), the task file (``TransformArgs``)
and ``trainer.py`` (the rest).

Import rule
-----------
Module scope imports only the standard library and ``train.api``. The two
``train.common`` helpers are imported inside the methods that call them (D-012).
"""

import dataclasses
from typing import Any, Dict, Iterable, Literal, Mapping, Optional

from .config import AUTO, Auto, StrictConfig, validate
from .errors import ConfigError, ConfigValidationError

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AutoContext:
    """What is known when ``"auto"`` fields are resolved; see the module docstring."""

    family: str
    variant: str
    dataset: str
    num_classes: int
    image_size: int
    defaults: Mapping[str, Any]


def _is_auto(value: Any) -> bool:
    return isinstance(value, str) and value == AUTO


def _recipe_values(
    group: StrictConfig, context: AutoContext, key: str, skip: Iterable[str] = ()
) -> Dict[str, Any]:
    """Return the recipe value of every AUTO field of ``group`` the recipe has.

    :param group: the group being resolved.
    :param context: supplies ``context.defaults[key]``.
    :param key: the group's field name in the task config.
    :param skip: field names that must stay as they are.
    """
    table = context.defaults.get(key, {})
    skipped = set(skip)
    return {
        field.name: table[field.name]
        for field in dataclasses.fields(group)
        if field.name not in skipped
        and _is_auto(getattr(group, field.name))
        and field.name in table
    }


def _with_values(group: StrictConfig, key: str, values: Mapping[str, Any]) -> Any:
    """Return a new ``group`` with ``values`` applied, held to the field types.

    A recipe is data; a typo in it (a string for ``lr``) must fail here and
    name the group, not surface later inside the optimizer.
    """
    merged = {f.name: getattr(group, f.name) for f in dataclasses.fields(group)}
    merged.update(values)
    try:
        return validate(type(group), merged)
    except ConfigValidationError as exc:
        raise ConfigValidationError(
            [f"Family recipe for {key!r}: {line}" for line in exc.errors]
        ) from exc


# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OptimArgs(StrictConfig):
    """Optimizer and learning-rate schedule. Read by ``optim.py``."""

    optimizer: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={"help": "optimizer name: adamw, adam or sgd; auto takes the family's"},
    )
    lr: Auto[float] = dataclasses.field(
        default=AUTO, metadata={"help": "peak learning rate"}
    )
    weight_decay: Auto[Optional[float]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "decoupled weight decay (adamw only); null for none"},
    )
    momentum: Auto[Optional[float]] = dataclasses.field(
        default=AUTO, metadata={"help": "momentum (sgd only); null for none"}
    )
    beta_1: Auto[Optional[float]] = dataclasses.field(
        default=AUTO, metadata={"help": "Adam beta_1; null for the optimizer's own"}
    )
    beta_2: Auto[Optional[float]] = dataclasses.field(
        default=AUTO, metadata={"help": "Adam beta_2; null for the optimizer's own"}
    )
    epsilon: Auto[Optional[float]] = dataclasses.field(
        default=AUTO, metadata={"help": "Adam epsilon; null for the optimizer's own"}
    )
    schedule: Auto[Literal["cosine_decay", "constant"]] = dataclasses.field(
        default=AUTO, metadata={"help": "learning-rate schedule after warmup"}
    )
    warmup_epochs: Auto[int] = dataclasses.field(
        default=AUTO, metadata={"help": "epochs of linear warmup; 0 for none"}
    )
    warmup_start_lr: Auto[float] = dataclasses.field(
        default=AUTO, metadata={"help": "learning rate at the first warmup step"}
    )
    min_lr_ratio: Auto[float] = dataclasses.field(
        default=AUTO,
        metadata={"help": "final cosine learning rate as a fraction of lr"},
    )
    decay_over: Auto[Literal["total", "post_warmup"]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "cosine length: the whole run (total) or the steps after "
            "warmup (post_warmup)"
        },
    )
    clipnorm: Auto[Optional[float]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "clip each variable's gradient to this norm; null for none"},
    )
    global_clipnorm: Auto[Optional[float]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "clip the global gradient norm to this value; null for none"},
    )

    def resolve_auto(self, context: AutoContext) -> "OptimArgs":
        """Fill ``"auto"`` fields from ``context.defaults["optim_args"]``.

        The two clipping fields exclude each other (``optim.py`` refuses both).
        When the caller sets one to a number, the other is resolved to ``None``
        rather than to the family's value, so an explicit choice never collides
        with the recipe.
        """
        found = _recipe_values(self, context, "optim_args")
        # DECISION plan-2026-10-30T132432-a865fa08/D-028: do NOT fill the
        # second clipping field from the recipe when the caller set the first.
        # optim.py refuses both together, so recipe filling would turn a
        # lone ``optim_args.clipnorm=0.5`` into a refusal the user never caused.
        if self.clipnorm is not None and not _is_auto(self.clipnorm):
            if _is_auto(self.global_clipnorm):
                found["global_clipnorm"] = None
        if self.global_clipnorm is not None and not _is_auto(self.global_clipnorm):
            if _is_auto(self.clipnorm):
                found["clipnorm"] = None
        return _with_values(self, "optim_args", found)


@dataclasses.dataclass(frozen=True)
class TransformArgs(StrictConfig):
    """Input size, augmentation and normalization. Read by the task file."""

    image_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={"help": "side length the images are resized to; auto is the data's own"},
    )
    augment: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={"help": "augmentation preset from train.common.image_augment.PRESETS"},
    )
    random_erasing_prob: Auto[float] = dataclasses.field(
        default=AUTO, metadata={"help": "probability of random erasing"}
    )
    normalize: Literal["standardize", "unit"] = dataclasses.field(
        default="standardize",
        metadata={"help": "standardize with the dataset statistics, or leave in [0, 1]"},
    )

    def resolve_auto(self, context: AutoContext) -> "TransformArgs":
        """Fill ``"auto"`` fields; ``image_size`` comes from the data.

        :raises ConfigError: when ``augment``, given or from the recipe, is not
            a name in ``train.common.image_augment.PRESETS``.
        """
        from train.common.image_augment import PRESETS

        found = _recipe_values(self, context, "transform_args", skip=("image_size",))
        if _is_auto(self.image_size):
            found["image_size"] = context.image_size
        resolved = _with_values(self, "transform_args", found)
        if not _is_auto(resolved.augment) and resolved.augment not in PRESETS:
            raise ConfigError(
                f"Invalid value for 'transform_args.augment': "
                f"{resolved.augment!r} is not a preset; choose one of "
                f"{', '.join(PRESETS)}."
            )
        return resolved


@dataclasses.dataclass(frozen=True)
class MetricArgs(StrictConfig):
    """The metric that selects the best epoch, and the success bar. Read by ``trainer.py``."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={"help": "metric that picks the best epoch, e.g. val_accuracy"},
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "min or max; auto is read from the metric's name"},
    )
    top_k: int = dataclasses.field(
        default=5, metadata={"help": "k of the extra top-k accuracy metric"}
    )
    success_threshold: Auto[float] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "best val_accuracy at or above this counts as success; "
            "auto is twice chance, kept between 0.05 and 0.95"
        },
    )

    def resolve_auto(self, context: AutoContext) -> "MetricArgs":
        """Fill ``"auto"`` fields.

        ``watch_metric`` comes from the recipe; ``watch_mode`` from the
        resolved metric's name through ``resolve_monitor_mode``;
        ``success_threshold`` from the class count.
        """
        from train.common.callbacks import resolve_monitor_mode

        if context.num_classes < 1:
            raise ConfigError(
                f"The data reports {context.num_classes} classes; "
                "success_threshold needs at least 1."
            )
        found = _recipe_values(
            self, context, "metric_args", skip=("watch_mode", "success_threshold")
        )
        metric = found.get("watch_metric", self.watch_metric)
        if _is_auto(self.watch_mode) and not _is_auto(metric):
            found["watch_mode"] = resolve_monitor_mode(metric)
        if _is_auto(self.success_threshold):
            found["success_threshold"] = min(max(2 / context.num_classes, 0.05), 0.95)
        return _with_values(self, "metric_args", found)


@dataclasses.dataclass(frozen=True)
class CheckpointArgs(StrictConfig):
    """Which weights are kept and when training stops early. Read by ``trainer.py``."""

    save_best: bool = dataclasses.field(
        default=True, metadata={"help": "keep best_model.keras and score it after training"}
    )
    save_last: bool = dataclasses.field(
        default=True, metadata={"help": "write final_model.keras after training"}
    )
    early_stopping_patience: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={"help": "epochs without improvement before stopping"},
    )

    def resolve_auto(self, context: AutoContext) -> "CheckpointArgs":
        """Fill ``early_stopping_patience`` from ``context.defaults["checkpoint_args"]``."""
        return _with_values(
            self, "checkpoint_args", _recipe_values(self, context, "checkpoint_args")
        )


@dataclasses.dataclass(frozen=True)
class LoggerArgs(StrictConfig):
    """Logging sinks. Read by ``trainer.py``."""

    tensorboard: Optional[Dict[str, Any]] = dataclasses.field(
        default_factory=dict,
        metadata={
            "help": "{} writes TensorBoard logs, null turns them off; a "
            "non-empty dict is refused (no option is wired)"
        },
    )

    def __post_init__(self) -> None:
        # DECISION plan-2026-10-30T132432-a865fa08/D-022: a TensorBoard dict
        # with keys is refused, not ignored. Do NOT accept options and drop
        # them: create_callbacks only switches TensorBoard on or off, so a
        # silently ignored option looks like it worked.
        if self.tensorboard:
            raise ConfigValidationError(
                [
                    "Invalid value for 'logger_args.tensorboard': expected {} "
                    "or null, got a dict with keys "
                    f"{sorted(self.tensorboard)} (no TensorBoard option is wired)"
                ]
            )


@dataclasses.dataclass(frozen=True)
class AnalysisArgs(StrictConfig):
    """Analysis and figures around training. Read by ``trainer.py``."""

    epoch_analysis_every: int = dataclasses.field(
        default=0,
        metadata={"help": "run the epoch analyzer every N epochs; 0 turns it off"},
    )
    model_analysis: bool = dataclasses.field(
        default=True, metadata={"help": "run the full model analysis once after training"}
    )
    figures: bool = dataclasses.field(
        default=True, metadata={"help": "write the classification figures"}
    )
    conformal_alpha: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "miscoverage level for conformal prediction sets; null skips it"},
    )


@dataclasses.dataclass(frozen=True)
class DebugArgs(StrictConfig):
    """Switches for short or defensive runs. Read by ``trainer.py``."""

    terminate_on_nan: bool = dataclasses.field(
        default=True, metadata={"help": "stop the run when the loss turns NaN"}
    )
    smoke: bool = dataclasses.field(
        default=False,
        metadata={"help": "one epoch on at most 512 samples; refused with an explicit epochs"},
    )
