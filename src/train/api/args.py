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
    :param image_size: side length of the (square) input image; ``None`` for a segmentation run
        (its size is a ``[height, width]`` pair and lives in the task's own facts).
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
    :raises ConfigValidationError: when a recipe group is not a mapping, names
        a key the group does not have (AUTO or not), or holds a value that
        does not fit the field.
    :raises ConfigError: when ``TransformArgs.augment`` names no preset, or the
        data's ``image_size`` or ``num_classes`` cannot fill an AUTO field.

The groups carry only data and the AUTO rules. The code that reads the fields
lands with them: ``optim.py`` (``OptimArgs``), the task file (``TransformArgs``)
and ``trainer.py`` (the rest).

Import rule
-----------
Module scope imports only the standard library and ``train.api``. The two
``train.common`` helpers are imported inside the methods that call them (D-012).
"""

import dataclasses
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from .config import (
    AUTO,
    Auto,
    StrictConfig,
    join_path,
    unknown_key_message,
    validate,
)
from .errors import ConfigError, ConfigValidationError

# ---------------------------------------------------------------------

#: ``success_threshold = auto`` is this many times chance (``1 / num_classes``),
#: kept between the floor and the ceiling.
SUCCESS_CHANCE_MULTIPLE = 2
SUCCESS_THRESHOLD_FLOOR = 0.05
SUCCESS_THRESHOLD_CEILING = 0.95

#: Momentum an sgd run gets when the caller switched to sgd from another
#: family optimizer and gave none (the legacy ``--optimizer sgd`` default of
#: ``train_vit`` and ``train_resnet``).
SGD_MOMENTUM = 0.9

#: The one augmentation preset that takes a random-erasing probability
#: (``train.common.image_augment.build_augment``).
ERASING_PRESET = "cifar_autoaugment"

#: ``OptimArgs`` fields that belong to some optimizers and not to others.
_OPTIMIZER_OPTIONS = ("beta_1", "beta_2", "epsilon", "momentum", "weight_decay")

#: Options two different optimizers both take, keyed by the pair of names.
#: ``weight_decay`` is not here: for adamw it is decoupled decay, for sgd it is
#: the strength of an L2 regularizer on the model, two different quantities.
_SHARED_OPTIONS = {frozenset(("adam", "adamw")): ("beta_1", "beta_2", "epsilon")}

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AutoContext:
    """What is known when ``"auto"`` fields are resolved; see the module docstring."""

    family: str
    variant: str
    dataset: str
    num_classes: int
    image_size: Optional[int]
    defaults: Mapping[str, Any]


def _is_auto(value: Any) -> bool:
    """Whether ``value`` is the string ``"auto"`` (the one definition; ``trainer`` and ``segmentation_args`` import it).

    :param value: any field value; a non-string (``None``, a number, a list) is never ``"auto"``.
    :return: ``True`` only for the exact string :data:`~train.api.config.AUTO`. Never raises.
    """
    return isinstance(value, str) and value == AUTO


def _range_line(prefix: str, name: str, expected: str, value: Any) -> str:
    """Return the line for a number outside its range, naming the dotted key."""
    return f"Invalid value for {join_path(prefix, name)!r}: expected {expected}, got {value!r}"


def _recipe_values(
    group: StrictConfig, context: AutoContext, key: str, skip: Iterable[str] = ()
) -> Dict[str, Any]:
    """Return the recipe value of every AUTO field of ``group`` the recipe has.

    :param group: the group being resolved.
    :param context: supplies ``context.defaults[key]``.
    :param key: the group's field name in the task config.
    :param skip: field names that must stay as they are.
    :raises ConfigValidationError: when the recipe's entry for the group is not
        a mapping, or holds a key that is not a field of the group.
    """
    table = context.defaults.get(key, {})
    if not isinstance(table, Mapping):
        raise ConfigValidationError(
            [
                f"Family recipe for {key!r}: expected a mapping of "
                f"{type(group).__name__} fields, got {table!r} "
                f"({type(table).__name__})"
            ]
        )
    names = [field.name for field in dataclasses.fields(group)]
    # A key that is no field is a typo whether or not its field is AUTO, so
    # it is named here and not dropped (D-034).
    unknown = [name for name in table if name not in names]
    if unknown:
        raise ConfigValidationError(
            [
                f"Family recipe for {key!r}: {unknown_key_message(key, str(name), names)}"
                for name in unknown
            ]
        )
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
        metadata={
            "help": "weight decay: decoupled decay for adamw, the strength of an L2 "
            "regularizer on the model for sgd, refused for adam; null for none"
        },
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
        self._drop_foreign_recipe_values(context, found)
        return _with_values(self, "optim_args", found)

    def _drop_foreign_recipe_values(
        self, context: AutoContext, found: Dict[str, Any]
    ) -> None:
        """Edit ``found`` so a recipe never hands an optimizer another one's options.

        :param context: supplies the recipe's own ``optimizer``.
        :param found: the recipe values chosen so far; changed in place.

        When the optimizer in force (explicit or the recipe's) is not the
        recipe's, every AUTO field that belongs to one optimizer and not the
        other resolves to ``None`` (``momentum`` to :data:`SGD_MOMENTUM` for
        sgd), and an AUTO ``lr`` is refused. A constant schedule has no
        warmup, so an AUTO ``warmup_epochs`` resolves to 0 there.

        :raises ConfigError: when the optimizer is switched and ``lr`` is AUTO.
        """
        # DECISION plan-2026-10-30T132432-a865fa08/D-039: do NOT fill
        # beta_1, beta_2, epsilon, momentum or weight_decay from a recipe that
        # belongs to a different optimizer. ``optimizer=sgd`` on an adamw
        # family would otherwise receive the recipe's betas and decay, and
        # optim.py refuses those; the user would be blamed for values the
        # library chose. Options two optimizers share (adam and adamw take the
        # same betas and epsilon) are kept. Equally, an AUTO warmup_epochs is
        # not filled when the schedule is constant, which has no warmup.
        recipe = context.defaults.get("optim_args", {})
        chosen = found.get("optimizer", self.optimizer)
        if isinstance(chosen, str) and not _is_auto(chosen):
            kind = chosen.strip().lower()
            recipe_kind = recipe.get("optimizer")
            recipe_kind = (
                recipe_kind.strip().lower() if isinstance(recipe_kind, str) else None
            )
            if kind != recipe_kind:
                # DECISION plan-2026-10-30T132432-a865fa08/D-045: a switch to
                # another optimizer needs an explicit ``lr``. Do NOT keep the
                # recipe's: a peak rate tuned for adamw (3e-4) trains an sgd
                # run badly, and a rate tuned for sgd (0.1) wrecks an adamw
                # run, with nothing said (review-iter-2 concern 10).
                if recipe_kind is not None and "lr" in recipe and _is_auto(self.lr):
                    raise ConfigError(
                        f"Invalid value for 'optim_args.lr': optim_args.optimizer={kind} "
                        f"replaces the family's {recipe_kind}, whose peak learning rate "
                        f"{recipe['lr']!r} was tuned for it; set optim_args.lr"
                    )
                shared = _SHARED_OPTIONS.get(frozenset((kind, recipe_kind)), ())
                for name in _OPTIMIZER_OPTIONS:
                    if name not in shared and _is_auto(getattr(self, name)):
                        found[name] = None
                if kind == "sgd" and _is_auto(self.momentum):
                    found["momentum"] = SGD_MOMENTUM
        schedule = found.get("schedule", self.schedule)
        if schedule == "constant" and _is_auto(self.warmup_epochs):
            found["warmup_epochs"] = 0


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

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit size or probability that cannot be right."""
        lines: List[str] = []
        size, probability = self.image_size, self.random_erasing_prob
        if not _is_auto(size) and size < 1:
            lines.append(_range_line(prefix, "image_size", "a positive int", size))
        if not _is_auto(probability) and not 0 <= probability <= 1:
            lines.append(
                _range_line(prefix, "random_erasing_prob", "a probability in [0, 1]", probability)
            )
        return lines

    def resolve_auto(self, context: AutoContext) -> "TransformArgs":
        """Fill ``"auto"`` fields; ``image_size`` comes from the data.

        An AUTO ``random_erasing_prob`` takes the recipe's value when the
        augment in force (explicit or the recipe's) is ``cifar_autoaugment``
        and is 0.0 under any other preset.

        :raises ConfigError: when ``augment``, given or from the recipe, is not
            a name in ``train.common.image_augment.PRESETS``, or when an
            explicit nonzero ``random_erasing_prob`` meets a preset other than
            ``cifar_autoaugment`` (D-042).
        """
        from train.common.image_augment import PRESETS

        found = _recipe_values(self, context, "transform_args", skip=("image_size",))
        # DECISION plan-2026-10-30T132432-a865fa08/D-041: an AUTO
        # random_erasing_prob is the recipe's value only under the preset it
        # belongs to. Do NOT take it from the recipe whatever the augment is:
        # the recipe's erasing probability is the cifar_autoaugment recipe's
        # (train_cliffordnet.py:684-685 gives 0.25 to that preset on ANY dataset
        # and 0.0 to every other), and image_augment.build_augment refuses a
        # nonzero probability with another preset.
        augment = found.get("augment", self.augment)
        if _is_auto(self.random_erasing_prob) and augment != ERASING_PRESET:
            found["random_erasing_prob"] = 0.0
        if _is_auto(self.image_size):
            size = context.image_size
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ConfigError(
                    f"The data reports image_size {size!r}; "
                    "transform_args.image_size = auto needs a positive integer. "
                    "Set transform_args.image_size."
                )
            found["image_size"] = size
        resolved = _with_values(self, "transform_args", found)
        if not _is_auto(resolved.augment) and resolved.augment not in PRESETS:
            raise ConfigError(
                f"Invalid value for 'transform_args.augment': "
                f"{resolved.augment!r} is not a preset; choose one of "
                f"{', '.join(PRESETS)}."
            )
        # DECISION plan-2026-10-30T132432-a865fa08/D-042: an EXPLICIT nonzero
        # random_erasing_prob under a preset that takes none is refused HERE, as a
        # ConfigError. Do NOT leave it to ``build_augment``: its ValueError reaches
        # the user as a traceback and exit 1 (the code for a failed run), from the
        # loader, after the run directory is claimed. An AUTO probability never gets
        # here nonzero (it is 0.0 under any other preset, D-041).
        probability = resolved.random_erasing_prob
        if (
            not _is_auto(probability)
            and probability != 0
            and not _is_auto(resolved.augment)
            and resolved.augment != ERASING_PRESET
        ):
            raise ConfigError(
                f"Invalid value for 'transform_args.random_erasing_prob': {probability!r} "
                f"applies to the {ERASING_PRESET!r} preset only, but "
                f"transform_args.augment is {resolved.augment!r}; set "
                f"transform_args.augment={ERASING_PRESET} or drop the probability"
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
            f"auto is {SUCCESS_CHANCE_MULTIPLE} times chance, kept between "
            f"{SUCCESS_THRESHOLD_FLOOR} and {SUCCESS_THRESHOLD_CEILING}"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit ``top_k`` or ``success_threshold`` outside its range."""
        lines: List[str] = []
        if self.top_k < 1:
            lines.append(_range_line(prefix, "top_k", "an int >= 1", self.top_k))
        threshold = self.success_threshold
        if not _is_auto(threshold) and not 0 <= threshold <= 1:
            lines.append(
                _range_line(prefix, "success_threshold", "a fraction in [0, 1]", threshold)
            )
        return lines

    def resolve_auto(self, context: AutoContext) -> "MetricArgs":
        """Fill ``"auto"`` fields.

        ``watch_metric`` comes from the recipe; ``watch_mode`` from the
        resolved metric's name through ``resolve_monitor_mode``;
        ``success_threshold`` from the class count.
        """
        from train.common.callbacks import resolve_monitor_mode

        classes = context.num_classes
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 1:
            raise ConfigError(
                f"The data reports {classes!r} classes; "
                "success_threshold needs at least 1."
            )
        found = _recipe_values(
            self, context, "metric_args", skip=("watch_mode", "success_threshold")
        )
        metric = found.get("watch_metric", self.watch_metric)
        if _is_auto(self.watch_mode) and not _is_auto(metric):
            found["watch_mode"] = resolve_monitor_mode(metric)
        if _is_auto(self.success_threshold):
            auto_threshold = SUCCESS_CHANCE_MULTIPLE / classes
            found["success_threshold"] = min(
                max(auto_threshold, SUCCESS_THRESHOLD_FLOOR),
                SUCCESS_THRESHOLD_CEILING,
            )
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
    backup_every_steps: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "also back the run up every this many batches INSIDE an epoch (an int >= 2 "
            "and below the steps per epoch), so a killed epoch loses at most that many batches "
            "of weights; null backs up at each epoch end only. A resume from such a backup "
            "continues the killed epoch on the weights of the last backup and runs only the "
            "batches of it that backup had not taken, from a fresh pass of the data (some "
            "examples of that epoch are seen twice and some not at all), then the remaining "
            "epochs in full; the optimizer's step count and the learning rate of every step "
            "are the unkilled run's. That epoch's row holds the train metrics of its remaining "
            "batches only. Each backup writes the weights and optimizer "
            "state twice (the previous copy is kept): src/train/README.md has the measured "
            "cost per save"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit negative patience and a ``backup_every_steps`` below 2."""
        lines = []
        patience = self.early_stopping_patience
        if not _is_auto(patience) and patience < 0:
            lines.append(_range_line(prefix, "early_stopping_patience", "an int >= 0", patience))
        # DECISION plan-2026-10-31T180033-2faf890a/D-038: 1 is refused, not rounded up. A
        # copy after the first batch of an epoch writes the same metadata as the epoch-end
        # copy (`batch` 0), and `settle_backup` tells the two apart by that index.
        every = self.backup_every_steps
        if every is not None and every < 2:
            lines.append(_range_line(prefix, "backup_every_steps", "null or an int >= 2", every))
        return lines

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

    def problems(self, prefix: str) -> List[str]:
        """Name a non-empty ``tensorboard`` dict as a problem at its own path."""
        # DECISION plan-2026-10-30T132432-a865fa08/D-022: a TensorBoard dict
        # with keys is refused, not ignored. Do NOT accept options and drop
        # them: create_callbacks only switches TensorBoard on or off, so a
        # silently ignored option looks like it worked. D-034: it is reported
        # here, from validate's walk, and NOT raised from __post_init__, which
        # aborted the walk and hid every other bad key, and which a later
        # in-place edit of the dict bypassed.
        if not self.tensorboard:
            return []
        return [
            f"Invalid value for {join_path(prefix, 'tensorboard')!r}: expected "
            "{} or null, got a dict with keys "
            f"{sorted(map(str, self.tensorboard))} (no TensorBoard option is wired)"
        ]


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
        default=True, metadata={"help": "write the task's figures"}
    )
    conformal_alpha: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "miscoverage level in (0, 1) for the conformal prediction sets of a classifier; "
            "null skips it; image_segmentation refuses a value (a set of classes for a whole image "
            "is not a segmentation output)"
        },
    )
    eval_batch_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "images per pass of the predictions made after training: the figures' "
            "predictions (every dataset), the model analyzer's predictions (it caps its own 32 at "
            "this value when this is lower; its information-flow pass sizes its own batch from the "
            "analyzer's memory budget and does not read it) and the final-model load check (the "
            "same cap), and the validation and test splits of the in-memory datasets (CIFAR, "
            "MNIST). The validation and test splits of ImageNet, Imagenette, folder and csv data "
            "are batched at batch_size and ignore it. auto is the recipe's value, and 256 where "
            "the recipe says nothing; the cliffordnet recipes on ImageNet and Imagenette set the "
            "batch they train at (8 or 16), because 256 images at 224 px ask for 18 GB. Those "
            "recipe values are keyed to the dataset: folder and csv data take 256 whatever the "
            "family or the image size, so a cliffordnet run on files of 224 px should set it (and "
            "batch_size) itself. It changes no weight; it can change an evaluation number in the "
            "last digits. image_segmentation batches the selection and test splits of every source at "
            "it (its recipes set the training batch, 16)"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an analysis period below 0, a miscoverage level outside (0, 1), a batch below 1."""
        lines: List[str] = []
        batch = self.eval_batch_size
        if not _is_auto(batch) and (isinstance(batch, bool) or batch < 1):
            lines.append(_range_line(prefix, "eval_batch_size", "an int >= 1", batch))
        if self.epoch_analysis_every < 0:
            lines.append(
                _range_line(prefix, "epoch_analysis_every", "an int >= 0 (0 is off)", self.epoch_analysis_every)
            )
        alpha = self.conformal_alpha
        if alpha is not None and not 0 < alpha < 1:
            lines.append(
                _range_line(prefix, "conformal_alpha", "null or a level strictly between 0 and 1", alpha)
            )
        return lines

    def resolve_auto(self, context: AutoContext) -> "AnalysisArgs":
        """Fill ``eval_batch_size``: the recipe's ``analysis_args`` entry, else the shared 256.

        The fallback is :data:`train.common.classification_data.EVAL_BATCH_SIZE`, the constant
        the in-memory pipelines and the figure prediction used before the key existed, so a
        run whose recipe says nothing evaluates exactly as it did.
        """
        # DECISION plan-2026-10-31T180033-2faf890a/D-040
        # The one source of the fallback is the legacy constant, imported here and NOT copied:
        # a second 256 would have to be kept equal by hand. Do NOT make ``auto`` the training
        # batch: CIFAR trains at 128 and has always evaluated at 256, and a different batch can
        # move an evaluation number in the last digits (the golden would no longer be exact).
        from train.common.classification_data import EVAL_BATCH_SIZE

        found = _recipe_values(self, context, "analysis_args")
        if not _is_auto(self.eval_batch_size):
            return self
        value = found.get("eval_batch_size", EVAL_BATCH_SIZE)
        # Not `_with_values`: it validates the WHOLE group again and words any error as the
        # recipe's, so a hand-built group with a bad `conformal_alpha` would be reported as
        # "Family recipe for 'analysis_args'" (test_a_bad_request_is_refused_before_the_run_is_claimed).
        # Only the recipe's own value is checked here.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigValidationError([
                "Family recipe for 'analysis_args': "
                + _range_line("", "eval_batch_size", "an int >= 1", value)
            ])
        return dataclasses.replace(self, eval_batch_size=value)


@dataclasses.dataclass(frozen=True)
class DebugArgs(StrictConfig):
    """Switches for short or defensive runs. Read by ``trainer.py``."""

    terminate_on_nan: bool = dataclasses.field(
        default=True, metadata={"help": "stop the run when the loss turns NaN"}
    )
    smoke: bool = dataclasses.field(
        default=False,
        metadata={"help": "one epoch on at most 512 train samples (the test split is whole); refused with an explicit epochs"},
    )
