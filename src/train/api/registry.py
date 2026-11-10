"""Registries of model families and tasks, stored as strings and imported lazily.

A :class:`ModelSpec` or :class:`TaskSpec` names its code as ``"module:attr"``
text, so storing a spec imports nothing and the registry can sit on the CLI
help path (D-012). The code is imported by the ``load_*`` methods, which the
commands that need it call.

Interface contract
------------------
``register_model(spec)`` / ``register_task(spec)``
    Add a spec to :data:`MODELS` / :data:`TASKS` and return it.

    :raises ValueError: when the family or task name is already registered, or a
        task is named like a built-in command (``cli.COMMANDS``), which
        ``cli`` looks up first and so would hide it. That is a bug in the registering code, not a user error, so it is not
        a :class:`~train.api.errors.DLTrainError` and reaches the user as a
        traceback.

``parse_model_name(name, task=None)``
    :param name: ``"family/variant"``, e.g. ``"vit/vit_pico"``.
    :param task: the task the model is for; ``None`` skips the task check.
    :return: ``(family, variant)``.
    :raises UnknownModelError: for a missing slash, an unknown family or an
        unknown variant; the message carries a ``difflib`` suggestion. The
        variant check imports the family's variants table.
    :raises ConfigError: when ``task`` is given and is not in the family's
        ``tasks``, in either direction (a segmentation family under the
        classifier, a classifier under the segmenter). The message names the
        key, the tasks the family trains and the families of ``task``. It is
        raised before the variants table is imported.

``list_models(task=None)``
    :param task: keep only families that list this task; ``None`` keeps all.
    :return: sorted ``"family/variant"`` names. Imports every matching family's
        variants table, so it is not on the help path.
    :raises ConfigError: when ``task`` names no registered task.

``list_tasks()``
    :return: the registered :class:`TaskSpec` rows, sorted by name.

``ModelSpec.stride_of(variant, model_args=None)``
    :param variant: variant name.
    :param model_args: the caller's ``model_args`` mapping, for a rule that a
        key of it changes (ConvUNext's ``depth``).
    :return: the number every side of a segmentation ``image_size`` must be a
        multiple of for this family (``None`` for a family with no such rule:
        the classifiers, whose ``image_size`` stays one square side, D-007).
    :raises ConfigError: from a stride function, for a ``model_args`` value it
        cannot use.

``ModelSpec.recipe(dataset, variant)``
    :return: a deep copy of the family's ``defaults`` with the ``per_dataset``
        and ``per_variant`` overlays merged in. This, not ``defaults``, is what
        goes into an ``AutoContext``.

``load_attr(path)``
    :param path: ``"module:attr"`` or ``"module:Class.attr"``.
    :return: the object. An import or attribute failure propagates; it means
        the spec is wrong.
"""

import copy
import difflib
import importlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .errors import ConfigError, UnknownModelError

# ---------------------------------------------------------------------

_ATTR_PATH = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")

#: Keys of ``ModelSpec.defaults`` that hold overlays, not recipe values.
OVERLAY_KEYS = ("per_dataset", "per_variant")

#: The properties a ``ModelSpec`` may declare in ``properties``. A property is a promise a task checks
#: about the family's models, not a capability the sequence switches on: ``"bias_free"`` is read by
#: the denoising task, which then verifies it on the built model (statically and numerically).
#: Adding a name here is the only way to let a spec declare one.
KNOWN_PROPERTIES = ("bias_free",)

#: The ``DataFacts.dataset`` names of file-based data (the ``format`` of a folder or
#: csv config). ``test_folder_and_csv_data`` holds them equal to the non-builtin
#: formats of ``train.api.data_args``, which this module does not import.
FILE_DATASETS = ("folder", "csv")

#: What every family's recipe says about file-based data, laid over its base recipe
#: before its own overlays. NEW BEHAVIOUR, not a legacy trainer's default (no legacy
#: trainer reads files): a horizontal flip, the one augmentation that is safe for any
#: photographs, in place of the base recipe's.
# DECISION plan-2026-10-30T132432-a865fa08/D-060: Do NOT give file data a family's per-dataset recipe (cifar's
# pad-crop and autoaugment are tuned for 32 px tiles); the overlay is one flip, and an
# explicit ``transform_args.augment`` still wins (D-010).
FILE_DATA_OVERLAY: Dict[str, Any] = {"transform_args": {"augment": "flip"}}


def load_attr(path: str) -> Any:
    """Import ``path`` (``"module:attr"``) and return the attribute.

    :param path: module and dotted attribute, split at the colon.
    :return: the object the path names.
    """
    module_name, _, attr_path = path.partition(":")
    value = importlib.import_module(module_name)
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value


def _require_attr_path(owner: str, field_name: str, text: str) -> None:
    if not isinstance(text, str) or not _ATTR_PATH.match(text):
        raise ValueError(
            f"{owner}.{field_name} must be 'module:attr' text, got {text!r}"
        )


@dataclass(frozen=True)
class ModelSpec:
    """One model family, as data.

    :param family: the first half of ``family/variant``; no slash.
    :param factory: ``"module:callable"`` building a model from a variant name.
    :param variants_attr: ``"module:Class.MODEL_VARIANTS"``, a mapping whose keys
        are the variant names.
    :param tasks: the task names this family can train.
    :param defaults: the family recipe as a nested mapping (see ``AutoContext``).
        Two sibling keys, ``per_dataset`` and ``per_variant``, hold overlays
        that :meth:`recipe` applies; they are not read anywhere else.
    :param resolve_model_args: optional ``"module:callable"`` called as
        ``f(context, optim_args, model_args)`` (an ``AutoContext``, the
        resolved ``OptimArgs`` and the caller's free mapping) that returns the
        factory keyword arguments; see :mod:`train.api.recipes`.
    :param stride: the size rule of a segmentation family: every side of its
        ``image_size`` must be a multiple of this. A positive int, or
        ``"module:callable"`` text of ``f(variant, model_args) -> int`` for a
        rule that depends on the variant or on a ``model_args`` key. ``None``
        (the default) is no rule: the classifiers keep their square ``image_size``
        and their own checks.
    :param observe: for a family whose models are functional (no class to be an
        instance of, no table keys in ``get_config()``): ``"module:callable"`` of
        ``f(model) -> Optional[Mapping[str, Any]]``. It reads the family's own layers
        and returns the variant-table keys they show (``{"depth": 3, ...}``), or
        ``None`` when the model holds no layer this family builds (a model of another
        family). ``train.api.predictor.check_model_against_record`` refuses ``None``
        and any key that differs from the record's variant. ``None`` (the default):
        the family is checked by its class and ``get_config()``.
    :param properties: promises about the family's models that a task reads and verifies,
        drawn from :data:`KNOWN_PROPERTIES` (a tuple of distinct names). ``()`` (the default):
        none, and no task reads anything here. Nothing in the shared sequence switches on it.
    """

    family: str
    factory: str
    variants_attr: str
    tasks: Tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    resolve_model_args: Optional[str] = None
    stride: Optional[Union[int, str]] = None
    observe: Optional[str] = None
    properties: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.family or "/" in self.family:
            raise ValueError(f"ModelSpec.family must be non-empty without '/', got {self.family!r}")
        _require_attr_path("ModelSpec", "factory", self.factory)
        _require_attr_path("ModelSpec", "variants_attr", self.variants_attr)
        if self.resolve_model_args is not None:
            _require_attr_path("ModelSpec", "resolve_model_args", self.resolve_model_args)
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError(f"ModelSpec.tasks must be a non-empty tuple, got {self.tasks!r}")
        if self.observe is not None:
            _require_attr_path("ModelSpec", "observe", self.observe)
        if (
            not isinstance(self.properties, tuple)
            or any(name not in KNOWN_PROPERTIES for name in self.properties)
            or len(set(self.properties)) != len(self.properties)
        ):
            raise ValueError(
                f"ModelSpec.properties must be a tuple of distinct names from "
                f"{KNOWN_PROPERTIES}, got {self.properties!r}"
            )
        if isinstance(self.stride, str):
            _require_attr_path("ModelSpec", "stride", self.stride)
        elif self.stride is not None and (
            isinstance(self.stride, bool) or not isinstance(self.stride, int) or self.stride < 1
        ):
            raise ValueError(
                f"ModelSpec.stride must be a positive int, 'module:callable' text or None, "
                f"got {self.stride!r}"
            )

    def recipe(self, dataset: str, variant: str) -> Dict[str, Any]:
        """Return the recipe for one dataset and variant, overlays applied.

        :param dataset: dataset name, e.g. ``"cifar10"``.
        :param variant: variant name, e.g. ``"cn_64_5"``.
        :return: a deep copy of :attr:`defaults` without its ``per_dataset``
            and ``per_variant`` keys, with :data:`FILE_DATA_OVERLAY` (for the
            dataset names ``"folder"`` and ``"csv"``), ``per_dataset[dataset]`` and
            then ``per_variant[variant]`` laid over it. An overlay entry that is a
            mapping is merged key by key into the base entry of the same name
            (a dataset changes ``optim_args.lr`` and leaves its other keys); any
            other value replaces. A dataset or variant with no overlay gets
            the base recipe, unchanged.
        """
        merged = {
            key: copy.deepcopy(value)
            for key, value in self.defaults.items()
            if key not in OVERLAY_KEYS
        }
        overlays = [
            FILE_DATA_OVERLAY if dataset in FILE_DATASETS else {},
            self.defaults.get("per_dataset", {}).get(dataset, {}),
            self.defaults.get("per_variant", {}).get(variant, {}),
        ]
        for overlay in overlays:
            for key, value in overlay.items():
                if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                    merged[key] = {**merged[key], **copy.deepcopy(dict(value))}
                else:
                    merged[key] = copy.deepcopy(value)
        return merged

    def load_factory(self) -> Any:
        """Import and return the model factory."""
        return load_attr(self.factory)

    def load_variants(self) -> Mapping[str, Any]:
        """Import and return the variants mapping."""
        return load_attr(self.variants_attr)

    def load_resolver(self) -> Optional[Any]:
        """Import and return ``resolve_model_args``, or ``None`` when unset."""
        if self.resolve_model_args is None:
            return None
        return load_attr(self.resolve_model_args)

    def load_observer(self) -> Optional[Any]:
        """Import and return ``observe``, or ``None`` when unset."""
        if self.observe is None:
            return None
        return load_attr(self.observe)

    def stride_of(
        self, variant: str, model_args: Optional[Mapping[str, Any]] = None
    ) -> Optional[int]:
        """Return the multiple each side of a segmentation ``image_size`` must be; see the module contract."""
        if self.stride is None or isinstance(self.stride, int):
            return self.stride
        return int(load_attr(self.stride)(variant, dict(model_args or {})))


@dataclass(frozen=True)
class TaskSpec:
    """One task, as data.

    :param name: the command word, e.g. ``"image_classification"``. It must not
        equal a built-in command; ``cli`` looks built-ins up first.
    :param config_class: ``"module:Class"`` of the task's ``StrictConfig``.
    :param function: ``"module:callable"`` of the keyword-only task function.
    :param summary: one line shown by ``help`` and ``list_tasks``.
    """

    name: str
    config_class: str
    function: str
    summary: str

    def __post_init__(self) -> None:
        if not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ValueError(f"TaskSpec.name must be lower snake case, got {self.name!r}")
        _require_attr_path("TaskSpec", "config_class", self.config_class)
        _require_attr_path("TaskSpec", "function", self.function)

    def load_config_class(self) -> Any:
        """Import and return the config class."""
        return load_attr(self.config_class)

    def load_function(self) -> Any:
        """Import and return the task function."""
        return load_attr(self.function)


# ---------------------------------------------------------------------
# DECISION plan-2026-10-30T132432-a865fa08/D-029: both registries start empty
# and a spec holds strings. Do NOT store the factory or the config class as
# objects: importing a family module loads TensorFlow, and this module is
# imported by cli.py on the help path.

MODELS: Dict[str, ModelSpec] = {}
TASKS: Dict[str, TaskSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to :data:`MODELS`; see the module contract."""
    if spec.family in MODELS:
        raise ValueError(f"model family {spec.family!r} is already registered")
    MODELS[spec.family] = spec
    return spec


def register_task(spec: TaskSpec) -> TaskSpec:
    """Add ``spec`` to :data:`TASKS`; see the module contract."""
    # Imported here because cli imports this module at its own import time.
    from .cli import COMMANDS

    if spec.name in COMMANDS:
        raise ValueError(
            f"task {spec.name!r} has the name of a built-in command "
            f"({', '.join(COMMANDS)}), which would shadow it"
        )
    if spec.name in TASKS:
        raise ValueError(f"task {spec.name!r} is already registered")
    TASKS[spec.name] = spec
    return spec


# ---------------------------------------------------------------------


def _suggest(name: str, choices: List[str]) -> str:
    close = difflib.get_close_matches(name, choices, n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def _variant_names(spec: ModelSpec) -> List[str]:
    return sorted(spec.load_variants())


def parse_model_name(name: str, task: Optional[str] = None) -> Tuple[str, str]:
    """Split ``"family/variant"`` and check both halves; see the module contract."""
    family, slash, variant = name.partition("/")
    if not slash or not family or not variant or "/" in variant:
        raise UnknownModelError(
            f"Invalid model {name!r}: write it as 'family/variant', for example "
            f"'vit/vit_pico'.{_suggest(name, sorted(MODELS))}"
        )
    spec = MODELS.get(family)
    if spec is None:
        known = ", ".join(sorted(MODELS)) or "none registered"
        raise UnknownModelError(
            f"Unknown model family {family!r} in {name!r} (families: {known})."
            f"{_suggest(family, sorted(MODELS))}"
        )
    # DECISION plan-2026-10-31T180033-2faf890a/D-072: the task is checked HERE, before the variants
    # table is imported and so before any model code is. Do NOT leave it to the task's own
    # resolver: a segmentation family handed to the classifier would reach its build as "an
    # unknown key" or a shape error, after the user has waited for the data.
    if task is not None and task not in spec.tasks:
        siblings = sorted(other for other, row in MODELS.items() if task in row.tasks)
        raise ConfigError(
            f"Invalid value for 'model': the family {family!r} trains {' and '.join(spec.tasks)}, "
            f"not {task!r}; the families of {task!r} are "
            f"{', '.join(siblings) if siblings else 'none registered'}"
        )
    variants = _variant_names(spec)
    if variant not in variants:
        hint = _suggest(variant, variants)
        raise UnknownModelError(
            f"Unknown variant {variant!r} for family {family!r}.{hint} "
            f"Run 'python -m train list_models' for the variants."
        )
    return family, variant


def list_tasks() -> List[TaskSpec]:
    """Return the registered tasks sorted by name; see the module contract."""
    return [TASKS[name] for name in sorted(TASKS)]


def list_models(task: Optional[str] = None) -> List[str]:
    """Return sorted ``family/variant`` names; see the module contract."""
    if task is not None and task not in TASKS:
        known = ", ".join(sorted(TASKS)) or "none registered"
        raise ConfigError(
            f"Unknown task {task!r} (tasks: {known}).{_suggest(task, sorted(TASKS))}"
        )
    names: List[str] = []
    for family in sorted(MODELS):
        spec = MODELS[family]
        if task is None or task in spec.tasks:
            names.extend(f"{family}/{variant}" for variant in _variant_names(spec))
    return names
