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

``parse_model_name(name)``
    :param name: ``"family/variant"``, e.g. ``"vit/vit_pico"``.
    :return: ``(family, variant)``.
    :raises UnknownModelError: for a missing slash, an unknown family or an
        unknown variant; the message carries a ``difflib`` suggestion. The
        variant check imports the family's variants table.

``list_models(task=None)``
    :param task: keep only families that list this task; ``None`` keeps all.
    :return: sorted ``"family/variant"`` names. Imports every matching family's
        variants table, so it is not on the help path.
    :raises ConfigError: when ``task`` names no registered task.

``list_tasks()``
    :return: the registered :class:`TaskSpec` rows, sorted by name.

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
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .errors import ConfigError, UnknownModelError

# ---------------------------------------------------------------------

_ATTR_PATH = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")

#: Keys of ``ModelSpec.defaults`` that hold overlays, not recipe values.
OVERLAY_KEYS = ("per_dataset", "per_variant")


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
    """

    family: str
    factory: str
    variants_attr: str
    tasks: Tuple[str, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)
    resolve_model_args: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.family or "/" in self.family:
            raise ValueError(f"ModelSpec.family must be non-empty without '/', got {self.family!r}")
        _require_attr_path("ModelSpec", "factory", self.factory)
        _require_attr_path("ModelSpec", "variants_attr", self.variants_attr)
        if self.resolve_model_args is not None:
            _require_attr_path("ModelSpec", "resolve_model_args", self.resolve_model_args)
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError(f"ModelSpec.tasks must be a non-empty tuple, got {self.tasks!r}")

    def recipe(self, dataset: str, variant: str) -> Dict[str, Any]:
        """Return the recipe for one dataset and variant, overlays applied.

        :param dataset: dataset name, e.g. ``"cifar10"``.
        :param variant: variant name, e.g. ``"cn_64_5"``.
        :return: a deep copy of :attr:`defaults` without its ``per_dataset``
            and ``per_variant`` keys, with ``per_dataset[dataset]`` and then
            ``per_variant[variant]`` laid over it. An overlay entry that is a
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
        for table, name in (("per_dataset", dataset), ("per_variant", variant)):
            for key, value in self.defaults.get(table, {}).get(name, {}).items():
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


def parse_model_name(name: str) -> Tuple[str, str]:
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
