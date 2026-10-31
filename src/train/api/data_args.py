"""The built-in datasets of an image-classification run, as a config and a loader.

``BuiltinImageData`` names one of the five datasets the repository's shared
ingestion path serves (``train.common.classification_data``) and the options of
its selection protocol: a held-out slice of train selects, the test split is only
scored. Every option that does not apply to the chosen dataset is REFUSED, never
ignored, as the legacy trainers do.

Interface contract
------------------
``BuiltinImageData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; the
    other fields default to ``None`` (the loader's own default).
    ``problems(prefix)`` names every option that does not fit the dataset.

``load_data_config(data)``
    :param data: a ``BuiltinImageData`` (returned as is), a mapping, or the path
        of a YAML file holding one. A mapping without ``format`` is a
        ``builtin`` one.
    :return: the validated data config.
    :raises ConfigError: for an unknown ``format``, a file that cannot be read,
        or any invalid key (every problem at once, as ``validate`` reports it;
        a file's lines name the file).

``DataFacts``
    Frozen record of what is known about the data BEFORE it is loaded:
    ``dataset``, ``num_classes`` and ``image_size``. The family recipe and the
    ``"auto"`` fields need these before the augmentation hook (a recipe
    decision) can be handed to the loader.

``builtin_facts(data, requested_image_size)``
    :param data: a :class:`BuiltinImageData`.
    :param requested_image_size: the explicit ``transform_args.image_size``, or
        ``None`` for the dataset's own.
    :return: the :class:`DataFacts`.
    :raises ConfigError: when an in-memory dataset (mnist, cifar) is asked for a
        size other than its native one: the loader keeps its figure arrays at the
        native size, so a resized model could not predict on them.

``load_builtin_data(data, *, batch_size, seed, augment, normalize, image_size, loader)``
    :param data: a :class:`BuiltinImageData`.
    :param batch_size: batch size of the three pipelines.
    :param seed: seed of the split, the shuffle and the augmentation.
    :param augment: the loader hook (``train.common.image_augment.build_augment``)
        or ``None``.
    :param normalize: ``"standardize"`` or ``"unit"``.
    :param image_size: the square input side; used for imagenet and imagenette.
    :param loader: the recipe's ``loader`` mapping, whose keys are arguments of
        ``load_classification_data`` the family decides (``channels``,
        ``imagenet_augment_color``).
    :return: the ``ClassificationData`` of the shared loader.
    :raises ConfigError: for a ``loader`` key that is not in :data:`LOADER_KEYS`
        (a typo in a recipe), or an option the loader refuses for this dataset.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); the
dataset table and the loader are imported inside the functions.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple

from .config import StrictConfig, join_path, load_yaml_mapping, validate
from .errors import ConfigError, ConfigValidationError

# ---------------------------------------------------------------------

#: The dataset names. ``test_data_args`` holds them equal to the keys of
#: ``dl_techniques.datasets.vision.classification.CLASSIFICATION_DATASETS``, which
#: cannot be imported here (it loads TensorFlow).
DATASETS: Tuple[str, ...] = ("mnist", "cifar10", "cifar100", "imagenette", "imagenet")

#: Datasets read whole into memory; they are never resized.
IN_MEMORY_DATASETS = ("mnist", "cifar10", "cifar100")

#: The arguments of ``load_classification_data`` a recipe's ``loader`` may set.
LOADER_KEYS = ("channels", "imagenet_augment_color")

#: The ``format`` values and the class each one builds. Later formats register here.
FORMATS = ("builtin",)

_DatasetName = Literal[DATASETS]  # type: ignore[valid-type]


@dataclasses.dataclass(frozen=True)
class DataFacts:
    """What is known about the data before it is loaded."""

    dataset: str
    num_classes: int
    image_size: int


@dataclasses.dataclass(frozen=True)
class BuiltinImageData(StrictConfig):
    """A dataset of the shared loader and its selection options."""

    dataset: _DatasetName = dataclasses.field(
        metadata={"help": "mnist, cifar10, cifar100, imagenette or imagenet"}
    )
    format: Literal["builtin"] = dataclasses.field(
        default="builtin", metadata={"help": "where the data comes from"}
    )
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "fraction of train held out to select the best epoch, in (0, 1); "
            "mnist, cifar and imagenette only; null is the loader's 0.1"
        },
    )
    holdout_per_class: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "selection images per class taken out of train; imagenet only; "
            "null is the loader's default"
        },
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "cap on the train pool and the test split; null is no cap"},
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "test images kept for figures and analysis; imagenet and "
            "imagenette only; null is the loader's default"
        },
    )
    imagenet_root: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "extracted ILSVRC2012 tree; imagenet only; null is the loader's"},
    )
    devkit_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "ILSVRC2012 devkit; imagenet only; null is the loader's"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option that does not fit ``dataset``, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(
                f"Invalid value for {join_path(prefix, name)!r}: {why}, "
                f"got {getattr(self, name)!r}"
            )

        in_memory = self.dataset in IN_MEMORY_DATASETS
        imagenet = self.dataset == "imagenet"
        split = self.validation_split
        if split is not None:
            if imagenet:
                bad("validation_split", "it does not apply to imagenet (use holdout_per_class)")
            elif not 0 < split < 1:
                bad("validation_split", "expected a fraction strictly between 0 and 1")
        if self.holdout_per_class is not None:
            if not imagenet:
                bad("holdout_per_class", f"it applies to imagenet only, not {self.dataset}")
            elif self.holdout_per_class < 1:
                bad("holdout_per_class", "expected a positive int")
        if self.max_samples is not None and self.max_samples < 1:
            bad("max_samples", "expected a positive int or null")
        if self.eval_samples is not None:
            if in_memory:
                bad("eval_samples", f"it applies to imagenet and imagenette only, not {self.dataset}")
            elif self.eval_samples < 0:
                bad("eval_samples", "expected an int >= 0")
        for name in ("imagenet_root", "devkit_dir"):
            if getattr(self, name) is not None and not imagenet:
                bad(name, f"it applies to imagenet only, not {self.dataset}")
        return lines


def load_data_config(data: Any) -> BuiltinImageData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    if isinstance(data, BuiltinImageData):
        return data
    source = ""
    if isinstance(data, (str, Path)):
        source = f"data file {str(data)!r}: "
        data = load_yaml_mapping(data)
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Invalid value for 'data': expected a mapping or the path of a YAML "
            f"file, got {type(data).__name__}"
        )
    given = dict(data)
    given.setdefault("format", "builtin")
    if given["format"] not in FORMATS:
        raise ConfigError(
            f"Invalid value for 'data.format': {given['format']!r} is not a data "
            f"format; choose one of {', '.join(FORMATS)}"
        )
    try:
        return validate(BuiltinImageData, given)
    except ConfigValidationError as exc:
        if not source:
            raise
        raise ConfigValidationError([source + line for line in exc.errors]) from exc


def builtin_facts(data: BuiltinImageData, requested_image_size: Optional[int]) -> DataFacts:
    """Return the class count and input size of ``data`` without loading it."""
    from dl_techniques.datasets.vision.classification import get_spec

    spec = get_spec(data.dataset)
    native = spec.input_shape[0]
    size = native if requested_image_size is None else requested_image_size
    if spec.in_memory and size != native:
        raise ConfigError(
            f"Invalid value for 'transform_args.image_size': {data.dataset} images are "
            f"{native} px and in-memory datasets are not resized, got {size!r}; "
            "leave it at 'auto'"
        )
    return DataFacts(dataset=data.dataset, num_classes=spec.num_classes, image_size=size)


def load_builtin_data(
    data: BuiltinImageData,
    *,
    batch_size: int,
    seed: int,
    augment: Any,
    normalize: str,
    image_size: int,
    loader: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Load ``data`` through the shared ingestion path; see the module contract."""
    from train.common.classification_data import DEFAULT_EVAL_SAMPLES, load_classification_data

    options = dict(loader or {})
    unknown = [key for key in options if key not in LOADER_KEYS]
    if unknown:
        raise ConfigError(
            f"Family recipe for 'loader': {', '.join(map(repr, unknown))} is not an "
            f"option of the data loader (choose from {', '.join(LOADER_KEYS)})"
        )
    extra = {}
    if data.dataset not in IN_MEMORY_DATASETS:
        extra["image_size"] = image_size
        extra["eval_samples"] = (
            DEFAULT_EVAL_SAMPLES if data.eval_samples is None else data.eval_samples
        )
    if data.dataset == "imagenet":
        extra.update(
            holdout_per_class=data.holdout_per_class,
            imagenet_root=data.imagenet_root,
            devkit_dir=data.devkit_dir,
        )
    else:
        extra["validation_split"] = data.validation_split
    return load_classification_data(
        data.dataset,
        batch_size=batch_size,
        seed=seed,
        normalize=normalize,
        augment=augment,
        max_samples=data.max_samples,
        max_samples_flag="data.max_samples",
        **options,
        **extra,
    )
