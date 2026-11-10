"""The image data of a classification run, as three configs and their loaders.

``BuiltinImageData`` names one of the five datasets the repository's shared
ingestion path serves (``train.common.classification_data``) and the options of
its selection protocol: a held-out slice of train selects, the test split is only
scored. ``FolderImageData`` (one directory of images per class) and
``CsvImageData`` (a CSV file listing image paths and labels) take the same
protocol from files you have. The ``format`` key says which one a mapping is.
Every option that does not apply to the chosen dataset is REFUSED, never
ignored, as the legacy trainers do.

Interface contract
------------------
``BuiltinImageData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; the
    other fields default to ``None`` (the loader's own default).
    ``problems(prefix)`` names every option that does not fit the dataset.

``FolderImageData`` / ``CsvImageData``
    Frozen configs with ``format`` ``"folder"`` / ``"csv"``. ``train_dir`` /
    ``train_csv`` is required; the test directory or file is optional (a run
    with none skips test metrics, figures, conformal and the model analysis, and
    its summary says so). ``max_samples`` is the smoke cap. A relative path is
    relative to the directory of the YAML file that holds it, or to the working
    directory for a mapping or the command line. ``transform_args.image_size``
    must be set: files have no native size.

``ImageData``
    The union of the three; the type of the task config's ``data`` field.

``load_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML
        file holding one. A mapping without ``format`` is a ``builtin`` one.
    :return: the validated data config.
    :raises ConfigError: for an unknown ``format``, a file that cannot be read,
        or any invalid key (every problem at once, as ``validate`` reports it;
        a file's lines name the file).

``read_data_file(path)``
    :return: the mapping in a YAML file, its relative ``train_dir``,
        ``test_dir``, ``train_csv`` and ``test_csv`` made absolute against the
        file's directory. The command line's ``data=data.yaml`` uses it too.

``DataFacts``
    Frozen record of what is known about the data BEFORE it is loaded:
    ``dataset`` (the dataset name, or ``"folder"`` / ``"csv"``), ``num_classes``
    and ``image_size``. The family recipe and the ``"auto"`` fields need these
    before the augmentation hook (a recipe decision) can be handed to the loader.

``image_facts(data, requested_image_size)``
    :param data: any of the three configs.
    :param requested_image_size: the explicit ``transform_args.image_size``, or
        ``None`` for the dataset's own.
    :return: the :class:`DataFacts`. A folder or CSV source is LISTED (every class
        directory, column and file checked) but no image is read and no tensor is
        made, so it may run before ``setup_gpu``.
    :raises ConfigError: for an in-memory dataset asked for a size other than its
        native one (the loader keeps its figure arrays at the native size, so a
        resized model could not predict on them); for a folder or CSV source
        with no ``image_size``, with a class that has no image or only one, a missing
        column or file, test classes that are not the train classes, a file listed
        twice, under two classes (a symlink counts as the file it points to) or in both
        the train and the test split, or two csv labels of one integer value (``01`` and
        ``1``).

``load_image_data(data, *, batch_size, seed, augment, augment_from_mean, normalize, image_size, loader, eval_batch_size)``
    :param data: any of the three configs.
    :param batch_size: batch size of the three pipelines.
    :param seed: seed of the split, the shuffle and the augmentation.
    :param augment: the loader hook (``train.common.image_augment.build_augment``)
        or ``None``.
    :param augment_from_mean: file data only, instead of ``augment``: a function
        of the fit images' per-channel mean returning the hook. Built-in data has
        published statistics, so it takes ``augment``.
    :param normalize: ``"standardize"`` or ``"unit"``. File data standardizes with
        the mean and standard deviation of one pass over its fit split.
    :param image_size: the square input side; used for imagenet, imagenette and
        file data.
    :param loader: the recipe's ``loader`` mapping, whose keys are arguments of
        ``load_classification_data`` the family decides (``channels``,
        ``imagenet_augment_color``); file data takes none.
    :param eval_batch_size: ``None`` (the loader's default, 256) or the rows per batch of the
        selection and test splits of the in-memory datasets (``analysis_args.eval_batch_size``).
        ImageNet, Imagenette and file data batch those splits at ``batch_size`` and ignore it.
    :return: the ``ClassificationData`` of the shared loader.
    :raises ConfigError: for a ``loader`` key that is not in :data:`LOADER_KEYS`
        (a typo in a recipe) or that file data cannot take, or an option the
        loader refuses for this data.

``builtin_facts`` / ``load_builtin_data``
    The built-in halves of the two functions above. ``builtin_facts`` also calls
    ``check_builtin_sources``.

``check_builtin_sources(data)``
    :param data: a :class:`BuiltinImageData`.
    :raises ConfigError: for ``imagenet`` when the root (``data.imagenet_root`` or the loader's
        default) is not a directory, has no ``train/`` or no ``val/``, or the devkit
        (``data.devkit_dir`` or ``<root>/devkit``, then its ``data/``) lacks ``meta.mat`` or
        ``ILSVRC2012_validation_ground_truth.txt``; for ``imagenette`` when no
        ``imagenette/320px-v2/<version>/dataset_info.json`` is under ``$TFDS_DATA_DIR`` (or
        ``~/tensorflow_datasets``). The message names the key and the path looked at.
        ``Path`` tests only: no TFDS import, no tensor, nothing opened or written, so it runs
        before ``open_run``. Other datasets return.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); the
dataset table and the loader are imported inside the functions.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union

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

#: Keys of a file-based data config that name a path; a YAML file's relative ones
#: are made absolute against the file (:func:`read_data_file`).
PATH_KEYS = ("train_dir", "test_dir", "train_csv", "test_csv")

_DatasetName = Literal[DATASETS]  # type: ignore[valid-type]


@dataclasses.dataclass(frozen=True)
class DataFacts:
    """What is known about the data before it is loaded.

    :param image_size: the side of a classification image; ``None`` for a segmentation run, whose
        size is a ``(height, width)`` pair that the task keeps (``AutoContext.image_size`` is then
        ``None`` too, and nothing the segmentation recipes resolve reads it).
    """

    dataset: str
    num_classes: int
    image_size: Optional[int]


@dataclasses.dataclass(frozen=True)
class BuiltinImageData(StrictConfig):
    """A dataset of the shared loader and its selection options."""

    # DECISION plan-2026-10-31T180033-2faf890a/D-037: the help names the one key that bounds the
    # lost epoch, `checkpoint_args.backup_every_steps` (D-038 added it; the knob guard refuses a
    # key the config lacks, so do NOT name a `checkpoint_args.*` key here that is not a field).
    # The rates are the lowest and highest cold ImageNet training row of
    # tests/test_train/test_api/memory_fit_evidence.json; the 4.6 and 6.7 hours, the 9.6 and 19.1
    # days are the README's probe and its one measured epoch. All are pinned by
    # test_the_help_and_the_commands_state_the_imagenet_cost.py: change them together.
    dataset: _DatasetName = dataclasses.field(
        metadata={
            "help": "mnist, cifar10, cifar100, imagenette or imagenet. imagenet is read from "
            "the data disk, a spinning disk on the development machine: 96 to 107 images per "
            "second were measured for the first read of files in training runs (44 to 115 for "
            "cold single-file reads), a disk-read bound of 3 to 4 hours for one epoch of its "
            "1,271,167 images, at which the 50-epoch recipe takes about a week (100 epochs about "
            "two). The whole-train-set runs of resnet18 were slower: a 150-step probe projected "
            "4.6 hours of training per epoch (9.6 days for 50 epochs, 19.1 for 100), and the one "
            "measured epoch took about 6.7 hours of training on the RTX 4090 with CPU test jobs "
            "sharing the host. A killed epoch is lost; "
            "checkpoint_args.backup_every_steps bounds the loss to that many batches of training. "
            "src/train/README.md has the numbers and the exact 50 and 100 epoch commands, which "
            "were not run"
        }
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
        metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"},
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


def _file_problems(data: Any, prefix: str, text_keys: Tuple[str, ...]) -> List[str]:
    """Name every file-data option out of range, at its dotted path; see the two classes.

    :param text_keys: the fields that must be non-empty text when they are not ``None``.
    """
    lines: List[str] = []

    def bad(name: str, why: str) -> None:
        lines.append(
            f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(data, name)!r}"
        )

    split = data.validation_split
    if split is not None and not 0 < split < 1:
        bad("validation_split", "expected a fraction strictly between 0 and 1")
    if data.max_samples is not None and data.max_samples < 1:
        bad("max_samples", "expected a positive int or null")
    if data.eval_samples is not None and data.eval_samples < 0:
        bad("eval_samples", "expected an int >= 0")
    for name in text_keys:
        value = getattr(data, name)
        if value is not None and not value.strip():
            bad(name, "expected a non-empty text")
    return lines


#: The help of ``fingerprint`` on folder and CSV data (one text for both).
_FINGERPRINT_HELP = (
    "what the run records to tell, at a resume, that the files are the same: 'listing' is every "
    "file's relative path, class and size, in the order they are listed (no file is opened; a CSV "
    "whose rows were reordered is another listing); 'content' also hashes every file's "
    "bytes, so a swapped or edited file of the same size is caught, at the cost of reading all "
    "the data once (a pass over the whole disk tree, minutes on a large one on a spinning disk; "
    "the seconds are logged)"
)


@dataclasses.dataclass(frozen=True)
class FolderImageData(StrictConfig):
    """A directory of images per class: ``train_dir/<class>/*.png``, and optionally a test one."""

    train_dir: str = dataclasses.field(
        metadata={"help": "root holding one directory of images per class (png, jpg, jpeg, bmp)"}
    )
    format: Literal["folder"] = dataclasses.field(
        default="folder", metadata={"help": "where the data comes from"}
    )
    test_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "root with the same class directories, only scored; null means no test "
            "split: test metrics, figures, conformal and the model analysis are skipped"
        },
    )
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "fraction of train held out to select the best epoch, in (0, 1); "
            "null is the loader's 0.1"
        },
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"},
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "test images kept for figures and analysis; null is the loader's default"},
    )
    fingerprint: Literal["listing", "content"] = dataclasses.field(
        default="listing",
        metadata={"help": _FINGERPRINT_HELP},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        return _file_problems(self, prefix, ("train_dir", "test_dir"))


@dataclasses.dataclass(frozen=True)
class CsvImageData(StrictConfig):
    """A CSV file listing image paths and labels, and optionally a test one."""

    train_csv: str = dataclasses.field(
        metadata={"help": "CSV with one row per image; a relative image path starts at the CSV's directory"}
    )
    format: Literal["csv"] = dataclasses.field(
        default="csv", metadata={"help": "where the data comes from"}
    )
    test_csv: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "CSV of the test images with the same classes, only scored; null means no "
            "test split: test metrics, figures, conformal and the model analysis are skipped"
        },
    )
    path_column: str = dataclasses.field(
        default="path", metadata={"help": "column holding the image path"}
    )
    label_column: str = dataclasses.field(
        default="label", metadata={"help": "column holding the class"}
    )
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "fraction of train held out to select the best epoch, in (0, 1); "
            "null is the loader's 0.1"
        },
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"},
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "test images kept for figures and analysis; null is the loader's default"},
    )
    fingerprint: Literal["listing", "content"] = dataclasses.field(
        default="listing",
        metadata={"help": _FINGERPRINT_HELP},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        return _file_problems(
            self, prefix, ("train_csv", "test_csv", "path_column", "label_column")
        )


#: The ``format`` values and the class each one builds, in the order of :data:`ImageData`.
FORMATS: Dict[str, Type[StrictConfig]] = {
    "builtin": BuiltinImageData,
    "folder": FolderImageData,
    "csv": CsvImageData,
}

#: The type of the task config's ``data`` field. The first member is the default format.
ImageData = Union[BuiltinImageData, FolderImageData, CsvImageData]


def read_data_file(path: Union[str, Path], path_keys: Tuple[str, ...] = PATH_KEYS) -> Dict[str, Any]:
    """Return the mapping in the YAML file ``path``, its relative paths anchored to the file.

    :param path_keys: the keys that name a path; the segmentation data classes pass their own
        (``train.api.segmentation_args.SEG_PATH_KEYS``).
    :raises ConfigError: from ``load_yaml_mapping`` (unreadable, malformed, not a mapping).
    """
    mapping = dict(load_yaml_mapping(path))
    base = Path(path).resolve().parent
    for key in path_keys:
        value = mapping.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            mapping[key] = os.path.normpath(base / value)
    return mapping


def load_data_config(data: Any) -> ImageData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    if isinstance(data, tuple(FORMATS.values())):
        return data
    source = ""
    if isinstance(data, (str, Path)):
        source = f"data file {str(data)!r}: "
        data = read_data_file(data)
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Invalid value for 'data': expected a mapping or the path of a YAML "
            f"file, got {type(data).__name__}"
        )
    given = dict(data)
    given.setdefault("format", "builtin")
    kind = given["format"]
    if not isinstance(kind, str) or kind not in FORMATS:
        raise ConfigError(
            f"Invalid value for 'data.format': {kind!r} is not a data "
            f"format; choose one of {', '.join(FORMATS)}"
        )
    try:
        return validate(FORMATS[kind], given)
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
    check_builtin_sources(data)
    return DataFacts(dataset=data.dataset, num_classes=spec.num_classes, image_size=size)


def _tfds_data_dir() -> Path:
    """Return the directory TFDS reads when no ``data_dir`` is given, without importing it.

    ``$TFDS_DATA_DIR`` when the variable is set, EVEN EMPTY (TFDS then reads ``.``), with a leading
    ``~`` expanded as TFDS expands it (``TFDS_DATA_DIR='~/t'`` is ``$HOME/t``); else
    ``~/tensorflow_datasets``. TFDS's own answer is pinned in
    ``test_the_pre_claim_tfds_directory_is_the_one_tfds_reads.py``.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-065
    # `expanduser`, and `is not None` rather than a truthiness test: TFDS reads `~/t` as
    # `$HOME/t` and `''` as `.` (list_data_dirs). Do NOT return `Path(given)` for a tilde (a
    # false refusal of a layout the loader reads, review-iter-2 concern 3) and do NOT treat an
    # empty variable as unset (mutant da_empty_var: the default directory is then looked in,
    # which TFDS does not do).
    given = os.environ.get("TFDS_DATA_DIR")
    if given is not None:
        return Path(os.path.expanduser(given))
    return Path(os.path.expanduser("~")) / "tensorflow_datasets"


def check_builtin_sources(data: BuiltinImageData) -> None:
    """Refuse an ImageNet tree or Imagenette record set that is not on disk; see the module contract.

    Interface contract: ``data`` is any built-in data config. In-memory datasets download
    through Keras and have nothing to look for, so they return. Only ``Path`` tests run: no
    TFDS import, no tensor, no file is opened or written, so it is safe before ``open_run`` and
    before ``setup_gpu``. Each test is the one the loader makes first, so a tree that passes
    here is the tree the loader opens (a missing ``val/`` or devkit file is a
    ``FileNotFoundError`` there, after the claim and with exit 1).

    :param data: the data config.
    :raises ConfigError: naming the dotted key and the path looked at.
    """
    if data.dataset == "imagenet":
        _check_imagenet_tree(data)
    elif data.dataset == "imagenette":
        _check_imagenette_records()


def _check_imagenet_tree(data: BuiltinImageData) -> None:
    """Refuse a missing ImageNet root, ``train/``, ``val/`` or devkit file; see :func:`check_builtin_sources`."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-034: path tests only, and exactly the loader's
    # own. Do NOT import ``train.common.imagenet`` or call the loader: they open files and the
    # first dataset op creates TensorFlow's devices before ``setup_gpu``. Do NOT test more than
    # the loader needs (an image count, a class count): a false refusal of the real tree is the
    # worst failure here.
    from dl_techniques.datasets.vision import imagenet2012 as loader

    key = "data.imagenet_root"
    root = loader.IMAGENET2012_DEFAULT_ROOT if data.imagenet_root is None else data.imagenet_root
    origin = "" if data.imagenet_root is not None else " (null: the loader's default)"
    if not os.path.isdir(root):
        raise ConfigError(
            f"Invalid value for {key!r}: the ImageNet root {root!r}{origin} is not a directory; "
            "point it at the extracted ILSVRC2012 tree (train/<wnid>/, val/, devkit/)"
        )
    for split in ("train", "val"):
        if not os.path.isdir(os.path.join(root, split)):
            raise ConfigError(
                f"Invalid value for {key!r}: {os.path.join(root, split)!r} is not a directory; "
                f"the ImageNet root {root!r}{origin} must hold {split}/ "
                "(every run reads train for the classes and val as the test split)"
            )
    _check_imagenet_not_empty(root, key)
    devkit = data.devkit_dir if data.devkit_dir is not None else os.path.join(root, "devkit")
    missing = [
        name
        for name in (loader._META_FILE, loader._VAL_GROUND_TRUTH_FILE)
        if loader._find_devkit_file(devkit, name) is None
    ]
    if missing:
        raise ConfigError(
            f"Invalid value for 'data.devkit_dir': {', '.join(missing)} not found in {devkit!r} "
            f"or {os.path.join(devkit, 'data')!r}"
            + ("" if data.devkit_dir is not None else " (null: <imagenet_root>/devkit)")
            + "; the validation labels need the extracted ILSVRC2012 devkit"
        )


def _holds_an_image(directory: str, suffixes: Tuple[str, ...]) -> bool:
    """Whether ``directory`` has a file with one of ``suffixes`` directly in it (stops at the first)."""
    with os.scandir(directory) as entries:
        return any(entry.is_file() and entry.name.lower().endswith(suffixes) for entry in entries)


def _check_imagenet_not_empty(root: str, key: str) -> None:
    """Refuse an ImageNet ``train/`` with no class folder or no image, and a ``val/`` with no image.

    The loader's own first failures (``_list_classes``: "No class folders under train";
    ``_load_or_build_index``: "No images found under ..."), seen here before the claim. It looks only
    as far as the loader's scan would have to: ``train/`` needs one class folder and ONE image in
    some class folder, ``val/`` one image directly in it (a subfolder is not scanned), found by
    ``os.scandir`` that stops at the first. An unreadable directory is named, not a raw
    ``PermissionError``.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-065
    # The loader's own first failures, seen before the claim, and no further: ONE class folder, ONE
    # image in some class folder, ONE image in val/. Do NOT count images or classes (a false refusal
    # of the real tree is the worst failure, and `os.scandir` that stops at the first hit costs one
    # directory listing), and do NOT look inside sub-folders of val/ (the loader's scan does not).
    from dl_techniques.datasets.vision import imagenet2012 as loader

    suffixes = loader._IMAGE_SUFFIXES
    train, val = os.path.join(root, "train"), os.path.join(root, "val")
    try:
        with os.scandir(train) as entries:
            classes = sorted(entry.name for entry in entries if entry.is_dir())
        if not classes:
            raise ConfigError(
                f"Invalid value for {key!r}: {train!r} has no class folder; the loader takes the "
                "sorted wnid folders under train/ as the classes"
            )
        if not any(_holds_an_image(os.path.join(train, name), suffixes) for name in classes):
            raise ConfigError(
                f"Invalid value for {key!r}: none of the class folders under {train!r} holds an "
                f"image ({', '.join(suffixes)}), so there is no training data: no image was found"
            )
        if not _holds_an_image(val, suffixes):
            raise ConfigError(
                f"Invalid value for {key!r}: {val!r} holds no image ({', '.join(suffixes)}) directly "
                "in it, and the validation images are the test split; no image was found"
            )
    except OSError as exc:
        raise ConfigError(
            f"Invalid value for {key!r}: {exc.filename or root!r} cannot be read "
            f"({exc.strerror or exc}); the ImageNet tree must be readable"
        ) from exc


def _check_imagenette_records() -> None:
    """Refuse an Imagenette ``320px-v2`` record set that is not under the TFDS directory."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-034: look for a version directory that holds
    # ``dataset_info.json`` and do NOT name a version: the builder's version is only known by
    # importing TFDS, which this check must not do. The loader still checks the exact one.
    from dl_techniques.datasets.vision.imagenette import (
        IMAGENETTE_TFDS_VERSION,
        find_prepared_records,
    )

    root = _tfds_data_dir()
    origin = (
        f"TFDS directory {str(root)!r} from "
        f"{'$TFDS_DATA_DIR' if 'TFDS_DATA_DIR' in os.environ else 'the default ~/tensorflow_datasets'}"
    )
    try:
        here, found = find_prepared_records(root)
    except OSError as exc:
        raise ConfigError(
            f"Invalid value for 'data.dataset': imagenette reads TFDS records, and "
            f"{exc.filename or str(root)!r} cannot be read ({exc.strerror or exc}) ({origin}); "
            "fix the permission, or set TFDS_DATA_DIR to a directory you can read"
        ) from exc
    if not found:
        try:
            holds = sorted(child.name for child in here.iterdir()) if here.is_dir() else []
        except OSError:
            holds = []
        present = f"; that directory holds {', '.join(holds)}" if holds else ""
        raise ConfigError(
            f"Invalid value for 'data.dataset': imagenette reads the TFDS records of "
            f"{here.relative_to(root).as_posix()} version {IMAGENETTE_TFDS_VERSION}, and none are "
            f"under {str(here / IMAGENETTE_TFDS_VERSION)!r}{present} ({origin}); "
            "nothing is downloaded. Set TFDS_DATA_DIR to the directory that holds imagenette/"
        )


def load_builtin_data(
    data: BuiltinImageData,
    *,
    batch_size: int,
    seed: int,
    augment: Any,
    normalize: str,
    image_size: int,
    loader: Optional[Mapping[str, Any]] = None,
    eval_batch_size: Optional[int] = None,
) -> Any:
    """Load ``data`` through the shared ingestion path; see the module contract.

    :param eval_batch_size: rows per batch of the selection and test splits of the in-memory
        datasets (``analysis_args.eval_batch_size``); ``None`` keeps the loader's default. The
        other datasets batch those splits at ``batch_size`` and ignore it.
    """
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
    if data.dataset in IN_MEMORY_DATASETS and eval_batch_size is not None:
        extra["eval_batch_size"] = eval_batch_size
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


def _list_files(data: Union[FolderImageData, CsvImageData]) -> Any:
    """Return the :class:`~train.common.classification_data.FileListing` of file data.

    Lists the files only: no image is read and no tensor is made.

    :raises ConfigError: naming the dotted key of the first thing wrong.
    """
    from train.common.classification_data import DataSourceError, read_image_csv, scan_image_folder

    try:
        if isinstance(data, FolderImageData):
            return scan_image_folder(data.train_dir, data.test_dir)
        return read_image_csv(
            data.train_csv,
            data.test_csv,
            path_column=data.path_column,
            label_column=data.label_column,
        )
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc


def file_facts(
    data: Union[FolderImageData, CsvImageData], requested_image_size: Optional[int]
) -> DataFacts:
    """Return the class count and input size of file data; see :func:`image_facts`."""
    if requested_image_size is None:
        raise ConfigError(
            f"Invalid value for 'transform_args.image_size': {data.format} images have no "
            "native size, so the input side must be given, e.g. transform_args.image_size=224"
        )
    listing = _list_files(data)
    return DataFacts(
        dataset=data.format, num_classes=listing.num_classes, image_size=requested_image_size
    )


def image_facts(data: ImageData, requested_image_size: Optional[int]) -> DataFacts:
    """Return the facts of any data config before it is loaded; see the module contract."""
    if data.format == "builtin":
        return builtin_facts(data, requested_image_size)
    return file_facts(data, requested_image_size)


def load_file_data(
    data: Union[FolderImageData, CsvImageData],
    *,
    batch_size: int,
    seed: int,
    augment: Any,
    augment_from_mean: Any,
    normalize: str,
    image_size: int,
    loader: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Load folder or CSV data through the shared ingestion path; see the module contract."""
    from train.common.classification_data import (
        DEFAULT_EVAL_SAMPLES,
        DataSourceError,
        load_csv_data,
        load_folder_data,
    )

    if loader:
        raise ConfigError(
            f"Family recipe for 'loader': {', '.join(map(repr, loader))} applies to the "
            f"built-in datasets only, not to {data.format} data"
        )
    shared = dict(
        batch_size=batch_size,
        seed=seed,
        image_size=image_size,
        validation_split=data.validation_split,
        eval_samples=DEFAULT_EVAL_SAMPLES if data.eval_samples is None else data.eval_samples,
        normalize=normalize,
        augment=augment,
        augment_from_mean=augment_from_mean,
        max_samples=data.max_samples,
        max_samples_flag="data.max_samples",
        fingerprint=data.fingerprint,
    )
    try:
        if isinstance(data, FolderImageData):
            return load_folder_data(data.train_dir, test_dir=data.test_dir, **shared)
        return load_csv_data(
            data.train_csv,
            test_csv=data.test_csv,
            path_column=data.path_column,
            label_column=data.label_column,
            **shared,
        )
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc


def load_image_data(
    data: ImageData,
    *,
    batch_size: int,
    seed: int,
    augment: Any,
    normalize: str,
    image_size: int,
    loader: Optional[Mapping[str, Any]] = None,
    augment_from_mean: Any = None,
    eval_batch_size: Optional[int] = None,
) -> Any:
    """Load any data config through the shared ingestion path; see the module contract."""
    if data.format == "builtin":
        if augment_from_mean is not None:
            raise ConfigError(
                "augment_from_mean applies to folder and csv data; the built-in datasets "
                "have published statistics, so pass the built hook as augment"
            )
        return load_builtin_data(
            data,
            batch_size=batch_size,
            seed=seed,
            augment=augment,
            normalize=normalize,
            image_size=image_size,
            loader=loader,
            eval_batch_size=eval_batch_size,
        )
    return load_file_data(
        data,
        batch_size=batch_size,
        seed=seed,
        augment=augment,
        augment_from_mean=augment_from_mean,
        normalize=normalize,
        image_size=image_size,
        loader=loader,
    )
