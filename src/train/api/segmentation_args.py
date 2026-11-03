"""The segmentation arguments of ``python -m train``: data configs (and, later, transform, metric and loss groups).

``BuiltinSegData`` names Oxford-IIIT Pet from the local TFDS cache; ``FolderSegData`` (two
directories, images and masks, joined by file stem) and ``CsvSegData`` (a CSV file with an
``image`` and a ``mask`` column) take the same selection protocol from files you have: a held-out
slice of train selects, the test split is only scored. The ``format`` key says which one a mapping
is. Every option that does not apply to the chosen source is REFUSED, never ignored.

Interface contract
------------------
``BuiltinSegData``
    Frozen :class:`~train.api.config.StrictConfig`; ``dataset`` is ``oxford_pet`` (the TFDS record
    ``oxford_iiit_pet:4.*.*``, mask minus 1, classes ``pet, background, border``, no void). It has
    no class, ignore, threshold or fingerprint key: Pet fixes them.

``FolderSegData`` / ``CsvSegData``
    Frozen configs with ``format`` ``"folder"`` / ``"csv"``. ``train_images`` and ``train_masks`` /
    ``train_csv`` are required; the test pair of directories or the test file is optional (a run
    with none skips test metrics and figures, and its summary says so). The classes are STATED:
    ``class_names`` (one name per class) or ``num_classes`` (named ``"0"``, ``"1"``, ...), both
    allowed when they agree. ``ignore_index`` is the void value (a whole number outside the
    classes) or null. A relative path is relative to the directory of the YAML file that holds it,
    or to the working directory for a mapping or the command line. ``mask_threshold`` is reserved
    and refused in this version. ``transform_args.image_size`` must be set: files have no native
    size.

``SegData``
    The union of the three; the type of the segmentation task config's ``data`` field.

``load_seg_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one. A
        mapping without ``format`` is a ``builtin`` one.
    :return: the validated data config.
    :raises ConfigError: for an unknown ``format``, a file that cannot be read, or any invalid key
        (every problem at once; a file's lines name the file).

``read_seg_data_file(path)``
    :return: the mapping in a YAML file with its relative path keys made absolute against the file.

``SegFacts`` / ``seg_facts(data, requested_image_size)``
    :param data: any of the three configs.
    :param requested_image_size: the explicit ``(height, width)`` or ``None``.
    :return: the :class:`SegFacts` (dataset name, class count and names, ignore index, size)
        known BEFORE the data is loaded. A folder or CSV source is LISTED and a
        sample of its masks is read with PIL (no TensorFlow op), so it may run before ``setup_gpu``
        and before the run directory is claimed.
    :raises ConfigError: for a source that cannot be used (a missing directory or file, an unpaired
        stem, a file listed twice or in both splits, a mask that is a JPEG, a colour image, a pair
        of two sizes after the EXIF orientation, or a value that is neither a class nor
        ``ignore_index``), for file data with no size, and for a Pet request whose arrays could
        not be embedded in a TensorFlow graph. Every message names the dotted key.

``check_seg_sources(data)``
    Pet only: ``Path`` tests for ``dataset_info.json`` under ``$TFDS_DATA_DIR/oxford_iiit_pet/4.*.*``.
    No TFDS import, nothing opened or written; other sources return.

``load_seg_data(data, *, batch_size, seed, image_size, augment, eval_batch_size)``
    :return: the :class:`~train.common.segmentation_data.SegmentationData` of the shared loader.
    :raises ConfigError: as :func:`seg_facts`, for the files the run will use (all of them are
        checked here, before ``fit``, and the duration is logged).

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); the data module and its
TensorFlow import are inside the functions.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union

from .config import StrictConfig, join_path, validate
from .data_args import _tfds_data_dir, read_data_file
from .errors import ConfigError, ConfigValidationError

# ---------------------------------------------------------------------

#: The built-in segmentation datasets.
SEG_DATASETS: Tuple[str, ...] = ("oxford_pet",)

#: Keys of a file-based segmentation config that name a path (a YAML file's relative ones are
#: made absolute against the file).
SEG_PATH_KEYS = ("train_images", "train_masks", "test_images", "test_masks", "train_csv", "test_csv")

#: The TFDS directory and version glob of Oxford-IIIT Pet (``train.common.segmentation_data.PET_TFDS_NAME``).
PET_TFDS_DIRECTORY = "oxford_iiit_pet"

_SegDatasetName = Literal[SEG_DATASETS]  # type: ignore[valid-type]

#: The help of ``fingerprint`` on folder and CSV segmentation data (one text for both).
_FINGERPRINT_HELP = (
    "what the run records to tell, at a resume, that the files are the same: 'listing' is every "
    "pair's relative image path, relative mask path and the two sizes, in the order they are listed "
    "(no file is opened; a CSV whose rows were reordered is another listing); 'content' also "
    "hashes the bytes of every image and mask, so a swapped or edited file of the same size, a "
    "mask alone included, is caught, at the cost of reading all the data once (minutes on a large "
    "set on a spinning disk; the seconds are logged)"
)


@dataclasses.dataclass(frozen=True)
class SegFacts:
    """What is known about segmentation data before it is loaded.

    :param dataset: the dataset name, or ``"folder"`` / ``"csv"``.
    :param num_classes: classes (void is not one).
    :param class_names: one name per class.
    :param ignore_index: the void value, or ``None``.
    :param image_size: the requested ``(height, width)``, or ``None`` for Pet when none was asked.
    """

    dataset: str
    num_classes: int
    class_names: Tuple[str, ...]
    ignore_index: Optional[int]
    image_size: Optional[Tuple[int, int]]


def _range_problems(data: Any, prefix: str) -> List[str]:
    """Name the split, cap and sample counts out of range, at their dotted paths."""
    lines: List[str] = []

    def bad(name: str, why: str) -> None:
        lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(data, name)!r}")

    split = data.validation_split
    if split is not None and not 0 < split < 1:
        bad("validation_split", "expected a fraction strictly between 0 and 1")
    if data.max_samples is not None and data.max_samples < 1:
        bad("max_samples", "expected a positive int or null")
    if data.eval_samples is not None and data.eval_samples < 0:
        bad("eval_samples", "expected an int >= 0")
    return lines


@dataclasses.dataclass(frozen=True)
class BuiltinSegData(StrictConfig):
    """Oxford-IIIT Pet from the local TFDS cache and its selection options."""

    dataset: _SegDatasetName = dataclasses.field(
        metadata={
            "help": "oxford_pet: the TFDS record oxford_iiit_pet 4.x read from TFDS_DATA_DIR "
            "(never downloaded), 3 classes (pet, background, border), no void, 3680 train and "
            "3669 test images"
        }
    )
    format: Literal["builtin"] = dataclasses.field(
        default="builtin", metadata={"help": "where the data comes from"}
    )
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "fraction of train held out to select the best epoch, in (0, 1); "
            "null is the loader's 0.1"
        },
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "cap on the train pool and the test split; null is no cap"}
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "test pairs kept in memory for figures and analysis; null is the loader's 200"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        return _range_problems(self, prefix)


def _class_problems(data: Any, prefix: str) -> List[str]:
    """Name what is wrong with the stated classes, the ignore index and the reserved threshold."""
    lines: List[str] = []

    def bad(name: str, why: str) -> None:
        lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(data, name)!r}")

    names, count = data.class_names, data.num_classes
    if names is None and count is None:
        lines.append(
            f"Invalid value for {join_path(prefix, 'class_names')!r}: a segmentation source does not "
            "say how many classes its masks hold, so state them: data.class_names=[...] "
            "(one name per class) or data.num_classes=N")
    if names is not None:
        if any(not name.strip() for name in names):
            bad("class_names", "expected non-empty names")
        elif len(set(names)) != len(names):
            bad("class_names", "expected distinct names")
        elif len(names) < 2:
            bad("class_names", "expected at least 2 classes")
        elif count is not None and count != len(names):
            bad("num_classes", f"it disagrees with the {len(names)} class_names")
    if count is not None and count < 2:
        bad("num_classes", "expected at least 2 classes")
    classes = len(names) if names is not None else count
    ignore = data.ignore_index
    if ignore is not None:
        if ignore < 0:
            bad("ignore_index", "expected a whole number of 0 or more (a mask value)")
        elif classes is not None and ignore < classes:
            bad("ignore_index", f"it must lie outside the classes 0 to {classes - 1}")
    if data.mask_threshold is not None:
        bad("mask_threshold", "it is reserved: this version does not accept it, so a mask stored "
                              "as 0 and 255 cannot be used yet")
    return lines


def _file_problems(data: Any, prefix: str, text_keys: Tuple[str, ...]) -> List[str]:
    """Name every file-data option out of range, at its dotted path."""
    lines = _range_problems(data, prefix) + _class_problems(data, prefix)
    for name in text_keys:
        value = getattr(data, name)
        if value is not None and not value.strip():
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: expected a non-empty text, got {value!r}")
    return lines


#: The keys the folder and CSV configs have in common after their paths.
_CLASS_HELP = {
    "class_names": "one name per class, in the order of the mask values 0, 1, ...; state this or num_classes",
    "num_classes": "number of classes (mask values 0 to N-1, named '0', '1', ...); state this or class_names",
    "ignore_index": "the mask value that marks void pixels (a whole number outside the classes): "
    "left out of the loss, mIoU and accuracy; null means every pixel is a class",
    "mask_threshold": "reserved, not accepted in this version (maps a 0/255 mask to two classes)",
}


@dataclasses.dataclass(frozen=True)
class FolderSegData(StrictConfig):
    """Two directories, images and masks, joined by file stem; optionally a test pair of them."""

    train_images: str = dataclasses.field(
        metadata={"help": "directory of the train images (png, jpg, jpeg, bmp), joined to train_masks by file stem"}
    )
    train_masks: str = dataclasses.field(
        metadata={"help": "directory of the train masks, one file per image with the same stem; "
                  "single-channel PNG of class ids (a palette PNG is read as its indices)"}
    )
    format: Literal["folder"] = dataclasses.field(
        default="folder", metadata={"help": "where the data comes from"}
    )
    test_images: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "directory of the test images, only scored; null means no test split: test "
                  "metrics and figures are skipped"},
    )
    test_masks: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "directory of the test masks; given together with test_images"}
    )
    class_names: Optional[List[str]] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["class_names"]})
    num_classes: Optional[int] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["num_classes"]})
    ignore_index: Optional[int] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["ignore_index"]})
    mask_threshold: Optional[float] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["mask_threshold"]})
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "fraction of train held out to select the best epoch, in (0, 1); null is the loader's 0.1"},
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "cap on the train pool and the test split; null is no cap"}
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "test pairs kept in memory for figures and analysis; null is the loader's 200"},
    )
    fingerprint: Literal["listing", "content"] = dataclasses.field(
        default="listing", metadata={"help": _FINGERPRINT_HELP}
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        lines = _file_problems(self, prefix, ("train_images", "train_masks", "test_images", "test_masks"))
        if (self.test_images is None) != (self.test_masks is None):
            missing = "test_masks" if self.test_masks is None else "test_images"
            lines.append(
                f"Invalid value for {join_path(prefix, missing)!r}: test_images and test_masks go "
                "together, give both or neither")
        return lines


@dataclasses.dataclass(frozen=True)
class CsvSegData(StrictConfig):
    """A CSV file with an image column and a mask column; optionally a test one."""

    train_csv: str = dataclasses.field(
        metadata={"help": "CSV with one row per pair; a relative path starts at the CSV's directory"}
    )
    format: Literal["csv"] = dataclasses.field(default="csv", metadata={"help": "where the data comes from"})
    test_csv: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "CSV of the test pairs, only scored; null means no test split: test metrics "
                  "and figures are skipped"},
    )
    image_column: str = dataclasses.field(default="image", metadata={"help": "column holding the image path"})
    mask_column: str = dataclasses.field(default="mask", metadata={"help": "column holding the mask path"})
    class_names: Optional[List[str]] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["class_names"]})
    num_classes: Optional[int] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["num_classes"]})
    ignore_index: Optional[int] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["ignore_index"]})
    mask_threshold: Optional[float] = dataclasses.field(default=None, metadata={"help": _CLASS_HELP["mask_threshold"]})
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "fraction of train held out to select the best epoch, in (0, 1); null is the loader's 0.1"},
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None, metadata={"help": "cap on the train pool and the test split; null is no cap"}
    )
    eval_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "test pairs kept in memory for figures and analysis; null is the loader's 200"},
    )
    fingerprint: Literal["listing", "content"] = dataclasses.field(
        default="listing", metadata={"help": _FINGERPRINT_HELP}
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        return _file_problems(self, prefix, ("train_csv", "test_csv", "image_column", "mask_column"))


#: The ``format`` values and the class each one builds, in the order of :data:`SegData`.
SEG_FORMATS: Dict[str, Type[StrictConfig]] = {
    "builtin": BuiltinSegData,
    "folder": FolderSegData,
    "csv": CsvSegData,
}

#: The type of the segmentation task config's ``data`` field. The first member is the default format.
SegData = Union[BuiltinSegData, FolderSegData, CsvSegData]


def read_seg_data_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Return the mapping in the YAML file ``path``, its relative paths anchored to the file."""
    return read_data_file(path, SEG_PATH_KEYS)


def load_seg_data_config(data: Any) -> SegData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    if isinstance(data, tuple(SEG_FORMATS.values())):
        return data
    source = ""
    if isinstance(data, (str, Path)):
        source = f"data file {str(data)!r}: "
        data = read_seg_data_file(data)
    if not isinstance(data, Mapping):
        raise ConfigError(
            "Invalid value for 'data': expected a mapping or the path of a YAML file, "
            f"got {type(data).__name__}")
    given = dict(data)
    given.setdefault("format", "builtin")
    kind = given["format"]
    if not isinstance(kind, str) or kind not in SEG_FORMATS:
        raise ConfigError(
            f"Invalid value for 'data.format': {kind!r} is not a data format; choose one of "
            f"{', '.join(SEG_FORMATS)}")
    try:
        return validate(SEG_FORMATS[kind], given)
    except ConfigValidationError as exc:
        if not source:
            raise
        raise ConfigValidationError([source + line for line in exc.errors]) from exc


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------

def _pet_records(root: Path) -> bool:
    """Whether a version directory of Pet with ``dataset_info.json`` is under ``root``."""
    here = root / PET_TFDS_DIRECTORY
    if not here.is_dir():
        return False
    return any(
        child.is_dir() and child.name.startswith("4.") and (child / "dataset_info.json").is_file()
        for child in here.iterdir())


def check_seg_sources(data: SegData) -> None:
    """Refuse a Pet record set that is not under the TFDS directory; see the module contract."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-074
    # Path tests only, and the directory TFDS itself reads (`data_args._tfds_data_dir`, pinned to
    # TFDS's own answer): do NOT import TFDS or open a file here, since the first dataset op
    # creates TensorFlow's devices before `setup_gpu`, and do NOT test more than the loader's own
    # first failure (a version directory that holds dataset_info.json), because a false refusal of
    # the real cache is the worst failure.
    if data.format != "builtin":
        return
    root = _tfds_data_dir()
    try:
        found = _pet_records(root)
    except OSError as exc:
        raise ConfigError(
            f"Invalid value for 'data.dataset': oxford_pet reads TFDS records and "
            f"{exc.filename or str(root)!r} cannot be read ({exc.strerror or exc}); fix the "
            "permission, or set TFDS_DATA_DIR to a directory you can read") from exc
    if not found:
        raise ConfigError(
            f"Invalid value for 'data.dataset': oxford_pet reads the TFDS records of "
            f"{PET_TFDS_DIRECTORY} version 4.x, and none are under {str(root / PET_TFDS_DIRECTORY)!r} "
            "(TFDS directory from "
            f"{'$TFDS_DATA_DIR' if 'TFDS_DATA_DIR' in os.environ else 'the default ~/tensorflow_datasets'}); "
            "nothing is downloaded. Set TFDS_DATA_DIR to the directory that holds oxford_iiit_pet/")


def _class_names_of(data: Union[FolderSegData, CsvSegData]) -> Tuple[str, ...]:
    if data.class_names is not None:
        return tuple(data.class_names)
    return tuple(str(index) for index in range(data.num_classes))


def list_pairs(data: Union[FolderSegData, CsvSegData]) -> Any:
    """Return the :class:`~train.common.segmentation_data.PairListing` of file data (nothing decoded).

    :raises ConfigError: naming the dotted key of the first thing wrong.
    """
    from train.common.classification_data import DataSourceError
    from train.common.segmentation_data import read_segmentation_csv, scan_segmentation_folder

    try:
        if isinstance(data, FolderSegData):
            return scan_segmentation_folder(
                data.train_images, data.train_masks, data.test_images, data.test_masks)
        return read_segmentation_csv(
            data.train_csv, data.test_csv, image_column=data.image_column, mask_column=data.mask_column)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc


def seg_facts(data: SegData, requested_image_size: Optional[Tuple[int, int]]) -> SegFacts:
    """Return the facts of any segmentation data config before it is loaded; see the module contract."""
    if data.format == "builtin":
        check_seg_sources(data)
        from train.common.segmentation_data import PET_CLASS_NAMES, PET_MAX_BYTES, PET_TRAIN_SIZE

        if requested_image_size is not None:
            held = PET_TRAIN_SIZE * requested_image_size[0] * requested_image_size[1] * 3
            if held > PET_MAX_BYTES:
                raise ConfigError(
                    f"Invalid value for 'transform_args.image_size': oxford_pet is held in memory "
                    f"as bytes and {PET_TRAIN_SIZE} train images at {requested_image_size[0]}x"
                    f"{requested_image_size[1]} are {held / 1e9:.1f} GB, above the "
                    f"{PET_MAX_BYTES / 1e9:.1f} GB a TensorFlow graph can embed; use a smaller size, "
                    "or the folder format")
        return SegFacts(data.dataset, len(PET_CLASS_NAMES), tuple(PET_CLASS_NAMES), None, requested_image_size)
    if requested_image_size is None:
        raise ConfigError(
            f"Invalid value for 'transform_args.image_size': {data.format} images have no native "
            "size, so the input size must be given, e.g. transform_args.image_size=128 or "
            "transform_args.image_size=[128,256]")
    from train.common.classification_data import DataSourceError
    from train.common.segmentation_data import check_listing_sample

    names = _class_names_of(data)
    listing = list_pairs(data)
    try:
        check_listing_sample(listing, num_classes=len(names), ignore_index=data.ignore_index)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc
    return SegFacts(data.format, len(names), names, data.ignore_index, requested_image_size)


def load_seg_data(
    data: SegData,
    *,
    batch_size: int,
    seed: int,
    image_size: Tuple[int, int],
    augment: Any = None,
    eval_batch_size: Optional[int] = None,
) -> Any:
    """Load any segmentation data config through the shared ingestion path; see the module contract.

    :param image_size: ``(height, width)`` every image and mask is resized to.
    :param augment: the train hook ``(seed, image, mask) -> (image, mask)`` or ``None``.
    """
    from train.common import segmentation_data as loader
    from train.common.classification_data import DataSourceError

    shared = dict(
        batch_size=batch_size, seed=seed, size=tuple(image_size), augment=augment,
        eval_batch_size=eval_batch_size, validation_split=data.validation_split,
        eval_samples=data.eval_samples, max_samples=data.max_samples)
    if data.format == "builtin":
        return loader.load_pet_data(**shared)
    names = _class_names_of(data)
    try:
        return loader.load_pair_data(
            list_pairs(data), num_classes=len(names), class_names=names,
            ignore_index=data.ignore_index, fingerprint=data.fingerprint, **shared)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc
