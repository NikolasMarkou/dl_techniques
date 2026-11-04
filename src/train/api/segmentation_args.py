"""The segmentation arguments of ``python -m train``: data configs, the transform, metric and loss groups.

``BuiltinSegData`` names Oxford-IIIT Pet from the local TFDS cache; ``FolderSegData`` (two
directories, images and masks, joined by file stem) and ``CsvSegData`` (a CSV file with an
``image`` and a ``mask`` column) take the same selection protocol from files you have: a held-out
slice of train selects, the test split is only scored. The ``format`` key says which one a mapping
is. Every option that does not apply to the chosen source is REFUSED, never ignored.

Interface contract
------------------
``BuiltinSegData``
    Frozen :class:`~train.api.config.StrictConfig`; ``dataset`` is ``oxford_pet`` (the TFDS record
    ``oxford_iiit_pet:4.0.0`` (``segmentation_data.PET_TFDS_NAME``), mask minus 1, classes ``pet, background, border``, no void). It has
    no class, ignore, threshold or fingerprint key: Pet fixes them.

``FolderSegData`` / ``CsvSegData``
    Frozen configs with ``format`` ``"folder"`` / ``"csv"``. ``train_images`` and ``train_masks`` /
    ``train_csv`` are required; the test pair of directories or the test file is optional (a run
    with none skips test metrics and figures, and its summary says so). The classes are STATED:
    ``class_names`` (one name per class) or ``num_classes`` (named ``"0"``, ``"1"``, ...), both
    allowed when they agree. ``ignore_index`` is the void value (a whole number outside the
    classes) or null. A relative path is relative to the directory of the YAML file that holds it,
    or to the working directory for a mapping or the command line. ``mask_threshold`` (``0 < T < 255``)
    reads a black and white or grey mask as ``value > T`` (class 1) and the rest as class 0, for an
    8-bit grey or 1-bit PNG; it needs exactly 2 classes, no ``ignore_index``, and is refused for a
    palette or 16-bit mask; without it a mask of exactly 0 and 255 is refused with a pointer at it. ``transform_args.image_size`` must be set: files have no native
    size.

``SegData``
    The union of the three; the type of the segmentation task config's ``data`` field.

``SegTransformArgs``
    ``image_size`` (a positive int, a square, or ``[height, width]``; ``"auto"`` is the family
    recipe's) and ``augment`` (``none``, ``flip`` or ``dihedral``; ``"auto"`` is the recipe's).
    There is no ``normalize`` key: segmentation inputs are in [0, 1], no statistics pass runs, and the
    run records ``normalize`` as zeros and ones. ``resolve_auto(context)`` fills the ``"auto"``
    fields from the recipe (``context.defaults["transform_args"]``) and leaves one the recipe has
    no entry for as ``"auto"``, which ``assert_resolved`` reports by dotted path.
    :raises ConfigError: from ``resolve_auto`` for an ``augment`` that is no preset, and for
        ``dihedral`` with a non-square ``image_size`` (a quarter turn swaps the sides).

``SegMetricArgs``
    ``watch_metric`` (the metric that picks the best epoch: ``val_miou``, ``val_accuracy`` or
    ``val_loss``; ``"auto"`` is the recipe's), ``watch_mode`` (``min`` or ``max``; ``"auto"`` is read
    from the metric's name) and ``success_threshold`` (a best ``val_miou`` at or above it counts as
    success; ``null``, the default, is the mIoU of predicting the majority class on the selection
    set, which the task works out once the data is loaded). There is no ``top_k``: a segmentation
    run reports ``accuracy`` and ``miou`` only.

``SegLossArgs``
    ``name`` (``ce`` the default, ``dice``, ``ce_dice`` or ``focal``; the losses of
    :mod:`dl_techniques.losses.sparse_segmentation_loss`, which take logits and sparse targets, leave the
    void value out of the value and of the gradient, and score ONE value per image; ``ce`` is Keras'
    own, pooled over the batch's valid pixels), ``dice_smooth`` (read by ``dice`` and ``ce_dice``;
    positive; ``null`` is 1e-5) and ``focal_gamma`` (read by ``focal``; at least 0; ``null`` is 2.0).
    :data:`LOSS_KNOBS` says which loss reads which knob. A knob the chosen loss does not read is
    REFUSED by its dotted key, never ignored.

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
    Pet only: ``Path`` tests for ``dataset_info.json`` under ``$TFDS_DATA_DIR/oxford_iiit_pet/4.0.0`` (the one version the loader reads).
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

from .args import AutoContext, _is_auto, _recipe_values, _with_values
from .config import AUTO, Auto, StrictConfig, join_path, validate
from .data_args import _tfds_data_dir, read_data_file
from .errors import ConfigError, ConfigValidationError
from .segmentation_models import sides_of

# ---------------------------------------------------------------------

#: The built-in segmentation datasets.
SEG_DATASETS: Tuple[str, ...] = ("oxford_pet",)

#: Keys of a file-based segmentation config that name a path (a YAML file's relative ones are
#: made absolute against the file).
SEG_PATH_KEYS = ("train_images", "train_masks", "test_images", "test_masks", "train_csv", "test_csv")


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
            "help": "oxford_pet: the TFDS record oxford_iiit_pet 4.0.0 read from TFDS_DATA_DIR "
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
        default=None, metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"}
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
    threshold = data.mask_threshold
    if threshold is not None:
        # DECISION plan-2026-11-03T221936-98b866fb/D-016
        # The threshold turns a grey mask into TWO classes with no void: 3 classes cannot come out of one
        # cut, and a void value would be a third kind of pixel the cut cannot make. Do NOT relax either to
        # "ignore it": a quiet no-op key is how a 255 mask trains the wrong classes.
        if isinstance(threshold, bool) or not 0 < threshold < 255:
            bad("mask_threshold", "expected a number between 0 and 255 (exclusive): a value above it is "
                                  "class 1 and the rest class 0")
        if classes is not None and classes != 2:
            bad("mask_threshold", f"it needs exactly 2 classes (value > T is class 1, the rest class 0) "
                                  f"and {classes} are stated")
        if ignore is not None:
            bad("mask_threshold", f"it cannot be used with data.ignore_index (a thresholded mask has "
                                  f"two classes and no void), and ignore_index is {ignore}")
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
    "mask_threshold": "T, 0 < T < 255: read an 8-bit grey or 1-bit mask as value > T = class 1, the rest "
    "class 0 (a black and white mask stored as 0 and 255); needs 2 classes and no ignore_index; null "
    "reads the stored values",
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
        default=None, metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"}
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
        default=None, metadata={"help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"}
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
# The transform group
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegTransformArgs(StrictConfig):
    """Input size and augmentation of a segmentation run; read by the segmentation task."""

    image_size: Auto[Union[int, List[int]]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "size every image and mask is resized to: one int (a square) or [height, width]; "
            "each side must be a multiple of the family's stride (2**depth for convunext, 16 for "
            "accunet, 32 for resnet_seg and 16 for convnext_seg, fewer with the cifar stem or "
            "cifar10); auto is the family recipe's. The image is resized bilinearly, the mask by "
            "nearest neighbour, and nothing is padded"
        },
    )
    augment: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "none, flip (horizontal) or dihedral (flips and quarter turns, square sizes "
            "only), applied to the image and the mask together; auto is the family recipe's"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit ``image_size`` that is no positive int or pair of them."""
        size = self.image_size
        if _is_auto(size):
            return []

        def side(value: Any) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        if side(size) or (isinstance(size, (list, tuple)) and len(size) == 2 and all(map(side, size))):
            return []
        return [
            f"Invalid value for {join_path(prefix, 'image_size')!r}: expected one positive int "
            f"(a square) or [height, width] of two, got {size!r}"
        ]

    def resolve_auto(self, context: AutoContext) -> "SegTransformArgs":
        """Fill ``"auto"`` fields from the recipe; see the module contract."""
        from train.common.segmentation_data import SEG_AUGMENTS

        found = _recipe_values(self, context, "transform_args")
        resolved = _with_values(self, "transform_args", found)
        if not _is_auto(resolved.augment) and resolved.augment not in SEG_AUGMENTS:
            raise ConfigError(
                f"Invalid value for 'transform_args.augment': {resolved.augment!r} is not a "
                f"segmentation preset; choose one of {', '.join(SEG_AUGMENTS)}."
            )
        if resolved.augment == "dihedral" and not _is_auto(resolved.image_size):
            height, width = sides_of(resolved.image_size)
            if height != width:
                raise ConfigError(
                    f"Invalid value for 'transform_args.augment': dihedral turns the image by "
                    f"quarter turns, which swaps the sides of a {height}x{width} (height x width) "
                    "image; use flip, or a square transform_args.image_size"
                )
        return resolved


# ---------------------------------------------------------------------
# The metric group
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegMetricArgs(StrictConfig):
    """The metric that selects the best epoch and the success bar; read by the trainer and the segmentation task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={"help": "metric that picks the best epoch: val_miou (the default recipe), val_accuracy or val_loss"},
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO, metadata={"help": "min or max; auto is read from the metric's name"}
    )
    success_threshold: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "a best val_miou at or above this counts as success; null is the mIoU of "
            "predicting the majority class of the selection set for every pixel, worked out after "
            "the data is loaded and stated in the run summary"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit ``success_threshold`` outside [0, 1]."""
        threshold = self.success_threshold
        if threshold is not None and not 0 <= threshold <= 1:
            return [
                f"Invalid value for {join_path(prefix, 'success_threshold')!r}: expected a fraction "
                f"in [0, 1] (an mIoU), got {threshold!r}"
            ]
        return []

    def resolve_auto(self, context: AutoContext) -> "SegMetricArgs":
        """Fill ``"auto"`` fields: ``watch_metric`` from the recipe, ``watch_mode`` from its name."""
        from train.common.callbacks import resolve_monitor_mode

        found = _recipe_values(self, context, "metric_args", skip=("watch_mode",))
        metric = found.get("watch_metric", self.watch_metric)
        if _is_auto(self.watch_mode) and not _is_auto(metric):
            found["watch_mode"] = resolve_monitor_mode(metric)
        return _with_values(self, "metric_args", found)


#: The segmentation losses and, for each, the ``loss_args`` knobs it reads. The keys are the values of
#: ``loss_args.name``; ``train.common.segmentation_run`` builds the same four (a test holds the two equal).
LOSS_KNOBS: Dict[str, Tuple[str, ...]] = {
    "ce": (),
    "dice": ("dice_smooth",),
    "ce_dice": ("dice_smooth",),
    "focal": ("focal_gamma",),
}

_SegLossName = Literal[tuple(LOSS_KNOBS)]  # type: ignore[valid-type]


@dataclasses.dataclass(frozen=True)
class SegLossArgs(StrictConfig):
    """The loss of a segmentation run and the knobs it reads; see the module contract."""

    name: _SegLossName = dataclasses.field(
        default="ce",
        metadata={
            "help": "the loss: ce (Keras' sparse cross-entropy, void left out, pooled over the batch's "
            "valid pixels), dice, ce_dice (cross-entropy plus dice) or focal; the last three score one "
            "value per image and leave void out of the value and the gradient"
        },
    )
    dice_smooth: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "smoothing of the dice term, in its numerator and denominator; read by dice and "
            "ce_dice only; positive; null is 1e-5"
        },
    )
    focal_gamma: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "focusing parameter of the focal loss (0 is the cross-entropy); read by focal "
            "only; at least 0; null is 2.0"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a knob out of range, and a knob the chosen loss does not read."""
        lines: List[str] = []
        if self.dice_smooth is not None and not self.dice_smooth > 0:
            lines.append(
                f"Invalid value for {join_path(prefix, 'dice_smooth')!r}: expected a positive number "
                f"(it keeps an image with no valid pixel at loss 0), got {self.dice_smooth!r}"
            )
        if self.focal_gamma is not None and not self.focal_gamma >= 0:
            lines.append(
                f"Invalid value for {join_path(prefix, 'focal_gamma')!r}: expected a number of at "
                f"least 0, got {self.focal_gamma!r}"
            )
        for knob in ("dice_smooth", "focal_gamma"):
            if getattr(self, knob) is not None and knob not in LOSS_KNOBS[self.name]:
                readers = [name for name, knobs in LOSS_KNOBS.items() if knob in knobs]
                lines.append(
                    f"Invalid value for {join_path(prefix, knob)!r}: the {self.name!r} loss does not "
                    f"read it ({' and '.join(readers)} do); leave it null or set "
                    f"{join_path(prefix, 'name')}={readers[0]}"
                )
        return lines


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------

def _pet_records(root: Path) -> bool:
    """Whether the ONE version of Pet the loader reads (``PET_TFDS_VERSION``) is prepared under ``root``."""
    from train.common.segmentation_data import PET_TFDS_DIRECTORY, PET_TFDS_VERSION

    return (root / PET_TFDS_DIRECTORY / PET_TFDS_VERSION / "dataset_info.json").is_file()


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
        from train.common.segmentation_data import PET_TFDS_DIRECTORY, PET_TFDS_VERSION

        here = root / PET_TFDS_DIRECTORY
        try:
            present = sorted(child.name for child in here.iterdir()) if here.is_dir() else []
        except OSError:
            present = []
        held = f" (it holds {', '.join(present)})" if present else ""
        raise ConfigError(
            f"Invalid value for 'data.dataset': oxford_pet reads the TFDS records of "
            f"{PET_TFDS_DIRECTORY} version {PET_TFDS_VERSION}, and none are under "
            f"{str(here / PET_TFDS_VERSION)!r}{held} "
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
        check_listing_sample(listing, num_classes=len(names), ignore_index=data.ignore_index,
                             mask_threshold=data.mask_threshold)
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
            ignore_index=data.ignore_index, fingerprint=data.fingerprint,
            mask_threshold=data.mask_threshold, **shared)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc
