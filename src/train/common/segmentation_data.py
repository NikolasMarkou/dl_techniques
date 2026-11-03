"""Image and mask data for the segmentation trainers: one decoder, listed pairs, one pipeline.

The trainers of ``train.api`` read pairs of files (an image and the mask of its classes) from two
directories joined by file stem, or from a CSV file, and Oxford-IIIT Pet from the local TFDS cache.
Every pixel of an image or a mask that reaches a model or a check is decoded by ONE function,
:func:`decode_image` / :func:`decode_mask`, built on PIL, which is also what the predictor
calls: ``tf.io.decode_image`` ignores the EXIF orientation of a JPEG, and on a measured set (SUT-Crack)
119 of 130 photographs are stored sideways (orientation 6) beside masks that are upright, so a
TensorFlow-native pair loader would train rotated photographs against upright masks and
would not fail.

Interface contract
------------------
``decode_image(path)``
    :param path: an image file.
    :return: ``uint8`` ``(H, W, 3)`` after ``ImageOps.exif_transpose`` and PIL's conversion to RGB
        (an RGBA image loses its alpha, a grey one is repeated over the channels). A 16-bit grey
        image (PIL ``I;16*``, or ``I`` with values in 0..65535) keeps its high byte, ``value >> 8``, so
        0..65535 becomes 0..255 (the rule of a 16-bit RGB PNG, which PIL opens with its high byte, and of
        the classification decoder); PIL's own conversion would clip it to 255. A 12-bit image
        (0..4095) therefore reaches only 0..15: the scale is fixed, never the image's own range.
    :raises PairError: for a file that cannot be decoded, or an image whose mode has no byte scale
        (float ``F``, or ``I`` holding values outside 0..65535): never a silent clip.

``decode_mask(path)``
    :param path: a mask file.
    :return: ``int32`` ``(H, W)`` of the raw stored values, after ``exif_transpose``: palette
        indices for a palette PNG (the palette is never applied), the grey value for a grey one,
        ``0`` and ``255`` for a 1-bit PNG, the value for a 16-bit grey PNG.
    :raises PairError: for a JPEG (lossy, the stored value is not the class), an RGB, RGBA or
        other multi-channel mask, or a file that cannot be decoded.

``image_to_rgb(image)``
    :param image: an opened ``PIL.Image.Image``.
    :return: what :func:`decode_image` returns for a file holding that image.

``read_pair(image_path, mask_path)``
    :return: ``(image, mask)`` as above.
    :raises PairError: when the two differ in size after the EXIF orientation, naming both files.

``scan_segmentation_folder(...)`` / ``read_segmentation_csv(...)``
    List the pairs of two directories joined by stem, or of a CSV file, reading no pixel except as
    the next function does. They refuse an unpaired stem, two files of one stem, a file listed twice
    in one split, and a file in both splits (image or mask, compared by resolved path and by file
    identity). Both return a :class:`PairListing`.

``check_pairs(images, masks, *, num_classes, ignore_index, ...)``
    The scan of the masks (the value check) and of the sizes. ``train.api`` calls it on a sample
    before the run directory is claimed (PIL only, no TensorFlow op) and the loaders call it on every
    file they use before ``fit``, logging the duration.

    Also refused: a 1-bit mask while ``ignore_index`` is 255 (a 1-bit PNG reads as 0 and 255, so the
    foreground would be void; D-087). The :class:`PairScan` it returns counts the pixels of each class.

``refuse_absent_classes(scan, class_names, *, mask_key, where, ignore_index)``
    :param scan: the :class:`PairScan` of the masks the model will be fit on.
    :raises DataSourceError: naming the first five declared classes with no pixel in ``scan`` (D-087).
        ``load_pair_data`` calls it on the FIT split's scan, before any pipeline is built.

``pair_fingerprint(listing, mode)``
    A sha256 over ``(split, relative image path, relative mask path, both sizes)`` in listing
    order, or with ``mode="content"`` also both files' bytes, so a change in a mask alone is seen.

``load_pair_data(listing, ...)`` / ``load_pet_data(...)``
    Build the :class:`SegmentationData` of a listing, or of the local Oxford-IIIT Pet cache. The
    pipelines yield ``(image, mask)``: ``float32`` ``(H, W, 3)`` in [0, 1] and ``int32`` ``(H, W)``;
    void pixels keep their ``ignore_index`` (the mask is resized by nearest neighbour only and
    nothing is padded, so no pixel gets a class the file did not give it).

``build_segmentation_augment(preset, size)``
    :param preset: one of :data:`SEG_AUGMENTS`: ``none``, ``flip`` (horizontal) or ``dihedral``
        (independent horizontal and vertical flips and a quarter turn).
    :param size: ``(height, width)`` of the images the hook will see.
    :return: ``None`` for ``none``, else a seeded hook ``(seed, image, mask) -> (image, mask)`` that
        draws ONE geometric transform from the element's seed and applies it to the image and the
        mask together, so the mask stays the function of the image it was (stateless: the draw is
        a function of the loader's seed and the element's seed pair). Nothing is padded and no
        pixel is interpolated, so no class appears that the mask did not hold and void stays void;
        photometric ops are absent, because they would be image-only ops.
    :raises ValueError: for another preset, or ``dihedral`` with ``height != width`` (a quarter turn
        swaps the sides of a non-square image).

``resize_pair(image, mask, size)``
    The one resize: bilinear with antialiasing for the image (rounded to a byte, as the legacy
    trainer's Pet loader does), nearest for the mask.

``load_oxford_pet_arrays(size)``
    The one function that touches TFDS; tests replace it.

Import rule
-----------
This module imports TensorFlow (as ``classification_data`` does) and is imported inside functions by
``train.api``; no function runs a TensorFlow op until a pipeline is built.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from dl_techniques.datasets.vision.augment_hook import Augment, seeded, with_hook
from dl_techniques.datasets.vision.splits import seeded_split_indices
from dl_techniques.utils.logger import logger
from train.common.classification_data import (
    DEFAULT_VALIDATION_SPLIT,
    FINGERPRINT_MODES,
    FINGERPRINT_SCHEME,
    IMAGE_SUFFIXES,
    DataSourceError,
    _examples,
    _link_state,
    _refuse_shared,
    _resolver,
    steps_per_epoch_for,
)

#: Oxford-IIIT Pet as the legacy trainer reads it (``train.convunext.common``): TFDS mask minus 1.
# DECISION plan-2026-10-31T180033-2faf890a/D-087
# The version is NAMED, in one place, as D-065 did for Imagenette. Do NOT go back to "any 4.*
# directory with a dataset_info.json" in the pre-claim check (`segmentation_args._pet_records`) or
# to the glob ``4.*.*`` in the loader: a TFDS directory that holds only ``4.9.9`` passed the check,
# the run directory was claimed, and the loader died with a traceback (review-iter-3 concern 7).
#: The TFDS directory of Oxford-IIIT Pet under ``TFDS_DATA_DIR``.
PET_TFDS_DIRECTORY = "oxford_iiit_pet"
#: The ONE version the loader reads and the pre-claim check looks for.
#: ``tests/test_train/test_api/test_the_builtin_sources_are_refused_before_the_claim.py`` holds it
#: equal to the installed TFDS builder's default version.
PET_TFDS_VERSION = "4.0.0"
PET_TFDS_NAME = f"{PET_TFDS_DIRECTORY}:{PET_TFDS_VERSION}"
PET_CLASS_NAMES: Tuple[str, ...] = ("pet", "background", "border")
#: Images per decode batch of :func:`load_oxford_pet_arrays`.
PET_DECODE_BATCH = 128
#: Test images kept in memory for figures and the analyzer when ``eval_samples`` is null.
DEFAULT_EVAL_SAMPLES = 200
#: Pairs the pre-claim check reads (evenly spaced over the train list).
PRE_CLAIM_PAIRS = 32
#: Pairs in the Pet train split of the TFDS 4.x cache (the test split holds 3669).
PET_TRAIN_SIZE = 3680
#: Bytes of the in-memory Pet image array above which TensorFlow cannot embed it in a graph (2 GB limit).
PET_MAX_BYTES = 1_900_000_000
#: The only value pair of a binary mask stored as black and white.
BINARY_PAIR = (0, 255)

#: The EXIF orientations that swap the two sides of the image.
_SWAPPING_ORIENTATIONS = (5, 6, 7, 8)
_ORIENTATION_TAG = 0x0112
#: PIL modes of a single-channel mask.
_MASK_MODES = ("1", "L", "P", "I;16", "I;16L", "I;16B", "I")
#: PIL modes that hold a 16-bit grey image (``I`` is PIL's 32-bit integer mode, which older Pillow
#: used for a 16-bit PNG); PIL's conversion to RGB clips all of them at 255.
_SIXTEEN_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I")


class PairError(ValueError):
    """A file of an image and mask pair that cannot be used; the message names the file(s)."""

    def __init__(self, message: str, which: str = "image") -> None:
        super().__init__(message)
        #: ``"image"`` or ``"mask"``: which side of the pair the message is about.
        self.which = which


# ---------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------

def _open(path: str, which: str) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
        return image
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise PairError(f"{path!r} cannot be decoded as an image ({exc})", which) from exc


def image_to_rgb(image: Image.Image) -> np.ndarray:
    """One opened PIL image as ``uint8`` ``(H, W, 3)``; see the module contract.

    The one place an image becomes the model's colour space: the EXIF orientation applied, then PIL's
    conversion to RGB. :func:`decode_image` (a file) and the segmentation predictor (a PIL image given
    by the caller) both end here, so a file and the same image opened by hand give the same pixels.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in _SIXTEEN_BIT_MODES:
        # DECISION plan-2026-10-31T180033-2faf890a/D-087
        # PIL's `convert("RGB")` CLIPS a 16-bit grey image at 255 (0.4% of a 0..65535 ramp survives).
        # Do NOT call it on these modes and do NOT rescale by the image's own min and max (the
        # decode would then depend on the image, so training and `predict` could disagree).
        values = np.asarray(image).astype(np.int64)
        if image.mode == "I" and values.size and (values.min() < 0 or values.max() > 65535):
            raise PairError(
                f"an image of PIL mode I holds values from {int(values.min())} to {int(values.max())}, "
                "outside the 16-bit range 0..65535, so there is no byte scale for it; convert it to "
                "an 8-bit or a 16-bit PNG")
        grey = (values >> 8).astype(np.uint8)
        return np.repeat(grey[..., None], 3, axis=-1)
    if image.mode == "F":
        raise PairError(
            "an image of PIL mode F (floating point) has no fixed value range, so there is no byte "
            "scale for it; convert it to an 8-bit or a 16-bit PNG")
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def decode_image(path: str) -> np.ndarray:
    """One image file as ``uint8`` ``(H, W, 3)``; see the module contract."""
    return image_to_rgb(_open(path, "image"))


def decode_mask(path: str) -> np.ndarray:
    """One mask file as ``int32`` ``(H, W)`` of its stored values; see the module contract."""
    return _decode_mask_and_mode(path)[0]


def _decode_mask_and_mode(path: str) -> Tuple[np.ndarray, str]:
    """:func:`decode_mask` and the PIL mode the file was stored in (``"1"`` is the 1-bit case)."""
    mask = _open(path, "mask")
    if mask.format in ("JPEG", "MPO"):
        raise PairError(
            f"{path!r} is a JPEG; JPEG is lossy, so the value stored for a pixel is not the class "
            "it was drawn with; save the masks as PNG", "mask")
    if mask.mode not in _MASK_MODES:
        raise PairError(
            f"{path!r} is a {mask.mode} image; a mask is one channel of class ids (an 8-bit grey "
            "or palette PNG, or a 1-bit PNG), not a picture; an RGB mask cannot be read as ids",
            "mask")
    mode = mask.mode
    mask = ImageOps.exif_transpose(mask)
    values = np.asarray(mask)
    if mode == "1":
        values = values.astype(np.int32) * 255
    return values.astype(np.int32), mode


def image_size_of(path: str) -> Tuple[int, int]:
    """``(height, width)`` of an image after its EXIF orientation, read from the header."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(_ORIENTATION_TAG)
            mode = image.mode
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise PairError(f"{path!r} cannot be decoded as an image ({exc})") from exc
    if mode == "F":
        # The mode is in the header, so the refusal can come before the run directory is claimed.
        raise PairError(f"{path!r} is a PIL mode F (floating point) image: it has no fixed value "
                        "range, so there is no byte scale for it; convert it to an 8-bit or a "
                        "16-bit PNG")
    return (width, height) if orientation in _SWAPPING_ORIENTATIONS else (height, width)


def _refuse_unequal(image_path: str, mask_path: str, image_hw: Tuple[int, int],
                    mask_hw: Tuple[int, int]) -> None:
    if image_hw != mask_hw:
        raise PairError(
            f"the image {image_path!r} is {image_hw[0]}x{image_hw[1]} (height x width) and its mask "
            f"{mask_path!r} is {mask_hw[0]}x{mask_hw[1]} after the EXIF orientation of each is "
            "applied; a pair must have one size (a sideways photograph beside an upright mask is "
            "the usual cause: re-save the mask in the photograph's orientation)", "mask")


def read_pair(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Both files of a pair, decoded and size-checked; see the module contract."""
    image, mask = decode_image(image_path), decode_mask(mask_path)
    _refuse_unequal(image_path, mask_path, image.shape[:2], mask.shape)
    return image, mask


# ---------------------------------------------------------------------
# Listing: two directories joined by stem, or a CSV file
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PairListing:
    """The image and mask files of a folder or CSV source, listed and paired, nothing decoded.

    :param source: ``"folder"`` or ``"csv"``.
    :param train_images, train_masks: Every train pair, in listing order.
    :param test_images, test_masks: The same for the test split, or ``None`` and ``None``.
    :param train_bases, test_bases: ``(image base, mask base)``, what a path is relative to in
        :func:`pair_fingerprint`: the two directories of a folder source, the CSV file's directory
        (twice) for a CSV one; ``test_bases`` is ``None`` without a test split.
    """

    source: str
    train_images: List[str]
    train_masks: List[str]
    test_images: Optional[List[str]]
    test_masks: Optional[List[str]]
    train_bases: Tuple[str, str]
    test_bases: Optional[Tuple[str, str]] = None


def source_keys(source: str) -> Dict[str, str]:
    """The :class:`DataSourceError` keys of a source: ``train_image``, ``train_mask``, ``test_image``, ``test_mask``."""
    if source == "folder":
        return {"train_image": "train_images", "train_mask": "train_masks",
                "test_image": "test_images", "test_mask": "test_masks"}
    return {"train_image": "train_csv", "train_mask": "train_csv",
            "test_image": "test_csv", "test_mask": "test_csv"}


def _has_suffix(name: str) -> bool:
    return name.lower().endswith(IMAGE_SUFFIXES)


def _stems(root: str, key: str) -> Dict[str, str]:
    """``{stem: path}`` of the image-suffixed files directly inside ``root``, hidden entries skipped.

    :raises DataSourceError: ``root`` is no directory, two files share a stem (``a.png`` and
        ``a.jpg``), a name is not UTF-8 or a symbolic link leads nowhere.
    """
    if not os.path.isdir(root):
        raise DataSourceError(key, f"{root!r} is not a directory")
    found: Dict[str, str] = {}
    problems: List[str] = []
    for entry in sorted(os.scandir(root), key=lambda item: item.name):
        if entry.name.startswith(".") or not _has_suffix(entry.name):
            continue
        try:
            entry.name.encode("utf-8")
        except UnicodeEncodeError:
            problems.append(f"{entry.name!r} is not a valid UTF-8 file name")
            continue
        if entry.is_symlink() and _link_state(entry.path) is not None:
            problems.append(f"{entry.name!r} is a symbolic link that leads nowhere")
            continue
        if not entry.is_file():
            continue
        stem = os.path.splitext(entry.name)[0]
        if stem in found:
            problems.append(
                f"{os.path.basename(found[stem])!r} and {entry.name!r} have the same stem, so "
                "which one is the file of the pair cannot be told")
        else:
            found[stem] = entry.path
    if problems:
        raise DataSourceError(key, f"{root!r}: {_examples(problems, len(problems))}")
    return found


def _pair_by_stem(images_dir: str, masks_dir: str, image_key: str, mask_key: str,
                  ) -> Tuple[List[str], List[str]]:
    """The images and masks of two directories joined by stem, in sorted stem order.

    :raises DataSourceError: an unpaired stem, naming the two counts and the first stems of each side.
    """
    images, masks = _stems(images_dir, image_key), _stems(masks_dir, mask_key)
    lone_images = sorted(set(images) - set(masks))
    lone_masks = sorted(set(masks) - set(images))
    if lone_images or lone_masks:
        raise DataSourceError(
            image_key,
            f"{images_dir!r} holds {len(images)} images and {masks_dir!r} holds {len(masks)} masks, "
            f"and they are joined by file stem: {len(lone_images)} image(s) have no mask "
            f"({_examples([repr(s) for s in lone_images], len(lone_images)) or 'none'}) and "
            f"{len(lone_masks)} mask(s) have no image "
            f"({_examples([repr(s) for s in lone_masks], len(lone_masks)) or 'none'})")
    stems = sorted(images)
    if not stems:
        raise DataSourceError(image_key, f"{images_dir!r} and {masks_dir!r} hold no pairs")
    return [images[s] for s in stems], [masks[s] for s in stems]


def _refuse_listed_twice(paths: Sequence[str], resolve: Callable[[str], str], key: str,
                         what: str) -> List[str]:
    """Resolve ``paths`` and refuse a file listed more than once; return the resolved paths."""
    resolved = [resolve(path) for path in paths]
    seen: Set[str] = set()
    twice: List[str] = []
    for path, real in zip(paths, resolved):
        if real in seen:
            twice.append(repr(path))
        seen.add(real)
    if twice:
        raise DataSourceError(
            key, f"{len(twice)} {what} file(s) are listed more than once (files are compared by "
                 f"resolved path, so a symlink is the file it points to): "
                 f"{_examples(twice, len(twice))}; a pair in the list twice can land in both the "
                 "fit and the selection split, and a mask shared by two images is the sign of a "
                 "mispaired list")
    return resolved


def _refuse_leaks(
        listing_train: Tuple[Sequence[str], Sequence[str]], train_real: Tuple[List[str], List[str]],
        listing_test: Tuple[Sequence[str], Sequence[str]], test_real: Tuple[List[str], List[str]],
        keys: Dict[str, str]) -> None:
    """Refuse a test image or a test mask that is also a train file: a mask belongs to one split."""
    _refuse_shared(listing_train[0], train_real[0], listing_test[0], test_real[0], keys["test_image"])
    _refuse_shared(listing_train[1], train_real[1], listing_test[1], test_real[1], keys["test_mask"])


def scan_segmentation_folder(
        train_images: str, train_masks: str, test_images: Optional[str] = None,
        test_masks: Optional[str] = None,
) -> PairListing:
    """List the pairs of ``train_images`` and ``train_masks`` (and the test pair of directories), reading none.

    :raises DataSourceError: see the module contract; also fewer than 2 train pairs (one to fit
        on, one to select on) and a test directory given without its mask directory.
    """
    keys = source_keys("folder")
    if (test_images is None) != (test_masks is None):
        raise DataSourceError(
            "test_masks" if test_masks is None else "test_images",
            "test_images and test_masks go together: give both or neither")
    images, masks = _pair_by_stem(train_images, train_masks, keys["train_image"], keys["train_mask"])
    if len(images) < 2:
        raise DataSourceError(
            keys["train_image"], f"{train_images!r} holds {len(images)} pair; a run needs at least 2 "
                                 "(one to fit on, one to select on)")
    resolve = _resolver()
    train_real = (_refuse_listed_twice(images, resolve, keys["train_image"], "image"),
                  _refuse_listed_twice(masks, resolve, keys["train_mask"], "mask"))
    test = None
    if test_images is not None:
        t_images, t_masks = _pair_by_stem(test_images, test_masks, keys["test_image"], keys["test_mask"])
        test_real = (_refuse_listed_twice(t_images, resolve, keys["test_image"], "image"),
                     _refuse_listed_twice(t_masks, resolve, keys["test_mask"], "mask"))
        _refuse_leaks((images, masks), train_real, (t_images, t_masks), test_real, keys)
        test = (t_images, t_masks, (os.path.abspath(test_images), os.path.abspath(test_masks)))
    return PairListing(
        "folder", images, masks, None if test is None else test[0], None if test is None else test[1],
        (os.path.abspath(train_images), os.path.abspath(train_masks)),
        None if test is None else test[2])


def _pair_rows(csv_path: str, key: str, image_column: str, mask_column: str,
               ) -> Tuple[List[str], List[str]]:
    """``(image paths, mask paths)`` of one CSV file; every file must exist and have an image suffix."""
    if not os.path.isfile(csv_path):
        raise DataSourceError(key, f"{csv_path!r} is not a file")
    base = os.path.dirname(os.path.abspath(csv_path))
    images: List[str] = []
    masks: List[str] = []
    problems: List[str] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for column_key, column in (("image_column", image_column), ("mask_column", mask_column)):
            if column not in columns:
                raise DataSourceError(
                    column_key, f"column {column!r} is not in {csv_path!r} (columns: {columns})")
        for number, row in enumerate(reader, start=2):
            pair = []
            for column in (image_column, mask_column):
                cell = row.get(column)
                if cell is None or not cell.strip():
                    problems.append(f"line {number} has an empty {column!r}")
                    break
                cell = cell.strip()
                full = cell if os.path.isabs(cell) else os.path.join(base, cell)
                if not _has_suffix(full):
                    problems.append(f"line {number}: {cell!r} is not one of {IMAGE_SUFFIXES}")
                elif not os.path.isfile(full):
                    problems.append(f"line {number}: {full!r} does not exist")
                pair.append(full)
            else:
                images.append(pair[0])
                masks.append(pair[1])
    if problems:
        shown = "; ".join(problems[:5]) + (f"; and {len(problems) - 5} more" if len(problems) > 5 else "")
        raise DataSourceError(key, f"{csv_path!r}: {shown}")
    if not images:
        raise DataSourceError(key, f"{csv_path!r} holds no rows")
    return images, masks


def read_segmentation_csv(
        train_csv: str, test_csv: Optional[str] = None, *, image_column: str = "image",
        mask_column: str = "mask",
) -> PairListing:
    """List the pairs a CSV file names (and a test CSV), reading no pixel.

    A relative path is relative to the CSV file's directory; the file is read as ``utf-8-sig``.

    :raises DataSourceError: see the module contract; also a missing column or an empty cell, a
        path that does not exist or has no image suffix, and fewer than 2 train rows.
    """
    keys = source_keys("csv")
    images, masks = _pair_rows(train_csv, keys["train_image"], image_column, mask_column)
    if len(images) < 2:
        raise DataSourceError(
            "train_csv", f"{train_csv!r} holds {len(images)} row; a run needs at least 2 "
                         "(one to fit on, one to select on)")
    resolve = _resolver()
    train_real = (_refuse_listed_twice(images, resolve, keys["train_image"], "image"),
                  _refuse_listed_twice(masks, resolve, keys["train_mask"], "mask"))
    t_images = t_masks = test_base = None
    if test_csv is not None:
        t_images, t_masks = _pair_rows(test_csv, keys["test_image"], image_column, mask_column)
        test_real = (_refuse_listed_twice(t_images, resolve, keys["test_image"], "image"),
                     _refuse_listed_twice(t_masks, resolve, keys["test_mask"], "mask"))
        _refuse_leaks((images, masks), train_real, (t_images, t_masks), test_real, keys)
        folder = os.path.dirname(os.path.abspath(test_csv))
        test_base = (folder, folder)
    folder = os.path.dirname(os.path.abspath(train_csv))
    return PairListing("csv", images, masks, t_images, t_masks, (folder, folder), test_base)


# ---------------------------------------------------------------------
# The check of the masks and the sizes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PairScan:
    """What :func:`check_pairs` read.

    :param pairs: pairs checked.
    :param values: the distinct mask values seen, sorted.
    :param seconds: wall time.
    :param class_pixels: pixels seen of each class ``0 .. num_classes - 1`` (void and any other value
        not counted); :func:`refuse_absent_classes` reads it.
    """

    pairs: int
    values: Tuple[int, ...]
    seconds: float
    class_pixels: Tuple[int, ...] = ()


def _allowed(num_classes: int, ignore_index: Optional[int]) -> Callable[[int], bool]:
    return lambda value: 0 <= value < num_classes or value == ignore_index


def check_pairs(
        images: Sequence[str], masks: Sequence[str], *, num_classes: int, ignore_index: Optional[int],
        image_key: str, mask_key: str, image_pixels: bool = False,
        mask_threshold_key: str = "data.mask_threshold",
) -> PairScan:
    """Decode every mask, compare the size of every pair and check the values against the classes.

    :param images, masks: the pairs, same length.
    :param num_classes: values ``0 .. num_classes - 1`` are classes.
    :param ignore_index: the one other value a mask may hold (void), or ``None``.
    :param image_key, mask_key: the :class:`DataSourceError` keys of a failure in an image and in a
        mask (the same key for a CSV source, whose one file holds both).
    :param image_pixels: decode each image fully (a corrupt file is found now) instead of reading
        its header; masks are always decoded, because their values are what is checked.
    :param mask_threshold_key: the key the binary-mask message points at.
    :return: the :class:`PairScan`.
    :raises DataSourceError: a file that cannot be decoded, a pair of two sizes, a JPEG or colour
        mask (all naming the file), a mask whose values are exactly ``{0, 255}`` while 255 is no
        class (a black and white mask: the message points at ``mask_threshold_key``), or a value that
        is neither a class nor ``ignore_index`` (naming the first file and the value).
    """
    allowed = _allowed(num_classes, ignore_index)
    started = time.perf_counter()
    union: Set[int] = set()
    first_bad: Optional[Tuple[str, int]] = None
    class_pixels = np.zeros(num_classes, dtype=np.int64)
    for image_path, mask_path in zip(images, masks):
        try:
            mask, mode = _decode_mask_and_mode(mask_path)
            image_hw = (decode_image(image_path).shape[:2] if image_pixels
                        else image_size_of(image_path))
            _refuse_unequal(image_path, mask_path, tuple(image_hw), mask.shape)
        except PairError as exc:
            raise DataSourceError(mask_key if exc.which == "mask" else image_key, str(exc)) from exc
        if mode == "1" and ignore_index == BINARY_PAIR[1]:
            # DECISION plan-2026-10-31T180033-2faf890a/D-087
            # The decoder reads a 1-bit PNG as 0 and 255 (D-074), and 255 is the conventional void
            # value. Do NOT let the pair through to `fit`: every foreground pixel becomes void, the
            # run trains on one class and reports "best 1.0000" against a baseline of 1.0000.
            raise DataSourceError(
                mask_key, f"{mask_path!r} is a 1-bit mask, which is read as the values 0 and "
                     f"{BINARY_PAIR[1]} (white is {BINARY_PAIR[1]}), and data.ignore_index is "
                     f"{ignore_index}: every white (foreground) pixel would be void and the run "
                     "would train on one class; save the masks as 8-bit PNGs holding the class ids "
                     f"0 and 1, or give data.ignore_index another value")
        counts = np.bincount(mask.reshape(-1))
        present = np.flatnonzero(counts).tolist()
        union.update(present)
        class_pixels += np.pad(counts[:num_classes], (0, max(0, num_classes - len(counts))))
        if first_bad is None:
            bad = [value for value in present if not allowed(value)]
            if bad:
                first_bad = (mask_path, bad[0])
    values = tuple(sorted(union))
    if first_bad is not None:
        if values == BINARY_PAIR:
            raise DataSourceError(
                mask_key, f"every mask holds only the values 0 and 255 (e.g. {first_bad[0]!r}): a black "
                     f"and white mask, whose white is not a class id; {mask_threshold_key} is the "
                     "key that maps such a mask to two classes (this version does not accept it "
                     "yet), or save the masks with the class ids 0 and 1")
        raise DataSourceError(
            mask_key, f"{first_bad[0]!r} holds the value {first_bad[1]}, which is neither a class "
                 f"(0 to {num_classes - 1}) nor the ignore index ({ignore_index!r}); if {first_bad[1]} "
                 "marks pixels to leave out, set data.ignore_index to it, otherwise fix the mask "
                 f"(values seen in the {len(images)} mask(s) checked: {list(values)[:12]})")
    return PairScan(len(images), values, time.perf_counter() - started,
                    tuple(int(n) for n in class_pixels))


def refuse_absent_classes(scan: PairScan, class_names: Sequence[str], *, mask_key: str, where: str,
                          ignore_index: Optional[int]) -> None:
    """Refuse a declared class that no pixel of the scanned masks belongs to.

    :param scan: the :class:`PairScan` of the masks the model will be fit on.
    :param class_names: the declared classes, in id order.
    :param mask_key: the :class:`DataSourceError` key to name.
    :param where: what ``scan`` covers, in words, for the message ("the 12 masks of the fit split").
    :param ignore_index: the void value, named in the message as the usual cause.
    :raises DataSourceError: naming the first class with no pixel and the count of such classes.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-087
    # A declared class with no pixel in the FIT split is refused before `fit`, not warned: the run
    # knew (review-iter-3 concern 5: 1-bit masks with ignore_index 255 trained a whole run on one class
    # and reported "best 1.0000" against a baseline of 1.0000). Do NOT test the union of fit and
    # selection pixels: a class only the selection split holds cannot be learned from the fit split.
    absent = [index for index, pixels in enumerate(scan.class_pixels) if pixels == 0]
    if not absent:
        return
    named = ", ".join(f"{index} ({class_names[index]!r})" for index in absent[:5])
    raise DataSourceError(
        mask_key, f"{'class' if len(absent) == 1 else 'classes'} {named}"
             f"{' and others' if len(absent) > 5 else ''} {'has' if len(absent) == 1 else 'have'} "
             f"no pixel in {where}, "
             "so the model would be fit on fewer classes than data.class_names declares and a "
             "verdict on it would be about the others; fix data.class_names (or the mask values), "
             + (f"and check that data.ignore_index={ignore_index} is not the value that marks "
                "one of these classes, " if ignore_index is not None else "")
             + "or give a larger data.max_samples")


def sample_indices(count: int, sample: int) -> List[int]:
    """``sample`` positions evenly spaced over ``range(count)`` (all of them when ``count`` is not larger)."""
    if count <= sample:
        return list(range(count))
    return sorted({int(i * count / sample) for i in range(sample)})


def check_listing_sample(listing: PairListing, *, num_classes: int, ignore_index: Optional[int]) -> PairScan:
    """The pre-claim check: :data:`PRE_CLAIM_PAIRS` train pairs, PIL only, no TensorFlow op."""
    keys = source_keys(listing.source)
    picks = sample_indices(len(listing.train_masks), PRE_CLAIM_PAIRS)
    return check_pairs(
        [listing.train_images[i] for i in picks], [listing.train_masks[i] for i in picks],
        num_classes=num_classes, ignore_index=ignore_index, image_key=keys["train_image"],
        mask_key=keys["train_mask"])


# ---------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------

_HASH_CHUNK = 1 << 20


def pair_fingerprint(listing: PairListing, mode: str = "listing") -> str:
    """A sha256 over the rows of a listing, in listing order; see the module contract.

    A row is ``(split, relative image path, relative mask path, image size in bytes, mask size in
    bytes)``. The rows are hashed in LISTING ORDER, not sorted, for the reason
    ``classification_data.listing_fingerprint`` gives (the seeded split permutes listing positions),
    and the scheme is that function's ``FINGERPRINT_SCHEME``. With ``mode="content"`` each row is
    followed by the bytes of its image and then of its mask, streamed.

    :raises ValueError: for another ``mode``.
    """
    if mode not in FINGERPRINT_MODES:
        raise ValueError(f"fingerprint mode must be one of {FINGERPRINT_MODES}, got {mode!r}")
    rows = []
    for split, images, masks, bases in (
            ("train", listing.train_images, listing.train_masks, listing.train_bases),
            ("test", listing.test_images, listing.test_masks, listing.test_bases)):
        if images is None:
            continue
        for image, mask in zip(images, masks):
            rows.append((split, os.path.relpath(image, bases[0]).replace(os.sep, "/"),
                         os.path.relpath(mask, bases[1]).replace(os.sep, "/"),
                         os.path.getsize(image), os.path.getsize(mask), image, mask))
    digest = hashlib.sha256(f"pairs-{listing.source}".encode("utf-8"))
    digest.update(b"\x00" + FINGERPRINT_SCHEME.encode("utf-8"))
    if mode == "content":
        digest.update(b"\x00content")
    started = time.perf_counter()
    total = 0
    for row in rows:
        digest.update(b"\n" + json.dumps(list(row[:5]), ensure_ascii=False).encode("utf-8"))
        if mode == "content":
            for path in row[5:]:
                digest.update(b"\n")
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                        digest.update(chunk)
                        total += len(chunk)
    if mode == "content":
        logger.info(f"{listing.source}: content fingerprint of {len(rows)} pairs, "
                    f"{total / 1e6:.1f} MB, in {time.perf_counter() - started:.2f} s")
    return digest.hexdigest()


# ---------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------

def resize_image_bytes(image: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Resize a ``uint8`` ``(h, w, 3)`` image to ``size`` (bilinear, antialiased); byte values as ``float32``."""
    resized = tf.image.resize(tf.cast(image, tf.float32), size, method="bilinear", antialias=True)
    return tf.round(tf.clip_by_value(resized, 0.0, 255.0))


def resize_pair(image: tf.Tensor, mask: tf.Tensor, size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resize ``uint8`` ``(h, w, 3)`` image and ``int32`` ``(h, w)`` mask to ``size`` (height, width).

    :return: the image as ``float32`` in [0, 1] (bilinear with antialiasing, rounded to a byte
        value as the legacy Pet loader does, then divided by 255) and the mask as ``int32``
        (nearest neighbour: no value that is not in the mask appears, and void stays void).
    """
    labels = tf.image.resize(tf.cast(mask, tf.int32)[..., None], size, method="nearest")[..., 0]
    return resize_image_bytes(image, size) / 255.0, labels


def _decode_pair_py(image_path: bytes, mask_path: bytes) -> Tuple[np.ndarray, np.ndarray]:
    return read_pair(os.fsdecode(image_path), os.fsdecode(mask_path))


def decode_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """:func:`read_pair` as a ``tf.data`` map: ``uint8`` ``(H, W, 3)`` and ``int32`` ``(H, W)``, sizes unknown."""
    image, mask = tf.numpy_function(_decode_pair_py, [image_path, mask_path], [tf.uint8, tf.int32])
    image.set_shape([None, None, 3])
    mask.set_shape([None, None])
    return image, mask


def pair_examples(
        images: Sequence[str], masks: Sequence[str], *, training: bool, seed: int,
        augment: Optional[Augment], size: Tuple[int, int],
) -> "tf.data.Dataset":
    """The unbatched ``(image, mask)`` examples of a list of pairs.

    Training reshuffles the PAIRS every pass (cheap), then decodes, resizes and augments (the hook
    sees the ``[0, 1]`` image and the ``int32`` mask); evaluation keeps the list's order.
    ``from_tensor_slices`` of the lists keeps the length known.
    """
    dataset = tf.data.Dataset.from_tensor_slices((list(images), list(masks)))
    if training:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda i, m: resize_pair(*decode_pair(i, m), size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = with_hook(dataset, augment, seed)
    return dataset


def array_pairs(
        x: np.ndarray, y: np.ndarray, *, training: bool, seed: int, augment: Optional[Augment],
) -> "tf.data.Dataset":
    """The unbatched examples of in-memory ``uint8`` images ``(N, H, W, 3)`` and ``uint8`` masks ``(N, H, W)``."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda image, mask: (tf.cast(image, tf.float32) / 255.0, tf.cast(mask, tf.int32)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = with_hook(dataset, augment, seed)
    return dataset


def _batched(dataset: "tf.data.Dataset", count: int, batch_size: int, *, drop_remainder: bool,
             ) -> "tf.data.Dataset":
    """Batch with the length asserted, so ``fit`` never meets an unknown-length dataset."""
    # DECISION plan-2026-10-31T180033-2faf890a/D-074
    # The cardinality is ASSERTED on every source, not left to the maps: a map keeps it, a
    # `filter` or a generator would not, and Keras then logs a phantom zero batch (loss n/(n+1)),
    # which hid a model that learned nothing in an earlier plan. Do NOT remove it because the
    # maps "preserve" the length: the test that reads `cardinality()` is what keeps that true.
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(count))
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# The augmentation of a pair
# ---------------------------------------------------------------------

#: The ``transform_args.augment`` presets of a segmentation run.
SEG_AUGMENTS = ("none", "flip", "dihedral")


def pair_hook(op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]) -> Augment:
    """The seeded ``(seed, image, mask)`` hook that applies the geometric ``op`` to both at once.

    :param op: ``op(seed, tensor) -> tensor`` on a ``(H, W, C)`` float tensor (a
        ``train.common.image_augment`` op): it must move pixels and change no value.
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-075
    # ONE call of `op` on the image and the mask stacked as a fourth channel, split afterwards:
    # a flip or a quarter turn is a pure permutation of pixels, so the class ids survive exactly
    # (float32 holds every id below 2**24). Do NOT call `op` twice, once per tensor: even with the
    # same seed that is two draws by contract only, and a bilinear or padded op would invent
    # class ids. The test that holds this is "a mask that is a function of the image stays that
    # function", over 200 seeds.
    @seeded
    def hook(seed: tf.Tensor, image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        stacked = tf.concat([image, tf.cast(mask, tf.float32)[..., None]], axis=-1)
        stacked = op(seed, stacked)
        return stacked[..., :3], tf.cast(stacked[..., 3], tf.int32)

    return hook


def build_segmentation_augment(preset: str, size: Tuple[int, int]) -> Optional[Augment]:
    """The train hook of a segmentation augmentation preset; see the module contract."""
    from train.common.image_augment import random_dihedral, random_flip

    if preset not in SEG_AUGMENTS:
        raise ValueError(f"Unknown segmentation augmentation {preset!r}; known: {list(SEG_AUGMENTS)}")
    if preset == "none":
        return None
    if preset == "dihedral":
        if size[0] != size[1]:
            raise ValueError(
                f"dihedral turns the image by quarter turns, which swaps the sides of a "
                f"{size[0]}x{size[1]} (height x width) image; use flip, or a square size")
        return pair_hook(random_dihedral())
    return pair_hook(random_flip())


# ---------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentationData:
    """What a segmentation trainer consumes, whatever the source.

    :param train_ds: shuffled, batched ``(image, mask)`` pipeline (drops the remainder).
    :param select_ds: the selection split: a held-out slice of train; never the test set.
    :param test_ds: the test split, or ``None`` when the source has none (``n_test`` 0, empty ``x_fig``).
    :param steps_per_epoch: batches in one epoch of ``train_ds``.
    :param n_train, n_select, n_test: pairs per split.
    :param x_fig, y_fig: the first ``eval_samples`` test pairs in memory: ``float32``
        ``(N, H, W, 3)`` in [0, 1] and ``int32`` ``(N, H, W)``.
    :param mean, std: zeros and ones: segmentation inputs are in [0, 1], never standardized.
    :param input_size: ``(height, width)`` every image and mask was resized to.
    :param num_classes, class_names: the classes; ``class_names`` has one name per class.
    :param ignore_index: the void value of the masks, or ``None``.
    :param selection_split, test_split: plain-language provenance of the two splits.
    :param image_files, mask_files: the files ``train_ds`` and ``select_ds`` read (fit split, then
        selection split, in step), for naming the file behind a decode error; ``None`` for Pet.
    :param fingerprint: :func:`pair_fingerprint`, or ``None`` for Pet, which carries none.
    :param mask_values: the distinct mask values the scan saw (before the resize), or ``()`` for Pet.
    """

    train_ds: "tf.data.Dataset"
    select_ds: "tf.data.Dataset"
    test_ds: Optional["tf.data.Dataset"]
    steps_per_epoch: int
    n_train: int
    n_select: int
    n_test: int
    x_fig: np.ndarray
    y_fig: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    input_size: Tuple[int, int]
    num_classes: int
    class_names: List[str]
    ignore_index: Optional[int]
    selection_split: str
    test_split: str
    image_files: Optional[Tuple[str, ...]] = None
    mask_files: Optional[Tuple[str, ...]] = None
    fingerprint: Optional[str] = None
    mask_values: Tuple[int, ...] = ()

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """``(H, W, 3)``."""
        return (self.input_size[0], self.input_size[1], 3)

    @property
    def select_steps(self) -> int:
        """Batches in one pass of ``select_ds``, read from the dataset."""
        return int(self.select_ds.cardinality())


def _figures(dataset: Optional["tf.data.Dataset"], count: int, size: Tuple[int, int],
             ) -> Tuple[np.ndarray, np.ndarray]:
    """The first ``count`` examples of an unbatched dataset as numpy arrays (empty without a dataset)."""
    if dataset is None or count < 1:
        return np.zeros((0, size[0], size[1], 3), np.float32), np.zeros((0, size[0], size[1]), np.int32)
    images, labels = [], []
    for image, mask in dataset.take(count).batch(64).as_numpy_iterator():
        images.append(image)
        labels.append(mask)
    return np.concatenate(images), np.concatenate(labels)


def _finish(
        *, train: "tf.data.Dataset", select: "tf.data.Dataset", test: Optional["tf.data.Dataset"],
        n_train: int, n_select: int, n_test: int, batch_size: int, eval_batch_size: int,
        eval_samples: int, size: Tuple[int, int], class_names: Sequence[str],
        ignore_index: Optional[int], max_samples_flag: str, **provenance,
) -> SegmentationData:
    """Batch the three unbatched pipelines and assemble the record."""
    steps = steps_per_epoch_for(n_train, batch_size, max_samples_flag)
    x_fig, y_fig = _figures(test, min(eval_samples, n_test), size)
    return SegmentationData(
        train_ds=_batched(train, n_train, batch_size, drop_remainder=True),
        select_ds=_batched(select, n_select, eval_batch_size, drop_remainder=False),
        test_ds=None if test is None else _batched(test, n_test, eval_batch_size, drop_remainder=False),
        steps_per_epoch=steps, n_train=n_train, n_select=n_select, n_test=n_test,
        x_fig=x_fig, y_fig=y_fig, mean=np.zeros(3, np.float32), std=np.ones(3, np.float32),
        input_size=size, num_classes=len(class_names), class_names=list(class_names),
        ignore_index=ignore_index, **provenance)


def load_pair_data(
        listing: PairListing, *, num_classes: int, class_names: Sequence[str],
        ignore_index: Optional[int], batch_size: int, seed: int, size: Tuple[int, int],
        validation_split: Optional[float], eval_samples: Optional[int], max_samples: Optional[int],
        fingerprint: str = "listing", augment: Optional[Augment] = None,
        eval_batch_size: Optional[int] = None, max_samples_flag: str = "data.max_samples",
) -> SegmentationData:
    """The record of a listed folder or CSV source.

    The protocol is the other loaders': a seeded ``validation_split`` cut of the TRAIN pairs is
    the selection split and the test pairs are only scored. Before any pipeline is built, every
    pair the run will use is checked (:func:`check_pairs`; the test images are decoded in full,
    so a corrupt test file never costs a finished run its report) and the duration is logged.

    :param size: ``(height, width)`` every image and mask is resized to.
    :param eval_samples: test pairs kept in memory, ``None`` is :data:`DEFAULT_EVAL_SAMPLES`.
    :param augment: the train hook ``(seed, image, mask) -> (image, mask)`` (marked with
        ``dl_techniques.datasets.vision.augment_hook.seeded``), or ``None``.
    :param eval_batch_size: rows per batch of the selection and test pipelines, ``None`` is ``batch_size``.
    :raises DataSourceError: a file or a mask value that cannot be used.
    :raises ValueError: a bad split, or a batch larger than the fit split.
    """
    keys = source_keys(listing.source)
    split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    train_images, train_masks = np.asarray(listing.train_images), np.asarray(listing.train_masks)
    n_test_files = 0 if listing.test_images is None else len(listing.test_images)
    fit_idx, select_idx, test_order = seeded_split_indices(
        len(train_images), n_test_files, seed, split, max_samples)
    steps_per_epoch_for(len(fit_idx), batch_size, max_samples_flag)
    used_train = np.concatenate([fit_idx, select_idx])
    test_images = test_masks = None
    if listing.test_images is not None:
        test_images = np.asarray(listing.test_images)[test_order]
        test_masks = np.asarray(listing.test_masks)[test_order]
    started = time.perf_counter()
    scans = [check_pairs(train_images[part].tolist(), train_masks[part].tolist(),
                         num_classes=num_classes, ignore_index=ignore_index,
                         image_key=keys["train_image"], mask_key=keys["train_mask"])
             for part in (fit_idx, select_idx)]
    values = set(scans[0].values) | set(scans[1].values)
    refuse_absent_classes(
        scans[0], class_names, mask_key=keys["train_mask"], ignore_index=ignore_index,
        where=f"any of the {len(fit_idx)} masks of the fit split")
    if test_images is not None:
        test_scan = check_pairs(test_images.tolist(), test_masks.tolist(), num_classes=num_classes,
                                ignore_index=ignore_index, image_key=keys["test_image"],
                                mask_key=keys["test_mask"], image_pixels=True)
        values.update(test_scan.values)
    logger.info(
        f"{listing.source}: checked {len(used_train)} train and {0 if test_images is None else len(test_images)} "
        f"test pairs (sizes, mask values {sorted(values)[:12]}) in {time.perf_counter() - started:.1f} s")
    fit_images, fit_masks = train_images[fit_idx].tolist(), train_masks[fit_idx].tolist()
    sel_images, sel_masks = train_images[select_idx].tolist(), train_masks[select_idx].tolist()

    def examples(images, masks, training):
        return pair_examples(images, masks, training=training, seed=seed if training else 0,
                             augment=augment, size=size)

    test = None if test_images is None else examples(test_images.tolist(), test_masks.tolist(), False)
    return _finish(
        train=examples(fit_images, fit_masks, True), select=examples(sel_images, sel_masks, False),
        test=test, n_train=len(fit_idx), n_select=len(select_idx),
        n_test=0 if test_images is None else len(test_images), batch_size=batch_size,
        eval_batch_size=batch_size if eval_batch_size is None else eval_batch_size,
        eval_samples=DEFAULT_EVAL_SAMPLES if eval_samples is None else eval_samples, size=size,
        class_names=class_names, ignore_index=ignore_index, max_samples_flag=max_samples_flag,
        selection_split=f"validation_split={split} of the train pairs (seeded permutation)",
        test_split="none: the data has no test split" if test is None else "the test pairs",
        image_files=tuple(fit_images) + tuple(sel_images), mask_files=tuple(fit_masks) + tuple(sel_masks),
        fingerprint=pair_fingerprint(listing, fingerprint), mask_values=tuple(sorted(values)))


# ---------------------------------------------------------------------
# Oxford-IIIT Pet from the local TFDS cache
# ---------------------------------------------------------------------

def load_oxford_pet_arrays(size: Tuple[int, int]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Decode ``oxford_iiit_pet`` (:data:`PET_TFDS_VERSION`) from the local TFDS cache into ``uint8`` arrays; the one TFDS seam.

    Images are resized to ``size`` (:func:`resize_pair`, the legacy Pet loader's arithmetic) and
    the label is the TFDS mask minus 1 (1 pet, 2 background, 3 border become 0, 1, 2). The order is
    the TFDS shard order, so the arrays are identical from run to run. Nothing is downloaded.

    :param size: ``(height, width)``.
    :return: ``((x_train, y_train), (x_test, y_test))``: ``uint8`` ``(N, H, W, 3)`` and ``uint8``
        ``(N, H, W)`` with values in ``{0, 1, 2}``; 3680 train and 3669 test pairs.
    :raises RuntimeError: when the cache cannot be read with ``download=False``.
    :raises ValueError: when a mask holds a value outside ``{1, 2, 3}``.
    """
    import tensorflow_datasets as tfds  # lazy: keeps the config path light

    def convert(example: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        image, mask = example["image"], tf.squeeze(example["segmentation_mask"], -1)
        labels = tf.image.resize(tf.cast(mask, tf.int32)[..., None], size, method="nearest")[..., 0]
        return tf.cast(resize_image_bytes(image, size), tf.uint8), labels - 1

    def split(name: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            dataset = tfds.load(PET_TFDS_NAME, split=name, download=False, shuffle_files=False)
        except Exception as exc:  # noqa: BLE001 - re-raised with the fix named
            raise RuntimeError(
                f"Could not read {PET_TFDS_NAME!r} split {name!r} from the local TFDS cache with "
                f"download=False: {type(exc).__name__}: {exc}. Set TFDS_DATA_DIR to the directory "
                f"that holds {PET_TFDS_DIRECTORY}/{PET_TFDS_VERSION} (nothing is downloaded).") from exc
        batches = tfds.as_numpy(dataset.map(convert, num_parallel_calls=tf.data.AUTOTUNE).batch(PET_DECODE_BATCH))
        images, labels = zip(*batches)
        x, y = np.concatenate(images), np.concatenate(labels)
        if y.min() < 0 or y.max() >= len(PET_CLASS_NAMES):
            raise ValueError(
                f"{PET_TFDS_NAME} split {name!r}: mask - 1 must lie in [0, {len(PET_CLASS_NAMES) - 1}], "
                f"got [{y.min()}, {y.max()}]")
        return x, y.astype(np.uint8)

    return split("train"), split("test")


def load_pet_data(
        *, batch_size: int, seed: int, size: Tuple[int, int], validation_split: Optional[float],
        eval_samples: Optional[int], max_samples: Optional[int], augment: Optional[Augment] = None,
        eval_batch_size: Optional[int] = None, max_samples_flag: str = "data.max_samples",
) -> SegmentationData:
    """The record of Oxford-IIIT Pet: 3 classes (pet, background, border), no void, 3669 test pairs.

    The cut is the legacy trainer's: a seeded permutation of the train split, truncated by
    ``max_samples``, whose first ``n_select`` entries select. The test pairs are only scored.
    """
    split = DEFAULT_VALIDATION_SPLIT if validation_split is None else validation_split
    (x_train, y_train), (x_test, y_test) = load_oxford_pet_arrays(size)
    fit_idx, select_idx, test_order = seeded_split_indices(
        len(x_train), len(x_test), seed, split, max_samples)
    steps_per_epoch_for(len(fit_idx), batch_size, max_samples_flag)
    x_test, y_test = x_test[test_order], y_test[test_order]
    return _finish(
        train=array_pairs(x_train[fit_idx], y_train[fit_idx], training=True, seed=seed, augment=augment),
        select=array_pairs(x_train[select_idx], y_train[select_idx], training=False, seed=0, augment=augment),
        test=array_pairs(x_test, y_test, training=False, seed=0, augment=augment),
        n_train=len(fit_idx), n_select=len(select_idx), n_test=len(x_test), batch_size=batch_size,
        eval_batch_size=batch_size if eval_batch_size is None else eval_batch_size,
        eval_samples=DEFAULT_EVAL_SAMPLES if eval_samples is None else eval_samples, size=size,
        class_names=PET_CLASS_NAMES, ignore_index=None, max_samples_flag=max_samples_flag,
        selection_split=f"validation_split={split} of the train split (seeded permutation)",
        test_split="the Oxford-IIIT Pet test split")
