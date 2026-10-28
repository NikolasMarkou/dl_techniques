"""
Local ImageNet ILSVRC2012 (ImageNet-1k) loader for the extracted-tar layout.

Reads straight from disk, with no TFDS involved::

    <root>/train/<wnid>/*.JPEG      1000 class folders, 1,281,167 images
    <root>/val/ILSVRC2012_val_*.JPEG  50,000 images, flat and unlabeled

and yields batched ``(image, label)`` pairs: ``image`` is float32
``(B, S, S, 3)``, ``label`` is int32 ``(B,)`` in ``[0, 999]``.

Class indexing
--------------
The class index of a wnid is its position in the SORTED list of the wnid
folders under ``train/`` (the standard ILSVRC order).  This is NOT the
ILSVRC2012 numeric id used by the devkit: those ids are assigned in a
different order, so the validation ground truth is translated
``ILSVRC id -> wnid -> sorted index`` through the devkit's ``meta.mat``.

Validation labels
-----------------
The image archives carry no validation labels.  They come from the
ILSVRC2012 devkit (``ILSVRC2012_devkit_t12.tar.gz``), which must be extracted
so that ``meta.mat`` and ``ILSVRC2012_validation_ground_truth.txt`` sit in
``devkit_dir`` (default ``<root>/devkit``) or in its ``data/`` subfolder.
Training never needs the devkit.

Pipeline
--------
The 1.28M-entry file list is scanned once and cached under ``<root>/.index/``
(the dataset sits on a spinning disk; a per-run glob is minutes).  Shuffling
is done on the path list before any file is read.  Decoding uses
``tf.io.decode_image``, which sniffs the magic bytes, so the PNG stored under
a ``.JPEG`` name and the CMYK JPEGs decode (measured on the real files; in
this TF build ``tf.io.decode_jpeg`` accepts the PNG too).

Held-out slice of train
-----------------------
ImageNet has no public test labels, so a trainer that wants a test set which
never influences model selection takes one from train.  With
``holdout_per_class=k`` the ``holdout`` split is ``k`` images of every class
(seeded, stratified, un-augmented) and ``train`` is everything else; the
official ``val`` split is then free to serve as the test set.  The default
``0`` leaves ``train`` whole.

Training augmentation is a random-resized crop (area 0.08-1, aspect 3/4-4/3)
plus a horizontal flip, and optionally (``augment_color``) random brightness,
contrast, saturation and hue.  Evaluation resizes the shorter side to
``image_size / 0.875`` and center-crops.  The returned dataset is finite and
does not repeat: Keras ``fit`` re-iterates it (reshuffling) every epoch.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.crops import (
    EVAL_CROP_FRACTION,
    RANDOM_RESIZED_CROP_AREA,
    RANDOM_RESIZED_CROP_RATIO,
)
from dl_techniques.utils.logger import logger


IMAGENET2012_DEFAULT_ROOT: str = "/media/arxwn/data0_4tb/datasets/imagenet2012"
IMAGENET2012_NUM_CLASSES: int = 1000

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

_SPLITS = ("train", "val", "holdout")
_NORMALIZE_MODES = ("imagenet", "unit", "none")
_IMAGE_SUFFIXES = (".jpeg", ".jpg")
_INDEX_DIRNAME = ".index"
_META_FILE = "meta.mat"
_VAL_GROUND_TRUTH_FILE = "ILSVRC2012_validation_ground_truth.txt"


@dataclass
class ImageNet2012Config:
    """Configuration for :func:`load_imagenet2012`.

    ``augment`` and ``shuffle`` apply to the ``train`` split only; ``val`` is
    always un-augmented and in file-name order.
    """

    root: str = IMAGENET2012_DEFAULT_ROOT
    devkit_dir: Optional[str] = None  # default: <root>/devkit
    image_size: int = 224
    batch_size: int = 128

    # "imagenet": (x/255 - mean) / std, "unit": x/255, "none": raw 0..255 floats.
    normalize: str = "imagenet"

    # Training only
    augment: bool = True
    # Random brightness (+-0.2), contrast (0.8-1.2), saturation (0.8-1.2) and
    # hue (+-0.1), applied on [0, 1] pixels after the crop and flip.  Only
    # takes effect together with ``augment``.
    augment_color: bool = False
    shuffle: bool = True

    seed: int = 0
    # Random (seeded) subset of the split for quick smoke runs.  The first N
    # files in order would be one or two classes, so it is not a prefix.
    max_images: Optional[int] = None
    drop_remainder: bool = False
    num_parallel_calls: int = tf.data.AUTOTUNE

    # A class-balanced slice of ``train`` for model selection.  With a positive
    # count the ``holdout`` split holds that many images per class (a seeded
    # draw, independent of ``seed``) and ``train`` holds every other image.
    holdout_per_class: int = 0
    holdout_seed: int = 0

    def __post_init__(self) -> None:
        if self.normalize not in _NORMALIZE_MODES:
            raise ValueError(
                f"normalize must be one of {_NORMALIZE_MODES}, got {self.normalize!r}"
            )
        if self.image_size <= 0 or self.batch_size <= 0:
            raise ValueError(
                f"image_size and batch_size must be positive, got "
                f"{self.image_size} and {self.batch_size}"
            )
        if self.max_images is not None and self.max_images <= 0:
            raise ValueError(f"max_images must be positive, got {self.max_images}")
        if self.holdout_per_class < 0:
            raise ValueError(
                f"holdout_per_class must be >= 0, got {self.holdout_per_class}"
            )


class ImageNet2012Data(NamedTuple):
    """A split ready for ``model.fit`` / ``model.evaluate``."""

    dataset: tf.data.Dataset
    num_samples: int
    steps_per_epoch: int
    num_classes: int
    #: wnids in class-index order (``class_names[label]`` is the wnid).
    class_names: List[str]


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------


def _list_classes(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet train directory not found: {train_dir}. Class indices "
            f"are the sorted wnid folders under train/, so it is needed for "
            f"every split."
        )
    with os.scandir(train_dir) as it:
        classes = sorted(e.name for e in it if e.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders under {train_dir}")
    if len(classes) != IMAGENET2012_NUM_CLASSES:
        logger.warning(
            f"{train_dir} has {len(classes)} class folders, expected "
            f"{IMAGENET2012_NUM_CLASSES}"
        )
    return classes


def _signature(split: str, split_dir: str, dirs: List[str]) -> str:
    """Fingerprint of every directory the index depends on.

    Adding or removing a file changes its directory's mtime, so a stale cache
    is detected without listing 1.28M entries.
    """
    h = hashlib.sha1()
    h.update(f"{split}:{len(dirs)}".encode())
    for d in dirs:
        h.update(f"{d}:{os.stat(os.path.join(split_dir, d) if d else split_dir).st_mtime_ns}".encode())
    return h.hexdigest()


def _scan(split: str, split_dir: str, classes: List[str]) -> List[str]:
    rel: List[str] = []
    if split == "train":
        for wnid in classes:
            with os.scandir(os.path.join(split_dir, wnid)) as it:
                names = sorted(
                    e.name for e in it
                    if e.is_file() and e.name.lower().endswith(_IMAGE_SUFFIXES)
                )
            rel.extend(f"{wnid}/{n}" for n in names)
    else:
        with os.scandir(split_dir) as it:
            rel = sorted(
                e.name for e in it
                if e.is_file() and e.name.lower().endswith(_IMAGE_SUFFIXES)
            )
    return rel


def _load_or_build_index(root: str, split: str, classes: List[str]) -> List[str]:
    """Relative image paths of ``split`` in a stable sorted order, cached."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"ImageNet {split} directory not found: {split_dir}")

    dirs = classes if split == "train" else [""]
    signature = _signature(split, split_dir, dirs)
    cache_file = os.path.join(root, _INDEX_DIRNAME, f"{split}.txt")

    if os.path.isfile(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        if lines and lines[0] == signature:
            rel = [ln for ln in lines[1:] if ln]
            logger.info(f"ImageNet {split}: {len(rel)} files from index cache {cache_file}")
            return rel

    t0 = time.time()
    rel = _scan(split, split_dir, classes)
    logger.info(f"ImageNet {split}: scanned {len(rel)} files in {time.time() - t0:.1f}s")
    if not rel:
        raise FileNotFoundError(f"No images found under {split_dir}")

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        tmp = f"{cache_file}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(signature + "\n" + "\n".join(rel) + "\n")
        os.replace(tmp, cache_file)
    except OSError as e:
        logger.warning(f"Could not write ImageNet index cache {cache_file}: {e}")
    return rel


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _find_devkit_file(devkit_dir: str, name: str) -> Optional[str]:
    for d in (devkit_dir, os.path.join(devkit_dir, "data")):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _read_val_labels(devkit_dir: str, classes: List[str], files: List[str]) -> np.ndarray:
    """Class index per validation file, via ``ILSVRC id -> wnid -> index``."""
    meta_path = _find_devkit_file(devkit_dir, _META_FILE)
    gt_path = _find_devkit_file(devkit_dir, _VAL_GROUND_TRUTH_FILE)
    if meta_path is None or gt_path is None:
        raise FileNotFoundError(
            f"ImageNet validation labels need the ILSVRC2012 devkit. Expected "
            f"{_META_FILE} and {_VAL_GROUND_TRUTH_FILE} in {devkit_dir} (or "
            f"{os.path.join(devkit_dir, 'data')}); extract "
            f"ILSVRC2012_devkit_t12.tar.gz there. Found meta.mat: "
            f"{meta_path is not None}, ground truth: {gt_path is not None}."
        )

    from scipy.io import loadmat

    synsets = loadmat(meta_path, squeeze_me=True, struct_as_record=False)["synsets"]
    id_to_wnid = {}
    for s in np.atleast_1d(synsets):
        ilsvrc_id = int(np.asarray(s.ILSVRC2012_ID).item())
        if ilsvrc_id <= IMAGENET2012_NUM_CLASSES:  # the rest are non-leaf synsets
            id_to_wnid[ilsvrc_id] = str(np.asarray(s.WNID).item())

    class_index = {w: i for i, w in enumerate(classes)}
    missing = sorted(set(id_to_wnid.values()) - set(class_index))
    if missing:
        raise ValueError(
            f"{len(missing)} wnids in {meta_path} have no folder under train/, "
            f"e.g. {missing[:3]}"
        )

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = [int(ln) for ln in f.read().split()]
    unknown = sorted(set(ground_truth) - set(id_to_wnid))
    if unknown:
        raise ValueError(
            f"{gt_path} holds ILSVRC ids absent from {meta_path}: {unknown[:5]}"
        )

    # Val file names carry their 1-based position in the ground-truth file.
    labels = np.empty(len(files), dtype=np.int32)
    for i, name in enumerate(files):
        pos = int(os.path.splitext(name)[0].rsplit("_", 1)[-1]) - 1
        if not 0 <= pos < len(ground_truth):
            raise ValueError(
                f"{name} is image #{pos + 1} but the ground truth lists "
                f"{len(ground_truth)} images"
            )
        labels[i] = class_index[id_to_wnid[ground_truth[pos]]]
    return labels


def _train_labels(files: List[str], classes: List[str]) -> np.ndarray:
    class_index = {w: i for i, w in enumerate(classes)}
    return np.fromiter(
        (class_index[f.split("/", 1)[0]] for f in files), dtype=np.int32, count=len(files)
    )


def _holdout_mask(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """True for the ``per_class`` images of every class that form the holdout.

    The draw for a class depends only on ``(seed, class index)`` and the
    class's own image count, so it does not move when other classes change.
    """
    mask = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        members = np.flatnonzero(labels == cls)
        if len(members) <= per_class:
            raise ValueError(
                f"holdout_per_class={per_class} leaves class {int(cls)} no train "
                f"image: it has {len(members)}"
            )
        mask[np.random.default_rng([seed, int(cls)]).choice(members, per_class, replace=False)] = True
    return mask


# ---------------------------------------------------------------------------
# tf.data
# ---------------------------------------------------------------------------


def _decode(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    return image


def _random_resized_crop(image: tf.Tensor, size: int, seed: tf.Tensor) -> tf.Tensor:
    begin, crop_size, _ = tf.image.stateless_sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], shape=[1, 1, 4]),
        seed=seed,
        area_range=RANDOM_RESIZED_CROP_AREA,
        aspect_ratio_range=RANDOM_RESIZED_CROP_RATIO,
        max_attempts=10,
        use_image_if_no_bounding_boxes=True,
    )
    crop = tf.slice(image, begin, crop_size)
    return tf.image.resize(crop, [size, size], method="bilinear", antialias=True)


def _center_crop(image: tf.Tensor, size: int) -> tf.Tensor:
    shape = tf.shape(image)
    short = tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)
    scale = tf.cast(math.ceil(size / EVAL_CROP_FRACTION), tf.float32) / short
    new_hw = tf.cast(tf.round(tf.cast(shape[:2], tf.float32) * scale), tf.int32)
    image = tf.image.resize(image, new_hw, method="bilinear", antialias=True)
    return tf.image.resize_with_crop_or_pad(image, size, size)


def _color_jitter(image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Brightness, contrast, saturation, hue on a 0..255 image; returns 0..255.

    The ops work on [0, 1] pixels, so a brightness delta of 0.2 is 20% of the
    range (applied to 0..255 values it would be a no-op).
    """
    image = image / 255.0
    image = tf.image.stateless_random_brightness(image, 0.2, seed=seed + 2)
    image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed=seed + 3)
    image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seed=seed + 4)
    image = tf.image.stateless_random_hue(image, 0.1, seed=seed + 5)
    return tf.clip_by_value(image, 0.0, 1.0) * 255.0


def _normalize(image: tf.Tensor, mode: str) -> tf.Tensor:
    if mode == "none":
        return image
    image = image / 255.0
    if mode == "imagenet":
        image = (image - tf.constant(IMAGENET_MEAN)) / tf.constant(IMAGENET_STD)
    return image


def _make_dataset(
    paths: np.ndarray, labels: np.ndarray, is_train: bool, cfg: ImageNet2012Config
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    shuffle = is_train and cfg.shuffle
    if shuffle:
        # Strings only at this point, so a full-size buffer is cheap.
        ds = ds.shuffle(len(paths), seed=cfg.seed, reshuffle_each_iteration=True)

    size, mode = cfg.image_size, cfg.normalize

    if is_train and cfg.augment:
        # One seed per element, attached BEFORE the parallel map, so an
        # element's augmentation does not depend on which worker finishes first.
        # Re-randomized per epoch (without it every epoch reuses one seed
        # stream); measured: still identical across two builds of the same seed.
        seeds = tf.data.Dataset.random(
            seed=cfg.seed, rerandomize_each_iteration=True
        ).batch(2)
        ds = tf.data.Dataset.zip((ds, seeds))

        def load(item, seed):
            path, label = item
            image = _random_resized_crop(_decode(path), size, seed)
            flip = tf.random.stateless_uniform([], seed=seed + 1) < 0.5
            image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
            if cfg.augment_color:
                image = _color_jitter(image, seed)
            return _normalize(image, mode), label
    else:
        def load(path, label):
            return _normalize(_center_crop(_decode(path), size), mode), label

    ds = ds.map(load, num_parallel_calls=cfg.num_parallel_calls, deterministic=not shuffle)
    ds = ds.batch(cfg.batch_size, drop_remainder=cfg.drop_remainder)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_imagenet2012(
    split: str, config: Optional[ImageNet2012Config] = None
) -> ImageNet2012Data:
    """Build a batched ``(image, label)`` dataset for one ImageNet split.

    :param split: ``"train"``, ``"val"`` or ``"holdout"``.  ``"holdout"`` is
        the ``holdout_per_class`` images per class that ``"train"`` leaves out
        (un-augmented, in file-name order); it needs ``holdout_per_class > 0``.
    :param config: Loader configuration; defaults to :class:`ImageNet2012Config`.
    :raises FileNotFoundError: Missing split folder, or ``split="val"`` without
        the devkit label files.
    :return: The dataset plus the counts a trainer needs.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    cfg = config if config is not None else ImageNet2012Config()

    if split == "holdout" and cfg.holdout_per_class == 0:
        raise ValueError("split 'holdout' needs holdout_per_class > 0, got 0")

    classes = _list_classes(os.path.join(cfg.root, "train"))
    source = "val" if split == "val" else "train"
    files = _load_or_build_index(cfg.root, source, classes)
    if source == "train":
        labels = _train_labels(files, classes)
        if cfg.holdout_per_class > 0:
            held = _holdout_mask(labels, cfg.holdout_per_class, cfg.holdout_seed)
            keep = held if split == "holdout" else ~held
            files = [f for f, k in zip(files, keep) if k]
            labels = labels[keep]
    else:
        devkit = cfg.devkit_dir or os.path.join(cfg.root, "devkit")
        labels = _read_val_labels(devkit, classes, files)

    if split != "holdout" and cfg.max_images is not None and cfg.max_images < len(files):
        keep = np.sort(
            np.random.default_rng(cfg.seed).choice(len(files), cfg.max_images, replace=False)
        )
        files = [files[i] for i in keep]
        labels = labels[keep]

    split_dir = os.path.join(cfg.root, source)
    paths = np.array([os.path.join(split_dir, f) for f in files])
    n = len(paths)
    steps = n // cfg.batch_size if cfg.drop_remainder else math.ceil(n / cfg.batch_size)
    logger.info(f"ImageNet {split}: {n} images, {len(classes)} classes, {steps} steps/epoch")

    dataset = _make_dataset(paths, labels, split == "train", cfg)
    return ImageNet2012Data(dataset, n, steps, len(classes), classes)
