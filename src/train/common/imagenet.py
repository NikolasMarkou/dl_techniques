"""ImageNet-1k wiring shared by the trainers that read the extracted dataset.

``dl_techniques.datasets.vision.imagenet2012.load_imagenet2012`` owns the
pipeline (decode, crop, augment, normalize). This module owns what every
trainer around it repeated: the ``--imagenet-root`` / ``--devkit-dir`` flags,
the root-exists check, the (train, select, test) triple with the same loader
settings, the class-count check, and a small in-memory head of a split for the
end-of-run figures. :func:`load_imagenet_selection_splits` takes the selection set
from a class-balanced slice of train so the official val split stays a clean test
set; trainers reach it through ``train.common.classification_data``, which adds the
other datasets and the normalization statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from dl_techniques.datasets.vision.imagenet2012 import (
    IMAGENET2012_DEFAULT_ROOT,
    ImageNet2012Config,
    ImageNet2012Data,
    load_imagenet2012,
)
from dl_techniques.utils.logger import logger

#: Selection images per class held out of train (10,000 of 1,281,167).
DEFAULT_HOLDOUT_PER_CLASS = 10


def add_imagenet_arguments(parser: argparse.ArgumentParser, note: str = "") -> None:
    """Add ``--imagenet-root`` and ``--devkit-dir`` to ``parser``.

    Args:
        parser: Parser to extend.
        note: Text put before both help strings, e.g. ``"Used only when
            --dataset imagenet: "``.
    """
    parser.add_argument(
        "--imagenet-root", type=str, default=IMAGENET2012_DEFAULT_ROOT,
        help=f"{note}folder holding train/<wnid>/ and the flat val/ "
             f"(default: {IMAGENET2012_DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--devkit-dir", type=str, default=None,
        help=f"{note}ILSVRC2012 devkit folder with meta.mat and the val ground "
             f"truth (default: <imagenet-root>/devkit)",
    )


def require_imagenet_root(root: str) -> None:
    """Raise ``ValueError`` when ``root`` is not an existing path."""
    if not Path(root).exists():
        raise ValueError(f"ImageNet root does not exist: {root}")


def add_holdout_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--holdout-per-class`` to ``parser`` (see :func:`load_imagenet_selection_splits`)."""
    parser.add_argument(
        "--holdout-per-class", type=int, default=DEFAULT_HOLDOUT_PER_CLASS,
        help="ImageNet images per class taken out of train for early stopping and "
             "checkpoint selection; the official val split is then used only as the "
             f"test set (default: {DEFAULT_HOLDOUT_PER_CLASS}, i.e. "
             f"{DEFAULT_HOLDOUT_PER_CLASS * 1000} images)",
    )


def _loader_kwargs(
        root: str, devkit_dir: Optional[str], image_size: int, batch_size: int,
        seed: int, augment: bool, augment_color: bool, num_parallel_calls: int,
        normalize: str,
) -> dict:
    return dict(
        root=root, devkit_dir=devkit_dir, image_size=image_size,
        batch_size=batch_size, augment=augment, augment_color=augment_color,
        seed=seed, num_parallel_calls=num_parallel_calls, normalize=normalize,
    )


def _check_num_classes(
        data: ImageNet2012Data, num_classes: Optional[int], root: str
) -> None:
    if num_classes is not None and data.num_classes != num_classes:
        raise ValueError(
            f"num_classes={num_classes} but {root} has {data.num_classes} classes"
        )


def load_imagenet_selection_splits(
        root: str,
        devkit_dir: Optional[str],
        image_size: int,
        batch_size: int,
        seed: int,
        augment: bool,
        holdout_per_class: int = DEFAULT_HOLDOUT_PER_CLASS,
        normalize: str = "imagenet",
        augment_color: bool = False,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        num_classes: Optional[int] = None,
        max_images: Optional[int] = None,
) -> Tuple[ImageNet2012Data, ImageNet2012Data, ImageNet2012Data]:
    """Build ``(train, select, test)`` where ``test`` never influences selection.

    ``select`` is ``holdout_per_class`` images per class taken out of train, for
    early stopping and checkpoint selection; ``train`` is every other train
    image; ``test`` is the official val split. As in :func:`load_imagenet_selection_splits`
    train drops its ragged last batch, and all three are finite.

    Args:
        root: ImageNet folder holding ``train/`` and ``val/``.
        devkit_dir: Devkit folder; ``None`` means ``<root>/devkit``.
        image_size: Square output side.
        batch_size: Batch size of all three splits.
        seed: Seed of the train shuffle and augmentation.
        augment: Random-resized-crop and flip on train.
        holdout_per_class: Selection images per class; must be positive.
        normalize: ``"imagenet"`` (mean/std), ``"unit"`` ([0, 1]) or ``"none"``.
        augment_color: Brightness, contrast, saturation and hue on train.
        num_parallel_calls: Parallel decode calls (default: autotuned).
        num_classes: When given, the number of class folders under ``train/``
            must equal it.
        max_images: Smoke-run cap on train and on test; ``select`` is never capped.

    Raises:
        ValueError: ``holdout_per_class`` is not positive, or ``num_classes``
            differs from the number of class folders.
    """
    if holdout_per_class < 1:
        raise ValueError(
            f"holdout_per_class must be positive to take a selection set out of "
            f"train, got {holdout_per_class}"
        )
    kwargs = _loader_kwargs(root, devkit_dir, image_size, batch_size, seed, augment,
                            augment_color, num_parallel_calls, normalize)
    kwargs["holdout_per_class"] = holdout_per_class
    train = load_imagenet2012(
        "train", ImageNet2012Config(drop_remainder=True, max_images=max_images, **kwargs))
    select = load_imagenet2012("holdout", ImageNet2012Config(**kwargs))
    test = load_imagenet2012("val", ImageNet2012Config(max_images=max_images, **kwargs))
    _check_num_classes(train, num_classes, root)
    logger.info(
        f"ImageNet: {train.num_samples} train ({train.steps_per_epoch} steps/epoch), "
        f"{select.num_samples} selection ({holdout_per_class} per class, held out of train), "
        f"{test.num_samples} test (official val), {train.num_classes} classes"
    )
    return train, select, test


def eval_head(val: ImageNet2012Data, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """The first ``n`` val images and labels as numpy, in the loader's input scale.

    Val is unshuffled and unaugmented, so the head is the same on every call.
    ``n`` larger than the split returns the whole split.

    Returns:
        ``(images (n, H, W, 3) float32, labels (n,) int32)``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    xs, ys = [], []
    for x, y in val.dataset.unbatch().take(n).batch(256):
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)
