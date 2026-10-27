"""ImageNet-1k wiring shared by the trainers that read the extracted dataset.

``dl_techniques.datasets.vision.imagenet2012.load_imagenet2012`` owns the
pipeline (decode, crop, augment, normalize). This module owns what every
trainer around it repeated: the ``--imagenet-root`` / ``--devkit-dir`` flags,
the root-exists check, the train/val pair with the same loader settings, the
class-count check, and a small in-memory head of val for the end-of-run
figures. ``train_resnet``, ``train_vit`` and ``train_cliffordnet`` use it.
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


def load_imagenet_splits(
        root: str,
        devkit_dir: Optional[str],
        image_size: int,
        batch_size: int,
        seed: int,
        augment: bool,
        augment_color: bool = False,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        num_classes: Optional[int] = None,
) -> Tuple[ImageNet2012Data, ImageNet2012Data]:
    """Build the ``(train, val)`` pair every ImageNet trainer feeds to ``fit``.

    Train drops its ragged last batch (a fixed step count per epoch); val keeps
    every image. Both datasets are finite and do not repeat.

    Args:
        root: ImageNet folder holding ``train/`` and ``val/``.
        devkit_dir: Devkit folder; ``None`` means ``<root>/devkit``.
        image_size: Square output side.
        batch_size: Batch size of both splits.
        seed: Seed of the train shuffle and augmentation.
        augment: Random-resized-crop and flip on train.
        augment_color: Brightness, contrast, saturation and hue on train.
        num_parallel_calls: Parallel decode calls (default: autotuned).
        num_classes: When given, the number of class folders under ``train/``
            must equal it.

    Raises:
        ValueError: ``num_classes`` differs from the number of class folders.
    """
    kwargs = dict(
        root=root, devkit_dir=devkit_dir, image_size=image_size,
        batch_size=batch_size, augment=augment, augment_color=augment_color,
        seed=seed, num_parallel_calls=num_parallel_calls,
    )
    train = load_imagenet2012("train", ImageNet2012Config(drop_remainder=True, **kwargs))
    val = load_imagenet2012("val", ImageNet2012Config(**kwargs))
    if num_classes is not None and train.num_classes != num_classes:
        raise ValueError(
            f"num_classes={num_classes} but {root} has {train.num_classes} classes"
        )
    logger.info(
        f"ImageNet: {train.num_samples} train ({train.steps_per_epoch} steps/epoch), "
        f"{val.num_samples} val, {train.num_classes} classes"
    )
    return train, val


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
