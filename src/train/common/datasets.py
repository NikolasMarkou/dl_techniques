"""Dataset utilities for training scripts that the classification loader does not cover.

MNIST, CIFAR, Imagenette and ImageNet classification data come from
``train.common.classification_data``; the self-supervised trainers' image streams from
``train.common.image_streams``. What is left here: image-file discovery
(:func:`collect_image_paths`) and :func:`fill_batch`.
"""

import numpy as np
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Default image file extensions for the denoiser path-collection helper
# below. This is the shared default for the byte-identical rglob +
# extension-filter preambles previously duplicated across the denoiser
# trainers (bfcnn / bfunet / cliffordnet) and their framework
# ``_create_file_list`` methods. Callers may pass their own ``extensions``
# set (e.g. ``config.image_extensions``); when they don't, this set is used.
# Matching is case-insensitive (both lower- and upper-case suffixes match).
# ---------------------------------------------------------------------

DEFAULT_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------------

def collect_image_paths(
        directories: List[str],
        extensions: Optional[Set[str]] = None,
        max_files: Optional[int] = None,
        shuffle_seed: Optional[int] = None,
        sort: bool = True,
) -> List[str]:
    """Recursively collect image file paths from a list of directories.

    Shared extraction of the byte-identical ``rglob`` + extension-filter +
    cap-then-shuffle path-collection preamble previously duplicated across the
    denoiser trainers (``train/bfunet/train_bfcnn_denoiser.py`` /
    ``train_convunext_denoiser.py`` /
    ``train_unet_denoiser.py``) and the ``DatasetBuilder._create_file_list``
    methods.

    For each directory it recursively scans (``rglob("*")``) for files whose
    suffix matches ``extensions`` (case-insensitive). A directory that does not
    exist is skipped with a warning; an error while scanning one that does
    exist raises (a partial listing must not become the training set).

    This helper does NOT raise when no files are found — it returns the
    (possibly empty) list. Callers keep their own ``if not paths: raise
    ValueError(...)`` guard so the error message stays caller-specific.

    Args:
        directories: Directories to scan recursively for image files.
        extensions: Set of file suffixes to accept (e.g. ``{".jpg", ".png"}``).
            Matching is case-insensitive — both the lower- and upper-case form
            of each suffix is accepted. When ``None`` (the default),
            :data:`DEFAULT_IMAGE_EXTENSIONS` is used.
        max_files: Optional cap on the number of returned paths. When set and
            smaller than the number of collected paths, the list is shuffled
            (see ``shuffle_seed``) and then truncated to ``max_files``. The
            shuffle happens ONLY when a cap is applied — this matches the
            original preamble semantics exactly.
        shuffle_seed: When ``max_files`` triggers a shuffle, use a seeded
            ``np.random.RandomState(shuffle_seed)`` (deterministic) if this is
            not ``None``; otherwise use the global ``np.random.shuffle``
            (non-deterministic, matching the original behaviour).
        sort: When ``True`` (the default) the collected paths are sorted before
            the optional cap-then-shuffle step. This preserves deterministic
            ordering for paired-path denoisers (target/condition alignment).

    Returns:
        A list of matching file paths as strings. Possibly empty.

    Raises:
        OSError: Whatever listing an existing directory raises (permissions, an
            I/O error, a directory removed mid-scan).
    """
    if extensions is None:
        extensions = DEFAULT_IMAGE_EXTENSIONS

    # Build a case-insensitive match set (both lower- and upper-case suffixes),
    # mirroring the original ``{ext.lower()} | {ext.upper()}`` construction.
    extensions_set = {ext.lower() for ext in extensions}
    extensions_set.update({ext.upper() for ext in extensions})

    all_file_paths: List[str] = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning(f"Directory not found, skipping: {directory}")
            continue
        # A scan that fails raises (D-183): a warning here trained on whatever
        # part of the directory was listed before the error.
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions_set:
                all_file_paths.append(str(file_path))

    logger.info(f"Found a total of {len(all_file_paths)} files.")

    if sort:
        all_file_paths = sorted(all_file_paths)

    # Cap-then-shuffle: shuffle ONLY when a cap is actually applied, exactly as
    # the original denoiser preambles did.
    if max_files is not None and max_files < len(all_file_paths):
        logger.info(f"Limiting to {max_files} files as per configuration.")
        if shuffle_seed is not None:
            np.random.RandomState(shuffle_seed).shuffle(all_file_paths)
        else:
            np.random.shuffle(all_file_paths)
        all_file_paths = all_file_paths[:max_files]

    return all_file_paths


# ---------------------------------------------------------------------
# Batch filling for index-addressed PyDatasets
# ---------------------------------------------------------------------

def fill_batch(
        owner: str,
        idx: int,
        batch_size: int,
        indices: Sequence[int],
        load_one: Callable[[int], Optional[Any]],
        describe: Callable[[int], str],
        failure: str,
) -> List[Any]:
    """The samples of batch ``idx``: the next window of files, a skipped file refilled.

    Interface contract (3 callers: ``train.common.megadepth.MegaDepthDataset``,
    ``train.omnipoint.data.MegaDepthPinholeDataset`` and
    ``train.omnipoint.kitti_depth.KittiDepthDataset``). Attempt ``a`` (from
    0, at most ``3 * batch_size`` and at most ``len(indices)``) reads file ``indices[(idx * batch_size + a)
    % len(indices)]`` through ``load_one(file_index)``; a ``None`` result is
    skipped, and filling stops when ``batch_size`` results are kept. When every
    file loads, batch ``i`` holds exactly files ``i*B .. i*B+B-1`` of
    ``indices``, so the batches of an epoch are disjoint and read each file
    once. A skipped file is refilled from the files after the window, which
    the next batch reads again (the D-126 trade-off). No batch reads a file
    twice: with fewer files than ``batch_size`` it holds each once (D-199).

    Args:
        owner: The dataset class name, for the error message.
        idx: The batch index.
        batch_size: The number of samples wanted.
        indices: The epoch's file order (shuffled or not).
        load_one: Loads one file by its index; ``None`` skips it.
        describe: Names a file by its index, for the error message.
        failure: Completes ``"no <failure> in N attempts"``, for example
            ``"KITTI pair loaded"``.

    Returns:
        Between 1 and ``batch_size`` results of ``load_one``, in reading order.

    Raises:
        RuntimeError: When no file of the attempts gives a result. The message
            names the batch and every file tried.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-193
    # Do NOT add the number of kept samples to this offset: the attempt count
    # already counts every read, kept or skipped, so the sum advanced by 2 per
    # sample. Consecutive batches overlapped by half and the files at odd
    # offsets were never read (D-126, then again in both OmniPoint loaders).
    kept: List[Any] = []
    tried: List[int] = []
    for attempt in range(min(batch_size * 3, len(indices))):
        if len(kept) == batch_size:
            break
        file_index = int(indices[(idx * batch_size + attempt) % len(indices)])
        tried.append(file_index)
        result = load_one(file_index)
        if result is not None:
            kept.append(result)
    if not kept:
        raise RuntimeError(
            f"{owner} batch {idx}: no {failure} in {len(tried)} attempts; tried "
            f"{[describe(i) for i in tried]}")
    return kept
