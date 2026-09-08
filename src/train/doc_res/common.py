"""Shared building blocks for the DocRes trainer.

One task per training run (plan assumption A7): upstream draws one task per
*batch* from five concurrent loaders, which needs a custom training step; this
port trains a single task on stock ``fit()`` instead.

Everything task-specific is read off
:data:`dl_techniques.datasets.document_restoration.tasks.TASKS`. Nothing in
this module branches on a task string -- the prompt generator, the supervised
channel count, the loss and the post-processing mode all arrive as fields of a
:class:`TaskSpec`, and the two places that must behave differently per task
(the loss object and the ground-truth adapter) are dispatch dicts keyed by the
table's ``LOSS_*`` constants. ``test_pipeline.py`` proves this by fabricating a
spec: changing the table changes the behaviour.

The data path generalises ``src/train/bfunet/common.py`` from a single
``(noisy, clean)`` image pair to a ``(rgb ++ prompt) -> ground truth`` triplet:

* three files per sample -- the page, its precomputed DTSPrompt sidecar (written
  once by ``prepare_doc_res_data.py``; the pipeline never runs a prompt
  generator per batch) and its ground-truth page;
* decoded together into one ``(H, W, 9)`` uint8 tensor, so the random crop is
  necessarily the SAME crop for all three;
* ONE normalisation site (``/255.0``), exactly as bfunet has;
* a degenerate-page filter, a multi-patch ``flat_map`` (one decode, many crops)
  and ``tf.ensure_shape`` at every stage boundary.

Public surface:
    * :class:`DocResTrainingConfig` -- the run knobs. Every field is consumed by
      something other than the config dump; see ``src/train/CLAUDE.md``.
    * :func:`add_common_arguments` -- the task-agnostic CLI flags.
    * :func:`config_from_args` -- namespace -> config, the one wiring site.
    * :func:`collect_task_triplets` / :func:`split_triplets` -- path worklists.
    * :func:`create_dataset` -- the tf.data pipeline.
    * :func:`build_task_loss` -- the table-driven loss.
    * :func:`train` -- stock ``fit()``.
    * :func:`require_task_data` -- re-exported startup check (see below).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.datasets.document_restoration.tasks import (
    LOSS_CATEGORICAL_CROSSENTROPY,
    LOSS_L1,
    N_OUTPUT_CHANNELS,
    N_PROMPT_CHANNELS,
    TaskSpec,
    get_task,
    task_names,
)
from dl_techniques.models.vision.image_restoration.doc_res.model import (
    SPATIAL_DIVISOR,
    create_doc_res,
)
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger
from train.common import create_callbacks, set_seeds
from train.common.config_io import save_config_json
from train.common.run_io import save_training_history_json

# `require_task_data` is NOT redefined here. It already exists in the staging
# script, where the dataset manifest that knows *why* a corpus is missing (a
# registration gate, a dead host, a Drive folder a script cannot page through)
# lives. A second implementation here would be a second, weaker answer to the
# same question. Re-exported so a trainer imports it from one place.
from train.doc_res.prepare_doc_res_data import (  # noqa: F401
    ARCHIVE_DIRNAME,
    DEFAULT_DATASET_ROOT,
    IMAGE_SUFFIXES,
    PROMPT_DIRNAME,
    MissingTrainingDataError,
    dataset_dir,
    is_ground_truth_path,
    iter_input_images,
    require_task_data,
    sidecar_path,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIXEL_SCALE: float = 255.0
"""THE single normalisation divisor for the whole pipeline. Upstream DocRes
normalises with a flat ``/255.0`` for every task and applies no mean/std
(``inference.py:104,147,186,216,241``). Do NOT add a second normalisation
anywhere downstream: the page, the prompt and the ground truth all pass through
:func:`decode_triplet` and nothing else rescales them."""

N_RGB_CHANNELS: int = 3
"""Channels of the source page after ``convert("RGB")``."""

N_INPUT_CHANNELS: int = N_RGB_CHANNELS + N_PROMPT_CHANNELS
"""6 -- the model's input contract."""

N_STACKED_CHANNELS: int = N_RGB_CHANNELS + N_PROMPT_CHANNELS + N_OUTPUT_CHANNELS
"""9 -- page, prompt and ground truth carried through the crop as one tensor."""

BINARIZATION_GT_THRESHOLD_U8: int = 155
"""Ink/background split of a binarization ground-truth page, on the 0-255
scale. Upstream's loader thresholds the GT with ``bin_map[bin_map > 155] = 255``
(``loaders/docres_loader.py:110-115``) before building its class target."""

BINARY_INK_CLASS_INDEX: int = 1
"""Which of the two supervised channels means *ink*. Upstream never states it
(the finding notes the channel-vs-class assignment was not traceable without a
checkpoint), so this port FIXES the convention here rather than leaving it
implicit: class 0 is background, class 1 is ink. The inference shim reads this
constant instead of restating the choice."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DocResTrainingConfig:
    """Knobs for one DocRes training run.

    Defaults follow the upstream recipe (``train.py:201-208``): 256 px crops,
    batch 10, ``AdamW(lr=2e-4, weight_decay=5e-4)`` and a cosine decay to
    ``1e-6``. Every field is read by something other than the config dump; a
    field that is only serialized is a knob that silently does nothing and is
    deleted rather than wired (``src/train/CLAUDE.md``,
    ``tests/test_train/test_config_fields_are_live.py``).

    :param task: A key of ``TASKS``. The single task this run trains.
    :param dataset_root: The staging volume holding ``doc_res/<task>/<set>/``.
    :param model_variant: A key of ``DocRes.MODEL_VARIANTS``.
    :param patch_size: Square crop size. Must be a positive multiple of 8 --
        the model refuses a non-divisible input (D-016) and this is where that
        is caught, rather than mid-epoch inside a pixel-unshuffle.
    :param batch_size: Patches per step.
    :param epochs: Training epochs.
    :param steps_per_epoch: Steps per epoch; the dataset repeats forever, so
        this is what defines an epoch.
    :param validation_steps: Validation batches per epoch.
    :param learning_rate: Peak (and initial) learning rate.
    :param final_learning_rate: Cosine floor, upstream's ``eta_min``.
    :param weight_decay: Decoupled AdamW weight decay. Applied by the optimizer
        ONLY -- no ``kernel_regularizer`` is ever attached, which would decay
        the same parameter twice.
    :param patches_per_image: Crops taken from each decoded page.
    :param patch_shuffle_buffer: Shuffle buffer over patch tensors, so a batch
        is not all crops of one page.
    :param dataset_shuffle_buffer: Shuffle buffer over path triplets.
    :param val_split: Fraction of pages held out for validation.
    :param max_train_files: Cap on pages used, or ``None`` for all of them.
    :param seed: Seed for the split and for ``set_seeds``.
    :param patience: Early-stopping patience.
    :param output_dir: Root under which the timestamped run directory is made.
    """

    task: str = "binarization"
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    model_variant: str = "docres"

    patch_size: int = 256
    batch_size: int = 10
    epochs: int = 50
    steps_per_epoch: int = 200
    validation_steps: int = 20

    learning_rate: float = 2e-4
    final_learning_rate: float = 1e-6
    weight_decay: float = 5e-4

    patches_per_image: int = 8
    patch_shuffle_buffer: int = 64
    dataset_shuffle_buffer: int = 512
    val_split: float = 0.1
    max_train_files: Optional[int] = None

    seed: int = 42
    patience: int = 15
    output_dir: str = "results"

    def __post_init__(self) -> None:
        """Validate the knobs at construction time.

        :raises ValueError: On an unknown task, a non-positive count, a
            ``patch_size`` that is not a multiple of :data:`SPATIAL_DIVISOR`,
            or a ``val_split`` outside ``[0, 1)``. NOT on an unknown
            ``model_variant``: that is checked by ``DocRes.from_variant``, the
            one authority on the legal keys, when :func:`build_model` runs
            (D-031).
        """
        get_task(self.task)  # raises, listing the legal names

        if self.patch_size <= 0 or self.patch_size % SPATIAL_DIVISOR != 0:
            raise ValueError(
                f"patch_size must be a positive multiple of {SPATIAL_DIVISOR}, "
                f"got {self.patch_size}. DocRes downsamples three times and "
                "refuses a non-divisible input; catching it here means the run "
                "fails at startup instead of part-way through the first epoch."
            )
        positive = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.validation_steps,
            "patches_per_image": self.patches_per_image,
            "patch_shuffle_buffer": self.patch_shuffle_buffer,
            "dataset_shuffle_buffer": self.dataset_shuffle_buffer,
            "patience": self.patience,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not 0.0 < self.final_learning_rate <= self.learning_rate:
            raise ValueError(
                "final_learning_rate must be positive and no greater than "
                f"learning_rate, got {self.final_learning_rate} vs "
                f"{self.learning_rate}"
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.max_train_files is not None and self.max_train_files <= 0:
            raise ValueError(
                f"max_train_files must be positive or None, got "
                f"{self.max_train_files}"
            )

    @property
    def spec(self) -> TaskSpec:
        """The resolved task specification for :attr:`task`."""
        return get_task(self.task)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the task-agnostic DocRes training flags.

    Shared by the trainer entry point and by any future per-task script, so a
    flag is declared once. ``--gpu`` is deliberately NOT here: it is consumed by
    ``setup_gpu`` in ``main()`` and is not a config field.

    :param parser: The parser to extend.
    :return: The same parser, for chaining.
    """
    defaults = DocResTrainingConfig()

    parser.add_argument(
        "--task", type=str, default=defaults.task, choices=list(task_names()),
        help="Which DocRes task to train (one task per run).",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=defaults.dataset_root,
        help="Staging volume holding doc_res/<task>/<dataset>/.",
    )
    parser.add_argument(
        "--model-variant", type=str, default=defaults.model_variant,
        help="DocRes variant key.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=defaults.patch_size,
        help="Square crop size; must be a multiple of 8.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=defaults.batch_size,
        help="Patches per optimizer step.",
    )
    parser.add_argument(
        "--epochs", type=int, default=defaults.epochs, help="Training epochs.",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=defaults.steps_per_epoch,
        help="Steps per epoch (the dataset repeats forever).",
    )
    parser.add_argument(
        "--validation-steps", type=int, default=defaults.validation_steps,
        help="Validation batches per epoch.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--final-learning-rate", type=float, default=defaults.final_learning_rate,
        help="Cosine floor learning rate.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay,
        help="Decoupled AdamW weight decay (never also an L2 regularizer).",
    )
    parser.add_argument(
        "--patches-per-image", type=int, default=defaults.patches_per_image,
        help="Crops taken from each decoded page.",
    )
    parser.add_argument(
        "--patch-shuffle-buffer", type=int, default=defaults.patch_shuffle_buffer,
        help="Shuffle buffer over patch tensors.",
    )
    parser.add_argument(
        "--dataset-shuffle-buffer", type=int,
        default=defaults.dataset_shuffle_buffer,
        help="Shuffle buffer over path triplets.",
    )
    parser.add_argument(
        "--val-split", type=float, default=defaults.val_split,
        help="Fraction of pages held out for validation.",
    )
    parser.add_argument(
        "--max-train-files", type=int, default=defaults.max_train_files,
        help="Cap on pages used; omit for all of them.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.seed, help="Random seed.",
    )
    parser.add_argument(
        "--patience", type=int, default=defaults.patience,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=defaults.output_dir,
        help="Root for the timestamped run directory.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> DocResTrainingConfig:
    """Build a config from a parsed namespace.

    The ONE wiring site between :func:`add_common_arguments` and
    :class:`DocResTrainingConfig`; a flag that does not arrive here silently
    does nothing, which ``tests/test_train/test_doc_res/`` pins.

    :param args: A namespace produced by a parser carrying the common flags.
    :return: The config.
    """
    return DocResTrainingConfig(
        task=args.task,
        dataset_root=args.dataset_root,
        model_variant=args.model_variant,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        weight_decay=args.weight_decay,
        patches_per_image=args.patches_per_image,
        patch_shuffle_buffer=args.patch_shuffle_buffer,
        dataset_shuffle_buffer=args.dataset_shuffle_buffer,
        val_split=args.val_split,
        max_train_files=args.max_train_files,
        seed=args.seed,
        patience=args.patience,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Pairing an input page with its ground truth
# ---------------------------------------------------------------------------

_GT_MARKER_RE = re.compile(r"[_\-]?(?:skel|est)?gt$", re.IGNORECASE)
"""Trailing ground-truth marker on a stem: ``_GT``, ``_gt``, ``_estGT``,
``_skelGT``, ``-GT``. All five spellings are present in the staged DIBCO
tree."""

_NOISE_TOKEN_RE = re.compile(r"noise[a-z]?", re.IGNORECASE)
"""NoisyOffice does not mark its ground truth with a GT token at all: the pair
is ``Fontfre_Noisec_TE.png`` -> ``Fontfre_Clean_TE.png``. Folding every
``Noise<x>`` token to ``clean`` makes the two stems agree."""


def pairing_key(stem: str) -> str:
    """Normalise a filename stem to the key an input and its GT share.

    :param stem: A filename without its suffix.
    :return: The lowercase pairing key.
    """
    return _NOISE_TOKEN_RE.sub("clean", _GT_MARKER_RE.sub("", stem)).lower()


def iter_ground_truth_images(dir_path: Path) -> List[Path]:
    """Every ground-truth image staged under one dataset directory.

    The complement of ``iter_input_images``: same exclusions (bookkeeping
    directories, hidden paths, non-image suffixes), opposite side of
    ``is_ground_truth_path``.

    :param dir_path: The dataset directory.
    :return: Sorted list of ground-truth image paths.
    """
    if not dir_path.is_dir():
        return []
    out: List[Path] = []
    for path in sorted(dir_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(dir_path).as_posix()
        parts = rel.split("/")
        if parts[0] in (ARCHIVE_DIRNAME, PROMPT_DIRNAME):
            continue
        if any(part.startswith(".") for part in parts):
            continue
        if is_ground_truth_path(rel):
            out.append(path)
    return out


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from an image header, or ``None`` if unreadable.

    Only the header is read -- PIL does not decode the pixels for ``.size`` --
    so this is cheap enough to run over every candidate at startup.

    :param path: The image.
    :return: The size, or ``None``.
    """
    from PIL import Image

    try:
        with Image.open(path) as handle:
            return handle.size
    except Exception as exc:  # noqa: BLE001
        logger.warning("cannot read image header for %s: %s", path, exc)
        return None


@dataclass(frozen=True)
class SampleTriplet:
    """One training sample as three file paths.

    :param image: The input page.
    :param prompt: Its precomputed DTSPrompt sidecar.
    :param ground_truth: The supervision target page.
    """

    image: Path
    prompt: Path
    ground_truth: Path


# DECISION plan-2026-09-08T111844-de235227/D-029: pairing is by
# :func:`pairing_key` AND by pixel size. Do not simplify this to a
# stem-only match: NoisyOffice ships `clean_images_grayscale_
# doubleresolution/Fontfre_Clean_TE.png` (1080x516) beside
# `clean_images_grayscale/Fontfre_Clean_TE.png` (540x258) under the SAME
# stem, so a key-only match has three candidates for every one of its 216
# pages and can pick a target at twice the input's resolution. Nothing
# downstream would fail -- the 9-channel decode would raise only because
# the sizes disagree, which is exactly the check being described here, and
# a resize-to-fit "fix" would train on a silently misaligned pair. Ties
# among same-size candidates break by sorted path order, which selects
# `clean_images_binaryscale` for binarization. See D-029 in decisions.md.
def collect_dataset_triplets(dir_path: Path, spec: TaskSpec) -> List[SampleTriplet]:
    """Pair every input page in one staged dataset with its prompt and GT.

    Pairing is by :func:`pairing_key` AND by pixel size; see the D-029 anchor
    above this function for why the size half is not optional.

    An input with no sidecar or no size-compatible ground truth is DROPPED and
    counted, not silently ignored -- the count is logged.

    :param dir_path: The dataset directory.
    :param spec: The task spec; supplies the sidecar suffix.
    :return: The usable triplets, in sorted input order.
    """
    suffix = ".png" if spec.prompt_dtype == "uint8" else ".npy"

    by_key: Dict[str, List[Path]] = {}
    for gt in iter_ground_truth_images(dir_path):
        by_key.setdefault(pairing_key(gt.stem), []).append(gt)

    triplets: List[SampleTriplet] = []
    no_prompt = no_gt = no_size = 0
    for image in iter_input_images(dir_path):
        prompt = sidecar_path(dir_path, image, suffix)
        if not prompt.is_file():
            no_prompt += 1
            continue
        candidates = by_key.get(pairing_key(image.stem), [])
        if not candidates:
            no_gt += 1
            continue
        size = _image_size(image)
        matched = [c for c in sorted(candidates) if _image_size(c) == size]
        if size is None or not matched:
            no_size += 1
            continue
        triplets.append(
            SampleTriplet(image=image, prompt=prompt, ground_truth=matched[0])
        )

    if no_prompt or no_gt or no_size:
        logger.info(
            "%s: %d usable triplets (dropped %d without a prompt sidecar, %d "
            "without a ground-truth match, %d without a size-compatible "
            "ground truth)",
            dir_path.name, len(triplets), no_prompt, no_gt, no_size,
        )
    return triplets


def collect_task_triplets(config: DocResTrainingConfig) -> List[SampleTriplet]:
    """The whole worklist for one task, across every staged dataset.

    Calls :func:`require_task_data` first, so a task with no corpus fails at
    startup naming the real reason (a registration gate, a dead host) rather
    than producing an empty dataset.

    :param config: The run config.
    :return: The triplets, capped at ``max_train_files`` after a seeded shuffle.
    :raises MissingTrainingDataError: If the task has no staged training corpus.
    :raises ValueError: If a corpus is staged but no page could be paired.
    """
    root = Path(config.dataset_root)
    spec = config.spec
    datasets = require_task_data(root, config.task)

    triplets: List[SampleTriplet] = []
    for ds in datasets:
        found = collect_dataset_triplets(dataset_dir(root, ds), spec)
        logger.info("%s: %d training triplets", ds.name, len(found))
        triplets.extend(found)

    if not triplets:
        raise ValueError(
            f"task {config.task!r} has staged pages under "
            f"{root} but none could be paired with both a prompt sidecar and a "
            "ground-truth page. Re-run `python -m "
            f"train.doc_res.prepare_doc_res_data --tasks {config.task}` to "
            "(re)build the sidecars, and check that the corpus really ships "
            "ground truth."
        )

    rng = np.random.default_rng(config.seed)
    order = rng.permutation(len(triplets))
    triplets = [triplets[i] for i in order]
    if config.max_train_files is not None:
        triplets = triplets[: config.max_train_files]
    return triplets


def split_triplets(
        triplets: Sequence[SampleTriplet], config: DocResTrainingConfig
) -> Tuple[List[SampleTriplet], List[SampleTriplet]]:
    """Split a worklist into train and validation halves.

    The worklist is already seed-shuffled by :func:`collect_task_triplets`, so
    this is a deterministic prefix/suffix cut. At least one page stays on each
    side whenever there are two or more.

    :param triplets: The worklist.
    :param config: The run config; reads ``val_split``.
    :return: ``(train, validation)``.
    """
    total = len(triplets)
    n_val = int(round(total * config.val_split))
    if config.val_split > 0.0 and total >= 2:
        n_val = max(1, min(n_val, total - 1))
    else:
        n_val = 0
    return list(triplets[n_val:]), list(triplets[:n_val])


# ---------------------------------------------------------------------------
# The tf.data pipeline
# ---------------------------------------------------------------------------


# DECISION plan-2026-09-08T111844-de235227/D-028: decoding goes through PIL
# inside a `tf.numpy_function`, NOT through `tf.io.decode_image`. Do not
# "optimise" this into a native TF decode: `tf.io.decode_image` handles
# BMP/GIF/JPEG/PNG only, and the staged binarization corpus is full of TIFF
# -- `H01.tif` / `H06.tif` inputs in H-DIBCO 2010 and `*_GT.tiff` /
# `*_estGT.tiff` targets in six of the ten staged DIBCO years. A native
# decode raises per file at graph-execution time, which surfaces mid-epoch
# rather than at startup. The per-sample Python cost is paid on 352 pages
# whose reads dominate anyway, and the prompt is a PRECOMPUTED sidecar, so
# no DTSPrompt generator runs per batch either way. See D-028 in
# decisions.md.
def _decode_triplet_numpy(
        image_path: bytes, prompt_path: bytes, gt_path: bytes
) -> np.ndarray:
    """Read the three files of one sample into a single ``(H, W, 9)`` uint8.

    Decoding goes through PIL, not ``tf.io.decode_image``; see the D-028
    anchor above this function.

    :param image_path: The page, as the bytes tf.data hands a Python function.
    :param prompt_path: Its prompt sidecar.
    :param gt_path: Its ground-truth page.
    :return: The stacked ``(H, W, 9)`` uint8 array.
    :raises ValueError: If the three images do not share one size. That is a
        pairing defect, and a silent resize here would hide it.
    """
    from PIL import Image

    parts = []
    sizes = []
    for raw in (image_path, prompt_path, gt_path):
        # tf.data hands a Python function `np.bytes_` in graph mode and a 0-d
        # object array when the op is called eagerly; both must resolve to the
        # same path string.
        value = raw.item() if isinstance(raw, np.ndarray) else raw
        path = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        with Image.open(path) as handle:
            parts.append(np.asarray(handle.convert("RGB"), dtype=np.uint8))
            sizes.append(handle.size)
    if len(set(sizes)) != 1:
        raise ValueError(
            "page, prompt and ground truth must share one size, got "
            f"{sizes} for {[p for p in (image_path, prompt_path, gt_path)]}"
        )
    return np.concatenate(parts, axis=-1)


def decode_triplet(
        image_path: tf.Tensor,
        prompt_path: tf.Tensor,
        gt_path: tf.Tensor,
        config: DocResTrainingConfig,
) -> tf.Tensor:
    """Decode one sample and normalise it, ONCE.

    Returns the full-resolution ``(H, W, 9)`` float tensor -- page, prompt and
    ground truth stacked on the channel axis so that the later random crop is
    necessarily the same crop for all three. This is the expensive step (disk
    read plus decode); the streaming pipeline calls it once per page and takes
    ``patches_per_image`` crops from the result.

    :param image_path: Scalar string tensor: the page.
    :param prompt_path: Scalar string tensor: the prompt sidecar.
    :param gt_path: Scalar string tensor: the ground-truth page.
    :param config: The run config; reads ``patch_size``.
    :return: The ``(H, W, 9)`` float32 tensor in ``[0, 1]``, upscaled if either
        extent is below ``patch_size``.
    """
    stacked = tf.numpy_function(
        _decode_triplet_numpy,
        [image_path, prompt_path, gt_path],
        tf.uint8,
        name="decode_doc_res_triplet",
    )
    stacked.set_shape([None, None, N_STACKED_CHANNELS])

    # THE single normalisation site for the whole training pipeline (INV: one
    # `/255.0`, as in train/bfunet/common.py). Do NOT rescale again downstream.
    image = tf.cast(stacked, tf.float32) / PIXEL_SCALE

    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    min_size = config.patch_size

    def _upscale() -> tf.Tensor:
        scale = tf.cast(min_size, tf.float32) / tf.cast(
            tf.minimum(height, width), tf.float32
        )
        new_h = tf.cast(tf.math.ceil(tf.cast(height, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(width, tf.float32) * scale), tf.int32)
        return tf.image.resize(image, [new_h, new_w])

    return tf.cond(
        tf.logical_or(height < min_size, width < min_size),
        true_fn=_upscale,
        false_fn=lambda: image,
    )


def random_crop_patch(
        image: tf.Tensor, config: DocResTrainingConfig
) -> tf.Tensor:
    """Take one random ``patch_size`` crop of the stacked triplet.

    :param image: A ``(H, W, 9)`` tensor.
    :param config: The run config; reads ``patch_size``.
    :return: The ``(patch, patch, 9)`` crop.
    """
    return tf.image.random_crop(
        image, [config.patch_size, config.patch_size, N_STACKED_CHANNELS]
    )


def is_informative_page(image: tf.Tensor) -> tf.Tensor:
    """Whether a decoded page carries any supervision at all.

    An all-constant scan (a blank or fully saturated decode) has a constant
    prompt too and teaches nothing. Measured on the RGB channels only, so a
    legitimately blank ground-truth region does not disqualify a page.

    Named rather than inlined into ``create_dataset`` because the dataset
    repeats forever: a test that asserted "the degenerate page is dropped" by
    iterating the pipeline would spin forever instead of failing. The predicate
    is therefore tested directly.

    :param image: A decoded ``(H, W, 9)`` tensor.
    :return: Scalar boolean tensor.
    """
    rgb = image[..., :N_RGB_CHANNELS]
    return tf.math.reduce_max(rgb) > tf.math.reduce_min(rgb)


def _ground_truth_l1(gt: tf.Tensor, spec: TaskSpec) -> tf.Tensor:
    """Supervision target for an L1 task: the leading GT channels."""
    return gt[..., : spec.n_supervised_channels]


def _ground_truth_two_class(gt: tf.Tensor, spec: TaskSpec) -> tf.Tensor:
    """Supervision target for a 2-class cross-entropy task: a one-hot map.

    Ink is ``gt <= 155/255`` (upstream's threshold, on the normalised scale)
    and lands in channel :data:`BINARY_INK_CLASS_INDEX`.

    :param gt: The ``(..., 3)`` ground-truth slice in ``[0, 1]``.
    :param spec: The task spec; must supervise exactly two channels.
    :return: The ``(..., 2)`` one-hot float32 target.
    :raises ValueError: If the spec does not supervise exactly two channels.
    """
    if spec.n_supervised_channels != 2:
        raise ValueError(
            f"task {spec.name!r} declares a two-class cross-entropy loss but "
            f"supervises {spec.n_supervised_channels} channels; a one-hot "
            "target needs exactly 2."
        )
    threshold = BINARIZATION_GT_THRESHOLD_U8 / PIXEL_SCALE
    ink = tf.cast(gt[..., :1] <= threshold, tf.float32)
    background = 1.0 - ink
    ordered = [background, ink]
    if BINARY_INK_CLASS_INDEX == 0:
        ordered = [ink, background]
    return tf.concat(ordered, axis=-1)


_GROUND_TRUTH_ADAPTERS = {
    LOSS_L1: _ground_truth_l1,
    LOSS_CATEGORICAL_CROSSENTROPY: _ground_truth_two_class,
}
"""Loss name -> ground-truth adapter. Keyed by the ``TASKS`` table's own
constants (D-022), so a task string is never compared anywhere in this module
and a table typo is a ``KeyError`` at build time rather than a silent
fallthrough to the wrong target."""


def split_stacked_sample(
        patch: tf.Tensor, spec: TaskSpec
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Turn one stacked ``(..., 9)`` patch into ``(model input, target)``.

    :param patch: The stacked crop.
    :param spec: The task spec.
    :return: ``(x, y)`` -- a 6-channel ``rgb ++ prompt`` input and the task's
        supervision target.
    :raises KeyError: If the spec names a loss with no ground-truth adapter.
    """
    rgb = patch[..., :N_RGB_CHANNELS]
    prompt = patch[..., N_RGB_CHANNELS:N_INPUT_CHANNELS]
    gt = patch[..., N_INPUT_CHANNELS:]
    x = tf.concat([rgb, prompt], axis=-1)
    y = _GROUND_TRUTH_ADAPTERS[spec.loss](gt, spec)
    return x, y


def create_dataset(
        triplets: Sequence[SampleTriplet],
        config: DocResTrainingConfig,
        spec: TaskSpec,
        is_training: bool,
) -> tf.data.Dataset:
    """Build the ``((rgb ++ prompt), ground truth)`` patch pipeline.

    :param triplets: The path worklist.
    :param config: The run config.
    :param spec: The task spec; supplies the loss and hence the GT adapter.
    :param is_training: Whether to shuffle, take multiple crops per page and
        drop a short final batch.
    :return: The batched, prefetched dataset.
    :raises ValueError: If ``triplets`` is empty.
    """
    if not triplets:
        raise ValueError("no (page, prompt, ground-truth) triplets for the dataset")

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            [str(t.image) for t in triplets],
            [str(t.prompt) for t in triplets],
            [str(t.ground_truth) for t in triplets],
        )
    )
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(config.dataset_shuffle_buffer, len(triplets)),
            reshuffle_each_iteration=True,
        )
    dataset = dataset.repeat()

    # Decode each page ONCE. `deterministic=False` lets fast reads flow past
    # slower ones (the corpus lives on a spinning volume).
    dataset = dataset.map(
        lambda i, p, g: decode_triplet(i, p, g, config),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    # Drop a degenerate page (see `is_informative_page`).
    dataset = dataset.filter(is_informative_page)

    if is_training and config.patches_per_image > 1:
        ppi = config.patches_per_image
        dataset = dataset.flat_map(
            lambda img: tf.data.Dataset.from_tensors(img)
            .repeat(ppi)
            .map(lambda im: random_crop_patch(im, config))
        )
        # The `ppi` crops above are consecutive crops of ONE page; shuffle the
        # patch tensors so a batch is not all crops of the same page.
        dataset = dataset.shuffle(
            buffer_size=config.patch_shuffle_buffer,
            reshuffle_each_iteration=True,
        )
    else:
        dataset = dataset.map(
            lambda img: random_crop_patch(img, config),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(
        lambda x: tf.ensure_shape(
            x, [config.patch_size, config.patch_size, N_STACKED_CHANNELS]
        )
    )
    dataset = dataset.map(
        lambda x: split_stacked_sample(x, spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(
                x, [config.patch_size, config.patch_size, N_INPUT_CHANNELS]
            ),
            tf.ensure_shape(
                y,
                [
                    config.patch_size,
                    config.patch_size,
                    spec.n_supervised_channels,
                ],
            ),
        )
    )

    dataset = dataset.batch(config.batch_size, drop_remainder=is_training)
    return dataset.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="train.doc_res")
class SupervisedSliceLoss(keras.losses.Loss):
    """Apply a base loss to the leading ``n_channels`` of the prediction.

    Every DocRes task uses the same 3-channel head, and two of the five
    supervise only the first two channels (``train.py:144-159``): dewarping's
    L1 sees the flow pair, binarization's cross-entropy sees a logit pair. The
    third channel is architecturally present and never trained. Slicing lives
    here, in one place, rather than in a per-task branch at the compile site.

    :param base_loss: The loss applied to the sliced prediction.
    :param n_channels: How many leading prediction channels to keep. Must match
        the target's channel count.
    :param kwargs: Forwarded to ``keras.losses.Loss``.
    """

    def __init__(
            self,
            base_loss: keras.losses.Loss,
            n_channels: int,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        self.base_loss = base_loss
        self.n_channels = int(n_channels)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Slice the prediction, then delegate."""
        return self.base_loss(y_true, y_pred[..., : self.n_channels])

    def get_config(self) -> Dict[str, Any]:
        """Serialize the wrapped loss and the slice width."""
        config = super().get_config()
        config.update(
            {
                "base_loss": keras.saving.serialize_keras_object(self.base_loss),
                "n_channels": self.n_channels,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SupervisedSliceLoss":
        """Rebuild from :meth:`get_config`."""
        config = dict(config)
        config["base_loss"] = keras.saving.deserialize_keras_object(
            config["base_loss"]
        )
        return cls(**config)


_BASE_LOSS_BUILDERS = {
    LOSS_L1: lambda: keras.losses.MeanAbsoluteError(),
    LOSS_CATEGORICAL_CROSSENTROPY: lambda: keras.losses.CategoricalCrossentropy(
        from_logits=True
    ),
}
"""Loss name -> Keras object. The name->object dispatch D-022 prescribes: the
``TASKS`` table stays framework-free and a typo there is a ``KeyError`` here."""


def build_task_loss(spec: TaskSpec) -> keras.losses.Loss:
    """The loss for one task, read entirely off the table.

    :param spec: The task spec.
    :return: The loss, already restricted to the task's supervised channels.
    :raises KeyError: If the spec names a loss this module cannot build.
    """
    return SupervisedSliceLoss(
        base_loss=_BASE_LOSS_BUILDERS[spec.loss](),
        n_channels=spec.n_supervised_channels,
        name=f"{spec.loss}_first_{spec.n_supervised_channels}",
    )


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------


def build_optimizer(
        config: DocResTrainingConfig
) -> keras.optimizers.Optimizer:
    """AdamW on a warmup-free cosine decay, matching the upstream recipe.

    Weight decay is applied by the optimizer and by NOTHING else: no
    ``kernel_regularizer`` is ever attached to the model, because AdamW's
    decoupled decay plus an L2 penalty decays the same parameter twice
    (``src/train/CLAUDE.md``).

    :param config: The run config.
    :return: The optimizer.
    """
    total_steps = max(1, config.epochs * config.steps_per_epoch)
    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": config.learning_rate,
            "decay_steps": total_steps,
            # Upstream anneals to eta_min=1e-6; `alpha` is that floor as a
            # fraction of the peak.
            "alpha": config.final_learning_rate / config.learning_rate,
            "warmup_steps": 0,
        }
    )
    return optimizer_builder(
        {"type": "adamw", "weight_decay": config.weight_decay}, schedule
    )


def build_model(config: DocResTrainingConfig) -> keras.Model:
    """Create and compile the DocRes model for one task.

    :param config: The run config.
    :return: The compiled model.
    """
    model = create_doc_res(variant=config.model_variant)
    model.compile(
        optimizer=build_optimizer(config),
        loss=build_task_loss(config.spec),
    )
    return model


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(config: DocResTrainingConfig) -> Tuple[keras.Model, Any, str]:
    """Train DocRes on one task with stock ``fit()``.

    No custom ``train_step``: upstream's per-batch task sampling is the only
    thing that would need one, and this port trains a single task per run
    (plan assumption A7).

    :param config: The run config.
    :return: ``(model, history, results_dir)``.
    """
    set_seeds(config.seed)
    spec = config.spec

    triplets = collect_task_triplets(config)
    train_triplets, val_triplets = split_triplets(triplets, config)
    logger.info(
        "DocRes %s: %d pages (%d train / %d val)",
        config.task, len(triplets), len(train_triplets), len(val_triplets),
    )

    train_ds = create_dataset(train_triplets, config, spec, is_training=True)
    val_ds = (
        create_dataset(val_triplets, config, spec, is_training=False)
        if val_triplets
        else None
    )

    model = build_model(config)

    callbacks, results_dir = create_callbacks(
        model_name=config.model_variant,
        results_dir_prefix=f"doc_res_{config.task}",
        output_root=config.output_dir,
        monitor="val_loss" if val_ds is not None else "loss",
        # `resolve_monitor_mode` maps the `loss` token to 'min' from its
        # minimize set; passed explicitly anyway so the direction is stated at
        # the call site rather than inferred.
        monitor_mode="min",
        patience=config.patience,
        use_lr_schedule=True,
    )
    save_config_json(config, results_dir, "config.json")

    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=config.validation_steps if val_ds is not None else None,
        callbacks=callbacks,
        verbose=1,
    )
    save_training_history_json(history, results_dir)
    return model, history, results_dir
