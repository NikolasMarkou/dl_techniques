"""The depth-estimation arguments of ``python -m train``: the data group, the transform, metric and optimizer groups.

The ``depth_estimation`` task trains a relative-disparity model on MegaDepth (``train.common.depth_data``): the legacy
scene-disjoint split, its validation scenes dealt alternately into SELECTION scenes (they pick the best epoch and set
the bar) and TEST scenes (scored once after ``fit``), fixed crops of both loaded once, and a stateless train stream.
``DepthData`` names the source and those counts; ``DepthTransformArgs``, ``DepthMetricArgs`` and ``DepthOptimArgs`` are
the groups of the task config that differ from the shared ones. Every option that cannot apply is REFUSED, never ignored.

Interface contract
------------------
``TASK`` / ``MEGADEPTH`` / ``BUILTIN`` / ``AUGMENTS`` / ``DINOV2_PATCH`` / ``WATCH_METRIC`` / ``WATCH_MODE``
    The task's name, the one dataset (the recipe key), the one format, the two augment values (``flip``, ``none``), the
    patch every ``image_size`` is a multiple of (a test holds it equal to the model's), and the one metric and
    direction the bar describes (``val_loss``, ``min``; the recipe imports them).

``PRETRAINED`` / ``PRETRAINED_DEFAULT`` / ``PRETRAINED_LR`` / ``FROM_SCRATCH_LR`` / ``pretrained_start(model_args)``
    The ``model_args`` key that chooses the official DINOv2 start, its default (true), the encoder rate from that start
    and the rate from random weights (the legacy trainer's numbers; a test reads them there).
    ``pretrained_start`` is the ONE rule for "is this a pretrained start": ``model_args[PRETRAINED]`` (absent is
    :data:`PRETRAINED_DEFAULT`) is not ``False``. Pure, never raises; the learning rate, the head's 10x table and the
    weights check all follow it.

``DepthData``
    Frozen :class:`~train.api.config.StrictConfig`. Every count field maps to the keyword of the same name of
    ``train.common.depth_data.load_depth_batches``; ``max_samples`` (the smoke cap) caps the train file list together
    with ``max_train_files`` and never the fixed sets. ``problems(prefix)`` names every option out of range.

``depth_facts(data)``
    :return: ``DataFacts(MEGADEPTH, None, None)``.
    :raises ConfigError: for a ``root`` that is not a directory (checked with ``os`` only), and for a tree whose scene
        deal fails: no scene, or fewer than 3 validation scenes (``train.common.depth_data.deal_megadepth_scenes``, the
        loader's own rule, imported inside the call). That import loads TensorFlow but creates no tensor and touches no
        device (a fresh-interpreter test holds it), so this runs before ``setup_gpu`` and before the claim (D-023).

``load_depth_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one (a relative ``root`` is
        anchored to the file).
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping, or any invalid key.

``DepthTransformArgs``
    ``image_size`` (the square crop side; auto is the recipe's 252; a multiple of :data:`DINOV2_PATCH`) and
    ``augment`` (``flip``, a horizontal flip of train samples only, or ``none``).

``DepthMetricArgs``
    ``watch_metric`` and ``watch_mode``: ``val_loss`` and ``min`` are the only values accepted (the bar is a statement
    about the minimum ``val_loss`` of the selection window, so the saved checkpoint must be that epoch).

``DepthOptimArgs``
    The shared optimizer group; an ``"auto"`` ``lr`` from random weights (:func:`pretrained_start` false, read from the
    request's ``model_args`` over the recipe's) is :data:`FROM_SCRATCH_LR`, not the recipe's pretrained rate (D-022).

Import rule
-----------
Module scope imports the standard library and ``train.api`` only, so ``python -m train depth_estimation help`` loads
no TensorFlow and no numpy.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

from .args import AutoContext, OptimArgs, _is_auto, _recipe_values, _watch_values, _with_values
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, load_data_group, read_data_file
from .errors import ConfigError

# ---------------------------------------------------------------------

TASK = "depth_estimation"
MEGADEPTH = "megadepth"
BUILTIN = "builtin"
AUGMENTS: Tuple[str, ...] = ("flip", "none")

#: Where MegaDepth is read in place unless ``data.root`` says otherwise (the legacy trainer's default).
MEGADEPTH_ROOT = "/media/arxwn/data0_4tb/datasets/megadepth"

#: The DINOv2 patch side: the crop side is a multiple of it (the model refuses any other; a test holds it equal).
DINOV2_PATCH = 14

#: The one metric and direction the bar describes; the recipe imports them.
WATCH_METRIC = "val_loss"
WATCH_MODE = "min"

#: The ``model_args`` key of the official DINOv2 start and its default.
PRETRAINED = "pretrained"
PRETRAINED_DEFAULT = True

#: The encoder rate from the official weights (the DPT head trains at 10x it) and the rate from random weights: the
#: legacy trainer's ``PRETRAINED_LEARNING_RATE`` and ``FROM_SCRATCH_LEARNING_RATE`` (a test reads them there).
PRETRAINED_LR = 5e-6
FROM_SCRATCH_LR = 2e-4

_Augment = Literal[AUGMENTS]  # type: ignore[valid-type]


def pretrained_start(model_args: Mapping[str, Any]) -> bool:
    """Whether ``model_args`` asks for the official DINOv2 start; see the module contract."""
    return model_args.get(PRETRAINED, PRETRAINED_DEFAULT) is not False


@dataclasses.dataclass(frozen=True)
class DepthData(StrictConfig):
    """MegaDepth and the counts of a depth run; see the module contract."""

    format: Literal["builtin"] = dataclasses.field(
        default=BUILTIN, metadata={"help": "builtin, the only format: MegaDepth read in place under data.root"})
    dataset: str = dataclasses.field(default=MEGADEPTH, metadata={"help": "megadepth, the only dataset"})
    root: str = dataclasses.field(
        default=MEGADEPTH_ROOT,
        metadata={"help": "the MegaDepth root (<root>/<scene>/dense<N>/{imgs,depths}), read in place and never copied"})
    train_split: float = dataclasses.field(
        default=0.9,
        metadata={"help": "share of the scenes that train, in (0, 1); the rest validate and are dealt alternately into "
                          "selection and test scenes (0.9 is the legacy split, 176 train and 19 validation scenes), at "
                          "least 3 validation scenes are needed"})
    max_train_files: Optional[int] = dataclasses.field(
        default=10_000,
        metadata={"help": "cap on the train files the stream draws from, taken evenly across the train scenes; null "
                          "keeps every file"})
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={"help": "smoke cap on the train file list (with max_train_files, the smaller wins); the selection and "
                          "test crops are never capped; null is no cap"})
    n_select: int = dataclasses.field(
        default=128,
        metadata={"help": "fixed selection crops (they pick the best epoch and set the bar), rounded down to whole batches"})
    n_test: int = dataclasses.field(
        default=256, metadata={"help": "fixed test crops, scored once after fit, rounded down to whole batches"})
    layout_samples: int = dataclasses.field(
        default=512, metadata={"help": "train targets averaged into the layout baseline of the bar"})
    min_valid_ratio: float = dataclasses.field(
        default=0.1, metadata={"help": "share of valid depth pixels a crop needs to be a sample, in (0, 1]"})
    steps_per_epoch: int = dataclasses.field(
        default=100, metadata={"help": "batches in an epoch of the infinite train stream, at least 1"})
    workers: int = dataclasses.field(
        default=8, metadata={"help": "processes reading the train stream; 0 reads in the training process"})

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(self, name)!r}")

        if self.dataset != MEGADEPTH:
            bad("dataset", f"expected {MEGADEPTH!r}, the only dataset")
        if not self.root.strip():
            bad("root", "expected a directory")
        if not 0.0 < self.train_split < 1.0:
            bad("train_split", "expected a fraction strictly between 0 and 1")
        for name in ("n_select", "n_test", "layout_samples", "steps_per_epoch", "max_train_files", "max_samples"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                bad(name, "expected an int of at least 1 (or null where null is allowed)")
        if not 0.0 < self.min_valid_ratio <= 1.0:
            bad("min_valid_ratio", "expected a fraction in (0, 1]")
        if self.workers < 0:
            bad("workers", "expected an int of at least 0")
        return lines


# ---------------------------------------------------------------------
# What is known before the data is loaded
# ---------------------------------------------------------------------


def depth_facts(data: DepthData) -> DataFacts:
    """Refuse a root that is not a MegaDepth tree with 3 validation scenes; see the module contract."""
    if not os.path.isdir(data.root):
        raise ConfigError(
            f"Invalid value for 'data.root': {data.root!r} is not a directory; give the MegaDepth root "
            "(<root>/<scene>/dense<N>/{imgs,depths})")
    # DECISION plan-2026-11-17T110935-9c7b9096/D-023
    # The scene deal is checked by the loader's own rule before the claim. Do NOT count scenes with a second walk of the
    # tree here "to stay TensorFlow-free" (two listings drift apart); the import loads TensorFlow but creates no tensor
    # and touches no device, which is what setup_gpu needs.
    from train.common.depth_data import deal_megadepth_scenes

    try:
        deal_megadepth_scenes(data.root, data.train_split)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid value for 'data.root': {exc} (data.train_split decides how many scenes validate)") from exc
    return DataFacts(dataset=MEGADEPTH, num_classes=None, image_size=None)


def read_depth_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``, a relative ``root`` anchored to the file."""
    return read_data_file(path, ("root",))


def load_depth_data_config(data: Any) -> DepthData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(DepthData, data, read_depth_data_file)


# ---------------------------------------------------------------------
# The other groups of the task config
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DepthTransformArgs(StrictConfig):
    """The crop size and the augmentation of the depth data; read by the depth task."""

    image_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={"help": f"side of the square crops the model trains on, a multiple of {DINOV2_PATCH} (the DINOv2 "
                          "patch); the model is built at it and predict and export serve that size only; auto is the "
                          "recipe's 252"})
    augment: _Augment = dataclasses.field(
        default="flip",
        metadata={"help": "flip: a horizontal flip with probability 0.5 of every train sample (the only augmentation, "
                          "never on the selection or test crops); none: no augmentation"})

    def problems(self, prefix: str) -> List[str]:
        """Name an ``image_size`` that is not a positive multiple of the patch."""
        size = self.image_size
        if _is_auto(size) or (size >= DINOV2_PATCH and size % DINOV2_PATCH == 0):
            return []
        return [f"Invalid value for {join_path(prefix, 'image_size')!r}: expected a positive multiple of {DINOV2_PATCH} "
                f"(the DINOv2 patch side) or auto, got {size!r}"]

    def resolve_auto(self, context: AutoContext) -> "DepthTransformArgs":
        """Fill ``image_size`` from the recipe."""
        return _with_values(self, "transform_args", _recipe_values(self, context, "transform_args"))


@dataclasses.dataclass(frozen=True)
class DepthMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the depth task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={"help": "metric that picks the best epoch: val_loss on the fixed selection crops, the only value "
                          "accepted (the success bar is about the minimum val_loss); auto is val_loss"})
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO, metadata={"help": "min, the only value accepted (the best epoch is the minimum loss); auto is min"})

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``val_loss`` and a ``watch_mode`` other than ``min``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success "
                    f"bar is a statement about the minimum {WATCH_METRIC} of the selection window, so the saved best "
                    "checkpoint must be that epoch")
        return lines

    def resolve_auto(self, context: AutoContext) -> "DepthMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))


@dataclasses.dataclass(frozen=True)
class DepthOptimArgs(OptimArgs):
    """The shared optimizer group with an ``"auto"`` rate that follows the pretrained start."""

    # DECISION plan-2026-11-17T110935-9c7b9096/D-022
    # From random weights an "auto" lr is the legacy from-scratch 2e-4, not the recipe's 5e-6: 5e-6 is the paper's rate for
    # a PRETRAINED encoder and trains a random one measurably slower (legacy D-115). The rule reads model_args.pretrained,
    # the request's over the recipe's, through pretrained_start, the same rule the head table and the weights check
    # follow. Do NOT put it in OptimArgs.resolve_auto (the other tasks' recipes name their own rates) and do NOT refuse an
    # auto lr from scratch instead (a from-scratch control is a supported run whose bar miss is reported).
    def resolve_auto(self, context: AutoContext) -> "DepthOptimArgs":
        """Fill the shared fields, then a from-scratch ``lr`` when the run does not start from the official weights."""
        resolved = super().resolve_auto(context)
        model_args = {**context.defaults.get("model_args", {}), **context.model_args}
        if _is_auto(self.lr) and not pretrained_start(model_args):
            return dataclasses.replace(resolved, lr=FROM_SCRATCH_LR)
        return resolved
