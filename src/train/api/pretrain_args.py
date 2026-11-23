"""The pretrain arguments of ``python -m train``: the data group with its mask keys, the transform and metric groups.

The ``pretrain`` task trains a model on masked images: ``et_mim`` predicts the pixels of the masked patches
(ET masked completion), ``beit_mim`` predicts the code ids a frozen tokenizer gives them (BEiT masked image modelling). Both
take their images from the shared image streams (``cifar10`` and ``imagenette``): a seeded slice of the train rows
selects, the test split is scored once after ``fit``, and a labelled image is only an image here. ``PretrainData`` names
the dataset, how it is split and capped, the patch the images are cut into and how many patches are masked.
``PretrainTransformArgs`` and ``PretrainMetricArgs`` are the groups of the pretrain task config
(``train.api.tasks.pretrain``) that differ from the shared ones: there is no class count, so classification's
``MetricArgs`` (a ``top_k``, a ``success_threshold`` that is a multiple of chance) cannot serve. Every option that cannot
apply is REFUSED, never ignored.

Interface contract
------------------
``TASK``
    ``"pretrain"``, the task these groups belong to.

``ET`` / ``BEIT``
    ``"et_mim"`` and ``"beit_mim"``, the names of the task's two families: :mod:`train.api.pretrain_models` registers its
    rows under them, and the help of the mask keys names them.

``DATASETS`` / ``GEOMETRY``
    The datasets the shared streams serve (``train.common.image_streams.SUPPORTED_DATASETS``, held equal by a test) and,
    per dataset, ``(image_size, patch_size)``: cifar10 32 px at patch 4 (an 8 x 8 grid), imagenette 224 px at patch 16
    (14 x 14), the geometry both legacy trainers' command lines give. The recipes' ``per_dataset`` overlays are built
    from this one table, and :func:`pretrain_facts` reads it, so a size lives in one place.

``PretrainData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; ``format`` is always ``"builtin"`` (the shared
    run sequence reads it). ``patch_size`` and the four mask keys are ``"auto"`` until the family's recipe fills them:
    ET masks by ``mask_ratio`` (the share of the patches in the loss) and ``mask_token_frac`` (the share of those
    occluded in the input), BEiT by blocks, ``num_mask_patches`` in all and ``min_mask_patches_per_block`` each. A key of the
    OTHER family resolves to ``None`` and an explicit value for it is refused by name; ``null`` for a key the family masks with is
    refused. ``problems(prefix)`` names every option out of range; ``resolve_auto(context)`` fills the keys from the recipe and the
    grid (``context.image_size``), and refuses a patch that does not divide the image, a ``num_mask_patches`` above the
    number of patches and a block minimum above the budget.

``load_pretrain_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one.
    :return: the validated config.
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping, or any invalid key (every problem
        at once; a file's lines name the file).

``read_pretrain_data_file(path)``
    :return: the mapping in a YAML file (no key of this config names a path).

``pretrain_facts(data, requested_image_size)``
    :param requested_image_size: ``transform_args.image_size``, or ``None`` for the dataset's own (:data:`GEOMETRY`).
    :return: ``DataFacts(dataset, None, image_size)``: no class count. No file is opened and no TensorFlow op runs, so it
        runs before ``setup_gpu`` and before the run directory is claimed.
    :raises ConfigError: for imagenette without its prepared records (the same path test the classifiers make).

``auto_mask_counts(num_patches)``
    :return: ``(num_mask_patches, min_mask_patches_per_block)`` of the BEiT recipe's fractions, 75 and 16 of 196 patches, at
        ``num_patches``: the minimum clamped to the budget, both at least 1 (24 and 5 at an 8 x 8 grid).

``resolve_mask_counts(num_mask_patches, min_mask_patches_per_block, num_patches)``
    :param num_mask_patches: ``"auto"`` or an int; likewise the minimum.
    :return: the two counts, ``"auto"`` filled by :func:`auto_mask_counts`.
    :raises ConfigError: naming ``data.num_mask_patches`` above ``num_patches`` and ``data.min_mask_patches_per_block`` above
        the budget.

``PretrainTransformArgs``
    ``image_size`` (``"auto"`` is the size the data reports: the dataset's own, or the one asked for) and ``augment`` (the
    shared loader's flip and pad-crop of the train stream; the selection and test streams are never augmented).

``PretrainMetricArgs``
    ``watch_metric`` (auto is the recipe's ``val_loss``; the only value accepted) and ``watch_mode`` (auto is ``min``; the
    only value accepted): the success bar is a statement about the MINIMUM ``val_loss`` of the selection window, so the
    checkpoint that is saved and reloaded must be that epoch, and ``problems`` refuses any other metric or direction by
    key. The bar is the task's (a baseline measured on the run's own selection stream), not a key.

Import rule
-----------
Module scope imports only the standard library and ``train.api``; ``train.common`` is imported inside the one method that
needs it, so ``python -m train pretrain help`` stays free of TensorFlow.
"""

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .args import AutoContext, _is_auto, _recipe_values, _watch_values, _with_values
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, check_builtin_sources, load_data_group, read_data_file
from .errors import ConfigError

# ---------------------------------------------------------------------

TASK = "pretrain"

# DECISION plan-2026-11-19T081214-5c6cc105/D-022: the two family names live HERE, beside TASK, because the help of the
# mask keys below names them and `pretrain_models`, which registers the rows under them, imports this module (the other
# way round is a cycle). Do NOT define them again there or type them into a help text or a message: a rename then misses
# the typed copy, as the help of these four keys would have.
ET, BEIT = "et_mim", "beit_mim"

#: The datasets of the shared image streams; ``test_pretrain_families`` holds it equal to
#: ``train.common.image_streams.SUPPORTED_DATASETS``.
DATASETS: Tuple[str, ...] = ("cifar10", "imagenette")

#: ``(image_size, patch_size)`` per dataset: what both legacy trainers' command lines resolve (``config_from_args``).
GEOMETRY: Dict[str, Tuple[int, int]] = {"cifar10": (32, 4), "imagenette": (224, 16)}

#: The BEiT recipe's masking: ``run_beit_pretraining.py`` masks 75 patches in blocks of at least 16 on its 14 x 14 grid of
#: 196 patches. ``test_pretrain_families`` holds the three equal to ``dl_techniques.datasets.vision.beit_masking`` and to the
#: legacy ``train.beit.train_mim.TrainingConfig``; they are copied here because that module imports TensorFlow.
BEIT_GRID_PATCHES = 196
BEIT_NUM_MASK_PATCHES = 75
BEIT_MIN_MASK_PATCHES_PER_BLOCK = 16

#: The four mask keys of :class:`PretrainData`: the first two are ET's, the last two BEiT's.
MASK_KEYS = ("mask_ratio", "mask_token_frac", "num_mask_patches", "min_mask_patches_per_block")

_Dataset = Literal[DATASETS]  # type: ignore[valid-type]


def auto_mask_counts(num_patches: int) -> Tuple[int, int]:
    """Return the BEiT recipe's mask counts at ``num_patches`` patches; see the module contract."""
    # DECISION plan-2026-11-12T015747-46e5318f/D-008
    # DECISION plan-2026-11-12T015747-46e5318f/D-018
    # The auto counts keep the BEiT recipe's FRACTIONS of the grid (75 and 16 of 196), not its absolute numbers: 75
    # patches cannot be drawn on cifar10's 8 x 8 grid of 64, so the legacy default fails at config time there. Measured
    # at 8 x 8 over 2,000 draws each: (24, 5) leaves 4.50 percent of the masks under-filled, harmless because the loss
    # weight `mask * N / n_masked` makes an under-filled mask score on the same scale. Do NOT default to 75, and do NOT
    # clamp an explicit count down to the grid: an explicit count above it is a request that cannot be honoured, and is
    # refused by name in `resolve_mask_counts`.
    num = max(1, round(BEIT_NUM_MASK_PATCHES / BEIT_GRID_PATCHES * num_patches))
    block = max(1, round(BEIT_MIN_MASK_PATCHES_PER_BLOCK / BEIT_GRID_PATCHES * num_patches))
    return num, min(block, num)


def resolve_mask_counts(num_mask_patches: Any, min_mask_patches_per_block: Any, num_patches: int) -> Tuple[int, int]:
    """Return the two BEiT mask counts at ``num_patches`` patches, ``"auto"`` filled; see the module contract."""
    side = round(num_patches ** 0.5)
    grid = f"{side} x {side}" if side * side == num_patches else f"{num_patches} patches"
    auto_num, auto_block = auto_mask_counts(num_patches)
    num = auto_num if _is_auto(num_mask_patches) else num_mask_patches
    if num > num_patches:
        raise ConfigError(
            f"Invalid value for 'data.num_mask_patches': {num!r} is above the {num_patches} patches of a {grid} grid "
            f"(the image cut into patches), so no mask of that size can be drawn; expected at most {num_patches}, "
            f"or leave it at auto ({auto_num} here)"
        )
    block = min(auto_block, num) if _is_auto(min_mask_patches_per_block) else min_mask_patches_per_block
    if block > num:
        raise ConfigError(
            f"Invalid value for 'data.min_mask_patches_per_block': {block!r} is above data.num_mask_patches ({num!r}): "
            "a block cannot be larger than the whole mask; expected at most the budget"
        )
    return num, block


@dataclasses.dataclass(frozen=True)
class PretrainData(StrictConfig):
    """The images of a pretrain run: the dataset, its split, the patch and the mask; see the module contract."""

    dataset: _Dataset = dataclasses.field(
        metadata={
            "help": "cifar10 or imagenette: the images the model is trained to complete or to tokenize; their labels are "
            "not used. A seeded slice of the train rows selects the best epoch and the test split is scored once after fit"
        }
    )
    format: Literal["builtin"] = dataclasses.field(
        default="builtin", metadata={"help": "where the data comes from; the only value"}
    )
    validation_split: Optional[float] = dataclasses.field(
        default=None,
        metadata={"help": "fraction of train held out to select the best epoch, in (0, 1); null is the loader's 0.1"},
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "cap on the train pool, fit and selection together; the test split is never capped; null is no cap"
        },
    )
    patch_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "side of the square patches the image is cut into, and of the model's patch embedding; it must divide "
            "transform_args.image_size; auto is 4 for cifar10 (an 8 x 8 grid) and 16 for imagenette (14 x 14)"
        },
    )
    mask_ratio: Auto[Optional[float]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": f"{ET} only: the share of the patches in the loss, in (0, 1]; auto is the paper's 0.5; "
            f"refused for {BEIT}"
        },
    )
    mask_token_frac: Auto[Optional[float]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": f"{ET} only: the share of the loss patches occluded in the input, in (0, 1], the rest are "
            f"scored without being hidden; auto is the paper's 0.9; refused for {BEIT}; a value that occludes every loss "
            "patch is refused by the library when the data loads"
        },
    )
    num_mask_patches: Auto[Optional[int]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": f"{BEIT} only: how many patches are masked, in blocks; at most the number of patches; auto is the BEiT "
            f"recipe's 75 of 196 patches, scaled to the grid (24 at 8 x 8); refused for {ET}"
        },
    )
    min_mask_patches_per_block: Auto[Optional[int]] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": f"{BEIT} only: the smallest block of masked patches, at most num_mask_patches; auto is the BEiT "
            f"recipe's 16 of 196 patches, scaled to the grid (5 at 8 x 8); refused for {ET}"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range and every contradiction between options, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(self, name)!r}")

        split = self.validation_split
        if split is not None and not 0 < split < 1:
            bad("validation_split", "expected a fraction strictly between 0 and 1")
        if self.max_samples is not None and self.max_samples < 1:
            bad("max_samples", "expected a positive int or null")
        if not _is_auto(self.patch_size) and self.patch_size < 1:
            bad("patch_size", "expected an int of at least 1")
        for name in ("mask_ratio", "mask_token_frac"):
            value = getattr(self, name)
            if not _is_auto(value) and value is not None and not 0 < value <= 1:
                bad(name, "expected a fraction in (0, 1] or null")
        for name in ("num_mask_patches", "min_mask_patches_per_block"):
            value = getattr(self, name)
            if not _is_auto(value) and value is not None and value < 1:
                bad(name, "expected an int of at least 1 or null")
        return lines

    def resolve_auto(self, context: AutoContext) -> "PretrainData":
        """Fill ``"auto"`` from the family's recipe and the grid, and refuse a mask the family or the grid cannot take."""
        found = _recipe_values(self, context, "data")
        table = context.defaults.get("data", {})
        patch = found.get("patch_size", self.patch_size)
        size = context.image_size
        if _is_auto(patch):
            raise ConfigError(
                f"Invalid value for 'data.patch_size': the {context.family!r} recipe has no patch size for "
                f"{context.dataset!r}; set data.patch_size"
            )
        if not isinstance(size, int) or isinstance(size, bool) or size < 1 or size % patch:
            raise ConfigError(
                f"Invalid value for 'data.patch_size': {patch!r} does not divide transform_args.image_size ({size!r}), "
                "so the image cannot be cut into whole patches; set one that divides the other"
            )
        # A key whose recipe entry is null belongs to the other family: the recipe is the one place that says which
        # family masks how, so an explicit value for it is refused here and no table of families is kept beside it.
        foreign = [name for name in MASK_KEYS if name in table and table[name] is None]
        own = [name for name in MASK_KEYS if name not in foreign]
        for name in foreign:
            value = getattr(self, name)
            if not _is_auto(value) and value is not None:
                raise ConfigError(
                    f"Invalid value for 'data.{name}': the {context.family!r} family does not take it (its recipe masks "
                    f"with {', '.join('data.' + key for key in own)}), got {value!r}"
                )
            found[name] = None
        for name in own:
            if getattr(self, name) is None:
                raise ConfigError(
                    f"Invalid value for 'data.{name}': null is no value for the {context.family!r} family, which masks with "
                    f"it; leave it at auto"
                )
        counts = ("num_mask_patches", "min_mask_patches_per_block")
        if all(name in own and name not in table for name in counts):
            found["num_mask_patches"], found["min_mask_patches_per_block"] = resolve_mask_counts(
                self.num_mask_patches, self.min_mask_patches_per_block, (size // patch) ** 2
            )
        return _with_values(self, "data", found)


# ---------------------------------------------------------------------
# The config from a mapping or a file
# ---------------------------------------------------------------------


def read_pretrain_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``; no key of the data config names a path."""
    return read_data_file(path, ())


def load_pretrain_data_config(data: Any) -> PretrainData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(PretrainData, data, read_pretrain_data_file)


def pretrain_facts(data: PretrainData, requested_image_size: Optional[int]) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    check_builtin_sources(data)
    size = GEOMETRY[data.dataset][0] if requested_image_size is None else requested_image_size
    return DataFacts(dataset=data.dataset, num_classes=None, image_size=size)


# ---------------------------------------------------------------------
# The groups of the task config
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PretrainTransformArgs(StrictConfig):
    """The input size and the augmentation of the train stream; read by the pretrain task."""

    image_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "side length the images are resized to; it must be a multiple of data.patch_size; auto is the "
            "dataset's own, 32 for cifar10 and 224 for imagenette"
        },
    )
    augment: bool = dataclasses.field(
        default=True,
        metadata={
            "help": "flip and pad-crop the TRAIN images (the legacy trainers' default); the selection and test images are "
            "never augmented"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit ``image_size`` below 1."""
        if not _is_auto(self.image_size) and self.image_size < 1:
            return [
                f"Invalid value for {join_path(prefix, 'image_size')!r}: expected a positive int, got {self.image_size!r}"
            ]
        return []

    def resolve_auto(self, context: AutoContext) -> "PretrainTransformArgs":
        """Fill ``image_size`` from the data's facts."""
        found = _recipe_values(self, context, "transform_args", skip=("image_size",))
        if _is_auto(self.image_size):
            size = context.image_size
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ConfigError(
                    f"The data reports image_size {size!r}; transform_args.image_size = auto needs a positive integer. "
                    "Set transform_args.image_size."
                )
            found["image_size"] = size
        return _with_values(self, "transform_args", found)


#: The one metric and direction the bar can describe (the pretrain recipes carry the same metric; a test holds them equal).
WATCH_METRIC = "val_loss"
WATCH_MODE = "min"


@dataclasses.dataclass(frozen=True)
class PretrainMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the pretrain task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: val_loss on the fixed selection masks, the only value accepted (the "
            "success bar is about the minimum val_loss, so the saved checkpoint must be that epoch); auto is val_loss"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "min, the only value accepted (val_loss is a loss: the best epoch is its minimum); auto is min"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``val_loss`` and a ``watch_mode`` other than ``min``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success "
                    f"bar is a statement about the minimum val_loss of the selection window, so the saved best checkpoint "
                    f"must be that epoch ({WATCH_METRIC!r} with mode {WATCH_MODE!r}); another metric or the other direction "
                    "would save weights the verdict does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "PretrainMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))
