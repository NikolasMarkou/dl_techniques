"""The ``dino`` arguments of ``python -m train``: the data group, the metric group, the geometry and the facts.

The ``dino`` task trains the DINOv1 self-supervised ViT (a student, an EMA teacher, multi-crop views, a centred loss) on the images of the
shared image streams (``cifar10`` and ``imagenette``): a seeded slice of the train rows selects the best epoch by a k-NN probe of the
backbone's features, the test split is scored once after ``fit``, and a labelled image is only an image to the training (its label is read by
the probe alone). ``DinoData`` names the dataset, how it is split and capped, the patch and the number of local crops; ``DinoMetricArgs`` is
the metric group of the task config of this task, and ``DinoTransformArgs`` its transform group: the size every view is rendered at. Every option
that cannot apply is REFUSED, never ignored.

Interface contract
------------------
``TASK`` / ``FAMILY``
    ``"dino"`` and ``"dino_v1"``: the task, and its one family, whose row :mod:`train.api.dino_models` builds. They live HERE, beside the groups
    whose help names them.

``WATCH_METRIC`` / ``WATCH_MODE``
    ``"dino_knn_top1_k20"`` and ``"max"``, the only metric and direction :class:`DinoMetricArgs` accepts: the key the k-NN probe writes into
    every epoch's logs (a test holds it equal to the probe's own key format).

``GEOMETRY`` / ``MIN_TOKENS``
    ``{dataset: (image_size, patch_size)}``: cifar10 32 px at patch 4 (an 8 x 8 grid), imagenette 96 px at patch 16 (6 x 6), the geometries the
    recipe starts from (the legacy trainer's 224 px was never run; 96 px is the record's scale). One row per dataset the shared streams serve
    (``pretrain_args.DATASETS``). ``MIN_TOKENS`` is the fewest patch tokens a view may have.

``DinoData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; ``format`` is always ``"builtin"``. ``patch_size`` and
    ``n_local_crops`` are ``"auto"`` until the recipe fills them. ``problems(prefix)`` names every option out of range;
    ``resolve_auto(context)`` fills both from the recipe and refuses a recipe that has none for the dataset. Whether the patch tiles the image
    is :func:`check_geometry`'s, which needs the resolved image size.

``DinoTransformArgs``
    ``image_size`` (``"auto"`` is the size the data reports: the dataset's own in :data:`GEOMETRY`, or the one asked for): the side of the GLOBAL
    crop, which every view, the local crops too, is rendered at. There is no ``augment`` key, unlike the pretrain task's group: the multi-crop map
    applies the DINO augmentations itself, so a switch for the flip and pad-crop of the shared loader would be a key nothing reads.

``read_dino_data_file(path)`` / ``load_dino_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one.
    :return: the validated config / the mapping in the YAML file (no key of this config names a path).
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping; ``ConfigValidationError`` for any invalid key
        (every problem at once; a file's lines name the file).

``dino_facts(data, requested_image_size)``
    :param requested_image_size: ``transform_args.image_size``, or ``None`` for the dataset's own (:data:`GEOMETRY`).
    :return: ``DataFacts(dataset, None, image_size)``: no class count. No file is opened and no TensorFlow op runs, so it runs before
        ``setup_gpu`` and before the run directory is claimed.
    :raises ConfigError: for imagenette without its prepared records (the same path test the classifiers make).

``geometry_problem(patch_size, side)``
    :param patch_size, side: any values; no check of type is assumed.
    :return: why ``patch_size`` cannot cut a square side of ``side`` pixels into views of at least :data:`MIN_TOKENS` patch tokens (not a
        positive int, not a power of 2, not dividing the side, or too few tokens), as a phrase that continues "<patch> ...", or ``None`` when it
        can. The ONE statement of the rule: :func:`check_geometry` and the model adapter both read it.

``check_geometry(patch_size, image_size)``
    :raises ConfigError: naming ``data.patch_size`` and ``transform_args.image_size`` for a patch :func:`geometry_problem` refuses. Pure, with no
        file and no tensor: it is a check for before the run directory is claimed.

``DinoMetricArgs``
    ``watch_metric`` (auto is ``dino_knn_top1_k20``; the only value accepted) and ``watch_mode`` (auto is ``max``; the only value accepted):
    the bar is a statement about the BEST selection k-NN accuracy, so the checkpoint that is saved and reloaded must be that epoch, and
    ``problems`` refuses any other metric or direction by key. ``val_loss`` is refused because a DINO run has no validation data (a
    validation pass would also advance the loss's centre).

Import rule
-----------
Module scope imports only the standard library and ``train.api``; no ``train.common`` and no model code, so ``python -m train dino help`` stays
free of TensorFlow.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from .args import AutoContext, _is_auto, _recipe_values, _watch_values, _with_values, is_count
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, check_builtin_sources, load_data_group, read_data_file
from .errors import ConfigError
from .pretrain_args import DATASETS

# ---------------------------------------------------------------------

TASK = "dino"
FAMILY = "dino_v1"

#: The one metric and direction the bar can describe; ``test_dino_families`` holds the key equal to the probe's own.
WATCH_METRIC = "dino_knn_top1_k20"
WATCH_MODE = "max"

#: ``(image_size, patch_size)`` per dataset; the recipes' ``per_dataset`` overlays are built from this one table and :func:`dino_facts` reads it.
GEOMETRY = {"cifar10": (32, 4), "imagenette": (96, 16)}

#: The fewest patch tokens a view may have: the library's patch 16 on a 32 px image is a 2 x 2 grid (4 tokens), which no run measured.
MIN_TOKENS = 5

_Dataset = Literal[DATASETS]  # type: ignore[valid-type]


def _geometry_words() -> str:
    """The patch and the grid of every dataset, from the one table (:data:`GEOMETRY`), for the help of ``patch_size``."""
    return " and ".join(
        f"{patch} for {name} ({size} px, grid {size // patch} x {size // patch})" for name, (size, patch) in GEOMETRY.items()
    )


@dataclasses.dataclass(frozen=True)
class DinoData(StrictConfig):
    """The images of a DINO run: the dataset, its split, the patch and the local crops; see the module contract."""

    dataset: _Dataset = dataclasses.field(
        metadata={
            "help": "cifar10 or imagenette: the images the student and the teacher learn from; their labels are used by the k-NN probe "
            "only. A seeded slice of the train rows selects the best epoch and the test split is scored once after fit"
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
            "help": "side of the square patch of one token: a power of 2 that divides transform_args.image_size and leaves a view at least "
            f"{MIN_TOKENS} tokens; auto is {_geometry_words()}"
        },
    )
    n_local_crops: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "local crops per image, in addition to the two global crops; every crop is rendered at transform_args.image_size. At "
            "least 1 (a run with none is the two-view baseline); auto is the recipe's"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range, at its dotted path."""
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
        if not _is_auto(self.n_local_crops) and self.n_local_crops < 1:
            bad("n_local_crops", "expected an int of at least 1 (a run with no local crop is the two-view baseline, which no run measured)")
        return lines

    def resolve_auto(self, context: AutoContext) -> "DinoData":
        """Fill ``patch_size`` and ``n_local_crops`` from the family's recipe for the dataset, and refuse a recipe that has none."""
        found = _recipe_values(self, context, "data")
        for name in ("patch_size", "n_local_crops"):
            if _is_auto(found.get(name, getattr(self, name))):
                raise ConfigError(
                    f"Invalid value for 'data.{name}': the {context.family!r} recipe has no {name} for {context.dataset!r}; set data.{name}"
                )
        return _with_values(self, "data", found)


# DECISION plan-2026-11-26T224507-e3381378/D-051
# The group is the task's OWN and has one key. The pretrain task's `PretrainTransformArgs` was the plan's first choice, and printing its help showed two
# defects for this task: its `image_size` says "224 for imagenette" (the DINO recipe is 96 px), and its `augment` key switches the shared loader's flip and
# pad-crop, which the multi-crop map never reads (a key nothing reads is a lie the config-fields census would also catch). Do NOT reuse it, and do NOT add an
# `augment` key "for symmetry": the DINO augmentations are the map's, fixed, and `stateless_augmentation` is refused by name in `dino_models.REFUSED`.
@dataclasses.dataclass(frozen=True)
class DinoTransformArgs(StrictConfig):
    """The size every view is rendered at; read by the dino task."""

    image_size: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "side length of the global crops and of every local crop (all views have the same resolution); it must be a multiple of "
            f"data.patch_size; auto is the dataset's own: {', '.join(f'{size} for {name}' for name, (size, _) in GEOMETRY.items())}"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name an explicit ``image_size`` below 1."""
        if not _is_auto(self.image_size) and self.image_size < 1:
            return [f"Invalid value for {join_path(prefix, 'image_size')!r}: expected a positive int, got {self.image_size!r}"]
        return []

    def resolve_auto(self, context: AutoContext) -> "DinoTransformArgs":
        """Fill ``image_size`` from the data's facts."""
        found = _recipe_values(self, context, "transform_args", skip=("image_size",))
        if _is_auto(self.image_size):
            if not is_count(context.image_size):
                raise ConfigError(
                    f"The data reports image_size {context.image_size!r}; transform_args.image_size = auto needs a positive integer. "
                    "Set transform_args.image_size."
                )
            found["image_size"] = context.image_size
        return _with_values(self, "transform_args", found)


# ---------------------------------------------------------------------
# The config from a mapping or a file, and what is known before the data is loaded
# ---------------------------------------------------------------------


def read_dino_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``; no key of the data config names a path."""
    return read_data_file(path, ())


def load_dino_data_config(data: Any) -> DinoData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(DinoData, data, read_dino_data_file)


def dino_facts(data: DinoData, requested_image_size: Optional[int]) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    check_builtin_sources(data)
    size = GEOMETRY[data.dataset][0] if requested_image_size is None else requested_image_size
    return DataFacts(dataset=data.dataset, num_classes=None, image_size=size)


# ---------------------------------------------------------------------
# The geometry, refused before the claim
# ---------------------------------------------------------------------


def geometry_problem(patch_size: Any, side: Any) -> Optional[str]:
    """Return why ``patch_size`` cannot tile a square of ``side`` pixels into acceptable views, ``None`` when it can."""
    if not is_count(patch_size):
        return "is not a positive int"
    # DECISION plan-2026-11-26T224507-e3381378/D-042
    # A power of 2 is the plan's rule (Step 5: "a power of two dividing the size, at least 5 tokens"), NOT a limit of the library: DINOv1 builds
    # a patch of 12 on a 96 px image. The two offered patches (16 at 96 px, 4 at 32 px) are powers of 2 and so is every size the record ran,
    # so a patch outside the set is a configuration no run has seen. Do NOT relax it to "any divisor" without a run, and do NOT call it a
    # limit of the model in the message. The same one function is read by the adapter, so the front door and the factory agree.
    if patch_size & (patch_size - 1):
        return "is not a power of 2: the offered patches (16 at 96 px, 4 at 32 px) are, and no run used another"
    if not is_count(side):
        return f"cannot tile an image whose side is {side!r}, which is not a positive int"
    if side % patch_size:
        return f"does not divide the image side {side}: the patch grid would round down and cut off the border"
    tokens = (side // patch_size) ** 2
    if tokens < MIN_TOKENS:
        return (
            f"leaves {tokens} tokens per view, fewer than the {MIN_TOKENS} this task asks of a view (the library's patch 16 on a "
            "32 px image is a 2 x 2 grid), and no run measured a view that small"
        )
    return None


def check_geometry(patch_size: Any, image_size: Any) -> None:
    """Refuse a patch that cannot tile the image into views of at least :data:`MIN_TOKENS` tokens; see the module contract."""
    reason = geometry_problem(patch_size, image_size)
    if reason is not None:
        raise ConfigError(
            f"Invalid value for 'data.patch_size': {patch_size!r} {reason} (transform_args.image_size is {image_size!r}); the "
            f"patch must be a power of 2 that divides the image side and leaves at least {MIN_TOKENS} tokens per view"
        )


# ---------------------------------------------------------------------
# The metric group of the task config
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DinoMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the dino task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: dino_knn_top1_k20, the k-NN top-1 of the backbone's features on the fixed selection "
            "split, the only value accepted (the bar is about its best value, and a DINO run has no val_loss: it has no validation data); "
            "auto is dino_knn_top1_k20"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "max, the only value accepted (an accuracy: the best epoch is its maximum); auto is max"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``dino_knn_top1_k20`` and a ``watch_mode`` other than ``max``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success bar is a "
                    f"statement about the best {WATCH_METRIC} of the selection window, so the saved best checkpoint must be that epoch "
                    f"({WATCH_METRIC!r} with mode {WATCH_MODE!r}); another metric or the other direction would save weights the verdict "
                    "does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "DinoMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))
