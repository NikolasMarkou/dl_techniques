"""The ``beit_tokenizer`` arguments of ``python -m train``: the data group, the metric group, the geometry and the codebook checks.

The ``beit_tokenizer`` task trains the BEiT stage-0 tokenizer, the library ``VQVAERotationTrick``, to reconstruct the images of the shared
image streams (``cifar10`` and ``imagenette``) through a codebook: one code per ``data.patch_size`` x ``data.patch_size`` cell of the
image, so the code grid is ``image_size // patch_size`` on a side, which is the patch grid a ``beit_mim`` run needs. A seeded slice of the
train rows selects the best epoch, the test split is scored once after ``fit``, and a labelled image is only an image here.
``TokenizerData`` names the dataset, how it is split and capped, and the patch; ``TokenizerMetricArgs`` is the metric group of the task
config of this task. The transform group is the pretrain task's (``train.api.pretrain_args.PretrainTransformArgs``:
the image size and the augmentation of the train stream), not a copy. Every option that cannot apply is REFUSED, never ignored.

Interface contract
------------------
``TASK`` / ``FAMILY``
    ``"beit_tokenizer"`` and ``"vq_vae_rotation"`` (the library package's name): the task, and its one family, whose row
    :mod:`train.api.tokenizer_models` builds. They live HERE, beside the groups whose help names them.

``WATCH_METRIC`` / ``WATCH_MODE``
    ``"val_reconstruction_loss"`` and ``"min"``, the only metric and direction :class:`TokenizerMetricArgs` accepts: the key the library
    model's own ``test_step`` reports for the reconstruction term (a test holds it equal to the built model's metric names).

``TokenizerData``
    Frozen :class:`~train.api.config.StrictConfig`. ``dataset`` is required; ``format`` is always ``"builtin"``. ``patch_size`` is ``"auto"``
    until the recipe fills it (the geometry table :data:`train.api.pretrain_args.GEOMETRY`, one patch per dataset).
    ``problems(prefix)`` names every option out of range; ``resolve_auto(context)`` fills the patch from the recipe and refuses a recipe
    that has none for the dataset. Whether the patch cuts the image is :func:`check_geometry`'s, which needs the resolved image size.

``read_tokenizer_data_file(path)`` / ``load_tokenizer_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one.
    :return: the validated config / the mapping in the YAML file (no key of this config names a path).
    :raises ConfigError: for a file that cannot be read or a value that is not a mapping; ``ConfigValidationError`` for any invalid key
        (every problem at once; a file's lines name the file).

``tokenizer_facts(data, requested_image_size)``
    :param requested_image_size: ``transform_args.image_size``, or ``None`` for the dataset's own.
    :return: ``DataFacts(dataset, None, image_size)``: no class count. No file is opened and no TensorFlow op runs, so it runs before
        ``setup_gpu`` and before the run directory is claimed.
    :raises ConfigError: for imagenette without its prepared records (the same path test the classifiers make).

``geometry_problem(patch_size, side)``
    :param patch_size, side: any values; no check of type is assumed.
    :return: why ``patch_size`` cannot cut a square side of ``side`` pixels into whole code cells (not a positive int, below 2, not a
        power of 2, or not dividing the side), as a phrase that continues "<patch> ...", or ``None`` when it can. The ONE statement of the
        rule: the adapter and :func:`check_geometry` both read it, and a test holds it equal to a real build of the library class with a
        residual block (a patch of 1 also builds with none, and the task refuses it all the same: it asks for a cell of at least 2 x 2 pixels).

``check_geometry(patch_size, image_size, num_embeddings, *, floor, smoke)``
    :param floor: the least number of codes a codebook may have, the perplexity floor of the bar
        (``train.common.pretrain_run.MIN_TOKENIZER_PERPLEXITY``, which the caller passes: this module imports no ``train.common``, so the
        number is never copied here). A codebook of K codes cannot reach a perplexity above K.
    :param smoke: ``debug_args.smoke``: a smoke run applies no bar, so its codebook may be as small as it likes (but a positive int).
    :raises ConfigError: naming ``data.patch_size`` and ``transform_args.image_size`` for a patch that cannot cut the image into whole
        code cells, and ``model_args.num_embeddings`` for a codebook that is not a positive int or, outside a smoke run, is under
        ``floor``. Pure, with no file and no tensor: it is a check for before the run directory is claimed.

``TokenizerMetricArgs``
    ``watch_metric`` (auto is ``val_reconstruction_loss``; the only value accepted) and ``watch_mode`` (auto is ``min``; the only value
    accepted): the bar is a statement about the MINIMUM ``val_reconstruction_loss`` of the selection window, so the checkpoint that is
    saved and reloaded must be that epoch, and ``problems`` refuses any other metric or direction by key. ``val_loss`` is refused
    because it carries the quantizer's own terms, which the constant image's score has no counterpart for.

Import rule
-----------
Module scope imports only the standard library and ``train.api``; no ``train.common`` and no model code, so
``python -m train beit_tokenizer help`` stays free of TensorFlow.
"""

import dataclasses
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from .args import AutoContext, _is_auto, _recipe_values, _watch_values, _with_values, is_count
from .config import AUTO, Auto, StrictConfig, join_path
from .data_args import DataFacts, check_builtin_sources, load_data_group, read_data_file
from .errors import ConfigError
from .pretrain_args import DATASETS, GEOMETRY

# ---------------------------------------------------------------------

TASK = "beit_tokenizer"
FAMILY = "vq_vae_rotation"

#: The one metric and direction the bar can describe; ``test_beit_tokenizer_families`` holds the metric equal to the library model's.
WATCH_METRIC = "val_reconstruction_loss"
WATCH_MODE = "min"

_Dataset = Literal[DATASETS]  # type: ignore[valid-type]


def _geometry_words() -> str:
    """The patch and the code grid of every dataset, from the one table (:data:`GEOMETRY`), for the help of ``patch_size``."""
    return " and ".join(f"{patch} for {name} (grid {size // patch} x {size // patch})" for name, (size, patch) in GEOMETRY.items())


@dataclasses.dataclass(frozen=True)
class TokenizerData(StrictConfig):
    """The images of a tokenizer run: the dataset, its split and the patch one code describes; see the module contract."""

    dataset: _Dataset = dataclasses.field(
        metadata={
            "help": "cifar10 or imagenette: the images the tokenizer learns to reconstruct through its codebook; their labels are not "
            "used. A seeded slice of the train rows selects the best epoch and the test split is scored once after fit"
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
            "help": "side of the square cell of the image that ONE code describes, the encoder's total stride: a power of 2, at least 2, "
            "that divides transform_args.image_size. The code grid is image_size / patch_size on a side, and a beit_mim run reads it only "
            f"with its own data.patch_size equal to this; auto is {_geometry_words()}"
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
        return lines

    def resolve_auto(self, context: AutoContext) -> "TokenizerData":
        """Fill ``patch_size`` from the family's recipe for the dataset, and refuse a recipe that has none."""
        found = _recipe_values(self, context, "data")
        if _is_auto(found.get("patch_size", self.patch_size)):
            raise ConfigError(
                f"Invalid value for 'data.patch_size': the {context.family!r} recipe has no patch size for {context.dataset!r}; "
                "set data.patch_size"
            )
        return _with_values(self, "data", found)


# ---------------------------------------------------------------------
# The config from a mapping or a file, and what is known before the data is loaded
# ---------------------------------------------------------------------


def read_tokenizer_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``; no key of the data config names a path."""
    return read_data_file(path, ())


def load_tokenizer_data_config(data: Any) -> TokenizerData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    return load_data_group(TokenizerData, data, read_tokenizer_data_file)


def tokenizer_facts(data: TokenizerData, requested_image_size: Optional[int]) -> DataFacts:
    """Return the facts of a data config before it is loaded; see the module contract."""
    check_builtin_sources(data)
    size = GEOMETRY[data.dataset][0] if requested_image_size is None else requested_image_size
    return DataFacts(dataset=data.dataset, num_classes=None, image_size=size)


# ---------------------------------------------------------------------
# The geometry and the codebook, refused before the claim
# ---------------------------------------------------------------------


def geometry_problem(patch_size: Any, side: Any) -> Optional[str]:
    """Return why ``patch_size`` cannot cut a side of ``side`` pixels into whole code cells, ``None`` when it can."""
    if not is_count(patch_size):
        return "is not a positive int"
    # DECISION plan-2026-11-25T124419-fa94f59c/D-065
    # A patch of 1 is refused HERE, before the claim: `1 & 0 == 0`, so the power-of-2 test below passes it, and the library then fails
    # AFTER the claim with the default model (its encoder has no stride-2 stage, so the first residual block adds the 3-channel image to a
    # hidden_channels-wide tensor). That is NOT the reason the message gives (D-080): the library builds a patch of 1 with no residual
    # block and with a width of 3, so the rule is the task's own, a code that describes a cell of at least 2 x 2 pixels. Do NOT lean on
    # the power-of-2 test alone, do NOT let the library refuse it, and do NOT give a failure of the library as the reason in the message:
    # it is false for those models. The families test of the pre-claim rule builds the library class at every patch and side with one
    # residual block and holds this function equal to what it can build, and another builds the corners where it can.
    if patch_size < 2:
        return (
            "is below 2: the encoder is a stack of stride-2 convolutions, so a patch of 1 leaves it no stage, and this task asks for a "
            "code that describes a cell of at least 2 x 2 pixels, not one pixel"
        )
    if patch_size & (patch_size - 1):
        return (
            "is not a power of 2: the encoder is a stack of stride-2 convolutions, so one code can only describe a cell "
            "whose side is a power of 2"
        )
    if not is_count(side):
        return f"cannot cut an image whose side is {side!r}, which is not a positive int"
    if side % patch_size:
        return (
            f"does not divide the image side {side}: the convolution stack would round the code grid down and every code "
            "would describe a cell the image does not have"
        )
    return None


def check_geometry(patch_size: Any, image_size: Any, num_embeddings: Any, *, floor: float, smoke: bool) -> None:
    """Refuse a patch that cannot cut the image and a codebook the bar could never be met with; see the module contract."""
    reason = geometry_problem(patch_size, image_size)
    if reason is not None:
        raise ConfigError(
            f"Invalid value for 'data.patch_size': {patch_size!r} {reason} (transform_args.image_size is {image_size!r}); the "
            "patch must be a power of 2, at least 2, that divides the image side"
        )
    if not is_count(num_embeddings):
        raise ConfigError(
            f"Invalid value for 'model_args.num_embeddings': expected a positive int, the codebook size, got {num_embeddings!r}"
        )
    # DECISION plan-2026-11-25T124419-fa94f59c/D-035
    # The floor is the CALLER's: this module imports no `train.common`, and the one definition of the floor is
    # `pretrain_run.MIN_TOKENIZER_PERPLEXITY` (a toy-run test patches it, and then this check and the verdict move together). Do NOT
    # copy 64 here, and do NOT refuse a smoke run: it applies no bar. A codebook of K codes cannot reach a perplexity above K, so a
    # smaller one could never meet the bar and the refusal is arithmetic, not taste.
    if not smoke and num_embeddings < floor:
        raise ConfigError(
            f"Invalid value for 'model_args.num_embeddings': {num_embeddings!r} is below the {floor:g} effective codes the bar asks of "
            f"a tokenizer: a codebook of {num_embeddings} codes cannot reach a perplexity above {num_embeddings}, so the run could "
            f"never meet it; expected at least {floor:g}, or set debug_args.smoke=true for a run with no bar"
        )


# ---------------------------------------------------------------------
# The metric group of the task config
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TokenizerMetricArgs(StrictConfig):
    """The metric that selects the checkpoint; read by the beit_tokenizer task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch: val_reconstruction_loss on the fixed selection split, the only value accepted "
            "(the bar is about its minimum, and the constant image's score has no codebook terms, which val_loss carries); auto is "
            "val_reconstruction_loss"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO,
        metadata={"help": "min, the only value accepted (a reconstruction loss: the best epoch is its minimum); auto is min"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a ``watch_metric`` other than ``val_reconstruction_loss`` and a ``watch_mode`` other than ``min``."""
        lines: List[str] = []
        for name, only in (("watch_metric", WATCH_METRIC), ("watch_mode", WATCH_MODE)):
            value = getattr(self, name)
            if not _is_auto(value) and value != only:
                lines.append(
                    f"Invalid value for {join_path(prefix, name)!r}: expected auto or {only!r}, got {value!r}; the success bar is a "
                    f"statement about the minimum val_reconstruction_loss of the selection window, so the saved best checkpoint must "
                    f"be that epoch ({WATCH_METRIC!r} with mode {WATCH_MODE!r}); another metric or the other direction would save "
                    "weights the verdict does not describe"
                )
        return lines

    def resolve_auto(self, context: AutoContext) -> "TokenizerMetricArgs":
        """Fill ``watch_metric`` from the recipe and ``watch_mode`` from its name."""
        return _with_values(self, "metric_args", _watch_values(self, context, "metric_args"))
