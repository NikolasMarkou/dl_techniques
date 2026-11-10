"""The denoising arguments of ``python -m train``: the data group and what is read from it.

``DenoisingData`` names the image folders a denoiser trains and validates on, how many files of them
it uses, the noise it adds and the curriculum of that noise. A denoiser has no labels: the clean image
is the target, so there is no class, mask or test split here; the noise is added by the pipeline
(``train.common.denoising_data``), on the fly, from a seed. Every option that cannot apply is REFUSED, never
ignored.

This module is the first of the denoising task's argument modules; the transform, metric and loss groups
and the task config that holds them are added with the task. The group is not yet a field of any
task config.

Interface contract
------------------
``DenoisingData``
    Frozen :class:`~train.api.config.StrictConfig`. ``train_dirs`` and ``val_dirs`` are required: lists of
    directories searched recursively for images. Every noise and curriculum key lives here (the noise
    defines what the model sees, so there is one home for it; ``noise_clip`` is the one that is a decision
    the user may overrule). ``format`` is always ``"folder"``: the shared run sequence reads it. A relative path is
    relative to the directory of the YAML file that holds it, or to the working directory for a mapping or
    the command line. ``problems(prefix)`` names every option that is out of range or contradicts another.

``load_denoising_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one.
    :return: the validated config.
    :raises ConfigError: for a file that cannot be read, or any invalid key (every problem at once; a file's
        lines name the file).

``read_denoising_data_file(path)``
    :return: the mapping in a YAML file with its relative ``train_dirs`` and ``val_dirs`` entries made
        absolute against the file.

``denoising_facts(data)``
    :return: the :class:`~train.api.data_args.DataFacts` (dataset ``"folder"``, no classes, no image size).
        Every directory is LISTED (no pixel is read, no TensorFlow op runs), so it runs before ``setup_gpu``
        and before the run directory is claimed.
    :raises ConfigError: naming ``data.train_dirs`` or ``data.val_dirs`` and the directory, for one that does
        not exist, holds no image, cannot be listed, or shares a file with the other list.

``load_denoising(data, *, patch_size, channels, patches_per_image, batch_size, seed, augment, eval_batch_size)``
    :return: the :class:`~train.common.denoising_data.DenoisingBatches` of the shared loader. Every file the run
        uses is decoded once before ``fit``.
    :raises ConfigError: as :func:`denoising_facts`, and for an undecodable file (named) or a batch larger
        than an epoch's patches.

``curriculum_callback(data, sigma_max_var)``
    :param sigma_max_var: the ``DenoisingBatches.sigma_max_var`` the pipeline reads.
    :return: the :class:`~dl_techniques.callbacks.noise_sigma_curriculum.NoiseSigmaCurriculumCallback`
        that assigns it at the start of every epoch from ``noise_sigma_max_start`` to ``noise_sigma_max``
        over ``curriculum_epochs`` with ``curriculum_schedule``.
    :raises ConfigError: while ``curriculum_epochs`` is still ``"auto"`` (the task resolves it from the epochs).

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); the data module and its
TensorFlow import are inside the functions.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

from .config import AUTO, Auto, StrictConfig, join_path, validate
from .data_args import DataFacts, read_data_file
from .errors import ConfigError, ConfigValidationError

# ---------------------------------------------------------------------

#: The noise kinds; ``test_common_denoising_data`` holds it equal to ``train.common.denoising_data.NOISE_KINDS``.
NOISE_KINDS: Tuple[str, ...] = ("additive", "multiplicative", "composite")

#: The curriculum schedules; ``test_common_denoising_data`` holds it equal to the callback's ``SCHEDULES``.
CURRICULUM_SCHEDULES: Tuple[str, ...] = ("linear", "cosine", "exp")

#: Keys of a denoising data config that name directories; a YAML file's relative ones are made absolute
#: against the file.
DENOISING_PATH_KEYS = ("train_dirs", "val_dirs")

_NoiseKind = Literal[NOISE_KINDS]  # type: ignore[valid-type]
_Schedule = Literal[CURRICULUM_SCHEDULES]  # type: ignore[valid-type]

_FINGERPRINT_HELP = (
    "what the run records to tell, at a resume, that the files are the same: 'listing' is every used file's "
    "directory number, relative path and size, in the order they are listed (no file is opened); 'content' also "
    "hashes the bytes of every used file, so a swapped or edited file of the same size is caught, at the cost "
    "of reading the data once (the seconds are logged)"
)


def _real(path: str) -> str:
    """The resolved absolute path of ``path`` (symlinks followed where they exist); no file is opened."""
    return os.path.realpath(path)


def _nested(first: str, second: str) -> bool:
    """Whether two resolved paths are the same directory or one holds the other."""
    return os.path.commonpath([first, second]) in (first, second)


@dataclasses.dataclass(frozen=True)
class DenoisingData(StrictConfig):
    """Image folders for a denoiser, its noise and the curriculum of that noise."""

    # DECISION plan-2026-11-10T095307-d483ffab/D-016
    # Every noise and curriculum key lives HERE, under `data.*`, with the directory keys and the caps: noise
    # defines what the model sees, so resume's data check compares it with the files, and a separate `noise_args`
    # group would be a home for one concept in two places (and a new group in every guard). Do NOT move the
    # `noise_*` or `curriculum_*` keys into a group of their own.

    train_dirs: List[str] = dataclasses.field(
        metadata={
            "help": "directories of the train images (png, jpg, jpeg, bmp), searched recursively; with "
            "max_train_files every directory weighs the same (a small one repeats its files), so COCO "
            "does not drown DIV2K"
        }
    )
    val_dirs: List[str] = dataclasses.field(
        metadata={
            "help": "directories of the validation images; fixed crops with fixed noise, so val_psnr is a "
            "function of the weights alone; a file may not also be in train_dirs"
        }
    )
    format: Literal["folder"] = dataclasses.field(
        default="folder", metadata={"help": "where the data comes from; the only value"}
    )
    max_train_files: Optional[int] = dataclasses.field(
        default=10000,
        metadata={
            "help": "files used per epoch, split equally over train_dirs (the first directories get the "
            "remainder); a directory with fewer files repeats them; at least the number of train_dirs; null "
            "uses every file once and the directories weigh by size"
        },
    )
    max_val_files: Optional[int] = dataclasses.field(
        default=500,
        metadata={"help": "validation files used, a subset chosen by val_seed alone; null uses all"},
    )
    max_samples: Optional[int] = dataclasses.field(
        default=None,
        metadata={
            "help": "cap on the train files and on the validation files, applied after the two caps above "
            "(the smoke run sets it); null is no cap"
        },
    )
    fingerprint: Literal["listing", "content"] = dataclasses.field(
        default="listing", metadata={"help": _FINGERPRINT_HELP}
    )
    noise_kind: _NoiseKind = dataclasses.field(
        default="additive",
        metadata={
            "help": "additive: y = x + n s; multiplicative: y = x (1 + n s); composite: y = x (1 + n s) + n' r s "
            "(r is noise_composite_ratio); n and n' are standard normal and s is the sigma of the patch, on the "
            "0-1 scale of the pixels (sigma_255 = 255 s). Only additive noise keeps the residual a score"
        },
    )
    noise_sigma_min: float = dataclasses.field(
        default=0.0, metadata={"help": "lowest sigma of a patch; at least 0 and below noise_sigma_max"}
    )
    noise_sigma_max: float = dataclasses.field(
        default=0.25,
        metadata={
            "help": "the curriculum's last upper bound of a patch's sigma, and the upper bound of the validation "
            "sigma; above noise_sigma_min"
        },
    )
    noise_sigma_max_start: float = dataclasses.field(
        default=0.025,
        metadata={
            "help": "upper bound of a patch's sigma at epoch 0; not below noise_sigma_min; above 0 for the exp "
            "schedule"
        },
    )
    noise_composite_ratio: Optional[float] = dataclasses.field(
        default=None,
        metadata={
            "help": "composite noise only: the ratio r of the additive part to the multiplicative one; null is "
            "0.5, the legacy ratio; refused for any other noise_kind"
        },
    )
    # DECISION plan-2026-11-10T095307-d483ffab/D-009
    # `noise_clip` defaults to FALSE. The legacy trainer clipped the noisy input to [0, 1] always; the clip makes
    # the noise non-Gaussian near 0 and 1, so the Tweedie identity a bias-free denoiser is used for holds only
    # unclipped, and the output is never clipped except when a PNG is written. Do NOT flip the default to
    # reproduce legacy numbers: the parity arm sets `data.noise_clip=true` by name, and
    # `eval_per_pixel_uncertainty` (which always clips) disagrees with an unclipped run, a stated limit.
    noise_clip: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "clip the noisy input to [0, 1] (the legacy behaviour; breaks the Gaussian noise near 0 and 1 "
            "and so the Miyasawa/Tweedie reading of the residual); false by default"
        },
    )
    curriculum_schedule: _Schedule = dataclasses.field(
        default="linear",
        metadata={"help": "how the sigma upper bound goes from noise_sigma_max_start to noise_sigma_max: linear, cosine or exp"},
    )
    curriculum_epochs: Auto[int] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "epochs the upper bound takes to reach noise_sigma_max (1 or more, at most epochs); auto is "
            "round(0.8 x epochs), so the last fifth of a run trains at the full range"
        },
    )
    val_seed: int = dataclasses.field(
        default=0,
        metadata={
            "help": "seed of the validation crops, sigmas and noise, and of the choice of validation files; not "
            "tied to seed, so two runs of different seeds validate on the same patches"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name every option out of range and every contradiction between options, at its dotted path."""
        lines: List[str] = []

        def bad(name: str, why: str) -> None:
            lines.append(f"Invalid value for {join_path(prefix, name)!r}: {why}, got {getattr(self, name)!r}")

        for name in ("train_dirs", "val_dirs"):
            directories = getattr(self, name)
            if not directories:
                bad(name, "expected at least one directory")
            elif any(not entry.strip() for entry in directories):
                bad(name, "expected non-empty directory names")
            else:
                real = [_real(entry) for entry in directories]
                if len(set(real)) != len(real):
                    bad(name, "expected distinct directories (one listed twice would weigh double)")
        if self.train_dirs and self.val_dirs and all(entry.strip() for entry in self.train_dirs + self.val_dirs):
            for trained in self.train_dirs:
                for validated in self.val_dirs:
                    if _nested(_real(trained), _real(validated)):
                        lines.append(
                            f"Invalid value for {join_path(prefix, 'val_dirs')!r}: {validated!r} is the same "
                            f"directory as, or lies inside or around, train directory {trained!r}; the "
                            "listing is recursive, so a validation image the model trained on says nothing "
                            "about it")
        for name in ("max_train_files", "max_val_files", "max_samples"):
            value = getattr(self, name)
            if value is not None and value < 1:
                bad(name, "expected a positive int or null")
        for name in ("max_train_files", "max_samples"):
            value = getattr(self, name)
            if value is not None and self.train_dirs and 1 <= value < len(self.train_dirs):
                bad(name, f"expected at least the {len(self.train_dirs)} train_dirs (each needs a file), or null")
        for name in ("noise_sigma_min", "noise_sigma_max", "noise_sigma_max_start"):
            if getattr(self, name) < 0:
                bad(name, "expected a number of at least 0")
        if self.noise_sigma_max <= self.noise_sigma_min:
            lines.append(
                f"Invalid value for {join_path(prefix, 'noise_sigma_max')!r}: expected a value above "
                f"noise_sigma_min ({self.noise_sigma_min!r}), got {self.noise_sigma_max!r}")
        if self.noise_sigma_min > self.noise_sigma_max_start:
            lines.append(
                f"Invalid value for {join_path(prefix, 'noise_sigma_min')!r}: expected a value not above "
                f"noise_sigma_max_start ({self.noise_sigma_max_start!r}), got {self.noise_sigma_min!r}")
        if self.curriculum_schedule == "exp" and self.noise_sigma_max_start <= 0:
            lines.append(
                f"Invalid value for {join_path(prefix, 'noise_sigma_max_start')!r}: the exp schedule goes "
                f"from the start to the end by a constant ratio and cannot leave 0; expected a value above 0 "
                f"or another curriculum_schedule, got {self.noise_sigma_max_start!r}")
        ratio = self.noise_composite_ratio
        if ratio is not None:
            if self.noise_kind != "composite":
                lines.append(
                    f"Invalid value for {join_path(prefix, 'noise_composite_ratio')!r}: it is the ratio of the "
                    f"additive part of COMPOSITE noise and noise_kind is {self.noise_kind!r}; leave it null or "
                    f"set {join_path(prefix, 'noise_kind')}=composite")
            elif ratio < 0:
                bad("noise_composite_ratio", "expected a number of at least 0")
        epochs = self.curriculum_epochs
        if epochs != AUTO and (not isinstance(epochs, int) or epochs < 1):
            bad("curriculum_epochs", "expected auto or an int of at least 1")
        if self.val_seed < 0:
            bad("val_seed", "expected an int of at least 0")
        return lines


# ---------------------------------------------------------------------
# The config from a mapping or a file
# ---------------------------------------------------------------------

def read_denoising_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``, its relative directory entries anchored to the file."""
    mapping = read_data_file(path, ())
    base = Path(path).resolve().parent
    for key in DENOISING_PATH_KEYS:
        value = mapping.get(key)
        if isinstance(value, list):
            mapping[key] = [
                os.path.normpath(base / entry) if isinstance(entry, str) and entry and not os.path.isabs(entry)
                else entry
                for entry in value
            ]
    return mapping


def load_denoising_data_config(data: Any) -> DenoisingData:
    """Return the data config for a mapping, a YAML path or an instance; see the module contract."""
    if isinstance(data, DenoisingData):
        return data
    source = ""
    if isinstance(data, (str, Path)):
        source = f"data file {str(data)!r}: "
        data = read_denoising_data_file(data)
    if not isinstance(data, Mapping):
        raise ConfigError(
            "Invalid value for 'data': expected a mapping or the path of a YAML file, "
            f"got {type(data).__name__}")
    try:
        return validate(DenoisingData, dict(data))
    except ConfigValidationError as exc:
        if not source:
            raise
        raise ConfigValidationError([source + line for line in exc.errors]) from exc


# ---------------------------------------------------------------------
# What is known before the data is loaded, and the loader
# ---------------------------------------------------------------------

def denoising_facts(data: DenoisingData) -> DataFacts:
    """List the folders (no pixel read) and return their facts; see the module contract."""
    from train.common.classification_data import DataSourceError
    from train.common.denoising_data import list_sources

    try:
        list_sources(data.train_dirs, data.val_dirs)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc
    return DataFacts(dataset=data.format, num_classes=None, image_size=None)


def noise_spec(data: DenoisingData) -> Any:
    """Return the :class:`~train.common.denoising_data.NoiseSpec` of a data config."""
    from train.common.denoising_data import DEFAULT_COMPOSITE_RATIO, NoiseSpec

    ratio = DEFAULT_COMPOSITE_RATIO if data.noise_composite_ratio is None else data.noise_composite_ratio
    return NoiseSpec(
        kind=data.noise_kind, sigma_min=data.noise_sigma_min, sigma_max=data.noise_sigma_max,
        sigma_max_start=data.noise_sigma_max_start, composite_ratio=ratio, clip=data.noise_clip)


def load_denoising(
    data: DenoisingData,
    *,
    patch_size: int,
    channels: int,
    patches_per_image: int,
    batch_size: int,
    seed: int,
    augment: Any = None,
    eval_batch_size: Optional[int] = None,
) -> Any:
    """Load the folders through the shared denoising loader; see the module contract."""
    from train.common.classification_data import DataSourceError
    from train.common.denoising_data import load_denoising_data

    try:
        return load_denoising_data(
            data.train_dirs, data.val_dirs, patch_size=patch_size, channels=channels,
            patches_per_image=patches_per_image, batch_size=batch_size, seed=seed, noise=noise_spec(data),
            val_seed=data.val_seed, max_train_files=data.max_train_files, max_val_files=data.max_val_files,
            max_samples=data.max_samples, fingerprint=data.fingerprint, augment=augment,
            eval_batch_size=eval_batch_size)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc


def curriculum_callback(data: DenoisingData, sigma_max_var: Any) -> Any:
    """Return the callback that assigns ``sigma_max_var`` every epoch; see the module contract."""
    from dl_techniques.callbacks.noise_sigma_curriculum import NoiseSigmaCurriculumCallback

    if data.curriculum_epochs == AUTO:
        raise ConfigError(
            "Invalid value for 'data.curriculum_epochs': still 'auto'; the task resolves it from the epochs "
            "before it builds the curriculum")
    return NoiseSigmaCurriculumCallback(
        sigma_max_var=sigma_max_var, sigma_max_start=data.noise_sigma_max_start,
        sigma_max_end=data.noise_sigma_max, total_epochs=data.curriculum_epochs,
        schedule=data.curriculum_schedule)
