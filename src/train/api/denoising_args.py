"""The denoising arguments of ``python -m train``: the data group, the transform, metric, loss and optimizer groups.

``DenoisingData`` names the image folders a denoiser trains and validates on, how many files of them
it uses, the noise it adds and the curriculum of that noise. A denoiser has no labels: the clean image
is the target, so there is no class, mask or test split here; the noise is added by the pipeline
(``train.common.denoising_data``), on the fly, from a seed. Every option that cannot apply is REFUSED, never
ignored. ``DenoiseTransformArgs``, ``DenoiseMetricArgs``, ``DenoiseLossArgs`` and ``DenoiseOptimArgs`` are the
groups of the ``image_denoising`` task config (``train.api.tasks.image_denoising``) that differ from the shared
ones.

Interface contract
------------------
``DenoisingData``
    Frozen :class:`~train.api.config.StrictConfig`. ``train_dirs`` and ``val_dirs`` are required: lists of
    directories searched recursively for images. Every noise and curriculum key lives here (the noise
    defines what the model sees, so there is one home for it; ``noise_clip`` is the one that is a decision
    the user may overrule). ``format`` is always ``"folder"``: the shared run sequence reads it. A relative path is
    relative to the directory of the YAML file that holds it, or to the working directory for a mapping or
    the command line. ``problems(prefix)`` names every option that is out of range or contradicts another;
    ``resolve_auto(context)`` fills ``curriculum_epochs`` ``"auto"`` with :func:`auto_curriculum_epochs` of the run's
    epochs (``AutoContext.epochs``).

``load_denoising_data_config(data)``
    :param data: a config (returned as is), a mapping, or the path of a YAML file holding one.
    :return: the validated config.
    :raises ConfigError: for a file that cannot be read, or any invalid key (every problem at once; a file's
        lines name the file).

``read_denoising_data_file(path)``
    :return: the mapping in a YAML file with its relative ``train_dirs`` and ``val_dirs`` entries made
        absolute against the file.

``split_test_dir(entry)`` / ``test_named_dirs(data)``
    A ``data.test_dirs`` entry is a directory (named by its last component) or ``name=directory`` (the name is what
    the summary's ``held_out`` is keyed by; ``CBSD68/original`` would otherwise be "original"). ``test_named_dirs``
    is ``[(name, directory), ...]`` of a data config, ``[]`` without ``test_dirs``.

``denoising_facts(data)``
    :return: the :class:`~train.api.data_args.DataFacts` (dataset ``"folder"``, no classes, no image size).
        Every directory is LISTED (no pixel is read, no TensorFlow op runs), so it runs before ``setup_gpu``
        and before the run directory is claimed.
    :raises ConfigError: naming ``data.train_dirs``, ``data.val_dirs`` or ``data.test_dirs`` and the directory, for
        one that does not exist, holds no image, cannot be listed, or shares a file with another list (a held-out
        image that is also a train or validation image is refused).

``held_out_sets(data, channels=None)``
    :param channels: when given, every held-out file is decoded once (as a channel count) so an unreadable one is named
        before ``fit``; ``None`` only lists (no TensorFlow op).
    :return: ``{name: image files}`` of ``data.test_dirs`` (``{}`` without it).
    :raises ConfigError: as :func:`denoising_facts` for a directory that cannot be listed, and naming the file for one
        that cannot be decoded.

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

``DenoiseTransformArgs``
    ``patch_size`` (the side of the square crops the model trains and validates on, at least 16: SSIM's window is 11),
    ``patches_per_image`` (crops cut from each decoded image), ``augment`` (``none``, ``flip`` or ``dihedral``; auto is
    the recipe's, which for folder data is the flip of ``registry.FILE_DATA_OVERLAY``) and ``channels`` (1 or 3).

``DenoiseMetricArgs``
    ``watch_metric`` (auto is the recipe's ``val_psnr``), ``watch_mode`` (auto is read from the name),
    ``success_gain_db`` (auto is :data:`SUCCESS_GAIN_DB`: the run succeeds when its best ``val_psnr`` inside the
    selection window beats the PSNR of the noisy input by this many dB) and ``test_sigmas_255`` (the noise levels,
    on the 0-255 scale, of the held-out cells).

``DenoiseLossArgs``
    ``name``: ``mse``, ``mae`` or ``charbonnier``. A bias-free family accepts only ``mse`` (the task refuses the
    others before the run directory is claimed): an L1-type loss estimates something other than the conditional
    mean, which is what the bias-free denoiser's residual-as-score reading needs.

``DenoiseOptimArgs``
    :class:`~train.api.args.OptimArgs` whose ``warmup_epochs`` ``"auto"`` is :func:`auto_warmup_epochs` of the run's
    epochs (the legacy rule) instead of the recipe's fixed number.

``auto_curriculum_epochs(epochs)`` / ``auto_warmup_epochs(epochs)``
    ``max(1, round(0.8 * epochs))`` and ``max(1, round(0.1 * epochs))``: what ``data.curriculum_epochs`` and
    ``optim_args.warmup_epochs`` resolve to when they are ``"auto"``.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012); the data module and its
TensorFlow import are inside the functions.
"""

import dataclasses
import os
import re
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

from .args import AutoContext, OptimArgs, _is_auto, _recipe_values, _with_values
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
DENOISING_PATH_KEYS = ("train_dirs", "val_dirs", "test_dirs")

#: The keys of ``results_summary.json``'s ``held_out`` block of a run that scored nothing (``status``, ``reason``,
#: ``sigmas_255``): a held-out set may not be called one of them, so a reader can tell the two shapes apart.
RESERVED_HELD_OUT_NAMES = ("status", "reason", "sigmas_255")

#: What a held-out set's name may be: it is a JSON key and a figure panel title.
_TEST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: The training-patch augmentations; ``test_common_denoising_data`` holds it equal to ``train.common.denoising_data.PATCH_AUGMENTS``.
PATCH_AUGMENTS: Tuple[str, ...] = ("none", "flip", "dihedral")

#: The losses of ``loss_args.name``; ``test_image_denoising_task`` holds it equal to ``train.common.denoising_run.LOSS_NAMES``.
LOSS_NAMES: Tuple[str, ...] = ("mse", "mae", "charbonnier")

#: The smallest ``patch_size``: SSIM's window is 11 pixels, so a smaller patch has no SSIM to report.
MIN_PATCH_SIZE = 16

#: The channel counts a denoiser reads and writes: grey or RGB.
CHANNEL_COUNTS = (1, 3)

#: ``metric_args.test_sigmas_255`` when the caller gives none: the three levels every denoising paper reports.
DEFAULT_TEST_SIGMAS_255: Tuple[float, ...] = (15.0, 25.0, 50.0)

# DECISION plan-2026-11-10T095307-d483ffab/D-013
# `metric_args.success_gain_db` auto is 1.0 dB: an INFERRED number, not a literature value. A trained denoiser
# beats its noisy input by 5 to 15 dB over the range U(0, 0.25) (the legacy run: 23.6 dB after one epoch against 20.7
# dB of input), so 1 dB is a bar a model that learned anything clears with room and one that learned nothing (an
# untrained U-Net sits well below its input) cannot. Do NOT make it an absolute PSNR: the input PSNR moves with the
# sigma range, so a fixed number would pass or fail the same model by the noise it was given.
SUCCESS_GAIN_DB = 1.0

#: ``data.curriculum_epochs`` auto is this fraction of the epochs (D-013): the last fifth trains at the full range.
CURRICULUM_FRACTION = 0.8

#: ``optim_args.warmup_epochs`` auto is this fraction of the epochs (the legacy ``max(1, round(0.1 * epochs))``).
WARMUP_FRACTION = 0.1

_NoiseKind = Literal[NOISE_KINDS]  # type: ignore[valid-type]
_Schedule = Literal[CURRICULUM_SCHEDULES]  # type: ignore[valid-type]
_Augment = Literal[PATCH_AUGMENTS]  # type: ignore[valid-type]
_LossName = Literal[LOSS_NAMES]  # type: ignore[valid-type]


def auto_curriculum_epochs(epochs: int) -> int:
    """Return what ``data.curriculum_epochs`` ``"auto"`` is for a run of ``epochs``: ``max(1, round(0.8 * epochs))``."""
    return max(1, round(CURRICULUM_FRACTION * epochs))


def auto_warmup_epochs(epochs: int) -> int:
    """Return what ``optim_args.warmup_epochs`` ``"auto"`` is for a run of ``epochs``: ``max(1, round(0.1 * epochs))``."""
    return max(1, round(WARMUP_FRACTION * epochs))

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


def _explicit_name(entry: str) -> Optional[Tuple[str, str]]:
    """``(name, directory)`` of an entry written ``name=directory``, else ``None``.

    The name is taken only when it holds no path separator, so a directory with an ``=`` in a parent's name stays bare.
    """
    name, separator, directory = entry.partition("=")
    if separator and name and "/" not in name and os.sep not in name:
        return name, directory
    return None


def split_test_dir(entry: str) -> Tuple[str, str]:
    """``(name, directory)`` of a ``data.test_dirs`` entry: ``name=directory``, or a bare directory named by its last part."""
    return _explicit_name(entry) or (os.path.basename(os.path.normpath(entry)), entry)


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
    test_dirs: Optional[List[str]] = dataclasses.field(
        default=None,
        metadata={
            "help": "held-out image sets, scored on WHOLE images at the end of the run with the run's noise at "
            "metric_args.test_sigmas_255, for the best and the final model: each entry is a directory (named by its "
            "last component) or name=directory; a file may not also be in train_dirs or val_dirs; null scores nothing"
        },
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
        lines.extend(self._test_dir_problems(prefix))
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

    def _test_dir_problems(self, prefix: str) -> List[str]:
        """Name a ``test_dirs`` that is empty, badly named, repeated, or the same directory as, or inside, another one."""
        entries = self.test_dirs
        if entries is None:
            return []
        key = join_path(prefix, "test_dirs")
        if not entries:
            return [f"Invalid value for {key!r}: expected at least one directory or null, got []"]
        if any(not entry.strip() for entry in entries):
            return [f"Invalid value for {key!r}: expected non-empty entries, got {entries!r}"]
        lines: List[str] = []
        named = [split_test_dir(entry) for entry in entries]
        for name, directory in named:
            if not _TEST_NAME.match(name) or name in RESERVED_HELD_OUT_NAMES:
                lines.append(
                    f"Invalid value for {key!r}: the held-out set name {name!r} (entry {directory!r}) must start with a "
                    "letter or digit, hold only letters, digits, '_', '.' and '-', and not be one of "
                    f"{list(RESERVED_HELD_OUT_NAMES)}; give it as name=directory")
            if not directory.strip():
                lines.append(f"Invalid value for {key!r}: the entry for {name!r} names no directory")
        names = [name for name, _ in named]
        for name in sorted({name for name in names if names.count(name) > 1}):
            lines.append(
                f"Invalid value for {key!r}: two held-out sets are named {name!r}; the summary is keyed by name, so "
                "give each as name=directory")
        if lines:
            return lines
        real = [(name, _real(directory)) for name, directory in named]
        for index, (name, path) in enumerate(real):
            for other, other_path in real[index + 1:]:
                if _nested(path, other_path):
                    lines.append(
                        f"Invalid value for {key!r}: {name!r} and {other!r} are the same directory or one lies inside "
                        "the other; the listing is recursive, so their images would be scored twice")
            for role in ("train_dirs", "val_dirs"):
                for used in getattr(self, role):
                    if used.strip() and _nested(path, _real(used)):
                        lines.append(
                            f"Invalid value for {key!r}: {name!r} is the same directory as, or lies inside or around, "
                            f"{join_path(prefix, role)} entry {used!r}; the listing is recursive, so a held-out image "
                            "the model trained or selected on says nothing about it")
        return lines

    def resolve_auto(self, context: AutoContext) -> "DenoisingData":
        """Fill ``curriculum_epochs`` ``"auto"`` with :func:`auto_curriculum_epochs` of the run's epochs."""
        if not _is_auto(self.curriculum_epochs):
            return self
        epochs = context.epochs
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ConfigError(
                f"Invalid value for 'data.curriculum_epochs': auto is {CURRICULUM_FRACTION:g} x the epochs, and the "
                f"run's epochs are {epochs!r}; set data.curriculum_epochs or epochs")
        return dataclasses.replace(self, curriculum_epochs=auto_curriculum_epochs(epochs))


# ---------------------------------------------------------------------
# The config from a mapping or a file
# ---------------------------------------------------------------------

def read_denoising_data_file(path: Union[str, Path]) -> dict:
    """Return the mapping in the YAML file ``path``, its relative directory entries anchored to the file."""
    mapping = read_data_file(path, ())
    base = Path(path).resolve().parent
    def anchored(key: str, entry: Any) -> Any:
        if not isinstance(entry, str) or not entry:
            return entry
        explicit = _explicit_name(entry) if key == "test_dirs" else None
        if explicit is not None:
            # `name=directory`: only the directory is relative to the file.
            name, directory = explicit
            return f"{name}={directory if os.path.isabs(directory) else os.path.normpath(base / directory)}"
        return entry if os.path.isabs(entry) else os.path.normpath(base / entry)

    for key in DENOISING_PATH_KEYS:
        value = mapping.get(key)
        if isinstance(value, list):
            mapping[key] = [anchored(key, entry) for entry in value]
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

    from train.common.denoising_data import list_test_sets

    try:
        train, val = list_sources(data.train_dirs, data.val_dirs)
        list_test_sets(test_named_dirs(data), train, val)
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc
    return DataFacts(dataset=data.format, num_classes=None, image_size=None)


def test_named_dirs(data: DenoisingData) -> List[Tuple[str, str]]:
    """Return ``[(name, directory), ...]`` of ``data.test_dirs`` (empty without it); see the module contract."""
    return [split_test_dir(entry) for entry in (data.test_dirs or [])]


def held_out_sets(data: DenoisingData, *, channels: Optional[int] = None) -> dict:
    """List (and, with ``channels``, decode once) the held-out sets of a data config; see the module contract."""
    from train.common.classification_data import DataSourceError
    from train.common.denoising_data import list_test_sets, scan_images

    try:
        sets = list_test_sets(test_named_dirs(data))
        if channels is not None and sets:
            scan_images([path for files in sets.values() for path in files], channels=channels, key="test_dirs")
        return sets
    except DataSourceError as exc:
        raise ConfigError(f"Invalid value for {join_path('data', exc.key)!r}: {exc}") from exc


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


# ---------------------------------------------------------------------
# The groups of the task config
# ---------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DenoiseTransformArgs(StrictConfig):
    """The crops a denoiser trains on and how they are augmented; read by the denoising task."""

    patch_size: int = dataclasses.field(
        default=256,
        metadata={
            "help": f"side of the square crops (at least {MIN_PATCH_SIZE}: SSIM's window is 11 pixels); the training "
            "crops, the fixed validation crops and the model's training input are this size; a U-Net family needs a "
            "multiple of 2**depth (8 for tiny and small, 16 for base and large, 32 for xlarge), which is refused "
            "before the run directory is claimed; an image with a side below it is upscaled, as the legacy trainers do"
        },
    )
    patches_per_image: int = dataclasses.field(
        default=4,
        metadata={
            "help": "crops cut from each decoded image per epoch (and fixed crops per validation image); an epoch "
            "is the used train files x this many patches"
        },
    )
    augment: Auto[_Augment] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "none, flip (horizontal) or dihedral (flips and quarter turns), applied to the clean training "
            "crop before the noise is added; validation is never augmented; auto is the family recipe's, a flip "
            "for folder data"
        },
    )
    channels: int = dataclasses.field(
        default=3,
        metadata={"help": "1 (grey) or 3 (RGB): every image is decoded to this many channels and the model reads and writes them"},
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a ``patch_size`` below the SSIM window, a count below 1 and a channel count other than 1 or 3."""
        lines: List[str] = []
        if self.patch_size < MIN_PATCH_SIZE:
            lines.append(
                f"Invalid value for {join_path(prefix, 'patch_size')!r}: expected an int of at least {MIN_PATCH_SIZE} "
                f"(SSIM's window is 11 pixels), got {self.patch_size!r}")
        if self.patches_per_image < 1:
            lines.append(
                f"Invalid value for {join_path(prefix, 'patches_per_image')!r}: expected an int of at least 1, "
                f"got {self.patches_per_image!r}")
        if self.channels not in CHANNEL_COUNTS:
            lines.append(
                f"Invalid value for {join_path(prefix, 'channels')!r}: expected one of {list(CHANNEL_COUNTS)}, "
                f"got {self.channels!r}")
        return lines

    def resolve_auto(self, context: AutoContext) -> "DenoiseTransformArgs":
        """Fill ``augment`` from the recipe; a recipe that says nothing is ``none``."""
        found = _recipe_values(self, context, "transform_args")
        if _is_auto(self.augment):
            found.setdefault("augment", "none")
        return _with_values(self, "transform_args", found)


@dataclasses.dataclass(frozen=True)
class DenoiseMetricArgs(StrictConfig):
    """The metric that selects the checkpoint, the success bar and the held-out noise levels; read by the denoising task."""

    watch_metric: Auto[str] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": "metric that picks the best epoch among those from the end of the curriculum on (the window starts "
            "there whatever the metric): val_psnr (the default recipe), val_ssim, val_mae or val_loss on the fixed "
            "validation set; the training names loss, psnr, mae and ssim are accepted too but are averaged over the "
            "epoch's fresh noise draws, so they are noisier than the validation ones"
        },
    )
    watch_mode: Auto[Literal["min", "max"]] = dataclasses.field(
        default=AUTO, metadata={"help": "min or max; auto is read from the metric's name"}
    )
    success_gain_db: Auto[float] = dataclasses.field(
        default=AUTO,
        metadata={
            "help": f"the run succeeds when its best val_psnr from the end of the curriculum on beats the PSNR of the "
            f"noisy validation set by at least this many dB; auto is {SUCCESS_GAIN_DB:g} dB, an inferred number, "
            "not a literature value: a model that learned anything clears it and an untrained one does not"
        },
    )
    test_sigmas_255: List[float] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_TEST_SIGMAS_255),
        metadata={
            "help": "noise levels on the 0-255 scale (sigma_255 = 255 x sigma) of the held-out cells, each scored for "
            "the best and the final model; 15, 25 and 50 are the levels denoising papers report"
        },
    )

    def problems(self, prefix: str) -> List[str]:
        """Name a negative ``success_gain_db`` and a ``test_sigmas_255`` that is empty, repeats or leaves (0, 255]."""
        lines: List[str] = []
        gain = self.success_gain_db
        if not _is_auto(gain) and gain < 0:
            lines.append(
                f"Invalid value for {join_path(prefix, 'success_gain_db')!r}: expected auto or a number of at "
                f"least 0, got {gain!r}")
        sigmas = self.test_sigmas_255
        if not sigmas or len(set(sigmas)) != len(sigmas) or any(not 0 < value <= 255 for value in sigmas):
            lines.append(
                f"Invalid value for {join_path(prefix, 'test_sigmas_255')!r}: expected a non-empty list of distinct "
                f"numbers in (0, 255], got {sigmas!r}")
        return lines

    def resolve_auto(self, context: AutoContext) -> "DenoiseMetricArgs":
        """Fill ``watch_metric`` from the recipe, ``watch_mode`` from its name and ``success_gain_db`` with :data:`SUCCESS_GAIN_DB`."""
        from train.common.callbacks import resolve_monitor_mode

        found = _recipe_values(self, context, "metric_args", skip=("watch_mode", "success_gain_db"))
        metric = found.get("watch_metric", self.watch_metric)
        if _is_auto(self.watch_mode) and not _is_auto(metric):
            found["watch_mode"] = resolve_monitor_mode(metric)
        if _is_auto(self.success_gain_db):
            found["success_gain_db"] = SUCCESS_GAIN_DB
        return _with_values(self, "metric_args", found)


@dataclasses.dataclass(frozen=True)
class DenoiseLossArgs(StrictConfig):
    """The loss of a denoising run; read by the denoising task."""

    name: _LossName = dataclasses.field(
        default="mse",
        metadata={
            "help": "mse (the mean squared error: the loss whose minimiser is the conditional mean, which a bias-free "
            "denoiser's residual-as-score reading needs), mae, or charbonnier (a smooth L1, epsilon 1e-3); mae and "
            "charbonnier are refused for a bias-free family (bfcnn, bfunet, bfconvunext)"
        },
    )


@dataclasses.dataclass(frozen=True)
class DenoiseOptimArgs(OptimArgs):
    """The shared optimizer group with a warmup that follows the length of the run."""

    # DECISION plan-2026-11-10T095307-d483ffab/D-024
    # `warmup_epochs` auto is `max(1, round(0.1 * epochs))`, the legacy rule (`bfunet/common.py` `warmup_epochs`),
    # NOT the recipe's 10: the recipe's number is that rule at its 100 epochs, and a fixed 10 against a 5-epoch run
    # (the length of a smoke run or a short parity run) never leaves its warmup, so the run trains at a fraction of
    # `lr` throughout. Do NOT put the derivation in `OptimArgs.resolve_auto` (the other tasks' recipes name their own
    # numbers and the goldens pin them) and do NOT drop `warmup_epochs` from the recipe (it documents the 100-epoch
    # value and the run still logs "Recipe value not used" when the derived number differs, which is true).
    def resolve_auto(self, context: AutoContext) -> "DenoiseOptimArgs":
        """Fill the shared fields, then an ``"auto"`` warmup from the run's epochs unless the schedule is constant."""
        resolved = super().resolve_auto(context)
        if _is_auto(self.warmup_epochs) and resolved.schedule != "constant" and isinstance(context.epochs, int):
            return dataclasses.replace(resolved, warmup_epochs=auto_warmup_epochs(context.epochs))
        return resolved
