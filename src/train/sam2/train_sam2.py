"""SAM 2 video trainer: stock ``compile()``/``fit()`` over ``SAM2TrainingModel``.

Run it::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam2.train_sam2 --smoke

    # The control that isolates the VIDEO machinery from the SAM 2 machinery
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam2.train_sam2 --smoke --num-frames 1

What this trainer is, and is not
--------------------------------
It proves the multi-frame training path RUNS with live gradients: the graph
traces under stock ``fit()``, the object-score head and the mask decoder both
receive gradient, and the loss descends on synthetic clips. **It is a WIRING
result, not a segmentation result.** No official Meta SAM 2 checkpoint has ever
been loaded in this repository, no annotated video-object-segmentation dataset
exists on this machine, and this trainer therefore makes no accuracy or
convergence claim whatsoever.

``jit_compile=False`` is MANDATORY, and it has ONE home
-------------------------------------------------------
Keras 3.8's ``fit()`` defaults to ``jit_compile='auto'``, which selects XLA on a
GPU, and ``Hiera``'s stem interpolates its learned positional embedding with a
BICUBIC resize that has no XLA GPU kernel (MEASURED; ``decisions.md`` D-055).
This trainer never re-spells the setting: it compiles through
:func:`dl_techniques.models.sam2.training_model.compile_sam2_video_trainer`,
which is the single home of the three-key ``loss=`` dict, of the mandatory
object-score BCE, and of ``jit_compile=False`` (D-064).

Argument wiring, and why it is a table and not a call
-----------------------------------------------------
This repository has a RECORDED defect class: a trainer's ``main()`` lists each
config field by hand, one line is omitted, and the corresponding CLI flag
becomes a **silent no-op** -- the run completes, the artifact is wrong, and no
test notices (`plans/LESSONS.md`; the bfunet trainer shipped exactly this with
``--high-freq-blocks`` and ``--filter-multiplier``). So the argparse ``dest`` ->
config-field wiring lives in ONE table, :data:`CLI_TO_CONFIG`, and
:func:`config_from_argv` builds the config by iterating it.
``tests/test_train/test_sam2/test_train_sam2.py`` drives every parser flag with
a sentinel through the FULL ``argv -> parse -> config`` path, and then drives
every config field one layer further, to a CONSUMER.

The ``--smoke`` preset and provenance
-------------------------------------
``--smoke`` is applied in the config BUILDER through the shared
:func:`train.common.args.explicitly_set_flags`, so a flag the caller actually
typed beats the preset -- **including one typed at its own parser default**,
which a parsed-value-vs-default comparison structurally cannot express. The
preset touches only :data:`SMOKE_PRESET`'s fields, and every one of them changes
*how much* is measured (clips, batch, epochs, patience), never *what*: not
``num_frames``, not ``occlusion_frames``, not the variant, not the loss weights,
not the seed.

``tensorflow`` is imported here for ``tf.data`` and the in-process GPU memory
reading, exactly as ``train/sam/train_sam.py`` and ``train/sam2/data.py`` do.
The ``keras.ops``-only invariant binds library forward paths, not trainers.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import keras
import tensorflow as tf

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
    save_config_json,
)
# `explicitly_set_flags` is NOT re-exported from `train.common`; it lives in
# `train.common.args`, which is how every other adopter imports it.
from train.common.args import explicitly_set_flags

from dl_techniques.models.sam2.hiera import Hiera
from dl_techniques.models.sam2.model import SAM2, create_sam2
from dl_techniques.models.sam2.training_model import (
    SAM2TrainingModel,
    compile_sam2_video_trainer,
)
from dl_techniques.utils.logger import logger

# `_occlusion_window` is the SINGLE home of "which occlusion windows are
# legal" (it also owns the frame-0 refusal). It is private to this package and
# imported rather than restated, so the CLI cannot accept a window the data
# source then refuses several seconds later, and neither can drift from the
# other.
from train.sam2.data import _occlusion_window, build_sam2_video_dataset

#: The two shipped SAM 2 variants. Read from the model's own table, never
#: restated: `SAM2.MODEL_VARIANTS` is the single home.
VARIANTS: Tuple[str, ...] = tuple(sorted(SAM2.MODEL_VARIANTS))

#: `hiera_l`'s measured parameter count (iteration 1, step 8). Quoted in the
#: `--smoke` refusal so the message names the cost rather than gesturing at it.
HIERA_L_PARAMETERS = 220_941_537


def variant_image_size(variant: str) -> int:
    """Input resolution of a SAM 2 variant.

    :param variant: A key of ``SAM2.MODEL_VARIANTS``.
    :type variant: str
    :return: The variant's square input side, in pixels, READ from
        ``Hiera.MODEL_VARIANTS`` -- the single home of every trunk geometry.
        ``SAM2.from_variant`` reads the same entry, so the data pipeline and
        the model cannot disagree about the input size.
    :rtype: int
    :raises ValueError: If ``variant`` is not a shipped variant.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; known variants are {list(VARIANTS)}"
        )
    return int(Hiera.MODEL_VARIANTS[variant]["image_size"])


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class SAM2TrainingConfig:
    """Every knob the SAM 2 video trainer reads.

    Field names are the target half of :data:`CLI_TO_CONFIG`; a field no CLI
    flag names must be listed in :data:`DERIVED_FIELDS` or the completeness
    guard fails. ``image_size`` is deliberately NOT a field: it belongs to the
    variant, and :func:`variant_image_size` reads it from the model's own
    table, so a mismatched pair is not expressible.
    """

    # Data
    num_frames: int = 4
    occlusion_frames: int = 1
    occlusion_start: Optional[int] = None
    num_clips_train: int = 256
    num_clips_val: int = 32
    num_background_points: int = 0
    include_box: bool = False

    # Model
    variant: str = "tiny"

    # Loss mixing. The focal:dice ratio lives INSIDE `SAM2GatedMaskLoss`; these
    # three weights balance the whole gated mask term, the object-score BCE and
    # the IoU regression against each other, matching upstream's
    # `loss_mask`/`loss_class`/`loss_iou` weights.
    mask_weight: float = 1.0
    object_score_weight: float = 1.0
    iou_weight: float = 1.0

    # Training
    batch_size: int = 4
    epochs: int = 20
    steps_per_epoch: Optional[int] = None
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Preset
    smoke: bool = False

    @property
    def image_size(self) -> int:
        """Square input side of the configured variant, in pixels."""
        return variant_image_size(self.variant)

    @property
    def mask_size(self) -> int:
        """Side of the square ground-truth grid.

        ``low_res_logits`` is emitted at ``feature_grid * 4 ==
        image_size // 4``, which is exactly ``train.sam.data``'s
        ``MASK_DIVISOR`` factor and therefore exactly what
        :func:`build_sam2_video_dataset` defaults to. Stated here as a property
        rather than passed around, so nothing can set it to something else.
        """
        return self.image_size // 4

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r}; known variants are "
                f"{list(VARIANTS)}"
            )
        if self.num_frames < 1:
            raise ValueError(
                f"num_frames must be >= 1 (1 degenerates to the image path, "
                f"which is the control run); got {self.num_frames}"
            )
        # ONE home for window legality, including the frame-0 refusal: the data
        # source validates through the same function, per clip.
        _occlusion_window(
            self.num_frames,
            self.occlusion_frames,
            1 if self.occlusion_frames else None,
        )
        if self.occlusion_start is not None:
            _occlusion_window(
                self.num_frames, self.occlusion_frames, self.occlusion_start
            )
        if self.num_clips_train <= 0 or self.num_clips_val <= 0:
            raise ValueError(
                f"num_clips_train ({self.num_clips_train}) and num_clips_val "
                f"({self.num_clips_val}) must both be > 0; the validation set "
                f"is what `monitor='val_loss'` reads."
            )
        if min(self.num_clips_train, self.num_clips_val) < self.batch_size:
            raise ValueError(
                f"num_clips_train ({self.num_clips_train}) and num_clips_val "
                f"({self.num_clips_val}) must both be >= batch_size "
                f"({self.batch_size}). The clip pipeline batches with "
                f"drop_remainder=True -- the batch axis must be STATIC (D-068) "
                f"-- so a split smaller than one batch yields ZERO batches, "
                f"and an empty validation set makes `monitor='val_loss'` fail "
                f"a whole epoch in."
            )
        if self.num_background_points < 0:
            raise ValueError(
                f"num_background_points must be >= 0; got "
                f"{self.num_background_points}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0; got {self.epochs}")
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0 or None (None = a full pass over "
                f"num_clips_train); got {self.steps_per_epoch}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}"
            )
        if min(self.mask_weight, self.object_score_weight,
               self.iou_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.object_score_weight == 0.0:
            raise ValueError(
                "object_score_weight must be > 0. Upstream's `loss_class` is "
                "mandatory at weight 1: every consumer of "
                "`object_score_logits` in this package thresholds it HARD at "
                "> 0, so at weight 0 the occlusion head has no differentiable "
                "consumer at all and trains not at all, with a finite, falling "
                "and meaningless loss."
            )
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be > 0; got "
                f"{self.early_stopping_patience}"
            )
        if self.smoke and self.variant != "tiny":
            raise ValueError(
                f"--smoke refuses variant {self.variant!r}: it is a "
                f"{HIERA_L_PARAMETERS:,}-parameter model at "
                f"{variant_image_size(self.variant)}px, which has never been "
                f"forward-passed in this repository and cannot be a smoke run "
                f"on a 12 GB card. Use --variant tiny, or drop --smoke and "
                f"choose the run size yourself."
            )
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"sam2_{self.variant}_{timestamp}"


#: argparse ``dest`` -> :class:`SAM2TrainingConfig` field. THE wiring, in one
#: place. Adding a flag without adding its row here fails
#: `test_every_cli_flag_is_wired_to_a_config_field`; deleting a row fails
#: `test_every_cli_flag_reaches_the_config_field_it_names`, by flag name.
CLI_TO_CONFIG: Dict[str, str] = {
    "num_frames": "num_frames",
    "occlusion_frames": "occlusion_frames",
    "occlusion_start": "occlusion_start",
    "num_clips_train": "num_clips_train",
    "num_clips_val": "num_clips_val",
    "num_background_points": "num_background_points",
    "include_box": "include_box",
    "variant": "variant",
    "mask_weight": "mask_weight",
    "object_score_weight": "object_score_weight",
    "iou_weight": "iou_weight",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "steps_per_epoch": "steps_per_epoch",
    "learning_rate": "learning_rate",
    "early_stopping_patience": "early_stopping_patience",
    "seed": "seed",
    "output_dir": "output_dir",
    "experiment_name": "experiment_name",
    "smoke": "smoke",
}

#: argparse dests that deliberately do NOT reach the config: they act on the
#: process, not on the run's parameters.
NON_CONFIG_DESTS: Set[str] = {"help", "gpu"}

#: Config fields no CLI flag names.
DERIVED_FIELDS: Set[str] = set()

#: The ``--smoke`` preset. Every entry changes HOW MUCH is measured; none
#: changes WHAT is measured -- `num_frames`, `occlusion_frames`, `variant`, the
#: three loss weights and the seed are absent ON PURPOSE, because each of them
#: redefines the quantity rather than shrinking it. Any field the caller typed
#: explicitly wins over its entry here; see :func:`parse_arguments`.
SMOKE_PRESET: Dict[str, Any] = {
    "num_clips_train": 8,
    "num_clips_val": 4,
    "batch_size": 2,
    "epochs": 3,
    "early_stopping_patience": 2,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    :return: The parser. Every option is a long (``--``) spelling plus
        argparse's own ``-h``: :func:`explicitly_set_flags` REFUSES a parser
        carrying any other short option, because it cannot see attached
        (``-b8``) or grouped (``-vb 8``) forms and would report a typed flag as
        not-typed.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Train SAM 2 on synthetic video with stock compile()/fit()",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = SAM2TrainingConfig()

    data = parser.add_argument_group("data")
    data.add_argument("--num-frames", type=int, default=defaults.num_frames,
                      help="Clip length T. The bound of an UNROLLED Python "
                           "loop, so it is static: a different T is a "
                           "different graph. 1 is the image-path control.")
    data.add_argument("--occlusion-frames", type=int,
                      default=defaults.occlusion_frames,
                      help="Consecutive fully-occluded frames per clip. Every "
                           "emitted clip has exactly this many all-zero "
                           "ground-truth frames; frame 0 is never one of them.")
    data.add_argument("--occlusion-start", type=int,
                      default=defaults.occlusion_start,
                      help="First occluded frame; default None draws it.")
    data.add_argument("--num-clips-train", type=int,
                      default=defaults.num_clips_train,
                      help="Clips in one training epoch.")
    data.add_argument("--num-clips-val", type=int,
                      default=defaults.num_clips_val)
    data.add_argument("--num-background-points", type=int,
                      default=defaults.num_background_points,
                      help="Background points added to the FRAME-0 prompt.")
    data.add_argument("--include-box", action=argparse.BooleanOptionalAction,
                      default=defaults.include_box,
                      help="Add a jittered frame-0 box prompt beside the "
                           "point.")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=list(VARIANTS),
                       default=defaults.variant,
                       help="'tiny' is a structurally faithful reduced-WIDTH "
                            "model at 64px; 'hiera_l' is SAM 2.1-L, "
                            f"{HIERA_L_PARAMETERS:,} parameters at 1024px, "
                            "never forward-passed in this repository.")

    loss = parser.add_argument_group("loss")
    loss.add_argument("--mask-weight", type=float, default=defaults.mask_weight)
    loss.add_argument("--object-score-weight", type=float,
                      default=defaults.object_score_weight,
                      help="Weight of the MANDATORY object-score BCE "
                           "(upstream's loss_class: 1). Zero is refused.")
    loss.add_argument("--iou-weight", type=float, default=defaults.iou_weight)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--steps-per-epoch", type=int,
                          default=defaults.steps_per_epoch,
                          help="Default None = one full pass over "
                               "--num-clips-train.")
    training.add_argument("--learning-rate", type=float,
                          default=defaults.learning_rate)
    training.add_argument("--early-stopping-patience", type=int,
                          default=defaults.early_stopping_patience)
    training.add_argument("--seed", type=int, default=defaults.seed)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help="Relative paths resolve against the REPO ROOT, "
                             "never the current directory, so `python -m "
                             "train.sam2.train_sam2` from anywhere writes to "
                             "<repo>/results/.")
    output.add_argument("--experiment-name", type=str,
                        default=defaults.experiment_name)
    output.add_argument("--gpu", type=int, default=None,
                        help="GPU index for setup_gpu (process-level; not a "
                             "config field).")
    output.add_argument("--smoke", action=argparse.BooleanOptionalAction,
                        default=defaults.smoke,
                        help=f"Preset: {SMOKE_PRESET}. Any flag you type "
                             f"explicitly wins over it.")
    return parser


def parse_arguments(
        argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, SAM2TrainingConfig]:
    """Run the FULL ``argv -> parse -> config`` path.

    This is the single entry point ``main`` uses and the single entry point the
    wiring guard drives, so there is no path a test can pass while the trainer
    fails.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``. The namespace is returned only so
        ``main`` can read the process-level dests in :data:`NON_CONFIG_DESTS`
        (``--gpu``); everything else must come from the config.
    :rtype: Tuple[argparse.Namespace, SAM2TrainingConfig]
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit_dests = explicitly_set_flags(parser, argv)
    explicit_fields = {
        CLI_TO_CONFIG[dest] for dest in explicit_dests if dest in CLI_TO_CONFIG
    }

    values = {
        field: getattr(args, dest) for dest, field in CLI_TO_CONFIG.items()
    }
    if values.get("smoke"):
        # DECISION plan-2026-08-04T044628-4c240b4c/D-041
        # Apply the preset HERE, in the BUILDER, gated on PROVENANCE. Do NOT
        # move it into `SAM2TrainingConfig.__post_init__`: by then the argv
        # tokens are gone, so `--smoke --epochs 20` (20 being the flag's own
        # default) is indistinguishable from a bare `--smoke`, and the preset
        # silently overrides a value the caller really typed. And do NOT add a
        # field to SMOKE_PRESET that changes WHAT is measured -- `num_frames`,
        # `occlusion_frames`, `variant`, the loss weights, the seed -- only how
        # much; `test_the_preset_changes_how_much_not_what` pins that list.
        for field, preset_value in SMOKE_PRESET.items():
            if field not in explicit_fields:
                values[field] = preset_value
    return args, SAM2TrainingConfig(**values)


def config_from_argv(
        argv: Optional[Sequence[str]] = None,
) -> SAM2TrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: A validated :class:`SAM2TrainingConfig`.
    :rtype: SAM2TrainingConfig
    """
    return parse_arguments(argv)[1]


def resolved_output_dir(config: SAM2TrainingConfig) -> Path:
    """Resolve the run directory, anchoring a relative path at the REPO ROOT.

    :param config: The run's config.
    :type config: SAM2TrainingConfig
    :return: ``<repo>/<output_dir>/<experiment_name>`` for a relative
        ``output_dir``, or ``<output_dir>/<experiment_name>`` for an absolute
        one.
    :rtype: pathlib.Path
    """
    # DECISION plan-2026-08-04T044628-4c240b4c/D-041
    # Anchor a relative path at the REPO ROOT, never at the cwd. Do NOT
    # "simplify" this to `Path(config.output_dir) / name`: the editable install
    # makes `python -m train.sam2.train_sam2` resolve from ANY working
    # directory, so the plain form writes a stray `results/` tree wherever the
    # user happened to be standing -- including `src/results/`, which the repo
    # convention names explicitly as the wrong place. Pinned by
    # `test_the_resolved_path_is_not_under_src`.
    root = Path(config.output_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    return root / str(config.experiment_name)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def create_dataset(
        config: SAM2TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build the training and validation clip datasets.

    :param config: The run's config.
    :type config: SAM2TrainingConfig
    :return: ``(train_dataset, val_dataset)``, each yielding
        ``(inputs, targets)`` dicts ready for ``SAM2TrainingModel.fit``.
    :rtype: Tuple[tensorflow.data.Dataset, tensorflow.data.Dataset]

    .. note::

       The two datasets are drawn from DIFFERENT seeds, so validation clips are
       genuinely unseen. That is not a benchmark protocol; this trainer makes
       no accuracy claim.
    """
    common = dict(
        num_frames=config.num_frames,
        image_size=config.image_size,
        mask_size=config.mask_size,
        batch_size=config.batch_size,
        occlusion_frames=config.occlusion_frames,
        occlusion_start=config.occlusion_start,
        num_background_points=config.num_background_points,
        include_box=config.include_box,
    )
    train_dataset = build_sam2_video_dataset(
        num_clips=config.num_clips_train,
        seed=config.seed,
        shuffle_buffer=min(config.num_clips_train, 64),
        **common,
    )
    val_dataset = build_sam2_video_dataset(
        num_clips=config.num_clips_val,
        seed=config.seed + 10_000,
        **common,
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def create_sam2_training_model(
        config: SAM2TrainingConfig,
) -> SAM2TrainingModel:
    """Build and compile the trainable multi-frame wrapper.

    :param config: The run's config.
    :type config: SAM2TrainingConfig
    :return: A compiled :class:`SAM2TrainingModel`.
    :rtype: SAM2TrainingModel

    .. note::

       ``multimask_output=False`` is passed EXPLICITLY rather than left to the
       variant table: the wrapper refuses ``True`` outright (the frame axis is
       folded into the mask axis, so at ``M > 1`` the two interleave with no
       shape symptom), and a variant table that changed its default would
       otherwise turn that refusal into a crash at construction.

       Compilation goes through ``compile_sam2_video_trainer``, the ONE home of
       the three-key ``loss=`` dict, of the mandatory object-score BCE and of
       ``jit_compile=False``. Do not re-spell any of the three here.
    """
    sam2 = create_sam2(config.variant, multimask_output=False)
    model = SAM2TrainingModel(
        sam2, num_frames=config.num_frames, seed=config.seed
    )
    compile_sam2_video_trainer(
        model,
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        mask_weight=config.mask_weight,
        object_score_weight=config.object_score_weight,
        iou_weight=config.iou_weight,
    )
    return model


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def train_sam2(config: SAM2TrainingConfig) -> Tuple[SAM2TrainingModel, Any]:
    """Run the training.

    :param config: The run's config.
    :type config: SAM2TrainingConfig
    :return: ``(model, history)``.
    :rtype: Tuple[SAM2TrainingModel, Any]
    """
    output_dir = resolved_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")
    logger.info(
        "SAM 2 video training run '%s' (%s, %dpx, T=%d) -> %s. This is a "
        "WIRING result, not a segmentation result: no Meta SAM 2 checkpoint "
        "has ever been loaded in this repository.",
        config.experiment_name, config.variant, config.image_size,
        config.num_frames, output_dir,
    )

    train_dataset, val_dataset = create_dataset(config)
    model = create_sam2_training_model(config)

    callbacks, _ = create_common_callbacks(
        model_name=str(config.experiment_name),
        results_dir_prefix="sam2",
        run_dir=str(output_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training finished in %.2f s", time.time() - start)
    # IN-PROCESS peak, never `nvidia-smi` polling: `setup_gpu`'s
    # `set_memory_growth` is a repo-wide no-op and TF pre-allocates most of the
    # card, so an external reading measures the allocator, not this run.
    for device in tf.config.list_physical_devices("GPU"):
        try:
            info = tf.config.experimental.get_memory_info(
                f"GPU:{device.name.split(':')[-1]}"
            )
            logger.info(
                "GPU peak on %s: %.1f MiB (current %.1f MiB)",
                device.name,
                info["peak"] / 1024 ** 2,
                info["current"] / 1024 ** 2,
            )
        except Exception as error:  # pragma: no cover - reporting path
            logger.warning("Could not read GPU memory info: %s", error)

    try:
        history_dict = {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as handle:
            json.dump(history_dict, handle, indent=2)
    except Exception as error:  # pragma: no cover - reporting path
        logger.warning("Failed to write training history: %s", error)

    try:
        model.save(output_dir / "final_model.keras")
        logger.info("Final model saved to %s", output_dir / "final_model.keras")
    except Exception as error:  # pragma: no cover - reporting path
        logger.error("Failed to save the final model: %s", error)

    return model, history


def main() -> None:
    """Parse the CLI, set the process up, and train."""
    args, config = parse_arguments()
    setup_gpu(gpu_id=args.gpu)
    set_seeds(config.seed)
    logger.info(
        "Config: %s",
        {k: v for k, v in asdict(config).items() if k in CLI_TO_CONFIG.values()},
    )
    train_sam2(config)
    logger.info("SAM 2 video training completed.")


if __name__ == "__main__":
    main()
