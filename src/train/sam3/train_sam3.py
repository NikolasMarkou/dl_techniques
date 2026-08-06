"""SAM 3 trainer: stock ``compile()``/``fit()`` over :class:`Sam3TrainingModel`.

Run it. EVERY command below was EXECUTED, on a 4070, before it was written
here -- a command in a docstring is a CLAIM, and this repository has shipped a
control command that exited 1 by construction after being advertised for two
steps::

    # Wiring proof. Exit 0 in 1 m 51 s; val_loss 7.1194 -> 6.0831, achieved
    # box IoU 0.1824 after two epochs.
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam3.train_sam3 --smoke

    # The frozen arm of the frozen-vs-joint A/B. Exit 0 in 1 m 35 s; the log
    # reports "247 of 323 trainable variables remain".
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam3.train_sam3 --smoke --freeze-trunk

    # With mask supervision on. Exit 0 in 1 m 58 s; `val_mask_iou` becomes a
    # real 0.0 instead of `nan`, and the two mask terms stop being exactly 0.
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam3.train_sam3 --smoke --include-masks

    # Every decoder layer supervised, not just the last. Exit 0 in 2 m 12 s
    # against 2 m 05 s for the same command WITHOUT the flag, measured back to
    # back on the same card in the same minute -- a 1.05x wall-clock ratio, and
    # the log reports "aux blocks=2" instead of "aux blocks=0". The ratio is
    # DILUTED by import, build and save: a 2-epoch smoke run is 16 training
    # steps, so it is a floor on the per-step cost, not an estimate of it.
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam3.train_sam3 --smoke --deep-supervision

A full-length run is the same command with a larger ``--epochs`` and
``--num-train-samples`` and a pinned ``--seed``. No such run has been executed
as of this file's authorship, so no wall time, loss or IoU for one is quoted
here.

What this trainer is, and is not
--------------------------------
It trains the SAM 3 port on a SYNTHETIC text-prompted detection task, from
RANDOM INITIALIZATION. No Meta SAM 3 checkpoint has ever been loaded in this
repository, the default ``small`` variant is not a published size, and nothing
here is a benchmark protocol. What it produces is the evidence for a
LEARNABILITY claim: achieved box IoU, mask IoU and presence accuracy on matched
pairs -- not a loss curve.

Why a metric and not a loss
---------------------------
A falling loss proves nothing here. A constant mask output has a perfectly
stable loss, and at random initialization ``pred_masks`` really is near
constant. So :func:`evaluate_sam3` reports achieved IoU on the SAME Hungarian
assignment the loss uses, plus the UNIQUE-VALUE COUNT of ``pred_masks``, which
is 1 exactly when the head has collapsed to a constant. The per-term losses are
reported beside them (:meth:`Sam3DetectionLoss.compute_terms`), because a
falling TOTAL can hide a term that is doing nothing.

Argument wiring, and why it is a table and not a call
----------------------------------------------------
This repository has a RECORDED defect class: a trainer's ``main()`` lists each
config field by hand, one line is omitted, and the corresponding CLI flag
becomes a **silent no-op** -- the run completes, the artifact is wrong, and no
test notices (`plans/LESSONS.md`; `train/bfunet` shipped exactly this with
``--high-freq-blocks`` and ``--filter-multiplier``). So the argparse ``dest`` ->
config-field wiring lives in ONE table, :data:`CLI_TO_CONFIG`, and
:func:`config_from_argv` builds the config by iterating it. The table is what
``tests/test_train/test_sam3/test_train_sam3.py`` walks: every parser flag is
driven with a sentinel through the FULL ``argv -> parse -> config`` path and the
resulting field is read back.

The ``--smoke`` preset and provenance
-------------------------------------
``--smoke`` is applied in the config BUILDER, gated on
:func:`train.common.args.explicitly_set_flags`, so a flag the caller actually
typed beats the preset -- **including one typed at its own parser default**,
which a parsed-value-vs-default comparison structurally cannot express. Every
preset entry changes HOW MUCH is measured; none changes WHAT is measured, and
``batch_size`` is deliberately absent from the preset even though the SAM 1 and
SAM 2 templates carry it (decisions.md D-030).
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import ops

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
    create_learning_rate_schedule,
    save_config_json,
)
# `explicitly_set_flags` is NOT re-exported from `train.common`; it lives in
# `train.common.args`, which is how every other adopter imports it.
from train.common.args import explicitly_set_flags

from dl_techniques.losses.sam3_detection_loss import (
    Sam3DetectionLoss,
    box_cxcywh_to_xyxy,
    iou_and_generalized_iou,
    unpack_targets,
)
from dl_techniques.models.sam3.sam3_image import Sam3Image
from dl_techniques.models.sam3.training_model import (
    Sam3TrainingModel,
    compile_sam3_trainer,
    pack_predictions,
)
from dl_techniques.utils.logger import logger

from train.sam3.data import build_sam3_dataset

#: Variants this trainer accepts. Read from the model's own table so the two
#: cannot drift; `sam3` (821,708,598 params, 10 GiB forward peak) is excluded
#: because it does not fit on the 12 GB card with AdamW moments.
VARIANTS: Tuple[str, ...] = tuple(
    name for name in sorted(Sam3Image.MODEL_VARIANTS) if name != "sam3")

#: Schedules :func:`train.common.create_learning_rate_schedule` implements.
LR_SCHEDULES: Tuple[str, ...] = ("cosine", "exponential", "constant")

#: Every key :func:`evaluate_sam3` returns, ALWAYS, in this order. The set is
#: fixed because ``keras.callbacks.CSVLogger`` FREEZES its column set on the
#: first epoch it sees: a metric that appears at epoch 3 is silently dropped
#: from every row. An unevaluated metric is therefore written as ``nan``, never
#: omitted (decisions.md D-028).
#:
#: The four ``*_std_*`` keys are the DEGENERACY GUARD FOR THE SUPERVISED HEADS.
#: Until iteration 1's review the only guard was ``pred_mask_unique_values``,
#: which watches ``pred_masks`` -- the one head that is OFF by default and was
#: off in five of six step-7 arms. MEASURED on the saved step-7 checkpoints:
#: ``box_std_across_images`` 0.00048 against ``box_std_across_queries`` 0.13951
#: (a 290x ratio on seed 1, 4363x on seed 3) and a ``presence_logit`` spread of
#: 1.4e-04, i.e. the box and presence heads had converged to constants that do
#: not read the image at all, while every shipped guard read green. An IoU
#: reported without these numbers beside it cannot distinguish "learned the
#: task" from "learned the dataset's mean box" (decisions.md D-039).
EVAL_METRIC_KEYS: Tuple[str, ...] = (
    "box_iou",
    "mask_iou",
    "presence_accuracy",
    "pred_mask_unique_values",
    "box_std_across_images",
    "box_std_across_queries",
    "logit_std_across_images",
    "presence_logit_std_across_images",
    "num_matched_pairs",
    "loss_ce",
    "presence_loss",
    "loss_bbox",
    "loss_giou",
    "loss_mask",
    "loss_dice",
)

#: The log key `EarlyStopping` and `ModelCheckpoint` select on. It is an
#: ACHIEVED metric, maximized, and it must be a `val_`-prefixed member of
#: :data:`EVAL_METRIC_KEYS` -- pinned by a test, because a typo here degrades
#: silently into "select epoch 1" rather than raising.
SELECTION_METRIC: str = "val_box_iou"


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class Sam3TrainingConfig:
    """Every knob the SAM 3 trainer reads.

    Field names are the target half of :data:`CLI_TO_CONFIG`; a field no CLI
    flag names must be listed in :data:`DERIVED_FIELDS` or the completeness
    guard fails.
    """

    # Data
    num_train_samples: int = 512
    num_val_samples: int = 128
    max_instances: int = 8
    max_per_category: int = 3
    zero_instance_rate: float = 0.25

    # Model
    variant: str = "small"
    include_masks: bool = False
    freeze_trunk: bool = False
    deep_supervision: bool = False

    # Optimizer. Reference-derived; see `create_optimizer` for the SIGNED,
    # NAMED divergences from the reference's own recipe.
    learning_rate: float = 8e-4
    weight_decay: float = 0.1
    gradient_clip_norm: float = 0.1
    warmup_steps: int = 20
    lr_schedule: str = "cosine"

    # Training
    batch_size: int = 4
    epochs: int = 30
    early_stopping_patience: int = 10

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Preset
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r}; this trainer accepts "
                f"{list(VARIANTS)}. The released 'sam3' geometry is refused on "
                f"purpose: 821,708,598 parameters at a 10,072.9 MiB forward "
                f"peak leaves no room for AdamW moments on a 12 GB card.")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"unknown lr_schedule {self.lr_schedule!r}; known schedules "
                f"are {list(LR_SCHEDULES)}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0; got {self.epochs}")
        for name in ("num_train_samples", "num_val_samples"):
            count = getattr(self, name)
            if count < self.batch_size:
                raise ValueError(
                    f"{name}={count} is smaller than batch_size="
                    f"{self.batch_size}. The dataset drops its remainder "
                    f"(D-023 -- a static batch axis is mandatory), so the "
                    f"split would be EMPTY and `fit` would see no steps.")
        if self.max_instances <= 0 or self.max_per_category <= 0:
            raise ValueError(
                f"max_instances ({self.max_instances}) and max_per_category "
                f"({self.max_per_category}) must both be > 0")
        if self.max_per_category > self.max_instances:
            raise ValueError(
                f"max_per_category ({self.max_per_category}) exceeds "
                f"max_instances ({self.max_instances}): targets would be "
                f"silently truncated.")
        if not 0.0 <= self.zero_instance_rate <= 1.0:
            raise ValueError(
                f"zero_instance_rate must be in [0, 1]; got "
                f"{self.zero_instance_rate}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0; got {self.weight_decay}")
        if self.gradient_clip_norm < 0.0:
            raise ValueError(
                f"gradient_clip_norm must be >= 0 (0 disables clipping); got "
                f"{self.gradient_clip_norm}")
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0; got {self.warmup_steps}")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be > 0; got "
                f"{self.early_stopping_patience}")
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"sam3_{self.variant}_{timestamp}"

    @property
    def steps_per_epoch(self) -> int:
        """Batches in one training epoch, after the remainder is dropped."""
        return self.num_train_samples // self.batch_size


#: argparse ``dest`` -> :class:`Sam3TrainingConfig` field. THE wiring, in one
#: place. Adding a flag without adding its row here fails
#: ``test_every_cli_flag_is_wired_to_a_config_field``; deleting a row fails
#: ``test_every_cli_flag_reaches_the_config_field_it_names``, by flag name.
CLI_TO_CONFIG: Dict[str, str] = {
    "num_train_samples": "num_train_samples",
    "num_val_samples": "num_val_samples",
    "max_instances": "max_instances",
    "max_per_category": "max_per_category",
    "zero_instance_rate": "zero_instance_rate",
    "variant": "variant",
    "include_masks": "include_masks",
    "freeze_trunk": "freeze_trunk",
    "deep_supervision": "deep_supervision",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "gradient_clip_norm": "gradient_clip_norm",
    "warmup_steps": "warmup_steps",
    "lr_schedule": "lr_schedule",
    "batch_size": "batch_size",
    "epochs": "epochs",
    "early_stopping_patience": "early_stopping_patience",
    "seed": "seed",
    "output_dir": "output_dir",
    "experiment_name": "experiment_name",
    "smoke": "smoke",
}

#: argparse dests that deliberately do NOT reach the config: they act on the
#: process, not on the run's parameters.
NON_CONFIG_DESTS: Set[str] = {"help", "gpu"}

#: Config fields no CLI flag names. ``steps_per_epoch`` is a property, not a
#: dataclass field, so it is not listed here.
DERIVED_FIELDS: Set[str] = set()

#: The ``--smoke`` preset. Every entry changes HOW MUCH is measured; none
#: changes WHAT is measured. ``batch_size`` is deliberately ABSENT even though
#: the SAM 1 and SAM 2 presets carry it -- see decisions.md D-030.
SMOKE_PRESET: Dict[str, Any] = {
    "num_train_samples": 32,
    "num_val_samples": 8,
    "epochs": 2,
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
        description=("Train SAM 3 with stock compile()/fit() over "
                     "Sam3TrainingModel on a synthetic text-prompted task"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = Sam3TrainingConfig()

    data = parser.add_argument_group("data")
    data.add_argument("--num-train-samples", type=int,
                      default=defaults.num_train_samples,
                      help="Images in one training epoch.")
    data.add_argument("--num-val-samples", type=int,
                      default=defaults.num_val_samples)
    data.add_argument("--max-instances", type=int,
                      default=defaults.max_instances,
                      help="N_max, the padded GT slot count.")
    data.add_argument("--max-per-category", type=int,
                      default=defaults.max_per_category)
    data.add_argument("--zero-instance-rate", type=float,
                      default=defaults.zero_instance_rate,
                      help="Probability the prompt names an ABSENT category. "
                           "This is what supervises the presence head.")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=list(VARIANTS),
                       default=defaults.variant,
                       help="'small' is a trainable-on-12-GB geometry, NOT a "
                            "published SAM 3 size; 'tiny' is degenerate and "
                            "exists for development.")
    model.add_argument("--include-masks",
                       action=argparse.BooleanOptionalAction,
                       default=defaults.include_masks,
                       help="Supervise pred_masks. OFF by default, matching "
                            "the reference's one shipped config, whose mask "
                            "loss block is commented out (D-009).")
    model.add_argument("--freeze-trunk",
                       action=argparse.BooleanOptionalAction,
                       default=defaults.freeze_trunk,
                       help="Freeze the image trunk. The frozen arm of the "
                            "frozen-vs-joint A/B.")
    model.add_argument("--deep-supervision",
                       action=argparse.BooleanOptionalAction,
                       default=defaults.deep_supervision,
                       help="Supervise EVERY decoder layer, not just the last "
                            "(the reference's aux_outputs). Adds L-1 packed "
                            "blocks and runs the Hungarian matcher once per "
                            "block, so a step costs more. MEASURED at "
                            "small/60ep/synthetic, 3 seeds, this flag the only "
                            "changed variable: box IoU rose on 3 of 3 seeds "
                            "(+0.110..+0.140), box_std_across_images improved "
                            "on 0 of 3. Read that gain against a TRIVIAL "
                            "baseline, not against zero: on those same 3 "
                            "splits an UNTRAINED, image-independent 5x5 grid "
                            "of fixed boxes scores 0.357/0.331/0.343, above "
                            "the no-flag arm on 3 of 3 and within "
                            "0.025..0.044 of this flag's own "
                            "0.401/0.369/0.368. The direction is real on 3 of "
                            "3 seeds; the margin over a predictor that reads "
                            "nothing is small.")

    optimizer = parser.add_argument_group("optimizer")
    optimizer.add_argument("--learning-rate", type=float,
                           default=defaults.learning_rate)
    optimizer.add_argument("--weight-decay", type=float,
                           default=defaults.weight_decay)
    optimizer.add_argument("--gradient-clip-norm", type=float,
                           default=defaults.gradient_clip_norm,
                           help="Global L2 gradient-norm clip; 0 disables it.")
    optimizer.add_argument("--warmup-steps", type=int,
                           default=defaults.warmup_steps,
                           help="Linear LR warmup, in STEPS (the reference's "
                                "own unit and value).")
    optimizer.add_argument("--lr-schedule", choices=list(LR_SCHEDULES),
                           default=defaults.lr_schedule)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--early-stopping-patience", type=int,
                          default=defaults.early_stopping_patience)
    training.add_argument("--seed", type=int, default=defaults.seed)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help="Relative paths resolve against the REPO ROOT, "
                             "never the current directory, so a run from "
                             "anywhere writes to <repo>/results/.")
    output.add_argument("--experiment-name", type=str,
                        default=defaults.experiment_name)
    output.add_argument("--gpu", type=int, default=None,
                        help="GPU index for setup_gpu (process-level; not a "
                             "config field).")
    output.add_argument("--smoke", action=argparse.BooleanOptionalAction,
                        default=defaults.smoke,
                        help="Shrink the run to a wiring proof. Any flag you "
                             "type explicitly wins over the preset.")
    return parser


def parse_arguments(
        argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, Sam3TrainingConfig]:
    """Run the FULL ``argv -> parse -> config`` path.

    Interface contract: this is the single entry point ``main`` uses and the
    single entry point the wiring guard drives, so there is no path a test can
    pass while the trainer fails.

    :param argv: Tokens without the program name. ``None`` reads
        ``sys.argv[1:]``.
    :type argv: Optional[Sequence[str]]
    :return: ``(namespace, config)``. The namespace is returned only so ``main``
        can read the process-level dests in :data:`NON_CONFIG_DESTS`.
    :rtype: Tuple[argparse.Namespace, Sam3TrainingConfig]
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
        # DECISION plan-2026-08-03T191222-1d751f81/D-041
        # Apply the preset HERE, in the builder, gated on PROVENANCE. Do NOT
        # move it into `Sam3TrainingConfig.__post_init__`: by then the argv
        # tokens are gone, so `--smoke --epochs 30` (30 being the flag's own
        # default) is indistinguishable from a bare `--smoke`, and the preset
        # silently overrides a value the caller really typed. And do NOT add a
        # field to SMOKE_PRESET that changes WHAT is measured (variant, seed,
        # learning rate, batch size, mask switch, DEEP-SUPERVISION switch,
        # zero-instance rate) -- only
        # how much; `test_the_preset_changes_how_much_not_what` pins that list.
        for field, preset_value in SMOKE_PRESET.items():
            if field not in explicit_fields:
                values[field] = preset_value
    return args, Sam3TrainingConfig(**values)


def config_from_argv(
        argv: Optional[Sequence[str]] = None) -> Sam3TrainingConfig:
    """The config half of :func:`parse_arguments`.

    :param argv: Tokens without the program name.
    :type argv: Optional[Sequence[str]]
    :return: A validated :class:`Sam3TrainingConfig`.
    :rtype: Sam3TrainingConfig
    """
    return parse_arguments(argv)[1]


def resolved_output_dir(config: Sam3TrainingConfig) -> Path:
    """Resolve the run directory, anchoring a relative path at the REPO ROOT.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :return: ``<repo>/<output_dir>/<experiment_name>`` for a relative
        ``output_dir``, or ``<output_dir>/<experiment_name>`` for an absolute
        one.
    :rtype: Path
    """
    # The editable install puts `<repo>/src` on `sys.path`, so
    # `python -m train.sam3.train_sam3` resolves from ANY working directory and
    # a bare `Path(config.output_dir)` would write a stray results tree wherever
    # the user happened to be standing -- including `src/results/`, which the
    # repo convention names explicitly as the wrong place.
    root = Path(config.output_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    return root / str(config.experiment_name)


# ---------------------------------------------------------------------------
# MODEL + OPTIMIZER
# ---------------------------------------------------------------------------
def create_optimizer(config: Sam3TrainingConfig) -> keras.optimizers.Optimizer:
    """Build the optimizer, derived from the reference by SIGNED divergences.

    Interface contract: returns a configured, unbuilt Keras optimizer. It reads
    only ``config`` and raises nothing.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :return: An ``AdamW`` carrying the schedule, the clip and the decay
        exclusions.
    :rtype: keras.optimizers.Optimizer
    """
    # DECISION plan-2026-08-05T124709-6c4fac48/D-027
    # These numbers are derived from the pinned reference clone
    # (96914d2425f90a64f45ca977c2b5165418099543,
    # `roboflow_v100_full_ft_100_images.yaml`), NOT invented and NOT copied
    # wholesale. The reference recipe fine-tunes an 822M PRETRAINED model on
    # 100 images; this trains 5,881,614 parameters from RANDOM INIT on a
    # synthetic task, so several of its terms would be actively wrong here.
    # `this_port = reference + PORT_ONLY(...) - REFERENCE_ONLY(...)`:
    #   ADOPTED EXACT: AdamW; global L2 grad clip `max_norm 0.1`; `wd 0.1`
    #     zeroed on bias and LayerNorm parameters; `warmup_steps 20` (the
    #     reference counts warmup in STEPS, not epochs -- `trainer.py:825`
    #     passes `step=int(exact_epoch * iters_per_epoch)`).
    #   - REFERENCE_ONLY(lr_scale 0.1): a FINE-TUNING discount on the
    #     pretraining LR. From random init there is no pretrained weight to
    #     protect, so the port takes the UNSCALED transformer LR 8e-4 rather
    #     than the fine-tune's 8e-5.
    #   - REFERENCE_ONLY(3 param groups: lr_vision_backbone 2.5e-5,
    #     lr_language_backbone 5e-6) and - REFERENCE_ONLY(layer_decay 0.9 on
    #     the trunk): both exist to keep a fine-tune from destroying pretrained
    #     backbones. At random init a 10x-lower trunk LR only STARVES the trunk,
    #     and it would silently confound step 7's frozen-vs-joint A/B, which is
    #     exactly a question about how much the trunk learns. ONE group.
    #   - REFERENCE_ONLY(InverseSquareRootParamScheduler + linear cooldown):
    #     replaced by + PORT_ONLY(the repo's shared warmup+cosine schedule).
    #     Writing the inverse-sqrt form would require a new public
    #     `LearningRateSchedule` class, which this plan's Complexity Budget
    #     makes a STOP-and-renegotiate trigger; cosine is monotone-decreasing
    #     with an implicit cooldown to `alpha` and is already shared code.
    #   - REFERENCE_ONLY(amp bfloat16): no mixed precision here. The matcher
    #     crosses an eager boundary and this plan makes no fp16/bf16 claim.
    # Do NOT "restore fidelity" by copying 8e-5 back: that is a fine-tuning
    # number applied to a from-scratch run. See decisions.md D-027.
    learning_rate: Any = create_learning_rate_schedule(
        initial_lr=config.learning_rate,
        schedule_type=config.lr_schedule,
        total_epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        warmup_steps=config.warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        global_clipnorm=(config.gradient_clip_norm
                         if config.gradient_clip_norm > 0.0 else None),
    )
    # The reference zeroes weight decay on `*bias*` and on LayerNorm modules.
    # Keras spells LayerNorm's two parameters `gamma` and `beta`, and matches
    # these patterns with `re.search` against the variable name.
    optimizer.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
    return optimizer


def create_training_model(
        config: Sam3TrainingConfig) -> Sam3TrainingModel:
    """Build and compile the trainable wrapper.

    Interface contract: returns a BUILT, COMPILED :class:`Sam3TrainingModel` at
    ``config.variant``, with ``jit_compile=False`` and, when
    ``config.freeze_trunk`` is set, a non-trainable image trunk. The model and
    the loss agree on BOTH packed-layout axes -- ``include_masks`` (the channel
    width) and ``num_aux_layers`` (the row stride) -- because
    :func:`compile_sam3_trainer` raises on either disagreement.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :return: The compiled wrapper.
    :rtype: Sam3TrainingModel
    """
    model = Sam3TrainingModel(
        Sam3Image.from_variant(config.variant),
        include_masks=config.include_masks,
        deep_supervision=config.deep_supervision,
    )
    model.build(None)

    if config.freeze_trunk:
        # DECISION plan-2026-08-05T124709-6c4fac48/D-029
        # Freeze the IMAGE TRUNK ONLY, and freeze it BEFORE `compile`. Do NOT
        # set `model.trainable = False` on the wrapper and then re-enable parts:
        # Keras propagates `trainable` down the whole tree, so a later
        # `model.trainable = True` anywhere would SILENTLY UNDO this and the run
        # would look like the joint arm while claiming to be the frozen one.
        # The A/B this feeds (step 7c) is only meaningful if the two arms
        # differ in exactly one thing, so the assertion that belongs with this
        # flag is a trainable-variable COUNT, not the flag's own value --
        # `test_freeze_trunk_drops_the_trainable_variable_count` measures it in
        # both directions. See decisions.md D-029.
        model.sam3.backbone.trainable = False
        logger.info(
            "freeze_trunk: image trunk frozen -- %d of %d trainable variables "
            "remain", len(model.trainable_variables),
            len(model.trainable_variables) + len(model.non_trainable_variables))

    compile_sam3_trainer(
        model,
        optimizer=create_optimizer(config),
        # DECISION plan-2026-08-05T124709-6c4fac48/D-040
        # `pad_n_queries` is DERIVED from the variant's own Q, never left at the
        # loss's reference default of 200. Do NOT "restore fidelity" by pinning
        # 200 here: BOTH shipped reference configs set it to the model's own
        # query count (`roboflow_*.yaml:100` writes 200 literally beside
        # `num_queries=200`; `odinw_text_only_train.yaml:102` writes
        # `${scratch.num_queries}` outright), so 200 is the reference's Q, not a
        # constant. Carried into a Q=32 variant it divides the ENTIRE
        # classification term by exactly 6.25 -- MEASURED on a real `small`
        # batch, raw `loss_ce` 0.043937 at 200 vs 0.274605 at 32 (ratio
        # 6.2500), and over the whole 64-image split its weighted share of the
        # total moves 9.1% -> 38.4%. See decisions.md D-040.
        # `num_aux_layers` is read off the MODEL, never re-derived from
        # `config.deep_supervision` here: the model derives it from its own
        # decoder depth (`L - 1`, hence 0 for a single-layer decoder even at
        # `deep_supervision=True`), and a second derivation is a second home for
        # the same fact. `compile_sam3_trainer` raises on a disagreement, so a
        # dropped argument here fails loudly rather than mis-slicing.
        loss=Sam3DetectionLoss(include_masks=config.include_masks,
                               pad_n_queries=model.num_queries,
                               num_aux_layers=model.num_aux_layers),
    )
    return model


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def create_datasets(
        config: Sam3TrainingConfig,
        model: Sam3TrainingModel,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build the training and validation datasets.

    Interface contract: both datasets are batched with a STATIC batch axis
    (``drop_remainder=True``, D-023) and every geometry is derived from
    ``model``. The two are drawn from DIFFERENT seeds, so validation is
    genuinely unseen; neither is a benchmark protocol.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :param model: The wrapper whose geometry the pipeline derives from.
    :type model: Sam3TrainingModel
    :return: ``(train_dataset, val_dataset)``.
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset]
    """
    common = dict(
        model=model,
        batch_size=config.batch_size,
        max_instances=config.max_instances,
        zero_instance_rate=config.zero_instance_rate,
        max_per_category=config.max_per_category,
    )
    train_dataset = build_sam3_dataset(
        num_samples=config.num_train_samples,
        seed=config.seed,
        shuffle_buffer=min(config.num_train_samples, 256),
        **common,
    )
    val_dataset = build_sam3_dataset(
        num_samples=config.num_val_samples,
        seed=config.seed + 10_000,
        **common,
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# EVALUATION -- the numbers a learnability claim is made of
# ---------------------------------------------------------------------------
def evaluate_sam3(
        model: Sam3TrainingModel,
        dataset: tf.data.Dataset,
        loss: Optional[Sam3DetectionLoss] = None,
        max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Measure achieved IoU, presence accuracy and the per-term losses.

    Interface contract: returns EXACTLY the keys of :data:`EVAL_METRIC_KEYS`,
    every one a Python float, with ``nan`` for a metric this configuration
    cannot measure (``mask_iou`` when ``include_masks`` is off, any metric on an
    empty dataset). It runs the model at ``training=False`` EXPLICITLY (H-9),
    performs ONE forward pass per batch and never raises. The six per-term
    losses are the MAIN (last) decoder layer's, at either setting of
    ``deep_supervision`` -- :meth:`Sam3DetectionLoss.compute_terms` reports the
    main block only -- so every logged term stays comparable across a
    deep-supervision A/B while the TOTAL the optimizer sees does not.

    ``box_iou`` and ``mask_iou`` are averaged over MATCHED PAIRS ONLY, using the
    same Hungarian assignment the loss uses -- an unmatched query has no ground
    truth to be scored against. ``pred_mask_unique_values`` is the MINIMUM over
    batches of the number of distinct values in ``pred_masks``: it is ``1``
    exactly when the mask head has collapsed to a constant, which is the
    degenerate output a stable loss cannot distinguish from a result.

    The ``*_std_*`` keys answer a question no other metric here can: **is the
    head's output CONSTANT across images?** That is NECESSARY for "the head
    reads the image" and is NOT sufficient for it, and the difference is
    measured rather than argued: on the step-8 validation split a hand-written
    predictor that reads NOTHING -- a fixed 5x5 box grid plus pure
    ``N(0, 0.05)`` noise -- scores ``box_std_across_images`` 4.91e-02, and a
    uniform-random-center predictor scores 1.42e-01, ABOVE the 7.02e-02 of an
    ORACLE head fed each image's own ground-truth boxes (seed 1; the other two
    seeds agree, see the plan's ``findings/chance-floor-and-instrument.md``).
    So a LOW reading convicts -- the shipped runs read 6.9e-06, four orders
    below every one of those -- while a high reading acquits nothing on its
    own. ``box_std_across_images`` is the standard
    deviation over the IMAGE axis, taken per (query, coordinate) and then
    averaged; ``box_std_across_queries`` is the same statistic over the QUERY
    axis. A head that has learned a fixed dataset prior has the first near zero
    and the second large -- and the Hungarian matcher will still score it a
    non-zero IoU, because it assigns the best-of-Q constant boxes to whatever
    ground truth is present. The two are reported together on purpose: neither
    is interpretable alone, and their RATIO is the diagnosis.

    :param model: A built :class:`Sam3TrainingModel`.
    :type model: Sam3TrainingModel
    :param dataset: A dataset of ``(inputs, packed_target)``.
    :type dataset: tf.data.Dataset
    :param loss: The loss whose matcher and terms are used. ``None`` takes the
        model's compiled loss.
    :type loss: Optional[Sam3DetectionLoss]
    :param max_batches: Stop after this many batches. ``None`` uses all.
    :type max_batches: Optional[int]
    :return: One float per key of :data:`EVAL_METRIC_KEYS`.
    :rtype: Dict[str, float]
    """
    if loss is None:
        loss = model.loss
    totals: Dict[str, float] = {key: 0.0 for key in EVAL_METRIC_KEYS}
    box_iou_sum = 0.0
    mask_iou_sum = 0.0
    matched_total = 0.0
    presence_correct = 0.0
    presence_total = 0.0
    unique_minimum: Optional[int] = None
    batches = 0
    # Kept per IMAGE, not per batch: an across-image statistic computed inside
    # a batch of 4 and then averaged is not the same number as one computed
    # over the whole split, and the split is what the claim is made on. That
    # difference is pinned by
    # `test_the_across_image_std_is_over_the_WHOLE_SPLIT_not_per_batch`, whose
    # heads are constant WITHIN a batch and varying BETWEEN batches -- the only
    # arm the two spellings score differently. MEASURED on
    # `results/step71_joint_seed1/final_model.keras`: 2.0135e-05 whole-split vs
    # 1.4989e-05 per-batch (-26 %).
    # KNOWN COST, named rather than fixed: this retains every batch's boxes,
    # logits and presence for the WHOLE split, so host memory is linear in
    # split size. Trivial at 64 images x 32 queries and it is called once per
    # epoch by `_Sam3EvalCallback` with no `max_batches`. A running Welford
    # accumulator gives the same number without the retention and is the fix
    # the first real validation set will require.
    head_boxes: List[np.ndarray] = []
    head_logits: List[np.ndarray] = []
    head_presence: List[np.ndarray] = []

    for index, (inputs, y_true) in enumerate(dataset):
        if max_batches is not None and index >= max_batches:
            break
        # DECISION plan-2026-08-06T055747-1e650383/D-006
        # Pack the SAME number of blocks the compiled loss slices. Do NOT
        # "simplify" this back to the single `pack_predictions(outputs, ...)`
        # call: `unpack_predictions` derives `Q` as
        # `rows // (1 + num_aux_layers) - 1` and validates NOTHING, so handing a
        # deep-supervision loss a main-block-only tensor does not raise -- it
        # reads a smaller Q and reports six per-term losses computed on a
        # fabricated slice (MEASURED on `tiny`: Q=5 read as Q=2). One forward
        # pass either way: at `num_aux_layers > 0` the main-layer dict is
        # element 0 of `call_per_layer`'s list and is bit-equal to `call`'s
        # output, so this branch does not double the eval cost.
        # See decisions.md D-006.
        if model.num_aux_layers:
            per_layer = model.sam3.call_per_layer(inputs, training=False)
            outputs = per_layer[0]
            packed = pack_predictions(outputs,
                                      include_masks=model.include_masks,
                                      aux_outputs=per_layer[1:])
        else:
            outputs = model.sam3(inputs, training=False)
            packed = pack_predictions(outputs,
                                      include_masks=model.include_masks)
        terms = loss.compute_terms(y_true, packed)
        targets = unpack_targets(ops.cast(y_true, "float32"),
                                 model.include_masks)

        pred_boxes = ops.cast(outputs["pred_boxes"], "float32")
        assignment, is_matched = loss.matcher(
            outputs["pred_logits"], pred_boxes,
            targets["target_boxes"], targets["target_valid"])
        gathered = ops.take_along_axis(
            targets["target_boxes"], assignment[:, :, None], axis=1)
        # `iou_and_generalized_iou` reads **xyxy**; every box on this path is
        # normalized `cxcywh` (H-5). Feeding it `cxcywh` directly does NOT
        # raise -- it silently computes an overlap between two rectangles that
        # are not the boxes, and MEASURED at this step it returned a flat 0.0
        # box IoU that looked exactly like "the model has not learned".
        iou, _ = iou_and_generalized_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(gathered))
        # `ops.where`, NOT `iou * is_matched`: the multiplicative spelling
        # propagates `nan * 0.0 = nan`, so ONE both-degenerate pair in a padded
        # column poisons the whole split's mean (review-iter-1 NOTE 10).
        box_iou_sum += float(ops.sum(ops.where(is_matched > 0.0, iou,
                                               ops.zeros_like(iou))))
        matched_total += float(ops.sum(is_matched))
        head_boxes.append(np.asarray(pred_boxes))
        head_logits.append(
            np.asarray(ops.cast(outputs["pred_logits"], "float32")))
        head_presence.append(np.asarray(
            ops.cast(outputs["presence_logit"], "float32")).reshape(-1))

        if model.include_masks and targets["target_masks"] is not None:
            flat = ops.reshape(
                ops.cast(outputs["pred_masks"], "float32"),
                (ops.shape(outputs["pred_masks"])[0],
                 ops.shape(outputs["pred_masks"])[1], -1))
            predicted = ops.cast(ops.sigmoid(flat) > 0.5, "float32")
            truth = ops.take_along_axis(
                targets["target_masks"], assignment[:, :, None], axis=1)
            intersection = ops.sum(predicted * truth, axis=-1)
            union = ops.sum(
                ops.maximum(predicted, truth), axis=-1) + 1e-6
            mask_iou_sum += float(
                ops.sum((intersection / union) * is_matched))

        presence = ops.cast(
            ops.sigmoid(ops.cast(outputs["presence_logit"], "float32")) > 0.5,
            "float32")
        presence_correct += float(
            ops.sum(ops.cast(presence == targets["keep_loss"], "float32")))
        presence_total += float(ops.size(presence))

        distinct = int(np.unique(np.asarray(outputs["pred_masks"])).size)
        unique_minimum = (distinct if unique_minimum is None
                          else min(unique_minimum, distinct))

        for key in EVAL_METRIC_KEYS:
            if key in terms:
                totals[key] += float(terms[key])
        batches += 1

    nan = float("nan")
    if batches == 0:
        return {key: nan for key in EVAL_METRIC_KEYS}
    metrics = {key: totals[key] / batches for key in EVAL_METRIC_KEYS}
    metrics["box_iou"] = (box_iou_sum / matched_total if matched_total > 0.0
                          else nan)
    metrics["mask_iou"] = (mask_iou_sum / matched_total
                           if model.include_masks and matched_total > 0.0
                           else nan)
    metrics["presence_accuracy"] = (presence_correct / presence_total
                                    if presence_total > 0.0 else nan)
    metrics["pred_mask_unique_values"] = float(unique_minimum or 0)
    metrics["num_matched_pairs"] = matched_total / batches
    boxes = np.concatenate(head_boxes)
    logits = np.concatenate(head_logits)
    presence = np.concatenate(head_presence)
    metrics["box_std_across_images"] = float(boxes.std(axis=0).mean())
    metrics["box_std_across_queries"] = float(boxes.std(axis=1).mean())
    metrics["logit_std_across_images"] = float(logits.std(axis=0).mean())
    metrics["presence_logit_std_across_images"] = float(presence.std())
    return metrics


# NOT @keras.saving.register_keras_serializable: callbacks are never serialized
# as part of a model (the `StepCheckpointCallback` precedent).
class _Sam3EvalCallback(keras.callbacks.Callback):
    """Write :func:`evaluate_sam3`'s metrics into every epoch's ``logs``.

    Module-private on purpose: it has exactly one production call site
    (:func:`train_sam3`) and all of its content is :func:`evaluate_sam3`, which
    IS public. Promoting it would spend this plan's last Complexity-Budget
    class slot on a five-line adapter (decisions.md D-028).

    :param dataset: The validation dataset to measure on.
    :type dataset: tf.data.Dataset
    :param prefix: Key prefix, so the metrics land beside ``val_loss``.
    :type prefix: str
    """

    def __init__(self, dataset: tf.data.Dataset,
                 prefix: str = "val_") -> None:
        super().__init__()
        self.dataset = dataset
        self.prefix = prefix

    def on_epoch_end(self, epoch: int,
                     logs: Optional[Dict[str, Any]] = None) -> None:
        """Measure, then write EVERY key -- filling failures with ``nan``.

        :param epoch: Zero-based epoch index.
        :type epoch: int
        :param logs: The epoch's log dict, mutated in place.
        :type logs: Optional[Dict[str, Any]]
        """
        if logs is None:
            return
        # DECISION plan-2026-08-05T124709-6c4fac48/D-028
        # Write EVERY key of EVAL_METRIC_KEYS on EVERY epoch, `nan` included,
        # and keep this callback BEFORE `CSVLogger` in the callback list. Do NOT
        # "optimize" by omitting a metric this configuration cannot measure:
        # MEASURED behaviour of `keras.callbacks.CSVLogger` is that it FREEZES
        # its column set from the first epoch's `logs` keys, so a key that first
        # appears at epoch 3 never reaches `training_log.csv` at all, and a key
        # that disappears breaks the row. Ordering matters for the same reason:
        # a callback appended AFTER CSVLogger writes into a dict CSVLogger has
        # already consumed. See decisions.md D-028.
        try:
            metrics = evaluate_sam3(self.model, self.dataset)
        except Exception as error:  # pragma: no cover - reporting path
            logger.warning("evaluate_sam3 failed at epoch %d: %s", epoch, error)
            metrics = {key: float("nan") for key in EVAL_METRIC_KEYS}
        for key in EVAL_METRIC_KEYS:
            logs[f"{self.prefix}{key}"] = float(
                metrics.get(key, float("nan")))


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def build_callbacks(config: Sam3TrainingConfig, output_dir: Path,
                    val_dataset: tf.data.Dataset) -> Any:
    """Build the callback list, with the metrics callback ahead of the logger.

    Interface contract: returns a list whose ``_Sam3EvalCallback`` precedes
    every callback that READS the epoch logs -- ``CSVLogger``,
    ``ModelCheckpoint`` and ``EarlyStopping``. This function exists so that
    ordering is a testable property rather than an inline detail of a function
    that only runs inside a real training job.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :param output_dir: The already-created run directory.
    :type output_dir: Path
    :param val_dataset: The dataset the metrics are measured on.
    :type val_dataset: tf.data.Dataset
    :return: The callback list.
    :rtype: Any
    """
    callbacks, _ = create_common_callbacks(
        model_name=str(config.experiment_name),
        results_dir_prefix="sam3",
        run_dir=str(output_dir),
        # NEITHER `monitor` NOR `patience` is passed, deliberately. Both are
        # read ONLY by the `EarlyStopping` / `ModelCheckpoint` that the next
        # statement filters out, and `use_lr_schedule=True` suppresses the one
        # other reader (`ReduceLROnPlateau`, which hardcodes `val_loss`
        # anyway). Passing `monitor="val_loss"` here was dead but READ, at a
        # glance, as if `val_loss` were still the selection scalar -- the exact
        # opposite of what D-041 decided. The real values are set below.
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    # DECISION plan-2026-08-05T124709-6c4fac48/D-041
    # Select checkpoints on ACHIEVED box IoU, never on `val_loss`. Do NOT revert
    # to `monitor="val_loss"`: MEASURED at iteration 1, `presence_loss` is
    # **61.7%** of `val_loss` while its own head's logit spread is 1.4e-04, i.e.
    # the majority of the selection scalar is a provably constant term. On seed
    # 3 that selected epoch 6 over epoch 29 on a **0.07%** val_loss margin and
    # cost box IoU 0.2360 vs 0.2724 -- and 0.2360 became the headline. The two
    # callbacks are REBUILT rather than re-configured because
    # `train.common.create_callbacks` derives `mode` as
    # `'max' if 'accuracy' in monitor else 'min'`, so passing `val_box_iou`
    # through it would silently select the WORST epoch. See decisions.md D-041.
    callbacks = [callback for callback in callbacks
                 if not isinstance(callback, (keras.callbacks.EarlyStopping,
                                              keras.callbacks.ModelCheckpoint))]
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=SELECTION_METRIC, mode="max", verbose=1,
        patience=config.early_stopping_patience, restore_best_weights=True))
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=str(output_dir / "best_model.keras"), monitor=SELECTION_METRIC,
        mode="max", save_best_only=True, verbose=1))
    # Inserted at the FRONT so it runs before `CSVLogger` (D-028) and before
    # `EarlyStopping`/`ModelCheckpoint`, which read the same `logs` dict. A
    # callback APPENDED here would write into a dict `CSVLogger` has already
    # consumed, and every metric would be missing from `training_log.csv`.
    callbacks.insert(0, _Sam3EvalCallback(val_dataset))
    return callbacks


def train_sam3(config: Sam3TrainingConfig
               ) -> Tuple[Sam3TrainingModel, Any, Dict[str, float]]:
    """Run the training and the final evaluation.

    :param config: The run's config.
    :type config: Sam3TrainingConfig
    :return: ``(model, history, final_metrics)``.
    :rtype: Tuple[Sam3TrainingModel, Any, Dict[str, float]]
    """
    output_dir = resolved_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")
    logger.info("SAM 3 run '%s' -> %s", config.experiment_name, output_dir)

    model = create_training_model(config)
    train_dataset, val_dataset = create_datasets(config, model)

    callbacks = build_callbacks(config, output_dir, val_dataset)

    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training finished in %.2f s", time.time() - start)

    metrics = evaluate_sam3(model, val_dataset)
    # BOTH numbers, always. `restore_best_weights=True` means `metrics` is the
    # SELECTED epoch, not the last one; reporting only one of the two is how a
    # selection rule silently picks the worse checkpoint (decisions.md D-041).
    curve = history.history.get(SELECTION_METRIC, [])
    last_epoch = float(curve[-1]) if curve else float("nan")
    logger.info(
        "ACHIEVED (not a loss): box IoU %.4f SELECTED / %.4f last epoch, mask "
        "IoU %.4f, presence acc %.4f, pred_masks distinct values %.0f",
        metrics["box_iou"], last_epoch, metrics["mask_iou"],
        metrics["presence_accuracy"], metrics["pred_mask_unique_values"])
    logger.info(
        "IMAGE DEPENDENCE (does the head read the image?): box std across "
        "images %.5f vs across queries %.5f (ratio %.1f), pred_logits std "
        "across images %.5f, presence_logit std %.3e",
        metrics["box_std_across_images"], metrics["box_std_across_queries"],
        metrics["box_std_across_queries"]
        / max(metrics["box_std_across_images"], 1e-12),
        metrics["logit_std_across_images"],
        metrics["presence_logit_std_across_images"])

    # IN-PROCESS peak, never `nvidia-smi` polling: TF pre-allocates ~85 % of the
    # card, so an external reading measures the allocator, not this run.
    for device in tf.config.list_physical_devices("GPU"):
        try:
            info = tf.config.experimental.get_memory_info(
                f"GPU:{device.name.split(':')[-1]}")
            logger.info("GPU peak on %s: %.1f MiB", device.name,
                        info["peak"] / 1024 ** 2)
        except Exception as error:  # pragma: no cover - reporting path
            logger.warning("Could not read GPU memory info: %s", error)

    try:
        payload = {
            "history": {key: [float(v) for v in values]
                        for key, values in history.history.items()},
            "final_metrics": metrics,
            "selection_metric": SELECTION_METRIC,
            "last_epoch_selection_metric": last_epoch,
        }
        with open(output_dir / "training_history.json", "w") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as error:  # pragma: no cover - reporting path
        logger.warning("Failed to write training history: %s", error)

    try:
        model.save(output_dir / "final_model.keras")
        logger.info("Final model saved to %s",
                    output_dir / "final_model.keras")
    except Exception as error:  # pragma: no cover - reporting path
        logger.error("Failed to save the final model: %s", error)

    return model, history, metrics


def main() -> None:
    """Parse the CLI, set the process up, and train."""
    args, config = parse_arguments()
    setup_gpu(gpu_id=args.gpu)
    set_seeds(config.seed)
    logger.info("Config: %s", {field.name: getattr(config, field.name)
                               for field in fields(config)})
    train_sam3(config)
    logger.info("SAM 3 training completed.")


if __name__ == "__main__":
    main()
