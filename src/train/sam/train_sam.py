"""
SAM trainer: stock ``compile()``/``fit()`` over :class:`SAMTrainingModel`.

Run it::

    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam.train_sam --smoke

    # Real synthetic run
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam.train_sam \\
        --epochs 50 --batch-size 4 --num-train-samples 2048

    # COCO 2017 instance masks
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 \\
        .venv/bin/python -m train.sam.train_sam \\
        --data-source coco --coco-max-images 200

What this trainer is, and is not
--------------------------------
It proves the training path RUNS with live gradients. It makes **no accuracy
claim**: no official Meta SAM checkpoint has ever been loaded in this
repository, and ``vit_l``/``vit_h`` have never been forward-passed. The
default model is the reduced ``tiny`` geometry, which is real SAM geometry at
reduced width, not a released variant.

Argument wiring, and why it is a table and not a call
----------------------------------------------------
This repository has a RECORDED defect class: a trainer's ``main()`` lists each
config field by hand, one line is omitted, and the corresponding CLI flag
becomes a **silent no-op** — the run completes, the artifact is wrong, and no
test notices (`plans/LESSONS.md`; the bfunet trainer shipped exactly this with
``--high-freq-blocks`` and ``--filter-multiplier``). So the argparse ``dest`` →
config-field wiring lives in ONE table, :data:`CLI_TO_CONFIG`, and
:func:`config_from_argv` builds the config by iterating it. The table is what
``tests/test_train/test_sam/test_train_sam.py`` walks: every parser flag is
driven with a sentinel value through the FULL ``argv → parse → config`` path
and the resulting field is read back. Deleting one row of the table makes that
guard fail, by flag name.

The ``--smoke`` preset and provenance
-------------------------------------
``--smoke`` is applied through
:func:`train.common.args.explicitly_set_flags`, so a flag the caller actually
typed beats the preset — **including one typed at its own parser default**,
which a parsed-value-vs-default comparison structurally cannot express. The
preset touches only :data:`SMOKE_PRESET`'s fields, and every one of them
changes *how much* is measured (samples, batch, epochs, patience), never
*what* is measured: not the data source, not the model geometry, not the loss
weights, not the round count.
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
# `train.common.args`, which is how the three other adopters import it.
from train.common.args import explicitly_set_flags

from dl_techniques.losses.sam_mask_loss import SAMIoULoss, SAMMaskLoss
from dl_techniques.models.SAM.SAM1.image_encoder import ImageEncoderViT
from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder
from dl_techniques.models.SAM.SAM1.model import SAM
from dl_techniques.models.SAM.SAM1.prompt_encoder import PromptEncoder
from dl_techniques.models.SAM.SAM1.training_model import (
    IOU_SUPERVISION,
    LOW_RES_LOGITS,
    SAMTrainingModel,
    TRAINING_REFINEMENT_ROUNDS,
)
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
from dl_techniques.utils.logger import logger

from train.sam.data import DATA_SOURCES, MASK_DIVISOR, build_sam_dataset

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
#: SAM's patch size, fixed across every released variant.
PATCH_SIZE = 16
#: The released variants (`SAM.from_variant`) are hard-wired to 1024 px inputs.
VARIANT_IMAGE_SIZE = 1024
#: The reduced default. Real SAM geometry (patch 16, window 14 against a 16x16
#: token grid so window padding actually happens, a non-empty
#: `global_attn_indexes`, `use_rel_pos=True`) at reduced WIDTH, so the whole
#: training path is exercised on a 12 GB card. It is deliberately the same
#: geometry `tests/test_models/test_sam/test_correctness.py` gates the model
#: with -- 202 weights / 321,862 parameters -- so a trainer run and the model
#: gate are measuring the same object.
TINY_GEOMETRY = {
    "embed_dim": 64,
    "depth": 4,
    "num_heads": 4,
    "out_chans": 32,
    "window_size": 14,
    "global_attn_indexes": (1, 3),
    "mask_in_chans": 8,
    "transformer_depth": 2,
    "transformer_num_heads": 2,
    "transformer_mlp_dim": 64,
    "iou_head_hidden_dim": 32,
}
VARIANTS = ("tiny", "vit_b", "vit_l", "vit_h")


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
@dataclass
class SAMTrainingConfig:
    """
    Every knob the SAM trainer reads.

    Field names are the target half of :data:`CLI_TO_CONFIG`; a field that no
    CLI flag names must be listed in :data:`DERIVED_FIELDS` or the completeness
    guard fails.
    """

    # Data
    data_source: str = "synthetic"
    image_size: int = 256
    num_train_samples: int = 2048
    num_val_samples: int = 128
    max_instances: int = 3
    include_box: bool = False
    num_background_points: int = 0
    coco_split: str = "train2017"
    coco_val_split: str = "val2017"
    coco_root: Optional[str] = None
    coco_max_images: Optional[int] = None

    # Model
    variant: str = "tiny"
    num_refinement_rounds: int = TRAINING_REFINEMENT_ROUNDS
    multimask_output: bool = False

    # Loss mixing. `SAMMaskLoss` carries focal:dice internally (D-036
    # re-derived 20:1 on this repo's code); `iou_weight` balances the IoU term
    # against the whole mask term in `compile(loss_weights=...)`.
    focal_weight: float = 20.0
    dice_weight: float = 1.0
    iou_weight: float = 1.0

    # Training
    batch_size: int = 4
    epochs: int = 50
    steps_per_epoch: Optional[int] = None
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Preset
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.data_source not in DATA_SOURCES:
            raise ValueError(
                f"unknown data_source {self.data_source!r}; known sources are "
                f"{sorted(DATA_SOURCES)}"
            )
        if self.variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {self.variant!r}; known variants are "
                f"{list(VARIANTS)}"
            )
        if self.variant != "tiny" and self.image_size != VARIANT_IMAGE_SIZE:
            raise ValueError(
                f"variant {self.variant!r} is built by SAM.from_variant, which "
                f"hard-wires a {VARIANT_IMAGE_SIZE}px input; got image_size="
                f"{self.image_size}. Pass --image-size {VARIANT_IMAGE_SIZE}, or "
                f"use --variant tiny for a reduced-width model at your size."
            )
        if self.image_size <= 0 or self.image_size % (PATCH_SIZE * MASK_DIVISOR):
            raise ValueError(
                f"image_size={self.image_size} must be a positive multiple of "
                f"{PATCH_SIZE * MASK_DIVISOR} (patch {PATCH_SIZE} x the "
                f"low-res-mask factor {MASK_DIVISOR}), so the GT mask grid "
                f"lands exactly on `low_res_logits`' resolution."
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0; got {self.epochs}")
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0 or None (None = a full pass over "
                f"num_train_samples); got {self.steps_per_epoch}"
            )
        if self.num_train_samples <= 0 or self.num_val_samples <= 0:
            raise ValueError(
                f"num_train_samples ({self.num_train_samples}) and "
                f"num_val_samples ({self.num_val_samples}) must both be > 0; "
                f"the validation set is what `monitor='val_loss'` reads."
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0; got {self.learning_rate}"
            )
        if self.num_refinement_rounds < 1:
            raise ValueError(
                f"num_refinement_rounds must be >= 1 (1 = no refinement); got "
                f"{self.num_refinement_rounds}"
            )
        if self.max_instances <= 0:
            raise ValueError(
                f"max_instances must be > 0; got {self.max_instances}"
            )
        if self.num_background_points < 0:
            raise ValueError(
                f"num_background_points must be >= 0; got "
                f"{self.num_background_points}"
            )
        if self.focal_weight < 0 or self.dice_weight < 0 or self.iou_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be > 0; got "
                f"{self.early_stopping_patience}"
            )
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"sam_{self.variant}_{timestamp}"


#: argparse ``dest`` -> :class:`SAMTrainingConfig` field. THE wiring, in one
#: place. Adding a flag without adding its row here fails
#: `test_every_cli_flag_is_wired_to_a_config_field`; deleting a row fails
#: `test_every_cli_flag_reaches_the_config_field_it_names`, by flag name.
CLI_TO_CONFIG: Dict[str, str] = {
    "data_source": "data_source",
    "image_size": "image_size",
    "num_train_samples": "num_train_samples",
    "num_val_samples": "num_val_samples",
    "max_instances": "max_instances",
    "include_box": "include_box",
    "num_background_points": "num_background_points",
    "coco_split": "coco_split",
    "coco_val_split": "coco_val_split",
    "coco_root": "coco_root",
    "coco_max_images": "coco_max_images",
    "variant": "variant",
    "num_refinement_rounds": "num_refinement_rounds",
    "multimask_output": "multimask_output",
    "focal_weight": "focal_weight",
    "dice_weight": "dice_weight",
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
#: changes WHAT is measured (source, geometry, round count, loss weights and
#: seed are all absent on purpose). Any field the caller typed explicitly wins
#: over its entry here -- see :func:`config_from_argv`.
SMOKE_PRESET: Dict[str, Any] = {
    "num_train_samples": 32,
    "num_val_samples": 8,
    "batch_size": 2,
    "epochs": 3,
    "early_stopping_patience": 2,
}


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser.

    Returns:
        The parser. Every option is a long (``--``) spelling plus argparse's
        own ``-h``: :func:`explicitly_set_flags` REFUSES a parser carrying any
        other short option, because it cannot see attached (``-b8``) or grouped
        (``-vb 8``) forms and would report a typed flag as not-typed.
    """
    parser = argparse.ArgumentParser(
        description="Train SAM with stock compile()/fit() over SAMTrainingModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = SAMTrainingConfig()

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-source", choices=sorted(DATA_SOURCES),
        default=defaults.data_source,
        help="Per-instance source. 'synthetic' needs no COCO on disk and no "
             "pycocotools, so an I/O problem can never masquerade as a model "
             "problem.")
    data.add_argument("--image-size", type=int, default=defaults.image_size)
    data.add_argument("--num-train-samples", type=int,
                      default=defaults.num_train_samples,
                      help="Instance records in one training epoch.")
    data.add_argument("--num-val-samples", type=int,
                      default=defaults.num_val_samples)
    data.add_argument("--max-instances", type=int,
                      default=defaults.max_instances,
                      help="Instances taken from any one image.")
    data.add_argument("--include-box", action=argparse.BooleanOptionalAction,
                      default=defaults.include_box,
                      help="Add a jittered box prompt beside the point.")
    data.add_argument("--num-background-points", type=int,
                      default=defaults.num_background_points)
    data.add_argument("--coco-split", type=str, default=defaults.coco_split)
    data.add_argument("--coco-val-split", type=str,
                      default=defaults.coco_val_split)
    data.add_argument("--coco-root", type=str, default=defaults.coco_root)
    data.add_argument("--coco-max-images", type=int,
                      default=defaults.coco_max_images,
                      help="Cap on COCO images read (a smoke-run lever).")

    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=list(VARIANTS),
                       default=defaults.variant,
                       help="'tiny' is a reduced-WIDTH model at real SAM "
                            "geometry; the vit_* variants are 1024px and have "
                            "never been trained in this repository.")
    model.add_argument("--num-refinement-rounds", type=int,
                       default=defaults.num_refinement_rounds)
    model.add_argument("--multimask-output",
                       action=argparse.BooleanOptionalAction,
                       default=defaults.multimask_output)

    loss = parser.add_argument_group("loss")
    loss.add_argument("--focal-weight", type=float,
                      default=defaults.focal_weight)
    loss.add_argument("--dice-weight", type=float, default=defaults.dice_weight)
    loss.add_argument("--iou-weight", type=float, default=defaults.iou_weight)

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--steps-per-epoch", type=int,
                          default=defaults.steps_per_epoch,
                          help="Default None = one full pass over "
                               "--num-train-samples.")
    training.add_argument("--learning-rate", type=float,
                          default=defaults.learning_rate)
    training.add_argument("--early-stopping-patience", type=int,
                          default=defaults.early_stopping_patience)
    training.add_argument("--seed", type=int, default=defaults.seed)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=str, default=defaults.output_dir,
                        help="Relative paths resolve against the REPO ROOT, "
                             "never the current directory, so `python -m "
                             "train.sam.train_sam` from anywhere writes to "
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
) -> Tuple[argparse.Namespace, SAMTrainingConfig]:
    """
    Run the FULL ``argv -> parse -> config`` path.

    This is the single entry point ``main`` uses and the single entry point the
    wiring guard drives, so there is no path a test can pass while the trainer
    fails.

    Args:
        argv: Tokens without the program name. ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``(namespace, config)``. The namespace is returned only so ``main`` can
        read the process-level dests in :data:`NON_CONFIG_DESTS` (``--gpu``);
        everything else must come from the config.
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
        # move it into `SAMTrainingConfig.__post_init__` the way
        # `train/superpoint/train_superpoint.py:174` does: by then the argv
        # tokens are gone, so `--smoke --epochs 50` (50 being the flag's own
        # default) is indistinguishable from a bare `--smoke`, and the preset
        # silently overrides a value the caller really typed. MEASURED: a
        # value-vs-default reimplementation of `explicit_fields` fires exactly
        # `test_an_explicitly_typed_DEFAULT_beats_the_preset` (`assert 3 == 50`).
        # And do NOT add a field to SMOKE_PRESET that changes WHAT is measured
        # (source, geometry, round count, loss weights, seed) -- only how much;
        # `test_the_preset_changes_how_much_not_what` pins that list.
        for field, preset_value in SMOKE_PRESET.items():
            if field not in explicit_fields:
                values[field] = preset_value
    return args, SAMTrainingConfig(**values)


def config_from_argv(
    argv: Optional[Sequence[str]] = None,
) -> SAMTrainingConfig:
    """
    The config half of :func:`parse_arguments`.

    Args:
        argv: Tokens without the program name.

    Returns:
        A validated :class:`SAMTrainingConfig`.
    """
    return parse_arguments(argv)[1]


def resolved_output_dir(config: SAMTrainingConfig) -> Path:
    """
    Resolve the run directory, anchoring a relative path at the REPO ROOT.

    `python -m train.sam.train_sam` resolves from any working directory (the
    editable install puts `<repo>/src` on `sys.path`), so a bare
    ``results`` would otherwise create a stray results tree wherever the user
    happened to be standing -- and the repo convention is repo-root
    ``results/``, never ``src/results/``.

    Args:
        config: The run's config.

    Returns:
        ``<repo>/<output_dir>/<experiment_name>`` for a relative
        ``output_dir``, or ``<output_dir>/<experiment_name>`` for an absolute
        one.
    """
    # DECISION plan-2026-08-03T191222-1d751f81/D-041
    # Anchor a relative path at the REPO ROOT, never at the cwd. Do NOT
    # "simplify" this to `Path(config.output_dir) / name`: D-034 measured that
    # the editable install makes `python -m train.sam.train_sam` resolve from
    # ANY working directory, so the plain form writes a stray `results/` tree
    # wherever the user happened to be standing -- including `src/results/`,
    # which the repo convention names explicitly as the wrong place. Pinned by
    # `test_the_resolved_path_is_not_under_src`.
    root = Path(config.output_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    return root / str(config.experiment_name)


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def create_dataset(
    config: SAMTrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build the training and validation datasets.

    Args:
        config: The run's config.

    Returns:
        ``(train_dataset, val_dataset)``, each yielding
        ``(inputs, y_true)`` dicts ready for ``SAMTrainingModel.fit``.

    Note:
        For the synthetic source the two datasets are drawn from DIFFERENT
        seeds, so validation is genuinely unseen data. For COCO they are
        different SPLITS (``--coco-split`` vs ``--coco-val-split``). Neither is
        a benchmark protocol; this trainer makes no accuracy claim.
    """
    common = dict(
        image_size=config.image_size,
        batch_size=config.batch_size,
        max_instances=config.max_instances,
        num_background_points=config.num_background_points,
        include_box=config.include_box,
        source=config.data_source,
    )
    train_kwargs: Dict[str, Any] = {}
    val_kwargs: Dict[str, Any] = {}
    if config.data_source == "coco":
        train_kwargs = {
            "split": config.coco_split,
            "coco_root": config.coco_root,
            "max_images": config.coco_max_images,
        }
        val_kwargs = dict(train_kwargs, split=config.coco_val_split)

    train_dataset = build_sam_dataset(
        num_samples=config.num_train_samples,
        seed=config.seed,
        shuffle_buffer=min(config.num_train_samples, 256),
        source_kwargs=train_kwargs,
        **common,
    )
    val_dataset = build_sam_dataset(
        num_samples=config.num_val_samples,
        seed=config.seed + 10_000,
        source_kwargs=val_kwargs,
        **common,
    )
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def create_sam(config: SAMTrainingConfig) -> SAM:
    """
    Build the SAM to train.

    Args:
        config: The run's config.

    Returns:
        An unbuilt :class:`SAM`. ``tiny`` is assembled here at reduced width;
        every other variant delegates to ``SAM.from_variant``.
    """
    if config.variant != "tiny":
        return SAM.from_variant(config.variant)

    grid = config.image_size // PATCH_SIZE
    geometry = TINY_GEOMETRY
    encoder = ImageEncoderViT(
        img_size=config.image_size,
        patch_size=PATCH_SIZE,
        embed_dim=geometry["embed_dim"],
        depth=geometry["depth"],
        num_heads=geometry["num_heads"],
        out_chans=geometry["out_chans"],
        use_rel_pos=True,
        window_size=geometry["window_size"],
        global_attn_indexes=geometry["global_attn_indexes"],
    )
    prompt_encoder = PromptEncoder(
        embed_dim=geometry["out_chans"],
        image_embedding_size=(grid, grid),
        input_image_size=(config.image_size, config.image_size),
        mask_in_chans=geometry["mask_in_chans"],
    )
    transformer = TwoWayTransformer(
        depth=geometry["transformer_depth"],
        embedding_dim=geometry["out_chans"],
        num_heads=geometry["transformer_num_heads"],
        mlp_dim=geometry["transformer_mlp_dim"],
    )
    mask_decoder = MaskDecoder(
        transformer_dim=geometry["out_chans"],
        transformer=transformer,
        iou_head_hidden_dim=geometry["iou_head_hidden_dim"],
    )
    return SAM(
        image_encoder=encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )


def create_training_model(config: SAMTrainingConfig) -> SAMTrainingModel:
    """
    Build and compile the trainable wrapper.

    Args:
        config: The run's config.

    Returns:
        A compiled :class:`SAMTrainingModel`.

    Note:
        The ``loss=`` dict keys a SUBSET of the model's output keys
        (``iou_predictions`` stays unsupervised, and its target is carried
        inside ``iou_supervision``). ``y_true``'s keys must match the keys
        ``loss=`` COVERS -- supplying an entry for an uncovered key raises
        ``ValueError: y_true and y_pred have different structures`` (D-036).
    """
    model = SAMTrainingModel(
        create_sam(config),
        multimask_output=config.multimask_output,
        seed=config.seed,
        num_refinement_rounds=config.num_refinement_rounds,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            LOW_RES_LOGITS: SAMMaskLoss(
                focal_weight=config.focal_weight,
                dice_weight=config.dice_weight,
            ),
            IOU_SUPERVISION: SAMIoULoss(),
        },
        loss_weights={LOW_RES_LOGITS: 1.0, IOU_SUPERVISION: config.iou_weight},
    )
    return model


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def train_sam(config: SAMTrainingConfig) -> Tuple[SAMTrainingModel, Any]:
    """
    Run the training.

    Args:
        config: The run's config.

    Returns:
        ``(model, history)``.
    """
    output_dir = resolved_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")
    logger.info("SAM training run '%s' -> %s", config.experiment_name, output_dir)

    train_dataset, val_dataset = create_dataset(config)
    model = create_training_model(config)

    callbacks, _ = create_common_callbacks(
        model_name=str(config.experiment_name),
        results_dir_prefix="sam",
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
    # `set_memory_growth` is a repo-wide no-op and TF pre-allocates ~85% of the
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
    train_sam(config)
    logger.info("SAM training completed.")


if __name__ == "__main__":
    main()
