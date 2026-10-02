"""
LightGlue Training Script (stage 1: homography pairs)
====================================================================

Trains `LightGlue` (`dl_techniques.models.vision.keypoints.lightglue`) on homography
pairs made from a folder of ordinary photographs (COCO ``train2017``), with a FROZEN,
already trained SuperPoint as the keypoint and descriptor source. Pattern 4 plus the
hkan CLI and run-artifact shell.

Each step (see `train.lightglue.pipeline`): the frozen SuperPoint runs on both images
inside the graph, keypoints are decoded (NMS, threshold, top-k, padding mask), the
ground-truth matches come from the sampled homography, LightGlue is called with the
padding masks and the objective is registered with `add_loss`. Stock `fit` with no
`y` and no custom `train_step`.

The SuperPoint checkpoint fixes the image size (its descriptor map is resized to the
construction-time size) and the LightGlue input width (its `descriptor_dim`); both are
derived from it. There is no preset or smoke flag: a quick plumbing run passes explicit
small values for the size flags.

Each run writes one directory under repo-root `results/` (or `--output-dir`):
`config.json`, `run.log`, `training_log.csv`, `training_history.json`, `best_model.keras`
(the LightGlue ALONE at the best monitored epoch), `lightglue.keras` (the LightGlue at
the end of training) and `results_summary.json` (strict JSON). A reused
`--experiment-name` is refused before anything is written.

A float32 front end is required; there is no mixed-precision flag.

Usage:
    python -m train.lightglue.train_lightglue \
        --superpoint-checkpoint results/<superpoint run>/final_model.keras
"""

import argparse
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import keras
import numpy as np

from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from dl_techniques.optimization import learning_rate_schedule_builder, optimizer_builder
from dl_techniques.utils.logger import logger
from train.common import (
    default_experiment_name, prepare_run_dir, save_training_history_json, set_seeds,
    setup_gpu,
)
from train.common.callbacks import best_checkpoint_path, create_callbacks
from train.common.run_artifacts import attach_run_log, refuse_existing_run, write_summary_json
from train.common.run_summary import describe_devices
from train.lightglue.data import list_images, make_pair_dataset
from train.lightglue.pipeline import LightGlueCheckpoint, LightGlueTrainingModel, load_superpoint

# ---------------------------------------------------------------------

# `parents[3]` reaches the repo root from THIS file
# (src/train/lightglue/train_lightglue.py: [0] lightglue, [1] train, [2] src, [3] <repo>).
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "lightglue"
DEFAULT_COCO_DIR = "/media/arxwn/data0_4tb/datasets/coco_2017/train2017"
LIGHTGLUE_MODEL_NAME = "lightglue.keras"
TRAINING_HISTORY_NAME = "training_history.json"
#: Last entry of the validation split's seed list (the bytes of "SPLT"), non-zero
#: because numpy drops trailing zeros of a seed list.
VALIDATION_SPLIT_TAG = 0x53504C54
#: Files whose presence means the experiment directory already holds a run.
RUN_ARTIFACTS = ("results_summary.json", "config.json", "run.log", "training_log.csv",
                 LIGHTGLUE_MODEL_NAME)


@dataclass
class LightGlueTrainConfig:
    """Every setting of one LightGlue stage-1 run (each field is read by `main` or the pipeline).

    :param superpoint_checkpoint: Trained SuperPoint ``.keras`` file (the frozen front end).
    :param coco_dir: Folder of training photographs (jpg/png, not recursive).
    :param val_images: Images held out (fixed split from ``seed``) for ``val_*`` metrics;
        0 trains with the training loss as the monitor.
    :param image_size: Optional square size, checked against the SuperPoint checkpoint
        (which fixes it); ``None`` takes the checkpoint's size.
    :param max_keypoints: Padded keypoints per image.
    :param nms_radius: Detection NMS radius in pixels.
    :param detection_threshold: Minimum heatmap probability of a keypoint.
    :param border: Pixels at the image edge without keypoints.
    :param pos_threshold: Reprojection error (pixels) of a positive pair, also the
        dustbin threshold.
    :param batch_size: Image pairs per step.
    :param epochs: Epochs requested (early stopping may end earlier).
    :param steps_per_epoch: Steps per epoch; ``None`` is one pass over the training images.
    :param validation_steps: Validation batches; ``None`` is the whole held-out set.
    :param learning_rate: Peak learning rate.
    :param weight_decay: Decoupled AdamW weight decay (biases and norm parameters excluded).
    :param warmup_steps: Linear warmup steps before the cosine decay.
    :param clip_norm: Global gradient-norm clip; 0 disables clipping.
    :param patience: Early-stopping patience in epochs.
    :param num_layers: LightGlue layers.
    :param descriptor_dim: LightGlue internal width.
    :param num_heads: LightGlue attention heads.
    :param seed: Seed of the weights, the pair sampling and the validation split.
    :param gpu: GPU index (``CUDA_VISIBLE_DEVICES``), ``None`` for all with memory growth.
    :param output_dir: Base directory of the run, ``None`` for repo-root ``results``.
    :param experiment_name: Run directory name, ``None`` for ``lightglue_<timestamp>``.
    """

    superpoint_checkpoint: str
    coco_dir: str = DEFAULT_COCO_DIR
    val_images: int = 256
    image_size: Optional[int] = None
    max_keypoints: int = 512
    nms_radius: int = 4
    detection_threshold: float = 0.005
    border: int = 4
    pos_threshold: float = 3.0
    batch_size: int = 16
    epochs: int = 10
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    clip_norm: float = 1.0
    patience: int = 5
    num_layers: int = 9
    descriptor_dim: int = 256
    num_heads: int = 4
    seed: int = 42
    gpu: Optional[int] = None
    output_dir: Optional[str] = None
    experiment_name: Optional[str] = None


# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the trainer's raw `argparse.ArgumentParser` (defaults come from the config)."""
    default = LightGlueTrainConfig(superpoint_checkpoint="")
    parser = argparse.ArgumentParser(
        description="Train LightGlue on COCO homography pairs with a frozen SuperPoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--superpoint-checkpoint", type=str, required=True,
                        help="Trained SuperPoint .keras file; it fixes the image size and the "
                             "LightGlue input width.")
    parser.add_argument("--coco-dir", type=str, default=default.coco_dir,
                        help="Folder of training photographs (jpg/png, not recursive).")
    parser.add_argument("--val-images", type=int, default=default.val_images,
                        help="Images held out for validation (fixed split from --seed); 0 "
                             "monitors the training loss.")
    parser.add_argument("--image-size", type=int, default=default.image_size,
                        help="Square size, checked against the SuperPoint checkpoint "
                             "(default: the checkpoint's size).")
    parser.add_argument("--max-keypoints", type=int, default=default.max_keypoints,
                        help="Padded keypoints per image.")
    parser.add_argument("--nms-radius", type=int, default=default.nms_radius,
                        help="Detection NMS radius in pixels.")
    parser.add_argument("--detection-threshold", type=float, default=default.detection_threshold,
                        help="Minimum heatmap probability of a keypoint.")
    parser.add_argument("--border", type=int, default=default.border,
                        help="Pixels at the image edge without keypoints.")
    parser.add_argument("--pos-threshold", type=float, default=default.pos_threshold,
                        help="Reprojection error (pixels) of a positive pair; also the "
                             "dustbin threshold (glue-factory: 3 for both).")
    parser.add_argument("--batch-size", type=int, default=default.batch_size,
                        help="Image pairs per step.")
    parser.add_argument("--epochs", type=int, default=default.epochs, help="Epochs requested.")
    parser.add_argument("--steps-per-epoch", type=int, default=default.steps_per_epoch,
                        help="Steps per epoch (default: one pass over the training images).")
    parser.add_argument("--validation-steps", type=int, default=default.validation_steps,
                        help="Validation batches (default: the whole held-out set).")
    parser.add_argument("--learning-rate", type=float, default=default.learning_rate,
                        help="Peak learning rate (AdamW, linear warmup then cosine decay).")
    parser.add_argument("--weight-decay", type=float, default=default.weight_decay,
                        help="Decoupled AdamW weight decay; biases and norms are excluded.")
    parser.add_argument("--warmup-steps", type=int, default=default.warmup_steps,
                        help="Linear warmup steps.")
    parser.add_argument("--clip-norm", type=float, default=default.clip_norm,
                        help="Global gradient-norm clip; 0 disables it.")
    parser.add_argument("--patience", type=int, default=default.patience,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--num-layers", type=int, default=default.num_layers,
                        help="LightGlue layers (paper: 9).")
    parser.add_argument("--descriptor-dim", type=int, default=default.descriptor_dim,
                        help="LightGlue internal width (paper: 256).")
    parser.add_argument("--num-heads", type=int, default=default.num_heads,
                        help="LightGlue attention heads (paper: 4).")
    parser.add_argument("--seed", type=int, default=default.seed,
                        help="Seed of the weights, the pair sampling and the validation split.")
    parser.add_argument("--gpu", type=int, default=default.gpu, help="GPU index to use.")
    parser.add_argument("--output-dir", type=str, default=default.output_dir,
                        help="Base directory of the run (default: <repo>/results).")
    parser.add_argument("--experiment-name", type=str, default=default.experiment_name,
                        help="Run directory name (default: lightglue_<timestamp>).")
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate the CLI arguments.

    :param argv: Argument list; ``None`` defers to ``sys.argv[1:]``.
    :return: The parsed namespace (one attribute per `LightGlueTrainConfig` field).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    for flag in ("max_keypoints", "batch_size", "epochs", "patience", "num_layers",
                 "descriptor_dim", "num_heads"):
        if getattr(args, flag) < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1, got {getattr(args, flag)}")
    for flag in ("steps_per_epoch", "validation_steps", "image_size"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            parser.error(f"--{flag.replace('_', '-')} must be >= 1, got {value}")
    for flag in ("val_images", "nms_radius", "border", "warmup_steps"):
        if getattr(args, flag) < 0:
            parser.error(f"--{flag.replace('_', '-')} must be >= 0, got {getattr(args, flag)}")
    for flag in ("learning_rate", "pos_threshold"):
        if getattr(args, flag) <= 0:
            parser.error(f"--{flag.replace('_', '-')} must be > 0, got {getattr(args, flag)}")
    if args.weight_decay < 0 or args.clip_norm < 0:
        parser.error("--weight-decay and --clip-norm must be >= 0")
    if not 0.0 <= args.detection_threshold < 1.0:
        parser.error(f"--detection-threshold must be in [0, 1), got {args.detection_threshold}")
    if args.descriptor_dim % args.num_heads or (args.descriptor_dim // args.num_heads) % 2:
        parser.error(
            f"--descriptor-dim ({args.descriptor_dim}) must split into --num-heads "
            f"({args.num_heads}) heads of even width")
    return args


def config_from_args(args: argparse.Namespace) -> LightGlueTrainConfig:
    """Build the config from a namespace whose attributes carry the field names."""
    return LightGlueTrainConfig(
        **{field.name: getattr(args, field.name) for field in fields(LightGlueTrainConfig)})


# ---------------------------------------------------------------------


def split_images(paths: List[str], val_images: int, seed: int):
    """Hold ``val_images`` files out with a seed-fixed permutation.

    :param paths: Sorted image paths.
    :param val_images: Held-out count (0 for none).
    :param seed: Split seed.
    :return: ``(train_paths, val_paths)``, each sorted.
    :raises ValueError: The split would leave no training image.
    """
    if val_images >= len(paths):
        raise ValueError(
            f"--val-images {val_images} leaves no training image out of {len(paths)}")
    order = np.random.default_rng([seed, VALIDATION_SPLIT_TAG]).permutation(len(paths))
    val = sorted(paths[i] for i in order[:val_images])
    train = sorted(paths[i] for i in order[val_images:])
    return train, val


def build_optimizer(config: LightGlueTrainConfig, steps_per_epoch: int):
    """AdamW through `optimizer_builder` with a warmup + cosine schedule.

    Weight decay is applied once, by the optimizer (no kernel regularizer in the model).
    The clipping key is `gradient_clipping_by_norm` (global norm); `optimizer_builder`
    renames its keys, so a literal `clipnorm` would be dropped silently.

    :param config: The run config.
    :param steps_per_epoch: Optimizer steps per epoch.
    :return: The configured optimizer.
    """
    total_steps = steps_per_epoch * config.epochs
    schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": config.learning_rate,
        "decay_steps": max(1, total_steps - config.warmup_steps),
        "warmup_steps": config.warmup_steps,
        "alpha": 0.01,
    })
    optimizer_config: Dict[str, Any] = {
        "type": "adamw",
        "weight_decay": config.weight_decay,
        "exclude_from_weight_decay": ["bias", "gamma", "beta"],
    }
    if config.clip_norm > 0:
        optimizer_config["gradient_clipping_by_norm"] = config.clip_norm
    return optimizer_builder(optimizer_config, schedule)


# ---------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the LightGlue trainer.

    The first statement parses argv, so ``--help`` exits 0 with a ``usage:`` line
    before any GPU, dataset or model work.

    :param argv: Argument list; ``None`` defers to ``sys.argv[1:]``.
    :return: Process exit code: 0 on success, 2 when a preflight check (image folder,
        SuperPoint checkpoint, image size, validation split) fails before any file is written.
    """
    args = parse_arguments(argv)
    config = config_from_args(args)

    setup_gpu(gpu_id=config.gpu)

    base_dir = Path(config.output_dir) if config.output_dir else REPO_ROOT / "results"
    run_dir = base_dir / (config.experiment_name or default_experiment_name(EXPERIMENT_NAME))
    refuse_existing_run(run_dir, artifact_names=RUN_ARTIFACTS)

    # Preflight: everything that can be wrong with the inputs, before the run directory
    # exists, so a bad path does not use up the experiment name.
    try:
        paths = list(list_images(config.coco_dir))
        train_paths, val_paths = split_images(paths, config.val_images, config.seed)
        requested = None if config.image_size is None else (config.image_size, config.image_size)
        superpoint = load_superpoint(config.superpoint_checkpoint, image_size=requested)
        if len(val_paths) and len(val_paths) < config.batch_size:
            raise ValueError(
                f"{len(val_paths)} validation images cannot fill one batch of {config.batch_size}")
    except (FileNotFoundError, ValueError, TypeError) as error:
        logger.error(f"Preflight failed, nothing written: {error}")
        return 2

    image_size = (superpoint.input_height, superpoint.input_width)
    steps_per_epoch = config.steps_per_epoch or max(1, len(train_paths) // config.batch_size)
    validation_steps = None
    if val_paths:
        validation_steps = len(val_paths) // config.batch_size
        if config.validation_steps is not None:
            validation_steps = min(validation_steps, config.validation_steps)

    prepare_run_dir(config, output_dir=run_dir)
    with attach_run_log(run_dir):
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Config: {config}")
        set_seeds(config.seed)

        lightglue = LightGlue(
            input_dim=superpoint.descriptor_dim, descriptor_dim=config.descriptor_dim,
            num_layers=config.num_layers, num_heads=config.num_heads)
        model = LightGlueTrainingModel(
            superpoint, lightglue, max_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold, nms_radius=config.nms_radius,
            border=config.border, pos_threshold=config.pos_threshold)

        train_ds = make_pair_dataset(
            train_paths, image_size, config.batch_size, seed=config.seed, repeat=True,
            ignore_errors=True)
        val_ds = None if not val_paths else make_pair_dataset(
            val_paths, image_size, config.batch_size, seed=config.seed + 1, shuffle=False)

        # DECISION plan-2026-10-02T084508-dd2c07ac/D-014
        # jit_compile=False and no loss: the objective is add_loss inside the wrapper's
        # call. Do NOT add a custom train_step "to reach the labels"; labels and loss are
        # computed in call() precisely so stock fit runs. See decisions.md D-014.
        model.compile(optimizer=build_optimizer(config, steps_per_epoch), jit_compile=False)
        model.build({"image0": (None, *image_size, 1)})

        monitor = "val_loss" if val_paths else "loss"
        callbacks, results_dir = create_callbacks(
            model_name=run_dir.name, results_dir_prefix=EXPERIMENT_NAME, run_dir=str(run_dir),
            monitor=monitor, patience=config.patience, use_lr_schedule=True,
            include_terminate_on_nan=True, include_analyzer=False)
        # The stock ModelCheckpoint would save the wrapper with the frozen SuperPoint at
        # every improvement; the LightGlue alone is the artifact (D-014).
        callbacks = [cb for cb in callbacks if not isinstance(cb, keras.callbacks.ModelCheckpoint)]
        checkpoint = LightGlueCheckpoint(best_checkpoint_path(results_dir), monitor=monitor)
        callbacks.append(checkpoint)

        sample = next(iter(train_ds))
        label_stats = model.batch_statistics(sample)
        logger.info(f"Label statistics of one training batch: {label_stats}")

        lightglue_params = int(sum(int(np.prod(w.shape)) for w in lightglue.trainable_weights))
        superpoint_params = int(superpoint.count_params())
        logger.info(
            f"LightGlue trainable params {lightglue_params:,}; frozen SuperPoint {superpoint_params:,}")

        start = time.time()
        history = model.fit(
            train_ds, epochs=config.epochs, steps_per_epoch=steps_per_epoch,
            validation_data=val_ds, validation_steps=validation_steps,
            callbacks=callbacks, verbose=2)
        fit_seconds = time.time() - start
        save_training_history_json(history, run_dir)

        final_path = run_dir / LIGHTGLUE_MODEL_NAME
        lightglue.save(str(final_path))
        reloaded = keras.saving.load_model(str(final_path), compile=False)
        reload_verified = all(
            np.array_equal(a, b) for a, b in zip(lightglue.get_weights(), reloaded.get_weights()))
        if not reload_verified:
            raise RuntimeError(f"{final_path} did not reload with identical weights")

        last = {key: float(values[-1]) for key, values in history.history.items() if values}
        summary: Dict[str, Any] = {
            "model": EXPERIMENT_NAME,
            "seed": config.seed,
            "image_size": list(image_size),
            "epochs_requested": config.epochs,
            "epochs_run": len(history.history.get("loss", [])),
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "batch_size": config.batch_size,
            "train_images": len(train_paths),
            "val_images": len(val_paths),
            "params": {"lightglue_trainable": lightglue_params, "superpoint_frozen": superpoint_params},
            "devices": describe_devices(),
            "monitor": monitor,
            "final": last,
            "best_checkpoint": {
                "monitor": monitor, "value": checkpoint.best, "epoch": checkpoint.best_epoch,
                "path": best_checkpoint_path(results_dir) if checkpoint.best is not None else None,
            },
            "label_statistics": label_stats,
            "lightglue_model": str(final_path),
            "lightglue_reload_verified": reload_verified,
            "fit_seconds": fit_seconds,
        }
        write_summary_json(run_dir, summary)
        logger.info(f"Done: final {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------
