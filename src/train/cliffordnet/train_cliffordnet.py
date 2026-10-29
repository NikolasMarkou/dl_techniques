"""
Training script for the CliffordNet vision backbones.

Trains CliffordNet (isotropic) or HierarchicalCliffordNet (``--variant hier_*`` or
``custom_hier``) on CIFAR-10, CIFAR-100, Imagenette or ImageNet-1k (``--dataset``).
The CIFAR recipe follows the protocol from
arXiv:2601.06793v2 and its authors' train.py (github.com/ParaMind2025/CAN):
200 epochs, batch 128, AdamW (lr 1e-3, weight decay 0.1), cosine decay with NO
warm-up ("without using warm-up training epochs"), DropPath 0.3 (0.4 for cn_64_5),
AutoAugment (CIFAR-10 policy), random flip/crop, per-channel normalisation, and
random erasing 0.25. Remaining differences from the authors' run: the cosine here
decays per step and ends at 1e-5 (alpha 0.01), theirs per epoch to 0.

The augmentation is ``train.common.image_augment``'s (``--augment``, default
``cifar_autoaugment`` on CIFAR: flip, pad-4 crop, the CIFAR-10 AutoAugment policy, random
erasing), seeded from ``--seed`` and applied to the [0, 1] image before standardization.
It is a hook of ``load_classification_data``, so ``--augment`` also works on ImageNet and
Imagenette on top of their random-resized-crop.

Label smoothing is opt-in (``--label-smoothing``, off in the paper recipe).

``--dataset imagenet`` trains on the extracted ImageNet-1k (``--imagenet-root``,
``--devkit-dir``) through ``train.common.imagenet``: random-resized-crop and flip,
ImageNet mean/std, no AutoAugment and no random erasing. The end-of-run figures,
model analysis and save/reload check use the first ``--eval-samples`` val images;
training-time validation and the final evaluation use all 50,000.

``--dataset imagenette`` trains on the prepared TFDS Imagenette records at ``--image-size``
through the shared loader: random-resized-crop and flip, ImageNet mean/std, selection on a
seeded ``--validation-split`` slice of train, the labelled validation split as the test set.
``--image-size`` defaults to 224 for Imagenette and ImageNet. The isotropic variants keep
the full input resolution in every block, so ``nano`` at 224 px and batch 32 asked XLA for a
34.6 GB buffer and ran out of memory on the 24 GB RTX 4090; pass ``--image-size 128``.

The length of a run is ``--epochs`` (default 200) or, exclusively, ``--train-steps N``: N
optimizer steps as ``N // K`` rounds of ``K = --eval-every-steps`` (default 250, must divide
N) over a repeating train stream. Validation, the best checkpoint, the dashboard and
``training_log.csv`` work per round, the cosine schedule spans exactly N steps, and a run of
fewer steps than one epoch (CIFAR at batch 32 is 1,406 steps per epoch) is well defined.
``config.json`` records both values and ``training_summary.txt`` reports "Steps" instead of
"Epochs".

A run writes its directory under ``--output-dir`` through ``train.common``: ``config.json``,
``training_log.csv``, ``training_history.json``, ``best_model.keras`` and the final model,
``visualizations/`` (dashboard, confusion matrix and the other classification figures),
``model_analysis/``, ``epoch_analysis/`` when ``--epoch-analysis-every`` is positive, and
``training_summary.txt``. The best checkpoint is selected on ``val_accuracy`` of the held-out
slice of train; the test split scores it once.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.classification import get_spec
from dl_techniques.models.vision.cliffordnet import CliffordNet, HierarchicalCliffordNet
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

from train.common import (
    setup_gpu,
    set_seeds,
    create_base_argument_parser,
    require_imagenet_root,
    create_callbacks,
    validate_model_loading,
    run_model_analysis,
    save_config_json,
)
from train.common import run_summary
from train.common.callbacks import LearningRateLogger
from train.common.classification_viz import TrainingDashboardCallback
from train.common.conformal import add_conformal_arguments, run_conformal
from train.common.classification_data import (
    add_selection_arguments,
    load_classification_data,
)
from train.common.image_augment import add_augmentation_arguments, build_augment
from train.common.run_io import save_training_history_json
from train.convnext.common import build_loss


# ---------------------------------------------------------------------

Variant = str  # a key of CliffordNet.MODEL_VARIANTS, or "custom"

# DropPath of the authors' networks.py: 0.4 for cliffordnet_64_5, 0.3 for the rest.
_DEFAULT_STOCHASTIC_DEPTH = 0.3
_STOCHASTIC_DEPTH_BY_VARIANT: Dict[str, float] = {
    "cn_64_5": 0.4,
    # The authors' HierarchicalCliffordNet constructor default.
    "hier_345": 0.1,
    "custom_hier": 0.1,
}
# Variants built by HierarchicalCliffordNet: the presets of its own table plus "custom_hier".
_HIERARCHICAL_VARIANTS = (*HierarchicalCliffordNet.MODEL_VARIANTS, "custom_hier")


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------


def _parse_int_list(text: Optional[str], flag: str) -> Optional[List[int]]:
    """Parse a comma-separated integer list; ``None`` or empty gives ``None``.

    :param text: The flag's value.
    :param flag: The flag name, for the error message.
    :raises ValueError: If an entry is not an integer.
    """
    if not text:
        return None
    try:
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid {flag} value {text!r}. Expected comma-separated integers."
        ) from exc


def build_model(
    variant: Variant,
    num_classes: int,
    channels: int,
    depth: int,
    shifts: List[int],
    cli_mode: str,
    ctx_mode: str,
    use_global_context: bool,
    layer_scale_init: float,
    stochastic_depth_rate: float,
    dropout_rate: float,
    weight_decay: float,
    stage_depths: Optional[List[int]] = None,
    stage_dims: Optional[List[int]] = None,
) -> keras.Model:
    """Instantiate a CliffordNet or HierarchicalCliffordNet model.

    ``channels``, ``depth`` and ``use_global_context`` belong to the isotropic
    variants; ``stage_depths`` and ``stage_dims`` to ``custom_hier``. The
    hierarchical model has no global branch, so it refuses
    ``use_global_context=True`` instead of dropping it.

    Note: weight decay is handled solely by the AdamW optimiser
    (decoupled weight decay), matching the reference PyTorch recipe.
    No L2 kernel regularizer is applied — that would double-penalise.
    """
    shared_kwargs: Dict[str, Any] = dict(
        cli_mode=cli_mode,
        ctx_mode=ctx_mode,
        use_global_context=use_global_context,
        layer_scale_init=layer_scale_init,
        stochastic_depth_rate=stochastic_depth_rate,
        dropout_rate=dropout_rate,
    )

    if variant in _HIERARCHICAL_VARIANTS:
        if use_global_context:
            raise ValueError(
                f"{variant!r} is hierarchical and has no global-context branch"
            )
        hier_kwargs = {k: v for k, v in shared_kwargs.items() if k != "use_global_context"}
        if variant == "custom_hier":
            model = HierarchicalCliffordNet(
                num_classes=num_classes,
                stage_depths=stage_depths,
                stage_dims=stage_dims,
                shifts=shifts,
                **hier_kwargs,
            )
        else:
            model = HierarchicalCliffordNet.from_variant(
                variant, num_classes=num_classes, **hier_kwargs
            )
    elif variant in CliffordNet.MODEL_VARIANTS:
        model = CliffordNet.from_variant(
            variant, num_classes=num_classes, **shared_kwargs
        )
    elif variant == "custom":
        model = CliffordNet(
            num_classes=num_classes,
            channels=channels,
            depth=depth,
            shifts=shifts,
            **shared_kwargs,
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    logger.info(
        f"Created CliffordNet-{variant} | classes={num_classes} | "
        f"cli={cli_mode}, ctx={ctx_mode}, global={use_global_context}"
    )
    return model


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------


def train_model(args) -> None:
    """End-to-end training routine for CliffordNet.

    Loads the dataset through ``train.common.classification_data``, builds the model, fits
    it for ``args.epochs`` epochs, or for ``args.train_steps`` steps in ``args.epochs``
    rounds of ``args.eval_every_steps`` over a repeating stream, then saves and reloads the
    final model, scores the best checkpoint on the test split, writes the classification
    figures and the model analysis, and writes ``training_summary.txt``.

    :param args: The parsed and resolved namespace of :func:`main`. ``args.epochs`` is
        always an integer here; with ``--train-steps`` it is ``train_steps // eval_every_steps``.
    :raises Exception: Any failure of loading, fitting, saving or analysis propagates; a run
        never exits 0 with a missing artifact.
    """
    logger.info("Starting CliffordNet training")
    setup_gpu(gpu_id=args.gpu)
    set_seeds(args.seed)

    # ---- Data --------------------------------------------------------
    # x_eval / y_eval are the in-memory test images the end-of-run figures, the model
    # analysis and the save/reload check read (standardized, like the datasets that feed
    # fit and evaluate); the augmentation is train.common.image_augment's, on the [0, 1]
    # image before the standardization.
    augment = build_augment(
        args.augment, random_erasing_prob=args.random_erasing_prob,
        fill=get_spec(args.dataset).mean)
    load_kwargs: Dict[str, Any] = dict(
        batch_size=args.batch_size, seed=args.seed, normalize="standardize",
        max_samples=args.max_samples, augment=augment)
    if args.dataset == "imagenet":
        require_imagenet_root(args.imagenet_root)
        load_kwargs.update(
            imagenet_augment=True, holdout_per_class=args.holdout_per_class,
            eval_samples=args.eval_samples, imagenet_root=args.imagenet_root,
            devkit_dir=args.devkit_dir, image_size=args.image_size)
    elif args.dataset == "imagenette":
        load_kwargs.update(
            imagenet_augment=True, crop_policy="random_resized_crop",
            validation_split=args.validation_split, eval_samples=args.eval_samples,
            image_size=args.image_size)
    else:
        load_kwargs.update(validation_split=args.validation_split)
    data = load_classification_data(args.dataset, **load_kwargs)
    # Selection (fit validation, the best checkpoint) is a held-out slice of train; the
    # dataset's own test split is the test set.
    train_ds, val_ds, test_ds = data.train_ds, data.select_ds, data.test_ds
    steps_per_epoch = data.steps_per_epoch
    num_classes = data.num_classes
    input_shape = data.input_shape
    class_names = data.class_names
    x_eval, y_eval = data.x_fig, data.y_fig
    # The images are standardized, so the figures un-standardize them.
    figure_kwargs = dict(mean=data.mean, std=data.std)
    # fit and the final evaluate see all of their split; the figures see the head.
    figure_ds = tf.data.Dataset.from_tensor_slices(x_eval).batch(args.batch_size)
    file_based = args.dataset in ("imagenet", "imagenette")

    # ---- Shifts and stage lists parsing ------------------------------
    shifts: List[int] = _parse_int_list(args.shifts, "--shifts") or [1, 2]
    stage_depths = _parse_int_list(args.stage_depths, "--stage-depths")
    stage_dims = _parse_int_list(args.stage_dims, "--stage-dims")

    # ---- Model -------------------------------------------------------
    model = build_model(
        variant=args.variant,
        num_classes=num_classes,
        channels=args.channels,
        depth=args.depth,
        shifts=shifts,
        stage_depths=stage_depths,
        stage_dims=stage_dims,
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context=args.use_global_context,
        layer_scale_init=args.layer_scale_init,
        stochastic_depth_rate=args.stochastic_depth_rate,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
    )

    # Warm-up build
    _ = model(np.zeros((1,) + input_shape, dtype="float32"), training=False)
    model.summary(print_fn=logger.info)

    # ---- Optimiser & schedule ----------------------------------------
    # --train-steps: the run is `epochs` virtual epochs of `eval_every_steps` optimizer
    # steps over a repeating train stream, so the schedule spans exactly train_steps.
    by_steps = args.train_steps is not None
    total_steps = args.train_steps if by_steps else args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = max(1, total_steps - warmup_steps)

    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.learning_rate,
        "decay_steps": decay_steps,
        "alpha": 1e-2,
        "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8,
    })

    optimizer = optimizer_builder({
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": args.weight_decay,
    }, lr_schedule)

    logger.info(
        f"Schedule: cosine_decay | total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}, decay_steps={decay_steps}, "
        f"peak_lr={args.learning_rate}"
    )

    # ---- Compile -----------------------------------------------------
    metrics: List[Any] = ["accuracy"]
    if num_classes > 10:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")
        )

    if args.label_smoothing > 0.0:
        logger.info(
            f"Label smoothing {args.label_smoothing:g}: loss and val_loss include "
            "the smoothing and are not comparable with unsmoothed runs"
        )
    model.compile(
        optimizer=optimizer,
        loss=build_loss(args.label_smoothing),
        metrics=metrics,
    )

    # ---- Callbacks ---------------------------------------------------
    custom_objects = {
        "CliffordNet": CliffordNet,
        "HierarchicalCliffordNet": HierarchicalCliffordNet,
    }
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.variant}_{args.dataset}",
        results_dir_prefix="cliffordnet",
        output_root=args.output_dir,
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
        # DECISION plan-2026-10-23T100412-e7eaaa02/D-002
        # Per-epoch analysis stays opt-in: do NOT make it the default again.
        # The shared default (every epoch) was removed in b6903e109 for cost.
        include_analyzer=args.epoch_analysis_every > 0,
        analyzer_epoch_frequency=max(1, args.epoch_analysis_every),
        # start at N so the snapshots land on epochs N, 2N, ... (default start 1
        # would give 1, N+1, ...).
        analyzer_start_epoch=max(1, args.epoch_analysis_every),
    )
    callbacks.insert(0, keras.callbacks.TerminateOnNaN())
    # Standard classification visualizations (src/train/AGENTS.md): the learning
    # rate reaches the CSV and the dashboard, which is drawn last and redrawn on a
    # cadence into <run>/visualizations/.
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    callbacks.insert(0, LearningRateLogger(at_epoch_start=True))

    def _baseline(untrained: keras.Model) -> Dict[str, float]:
        measured = untrained.evaluate(val_ds, verbose=0, return_dict=True)
        return {"loss": measured["loss"], "accuracy": measured["accuracy"]}

    callbacks.append(TrainingDashboardCallback(
        out_path=os.path.join(vis_dir, "training_dashboard.png"),
        baseline_fn=_baseline,
        title=f"cliffordnet_{args.variant}_{args.dataset} (seed {args.seed})",
    ))

    # DECISION plan-2026-10-14T062616-81938e81/D-177: the config is written here,
    # before training, and NOT wrapped. A failed write raises before any compute
    # is spent. Do NOT move it back after fit inside try/except-and-warn: the run
    # then exited 0 with config.json missing. See decisions.md D-177.
    save_config_json({
        "dataset": args.dataset, "variant": args.variant,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "train_steps": args.train_steps, "eval_every_steps": args.eval_every_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stochastic_depth_rate": args.stochastic_depth_rate,
        "dropout_rate": args.dropout_rate,
        "label_smoothing": args.label_smoothing,
        "cli_mode": args.cli_mode, "ctx_mode": args.ctx_mode,
        "use_global_context": args.use_global_context,
        "warmup_epochs": args.warmup_epochs,
        "augment": args.augment,
        "random_erasing_prob": args.random_erasing_prob,
        "stage_depths": stage_depths, "stage_dims": stage_dims,
        "epoch_analysis_every": args.epoch_analysis_every,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        **({"image_size": args.image_size, "validation_split": args.validation_split,
            "eval_samples": args.eval_samples}
           if args.dataset == "imagenette" else {}),
        **({"imagenet_root": args.imagenet_root, "devkit_dir": args.devkit_dir,
            "eval_samples": args.eval_samples,
            "conformal_alpha": None if args.no_conformal else args.conformal_alpha}
           if args.dataset == "imagenet" else {}),
    }, results_dir)

    logger.info(
        f"Training | dataset={args.dataset}, variant={args.variant}, "
        f"epochs={args.epochs}, batch={args.batch_size}, lr={args.learning_rate}"
        + (f", train_steps={args.train_steps} (every {args.eval_every_steps})"
           if by_steps else "")
    )

    # ---- Training ----------------------------------------------------
    history = model.fit(
        train_ds.repeat() if by_steps else train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.eval_every_steps if by_steps else None,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save final model & validate ---------------------------------
    test_sample = x_eval[:4]
    pre_save_output = model.predict(test_sample, verbose=0)

    final_path = os.path.join(
        results_dir,
        f"cliffordnet_{args.variant}_{args.dataset}_final.keras",
    )
    # A save or reload error surfaces (D-085); a value mismatch is logged by
    # validate_model_loading itself.
    model.save(final_path)
    logger.info(f"Final model saved to: {final_path}")
    # After the save: a failed history write raises (D-174).
    save_training_history_json(history, results_dir)
    validate_model_loading(final_path, test_sample, pre_save_output, custom_objects)

    # ---- Load best checkpoint & evaluate -----------------------------
    best_path = os.path.join(results_dir, "best_model.keras")
    best_model = model
    if os.path.exists(best_path):
        # A best checkpoint that exists but does not load raises (D-183); the old
        # fallback scored the last-epoch weights under the best checkpoint's name.
        best_model = keras.models.load_model(best_path, custom_objects=custom_objects)
        logger.info("Loaded best checkpoint for final evaluation")

    test_results = best_model.evaluate(test_ds, verbose=1, return_dict=True)
    logger.info(f"Test results: {test_results}")

    # ---- Conformal prediction sets (ImageNet), over ALL of the official val split ----
    # Before the figures and the model analysis, which are the heavy steps. The checkpoint
    # was selected on the held-out slice of train, not on this split.
    conformal: Optional[Dict[str, Any]] = None
    if args.dataset == "imagenet" and not args.no_conformal:
        conformal = run_conformal(
            best_model, test_ds, alpha=args.conformal_alpha, seed=args.seed,
            out_dir=Path(results_dir), input_shape=input_shape,
            model_name=f"cliffordnet_{args.variant}_imagenet",
            weights="best checkpoint, selected on the held-out slice of train "
                    "(val_accuracy)" if os.path.exists(best_path) else "last epoch",
            selection_overlaps_calibration=False,
        )

    # ---- Classification figures --------------------------------------
    # The model emits logits; every probability consumer gets a softmax. The eval
    # data is ordered, so the predictions line up with y_eval. x_eval is
    # standardized, so the figures get the mean/std to un-standardize it.
    logits = best_model.predict(figure_ds, verbose=0)
    probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))
    run_summary.write_classification_figures(
        Path(vis_dir), x_eval, y_eval, probs, class_names, **figure_kwargs,
    )

    # ---- Post-training analysis --------------------------------------
    run_model_analysis(
        model=best_model,
        test_data=(x_eval, y_eval),
        training_history=history,
        model_name=f"cliffordnet_{args.variant}_{args.dataset}",
        results_dir=results_dir,
    )

    # ---- Training summary --------------------------------------------
    val_acc_history: List[float] = history.history.get("val_accuracy", [])
    best_val_acc = max(val_acc_history) if val_acc_history else float("nan")
    trained_epochs = len(history.history.get("loss", []))

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CliffordNet Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Num classes: {num_classes}\n")
        if args.dataset == "imagenette":
            f.write(f"Image size: {args.image_size}\n")
        if args.dataset == "imagenet":
            f.write(f"ImageNet root: {args.imagenet_root}\n")
        f.write(f"Augmentation: {args.augment}"
                + (f" (random erasing {args.random_erasing_prob})"
                   if args.augment == "cifar_autoaugment" else "") + "\n")
        if file_based:
            f.write(f"Figures and analysis on the first {len(x_eval)} val images\n")
        f.write(f"Parameters: {model.count_params():,}\n\n")
        if by_steps:
            f.write(f"Steps: {trained_epochs * args.eval_every_steps} "
                    f"({trained_epochs} x {args.eval_every_steps}), "
                    f"Batch: {args.batch_size}\n")
        else:
            f.write(f"Epochs: {trained_epochs}, Batch: {args.batch_size}\n")
        f.write(f"LR: {args.learning_rate}, Weight decay: {args.weight_decay}\n")
        f.write(f"Stochastic depth: {args.stochastic_depth_rate}\n")
        f.write(f"Label smoothing: {args.label_smoothing}\n\n")
        f.write("Test Results:\n")
        for key, val in test_results.items():
            f.write(f"  {key}: {val:.4f}\n")
        if conformal is not None:
            splits = conformal["repeated_splits"]
            f.write(
                f"\nConformal sets (alpha {conformal['alpha']}, tau {conformal['tau']}): "
                f"coverage {splits['coverage_mean']:.4f} on random half splits "
                f"(min {splits['coverage_min']:.4f}), mean set size "
                f"{splits['mean_size']:.2f}\n"
            )
        f.write(f"\nBest val_accuracy: {best_val_acc:.4f}\n")

    logger.info(f"Summary written to: {summary_path}")
    logger.info(f"Training complete. Best val_accuracy={best_val_acc:.4f}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    """Entry point for the CliffordNet training script.

    Parses the command line and resolves every flag that depends on another: the
    augmentation preset and random-erasing probability per dataset, ``--eval-samples``
    (2000 for Imagenette and ImageNet, refused for CIFAR), the model flags from the variant's
    preset when not given, the DropPath rate per variant, and the length of the run
    (``--epochs`` defaulting to 200, or ``--train-steps`` with ``--eval-every-steps``
    defaulting to 250, the two refused together). A flag that does not apply to the chosen
    dataset or architecture is a parser error, never ignored. Then runs :func:`train_model`.
    """
    parser = create_base_argument_parser(
        description="Train CliffordNet (geometric-algebra vision backbone)",
        default_dataset="cifar100",
        dataset_choices=["cifar10", "cifar100", "imagenette", "imagenet"],
    )
    add_selection_arguments(parser)
    # None means "not given": --eval-samples is refused for the CIFAR datasets (below).
    parser.set_defaults(eval_samples=None)
    add_conformal_arguments(parser, note="Used only when --dataset imagenet: ")

    # Model variant
    parser.add_argument('--variant', type=str, default='nano',
                        choices=[*CliffordNet.MODEL_VARIANTS,
                                 *HierarchicalCliffordNet.MODEL_VARIANTS,
                                 'custom', 'custom_hier'],
                        help='A preset of CliffordNet.MODEL_VARIANTS or of '
                             'HierarchicalCliffordNet.MODEL_VARIANTS, custom '
                             '(isotropic, --channels/--depth/--shifts) or '
                             'custom_hier (--stage-depths/--stage-dims/--shifts).')

    # Custom variant overrides
    # None means "not given"; the isotropic variants resolve them to 128 / 12 in main().
    parser.add_argument('--channels', type=int, default=None,
                        help='Feature dimension D (custom variant only; default 128).')
    parser.add_argument('--depth', type=int, default=None,
                        help='Number of blocks L (custom variant only; default 12).')
    parser.add_argument('--stage-depths', type=str, default=None, dest='stage_depths',
                        help="custom_hier only: blocks per stage, e.g. '3,4,5'.")
    parser.add_argument('--stage-dims', type=str, default=None, dest='stage_dims',
                        help="custom_hier only: width per stage, e.g. '32,64,128' "
                             "(default: 32 doubling per stage).")
    parser.add_argument('--shifts', type=str, default=None,
                        help="Comma-separated shift offsets, e.g. '1,2,4,8,16'.")

    # Algebraic / context options
    # None means "the preset's value" (custom: full / diff / off), resolved in main().
    parser.add_argument('--cli-mode', type=str, default=None,
                        choices=['inner', 'wedge', 'full'], dest='cli_mode')
    parser.add_argument('--ctx-mode', type=str, default=None,
                        choices=['diff', 'abs'], dest='ctx_mode')
    parser.add_argument('--use-global-context', action='store_true', default=None,
                        dest='use_global_context',
                        help='Add the gFFN-G branch (on in lite_g and nano_g).')

    # Regularisation
    parser.add_argument('--layer-scale-init', type=float, default=1e-5,
                        dest='layer_scale_init')
    parser.add_argument('--stochastic-depth-rate', type=float, default=None,
                        dest='stochastic_depth_rate',
                        help='Maximum DropPath rate, linearly ramped over depth. '
                             'Default: 0.4 for cn_64_5, else 0.3.')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        dest='dropout_rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        dest='label_smoothing',
                        help='Label smoothing in [0, 1) (0 = off, the stock loss). '
                             'Above 0 the loss and val_loss include the smoothing '
                             'and are not comparable with unsmoothed runs.')

    # Augmentation
    add_augmentation_arguments(parser)
    parser.add_argument('--random-erasing-prob', type=float, default=None,
                        dest='random_erasing_prob',
                        help='Random-erasing probability of --augment cifar_autoaugment '
                             '(default 0.25); refused with any other preset.')

    # Analysis
    parser.add_argument('--epoch-analysis-every', type=int, default=0,
                        dest='epoch_analysis_every',
                        help='Run the data-free weight/spectral model analyzer '
                             'every N epochs into <run>/epoch_analysis/ '
                             '(0 = off).')

    # Step budget
    parser.add_argument('--train-steps', type=int, default=None, dest='train_steps',
                        help='Train for exactly this many optimizer steps instead of '
                             '--epochs: virtual epochs of --eval-every-steps over a '
                             'repeating train stream, the LR schedule spans the budget.')
    parser.add_argument('--eval-every-steps', type=int, default=None,
                        dest='eval_every_steps',
                        help='Steps between validations with --train-steps '
                             '(default 250; must divide --train-steps).')

    # Warmup
    parser.add_argument('--warmup-epochs', type=int, default=0, dest='warmup_epochs',
                        help='Linear warm-up epochs (paper v2: none).')

    # Output
    parser.add_argument('--output-dir', type=str, default='results', dest='output_dir',
                        help='Directory under which the timestamped run directory '
                             'is created (default: results).')

    # Override defaults from base parser to match the reference recipe
    parser.set_defaults(
        weight_decay=0.1,
        batch_size=128,
        epochs=None,
        patience=30,
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for Python, NumPy and Keras (train.common.set_seeds)')

    args = parser.parse_args()
    if args.train_steps is None:
        if args.eval_every_steps is not None:
            parser.error("--eval-every-steps only applies with --train-steps")
        args.epochs = 200 if args.epochs is None else args.epochs
    else:
        if args.epochs is not None:
            parser.error("--train-steps and --epochs are mutually exclusive")
        if args.train_steps <= 0:
            parser.error(f"--train-steps must be positive, got {args.train_steps}")
        args.eval_every_steps = 250 if args.eval_every_steps is None else args.eval_every_steps
        if args.eval_every_steps <= 0 or args.train_steps % args.eval_every_steps:
            parser.error(f"--eval-every-steps {args.eval_every_steps} must be positive "
                         f"and divide --train-steps {args.train_steps}")
        args.epochs = args.train_steps // args.eval_every_steps
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error(f"--label-smoothing must be in [0, 1), got {args.label_smoothing}")
    # A flag that does not apply to the chosen dataset is refused, never ignored.
    # The CIFAR recipe of the paper is the default there; ImageNet and Imagenette keep the
    # loader's random-resized-crop and flip unless a preset is asked for on top.
    if args.augment is None:
        args.augment = "cifar_autoaugment" if args.dataset.startswith("cifar") else "none"
    if args.random_erasing_prob is not None and args.augment != "cifar_autoaugment":
        parser.error("--random-erasing-prob only applies with --augment cifar_autoaugment")
    if args.random_erasing_prob is None:
        args.random_erasing_prob = 0.25 if args.augment == "cifar_autoaugment" else 0.0
    if args.dataset in ("imagenet", "imagenette"):
        args.eval_samples = 2000 if args.eval_samples is None else args.eval_samples
        if args.eval_samples <= 0:
            parser.error(f"--eval-samples must be positive, got {args.eval_samples}")
    elif args.eval_samples is not None:
        parser.error("--eval-samples only applies with --dataset imagenet or imagenette")
    # DECISION plan-2026-10-23T100412-e7eaaa02/D-009
    # Flags the user did not pass take the PRESET's value. Do NOT give these
    # three flags their own defaults again: always passing cli_mode='full' and
    # use_global_context=False as overrides made `--variant lite_g` train without
    # its global branch and would have turned cn_32_5 / cn_64_5 into 'full'.
    hierarchical = args.variant in _HIERARCHICAL_VARIANTS
    # A flag that does not apply to the chosen architecture is refused, never ignored.
    if hierarchical:
        for flag, given in (("--channels", args.channels), ("--depth", args.depth),
                            ("--use-global-context", args.use_global_context)):
            if given is not None:
                parser.error(
                    f"{flag} does not apply to the hierarchical variant "
                    f"{args.variant!r} (it has no such parameter)"
                )
    if args.variant != "custom_hier":
        for flag, given in (("--stage-depths", args.stage_depths),
                            ("--stage-dims", args.stage_dims)):
            if given is not None:
                parser.error(f"{flag} only applies with --variant custom_hier")
    if not hierarchical:
        args.channels = 128 if args.channels is None else args.channels
        args.depth = 12 if args.depth is None else args.depth
    preset = {
        **CliffordNet.MODEL_VARIANTS, **HierarchicalCliffordNet.MODEL_VARIANTS
    }.get(args.variant, {})
    for name, fallback in (("cli_mode", "full"), ("ctx_mode", "diff"),
                           ("use_global_context", False)):
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, fallback))
    if args.stochastic_depth_rate is None:
        args.stochastic_depth_rate = _STOCHASTIC_DEPTH_BY_VARIANT.get(
            args.variant, _DEFAULT_STOCHASTIC_DEPTH
        )

    try:
        train_model(args)
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
