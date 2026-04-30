"""
Training script comparing vanilla ``CliffordNetBlock`` against the new
``CliffordNetBlockDS`` variant on CIFAR-100.

Both backbones are isotropic 5-block stacks at a single channel resolution
(no in-backbone downsampling). The DS variant differs only in its context
stream:

- Vanilla: two stacked 3x3 ``DepthwiseConv2D`` + one ``BatchNormalization``
  (effective 7x7 receptive field).
- DS:      one 7x7 ``DepthwiseConv2D`` + ``BatchNormalization`` (same
  receptive field, fewer parameters / FLOPs in the context stream).

Strides are fixed to 1 and ``skip_pool`` is therefore inert for the DS
variant in this comparison; the experiment isolates the *single 7x7 vs
stacked 3x3* design choice. Everything else (stem, channels, head,
optimiser, schedule, augmentation, drop-path) is held constant.

Variants
--------
- **vanilla** -- 5 ``CliffordNetBlock`` (control).
- **ds**      -- 5 ``CliffordNetBlockDS`` (kernel=7, strides=1).

Usage
-----
.. code-block:: bash

    # Smoke test (3 epochs, batch 32)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \\
        --variant all --smoke-test --gpu 0

    # Full benchmark (100 epochs each, serial on GPU 0)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_compare_variants \\
        --variant all --epochs 100 --batch-size 128 --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# Local / library imports
# ---------------------------------------------------------------------

from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlock,
    CliffordNetBlockDS,
)
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger

from train.cliffordnet.train_cliffordnet import (
    build_eval_dataset,
    build_train_dataset,
)
from train.common import (
    create_callbacks,
    load_dataset,
    setup_gpu,
    validate_model_loading,
)


# ---------------------------------------------------------------------
# Default comparison configuration (overridable via CLI)
# ---------------------------------------------------------------------

# Held constant across all variants — only the block class differs.
# Channels and block count are CLI-tunable so the same script can run
# both the small (~600k) and large (~10M) sweeps without copy-paste.
_DEFAULT_CHANNELS: int = 128
_DEFAULT_NUM_BLOCKS: int = 5
_SHIFTS: List[int] = [1, 2]
_CLI_MODE: str = "full"
_CTX_MODE: str = "diff"
_USE_GLOBAL_CONTEXT: bool = False

# DS-only knobs that are constant across variants (strides=1 ⇒ skip_pool
# is inert and no pool layers are built). ``kernel_size`` lives inside
# each variant's ``ds_kwargs`` so it can be tuned per-variant.
_DS_STRIDES: int = 1
_DS_SKIP_POOL: str = "avg"


VARIANTS: Dict[str, Dict[str, Any]] = {
    "vanilla": dict(
        block_cls="CliffordNetBlock",
        ds_kwargs=None,
        description=(
            "Vanilla CliffordNetBlock x5 @ C=128. Context stream: two "
            "stacked 3x3 depthwise convs + single BN (effective 7x7 RF)."
        ),
    ),
    "ds": dict(
        block_cls="CliffordNetBlockDS",
        ds_kwargs=dict(kernel_size=7, use_ctx_bn=True, ctx_activation="silu"),
        description=(
            "CliffordNetBlockDS x5 @ C=128, kernel_size=7, strides=1. "
            "Context stream: one 7x7 depthwise conv + BN + SiLU. No "
            "downsampling, skip_pool inert."
        ),
    ),
    "ds_k5": dict(
        block_cls="CliffordNetBlockDS",
        ds_kwargs=dict(kernel_size=5, use_ctx_bn=True, ctx_activation="silu"),
        description=(
            "CliffordNetBlockDS x5 @ C=128, kernel_size=5, strides=1. "
            "Same as ds but with a 5x5 depthwise (smaller RF, fewer "
            "spatial weights: 25 vs 49 per channel)."
        ),
    ),
    "ds_plain": dict(
        block_cls="CliffordNetBlockDS",
        ds_kwargs=dict(kernel_size=7, use_ctx_bn=False, ctx_activation=None),
        description=(
            "CliffordNetBlockDS x5 @ C=128, kernel_size=7, strides=1, "
            "no BN and no activation on the context stream. The DWConv "
            "carries a learnable bias in lieu of BN's affine."
        ),
    ),
    "ds_plain_k5": dict(
        block_cls="CliffordNetBlockDS",
        ds_kwargs=dict(kernel_size=5, use_ctx_bn=False, ctx_activation=None),
        description=(
            "CliffordNetBlockDS x5 @ C=128, kernel_size=5, strides=1, "
            "no BN and no activation on the context stream. Same as "
            "ds_plain but with a 5x5 depthwise (smaller RF, fewer "
            "spatial weights: 25 vs 49 per channel)."
        ),
    ),
}


# ---------------------------------------------------------------------
# Stem (shared across variants)
# ---------------------------------------------------------------------


def _build_stem(
    inputs: keras.KerasTensor, channels: int
) -> keras.KerasTensor:
    """CIFAR-friendly patch-2 stem: 3x3 stride-2 Conv2D + BN.

    32x32 -> 16x16 spatially, channel count = ``channels``.
    """
    x = keras.layers.Conv2D(
        channels,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=True,
        name="stem_conv",
    )(inputs)
    return keras.layers.BatchNormalization(name="stem_norm")(x)


# ---------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------


def build_variant(
    variant_name: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    channels: int = _DEFAULT_CHANNELS,
    num_blocks: int = _DEFAULT_NUM_BLOCKS,
    stochastic_depth_rate: float = 0.1,
    layer_scale_init: float = 1e-5,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """Build a 5-block isotropic backbone using the requested block class.

    :param variant_name: One of ``"vanilla"`` or ``"ds"``.
    :param num_classes: Number of classification outputs.
    :param input_shape: Image shape ``(H, W, 3)``.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled
        across the 5 blocks).
    :param layer_scale_init: Initial LayerScale weight value per block.
    :param dropout_rate: Dropout applied before the classifier head.
    :return: A built Keras functional ``Model``.
    """
    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_name!r}. "
            f"Available: {list(VARIANTS.keys())}"
        )

    if num_blocks <= 1:
        drop_rates = [0.0] * num_blocks
    else:
        step = stochastic_depth_rate / (num_blocks - 1)
        drop_rates = [round(i * step, 6) for i in range(num_blocks)]

    inputs = keras.layers.Input(shape=input_shape, name="input")
    x = _build_stem(inputs, channels)

    cfg = VARIANTS[variant_name]
    if cfg["ds_kwargs"] is None:
        for i in range(num_blocks):
            x = CliffordNetBlock(
                channels=channels,
                shifts=_SHIFTS,
                cli_mode=_CLI_MODE,
                ctx_mode=_CTX_MODE,
                use_global_context=_USE_GLOBAL_CONTEXT,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[i],
                name=f"block{i}",
            )(x)
    else:
        ds_kwargs = cfg["ds_kwargs"]
        for i in range(num_blocks):
            x = CliffordNetBlockDS(
                channels=channels,
                shifts=_SHIFTS,
                cli_mode=_CLI_MODE,
                ctx_mode=_CTX_MODE,
                use_global_context=_USE_GLOBAL_CONTEXT,
                strides=_DS_STRIDES,
                skip_pool=_DS_SKIP_POOL,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[i],
                name=f"block{i}",
                **ds_kwargs,
            )(x)

    # Head: GAP -> LayerNorm -> (Dropout) -> Dense
    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, name="classifier")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=variant_name)


# ---------------------------------------------------------------------
# Per-variant training
# ---------------------------------------------------------------------


def _smoke_forward(
    model: keras.Model, input_shape: Tuple[int, int, int]
) -> None:
    """Run a 1-batch forward+backward to confirm the variant trains."""
    x = np.random.normal(size=(2,) + input_shape).astype(np.float32)
    y = np.random.randint(0, 100, size=(2,)).astype(np.int32)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )(y, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    n_grads = sum(1 for g in grads if g is not None)
    logger.info(
        f"  smoke-forward OK | output={tuple(logits.shape)} | "
        f"loss={float(loss):.4f} | trainable_grads={n_grads}/"
        f"{len(model.trainable_weights)}"
    )


def train_one_variant(
    variant_name: str,
    args: argparse.Namespace,
    run_root: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    input_shape: Tuple[int, int, int],
) -> Optional[Dict[str, Any]]:
    """Train one variant end-to-end. Returns a result dict or ``None`` on failure."""
    logger.info(f"========== Variant: {variant_name} ==========")
    t0 = time.time()

    # ---- Build + smoke ----
    try:
        model = build_variant(
            variant_name,
            num_classes=num_classes,
            input_shape=input_shape,
            channels=args.channels,
            num_blocks=args.num_blocks,
            stochastic_depth_rate=args.stochastic_depth_rate,
            layer_scale_init=args.layer_scale_init,
            dropout_rate=args.dropout_rate,
        )
        _smoke_forward(model, input_shape)
        param_count = int(model.count_params())
        logger.info(f"  parameters: {param_count:,}")
    except Exception as exc:
        logger.error(
            f"  BUILD FAILED for {variant_name}: {exc}", exc_info=True
        )
        return None

    # ---- Datasets ----
    epochs = args.epochs
    batch_size = args.batch_size
    if args.smoke_test:
        epochs = 3
        batch_size = 32

    train_ds = build_train_dataset(
        x_train, y_train, batch_size, dataset_name="cifar100",
        random_erasing_prob=0.0 if args.smoke_test else 0.25,
    )
    val_ds = build_eval_dataset(
        x_test, y_test, batch_size, dataset_name="cifar100"
    )
    steps_per_epoch = len(x_train) // batch_size

    # ---- Optimiser + schedule ----
    total_steps = epochs * steps_per_epoch
    warmup_epochs = min(args.warmup_epochs, max(1, epochs // 5))
    warmup_steps = warmup_epochs * steps_per_epoch
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

    metrics: List[Any] = ["accuracy"]
    if num_classes > 10:
        metrics.append(
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top5_accuracy"
            )
        )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=metrics,
    )

    # ---- Callbacks ----
    relative_prefix = os.path.relpath(
        os.path.join(run_root, variant_name), start="results"
    )
    callbacks, results_dir = create_callbacks(
        model_name="compare_variants",
        results_dir_prefix=relative_prefix,
        monitor="val_accuracy",
        patience=args.patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    logger.info(
        f"  Training | epochs={epochs}, batch_size={batch_size}, "
        f"steps/epoch={steps_per_epoch}, peak_lr={args.learning_rate}, "
        f"warmup_epochs={warmup_epochs}, results_dir={results_dir}"
    )

    # ---- Fit ----
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=2,
        )
    except Exception as exc:
        logger.error(
            f"  TRAIN FAILED for {variant_name}: {exc}", exc_info=True
        )
        return None

    wall = time.time() - t0

    val_acc_history = history.history.get("val_accuracy", [])
    val_top5_history = history.history.get("val_top5_accuracy", [])
    best_val_acc = max(val_acc_history) if val_acc_history else float("nan")
    final_val_acc = val_acc_history[-1] if val_acc_history else float("nan")
    best_val_top5 = max(val_top5_history) if val_top5_history else float("nan")

    # ---- Save artifacts ----
    try:
        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump({
                "variant": variant_name,
                "description": VARIANTS[variant_name]["description"],
                "block_cls": VARIANTS[variant_name]["block_cls"],
                "channels": args.channels,
                "num_blocks": args.num_blocks,
                "shifts": _SHIFTS,
                "cli_mode": _CLI_MODE,
                "ctx_mode": _CTX_MODE,
                "use_global_context": _USE_GLOBAL_CONTEXT,
                "ds_kwargs": VARIANTS[variant_name]["ds_kwargs"],
                "ds_strides": _DS_STRIDES,
                "ds_skip_pool": _DS_SKIP_POOL,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "stochastic_depth_rate": args.stochastic_depth_rate,
                "layer_scale_init": args.layer_scale_init,
                "parameters": param_count,
            }, f, indent=2)
        with open(os.path.join(results_dir, "training_history.json"), "w") as f:
            json.dump(
                {k: [float(v) for v in vals]
                 for k, vals in history.history.items()},
                f, indent=2,
            )
    except Exception as exc:
        logger.warning(f"  failed to write JSON artifacts: {exc}")

    # ---- Round-trip check ----
    if not args.skip_save:
        test_sample = x_test[:4]
        try:
            stats_mean = np.array(
                [0.5071, 0.4867, 0.4408], dtype=np.float32
            )
            stats_std = np.array(
                [0.2675, 0.2565, 0.2761], dtype=np.float32
            )
            test_norm = (test_sample - stats_mean) / stats_std
            pre_save = model.predict(test_norm, verbose=0)
            final_path = os.path.join(
                results_dir, f"{variant_name}_final.keras"
            )
            model.save(final_path)
            logger.info(f"  saved final model to {final_path}")
            validate_model_loading(
                final_path, test_norm, pre_save,
                custom_objects={
                    "CliffordNetBlock": CliffordNetBlock,
                    "CliffordNetBlockDS": CliffordNetBlockDS,
                },
            )
        except Exception as exc:
            logger.warning(f"  save / validate failed: {exc}")

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Variant: {variant_name}\n")
        f.write(f"Block class: {VARIANTS[variant_name]['block_cls']}\n")
        f.write(f"Parameters: {param_count:,}\n")
        f.write(f"Epochs: {epochs}, Batch: {batch_size}\n")
        f.write(f"Wall time: {wall / 60:.1f} min\n")
        f.write(f"Best val_accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final val_accuracy: {final_val_acc:.4f}\n")
        f.write(f"Best val_top5_accuracy: {best_val_top5:.4f}\n")

    logger.info(
        f"  DONE {variant_name} | best_val_acc={best_val_acc:.4f} | "
        f"params={param_count:,} | wall={wall / 60:.1f} min"
    )

    return {
        "variant": variant_name,
        "block_cls": VARIANTS[variant_name]["block_cls"],
        "parameters": param_count,
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": final_val_acc,
        "best_val_top5_accuracy": best_val_top5,
        "wall_seconds": round(wall, 1),
        "epochs_trained": epochs,
        "results_dir": results_dir,
    }


# ---------------------------------------------------------------------
# Multi-variant orchestration
# ---------------------------------------------------------------------


def _write_comparison_csv(
    rows: List[Dict[str, Any]], path: str
) -> None:
    """Write the cross-variant comparison table to disk."""
    if not rows:
        logger.warning("No rows to write to comparison.csv")
        return
    fieldnames = [
        "variant",
        "block_cls",
        "parameters",
        "best_val_accuracy",
        "final_val_accuracy",
        "best_val_top5_accuracy",
        "wall_seconds",
        "epochs_trained",
        "results_dir",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(f"Comparison written to {path}")


def train_all_variants(args: argparse.Namespace) -> None:
    """Train every variant in :data:`VARIANTS` (or the one in args.variant)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(
        "results", f"cliffordnet_compare_variants_{timestamp}"
    )
    os.makedirs(run_root, exist_ok=True)
    logger.info(f"Run root: {run_root}")

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = (
        load_dataset("cifar100", batch_size=args.batch_size)
    )

    if args.variant == "all":
        variant_names = list(VARIANTS.keys())
    else:
        if args.variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant {args.variant!r}. "
                f"Available: {list(VARIANTS.keys())} or 'all'."
            )
        variant_names = [args.variant]

    rows: List[Dict[str, Any]] = []
    for variant_name in variant_names:
        try:
            row = train_one_variant(
                variant_name,
                args,
                run_root,
                x_train, y_train,
                x_test, y_test,
                num_classes,
                input_shape,
            )
            if row is not None:
                rows.append(row)
        except Exception as exc:
            logger.error(
                f"Unhandled error in variant {variant_name}: {exc}",
                exc_info=True,
            )
            continue

        # Persist after every variant so a mid-run crash still yields data.
        _write_comparison_csv(rows, os.path.join(run_root, "comparison.csv"))

    logger.info("All variants attempted.")
    if rows:
        logger.info("Final ranking by best_val_accuracy:")
        for r in sorted(
            rows, key=lambda r: r["best_val_accuracy"], reverse=True
        ):
            logger.info(
                f"  {r['variant']:<10} ({r['block_cls']:<22}) "
                f"acc={r['best_val_accuracy']:.4f} "
                f"params={r['parameters']:>9,d} "
                f"wall={r['wall_seconds'] / 60:>5.1f} min"
            )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "CIFAR-100 comparison: vanilla CliffordNetBlock vs "
            "CliffordNetBlockDS (5 isotropic blocks, no downsampling)."
        )
    )
    parser.add_argument(
        "--variant", type=str, default="all",
        choices=["all", "vanilla", "ds", "ds_k5", "ds_plain", "ds_plain_k5"],
        help="Variant to train, or 'all' to run both serially.",
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (None -> all visible).")
    parser.add_argument("--channels", type=int, default=_DEFAULT_CHANNELS,
                        help=f"Backbone channel width (default {_DEFAULT_CHANNELS}).")
    parser.add_argument("--num-blocks", type=int, default=_DEFAULT_NUM_BLOCKS,
                        dest="num_blocks",
                        help=f"Number of isotropic blocks (default {_DEFAULT_NUM_BLOCKS}).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128,
                        dest="batch_size")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        dest="learning_rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        dest="weight_decay")
    parser.add_argument("--patience", type=int, default=30,
                        help="EarlyStopping patience (val_accuracy).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        dest="warmup_epochs")
    parser.add_argument("--stochastic-depth-rate", type=float, default=0.1,
                        dest="stochastic_depth_rate")
    parser.add_argument("--layer-scale-init", type=float, default=1e-5,
                        dest="layer_scale_init")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        dest="dropout_rate")
    parser.add_argument("--smoke-test", action="store_true",
                        dest="smoke_test",
                        help="3-epoch / batch-32 / no-aug sanity loop.")
    parser.add_argument("--skip-save", action="store_true",
                        dest="skip_save",
                        help="Skip model.save and round-trip check.")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = _parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    setup_gpu(gpu_id=None)
    try:
        train_all_variants(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
