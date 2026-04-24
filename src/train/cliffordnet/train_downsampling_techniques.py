"""
Training script for the CliffordNet downsampling experiment.

Trains 5 hierarchical / multi-stage downsampling variants of CliffordNet on
CIFAR-100, plus the existing isotropic baseline (V0) for reference.  The
underlying ``CliffordNetBlock`` is dim-preserving, so each variant is built
by composing a custom stem, several stages of identical
``CliffordNetBlock`` layers, and one of four candidate inter-stage
**downsamplers**:

- ``strided_conv``       -- 3x3 stride-2 Conv2D + LayerNorm.
- ``avgpool_proj``       -- AvgPool2D(2) + 1x1 Conv2D + LayerNorm.
- ``patch_merging``      -- Swin-style 2x2 patch concat + LayerNorm + Dense.
- ``dwsep_strided``      -- Depthwise 3x3 stride-2 + 1x1 Conv2D + BN.

Variants
--------
- **V0_baseline_isotropic**   -- existing ``CliffordNet.nano`` (control).
- **V1_3stage_strided_conv**  -- 3 stages 64-128-256, strided 3x3 conv.
- **V2_3stage_avgpool**       -- 3 stages 64-128-256, AvgPool + 1x1 proj.
- **V3_3stage_patch_merging** -- 3 stages 64-128-256, Swin patch-merge.
- **V4_4stage_aggressive**    -- 4 stages 64-128-256-512, patch-merge.
- **V5_2stage_aggressive_stem** -- patch-stem stride 4, 2 stages 128-256,
  depthwise-separable strided downsampler.

The accompanying report lives at
``src/dl_techniques/models/cliffordnet/DOWNSAMPLING.md``.

Usage
-----
.. code-block:: bash

    # Cheap sanity check (3 epochs, batch 32, all variants)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \\
        --variant all --smoke-test --gpu 0

    # Full benchmark (100 epochs each, all variants serial on GPU 0)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \\
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

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.layers.patch_merging import PatchMerging
from dl_techniques.models.cliffordnet import CliffordNet
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
# Variant registry
# ---------------------------------------------------------------------

# Each variant entry defines:
#   stem            : "patch1" | "patch2" | "patch4"  (CIFAR-friendly stems)
#   stages          : list of (channels, n_blocks)
#   downsamplers    : list of strings of length len(stages) - 1; each is one of
#                     {"strided_conv", "avgpool_proj", "patch_merging",
#                      "dwsep_strided"}
#   shifts          : SparseRollingGeometricProduct shifts (kept = [1, 2]
#                     across variants for a single-axis comparison)

_DEFAULT_SHIFTS: List[int] = [1, 2]


VARIANTS: Dict[str, Dict[str, Any]] = {
    "V0_baseline_isotropic": dict(
        stem="patch2",
        stages=[(128, 12)],
        downsamplers=[],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "Existing CliffordNet.nano baseline: patch_size=2 stem, "
            "12 isotropic blocks @ C=128, no in-backbone downsampling."
        ),
    ),
    "V1_3stage_strided_conv": dict(
        stem="patch1",
        stages=[(64, 4), (128, 4), (256, 4)],
        downsamplers=["strided_conv", "strided_conv"],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "3-stage hierarchy 64-128-256 channels, strided 3x3 Conv2D "
            "downsampler, channels doubled at each transition."
        ),
    ),
    "V2_3stage_avgpool": dict(
        stem="patch1",
        stages=[(64, 4), (128, 4), (256, 4)],
        downsamplers=["avgpool_proj", "avgpool_proj"],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "Same shape as V1 but downsampling is parameter-light: "
            "AvgPool2D(2) followed by a 1x1 channel-expansion conv."
        ),
    ),
    "V3_3stage_patch_merging": dict(
        stem="patch1",
        stages=[(64, 4), (128, 4), (256, 4)],
        downsamplers=["patch_merging", "patch_merging"],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "Swin-style patch merging downsampler (concat 2x2 + LN + Dense). "
            "PatchMerging produces 2*C channels natively, matching the "
            "doubling in the channel schedule."
        ),
    ),
    "V4_4stage_aggressive": dict(
        stem="patch1",
        stages=[(64, 2), (128, 2), (256, 4), (512, 4)],
        downsamplers=["patch_merging", "patch_merging", "patch_merging"],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "Deeper 4-stage hierarchy 64-128-256-512 with PatchMerging at "
            "each transition. Most aggressive channel expansion (8x); "
            "highest parameter count of all variants."
        ),
    ),
    "V5_2stage_aggressive_stem": dict(
        stem="patch4",
        stages=[(128, 6), (256, 6)],
        downsamplers=["dwsep_strided"],
        shifts=_DEFAULT_SHIFTS,
        description=(
            "Aggressive 4x stem followed by 2 stages 128-256 with a "
            "depthwise-separable strided downsampler. Tests the "
            "front-load-spatial-reduction strategy."
        ),
    ),
}


# ---------------------------------------------------------------------
# Stem / downsampler helpers
# ---------------------------------------------------------------------


def _build_stem(
    inputs: keras.KerasTensor, stem: str, channels: int
) -> keras.KerasTensor:
    """Build a CIFAR-friendly stem and return its output tensor.

    :param inputs: Image tensor ``(B, 32, 32, 3)``.
    :param stem: One of ``"patch1"``, ``"patch2"``, ``"patch4"``.
    :param channels: Output channel dimension of the stem.
    :return: Stem feature map.
    """
    if stem == "patch1":
        # Two 3x3 stride-1 convs (no spatial downsampling).
        x = keras.layers.Conv2D(
            channels // 2,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="stem_conv1",
        )(inputs)
        x = keras.layers.BatchNormalization(name="stem_bn1")(x)
        x = keras.layers.Activation("silu", name="stem_act1")(x)
        x = keras.layers.Conv2D(
            channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            name="stem_conv2",
        )(x)
    elif stem == "patch2":
        # Single 3x3 stride-2 conv (32 -> 16).
        x = keras.layers.Conv2D(
            channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=True,
            name="stem_conv",
        )(inputs)
    elif stem == "patch4":
        # Two stride-2 3x3 convs (32 -> 8).
        x = keras.layers.Conv2D(
            channels // 2,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name="stem_conv1",
        )(inputs)
        x = keras.layers.BatchNormalization(name="stem_bn1")(x)
        x = keras.layers.Activation("silu", name="stem_act1")(x)
        x = keras.layers.Conv2D(
            channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name="stem_conv2",
        )(x)
    else:
        raise ValueError(f"Unknown stem type: {stem!r}")

    return keras.layers.BatchNormalization(name="stem_norm")(x)


def _build_downsampler(
    x: keras.KerasTensor,
    kind: str,
    in_channels: int,
    out_channels: int,
    name_prefix: str,
) -> keras.KerasTensor:
    """Apply one of the four candidate inter-stage downsamplers.

    :param x: Input feature map ``(B, H, W, in_channels)``.
    :param kind: Downsampler type (see module docstring).
    :param in_channels: Channel count of the input feature map.
    :param out_channels: Desired output channel count.
    :param name_prefix: Prefix used to name the sub-layers.
    :return: Downsampled feature map ``(B, H/2, W/2, out_channels)``.
    """
    if kind == "strided_conv":
        x = keras.layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=True,
            name=f"{name_prefix}_conv",
        )(x)
        x = keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_ln"
        )(x)
        return x

    if kind == "avgpool_proj":
        x = keras.layers.AveragePooling2D(
            pool_size=2, strides=2, padding="same", name=f"{name_prefix}_pool"
        )(x)
        x = keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name=f"{name_prefix}_proj",
        )(x)
        x = keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name_prefix}_ln"
        )(x)
        return x

    if kind == "patch_merging":
        x = PatchMerging(dim=in_channels, name=f"{name_prefix}_pm")(x)
        # PatchMerging emits 2*in_channels; if out_channels differs, project.
        merged_dim = 2 * in_channels
        if out_channels != merged_dim:
            x = keras.layers.Dense(
                out_channels, name=f"{name_prefix}_proj"
            )(x)
        return x

    if kind == "dwsep_strided":
        x = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            name=f"{name_prefix}_dw",
        )(x)
        x = keras.layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=f"{name_prefix}_pw",
        )(x)
        x = keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
        return x

    raise ValueError(f"Unknown downsampler: {kind!r}")


# ---------------------------------------------------------------------
# Variant builder
# ---------------------------------------------------------------------


def build_variant(
    variant_name: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    stochastic_depth_rate: float = 0.1,
    layer_scale_init: float = 1e-5,
    dropout_rate: float = 0.0,
) -> keras.Model:
    """Build a single variant by composing stem + stages + downsamplers + head.

    :param variant_name: One of the keys in :data:`VARIANTS`.
    :param num_classes: Number of classification outputs.
    :param input_shape: Image shape ``(H, W, 3)``.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled
        across **all** blocks in the model regardless of stage).
    :param layer_scale_init: Initial value for the per-block LayerScale weight.
    :param dropout_rate: Dropout applied before the classifier head.
    :return: A built Keras functional ``Model``.
    """
    if variant_name == "V0_baseline_isotropic":
        # Reuse the existing CliffordNet.nano factory verbatim so the
        # baseline is bitwise-comparable to the existing training script.
        return CliffordNet.nano(
            num_classes=num_classes,
            stochastic_depth_rate=stochastic_depth_rate,
            layer_scale_init=layer_scale_init,
            dropout_rate=dropout_rate,
        )

    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_name!r}. "
            f"Available: {list(VARIANTS.keys())}"
        )

    cfg = VARIANTS[variant_name]
    stages: List[Tuple[int, int]] = cfg["stages"]
    downsamplers: List[str] = cfg["downsamplers"]
    shifts: List[int] = cfg["shifts"]

    if len(downsamplers) != len(stages) - 1:
        raise ValueError(
            f"{variant_name}: expected {len(stages) - 1} downsamplers, "
            f"got {len(downsamplers)}"
        )

    # Linear DropPath schedule across every block in the model.
    total_blocks = sum(n for _, n in stages)
    if total_blocks <= 1:
        drop_rates = [0.0] * total_blocks
    else:
        step = stochastic_depth_rate / (total_blocks - 1)
        drop_rates = [round(i * step, 6) for i in range(total_blocks)]

    inputs = keras.layers.Input(shape=input_shape, name="input")

    # Stem
    first_stage_channels = stages[0][0]
    x = _build_stem(inputs, cfg["stem"], first_stage_channels)

    # Stages + interleaved downsamplers
    block_idx = 0
    for stage_idx, (channels, n_blocks) in enumerate(stages):
        for _ in range(n_blocks):
            x = CliffordNetBlock(
                channels=channels,
                shifts=shifts,
                cli_mode="full",
                ctx_mode="diff",
                use_global_context=False,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[block_idx],
                name=f"stage{stage_idx}_block{_}",
            )(x)
            block_idx += 1
        if stage_idx < len(stages) - 1:
            next_channels = stages[stage_idx + 1][0]
            x = _build_downsampler(
                x,
                kind=downsamplers[stage_idx],
                in_channels=channels,
                out_channels=next_channels,
                name_prefix=f"down{stage_idx}",
            )

    # Head: GAP -> LayerNorm -> (Dropout) -> Dense
    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="head_norm")(x)
    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = keras.layers.Dense(num_classes, name="classifier")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=variant_name)
    return model


# ---------------------------------------------------------------------
# Per-variant training
# ---------------------------------------------------------------------


def _smoke_forward(model: keras.Model, input_shape: Tuple[int, int, int]) -> None:
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
    """Train one variant end-to-end. Returns a result dict or ``None`` on failure.

    Records:
        - best_val_accuracy, final_val_accuracy, val_top5_accuracy
        - parameters, wall_seconds
        - results_dir
    """
    logger.info(f"========== Variant: {variant_name} ==========")
    t0 = time.time()

    # ---- Build + smoke ----
    try:
        model = build_variant(
            variant_name,
            num_classes=num_classes,
            input_shape=input_shape,
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
    callbacks, results_dir = create_callbacks(
        model_name=f"downsampling_{variant_name}",
        results_dir_prefix=os.path.join(run_root, variant_name),
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
                "description": VARIANTS[variant_name]["description"]
                if variant_name in VARIANTS else "",
                "stages": VARIANTS[variant_name]["stages"]
                if variant_name in VARIANTS else None,
                "downsamplers": VARIANTS[variant_name]["downsamplers"]
                if variant_name in VARIANTS else None,
                "stem": VARIANTS[variant_name]["stem"]
                if variant_name in VARIANTS else None,
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

    # ---- Round-trip check (skip for V0 due to the known Keras-3.8
    #      .keras + by_name=True bug; V0 round-trip is already covered by
    #      the existing CliffordNet test suite) ----
    if not args.skip_save and variant_name != "V0_baseline_isotropic":
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
                    "PatchMerging": PatchMerging,
                },
            )
        except Exception as exc:
            logger.warning(f"  save / validate failed: {exc}")

    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Variant: {variant_name}\n")
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
    setup_gpu(gpu_id=args.gpu)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(
        "results", f"cliffordnet_downsampling_{timestamp}"
    )
    os.makedirs(run_root, exist_ok=True)
    logger.info(f"Run root: {run_root}")

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = (
        load_dataset("cifar100", batch_size=args.batch_size)
    )

    # ---- Determine variants to run ----
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

        # Persist after every variant (so a mid-run crash still yields data)
        _write_comparison_csv(rows, os.path.join(run_root, "comparison.csv"))

    logger.info("All variants attempted.")
    if rows:
        logger.info("Final ranking by best_val_accuracy:")
        for r in sorted(
            rows, key=lambda r: r["best_val_accuracy"], reverse=True
        ):
            logger.info(
                f"  {r['variant']:<32} acc={r['best_val_accuracy']:.4f} "
                f"params={r['parameters']:>9,d} "
                f"wall={r['wall_seconds'] / 60:>5.1f} min"
            )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CliffordNet CIFAR-100 downsampling-techniques experiment."
    )
    parser.add_argument(
        "--variant", type=str, default="all",
        help=(
            "Variant key from VARIANTS, or 'all' to train every variant "
            "serially. See module docstring for the catalog."
        ),
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (None -> all visible).")
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
    try:
        train_all_variants(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
