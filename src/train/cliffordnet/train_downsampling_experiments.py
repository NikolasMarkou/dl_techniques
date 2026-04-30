"""
CIFAR-100 downsampling-design-space experiments for ``CliffordNetBlockDSv2``.

Implements the experimental campaign described in
``analyses/analysis_2026-04-30_41b5e415/summary.md`` -- 11 variants drawn
from the 12-variant catalogue (V0-V7, V10-V12; V8 and V9 deferred).
Every variant builds a 4-stage backbone

    stem (patch1, no spatial reduction)
    stage 0: 32x32, C=96, 2 isotropic CliffordNetBlockDSv2 blocks
    stage 1: 16x16, C=192, 1 strided + 1 isotropic block (3 transitions total)
    stage 2:  8x8,  C=384, 1 strided + 3 isotropic blocks
    stage 3:  4x4,  C=768, 1 strided + 3 isotropic blocks
    head:   GAP -> LN -> (Dropout) -> Dense(num_classes)

(~10M params, comparable to E05 in ``VARIATIONS_COMPARISON.md``.)

The variant axes (A-H from the analysis) are encoded in each variant's
``transition`` block kwargs (see :data:`VARIANTS`):

* A (stream pool): ``stream_pool``
* B (skip pool):   ``skip_pool``
* C (k >= 2s):     ``kernel_size``
* D (ctx_mode):    ``ctx_mode`` of the strided block
* E (channel exp): ``internal_expansion`` flag
* G (norm type):   ``ctx_norm_type``
* H (LayerScale):  ``--layer-scale-init`` CLI arg (uniform across blocks)

Variants V8 (full-res product) and V9 (grade-aware pool) are NOT
implemented in this script -- they require structural refactors of
``CliffordNetBlockDSv2.call`` and a constructive grade-grouping design,
respectively (see ``DECISIONS.md`` D-001).

Usage
-----
.. code-block:: bash

    # Smoke test (3 epochs, batch 32, all 11 variants)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \\
        --variant all --smoke-test --gpu 0

    # Full benchmark (100 epochs each, serial on GPU 0)
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_experiments \\
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

from dl_techniques.layers.blur_pool import BlurPool2D
from dl_techniques.layers.pixel_unshuffle import PixelUnshuffle2D
from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlockDSv2,
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
# Architecture constants (held constant across variants)
# ---------------------------------------------------------------------

# (channels, n_blocks_total_in_stage). For stages 1..3, the FIRST block of
# the stage is the strided transition block; the remaining n-1 blocks are
# isotropic (strides=1). Stage 0 has no transition; all n blocks are isotropic.
_STAGES: List[Tuple[int, int]] = [(96, 2), (192, 2), (384, 4), (768, 4)]
_SHIFTS: List[int] = [1, 2]
_CLI_MODE: str = "full"
_USE_GLOBAL_CONTEXT: bool = False
_DEFAULT_CTX_MODE_ISO: str = "diff"  # isotropic block ctx_mode (always diff)
_DEFAULT_KERNEL_SIZE_ISO: int = 7    # isotropic block context-DW kernel
_DEFAULT_NORM_TYPE_ISO: str = "bn"   # isotropic block ctx norm


# ---------------------------------------------------------------------
# Variant catalogue (V0-V7, V10-V12 from analysis summary section 4)
# ---------------------------------------------------------------------

VARIANTS: Dict[str, Dict[str, Any]] = {
    "V0_baseline_avg_avg": dict(
        description=(
            "Baseline. Both stream and skip use avg-2x2 pool. ctx_mode=diff, "
            "kernel_size=7, BN context norm, external channel expansion."
        ),
        transition=dict(
            stream_pool="avg",
            skip_pool="avg",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
    "V1_blur_blur": dict(
        description=(
            "BlurPool stream + BlurPool skip. Anti-aliased on both paths."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="blur",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
    "V2_blur_pxsh": dict(
        description=(
            "Strongest single change predicted: BlurPool stream + lossless "
            "pixel-unshuffle skip (axis A + B decouple, +0.5..+1.0pp)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
    "V3_gauss_pxsh": dict(
        description=(
            "Learnable Gaussian-init DW (k=5) stream + pixel-unshuffle skip."
        ),
        transition=dict(
            stream_pool="gaussian_dw",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
    "V4_blur_pxsh_int": dict(
        description=(
            "V2 + internal channel expansion. Principled bundle "
            "(+0.7..+1.4pp predicted)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=True,
    ),
    "V5_blur_pxsh_int_gn": dict(
        description=(
            "V4 + GroupNorm context-stream norm at all stages "
            "(targets H/4 and below; +0.1..+0.4pp predicted)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="gn",
        ),
        internal_expansion=True,
    ),
    "V6_blur_pxsh_int_pyrdiff": dict(
        description=(
            "V4 + ctx_mode=pyramid_diff at strides>1 (Laplacian-pyramid "
            "level residual; +0.1..+0.3pp predicted)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="pyramid_diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=True,
    ),
    "V7_blur_pxsh_int_abs": dict(
        description=(
            "V4 + ctx_mode=abs at strides>1 only (drop the broken "
            "Laplacian semantics at stage transitions; +0.1..+0.3pp)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="abs",
            ctx_norm_type="bn",
        ),
        internal_expansion=True,
    ),
    "V10_resnetd": dict(
        description=(
            "ResNet-D both paths: AvgPool then 1x1. Literature control."
        ),
        transition=dict(
            stream_pool="resnetd",
            skip_pool="resnetd",
            kernel_size=7,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
    "V11_kitchen_sink": dict(
        description=(
            "Kitchen sink: V5 + V6 stack (BlurPool + pixel-unshuffle + "
            "internal expansion + GroupNorm + pyramid_diff). "
            "High-risk / high-reward (+0.8..+1.5pp predicted)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=7,
            ctx_mode="pyramid_diff",
            ctx_norm_type="gn",
        ),
        internal_expansion=True,
    ),
    "V12_blur_pxsh_k3": dict(
        description=(
            "Negative control: BlurPool + pixel-unshuffle + k=3,s=2 "
            "(violates k>=2s anti-alias support; -0.3..-0.8pp predicted)."
        ),
        transition=dict(
            stream_pool="blur",
            skip_pool="pixel_unshuffle",
            kernel_size=3,
            ctx_mode="diff",
            ctx_norm_type="bn",
        ),
        internal_expansion=False,
    ),
}


# ---------------------------------------------------------------------
# Stem (patch1: 32x32 -> 32x32, no spatial reduction)
# ---------------------------------------------------------------------


def _build_stem(
    inputs: keras.KerasTensor, channels: int
) -> keras.KerasTensor:
    """Patch1 stem: two stride-1 3x3 convs + BN. Preserves 32x32 spatial.

    The stride-2 transitions of the 4-stage backbone all happen INSIDE
    ``CliffordNetBlockDSv2`` -- the stem must therefore not reduce
    spatial dims so that stage 0 sees full 32x32 features.
    """
    x = keras.layers.Conv2D(
        channels // 2, kernel_size=3, strides=1, padding="same",
        use_bias=False, name="stem_conv1",
    )(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn1")(x)
    x = keras.layers.Activation("silu", name="stem_act1")(x)
    x = keras.layers.Conv2D(
        channels, kernel_size=3, strides=1, padding="same",
        use_bias=False, name="stem_conv2",
    )(x)
    return keras.layers.BatchNormalization(name="stem_norm")(x)


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
    """Build a single variant by composing stem + 4 stages + head."""
    if variant_name not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant_name!r}. "
            f"Available: {list(VARIANTS.keys())}"
        )
    cfg = VARIANTS[variant_name]
    transition_kwargs: Dict[str, Any] = cfg["transition"]
    internal_expansion: bool = cfg["internal_expansion"]

    # Linear DropPath schedule across every block in the model.
    total_blocks = sum(n for _, n in _STAGES)
    if total_blocks <= 1:
        drop_rates = [0.0] * total_blocks
    else:
        step = stochastic_depth_rate / (total_blocks - 1)
        drop_rates = [round(i * step, 6) for i in range(total_blocks)]

    inputs = keras.layers.Input(shape=input_shape, name="input")
    first_stage_channels = _STAGES[0][0]
    x = _build_stem(inputs, first_stage_channels)

    block_idx = 0
    for stage_idx, (channels, n_blocks) in enumerate(_STAGES):
        if stage_idx == 0:
            # No transition into stage 0 -- emit n_blocks isotropic blocks.
            for _ in range(n_blocks):
                x = CliffordNetBlockDSv2(
                    channels=channels,
                    shifts=_SHIFTS,
                    cli_mode=_CLI_MODE,
                    ctx_mode=_DEFAULT_CTX_MODE_ISO,
                    use_global_context=_USE_GLOBAL_CONTEXT,
                    kernel_size=_DEFAULT_KERNEL_SIZE_ISO,
                    strides=1,
                    stream_pool="avg",  # inert at strides=1
                    skip_pool="avg",
                    out_channels=None,
                    ctx_norm_type=_DEFAULT_NORM_TYPE_ISO,
                    layer_scale_init=layer_scale_init,
                    drop_path_rate=drop_rates[block_idx],
                    name=f"stage{stage_idx}_block{block_idx - sum(s[1] for s in _STAGES[:stage_idx])}",
                )(x)
                block_idx += 1
            continue

        # Transition into stage_idx > 0: strided block + (n-1) isotropic.
        prev_channels = _STAGES[stage_idx - 1][0]
        # NB: the transition block reads `prev_channels` and produces
        # `channels` either internally (out_channels) or via an external
        # 1x1 Conv2D applied AFTER the strided block.
        if internal_expansion:
            x = CliffordNetBlockDSv2(
                channels=prev_channels,
                shifts=_SHIFTS,
                cli_mode=_CLI_MODE,
                use_global_context=_USE_GLOBAL_CONTEXT,
                strides=2,
                out_channels=channels,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[block_idx],
                name=f"stage{stage_idx}_transition",
                **transition_kwargs,
            )(x)
        else:
            x = CliffordNetBlockDSv2(
                channels=prev_channels,
                shifts=_SHIFTS,
                cli_mode=_CLI_MODE,
                use_global_context=_USE_GLOBAL_CONTEXT,
                strides=2,
                out_channels=None,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[block_idx],
                name=f"stage{stage_idx}_transition",
                **transition_kwargs,
            )(x)
            # External 1x1 channel expansion after spatial downsample.
            x = keras.layers.Conv2D(
                channels, kernel_size=1, strides=1, padding="same",
                use_bias=True, name=f"stage{stage_idx}_expand",
            )(x)
        block_idx += 1

        # Remaining (n_blocks - 1) isotropic blocks at this stage's channels.
        for _ in range(n_blocks - 1):
            x = CliffordNetBlockDSv2(
                channels=channels,
                shifts=_SHIFTS,
                cli_mode=_CLI_MODE,
                ctx_mode=_DEFAULT_CTX_MODE_ISO,
                use_global_context=_USE_GLOBAL_CONTEXT,
                kernel_size=_DEFAULT_KERNEL_SIZE_ISO,
                strides=1,
                stream_pool="avg",
                skip_pool="avg",
                out_channels=None,
                ctx_norm_type=_DEFAULT_NORM_TYPE_ISO,
                layer_scale_init=layer_scale_init,
                drop_path_rate=drop_rates[block_idx],
                name=f"stage{stage_idx}_block{block_idx - sum(s[1] for s in _STAGES[:stage_idx])}",
            )(x)
            block_idx += 1

    # Head: GAP -> LN -> (Dropout) -> Dense
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
        model_name="downsampling_experiments",
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
                "stages": _STAGES,
                "shifts": _SHIFTS,
                "cli_mode": _CLI_MODE,
                "transition_kwargs": VARIANTS[variant_name]["transition"],
                "internal_expansion": VARIANTS[variant_name][
                    "internal_expansion"
                ],
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
                    "CliffordNetBlockDSv2": CliffordNetBlockDSv2,
                    "BlurPool2D": BlurPool2D,
                    "PixelUnshuffle2D": PixelUnshuffle2D,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(
        "results", f"cliffordnet_downsampling_experiments_{timestamp}"
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

        # Persist after every variant (so a mid-run crash still yields data).
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
        description=(
            "CIFAR-100 downsampling-design-space experiments for "
            "CliffordNetBlockDSv2 (V0-V7, V10-V12). See "
            "analyses/analysis_2026-04-30_41b5e415/summary.md."
        )
    )
    parser.add_argument(
        "--variant", type=str, default="all",
        help=(
            "Variant key from VARIANTS, or 'all' to train every variant "
            "serially. See module docstring + analysis summary."
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
