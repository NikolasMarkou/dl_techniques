"""Energy Transformer masked-image-completion training (Pattern 1).

Self-supervised pretraining of the paper's §3 image model (arXiv:2302.07253): patch-embed ->
learnable MASK token -> positional embedding -> ONE Energy Transformer block running T=12
internal energy-descent steps -> affine decoder back to raw patch pixels.

**No custom ``train_step``.** The occlusion mask reaches the LOSS through Keras' sanctioned
channel, ``sample_weight``, supplied as the third element of each ``tf.data`` batch::

    ((image, input_mask), target_patches, loss_weight)

with ``loss_weight = 1{i in S} * (N / n_loss)``, so that a stock
``model.compile(loss="mse")`` + ``model.fit(ds)`` computes exactly
``mean_{i in S} MSE(recon_i, target_i)``. See
``dl_techniques.datasets.vision.masked_patches``.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.energy_transformer.train_masked_completion \\
        --dataset imagenette --variant tiny --image-size 224 --patch-size 16 \\
        --epochs 100 --batch-size 32 --gpu 1
"""

import gc
import json
import time
import keras
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    set_seeds,
    save_config_json,
    create_callbacks as create_common_callbacks,
)
from train.energy_transformer.common import (
    SUPPORTED_DATASETS,
    build_optimizer,
    build_raw_image_dataset,
    EnergyTraceCallback,
)
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.masked_patches import make_masked_patch_map_fn
from dl_techniques.models.energy_transformer import create_energy_transformer_mim

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for Energy Transformer masked-image-completion training.

    Optimization defaults are the paper's Table 4 (image / ImageNet-1k):
    ``lr 5e-4``, ``weight_decay 0.05``, ``gradient_clipping 1.0``, cosine decay with warmup,
    AdamW.
    """

    # Data
    dataset: str = "imagenette"  # imagenette | cifar10
    image_size: int = 224
    patch_size: int = 16
    batch_size: int = 32
    augment_data: bool = True

    # Model
    variant: str = "tiny"  # tiny | small | base
    num_steps: int = 12  # T -- the block's internal descent steps
    mask_ratio: float = 0.5  # |S| / N  (paper: 0.5)
    mask_token_frac: float = 0.9  # |input_mask| / |S|  (paper's 90/10 rule)

    # Training (paper Table 4)
    epochs: int = 100
    learning_rate: float = 5e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    # LOAD-BEARING. The paper reports the model FAILS TO TRAIN at learning rates above 1e-4
    # WITHOUT gradient clipping -- the energy descent is unrolled T=12 times, so the backward
    # pass composes 12 Jacobians and the gradient norm spikes. Do not set this to 0 "to
    # simplify" and then raise the lr; that combination is the paper's documented failure.
    gradient_clipping: float = 1.0

    # Monitoring
    early_stopping_patience: int = 15

    # Debug
    max_steps: Optional[int] = None  # cap steps_per_epoch (smoke runs); None = full epoch

    # Output -- H9: repo-root `results/`, NEVER `src/results/`.
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Runtime
    seed: int = 42
    gpu: Optional[int] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"et_mim_{self.dataset}_{self.variant}_{timestamp}"

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}"
            )
        # mask_ratio / mask_token_frac are validated (with the exact n_loss / n_input
        # arithmetic, including the n_input == n_loss rejection) by make_masked_patch_map_fn.


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_datasets(config: TrainingConfig) -> Tuple[Any, Any, int, int]:
    """Build the masked-patch train/val pipelines.

    Returns:
        ``(train_ds, val_ds, steps_per_epoch, val_steps)``. Both datasets yield the frozen
        element ``((image, input_mask), target_patches, loss_weight)``.
    """
    map_fn = make_masked_patch_map_fn(
        patch_size=config.patch_size,
        image_size=config.image_size,
        mask_ratio=config.mask_ratio,
        mask_token_frac=config.mask_token_frac,
        seed=config.seed,
    )

    train_ds, num_train, _ = build_raw_image_dataset(
        config.dataset,
        config.image_size,
        config.batch_size,
        is_training=True,
        augment=config.augment_data,
        element_map_fn=map_fn,
        seed=config.seed,
    )
    val_ds, num_val, _ = build_raw_image_dataset(
        config.dataset,
        config.image_size,
        config.batch_size,
        is_training=False,
        element_map_fn=map_fn,
        seed=config.seed,
    )

    steps_per_epoch = max(1, num_train // config.batch_size)
    val_steps = max(1, num_val // config.batch_size)
    if config.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps)
        val_steps = min(val_steps, config.max_steps)
    return train_ds, val_ds, steps_per_epoch, val_steps


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_masked_completion(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate masked-image-completion pretraining.

    Returns:
        Dict with ``model``, ``best_val_loss``, ``first_loss``, ``final_loss``, ``run_dir``,
        ``history``.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | variant={config.variant} "
        f"dataset={config.dataset} image={config.image_size} patch={config.patch_size} "
        f"mask_ratio={config.mask_ratio} mask_token_frac={config.mask_token_frac}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Data ----
    train_ds, val_ds, steps_per_epoch, val_steps = build_datasets(config)
    logger.info(f"Steps per epoch: {steps_per_epoch}, val steps: {val_steps}")

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): AdamW's decay comes from optimizer_builder alone.
    input_shape = (config.image_size, config.image_size, 3)
    model = create_energy_transformer_mim(
        variant=config.variant,
        input_shape=input_shape,
        patch_size=config.patch_size,
        num_steps=config.num_steps,
    )
    num_patches = model.num_patches
    # Probe build so summary()/count_params() work before fit.
    model.build([(None,) + input_shape, (None, num_patches)])
    model.summary(print_fn=logger.info)

    # ---- Optimization ----
    optimizer = build_optimizer(config, steps_per_epoch)

    # STOCK compile. The mask is a sample_weight in the dataset, not a train_step override.
    model.compile(optimizer=optimizer, loss="mse")

    # ---- Callbacks ----
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="et_mim",
        run_dir=str(run_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    # One FIXED val batch, held for the whole run, so the epoch-to-epoch traces are comparable.
    probe_inputs, _, _ = next(iter(val_ds))
    callbacks.append(EnergyTraceCallback(
        probe_inputs=probe_inputs,
        csv_path=str(run_dir / "energy_trace.csv"),
    ))

    # ---- Train ----
    start = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {(time.time() - start) / 3600.0:.3f} hours")

    loss_curve = history.history.get("loss", []) or [float("nan")]
    val_curve = history.history.get("val_loss", []) or [float("nan")]

    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(run_dir / "training_history.json", "w") as handle:
            json.dump(history_dict, handle, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        logger.warning(f"Failed to save training history: {exc}")

    gc.collect()
    return {
        "model": model,
        "best_val_loss": float(min(val_curve)),
        "first_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "run_dir": str(run_dir),
        "history": history,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="Train the Energy Transformer masked-image-completion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--image-size", type=int, default=None,
                        help="Auto: 224 for imagenette, 32 for cifar10")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Auto: 16 for imagenette, 4 for cifar10")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-augmentation", dest="augment_data", action="store_false")

    # Model
    parser.add_argument("--variant", type=str, default="tiny",
                        choices=["tiny", "small", "base"])
    parser.add_argument("--num-steps", type=int, default=12,
                        help="T: the ET block's internal energy-descent steps")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="Fraction of tokens entering the loss set S")
    parser.add_argument("--mask-token-frac", type=float, default=0.9,
                        help="Fraction of S that is additionally occluded (paper's 90/10 rule)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                        help="Clip by global norm. The paper's model does not train above "
                             "lr 1e-4 without this.")

    # Monitoring
    parser.add_argument("--early-stopping-patience", type=int, default=15)

    # Debug
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps_per_epoch (and val steps). Smoke runs only.")

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Map a parsed ``Namespace`` onto a :class:`TrainingConfig`. PURE -- no side effects.

    Every flag in :func:`parse_arguments` must land in a field here. A flag that does not is a
    SILENT NO-OP: the run trains at the default while the command line says otherwise. That
    trap has bitten this repo before, which is why this mapping is an importable function with
    a dedicated test (``tests/test_train/test_energy_transformer/test_cli_wiring.py``) rather
    than a block inside ``main()``.
    """
    dataset = args.dataset.lower()
    image_size = args.image_size if args.image_size is not None else (
        32 if dataset == "cifar10" else 224
    )
    patch_size = args.patch_size if args.patch_size is not None else (
        4 if dataset == "cifar10" else 16
    )

    return TrainingConfig(
        dataset=dataset,
        image_size=image_size,
        patch_size=patch_size,
        batch_size=args.batch_size,
        augment_data=args.augment_data,
        variant=args.variant,
        num_steps=args.num_steps,
        mask_ratio=args.mask_ratio,
        mask_token_frac=args.mask_token_frac,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        early_stopping_patience=args.early_stopping_patience,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        gpu=args.gpu,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    config = config_from_args(parse_arguments())

    logger.info(
        f"Config: variant={config.variant}, dataset={config.dataset}, "
        f"{config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}, clip={config.gradient_clipping}"
    )

    try:
        result = train_masked_completion(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== MIM TRAINING DONE === first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} best_val_loss={result['best_val_loss']:.6f} "
        f"run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
