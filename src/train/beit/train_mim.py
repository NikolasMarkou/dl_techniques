"""BEiT stage-1: masked-image-modeling pre-training (Pattern 1).

BEiT's pre-training objective: block-wise mask ~40% of the patch grid, feed the masked
image through the BEiT encoder, and predict the FROZEN stage-0 tokenizer's discrete
visual-token id at every masked position.

**No custom ``train_step``** (H-8). The mask reaches the LOSS through Keras' sanctioned
channel, ``sample_weight``, supplied as the third element of each ``tf.data`` batch::

    ((image, bool_mask), target_ids, sample_weight)

with ``sample_weight`` exactly the mask (1.0 masked / 0.0 unmasked), so a stock
``model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))`` +
``model.fit(ds)`` computes the cross-entropy over the masked set only. See
``dl_techniques.datasets.vision.beit_masking``.

**The two grids must be the same grid.** The encoder's patch grid is
``image_size // patch_size``; the tokenizer's code grid is a property of the stage-0
checkpoint. If they differ, every MIM target is read from the wrong spatial position and
training still produces a finite, plausible, completely wrong loss. So the tokenizer is
loaded and its grid VERIFIED (by an actual forward pass, inside
:func:`train.beit.common.load_frozen_tokenizer`) BEFORE the data pipeline is built.

The codebook size is likewise NOT a CLI flag: the MIM head's ``vocab_size`` is read off
the loaded tokenizer, so a head narrower or wider than the codebook is unrepresentable
rather than merely discouraged.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.beit.train_mim \\
        --dataset imagenette --variant tiny --image-size 224 --patch-size 16 \\
        --tokenizer-checkpoint results/beit_tokenizer_.../final_model.keras \\
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
from typing import Any, Callable, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    set_seeds,
    save_config_json,
    create_callbacks as create_common_callbacks,
)
from train.beit.common import (
    SUPPORTED_DATASETS,
    build_optimizer,
    build_raw_image_dataset,
    load_frozen_tokenizer,
)
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.vision.beit_masking import (
    BEIT_MIN_MASK_PATCHES_PER_BLOCK,
    BEIT_NUM_MASK_PATCHES,
    make_beit_mim_map_fn,
)
from dl_techniques.models.beit import create_beit_mim

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for BEiT stage-1 MIM pre-training.

    Owned by THIS trainer (no shared config dataclass across the three BEiT trainers --
    a flag that never reaches its field must be a local, greppable, testable defect).

    ``vocab_size`` is deliberately ABSENT: it is read off the stage-0 tokenizer at build
    time, so it cannot drift from the codebook it indexes.
    """

    # Data
    dataset: str = "imagenette"  # imagenette | cifar10
    image_size: int = 224
    patch_size: int = 16
    batch_size: int = 32
    augment_data: bool = True

    # Tokenizer (stage 0). Its code grid MUST equal image_size // patch_size; verified
    # by a forward pass in `load_frozen_tokenizer` before the data pipeline is built.
    tokenizer_checkpoint: Optional[str] = None

    # Masking (BEiT's run_beit_pretraining.py defaults)
    num_mask_patches: int = BEIT_NUM_MASK_PATCHES  # 75 of 196 == 38.3%
    min_mask_patches_per_block: int = BEIT_MIN_MASK_PATCHES_PER_BLOCK  # 16

    # Model
    variant: str = "base"  # tiny | small | base | large
    drop_path_rate: float = 0.1

    # Training
    epochs: int = 100
    learning_rate: float = 1.5e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    gradient_clipping: float = 3.0

    # Monitoring
    early_stopping_patience: int = 15

    # Debug
    max_steps: Optional[int] = None  # cap steps_per_epoch (smoke runs); None = full epoch

    # Output -- H13: repo-root `results/`, NEVER `src/results/`.
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Runtime
    seed: int = 42
    gpu: Optional[int] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"beit_mim_{self.dataset}_{self.variant}_{timestamp}"

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size "
                f"({self.patch_size})"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if not (0.0 <= self.drop_path_rate < 1.0):
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {self.drop_path_rate}"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: "
                f"{sorted(SUPPORTED_DATASETS)}"
            )

        # Mask budget vs the grid this configuration actually produces. The masking
        # generator validates the same things, but only once the tf.data pipeline is
        # being built -- minutes later, behind a tokenizer load.
        num_patches = self.num_patches
        if self.num_mask_patches <= 0 or self.num_mask_patches > num_patches:
            raise ValueError(
                f"num_mask_patches must be in [1, {num_patches}] for a "
                f"{self.patch_grid} patch grid, got {self.num_mask_patches}"
            )
        if self.min_mask_patches_per_block <= 0:
            raise ValueError(
                "min_mask_patches_per_block must be positive, got "
                f"{self.min_mask_patches_per_block}"
            )
        if self.min_mask_patches_per_block > self.num_mask_patches:
            raise ValueError(
                f"min_mask_patches_per_block ({self.min_mask_patches_per_block}) exceeds "
                f"the mask budget num_mask_patches ({self.num_mask_patches})"
            )

        if self.tokenizer_checkpoint is None:
            raise ValueError(
                "tokenizer_checkpoint is required: BEiT's MIM targets are the discrete "
                "code ids of a stage-0 tokenizer. Train one with "
                "`python -m train.beit.train_tokenizer` first."
            )
        # A typo'd path must fail HERE, before the data pipeline warm-up, and it must
        # fail rather than degrade into some other objective.
        if not str(self.tokenizer_checkpoint).endswith(".keras"):
            raise ValueError(
                "tokenizer_checkpoint must be a .keras checkpoint, got "
                f"{self.tokenizer_checkpoint!r}"
            )
        if not Path(self.tokenizer_checkpoint).exists():
            raise FileNotFoundError(
                f"tokenizer_checkpoint not found: {self.tokenizer_checkpoint}"
            )

    @property
    def patch_grid(self) -> Tuple[int, int]:
        """The encoder's ``(gh, gw)`` patch grid. The tokenizer must match it exactly."""
        side = self.image_size // self.patch_size
        return (side, side)

    @property
    def num_patches(self) -> int:
        return self.patch_grid[0] * self.patch_grid[1]


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_datasets(
        config: TrainingConfig,
        tokenizer_fn: Callable[[Any], Any],
) -> Tuple[Any, Any, int, int]:
    """Build the BEiT MIM train/val pipelines.

    Both datasets yield ``((image, bool_mask), target_ids, sample_weight)``.

    Args:
        config: The trainer's configuration.
        tokenizer_fn: A BATCHED tokenizer callable from
            :func:`train.beit.common.load_frozen_tokenizer`. Its grid has already been
            verified against ``config.patch_grid``.

    Returns:
        ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
    """
    map_fn = make_beit_mim_map_fn(
        # The map fn hands over ONE unbatched image; the tokenizer wants a batch.
        tokenizer_fn=lambda img: tokenizer_fn(img[None])[0],
        grid_size=config.patch_grid,
        num_masking_patches=config.num_mask_patches,
        min_num_patches=config.min_mask_patches_per_block,
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

def train_mim(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate BEiT MIM pre-training.

    Returns:
        Dict with ``model``, ``best_val_loss``, ``first_loss``, ``final_loss``,
        ``run_dir``, ``vocab_size``, ``patch_grid``, ``history``.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | variant={config.variant} "
        f"dataset={config.dataset} image={config.image_size} patch={config.patch_size} "
        f"grid={config.patch_grid} mask={config.num_mask_patches}/{config.num_patches} "
        f"tokenizer={config.tokenizer_checkpoint}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Tokenizer FIRST: the grid guard runs before anything expensive ----
    # `load_frozen_tokenizer` raises if the code grid at this image size is not
    # `config.patch_grid`. That check is the only thing standing between a config typo
    # and an entire pre-training run against spatially misaligned targets.
    input_shape = (config.image_size, config.image_size, 3)
    tokenizer_fn = load_frozen_tokenizer(
        config.tokenizer_checkpoint,
        expected_grid=config.patch_grid,
        image_shape=input_shape,
    )
    vocab_size = int(tokenizer_fn.model.num_embeddings)

    # ---- Data ----
    train_ds, val_ds, steps_per_epoch, val_steps = build_datasets(config, tokenizer_fn)
    logger.info(f"Steps per epoch: {steps_per_epoch}, val steps: {val_steps}")

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): AdamW's decay comes from optimizer_builder alone.
    model = create_beit_mim(
        variant=config.variant,
        input_shape=input_shape,
        patch_size=config.patch_size,
        vocab_size=vocab_size,
        drop_path_rate=config.drop_path_rate,
    )
    # Probe build so summary()/count_params() work before fit.
    model.build([(None,) + input_shape, (None, config.num_patches)])
    model.summary(print_fn=logger.info)

    # ---- Optimization ----
    optimizer = build_optimizer(config, steps_per_epoch)

    # STOCK compile. The head emits LOGITS over the codebook; the mask is a sample_weight
    # in the dataset, NOT a train_step override (H-8).
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # ---- Callbacks ----
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="beit_mim",
        run_dir=str(run_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

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

    final_model_path = run_dir / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Saved MIM model to {final_model_path}")

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
        "final_model_path": str(final_model_path),
        "vocab_size": vocab_size,
        "patch_grid": config.patch_grid,
        "history": history,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="BEiT stage-1 masked-image-modeling pre-training",
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

    # Tokenizer
    parser.add_argument("--tokenizer-checkpoint", type=str, default=None,
                        help="REQUIRED. Stage-0 VQVAERotationTrick .keras checkpoint. Its "
                             "code grid must equal image-size // patch-size; the run "
                             "aborts at startup if it does not.")

    # Masking
    parser.add_argument("--num-mask-patches", type=int, default=BEIT_NUM_MASK_PATCHES,
                        help="Block-wise mask budget in patches (BEiT: 75 of 196).")
    parser.add_argument("--min-mask-patches-per-block", type=int,
                        default=BEIT_MIN_MASK_PATCHES_PER_BLOCK,
                        help="Minimum per-block target area (BEiT's script passes 16).")

    # Model
    parser.add_argument("--variant", type=str, default="base",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--drop-path-rate", type=float, default=0.1,
                        help="Maximum stochastic-depth rate (linear ramp across depth).")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=3.0,
                        help="Clip by global norm (BEiT pre-training uses 3.0).")

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

    Every flag in :func:`parse_arguments` must land in a field here. A flag that does not
    is a SILENT NO-OP: the run trains at the default while the command line says
    otherwise. Guarded by ``tests/test_train/test_beit/test_cli_wiring.py``.
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
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        num_mask_patches=args.num_mask_patches,
        min_mask_patches_per_block=args.min_mask_patches_per_block,
        variant=args.variant,
        drop_path_rate=args.drop_path_rate,
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
        f"grid={config.patch_grid}, mask={config.num_mask_patches}, "
        f"{config.epochs} epochs, batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}, "
        f"clip={config.gradient_clipping}, tokenizer={config.tokenizer_checkpoint}"
    )

    try:
        result = train_mim(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== MIM PRE-TRAINING DONE === first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} best_val_loss={result['best_val_loss']:.6f} "
        f"vocab={result['vocab_size']} checkpoint={result['final_model_path']}"
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
