"""BEiT stage-0: discrete visual-tokenizer training (Pattern 1).

Trains the VQ-VAE that produces BEiT's MIM targets. BEiT v1 uses a frozen, externally
pre-trained DALL-E dVAE (Gumbel-softmax, 8192 codes); this repo has no Gumbel dVAE, so
the tokenizer is VQ-style — :class:`VQVAERotationTrick`, hard nearest-neighbour — exactly
as BEiT v2's own VQ-KD tokenizer is. That deviation is recorded as X-1 in
``models/beit/README.md`` and in ``decisions.md`` D-002 (class name corrected by D-010).

The ONLY thing stage 1 consumes from this stage is
``encode_to_indices(image) -> (B, gh, gw)`` integer code ids, and the ONLY thing that has
to line up is ``gh, gw == image_size // patch_size`` of the BEiT encoder. So the geometry
here is load-bearing and is validated in ``TrainingConfig.__post_init__``, not discovered
an hour into stage 1.

**Measured geometry (step-9 falsification probe, 2026-08-11).** Both candidate schemes
reach BEiT's 14x14 code grid through the auto-build path::

    input_shape=(224,224,3), downsample_factor=16  ->  encode_to_indices (1, 14, 14)
    input_shape=(112,112,3), downsample_factor=8   ->  encode_to_indices (1, 14, 14)
    input_shape=(64,64,3),   downsample_factor=16  ->  (1, 4, 4)   [control]

The DEFAULTS below take the first (D-004's single-resolution scheme): the SAME image
tensor feeds the BEiT encoder at ``patch_size=16`` and this tokenizer at
``downsample_factor=16``, so the MIM pipeline has one image transform and one ``tf.data``
branch. The reference's 112/8 split exists only because its dVAE was a fixed, externally
trained /8 model; we train our own, so that constraint does not bind. It remains
reachable: ``--image-size 112 --downsample-factor 8``.

**No custom ``train_step``** is added here (H-8). :class:`VQVAERotationTrick` already
supplies its own (reconstruction + VQ losses, reported under the ``loss`` key), so this
trainer uses stock ``compile()`` + ``fit()`` and monitors ``val_loss``.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.beit.train_tokenizer \\
        --dataset imagenette --image-size 224 --downsample-factor 16 \\
        --num-embeddings 8192 --epochs 50 --batch-size 32 --gpu 1
"""

import gc
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
)
from train.common.run_io import default_experiment_name, prepare_run_dir, save_training_history_json
from train.beit.common import (
    SUPPORTED_DATASETS,
    build_optimizer,
    build_raw_image_dataset,
)
from dl_techniques.utils.logger import logger
from dl_techniques.models.vq_vae_rotation.model import VQVAERotationTrick

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for BEiT stage-0 discrete-tokenizer training.

    Owned by THIS trainer. There is deliberately no shared config dataclass across the
    three BEiT trainers: a flag that never reaches its field must be a local, greppable,
    testable defect rather than an inherited one.
    """

    # Data
    dataset: str = "imagenette"  # imagenette | cifar10
    image_size: int = 224
    batch_size: int = 32
    augment_data: bool = True

    # Model (auto-build path of VQVAERotationTrick)
    downsample_factor: int = 16  # code grid = image_size // downsample_factor
    num_embeddings: int = 8192  # BEiT's codebook size (DALL-E dVAE parity)
    embedding_dim: int = 32
    hidden_channels: int = 128
    num_res_blocks: int = 2
    commitment_cost: float = 0.25
    use_ema: bool = False

    # Training
    epochs: int = 50
    learning_rate: float = 5e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0

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
            self.experiment_name = default_experiment_name("beit_tokenizer", self.dataset)

        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.downsample_factor <= 0:
            raise ValueError(
                f"downsample_factor must be positive, got {self.downsample_factor}"
            )
        # The auto-build encoder is a stack of stride-2 convs, so the factor must be a
        # power of 2. VQVAERotationTrick raises this too, but only after the data
        # pipeline has already been built.
        if self.downsample_factor & (self.downsample_factor - 1) != 0:
            raise ValueError(
                f"downsample_factor must be a power of 2, got {self.downsample_factor}"
            )
        # THE geometry check. A non-integral grid does not fail loudly downstream: the
        # conv stack silently floors, and stage 1 then reads every MIM target from the
        # wrong spatial position while producing a finite, plausible loss.
        if self.image_size % self.downsample_factor != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by downsample_factor "
                f"({self.downsample_factor}); it yields a non-integral code grid of "
                f"{self.image_size / self.downsample_factor} x "
                f"{self.image_size / self.downsample_factor}. Stage 1 requires the code "
                f"grid to equal the BEiT encoder's patch grid exactly."
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.num_embeddings <= 0:
            raise ValueError(
                f"num_embeddings must be positive, got {self.num_embeddings}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.hidden_channels <= 0:
            raise ValueError(
                f"hidden_channels must be positive, got {self.hidden_channels}"
            )
        if self.num_res_blocks < 0:
            raise ValueError(
                f"num_res_blocks must be non-negative, got {self.num_res_blocks}"
            )
        if self.commitment_cost < 0.0:
            raise ValueError(
                f"commitment_cost must be non-negative, got {self.commitment_cost}"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: "
                f"{sorted(SUPPORTED_DATASETS)}"
            )

    @property
    def code_grid(self) -> Tuple[int, int]:
        """The ``(gh, gw)`` code grid this configuration produces."""
        side = self.image_size // self.downsample_factor
        return (side, side)


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_datasets(config: TrainingConfig) -> Tuple[Any, Any, int, int]:
    """Build the raw-image train/val pipelines.

    NO ``element_map_fn``: the tokenizer reconstructs the raw image, so the stock
    ``(image, label)`` element is what it wants. ``VQVAERotationTrick.train_step`` takes
    ``data[0]`` and ignores the label.

    Returns:
        ``(train_ds, val_ds, steps_per_epoch, val_steps)``.
    """
    train_ds, num_train, _ = build_raw_image_dataset(
        config.dataset,
        config.image_size,
        config.batch_size,
        is_training=True,
        augment=config.augment_data,
        seed=config.seed,
    )
    val_ds, num_val, _ = build_raw_image_dataset(
        config.dataset,
        config.image_size,
        config.batch_size,
        is_training=False,
        seed=config.seed,
    )

    steps_per_epoch = max(1, num_train // config.batch_size)
    val_steps = max(1, num_val // config.batch_size)
    if config.max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps)
        val_steps = min(val_steps, config.max_steps)
    return train_ds, val_ds, steps_per_epoch, val_steps


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def build_tokenizer(config: TrainingConfig) -> VQVAERotationTrick:
    """Construct the auto-build tokenizer and VERIFY its code grid by a forward pass.

    The grid is measured, not derived: ``encode_to_indices`` is actually run on a zero
    image at ``config.image_size``, and its shape is compared against
    ``config.code_grid``. Deriving the "expected" grid from ``downsample_factor`` would
    make the check agree with the implementation by construction.

    Args:
        config: The trainer's configuration.

    Returns:
        A built :class:`VQVAERotationTrick`.

    Raises:
        RuntimeError: If the realized code grid is not ``config.code_grid``.
    """
    input_shape = (config.image_size, config.image_size, 3)
    model = VQVAERotationTrick(
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        input_shape=input_shape,
        downsample_factor=config.downsample_factor,
        hidden_channels=config.hidden_channels,
        num_res_blocks=config.num_res_blocks,
        commitment_cost=config.commitment_cost,
        use_ema=config.use_ema,
    )

    # A real forward pass, not `build(...)`: VQVAERotationTrick does not override
    # `build`, so `model.build(shape)` marks it built WITHOUT creating the sub-model
    # weights, and `summary()` / `count_params()` then raise.
    probe = np.zeros((1,) + input_shape, dtype="float32")
    _ = model(probe, training=False)

    code_ids = model.encode_to_indices(probe)
    actual_grid = tuple(int(v) for v in code_ids.shape[1:])
    if actual_grid != config.code_grid:
        raise RuntimeError(
            f"Tokenizer code grid is {actual_grid}, expected {config.code_grid} at "
            f"image_size={config.image_size}, downsample_factor="
            f"{config.downsample_factor}. Stage 1 would read every MIM target from the "
            f"wrong spatial position and still produce a finite, plausible loss."
        )
    logger.info(
        f"tokenizer: input={input_shape} -> code grid {actual_grid} "
        f"(N={actual_grid[0] * actual_grid[1]} tokens), codebook={config.num_embeddings}, "
        f"embedding_dim={config.embedding_dim}"
    )
    return model


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_tokenizer(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate stage-0 tokenizer training.

    Returns:
        Dict with ``model``, ``best_val_loss``, ``first_loss``, ``final_loss``,
        ``run_dir``, ``final_model_path``, ``code_grid``, ``history``.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | dataset={config.dataset} "
        f"image={config.image_size} downsample={config.downsample_factor} "
        f"grid={config.code_grid} codebook={config.num_embeddings}"
    )

    run_dir = prepare_run_dir(config)

    # ---- Data ----
    train_ds, val_ds, steps_per_epoch, val_steps = build_datasets(config)
    logger.info(f"Steps per epoch: {steps_per_epoch}, val steps: {val_steps}")

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): AdamW's decay comes from optimizer_builder alone.
    model = build_tokenizer(config)
    model.summary(print_fn=logger.info)

    # ---- Optimization ----
    optimizer = build_optimizer(config, steps_per_epoch)

    # STOCK compile: VQVAERotationTrick supplies its OWN train_step/test_step (recon + VQ
    # losses under the `loss` key). No loss argument, and NO custom train_step is added
    # here -- H-8 forbids one, and the class already has what it needs.
    model.compile(optimizer=optimizer)

    # ---- Callbacks ----
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="beit_tokenizer",
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

    # `final_model.keras` is what stage 1 should consume. `best_model.keras` (written by
    # ModelCheckpoint) is min-val_loss-selected, which is the right choice here -- there
    # is no curriculum -- but the final weights are the ones whose codebook statistics
    # match the end of training.
    final_model_path = run_dir / "final_model.keras"
    model.save(final_model_path)
    logger.info(f"Saved tokenizer to {final_model_path}")

    save_training_history_json(history, run_dir)

    gc.collect()
    return {
        "model": model,
        "best_val_loss": float(min(val_curve)),
        "first_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "run_dir": str(run_dir),
        "final_model_path": str(final_model_path),
        "code_grid": config.code_grid,
        "history": history,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="Train the BEiT stage-0 discrete visual tokenizer (VQ-VAE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="imagenette",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--image-size", type=int, default=None,
                        help="Auto: 224 for imagenette, 32 for cifar10")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-augmentation", dest="augment_data", action="store_false")

    # Model
    parser.add_argument("--downsample-factor", type=int, default=None,
                        help="Power of 2. Code grid = image-size // this. "
                             "Auto: 16 for imagenette (224 -> 14x14), 8 for cifar10 "
                             "(32 -> 4x4).")
    parser.add_argument("--num-embeddings", type=int, default=8192,
                        help="Codebook size. BEiT v1's dVAE uses 8192.")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--use-ema", action="store_true",
                        help="EMA codebook updates instead of gradient updates.")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                        help="Clip by global norm.")

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
    otherwise. That trap has bitten this repo before, which is why this mapping is an
    importable function with a dedicated test
    (``tests/test_train/test_beit/test_cli_wiring.py``) rather than a block in ``main()``.
    """
    dataset = args.dataset.lower()
    image_size = args.image_size if args.image_size is not None else (
        32 if dataset == "cifar10" else 224
    )
    downsample_factor = args.downsample_factor if args.downsample_factor is not None else (
        8 if dataset == "cifar10" else 16
    )

    return TrainingConfig(
        dataset=dataset,
        image_size=image_size,
        batch_size=args.batch_size,
        augment_data=args.augment_data,
        downsample_factor=downsample_factor,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        hidden_channels=args.hidden_channels,
        num_res_blocks=args.num_res_blocks,
        commitment_cost=args.commitment_cost,
        use_ema=args.use_ema,
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
        f"Config: dataset={config.dataset}, image={config.image_size}, "
        f"downsample={config.downsample_factor}, grid={config.code_grid}, "
        f"codebook={config.num_embeddings}, {config.epochs} epochs, "
        f"batch={config.batch_size}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}"
    )

    try:
        result = train_tokenizer(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== TOKENIZER TRAINING DONE === first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} best_val_loss={result['best_val_loss']:.6f} "
        f"grid={result['code_grid']} checkpoint={result['final_model_path']}"
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
