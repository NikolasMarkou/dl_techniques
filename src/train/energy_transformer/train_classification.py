"""Energy Transformer image-classification training (Pattern 1).

Supervised training of :class:`EnergyTransformerClassifier` — the SAME Energy Transformer
trunk as ``train_masked_completion.py``, with a LayerNorm -> mean-pool -> ``Dense`` logits head
instead of the MIM decoder. The trunk can be WARM-STARTED from a masked-completion checkpoint:

    --pretrained-encoder results/et_mim_.../best_model.keras

which is the whole point of the pair of models. The transfer is asserted, not logged: a
warm-start that silently moves ZERO layers trains from random init while the command line says
"pretrained", and the (poor) accuracy is then blamed on the pretraining objective. See
:func:`warm_start_encoder`.

**No custom ``train_step``** (H6): stock ``compile`` + ``fit``. The classifier passes NO
``input_mask``, so the backbone's ``mask_token`` sub-layer is never called — but it IS still
created and built (invariant I6), which is exactly what keeps the two trunks weight-identical
and the transfer complete.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.energy_transformer.train_classification \\
        --dataset imagenette --variant tiny --image-size 224 --patch-size 16 \\
        --epochs 100 --batch-size 32 --gpu 1 \\
        --pretrained-encoder results/et_mim_imagenette_tiny_.../best_model.keras
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
from dl_techniques.utils.weight_transfer import (
    TransferReport,
    load_weights_from_checkpoint,
)
from dl_techniques.models.energy_transformer import (
    BACKBONE_NAME,
    create_energy_transformer_classifier,
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for Energy Transformer classification training.

    Deliberately PARALLEL to ``train_masked_completion.TrainingConfig`` (same optimization
    defaults, paper Table 4). The only structural additions are ``num_classes``,
    ``dropout_rate`` and ``pretrained_encoder``.

    **One learning rate.** There is no trunk-lr multiplier, no layer-wise lr decay and no
    freeze flag: a warm-started trunk trains at exactly the same lr as the head. Adding a
    second lr is a real experiment, not a default — it belongs behind a measured comparison,
    not inside the first version of this trainer.
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
    dropout_rate: float = 0.0  # before the final Dense
    num_classes: Optional[int] = None  # None -> from the dataset

    # Warm start. Points at an EnergyTransformerMIM `.keras` checkpoint; only the
    # `et_backbone` trunk transfers (the MIM `decoder_*` head is skipped by prefix).
    pretrained_encoder: Optional[str] = None

    # Training (paper Table 4)
    epochs: int = 100
    learning_rate: float = 5e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    # LOAD-BEARING, same as the MIM trainer: the backward pass composes T=12 Jacobians through
    # the unrolled energy descent and the gradient norm spikes. The paper's model does not
    # train above lr 1e-4 without clipping.
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
            self.experiment_name = f"et_cls_{self.dataset}_{self.variant}_{timestamp}"

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
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive when set, got {self.num_classes}"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}"
            )
        if self.pretrained_encoder is not None:
            # A typo'd path must fail HERE, before an hour of data pipeline warm-up, and it
            # must fail rather than degrade to a random-init run.
            if not str(self.pretrained_encoder).endswith(".keras"):
                raise ValueError(
                    "pretrained_encoder must be a .keras checkpoint, got "
                    f"{self.pretrained_encoder!r}"
                )
            if not Path(self.pretrained_encoder).exists():
                raise FileNotFoundError(
                    f"pretrained_encoder checkpoint not found: {self.pretrained_encoder}"
                )


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_datasets(config: TrainingConfig) -> Tuple[Any, Any, int, int, int]:
    """Build the ``(image, label)`` train/val pipelines.

    NO ``element_map_fn``: the classifier is fed raw images, with no occlusion mask. (The
    backbone's ``mask_token`` weight still exists and is still built — I6 — which is what
    keeps this trunk weight-identical to the MIM trunk.)

    Returns:
        ``(train_ds, val_ds, steps_per_epoch, val_steps, num_classes)``.
    """
    train_ds, num_train, num_classes = build_raw_image_dataset(
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
    return train_ds, val_ds, steps_per_epoch, val_steps, num_classes


# ---------------------------------------------------------------------
# WARM START
# ---------------------------------------------------------------------

# The MIM decoder (`decoder_norm`, `decoder_proj`) is the ONLY thing in the source checkpoint
# that must not cross over; the classifier's own head is named `head_*` and simply does not
# exist in the source. Listing `head_` here too keeps `missing_in_source` clean, so that list
# staying EMPTY is a real signal.
WARM_START_SKIP_PREFIXES: Tuple[str, ...] = ("decoder_", "head_")


def warm_start_encoder(
        model: keras.Model,
        ckpt_path: str,
        skip_prefixes: Tuple[str, ...] = WARM_START_SKIP_PREFIXES,
) -> TransferReport:
    """Transfer the ``et_backbone`` trunk from an ``EnergyTransformerMIM`` checkpoint.

    ``load_weights_from_checkpoint`` matches layers BY NAME (``model.load_weights(...,
    by_name=True)`` is broken for ``.keras`` files in Keras 3.8). Both models compose their
    backbone under the same name, ``"et_backbone"``, and build it from the same config, so the
    whole trunk moves in ONE ``set_weights`` call — bit-exact, not approximate.

    **The transfer is ASSERTED, not logged.** A run that transfers zero layers is
    indistinguishable from a random-init run except by its accuracy, and by then the blame
    lands on the pretraining objective rather than on the plumbing. So:

    * ``et_backbone`` MUST appear in ``report.loaded`` — not merely "something loaded";
    * a shape mismatch on the trunk (a config drift between the two runs) is FATAL, not a
      warning, because ``strict=False`` would otherwise skip it and leave the trunk at init.

    Args:
        model: A BUILT :class:`EnergyTransformerClassifier`.
        ckpt_path: Path to the source ``.keras`` MIM checkpoint.
        skip_prefixes: Source-layer name prefixes NOT to transfer.

    Returns:
        The :class:`TransferReport` (fields: ``loaded``, ``skipped_by_prefix``,
        ``shape_mismatch``, ``missing_in_source``, ``unused_in_source``).

    Raises:
        RuntimeError: If the trunk did not transfer, for any reason.
    """
    logger.info(f"Warm-starting the encoder from {ckpt_path}")
    report = load_weights_from_checkpoint(
        target=model,
        ckpt_path=ckpt_path,
        skip_prefixes=skip_prefixes,
        strict=False,
    )

    mismatched = [name for name, _, _ in report.shape_mismatch]
    logger.info(
        f"warm start: loaded={report.loaded} "
        f"skipped_by_prefix={report.skipped_by_prefix} "
        f"shape_mismatch={mismatched} "
        f"missing_in_source={report.missing_in_source} "
        f"unused_in_source={report.unused_in_source}"
    )

    if BACKBONE_NAME in mismatched:
        raise RuntimeError(
            f"Warm start FAILED: the {BACKBONE_NAME!r} trunk shapes do not match the "
            f"checkpoint {ckpt_path!r}. The classifier and the MIM model were built from "
            f"DIFFERENT backbone configs (variant / image-size / patch-size / num-steps). "
            f"Mismatch detail:\n{report.summary_string()}"
        )
    if BACKBONE_NAME not in report.loaded:
        raise RuntimeError(
            f"Warm start FAILED: transferred 0 weights into {BACKBONE_NAME!r} from "
            f"{ckpt_path!r} (loaded={report.loaded}). The run would have trained from RANDOM "
            f"INIT while claiming to be pretrained. Is that checkpoint an "
            f"EnergyTransformerMIM?\n{report.summary_string()}"
        )

    num_arrays = len(model.get_layer(BACKBONE_NAME).get_weights())
    logger.info(
        f"Warm start OK: {report.num_loaded} layer(s) loaded "
        f"({BACKBONE_NAME}: {num_arrays} weight arrays), "
        f"{len(report.skipped_by_prefix)} skipped by prefix"
    )
    return report


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_classification(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate supervised classification training.

    Returns:
        Dict with ``model``, ``best_val_accuracy``, ``first_loss``, ``final_loss``,
        ``run_dir``, ``transfer_report`` (``None`` without ``--pretrained-encoder``),
        ``history``.
    """
    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | variant={config.variant} "
        f"dataset={config.dataset} image={config.image_size} patch={config.patch_size} "
        f"pretrained_encoder={config.pretrained_encoder}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Data ----
    train_ds, val_ds, steps_per_epoch, val_steps, dataset_classes = build_datasets(config)
    if config.num_classes is None:
        num_classes = dataset_classes
    elif config.num_classes != dataset_classes:
        # A head narrower than the label range gives a nan/garbage loss; a wider one silently
        # wastes capacity. Neither should be a shrug.
        raise ValueError(
            f"num_classes={config.num_classes} contradicts dataset {config.dataset!r}, which "
            f"has {dataset_classes} classes."
        )
    else:
        num_classes = config.num_classes
    logger.info(f"Steps per epoch: {steps_per_epoch}, val steps: {val_steps}, "
                f"num_classes: {num_classes}")

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): AdamW's decay comes from optimizer_builder alone.
    input_shape = (config.image_size, config.image_size, 3)
    model = create_energy_transformer_classifier(
        variant=config.variant,
        input_shape=input_shape,
        patch_size=config.patch_size,
        num_classes=num_classes,
        dropout_rate=config.dropout_rate,
        num_steps=config.num_steps,
    )
    # Probe build BEFORE the transfer -- load_weights_from_checkpoint requires a built target
    # (its layers must already be weight-shaped for the set_weights to land).
    model.build((None,) + input_shape)
    model.summary(print_fn=logger.info)

    # ---- Warm start (asserted) ----
    transfer_report: Optional[TransferReport] = None
    if config.pretrained_encoder is not None:
        transfer_report = warm_start_encoder(model, config.pretrained_encoder)
    else:
        logger.info("No --pretrained-encoder given: training the trunk from RANDOM INIT.")

    # ---- Optimization ----
    optimizer = build_optimizer(config, steps_per_epoch)

    # STOCK compile. The head emits LOGITS (no softmax) -> from_logits=True.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # ---- Callbacks ----
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="et_cls",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )
    # One FIXED val batch (images only -- the classifier passes no mask), held for the whole
    # run so the epoch-to-epoch energy traces are comparable.
    probe_images, _ = next(iter(val_ds))
    callbacks.append(EnergyTraceCallback(
        probe_inputs=probe_images,
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
    val_acc_curve = history.history.get("val_accuracy", []) or [float("nan")]

    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(run_dir / "training_history.json", "w") as handle:
            json.dump(history_dict, handle, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        logger.warning(f"Failed to save training history: {exc}")

    gc.collect()
    return {
        "model": model,
        "best_val_accuracy": float(max(val_acc_curve)),
        "first_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "run_dir": str(run_dir),
        "transfer_report": transfer_report,
        "history": history,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="Train the Energy Transformer image classifier",
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
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                        help="Dropout before the final Dense")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Default: from the dataset. Must agree with it if given.")

    # Warm start
    parser.add_argument("--pretrained-encoder", type=str, default=None,
                        help="EnergyTransformerMIM .keras checkpoint. Transfers the "
                             "'et_backbone' trunk ONLY; the MIM decoder is skipped. The "
                             "transfer is asserted -- a 0-layer transfer aborts the run.")

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
        dropout_rate=args.dropout_rate,
        num_classes=args.num_classes,
        pretrained_encoder=args.pretrained_encoder,
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
        f"opt={config.optimizer_type}, wd={config.weight_decay}, clip={config.gradient_clipping}, "
        f"pretrained_encoder={config.pretrained_encoder}"
    )

    try:
        result = train_classification(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== CLASSIFICATION TRAINING DONE === first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} "
        f"best_val_accuracy={result['best_val_accuracy']:.4f} "
        f"run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------
