"""
Head comparison on CIFAR-100 (ViT trunk, five output heads).

Compares a softmax head against harmonic heads (flat HarMax, hierarchical
HarMax) on a FIXED ViT trunk, one run per (head, seed). Every run writes
the standard run-directory contract (``config.json``, ``best_model.keras``,
``training_log.csv``, ``training_history.json``) plus ``test_metrics.json``
(fine/coarse/tail accuracy, ECE), ``test_features.npz`` (trunk features +
labels for representation analysis) and ``head_prototypes.npz``.

Arms (``--head``):

- ``softmax`` -- Dense(100) + softmax, cross-entropy. Baseline.
- ``harmonic_logits`` -- HarmonicDense(``logits``) + cross-entropy
  (``from_logits=True``).
- ``harmonic_dist`` -- HarmonicDense(``distances``) + HarMax +
  cross-entropy (``from_logits=False``).
- ``hier_fixed`` -- HierarchicalHarmonicHead, identity assignment, fixed n.
- ``hier_full`` -- HierarchicalHarmonicHead + periodic reassignment +
  per-level exponent annealing.

Usage:
    python -m train.head_comparison.train_heads --head hier_full \\
        --epochs 50 --seed 0 --output-dir results --experiment-name pilot
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import keras
from keras.datasets import cifar100 as cifar100_datasets

from train.common import (
    setup_gpu,
    create_callbacks as create_common_callbacks,
    load_dataset,
)
from train.common.run_io import prepare_run_dir, save_training_history_json
from train.common.seed import set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.vit import create_vit
from dl_techniques.layers.structured_linear.harmonic_dense import HarmonicDense
from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
    HierarchicalHarmonicHead,
)
from dl_techniques.layers.activations.harmax import HarMax
from dl_techniques.callbacks.periodic_reassign import PeriodicReassignCallback
from dl_techniques.callbacks.anneal_harmonic_exponent import (
    HarmonicExponentAnnealingCallback,
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

HEAD_ARMS = (
    "softmax",
    "harmonic_logits",
    "harmonic_dist",
    "hier_fixed",
    "hier_full",
)

NUM_FINE = 100
NUM_COARSE = 20
ECE_BINS = 15


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeadComparisonConfig:
    """Configuration for one (head, seed) comparison run. Every field is live."""

    # Model
    model_variant: str = "vit_pico"
    patch_size: int = 4
    image_size: int = 32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    head: str = "softmax"

    # Harmonic heads
    harmonic_n: float = 4.0
    hier_branching: str = "10,10"

    # Hierarchical schedule
    reassign_every: int = 2
    reassign_start: int = 1
    anneal: bool = True
    anneal_schedule: str = "linear"

    # Training
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    gradient_clipping: float = 1.0
    momentum: float = 0.9
    early_stopping_patience: int = 15
    seed: int = 0

    # Analysis / IO
    analyzer: bool = True
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    gpu: Optional[int] = None

    def __post_init__(self) -> None:
        if self.head not in HEAD_ARMS:
            raise ValueError(f"head must be one of {HEAD_ARMS}, got {self.head!r}")
        if self.harmonic_n <= 0:
            raise ValueError(f"harmonic_n must be positive, got {self.harmonic_n}")
        branching = tuple(int(b) for b in self.hier_branching.split(","))
        if any(b <= 0 for b in branching) or int(np.prod(branching)) < NUM_FINE:
            raise ValueError(
                f"hier_branching must cover {NUM_FINE} classes, "
                f"got {self.hier_branching!r}"
            )
        if self.reassign_every < 1:
            raise ValueError(f"reassign_every must be >= 1, got {self.reassign_every}")
        if self.reassign_start < 1:
            raise ValueError(f"reassign_start must be >= 1, got {self.reassign_start}")
        if self.anneal_schedule not in ("linear", "cosine", "exp"):
            raise ValueError(
                f"anneal_schedule must be linear/cosine/exp, "
                f"got {self.anneal_schedule!r}"
            )

    @property
    def branching(self) -> Tuple[int, ...]:
        """Parsed hierarchical branching tuple."""
        return tuple(int(b) for b in self.hier_branching.split(","))


# =============================================================================
# METRICS
# =============================================================================

def expected_calibration_error(
        probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Mean |accuracy - confidence| over equal-width confidence bins.

    :param probs: ``(N, C)`` predicted distributions.
    :type probs: np.ndarray
    :param labels: ``(N,)`` integer labels.
    :type labels: np.ndarray
    :param n_bins: Number of bins.
    :type n_bins: int
    :return: ECE in [0, 1].
    :rtype: float
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def derive_fine_to_coarse(
        y_fine: np.ndarray, y_coarse: np.ndarray
) -> np.ndarray:
    """Recover the fixed fine->coarse table by majority vote.

    Each CIFAR-100 fine class belongs to exactly one superclass, so the
    vote must be unanimous; anything less raises (data integrity guard).

    :param y_fine: ``(N,)`` fine labels in ``[0, 100)``.
    :type y_fine: np.ndarray
    :param y_coarse: ``(N,)`` coarse labels in ``[0, 20)``.
    :type y_coarse: np.ndarray
    :return: ``(100,)`` coarse id per fine id.
    :rtype: np.ndarray
    :raises ValueError: If a fine class maps to two superclasses, or a
        fine class is absent.
    """
    table = np.full(NUM_FINE, -1, dtype=np.int64)
    for fine in range(NUM_FINE):
        votes = y_coarse[y_fine == fine]
        if votes.size == 0:
            raise ValueError(f"Fine class {fine} has no samples.")
        ids, counts = np.unique(votes, return_counts=True)
        if len(ids) != 1:
            raise ValueError(
                f"Fine class {fine} maps to superclasses {ids.tolist()}; "
                f"the CIFAR-100 taxonomy is one-to-one."
            )
        table[fine] = ids[0]
    return table


def to_probabilities(raw: np.ndarray, from_logits: bool) -> np.ndarray:
    """Convert head outputs to probabilities for value-based metrics.

    Arms with ``from_logits=True`` losses emit unnormalized logits;
    argmax metrics are unaffected either way, but ECE needs probabilities.

    :param raw: ``(N, C)`` head outputs.
    :type raw: np.ndarray
    :param from_logits: Whether to apply a softmax first.
    :type from_logits: bool
    :return: ``(N, C)`` non-negative rows summing to 1.
    :rtype: np.ndarray
    """
    raw = np.asarray(raw, dtype=np.float64)
    if not from_logits:
        return raw
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def tail_decile_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_FINE
) -> float:
    """Mean per-class accuracy over the worst decile of classes.

    :param y_true: ``(N,)`` integer labels.
    :type y_true: np.ndarray
    :param y_pred: ``(N,)`` integer predictions.
    :type y_pred: np.ndarray
    :param num_classes: Class count.
    :type num_classes: int
    :return: Tail-decile accuracy.
    :rtype: float
    """
    per_class = np.array([
        (y_pred[y_true == c] == c).mean() if (y_true == c).any() else np.nan
        for c in range(num_classes)
    ])
    per_class = per_class[~np.isnan(per_class)]
    k = max(1, len(per_class) // 10)
    return float(np.sort(per_class)[:k].mean())


# =============================================================================
# MODEL
# =============================================================================

def build_model(config: HeadComparisonConfig) -> Tuple[keras.Model, Any, Any]:
    """Build trunk + head for one arm.

    Returns the uncompiled model, the head layer, and the trunk-output
    tensor (for the feature extractor used by reassignment and analysis).
    Compilation (arm-specific loss) happens in :func:`train_heads`.

    :param config: Run configuration.
    :type config: HeadComparisonConfig
    :return: ``(model, head_layer, trunk_output_tensor)``.
    :rtype: Tuple[keras.Model, Any, Any]
    """
    use_adamw = config.optimizer_type.lower() == "adamw"
    kernel_reg = (
        None
        if use_adamw
        else (keras.regularizers.L2(config.weight_decay) if config.weight_decay > 0 else None)
    )
    backbone = create_vit(
        variant=config.model_variant,
        input_shape=(config.image_size, config.image_size, 3),
        patch_size=config.patch_size,
        include_top=False,
        pooling="cls",
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        kernel_regularizer=kernel_reg,
    )
    inputs = keras.Input((config.image_size, config.image_size, 3))
    trunk_out = backbone(inputs)
    n = config.harmonic_n
    # Losses follow the arm at compile time in train_heads():
    # softmax / harmonic_logits train from_logits=True, the rest
    # (already-normalized probabilities) with from_logits=False.
    if config.head == "softmax":
        outputs = keras.layers.Dense(NUM_FINE, name="head")(trunk_out)
    elif config.head == "harmonic_logits":
        outputs = HarmonicDense(NUM_FINE, n=n, output_mode="logits", name="head")(
            trunk_out
        )
    elif config.head == "harmonic_dist":
        distances = HarmonicDense(NUM_FINE, n=n, output_mode="distances")(trunk_out)
        outputs = HarMax(n=n, name="head")(distances)
    elif config.head in ("hier_fixed", "hier_full"):
        outputs = HierarchicalHarmonicHead(
            NUM_FINE, branching=config.branching, n=(1.0, n), name="head"
        )(trunk_out)
    else:  # pragma: no cover - __post_init__ rejects unknown arms
        raise ValueError(f"Unknown head {config.head!r}")
    model = keras.Model(inputs, outputs)
    head = model.get_layer("head")
    return model, head, trunk_out


def build_callbacks(
        config: HeadComparisonConfig,
        run_dir: str,
        model_input: Any,
        head: Any,
        trunk_out: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
) -> List[keras.callbacks.Callback]:
    """Assemble training callbacks: common set plus arm extras.

    :param config: Run configuration.
    :type config: HeadComparisonConfig
    :param run_dir: Prepared run directory.
    :type run_dir: str
    :param model_input: Model input tensor (extractor input).
    :type model_input: Any
    :param head: Head layer (reassign target for ``hier_full``).
    :type head: Any
    :param trunk_out: Trunk-output tensor for the feature extractor.
    :type trunk_out: Any
    :param x_train: Training inputs for class-means computation.
    :type x_train: np.ndarray
    :param y_train: Training fine labels.
    :type y_train: np.ndarray
    :return: Callback list.
    :rtype: List[keras.callbacks.Callback]
    """
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name or f"{config.head}",
        results_dir_prefix="heads",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
    )
    if config.head == "hier_full":
        # Prepended, not appended: on_epoch_end fires in list order, so the
        # reassignment lands BEFORE ModelCheckpoint / EarlyStopping snapshot.
        # Otherwise the final epoch's reassignment would mutate the kernel
        # past the saved best, leaving best_model.keras one E-step staler
        # than the in-memory model every post-hoc artifact is computed from
        # (measured 0.004 test-acc drift on hier_full_s0).
        extractor = keras.Model(model_input, trunk_out)
        extras = [PeriodicReassignCallback(
            head,
            x_train,
            y_train,
            NUM_FINE,
            feature_model=extractor,
            every_n_epochs=config.reassign_every,
            start_epoch=config.reassign_start,
        )]
        if config.anneal:
            extras.append(HarmonicExponentAnnealingCallback(
                schedule=config.anneal_schedule,
                n_init=[1.0, min(2.0, config.harmonic_n)],
                n_final=[2.0, config.harmonic_n],
                total_epochs=config.epochs,
                layer_names=["head"],
            ))
        callbacks = extras + callbacks
    return callbacks


# =============================================================================
# TRAINING
# =============================================================================

def train_heads(
        config: HeadComparisonConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Run one (head, seed) comparison arm end to end.

    :param config: Run configuration.
    :type config: HeadComparisonConfig
    :param gpu_id: GPU id for ``setup_gpu``.
    :type gpu_id: Optional[int]
    :return: Dict with ``model``, ``history``, ``test_metrics`` and paths.
    :rtype: Dict[str, Any]
    """
    setup_gpu(gpu_id)
    set_seeds(config.seed)
    if config.experiment_name is None:
        config.experiment_name = f"{config.head}_seed{config.seed}"
    logger.info(f"Experiment: {config.experiment_name}, head: {config.head}")

    run_dir = prepare_run_dir(config)

    # ---- Data (fine via the shared loader, coarse alongside it) ----
    (x_train, y_train), (x_test, y_test), _, _ = load_dataset("cifar100")
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    # Coarse (superclass) labels ride alongside the shared fine loader.
    (_, y_train_coarse), (_, y_test_coarse) = cifar100_datasets.load_data(
        label_mode="coarse"
    )
    y_train_coarse = np.asarray(y_train_coarse).reshape(-1)
    y_test_coarse = np.asarray(y_test_coarse).reshape(-1)
    fine_to_coarse = derive_fine_to_coarse(y_train, y_train_coarse)

    # ---- Model ----
    use_adamw = config.optimizer_type.lower() == "adamw"
    model, head, trunk_out = build_model(config)
    model.build((None, config.image_size, config.image_size, 3))

    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": (len(x_train) // config.batch_size) * config.epochs,
        "warmup_steps": (len(x_train) // config.batch_size) * config.warmup_epochs,
        "alpha": 0.01,
    })
    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if use_adamw:
        opt_config["weight_decay"] = config.weight_decay
    elif config.optimizer_type.lower() == "sgd":
        opt_config["momentum"] = config.momentum
    optimizer = optimizer_builder(opt_config, lr_schedule)
    # build_model returns the uncompiled graph; the loss follows the arm.
    from_logits = config.head in ("softmax", "harmonic_logits")
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    callbacks = build_callbacks(
        config, str(run_dir), model.input, head, trunk_out, x_train, y_train
    )
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    train_hours = (time.time() - start_time) / 3600.0
    save_training_history_json(history, str(run_dir))

    # ---- Post-hoc evaluation ----
    raw = np.asarray(model.predict(x_test, batch_size=512, verbose=0))
    probs = to_probabilities(
        raw, from_logits=config.head in ("softmax", "harmonic_logits")
    )
    y_pred = probs.argmax(axis=1)
    test_metrics = {
        "fine_accuracy": float((y_pred == y_test).mean()),
        "coarse_accuracy": float(
            (fine_to_coarse[y_pred] == y_test_coarse).mean()
        ),
        "tail_decile_accuracy": tail_decile_accuracy(y_test, y_pred),
        "ece": expected_calibration_error(probs, y_test),
        "train_hours": train_hours,
        "params": int(model.count_params()),
        "epochs_run": len(history.history.get("loss", [])),
    }
    with open(Path(run_dir) / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics: {test_metrics}")

    # ---- Artifacts for representation analysis ----
    extractor = keras.Model(model.input, trunk_out)
    test_features = np.asarray(
        extractor.predict(x_test, batch_size=512, verbose=0)
    )
    np.savez_compressed(
        Path(run_dir) / "test_features.npz",
        features=test_features, y_fine=y_test, y_coarse=y_test_coarse,
    )
    head_weights = {
        w.name.replace(":", "_"): np.asarray(w) for w in head.weights
    }
    np.savez_compressed(Path(run_dir) / "head_prototypes.npz", **head_weights)

    if config.analyzer:
        from train.common.evaluation import run_model_analysis
        run_model_analysis(
            model, (x_test, y_test), history,
            model_name=config.experiment_name, results_dir=str(run_dir),
        )

    gc.collect()
    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "run_dir": str(run_dir),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments (argv override for tests)."""
    parser = argparse.ArgumentParser(
        description="CIFAR-100 ViT head comparison: softmax vs harmonic vs hierarchical",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", type=str, default="vit_pico",
                        choices=["vit_pico", "vit_tiny", "vit_small",
                                 "vit_base", "vit_large", "vit_huge"])
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-type", type=str, default="adamw")
    parser.add_argument("--lr-schedule-type", type=str, default="cosine_decay")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--attention-dropout-rate", type=float, default=0.1)
    parser.add_argument("--head", type=str, default="softmax", choices=list(HEAD_ARMS))
    parser.add_argument("--harmonic-n", type=float, default=4.0)
    parser.add_argument("--hier-branching", type=str, default="10,10")
    parser.add_argument("--reassign-every", type=int, default=2)
    parser.add_argument("--reassign-start", type=int, default=1)
    parser.add_argument("--anneal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anneal-schedule", type=str, default="linear",
                        choices=["linear", "cosine", "exp"])
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyzer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HeadComparisonConfig:
    """Build a live config from parsed arguments."""
    return HeadComparisonConfig(
        model_variant=args.variant,
        patch_size=args.patch_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer_type,
        lr_schedule_type=args.lr_schedule_type,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        momentum=args.momentum,
        dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        head=args.head,
        harmonic_n=args.harmonic_n,
        hier_branching=args.hier_branching,
        reassign_every=args.reassign_every,
        reassign_start=args.reassign_start,
        anneal=args.anneal,
        anneal_schedule=args.anneal_schedule,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        analyzer=args.analyzer,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        gpu=args.gpu,
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """CLI entry point: parse argv first, then run one comparison arm."""
    args = parse_arguments(argv)
    config = config_from_args(args)
    return train_heads(config, gpu_id=config.gpu)


if __name__ == "__main__":
    main(sys.argv[1:])
