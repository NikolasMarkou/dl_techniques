"""Graph Energy Transformer node-anomaly training — variant B (Pattern 1).

Supervised node-anomaly detection with :class:`GraphAnomalyDetector`: a single shared
:class:`GraphEnergyTransformerBackbone` block descends for ``T`` steps over a bounded k-hop
subgraph, the head reads the TARGET node's LayerNormed ``g_1 || g_T`` state and maps it to a
single anomaly LOGIT. Trained on the CARE-GNN fraud benchmarks (Amazon / YelpChi) with a
synthetic, network-free fallback for smoke runs and CI.

**Severe class imbalance is handled through STOCK ``fit`` — NO custom ``train_step``** (house
rule). The dataset yields ``(inputs_dict, label)``; the anomalous class (label ``1``) is
up-weighted by ``pos_weight = ω = benign / anomalous`` (from the loader metadata). The clean
stock-fit path for per-class weighting is ``model.fit(..., class_weight={0: 1.0, 1: ω})``; if
that is incompatible with the dict-input signature in this Keras version, the trainer falls
back to yielding a per-sample ``sample_weight`` third element from the dataset (``ω`` where
``label == 1`` else ``1.0``). Both are stock-fit-legal. The active path is logged.

Optimization defaults follow the paper's Table 6: Adam, lr ``1e-3``, 100 epochs, batch 32.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.graph_energy_transformer.train_anomaly \\
        --dataset amazon --epochs 100 --batch-size 32 --num-steps 2 --gpu 1

    # network-free smoke:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.graph_energy_transformer.train_anomaly \\
        --dataset synthetic --epochs 1 --batch-size 16 --max-nodes 48

Results are written under repo-root ``results/`` (H9 — NEVER ``src/results/``).
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
from train.graph_energy_transformer.common import build_optimizer
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.graphs import build_fraud_subgraph_dataset
from dl_techniques.models.graph_energy_transformer import create_graph_anomaly_detector

# Real CARE-GNN fraud benchmarks + the synthetic fallback. `synthetic` routes the loader to a
# network-free generator; the two real keys download+cache under $GRAPH_CACHE.
SUPPORTED_DATASETS: Tuple[str, ...] = ("amazon", "yelpchi", "synthetic")


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for variant-B graph node-anomaly training.

    Optimization defaults are the paper's Table 6 (Adam, lr ``1e-3``, 100 epochs, batch 32).
    The model dims default to a small trunk suitable for the bounded k-hop subgraphs; the
    descent depth ``num_steps`` (``T ∈ {1, 2, 3}``) defaults to 2.
    """

    # Data
    dataset: str = "synthetic"  # amazon | yelpchi | synthetic
    batch_size: int = 32
    max_nodes: int = 64
    num_hops: int = 2
    # Synthetic-fallback graph knobs (ignored for the real benchmarks).
    synth_nodes: int = 2000
    synth_features: int = 25
    synth_anomaly_ratio: float = 0.1

    # Model (a small trunk over bounded subgraphs)
    embed_dim: int = 64
    num_heads: int = 4
    head_dim: int = 16
    hopfield_dim: int = 128
    num_steps: int = 2  # T -- the block's internal descent steps (paper T ∈ {1, 2, 3})
    step_size: float = 0.1  # alpha
    mlp_hidden_dim: int = 64
    mlp_dropout: float = 0.0

    # Training (paper Table 6)
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer_type: str = "adam"  # Table 6: plain Adam (no decoupled weight decay)
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 0
    weight_decay: float = 0.0  # only consumed when optimizer_type == 'adamw'
    # LOAD-BEARING: the backward pass composes T Jacobians through the unrolled energy descent
    # and the gradient norm spikes. Clip by global norm (same rationale as the image ET trainer).
    gradient_clipping: float = 1.0

    # Monitoring
    early_stopping_patience: int = 15

    # Runtime / precision
    mixed_precision: bool = False
    seed: int = 42
    gpu: Optional[int] = None

    # Debug
    max_steps: Optional[int] = None  # cap steps_per_epoch (smoke runs); None = full pass

    # Output -- H9: repo-root `results/`, NEVER `src/results/`.
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"graph_et_anomaly_{self.dataset}_{timestamp}"

        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {self.max_nodes}")
        if self.num_hops <= 0:
            raise ValueError(f"num_hops must be positive, got {self.num_hops}")
        for _n, _v in (
            ("embed_dim", self.embed_dim), ("num_heads", self.num_heads),
            ("head_dim", self.head_dim), ("hopfield_dim", self.hopfield_dim),
            ("mlp_hidden_dim", self.mlp_hidden_dim),
        ):
            if _v <= 0:
                raise ValueError(f"{_n} must be positive, got {_v}")
        if not (1 <= self.num_steps <= 3):
            # Paper Table 6 sweeps T ∈ {1, 2, 3}; refuse an out-of-range descent depth loudly.
            raise ValueError(f"num_steps must be in [1, 3], got {self.num_steps}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if not (0.0 <= self.mlp_dropout <= 1.0):
            raise ValueError(f"mlp_dropout must be in [0, 1], got {self.mlp_dropout}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def build_datasets(
        config: TrainingConfig,
) -> Tuple[Any, Any, float, int]:
    """Build the ``(inputs_dict, label)`` train/val subgraph pipelines.

    Returns:
        ``(train_ds, val_ds, pos_weight, num_features)``. ``pos_weight`` (= ω = benign /
        anomalous) drives the class-weighted BCE; ``num_features`` is the node-feature dim ``F``
        the model's ``node_feature_dim`` must match.
    """
    synthetic = config.dataset == "synthetic"

    train_ds, train_meta = build_fraud_subgraph_dataset(
        name=config.dataset,
        split="train",
        batch_size=config.batch_size,
        num_hops=config.num_hops,
        max_nodes=config.max_nodes,
        synthetic=synthetic,
        n_synth_nodes=config.synth_nodes,
        synth_num_features=config.synth_features,
        synth_anomaly_ratio=config.synth_anomaly_ratio,
        seed=config.seed,
    )
    val_ds, _ = build_fraud_subgraph_dataset(
        name=config.dataset,
        split="val",
        batch_size=config.batch_size,
        num_hops=config.num_hops,
        max_nodes=config.max_nodes,
        synthetic=synthetic,
        n_synth_nodes=config.synth_nodes,
        synth_num_features=config.synth_features,
        synth_anomaly_ratio=config.synth_anomaly_ratio,
        seed=config.seed,
        shuffle=False,
    )

    pos_weight = float(train_meta["pos_weight"])
    num_features = int(train_meta["num_features"])
    logger.info(
        f"dataset={config.dataset} F={num_features} "
        f"anomaly_ratio={train_meta['anomaly_ratio']:.4f} pos_weight(ω)={pos_weight:.4f} "
        f"split_sizes={train_meta['split_sizes']}"
    )
    return train_ds, val_ds, pos_weight, num_features


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_anomaly(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate variant-B node-anomaly training.

    Returns:
        Dict with ``model``, ``best_val_auc``, ``first_loss``, ``final_loss``, ``run_dir``,
        ``pos_weight``, ``weighting`` (``'class_weight'`` or ``'sample_weight'``), ``history``.
    """
    if config.mixed_precision:
        # The graph backbone replicates the image ET's variable_dtype fix (D-002), so a
        # mixed_float16 policy runs the ET block in float32 while the head stays fp16-safe.
        keras.mixed_precision.set_global_policy("mixed_float16")

    setup_gpu(config.gpu)
    set_seeds(config.seed)

    logger.info(
        f"Experiment: {config.experiment_name} | dataset={config.dataset} "
        f"embed_dim={config.embed_dim} heads={config.num_heads}x{config.head_dim} "
        f"K={config.hopfield_dim} T={config.num_steps} alpha={config.step_size} "
        f"mixed_precision={config.mixed_precision}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Data ----
    train_ds, val_ds, pos_weight, num_features = build_datasets(config)

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): weight decay, if any, comes from optimizer_builder.
    model = create_graph_anomaly_detector(
        node_feature_dim=num_features,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        hopfield_dim=config.hopfield_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        num_steps=config.num_steps,
        mlp_dropout=config.mlp_dropout,
        step_size=config.step_size,
        seed=config.seed,
    )
    # Materialize weights with a real forward pass over one probe batch. build() alone is
    # config-driven, but a dummy forward is the surest way to shape every lazy sub-layer before
    # compile (subclassed lazy-build serialization hazard).
    probe_inputs, _ = next(iter(train_ds))
    _ = model(probe_inputs, training=False)
    model.summary(print_fn=logger.info)

    # ---- Steps per epoch (cap for smoke runs) ----
    steps_per_epoch: Optional[int] = None
    val_steps: Optional[int] = None
    if config.max_steps is not None:
        steps_per_epoch = config.max_steps
        val_steps = config.max_steps
        train_ds = train_ds.repeat()

    # ---- Optimization ----
    # steps_per_epoch for the LR horizon: unknown a priori for the from_generator dataset, so
    # use a coarse estimate from the train split size (cosine decay only needs a horizon).
    split_sizes_est = max(1, config.synth_nodes if config.dataset == "synthetic" else 10000)
    horizon_steps = steps_per_epoch or max(1, split_sizes_est // config.batch_size)
    optimizer = build_optimizer(config, horizon_steps)

    # STOCK compile. The head emits a LOGIT (no sigmoid) -> from_logits=True on loss AND AUC.
    # BinaryAccuracy has no from_logits; threshold=0.0 is the logit-space decision boundary.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.AUC(from_logits=True, name="auc"),
            keras.metrics.BinaryAccuracy(threshold=0.0, name="binary_accuracy"),
        ],
        # XLA left to the Keras default ("auto", on). Step 9 replaced the head's dynamic
        # per-target `take_along_axis` gather with a STATIC index-0 slice (the sampler always
        # puts the target at index 0), so the head is now XLA-safe and the old `jit_compile=False`
        # workaround is gone. The fp16/XLA training path is exercised by the dedicated guard test
        # (tests/test_models/test_graph_energy_transformer/test_model.py).
    )

    # ---- Callbacks ----
    # Monitor val_loss (min): create_common_callbacks picks 'max' only when 'accuracy' is in the
    # monitor name, so a 'val_auc' monitor would silently get the WRONG (min) mode. val_loss is
    # the correct-by-default checkpoint signal under imbalance. No EnergyTraceCallback: the graph
    # backbone surfaces no energy trace (see common.py). No analyzer: it is not dict-input aware.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="graph_et_anomaly",
        run_dir=str(run_dir),
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    # ---- Class imbalance via STOCK fit (no custom train_step) ----
    # Prefer class_weight (leaves the dataset contract untouched). If this Keras version rejects
    # class_weight for the (dict-inputs, scalar-label) signature, fall back to a per-sample
    # sample_weight yielded from the dataset. Both are stock-fit-legal.
    class_weight = {0: 1.0, 1: float(pos_weight)}
    weighting = "class_weight"
    logger.info(f"Class imbalance: weighting={weighting} class_weight={class_weight}")

    def _fit(cw: Optional[Dict[int, float]], tds: Any, vds: Any):
        return model.fit(
            tds,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=vds,
            validation_steps=val_steps,
            class_weight=cw,
            callbacks=callbacks,
            verbose=1,
        )

    start = time.time()
    try:
        history = _fit(class_weight, train_ds, val_ds)
    except (ValueError, TypeError) as exc:
        # class_weight incompatible with the dict-input signature -> documented fallback.
        logger.warning(
            f"class_weight rejected by fit ({exc}); falling back to a per-sample sample_weight "
            f"map (ω={pos_weight:.4f} on label==1)."
        )
        weighting = "sample_weight"
        train_ds, val_ds, pos_weight, num_features = build_datasets(config)
        train_ds = _attach_sample_weight(train_ds, pos_weight)
        val_ds = _attach_sample_weight(val_ds, pos_weight)
        if config.max_steps is not None:
            train_ds = train_ds.repeat()
        history = _fit(None, train_ds, val_ds)
    logger.info(f"Training completed in {(time.time() - start) / 3600.0:.3f} hours")

    loss_curve = history.history.get("loss", []) or [float("nan")]
    val_auc_curve = history.history.get("val_auc", []) or [float("nan")]

    try:
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(run_dir / "training_history.json", "w") as handle:
            json.dump(history_dict, handle, indent=2)
    except Exception as exc:  # pragma: no cover - best-effort artifact
        logger.warning(f"Failed to save training history: {exc}")

    gc.collect()
    return {
        "model": model,
        "best_val_auc": float(max(val_auc_curve)),
        "first_loss": float(loss_curve[0]),
        "final_loss": float(loss_curve[-1]),
        "run_dir": str(run_dir),
        "results_dir": results_dir,
        "pos_weight": float(pos_weight),
        "weighting": weighting,
        "history": history,
    }


def _attach_sample_weight(dataset: Any, pos_weight: float) -> Any:
    """Map ``(inputs, label) -> (inputs, label, sample_weight)`` with ``ω`` on the anomalies.

    The fallback weighting path: ``sample_weight = ω`` where ``label == 1`` else ``1.0``. This
    is the same up-weighting ``class_weight`` would apply, expressed as a per-sample tensor —
    stock-fit-legal, no custom ``train_step``.
    """
    import tensorflow as tf

    def _map(inputs: Any, label: Any):
        w = tf.where(
            tf.equal(tf.cast(label, tf.int32), 1),
            tf.cast(pos_weight, tf.float32),
            tf.cast(1.0, tf.float32),
        )
        return inputs, label, w

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the wiring test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="Train the Graph Energy Transformer node-anomaly detector (variant B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=64,
                        help="Subgraph node cap N (padded to exactly this).")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="BFS depth for the k-hop subgraph sampler.")
    parser.add_argument("--synth-nodes", type=int, default=2000,
                        help="Node count for the synthetic fallback graph.")
    parser.add_argument("--synth-features", type=int, default=25,
                        help="Feature dim for the synthetic fallback graph.")
    parser.add_argument("--synth-anomaly-ratio", type=float, default=0.1,
                        help="Anomaly fraction for the synthetic fallback graph.")

    # Model
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--hopfield-dim", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=2,
                        help="T: the ET block's internal energy-descent steps (paper T ∈ {1,2,3}).")
    parser.add_argument("--step-size", type=float, default=0.1,
                        help="alpha: descent step size.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-dropout", type=float, default=0.0)

    # Training (paper Table 6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Only consumed when --optimizer adamw.")
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                        help="Clip by global norm; the unrolled descent spikes the gradient norm.")

    # Monitoring
    parser.add_argument("--early-stopping-patience", type=int, default=15)

    # Runtime / precision
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true",
                        help="Enable mixed_float16 (the graph backbone applies the D-002 "
                             "variable_dtype fix so the ET block runs in float32).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Debug
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps_per_epoch (and val steps). Smoke runs only.")

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Map a parsed ``Namespace`` onto a :class:`TrainingConfig`. PURE -- no side effects.

    Every flag in :func:`parse_arguments` must land in a field here. A flag that does not is a
    SILENT NO-OP: the run trains at the default while the command line says otherwise. That trap
    has bitten this repo before (bfunet trainer arg wiring), which is why this mapping is an
    importable pure function with a dedicated CLI-wiring test rather than a block inside
    ``main()``.
    """
    return TrainingConfig(
        dataset=args.dataset.lower(),
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        num_hops=args.num_hops,
        synth_nodes=args.synth_nodes,
        synth_features=args.synth_features,
        synth_anomaly_ratio=args.synth_anomaly_ratio,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        hopfield_dim=args.hopfield_dim,
        num_steps=args.num_steps,
        step_size=args.step_size,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        early_stopping_patience=args.early_stopping_patience,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        gpu=args.gpu,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    config = config_from_args(parse_arguments())

    logger.info(
        f"Config: dataset={config.dataset}, {config.epochs} epochs, batch={config.batch_size}, "
        f"max_nodes={config.max_nodes}, T={config.num_steps}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, clip={config.gradient_clipping}, "
        f"mixed_precision={config.mixed_precision}"
    )

    try:
        result = train_anomaly(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== ANOMALY TRAINING DONE === weighting={result['weighting']} "
        f"pos_weight={result['pos_weight']:.4f} first_loss={result['first_loss']:.6f} "
        f"final_loss={result['final_loss']:.6f} best_val_auc={result['best_val_auc']:.4f} "
        f"run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()
