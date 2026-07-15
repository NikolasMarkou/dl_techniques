"""Graph Energy Transformer graph-classification training — variant C-lite (Pattern 1).

Supervised whole-graph classification with :class:`GraphClassifier`: a shared
:class:`GraphEnergyTransformerBackbone` (``use_cls=True``, ``use_pe=True``) prepends a CLS
token, adds Laplacian positional encodings, descends ``S=4`` ET blocks over the binary graph
adjacency (rank-3 ``attention_mask``) with eq.-27 saddle-escape noise, and the head maps the
CLS token to per-class LOGITS. Trained on the TUDataset benchmarks (MUTAG / PROTEINS / NCI1 /
...) via :func:`build_tudataset_graph_dataset`, which downloads + caches OUTSIDE the repo.

**Label smoothing through STOCK ``fit`` — NO custom ``train_step``** (house rule). The dataset
yields ``(inputs_dict, integer_label)``. Keras 3.8's ``SparseCategoricalCrossentropy`` has NO
``label_smoothing`` argument, so to honor the paper's ``label_smoothing=0.05`` we ONE-HOT the
labels in the ``tf.data`` pipeline (a ``.map`` to ``tf.one_hot(label, num_classes)``) and
compile with ``keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)``.
The one-hot lives in the dataset map, not a custom loss or train_step. (If the one-hot path is
ever undesirable, the documented fallback is ``SparseCategoricalCrossentropy(from_logits=True)``
with label smoothing dropped — but the one-hot path is the default and is what runs here.)

Optimization defaults follow the paper's Table 9: AdamW, cosine schedule with warmup,
``label_smoothing=0.05``, 300 epochs, batch 32, and the Table-9 model dims (``embed_dim=128``,
``num_heads=12``, ``head_dim=64``, ``hopfield_dim=512``, ``num_blocks=S=4``, ``step_size=0.01``,
``noise_std=0.02``, ``pe_dim=15``) supplied by :func:`create_graph_classifier`.

**Gradient centralization.** Table 9 lists gradient centralization ON, but the repo's
``optimizer_builder`` (consumed via :func:`build_optimizer`) exposes gradient CLIPPING (by global
norm) and decoupled weight decay, NOT gradient centralization. Rather than hand-roll a
centralizing optimizer wrapper, we OMIT centralization and keep clip-by-global-norm as the
gradient stabilizer (same rationale + call site as the variant-B trainer). This is a documented
deviation from Table 9, not a silent one.

Usage:
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.graph_energy_transformer.train_classification \\
        --dataset mutag --epochs 300 --batch-size 32 --gpu 1

    # fast smoke on real MUTAG (tiny, network-cached):
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 python -m train.graph_energy_transformer.train_classification \\
        --dataset mutag --epochs 1 --batch-size 16

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
from dl_techniques.datasets.graphs import build_tudataset_graph_dataset
from dl_techniques.models.graph_energy_transformer import create_graph_classifier

# TUDataset graph-classification benchmarks. The loader parses any TUDataset whose ZIP follows
# the standard `<NAME>_A.txt` / `_graph_indicator.txt` / `_graph_labels.txt` layout; these are the
# common small ones. The value passed to the loader is upper-cased (`mutag` -> `MUTAG`).
SUPPORTED_DATASETS: Tuple[str, ...] = ("mutag", "proteins", "nci1", "enzymes", "dd", "ptc_mr")


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for variant-C-lite graph classification training.

    Optimization + model defaults are the paper's Table 9: AdamW, cosine schedule with warmup,
    ``label_smoothing=0.05``, 300 epochs, batch 32, and the Table-9 trunk dims. TUDataset graphs
    are small, so ``max_nodes`` caps the dense pad (graphs above the cap are dropped by the
    loader, never truncated).
    """

    # Data
    dataset: str = "mutag"  # mutag | proteins | nci1 | ...
    batch_size: int = 32
    max_nodes: Optional[int] = None  # None -> per-batch dynamic pad; int -> fixed cap
    k_pe: int = 15  # Laplacian-PE width (must match model pe_dim)
    add_self_loops: bool = True
    sign_flip: bool = True  # per-epoch PE sign-flip augmentation

    # Model (Table 9 defaults, forwarded to create_graph_classifier)
    embed_dim: int = 128
    num_heads: int = 12
    head_dim: int = 64
    hopfield_dim: int = 512
    num_blocks: int = 4  # S -- stacked ET blocks (Table 9)
    num_steps: int = 12  # T -- descent steps per block
    step_size: float = 0.01  # alpha (Table 9)
    noise_std: float = 0.02  # eq.-27 saddle-escape (training only; Table 9)
    head_dropout: float = 0.0

    # Training (paper Table 9)
    epochs: int = 300
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"  # Table 9: AdamW (decoupled weight decay)
    lr_schedule_type: str = "cosine_decay"  # cosine schedule with warmup (Table 9)
    warmup_epochs: int = 5
    weight_decay: float = 1e-4  # only consumed when optimizer_type == 'adamw'
    label_smoothing: float = 0.05  # Table 9; applied via one-hot + CategoricalCrossentropy
    # LOAD-BEARING: the backward pass composes S*T Jacobians through the unrolled energy descent
    # and the gradient norm spikes. Clip by global norm (same rationale as the image ET trainer).
    # NOTE: Table-9 gradient CENTRALIZATION is not offered by the repo optimizer builder and is
    # omitted (see module docstring); clip-by-global-norm is the retained gradient stabilizer.
    gradient_clipping: float = 1.0

    # Monitoring
    early_stopping_patience: int = 30

    # Runtime / precision
    mixed_precision: bool = False
    # jit_compile: None -> Keras default ("auto", XLA on). Variant C uses a STATIC index-0 CLS
    # slice (no dynamic gather like variant B), so it is expected to compile under XLA; set
    # False only if a run surfaces an XLA error. The dedicated fp16/XLA guard test exercises
    # jit_compile=True explicitly regardless of this trainer knob.
    jit_compile: Optional[bool] = None
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
            self.experiment_name = f"graph_et_classify_{self.dataset}_{timestamp}"

        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset {self.dataset!r}; supported: {sorted(SUPPORTED_DATASETS)}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_nodes is not None and self.max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive when set, got {self.max_nodes}")
        if self.k_pe <= 0:
            raise ValueError(f"k_pe must be positive, got {self.k_pe}")
        for _n, _v in (
            ("embed_dim", self.embed_dim), ("num_heads", self.num_heads),
            ("head_dim", self.head_dim), ("hopfield_dim", self.hopfield_dim),
            ("num_blocks", self.num_blocks), ("num_steps", self.num_steps),
        ):
            if _v <= 0:
                raise ValueError(f"{_n} must be positive, got {_v}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
        if not (0.0 <= self.head_dropout <= 1.0):
            raise ValueError(f"head_dropout must be in [0, 1], got {self.head_dropout}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive when set, got {self.max_steps}")


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def _one_hot_labels(dataset: Any, num_classes: int) -> Any:
    """Map ``(inputs, int_label) -> (inputs, one_hot_label)`` for label-smoothed categorical CE.

    Keras 3.8's ``SparseCategoricalCrossentropy`` has NO ``label_smoothing``; to honor Table 9's
    ``label_smoothing=0.05`` we one-hot the integer labels here (in the ``tf.data`` pipeline) and
    compile with ``CategoricalCrossentropy(label_smoothing=...)``. This keeps STOCK ``fit`` with
    NO custom ``train_step`` / ``compute_loss``.
    """
    import tensorflow as tf

    def _map(inputs: Any, label: Any):
        onehot = tf.one_hot(tf.cast(label, tf.int32), depth=num_classes, dtype=tf.float32)
        return inputs, onehot

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE)


def build_datasets(
        config: TrainingConfig,
) -> Tuple[Any, Any, Any, int, int]:
    """Build one-hot ``(inputs_dict, one_hot_label)`` train/val/test TUDataset pipelines.

    Returns:
        ``(train_ds, val_ds, test_ds, num_features, num_classes)``. ``num_features`` is the
        node-feature dim ``F`` the model's ``node_feature_dim`` must match; ``num_classes`` sizes
        both the one-hot depth and the classifier head.
    """
    # The loader is case-sensitive: the TUDataset host serves UPPER-cased archives (`MUTAG.zip`,
    # extracting to `MUTAG/MUTAG_A.txt`). Our CLI keys are lower-cased for ergonomics, so map to
    # the canonical host name here. `.upper()` is exact for every SUPPORTED_DATASETS key
    # (MUTAG / PROTEINS / NCI1 / ENZYMES / DD / PTC_MR).
    ds_name = config.dataset.upper()

    train_ds, train_meta = build_tudataset_graph_dataset(
        name=ds_name,
        split="train",
        batch_size=config.batch_size,
        k_pe=config.k_pe,
        max_nodes=config.max_nodes,
        add_self_loops=config.add_self_loops,
        sign_flip=config.sign_flip,
        seed=config.seed,
        shuffle=True,
    )
    val_ds, _ = build_tudataset_graph_dataset(
        name=ds_name,
        split="val",
        batch_size=config.batch_size,
        k_pe=config.k_pe,
        max_nodes=config.max_nodes,
        add_self_loops=config.add_self_loops,
        sign_flip=False,
        seed=config.seed,
        shuffle=False,
    )
    test_ds, _ = build_tudataset_graph_dataset(
        name=ds_name,
        split="test",
        batch_size=config.batch_size,
        k_pe=config.k_pe,
        max_nodes=config.max_nodes,
        add_self_loops=config.add_self_loops,
        sign_flip=False,
        seed=config.seed,
        shuffle=False,
    )

    num_features = int(train_meta["num_node_features"])
    num_classes = int(train_meta["num_classes"])
    logger.info(
        f"dataset={config.dataset} F={num_features} C={num_classes} "
        f"k_pe={config.k_pe} split_sizes={train_meta['split_sizes']}"
    )

    train_ds = _one_hot_labels(train_ds, num_classes)
    val_ds = _one_hot_labels(val_ds, num_classes)
    test_ds = _one_hot_labels(test_ds, num_classes)
    return train_ds, val_ds, test_ds, num_features, num_classes


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_classification(config: TrainingConfig) -> Dict[str, Any]:
    """Orchestrate variant-C-lite graph classification training.

    Returns:
        Dict with ``model``, ``best_val_accuracy``, ``first_loss``, ``final_loss``, ``run_dir``,
        ``num_classes``, ``jit_compile``, ``history``.
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
        f"K={config.hopfield_dim} S={config.num_blocks} T={config.num_steps} "
        f"alpha={config.step_size} noise_std={config.noise_std} "
        f"label_smoothing={config.label_smoothing} mixed_precision={config.mixed_precision}"
    )

    run_dir = Path(config.output_dir) / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(run_dir), "config.json")

    # ---- Data ----
    train_ds, val_ds, test_ds, num_features, num_classes = build_datasets(config)

    # ---- Model ----
    # No kernel_regularizer anywhere (H10): weight decay, if any, comes from optimizer_builder.
    model = create_graph_classifier(
        node_feature_dim=num_features,
        num_classes=num_classes,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        hopfield_dim=config.hopfield_dim,
        num_blocks=config.num_blocks,
        num_steps=config.num_steps,
        step_size=config.step_size,
        noise_std=config.noise_std,
        pe_dim=config.k_pe,
        head_dropout=config.head_dropout,
        seed=config.seed,
    )
    # Materialize weights with a real forward pass over one probe batch before compile
    # (subclassed lazy-build serialization hazard — build() alone leaves lazy sub-layers unshaped).
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
    # steps_per_epoch for the LR horizon: the TUDataset generator yields one finite pass per
    # epoch. Estimate the horizon from the train split size when not smoke-capped.
    horizon_steps = steps_per_epoch or max(1, split_size_estimate(config, num_classes))
    optimizer = build_optimizer(config, horizon_steps)

    # STOCK compile. Head emits LOGITS (no softmax in-graph) -> from_logits=True. Labels are
    # one-hot (see _one_hot_labels), so CategoricalCrossentropy carries label_smoothing=0.05.
    compile_kwargs: Dict[str, Any] = dict(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=config.label_smoothing
        ),
        metrics=["accuracy"],
    )
    # jit_compile: leave to the Keras default ("auto") unless explicitly set. Variant C uses a
    # STATIC index-0 CLS slice (unlike variant B's dynamic gather), so XLA is expected to work.
    if config.jit_compile is not None:
        compile_kwargs["jit_compile"] = config.jit_compile
    model.compile(**compile_kwargs)

    # ---- Callbacks ----
    # Classification -> monitor val_accuracy (max). create_common_callbacks selects 'max' when
    # 'accuracy' is in the monitor name. No EnergyTraceCallback (the graph backbone surfaces no
    # energy trace) and no analyzer (not dict-input aware).
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="graph_et_classify",
        run_dir=str(run_dir),
        monitor="val_accuracy",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

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
        "results_dir": results_dir,
        "num_classes": num_classes,
        "jit_compile": config.jit_compile,
        "history": history,
    }


def split_size_estimate(config: TrainingConfig, num_classes: int) -> int:
    """Coarse train-split step count for the LR horizon.

    The exact train-split size is only known after the loader runs; use a small conservative
    estimate that keeps the cosine horizon sane for the tiny TUDataset benchmarks (MUTAG has
    ~150 train graphs). The horizon only needs to be the right order of magnitude.
    """
    approx_train_graphs = {
        "mutag": 150, "proteins": 890, "nci1": 3288, "enzymes": 480,
        "dd": 942, "ptc_mr": 275,
    }.get(config.dataset, 500)
    return max(1, approx_train_graphs // config.batch_size)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_arguments(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse the CLI. ``argv=None`` reads ``sys.argv`` (the wiring test passes an explicit list)."""
    parser = argparse.ArgumentParser(
        description="Train the Graph Energy Transformer graph classifier (variant C-lite)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--dataset", type=str, default="mutag",
                        choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Fixed node cap N (graphs above it are dropped). None = per-batch pad.")
    parser.add_argument("--k-pe", type=int, default=15,
                        help="Laplacian positional-encoding width (must match model pe_dim).")
    parser.add_argument("--no-self-loops", dest="add_self_loops", action="store_false",
                        help="Disable adjacency self-loops (default: enabled).")
    parser.add_argument("--no-sign-flip", dest="sign_flip", action="store_false",
                        help="Disable per-epoch PE sign-flip augmentation (default: enabled).")

    # Model (Table 9)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--hopfield-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4,
                        help="S: stacked ET blocks (Table 9).")
    parser.add_argument("--num-steps", type=int, default=12,
                        help="T: energy-descent steps per block.")
    parser.add_argument("--step-size", type=float, default=0.01,
                        help="alpha: descent step size (Table 9).")
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="eq.-27 saddle-escape noise std (training only; Table 9).")
    parser.add_argument("--head-dropout", type=float, default=0.0)

    # Training (paper Table 9)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--lr-schedule", type=str, default="cosine_decay",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Only consumed when --optimizer adamw.")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Applied via one-hot labels + CategoricalCrossentropy (Table 9).")
    parser.add_argument("--gradient-clipping", type=float, default=1.0,
                        help="Clip by global norm; the unrolled descent spikes the gradient norm.")

    # Monitoring
    parser.add_argument("--early-stopping-patience", type=int, default=30)

    # Runtime / precision
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true",
                        help="Enable mixed_float16 (the graph backbone applies the D-002 "
                             "variable_dtype fix so the ET block runs in float32).")
    jit_group = parser.add_mutually_exclusive_group()
    jit_group.add_argument("--jit-compile", dest="jit_compile", action="store_const", const=True,
                           default=None, help="Force XLA on (default: Keras 'auto').")
    jit_group.add_argument("--no-jit-compile", dest="jit_compile", action="store_const",
                           const=False, help="Force XLA off.")
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
        k_pe=args.k_pe,
        add_self_loops=args.add_self_loops,
        sign_flip=args.sign_flip,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        hopfield_dim=args.hopfield_dim,
        num_blocks=args.num_blocks,
        num_steps=args.num_steps,
        step_size=args.step_size,
        noise_std=args.noise_std,
        head_dropout=args.head_dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clipping=args.gradient_clipping,
        early_stopping_patience=args.early_stopping_patience,
        mixed_precision=args.mixed_precision,
        jit_compile=args.jit_compile,
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
        f"S={config.num_blocks}, T={config.num_steps}, lr={config.learning_rate}, "
        f"opt={config.optimizer_type}, wd={config.weight_decay}, "
        f"label_smoothing={config.label_smoothing}, clip={config.gradient_clipping}, "
        f"mixed_precision={config.mixed_precision}"
    )

    try:
        result = train_classification(config)
    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        raise

    logger.info(
        f"=== CLASSIFICATION TRAINING DONE === num_classes={result['num_classes']} "
        f"first_loss={result['first_loss']:.6f} final_loss={result['final_loss']:.6f} "
        f"best_val_accuracy={result['best_val_accuracy']:.4f} run_dir={result['run_dir']}"
    )


if __name__ == "__main__":
    main()
