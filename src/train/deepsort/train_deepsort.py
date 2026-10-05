"""DeepSORT appearance-embedding training (Pattern 1: vision classification).

Trains :class:`DeepSortAppearanceNet` as a person re-identification model.
Two objectives behind ``--loss-mode`` (transcribing
``cosine_metric_learning``):

- ``cosine-softmax`` (default, the mode the reference descriptor trains
  with): identity classification over cosine-softmax logits (stock sparse
  softmax CE ``from_logits=True``; the cosine parameterization lives in the
  :class:`CosineClassifier` head, not in the loss);
- ``triplet``: batch-hard softmargin triplet loss over the L2-normalized
  features (:class:`SoftmarginTripletLoss`).

Batches are always PK (``--p-ids`` identities x ``--k-shots`` shots) sampled
by :func:`pk_batch_generator`; the triplet mode needs K >= 2. Two pair
sources behind ``--data-source``:

- ``synthetic`` (default): seeded color/stripe identities -- offline,
  deterministic, what the smoke tests use;
- ``market1501``: the standard on-disk Market1501 layout under
  ``--market1501-root`` (no TFDS builder exists in the pinned version).

Validation holds out disjoint identities (10%, transcribed) and reports
CMC rank-1 plus mAP over cosine ranking alongside ``val_loss``.

Usage:
    MPLBACKEND=Agg .venv/bin/python -m train.deepsort.train_deepsort \\
        --loss-mode cosine-softmax --epochs 5 --p-ids 8 --k-shots 4 --gpu 1

    MPLBACKEND=Agg .venv/bin/python -m train.deepsort.train_deepsort \\
        --data-source market1501 --market1501-root /data/Market-1501-v15.09.15 \\
        --loss-mode triplet --epochs 50 --gpu 1
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from train.common import setup_gpu, create_callbacks as create_common_callbacks
from train.common.run_io import (
    default_experiment_name,
    prepare_run_dir,
    save_training_history_json,
)
from train.common import set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.vision.deepsort import create_deepsort_embedding
from dl_techniques.losses.reid_cosine_loss import SoftmarginTripletLoss
from dl_techniques.datasets.vision.reid import (
    REID_IMAGE_SHAPE,
    synthetic_reid_generator,
    read_market1501_split,
    reindex_person_ids,
    create_id_validation_split,
    cmc_and_map,
    pk_batch_generator,
    _load_image_128x64,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepSortTrainingConfig:
    """Configuration for DeepSORT appearance-embedding training."""

    # Data
    data_source: str = "synthetic"
    market1501_root: Optional[str] = None
    num_synthetic_ids: int = 24
    shots_per_id: int = 8
    p_ids: int = 8
    k_shots: int = 4
    validation_id_fraction: float = 0.1
    augment: bool = True
    seed: int = 0
    prefetch_buffer: int = tf.data.AUTOTUNE

    # Model / objective
    loss_mode: str = "cosine-softmax"
    dropout_rate: float = 0.4

    # Training
    epochs: int = 5
    steps_per_epoch: int = 20
    val_steps: int = 5
    learning_rate: float = 1e-3
    optimizer_type: str = "adam"
    lr_schedule_type: str = "constant"
    warmup_epochs: int = 0
    weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    momentum: float = 0.9  # used only when optimizer_type='sgd'

    # Monitoring
    early_stopping_patience: int = 10

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            self.experiment_name = default_experiment_name(
                "deepsort", f"{self.loss_mode}_{self.data_source}"
            )
        if self.data_source not in ("synthetic", "market1501"):
            raise ValueError(
                f"data_source must be 'synthetic' or 'market1501', got {self.data_source!r}"
            )
        if self.loss_mode not in ("cosine-softmax", "triplet"):
            raise ValueError(
                f"loss_mode must be 'cosine-softmax' or 'triplet', got {self.loss_mode!r}"
            )
        if self.p_ids <= 0 or self.k_shots <= 0:
            raise ValueError("p_ids and k_shots must be positive")
        if self.loss_mode == "triplet" and self.k_shots < 2:
            raise ValueError("triplet mode needs k_shots >= 2 (positives per anchor)")
    @property
    def batch_size(self) -> int:
        """Effective batch size, always ``p_ids * k_shots`` (not a field)."""
        return self.p_ids * self.k_shots
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.0 < self.validation_id_fraction < 1.0:
            raise ValueError("validation_id_fraction must be in (0, 1)")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")


# =============================================================================
# DATA PIPELINE
# =============================================================================

def load_split_arrays(
    config: DeepSortTrainingConfig,
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int],
           List[np.ndarray], List[int], int]:
    """Materialize train/fit-val/retrieval images with disjoint val identities.

    Train and fit-validation share the training identities (fit-val holds
    out the last shot of every train identity, so the cosine-softmax head
    always sees in-range labels); retrieval evaluation uses disjoint
    identities.

    Returns:
        ``(train_images, train_ids, val_fit_images, val_fit_ids,
        val_images, val_ids, num_train_ids)`` with contiguous identity
        labels; ``num_train_ids`` sizes the cosine-softmax head.
    :raises ValueError: If ``shots_per_id`` leaves no holdout shot.
    """
    if config.data_source == "synthetic":
        pairs = list(
            synthetic_reid_generator(
                config.num_synthetic_ids, config.shots_per_id, seed=config.seed
            )
        )
        images = [image for image, _ in pairs]
        raw_ids = [identity for _, identity in pairs]
    else:
        if not config.market1501_root:
            raise ValueError(
                "market1501_root is required for data_source='market1501'"
            )
        filenames, raw_pids, _ = read_market1501_split(config.market1501_root)
        images = [_load_image_128x64(path) for path in filenames]
        raw_ids = raw_pids
    ids, _ = reindex_person_ids(raw_ids)
    train_idx, val_idx = create_id_validation_split(
        np.array(ids), fraction=config.validation_id_fraction, seed=config.seed
    )
    train_images = [images[i] for i in train_idx.tolist()]
    train_ids = [ids[i] for i in train_idx.tolist()]
    val_images = [images[i] for i in val_idx.tolist()]
    val_ids = [ids[i] for i in val_idx.tolist()]
    train_ids, id_mapping = reindex_person_ids(train_ids)
    num_train_ids = len(id_mapping)
    # Fit-validation: last shot of every training identity (same label
    # space as the classifier). Needs shots_per_id >= 2 by construction.
    by_id: Dict[int, List[int]] = {}
    for pos, identity in enumerate(train_ids):
        by_id.setdefault(identity, []).append(pos)
    if min(len(v) for v in by_id.values()) < 2:
        raise ValueError(
            "shots_per_id must leave a holdout shot for fit-validation "
            "(need >= 2 shots per training identity)"
        )
    val_fit_positions = [v[-1] for v in by_id.values()]
    keep = [i for i in range(len(train_images)) if i not in set(val_fit_positions)]
    val_fit_images = [train_images[i] for i in val_fit_positions]
    val_fit_ids = [train_ids[i] for i in val_fit_positions]
    train_images = [train_images[i] for i in keep]
    train_ids = [train_ids[i] for i in keep]
    return (
        train_images, train_ids, val_fit_images, val_fit_ids,
        val_images, val_ids, num_train_ids,
    )


def create_pk_dataset(
    images: List[np.ndarray],
    ids: List[int],
    config: DeepSortTrainingConfig,
    train: bool,
) -> tf.data.Dataset:
    """PK-batch ``tf.data`` pipeline over in-memory images."""
    split_seed = config.seed + (0 if train else 1_000_000)
    signature = (
        tf.TensorSpec((config.batch_size,) + REID_IMAGE_SHAPE, tf.float32),
        tf.TensorSpec((config.batch_size,), tf.int32),
    )
    return tf.data.Dataset.from_generator(
        lambda: pk_batch_generator(
            images, ids, config.p_ids, config.k_shots,
            seed=split_seed, augment=config.augment and train,
        ),
        output_signature=signature,
    ).prefetch(config.prefetch_buffer)


# =============================================================================
# CALLBACKS
# =============================================================================

def create_callbacks(
    config: DeepSortTrainingConfig, run_dir: str
) -> Tuple[List[keras.callbacks.Callback], str]:
    """Early-stop / checkpoint / CSV bundle monitoring ``val_loss``."""
    # run_dir=run_dir is REQUIRED: without it the common factory derives its
    # own directory and the run splits across two result trees.
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="deepsort",
        run_dir=run_dir,
        monitor="val_loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=config.lr_schedule_type != "constant",
    )
    return callbacks, results_dir


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_rank1_map(
    model: keras.Model, val_images: List[np.ndarray], val_ids: List[int]
) -> Dict[str, float]:
    """CMC + mAP of the trunk features: shot 0 per ID queries the rest.

    Runs in training mode so fresh BatchNorm statistics apply (see the
    init-scale note in ``appearance.py``); untrained weights give chance
    scores, trained weights rank identities.

    :param model: Appearance net (features or top -- trunk output used).
    :type model: keras.Model
    :param val_images: Held-out identity images.
    :type val_images: list
    :param val_ids: Contiguous identity per image.
    :type val_ids: list
    :return: ``cmc@1/5/10`` + ``mAP``.
    :rtype: dict
    """
    images = np.stack(val_images, axis=0).astype(np.float32)
    outputs = model(images, training=True)
    features = np.asarray(outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
    ids = np.array(val_ids)
    query_idx, gallery_idx = [], []
    for identity in sorted(set(ids.tolist())):
        members = np.nonzero(ids == identity)[0]
        query_idx.append(int(members[0]))
        gallery_idx.extend(int(m) for m in members[1:])
    if not gallery_idx:
        return {"cmc@1": 0.0, "cmc@5": 0.0, "cmc@10": 0.0, "mAP": 0.0}
    return cmc_and_map(
        features[query_idx], ids[query_idx], features[gallery_idx], ids[gallery_idx]
    )


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train_deepsort(
    config: DeepSortTrainingConfig, gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Orchestrate the DeepSORT embedding training pipeline.

    Returns:
        Dict with keys ``model``, ``best_val_loss``, ``rank1``, ``map``,
        ``epochs_run``, ``total_epochs`` and ``history``.
    """
    setup_gpu(gpu_id)
    set_seeds(config.seed)

    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(
        f"Data: {config.data_source}, loss={config.loss_mode}, "
        f"PK={config.p_ids}x{config.k_shots}"
    )

    output_dir = prepare_run_dir(config)

    (train_images, train_ids, val_fit_images, val_fit_ids,
     val_images, val_ids, num_train_ids) = load_split_arrays(config)
    logger.info(
        f"Identities: {num_train_ids} train / "
        f"{len(set(val_ids))} val; "
        f"{len(train_images)} train / {len(val_fit_images)} fit-val images / "
        f"{len(val_images)} retrieval images"
    )
    train_ds = create_pk_dataset(train_images, train_ids, config, train=True)
    val_ds = create_pk_dataset(val_fit_images, val_fit_ids, config, train=False)

    use_top = config.loss_mode == "cosine-softmax"
    model = create_deepsort_embedding(
        include_top=use_top,
        num_classes=num_train_ids if use_top else None,
        dropout_rate=config.dropout_rate,
    )
    model.build((None,) + REID_IMAGE_SHAPE)
    model.summary()

    # The shared schedule builder knows decay schedules only; the reference
    # trains at a fixed LR, so "constant" passes the bare float through and
    # the plateau callback (not an external schedule) drives decay.
    if config.lr_schedule_type == "constant":
        lr: Any = config.learning_rate
    else:
        lr = learning_rate_schedule_builder({
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": config.steps_per_epoch * config.epochs,
            "warmup_steps": config.steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        })

    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    use_adamw = config.optimizer_type.lower() == "adamw"
    if use_adamw:
        opt_config["weight_decay"] = config.weight_decay
    elif config.optimizer_type.lower() == "sgd":
        opt_config["momentum"] = config.momentum
    optimizer = optimizer_builder(opt_config, lr)

    if use_top:
        # Two outputs (features, logits): the feature tap carries no loss.
        loss_fn = [
            None,
            keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ]
    else:
        loss_fn = SoftmarginTripletLoss()
    model.compile(optimizer=optimizer, loss=loss_fn)

    callbacks, _ = create_callbacks(config, str(output_dir))

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=config.val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {(time.time() - start_time) / 3600.0:.2f} hours")

    save_training_history_json(history, output_dir)

    retrieval = evaluate_rank1_map(model, val_images, val_ids)
    logger.info(
        f"Retrieval: rank-1={retrieval['cmc@1']:.4f} mAP={retrieval['mAP']:.4f}"
    )

    val_curve = history.history.get("val_loss", [float("inf")]) or [float("inf")]
    gc.collect()
    return {
        "model": model,
        "best_val_loss": float(min(val_curve)),
        "rank1": float(retrieval["cmc@1"]),
        "map": float(retrieval["mAP"]),
        "epochs_run": int(len(history.history.get("loss", []))),
        "total_epochs": int(config.epochs),
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the DeepSORT appearance embedding (synthetic or Market1501)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-source", type=str, default="synthetic",
                        choices=["synthetic", "market1501"])
    parser.add_argument("--market1501-root", type=str, default=None)
    parser.add_argument("--num-synthetic-ids", type=int, default=24)
    parser.add_argument("--shots-per-id", type=int, default=8)
    parser.add_argument("--p-ids", type=int, default=8)
    parser.add_argument("--k-shots", type=int, default=4)
    parser.add_argument("--validation-id-fraction", type=float, default=0.1)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-mode", type=str, default="cosine-softmax",
                        choices=["cosine-softmax", "triplet"])
    parser.add_argument("--dropout-rate", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--val-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["cosine_decay", "exponential_decay", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    return parser


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments (``argv`` override keeps tests hermetic)."""
    return _build_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DeepSortTrainingConfig:
    """Build the training config from a parsed namespace (flag liveness point)."""
    return DeepSortTrainingConfig(
        data_source=args.data_source,
        market1501_root=args.market1501_root,
        num_synthetic_ids=args.num_synthetic_ids,
        shots_per_id=args.shots_per_id,
        p_ids=args.p_ids,
        k_shots=args.k_shots,
        validation_id_fraction=args.validation_id_fraction,
        augment=args.augment,
        seed=args.seed,
        loss_mode=args.loss_mode,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        lr_schedule_type=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        momentum=args.momentum,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: parse argv first, then train. Returns a process exit code."""
    args = parse_arguments(argv)
    config = config_from_args(args)
    logger.info(
        f"Config: {config.epochs} epochs, PK={config.p_ids}x{config.k_shots}, "
        f"lr={config.learning_rate}, opt={config.optimizer_type}, "
        f"source={config.data_source}, loss={config.loss_mode}"
    )
    try:
        result = train_deepsort(config, gpu_id=args.gpu)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    logger.info(
        f"Best val_loss={result['best_val_loss']:.4f} "
        f"rank-1={result['rank1']:.4f} mAP={result['map']:.4f}"
    )
    if not np.isfinite(result["best_val_loss"]):
        logger.error("Training diverged (non-finite val_loss)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
