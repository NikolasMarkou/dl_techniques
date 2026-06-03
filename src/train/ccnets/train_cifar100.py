"""CCNet on CIFAR-100 (training script).

Scales the image CCNet (``train/ccnets/train_mnist.py``) to 32x32x3 natural
images and 100 fine-grained classes.

Expectations, stated honestly up front (``models/ccnets/PRINCIPLES_CCNETS.md``,
P1/P2): a CIFAR class label plus a modest latent ``E`` do **not** determine a
32x32 natural photograph the way ``(digit, style)`` determine an MNIST digit -- a
natural image carries far more information than its class. So the Producer
``P(X|Y,E)`` is an *underdetermined* conditional generator: expect blurry,
class-average reconstructions. The well-posed, meaningful signal here is the
Reasoner's classification accuracy. This script measures how the paradigm
degrades as the necessity-&-sufficiency condition weakens.

The architecture (``Cifar100Explainer/Reasoner/Producer`` + the
``create_cifar100_ccnet`` factory) now lives in
``dl_techniques.models.ccnets``; this script keeps the training-side layer:
config dataclasses, data prep, the ``CCNetTrainer`` driver, evaluation, and
plotting.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.train_cifar100 --gpu 0 --epochs 40
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.common import setup_gpu, set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import CCNetTrainer
from dl_techniques.models.ccnets.architectures.cifar100 import (
    ModelConfig,
    create_cifar100_ccnet,
)


# =====================================================================
# CONFIG (training-side; architecture ModelConfig is imported)
# =====================================================================

@dataclass
class TrainingConfig:
    epochs: int = 40
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4})
    loss_fn: str = 'l1'                       # L1 is sharper than L2 for images
    gradient_clip_norm: Optional[float] = 1.0
    kl_annealing_epochs: Optional[int] = 10
    explainer_weights: Dict[str, float] = field(
        default_factory=lambda: {'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3})
    reasoner_weights: Dict[str, float] = field(
        default_factory=lambda: {'inference': 1.0, 'reconstruction': 0.1})
    producer_weights: Dict[str, float] = field(
        default_factory=lambda: {'generation': 1.0, 'reconstruction': 1.0})


@dataclass
class DataConfig:
    batch_size: int = 128
    augment: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "results/ccnets_cifar100"
    gpu: Optional[int] = None


# =====================================================================
# CONSTRUCTION + DATA
# =====================================================================

def build_cifar100_ccnet(config: ExperimentConfig):
    """Build the CIFAR-100 orchestrator from the experiment config."""
    return create_cifar100_ccnet(
        config.model,
        loss_fn=config.training.loss_fn,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )


def prepare_cifar100(config: ExperimentConfig):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_oh = keras.utils.to_categorical(y_train, 100)
    y_test_oh = keras.utils.to_categorical(y_test, 100)

    aug = None
    if config.data.augment:
        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        ])

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_oh)).shuffle(50000)
    if aug is not None:
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(config.data.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, x_test, y_test.squeeze()


# =====================================================================
# EVAL + VIZ
# =====================================================================

def evaluate(orchestrator, x_test, y_test, results_dir) -> str:
    """Reasoner top-1 / top-5 accuracy + a reconstruction grid."""
    top1 = top5 = total = 0
    for start in range(0, len(x_test), 256):
        xb = x_test[start:start + 256]
        yb = keras.utils.to_categorical(y_test[start:start + 256], 100).astype("float32")
        tensors = orchestrator.forward_pass(
            tf.convert_to_tensor(xb), tf.convert_to_tensor(yb), training=False)
        probs = keras.ops.convert_to_numpy(tensors["y_inferred"])
        labels = y_test[start:start + 256]
        top1 += int(np.sum(np.argmax(probs, axis=-1) == labels))
        top5 += int(np.sum([labels[i] in np.argsort(probs[i])[-5:]
                            for i in range(len(labels))]))
        total += len(xb)
    report = (f"Reasoner accuracy (CIFAR-100 test): top-1 = {top1/total:.4f}, "
              f"top-5 = {top5/total:.4f}")
    logger.info(report)

    # Reconstruction grid: original / reconstructed / generated for 6 images.
    n = 6
    xb = x_test[:n]
    yb = keras.utils.to_categorical(y_test[:n], 100).astype("float32")
    t = orchestrator.forward_pass(tf.convert_to_tensor(xb), tf.convert_to_tensor(yb),
                                  training=False)
    xr = np.clip(keras.ops.convert_to_numpy(t["x_reconstructed"]), 0, 1)
    xg = np.clip(keras.ops.convert_to_numpy(t["x_generated"]), 0, 1)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        for row, (img, name) in enumerate(
                [(xb[i], "original"), (xr[i], "reconstructed"), (xg[i], "generated")]):
            axes[row, i].imshow(img)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(name, rotation=90, size="large")
                axes[row, i].axis("on")
                axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
    fig.suptitle("CCNet CIFAR-100 — original / reconstructed / generated",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "reconstruction_grid.png"), dpi=120)
    plt.close(fig)
    return report


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (title, keys) in zip(axes, [
        ("Fundamental losses", ["generation_loss", "reconstruction_loss", "inference_loss"]),
        ("Module errors", ["explainer_error", "reasoner_error", "producer_error"]),
        ("Batch accuracy", ["batch_accuracy"]),
    ]):
        for key in keys:
            if key in history and history[key]:
                ax.plot(history[key], label=key, linewidth=2)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("CCNet CIFAR-100 training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# =====================================================================
# RUN
# =====================================================================

def run_experiment(config: ExperimentConfig):
    os.makedirs(config.results_dir, exist_ok=True)

    logger.info("Building CIFAR-100 CCNet...")
    orchestrator = build_cifar100_ccnet(config)

    logger.info("Loading CIFAR-100...")
    train_ds, x_test, y_test = prepare_cifar100(config)

    logger.info(f"Training for {config.training.epochs} epochs...")
    trainer = CCNetTrainer(orchestrator, kl_annealing_epochs=config.training.kl_annealing_epochs)
    trainer.train(train_ds, config.training.epochs)

    orchestrator.save_models(os.path.join(config.results_dir, "cifar100_ccnet"))
    plot_history(trainer.history, os.path.join(config.results_dir, "training_history.png"))
    report = evaluate(orchestrator, x_test, y_test, config.results_dir)
    with open(os.path.join(config.results_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    logger.info(f"Artifacts saved to {config.results_dir}")
    return orchestrator


# =====================================================================
# ENTRY POINT
# =====================================================================

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Map parsed CLI args onto the experiment config dataclasses."""
    config = ExperimentConfig()
    config.gpu = args.gpu
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    if args.smoke:
        config.model.explanation_dim = 16
        config.training.epochs = 1
        config.training.kl_annealing_epochs = 1
        config.data.batch_size = 64
        config.data.augment = False
        config.results_dir = "results/ccnets_cifar100_smoke"

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CCNet on CIFAR-100.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index.")
    parser.add_argument('--epochs', type=int, default=None, help="Training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--smoke', action='store_true',
                        help="Tiny CI-safe run (1 epoch, small dims).")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    set_seeds(args.seed)

    config = build_config(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
