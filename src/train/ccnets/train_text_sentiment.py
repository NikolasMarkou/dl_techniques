"""CCNet for sentiment-conditioned text (training script).

Extends the Causal Cooperative Networks paradigm (``dl_techniques/models/ccnets``)
from images to discrete token sequences, on the IMDB sentiment corpus:

    X = a movie review (token sequence)   -- the observation (effect)
    Y = sentiment (negative / positive)   -- the explicit cause (label)
    E = latent style / content            -- the latent cause

What differs from the image task (``train/ccnets/train_mnist.py``):

* **The Producer ``P(X|Y,E)``.** Two variants, selected by ``ModelConfig.producer_type``:
    - ``'autoregressive'`` (default): a causal Transformer decoder, teacher-forced on
      ``x_input``. Token ``i`` is predicted from tokens ``< i`` plus a conditioning
      prefix built from ``(Y, E)``. Used via ``ARTextCCNetOrchestrator``.
    - ``'nonautoregressive'``: emits the whole token-logit sequence in one shot from
      ``(Y, E)``. Simpler, but a small latent cannot carry a long review.
  Both keep the differentiable label path (PRINCIPLES_CCNETS.md, P4).
* **Token-space losses.** ``TextCCNetOrchestrator`` overrides ``compute_losses``
  with masked cross-entropy (generation/reconstruction) and masked KL (inference).

The architecture (Sentiment networks + the token-space orchestrators +
``create_text_ccnet``) now lives in ``dl_techniques.models.ccnets``; this script
keeps the training-side layer: config dataclasses, IMDB data prep, decoding,
evaluation, and plotting.

PROTOTYPE SCOPE: a movie review is not *determined* by its sentiment, so the CCNet
necessity-&-sufficiency condition (PRINCIPLES_CCNETS.md, P1/P2) only partly holds.
Sentiment classification (the Reasoner) is the strong, well-posed signal. The
three-Producer-variant findings are in ``train/ccnets/README.md``.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.train_text_sentiment --gpu 0 --epochs 10
"""

import os
import argparse
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.common import setup_gpu, set_seeds
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import CCNetTrainer
from dl_techniques.models.ccnets.architectures.text import (
    ModelConfig,
    TextCCNetOrchestrator,
    create_text_ccnet,
)


# =====================================================================
# CONFIGURATION (training-side; architecture ModelConfig is imported)
# =====================================================================

@dataclass
class TrainingConfig:
    """Training parameters."""
    epochs: int = 10
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4}
    )
    gradient_clip_norm: Optional[float] = 1.0
    kl_annealing_epochs: Optional[int] = 4
    # Reasoner: the cross-entropy anchor leads; reconstruction is a gentle
    # cooperative nudge (token CE is ~10x larger in magnitude). See P5.
    explainer_weights: Dict[str, float] = field(
        default_factory=lambda: {'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3}
    )
    reasoner_weights: Dict[str, float] = field(
        default_factory=lambda: {'inference': 1.0, 'reconstruction': 0.1}
    )
    producer_weights: Dict[str, float] = field(
        default_factory=lambda: {'generation': 1.0, 'reconstruction': 1.0}
    )


@dataclass
class DataConfig:
    """Data parameters."""
    batch_size: int = 32
    shuffle_buffer: int = 25000


@dataclass
class ExperimentConfig:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "results/ccnets_text_sentiment"
    gpu: Optional[int] = None


# =====================================================================
# CONSTRUCTION
# =====================================================================

def build_text_ccnet(config: ExperimentConfig) -> TextCCNetOrchestrator:
    """Build the sentiment orchestrator from the experiment config."""
    return create_text_ccnet(
        config.model,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )


def prepare_imdb_data(
    config: ExperimentConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """Load and pad the IMDB sentiment dataset."""
    mc, dc = config.model, config.data
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=mc.vocab_size
    )
    x_train = keras.utils.pad_sequences(
        x_train, maxlen=mc.max_len, padding="post", truncating="post"
    )
    x_test = keras.utils.pad_sequences(
        x_test, maxlen=mc.max_len, padding="post", truncating="post"
    )
    y_train_oh = keras.utils.to_categorical(y_train, mc.num_classes)
    y_test_oh = keras.utils.to_categorical(y_test, mc.num_classes)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(dc.shuffle_buffer)
        .batch(dc.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
        .batch(dc.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, x_test, y_test


# =====================================================================
# DECODING / EVALUATION
# =====================================================================

def build_decoder(vocab_size: int):
    """Return a function mapping an id sequence back to text."""
    word_index = keras.datasets.imdb.get_word_index()
    # IMDB reserves 0=pad, 1=start, 2=oov; real word ids are offset by 3.
    reverse = {idx + 3: word for word, idx in word_index.items()}
    reverse.update({0: "", 1: "", 2: "<oov>"})

    def decode(ids: np.ndarray) -> str:
        words = [reverse.get(int(i), "<oov>") for i in ids if int(i) != 0]
        return " ".join(w for w in words if w)

    return decode


def evaluate_and_report(
    orchestrator: TextCCNetOrchestrator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
) -> str:
    """Report Reasoner accuracy and print reconstruction / counterfactual samples."""
    mc = config.model
    decode = build_decoder(mc.vocab_size)
    lines: List[str] = []

    # --- Reasoner sentiment accuracy over the test set ---
    correct, total = 0, 0
    for start in range(0, len(x_test), 256):
        xb = x_test[start:start + 256]
        yb = keras.utils.to_categorical(y_test[start:start + 256], mc.num_classes)
        tensors = orchestrator.forward_pass(
            tf.convert_to_tensor(xb), tf.convert_to_tensor(yb.astype("float32")),
            training=False,
        )
        preds = np.argmax(keras.ops.convert_to_numpy(tensors["y_inferred"]), axis=-1)
        correct += int(np.sum(preds == y_test[start:start + 256]))
        total += len(xb)
    accuracy = correct / total
    lines.append(f"Reasoner sentiment accuracy (test): {accuracy:.4f}")
    lines.append("=" * 70)

    # --- Reconstruction + sentiment counterfactual on a few samples ---
    label_name = {0: "negative", 1: "positive"}
    for idx in (0, 1, 12, 25):
        x = tf.convert_to_tensor(x_test[idx:idx + 1])
        true_label = int(y_test[idx])
        y_true = keras.utils.to_categorical([true_label], mc.num_classes).astype("float32")

        tensors = orchestrator.forward_pass(x, tf.convert_to_tensor(y_true), training=False)
        recon_ids = np.argmax(keras.ops.convert_to_numpy(tensors["x_reconstructed"][0]), -1)

        # Counterfactual: regenerate with the opposite sentiment, same style E.
        flipped = keras.utils.to_categorical(
            [1 - true_label], mc.num_classes
        ).astype("float32")
        cf = keras.ops.convert_to_numpy(
            orchestrator.counterfactual_generation(x, tf.convert_to_tensor(flipped))
        )
        # The autoregressive orchestrator returns decoded token ids [B, T];
        # the non-autoregressive one returns logits [B, T, vocab].
        cf_ids = np.argmax(cf[0], axis=-1) if cf.ndim == 3 else cf[0]

        lines.append(f"[sample {idx}] true sentiment: {label_name[true_label]}")
        lines.append(f"  original     : {decode(x_test[idx])[:300]}")
        lines.append(f"  reconstructed: {decode(recon_ids)[:300]}")
        lines.append(f"  counterfactual ({label_name[1 - true_label]}): {decode(cf_ids)[:300]}")
        lines.append("-" * 70)

    report = "\n".join(lines)
    logger.info("\n" + report)
    return report


def plot_history(history: Dict[str, List[float]], out_path: str) -> None:
    """Save a grid of the CCNet training metrics."""
    panels = [
        ("Fundamental losses", ["generation_loss", "reconstruction_loss", "inference_loss"]),
        ("Module errors", ["explainer_error", "reasoner_error", "producer_error"]),
        ("Gradient norms", ["explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm"]),
        ("Sentiment accuracy", ["batch_accuracy"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, keys) in zip(axes.flatten(), panels):
        for key in keys:
            if key in history and history[key]:
                ax.plot(history[key], label=key, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("CCNet — sentiment-conditioned text", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_counterfactual_matrix(
    orchestrator: TextCCNetOrchestrator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    out_path: str,
) -> None:
    """Render a sentiment counterfactual matrix — the text analogue of the MNIST
    digit matrix.

    Each row is a source review; each cell re-decodes that review's latent style
    ``E`` under a different target sentiment ``Y``. The cell whose target matches
    the review's true sentiment is outlined in green.
    """
    mc = config.model
    decode = build_decoder(mc.vocab_size)
    label_name = {0: "negative", 1: "positive"}
    samples_per_class = 3

    # Balanced set of source reviews.
    idx: List[int] = []
    for label in range(mc.num_classes):
        idx.extend(np.where(y_test == label)[0][:samples_per_class].tolist())
    sources = x_test[idx]
    src_labels = y_test[idx]

    # One batched counterfactual decode per target sentiment.
    cf_by_target: Dict[int, np.ndarray] = {}
    for target in range(mc.num_classes):
        y_target = keras.utils.to_categorical(
            [target] * len(idx), mc.num_classes).astype("float32")
        out = keras.ops.convert_to_numpy(
            orchestrator.counterfactual_generation(
                tf.convert_to_tensor(sources), tf.convert_to_tensor(y_target))
        )
        # AR orchestrator returns token ids [N,T]; NAR returns logits [N,T,V].
        cf_by_target[target] = np.argmax(out, axis=-1) if out.ndim == 3 else out

    rows, cols = len(idx), mc.num_classes + 1
    headers = ["Original"] + [f"-> {label_name[t]}" for t in range(mc.num_classes)]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.3))
    fig.suptitle(
        'Sentiment Counterfactual Matrix\n'
        '"the same review, re-decoded under each sentiment"',
        fontsize=14, fontweight="bold",
    )

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ids = sources[i] if j == 0 else cf_by_target[j - 1][i]
            ax.text(0.03, 0.97, textwrap.fill(decode(ids)[:240], width=38),
                    va="top", ha="left", fontsize=7, family="monospace")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[j], fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"src: {label_name[src_labels[i]]}", fontsize=8)
            # Outline the cell whose target sentiment is the review's true one.
            if j > 0 and (j - 1) == src_labels[i]:
                for spine in ax.spines.values():
                    spine.set_edgecolor("green")
                    spine.set_linewidth(2.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =====================================================================
# RUN
# =====================================================================

def run_experiment(config: ExperimentConfig) -> TextCCNetOrchestrator:
    """Build, train, and evaluate the sentiment CCNet."""
    os.makedirs(config.results_dir, exist_ok=True)

    logger.info("Building sentiment CCNet...")
    orchestrator = build_text_ccnet(config)

    logger.info("Loading IMDB data...")
    train_ds, val_ds, x_test, y_test = prepare_imdb_data(config)

    logger.info(f"Training for {config.training.epochs} epochs...")
    trainer = CCNetTrainer(
        orchestrator, kl_annealing_epochs=config.training.kl_annealing_epochs
    )
    trainer.train(train_ds, config.training.epochs, validation_dataset=val_ds)

    # Save the trained modules immediately, before the (fallible) plotting and
    # evaluation steps, so a reporting bug can never cost the trained model.
    orchestrator.save_models(os.path.join(config.results_dir, "sentiment_ccnet"))

    plot_history(trainer.history, os.path.join(config.results_dir, "training_history.png"))
    plot_counterfactual_matrix(
        orchestrator, x_test, y_test, config,
        os.path.join(config.results_dir, "counterfactual_matrix.png"))

    report = evaluate_and_report(orchestrator, x_test, y_test, config)
    with open(os.path.join(config.results_dir, "samples.txt"), "w") as fh:
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
        config.model.vocab_size = 1000
        config.model.max_len = 24
        config.model.explanation_dim = 16
        config.model.embed_dim = 32
        config.model.encoder_hidden = 32
        config.model.producer_d_model = 32
        config.model.producer_layers = 1
        config.model.producer_ffn_dim = 64
        config.training.epochs = 1
        config.training.kl_annealing_epochs = 1
        config.data.batch_size = 32
        config.data.shuffle_buffer = 1000
        config.results_dir = "results/ccnets_text_sentiment_smoke"

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CCNet on IMDB sentiment.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device index.")
    parser.add_argument('--epochs', type=int, default=None, help="Training epochs.")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--smoke', action='store_true',
                        help="Tiny CI-safe run (1 epoch, small dims/vocab).")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    set_seeds(args.seed)

    config = build_config(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
