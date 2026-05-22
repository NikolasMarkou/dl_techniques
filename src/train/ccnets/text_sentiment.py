"""CCNet for sentiment-conditioned text (prototype).

Extends the Causal Cooperative Networks paradigm (``dl_techniques/models/ccnets``)
from images to discrete token sequences, on the IMDB sentiment corpus:

    X = a movie review (token sequence)   -- the observation (effect)
    Y = sentiment (negative / positive)   -- the explicit cause (label)
    E = latent style / content            -- the latent cause

What differs from the image task (``train/ccnets/mnist.py``):

* **The Producer ``P(X|Y,E)`` is a non-autoregressive text decoder.** It emits the
  full sequence of token logits ``[B, T, vocab]`` in one shot from ``(Y, E)``. This
  keeps the differentiable label path (PRINCIPLES_CCNETS.md, P4) trivially intact --
  no argmax, no sampled-token feedback -- at the cost of generation sharpness.
* **Token-space losses.** ``TextCCNetOrchestrator`` overrides ``compute_losses``:
    - generation / reconstruction -> masked sparse categorical cross-entropy
      between Producer logits and the input tokens;
    - inference -> masked KL divergence between the two Producer output
      distributions.
  The pixel-norm losses (L1/L2/Huber) in ``losses.py`` do not apply to discrete X.

Everything else -- the variational Explainer, the cross-entropy-anchored Reasoner
error, the per-module gradient tapes, the live ``kl_weight`` -- is reused unchanged
from the base ``CCNetOrchestrator``. Only ``compute_losses`` is task-specific,
which is Principle P11 (model-agnostic, contract-based) in action.

PROTOTYPE SCOPE: this demonstrates that the mechanism runs end-to-end on text --
cooperative gradient flow, token-space losses, sentiment counterfactuals. A small
latent + a non-autoregressive decoder will not reconstruct long reviews faithfully;
reconstruction is a structural demo, sentiment classification is the strong signal.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.text_sentiment
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import (
    CCNetConfig,
    CCNetOrchestrator,
    CCNetTrainer,
    wrap_keras_model,
)
from dl_techniques.models.ccnets.base import CCNetLosses


# =====================================================================
# CONFIGURATION
# =====================================================================

@dataclass
class ModelConfig:
    """Architecture parameters for the three CCNet modules."""
    vocab_size: int = 5000
    max_len: int = 80
    num_classes: int = 2          # negative / positive
    explanation_dim: int = 32

    embed_dim: int = 128
    encoder_hidden: int = 128     # GRU width for Explainer / Reasoner
    reasoner_dense_units: int = 64
    reasoner_dropout: float = 0.3

    producer_d_model: int = 128
    producer_layers: int = 2
    producer_heads: int = 4
    producer_ffn_dim: int = 256


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
# TOKEN-SPACE ORCHESTRATOR
# =====================================================================

class TextCCNetOrchestrator(CCNetOrchestrator):
    """CCNet orchestrator with token-space losses for discrete observations.

    Only ``compute_losses`` is overridden. ``forward_pass``, ``compute_model_errors``,
    ``train_step`` and the gradient-tape routing are inherited unchanged -- they
    operate on the three scalar losses regardless of how those scalars were formed.
    """

    def __init__(self, *args, pad_token: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token = pad_token

    def compute_losses(self, tensors: Dict[str, tf.Tensor]) -> CCNetLosses:
        x_input = tensors['x_input']               # [B, T] int token ids
        gen_logits = tensors['x_generated']        # [B, T, V]
        rec_logits = tensors['x_reconstructed']    # [B, T, V]

        # Mask out padding positions so they do not dilute the losses.
        mask = keras.ops.cast(
            keras.ops.not_equal(x_input, self.pad_token), dtype='float32'
        )                                          # [B, T]
        denom = keras.ops.sum(mask) + 1e-8

        # Generation / reconstruction: per-token cross-entropy in nats.
        ce_gen = keras.losses.sparse_categorical_crossentropy(
            x_input, gen_logits, from_logits=True
        )                                          # [B, T]
        ce_rec = keras.losses.sparse_categorical_crossentropy(
            x_input, rec_logits, from_logits=True
        )
        generation_loss = keras.ops.sum(ce_gen * mask) / denom
        reconstruction_loss = keras.ops.sum(ce_rec * mask) / denom

        # Inference loss: KL( P(x_generated) || P(x_reconstructed) ), per token.
        # Both come from the same Producer, so the gap reflects Y_truth vs Y_inferred.
        log_p = keras.ops.log_softmax(gen_logits, axis=-1)
        log_q = keras.ops.log_softmax(rec_logits, axis=-1)
        p = keras.ops.exp(log_p)
        kl = keras.ops.sum(p * (log_p - log_q), axis=-1)   # [B, T]
        inference_loss = keras.ops.sum(kl * mask) / denom

        return CCNetLosses(
            generation_loss=generation_loss,
            reconstruction_loss=reconstruction_loss,
            inference_loss=inference_loss,
        )


# =====================================================================
# MODULES — Explainer / Reasoner / Producer
# =====================================================================

@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentExplainer(keras.Model):
    """Models P(E|X): review tokens -> (mu, log_var) of the latent cause."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="embedding"
        )
        self.encoder = keras.layers.Bidirectional(
            keras.layers.GRU(config.encoder_hidden), name="encoder"
        )
        self.fc_mu = keras.layers.Dense(config.explanation_dim, name="mu")
        self.fc_log_var = keras.layers.Dense(config.explanation_dim, name="log_var")

    def call(self, x, training=None):
        h = self.encoder(self.embedding(x), training=training)
        return self.fc_mu(h), self.fc_log_var(h)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentReasoner(keras.Model):
    """Models P(Y|X,E): (review tokens, latent E) -> sentiment probabilities."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding = keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="embedding"
        )
        self.encoder = keras.layers.Bidirectional(
            keras.layers.GRU(config.encoder_hidden), name="encoder"
        )
        self.dense = keras.layers.Dense(
            config.reasoner_dense_units, activation="relu", name="dense"
        )
        self.dropout = keras.layers.Dropout(config.reasoner_dropout, name="dropout")
        self.classifier = keras.layers.Dense(
            config.num_classes, activation="softmax", name="classifier"
        )

    def call(self, x, e, training=None):
        h = self.encoder(self.embedding(x), training=training)
        h = keras.ops.concatenate([h, e], axis=-1)
        h = self.dropout(self.dense(h), training=training)
        return self.classifier(h)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_text")
class SentimentProducer(keras.Model):
    """Models P(X|Y,E): (sentiment, latent E) -> token logits [B, T, vocab].

    Non-autoregressive: the conditioning vector is broadcast across all T
    positions, differentiated by positional embeddings, then refined by
    self-attention blocks. The label enters via a bias-free Dense projection
    (PRINCIPLES_CCNETS.md, P4) so gradient flows back to the Reasoner.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        d = config.producer_d_model

        # P4: differentiable label path -- Dense on the probability vector,
        # never argmax + Embedding.
        self.label_projection = keras.layers.Dense(d, use_bias=False, name="label_projection")
        self.style_projection = keras.layers.Dense(d, name="style_projection")
        self.merge = keras.layers.Dense(d, activation="gelu", name="merge")
        self.position_embedding = keras.layers.Embedding(
            config.max_len, d, name="position_embedding"
        )

        self.blocks: List[Dict[str, keras.layers.Layer]] = []
        for i in range(config.producer_layers):
            self.blocks.append({
                "attn": keras.layers.MultiHeadAttention(
                    num_heads=config.producer_heads, key_dim=d // config.producer_heads,
                    name=f"attn_{i}"),
                "norm1": keras.layers.LayerNormalization(name=f"norm1_{i}"),
                "ffn1": keras.layers.Dense(config.producer_ffn_dim, activation="gelu",
                                           name=f"ffn1_{i}"),
                "ffn2": keras.layers.Dense(d, name=f"ffn2_{i}"),
                "norm2": keras.layers.LayerNormalization(name=f"norm2_{i}"),
            })

        self.to_logits = keras.layers.Dense(config.vocab_size, name="to_logits")

    def call(self, y, e, training=None):
        content = self.label_projection(y)        # [B, d]
        style = self.style_projection(e)          # [B, d]
        seed = self.merge(content + style)        # [B, d]

        # Broadcast the conditioning vector to every sequence position.
        x = keras.ops.expand_dims(seed, axis=1)               # [B, 1, d]
        x = keras.ops.repeat(x, self.config.max_len, axis=1)  # [B, T, d]
        positions = keras.ops.arange(self.config.max_len)
        x = x + self.position_embedding(positions)            # [B, T, d]

        for block in self.blocks:
            attn = block["attn"](x, x, training=training)     # non-causal self-attention
            x = block["norm1"](x + attn)
            ffn = block["ffn2"](block["ffn1"](x))
            x = block["norm2"](x + ffn)

        return self.to_logits(x)                              # [B, T, vocab]

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


# =====================================================================
# CONSTRUCTION
# =====================================================================

def create_sentiment_ccnet(config: ExperimentConfig) -> TextCCNetOrchestrator:
    """Build the three modules, wrap them, and assemble the orchestrator."""
    mc = config.model
    explainer = SentimentExplainer(mc)
    reasoner = SentimentReasoner(mc)
    producer = SentimentProducer(mc)

    # Build via dummy forward passes (the label dummy is a probability vector).
    dummy_x = keras.ops.zeros((1, mc.max_len), dtype="int32")
    dummy_y = keras.ops.zeros((1, mc.num_classes))
    dummy_e = keras.ops.zeros((1, mc.explanation_dim))
    explainer(dummy_x)
    reasoner(dummy_x, dummy_e)
    producer(dummy_y, dummy_e)

    ccnet_config = CCNetConfig(
        explanation_dim=mc.explanation_dim,
        loss_fn='l2',  # unused -- TextCCNetOrchestrator overrides compute_losses
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )
    return TextCCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config,
        pad_token=0,
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
        cf_logits = orchestrator.counterfactual_generation(x, tf.convert_to_tensor(flipped))
        cf_ids = np.argmax(keras.ops.convert_to_numpy(cf_logits[0]), -1)

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


# =====================================================================
# RUN
# =====================================================================

def run_experiment(config: ExperimentConfig) -> TextCCNetOrchestrator:
    """Build, train, and evaluate the sentiment CCNet."""
    setup_gpu(config.gpu)
    os.makedirs(config.results_dir, exist_ok=True)

    logger.info("Building sentiment CCNet...")
    orchestrator = create_sentiment_ccnet(config)

    logger.info("Loading IMDB data...")
    train_ds, val_ds, x_test, y_test = prepare_imdb_data(config)

    logger.info(f"Training for {config.training.epochs} epochs...")
    trainer = CCNetTrainer(
        orchestrator, kl_annealing_epochs=config.training.kl_annealing_epochs
    )
    trainer.train(train_ds, config.training.epochs, validation_dataset=val_ds)

    plot_history(trainer.history, os.path.join(config.results_dir, "training_history.png"))

    report = evaluate_and_report(orchestrator, x_test, y_test, config)
    with open(os.path.join(config.results_dir, "samples.txt"), "w") as fh:
        fh.write(report + "\n")

    orchestrator.save_models(os.path.join(config.results_dir, "sentiment_ccnet"))
    logger.info(f"Artifacts saved to {config.results_dir}")
    return orchestrator


if __name__ == "__main__":
    run_experiment(ExperimentConfig())
