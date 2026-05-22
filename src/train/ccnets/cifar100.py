"""CCNet on CIFAR-100 (experiment).

Scales the image CCNet (``train/ccnets/mnist.py``) to 32x32x3 natural images and
100 fine-grained classes.

Expectations, stated honestly up front (``models/ccnets/PRINCIPLES_CCNETS.md``,
P1/P2): a CIFAR class label plus a modest latent ``E`` do **not** determine a
32x32 natural photograph the way ``(digit, style)`` determine an MNIST digit -- a
natural image carries far more information than its class. So the Producer
``P(X|Y,E)`` is an *underdetermined* conditional generator: expect blurry,
class-average reconstructions. The well-posed, meaningful signal here is the
Reasoner's classification accuracy. This script measures how the paradigm
degrades as the necessity-&-sufficiency condition weakens.

Run:
    MPLBACKEND=Agg .venv/bin/python -m train.ccnets.cifar100
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from train.common import setup_gpu
from dl_techniques.utils.logger import logger
from dl_techniques.models.ccnets import (
    CCNetConfig, CCNetOrchestrator, CCNetTrainer, wrap_keras_model,
)
# Generic, channel-agnostic building blocks reused from the MNIST CCNet.
from train.ccnets.mnist import ConvBlock, DenseBlock, FiLMLayer


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class ModelConfig:
    """Architecture parameters for the CIFAR-100 CCNet."""
    num_classes: int = 100
    image_channels: int = 3
    explanation_dim: int = 64

    explainer_conv_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    explainer_conv_kernels: List[int] = field(default_factory=lambda: [3, 3, 3])

    reasoner_conv_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    reasoner_conv_kernels: List[int] = field(default_factory=lambda: [3, 3, 3])
    reasoner_dense_units: List[int] = field(default_factory=lambda: [512, 256])
    reasoner_dropout: float = 0.3

    producer_label_units: int = 256
    producer_initial_spatial: int = 8        # 8 -> 16 -> 32
    producer_initial_channels: int = 256
    producer_conv_filters: List[int] = field(default_factory=lambda: [128, 64])


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
# MODULES
# =====================================================================

@keras.saving.register_keras_serializable(package="ccnets_cifar100")
class Cifar100Explainer(keras.Model):
    """P(E|X): image -> (mu, log_var) of the latent cause."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.conv_blocks = [
            ConvBlock(filters=f, kernel_size=k, use_pooling=(i < 2),
                      pool_size=2, name=f"conv_block_{i}")
            for i, (f, k) in enumerate(zip(config.explainer_conv_filters,
                                           config.explainer_conv_kernels))
        ]
        self.global_pool = keras.layers.GlobalMaxPooling2D(name="global_pool")
        self.fc_mu = keras.layers.Dense(config.explanation_dim, name="mu")
        self.fc_log_var = keras.layers.Dense(config.explanation_dim, name="log_var")

    def build(self, input_shape):
        shape = input_shape
        for block in self.conv_blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)
        pooled = self.global_pool.compute_output_shape(shape)
        self.fc_mu.build(pooled)
        self.fc_log_var.build(pooled)
        super().build(input_shape)

    def call(self, x, training=None):
        for block in self.conv_blocks:
            x = block(x, training=training)
        features = self.global_pool(x)
        return self.fc_mu(features), self.fc_log_var(features)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_cifar100")
class Cifar100Reasoner(keras.Model):
    """P(Y|X,E): (image, latent E) -> class probabilities."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.conv_blocks = [
            ConvBlock(filters=f, kernel_size=k, use_pooling=True,
                      pool_size=2, name=f"conv_block_{i}")
            for i, (f, k) in enumerate(zip(config.reasoner_conv_filters,
                                           config.reasoner_conv_kernels))
        ]
        self.global_pool = keras.layers.GlobalMaxPooling2D(name="global_pool")
        self.dense_blocks = [
            DenseBlock(units=u, dropout_rate=config.reasoner_dropout, name=f"dense_{i}")
            for i, u in enumerate(config.reasoner_dense_units)
        ]
        self.classifier = keras.layers.Dense(
            config.num_classes, activation="softmax", name="classifier")

    def build(self, input_shape):
        shape = input_shape
        for block in self.conv_blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)
        pooled = self.global_pool.compute_output_shape(shape)
        current = (pooled[0], pooled[-1] + self.config.explanation_dim)
        for block in self.dense_blocks:
            block.build(current)
            current = block.compute_output_shape(current)
        self.classifier.build(current)
        super().build(input_shape)

    def call(self, x, e, training=None):
        for block in self.conv_blocks:
            x = block(x, training=training)
        features = keras.ops.concatenate([self.global_pool(x), e], axis=-1)
        for block in self.dense_blocks:
            features = block(features, training=training)
        return self.classifier(features)

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


@keras.saving.register_keras_serializable(package="ccnets_cifar100")
class Cifar100Producer(keras.Model):
    """P(X|Y,E): (class probs, latent E) -> 32x32x3 image.

    The label enters via a bias-free Dense projection of the probability vector
    (PRINCIPLES_CCNETS.md, P4); E modulates each upsampling block via FiLM.
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        s, c = config.producer_initial_spatial, config.producer_initial_channels

        self.label_projection = keras.layers.Dense(
            config.producer_label_units, use_bias=False, name="label_projection")
        self.fc_content = keras.layers.Dense(s * s * c, activation="relu", name="fc_content")
        self.reshape = keras.layers.Reshape((s, s, c), name="reshape")

        self.gen_blocks: List[Dict[str, keras.layers.Layer]] = []
        for i, f in enumerate(config.producer_conv_filters):
            self.gen_blocks.append({
                "upsample": keras.layers.UpSampling2D(size=2, name=f"up_{i}"),
                "conv": keras.layers.Conv2D(f, 3, padding="same", name=f"conv_{i}"),
                "bn": keras.layers.BatchNormalization(name=f"bn_{i}"),
                "film": FiLMLayer(name=f"film_{i}"),
                "act": keras.layers.Activation("gelu", name=f"act_{i}"),
            })
        self.conv_out_1 = keras.layers.Conv2D(32, 1, padding="same", name="conv_out_1")
        self.conv_out_2 = keras.layers.Conv2D(
            config.image_channels, 1, padding="same", activation="sigmoid", name="conv_out_2")

    def build(self, input_shape):
        c = self.config
        self.label_projection.build((None, c.num_classes))
        proj = (None, c.producer_label_units)
        self.fc_content.build(proj)
        s, ch = c.producer_initial_spatial, c.producer_initial_channels
        self.reshape.build((None, s * s * ch))
        shape = (None, s, s, ch)
        e_shape = (None, c.explanation_dim)
        for block in self.gen_blocks:
            up = block["upsample"].compute_output_shape(shape)
            block["conv"].build(up)
            conv = block["conv"].compute_output_shape(up)
            block["bn"].build(conv)
            block["film"].build([conv, e_shape])
            block["act"].build(conv)
            shape = conv
        self.conv_out_1.build(shape)
        self.conv_out_2.build(self.conv_out_1.compute_output_shape(shape))
        super().build(input_shape)

    def call(self, y, e, training=None):
        x = self.reshape(self.fc_content(self.label_projection(y)))
        for block in self.gen_blocks:
            x = block["upsample"](x)
            x = block["conv"](x)
            x = block["bn"](x, training=training)
            x = block["film"]([x, e])
            x = block["act"](x)
        return self.conv_out_2(self.conv_out_1(x))

    def get_config(self):
        config = super().get_config()
        config["config"] = self.config.__dict__
        return config

    @classmethod
    def from_config(cls, config):
        return cls(ModelConfig(**config.pop("config")), **config)


# =====================================================================
# CONSTRUCTION + DATA
# =====================================================================

def create_cifar100_ccnet(config: ExperimentConfig) -> CCNetOrchestrator:
    mc = config.model
    explainer = Cifar100Explainer(mc)
    reasoner = Cifar100Reasoner(mc)
    producer = Cifar100Producer(mc)

    dummy_x = keras.ops.zeros((1, 32, 32, mc.image_channels))
    dummy_y = keras.ops.zeros((1, mc.num_classes))
    dummy_e = keras.ops.zeros((1, mc.explanation_dim))
    explainer(dummy_x)
    reasoner(dummy_x, dummy_e)
    producer(dummy_y, dummy_e)

    ccnet_config = CCNetConfig(
        explanation_dim=mc.explanation_dim,
        loss_fn=config.training.loss_fn,
        learning_rates=config.training.learning_rates,
        gradient_clip_norm=config.training.gradient_clip_norm,
        explainer_weights=config.training.explainer_weights,
        reasoner_weights=config.training.reasoner_weights,
        producer_weights=config.training.producer_weights,
    )
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config,
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

def run_experiment(config: ExperimentConfig) -> CCNetOrchestrator:
    setup_gpu(config.gpu)
    os.makedirs(config.results_dir, exist_ok=True)

    logger.info("Building CIFAR-100 CCNet...")
    orchestrator = create_cifar100_ccnet(config)

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


if __name__ == "__main__":
    run_experiment(ExperimentConfig())
