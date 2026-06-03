"""CIFAR-100 CCNet architecture: Explainer, Reasoner, Producer + factory + hybrid orchestrator.

This module holds the image-task CCNet networks scaled to 32x32x3 natural
images and 100 fine-grained classes, migrated verbatim from the former
``train/ccnets/cifar100.py`` and ``train/ccnets/cifar100_hybrid.py`` training
scripts.

The three networks honor the CCNet three-network contract:

* ``Cifar100Explainer(x) -> (mu, log_var)`` — variational ``P(E|X)``.
* ``Cifar100Reasoner(x, e) -> y_probs`` — ``P(Y|X,E)``, concatenates ``E`` into
  the classifier head. Also exposes ``image_features(x)``: the E-independent
  conv backbone embedding, reused as the perceptual encoder ``phi`` by the
  :class:`HybridCCNetOrchestrator`.
* ``Cifar100Producer(y, e) -> x_hat`` — ``P(X|Y,E)``. The label path is a
  bias-free ``Dense`` on the class-probability vector (NOT an ``argmax`` +
  ``Embedding``), which keeps the label projection differentiable for the
  Reasoner's soft predictions (CCNet Invariant 1).

Honest scope (``models/ccnets/PRINCIPLES_CCNETS.md``, P1/P2): a CIFAR class
label plus a modest latent ``E`` do **not** determine a 32x32 natural photograph
the way ``(digit, style)`` determine an MNIST digit, so the Producer is an
underdetermined conditional generator — expect blurry, class-average
reconstructions. The well-posed signal here is the Reasoner's classification
accuracy.

:class:`HybridCCNetOrchestrator` ports LeWM-style latent-space verification: it
adds a feature-space term (computed via the Reasoner's ``image_features``
backbone) to the Producer and Explainer errors, leaving ``reasoner_error``
untouched so ``phi`` stays anchored by the classification objective.

Shared building blocks (``FiLMLayer``, ``ConvBlock``, ``DenseBlock``) are
imported from ``dl_techniques.models.ccnets.blocks``; the CCNet framework
symbols are imported absolutely from the framework submodules so this module
carries no dependency on ``train.*``.
"""

import keras
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from dl_techniques.models.ccnets.base import CCNetConfig, CCNetModelErrors
from dl_techniques.models.ccnets.orchestrators import CCNetOrchestrator
from dl_techniques.models.ccnets.utils import wrap_keras_model
from dl_techniques.models.ccnets.blocks import FiLMLayer, ConvBlock, DenseBlock


# Weight on the latent (perceptual) verification term used by the hybrid
# orchestrator. Untuned; kept modest so the pixel loss still grounds
# generation. The run logs both magnitudes.
LATENT_WEIGHT = 0.25


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class ModelConfig:
    """Architecture parameters for the CIFAR-100 CCNet.

    Only model-architecture fields live here; training-side fields
    (learning rates, loss weights, etc.) are supplied to
    ``create_cifar100_ccnet`` explicitly so the model package never imports a
    training dataclass.
    """
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

    def image_features(self, x, training=None):
        """E-independent image embedding (conv backbone + global pool).

        Reused as the perceptual encoder phi by the hybrid latent-verification
        orchestrator (:class:`HybridCCNetOrchestrator`)."""
        for block in self.conv_blocks:
            x = block(x, training=training)
        return self.global_pool(x)

    def call(self, x, e, training=None):
        features = keras.ops.concatenate(
            [self.image_features(x, training=training), e], axis=-1)
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

        # Differentiable label projection, never an Embedding: the Producer
        # receives soft class probabilities and a bias-free Dense keeps the
        # gradient path to the Reasoner intact (CCNet Invariant 1).
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
# HYBRID ORCHESTRATOR
# =====================================================================

class HybridCCNetOrchestrator(CCNetOrchestrator):
    """CCNet orchestrator with an added latent-space (perceptual) verification term.

    `perceptual_model` must expose `image_features(x, training)` — used as the
    encoder phi. The latent term enters only the Producer and Explainer errors;
    the three-tape design then guarantees phi (the Reasoner backbone) is not
    updated by it.
    """

    def __init__(self, *args, perceptual_model, latent_weight=LATENT_WEIGHT, **kwargs):
        super().__init__(*args, **kwargs)
        self.perceptual_model = perceptual_model
        self.latent_weight = latent_weight

    def _latent_losses(self, tensors):
        phi_x = self.perceptual_model.image_features(tensors["x_input"], training=False)
        phi_gen = self.perceptual_model.image_features(tensors["x_generated"], training=False)
        phi_rec = self.perceptual_model.image_features(tensors["x_reconstructed"], training=False)
        gen_latent = keras.ops.mean(keras.ops.square(phi_gen - phi_x))
        rec_latent = keras.ops.mean(keras.ops.square(phi_rec - phi_x))
        return gen_latent, rec_latent

    def compute_model_errors(self, losses, tensors) -> CCNetModelErrors:
        errors = super().compute_model_errors(losses, tensors)   # pixel-space errors
        gen_latent, rec_latent = self._latent_losses(tensors)
        w = self.latent_weight
        return CCNetModelErrors(
            explainer_error=errors.explainer_error + w * gen_latent,
            reasoner_error=errors.reasoner_error,                      # unchanged: phi stays anchored
            producer_error=errors.producer_error + w * (gen_latent + rec_latent),
        )


# =====================================================================
# FACTORY
# =====================================================================

def create_cifar100_ccnet(
    model_config: ModelConfig,
    *,
    loss_fn: str = 'l1',
    loss_fn_params: Optional[Dict[str, Any]] = None,
    learning_rates: Optional[Dict[str, float]] = None,
    gradient_clip_norm: Optional[float] = 1.0,
    use_mixed_precision: bool = False,
    explainer_weights: Optional[Dict[str, float]] = None,
    reasoner_weights: Optional[Dict[str, float]] = None,
    producer_weights: Optional[Dict[str, float]] = None,
    hybrid: bool = False,
    latent_weight: float = LATENT_WEIGHT,
) -> CCNetOrchestrator:
    """Create and initialize a CCNet for CIFAR-100 classification + generation.

    Builds the three sub-networks on one-hot dummy inputs, wraps them, and
    returns a configured :class:`CCNetOrchestrator` (or a
    :class:`HybridCCNetOrchestrator` when ``hybrid=True``).

    Args:
        model_config: CIFAR-100 architecture configuration.
        loss_fn: Base loss-function name for the orchestrator (``'l1'`` is
            sharper than ``'l2'`` for images).
        loss_fn_params: Extra parameters for the loss function.
        learning_rates: Per-network learning rates
            (``explainer``/``reasoner``/``producer``).
        gradient_clip_norm: Global gradient-norm clip value (``None`` disables).
        use_mixed_precision: Whether to enable mixed-precision training.
        explainer_weights: Explainer loss-term weights.
        reasoner_weights: Reasoner loss-term weights.
        producer_weights: Producer loss-term weights.
        hybrid: When ``True``, return a :class:`HybridCCNetOrchestrator` that
            adds a latent-space (perceptual) verification term using the
            Reasoner's ``image_features`` backbone as phi.
        latent_weight: Weight on the perceptual verification term (only used
            when ``hybrid=True``).

    Returns:
        A configured :class:`CCNetOrchestrator` (or :class:`HybridCCNetOrchestrator`).
    """
    loss_fn_params = loss_fn_params if loss_fn_params is not None else {}
    learning_rates = learning_rates if learning_rates is not None else {
        'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4
    }
    explainer_weights = explainer_weights if explainer_weights is not None else {
        'inference': 1.0, 'generation': 1.0, 'kl_divergence': 1e-3
    }
    reasoner_weights = reasoner_weights if reasoner_weights is not None else {
        'inference': 1.0, 'reconstruction': 0.1
    }
    producer_weights = producer_weights if producer_weights is not None else {
        'generation': 1.0, 'reconstruction': 1.0
    }

    mc = model_config
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
        loss_fn=loss_fn,
        loss_fn_params=loss_fn_params,
        learning_rates=learning_rates,
        gradient_clip_norm=gradient_clip_norm,
        use_mixed_precision=use_mixed_precision,
        explainer_weights=explainer_weights,
        reasoner_weights=reasoner_weights,
        producer_weights=producer_weights,
    )

    if hybrid:
        return HybridCCNetOrchestrator(
            explainer=wrap_keras_model(explainer),
            reasoner=wrap_keras_model(reasoner),
            producer=wrap_keras_model(producer),
            config=ccnet_config,
            perceptual_model=reasoner,           # phi = the Reasoner's image features
            latent_weight=latent_weight,
        )

    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config,
    )
