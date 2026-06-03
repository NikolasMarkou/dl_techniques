"""MNIST CCNet architecture: Explainer, Reasoner, Producer + factory.

This module holds the reference image-task CCNet networks for MNIST digit
classification and counterfactual generation, migrated verbatim from the
former ``train/ccnets/mnist.py`` training script.

The three networks honor the CCNet three-network contract:

* ``MNISTExplainer(x) -> (mu, log_var)`` — variational ``P(E|X)``.
* ``MNISTReasoner(x, e) -> y_probs`` — ``P(Y|X,E)``, concatenates ``E`` into
  the classifier head.
* ``MNISTProducer(y, e) -> x_hat`` — ``P(X|Y,E)``. The label path is a bias-free
  ``Dense`` on the class-probability vector (NOT an ``argmax`` + ``Embedding``),
  which keeps the label projection differentiable for the Reasoner's soft
  predictions (CCNet Invariant 1).

Shared building blocks are reused from the canonical library layers:
``FiLMLayer`` from ``dl_techniques.layers.film`` and ``ConvBlock`` /
``DenseBlock`` from ``dl_techniques.layers.standard_blocks`` (configured with
``normalization_type="batch_norm", activation_type="golu"`` to preserve the
original Conv/Dense + BatchNorm + GoLU behavior). The CCNet framework symbols
(``CCNetConfig``, ``CCNetOrchestrator``, ``wrap_keras_model``) are imported
absolutely from the framework submodules so this module carries no dependency
on ``train.*``.
"""

import keras
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from dl_techniques.models.ccnets.base import CCNetConfig
from dl_techniques.models.ccnets.orchestrators import CCNetOrchestrator
from dl_techniques.models.ccnets.utils import wrap_keras_model
from dl_techniques.layers.film import FiLMLayer
from dl_techniques.layers.standard_blocks import ConvBlock, DenseBlock
from dl_techniques.layers.activations.golu import GoLU
from dl_techniques.regularizers.soft_orthogonal import (
    SoftOrthonormalConstraintRegularizer
)


# =====================================================================
# MODEL CONFIGURATION
# =====================================================================

@dataclass
class ModelConfig:
    """MNIST CCNet architecture parameters.

    Only model-architecture fields live here; training-side fields
    (learning rates, loss weights, etc.) are supplied to ``create_mnist_ccnet``
    explicitly so the model package never imports a training dataclass.
    """
    explanation_dim: int = 16
    num_classes: int = 10

    explainer_conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    explainer_conv_kernels: List[int] = field(default_factory=lambda: [5, 3, 3])
    explainer_l2_regularization: float = 1e-4
    explainer_orthonormal_lambda: float = 1e-3
    explainer_orthonormal_l1: float = 0.0
    explainer_orthonormal_l2: float = 0.0
    explainer_use_matrix_scaling: bool = True

    reasoner_conv_filters: List[int] = field(default_factory=lambda: [32, 64])
    reasoner_conv_kernels: List[int] = field(default_factory=lambda: [5, 3])
    reasoner_dense_units: List[int] = field(default_factory=lambda: [512, 256])
    reasoner_dropout_rate: float = 0.2
    reasoner_l2_regularization: float = 1e-4

    producer_initial_dense_units: int = 256
    producer_initial_spatial_size: int = 7
    producer_initial_channels: int = 128
    producer_conv_filters: List[int] = field(default_factory=lambda: [128, 64])
    producer_style_units: List[int] = field(default_factory=lambda: [256, 128])
    producer_orthonormal_lambda: float = 1e-3
    producer_orthonormal_l1: float = 0.0
    producer_orthonormal_l2: float = 0.0
    producer_use_matrix_scaling: bool = True


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

@keras.saving.register_keras_serializable()
class MNISTExplainer(keras.Model):
    """Explainer network: extracts causal explanations via variational encoding."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        l2_reg = config.explainer_l2_regularization
        self.l2_regularizer = keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

        self.orthonormal_regularizer = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=config.explainer_orthonormal_lambda,
            l1_coefficient=config.explainer_orthonormal_l1,
            l2_coefficient=config.explainer_orthonormal_l2,
            use_matrix_scaling=config.explainer_use_matrix_scaling
        ) if config.explainer_orthonormal_lambda > 0 else None

        self.conv_blocks: List[ConvBlock] = []
        for i, (filters, kernel) in enumerate(zip(
            config.explainer_conv_filters, config.explainer_conv_kernels
        )):
            is_last = (i == len(config.explainer_conv_filters) - 1)
            self.conv_blocks.append(ConvBlock(
                filters=filters, kernel_size=kernel,
                normalization_type="batch_norm", activation_type="golu",
                use_pooling=not is_last, pool_size=2, pool_type="max",
                kernel_regularizer=self.l2_regularizer,
                name=f"conv_block_{i + 1}"
            ))

        self.global_pool = keras.layers.GlobalMaxPooling2D(name="global_pool")
        self.flatten = keras.layers.Flatten(name="flatten")
        self.fc_mu = keras.layers.Dense(
            config.explanation_dim, name="mu",
            kernel_regularizer=self.orthonormal_regularizer
        )
        self.fc_log_var = keras.layers.Dense(
            config.explanation_dim, name="log_var",
            kernel_regularizer=self.l2_regularizer
        )

    def build(self, input_shape):
        current_shape = input_shape
        for block in self.conv_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)
        pooled_shape = self.global_pool.compute_output_shape(current_shape)
        flat_shape = self.flatten.compute_output_shape(pooled_shape)
        self.fc_mu.build(flat_shape)
        self.fc_log_var.build(flat_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        for block in self.conv_blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        features = self.flatten(x)
        return self.fc_mu(features), self.fc_log_var(features)

    def get_config(self):
        config = super().get_config()
        config["config"] = {
            "explanation_dim": self.config.explanation_dim,
            "num_classes": self.config.num_classes,
            "explainer_conv_filters": self.config.explainer_conv_filters,
            "explainer_conv_kernels": self.config.explainer_conv_kernels,
            "explainer_l2_regularization": self.config.explainer_l2_regularization,
            "explainer_orthonormal_lambda": self.config.explainer_orthonormal_lambda,
            "explainer_orthonormal_l1": self.config.explainer_orthonormal_l1,
            "explainer_orthonormal_l2": self.config.explainer_orthonormal_l2,
            "explainer_use_matrix_scaling": self.config.explainer_use_matrix_scaling,
        }
        return config

    @classmethod
    def from_config(cls, config):
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


@keras.saving.register_keras_serializable()
class MNISTReasoner(keras.Model):
    """Reasoner network: classifies images using visual features and explanations."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        l2_reg = config.reasoner_l2_regularization
        self.l2_regularizer = keras.regularizers.L2(l2_reg) if l2_reg > 0 else None

        self.conv_blocks: List[ConvBlock] = []
        for i, (filters, kernel) in enumerate(zip(
            config.reasoner_conv_filters, config.reasoner_conv_kernels
        )):
            self.conv_blocks.append(ConvBlock(
                filters=filters, kernel_size=kernel,
                normalization_type="batch_norm", activation_type="golu",
                use_pooling=True, pool_size=2, pool_type="max",
                kernel_regularizer=self.l2_regularizer,
                name=f"conv_block_{i + 1}"
            ))

        self.flatten = keras.layers.Flatten(name="flatten")

        self.dense_blocks: List[DenseBlock] = []
        for i, units in enumerate(config.reasoner_dense_units):
            self.dense_blocks.append(DenseBlock(
                units=units, dropout_rate=config.reasoner_dropout_rate,
                normalization_type="batch_norm", activation_type="golu",
                kernel_regularizer=self.l2_regularizer,
                name=f"dense_block_{i + 1}"
            ))

        self.fc_output = keras.layers.Dense(
            config.num_classes, activation="softmax", name="classifier"
        )

    def build(self, input_shape):
        current_shape = input_shape
        for block in self.conv_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)
        flat_shape = self.flatten.compute_output_shape(current_shape)
        combined_shape = (flat_shape[0], flat_shape[-1] + self.config.explanation_dim)
        current_shape = combined_shape
        for block in self.dense_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)
        self.fc_output.build(current_shape)
        super().build(input_shape)

    def call(self, x, e, training=None):
        img_features = x
        for block in self.conv_blocks:
            img_features = block(img_features, training=training)
        img_features = self.flatten(img_features)
        combined = keras.ops.concatenate([img_features, e], axis=-1)
        for block in self.dense_blocks:
            combined = block(combined, training=training)
        return self.fc_output(combined)

    def get_config(self):
        config = super().get_config()
        config["config"] = {
            "explanation_dim": self.config.explanation_dim,
            "num_classes": self.config.num_classes,
            "reasoner_conv_filters": self.config.reasoner_conv_filters,
            "reasoner_conv_kernels": self.config.reasoner_conv_kernels,
            "reasoner_dense_units": self.config.reasoner_dense_units,
            "reasoner_dropout_rate": self.config.reasoner_dropout_rate,
            "reasoner_l2_regularization": self.config.reasoner_l2_regularization,
        }
        return config

    @classmethod
    def from_config(cls, config):
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


@keras.saving.register_keras_serializable()
class MNISTProducer(keras.Model):
    """Producer network: generates images using FiLM-modulated style conditioning."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.orthonormal_regularizer = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=config.producer_orthonormal_lambda,
            l1_coefficient=config.producer_orthonormal_l1,
            l2_coefficient=config.producer_orthonormal_l2,
            use_matrix_scaling=config.producer_use_matrix_scaling
        ) if config.producer_orthonormal_lambda > 0 else None

        # DECISION plan_2026-06-03_5c8c6d19/D-004
        # The label path MUST stay a bias-free Dense on the class-probability
        # vector. Do NOT replace with keras.layers.Embedding: Embedding needs
        # integer indices, forcing a non-differentiable argmax in the
        # orchestrator that severs the Reasoner from cooperative credit
        # assignment. For a one-hot input this Dense IS an embedding lookup, but
        # it stays differentiable for the Reasoner's soft predictions so
        # reconstruction_loss can train the Reasoner (CCNet Invariant 1; D-004).
        self.label_projection = keras.layers.Dense(
            config.producer_initial_dense_units,
            use_bias=False,
            name="label_projection"
        )

        s, c = config.producer_initial_spatial_size, config.producer_initial_channels
        self.fc_content = keras.layers.Dense(s * s * c, activation='relu', name="fc_content")
        self.reshape_content = keras.layers.Reshape((s, s, c), name="reshape_content")

        self.generation_blocks: List[Dict[str, keras.layers.Layer]] = []
        for i, filters in enumerate(config.producer_conv_filters):
            self.generation_blocks.append({
                'upsample': keras.layers.UpSampling2D(size=(2, 2), name=f"up_{i + 1}"),
                'conv': keras.layers.Conv2D(filters, 3, padding="same", name=f"conv_{i + 1}"),
                'bn': keras.layers.BatchNormalization(name=f"norm_{i + 1}"),
                'film': FiLMLayer(kernel_regularizer=self.orthonormal_regularizer, name=f"film_{i + 1}"),
                'activation': GoLU(name=f"act_{i + 1}")
            })

        self.conv_out_1 = keras.layers.Conv2D(32, 1, padding="same", activation="linear", name="conv_out_1")
        self.conv_out_2 = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="conv_out_2")

    def build(self, input_shape):
        # input_shape is the label tensor shape: [batch, num_classes] (probabilities).
        label_shape = input_shape
        explanation_shape = (label_shape[0], self.config.explanation_dim)

        self.label_projection.build(label_shape)
        projected_shape = self.label_projection.compute_output_shape(label_shape)
        self.fc_content.build(projected_shape)
        fc_shape = self.fc_content.compute_output_shape(projected_shape)
        self.reshape_content.build(fc_shape)

        s, c = self.config.producer_initial_spatial_size, self.config.producer_initial_channels
        current_shape = (label_shape[0], s, s, c)

        for block in self.generation_blocks:
            block['upsample'].build(current_shape)
            up_shape = block['upsample'].compute_output_shape(current_shape)
            block['conv'].build(up_shape)
            conv_shape = block['conv'].compute_output_shape(up_shape)
            block['bn'].build(conv_shape)
            block['film'].build([conv_shape, explanation_shape])
            block['activation'].build(conv_shape)
            current_shape = conv_shape

        self.conv_out_1.build(current_shape)
        out1_shape = self.conv_out_1.compute_output_shape(current_shape)
        self.conv_out_2.build(out1_shape)
        super().build(input_shape)

    def call(self, y, e, training=None):
        # y is a class-probability / one-hot vector [batch, num_classes].
        # The differentiable projection preserves gradient flow to the Reasoner.
        c = self.label_projection(y)
        c = self.fc_content(c)
        x = self.reshape_content(c)

        for block in self.generation_blocks:
            x = block['upsample'](x)
            x = block['conv'](x)
            x = block['bn'](x, training=training)
            x = block['film']([x, e])
            x = block['activation'](x)

        x = self.conv_out_1(x)
        return self.conv_out_2(x)

    def get_config(self):
        config = super().get_config()
        config["config"] = {
            "explanation_dim": self.config.explanation_dim,
            "num_classes": self.config.num_classes,
            "producer_initial_dense_units": self.config.producer_initial_dense_units,
            "producer_initial_spatial_size": self.config.producer_initial_spatial_size,
            "producer_initial_channels": self.config.producer_initial_channels,
            "producer_conv_filters": self.config.producer_conv_filters,
            "producer_style_units": self.config.producer_style_units,
            "producer_orthonormal_lambda": self.config.producer_orthonormal_lambda,
            "producer_orthonormal_l1": self.config.producer_orthonormal_l1,
            "producer_orthonormal_l2": self.config.producer_orthonormal_l2,
            "producer_use_matrix_scaling": self.config.producer_use_matrix_scaling,
        }
        return config

    @classmethod
    def from_config(cls, config):
        model_config = ModelConfig(**config.pop("config"))
        return cls(model_config, **config)


# =====================================================================
# FACTORY
# =====================================================================

def create_mnist_ccnet(
    model_config: ModelConfig,
    *,
    loss_fn: str = 'l2',
    loss_fn_params: Optional[Dict[str, Any]] = None,
    learning_rates: Optional[Dict[str, float]] = None,
    gradient_clip_norm: Optional[float] = 1.0,
    use_mixed_precision: bool = False,
    explainer_weights: Optional[Dict[str, float]] = None,
    reasoner_weights: Optional[Dict[str, float]] = None,
    producer_weights: Optional[Dict[str, float]] = None,
) -> CCNetOrchestrator:
    """Create and initialize a CCNet for MNIST digit classification.

    Builds the three sub-networks on one-hot dummy inputs, wraps them, and
    returns a configured :class:`CCNetOrchestrator`.

    Args:
        model_config: MNIST architecture configuration.
        loss_fn: Base loss-function name for the orchestrator.
        loss_fn_params: Extra parameters for the loss function.
        learning_rates: Per-network learning rates
            (``explainer``/``reasoner``/``producer``).
        gradient_clip_norm: Global gradient-norm clip value (``None`` disables).
        use_mixed_precision: Whether to enable mixed-precision training.
        explainer_weights: Explainer loss-term weights.
        reasoner_weights: Reasoner loss-term weights.
        producer_weights: Producer loss-term weights.

    Returns:
        A configured :class:`CCNetOrchestrator`.
    """
    loss_fn_params = loss_fn_params if loss_fn_params is not None else {}
    learning_rates = learning_rates if learning_rates is not None else {
        'explainer': 3e-4, 'reasoner': 3e-4, 'producer': 3e-4
    }
    explainer_weights = explainer_weights if explainer_weights is not None else {
        'inference': 1.0, 'generation': 1.0, 'kl_divergence': 0.001
    }
    reasoner_weights = reasoner_weights if reasoner_weights is not None else {
        'reconstruction': 1.0, 'inference': 1.0
    }
    producer_weights = producer_weights if producer_weights is not None else {
        'generation': 1.0, 'reconstruction': 1.0
    }

    explainer = MNISTExplainer(model_config)
    reasoner = MNISTReasoner(model_config)
    producer = MNISTProducer(model_config)

    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, model_config.num_classes))
    dummy_latent = keras.ops.zeros((1, model_config.explanation_dim))

    mu, _ = explainer(dummy_image)
    _ = producer(dummy_label, dummy_latent)
    _ = reasoner(dummy_image, dummy_latent)

    ccnet_config = CCNetConfig(
        explanation_dim=model_config.explanation_dim,
        loss_fn=loss_fn,
        loss_fn_params=loss_fn_params,
        learning_rates=learning_rates,
        gradient_clip_norm=gradient_clip_norm,
        use_mixed_precision=use_mixed_precision,
        explainer_weights=explainer_weights,
        reasoner_weights=reasoner_weights,
        producer_weights=producer_weights,
    )

    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=ccnet_config
    )
