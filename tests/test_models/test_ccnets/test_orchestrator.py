"""Tests for the CCNets cooperative-training framework.

These tests pin the two fatal defects fixed in commit b50737af (see
``src/dl_techniques/models/ccnets/FIXES.md``):

* the cooperative gradient path ``reconstruction_loss -> Producer -> Reasoner``
  must stay differentiable (no ``argmax`` severing it);
* the KL weight must be a live ``tf.Variable``, not a float frozen at
  ``@tf.function`` trace time.

plus the loss functions, control strategies, config, and model serialization.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.models.ccnets.base import (
    CCNetConfig,
    CCNetLosses,
    CCNetModelErrors,
)
from dl_techniques.models.ccnets.losses import (
    L1Loss,
    L2Loss,
    HuberLoss,
    PolynomialLoss,
)
from dl_techniques.models.ccnets.control import (
    StaticThresholdStrategy,
    AdaptiveDivergenceStrategy,
)
from dl_techniques.models.ccnets.orchestrators import CCNetOrchestrator
from dl_techniques.models.ccnets.utils import wrap_keras_model


# ---------------------------------------------------------------------
# Tiny test networks implementing the CCNet module contract
# ---------------------------------------------------------------------

IMG_SHAPE = (8, 8, 1)
NUM_CLASSES = 3
EXPLANATION_DIM = 4
BATCH = 4


@keras.saving.register_keras_serializable(package="test_ccnets")
class TinyExplainer(keras.Model):
    """Models P(E|X): image -> (mu, log_var)."""

    def __init__(self, explanation_dim=EXPLANATION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.explanation_dim = explanation_dim
        self.flatten = keras.layers.Flatten()
        self.hidden = keras.layers.Dense(16, activation="relu")
        self.fc_mu = keras.layers.Dense(explanation_dim)
        self.fc_log_var = keras.layers.Dense(explanation_dim)

    def build(self, input_shape):
        self.flatten.build(input_shape)
        flat = self.flatten.compute_output_shape(input_shape)
        self.hidden.build(flat)
        hidden = self.hidden.compute_output_shape(flat)
        self.fc_mu.build(hidden)
        self.fc_log_var.build(hidden)
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.hidden(self.flatten(x))
        return self.fc_mu(h), self.fc_log_var(h)

    def get_config(self):
        config = super().get_config()
        config["explanation_dim"] = self.explanation_dim
        return config


@keras.saving.register_keras_serializable(package="test_ccnets")
class TinyReasoner(keras.Model):
    """Models P(Y|X,E): (image, explanation) -> class probabilities."""

    def __init__(self, num_classes=NUM_CLASSES, explanation_dim=EXPLANATION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.flatten = keras.layers.Flatten()
        self.hidden = keras.layers.Dense(16, activation="relu")
        self.out = keras.layers.Dense(num_classes, activation="softmax")

    def build(self, input_shape):
        # Keras passes only the first call argument's shape (the image).
        self.flatten.build(input_shape)
        flat = self.flatten.compute_output_shape(input_shape)
        combined = (flat[0], flat[-1] + self.explanation_dim)
        self.hidden.build(combined)
        self.out.build(self.hidden.compute_output_shape(combined))
        super().build(input_shape)

    def call(self, x, e, training=None):
        combined = keras.ops.concatenate([self.flatten(x), e], axis=-1)
        return self.out(self.hidden(combined))

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["explanation_dim"] = self.explanation_dim
        return config


@keras.saving.register_keras_serializable(package="test_ccnets")
class TinyProducer(keras.Model):
    """Models P(X|Y,E): (label probs, explanation) -> image.

    The label projection is a bias-free Dense on the probability vector --
    the differentiable replacement for keras.layers.Embedding that keeps the
    Reasoner inside the cooperative loop.
    """

    def __init__(self, num_classes=NUM_CLASSES, img_shape=IMG_SHAPE,
                 explanation_dim=EXPLANATION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.img_shape = tuple(img_shape)
        self.explanation_dim = explanation_dim
        self.label_projection = keras.layers.Dense(8, use_bias=False)
        self.hidden = keras.layers.Dense(
            int(np.prod(self.img_shape)), activation="sigmoid"
        )
        self.reshape = keras.layers.Reshape(self.img_shape)

    def build(self, input_shape):
        # Keras passes only the first call argument's shape (the label vector).
        self.label_projection.build(input_shape)
        proj = self.label_projection.compute_output_shape(input_shape)
        combined = (proj[0], proj[-1] + self.explanation_dim)
        self.hidden.build(combined)
        self.reshape.build(self.hidden.compute_output_shape(combined))
        super().build(input_shape)

    def call(self, y, e, training=None):
        c = self.label_projection(y)
        combined = keras.ops.concatenate([c, e], axis=-1)
        return self.reshape(self.hidden(combined))

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["img_shape"] = list(self.img_shape)
        config["explanation_dim"] = self.explanation_dim
        return config


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_batch():
    """A deterministic (image, one-hot label) batch."""
    rng = np.random.default_rng(0)
    x = rng.random((BATCH, *IMG_SHAPE)).astype("float32")
    labels = rng.integers(0, NUM_CLASSES, size=BATCH)
    y = keras.utils.to_categorical(labels, NUM_CLASSES).astype("float32")
    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)


@pytest.fixture
def orchestrator():
    """A built CCNetOrchestrator over the tiny test networks."""
    explainer = TinyExplainer()
    reasoner = TinyReasoner()
    producer = TinyProducer()

    # Build via dummy forward passes (mirrors create_mnist_ccnet).
    dummy_img = keras.ops.zeros((1, *IMG_SHAPE))
    dummy_label = keras.ops.zeros((1, NUM_CLASSES))
    dummy_latent = keras.ops.zeros((1, EXPLANATION_DIM))
    explainer(dummy_img)
    reasoner(dummy_img, dummy_latent)
    producer(dummy_label, dummy_latent)

    config = CCNetConfig(explanation_dim=EXPLANATION_DIM)
    return CCNetOrchestrator(
        explainer=wrap_keras_model(explainer),
        reasoner=wrap_keras_model(reasoner),
        producer=wrap_keras_model(producer),
        config=config,
    )


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

class TestCCNetConfig:
    def test_defaults(self):
        config = CCNetConfig()
        assert config.explanation_dim == 128
        assert config.loss_fn == "l2"
        assert config.dynamic_weighting is False
        assert config.explainer_weights["kl_divergence"] == pytest.approx(0.01)

    def test_custom_weights(self):
        config = CCNetConfig(
            explanation_dim=32,
            reasoner_weights={"reconstruction": 0.5, "inference": 1.0},
        )
        assert config.explanation_dim == 32
        assert config.reasoner_weights["reconstruction"] == 0.5

    def test_unsupported_loss_raises(self):
        explainer = TinyExplainer()
        reasoner = TinyReasoner()
        producer = TinyProducer()
        with pytest.raises(ValueError, match="Unsupported loss function"):
            CCNetOrchestrator(
                wrap_keras_model(explainer),
                wrap_keras_model(reasoner),
                wrap_keras_model(producer),
                config=CCNetConfig(loss_fn="not_a_loss"),
            )


# ---------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------

class TestLossFunctions:
    def test_l1_loss(self):
        a = keras.ops.zeros((4, 4))
        b = keras.ops.ones((4, 4)) * 2.0
        assert float(L1Loss()(a, b)) == pytest.approx(2.0)

    def test_l2_loss(self):
        a = keras.ops.zeros((4, 4))
        b = keras.ops.ones((4, 4)) * 3.0
        assert float(L2Loss()(a, b)) == pytest.approx(9.0)

    def test_huber_loss_linear_and_quadratic(self):
        loss = HuberLoss(delta=1.0)
        small = loss(keras.ops.zeros((2, 2)), keras.ops.ones((2, 2)) * 0.5)
        large = loss(keras.ops.zeros((2, 2)), keras.ops.ones((2, 2)) * 5.0)
        # Quadratic regime: 0.5 * 0.5^2 = 0.125
        assert float(small) == pytest.approx(0.125)
        # Linear regime: 0.5 * 1 + 1 * (5 - 1) = 4.5
        assert float(large) == pytest.approx(4.5)

    def test_polynomial_loss(self):
        loss = PolynomialLoss(p=2.0)
        value = loss(keras.ops.zeros((4, 4)), keras.ops.ones((4, 4)) * 2.0)
        assert float(value) == pytest.approx(4.0, abs=1e-3)

    def test_polynomial_rejects_nonpositive_exponent(self):
        with pytest.raises(ValueError, match="must be positive"):
            PolynomialLoss(p=0.0)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

class TestForwardPass:
    def test_output_keys(self, orchestrator, sample_batch):
        x, y = sample_batch
        tensors = orchestrator.forward_pass(x, y, training=True)
        expected = {
            "x_input", "y_truth", "mu", "log_var", "e_latent",
            "y_inferred", "x_reconstructed", "x_generated",
        }
        assert expected.issubset(tensors.keys())

    def test_output_shapes(self, orchestrator, sample_batch):
        x, y = sample_batch
        tensors = orchestrator.forward_pass(x, y, training=False)
        assert tuple(tensors["mu"].shape) == (BATCH, EXPLANATION_DIM)
        assert tuple(tensors["y_inferred"].shape) == (BATCH, NUM_CLASSES)
        assert tuple(tensors["x_reconstructed"].shape) == (BATCH, *IMG_SHAPE)
        assert tuple(tensors["x_generated"].shape) == (BATCH, *IMG_SHAPE)

    def test_y_inferred_is_probability_distribution(self, orchestrator, sample_batch):
        x, y = sample_batch
        tensors = orchestrator.forward_pass(x, y, training=False)
        row_sums = keras.ops.convert_to_numpy(
            keras.ops.sum(tensors["y_inferred"], axis=-1)
        )
        np.testing.assert_allclose(row_sums, np.ones(BATCH), atol=1e-5)

    def test_no_nans(self, orchestrator, sample_batch):
        x, y = sample_batch
        tensors = orchestrator.forward_pass(x, y, training=True)
        for key in ("x_reconstructed", "x_generated", "e_latent"):
            assert not np.any(np.isnan(keras.ops.convert_to_numpy(tensors[key])))


# ---------------------------------------------------------------------
# Losses and errors
# ---------------------------------------------------------------------

class TestLossesAndErrors:
    def test_compute_losses_nonnegative(self, orchestrator, sample_batch):
        x, y = sample_batch
        losses = orchestrator.compute_losses(
            orchestrator.forward_pass(x, y, training=True)
        )
        assert isinstance(losses, CCNetLosses)
        for value in losses.to_dict().values():
            assert value >= 0.0

    def test_compute_model_errors(self, orchestrator, sample_batch):
        x, y = sample_batch
        tensors = orchestrator.forward_pass(x, y, training=True)
        losses = orchestrator.compute_losses(tensors)
        errors = orchestrator.compute_model_errors(losses, tensors)
        assert isinstance(errors, CCNetModelErrors)
        for value in errors.to_dict().values():
            assert np.isfinite(value)


# ---------------------------------------------------------------------
# Gradient flow -- the core regression tests
# ---------------------------------------------------------------------

class TestGradientFlow:
    def test_reconstruction_gradient_reaches_reasoner(self, orchestrator, sample_batch):
        """Regression for the H3 fix: argmax used to sever this gradient.

        With the differentiable label projection, reconstruction_loss must
        backpropagate into every Reasoner variable.
        """
        x, y = sample_batch
        reasoner_vars = orchestrator.reasoner.trainable_variables
        with tf.GradientTape() as tape:
            tensors = orchestrator.forward_pass(x, y, training=True)
            losses = orchestrator.compute_losses(tensors)
        grads = tape.gradient(losses.reconstruction_loss, reasoner_vars)

        assert all(g is not None for g in grads), (
            "reconstruction_loss gradient is severed from the Reasoner -- "
            "the cooperative causal mechanism is broken (see FIXES.md, H3)."
        )
        total_norm = sum(float(tf.norm(g)) for g in grads)
        assert total_norm > 1e-8

    def test_each_module_error_trains_its_own_weights(self, orchestrator, sample_batch):
        x, y = sample_batch
        with tf.GradientTape(persistent=True) as tape:
            tensors = orchestrator.forward_pass(x, y, training=True)
            losses = orchestrator.compute_losses(tensors)
            errors = orchestrator.compute_model_errors(losses, tensors)

        for module, error in (
            (orchestrator.explainer, errors.explainer_error),
            (orchestrator.reasoner, errors.reasoner_error),
            (orchestrator.producer, errors.producer_error),
        ):
            grads = tape.gradient(error, module.trainable_variables)
            assert any(g is not None for g in grads)
            assert sum(float(tf.norm(g)) for g in grads if g is not None) > 1e-8
        del tape


# ---------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------

class TestTrainStep:
    def test_returns_expected_metrics(self, orchestrator, sample_batch):
        x, y = sample_batch
        out = orchestrator.train_step(x, y)
        for key in (
            "generation_loss", "reconstruction_loss", "inference_loss",
            "explainer_error", "reasoner_error", "producer_error",
            "explainer_grad_norm", "reasoner_grad_norm", "producer_grad_norm",
            "batch_accuracy",
        ):
            assert key in out
            assert np.isfinite(float(out[key]))

    def test_training_reduces_total_error(self, orchestrator, sample_batch):
        """A handful of steps on a fixed batch must lower the total error."""
        x, y = sample_batch
        first = orchestrator.train_step(x, y)
        start = sum(float(first[k]) for k in
                    ("explainer_error", "reasoner_error", "producer_error"))
        for _ in range(15):
            last = orchestrator.train_step(x, y)
        end = sum(float(last[k]) for k in
                  ("explainer_error", "reasoner_error", "producer_error"))
        assert end < start

    def test_kl_weight_is_live_variable(self, orchestrator, sample_batch):
        """Regression for the H5 fix: the KL weight was a float baked into the
        @tf.function graph at trace time, making KL annealing a silent no-op.
        It must now be a tf.Variable read live every step."""
        x, y = sample_batch
        assert isinstance(orchestrator.kl_weight, tf.Variable)

        orchestrator.kl_weight.assign(0.0)
        low = float(orchestrator.train_step(x, y)["explainer_error"])
        orchestrator.kl_weight.assign(1000.0)
        high = float(orchestrator.train_step(x, y)["explainer_error"])

        assert high > low + 1.0, (
            "explainer_error did not respond to a 1000x KL-weight change -- "
            "the weight is frozen in the compiled graph (see FIXES.md, H5)."
        )


# ---------------------------------------------------------------------
# Control strategies
# ---------------------------------------------------------------------

class TestControlStrategies:
    def test_static_threshold(self):
        strategy = StaticThresholdStrategy(threshold=0.5)
        assert bool(strategy.should_train_reasoner({"batch_accuracy": 0.4})) is True
        assert bool(strategy.should_train_reasoner({"batch_accuracy": 0.9})) is False

    def test_adaptive_divergence_updates_state(self):
        strategy = AdaptiveDivergenceStrategy(patience=3)
        metrics = {
            "batch_accuracy": tf.constant(0.5),
            "explainer_error": tf.constant(1.0),
            "reasoner_error": tf.constant(0.2),
            "producer_error": tf.constant(1.0),
        }
        strategy.update_state(metrics)
        decision = strategy.should_train_reasoner(metrics)
        assert decision is not None

    def test_adaptive_divergence_static_ceiling(self):
        strategy = AdaptiveDivergenceStrategy(static_ceiling=0.99)
        metrics = {
            "batch_accuracy": tf.constant(0.999),
            "explainer_error": tf.constant(1.0),
            "reasoner_error": tf.constant(1.0),
            "producer_error": tf.constant(1.0),
        }
        assert bool(strategy.should_train_reasoner(metrics)) is False


# ---------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------

class TestInferenceUtilities:
    def test_counterfactual_generation_shape(self, orchestrator, sample_batch):
        x, _ = sample_batch
        y_target = keras.utils.to_categorical([1] * BATCH, NUM_CLASSES).astype("float32")
        out = orchestrator.counterfactual_generation(x, tf.convert_to_tensor(y_target))
        assert tuple(out.shape) == (BATCH, *IMG_SHAPE)

    def test_disentangle_causes(self, orchestrator, sample_batch):
        x, _ = sample_batch
        y_explicit, e_latent = orchestrator.disentangle_causes(x)
        assert tuple(y_explicit.shape) == (BATCH, NUM_CLASSES)
        assert tuple(e_latent.shape) == (BATCH, EXPLANATION_DIM)

    def test_evaluate_does_not_update_weights(self, orchestrator, sample_batch):
        x, y = sample_batch
        before = [
            keras.ops.convert_to_numpy(v).copy()
            for v in orchestrator.reasoner.trainable_variables
        ]
        orchestrator.evaluate(x, y)
        after = [
            keras.ops.convert_to_numpy(v)
            for v in orchestrator.reasoner.trainable_variables
        ]
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b, a)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

class TestSerialization:
    def test_save_and_load_round_trip(self, orchestrator, sample_batch, tmp_path):
        # counterfactual_generation uses the deterministic mu (no sampling),
        # so it is a stable fingerprint of the saved weights.
        x, _ = sample_batch
        y_target = tf.convert_to_tensor(
            keras.utils.to_categorical([1] * BATCH, NUM_CLASSES).astype("float32")
        )
        before = keras.ops.convert_to_numpy(
            orchestrator.counterfactual_generation(x, y_target)
        )

        base_path = str(tmp_path / "ccnet")
        orchestrator.save_models(base_path)
        for suffix in ("explainer", "reasoner", "producer"):
            assert (tmp_path / f"ccnet_{suffix}.keras").exists()

        orchestrator.load_models(base_path)
        after = keras.ops.convert_to_numpy(
            orchestrator.counterfactual_generation(x, y_target)
        )
        np.testing.assert_allclose(before, after, atol=1e-5)
