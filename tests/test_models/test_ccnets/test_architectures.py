"""Tests for the migrated CCNet task architectures.

Companion to ``test_orchestrator.py`` (which exercises the framework). This
module pins the contract + serialization round-trip of the concrete networks
relocated into ``dl_techniques.models.ccnets`` during the train->model
consolidation (plan_2026-06-03_5c8c6d19):

* MNIST / CIFAR-100 image networks + factories;
* text/sentiment networks (non-AR + AR producers) + factories;
* the package public surface (every name in ``__all__`` resolves).

The shared building blocks (``FiLMLayer``, ``ConvBlock``, ``DenseBlock``) are
the canonical library layers from ``dl_techniques.layers.film`` /
``dl_techniques.layers.standard_blocks`` and are tested under
``tests/test_layers/``; this suite tests the architectures that embed them, not
the layers themselves.

All dims are intentionally tiny so the suite stays within a few seconds of
wall-clock. Save/load uses ``model.save(...)`` + ``keras.models.load_model``;
the migrated classes are ``@register_keras_serializable``, so no
``custom_objects`` is required (the registry resolves them by their module
key).
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.ccnets.architectures.mnist import (
    MNISTExplainer, MNISTReasoner, MNISTProducer, create_mnist_ccnet,
    ModelConfig as MNISTModelConfig,
)
from dl_techniques.models.ccnets.architectures.cifar100 import (
    Cifar100Explainer, Cifar100Reasoner, Cifar100Producer,
    create_cifar100_ccnet, HybridCCNetOrchestrator,
    ModelConfig as CifarModelConfig,
)
from dl_techniques.models.ccnets.architectures.text import (
    SentimentExplainer, SentimentReasoner, SentimentProducer,
    ARSentimentProducer, TextCCNetOrchestrator, ARTextCCNetOrchestrator,
    create_text_ccnet, ModelConfig as TextModelConfig,
)
from dl_techniques.models.ccnets.orchestrators import CCNetOrchestrator


BATCH = 2


# =====================================================================
# Fixtures: tiny per-task configs
# =====================================================================

@pytest.fixture
def mnist_config():
    return MNISTModelConfig(
        explanation_dim=4,
        num_classes=3,
        explainer_conv_filters=[4, 8],
        explainer_conv_kernels=[3, 3],
        reasoner_conv_filters=[4, 8],
        reasoner_conv_kernels=[3, 3],
        reasoner_dense_units=[16],
        producer_initial_dense_units=16,
        producer_initial_spatial_size=7,
        producer_initial_channels=8,
        producer_conv_filters=[8, 4],
        producer_style_units=[16],
    )


@pytest.fixture
def cifar_config():
    return CifarModelConfig(
        num_classes=5,
        image_channels=3,
        explanation_dim=4,
        explainer_conv_filters=[4, 8, 16],
        explainer_conv_kernels=[3, 3, 3],
        reasoner_conv_filters=[4, 8, 16],
        reasoner_conv_kernels=[3, 3, 3],
        reasoner_dense_units=[16],
        producer_label_units=16,
        producer_initial_spatial=8,
        producer_initial_channels=16,
        producer_conv_filters=[8, 4],
    )


@pytest.fixture
def text_config():
    return TextModelConfig(
        vocab_size=30,
        max_len=6,
        num_classes=2,
        explanation_dim=4,
        embed_dim=8,
        encoder_hidden=8,
        reasoner_dense_units=8,
        producer_type='nonautoregressive',
        producer_d_model=8,
        producer_layers=1,
        producer_heads=2,
        producer_ffn_dim=16,
    )


@pytest.fixture
def ar_text_config():
    return TextModelConfig(
        vocab_size=30,
        max_len=6,
        num_classes=2,
        explanation_dim=4,
        embed_dim=8,
        encoder_hidden=8,
        reasoner_dense_units=8,
        producer_type='autoregressive',
        producer_d_model=8,
        producer_layers=1,
        producer_heads=2,
        producer_ffn_dim=16,
    )


# =====================================================================
# MNIST architectures
# =====================================================================

def _build_mnist(config):
    explainer = MNISTExplainer(config)
    reasoner = MNISTReasoner(config)
    producer = MNISTProducer(config)
    dummy_img = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, config.num_classes))
    dummy_latent = keras.ops.zeros((1, config.explanation_dim))
    explainer(dummy_img)
    reasoner(dummy_img, dummy_latent)
    producer(dummy_label, dummy_latent)
    return explainer, reasoner, producer


class TestMNISTArchitectures:
    def test_contract_shapes(self, mnist_config):
        explainer, reasoner, producer = _build_mnist(mnist_config)
        rng = np.random.default_rng(0)
        x = rng.random((BATCH, 28, 28, 1)).astype("float32")
        y = keras.utils.to_categorical(
            rng.integers(0, mnist_config.num_classes, BATCH),
            mnist_config.num_classes).astype("float32")
        e = rng.random((BATCH, mnist_config.explanation_dim)).astype("float32")

        out = explainer(x)
        assert isinstance(out, tuple) and len(out) == 2
        mu, log_var = out
        assert tuple(mu.shape) == (BATCH, mnist_config.explanation_dim)
        assert tuple(log_var.shape) == (BATCH, mnist_config.explanation_dim)

        y_inf = reasoner(x, e)
        assert tuple(y_inf.shape) == (BATCH, mnist_config.num_classes)

        x_hat = producer(y, e)
        assert tuple(x_hat.shape) == (BATCH, 28, 28, 1)

    def test_producer_label_projection_is_bias_free(self, mnist_config):
        _, _, producer = _build_mnist(mnist_config)
        assert producer.label_projection.use_bias is False

    def test_explainer_round_trip(self, mnist_config, tmp_path):
        explainer, _, _ = _build_mnist(mnist_config)
        x = np.random.default_rng(3).random((BATCH, 28, 28, 1)).astype("float32")
        mu_before = keras.ops.convert_to_numpy(explainer(x, training=False)[0])

        path = str(tmp_path / "mnist_explainer.keras")
        explainer.save(path)
        reloaded = keras.models.load_model(path)
        mu_after = keras.ops.convert_to_numpy(reloaded(x, training=False)[0])
        np.testing.assert_allclose(mu_before, mu_after, atol=1e-4)

    def test_reasoner_round_trip(self, mnist_config, tmp_path):
        _, reasoner, _ = _build_mnist(mnist_config)
        rng = np.random.default_rng(4)
        x = rng.random((BATCH, 28, 28, 1)).astype("float32")
        e = rng.random((BATCH, mnist_config.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(reasoner(x, e, training=False))

        path = str(tmp_path / "mnist_reasoner.keras")
        reasoner.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(x, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)

    def test_producer_round_trip(self, mnist_config, tmp_path):
        _, _, producer = _build_mnist(mnist_config)
        rng = np.random.default_rng(11)
        y = keras.utils.to_categorical(
            rng.integers(0, mnist_config.num_classes, BATCH),
            mnist_config.num_classes).astype("float32")
        e = rng.random((BATCH, mnist_config.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(producer(y, e, training=False))

        path = str(tmp_path / "mnist_producer.keras")
        producer.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(y, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)


# =====================================================================
# CIFAR-100 architectures
# =====================================================================

def _build_cifar(config):
    explainer = Cifar100Explainer(config)
    reasoner = Cifar100Reasoner(config)
    producer = Cifar100Producer(config)
    dummy_x = keras.ops.zeros((1, 32, 32, config.image_channels))
    dummy_y = keras.ops.zeros((1, config.num_classes))
    dummy_e = keras.ops.zeros((1, config.explanation_dim))
    explainer(dummy_x)
    reasoner(dummy_x, dummy_e)
    producer(dummy_y, dummy_e)
    return explainer, reasoner, producer


class TestCifar100Architectures:
    def test_contract_shapes(self, cifar_config):
        explainer, reasoner, producer = _build_cifar(cifar_config)
        rng = np.random.default_rng(0)
        x = rng.random((BATCH, 32, 32, cifar_config.image_channels)).astype("float32")
        y = keras.utils.to_categorical(
            rng.integers(0, cifar_config.num_classes, BATCH),
            cifar_config.num_classes).astype("float32")
        e = rng.random((BATCH, cifar_config.explanation_dim)).astype("float32")

        mu, log_var = explainer(x)
        assert tuple(mu.shape) == (BATCH, cifar_config.explanation_dim)
        assert tuple(log_var.shape) == (BATCH, cifar_config.explanation_dim)

        y_inf = reasoner(x, e)
        assert tuple(y_inf.shape) == (BATCH, cifar_config.num_classes)

        x_hat = producer(y, e)
        assert tuple(x_hat.shape) == (BATCH, 32, 32, cifar_config.image_channels)

    def test_image_features_rank(self, cifar_config):
        """image_features must be the E-independent [B, C] backbone embedding."""
        _, reasoner, _ = _build_cifar(cifar_config)
        x = np.random.default_rng(5).random(
            (BATCH, 32, 32, cifar_config.image_channels)).astype("float32")
        feats = reasoner.image_features(x, training=False)
        assert feats.shape.rank == 2
        assert int(feats.shape[0]) == BATCH

    def test_producer_label_projection_is_bias_free(self, cifar_config):
        _, _, producer = _build_cifar(cifar_config)
        assert producer.label_projection.use_bias is False

    def test_explainer_round_trip(self, cifar_config, tmp_path):
        explainer, _, _ = _build_cifar(cifar_config)
        x = np.random.default_rng(6).random(
            (BATCH, 32, 32, cifar_config.image_channels)).astype("float32")
        mu_before = keras.ops.convert_to_numpy(explainer(x, training=False)[0])

        path = str(tmp_path / "cifar_explainer.keras")
        explainer.save(path)
        reloaded = keras.models.load_model(path)
        mu_after = keras.ops.convert_to_numpy(reloaded(x, training=False)[0])
        np.testing.assert_allclose(mu_before, mu_after, atol=1e-4)

    def test_reasoner_round_trip(self, cifar_config, tmp_path):
        _, reasoner, _ = _build_cifar(cifar_config)
        rng = np.random.default_rng(12)
        x = rng.random((BATCH, 32, 32, cifar_config.image_channels)).astype("float32")
        e = rng.random((BATCH, cifar_config.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(reasoner(x, e, training=False))

        path = str(tmp_path / "cifar_reasoner.keras")
        reasoner.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(x, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)

    def test_producer_round_trip(self, cifar_config, tmp_path):
        _, _, producer = _build_cifar(cifar_config)
        rng = np.random.default_rng(13)
        y = keras.utils.to_categorical(
            rng.integers(0, cifar_config.num_classes, BATCH),
            cifar_config.num_classes).astype("float32")
        e = rng.random((BATCH, cifar_config.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(producer(y, e, training=False))

        path = str(tmp_path / "cifar_producer.keras")
        producer.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(y, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)


# =====================================================================
# Text architectures
# =====================================================================

class TestTextArchitectures:
    def test_explainer_reasoner_contract(self, text_config):
        c = text_config
        explainer = SentimentExplainer(c)
        reasoner = SentimentReasoner(c)
        dummy_x = keras.ops.zeros((1, c.max_len), dtype="int32")
        dummy_e = keras.ops.zeros((1, c.explanation_dim))
        explainer(dummy_x)
        reasoner(dummy_x, dummy_e)

        rng = np.random.default_rng(0)
        x = rng.integers(0, c.vocab_size, (BATCH, c.max_len)).astype("int32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")

        out = explainer(x)
        assert isinstance(out, tuple) and len(out) == 2
        mu, log_var = out
        assert tuple(mu.shape) == (BATCH, c.explanation_dim)
        assert tuple(log_var.shape) == (BATCH, c.explanation_dim)

        y_inf = reasoner(x, e)
        assert tuple(y_inf.shape) == (BATCH, c.num_classes)

    def test_nonautoregressive_producer_shape(self, text_config):
        c = text_config
        producer = SentimentProducer(c)
        dummy_y = keras.ops.zeros((1, c.num_classes))
        dummy_e = keras.ops.zeros((1, c.explanation_dim))
        producer(dummy_y, dummy_e)

        rng = np.random.default_rng(7)
        y = keras.utils.to_categorical(
            rng.integers(0, c.num_classes, BATCH), c.num_classes).astype("float32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")
        logits = producer(y, e)
        assert tuple(logits.shape) == (BATCH, c.max_len, c.vocab_size)

    def test_autoregressive_producer_shape(self, ar_text_config):
        c = ar_text_config
        producer = ARSentimentProducer(c)
        dummy_y = keras.ops.zeros((1, c.num_classes))
        dummy_e = keras.ops.zeros((1, c.explanation_dim))
        dummy_x = keras.ops.zeros((1, c.max_len), dtype="int32")
        producer(dummy_y, dummy_e, dummy_x)

        rng = np.random.default_rng(8)
        y = keras.utils.to_categorical(
            rng.integers(0, c.num_classes, BATCH), c.num_classes).astype("float32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")
        x = rng.integers(0, c.vocab_size, (BATCH, c.max_len)).astype("int32")
        logits = producer(y, e, x, training=False)
        assert tuple(logits.shape) == (BATCH, c.max_len, c.vocab_size)

    def test_both_producers_label_projection_bias_free(self, text_config, ar_text_config):
        """Guards the CCNet differentiable-label invariant on both producers."""
        na = SentimentProducer(text_config)
        ar = ARSentimentProducer(ar_text_config)
        assert na.label_projection.use_bias is False
        assert ar.label_projection.use_bias is False

    def test_explainer_round_trip(self, text_config, tmp_path):
        c = text_config
        explainer = SentimentExplainer(c)
        explainer(keras.ops.zeros((1, c.max_len), dtype="int32"))

        x = np.random.default_rng(9).integers(
            0, c.vocab_size, (BATCH, c.max_len)).astype("int32")
        mu_before = keras.ops.convert_to_numpy(explainer(x, training=False)[0])

        path = str(tmp_path / "text_explainer.keras")
        explainer.save(path)
        reloaded = keras.models.load_model(path)
        mu_after = keras.ops.convert_to_numpy(reloaded(x, training=False)[0])
        np.testing.assert_allclose(mu_before, mu_after, atol=1e-4)

    def test_reasoner_round_trip(self, text_config, tmp_path):
        c = text_config
        reasoner = SentimentReasoner(c)
        reasoner(keras.ops.zeros((1, c.max_len), dtype="int32"),
                 keras.ops.zeros((1, c.explanation_dim)))

        rng = np.random.default_rng(14)
        x = rng.integers(0, c.vocab_size, (BATCH, c.max_len)).astype("int32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(reasoner(x, e, training=False))

        path = str(tmp_path / "text_reasoner.keras")
        reasoner.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(x, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)

    def test_producer_round_trip(self, text_config, tmp_path):
        c = text_config
        producer = SentimentProducer(c)
        producer(keras.ops.zeros((1, c.num_classes)),
                 keras.ops.zeros((1, c.explanation_dim)))

        rng = np.random.default_rng(15)
        y = keras.utils.to_categorical(
            rng.integers(0, c.num_classes, BATCH), c.num_classes).astype("float32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")
        before = keras.ops.convert_to_numpy(producer(y, e, training=False))

        path = str(tmp_path / "text_producer.keras")
        producer.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(y, e, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)

    def test_ar_producer_round_trip(self, ar_text_config, tmp_path):
        c = ar_text_config
        producer = ARSentimentProducer(c)
        producer(keras.ops.zeros((1, c.num_classes)),
                 keras.ops.zeros((1, c.explanation_dim)),
                 keras.ops.zeros((1, c.max_len), dtype="int32"))

        rng = np.random.default_rng(16)
        y = keras.utils.to_categorical(
            rng.integers(0, c.num_classes, BATCH), c.num_classes).astype("float32")
        e = rng.random((BATCH, c.explanation_dim)).astype("float32")
        x = rng.integers(0, c.vocab_size, (BATCH, c.max_len)).astype("int32")
        before = keras.ops.convert_to_numpy(producer(y, e, x, training=False))

        path = str(tmp_path / "text_ar_producer.keras")
        producer.save(path)
        reloaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(reloaded(y, e, x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4)


# =====================================================================
# Factories
# =====================================================================

class TestFactories:
    def test_create_mnist_ccnet(self, mnist_config):
        orch = create_mnist_ccnet(mnist_config)
        assert isinstance(orch, CCNetOrchestrator)

    def test_create_cifar100_ccnet(self, cifar_config):
        orch = create_cifar100_ccnet(cifar_config)
        assert isinstance(orch, CCNetOrchestrator)
        assert not isinstance(orch, HybridCCNetOrchestrator)

    def test_create_cifar100_ccnet_hybrid(self, cifar_config):
        orch = create_cifar100_ccnet(cifar_config, hybrid=True)
        assert isinstance(orch, HybridCCNetOrchestrator)

    def test_create_text_ccnet_nonautoregressive(self, text_config):
        orch = create_text_ccnet(text_config)
        assert isinstance(orch, TextCCNetOrchestrator)
        assert not isinstance(orch, ARTextCCNetOrchestrator)

    def test_create_text_ccnet_autoregressive(self, ar_text_config):
        orch = create_text_ccnet(ar_text_config)
        assert isinstance(orch, ARTextCCNetOrchestrator)


# =====================================================================
# Package surface
# =====================================================================

class TestPackageSurface:
    def test_all_exports_resolve(self):
        import dl_techniques.models.ccnets as pkg

        assert hasattr(pkg, "__all__") and pkg.__all__
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"missing export: {name}"
            assert getattr(pkg, name) is not None

    def test_expected_symbols_present(self):
        import dl_techniques.models.ccnets as pkg

        expected = {
            # framework
            "CCNetConfig", "CCNetTrainer", "CCNetOrchestrator",
            "SequentialCCNetOrchestrator", "EarlyStoppingCallback",
            "wrap_keras_model",
            # architectures + factories
            "MNISTExplainer", "MNISTReasoner", "MNISTProducer", "create_mnist_ccnet",
            "Cifar100Explainer", "Cifar100Reasoner", "Cifar100Producer",
            "create_cifar100_ccnet", "HybridCCNetOrchestrator",
            "SentimentExplainer", "SentimentReasoner", "SentimentProducer",
            "ARSentimentProducer", "TextCCNetOrchestrator",
            "ARTextCCNetOrchestrator", "create_text_ccnet",
        }
        missing = expected - set(pkg.__all__)
        assert not missing, f"package __all__ missing: {missing}"
