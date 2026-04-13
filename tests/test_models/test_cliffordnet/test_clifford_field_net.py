"""Tests for CliffordFieldNet — CliffordNet augmented with gauge field theory."""

from __future__ import annotations

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.cliffordnet.field_net import (
    CliffordFieldBlock,
    CliffordFieldNet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_block_config():
    """Minimal CliffordFieldBlock config for fast tests."""
    return dict(
        channels=32,
        shifts=[1, 2],
        cli_mode="full",
        ctx_mode="diff",
        use_holonomy_context=True,
        use_parallel_transport_residual=True,
        use_gauge_attention=False,
    )


@pytest.fixture
def small_model_config():
    """Minimal CliffordFieldNet config for fast tests."""
    return dict(
        num_classes=10,
        channels=32,
        depth=2,
        patch_size=2,
        shifts=[1, 2],
        cli_mode="full",
        ctx_mode="diff",
        use_holonomy_context=True,
        use_parallel_transport_residual=True,
        use_gauge_attention=False,
        use_anomaly_detection=False,
    )


@pytest.fixture
def sample_images():
    """Random image batch (B=2, H=16, W=16, C=3)."""
    np.random.seed(42)
    return np.random.randn(2, 16, 16, 3).astype("float32")


@pytest.fixture
def sample_features():
    """Random feature batch (B=2, H=8, W=8, D=32)."""
    np.random.seed(42)
    return np.random.randn(2, 8, 8, 32).astype("float32")


# ===========================================================================
# CliffordFieldBlock
# ===========================================================================


class TestCliffordFieldBlock:
    """Tests for the CliffordFieldBlock layer."""

    def test_init(self, small_block_config):
        """Block can be instantiated."""
        block = CliffordFieldBlock(**small_block_config)
        assert block.channels == 32
        assert block.use_holonomy_context is True
        assert block.use_parallel_transport_residual is True
        assert block.use_gauge_attention is False

    def test_forward_pass(self, small_block_config, sample_features):
        """Block preserves input shape."""
        block = CliffordFieldBlock(**small_block_config)
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_forward_training(self, small_block_config, sample_features):
        """Block works in training mode."""
        block = CliffordFieldBlock(**small_block_config)
        out = block(sample_features, training=True)
        assert out.shape == sample_features.shape

    def test_no_holonomy(self, sample_features):
        """Block works without holonomy context."""
        block = CliffordFieldBlock(
            channels=32,
            shifts=[1, 2],
            use_holonomy_context=False,
            use_parallel_transport_residual=True,
        )
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_no_transport(self, sample_features):
        """Block works without parallel-transport residual."""
        block = CliffordFieldBlock(
            channels=32,
            shifts=[1, 2],
            use_holonomy_context=True,
            use_parallel_transport_residual=False,
        )
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_with_gauge_attention(self, sample_features):
        """Block works with gauge-invariant attention enabled."""
        block = CliffordFieldBlock(
            channels=32,
            shifts=[1, 2],
            use_gauge_attention=True,
            num_attention_heads=4,
        )
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_minimal_config(self, sample_features):
        """Block works with all optional features disabled."""
        block = CliffordFieldBlock(
            channels=32,
            shifts=[1, 2],
            use_holonomy_context=False,
            use_parallel_transport_residual=False,
            use_gauge_attention=False,
        )
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_abs_ctx_mode(self, sample_features):
        """Block works with ctx_mode='abs'."""
        block = CliffordFieldBlock(
            channels=32,
            shifts=[1, 2],
            ctx_mode="abs",
        )
        out = block(sample_features, training=False)
        assert out.shape == sample_features.shape

    def test_get_config(self, small_block_config):
        """Config round-trip preserves all parameters."""
        block = CliffordFieldBlock(**small_block_config)
        config = block.get_config()
        assert config["channels"] == 32
        assert config["shifts"] == [1, 2]
        assert config["use_holonomy_context"] is True
        assert config["use_parallel_transport_residual"] is True
        assert config["use_gauge_attention"] is False

    def test_gradient_flow(self, small_block_config, sample_features):
        """Gradients flow through the block without NaN."""
        block = CliffordFieldBlock(**small_block_config)
        x_var = tf.Variable(sample_features)
        with tf.GradientTape() as tape:
            out = block(x_var, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, x_var)
        assert grads is not None
        assert np.all(np.isfinite(grads.numpy()))

    def test_different_spatial_sizes(self, small_block_config):
        """Block works with non-square spatial dims."""
        block = CliffordFieldBlock(**small_block_config)
        x = np.random.randn(2, 6, 10, 32).astype("float32")
        out = block(x, training=False)
        assert out.shape == (2, 6, 10, 32)

    def test_invalid_channels(self):
        """Negative channels raises ValueError."""
        with pytest.raises(ValueError, match="channels must be positive"):
            CliffordFieldBlock(channels=-1, shifts=[1, 2])


# ===========================================================================
# CliffordFieldNet
# ===========================================================================


class TestCliffordFieldNet:
    """Tests for the CliffordFieldNet model."""

    def test_init(self, small_model_config):
        """Model can be instantiated."""
        model = CliffordFieldNet(**small_model_config)
        assert model.num_classes == 10
        assert model.channels == 32
        assert model.depth == 2
        assert len(model.blocks_list) == 2

    def test_forward_pass(self, small_model_config, sample_images):
        """Model produces correct output shape."""
        model = CliffordFieldNet(**small_model_config)
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_forward_training(self, small_model_config, sample_images):
        """Model works in training mode."""
        model = CliffordFieldNet(**small_model_config)
        logits = model(sample_images, training=True)
        assert logits.shape == (2, 10)

    def test_anomaly_detection(self, sample_images):
        """Model returns stress dict when anomaly detection is on."""
        model = CliffordFieldNet(
            num_classes=10,
            channels=32,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            use_anomaly_detection=True,
        )
        result = model(sample_images, training=False)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"logits", "stress", "anomaly_mask"}
        assert result["logits"].shape == (2, 10)
        assert result["stress"].shape == (2, 1)
        assert result["anomaly_mask"].shape == (2, 1)

    def test_gauge_attention(self, sample_images):
        """Model works with gauge-invariant attention enabled."""
        model = CliffordFieldNet(
            num_classes=10,
            channels=32,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            use_gauge_attention=True,
            num_attention_heads=4,
        )
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_all_features(self, sample_images):
        """Model works with ALL features enabled."""
        model = CliffordFieldNet(
            num_classes=10,
            channels=32,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            use_holonomy_context=True,
            use_parallel_transport_residual=True,
            use_gauge_attention=True,
            num_attention_heads=4,
            use_anomaly_detection=True,
        )
        result = model(sample_images, training=False)
        assert isinstance(result, dict)
        assert result["logits"].shape == (2, 10)

    def test_gradient_flow(self, small_model_config, sample_images):
        """Gradients flow through the model without NaN."""
        model = CliffordFieldNet(**small_model_config)
        x_var = tf.Variable(sample_images)
        with tf.GradientTape() as tape:
            out = model(x_var, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, x_var)
        assert grads is not None
        assert np.all(np.isfinite(grads.numpy()))

    def test_training_step(self, small_model_config, sample_images):
        """Model can execute a training step with compile+fit."""
        model = CliffordFieldNet(**small_model_config)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )
        labels = np.array([3, 7])
        history = model.fit(sample_images, labels, epochs=1, verbose=0)
        assert "loss" in history.history
        assert np.isfinite(history.history["loss"][0])

    def test_config_round_trip(self, small_model_config, sample_images):
        """get_config / from_config round-trip produces working model."""
        model = CliffordFieldNet(**small_model_config)
        _ = model(sample_images, training=False)  # build
        config = model.get_config()
        model2 = CliffordFieldNet.from_config(config)
        logits = model2(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_compute_output_shape(self, small_model_config):
        """compute_output_shape returns expected shape."""
        model = CliffordFieldNet(**small_model_config)
        out_shape = model.compute_output_shape((None, 16, 16, 3))
        assert out_shape == (None, 10)

    def test_compute_output_shape_anomaly(self):
        """compute_output_shape with anomaly detection returns dict."""
        model = CliffordFieldNet(
            num_classes=10,
            channels=32,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            use_anomaly_detection=True,
        )
        out_shape = model.compute_output_shape((None, 16, 16, 3))
        assert isinstance(out_shape, dict)
        assert out_shape["logits"] == (None, 10)
        assert out_shape["stress"] == (None, 1)

    def test_patch_size_1(self):
        """Model works with patch_size=1."""
        model = CliffordFieldNet(
            num_classes=10, channels=32, depth=2, patch_size=1, shifts=[1, 2],
        )
        imgs = np.random.randn(2, 8, 8, 3).astype("float32")
        logits = model(imgs, training=False)
        assert logits.shape == (2, 10)

    def test_patch_size_4(self):
        """Model works with patch_size=4."""
        model = CliffordFieldNet(
            num_classes=10, channels=32, depth=2, patch_size=4, shifts=[1, 2],
        )
        imgs = np.random.randn(2, 32, 32, 3).astype("float32")
        logits = model(imgs, training=False)
        assert logits.shape == (2, 10)

    def test_dropout(self, sample_images):
        """Model works with head dropout."""
        model = CliffordFieldNet(
            num_classes=10,
            channels=32,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            dropout_rate=0.1,
        )
        logits = model(sample_images, training=True)
        assert logits.shape == (2, 10)

    def test_invalid_num_classes(self):
        """Non-positive num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            CliffordFieldNet(num_classes=0, channels=32, depth=2, shifts=[1, 2])


# ===========================================================================
# Factory variants
# ===========================================================================


class TestCliffordFieldNetVariants:
    """Tests for CliffordFieldNet factory methods."""

    def test_from_variant_nano(self, sample_images):
        """Nano variant creates working model."""
        model = CliffordFieldNet.from_variant(
            "nano", num_classes=10, channels=32, depth=2
        )
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_from_variant_lite(self, sample_images):
        """Lite variant creates working model."""
        model = CliffordFieldNet.from_variant(
            "lite", num_classes=10, channels=32, depth=2
        )
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_from_variant_base(self, sample_images):
        """Base variant creates working model (with gauge attention)."""
        model = CliffordFieldNet.from_variant(
            "base", num_classes=10, channels=32, depth=2,
            num_attention_heads=4,
        )
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_convenience_nano(self, sample_images):
        """CliffordFieldNet.nano() convenience method."""
        model = CliffordFieldNet.nano(num_classes=10, channels=32, depth=2)
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_convenience_lite(self, sample_images):
        """CliffordFieldNet.lite() convenience method."""
        model = CliffordFieldNet.lite(num_classes=10, channels=32, depth=2)
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_convenience_base(self, sample_images):
        """CliffordFieldNet.base() convenience method."""
        model = CliffordFieldNet.base(
            num_classes=10, channels=32, depth=2, num_attention_heads=4,
        )
        logits = model(sample_images, training=False)
        assert logits.shape == (2, 10)

    def test_invalid_variant(self):
        """Unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            CliffordFieldNet.from_variant("nonexistent", num_classes=10)

    def test_variant_override(self, sample_images):
        """Factory kwargs override variant defaults."""
        model = CliffordFieldNet.from_variant(
            "nano",
            num_classes=10,
            channels=32,
            depth=2,
            use_anomaly_detection=True,
        )
        result = model(sample_images, training=False)
        assert isinstance(result, dict)
        assert "stress" in result
