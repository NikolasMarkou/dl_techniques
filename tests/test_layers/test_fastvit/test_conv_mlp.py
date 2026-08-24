"""
Test suite for FastVitConvMlp (FastViT / MobileCLIP2 MCi channel mixer).

Covers initialization + stored config, constructor validation, forward shape
over multiple spatial sizes (including a NON-square one) and channel widths,
`compute_output_shape` pre- and post-build, training-mode behaviour, gradient
flow, a `.keras` VALUE round trip, and the mandated behavioural pin
`test_first_conv_is_depthwise_7x7`.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.conv_mlp import FastVitConvMlp


class TestFastVitConvMlp:
    """Comprehensive test suite for the FastVitConvMlp layer."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        """Standard configuration used across the suite."""
        return {'dim': 16, 'hidden_dim': 32}

    @pytest.fixture
    def sample_input(self):
        """Deterministic rank-4 sample input, (B, H, W, C) = (2, 8, 8, 16)."""
        rng = np.random.default_rng(1234)
        return rng.normal(size=(2, 8, 8, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        layer = FastVitConvMlp(**basic_config)

        assert layer.dim == 16
        assert layer.hidden_dim == 32
        assert layer.out_dim == 16
        assert layer.kernel_size == 7
        assert layer.dropout_rate == 0.0
        assert not layer.built

    def test_defaults_hidden_and_out_dim(self):
        layer = FastVitConvMlp(dim=24)
        assert layer.hidden_dim == 24
        assert layer.out_dim == 24

    def test_default_kernel_initializer_is_trunc_normal_002(self):
        """timm's `_init_weights` uses trunc_normal_(std=0.02) on conv weights."""
        layer = FastVitConvMlp(dim=8)
        assert isinstance(layer.kernel_initializer, keras.initializers.TruncatedNormal)
        assert layer.kernel_initializer.stddev == pytest.approx(0.02)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)

    def test_default_kernel_initializer_is_per_instance(self):
        """Two instances must NOT share one Initializer object.

        A default argument like `kernel_initializer=TruncatedNormal(0.02)` is
        evaluated ONCE at import time, so every FastVitConvMlp in the process
        would hold the SAME object. That is harmless while `seed=None` but
        silently correlates every layer's draws the moment a seed is set, and it
        makes one layer's `set_config`-style mutation reach all the others.
        """
        a = FastVitConvMlp(dim=8)
        b = FastVitConvMlp(dim=8)
        assert a.kernel_initializer is not b.kernel_initializer, (
            "both FastVitConvMlp instances hold the SAME kernel_initializer "
            "object — the default argument is being evaluated at import time"
        )
        # Both are still the reference initializer.
        for layer in (a, b):
            assert isinstance(
                layer.kernel_initializer, keras.initializers.TruncatedNormal)
            assert layer.kernel_initializer.stddev == pytest.approx(0.02)

        # An EXPLICIT initializer is still honoured as-is (shared on purpose).
        shared = keras.initializers.TruncatedNormal(stddev=0.05)
        c = FastVitConvMlp(dim=8, kernel_initializer=shared)
        d = FastVitConvMlp(dim=8, kernel_initializer=shared)
        assert c.kernel_initializer is shared
        assert d.kernel_initializer is shared

    def test_config_completeness(self, basic_config):
        layer = FastVitConvMlp(**basic_config, kernel_size=3, dropout_rate=0.1)
        config = layer.get_config()

        for key in (
                'dim', 'hidden_dim', 'out_dim', 'kernel_size', 'activation',
                'dropout_rate', 'kernel_initializer', 'bias_initializer',
                'kernel_regularizer', 'bias_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = FastVitConvMlp.from_config(config)
        assert rebuilt.dim == layer.dim
        assert rebuilt.hidden_dim == layer.hidden_dim
        assert rebuilt.kernel_size == layer.kernel_size
        assert rebuilt.dropout_rate == layer.dropout_rate

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            FastVitConvMlp(dim=0)

    def test_invalid_hidden_dim(self):
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            FastVitConvMlp(dim=8, hidden_dim=-4)

    def test_invalid_out_dim_nonpositive(self):
        with pytest.raises(ValueError, match="out_dim must be positive"):
            FastVitConvMlp(dim=8, out_dim=0)

    def test_out_dim_must_equal_dim(self):
        with pytest.raises(ValueError, match="out_dim must equal dim"):
            FastVitConvMlp(dim=8, out_dim=16)

    def test_invalid_kernel_size_nonpositive(self):
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            FastVitConvMlp(dim=8, kernel_size=0)

    def test_invalid_kernel_size_even(self):
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            FastVitConvMlp(dim=8, kernel_size=6)

    @pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
    def test_invalid_dropout_rate(self, rate):
        with pytest.raises(ValueError, match=r"dropout_rate must be in \[0, 1\)"):
            FastVitConvMlp(dim=8, dropout_rate=rate)

    def test_invalid_input_rank(self, basic_config):
        layer = FastVitConvMlp(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            layer.build((None, 8, 16))

    def test_invalid_input_channels(self, basic_config):
        layer = FastVitConvMlp(**basic_config)
        with pytest.raises(ValueError, match="Input channel count must equal dim"):
            layer.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("height,width", [(8, 8), (12, 5), (4, 4), (1, 3)])
    @pytest.mark.parametrize("dim", [8, 32])
    def test_forward_shape(self, height, width, dim):
        layer = FastVitConvMlp(dim=dim, hidden_dim=2 * dim)
        x = np.random.default_rng(0).normal(
            size=(2, height, width, dim)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, height, width, dim)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_build(self):
        layer = FastVitConvMlp(dim=16, hidden_dim=64)
        assert not layer.built
        assert layer.compute_output_shape((None, 12, 5, 16)) == (None, 12, 5, 16)
        assert not layer.built

    def test_compute_output_shape_matches_forward(self, basic_config):
        layer = FastVitConvMlp(**basic_config)
        input_shape = (2, 12, 5, 16)
        predicted = layer.compute_output_shape(input_shape)

        x = np.random.default_rng(3).normal(size=input_shape).astype('float32')
        y = layer(x, training=False)

        assert tuple(y.shape) == tuple(predicted)

    # ------------------------------------------------------------------
    # behavioural pin (RED-proven against a plain / 3x3 first conv)
    # ------------------------------------------------------------------

    def test_first_conv_is_depthwise_7x7(self):
        """PIN: the BUILT first-conv kernel must be depthwise 7x7, shape (7,7,C,1).

        Reads the actual weight tensor rather than the constructor kwarg — a
        constructor assertion cannot see a plain Conv2D or a re-wired graph.
        """
        channels = 16
        layer = FastVitConvMlp(dim=channels, hidden_dim=32)
        layer.build((None, 8, 8, channels))

        kernel_shape = tuple(layer.dw_conv.weights[0].shape)
        assert kernel_shape == (7, 7, channels, 1), (
            f"first conv kernel must be a depthwise 7x7 kernel of shape "
            f"(7, 7, {channels}, 1), got {kernel_shape}"
        )

    # ------------------------------------------------------------------
    # training mode / dropout
    # ------------------------------------------------------------------

    def test_training_modes_differ_with_dropout(self, sample_input):
        layer = FastVitConvMlp(dim=16, hidden_dim=64, dropout_rate=0.5)
        keras.utils.set_random_seed(42)
        train_out = ops.convert_to_numpy(layer(sample_input, training=True))
        eval_out = ops.convert_to_numpy(layer(sample_input, training=False))

        assert not np.allclose(train_out, eval_out), (
            "dropout_rate=0.5 must make training and inference outputs differ"
        )

    def test_training_modes_equal_without_dropout(self, sample_input):
        layer = FastVitConvMlp(dim=16, hidden_dim=64, dropout_rate=0.0)
        # Run once with training=False first so BatchNorm moving stats are the
        # same on both reads (training=True would update them).
        eval_out = ops.convert_to_numpy(layer(sample_input, training=False))
        eval_out2 = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(eval_out, eval_out2, atol=1e-6, rtol=0)

        layer_no_bn_update = FastVitConvMlp(dim=16, hidden_dim=64, dropout_rate=0.0)
        layer_no_bn_update.build(sample_input.shape)
        layer_no_bn_update.norm.trainable = False
        a = ops.convert_to_numpy(layer_no_bn_update(sample_input, training=True))
        b = ops.convert_to_numpy(layer_no_bn_update(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        layer = FastVitConvMlp(dim=16, hidden_dim=32)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = ops.mean(ops.square(y))

        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0
        for var, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {var.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), (
                f"all-zero gradient for {var.path}"
            )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def test_serialization_cycle_value_identity(self, sample_input):
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = FastVitConvMlp(dim=16, hidden_dim=32, dropout_rate=0.1)(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'conv_mlp.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="FastVitConvMlp values differ after a .keras round trip"
        )
