"""
Regression test suite for NonLocalAttention.

This file is the regression gate for two pre-existing bugs fixed in
plan_2026-06-14_adaddf34:

- F18 / G1: the DEFAULT ``attention_mode='gaussian'`` forward crashed because
  ``query_conv`` stayed at full ``attention_channels`` while K/V were reduced
  to ``attention_channels // 8`` (``Q@Kᵀ`` contracted mismatched dims).
- F19 / G2: ``.keras`` round-trip corrupted ``kernel_size`` because the
  normalization used ``isinstance(kernel_size, tuple)``, which is False for the
  list/TrackedList form Keras produces on reload.

Tests follow dl-techniques Keras 3 conventions: class-based, pytest fixtures,
fixed-seed inputs, atol=1e-6 numerical tolerance, headless / GPU-agnostic.
"""

import os
import tempfile

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.layers.attention.non_local_attention import NonLocalAttention


class TestNonLocalAttention:
    """Regression + conformance suite for NonLocalAttention."""

    # =========================================================================
    # Fixtures
    # =========================================================================

    @pytest.fixture
    def sample_input(self):
        """Fixed 4D image input (batch=2, H=16, W=16, C=64)."""
        return keras.random.normal([2, 16, 16, 64], seed=42)

    @pytest.fixture
    def layer_config(self):
        """Default layer construction config (gaussian mode)."""
        return {
            'attention_channels': 32,
        }

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_initialization_defaults(self):
        """Construct with defaults; assert stored attributes."""
        layer = NonLocalAttention(attention_channels=32)

        assert layer.attention_channels == 32
        assert layer.attention_mode == 'gaussian'
        # Default kernel_size=(7,7) normalized to a 2-tuple.
        assert layer.kernel_size == (7, 7)
        # gaussian reduces the embedded dim to max(1, channels // 8).
        assert layer.key_value_channels == 32 // 8
        assert not layer.built

    def test_initialization_custom_parameters(self):
        """Construct with custom params; assert stored attributes."""
        layer = NonLocalAttention(
            attention_channels=64,
            kernel_size=(5, 5),
            attention_mode='dot_product',
            output_channels=16,
            dropout_rate=0.1,
            use_bias=True,
        )

        assert layer.attention_channels == 64
        assert layer.kernel_size == (5, 5)
        assert layer.attention_mode == 'dot_product'
        assert layer.output_channels == 16
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True
        # dot_product keeps the embedded dim == attention_channels.
        assert layer.key_value_channels == 64

    def test_invalid_attention_channels(self):
        """attention_channels <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="attention_channels must be positive"):
            NonLocalAttention(attention_channels=0)
        with pytest.raises(ValueError, match="attention_channels must be positive"):
            NonLocalAttention(attention_channels=-8)

    def test_invalid_dropout_rate(self):
        """dropout_rate >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            NonLocalAttention(attention_channels=32, dropout_rate=1.0)

    def test_invalid_attention_mode(self):
        """attention_mode not in {gaussian, dot_product} raises ValueError."""
        with pytest.raises(ValueError, match="attention_mode must be"):
            NonLocalAttention(attention_channels=32, attention_mode='bad')

    # =========================================================================
    # kernel_size normalization (F19 regression)
    # =========================================================================

    @pytest.mark.parametrize("kernel_size,expected", [
        (7, (7, 7)),
        ((5, 5), (5, 5)),
        ([7, 7], (7, 7)),
    ])
    def test_kernel_size_normalization(self, kernel_size, expected):
        """int / tuple / list kernel_size all normalize to a 2-tuple."""
        layer = NonLocalAttention(attention_channels=32, kernel_size=kernel_size)
        assert layer.kernel_size == expected
        assert isinstance(layer.kernel_size, tuple)

    # =========================================================================
    # Forward Pass Tests
    # =========================================================================

    def test_gaussian_forward(self, sample_input):
        """F18 regression: default gaussian forward must produce correct shape.

        This is the core F18 regression — it crashed before the fix.
        """
        layer = NonLocalAttention(attention_channels=32)
        output = layer(sample_input)

        assert output.shape == (2, 16, 16, 64)
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(np.isfinite(output_np))

    def test_dot_product_forward(self, sample_input):
        """dot_product mode forward produces correct shape, finite."""
        layer = NonLocalAttention(attention_channels=32, attention_mode='dot_product')
        output = layer(sample_input)

        assert output.shape == (2, 16, 16, 64)
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(np.isfinite(output_np))

    def test_small_channel_guard(self):
        """SC5: attention_channels=4 (not divisible by 8) forwards in gaussian.

        The max(1, channels // 8) guard clamps the embedded dim to 1.
        """
        layer = NonLocalAttention(attention_channels=4)
        # Guard yields embedded dim 1 (4 // 8 == 0 -> max(1, 0) == 1).
        assert layer.key_value_channels == 1

        inputs = keras.random.normal([2, 8, 8, 16], seed=7)
        output = layer(inputs)

        assert output.shape == (2, 8, 8, 16)
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(np.isfinite(output_np))

    def test_output_channels_param(self, sample_input):
        """output_channels controls the output last dim."""
        layer = NonLocalAttention(attention_channels=32, output_channels=16)
        output = layer(sample_input)
        assert output.shape == (2, 16, 16, 16)

    # =========================================================================
    # Serialization Tests
    # =========================================================================

    def test_keras_functional_round_trip(self, sample_input):
        """F19 regression: .keras Functional save/load forward-matches.

        This is the core F19 regression — the list/TrackedList kernel_size on
        reload broke this before the fix.
        """
        inputs = keras.Input(shape=(16, 16, 64))
        outputs = NonLocalAttention(
            attention_channels=32, name='non_local'
        )(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'non_local_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

        # kernel_size must survive the round-trip as a 2-tuple.
        loaded_layer = loaded_model.get_layer('non_local')
        assert loaded_layer.kernel_size == (7, 7)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_prediction),
            keras.ops.convert_to_numpy(loaded_prediction),
            rtol=1e-6, atol=1e-6,
            err_msg="Predictions differ after .keras round-trip",
        )

    def test_get_config_from_config_round_trip(self, sample_input):
        """get_config / from_config reconstructs a working layer."""
        layer = NonLocalAttention(
            attention_channels=64,
            kernel_size=[5, 5],
            attention_mode='dot_product',
            output_channels=16,
        )
        config = layer.get_config()

        reconstructed = NonLocalAttention.from_config(config)

        # Key config fields preserved.
        assert reconstructed.attention_channels == 64
        assert reconstructed.kernel_size == (5, 5)
        assert reconstructed.attention_mode == 'dot_product'
        assert reconstructed.output_channels == 16

        # Reconstructed layer forwards correctly.
        output = reconstructed(sample_input)
        assert output.shape == (2, 16, 16, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))

    # =========================================================================
    # Gradient Flow
    # =========================================================================

    def test_gradient_flow(self, sample_input):
        """All trainable weights receive non-None, finite gradients."""
        layer = NonLocalAttention(attention_channels=32)

        with tf.GradientTape() as tape:
            outputs = layer(sample_input, training=True)
            loss = tf.reduce_sum(outputs)

        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(grads) > 0
        assert all(g is not None for g in grads)
        for grad in grads:
            grad_np = keras.ops.convert_to_numpy(grad)
            assert np.all(np.isfinite(grad_np))

    # =========================================================================
    # Idempotent build (D-003)
    # =========================================================================

    def test_idempotent_build(self, sample_input):
        """A second build() call must be a no-op (no error)."""
        layer = NonLocalAttention(attention_channels=32)
        input_shape = (2, 16, 16, 64)

        layer.build(input_shape)
        assert layer.built

        # Second build must not raise (D-003 idempotency guard).
        layer.build(input_shape)
        assert layer.built

        # Layer still forwards correctly after the double build.
        output = layer(sample_input)
        assert output.shape == (2, 16, 16, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
