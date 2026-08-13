"""
Test suite for RepConditionalPosEnc (FastViT / MobileCLIP2 MCi positional encoding).

Covers initialization + stored config, constructor validation, forward shape over
multiple spatial sizes (including a NON-square one and a map SMALLER than the
7x7 kernel), `compute_output_shape` pre- and post-build, training-mode
determinism, gradient flow, a `.keras` VALUE round trip, and the mandated
behavioural pin `test_skip_connection_is_wired`.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.fastvit.rep_conditional_pos_enc import RepConditionalPosEnc


class TestRepConditionalPosEnc:
    """Comprehensive test suite for the RepConditionalPosEnc layer."""

    # ------------------------------------------------------------------
    # fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def basic_config(self):
        return {'dim': 16}

    @pytest.fixture
    def sample_input(self):
        rng = np.random.default_rng(4321)
        return rng.normal(size=(2, 8, 8, 16)).astype('float32')

    # ------------------------------------------------------------------
    # initialization / config
    # ------------------------------------------------------------------

    def test_initialization_stores_config(self, basic_config):
        layer = RepConditionalPosEnc(**basic_config)

        assert layer.dim == 16
        assert layer.dim_out == 16
        assert layer.spatial_shape == (7, 7)
        assert not layer.built

    def test_int_spatial_shape_is_normalized(self):
        layer = RepConditionalPosEnc(dim=8, spatial_shape=3)
        assert layer.spatial_shape == (3, 3)

    def test_asymmetric_spatial_shape(self):
        layer = RepConditionalPosEnc(dim=8, spatial_shape=(3, 5))
        assert layer.spatial_shape == (3, 5)

    def test_config_completeness(self):
        layer = RepConditionalPosEnc(dim=16, spatial_shape=(3, 5))
        config = layer.get_config()

        for key in (
                'dim', 'dim_out', 'spatial_shape', 'kernel_initializer',
                'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
        ):
            assert key in config, f"missing '{key}' in get_config()"

        rebuilt = RepConditionalPosEnc.from_config(config)
        assert rebuilt.dim == layer.dim
        assert rebuilt.dim_out == layer.dim_out
        assert rebuilt.spatial_shape == layer.spatial_shape

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            RepConditionalPosEnc(dim=0)

    def test_invalid_dim_out_nonpositive(self):
        with pytest.raises(ValueError, match="dim_out must be positive"):
            RepConditionalPosEnc(dim=8, dim_out=-1)

    def test_dim_out_must_equal_dim(self):
        with pytest.raises(ValueError, match="dim_out must equal dim"):
            RepConditionalPosEnc(dim=8, dim_out=16)

    def test_invalid_spatial_shape_even(self):
        with pytest.raises(ValueError, match="spatial_shape entries must be odd"):
            RepConditionalPosEnc(dim=8, spatial_shape=(4, 3))

    def test_invalid_spatial_shape_nonpositive(self):
        with pytest.raises(ValueError, match="spatial_shape entries must be positive"):
            RepConditionalPosEnc(dim=8, spatial_shape=(-3, 3))

    def test_invalid_spatial_shape_length(self):
        with pytest.raises(ValueError, match="exactly 2 entries"):
            RepConditionalPosEnc(dim=8, spatial_shape=(3, 3, 3))

    def test_invalid_spatial_shape_type(self):
        with pytest.raises(ValueError, match="spatial_shape entries must be ints"):
            RepConditionalPosEnc(dim=8, spatial_shape=(3.0, 3.0))

    def test_invalid_input_rank(self, basic_config):
        layer = RepConditionalPosEnc(**basic_config)
        with pytest.raises(ValueError, match="rank-4"):
            layer.build((None, 8, 16))

    def test_invalid_input_channels(self, basic_config):
        layer = RepConditionalPosEnc(**basic_config)
        with pytest.raises(ValueError, match="Input channel count must equal dim"):
            layer.build((None, 8, 8, 32))

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("height,width", [(8, 8), (12, 5), (4, 4), (2, 3)])
    @pytest.mark.parametrize("dim", [8, 32])
    def test_forward_shape(self, height, width, dim):
        """Includes maps SMALLER than the 7x7 kernel (the 5-stage 4x4 case)."""
        layer = RepConditionalPosEnc(dim=dim)
        x = np.random.default_rng(0).normal(
            size=(2, height, width, dim)).astype('float32')

        y = layer(x, training=False)

        assert tuple(y.shape) == (2, height, width, dim)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape_pre_build(self):
        layer = RepConditionalPosEnc(dim=16)
        assert not layer.built
        assert layer.compute_output_shape((None, 12, 5, 16)) == (None, 12, 5, 16)
        assert not layer.built

    def test_compute_output_shape_matches_forward(self, basic_config):
        layer = RepConditionalPosEnc(**basic_config)
        input_shape = (2, 12, 5, 16)
        predicted = layer.compute_output_shape(input_shape)

        x = np.random.default_rng(7).normal(size=input_shape).astype('float32')
        y = layer(x, training=False)

        assert tuple(y.shape) == tuple(predicted)

    def test_pos_conv_kernel_is_depthwise(self):
        channels = 16
        layer = RepConditionalPosEnc(dim=channels)
        layer.build((None, 8, 8, channels))

        assert tuple(layer.pos_conv.weights[0].shape) == (7, 7, channels, 1)
        # The reference uses a BIASED depthwise convolution.
        assert layer.pos_conv.use_bias is True
        assert len(layer.pos_conv.weights) == 2

    # ------------------------------------------------------------------
    # behavioural pin (RED-proven by dropping the `+ x`)
    # ------------------------------------------------------------------

    def test_skip_connection_is_wired(self, sample_input):
        """PIN: output == conv(x) + x, and is NOT conv(x) alone.

        `conv_out` is read straight off the sub-layer, so an injection that
        removes the `+ x` from `call()` moves only ONE side of the comparison.
        """
        layer = RepConditionalPosEnc(dim=16)
        out = ops.convert_to_numpy(layer(sample_input, training=False))
        conv_out = ops.convert_to_numpy(layer.pos_conv(sample_input))

        assert not np.allclose(out, conv_out, atol=1e-6), (
            "output equals the internal conv output alone: the `+ x` skip "
            "connection is missing"
        )
        np.testing.assert_allclose(
            out, conv_out + sample_input, atol=1e-6, rtol=0,
            err_msg="output is not exactly conv(x) + x"
        )

    # ------------------------------------------------------------------
    # training mode
    # ------------------------------------------------------------------

    def test_training_modes_are_identical(self, sample_input):
        """The layer holds no stochastic or statistic-bearing sub-layer."""
        layer = RepConditionalPosEnc(dim=16)
        a = ops.convert_to_numpy(layer(sample_input, training=True))
        b = ops.convert_to_numpy(layer(sample_input, training=False))
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------

    def test_gradients_flow_to_every_trainable_weight(self, sample_input):
        layer = RepConditionalPosEnc(dim=16)
        x = tf.convert_to_tensor(sample_input)

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = ops.mean(ops.square(y))

        grads = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) == 2
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
        outputs = RepConditionalPosEnc(dim=16)(inputs)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(sample_input, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rep_cpe.keras')
            model.save(path)
            loaded = keras.models.load_model(path)
            restored = ops.convert_to_numpy(loaded(sample_input, training=False))

        np.testing.assert_allclose(
            original, restored, atol=1e-6, rtol=0,
            err_msg="RepConditionalPosEnc values differ after a .keras round trip"
        )
