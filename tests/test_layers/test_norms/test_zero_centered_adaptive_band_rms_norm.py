"""
Comprehensive test suite for ZeroCenteredAdaptiveBandRMS.

Combines tests from AdaptiveBandRMS (adaptive log-RMS scaling) and
ZeroCenteredBandRMSNorm (zero-centering), plus a dedicated zero-mean
invariant test.

Covers:
- All tensor shapes (2D, 3D, 4D, 5D)
- Various axis configurations
- Serialization round-trip
- Edge cases and robustness
- Model integration
- Gradient flow and mixed precision
- Zero-mean property along normalization axis
"""

import pytest
import numpy as np
import tempfile
import os

import keras
import tensorflow as tf
from keras import ops
from dl_techniques.utils.logger import logger

from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
)


class TestZeroCenteredAdaptiveBandRMS:
    """Comprehensive test suite for ZeroCenteredAdaptiveBandRMS."""

    # =========================================================================
    # FIXTURES
    # =========================================================================

    @pytest.fixture
    def input_tensor_2d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 64).astype(np.float32)

    @pytest.fixture
    def input_tensor_3d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 20, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_4d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 16, 16, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_5d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(2, 8, 16, 16, 16).astype(np.float32)

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_defaults(self):
        layer = ZeroCenteredAdaptiveBandRMS()
        assert layer.max_band_width == 0.1
        assert layer.axis == -1
        assert layer.epsilon == 1e-7
        assert isinstance(layer.band_initializer, keras.initializers.Zeros)
        assert layer.band_regularizer is None
        assert layer.dense_layer is None
        logger.info("Default initialization test passed")

    def test_initialization_custom(self):
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()
        layer = ZeroCenteredAdaptiveBandRMS(
            max_band_width=0.3,
            axis=(1, 2),
            epsilon=1e-6,
            band_initializer=custom_initializer,
            band_regularizer=custom_regularizer,
        )
        assert layer.max_band_width == 0.3
        assert layer.axis == (1, 2)
        assert layer.epsilon == 1e-6
        assert isinstance(layer.band_initializer, keras.initializers.HeNormal)
        assert layer.band_regularizer == custom_regularizer
        logger.info("Custom initialization test passed")

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            ZeroCenteredAdaptiveBandRMS(max_band_width=-0.1)
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            ZeroCenteredAdaptiveBandRMS(max_band_width=1.5)
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            ZeroCenteredAdaptiveBandRMS(max_band_width=0.0)
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            ZeroCenteredAdaptiveBandRMS(max_band_width=1.0)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            ZeroCenteredAdaptiveBandRMS(epsilon=-1e-7)
        with pytest.raises(ValueError, match="epsilon must be positive"):
            ZeroCenteredAdaptiveBandRMS(epsilon=0)
        with pytest.raises(TypeError, match="axis must be int or tuple"):
            ZeroCenteredAdaptiveBandRMS(axis="invalid")
        with pytest.raises(TypeError, match="All elements in axis must be integers"):
            ZeroCenteredAdaptiveBandRMS(axis=(1, "x"))
        logger.info("Invalid parameters test passed")

    # =========================================================================
    # BUILD PROCESS TESTS
    # =========================================================================

    def test_build_process_2d(self, input_tensor_2d):
        layer = ZeroCenteredAdaptiveBandRMS()
        input_shape = input_tensor_2d.shape[1:]
        layer.build((None,) + input_shape)
        assert layer.built is True
        assert layer.dense_layer is not None
        assert layer.dense_layer.built is True
        assert layer.dense_layer.units == input_shape[-1]
        logger.info("2D build process test passed")

    def test_build_process_3d(self, input_tensor_3d):
        test_cases = [
            ("feature_wise", -1, input_tensor_3d.shape[-1]),
            ("sequence_wise", 1, input_tensor_3d.shape[1]),
            ("global", (1, 2), 1),
        ]
        for name, axis, expected_units in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis)
            layer.build((None,) + input_tensor_3d.shape[1:])
            assert layer.built is True
            assert layer.dense_layer.units == expected_units, name
            logger.info(f"3D build process test passed for {name}")

    def test_build_process_4d(self, input_tensor_4d):
        test_cases = [
            ("channel_wise", -1, input_tensor_4d.shape[-1]),
            ("spatial", (1, 2), input_tensor_4d.shape[1] * input_tensor_4d.shape[2]),
            ("global", (1, 2, 3), 1),
        ]
        for name, axis, expected_units in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis)
            layer.build((None,) + input_tensor_4d.shape[1:])
            assert layer.built is True
            assert layer.dense_layer.units == expected_units, name
            logger.info(f"4D build process test passed for {name}")

    def test_build_process_5d(self, input_tensor_5d):
        test_cases = [
            ("channel_wise", -1, input_tensor_5d.shape[-1]),
            ("spatial_3d", (1, 2, 3),
             input_tensor_5d.shape[1] * input_tensor_5d.shape[2] * input_tensor_5d.shape[3]),
            ("partial_spatial", (2, 3),
             input_tensor_5d.shape[2] * input_tensor_5d.shape[3]),
            ("global", (1, 2, 3, 4), 1),
        ]
        for name, axis, expected_units in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis)
            layer.build((None,) + input_tensor_5d.shape[1:])
            assert layer.built is True
            assert layer.dense_layer.units == expected_units
            logger.info(f"5D build process test passed for {name}")

    def test_invalid_axis_configurations(self):
        layer = ZeroCenteredAdaptiveBandRMS(axis=10)
        with pytest.raises(ValueError, match="axis .* is out of bounds"):
            layer.build((None, 32, 32, 64))
        layer = ZeroCenteredAdaptiveBandRMS(axis=(0, 1))
        with pytest.raises(ValueError, match="axis 0 .* cannot be normalized"):
            layer.build((None, 32, 32, 64))
        logger.info("Invalid axis configurations test passed")

    # =========================================================================
    # OUTPUT SHAPE TESTS
    # =========================================================================

    def test_output_shapes_all_dimensions(self):
        test_cases = [
            ("2D", (8, 64), -1),
            ("3D_feature", (4, 20, 32), -1),
            ("3D_sequence", (4, 20, 32), 1),
            ("3D_global", (4, 20, 32), (1, 2)),
            ("4D_channel", (4, 16, 16, 32), -1),
            ("4D_spatial", (4, 16, 16, 32), (1, 2)),
            ("4D_global", (4, 16, 16, 32), (1, 2, 3)),
            ("5D_channel", (2, 8, 16, 16, 16), -1),
            ("5D_spatial", (2, 8, 16, 16, 16), (1, 2, 3)),
        ]
        for name, input_shape, axis in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis)
            test_input = np.random.randn(*input_shape).astype(np.float32)
            output = layer(test_input)
            assert output.shape == test_input.shape, name
            assert layer.compute_output_shape(input_shape) == input_shape
            logger.info(f"Output shape test passed for {name}")

    # =========================================================================
    # FORWARD PASS TESTS
    # =========================================================================

    def test_forward_pass_numerical_properties(self):
        test_cases = [
            ("2D", (8, 64), -1),
            ("3D", (4, 20, 32), -1),
            ("4D", (4, 16, 16, 32), -1),
            ("5D", (2, 8, 16, 16, 16), -1),
        ]
        for name, input_shape, axis in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(max_band_width=0.2, axis=axis)
            test_input = np.random.randn(*input_shape).astype(np.float32)
            output = layer(test_input)
            assert not np.any(np.isnan(output.numpy())), name
            assert not np.any(np.isinf(output.numpy())), name

            # Output RMS along the normalization axis is in [1-alpha, 1].
            output_rms = ops.sqrt(
                ops.mean(ops.square(output), axis=axis, keepdims=False)
            )
            mean_rms = float(ops.mean(output_rms).numpy())
            # Allow slight slack for finite-sample noise.
            assert 0.78 <= mean_rms <= 1.02, (
                f"{name}: RMS {mean_rms} not in expected band ~[0.8, 1.0]"
            )
            logger.info(f"Forward pass numerical properties test passed for {name}")

    def test_zero_mean_property(self):
        """Output should have ~zero mean along the normalization axis."""
        test_cases = [
            ("2D", (8, 64), -1),
            ("3D", (4, 20, 32), -1),
            ("4D", (4, 16, 16, 32), -1),
            ("3D_seq_axis", (4, 20, 32), 1),
        ]
        for name, input_shape, axis in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis, max_band_width=0.1)
            # Use shifted input to verify centering removes the offset.
            test_input = (
                np.random.randn(*input_shape).astype(np.float32) + 5.0
            )
            output = layer(test_input).numpy()
            mean_along_axis = np.mean(output, axis=axis)
            np.testing.assert_allclose(
                mean_along_axis,
                np.zeros_like(mean_along_axis),
                atol=1e-5,
                err_msg=f"{name}: output not zero-mean along axis {axis}",
            )
            logger.info(f"Zero-mean property test passed for {name}")

    def test_forward_pass_different_axis_configurations(self, input_tensor_4d):
        for name, axis in [("single_axis", -1), ("tuple_axis", (1, 2)), ("global", (1, 2, 3))]:
            layer = ZeroCenteredAdaptiveBandRMS(axis=axis, max_band_width=0.15)
            output = layer(input_tensor_4d)
            assert output.shape == input_tensor_4d.shape, name
            assert not np.any(np.isnan(output.numpy())), name
            assert not np.any(np.isinf(output.numpy())), name
            logger.info(f"Different axis configurations test passed for {name}")

    def test_deterministic_behavior(self):
        controlled_input = np.ones((2, 8), dtype=np.float32) * 2.0
        layer = ZeroCenteredAdaptiveBandRMS(max_band_width=0.1, band_initializer="zeros")
        layer.build(controlled_input.shape)
        out1 = layer(controlled_input)
        out2 = layer(controlled_input)
        np.testing.assert_allclose(
            ops.convert_to_numpy(out1),
            ops.convert_to_numpy(out2),
            rtol=1e-6,
            atol=1e-6,
        )
        logger.info("Deterministic behavior test passed")

    # =========================================================================
    # SERIALIZATION TESTS
    # =========================================================================

    def test_serialization_cycle_2d(self, input_tensor_2d):
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        norm = ZeroCenteredAdaptiveBandRMS(
            max_band_width=0.25,
            epsilon=1e-6,
            band_regularizer=keras.regularizers.L2(1e-4),
            name="test_zc_adaptive_norm",
        )
        outputs = norm(inputs)
        model = keras.Model(inputs, outputs, name="test_model")
        original_output = model(input_tensor_2d)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(input_tensor_2d)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6,
                atol=1e-6,
            )
            loaded_norm = loaded_model.get_layer("test_zc_adaptive_norm")
            assert loaded_norm.max_band_width == 0.25
            assert loaded_norm.epsilon == 1e-6
            logger.info("2D serialization cycle test passed")

    def test_serialization_cycle_3d(self, input_tensor_3d):
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        norm = ZeroCenteredAdaptiveBandRMS(axis=(1, 2), max_band_width=0.2, name="zc_a")
        outputs = norm(inputs)
        model = keras.Model(inputs, outputs)
        original_output = model(input_tensor_3d)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "m.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(input_tensor_3d)
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6,
                atol=1e-6,
            )
            logger.info("3D serialization cycle test passed")

    def test_get_config_completeness(self):
        layer = ZeroCenteredAdaptiveBandRMS(
            max_band_width=0.3,
            axis=(1, 2),
            epsilon=1e-6,
            band_initializer="he_normal",
            band_regularizer=keras.regularizers.L2(1e-4),
        )
        config = layer.get_config()
        for key in ("max_band_width", "axis", "epsilon", "band_initializer", "band_regularizer"):
            assert key in config
        assert config["max_band_width"] == 0.3
        assert config["axis"] == (1, 2)
        assert config["epsilon"] == 1e-6
        logger.info("Config completeness test passed")

    # =========================================================================
    # MODEL INTEGRATION TESTS
    # =========================================================================

    def test_model_integration_transformer_style(self, input_tensor_3d):
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        x = keras.layers.Dense(64)(inputs)
        x = ZeroCenteredAdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dense(64)(x)
        x = ZeroCenteredAdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        y_pred = model.predict(input_tensor_3d, verbose=0)
        assert y_pred.shape == (input_tensor_3d.shape[0], 10)
        np.testing.assert_allclose(np.sum(y_pred, axis=-1), 1.0, rtol=1e-5)
        logger.info("Transformer-style integration test passed")

    def test_model_integration_cnn_multiaxis(self, input_tensor_4d):
        inputs = keras.Input(shape=input_tensor_4d.shape[1:])
        x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
        x = ZeroCenteredAdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = ZeroCenteredAdaptiveBandRMS(axis=(1, 2), max_band_width=0.15)(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = ZeroCenteredAdaptiveBandRMS(axis=(1, 2, 3), max_band_width=0.2)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = ZeroCenteredAdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        outputs = keras.layers.Dense(5, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        y_pred = model.predict(input_tensor_4d, verbose=0)
        assert y_pred.shape == (input_tensor_4d.shape[0], 5)
        logger.info("CNN multi-axis integration test passed")

    # =========================================================================
    # FACTORY INTEGRATION
    # =========================================================================

    def test_factory_creation(self):
        from dl_techniques.layers.norms import create_normalization_layer

        layer = create_normalization_layer(
            "zero_centered_adaptive_band_rms_norm",
            max_band_width=0.15,
            name="factory_norm",
        )
        assert isinstance(layer, ZeroCenteredAdaptiveBandRMS)
        assert layer.max_band_width == 0.15
        assert layer.name == "factory_norm"
        logger.info("Factory creation test passed")

    def test_factory_validation(self):
        from dl_techniques.layers.norms.factory import validate_normalization_config

        assert validate_normalization_config(
            "zero_centered_adaptive_band_rms_norm",
            max_band_width=0.1,
            epsilon=1e-7,
        )
        with pytest.raises(ValueError):
            validate_normalization_config(
                "zero_centered_adaptive_band_rms_norm",
                max_band_width=-1.0,
            )
        logger.info("Factory validation test passed")

    # =========================================================================
    # EDGE CASES AND ROBUSTNESS
    # =========================================================================

    def test_extreme_input_values(self):
        test_cases = [
            ("zeros", np.zeros((4, 32), dtype=np.float32)),
            ("very_small", np.ones((4, 32), dtype=np.float32) * 1e-10),
            ("very_large", np.ones((4, 32), dtype=np.float32) * 1e5),
            ("mixed_extreme", np.concatenate([
                np.ones((4, 16), dtype=np.float32) * 1e-8,
                np.ones((4, 16), dtype=np.float32) * 1e8,
            ], axis=-1)),
        ]
        for name, test_input in test_cases:
            layer = ZeroCenteredAdaptiveBandRMS(max_band_width=0.1)
            output = layer(test_input)
            assert not np.any(np.isnan(output.numpy())), name
            assert not np.any(np.isinf(output.numpy())), name
            logger.info(f"Extreme values test passed for {name}")

    def test_variable_batch_sizes(self):
        layer = ZeroCenteredAdaptiveBandRMS(axis=-1)
        for batch_size in [1, 2, 8, 16, 32]:
            test_input = np.random.randn(batch_size, 32).astype(np.float32)
            output = layer(test_input)
            assert output.shape == test_input.shape
            logger.info(f"Variable batch size {batch_size} test passed")

    def test_training_vs_inference_mode(self, input_tensor_2d):
        layer = ZeroCenteredAdaptiveBandRMS(
            band_regularizer=keras.regularizers.L2(0.01)
        )
        train_output = layer(input_tensor_2d, training=True)
        inference_output = layer(input_tensor_2d, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(train_output),
            ops.convert_to_numpy(inference_output),
            rtol=1e-6,
            atol=1e-6,
        )
        logger.info("Training vs inference mode test passed")

    def test_regularization_integration(self, input_tensor_2d):
        layer = ZeroCenteredAdaptiveBandRMS(
            band_regularizer=keras.regularizers.L2(0.1)
        )
        layer.build(input_tensor_2d.shape)
        _ = layer(input_tensor_2d)
        if layer.dense_layer is not None:
            weights = layer.dense_layer.get_weights()
            if len(weights) > 0:
                weights[0] = np.ones_like(weights[0]) * 0.1
                layer.dense_layer.set_weights(weights)
                _ = layer(input_tensor_2d)
                total_loss = sum(layer.losses)
                assert total_loss > 0
        logger.info("Regularization integration test passed")

    # =========================================================================
    # GRADIENT FLOW + MIXED PRECISION
    # =========================================================================

    def test_gradient_flow(self, input_tensor_2d):
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = ZeroCenteredAdaptiveBandRMS(max_band_width=0.2)(inputs)
        outputs = keras.layers.Dense(5)(x)
        model = keras.Model(inputs, outputs)
        tensor_input = tf.convert_to_tensor(input_tensor_2d)
        with tf.GradientTape() as tape:
            tape.watch(tensor_input)
            predictions = model(tensor_input)
            loss = ops.mean(ops.square(predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        for i, grad in enumerate(gradients):
            assert grad is not None, f"Gradient {i} is None"
            assert ops.all(ops.isfinite(grad)), f"Gradient {i} non-finite"
        logger.info("Gradient flow test passed")

    def test_mixed_precision_compatibility(self, input_tensor_2d):
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            layer = ZeroCenteredAdaptiveBandRMS(max_band_width=0.2)
            input_fp16 = ops.cast(input_tensor_2d, "float16")
            output = layer(input_fp16)
            assert output.dtype == input_fp16.dtype
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))
            logger.info("Mixed precision compatibility test passed")
        finally:
            keras.mixed_precision.set_global_policy("float32")


# =========================================================================
# INTEGRATION TESTS WITH OTHER LAYERS
# =========================================================================


class TestZeroCenteredAdaptiveBandRMSIntegration:
    """Integration tests with sibling layers."""

    def test_integration_with_standard_normalization(self):
        inputs = keras.Input(shape=(64,))
        x1 = keras.layers.Dense(128)(inputs)
        x1 = keras.layers.LayerNormalization()(x1)
        x1 = keras.layers.ReLU()(x1)
        x1 = keras.layers.Dense(32)(x1)
        x2 = keras.layers.Dense(128)(inputs)
        x2 = ZeroCenteredAdaptiveBandRMS(axis=-1)(x2)
        x2 = keras.layers.ReLU()(x2)
        x2 = keras.layers.Dense(32)(x2)
        model1 = keras.Model(inputs, x1)
        model2 = keras.Model(inputs, x2)
        test_input = np.random.randn(8, 64).astype(np.float32)
        out1 = model1.predict(test_input, verbose=0)
        out2 = model2.predict(test_input, verbose=0)
        assert out1.shape == out2.shape
        assert not np.any(np.isnan(out1))
        assert not np.any(np.isnan(out2))
        logger.info("Integration with standard normalization test passed")

    def test_integration_with_attention(self):
        seq_len, embed_dim = 20, 64
        inputs = keras.Input(shape=(seq_len, embed_dim))
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8
        )(inputs, inputs)
        x = ZeroCenteredAdaptiveBandRMS(axis=-1)(inputs + attn_output)
        ffn = keras.layers.Dense(embed_dim * 4, activation="relu")(x)
        ffn = keras.layers.Dense(embed_dim)(ffn)
        outputs = ZeroCenteredAdaptiveBandRMS(axis=-1)(x + ffn)
        model = keras.Model(inputs, outputs)
        test_input = np.random.randn(4, seq_len, embed_dim).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output))
        logger.info("Integration with attention test passed")


# =========================================================================
# BENCHMARK TESTS
# =========================================================================


class TestZeroCenteredAdaptiveBandRMSBenchmark:
    """Lightweight benchmark tests."""

    def test_forward_pass_speed(self):
        layer = ZeroCenteredAdaptiveBandRMS()
        test_input = np.random.randn(32, 128).astype(np.float32)
        layer.build(test_input.shape)
        result = layer(test_input)
        assert result.shape == test_input.shape

    def test_build_time(self):
        layer = ZeroCenteredAdaptiveBandRMS()
        layer.build((None, 256))
        assert layer.built is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings", "-x"])
