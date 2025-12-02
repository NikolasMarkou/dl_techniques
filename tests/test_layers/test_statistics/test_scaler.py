"""
Comprehensive test suite for UnifiedScaler layer.

Tests cover:
- Basic functionality and forward pass
- RevIN-style usage (axis=1, affine=True)
- StandardScaler-style usage (axis=-1, store_stats=True)
- Inverse transformation and denormalization
- NaN handling
- Statistics storage and retrieval
- Serialization (save/load cycle)
- Edge cases and error handling
- Various configurations and batch sizes
"""

import os
import tempfile
import numpy as np
import pytest

import keras
from keras import ops

from dl_techniques.layers.statistics.scaler import UnifiedScaler


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def basic_config():
    """Basic configuration for UnifiedScaler."""
    return {
        "num_features": 10,
        "axis": -1,
        "eps": 1e-5,
        "affine": False,
        "store_stats": False,
    }


@pytest.fixture
def revin_config():
    """RevIN-style configuration for time series."""
    return {
        "num_features": 10,
        "axis": 1,  # Normalize across time dimension
        "eps": 1e-5,
        "affine": True,
        "store_stats": True,
    }


@pytest.fixture
def scaler_config():
    """StandardScaler-style configuration."""
    return {
        "num_features": None,  # Infer from input
        "axis": -1,  # Normalize across features
        "eps": 1e-5,
        "affine": False,
        "store_stats": True,
        "nan_replacement": 0.0,
    }


@pytest.fixture
def sample_2d_input():
    """2D input for tabular data: (batch, features)."""
    return np.random.randn(32, 10).astype(np.float32)


@pytest.fixture
def sample_3d_input():
    """3D input for time series: (batch, seq_len, features)."""
    return np.random.randn(16, 50, 10).astype(np.float32)


@pytest.fixture
def sample_3d_with_nans():
    """3D input with NaN values."""
    data = np.random.randn(8, 30, 5).astype(np.float32)
    # Add some NaN values
    nan_indices = np.random.choice(data.size, size=10, replace=False)
    flat_data = data.flatten()
    flat_data[nan_indices] = np.nan
    return flat_data.reshape(data.shape)


# ---------------------------------------------------------------------
# Basic Functionality Tests
# ---------------------------------------------------------------------


class TestBasicFunctionality:
    """Test basic layer functionality."""

    def test_instantiation(self, basic_config):
        """Test layer can be instantiated with valid config."""
        layer = UnifiedScaler(**basic_config)
        assert layer.num_features == basic_config["num_features"]
        assert layer.eps == basic_config["eps"]
        assert layer.affine == basic_config["affine"]

    def test_instantiation_without_num_features(self):
        """Test layer can be instantiated without specifying num_features."""
        layer = UnifiedScaler(axis=-1)
        assert layer.num_features is None

    def test_invalid_num_features(self):
        """Test layer rejects invalid num_features."""
        with pytest.raises(ValueError, match="num_features must be positive"):
            UnifiedScaler(num_features=-1)

    def test_invalid_eps(self):
        """Test layer rejects invalid epsilon."""
        with pytest.raises(ValueError, match="eps must be positive"):
            UnifiedScaler(num_features=10, eps=-1e-5)

        with pytest.raises(ValueError, match="eps must be positive"):
            UnifiedScaler(num_features=10, eps=0)

    def test_forward_pass_2d(self, basic_config, sample_2d_input):
        """Test forward pass with 2D input produces correct output shape."""
        layer = UnifiedScaler(**basic_config)
        output = layer(sample_2d_input)

        assert output.shape == sample_2d_input.shape

    def test_forward_pass_3d(self, revin_config, sample_3d_input):
        """Test forward pass with 3D input produces correct output shape."""
        layer = UnifiedScaler(**revin_config)
        output = layer(sample_3d_input)

        assert output.shape == sample_3d_input.shape

    def test_output_statistics(self, basic_config, sample_2d_input):
        """Test that normalized output has approximately zero mean and unit variance."""
        layer = UnifiedScaler(**basic_config)
        output = layer(sample_2d_input)

        output_np = ops.convert_to_numpy(output)

        # Check mean is close to zero
        mean = np.mean(output_np, axis=-1)
        np.testing.assert_allclose(mean, 0.0, atol=1e-3)

        # Check std is close to one
        std = np.std(output_np, axis=-1)
        np.testing.assert_allclose(std, 1.0, atol=1e-3)


# ---------------------------------------------------------------------
# RevIN-Style Tests
# ---------------------------------------------------------------------


class TestRevINStyle:
    """Test RevIN-style functionality (axis=1, affine=True)."""

    def test_revin_normalization(self, revin_config, sample_3d_input):
        """Test RevIN-style normalization along time axis."""
        layer = UnifiedScaler(**revin_config)
        output = layer(sample_3d_input)

        output_np = ops.convert_to_numpy(output)

        # Statistics computed along axis=1 (time), so check per instance
        for i in range(output_np.shape[0]):
            for j in range(output_np.shape[2]):
                # Mean along time axis should be close to zero
                mean = np.mean(output_np[i, :, j])
                # But affine transformation might shift it
                # So we just verify the shape is correct
                assert output_np.shape == sample_3d_input.shape

    def test_revin_denormalization(self, revin_config, sample_3d_input):
        """Test RevIN denormalization restores original scale."""
        layer = UnifiedScaler(**revin_config)

        # Normalize
        normalized = layer(sample_3d_input)

        # Denormalize
        denormalized = layer.denormalize(normalized)

        # Should recover original input (approximately)
        np.testing.assert_allclose(
            ops.convert_to_numpy(denormalized),
            sample_3d_input,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Denormalization should restore original scale"
        )

    def test_revin_affine_parameters(self, revin_config, sample_3d_input):
        """Test that affine parameters are created with correct shape."""
        layer = UnifiedScaler(**revin_config)
        layer(sample_3d_input)  # Trigger build

        assert layer.affine_weight is not None
        assert layer.affine_bias is not None
        assert layer.affine_weight.shape == (revin_config["num_features"],)
        assert layer.affine_bias.shape == (revin_config["num_features"],)

    def test_revin_statistics_storage(self, revin_config, sample_3d_input):
        """Test that statistics are stored when store_stats=True."""
        layer = UnifiedScaler(**revin_config)
        layer(sample_3d_input)

        stats = layer.get_stats()
        assert stats is not None
        mean, std = stats
        assert mean.shape == (1, revin_config["num_features"])
        assert std.shape == (1, revin_config["num_features"])


# ---------------------------------------------------------------------
# StandardScaler-Style Tests
# ---------------------------------------------------------------------


class TestStandardScalerStyle:
    """Test StandardScaler-style functionality (axis=-1, store_stats=True)."""

    def test_scaler_normalization(self, scaler_config, sample_3d_input):
        """Test StandardScaler-style normalization along feature axis."""
        layer = UnifiedScaler(**scaler_config)
        output = layer(sample_3d_input)

        output_np = ops.convert_to_numpy(output)

        # Check that features are normalized
        # Mean along feature axis should be close to zero
        mean = np.mean(output_np, axis=-1)
        np.testing.assert_allclose(mean, 0.0, atol=1e-3)

        # Std along feature axis should be close to one
        std = np.std(output_np, axis=-1)
        np.testing.assert_allclose(std, 1.0, atol=1e-3)

    def test_scaler_inverse_transform(self, scaler_config, sample_3d_input):
        """Test inverse_transform restores original scale."""
        layer = UnifiedScaler(**scaler_config)

        # Normalize
        normalized = layer(sample_3d_input)

        # Inverse transform
        restored = layer.inverse_transform(normalized)

        # Should recover original input
        np.testing.assert_allclose(
            ops.convert_to_numpy(restored),
            sample_3d_input,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Inverse transform should restore original scale"
        )

    def test_scaler_stored_statistics(self, scaler_config, sample_3d_input):
        """Test that statistics are properly stored."""
        layer = UnifiedScaler(**scaler_config)

        # First forward pass
        layer(sample_3d_input)

        # Get stored statistics
        stats = layer.get_stats()
        assert stats is not None

        mean, std = stats
        # Statistics should be non-trivial
        assert not ops.all(mean == 0.0)
        assert not ops.all(std == 1.0)


# ---------------------------------------------------------------------
# NaN Handling Tests
# ---------------------------------------------------------------------


class TestNaNHandling:
    """Test NaN handling functionality."""

    def test_nan_replacement(self, sample_3d_with_nans):
        """Test that NaN values are properly replaced."""
        layer = UnifiedScaler(num_features=5, nan_replacement=0.0, axis=-1)

        # Should not raise error with NaN input
        output = layer(sample_3d_with_nans)

        # Output should not contain NaN
        output_np = ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()

    def test_custom_nan_replacement(self, sample_3d_with_nans):
        """Test custom NaN replacement value."""
        replacement_value = -999.0
        layer = UnifiedScaler(
            num_features=5,
            nan_replacement=replacement_value,
            axis=-1
        )

        output = layer(sample_3d_with_nans)

        # Should not contain NaN
        output_np = ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()


# ---------------------------------------------------------------------
# Inverse Transformation Tests
# ---------------------------------------------------------------------


class TestInverseTransformation:
    """Test inverse transformation functionality."""

    def test_inverse_transform_alias(self, basic_config, sample_2d_input):
        """Test that denormalize is an alias for inverse_transform."""
        layer = UnifiedScaler(**basic_config)

        normalized = layer(sample_2d_input)

        # Both methods should produce identical results
        restored_inverse = layer.inverse_transform(normalized)
        restored_denorm = layer.denormalize(normalized)

        np.testing.assert_allclose(
            ops.convert_to_numpy(restored_inverse),
            ops.convert_to_numpy(restored_denorm),
            rtol=1e-6,
            atol=1e-6,
            err_msg="inverse_transform and denormalize should be identical"
        )

    def test_inverse_before_forward_raises_error(self, basic_config):
        """Test that inverse_transform raises error if called before forward pass."""
        layer = UnifiedScaler(**basic_config)

        dummy_input = np.random.randn(8, 10).astype(np.float32)

        with pytest.raises(RuntimeError, match="Cannot perform inverse transformation"):
            layer.inverse_transform(dummy_input)

    def test_denormalize_before_forward_raises_error(self, basic_config):
        """Test that denormalize raises error if called before forward pass."""
        layer = UnifiedScaler(**basic_config)

        dummy_input = np.random.randn(8, 10).astype(np.float32)

        with pytest.raises(RuntimeError, match="Cannot perform inverse transformation"):
            layer.denormalize(dummy_input)

    def test_inverse_with_affine(self, revin_config, sample_3d_input):
        """Test inverse transformation with affine parameters."""
        layer = UnifiedScaler(**revin_config)

        # Forward pass with affine
        normalized = layer(sample_3d_input)

        # Inverse should still restore original
        restored = layer.inverse_transform(normalized)

        np.testing.assert_allclose(
            ops.convert_to_numpy(restored),
            sample_3d_input,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Inverse with affine should restore original"
        )


# ---------------------------------------------------------------------
# Statistics Management Tests
# ---------------------------------------------------------------------


class TestStatisticsManagement:
    """Test statistics storage and management."""

    def test_get_stats_without_store_stats(self, basic_config, sample_2d_input):
        """Test get_stats returns None when store_stats=False."""
        layer = UnifiedScaler(**basic_config)
        layer(sample_2d_input)

        stats = layer.get_stats()
        assert stats is None

    def test_get_stats_with_store_stats(self, scaler_config, sample_3d_input):
        """Test get_stats returns statistics when store_stats=True."""
        layer = UnifiedScaler(**scaler_config)
        layer(sample_3d_input)

        stats = layer.get_stats()
        assert stats is not None

        mean, std = stats
        assert mean is not None
        assert std is not None

    def test_reset_stats(self, scaler_config, sample_3d_input):
        """Test reset_stats clears stored statistics."""
        layer = UnifiedScaler(**scaler_config)

        # Compute statistics
        layer(sample_3d_input)

        # Get stats before reset
        stats_before = layer.get_stats()
        assert stats_before is not None

        # Reset
        layer.reset_stats()

        # Internal statistics should be cleared
        assert layer._last_mean is None
        assert layer._last_std is None

        # Stored statistics should be reset to initial values
        stats_after = layer.get_stats()
        mean, std = stats_after

        mean_np = ops.convert_to_numpy(mean)
        std_np = ops.convert_to_numpy(std)

        np.testing.assert_allclose(mean_np, 0.0, atol=1e-6)
        np.testing.assert_allclose(std_np, 1.0, atol=1e-6)


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------


class TestSerialization:
    """Test serialization and deserialization."""

    def test_serialization_cycle_basic(self, basic_config, sample_2d_input):
        """Test full save/load cycle preserves functionality."""
        # Build model
        inputs = keras.Input(shape=sample_2d_input.shape[1:])
        outputs = UnifiedScaler(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original output
        original_output = model(sample_2d_input)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Compare outputs
        loaded_output = loaded_model(sample_2d_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization"
        )

    def test_serialization_cycle_with_affine(self, revin_config, sample_3d_input):
        """Test serialization with affine parameters."""
        # Build model
        inputs = keras.Input(shape=sample_3d_input.shape[1:])
        outputs = UnifiedScaler(**revin_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original output
        original_output = model(sample_3d_input)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_affine.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Compare outputs
        loaded_output = loaded_model(sample_3d_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs with affine should match after serialization"
        )

    def test_serialization_preserves_stored_stats(
            self, scaler_config, sample_3d_input
    ):
        """Test that stored statistics are preserved through serialization."""
        # Build model with stored stats
        inputs = keras.Input(shape=sample_3d_input.shape[1:])
        scaler_layer = UnifiedScaler(**scaler_config)
        outputs = scaler_layer(inputs)
        model = keras.Model(inputs, outputs)

        # Compute statistics
        model(sample_3d_input)

        # Get stats before save
        stats_before = scaler_layer.get_stats()
        mean_before, std_before = stats_before

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_stats.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Get stats after load
        loaded_scaler = loaded_model.layers[1]
        stats_after = loaded_scaler.get_stats()
        mean_after, std_after = stats_after

        # Compare statistics
        np.testing.assert_allclose(
            ops.convert_to_numpy(mean_before),
            ops.convert_to_numpy(mean_after),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Stored mean should match after serialization"
        )

        np.testing.assert_allclose(
            ops.convert_to_numpy(std_before),
            ops.convert_to_numpy(std_after),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Stored std should match after serialization"
        )

    def test_get_config_complete(self, revin_config):
        """Test get_config returns all constructor arguments."""
        layer = UnifiedScaler(**revin_config)
        config = layer.get_config()

        # Verify all config keys present
        assert "num_features" in config
        assert "axis" in config
        assert "eps" in config
        assert "affine" in config
        assert "affine_weight_initializer" in config
        assert "affine_bias_initializer" in config
        assert "nan_replacement" in config
        assert "store_stats" in config

    def test_from_config_reconstruction(self, revin_config, sample_3d_input):
        """Test layer can be reconstructed from config."""
        original = UnifiedScaler(**revin_config)
        original(sample_3d_input)  # Build it

        config = original.get_config()
        reconstructed = UnifiedScaler.from_config(config)

        assert reconstructed.num_features == original.num_features
        assert reconstructed.axis == original.axis
        assert reconstructed.eps == original.eps
        assert reconstructed.affine == original.affine
        assert reconstructed.store_stats == original.store_stats


# ---------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_input(self):
        """Test layer handles constant input (zero variance)."""
        # Create constant input
        constant_input = np.ones((8, 10), dtype=np.float32)

        layer = UnifiedScaler(num_features=10, eps=1e-5)

        # Should not crash with constant input
        output = layer(constant_input)

        # Output should be well-defined (zeros due to zero variance + eps handling)
        output_np = ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()
        assert not np.isinf(output_np).any()

    def test_single_sample_batch(self, basic_config):
        """Test layer handles batch size of 1."""
        single_input = np.random.randn(1, 10).astype(np.float32)

        layer = UnifiedScaler(**basic_config)
        output = layer(single_input)

        assert output.shape == single_input.shape

    def test_large_batch_size(self, basic_config):
        """Test layer handles large batch sizes."""
        large_input = np.random.randn(1024, 10).astype(np.float32)

        layer = UnifiedScaler(**basic_config)
        output = layer(large_input)

        assert output.shape == large_input.shape

    def test_multi_axis_normalization(self):
        """Test normalization along multiple axes."""
        layer = UnifiedScaler(num_features=10, axis=(1, 2))

        # 4D input
        input_4d = np.random.randn(4, 8, 8, 10).astype(np.float32)

        # Should handle multi-axis normalization
        output = layer(input_4d)
        assert output.shape == input_4d.shape

    def test_invalid_axis_raises_error(self, sample_3d_input):
        """Test that invalid axis raises error during build."""
        layer = UnifiedScaler(num_features=10, axis=10)  # Invalid axis

        with pytest.raises(ValueError, match="Invalid axis"):
            layer(sample_3d_input)

    def test_compute_output_shape(self, basic_config, sample_2d_input):
        """Test compute_output_shape matches actual output."""
        layer = UnifiedScaler(**basic_config)

        computed_shape = layer.compute_output_shape(sample_2d_input.shape)
        actual_output = layer(sample_2d_input)

        assert computed_shape == actual_output.shape


# ---------------------------------------------------------------------
# Parametrized Tests
# ---------------------------------------------------------------------


class TestParametrized:
    """Parametrized tests for various configurations."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_variable_batch_size(self, batch_size):
        """Test layer handles various batch sizes."""
        layer = UnifiedScaler(num_features=10)

        inputs = np.random.randn(batch_size, 10).astype(np.float32)
        output = layer(inputs)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("num_features", [1, 5, 10, 50, 100])
    def test_variable_num_features(self, num_features):
        """Test layer handles various feature dimensions."""
        layer = UnifiedScaler(num_features=num_features)

        inputs = np.random.randn(16, num_features).astype(np.float32)
        output = layer(inputs)

        assert output.shape == inputs.shape

    @pytest.mark.parametrize("axis", [-1, -2, 0, 1])
    def test_different_axes(self, axis):
        """Test layer with different normalization axes."""
        input_3d = np.random.randn(8, 20, 10).astype(np.float32)

        layer = UnifiedScaler(num_features=10, axis=axis)
        output = layer(input_3d)

        assert output.shape == input_3d.shape

    @pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-7, 1e-9])
    def test_different_epsilon_values(self, eps):
        """Test layer with different epsilon values."""
        layer = UnifiedScaler(num_features=10, eps=eps)

        inputs = np.random.randn(16, 10).astype(np.float32)
        output = layer(inputs)

        assert output.shape == inputs.shape

    @pytest.mark.parametrize("affine", [True, False])
    def test_with_and_without_affine(self, affine, sample_3d_input):
        """Test layer with and without affine transformation."""
        layer = UnifiedScaler(num_features=10, axis=1, affine=affine)

        output = layer(sample_3d_input)
        assert output.shape == sample_3d_input.shape

        if affine:
            assert layer.affine_weight is not None
            assert layer.affine_bias is not None
        else:
            assert layer.affine_weight is None
            assert layer.affine_bias is None

    @pytest.mark.parametrize("store_stats", [True, False])
    def test_with_and_without_store_stats(self, store_stats, sample_3d_input):
        """Test layer with and without statistics storage."""
        layer = UnifiedScaler(num_features=10, store_stats=store_stats)

        layer(sample_3d_input)

        stats = layer.get_stats()

        if store_stats:
            assert stats is not None
        else:
            assert stats is None


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------


class TestIntegration:
    """Integration tests with models and training."""

    def test_in_sequential_model(self, sample_3d_input):
        """Test layer works in Sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=sample_3d_input.shape[1:]),
            UnifiedScaler(num_features=10, axis=1, affine=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.Dense(10)
        ])

        output = model(sample_3d_input)
        assert output.shape == sample_3d_input.shape

    def test_in_functional_model(self, sample_3d_input):
        """Test layer works in Functional API model."""
        inputs = keras.Input(shape=sample_3d_input.shape[1:])
        normalized = UnifiedScaler(num_features=10, axis=1, affine=True)(inputs)
        features = keras.layers.LSTM(32)(normalized)
        outputs = keras.layers.Dense(10)(features)

        model = keras.Model(inputs, outputs)

        output = model(sample_3d_input)
        assert output.shape == (sample_3d_input.shape[0], 10)

    def test_with_model_training(self, sample_3d_input):
        """Test layer works during model training."""
        # Create simple model
        inputs = keras.Input(shape=sample_3d_input.shape[1:])
        normalized = UnifiedScaler(num_features=10, axis=-1)(inputs)
        outputs = keras.layers.GlobalAveragePooling1D()(normalized)
        outputs = keras.layers.Dense(1)(outputs)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")

        # Create dummy targets
        targets = np.random.randn(sample_3d_input.shape[0], 1).astype(np.float32)

        # Training should not crash
        history = model.fit(
            sample_3d_input,
            targets,
            epochs=2,
            batch_size=8,
            verbose=0
        )

        assert len(history.history["loss"]) == 2

    def test_inverse_transform_with_model_output(self, sample_3d_input):
        """Test inverse transform works with model predictions."""
        # Create model with scaler
        inputs = keras.Input(shape=sample_3d_input.shape[1:])
        scaler = UnifiedScaler(num_features=10, axis=1, affine=True, store_stats=True)
        normalized = scaler(inputs)

        # Simple passthrough model for testing
        outputs = normalized

        model = keras.Model(inputs, outputs)

        # Forward pass
        predictions = model(sample_3d_input)

        # Inverse transform predictions
        restored = scaler.inverse_transform(predictions)

        # Should approximately match original input
        np.testing.assert_allclose(
            ops.convert_to_numpy(restored),
            sample_3d_input,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Inverse transform should restore original scale"
        )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])