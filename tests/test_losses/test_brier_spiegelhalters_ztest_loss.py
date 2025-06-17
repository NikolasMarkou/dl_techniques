"""
Tests for Calibration Loss Functions

This module contains comprehensive tests for the calibration loss functions:
- BrierScoreLoss
- SpiegelhalterZLoss
- CombinedCalibrationLoss
- BrierScoreMetric
- SpiegelhalterZMetric

Tests cover initialization, forward pass, serialization, edge cases, and integration scenarios.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Any, Dict, List, Tuple, Optional

from dl_techniques.utils.logger import logger

# Import the classes to test (adjust import path as needed)
from dl_techniques.losses.brier_spiegelhalters_ztest_loss import (
    BrierScoreLoss,
    SpiegelhalterZLoss,
    CombinedCalibrationLoss,
    BrierScoreMetric,
    SpiegelhalterZMetric
)


class TestBrierScoreLoss:
    """Test suite for BrierScoreLoss implementation."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample binary classification data for testing.

        Returns:
            Tuple of (y_true, y_pred) tensors.
        """
        batch_size = 32
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
        return y_true, y_pred

    @pytest.fixture
    def logits_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample data with logits for testing.

        Returns:
            Tuple of (y_true, logits) tensors.
        """
        batch_size = 32
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        logits = ops.cast(np.random.normal(0, 2, size=(batch_size, 1)), dtype="float32")
        return y_true, logits

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        loss = BrierScoreLoss()

        assert loss.from_logits is False
        assert loss.reduction == 'sum_over_batch_size'
        assert loss.name == "brier_score_loss"

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        loss = BrierScoreLoss(
            from_logits=True,
            reduction='none',
            name='custom_brier'
        )

        assert loss.from_logits is True
        assert loss.reduction == 'none'
        assert loss.name == "custom_brier"

    def test_forward_pass_probabilities(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test forward pass with probability inputs."""
        y_true, y_pred = sample_data
        loss = BrierScoreLoss()

        result = loss(y_true, y_pred)

        # Check result is valid
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))
        assert ops.all(result >= 0.0)  # Brier score is always non-negative

    def test_forward_pass_logits(self, logits_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test forward pass with logit inputs."""
        y_true, logits = logits_data
        loss = BrierScoreLoss(from_logits=True)

        result = loss(y_true, logits)

        # Check result is valid
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))
        assert ops.all(result >= 0.0)

    def test_perfect_predictions(self) -> None:
        """Test loss with perfect predictions."""
        y_true = ops.cast([[1.0], [0.0], [1.0], [0.0]], dtype="float32")
        y_pred = ops.cast([[1.0], [0.0], [1.0], [0.0]], dtype="float32")

        loss = BrierScoreLoss()
        result = loss(y_true, y_pred)

        # Perfect predictions should give loss close to 0
        assert ops.abs(result) < 1e-6

    def test_worst_predictions(self) -> None:
        """Test loss with worst possible predictions."""
        y_true = ops.cast([[1.0], [0.0], [1.0], [0.0]], dtype="float32")
        y_pred = ops.cast([[0.0], [1.0], [0.0], [1.0]], dtype="float32")

        loss = BrierScoreLoss()
        result = loss(y_true, y_pred)

        # Worst predictions should give loss of 1.0
        assert ops.abs(result - 1.0) < 1e-6

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the loss."""
        original_loss = BrierScoreLoss(from_logits=True, reduction='none')

        # Get config and recreate
        config = original_loss.get_config()
        recreated_loss = BrierScoreLoss.from_config(config)

        # Check configuration matches
        assert recreated_loss.from_logits == original_loss.from_logits
        assert recreated_loss.reduction == original_loss.reduction

    def test_model_integration(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test the loss in a model context."""
        y_true, y_pred = sample_data

        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu", input_shape=(1,)),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        # Compile with Brier score loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=BrierScoreLoss()
        )

        # Test that it doesn't crash
        x_dummy = ops.ones((32, 1))
        loss_value = model.evaluate(x_dummy, y_true, verbose=0)
        assert not np.isnan(loss_value)


class TestSpiegelhalterZLoss:
    """Test suite for SpiegelhalterZLoss implementation."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample binary classification data for testing.

        Returns:
            Tuple of (y_true, y_pred) tensors.
        """
        batch_size = 100  # Larger batch for Z-test
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
        return y_true, y_pred

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        loss = SpiegelhalterZLoss()

        assert loss.use_squared is True
        assert loss.from_logits is False
        assert loss.reduction == 'sum_over_batch_size'
        assert loss.name == "spiegelhalter_z_loss"

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        loss = SpiegelhalterZLoss(
            use_squared=False,
            from_logits=True,
            reduction='none',
            name='custom_z'
        )

        assert loss.use_squared is False
        assert loss.from_logits is True
        assert loss.reduction == 'none'
        assert loss.name == "custom_z"

    def test_forward_pass(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test forward pass computation."""
        y_true, y_pred = sample_data
        loss = SpiegelhalterZLoss()

        result = loss(y_true, y_pred)

        # Check result is valid
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))
        assert ops.all(result >= 0.0)  # Squared Z-stat is always non-negative

    def test_squared_vs_absolute(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test difference between squared and absolute Z-statistic."""
        y_true, y_pred = sample_data

        loss_squared = SpiegelhalterZLoss(use_squared=True)
        loss_abs = SpiegelhalterZLoss(use_squared=False)

        result_squared = loss_squared(y_true, y_pred)
        result_abs = loss_abs(y_true, y_pred)

        # Both should be valid but potentially different
        assert not ops.any(ops.isnan(result_squared))
        assert not ops.any(ops.isnan(result_abs))
        assert ops.all(result_squared >= 0.0)
        assert ops.all(result_abs >= 0.0)

    def test_well_calibrated_predictions(self) -> None:
        """Test Z-statistic with well-calibrated predictions."""
        # Create well-calibrated synthetic data
        n_samples = 1000
        probs = np.random.uniform(0.1, 0.9, n_samples)
        outcomes = np.random.binomial(1, probs)  # Generate outcomes based on probabilities

        y_true = ops.cast(outcomes.reshape(-1, 1), dtype="float32")
        y_pred = ops.cast(probs.reshape(-1, 1), dtype="float32")

        loss = SpiegelhalterZLoss()
        result = loss(y_true, y_pred)

        # Well-calibrated data should have relatively small Z-statistic
        # (though there's randomness involved)
        assert not ops.any(ops.isnan(result))

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the loss."""
        original_loss = SpiegelhalterZLoss(use_squared=False, from_logits=True)

        # Get config and recreate
        config = original_loss.get_config()
        recreated_loss = SpiegelhalterZLoss.from_config(config)

        # Check configuration matches
        assert recreated_loss.use_squared == original_loss.use_squared
        assert recreated_loss.from_logits == original_loss.from_logits


class TestCombinedCalibrationLoss:
    """Test suite for CombinedCalibrationLoss implementation."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample binary classification data for testing.

        Returns:
            Tuple of (y_true, y_pred) tensors.
        """
        batch_size = 100
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
        return y_true, y_pred

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        loss = CombinedCalibrationLoss()

        assert loss.alpha == 0.5
        assert loss.use_squared_z is True
        assert loss.from_logits is False

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        loss = CombinedCalibrationLoss(
            alpha=0.7,
            use_squared_z=False,
            from_logits=True
        )

        assert loss.alpha == 0.7
        assert loss.use_squared_z is False
        assert loss.from_logits is True

    def test_invalid_alpha(self) -> None:
        """Test that invalid alpha values raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be in the range"):
            CombinedCalibrationLoss(alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be in the range"):
            CombinedCalibrationLoss(alpha=1.5)

    def test_forward_pass(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test forward pass computation."""
        y_true, y_pred = sample_data
        loss = CombinedCalibrationLoss()

        result = loss(y_true, y_pred)

        # Check result is valid
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))

    def test_alpha_extremes(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test combined loss at alpha extremes."""
        y_true, y_pred = sample_data

        # Alpha = 1.0 should be equivalent to pure Brier score
        combined_loss_1 = CombinedCalibrationLoss(alpha=1.0)
        brier_loss = BrierScoreLoss()

        result_combined = combined_loss_1(y_true, y_pred)
        result_brier = brier_loss(y_true, y_pred)

        # Should be approximately equal (allowing for small numerical differences)
        assert ops.abs(result_combined - result_brier) < 1e-5

    def test_component_losses_exist(self) -> None:
        """Test that component losses are properly initialized."""
        loss = CombinedCalibrationLoss()

        assert hasattr(loss, 'brier_loss')
        assert hasattr(loss, 'z_loss')
        assert isinstance(loss.brier_loss, BrierScoreLoss)
        assert isinstance(loss.z_loss, SpiegelhalterZLoss)

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the loss."""
        original_loss = CombinedCalibrationLoss(
            alpha=0.3,
            use_squared_z=False,
            from_logits=True
        )

        # Get config and recreate
        config = original_loss.get_config()
        recreated_loss = CombinedCalibrationLoss.from_config(config)

        # Check configuration matches
        assert recreated_loss.alpha == original_loss.alpha
        assert recreated_loss.use_squared_z == original_loss.use_squared_z
        assert recreated_loss.from_logits == original_loss.from_logits


class TestBrierScoreMetric:
    """Test suite for BrierScoreMetric implementation."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample binary classification data for testing.

        Returns:
            Tuple of (y_true, y_pred) tensors.
        """
        batch_size = 32
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
        return y_true, y_pred

    def test_initialization(self) -> None:
        """Test metric initialization."""
        metric = BrierScoreMetric()

        assert metric.name == 'brier_score'
        assert metric.from_logits is False
        assert hasattr(metric, 'total_score')
        assert hasattr(metric, 'count')

    def test_update_and_result(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test metric update and result computation."""
        y_true, y_pred = sample_data
        metric = BrierScoreMetric()

        # Update metric state
        metric.update_state(y_true, y_pred)

        # Get result
        result = metric.result()

        # Check result is valid
        assert not ops.any(ops.isnan(result))
        assert ops.all(result >= 0.0)

    def test_reset_state(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test metric state reset."""
        y_true, y_pred = sample_data
        metric = BrierScoreMetric()

        # Update metric state
        metric.update_state(y_true, y_pred)
        result_before = metric.result()

        # Reset state
        metric.reset_state()
        result_after = metric.result()

        # Result should be 0 after reset
        assert ops.abs(result_after) < 1e-7
        assert result_before > 0  # Should have been non-zero before reset

    def test_multiple_batches(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = BrierScoreMetric()

        # Process multiple batches
        for _ in range(5):
            batch_size = 16
            y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
            y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
            metric.update_state(y_true, y_pred)

        result = metric.result()
        assert not ops.any(ops.isnan(result))
        assert ops.all(result >= 0.0)

    def test_serialization(self) -> None:
        """Test metric serialization."""
        original_metric = BrierScoreMetric(name='custom_brier', from_logits=True)

        config = original_metric.get_config()
        recreated_metric = BrierScoreMetric(**config)

        assert recreated_metric.name == original_metric.name
        assert recreated_metric.from_logits == original_metric.from_logits


class TestSpiegelhalterZMetric:
    """Test suite for SpiegelhalterZMetric implementation."""

    @pytest.fixture
    def sample_data(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Create sample binary classification data for testing.

        Returns:
            Tuple of (y_true, y_pred) tensors.
        """
        batch_size = 100  # Larger batch for Z-statistic
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.random.uniform(0.1, 0.9, size=(batch_size, 1)), dtype="float32")
        return y_true, y_pred

    def test_initialization(self) -> None:
        """Test metric initialization."""
        metric = SpiegelhalterZMetric()

        assert metric.name == 'spiegelhalter_z'
        assert metric.from_logits is False
        assert hasattr(metric, 'residual_sum')
        assert hasattr(metric, 'variance_sum')

    def test_update_and_result(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test metric update and result computation."""
        y_true, y_pred = sample_data
        metric = SpiegelhalterZMetric()

        # Update metric state
        metric.update_state(y_true, y_pred)

        # Get result
        result = metric.result()

        # Check result is valid (Z-statistic can be positive or negative)
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))

    def test_reset_state(self, sample_data: Tuple[keras.KerasTensor, keras.KerasTensor]) -> None:
        """Test metric state reset."""
        y_true, y_pred = sample_data
        metric = SpiegelhalterZMetric()

        # Update metric state
        metric.update_state(y_true, y_pred)

        # Reset state
        metric.reset_state()
        result_after = metric.result()

        # Result should be 0 after reset
        assert ops.abs(result_after) < 1e-7

    def test_serialization(self) -> None:
        """Test metric serialization."""
        original_metric = SpiegelhalterZMetric(name='custom_z', from_logits=True)

        config = original_metric.get_config()
        recreated_metric = SpiegelhalterZMetric(**config)

        assert recreated_metric.name == original_metric.name
        assert recreated_metric.from_logits == original_metric.from_logits


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_empty_tensors(self) -> None:
        """Test with empty tensors."""
        y_true = ops.cast(np.array([]).reshape(0, 1), dtype="float32")
        y_pred = ops.cast(np.array([]).reshape(0, 1), dtype="float32")

        # Metrics should handle empty tensors gracefully
        brier_metric = BrierScoreMetric()
        z_metric = SpiegelhalterZMetric()

        brier_metric.update_state(y_true, y_pred)
        z_metric.update_state(y_true, y_pred)

        brier_result = brier_metric.result()
        z_result = z_metric.result()

        # Should return 0 for empty inputs
        assert ops.abs(brier_result) < 1e-7
        assert ops.abs(z_result) < 1e-7

    def test_extreme_probabilities(self) -> None:
        """Test with extreme probability values."""
        # Test with probabilities very close to 0 and 1
        y_true = ops.cast([[1.0], [0.0], [1.0], [0.0]], dtype="float32")
        y_pred = ops.cast([[0.999999], [0.000001], [0.999999], [0.000001]], dtype="float32")

        brier_loss = BrierScoreLoss()
        z_loss = SpiegelhalterZLoss()

        brier_result = brier_loss(y_true, y_pred)
        z_result = z_loss(y_true, y_pred)

        # Should handle extreme values without NaN/Inf
        assert not ops.any(ops.isnan(brier_result))
        assert not ops.any(ops.isnan(z_result))
        assert not ops.any(ops.isinf(brier_result))
        assert not ops.any(ops.isinf(z_result))

    def test_identical_predictions(self) -> None:
        """Test with all identical predictions."""
        batch_size = 32
        y_true = ops.cast(np.random.choice([0, 1], size=(batch_size, 1)), dtype="float32")
        y_pred = ops.cast(np.full((batch_size, 1), 0.5), dtype="float32")  # All predictions = 0.5

        z_loss = SpiegelhalterZLoss()
        result = z_loss(y_true, y_pred)

        # Should handle identical predictions without issues
        assert not ops.any(ops.isnan(result))
        assert not ops.any(ops.isinf(result))


class TestModelSaveLoad:
    """Test model saving and loading with calibration losses."""

    def test_model_save_load_with_losses(self) -> None:
        """Test saving and loading a model with calibration losses."""
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu", input_shape=(10,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        # Compile with calibration loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=CombinedCalibrationLoss(alpha=0.6),
            metrics=[BrierScoreMetric(), SpiegelhalterZMetric()]
        )

        # Generate dummy data
        x_train = np.random.random((100, 10))
        y_train = np.random.choice([0, 1], size=(100, 1))

        # Train for a few steps
        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Generate predictions before saving
        x_test = np.random.random((20, 10))
        original_predictions = model.predict(x_test, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "calibration_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "CombinedCalibrationLoss": CombinedCalibrationLoss,
                    "BrierScoreMetric": BrierScoreMetric,
                    "SpiegelhalterZMetric": SpiegelhalterZMetric
                }
            )

            # Generate predictions with loaded model
            loaded_predictions = loaded_model.predict(x_test, verbose=0)

            # Predictions should match
            np.testing.assert_allclose(
                original_predictions,
                loaded_predictions,
                rtol=1e-5
            )

            logger.info("Model save/load test passed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])