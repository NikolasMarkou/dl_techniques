"""
Tests for the CapsuleMarginLoss implementation.

This module contains unit tests for the CapsuleMarginLoss function specifically
designed for Capsule Networks, including parameter validation, loss calculation,
serialization, and integration tests.
"""

import keras
from keras import ops
import pytest
import numpy as np
import tempfile
import os
from typing import Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.losses.capsule_margin_loss import (
    CapsuleMarginLoss,
    analyze_margin_loss_components
)


@pytest.fixture
def capsule_data() -> Tuple[np.ndarray, np.ndarray]:
    """Fixture providing capsule network test data.

    Returns
    -------
    tuple
        Tuple containing (y_true, y_pred) arrays for capsule network testing.
        y_true: one-hot encoded labels
        y_pred: capsule lengths (magnitudes) in range [0, 1]
    """
    # Create consistent test data
    np.random.seed(42)

    # Create one-hot encoded labels for 10 classes
    num_samples = 20
    num_classes = 10
    y_true = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Assign random classes to samples
    true_classes = np.random.randint(0, num_classes, num_samples)
    for i, cls in enumerate(true_classes):
        y_true[i, cls] = 1.0

    # Create capsule lengths (should be high for present classes, low for absent)
    y_pred = np.random.uniform(0.0, 0.3, size=(num_samples, num_classes)).astype(np.float32)

    # Make predicted lengths higher for true classes (but not perfect)
    for i, cls in enumerate(true_classes):
        if np.random.random() > 0.2:  # 80% correct predictions
            y_pred[i, cls] = np.random.uniform(0.7, 1.0)
        else:  # 20% incorrect predictions (low capsule length for true class)
            y_pred[i, cls] = np.random.uniform(0.0, 0.4)

    # Add some false positives (high capsule lengths for wrong classes)
    for i in range(num_samples):
        false_positive_class = np.random.choice([c for c in range(num_classes) if c != true_classes[i]])
        if np.random.random() > 0.8:  # 20% chance of false positive
            y_pred[i, false_positive_class] = np.random.uniform(0.6, 0.9)

    return y_true, y_pred


def test_capsule_margin_loss_initialization() -> None:
    """Test CapsuleMarginLoss initialization with various parameters."""
    logger.info("Testing CapsuleMarginLoss initialization")

    # Test default initialization
    loss = CapsuleMarginLoss()
    assert loss.positive_margin == 0.9
    assert loss.negative_margin == 0.1
    assert loss.downweight == 0.5
    logger.debug("Default initialization test passed")

    # Test custom initialization
    loss = CapsuleMarginLoss(
        positive_margin=0.85,
        negative_margin=0.15,
        downweight=0.3
    )
    assert loss.positive_margin == 0.85
    assert loss.negative_margin == 0.15
    assert loss.downweight == 0.3
    logger.debug("Custom initialization test passed")


def test_capsule_margin_loss_parameter_validation() -> None:
    """Test parameter validation in CapsuleMarginLoss initialization."""
    logger.info("Testing parameter validation")

    # Test invalid positive margin
    with pytest.raises(ValueError, match="positive_margin must be in range"):
        CapsuleMarginLoss(positive_margin=1.1)

    with pytest.raises(ValueError, match="positive_margin must be in range"):
        CapsuleMarginLoss(positive_margin=0.0)

    # Test invalid negative margin
    with pytest.raises(ValueError, match="negative_margin must be in range"):
        CapsuleMarginLoss(negative_margin=1.1)

    with pytest.raises(ValueError, match="negative_margin must be in range"):
        CapsuleMarginLoss(negative_margin=0.0)

    # Test positive margin <= negative margin
    with pytest.raises(ValueError, match="positive_margin .* must be greater than negative_margin"):
        CapsuleMarginLoss(positive_margin=0.3, negative_margin=0.4)

    with pytest.raises(ValueError, match="positive_margin .* must be greater than negative_margin"):
        CapsuleMarginLoss(positive_margin=0.5, negative_margin=0.5)

    # Test invalid downweight
    with pytest.raises(ValueError, match="downweight must be in range"):
        CapsuleMarginLoss(downweight=0.0)

    with pytest.raises(ValueError, match="downweight must be in range"):
        CapsuleMarginLoss(downweight=1.5)

    logger.debug("Parameter validation tests passed")


def test_capsule_margin_loss_calculation(capsule_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test that CapsuleMarginLoss correctly calculates loss values."""
    y_true, y_pred = capsule_data

    logger.info("Testing margin loss calculation")
    logger.info(f"Test data shape: y_true={y_true.shape}, y_pred={y_pred.shape}")

    # Create loss function
    loss_fn = CapsuleMarginLoss(
        positive_margin=0.9,
        negative_margin=0.1,
        downweight=0.5
    )

    # Convert to tensors
    y_true_tensor = ops.convert_to_tensor(y_true)
    y_pred_tensor = ops.convert_to_tensor(y_pred)

    # Calculate loss
    loss_value = loss_fn(y_true_tensor, y_pred_tensor)
    loss_numpy = ops.convert_to_numpy(loss_value)

    logger.info(f"Calculated loss: {loss_numpy}")

    # Verify loss is a tensor (scalar due to reduction)
    assert hasattr(loss_value, 'numpy'), "Loss should be a tensor"
    # Due to reduction strategy, loss is typically a scalar
    assert ops.shape(loss_value) == (), f"Loss should be scalar due to reduction, got shape {ops.shape(loss_value)}"

    # Verify loss values are non-negative
    assert np.all(loss_numpy >= 0), f"Loss values should be non-negative, got min={np.min(loss_numpy)}"

    # Manual calculation for verification
    positive_loss = y_true * np.square(np.maximum(0.0, 0.9 - y_pred))
    negative_loss = 0.5 * (1.0 - y_true) * np.square(np.maximum(0.0, y_pred - 0.1))
    expected_loss_per_sample = np.sum(positive_loss + negative_loss, axis=1)
    # Apply same reduction as the loss function (sum_over_batch_size by default)
    expected_loss = np.mean(expected_loss_per_sample)

    # Compare with manual calculation
    assert np.allclose(loss_numpy, expected_loss, atol=1e-5), \
        f"Loss calculation mismatch. Expected {expected_loss}, got {loss_numpy}"

    logger.debug("Loss calculation test passed")


def test_capsule_margin_loss_edge_cases() -> None:
    """Test CapsuleMarginLoss with edge cases."""
    logger.info("Testing edge cases")

    loss_fn = CapsuleMarginLoss()

    # Test with perfect predictions (capsule lengths = 1 for true classes, 0 for false)
    y_true_perfect = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    y_pred_perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    y_true_tensor = ops.convert_to_tensor(y_true_perfect)
    y_pred_tensor = ops.convert_to_tensor(y_pred_perfect)

    loss_perfect = ops.convert_to_numpy(loss_fn(y_true_tensor, y_pred_tensor))
    logger.info(f"Perfect prediction loss: {loss_perfect}")

    # Loss should be very small for perfect predictions
    assert np.all(loss_perfect < 0.1), f"Perfect predictions should have low loss, got {loss_perfect}"

    # Test with worst predictions (capsule lengths = 0 for true classes, 1 for false)
    y_pred_worst = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    y_pred_worst_tensor = ops.convert_to_tensor(y_pred_worst)

    loss_worst = ops.convert_to_numpy(loss_fn(y_true_tensor, y_pred_worst_tensor))
    logger.info(f"Worst prediction loss: {loss_worst}")

    # Loss should be much higher for worst predictions
    assert np.all(loss_worst > loss_perfect), \
        f"Worst predictions should have higher loss than perfect predictions"

    # Test with all zeros
    y_pred_zeros = np.zeros_like(y_true_perfect)
    y_pred_zeros_tensor = ops.convert_to_tensor(y_pred_zeros)

    loss_zeros = ops.convert_to_numpy(loss_fn(y_true_tensor, y_pred_zeros_tensor))
    logger.info(f"All zeros prediction loss: {loss_zeros}")

    # Should be finite and positive
    assert np.all(np.isfinite(loss_zeros)), "Loss should be finite"
    assert np.all(loss_zeros > 0), "Loss should be positive for imperfect predictions"

    logger.debug("Edge cases test passed")


def test_capsule_margin_loss_different_margins() -> None:
    """Test that different margin values produce different loss values."""
    logger.info("Testing different margin configurations")

    # Create test data
    y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.2, 0.1], [0.3, 0.7, 0.2]], dtype=np.float32)

    y_true_tensor = ops.convert_to_tensor(y_true)
    y_pred_tensor = ops.convert_to_tensor(y_pred)

    # Test different positive margins
    loss_high_margin = CapsuleMarginLoss(positive_margin=0.95, negative_margin=0.1)
    loss_low_margin = CapsuleMarginLoss(positive_margin=0.7, negative_margin=0.1)

    loss_high = ops.convert_to_numpy(loss_high_margin(y_true_tensor, y_pred_tensor))
    loss_low = ops.convert_to_numpy(loss_low_margin(y_true_tensor, y_pred_tensor))

    logger.info(f"High margin loss: {loss_high}")
    logger.info(f"Low margin loss: {loss_low}")

    # Higher positive margin should generally result in higher loss (stricter requirement)
    assert not np.allclose(loss_high, loss_low), \
        f"Different margins should produce different losses"

    # Test different downweight values
    loss_high_downweight = CapsuleMarginLoss(downweight=0.8)
    loss_low_downweight = CapsuleMarginLoss(downweight=0.2)

    loss_high_dw = ops.convert_to_numpy(loss_high_downweight(y_true_tensor, y_pred_tensor))
    loss_low_dw = ops.convert_to_numpy(loss_low_downweight(y_true_tensor, y_pred_tensor))

    logger.info(f"High downweight loss: {loss_high_dw}")
    logger.info(f"Low downweight loss: {loss_low_dw}")

    assert not np.allclose(loss_high_dw, loss_low_dw), \
        f"Different downweights should produce different losses"

    logger.debug("Different margins test passed")


def test_model_training_with_capsule_margin_loss(capsule_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test that a model can be trained using CapsuleMarginLoss."""
    y_true, y_pred = capsule_data

    logger.info("Testing model training with CapsuleMarginLoss")

    # Create a simple model that outputs capsule lengths
    # Use functional API to avoid warnings
    inputs = keras.Input(shape=(y_true.shape[1],))  # Input same as number of classes for simplicity
    x = keras.layers.Dense(32, activation='relu',
                          kernel_initializer='he_normal')(inputs)
    x = keras.layers.Dense(16, activation='relu',
                          kernel_initializer='he_normal')(x)
    # Output layer should produce capsule lengths (0 to 1)
    outputs = keras.layers.Dense(y_true.shape[1], activation='sigmoid',
                                kernel_initializer='glorot_uniform')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create synthetic input data (using y_true as features for simplicity)
    x_train = y_true + np.random.normal(0, 0.1, y_true.shape).astype(np.float32)

    # Compile with CapsuleMarginLoss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=CapsuleMarginLoss(),
        metrics=['mae']
    )

    logger.info("Starting model training with CapsuleMarginLoss")

    # Train for a few epochs
    history = model.fit(
        x_train, y_true,  # Using y_true as labels (capsule lengths should match one-hot)
        epochs=5,
        batch_size=8,
        verbose=0
    )

    initial_loss = history.history['loss'][0]
    final_loss = history.history['loss'][-1]

    logger.info(f"Training completed. Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

    # Check that the loss decreased during training (or at least didn't increase significantly)
    assert final_loss <= initial_loss * 1.1, \
        f"Expected loss to decrease or stay similar during training, got {initial_loss} -> {final_loss}"

    # Test model prediction
    predictions = model.predict(x_train[:2], verbose=0)
    logger.info(f"Sample predictions shape: {predictions.shape}")
    logger.info(f"Sample prediction values: {predictions[0][:3]}")

    # Predictions should be in valid range [0, 1] (sigmoid output)
    assert np.all(predictions >= 0) and np.all(predictions <= 1), \
        f"Predictions should be in range [0, 1], got min={np.min(predictions)}, max={np.max(predictions)}"

    logger.debug("Model training test passed")


def test_capsule_margin_loss_serialization() -> None:
    """Test that CapsuleMarginLoss can be properly serialized and deserialized."""
    logger.info("Testing CapsuleMarginLoss serialization")

    # Create a model with CapsuleMarginLoss
    inputs = keras.Input(shape=(5,))
    x = keras.layers.Dense(8, activation='relu')(inputs)
    outputs = keras.layers.Dense(3, activation='sigmoid')(x)  # 3 capsule lengths
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create the loss with custom parameters
    original_loss = CapsuleMarginLoss(
        positive_margin=0.85,
        negative_margin=0.15,
        downweight=0.3
    )

    logger.info(f"Created original loss with positive_margin={original_loss.positive_margin}, "
               f"negative_margin={original_loss.negative_margin}, downweight={original_loss.downweight}")

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=original_loss
    )

    # Save the model to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, 'capsule_model.keras')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Custom objects dictionary for loading
        custom_objects = {
            'CapsuleMarginLoss': CapsuleMarginLoss
        }

        # Load the model back
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        logger.info("Model loaded successfully")

        # Get the loss configuration
        loaded_loss = loaded_model.loss

        # Check that the configuration matches
        assert loaded_loss.positive_margin == original_loss.positive_margin, \
            f"Expected positive_margin to be {original_loss.positive_margin}, got {loaded_loss.positive_margin}"
        assert loaded_loss.negative_margin == original_loss.negative_margin, \
            f"Expected negative_margin to be {original_loss.negative_margin}, got {loaded_loss.negative_margin}"
        assert loaded_loss.downweight == original_loss.downweight, \
            f"Expected downweight to be {original_loss.downweight}, got {loaded_loss.downweight}"

        logger.info("Serialization test passed successfully")


def test_analyze_margin_loss_components(capsule_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the analyze_margin_loss_components function."""
    y_true, y_pred = capsule_data

    logger.info("Testing margin loss component analysis")

    # Create loss function
    loss_fn = CapsuleMarginLoss(
        positive_margin=0.9,
        negative_margin=0.1,
        downweight=0.5
    )

    # Convert to tensors
    y_true_tensor = ops.convert_to_tensor(y_true)
    y_pred_tensor = ops.convert_to_tensor(y_pred)

    # Analyze components
    analysis = analyze_margin_loss_components(loss_fn, y_true_tensor, y_pred_tensor)

    logger.info("Analysis results:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")

    # Check that all expected keys are present
    expected_keys = {
        'total_loss', 'positive_loss', 'negative_loss',
        'avg_present_length', 'avg_absent_length',
        'positive_margin', 'negative_margin', 'downweight',
        'positive_contrib_pct', 'negative_contrib_pct',
        'margin_gap', 'present_below_margin', 'absent_above_margin'
    }

    assert set(analysis.keys()) == expected_keys, \
        f"Missing or extra keys in analysis. Expected {expected_keys}, got {set(analysis.keys())}"

    # Check that values are reasonable
    assert analysis['total_loss'] >= 0, "Total loss should be non-negative"
    assert analysis['positive_loss'] >= 0, "Positive loss should be non-negative"
    assert analysis['negative_loss'] >= 0, "Negative loss should be non-negative"

    # Check percentage contributions sum to approximately 100% (only if total_loss > 0)
    if analysis['total_loss'] > 1e-9:
        total_contrib = analysis['positive_contrib_pct'] + analysis['negative_contrib_pct']
        assert abs(total_contrib - 100.0) < 1e-3, \
            f"Contribution percentages should sum to 100%, got {total_contrib}"

    # Check that margin values match the loss function
    assert analysis['positive_margin'] == loss_fn.positive_margin
    assert analysis['negative_margin'] == loss_fn.negative_margin
    assert analysis['downweight'] == loss_fn.downweight

    # Average lengths should be in valid range
    assert 0 <= analysis['avg_present_length'] <= 1, \
        f"Average present length should be in [0,1], got {analysis['avg_present_length']}"
    assert 0 <= analysis['avg_absent_length'] <= 1, \
        f"Average absent length should be in [0,1], got {analysis['avg_absent_length']}"

    logger.debug("Component analysis test passed")


def test_capsule_margin_loss_dtype_handling() -> None:
    """Test that CapsuleMarginLoss handles different dtypes correctly."""
    logger.info("Testing dtype handling")

    loss_fn = CapsuleMarginLoss()

    # Test with different dtypes
    y_true_float32 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    y_pred_float64 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)

    y_true_tensor = ops.convert_to_tensor(y_true_float32)
    y_pred_tensor = ops.convert_to_tensor(y_pred_float64)

    # Should handle dtype conversion gracefully
    loss_value = loss_fn(y_true_tensor, y_pred_tensor)
    loss_numpy = ops.convert_to_numpy(loss_value)

    assert np.isfinite(loss_numpy).all(), "Loss should be finite with mixed dtypes"
    logger.info(f"Mixed dtype loss: {loss_numpy}")

    # Test with integer labels (should be converted to float)
    y_true_int = np.array([[1, 0], [0, 1]], dtype=np.int32)
    y_true_int_tensor = ops.convert_to_tensor(y_true_int)

    loss_value_int = loss_fn(y_true_int_tensor, y_pred_tensor)
    loss_numpy_int = ops.convert_to_numpy(loss_value_int)

    # Should produce similar results
    assert np.allclose(loss_numpy, loss_numpy_int, atol=1e-5), \
        f"Loss should be similar with int32 vs float32 labels"

    logger.debug("Dtype handling test passed")


def test_capsule_margin_loss_batch_consistency() -> None:
    """Test that CapsuleMarginLoss produces consistent results across different batch sizes."""
    logger.info("Testing batch consistency")

    # Use reduction='none' to get per-sample losses for comparison
    loss_fn = CapsuleMarginLoss(reduction='none')

    # Create test data
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]], dtype=np.float32)

    # Calculate loss for full batch
    y_true_tensor = ops.convert_to_tensor(y_true)
    y_pred_tensor = ops.convert_to_tensor(y_pred)

    full_batch_loss = ops.convert_to_numpy(loss_fn(y_true_tensor, y_pred_tensor))
    logger.info(f"Full batch loss: {full_batch_loss}")

    # Calculate loss for individual samples
    individual_losses = []
    for i in range(len(y_true)):
        y_true_single = ops.convert_to_tensor(y_true[i:i+1])
        y_pred_single = ops.convert_to_tensor(y_pred[i:i+1])
        single_loss = ops.convert_to_numpy(loss_fn(y_true_single, y_pred_single))
        # single_loss should be a 1D array with one element when reduction='none'
        individual_losses.append(single_loss[0])

    individual_losses = np.array(individual_losses)
    logger.info(f"Individual losses: {individual_losses}")

    # Should be approximately equal
    assert np.allclose(full_batch_loss, individual_losses, atol=1e-6), \
        f"Batch and individual losses should match. Batch: {full_batch_loss}, Individual: {individual_losses}"

    logger.debug("Batch consistency test passed")


def test_capsule_margin_loss_warnings() -> None:
    """Test that CapsuleMarginLoss produces appropriate warnings for problematic parameters."""
    logger.info("Testing warning generation")

    # Test small margin gap warning
    with pytest.warns(UserWarning, match="Small margin gap"):
        CapsuleMarginLoss(positive_margin=0.3, negative_margin=0.25)

    # Test very low downweight warning
    with pytest.warns(UserWarning, match="Very low downweight"):
        CapsuleMarginLoss(downweight=0.05)

    logger.debug("Warning tests passed")


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])