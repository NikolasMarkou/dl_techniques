"""
Tests for the AnyLoss framework implementation.

This module contains unit tests for the AnyLoss framework including the
ApproximationFunction and various loss function implementations.
"""
import keras
import pytest
import numpy as np
import tensorflow as tf

# Import the classes to test - adjust the import path as needed
from dl_techniques.losses.any_loss import (
    ApproximationFunction,
    AnyLoss,
    AccuracyLoss,
    F1Loss,
    FBetaLoss,
    GeometricMeanLoss,
    BalancedAccuracyLoss,
    WeightedCrossEntropyWithAnyLoss
)

@pytest.fixture
def binary_data():
    """Fixture providing simple binary classification data for testing."""
    # Create consistent test data
    np.random.seed(42)

    # Create synthetic binary classification data
    y_true = np.array([
        [1], [0], [1], [1], [0], [0], [1], [0], [1], [0],
        [1], [0], [1], [1], [0], [0], [1], [0], [1], [0]
    ], dtype=np.float32)

    # Some very confident predictions (close to 0 or 1)
    # and some less confident ones (close to 0.5)
    y_pred = np.array([
        [0.92], [0.08], [0.89], [0.51], [0.48], [0.12], [0.99], [0.03], [0.85], [0.11],
        [0.93], [0.45], [0.78], [0.67], [0.01], [0.34], [0.91], [0.23], [0.95], [0.22]
    ], dtype=np.float32)

    return y_true, y_pred


def test_approximation_function_transforms_values(binary_data):
    """Test that ApproximationFunction properly transforms probabilities to near-binary values."""
    y_true, y_pred = binary_data

    # Create the approximation function layer
    approx_fn = ApproximationFunction(amplifying_scale=73.0)

    # Apply the function to predictions
    y_approx = approx_fn(y_pred).numpy()

    # Check that values close to 1 become very close to 1
    high_confidence_indices = np.where(y_pred > 0.8)[0]
    for idx in high_confidence_indices:
        assert y_approx[idx] > 0.99, f"Expected approx value for {y_pred[idx]} to be > 0.99, got {y_approx[idx]}"

    # Check that values close to 0 become very close to 0
    low_confidence_indices = np.where(y_pred < 0.2)[0]
    for idx in low_confidence_indices:
        assert y_approx[idx] < 0.01, f"Expected approx value for {y_pred[idx]} to be < 0.01, got {y_approx[idx]}"

    # Check that values close to 0.5 stay somewhat ambiguous
    mid_confidence_indices = np.where((y_pred > 0.45) & (y_pred < 0.55))[0]
    for idx in mid_confidence_indices:
        # Should not be too close to either 0 or 1
        assert y_approx[idx] > 0.1 and y_approx[idx] < 0.9, \
            f"Expected approx value for {y_pred[idx]} to remain somewhat ambiguous, got {y_approx[idx]}"


def test_f1_loss_calculation(binary_data):
    """Test that F1Loss correctly calculates loss values."""
    y_true, y_pred = binary_data

    # Calculate expected F1 score manually using non-amplified values
    # For this test, we'll use 0.5 as the threshold for binary predictions
    y_pred_binary = (y_pred > 0.5).astype(np.float32)

    tp = np.sum(y_true * y_pred_binary)
    fn = np.sum(y_true * (1.0 - y_pred_binary))
    fp = np.sum((1.0 - y_true) * y_pred_binary)

    # Calculate traditional F1 score (with a small epsilon to avoid division by zero)
    epsilon = tf.keras.backend.epsilon()
    expected_f1 = (2.0 * tp) / (2.0 * tp + fp + fn + epsilon)
    expected_loss = 1.0 - expected_f1

    # Create F1Loss with a very high amplifying scale to make it closer to hard classification
    f1_loss = F1Loss(amplifying_scale=1000.0, reduction="sum_over_batch_size")

    # Calculate the loss
    actual_loss = f1_loss(y_true, y_pred).numpy()

    # The loss should be close to our manual calculation, but not exactly the same
    # due to the approximation function
    assert np.isclose(actual_loss, expected_loss, atol=0.1), \
        f"Expected F1Loss to be close to {expected_loss}, got {actual_loss}"


def test_model_training_with_anyloss():
    """Test that a model can be trained using AnyLoss implementations."""
    # Create a simple dataset
    np.random.seed(42)
    x_train = np.random.normal(size=(100, 10)).astype(np.float32)

    # Create imbalanced dataset (20% positive, 80% negative)
    y_train = np.zeros((100, 1), dtype=np.float32)
    positive_indices = np.random.choice(100, size=20, replace=False)
    y_train[positive_indices] = 1.0

    # Create a simple model using Functional API to avoid warnings
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(16, activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=keras.regularizers.L2(0.001))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(8, activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(1, activation='sigmoid',
                          kernel_initializer='glorot_uniform',
                          kernel_regularizer=keras.regularizers.L2(0.001))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model with F1Loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=F1Loss(reduction="sum_over_batch_size"),
        metrics=['accuracy']
    )

    # Train for a few epochs
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=16,
        verbose=0
    )

    # Check that the loss decreased during training
    assert history.history['loss'][0] > history.history['loss'][-1], \
        "Expected loss to decrease during training"


def test_fbeta_loss_with_different_betas():
    """Test that FBetaLoss produces different results with different beta values."""
    # Create consistent test data
    np.random.seed(42)

    # Create synthetic binary classification data
    y_true = np.array([
        [1], [0], [1], [1], [0], [0], [1], [0], [1], [0],
        [1], [0], [1], [1], [0], [0], [1], [0], [1], [0]
    ], dtype=np.float32)

    # Some very confident predictions (close to 0 or 1)
    # and some less confident ones (close to 0.5)
    y_pred = np.array([
        [0.92], [0.08], [0.89], [0.51], [0.48], [0.12], [0.99], [0.03], [0.85], [0.11],
        [0.93], [0.45], [0.78], [0.67], [0.01], [0.34], [0.91], [0.23], [0.95], [0.22]
    ], dtype=np.float32)

    # Create FBetaLoss with different beta values
    # Use more extreme beta values to make differences more pronounced
    f1_loss = FBetaLoss(beta=1.0, reduction="sum_over_batch_size")  # Same as F1Loss
    f5_loss = FBetaLoss(beta=5.0, reduction="sum_over_batch_size")  # Much more weight on recall
    f01_loss = FBetaLoss(beta=0.1, reduction="sum_over_batch_size")  # Much more weight on precision

    # Calculate losses
    loss_f1 = f1_loss(y_true, y_pred).numpy()
    loss_f5 = f5_loss(y_true, y_pred).numpy()
    loss_f01 = f01_loss(y_true, y_pred).numpy()

    # Print the values for debugging
    print(f"F1 Loss (beta=1.0): {loss_f1}")
    print(f"F5 Loss (beta=5.0): {loss_f5}")
    print(f"F0.1 Loss (beta=0.1): {loss_f01}")

    # Use direct inequality assertions instead of numerical closeness
    assert loss_f1 != loss_f5, \
        f"Expected F1Loss and F5Loss to be different, got {loss_f1} and {loss_f5}"
    assert loss_f1 != loss_f01, \
        f"Expected F1Loss and F0.1Loss to be different, got {loss_f1} and {loss_f01}"
    assert loss_f5 != loss_f01, \
        f"Expected F5Loss and F0.1Loss to be different, got {loss_f5} and {loss_f01}"


def test_loss_serialization_and_deserialization():
    """Test that AnyLoss implementations can be properly serialized and deserialized."""
    # Create a model with an AnyLoss
    # Create a model with proper Input layer instead of input_shape parameter
    inputs = keras.Input(shape=(5,))
    x = keras.layers.Dense(4, activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create the loss with custom parameters
    original_loss = FBetaLoss(
        beta=1.5,
        amplifying_scale=80.0,
        from_logits=False,
        reduction="sum_over_batch_size"
    )

    # Compile the model
    model.compile(
        optimizer='adam',
        loss=original_loss
    )

    # Save the model to a temporary file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, 'model.keras')
        model.save(model_path)

        # Custom objects dictionary for loading
        custom_objects = {
            'FBetaLoss': FBetaLoss,
            'ApproximationFunction': ApproximationFunction
        }

        # Load the model back
        loaded_model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )

        # Get the loss configuration
        loaded_loss = loaded_model.loss

        # Check that the configuration matches
        assert loaded_loss.beta == original_loss.beta, \
            f"Expected beta to be {original_loss.beta}, got {loaded_loss.beta}"
        assert loaded_loss.amplifying_scale == original_loss.amplifying_scale, \
            f"Expected amplifying_scale to be {original_loss.amplifying_scale}, got {loaded_loss.amplifying_scale}"
        assert loaded_loss.from_logits == original_loss.from_logits, \
            f"Expected from_logits to be {original_loss.from_logits}, got {loaded_loss.from_logits}"


def test_geometric_mean_loss_on_imbalanced_data():
    """Test that GeometricMeanLoss works well on imbalanced datasets."""
    # Create a highly imbalanced dataset (10% positive, 90% negative)
    np.random.seed(42)
    y_true = np.zeros((100, 1), dtype=np.float32)
    y_true[:10] = 1.0  # Only 10% positive examples

    # Create predictions with varying degrees of confidence
    # Some correct, some incorrect predictions
    y_pred = np.random.uniform(0, 0.3, size=(100, 1)).astype(np.float32)  # Most predicted as negative
    y_pred[:5] = np.random.uniform(0.7, 1.0, size=(5, 1))  # 5 true positives (correctly predicted)
    y_pred[10:15] = np.random.uniform(0.7, 1.0, size=(5, 1))  # 5 false positives

    # Calculate using GeometricMeanLoss
    gmean_loss = GeometricMeanLoss(reduction="sum_over_batch_size")
    loss_value = gmean_loss(y_true, y_pred).numpy()

    # Calculate expected value manually
    # True positives: 5, False negatives: 5, False positives: 5, True negatives: 85
    sensitivity = 5 / 10  # TP / (TP + FN)
    specificity = 85 / 90  # TN / (TN + FP)
    expected_gmean = np.sqrt(sensitivity * specificity)
    expected_loss = 1.0 - expected_gmean

    # Check if the loss is in the reasonable range
    # We can't expect exact match due to the approximation function
    assert 0 <= loss_value <= 1, f"Loss value {loss_value} should be between 0 and 1"
    assert np.isclose(loss_value, expected_loss, atol=0.2), \
        f"Expected GeometricMeanLoss to be around {expected_loss}, got {loss_value}"


def test_amplifying_scale_effect():
    """Test the effect of different amplifying scale values on the approximation function."""
    # Create a range of probability values to test
    prob_values = np.linspace(0, 1, 11).reshape(-1, 1).astype(np.float32)

    # Create approximation functions with different scales
    approx_low = ApproximationFunction(amplifying_scale=10.0)
    approx_med = ApproximationFunction(amplifying_scale=73.0)
    approx_high = ApproximationFunction(amplifying_scale=200.0)

    # Apply the functions
    result_low = approx_low(prob_values).numpy()
    result_med = approx_med(prob_values).numpy()
    result_high = approx_high(prob_values).numpy()

    # Test that higher amplifying scales make the function sharper
    # For values > 0.5, higher scale should result in values closer to 1
    high_indices = np.where(prob_values > 0.5)[0]
    for idx in high_indices:
        if idx < len(result_low):  # Ensure index is valid
            assert result_low[idx] <= result_med[idx] <= result_high[idx], \
                f"At p={prob_values[idx]}, expected higher scales to produce higher values, got " \
                f"{result_low[idx]}, {result_med[idx]}, {result_high[idx]}"

    # For values < 0.5, higher scale should result in values closer to 0
    low_indices = np.where(prob_values < 0.5)[0]
    for idx in low_indices:
        if idx < len(result_low):  # Ensure index is valid
            assert result_low[idx] >= result_med[idx] >= result_high[idx], \
                f"At p={prob_values[idx]}, expected higher scales to produce lower values, got " \
                f"{result_low[idx]}, {result_med[idx]}, {result_high[idx]}"

    # For p=0.5, all scales should produce approximately 0.5
    mid_indices = np.where(np.isclose(prob_values, 0.5))[0]
    for idx in mid_indices:
        if idx < len(result_low):  # Ensure index is valid
            assert np.isclose(result_low[idx], 0.5, atol=0.1)
            assert np.isclose(result_med[idx], 0.5, atol=0.1)
            assert np.isclose(result_high[idx], 0.5, atol=0.1)


def test_balanced_accuracy_loss_vs_accuracy_loss():
    """Test that BalancedAccuracyLoss handles imbalanced data better than AccuracyLoss."""
    # Create a highly imbalanced dataset (10% positive, 90% negative)
    np.random.seed(42)
    y_true = np.zeros((100, 1), dtype=np.float32)
    y_true[:10] = 1.0  # Only 10% positive examples

    # Create predictions where all examples are predicted as negative
    # This gives 90% accuracy but 0% recall
    y_pred = np.zeros((100, 1), dtype=np.float32)

    # Add a small epsilon to avoid exact 0 or 1 values
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    # Calculate using AccuracyLoss
    accuracy_loss = AccuracyLoss(reduction="sum_over_batch_size")
    acc_loss_value = accuracy_loss(y_true, y_pred).numpy()

    # Calculate using BalancedAccuracyLoss
    balanced_loss = BalancedAccuracyLoss(reduction="sum_over_batch_size")
    bal_loss_value = balanced_loss(y_true, y_pred).numpy()

    # The balanced loss should be higher (worse) because it penalizes the
    # model for not predicting any positive examples
    assert bal_loss_value > acc_loss_value, \
        f"Expected BalancedAccuracyLoss {bal_loss_value} to be higher than AccuracyLoss {acc_loss_value}"

    # AccuracyLoss should be around 0.1 (1 - 0.9 accuracy)
    assert np.isclose(acc_loss_value, 0.1, atol=0.1), \
        f"Expected AccuracyLoss to be around 0.1, got {acc_loss_value}"

    # BalancedAccuracyLoss should be around 0.5 (1 - (0.5+0.0)/2) given specificity=1, sensitivity=0
    assert np.isclose(bal_loss_value, 0.5, atol=0.1), \
        f"Expected BalancedAccuracyLoss to be around 0.5, got {bal_loss_value}"


def test_weighted_cross_entropy_with_any_loss():
    """Test the WeightedCrossEntropyWithAnyLoss combination loss."""
    np.random.seed(42)
    y_true = np.array([[1], [0], [1], [0], [1]], dtype=np.float32)
    y_pred = np.array([[0.8], [0.3], [0.6], [0.2], [0.9]], dtype=np.float32)

    # Create component losses
    f1_loss = F1Loss(reduction="sum_over_batch_size")
    bce = keras.losses.BinaryCrossentropy(from_logits=False)

    # Calculate individual loss values
    f1_loss_value = f1_loss(y_true, y_pred).numpy()
    bce_value = bce(y_true, y_pred).numpy()

    # Test with different alpha values
    alphas = [0.0, 0.3, 0.7, 1.0]
    for alpha in alphas:
        # Create the combined loss
        combined_loss = WeightedCrossEntropyWithAnyLoss(
            anyloss=f1_loss,
            alpha=alpha,
            reduction="sum_over_batch_size"
        )

        # Calculate combined loss value
        combined_value = combined_loss(y_true, y_pred).numpy()

        # Check that the combined value is a weighted average of the components
        expected_value = alpha * f1_loss_value + (1 - alpha) * bce_value
        assert np.isclose(combined_value, expected_value, atol=1e-5), \
            f"With alpha={alpha}, expected combined loss to be {expected_value}, got {combined_value}"

        # Check limit cases
        if alpha == 0.0:
            # Should be equal to BCE
            assert np.isclose(combined_value, bce_value, atol=1e-5), \
                f"With alpha=0, expected combined loss to equal BCE value {bce_value}, got {combined_value}"
        elif alpha == 1.0:
            # Should be equal to F1Loss
            assert np.isclose(combined_value, f1_loss_value, atol=1e-5), \
                f"With alpha=1, expected combined loss to equal F1Loss value {f1_loss_value}, got {combined_value}"


def test_from_logits_parameter():
    """Test that from_logits parameter correctly handles logits vs probabilities."""
    np.random.seed(42)

    # Create binary labels
    y_true = np.array([[1], [0], [1], [0], [1]], dtype=np.float32)

    # Create logits (unbounded values)
    logits = np.array([[2.0], [-1.5], [0.8], [-2.2], [3.1]], dtype=np.float32)

    # Convert logits to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # Create two identical losses, one expecting logits, one expecting probabilities
    loss_with_logits = F1Loss(from_logits=True, reduction="sum_over_batch_size")
    loss_with_probs = F1Loss(from_logits=False, reduction="sum_over_batch_size")

    # Calculate loss values
    loss_value_from_logits = loss_with_logits(y_true, logits).numpy()
    loss_value_from_probs = loss_with_probs(y_true, probs).numpy()

    # Expect the values to be approximately the same
    assert np.isclose(loss_value_from_logits, loss_value_from_probs, atol=1e-5), \
        f"Expected the same loss value from logits and probabilities, got {loss_value_from_logits} and {loss_value_from_probs}"

    # Test that feeding probabilities to a loss expecting logits gives different results
    incorrect_loss_value = loss_with_logits(y_true, probs).numpy()
    assert not np.isclose(incorrect_loss_value, loss_value_from_probs, atol=1e-5), \
        f"Expected different loss values when mistakenly feeding probabilities to a loss expecting logits"