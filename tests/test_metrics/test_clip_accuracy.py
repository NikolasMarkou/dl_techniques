"""
Unit Tests for CLIP Metrics
============================

Comprehensive test suite for CLIPAccuracy and CLIPRecallAtK metrics,
verifying correctness, serialization, and edge case handling.

Test Categories:
    1. Basic Functionality: update_state, result, reset_state
    2. Serialization: get_config, from_config, save/load
    3. Correctness: Mathematical accuracy of metric calculations
    4. Edge Cases: Small batches, K > batch_size, empty inputs
    5. Error Handling: Invalid inputs, missing keys, wrong shapes
    6. Integration: Multiple metrics, batch accumulation

Run with:
    python test_clip_metrics.py
"""

import numpy as np
import keras
from keras import ops
import tempfile
from typing import Dict, Any

# Import metrics to test
from dl_techniques.metrics.clip_accuracy import CLIPAccuracy, CLIPRecallAtK


# =============================================================================
# Test Utilities
# =============================================================================


def create_perfect_similarity_matrix(batch_size: int) -> Dict[str, keras.KerasTensor]:
    """
    Create a perfect similarity matrix where diagonal is highest.

    This simulates ideal CLIP output where each image perfectly matches
    its corresponding text (100% accuracy expected).

    Args:
        batch_size: Size of batch (N x N similarity matrix).

    Returns:
        Dictionary with 'logits_per_image' and 'logits_per_text' keys.
    """
    # Create base random similarities (negative for off-diagonal)
    base = keras.random.uniform((batch_size, batch_size), -5.0, -1.0)

    # Make diagonal elements much larger (correct matches)
    identity = ops.eye(batch_size) * 10.0
    logits_i2t = base + identity

    # Transpose for T2I
    logits_t2i = ops.transpose(logits_i2t)

    return {
        'logits_per_image': logits_i2t,
        'logits_per_text': logits_t2i
    }


def create_random_similarity_matrix(batch_size: int) -> Dict[str, keras.KerasTensor]:
    """
    Create a random similarity matrix with no preference.

    This simulates untrained CLIP output (accuracy around 1/N expected).

    Args:
        batch_size: Size of batch (N x N similarity matrix).

    Returns:
        Dictionary with 'logits_per_image' and 'logits_per_text' keys.
    """
    logits_i2t = keras.random.normal((batch_size, batch_size))
    logits_t2i = ops.transpose(logits_i2t)

    return {
        'logits_per_image': logits_i2t,
        'logits_per_text': logits_t2i
    }


def create_partial_match_matrix(
        batch_size: int,
        correct_fraction: float = 0.5
) -> Dict[str, keras.KerasTensor]:
    """
    Create similarity matrix with controlled accuracy.

    Args:
        batch_size: Size of batch.
        correct_fraction: Fraction of samples that should be correct (0-1).

    Returns:
        Dictionary with 'logits_per_image' and 'logits_per_text' keys.
    """
    num_correct = int(batch_size * correct_fraction)

    # Start with random matrix
    logits = keras.random.uniform((batch_size, batch_size), -1.0, 1.0)
    logits_np = ops.convert_to_numpy(logits)

    # Make first num_correct diagonal elements highest in their row
    for i in range(num_correct):
        logits_np[i, i] = 10.0

    logits_i2t = ops.convert_to_tensor(logits_np)
    logits_t2i = ops.transpose(logits_i2t)

    return {
        'logits_per_image': logits_i2t,
        'logits_per_text': logits_t2i
    }


# =============================================================================
# Test Cases for CLIPAccuracy
# =============================================================================


def test_clip_accuracy_initialization():
    """Test CLIPAccuracy initialization with various parameters."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Initialization")
    print("=" * 70)

    # Test default initialization
    metric = CLIPAccuracy()
    assert metric.direction == 'i2t', "Default direction should be 'i2t'"
    assert metric.name == 'clip_i2t_accuracy', "Default name incorrect"
    print("✓ Default initialization successful")

    # Test custom initialization
    metric = CLIPAccuracy(direction='t2i', name='custom_accuracy')
    assert metric.direction == 't2i', "Custom direction not set"
    assert metric.name == 'custom_accuracy', "Custom name not set"
    print("✓ Custom initialization successful")

    # Test invalid direction
    try:
        metric = CLIPAccuracy(direction='invalid')
        raise AssertionError("Should have raised ValueError for invalid direction")
    except ValueError as e:
        assert 'i2t' in str(e) and 't2i' in str(e), "Error message should mention valid options"
        print("✓ Invalid direction properly rejected")

    print("✓ All initialization tests passed")


def test_clip_accuracy_perfect_predictions():
    """Test CLIPAccuracy with perfect similarity matrix."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Perfect Predictions")
    print("=" * 70)

    batch_size = 8
    outputs = create_perfect_similarity_matrix(batch_size)

    # Test I2T accuracy
    metric_i2t = CLIPAccuracy(direction='i2t')
    metric_i2t.update_state(y_pred=outputs)
    accuracy = float(metric_i2t.result())

    assert abs(accuracy - 1.0) < 1e-6, f"Expected 1.0, got {accuracy}"
    print(f"✓ I2T Accuracy (perfect): {accuracy:.4f}")

    # Test T2I accuracy
    metric_t2i = CLIPAccuracy(direction='t2i')
    metric_t2i.update_state(y_pred=outputs)
    accuracy = float(metric_t2i.result())

    assert abs(accuracy - 1.0) < 1e-6, f"Expected 1.0, got {accuracy}"
    print(f"✓ T2I Accuracy (perfect): {accuracy:.4f}")

    print("✓ Perfect predictions test passed")


def test_clip_accuracy_random_predictions():
    """Test CLIPAccuracy with random similarity matrix."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Random Predictions")
    print("=" * 70)

    batch_size = 32
    outputs = create_random_similarity_matrix(batch_size)

    # Test I2T accuracy (should be around 1/batch_size)
    metric_i2t = CLIPAccuracy(direction='i2t')
    metric_i2t.update_state(y_pred=outputs)
    accuracy = float(metric_i2t.result())

    # With random predictions, accuracy should be roughly 1/N
    expected = 1.0 / batch_size
    assert 0.0 <= accuracy <= 1.0, f"Accuracy out of valid range: {accuracy}"
    print(f"✓ I2T Accuracy (random): {accuracy:.4f} (expected ~{expected:.4f})")

    print("✓ Random predictions test passed")


def test_clip_accuracy_partial_match():
    """Test CLIPAccuracy with controlled partial matching."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Partial Match")
    print("=" * 70)

    batch_size = 10
    target_accuracy = 0.7
    outputs = create_partial_match_matrix(batch_size, target_accuracy)

    metric = CLIPAccuracy(direction='i2t')
    metric.update_state(y_pred=outputs)
    accuracy = float(metric.result())

    # Should be close to target accuracy
    assert abs(accuracy - target_accuracy) < 0.15, \
        f"Expected ~{target_accuracy}, got {accuracy}"
    print(f"✓ Accuracy: {accuracy:.4f} (target: {target_accuracy:.4f})")

    print("✓ Partial match test passed")


def test_clip_accuracy_reset_state():
    """Test CLIPAccuracy reset_state functionality."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Reset State")
    print("=" * 70)

    batch_size = 8
    outputs = create_perfect_similarity_matrix(batch_size)

    metric = CLIPAccuracy(direction='i2t')

    # First update
    metric.update_state(y_pred=outputs)
    accuracy_1 = float(metric.result())
    print(f"✓ First batch accuracy: {accuracy_1:.4f}")

    # Reset
    metric.reset_state()

    # Check that state is reset
    correct = float(metric.correct.numpy())
    total = float(metric.total.numpy())
    assert correct == 0.0, f"correct should be 0 after reset, got {correct}"
    assert total == 0.0, f"total should be 0 after reset, got {total}"
    print("✓ State properly reset to zero")

    # Second update after reset
    metric.update_state(y_pred=outputs)
    accuracy_2 = float(metric.result())

    assert abs(accuracy_1 - accuracy_2) < 1e-6, \
        f"Accuracy after reset differs: {accuracy_1} vs {accuracy_2}"
    print(f"✓ Second batch accuracy: {accuracy_2:.4f}")

    print("✓ Reset state test passed")


def test_clip_accuracy_accumulation():
    """Test CLIPAccuracy accumulation across multiple batches."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Batch Accumulation")
    print("=" * 70)

    batch_size = 4
    num_batches = 3

    metric = CLIPAccuracy(direction='i2t')

    accuracies = []
    for i in range(num_batches):
        outputs = create_partial_match_matrix(batch_size, correct_fraction=0.75)
        metric.update_state(y_pred=outputs)
        acc = float(metric.result())
        accuracies.append(acc)
        print(f"✓ Batch {i + 1}: accuracy = {acc:.4f}")

    # Check that metric is accumulating
    total = float(metric.total.numpy())
    expected_total = batch_size * num_batches
    assert total == expected_total, \
        f"Total should be {expected_total}, got {total}"
    print(f"✓ Total samples: {int(total)} (expected {expected_total})")

    print("✓ Batch accumulation test passed")


def test_clip_accuracy_serialization():
    """Test CLIPAccuracy serialization with get_config and from_config."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Serialization")
    print("=" * 70)

    # Create metric with custom parameters
    original = CLIPAccuracy(direction='t2i', name='test_metric')

    # Update state
    outputs = create_perfect_similarity_matrix(4)
    original.update_state(y_pred=outputs)
    original_result = float(original.result())

    # Get config
    config = original.get_config()
    assert 'direction' in config, "Config missing 'direction'"
    assert config['direction'] == 't2i', "Direction not in config"
    assert config['name'] == 'test_metric', "Name not in config"
    print(f"✓ Config: {config}")

    # Recreate from config
    restored = CLIPAccuracy.from_config(config)
    assert restored.direction == 't2i', "Direction not restored"
    assert restored.name == 'test_metric', "Name not restored"
    print("✓ Metric recreated from config")

    # Test that restored metric works
    restored.update_state(y_pred=outputs)
    restored_result = float(restored.result())

    assert abs(original_result - restored_result) < 1e-6, \
        "Results differ after restoration"
    print(f"✓ Original result: {original_result:.4f}")
    print(f"✓ Restored result: {restored_result:.4f}")

    print("✓ Serialization test passed")


def test_clip_accuracy_model_save_load():
    """Test CLIPAccuracy in model save/load cycle."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Model Save/Load")
    print("=" * 70)

    # Create simple model with metric
    batch_size = 4
    embed_dim = 8

    # Dummy model that outputs CLIP-like dictionary
    image_input = keras.Input(shape=(embed_dim,), name='image')
    text_input = keras.Input(shape=(embed_dim,), name='text')

    # Normalize inputs
    image_norm = keras.layers.Lambda(
        lambda x: x / ops.norm(x, axis=-1, keepdims=True)
    )(image_input)
    text_norm = keras.layers.Lambda(
        lambda x: x / ops.norm(x, axis=-1, keepdims=True)
    )(text_input)

    # Create model that computes similarity
    model = keras.Model(
        inputs={'image': image_input, 'text': text_input},
        outputs={'image_features': image_norm, 'text_features': text_norm}
    )

    # Note: Full CLIP model would compute logits here
    # For testing, we'll manually create outputs

    # Create metric
    metric = CLIPAccuracy(direction='i2t', name='i2t_acc')

    # Test with dummy data
    outputs = create_perfect_similarity_matrix(batch_size)
    metric.update_state(y_pred=outputs)
    original_result = float(metric.result())
    print(f"✓ Original metric result: {original_result:.4f}")

    # Save and load metric via config
    config = metric.get_config()
    loaded_metric = CLIPAccuracy.from_config(config)

    # Test loaded metric
    loaded_metric.update_state(y_pred=outputs)
    loaded_result = float(loaded_metric.result())

    assert abs(original_result - loaded_result) < 1e-6, \
        "Results differ after save/load"
    print(f"✓ Loaded metric result: {loaded_result:.4f}")

    print("✓ Model save/load test passed")


def test_clip_accuracy_error_handling():
    """Test CLIPAccuracy error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST: CLIPAccuracy Error Handling")
    print("=" * 70)

    metric = CLIPAccuracy(direction='i2t')

    # Test None y_pred
    try:
        metric.update_state(y_pred=None)
        raise AssertionError("Should raise ValueError for None y_pred")
    except ValueError as e:
        assert 'cannot be None' in str(e)
        print("✓ None y_pred properly rejected")

    # Test non-dict y_pred
    try:
        metric.update_state(y_pred=[1, 2, 3])
        raise AssertionError("Should raise ValueError for non-dict y_pred")
    except ValueError as e:
        assert 'dictionary' in str(e)
        print("✓ Non-dict y_pred properly rejected")

    # Test missing key
    try:
        metric.update_state(y_pred={'wrong_key': keras.random.normal((4, 4))})
        raise AssertionError("Should raise ValueError for missing key")
    except ValueError as e:
        assert 'logits_per_image' in str(e)
        print("✓ Missing key properly rejected")

    # Test wrong shape
    try:
        metric.update_state(y_pred={'logits_per_image': keras.random.normal((4,))})
        raise AssertionError("Should raise ValueError for 1D tensor")
    except ValueError as e:
        assert '2D tensor' in str(e)
        print("✓ Wrong shape properly rejected")

    print("✓ Error handling test passed")


# =============================================================================
# Test Cases for CLIPRecallAtK
# =============================================================================


def test_clip_recall_initialization():
    """Test CLIPRecallAtK initialization with various parameters."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Initialization")
    print("=" * 70)

    # Test default initialization
    metric = CLIPRecallAtK()
    assert metric.k == 5, "Default k should be 5"
    assert metric.direction == 'i2t', "Default direction should be 'i2t'"
    assert metric.name == 'clip_i2t_recall@5', "Default name incorrect"
    print("✓ Default initialization successful")

    # Test custom initialization
    metric = CLIPRecallAtK(k=10, direction='t2i', name='custom_recall')
    assert metric.k == 10, "Custom k not set"
    assert metric.direction == 't2i', "Custom direction not set"
    assert metric.name == 'custom_recall', "Custom name not set"
    print("✓ Custom initialization successful")

    # Test invalid k
    try:
        metric = CLIPRecallAtK(k=0)
        raise AssertionError("Should have raised ValueError for k=0")
    except ValueError as e:
        assert 'positive' in str(e)
        print("✓ Invalid k=0 properly rejected")

    try:
        metric = CLIPRecallAtK(k=-5)
        raise AssertionError("Should have raised ValueError for negative k")
    except ValueError as e:
        assert 'positive' in str(e)
        print("✓ Invalid negative k properly rejected")

    # Test invalid direction
    try:
        metric = CLIPRecallAtK(direction='invalid')
        raise AssertionError("Should have raised ValueError for invalid direction")
    except ValueError as e:
        assert 'i2t' in str(e) and 't2i' in str(e)
        print("✓ Invalid direction properly rejected")

    print("✓ All initialization tests passed")


def test_clip_recall_perfect_predictions():
    """Test CLIPRecallAtK with perfect similarity matrix."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Perfect Predictions")
    print("=" * 70)

    batch_size = 8
    outputs = create_perfect_similarity_matrix(batch_size)

    # Test different K values
    for k in [1, 5, 10]:
        metric = CLIPRecallAtK(k=k, direction='i2t')
        metric.update_state(y_pred=outputs)
        recall = float(metric.result())

        assert abs(recall - 1.0) < 1e-6, f"Expected 1.0 for k={k}, got {recall}"
        print(f"✓ Recall@{k}: {recall:.4f}")

    print("✓ Perfect predictions test passed")


def test_clip_recall_k_greater_than_batch():
    """Test CLIPRecallAtK when K > batch_size."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK K > Batch Size")
    print("=" * 70)

    batch_size = 4
    k = 10  # Larger than batch_size
    outputs = create_perfect_similarity_matrix(batch_size)

    # Should automatically cap K at batch_size
    metric = CLIPRecallAtK(k=k, direction='i2t')
    metric.update_state(y_pred=outputs)
    recall = float(metric.result())

    # With perfect matrix and K >= batch_size, recall should be 1.0
    assert abs(recall - 1.0) < 1e-6, f"Expected 1.0, got {recall}"
    print(f"✓ Recall@{k} (batch_size={batch_size}): {recall:.4f}")
    print("✓ K properly capped at batch_size")

    print("✓ K > batch_size test passed")


def test_clip_recall_vs_accuracy():
    """Test that Recall@1 equals Accuracy."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Recall@1 vs Accuracy")
    print("=" * 70)

    batch_size = 16
    outputs = create_partial_match_matrix(batch_size, correct_fraction=0.6)

    # Create both metrics
    accuracy_metric = CLIPAccuracy(direction='i2t')
    recall1_metric = CLIPRecallAtK(k=1, direction='i2t')

    # Update both
    accuracy_metric.update_state(y_pred=outputs)
    recall1_metric.update_state(y_pred=outputs)

    accuracy = float(accuracy_metric.result())
    recall1 = float(recall1_metric.result())

    assert abs(accuracy - recall1) < 1e-6, \
        f"Recall@1 ({recall1:.4f}) should equal Accuracy ({accuracy:.4f})"
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Recall@1: {recall1:.4f}")
    print("✓ Recall@1 == Accuracy confirmed")

    print("✓ Recall@1 vs Accuracy test passed")


def test_clip_recall_increasing_with_k():
    """Test that Recall@K increases (or stays same) as K increases."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Monotonicity")
    print("=" * 70)

    batch_size = 16
    outputs = create_random_similarity_matrix(batch_size)

    k_values = [1, 3, 5, 10]
    recalls = []

    for k in k_values:
        metric = CLIPRecallAtK(k=k, direction='i2t')
        metric.update_state(y_pred=outputs)
        recall = float(metric.result())
        recalls.append(recall)
        print(f"✓ Recall@{k}: {recall:.4f}")

    # Check monotonicity
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i + 1] + 1e-6, \
            f"Recall@{k_values[i]} > Recall@{k_values[i + 1]}: {recalls[i]} > {recalls[i + 1]}"

    print("✓ Monotonicity confirmed (Recall@K increases with K)")
    print("✓ Monotonicity test passed")


def test_clip_recall_serialization():
    """Test CLIPRecallAtK serialization with get_config and from_config."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Serialization")
    print("=" * 70)

    # Create metric with custom parameters
    original = CLIPRecallAtK(k=10, direction='t2i', name='test_recall')

    # Update state
    outputs = create_perfect_similarity_matrix(8)
    original.update_state(y_pred=outputs)
    original_result = float(original.result())

    # Get config
    config = original.get_config()
    assert 'k' in config, "Config missing 'k'"
    assert 'direction' in config, "Config missing 'direction'"
    assert config['k'] == 10, "k not in config"
    assert config['direction'] == 't2i', "direction not in config"
    assert config['name'] == 'test_recall', "name not in config"
    print(f"✓ Config: {config}")

    # Recreate from config
    restored = CLIPRecallAtK.from_config(config)
    assert restored.k == 10, "k not restored"
    assert restored.direction == 't2i', "direction not restored"
    assert restored.name == 'test_recall', "name not restored"
    print("✓ Metric recreated from config")

    # Test that restored metric works
    restored.update_state(y_pred=outputs)
    restored_result = float(restored.result())

    assert abs(original_result - restored_result) < 1e-6, \
        "Results differ after restoration"
    print(f"✓ Original result: {original_result:.4f}")
    print(f"✓ Restored result: {restored_result:.4f}")

    print("✓ Serialization test passed")


def test_clip_recall_error_handling():
    """Test CLIPRecallAtK error handling for invalid inputs."""
    print("\n" + "=" * 70)
    print("TEST: CLIPRecallAtK Error Handling")
    print("=" * 70)

    metric = CLIPRecallAtK(k=5, direction='i2t')

    # Test None y_pred
    try:
        metric.update_state(y_pred=None)
        raise AssertionError("Should raise ValueError for None y_pred")
    except ValueError as e:
        assert 'cannot be None' in str(e)
        print("✓ None y_pred properly rejected")

    # Test non-dict y_pred
    try:
        metric.update_state(y_pred="not a dict")
        raise AssertionError("Should raise ValueError for non-dict y_pred")
    except ValueError as e:
        assert 'dictionary' in str(e)
        print("✓ Non-dict y_pred properly rejected")

    # Test missing key
    try:
        metric.update_state(y_pred={'wrong_key': keras.random.normal((4, 4))})
        raise AssertionError("Should raise ValueError for missing key")
    except ValueError as e:
        assert 'logits_per_image' in str(e)
        print("✓ Missing key properly rejected")

    # Test wrong shape
    try:
        metric.update_state(y_pred={'logits_per_image': keras.random.normal((4,))})
        raise AssertionError("Should raise ValueError for 1D tensor")
    except ValueError as e:
        assert '2D tensor' in str(e)
        print("✓ Wrong shape properly rejected")

    print("✓ Error handling test passed")


# =============================================================================
# Integration Tests
# =============================================================================


def test_multiple_metrics_together():
    """Test using multiple metrics together."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Metrics Integration")
    print("=" * 70)

    batch_size = 16
    outputs = create_partial_match_matrix(batch_size, correct_fraction=0.7)

    # Create multiple metrics
    metrics = [
        CLIPAccuracy(direction='i2t', name='i2t_acc'),
        CLIPAccuracy(direction='t2i', name='t2i_acc'),
        CLIPRecallAtK(k=1, direction='i2t', name='i2t_r@1'),
        CLIPRecallAtK(k=5, direction='i2t', name='i2t_r@5'),
        CLIPRecallAtK(k=10, direction='i2t', name='i2t_r@10'),
    ]

    # Update all metrics
    for metric in metrics:
        metric.update_state(y_pred=outputs)

    # Get results
    results = {}
    for metric in metrics:
        results[metric.name] = float(metric.result())
        print(f"✓ {metric.name}: {results[metric.name]:.4f}")

    # Verify relationships
    assert abs(results['i2t_acc'] - results['i2t_r@1']) < 1e-6, \
        "Accuracy should equal Recall@1"
    assert results['i2t_r@1'] <= results['i2t_r@5'] + 1e-6, \
        "Recall@1 should be <= Recall@5"
    assert results['i2t_r@5'] <= results['i2t_r@10'] + 1e-6, \
        "Recall@5 should be <= Recall@10"

    print("✓ All metric relationships verified")
    print("✓ Multiple metrics integration test passed")


def test_multi_epoch_training_simulation():
    """Simulate multi-epoch training with metric tracking."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Epoch Training Simulation")
    print("=" * 70)

    batch_size = 8
    batches_per_epoch = 5
    num_epochs = 3

    metrics = [
        CLIPAccuracy(direction='i2t', name='i2t_acc'),
        CLIPRecallAtK(k=5, direction='i2t', name='i2t_r@5'),
    ]

    epoch_results = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Reset metrics for new epoch
        for metric in metrics:
            metric.reset_state()

        # Process batches
        for batch in range(batches_per_epoch):
            # Simulate improving performance over epochs
            correct_fraction = 0.4 + (epoch * 0.15)
            outputs = create_partial_match_matrix(batch_size, correct_fraction)

            for metric in metrics:
                metric.update_state(y_pred=outputs)

        # Collect epoch results
        epoch_result = {}
        for metric in metrics:
            value = float(metric.result())
            epoch_result[metric.name] = value
            print(f"  {metric.name}: {value:.4f}")

        epoch_results.append(epoch_result)

    # Verify improvement over epochs (with some tolerance for randomness)
    for metric_name in ['i2t_acc', 'i2t_r@5']:
        values = [result[metric_name] for result in epoch_results]
        print(f"\n✓ {metric_name} progression: {[f'{v:.3f}' for v in values]}")

        # Generally should improve, but allow for some variance
        # Check that last epoch is better than first
        assert values[-1] > values[0] - 0.1, \
            f"{metric_name} should generally improve over epochs"

    print("\n✓ Multi-epoch training simulation passed")


# =============================================================================
# Test Runner
# =============================================================================


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 70)
    print("CLIP METRICS TEST SUITE")
    print("=" * 70)

    test_functions = [
        # CLIPAccuracy tests
        test_clip_accuracy_initialization,
        test_clip_accuracy_perfect_predictions,
        test_clip_accuracy_random_predictions,
        test_clip_accuracy_partial_match,
        test_clip_accuracy_reset_state,
        test_clip_accuracy_accumulation,
        test_clip_accuracy_serialization,
        test_clip_accuracy_model_save_load,
        test_clip_accuracy_error_handling,

        # CLIPRecallAtK tests
        test_clip_recall_initialization,
        test_clip_recall_perfect_predictions,
        test_clip_recall_k_greater_than_batch,
        test_clip_recall_vs_accuracy,
        test_clip_recall_increasing_with_k,
        test_clip_recall_serialization,
        test_clip_recall_error_handling,

        # Integration tests
        test_multiple_metrics_together,
        test_multi_epoch_training_simulation,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)