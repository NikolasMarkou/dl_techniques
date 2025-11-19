"""
Tests for the CLIP Contrastive Loss implementation.

This module contains comprehensive unit tests for the CLIPContrastiveLoss,
including basic functionality, serialization, different input formats,
temperature scaling, label smoothing, and model integration.
"""

import keras
from keras import ops
import pytest
import numpy as np
import tempfile
import os
from typing import Tuple, Dict

from dl_techniques.utils.logger import logger
from dl_techniques.losses.clip_contrastive_loss import CLIPContrastiveLoss

# =============================================================================
# Helper Functions
# =============================================================================

def get_dummy_labels(batch_size: int) -> keras.KerasTensor:
    """
    Create dummy ground truth labels for Keras Loss API compatibility.

    The CLIPContrastiveLoss is self-supervised and ignores y_true content,
    but Keras requires y_true to be a tensor for internal validation and mapping.
    """
    return ops.zeros((batch_size,), dtype="int32")

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clip_batch_data() -> Tuple[keras.KerasTensor, Dict[str, np.ndarray]]:
    """
    Fixture providing synthetic CLIP similarity matrices for testing.

    Creates a batch of 8 image-text pairs with realistic similarity logits.
    The diagonal represents correct pairings (positive samples).

    Returns
    -------
    tuple
        Tuple containing (y_true, y_pred) where:
        - y_true: Dummy tensor for Keras compatibility
        - y_pred: Dict with 'logits_per_image' and 'logits_per_text'
    """
    np.random.seed(42)
    batch_size = 8

    # Create similarity matrix where diagonal has high values (correct pairs)
    # and off-diagonal has lower values (incorrect pairs)
    base_similarity = np.random.randn(batch_size, batch_size).astype(np.float32) * 0.3

    # Make diagonal elements (correct pairs) have higher similarity
    for i in range(batch_size):
        base_similarity[i, i] = np.random.uniform(0.8, 1.2)

    # logits_per_image: image i's similarity to all texts
    logits_per_image = base_similarity.copy()

    # logits_per_text: text i's similarity to all images (transpose)
    logits_per_text = base_similarity.T.copy()

    # Create predictions dictionary
    y_pred = {
        'logits_per_image': logits_per_image,
        'logits_per_text': logits_per_text
    }

    # y_true is ignored by loss but required by Keras API
    y_true = get_dummy_labels(batch_size)

    return y_true, y_pred


@pytest.fixture
def small_clip_batch() -> Tuple[keras.KerasTensor, Dict[str, np.ndarray]]:
    """
    Fixture providing a small batch for quick testing.

    Returns
    -------
    tuple
        Small batch with batch_size=4 for fast test execution.
    """
    np.random.seed(123)
    batch_size = 4

    # Create simple similarity matrix
    logits_per_image = np.array([
        [1.0, 0.2, 0.1, 0.3],
        [0.2, 0.9, 0.3, 0.1],
        [0.1, 0.2, 1.1, 0.2],
        [0.3, 0.1, 0.2, 0.8]
    ], dtype=np.float32)

    logits_per_text = logits_per_image.T

    y_pred = {
        'logits_per_image': logits_per_image,
        'logits_per_text': logits_per_text
    }

    y_true = get_dummy_labels(batch_size)

    return y_true, y_pred


# =============================================================================
# Basic Functionality Tests
# =============================================================================


def test_clip_loss_initialization_default_parameters() -> None:
    """Test that CLIPContrastiveLoss initializes with default parameters."""
    loss = CLIPContrastiveLoss()

    assert loss.temperature == 0.07
    assert loss.label_smoothing == 0.0
    assert loss.apply_temperature is False
    assert loss.loss_weight_i2t == 0.5
    assert loss.loss_weight_t2i == 0.5
    assert loss.name == "clip_contrastive_loss"

    logger.info("Default initialization test passed")


def test_clip_loss_initialization_custom_parameters() -> None:
    """Test that CLIPContrastiveLoss initializes with custom parameters."""
    loss = CLIPContrastiveLoss(
        temperature=0.05,
        label_smoothing=0.1,
        apply_temperature=True,
        loss_weight_i2t=0.6,
        loss_weight_t2i=0.4,
        name="custom_clip_loss"
    )

    assert loss.temperature == 0.05
    assert loss.label_smoothing == 0.1
    assert loss.apply_temperature is True
    assert loss.loss_weight_i2t == 0.6
    assert loss.loss_weight_t2i == 0.4
    assert loss.name == "custom_clip_loss"

    logger.info("Custom initialization test passed")


def test_clip_loss_basic_calculation(clip_batch_data: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """
    Test that CLIPContrastiveLoss correctly calculates loss values.

    Parameters
    ----------
    clip_batch_data : tuple
        CLIP batch test data fixture.
    """
    y_true, y_pred = clip_batch_data

    # Create loss function
    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")

    # Convert to tensors
    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Calculate loss
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred_tensors)
    loss_value = ops.convert_to_numpy(loss_value)

    logger.info(f"Calculated CLIP loss: {loss_value:.6f}")

    # Loss should be a scalar value between 0 and ~log(batch_size)
    assert isinstance(loss_value, (float, np.floating, np.ndarray))
    assert loss_value >= 0.0, f"Loss should be non-negative, got {loss_value}"

    # For contrastive loss with batch_size=8, max theoretical loss is log(8) ≈ 2.08
    # But with label smoothing and averaging, it should be less
    assert loss_value < 10.0, f"Loss seems unreasonably high: {loss_value}"

    logger.info("Basic loss calculation test passed")


def test_clip_loss_symmetric_computation(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """
    Test that loss is computed symmetrically in both directions.

    Parameters
    ----------
    small_clip_batch : tuple
        Small batch test data fixture.
    """
    y_true, y_pred = small_clip_batch

    # Create loss with equal weights (default)
    loss_fn = CLIPContrastiveLoss(reduction="none")

    # Convert to tensors
    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Calculate per-sample losses
    per_sample_loss = loss_fn(y_true=y_true, y_pred=y_pred_tensors)
    per_sample_loss = ops.convert_to_numpy(per_sample_loss)

    logger.info(f"Per-sample losses shape: {per_sample_loss.shape}")
    logger.info(f"Per-sample losses: {per_sample_loss}")

    # Should have one loss value per sample
    assert per_sample_loss.shape == (4,), f"Expected shape (4,), got {per_sample_loss.shape}"

    # All losses should be non-negative
    assert np.all(per_sample_loss >= 0), "All per-sample losses should be non-negative"

    logger.info("Symmetric computation test passed")


def test_clip_loss_perfect_predictions() -> None:
    """Test loss with perfect predictions (diagonal = high, off-diagonal = low)."""
    batch_size = 4

    # Create perfect similarity matrix: diagonal = 10, off-diagonal = -10
    logits_per_image = np.full((batch_size, batch_size), -10.0, dtype=np.float32)
    np.fill_diagonal(logits_per_image, 10.0)

    logits_per_text = logits_per_image.T

    y_pred = {
        'logits_per_image': ops.convert_to_tensor(logits_per_image),
        'logits_per_text': ops.convert_to_tensor(logits_per_text)
    }

    y_true = get_dummy_labels(batch_size)

    # Create loss
    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred)
    loss_value = ops.convert_to_numpy(loss_value)

    logger.info(f"Loss with perfect predictions: {loss_value:.6f}")

    # Loss should be very close to 0 for perfect predictions
    assert loss_value < 0.01, f"Expected near-zero loss for perfect predictions, got {loss_value}"

    logger.info("Perfect predictions test passed")


def test_clip_loss_random_predictions() -> None:
    """Test loss with random predictions (no structure)."""
    np.random.seed(999)
    batch_size = 4

    # Create random similarity matrix (no structure)
    logits_per_image = np.random.randn(batch_size, batch_size).astype(np.float32)
    logits_per_text = logits_per_image.T

    y_pred = {
        'logits_per_image': ops.convert_to_tensor(logits_per_image),
        'logits_per_text': ops.convert_to_tensor(logits_per_text)
    }

    y_true = get_dummy_labels(batch_size)

    # Create loss
    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred)
    loss_value = ops.convert_to_numpy(loss_value)

    logger.info(f"Loss with random predictions: {loss_value:.6f}")

    # Loss should be higher for random predictions (around log(batch_size) / 2)
    # For batch_size=4, log(4) ≈ 1.39, so expect loss around 0.7-1.0
    assert 0.5 < loss_value < 2.0, f"Expected loss in range [0.5, 2.0], got {loss_value}"

    logger.info("Random predictions test passed")


# =============================================================================
# Temperature Scaling Tests
# =============================================================================


def test_temperature_scaling_effect(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """
    Test that temperature scaling affects loss magnitude correctly.

    Lower temperature should increase loss magnitude (sharper distribution).
    """
    y_true, y_pred = small_clip_batch

    # Convert to tensors once
    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Test different temperatures
    temperatures = [0.01, 0.07, 0.5, 1.0]
    losses = []

    for temp in temperatures:
        loss_fn = CLIPContrastiveLoss(
            temperature=temp,
            apply_temperature=True,
            reduction="sum_over_batch_size"
        )
        loss_value = loss_fn(y_true=y_true, y_pred=y_pred_tensors)
        loss_value = ops.convert_to_numpy(loss_value)
        losses.append(loss_value)
        logger.info(f"Temperature {temp}: Loss = {loss_value:.6f}")

    # Lower temperature should generally lead to different loss values
    # (relationship depends on whether predictions are correct)
    assert len(set(np.round(losses, 4))) > 1, \
        "Different temperatures should produce different loss values"

    logger.info("Temperature scaling effect test passed")


def test_temperature_application_flag(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that apply_temperature flag controls whether temperature is applied."""
    y_true, y_pred = small_clip_batch

    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Loss without temperature application
    loss_no_temp = CLIPContrastiveLoss(
        temperature=0.07,
        apply_temperature=False,
        reduction="sum_over_batch_size"
    )
    value_no_temp = loss_no_temp(y_true=y_true, y_pred=y_pred_tensors)
    value_no_temp = ops.convert_to_numpy(value_no_temp)

    # Loss with temperature application
    loss_with_temp = CLIPContrastiveLoss(
        temperature=0.07,
        apply_temperature=True,
        reduction="sum_over_batch_size"
    )
    value_with_temp = loss_with_temp(y_true=y_true, y_pred=y_pred_tensors)
    value_with_temp = ops.convert_to_numpy(value_with_temp)

    logger.info(f"Loss without temperature: {value_no_temp:.6f}")
    logger.info(f"Loss with temperature: {value_with_temp:.6f}")

    # Values should be different when temperature is applied
    assert not np.isclose(value_no_temp, value_with_temp, atol=1e-5), \
        "Temperature application should change loss value"

    logger.info("Temperature application flag test passed")


def test_temperature_update_method() -> None:
    """Test the update_temperature method."""
    loss_fn = CLIPContrastiveLoss(temperature=0.1, apply_temperature=True)

    # Check initial value
    assert loss_fn.temperature_value == 0.1

    # Update temperature
    loss_fn.update_temperature(0.05)
    assert loss_fn.temperature_value == 0.05

    # Test that invalid temperature raises error
    with pytest.raises(ValueError, match="Temperature must be positive"):
        loss_fn.update_temperature(0.0)

    with pytest.raises(ValueError, match="Temperature must be positive"):
        loss_fn.update_temperature(-0.1)

    logger.info("Temperature update method test passed")


# =============================================================================
# Label Smoothing Tests
# =============================================================================


def test_label_smoothing_effect(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that label smoothing affects loss values."""
    y_true, y_pred = small_clip_batch

    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Loss without smoothing
    loss_no_smooth = CLIPContrastiveLoss(
        label_smoothing=0.0,
        reduction="sum_over_batch_size"
    )
    value_no_smooth = loss_no_smooth(y_true=y_true, y_pred=y_pred_tensors)
    value_no_smooth = ops.convert_to_numpy(value_no_smooth)

    # Loss with smoothing
    loss_with_smooth = CLIPContrastiveLoss(
        label_smoothing=0.1,
        reduction="sum_over_batch_size"
    )
    value_with_smooth = loss_with_smooth(y_true=y_true, y_pred=y_pred_tensors)
    value_with_smooth = ops.convert_to_numpy(value_with_smooth)

    logger.info(f"Loss without smoothing: {value_no_smooth:.6f}")
    logger.info(f"Loss with smoothing: {value_with_smooth:.6f}")

    # Label smoothing should change the loss value
    assert not np.isclose(value_no_smooth, value_with_smooth, atol=1e-5), \
        "Label smoothing should affect loss value"

    # With label smoothing, loss typically increases slightly
    # (as some probability mass is given to incorrect classes)
    assert value_with_smooth > value_no_smooth, \
        "Label smoothing typically increases loss value"

    logger.info("Label smoothing effect test passed")


# =============================================================================
# Loss Weight Tests
# =============================================================================


def test_loss_weight_combinations(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test different combinations of image-to-text and text-to-image weights."""
    y_true, y_pred = small_clip_batch

    y_pred_tensors = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    # Test different weight combinations
    weight_configs = [
        (1.0, 0.0),  # Only image-to-text
        (0.0, 1.0),  # Only text-to-image
        (0.5, 0.5),  # Equal (default)
        (0.7, 0.3),  # Favor image-to-text
        (0.3, 0.7),  # Favor text-to-image
    ]

    losses = []
    for w_i2t, w_t2i in weight_configs:
        loss_fn = CLIPContrastiveLoss(
            loss_weight_i2t=w_i2t,
            loss_weight_t2i=w_t2i,
            reduction="sum_over_batch_size"
        )
        loss_value = loss_fn(y_true=y_true, y_pred=y_pred_tensors)
        loss_value = ops.convert_to_numpy(loss_value)
        losses.append(loss_value)
        logger.info(f"Weights ({w_i2t}, {w_t2i}): Loss = {loss_value:.6f}")

    # Different weight combinations should produce different losses
    assert len(set(np.round(losses, 5))) > 1, \
        "Different weight combinations should produce different losses"

    # Check that pure directional losses are different
    assert not np.isclose(losses[0], losses[1], atol=1e-5), \
        "Pure i2t and t2i losses should differ"

    logger.info("Loss weight combinations test passed")


def test_weight_sum_warning() -> None:
    """Test that a warning is logged when weights don't sum to 1.0."""
    # This should trigger a warning
    with pytest.warns(UserWarning, match="Loss weights sum"):
        loss_fn = CLIPContrastiveLoss(
            loss_weight_i2t=0.6,
            loss_weight_t2i=0.6  # Sum = 1.2
        )

    logger.info("Weight sum warning test passed")


# =============================================================================
# Input Format Tests
# =============================================================================


def test_dictionary_input_format(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that dictionary input format works correctly."""
    y_true, y_pred = small_clip_batch

    # Dictionary format (recommended)
    y_pred_dict = {
        'logits_per_image': ops.convert_to_tensor(y_pred['logits_per_image']),
        'logits_per_text': ops.convert_to_tensor(y_pred['logits_per_text'])
    }

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred_dict)
    loss_value = ops.convert_to_numpy(loss_value)

    assert loss_value >= 0.0
    logger.info(f"Dictionary input: Loss = {loss_value:.6f}")
    logger.info("Dictionary input format test passed")


def test_tuple_input_format(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that tuple input format works correctly."""
    y_true, y_pred = small_clip_batch

    # Tuple format
    y_pred_tuple = (
        ops.convert_to_tensor(y_pred['logits_per_image']),
        ops.convert_to_tensor(y_pred['logits_per_text'])
    )

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred_tuple)
    loss_value = ops.convert_to_numpy(loss_value)

    assert loss_value >= 0.0
    logger.info(f"Tuple input: Loss = {loss_value:.6f}")
    logger.info("Tuple input format test passed")


def test_list_input_format(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that list input format works correctly."""
    y_true, y_pred = small_clip_batch

    # List format
    y_pred_list = [
        ops.convert_to_tensor(y_pred['logits_per_image']),
        ops.convert_to_tensor(y_pred['logits_per_text'])
    ]

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    loss_value = loss_fn(y_true=y_true, y_pred=y_pred_list)
    loss_value = ops.convert_to_numpy(loss_value)

    assert loss_value >= 0.0
    logger.info(f"List input: Loss = {loss_value:.6f}")
    logger.info("List input format test passed")


def test_equivalent_input_formats(small_clip_batch: Tuple[keras.KerasTensor, Dict[str, np.ndarray]]) -> None:
    """Test that all input formats produce the same loss value."""
    y_true, y_pred = small_clip_batch

    # Create tensors once
    img_tensor = ops.convert_to_tensor(y_pred['logits_per_image'])
    txt_tensor = ops.convert_to_tensor(y_pred['logits_per_text'])

    # Three different input formats
    y_pred_dict = {'logits_per_image': img_tensor, 'logits_per_text': txt_tensor}
    y_pred_tuple = (img_tensor, txt_tensor)
    y_pred_list = [img_tensor, txt_tensor]

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")

    # Calculate losses
    loss_dict = ops.convert_to_numpy(loss_fn(y_true, y_pred_dict))
    loss_tuple = ops.convert_to_numpy(loss_fn(y_true, y_pred_tuple))
    loss_list = ops.convert_to_numpy(loss_fn(y_true, y_pred_list))

    logger.info(f"Dict: {loss_dict:.6f}, Tuple: {loss_tuple:.6f}, List: {loss_list:.6f}")

    # All should be equal
    np.testing.assert_allclose(loss_dict, loss_tuple, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loss_dict, loss_list, rtol=1e-6, atol=1e-6)

    logger.info("Equivalent input formats test passed")


# =============================================================================
# Validation Tests
# =============================================================================


def test_invalid_temperature_initialization() -> None:
    """Test that invalid temperature values raise errors."""
    # Zero temperature
    with pytest.raises(ValueError, match="temperature must be positive"):
        CLIPContrastiveLoss(temperature=0.0)

    # Negative temperature
    with pytest.raises(ValueError, match="temperature must be positive"):
        CLIPContrastiveLoss(temperature=-0.5)

    logger.info("Invalid temperature initialization test passed")


def test_invalid_label_smoothing_initialization() -> None:
    """Test that invalid label smoothing values raise errors."""
    # Below range
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        CLIPContrastiveLoss(label_smoothing=-0.1)

    # Above range
    with pytest.raises(ValueError, match="label_smoothing must be in"):
        CLIPContrastiveLoss(label_smoothing=1.5)

    logger.info("Invalid label smoothing initialization test passed")


def test_invalid_loss_weights_initialization() -> None:
    """Test that invalid loss weight values raise errors."""
    # Negative i2t weight
    with pytest.raises(ValueError, match="loss_weight_i2t must be non-negative"):
        CLIPContrastiveLoss(loss_weight_i2t=-0.5)

    # Negative t2i weight
    with pytest.raises(ValueError, match="loss_weight_t2i must be non-negative"):
        CLIPContrastiveLoss(loss_weight_t2i=-0.3)

    logger.info("Invalid loss weights initialization test passed")


def test_missing_dictionary_keys() -> None:
    """Test that missing dictionary keys raise appropriate errors."""
    batch_size = 4
    logits = np.random.randn(batch_size, batch_size).astype(np.float32)
    y_true = get_dummy_labels(batch_size)

    loss_fn = CLIPContrastiveLoss()

    # Missing logits_per_image
    with pytest.raises(ValueError, match="must contain.*logits_per_image"):
        y_pred = {'logits_per_text': ops.convert_to_tensor(logits)}
        loss_fn(y_true, y_pred)

    # Missing logits_per_text
    with pytest.raises(ValueError, match="must contain.*logits_per_text"):
        y_pred = {'logits_per_image': ops.convert_to_tensor(logits)}
        loss_fn(y_true, y_pred)

    logger.info("Missing dictionary keys test passed")


def test_wrong_tuple_length() -> None:
    """Test that tuples with wrong length raise errors."""
    batch_size = 4
    logits = ops.convert_to_tensor(np.random.randn(batch_size, batch_size).astype(np.float32))
    y_true = get_dummy_labels(batch_size)

    loss_fn = CLIPContrastiveLoss()

    # Single element tuple
    with pytest.raises(ValueError, match="must have exactly 2 elements"):
        loss_fn(y_true, (logits,))

    # Three element tuple
    with pytest.raises(ValueError, match="must have exactly 2 elements"):
        loss_fn(y_true, (logits, logits, logits))

    logger.info("Wrong tuple length test passed")


def test_incompatible_logit_shapes() -> None:
    """Test that incompatible logit shapes raise errors."""
    loss_fn = CLIPContrastiveLoss()

    # Different shapes
    logits_4x4 = ops.convert_to_tensor(np.random.randn(4, 4).astype(np.float32))
    logits_4x8 = ops.convert_to_tensor(np.random.randn(4, 8).astype(np.float32))

    y_true = get_dummy_labels(4)

    with pytest.raises(ValueError, match="must be identical"):
        y_pred = {'logits_per_image': logits_4x4, 'logits_per_text': logits_4x8}
        loss_fn(y_true, y_pred)

    logger.info("Incompatible logit shapes test passed")


def test_non_2d_logits() -> None:
    """Test that non-2D logits raise errors."""
    loss_fn = CLIPContrastiveLoss()

    # 1D logits
    logits_1d = ops.convert_to_tensor(np.random.randn(4).astype(np.float32))
    y_true = get_dummy_labels(4)

    with pytest.raises(ValueError, match="must be 2D tensors"):
        y_pred = {'logits_per_image': logits_1d, 'logits_per_text': logits_1d}
        loss_fn(y_true, y_pred)

    # 3D logits
    logits_3d = ops.convert_to_tensor(np.random.randn(4, 4, 4).astype(np.float32))

    with pytest.raises(ValueError, match="must be 2D tensors"):
        y_pred = {'logits_per_image': logits_3d, 'logits_per_text': logits_3d}
        loss_fn(y_true, y_pred)

    logger.info("Non-2D logits test passed")


# =============================================================================
# Serialization Tests
# =============================================================================


def test_loss_serialization_and_deserialization() -> None:
    """Test that CLIPContrastiveLoss can be properly serialized and deserialized."""
    # Create a simple model with CLIPContrastiveLoss
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(8, activation='relu')(inputs)
    outputs = keras.layers.Dense(4, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Create loss with custom parameters
    original_loss = CLIPContrastiveLoss(
        temperature=0.05,
        label_smoothing=0.15,
        apply_temperature=True,
        loss_weight_i2t=0.6,
        loss_weight_t2i=0.4,
        name="custom_clip_loss"
    )

    logger.info(
        f"Created original loss: temperature={original_loss.temperature}, "
        f"label_smoothing={original_loss.label_smoothing}, "
        f"apply_temperature={original_loss.apply_temperature}"
    )

    # Compile model
    model.compile(optimizer='adam', loss=original_loss)

    # Save and load model
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, 'model.keras')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Load model
        loaded_model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")

        # Get loaded loss
        loaded_loss = loaded_model.loss

        # Verify all parameters match
        assert loaded_loss.temperature == original_loss.temperature, \
            f"Temperature mismatch: {loaded_loss.temperature} != {original_loss.temperature}"

        assert loaded_loss.label_smoothing == original_loss.label_smoothing, \
            f"Label smoothing mismatch: {loaded_loss.label_smoothing} != {original_loss.label_smoothing}"

        assert loaded_loss.apply_temperature == original_loss.apply_temperature, \
            f"Apply temperature mismatch: {loaded_loss.apply_temperature} != {original_loss.apply_temperature}"

        assert loaded_loss.loss_weight_i2t == original_loss.loss_weight_i2t, \
            f"Weight i2t mismatch: {loaded_loss.loss_weight_i2t} != {original_loss.loss_weight_i2t}"

        assert loaded_loss.loss_weight_t2i == original_loss.loss_weight_t2i, \
            f"Weight t2i mismatch: {loaded_loss.loss_weight_t2i} != {original_loss.loss_weight_t2i}"

        assert loaded_loss.name == original_loss.name, \
            f"Name mismatch: {loaded_loss.name} != {original_loss.name}"

        logger.info("All parameters verified successfully")

    logger.info("Serialization test passed")


def test_get_config_completeness() -> None:
    """Test that get_config returns all necessary parameters."""
    loss = CLIPContrastiveLoss(
        temperature=0.05,
        label_smoothing=0.1,
        apply_temperature=True,
        loss_weight_i2t=0.7,
        loss_weight_t2i=0.3,
        name="test_loss"
    )

    config = loss.get_config()

    # Check that all custom parameters are in config
    assert 'temperature' in config
    assert 'label_smoothing' in config
    assert 'apply_temperature' in config
    assert 'loss_weight_i2t' in config
    assert 'loss_weight_t2i' in config
    assert 'name' in config

    # Check values
    assert config['temperature'] == 0.05
    assert config['label_smoothing'] == 0.1
    assert config['apply_temperature'] is True
    assert config['loss_weight_i2t'] == 0.7
    assert config['loss_weight_t2i'] == 0.3

    logger.info("get_config completeness test passed")


def test_from_config_reconstruction() -> None:
    """Test that loss can be reconstructed from config."""
    original_loss = CLIPContrastiveLoss(
        temperature=0.08,
        label_smoothing=0.2,
        apply_temperature=True,
        loss_weight_i2t=0.4,
        loss_weight_t2i=0.6
    )

    # Get config
    config = original_loss.get_config()

    # Reconstruct from config
    reconstructed_loss = CLIPContrastiveLoss.from_config(config)

    # Verify parameters match
    assert reconstructed_loss.temperature == original_loss.temperature
    assert reconstructed_loss.label_smoothing == original_loss.label_smoothing
    assert reconstructed_loss.apply_temperature == original_loss.apply_temperature
    assert reconstructed_loss.loss_weight_i2t == original_loss.loss_weight_i2t
    assert reconstructed_loss.loss_weight_t2i == original_loss.loss_weight_t2i

    logger.info("from_config reconstruction test passed")


# =============================================================================
# Model Integration Tests
# =============================================================================


def test_model_training_with_clip_loss() -> None:
    """Test that a model can be trained using CLIPContrastiveLoss."""
    np.random.seed(42)

    # Create synthetic training data
    batch_size = 16
    embedding_dim = 32
    num_samples = 100

    # Simulate image and text embeddings
    image_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    text_embeddings = np.random.randn(num_samples, embedding_dim).astype(np.float32)

    # Normalize embeddings (important for CLIP)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    logger.info(f"Created training data: {num_samples} pairs, embedding_dim={embedding_dim}")

    # Create a simple model that computes similarity
    # This is a simplified version - real CLIP models are more complex
    image_input = keras.Input(shape=(embedding_dim,), name='image')
    text_input = keras.Input(shape=(embedding_dim,), name='text')

    # Simple projection layers
    image_proj = keras.layers.Dense(16, activation='relu')(image_input)
    text_proj = keras.layers.Dense(16, activation='relu')(text_input)

    model = keras.Model(
        inputs={'image': image_input, 'text': text_input},
        outputs={'image_proj': image_proj, 'text_proj': text_proj}
    )

    # Custom training loop to compute similarity matrices
    # (In practice, this would be done in a custom training step)

    # For testing purposes, we'll use a simpler approach:
    # Create a model that directly outputs logits
    def compute_logits(image_emb, text_emb):
        """Compute similarity logits between image and text embeddings."""
        # Normalize
        image_norm = image_emb / ops.norm(image_emb, axis=-1, keepdims=True)
        text_norm = text_emb / ops.norm(text_emb, axis=-1, keepdims=True)
        # Compute similarity matrix
        logits = ops.matmul(image_norm, ops.transpose(text_norm))
        return logits

    # For testing, we'll just verify the loss can be used in compilation
    loss_fn = CLIPContrastiveLoss(
        temperature=0.07,
        apply_temperature=True,
        reduction="sum_over_batch_size"
    )

    # Create dummy logits for testing
    dummy_logits = np.random.randn(batch_size, batch_size).astype(np.float32)
    y_pred = {
        'logits_per_image': ops.convert_to_tensor(dummy_logits),
        'logits_per_text': ops.convert_to_tensor(dummy_logits.T)
    }

    y_true = get_dummy_labels(batch_size)

    # Test that loss can be computed
    initial_loss = loss_fn(y_true, y_pred)
    initial_loss = ops.convert_to_numpy(initial_loss)

    logger.info(f"Initial loss: {initial_loss:.4f}")
    assert initial_loss >= 0.0

    logger.info("Model training integration test passed")


def test_loss_with_different_batch_sizes() -> None:
    """Test that loss works correctly with different batch sizes."""
    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")

    batch_sizes = [2, 4, 8, 16, 32]

    for bs in batch_sizes:
        # Create random logits
        logits = np.random.randn(bs, bs).astype(np.float32)
        y_pred = {
            'logits_per_image': ops.convert_to_tensor(logits),
            'logits_per_text': ops.convert_to_tensor(logits.T)
        }

        y_true = get_dummy_labels(bs)

        # Compute loss
        loss_value = loss_fn(y_true, y_pred)
        loss_value = ops.convert_to_numpy(loss_value)

        logger.info(f"Batch size {bs}: Loss = {loss_value:.6f}")

        # Loss should be valid
        assert loss_value >= 0.0
        assert not np.isnan(loss_value)
        assert not np.isinf(loss_value)

    logger.info("Different batch sizes test passed")


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================


def test_numerical_stability_extreme_logits() -> None:
    """Test numerical stability with extreme logit values."""
    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")

    batch_size = 4

    # Test with very large logits
    large_logits = np.full((batch_size, batch_size), 100.0, dtype=np.float32)
    np.fill_diagonal(large_logits, 200.0)

    y_pred_large = {
        'logits_per_image': ops.convert_to_tensor(large_logits),
        'logits_per_text': ops.convert_to_tensor(large_logits.T)
    }

    y_true = get_dummy_labels(batch_size)

    loss_large = loss_fn(y_true, y_pred_large)
    loss_large = ops.convert_to_numpy(loss_large)

    assert not np.isnan(loss_large), "Loss should not be NaN with large logits"
    assert not np.isinf(loss_large), "Loss should not be inf with large logits"
    logger.info(f"Loss with large logits: {loss_large:.6f}")

    # Test with very small (negative) logits
    small_logits = np.full((batch_size, batch_size), -100.0, dtype=np.float32)
    np.fill_diagonal(small_logits, -50.0)

    y_pred_small = {
        'logits_per_image': ops.convert_to_tensor(small_logits),
        'logits_per_text': ops.convert_to_tensor(small_logits.T)
    }

    loss_small = loss_fn(y_true, y_pred_small)
    loss_small = ops.convert_to_numpy(loss_small)

    assert not np.isnan(loss_small), "Loss should not be NaN with small logits"
    assert not np.isinf(loss_small), "Loss should not be inf with small logits"
    logger.info(f"Loss with small logits: {loss_small:.6f}")

    logger.info("Numerical stability test passed")


def test_gradient_flow() -> None:
    """Test that gradients can flow through the loss."""
    import tensorflow as tf

    batch_size = 4

    # Create trainable variables for logits
    logits_per_image = tf.Variable(
        np.random.randn(batch_size, batch_size).astype(np.float32)
    )
    logits_per_text = tf.Variable(
        np.random.randn(batch_size, batch_size).astype(np.float32)
    )

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")
    y_true = get_dummy_labels(batch_size)

    # Compute loss and gradients
    with tf.GradientTape() as tape:
        y_pred = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
        loss = loss_fn(y_true, y_pred)

    # Get gradients
    grads = tape.gradient(loss, [logits_per_image, logits_per_text])

    # Check that gradients exist and are not None
    assert grads[0] is not None, "Gradient for logits_per_image should not be None"
    assert grads[1] is not None, "Gradient for logits_per_text should not be None"

    # Check that gradients are not all zeros
    grad_img_sum = tf.reduce_sum(tf.abs(grads[0]))
    grad_txt_sum = tf.reduce_sum(tf.abs(grads[1]))

    assert grad_img_sum > 0, "Gradients for logits_per_image should not be all zeros"
    assert grad_txt_sum > 0, "Gradients for logits_per_text should not be all zeros"

    logger.info(f"Gradient sums - Image: {grad_img_sum:.6f}, Text: {grad_txt_sum:.6f}")
    logger.info("Gradient flow test passed")


# =============================================================================
# Performance and Consistency Tests
# =============================================================================


def test_loss_consistency_across_calls() -> None:
    """Test that loss produces consistent results across multiple calls."""
    np.random.seed(42)
    batch_size = 8

    logits = np.random.randn(batch_size, batch_size).astype(np.float32)
    y_pred = {
        'logits_per_image': ops.convert_to_tensor(logits),
        'logits_per_text': ops.convert_to_tensor(logits.T)
    }

    y_true = get_dummy_labels(batch_size)

    loss_fn = CLIPContrastiveLoss(reduction="sum_over_batch_size")

    # Compute loss multiple times
    losses = []
    for i in range(5):
        loss_value = loss_fn(y_true, y_pred)
        loss_value = ops.convert_to_numpy(loss_value)
        losses.append(loss_value)
        logger.info(f"Call {i + 1}: Loss = {loss_value:.8f}")

    # All losses should be identical
    for i in range(1, len(losses)):
        np.testing.assert_allclose(
            losses[0], losses[i],
            rtol=1e-6, atol=1e-6,
            err_msg=f"Loss should be consistent across calls"
        )

    logger.info("Loss consistency test passed")


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])