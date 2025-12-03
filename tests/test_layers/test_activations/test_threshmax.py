"""
Test suite for ThreshMax implementation.

This module provides comprehensive tests for:
- ThreshMax layer configuration and initialization
- Sparse probability distribution generation (Rank Preservation)
- Learnable slope dynamics (Gradient flow)
- Natural equilibrium handling (Uniform inputs)
- Serialization and deserialization
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Tuple

from dl_techniques.layers.activations.thresh_max import ThreshMax, thresh_max


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_logits() -> tf.Tensor:
    """Generate sample logits tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 10))  # Batch of 2, 10 classes


@pytest.fixture
def clear_winner_logits() -> tf.Tensor:
    """Generate logits with clear winner (should produce sparse output)."""
    return tf.constant([
        [1.0, 3.0, 0.5, -1.0],  # Class 1 is clear winner
        [-0.5, 2.5, 0.0, -2.0]  # Class 1 is clear winner
    ], dtype=tf.float32)


@pytest.fixture
def uniform_logits() -> tf.Tensor:
    """Generate uniform logits (should result in uniform output)."""
    return tf.constant([
        [2.0, 2.0, 2.0, 2.0],  # Perfectly uniform
        [1.5, 1.5, 1.5, 1.5]   # Perfectly uniform
    ], dtype=tf.float32)


@pytest.fixture
def default_params() -> Dict[str, Any]:
    """Default parameters for ThreshMax."""
    return {
        "axis": -1,
        "slope": 10.0,
        "epsilon": 1e-12,
        "trainable_slope": False
    }


# ---------------------------------------------------------------------
# Initialization & Configuration Tests
# ---------------------------------------------------------------------

def test_initialization(default_params: Dict[str, Any]) -> None:
    """Test initialization with default parameters."""
    layer = ThreshMax(**default_params)
    assert layer.axis == default_params["axis"]
    assert layer.epsilon == default_params["epsilon"]
    assert layer.slope_initial_value == default_params["slope"]
    assert layer.trainable_slope == default_params["trainable_slope"]
    # Check that weight is not created in init
    assert layer.slope_weight is None


def test_invalid_parameters() -> None:
    """Test that invalid parameters raise errors."""
    # Test epsilon <= 0
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ThreshMax(epsilon=0.0)

    # Test slope <= 0
    with pytest.raises(ValueError, match="slope must be positive"):
        ThreshMax(slope=-1.0)


def test_build_fixed_slope(default_params: Dict[str, Any]) -> None:
    """Test build process when slope is fixed (not trainable)."""
    layer = ThreshMax(**default_params)
    layer.build((None, 10))
    
    # Should be built but have no trainable weights
    assert layer.built
    assert len(layer.trainable_weights) == 0
    assert layer.slope_weight is None


def test_build_trainable_slope() -> None:
    """Test build process when slope is trainable."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    layer.build((None, 10))
    
    # Should have exactly one trainable weight
    assert len(layer.trainable_weights) == 1
    weight = layer.trainable_weights[0]
    
    assert "slope" in weight.name
    # Check if initialized correctly to the passed slope value
    # Note: We convert to numpy for value check
    assert np.isclose(weight.numpy(), 5.0)


# ---------------------------------------------------------------------
# Core Logic & Math Tests
# ---------------------------------------------------------------------

def test_sum_to_one(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test if output probabilities sum to 1."""
    layer = ThreshMax(**default_params)
    outputs = layer(sample_logits)
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_rank_preservation_sparsity(clear_winner_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that clear winners produce sparser distributions while preserving rank."""
    layer = ThreshMax(**default_params)

    # Get ThreshMax output
    thresh_max_output = layer(clear_winner_logits)

    # Get standard softmax output for comparison
    standard_softmax = tf.nn.softmax(clear_winner_logits, axis=-1)

    # 1. Check Rank Preservation:
    # The indices of the max values should be identical for both
    thresh_argmax = tf.argmax(thresh_max_output, axis=-1)
    standard_argmax = tf.argmax(standard_softmax, axis=-1)
    assert np.array_equal(thresh_argmax.numpy(), standard_argmax.numpy())

    # 2. Check Sparsity/Sharpening:
    # ThreshMax peak probability should be >= Standard Softmax peak
    thresh_max_peaks = tf.reduce_max(thresh_max_output, axis=-1)
    standard_peaks = tf.reduce_max(standard_softmax, axis=-1)
    
    assert tf.reduce_all(tf.greater_equal(thresh_max_peaks, standard_peaks))


def test_uniform_equilibrium(uniform_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """
    Test that uniform inputs naturally result in uniform outputs.
    
    With the optimized math (no explicit fallback), uniform inputs should
    produce:
    1. softmax(x) = 1/N
    2. diff = 0
    3. gate = 0.5
    4. gated = 1/N * 0.5
    5. norm = (1/N * 0.5) / sum(1/N * 0.5) = 1/N
    """
    layer = ThreshMax(**default_params)
    outputs = layer(uniform_logits)
    
    num_classes = uniform_logits.shape[-1]
    expected_prob = 1.0 / num_classes
    
    # Check that all outputs are approximately 1/N
    assert np.allclose(outputs.numpy(), expected_prob, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------
# Learnability & Gradient Tests
# ---------------------------------------------------------------------

def test_gradient_flow_to_inputs(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test gradient flow propagates back to inputs."""
    layer = ThreshMax(**default_params)

    with tf.GradientTape() as tape:
        tape.watch(sample_logits)
        output = layer(sample_logits)
        loss = tf.reduce_sum(output * output) # Dummy loss

    gradients = tape.gradient(loss, sample_logits)

    assert gradients is not None
    assert tf.reduce_all(tf.math.is_finite(gradients))


def test_gradient_flow_to_slope(sample_logits: tf.Tensor) -> None:
    """Test gradient flow propagates to the slope weight when trainable."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    
    with tf.GradientTape() as tape:
        # Build layer dynamically
        output = layer(sample_logits)
        loss = tf.reduce_sum(output) # Dummy loss

    # Calculate gradient with respect to trainable weights (slope)
    gradients = tape.gradient(loss, layer.trainable_weights)
    
    # Gradient should exist for the slope parameter
    assert len(gradients) == 1
    assert gradients[0] is not None
    assert tf.math.is_finite(gradients[0])


# ---------------------------------------------------------------------
# Functional Interface Tests
# ---------------------------------------------------------------------

def test_functional_interface_basics(clear_winner_logits: tf.Tensor) -> None:
    """Test the functional interface thresh_max."""
    # Test with default parameters
    output = thresh_max(clear_winner_logits)

    # Should have same shape
    assert output.shape == clear_winner_logits.shape

    # Should sum to 1
    row_sums = tf.reduce_sum(output, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_functional_interface_consistency(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that functional interface produces same results as layer (fixed mode)."""
    # Layer output
    layer = ThreshMax(**default_params)
    layer_output = layer(sample_logits)

    # Functional output
    # Note: functional interface doesn't accept 'trainable_slope'
    func_params = {k: v for k, v in default_params.items() if k != "trainable_slope"}
    func_output = thresh_max(sample_logits, **func_params)

    assert np.allclose(layer_output.numpy(), func_output.numpy(), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------
# Serialization & Integration Tests
# ---------------------------------------------------------------------

def test_serialization(default_params: Dict[str, Any]) -> None:
    """Test serialization including new trainable_slope parameter."""
    # Initialize with specific non-default values to verify restoration
    original_layer = ThreshMax(
        axis=1, 
        slope=5.0, 
        trainable_slope=True,
        epsilon=1e-9
    )

    config = original_layer.get_config()
    restored_layer = ThreshMax.from_config(config)

    assert restored_layer.axis == 1
    assert restored_layer.slope_initial_value == 5.0
    assert restored_layer.trainable_slope is True
    assert restored_layer.epsilon == 1e-9


def test_model_save_load_keras_v3(sample_logits: tf.Tensor, tmp_path) -> None:
    """Test saving and loading a Keras 3 model with ThreshMax."""
    input_shape = sample_logits.shape[1:]
    inputs = keras.Input(shape=input_shape)
    
    # Use trainable version to test complex state saving
    layer = ThreshMax(trainable_slope=True)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Force build and variable creation
    model(sample_logits)

    # Save model
    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    # Load model
    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"ThreshMax": ThreshMax}
    )

    # Verify outputs match
    original_out = model(sample_logits).numpy()
    loaded_out = loaded_model(sample_logits).numpy()
    
    assert np.allclose(original_out, loaded_out, rtol=1e-6, atol=1e-6)
    
    # Verify the slope weight was loaded correctly
    original_slope = model.layers[1].slope_weight.numpy()
    loaded_slope = loaded_model.layers[1].slope_weight.numpy()
    assert np.isclose(original_slope, loaded_slope)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
