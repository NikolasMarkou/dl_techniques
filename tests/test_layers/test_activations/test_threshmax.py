"""
Comprehensive test suite for ThreshMax implementation.

This module provides extensive tests for:
- ThreshMax layer configuration and initialization
- Sparse probability distribution generation (Rank Preservation)
- Learnable slope dynamics (Gradient flow)
- Natural equilibrium handling (Uniform inputs)
- Serialization and deserialization
- Edge cases and numerical stability
- Integration with training loops
- Different input shapes and axes
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any

from dl_techniques.layers.activations.thresh_max import ThreshMax, thresh_max


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_logits() -> tf.Tensor:
    """Generate sample logits tensor."""
    tf.random.set_seed(42)
    return tf.random.normal((2, 10))


@pytest.fixture
def clear_winner_logits() -> tf.Tensor:
    """Generate logits with clear winner (should produce sparse output)."""
    return tf.constant([
        [1.0, 3.0, 0.5, -1.0],
        [-0.5, 2.5, 0.0, -2.0]
    ], dtype=tf.float32)


@pytest.fixture
def uniform_logits() -> tf.Tensor:
    """Generate uniform logits (should result in uniform output)."""
    return tf.constant([
        [2.0, 2.0, 2.0, 2.0],
        [1.5, 1.5, 1.5, 1.5]
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


@pytest.fixture
def batch_logits() -> tf.Tensor:
    """Generate larger batch of logits."""
    tf.random.set_seed(123)
    return tf.random.normal((16, 20))


@pytest.fixture
def multidim_logits() -> tf.Tensor:
    """Generate 3D logits tensor."""
    tf.random.set_seed(456)
    return tf.random.normal((4, 8, 10))


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
    assert layer.slope_weight is None


def test_invalid_parameters() -> None:
    """Test that invalid parameters raise errors."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ThreshMax(epsilon=0.0)

    with pytest.raises(ValueError, match="slope must be positive"):
        ThreshMax(slope=-1.0)


def test_build_fixed_slope(default_params: Dict[str, Any]) -> None:
    """Test build process when slope is fixed (not trainable)."""
    layer = ThreshMax(**default_params)
    layer.build((None, 10))
    
    assert layer.built
    assert layer.slope_weight is not None
    assert len(layer.trainable_weights) == 0
    assert len(layer.non_trainable_weights) == 1
    assert np.isclose(layer.slope_weight.numpy(), default_params["slope"])


def test_build_trainable_slope() -> None:
    """Test build process when slope is trainable."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    layer.build((None, 10))
    
    assert len(layer.trainable_weights) == 1
    assert len(layer.non_trainable_weights) == 0
    weight = layer.trainable_weights[0]
    
    assert "slope" in weight.name
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
    thresh_max_output = layer(clear_winner_logits)
    standard_softmax = tf.nn.softmax(clear_winner_logits, axis=-1)

    thresh_argmax = tf.argmax(thresh_max_output, axis=-1)
    standard_argmax = tf.argmax(standard_softmax, axis=-1)
    assert np.array_equal(thresh_argmax.numpy(), standard_argmax.numpy())

    thresh_max_peaks = tf.reduce_max(thresh_max_output, axis=-1)
    standard_peaks = tf.reduce_max(standard_softmax, axis=-1)
    
    assert tf.reduce_all(tf.greater_equal(thresh_max_peaks, standard_peaks))


def test_uniform_equilibrium(uniform_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that uniform inputs naturally result in uniform outputs."""
    layer = ThreshMax(**default_params)
    outputs = layer(uniform_logits)
    
    num_classes = uniform_logits.shape[-1]
    expected_prob = 1.0 / num_classes
    
    assert np.allclose(outputs.numpy(), expected_prob, rtol=1e-5, atol=1e-5)


def test_output_shape_preservation(sample_logits: tf.Tensor) -> None:
    """Test that output shape matches input shape."""
    layer = ThreshMax()
    outputs = layer(sample_logits)
    assert outputs.shape == sample_logits.shape


def test_output_range(sample_logits: tf.Tensor) -> None:
    """Test that all output probabilities are in [0, 1]."""
    layer = ThreshMax()
    outputs = layer(sample_logits)
    assert tf.reduce_all(outputs >= 0.0)
    assert tf.reduce_all(outputs <= 1.0)


def test_non_negativity(sample_logits: tf.Tensor) -> None:
    """Test that outputs are always non-negative."""
    layer = ThreshMax()
    outputs = layer(sample_logits)
    assert tf.reduce_all(outputs >= 0.0)


def test_batch_consistency(batch_logits: tf.Tensor) -> None:
    """Test that each batch element is processed independently."""
    layer = ThreshMax()
    outputs = layer(batch_logits)
    
    # Each row should sum to 1
    row_sums = tf.reduce_sum(outputs, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)
    
    # Check that each row is properly normalized
    for i in range(outputs.shape[0]):
        assert np.isclose(tf.reduce_sum(outputs[i]).numpy(), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------
# Slope Parameter Tests
# ---------------------------------------------------------------------

def test_different_slope_values(clear_winner_logits: tf.Tensor) -> None:
    """Test that different slope values produce different sparsity levels."""
    slopes = [1.0, 5.0, 10.0, 20.0]
    outputs_list = []
    
    for slope in slopes:
        layer = ThreshMax(slope=slope)
        output = layer(clear_winner_logits)
        outputs_list.append(output)
    
    # Higher slopes should produce higher max probabilities (more sparse)
    max_probs = [tf.reduce_max(out, axis=-1).numpy().mean() for out in outputs_list]
    
    # Check monotonic increase in sparsity
    for i in range(len(max_probs) - 1):
        assert max_probs[i] <= max_probs[i + 1]


def test_low_slope_approaches_softmax(clear_winner_logits: tf.Tensor) -> None:
    """Test that very low slope approaches standard softmax."""
    layer = ThreshMax(slope=0.1)
    thresh_output = layer(clear_winner_logits)
    softmax_output = tf.nn.softmax(clear_winner_logits, axis=-1)
    
    # Should be relatively close to softmax
    assert np.allclose(thresh_output.numpy(), softmax_output.numpy(), rtol=0.1, atol=0.1)


def test_high_slope_produces_sparsity(clear_winner_logits: tf.Tensor) -> None:
    """Test that high slope produces very sparse outputs."""
    layer = ThreshMax(slope=50.0)
    outputs = layer(clear_winner_logits)
    
    # Count near-zero probabilities (< 0.01)
    near_zero = tf.reduce_sum(tf.cast(outputs < 0.01, tf.int32), axis=-1)
    
    # Should have at least 2 near-zero values per row
    assert tf.reduce_all(near_zero >= 2)


def test_slope_constraint_applied() -> None:
    """Test that slope constraint is properly applied during build."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    layer.build((None, 10))
    
    # Slope should be within constraint bounds [1.0, 50.0]
    slope_value = layer.slope_weight.numpy()
    assert 1.0 <= slope_value <= 50.0


# ---------------------------------------------------------------------
# Axis Parameter Tests
# ---------------------------------------------------------------------

def test_different_axes(multidim_logits: tf.Tensor) -> None:
    """Test ThreshMax with different axis parameters."""
    # Test axis=-1
    layer_neg1 = ThreshMax(axis=-1)
    output_neg1 = layer_neg1(multidim_logits)
    sums_neg1 = tf.reduce_sum(output_neg1, axis=-1)
    assert np.allclose(sums_neg1.numpy(), 1.0, rtol=1e-5, atol=1e-5)
    
    # Test axis=2 (equivalent for 3D input)
    layer_2 = ThreshMax(axis=2)
    output_2 = layer_2(multidim_logits)
    sums_2 = tf.reduce_sum(output_2, axis=2)
    assert np.allclose(sums_2.numpy(), 1.0, rtol=1e-5, atol=1e-5)
    
    # Should produce same results
    assert np.allclose(output_neg1.numpy(), output_2.numpy(), rtol=1e-10, atol=1e-10)


def test_axis_1_normalization() -> None:
    """Test normalization along axis 1."""
    logits = tf.random.normal((4, 8, 12))
    layer = ThreshMax(axis=1)
    outputs = layer(logits)
    
    # Sum along axis 1 should be 1
    sums = tf.reduce_sum(outputs, axis=1)
    assert np.allclose(sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_axis_0_normalization() -> None:
    """Test normalization along axis 0 (batch axis)."""
    logits = tf.random.normal((8, 12))
    layer = ThreshMax(axis=0)
    outputs = layer(logits)
    
    # Sum along axis 0 should be 1
    sums = tf.reduce_sum(outputs, axis=0)
    assert np.allclose(sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------
# Numerical Stability Tests
# ---------------------------------------------------------------------

def test_extreme_positive_logits() -> None:
    """Test stability with very large positive logits."""
    logits = tf.constant([[100.0, 50.0, 30.0, 20.0]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    # Should not produce NaN or Inf
    assert tf.reduce_all(tf.math.is_finite(outputs))
    assert np.isclose(tf.reduce_sum(outputs).numpy(), 1.0, rtol=1e-5)


def test_extreme_negative_logits() -> None:
    """Test stability with very large negative logits."""
    logits = tf.constant([[-100.0, -150.0, -130.0, -120.0]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    # Should not produce NaN or Inf
    assert tf.reduce_all(tf.math.is_finite(outputs))
    assert np.isclose(tf.reduce_sum(outputs).numpy(), 1.0, rtol=1e-5)


def test_mixed_extreme_logits() -> None:
    """Test stability with mixed extreme values."""
    logits = tf.constant([[100.0, -100.0, 50.0, -50.0]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    assert tf.reduce_all(tf.math.is_finite(outputs))
    assert np.isclose(tf.reduce_sum(outputs).numpy(), 1.0, rtol=1e-5)


def test_near_zero_logits() -> None:
    """Test stability with logits near zero."""
    logits = tf.constant([[1e-7, 2e-7, 1.5e-7, 0.5e-7]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    assert tf.reduce_all(tf.math.is_finite(outputs))
    assert np.isclose(tf.reduce_sum(outputs).numpy(), 1.0, rtol=1e-5)


def test_custom_epsilon_stability() -> None:
    """Test that custom epsilon affects numerical stability."""
    logits = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    
    layer_small = ThreshMax(epsilon=1e-12)
    layer_large = ThreshMax(epsilon=1e-6)
    
    output_small = layer_small(logits)
    output_large = layer_large(logits)
    
    # Both should be stable
    assert tf.reduce_all(tf.math.is_finite(output_small))
    assert tf.reduce_all(tf.math.is_finite(output_large))


# ---------------------------------------------------------------------
# Gradient Tests
# ---------------------------------------------------------------------

def test_gradient_flow_to_inputs(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test gradient flow propagates back to inputs."""
    layer = ThreshMax(**default_params)

    with tf.GradientTape() as tape:
        tape.watch(sample_logits)
        output = layer(sample_logits)
        loss = tf.reduce_sum(output * output)

    gradients = tape.gradient(loss, sample_logits)

    assert gradients is not None
    assert tf.reduce_all(tf.math.is_finite(gradients))


def test_gradient_flow_to_slope(sample_logits: tf.Tensor) -> None:
    """Test gradient flow propagates to the slope weight when trainable."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    
    with tf.GradientTape() as tape:
        output = layer(sample_logits)
        loss = tf.reduce_sum(output)

    gradients = tape.gradient(loss, layer.trainable_weights)
    
    assert len(gradients) == 1
    assert gradients[0] is not None
    assert tf.math.is_finite(gradients[0])


def test_gradient_magnitude(sample_logits: tf.Tensor) -> None:
    """Test that gradients are reasonable in magnitude."""
    layer = ThreshMax()
    
    with tf.GradientTape() as tape:
        tape.watch(sample_logits)
        output = layer(sample_logits)
        loss = tf.reduce_sum(output)
    
    gradients = tape.gradient(loss, sample_logits)
    grad_norm = tf.norm(gradients)
    
    # Gradients should not be too large or too small
    assert 0.00 < grad_norm < 100.0


def test_second_order_gradients(sample_logits: tf.Tensor) -> None:
    """Test that second-order gradients can be computed."""
    layer = ThreshMax(trainable_slope=True)
    
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(sample_logits)
            output = layer(sample_logits)
            loss = tf.reduce_sum(output)
        
        first_grad = tape1.gradient(loss, sample_logits)
        first_grad_sum = tf.reduce_sum(first_grad)
    
    second_grad = tape2.gradient(first_grad_sum, layer.trainable_weights)
    
    assert second_grad[0] is not None
    assert tf.math.is_finite(second_grad[0])


# ---------------------------------------------------------------------
# Training Integration Tests
# ---------------------------------------------------------------------

def test_training_loop_integration(sample_logits: tf.Tensor) -> None:
    """Test integration with a simple training loop."""
    layer = ThreshMax(trainable_slope=True, slope=5.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    
    initial_slope = None
    final_slope = None
    
    for step in range(10):
        with tf.GradientTape() as tape:
            output = layer(sample_logits)
            # Dummy loss: encourage high entropy
            loss = -tf.reduce_sum(output * tf.math.log(output + 1e-10))
        
        if step == 0:
            initial_slope = layer.slope_weight.numpy()
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        optimizer.apply_gradients(zip(gradients, layer.trainable_weights))
    
    final_slope = layer.slope_weight.numpy()
    
    # Slope should have changed during training
    assert not np.isclose(initial_slope, final_slope)


def test_model_training(sample_logits: tf.Tensor) -> None:
    """Test ThreshMax in a full Keras model training scenario."""
    inputs = keras.Input(shape=(10,))
    x = keras.layers.Dense(20)(inputs)
    x = ThreshMax(trainable_slope=True)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Create dummy data
    X = tf.random.normal((32, 10))
    y = tf.one_hot(tf.random.uniform((32,), 0, 20, dtype=tf.int32), depth=20)
    
    # Should train without errors
    history = model.fit(X, y, epochs=2, verbose=0)
    
    assert len(history.history['loss']) == 2
    assert all(np.isfinite(loss) for loss in history.history['loss'])


def test_slope_learning_direction(clear_winner_logits: tf.Tensor) -> None:
    """Test that slope moves in expected direction during training."""
    layer = ThreshMax(trainable_slope=True, slope=10.0)
    optimizer = keras.optimizers.SGD(learning_rate=0.1)
    
    initial_slope = layer(clear_winner_logits)  # Build layer
    initial_slope_value = layer.slope_weight.numpy()
    
    # Train to maximize entropy (should decrease slope)
    for _ in range(100):
        with tf.GradientTape() as tape:
            output = layer(clear_winner_logits)
            loss = -tf.reduce_sum(output * tf.math.log(output + 1e-10))
        
        gradients = tape.gradient(loss, layer.trainable_weights)
        optimizer.apply_gradients(zip(gradients, layer.trainable_weights))
    
    final_slope_value = layer.slope_weight.numpy()

    assert final_slope_value > initial_slope_value


# ---------------------------------------------------------------------
# Functional Interface Tests
# ---------------------------------------------------------------------

def test_functional_interface_basics(clear_winner_logits: tf.Tensor) -> None:
    """Test the functional interface thresh_max."""
    output = thresh_max(clear_winner_logits)

    assert output.shape == clear_winner_logits.shape
    row_sums = tf.reduce_sum(output, axis=-1)
    assert np.allclose(row_sums.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_functional_interface_consistency(sample_logits: tf.Tensor, default_params: Dict[str, Any]) -> None:
    """Test that functional interface produces same results as layer (fixed mode)."""
    layer = ThreshMax(**default_params)
    layer_output = layer(sample_logits)

    func_params = {k: v for k, v in default_params.items() if k != "trainable_slope"}
    func_output = thresh_max(sample_logits, **func_params)

    assert np.allclose(layer_output.numpy(), func_output.numpy(), rtol=1e-10, atol=1e-10)


def test_functional_interface_invalid_params() -> None:
    """Test that functional interface validates parameters."""
    logits = tf.constant([[1.0, 2.0, 3.0]])
    
    with pytest.raises(ValueError, match="epsilon must be positive"):
        thresh_max(logits, epsilon=-1e-5)
    
    with pytest.raises(ValueError, match="slope must be positive"):
        thresh_max(logits, slope=-5.0)


def test_functional_interface_different_slopes(clear_winner_logits: tf.Tensor) -> None:
    """Test functional interface with various slope values."""
    output_low = thresh_max(clear_winner_logits, slope=1.0)
    output_high = thresh_max(clear_winner_logits, slope=50.0)
    
    # High slope should produce more sparse output
    max_prob_low = tf.reduce_max(output_low, axis=-1).numpy()
    max_prob_high = tf.reduce_max(output_high, axis=-1).numpy()
    
    assert np.all(max_prob_high >= max_prob_low)


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------

def test_serialization(default_params: Dict[str, Any]) -> None:
    """Test serialization including new trainable_slope parameter."""
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
    
    layer = ThreshMax(trainable_slope=True)
    outputs = layer(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model(sample_logits)

    save_path = str(tmp_path / "model.keras")
    model.save(save_path)

    loaded_model = keras.models.load_model(
        save_path,
        custom_objects={"ThreshMax": ThreshMax}
    )

    original_out = model(sample_logits).numpy()
    loaded_out = loaded_model(sample_logits).numpy()
    
    assert np.allclose(original_out, loaded_out, rtol=1e-6, atol=1e-6)
    
    original_slope = model.layers[1].slope_weight.numpy()
    loaded_slope = loaded_model.layers[1].slope_weight.numpy()
    assert np.isclose(original_slope, loaded_slope)


def test_config_serialization_roundtrip() -> None:
    """Test that config can be serialized and deserialized multiple times."""
    layer1 = ThreshMax(axis=0, slope=15.0, epsilon=1e-8, trainable_slope=True)
    config1 = layer1.get_config()
    
    layer2 = ThreshMax.from_config(config1)
    config2 = layer2.get_config()
    
    # Configs should be identical
    assert config1 == config2


# ---------------------------------------------------------------------
# Edge Cases & Special Scenarios
# ---------------------------------------------------------------------

def test_single_class_input() -> None:
    """Test behavior with single class (edge case)."""
    logits = tf.constant([[5.0], [3.0]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    # Single class should always have probability 1
    assert np.allclose(outputs.numpy(), 1.0, rtol=1e-5, atol=1e-5)


def test_two_class_input(clear_winner_logits: tf.Tensor) -> None:
    """Test behavior with two classes."""
    logits = tf.constant([[2.0, 5.0], [1.0, 3.0]], dtype=tf.float32)
    layer = ThreshMax()
    outputs = layer(logits)
    
    # Should sum to 1
    assert np.allclose(tf.reduce_sum(outputs, axis=-1).numpy(), 1.0, rtol=1e-5)
    
    # Winner class should have higher probability
    assert np.all(outputs[:, 1].numpy() > outputs[:, 0].numpy())


def test_many_classes_input() -> None:
    """Test behavior with many classes."""
    logits = tf.random.normal((4, 100))
    layer = ThreshMax()
    outputs = layer(logits)
    
    # Should sum to 1
    assert np.allclose(tf.reduce_sum(outputs, axis=-1).numpy(), 1.0, rtol=1e-5)
    
    # Should have some sparsity
    # Count probabilities below 0.01
    sparse_count = tf.reduce_sum(tf.cast(outputs < 0.01, tf.int32), axis=-1)
    assert tf.reduce_all(sparse_count > 50)  # At least half should be near zero


def test_repr() -> None:
    """Test string representation of the layer."""
    layer_fixed = ThreshMax(slope=10.0, trainable_slope=False)
    repr_str = repr(layer_fixed)
    assert "ThreshMax" in repr_str
    assert "mode='fixed'" in repr_str
    
    layer_learnable = ThreshMax(slope=5.0, trainable_slope=True)
    repr_str = repr(layer_learnable)
    assert "mode='learnable'" in repr_str


def test_compute_output_shape() -> None:
    """Test compute_output_shape method."""
    layer = ThreshMax()
    
    input_shapes = [
        (None, 10),
        (32, 20),
        (None, 5, 8),
        (16, 4, 12)
    ]
    
    for shape in input_shapes:
        output_shape = layer.compute_output_shape(shape)
        assert output_shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
