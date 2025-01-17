"""
Test suite for custom Keras layers:
- LayerScale
- MultiplierType
- LearnableMultiplier

This module provides comprehensive tests for initialization,
behavior, serialization, and edge cases.
"""

import pytest
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any

from dl_techniques.layers.layer_scale import (
    LayerScale,
    MultiplierType,
    LearnableMultiplier
)


# Test fixtures
@pytest.fixture
def sample_shape() -> Tuple[int, int, int, int]:
    """Sample input shape for testing."""
    return (2, 16, 16, 64)


@pytest.fixture
def sample_input(sample_shape: Tuple[int, int, int, int]) -> tf.Tensor:
    """Generate sample input tensor."""
    tf.random.set_seed(42)
    return tf.random.normal(sample_shape)


# LayerScale Tests
class TestLayerScale:
    @pytest.fixture
    def layer_params(self) -> Dict[str, Any]:
        """Default parameters for LayerScale."""
        return {
            "init_values": 0.1,
            "projection_dim": 64
        }

    def test_initialization(self, layer_params: Dict[str, Any]) -> None:
        """Test LayerScale initialization."""
        layer = LayerScale(**layer_params)
        assert layer.init_values == layer_params["init_values"]
        assert layer.projection_dim == layer_params["projection_dim"]
        assert layer.gamma is None  # Not built yet

    def test_build(self, layer_params: Dict[str, Any], sample_shape: Tuple[int, int, int, int]) -> None:
        """Test LayerScale build method."""
        layer = LayerScale(**layer_params)
        layer.build(sample_shape)
        assert layer.gamma is not None
        assert layer.gamma.shape == (layer_params["projection_dim"],)

    def test_call(self, layer_params: Dict[str, Any], sample_input: tf.Tensor) -> None:
        """Test LayerScale call method."""
        layer = LayerScale(**layer_params)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

        # Test training vs inference mode
        train_output = layer(sample_input, training=True)
        infer_output = layer(sample_input, training=False)
        assert np.allclose(train_output.numpy(), infer_output.numpy())

    def test_serialization(self, layer_params: Dict[str, Any]) -> None:
        """Test LayerScale serialization."""
        layer = LayerScale(**layer_params)
        config = layer.get_config()

        # Recreate from config
        new_layer = LayerScale.from_config(config)
        assert new_layer.init_values == layer.init_values
        assert new_layer.projection_dim == layer.projection_dim


# MultiplierType Tests
class TestMultiplierType:
    @pytest.mark.parametrize("type_str", ["GLOBAL", "CHANNEL"])
    def test_valid_from_string(self, type_str: str) -> None:
        """Test valid string conversions."""
        mult_type = MultiplierType.from_string(type_str)
        assert isinstance(mult_type, MultiplierType)
        assert mult_type.to_string() == type_str

    @pytest.mark.parametrize("invalid_input", [None, 123, "", " "])
    def test_invalid_from_string(self, invalid_input: Any) -> None:
        """Test invalid string conversions."""
        with pytest.raises(ValueError):
            MultiplierType.from_string(invalid_input)

    def test_enum_values(self) -> None:
        """Test enum values."""
        assert MultiplierType.GLOBAL.value == 0
        assert MultiplierType.CHANNEL.value == 1


# LearnableMultiplier Tests
class TestLearnableMultiplier:
    @pytest.fixture
    def layer_params(self) -> Dict[str, Any]:
        """Default parameters for LearnableMultiplier."""
        return {
            "multiplier_type": "GLOBAL",
            "capped": True
        }

    def test_initialization(self, layer_params: Dict[str, Any]) -> None:
        """Test LearnableMultiplier initialization."""
        layer = LearnableMultiplier(**layer_params)
        assert layer.capped == layer_params["capped"]
        assert layer.multiplier_type == MultiplierType.GLOBAL
        assert layer.w_multiplier is None

    def test_build_global(self, layer_params: Dict[str, Any], sample_shape: Tuple[int, int, int, int]) -> None:
        """Test build with global multiplier."""
        layer = LearnableMultiplier(**layer_params)
        layer.build(sample_shape)
        assert layer.w_multiplier.shape == (1, 1, 1, 1)

    def test_build_channel(self, layer_params: Dict[str, Any], sample_shape: Tuple[int, int, int, int]) -> None:
        """Test build with channel multiplier."""
        layer_params["multiplier_type"] = "CHANNEL"
        layer = LearnableMultiplier(**layer_params)
        layer.build(sample_shape)
        assert layer.w_multiplier.shape == (1, 1, 1, sample_shape[-1])

    @pytest.mark.parametrize("capped", [True, False])
    def test_call_modes(self, layer_params: Dict[str, Any], sample_input: tf.Tensor, capped: bool) -> None:
        """Test different call modes."""
        layer_params["capped"] = capped
        layer = LearnableMultiplier(**layer_params)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

        # Values should be within expected range
        if capped:
            assert tf.reduce_max(output) <= tf.reduce_max(sample_input)

    def test_training_behavior(self, layer_params: Dict[str, Any], sample_input: tf.Tensor) -> None:
        """Test training behavior and gradient flow."""
        layer = LearnableMultiplier(**layer_params)

        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_serialization(self, layer_params: Dict[str, Any]) -> None:
        """Test serialization."""
        layer = LearnableMultiplier(**layer_params)
        config = layer.get_config()

        # Recreate from config
        new_layer = LearnableMultiplier.from_config(config)
        assert new_layer.capped == layer.capped
        assert new_layer.multiplier_type == layer.multiplier_type


@pytest.mark.integration
class TestIntegration:
    """Integration tests for layers working together."""

    def test_sequential_model(self, sample_input: tf.Tensor) -> None:
        """Test layers in a sequential model."""
        model = tf.keras.Sequential([
            LayerScale(init_values=0.1, projection_dim=64),
            LearnableMultiplier(multiplier_type="GLOBAL"),
            LearnableMultiplier(multiplier_type="CHANNEL")
        ])

        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_training_pipeline(self, sample_input: tf.Tensor) -> None:
        """Test layers in a training pipeline."""
        model = tf.keras.Sequential([
            LayerScale(init_values=0.1, projection_dim=64),
            LearnableMultiplier(multiplier_type="CHANNEL")
        ])

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            sample_input,
            sample_input,  # Using input as target for testing
            epochs=1,
            verbose=0
        )
        assert 'loss' in history.history

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
