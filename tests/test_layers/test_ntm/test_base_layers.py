"""
Tests for NTM base_layers module.

Tests cover:
    - DifferentiableAddressingHead
    - DifferentiableSelectCopy
    - SimpleSelectCopy
"""

import numpy as np
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.ntm.base_layers import (
    DifferentiableAddressingHead,
    DifferentiableSelectCopy,
    SimpleSelectCopy,
)


# ---------------------------------------------------------------------
# DifferentiableAddressingHead Tests
# ---------------------------------------------------------------------


class TestDifferentiableAddressingHead:
    """Tests for DifferentiableAddressingHead layer."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        head = DifferentiableAddressingHead(
            memory_size=32,
            content_dim=16,
        )
        assert head.memory_size == 32
        assert head.content_dim == 16
        assert head.num_shifts == 3
        assert head.use_content_addressing is True
        assert head.use_location_addressing is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        head = DifferentiableAddressingHead(
            memory_size=64,
            content_dim=32,
            controller_dim=128,
            num_shifts=5,
            use_content_addressing=False,
            use_location_addressing=True,
            sharpening_bias=2.0,
        )
        assert head.memory_size == 64
        assert head.content_dim == 32
        assert head.controller_dim == 128
        assert head.num_shifts == 5
        assert head.use_content_addressing is False
        assert head.sharpening_bias == 2.0

    def test_init_invalid_num_shifts(self):
        """Test that even num_shifts raises ValueError."""
        try:
            DifferentiableAddressingHead(
                memory_size=32,
                content_dim=16,
                num_shifts=4,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "odd" in str(e).lower()

    def test_init_invalid_memory_size(self):
        """Test that non-positive memory_size raises ValueError."""
        try:
            DifferentiableAddressingHead(
                memory_size=0,
                content_dim=16,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "positive" in str(e).lower()

    def test_build(self):
        """Test that build creates all necessary weights."""
        head = DifferentiableAddressingHead(
            memory_size=32,
            content_dim=16,
            controller_dim=64,
        )
        head.build((None, 64))

        assert head.built
        assert head.initial_weights is not None
        assert head.initial_weights.shape == (1, 32)

    def test_call_output_shape(self):
        """Test that call produces correct output shape."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        weights = head(memory, controller_state)

        assert ops.shape(weights) == (batch_size, memory_size)

    def test_call_weights_sum_to_one(self):
        """Test that output weights sum to 1 (valid probability distribution)."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        weights = head(memory, controller_state)
        weight_sums = ops.sum(weights, axis=-1)

        # Relaxed tolerance due to epsilon in sharpening denominator
        # sum = S / (S + eps) which is < 1.0
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weight_sums),
            np.ones(batch_size),
            rtol=1e-3,
            atol=1e-3,
            err_msg="Weights should sum to approximately 1",
        )

    def test_call_weights_non_negative(self):
        """Test that output weights are non-negative."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        weights = head(memory, controller_state)
        weights_np = keras.ops.convert_to_numpy(weights)

        assert np.all(weights_np >= 0), "Weights should be non-negative"

    def test_call_with_previous_weights(self):
        """Test call with explicit previous weights."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)
        previous_weights = ops.softmax(
            keras.random.normal((batch_size, memory_size), seed=44), axis=-1
        )

        weights = head(memory, controller_state, previous_weights=previous_weights)

        assert ops.shape(weights) == (batch_size, memory_size)

    def test_content_only_addressing(self):
        """Test with only content-based addressing enabled."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
            use_content_addressing=True,
            use_location_addressing=False,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        weights = head(memory, controller_state)

        assert ops.shape(weights) == (batch_size, memory_size)
        assert head.shift_proj is None

    def test_location_only_addressing(self):
        """Test with only location-based addressing enabled."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
            use_content_addressing=False,
            use_location_addressing=True,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        weights = head(memory, controller_state)

        assert ops.shape(weights) == (batch_size, memory_size)
        assert head.key_proj is None
        assert head.beta_proj is None

    def test_serialization(self):
        """Test get_config and from_config."""
        head = DifferentiableAddressingHead(
            memory_size=32,
            content_dim=16,
            controller_dim=64,
            num_shifts=5,
            sharpening_bias=1.5,
        )

        config = head.get_config()

        assert config["memory_size"] == 32
        assert config["content_dim"] == 16
        assert config["controller_dim"] == 64
        assert config["num_shifts"] == 5
        assert config["sharpening_bias"] == 1.5

        # Reconstruct from config
        head_restored = DifferentiableAddressingHead.from_config(config)

        assert head_restored.memory_size == head.memory_size
        assert head_restored.content_dim == head.content_dim
        assert head_restored.num_shifts == head.num_shifts

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        head = DifferentiableAddressingHead(
            memory_size=32,
            content_dim=16,
        )

        output_shape = head.compute_output_shape((None, 32, 16))
        assert output_shape == (None, 32)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        batch_size = 2
        memory_size = 16
        content_dim = 8
        controller_dim = 32

        head = DifferentiableAddressingHead(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = tf.Variable(
            keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        )
        controller_state = tf.Variable(
            keras.random.normal((batch_size, controller_dim), seed=43)
        )

        with tf.GradientTape() as tape:
            weights = head(memory, controller_state)
            loss = ops.sum(weights)

        grads = tape.gradient(loss, [memory, controller_state])

        assert grads[0] is not None, "Gradient w.r.t. memory should exist"
        assert grads[1] is not None, "Gradient w.r.t. controller_state should exist"


# ---------------------------------------------------------------------
# DifferentiableSelectCopy Tests
# ---------------------------------------------------------------------


class TestDifferentiableSelectCopy:
    """Tests for DifferentiableSelectCopy layer."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        layer = DifferentiableSelectCopy(
            memory_size=32,
            content_dim=16,
            controller_dim=64,
        )
        assert layer.memory_size == 32
        assert layer.content_dim == 16
        assert layer.controller_dim == 64
        assert layer.num_read_heads == 1
        assert layer.num_write_heads == 1

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        layer = DifferentiableSelectCopy(
            memory_size=64,
            content_dim=32,
            controller_dim=128,
            num_read_heads=2,
            num_write_heads=2,
            num_shifts=5,
        )
        assert layer.num_read_heads == 2
        assert layer.num_write_heads == 2
        assert layer.num_shifts == 5
        assert len(layer.read_heads) == 2
        assert len(layer.write_heads) == 2

    def test_init_invalid_params(self):
        """Test that invalid parameters raise ValueError."""
        try:
            DifferentiableSelectCopy(
                memory_size=-1,
                content_dim=16,
                controller_dim=64,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            DifferentiableSelectCopy(
                memory_size=32,
                content_dim=16,
                controller_dim=64,
                num_read_heads=0,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_build(self):
        """Test that build creates all necessary components."""
        layer = DifferentiableSelectCopy(
            memory_size=32,
            content_dim=16,
            controller_dim=64,
            num_read_heads=2,
            num_write_heads=1,
        )
        layer.build((None, 32, 16))

        assert layer.built
        assert len(layer.read_heads) == 2
        assert len(layer.write_heads) == 1
        assert len(layer.erase_projections) == 1
        assert len(layer.add_projections) == 1

    def test_call_output_shapes(self):
        """Test that call produces correct output shapes."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64
        num_read_heads = 2

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
            num_read_heads=num_read_heads,
            num_write_heads=1,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        new_memory, read_output, state_dict = layer(memory, controller_state)

        assert ops.shape(new_memory) == (batch_size, memory_size, content_dim)
        assert ops.shape(read_output) == (batch_size, num_read_heads * content_dim)
        assert len(state_dict["read_weights"]) == num_read_heads
        assert len(state_dict["write_weights"]) == 1

    def test_call_with_previous_weights(self):
        """Test call with explicit previous weights."""
        batch_size = 4
        memory_size = 32
        content_dim = 16
        controller_dim = 64

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
            num_read_heads=1,
            num_write_heads=1,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        prev_read_weights = [
            ops.softmax(keras.random.normal((batch_size, memory_size), seed=44), axis=-1)
        ]
        prev_write_weights = [
            ops.softmax(keras.random.normal((batch_size, memory_size), seed=45), axis=-1)
        ]

        new_memory, read_output, state_dict = layer(
            memory,
            controller_state,
            previous_read_weights=prev_read_weights,
            previous_write_weights=prev_write_weights,
        )

        assert ops.shape(new_memory) == (batch_size, memory_size, content_dim)

    def test_memory_modification(self):
        """Test that write operation modifies memory."""
        batch_size = 2
        memory_size = 16
        content_dim = 8
        controller_dim = 32

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        new_memory, _, _ = layer(memory, controller_state)

        memory_np = keras.ops.convert_to_numpy(memory)
        new_memory_np = keras.ops.convert_to_numpy(new_memory)

        # Memory should be modified (not identical)
        assert not np.allclose(memory_np, new_memory_np)

    def test_read_operation(self):
        """Test that read operation extracts content from memory."""
        batch_size = 2
        memory_size = 16
        content_dim = 8
        controller_dim = 32

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        # Create memory with distinct content
        memory = keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        controller_state = keras.random.normal((batch_size, controller_dim), seed=43)

        _, read_output, state_dict = layer(memory, controller_state)

        read_output_np = keras.ops.convert_to_numpy(read_output)
        read_weights = state_dict["read_weights"][0]
        read_weights_np = keras.ops.convert_to_numpy(read_weights)

        # Read weights should be valid probability distribution
        np.testing.assert_allclose(
            np.sum(read_weights_np, axis=-1),
            np.ones(batch_size),
            rtol=1e-5,
            atol=1e-5,
        )

        # Read output should have correct shape
        assert read_output_np.shape == (batch_size, content_dim)

    def test_serialization(self):
        """Test get_config and from_config."""
        layer = DifferentiableSelectCopy(
            memory_size=32,
            content_dim=16,
            controller_dim=64,
            num_read_heads=2,
            num_write_heads=1,
            num_shifts=5,
        )

        config = layer.get_config()

        assert config["memory_size"] == 32
        assert config["content_dim"] == 16
        assert config["num_read_heads"] == 2
        assert config["num_write_heads"] == 1

        layer_restored = DifferentiableSelectCopy.from_config(config)

        assert layer_restored.memory_size == layer.memory_size
        assert layer_restored.num_read_heads == layer.num_read_heads

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        batch_size = 2
        memory_size = 16
        content_dim = 8
        controller_dim = 32

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
        )

        memory = tf.Variable(
            keras.random.normal((batch_size, memory_size, content_dim), seed=42)
        )
        controller_state = tf.Variable(
            keras.random.normal((batch_size, controller_dim), seed=43)
        )

        with tf.GradientTape() as tape:
            new_memory, read_output, _ = layer(memory, controller_state)
            loss = ops.sum(new_memory) + ops.sum(read_output)

        grads = tape.gradient(loss, [memory, controller_state])

        assert grads[0] is not None
        assert grads[1] is not None


# ---------------------------------------------------------------------
# SimpleSelectCopy Tests
# ---------------------------------------------------------------------


class TestSimpleSelectCopy:
    """Tests for SimpleSelectCopy layer."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        layer = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=16,
        )
        assert layer.input_size == 10
        assert layer.output_size == 10
        assert layer.content_dim == 16
        assert layer.num_copies == 1
        assert layer.temperature == 1.0
        assert layer.use_content_query is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        layer = SimpleSelectCopy(
            input_size=20,
            output_size=15,
            content_dim=32,
            num_copies=3,
            temperature=0.5,
            use_content_query=False,
        )
        assert layer.input_size == 20
        assert layer.output_size == 15
        assert layer.num_copies == 3
        assert layer.temperature == 0.5
        assert layer.use_content_query is False

    def test_init_invalid_params(self):
        """Test that invalid parameters raise ValueError."""
        try:
            SimpleSelectCopy(
                input_size=0,
                output_size=10,
                content_dim=16,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            SimpleSelectCopy(
                input_size=10,
                output_size=10,
                content_dim=16,
                temperature=-1.0,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_build(self):
        """Test that build creates all necessary components."""
        layer = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=16,
            num_copies=2,
            use_content_query=True,
        )
        layer.build((None, 10, 16))

        assert layer.built
        assert layer.output_position_embeddings is not None
        assert layer.output_position_embeddings.shape == (10, 16)
        assert len(layer.content_transforms) == 2

    def test_build_non_content_query(self):
        """Test build with use_content_query=False."""
        layer = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=16,
            num_copies=2,
            use_content_query=False,
        )
        layer.build((None, 10, 16))

        assert layer.built
        assert len(layer.read_query_weights) == 2
        assert len(layer.write_query_weights) == 2

    def test_call_output_shape(self):
        """Test that call produces correct output shape."""
        batch_size = 4
        input_size = 10
        output_size = 8
        content_dim = 16
        num_copies = 2

        layer = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
            num_copies=num_copies,
        )

        inputs = keras.random.normal((batch_size, input_size, content_dim), seed=42)

        output, attention_info = layer(inputs)

        assert ops.shape(output) == (batch_size, output_size, content_dim)
        assert len(attention_info["read_weights"]) == num_copies
        assert len(attention_info["write_weights"]) == num_copies

    def test_call_attention_weights_valid(self):
        """Test that attention weights are valid probability distributions."""
        batch_size = 4
        input_size = 10
        output_size = 8
        content_dim = 16

        layer = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
        )

        inputs = keras.random.normal((batch_size, input_size, content_dim), seed=42)

        _, attention_info = layer(inputs)

        read_weights = attention_info["read_weights"][0]
        write_weights = attention_info["write_weights"][0]

        read_weights_np = keras.ops.convert_to_numpy(read_weights)
        write_weights_np = keras.ops.convert_to_numpy(write_weights)

        # Check sums
        np.testing.assert_allclose(
            np.sum(read_weights_np, axis=-1),
            np.ones(batch_size),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.sum(write_weights_np, axis=-1),
            np.ones(batch_size),
            rtol=1e-5,
            atol=1e-5,
        )

        # Check non-negative
        assert np.all(read_weights_np >= 0)
        assert np.all(write_weights_np >= 0)

    def test_temperature_effect(self):
        """Test that temperature affects attention sharpness."""
        batch_size = 4
        input_size = 10
        output_size = 10
        content_dim = 16

        layer_low_temp = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
            temperature=0.1,
        )

        layer_high_temp = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
            temperature=10.0,
        )

        inputs = keras.random.normal((batch_size, input_size, content_dim), seed=42)

        # Build both layers with same input
        layer_low_temp.build((None, input_size, content_dim))
        layer_high_temp.build((None, input_size, content_dim))

        _, attention_low = layer_low_temp(inputs)
        _, attention_high = layer_high_temp(inputs)

        read_weights_low = keras.ops.convert_to_numpy(attention_low["read_weights"][0])
        read_weights_high = keras.ops.convert_to_numpy(attention_high["read_weights"][0])

        # Low temperature should produce sharper (higher entropy) distribution
        entropy_low = -np.sum(
            read_weights_low * np.log(read_weights_low + 1e-10), axis=-1
        )
        entropy_high = -np.sum(
            read_weights_high * np.log(read_weights_high + 1e-10), axis=-1
        )

        # High temperature -> more uniform -> higher entropy
        assert np.mean(entropy_high) > np.mean(entropy_low)

    def test_serialization(self):
        """Test get_config and from_config."""
        layer = SimpleSelectCopy(
            input_size=10,
            output_size=8,
            content_dim=16,
            num_copies=2,
            temperature=0.5,
            use_content_query=False,
        )

        config = layer.get_config()

        assert config["input_size"] == 10
        assert config["output_size"] == 8
        assert config["num_copies"] == 2
        assert config["temperature"] == 0.5
        assert config["use_content_query"] is False

        layer_restored = SimpleSelectCopy.from_config(config)

        assert layer_restored.input_size == layer.input_size
        assert layer_restored.output_size == layer.output_size
        assert layer_restored.temperature == layer.temperature

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        layer = SimpleSelectCopy(
            input_size=10,
            output_size=8,
            content_dim=16,
            num_copies=2,
        )

        output_shape, attention_shapes = layer.compute_output_shape((None, 10, 16))

        assert output_shape == (None, 8, 16)
        assert len(attention_shapes["read_weights"]) == 2
        assert len(attention_shapes["write_weights"]) == 2

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        batch_size = 2
        input_size = 8
        output_size = 8
        content_dim = 16

        layer = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
        )

        inputs = tf.Variable(
            keras.random.normal((batch_size, input_size, content_dim), seed=42)
        )

        with tf.GradientTape() as tape:
            output, _ = layer(inputs)
            loss = ops.sum(output)

        grads = tape.gradient(loss, inputs)

        assert grads is not None


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])