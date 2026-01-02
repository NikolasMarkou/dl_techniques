"""
Tests for Differentiable Select-Copy Layers.

This module contains comprehensive tests for:
- DifferentiableAddressingHead
- DifferentiableSelectCopy
- SimpleSelectCopy
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.ntm.base_layers import (
    DifferentiableAddressingHead,
    DifferentiableSelectCopy,
    SimpleSelectCopy,
)


# =============================================================================
# DifferentiableAddressingHead Tests
# =============================================================================


class TestDifferentiableAddressingHead:
    """Tests for DifferentiableAddressingHead layer."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for addressing head."""
        return {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,  # Added explicit controller_dim for robustness
            "num_shifts": 3,
            "use_content_addressing": True,
            "use_location_addressing": True,
            "sharpening_bias": 1.0,
        }

    @pytest.fixture
    def sample_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample memory and controller state inputs."""
        batch_size = 4
        memory_size = 16
        content_dim = 32
        controller_dim = 64

        memory = np.random.randn(batch_size, memory_size, content_dim).astype(
            np.float32
        )
        controller_state = np.random.randn(batch_size, controller_dim).astype(
            np.float32
        )
        return memory, controller_state

    def test_instantiation_valid_config(self, default_config: dict) -> None:
        """Test layer can be instantiated with valid config."""
        layer = DifferentiableAddressingHead(**default_config)

        assert layer.memory_size == default_config["memory_size"]
        assert layer.content_dim == default_config["content_dim"]
        assert layer.controller_dim == default_config["controller_dim"]
        assert layer.num_shifts == default_config["num_shifts"]
        assert layer.use_content_addressing == default_config["use_content_addressing"]
        assert layer.use_location_addressing == default_config["use_location_addressing"]

    def test_invalid_num_shifts_even(self, default_config: dict) -> None:
        """Test that even num_shifts raises ValueError."""
        default_config["num_shifts"] = 4
        with pytest.raises(ValueError, match="num_shifts must be odd"):
            DifferentiableAddressingHead(**default_config)

    def test_invalid_memory_size_zero(self, default_config: dict) -> None:
        """Test that zero memory_size raises ValueError."""
        default_config["memory_size"] = 0
        with pytest.raises(ValueError, match="memory_size must be positive"):
            DifferentiableAddressingHead(**default_config)

    def test_invalid_content_dim_negative(self, default_config: dict) -> None:
        """Test that negative content_dim raises ValueError."""
        default_config["content_dim"] = -1
        with pytest.raises(ValueError, match="content_dim must be positive"):
            DifferentiableAddressingHead(**default_config)

    def test_forward_pass_output_shape(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test forward pass produces correct output shape."""
        layer = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs

        weights = layer(memory, controller_state)

        expected_shape = (memory.shape[0], default_config["memory_size"])
        assert weights.shape == expected_shape

    def test_output_is_valid_distribution(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test output weights sum to 1 and are non-negative."""
        layer = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs

        weights = layer(memory, controller_state)
        weights_np = ops.convert_to_numpy(weights)

        # Check non-negative
        assert np.all(weights_np >= 0), "Weights should be non-negative"

        # Check sum to 1
        sums = np.sum(weights_np, axis=-1)
        np.testing.assert_allclose(
            sums,
            np.ones_like(sums),
            rtol=1e-3,
            atol=1e-3,
            err_msg="Weights should sum to 1",
        )

    def test_with_previous_weights(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test forward pass with previous weights provided."""
        layer = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs

        # Create previous weights
        batch_size = memory.shape[0]
        previous_weights = np.ones((batch_size, default_config["memory_size"])) / default_config["memory_size"]
        previous_weights = previous_weights.astype(np.float32)

        weights = layer(memory, controller_state, previous_weights=previous_weights)

        expected_shape = (batch_size, default_config["memory_size"])
        assert weights.shape == expected_shape

    def test_content_only_addressing(
            self,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with content addressing only."""
        config = {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "use_content_addressing": True,
            "use_location_addressing": False,
        }
        layer = DifferentiableAddressingHead(**config)
        memory, controller_state = sample_inputs

        weights = layer(memory, controller_state)

        assert weights.shape == (memory.shape[0], config["memory_size"])

    def test_location_only_addressing(
            self,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with location addressing only."""
        config = {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "use_content_addressing": False,
            "use_location_addressing": True,
        }
        layer = DifferentiableAddressingHead(**config)
        memory, controller_state = sample_inputs

        weights = layer(memory, controller_state)

        assert weights.shape == (memory.shape[0], config["memory_size"])

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape returns correct shape."""
        layer = DifferentiableAddressingHead(**default_config)

        input_shape = (None, default_config["memory_size"], default_config["content_dim"])
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, default_config["memory_size"])

    def test_compute_output_shape_matches_actual(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test compute_output_shape matches actual output shape."""
        layer = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs

        computed_shape = layer.compute_output_shape(memory.shape)
        actual_output = layer(memory, controller_state)

        # Compare ignoring batch dimension
        assert computed_shape[1:] == actual_output.shape[1:]

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        layer = DifferentiableAddressingHead(**default_config)
        config = layer.get_config()

        assert "memory_size" in config
        assert "content_dim" in config
        assert "controller_dim" in config
        assert "num_shifts" in config
        assert "use_content_addressing" in config
        assert "use_location_addressing" in config
        assert "sharpening_bias" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config

    def test_from_config_reconstruction(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs
        original(memory, controller_state)  # Build

        config = original.get_config()
        reconstructed = DifferentiableAddressingHead.from_config(config)

        assert reconstructed.memory_size == original.memory_size
        assert reconstructed.content_dim == original.content_dim
        assert reconstructed.controller_dim == original.controller_dim
        assert reconstructed.num_shifts == original.num_shifts

    def test_serialization_cycle(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        memory, controller_state = sample_inputs

        # Create model wrapping the layer
        memory_input = keras.Input(shape=memory.shape[1:], name="memory")
        controller_input = keras.Input(shape=controller_state.shape[1:], name="controller")

        layer = DifferentiableAddressingHead(**default_config)
        outputs = layer(memory_input, controller_input)
        model = keras.Model(inputs=[memory_input, controller_input], outputs=outputs)

        original_output = model([memory, controller_state])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_addressing_head.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model([memory, controller_state])

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_variable_batch_size(
            self,
            default_config: dict,
            batch_size: int,
    ) -> None:
        """Test layer handles various batch sizes."""
        layer = DifferentiableAddressingHead(**default_config)

        memory = np.random.randn(
            batch_size, default_config["memory_size"], default_config["content_dim"]
        ).astype(np.float32)
        controller_state = np.random.randn(batch_size, 64).astype(np.float32)

        weights = layer(memory, controller_state)

        assert weights.shape[0] == batch_size

    @pytest.mark.parametrize("num_shifts", [1, 3, 5, 7])
    def test_different_num_shifts(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
            num_shifts: int,
    ) -> None:
        """Test layer works with various num_shifts values."""
        default_config["num_shifts"] = num_shifts
        layer = DifferentiableAddressingHead(**default_config)
        memory, controller_state = sample_inputs

        weights = layer(memory, controller_state)

        assert weights.shape == (memory.shape[0], default_config["memory_size"])


# =============================================================================
# DifferentiableSelectCopy Tests
# =============================================================================


class TestDifferentiableSelectCopy:
    """Tests for DifferentiableSelectCopy layer."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for select-copy layer."""
        return {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 2,
            "num_write_heads": 1,
            "num_shifts": 3,
            "use_content_addressing": True,
            "use_location_addressing": True,
        }

    @pytest.fixture
    def sample_inputs(self, default_config: dict) -> tuple[np.ndarray, np.ndarray]:
        """Sample memory and controller state inputs."""
        batch_size = 4

        memory = np.random.randn(
            batch_size, default_config["memory_size"], default_config["content_dim"]
        ).astype(np.float32)
        controller_state = np.random.randn(
            batch_size, default_config["controller_dim"]
        ).astype(np.float32)

        return memory, controller_state

    def test_instantiation_valid_config(self, default_config: dict) -> None:
        """Test layer can be instantiated with valid config."""
        layer = DifferentiableSelectCopy(**default_config)

        assert layer.memory_size == default_config["memory_size"]
        assert layer.content_dim == default_config["content_dim"]
        assert layer.controller_dim == default_config["controller_dim"]
        assert layer.num_read_heads == default_config["num_read_heads"]
        assert layer.num_write_heads == default_config["num_write_heads"]

    def test_invalid_memory_size(self, default_config: dict) -> None:
        """Test that invalid memory_size raises ValueError."""
        default_config["memory_size"] = 0
        with pytest.raises(ValueError, match="memory_size must be positive"):
            DifferentiableSelectCopy(**default_config)

    def test_invalid_content_dim(self, default_config: dict) -> None:
        """Test that invalid content_dim raises ValueError."""
        default_config["content_dim"] = -5
        with pytest.raises(ValueError, match="content_dim must be positive"):
            DifferentiableSelectCopy(**default_config)

    def test_invalid_controller_dim(self, default_config: dict) -> None:
        """Test that invalid controller_dim raises ValueError."""
        default_config["controller_dim"] = 0
        with pytest.raises(ValueError, match="controller_dim must be positive"):
            DifferentiableSelectCopy(**default_config)

    def test_invalid_num_read_heads(self, default_config: dict) -> None:
        """Test that invalid num_read_heads raises ValueError."""
        default_config["num_read_heads"] = 0
        with pytest.raises(ValueError, match="num_read_heads must be positive"):
            DifferentiableSelectCopy(**default_config)

    def test_invalid_num_write_heads(self, default_config: dict) -> None:
        """Test that invalid num_write_heads raises ValueError."""
        default_config["num_write_heads"] = -1
        with pytest.raises(ValueError, match="num_write_heads must be positive"):
            DifferentiableSelectCopy(**default_config)

    def test_forward_pass_output_shapes(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test forward pass produces correct output shapes."""
        layer = DifferentiableSelectCopy(**default_config)
        memory, controller_state = sample_inputs

        new_memory, read_content, state_dict = layer(memory, controller_state)

        # Check memory shape unchanged
        assert new_memory.shape == memory.shape

        # Check read content shape
        expected_read_dim = default_config["num_read_heads"] * default_config["content_dim"]
        assert read_content.shape == (memory.shape[0], expected_read_dim)

        # Check state dict
        assert "read_weights" in state_dict
        assert "write_weights" in state_dict
        assert len(state_dict["read_weights"]) == default_config["num_read_heads"]
        assert len(state_dict["write_weights"]) == default_config["num_write_heads"]

    def test_read_weights_are_valid_distributions(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test read weights are valid probability distributions."""
        layer = DifferentiableSelectCopy(**default_config)
        memory, controller_state = sample_inputs

        _, _, state_dict = layer(memory, controller_state)

        for weights in state_dict["read_weights"]:
            weights_np = ops.convert_to_numpy(weights)
            assert np.all(weights_np >= 0), "Weights should be non-negative"
            sums = np.sum(weights_np, axis=-1)
            np.testing.assert_allclose(
                sums,
                np.ones_like(sums),
                rtol=1e-5,
                atol=1e-5,
                err_msg="Weights should sum to 1",
            )

    def test_write_weights_are_valid_distributions(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test write weights are valid probability distributions."""
        layer = DifferentiableSelectCopy(**default_config)
        memory, controller_state = sample_inputs

        _, _, state_dict = layer(memory, controller_state)

        for weights in state_dict["write_weights"]:
            weights_np = ops.convert_to_numpy(weights)
            assert np.all(weights_np >= 0), "Weights should be non-negative"
            sums = np.sum(weights_np, axis=-1)
            np.testing.assert_allclose(
                sums,
                np.ones_like(sums),
                rtol=1e-3,
                atol=1e-3,
                err_msg="Weights should sum to 1",
            )

    def test_with_previous_weights(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test forward pass with previous weights provided."""
        layer = DifferentiableSelectCopy(**default_config)
        memory, controller_state = sample_inputs
        batch_size = memory.shape[0]

        # Create previous weights
        prev_read = [
            np.ones((batch_size, default_config["memory_size"]), dtype=np.float32) / default_config["memory_size"]
            for _ in range(default_config["num_read_heads"])
        ]
        prev_write = [
            np.ones((batch_size, default_config["memory_size"]), dtype=np.float32) / default_config["memory_size"]
            for _ in range(default_config["num_write_heads"])
        ]

        new_memory, read_content, state_dict = layer(
            memory,
            controller_state,
            previous_read_weights=prev_read,
            previous_write_weights=prev_write,
        )

        assert new_memory.shape == memory.shape

    def test_single_read_head(
            self,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with single read head."""
        config = {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 1,
            "num_write_heads": 1,
        }
        layer = DifferentiableSelectCopy(**config)

        memory = np.random.randn(4, 16, 32).astype(np.float32)
        controller_state = np.random.randn(4, 64).astype(np.float32)

        _, read_content, _ = layer(memory, controller_state)

        # Single head: read_content should have content_dim dimensions
        assert read_content.shape == (4, config["content_dim"])

    def test_multiple_read_heads(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with multiple read heads."""
        default_config["num_read_heads"] = 4
        layer = DifferentiableSelectCopy(**default_config)
        memory, controller_state = sample_inputs

        _, read_content, state_dict = layer(memory, controller_state)

        expected_dim = 4 * default_config["content_dim"]
        assert read_content.shape == (memory.shape[0], expected_dim)
        assert len(state_dict["read_weights"]) == 4

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape returns correct shapes."""
        layer = DifferentiableSelectCopy(**default_config)

        input_shape = (None, default_config["memory_size"], default_config["content_dim"])
        memory_shape, read_shape, state_shapes = layer.compute_output_shape(input_shape)

        assert memory_shape == (None, default_config["memory_size"], default_config["content_dim"])
        expected_read_dim = default_config["num_read_heads"] * default_config["content_dim"]
        assert read_shape == (None, expected_read_dim)

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        layer = DifferentiableSelectCopy(**default_config)
        config = layer.get_config()

        assert "memory_size" in config
        assert "content_dim" in config
        assert "controller_dim" in config
        assert "num_read_heads" in config
        assert "num_write_heads" in config
        assert "num_shifts" in config
        assert "use_content_addressing" in config
        assert "use_location_addressing" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config

    def test_from_config_reconstruction(
            self,
            default_config: dict,
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = DifferentiableSelectCopy(**default_config)
        config = original.get_config()
        reconstructed = DifferentiableSelectCopy.from_config(config)

        assert reconstructed.memory_size == original.memory_size
        assert reconstructed.content_dim == original.content_dim
        assert reconstructed.controller_dim == original.controller_dim
        assert reconstructed.num_read_heads == original.num_read_heads
        assert reconstructed.num_write_heads == original.num_write_heads

    def test_serialization_cycle(
            self,
            default_config: dict,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        memory, controller_state = sample_inputs

        # Create model wrapping the layer
        memory_input = keras.Input(shape=memory.shape[1:], name="memory")
        controller_input = keras.Input(shape=controller_state.shape[1:], name="controller")

        layer = DifferentiableSelectCopy(**default_config)
        new_memory, read_content, _ = layer(memory_input, controller_input)
        model = keras.Model(
            inputs=[memory_input, controller_input],
            outputs=[new_memory, read_content],
        )

        original_outputs = model([memory, controller_state])

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_select_copy.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_outputs = loaded_model([memory, controller_state])

        for orig, loaded in zip(original_outputs, loaded_outputs):
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(loaded),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Outputs should match after serialization",
            )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
            self,
            default_config: dict,
            batch_size: int,
    ) -> None:
        """Test layer handles various batch sizes."""
        layer = DifferentiableSelectCopy(**default_config)

        memory = np.random.randn(
            batch_size, default_config["memory_size"], default_config["content_dim"]
        ).astype(np.float32)
        controller_state = np.random.randn(
            batch_size, default_config["controller_dim"]
        ).astype(np.float32)

        new_memory, read_content, _ = layer(memory, controller_state)

        assert new_memory.shape[0] == batch_size
        assert read_content.shape[0] == batch_size

    def test_content_only_mode(
            self,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with content addressing only."""
        config = {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 1,
            "num_write_heads": 1,
            "use_content_addressing": True,
            "use_location_addressing": False,
        }
        layer = DifferentiableSelectCopy(**config)

        memory = np.random.randn(4, 16, 32).astype(np.float32)
        controller_state = np.random.randn(4, 64).astype(np.float32)

        new_memory, read_content, _ = layer(memory, controller_state)

        assert new_memory.shape == memory.shape

    def test_location_only_mode(
            self,
            sample_inputs: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Test with location addressing only."""
        config = {
            "memory_size": 16,
            "content_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 1,
            "num_write_heads": 1,
            "use_content_addressing": False,
            "use_location_addressing": True,
        }
        layer = DifferentiableSelectCopy(**config)

        memory = np.random.randn(4, 16, 32).astype(np.float32)
        controller_state = np.random.randn(4, 64).astype(np.float32)

        new_memory, read_content, _ = layer(memory, controller_state)

        assert new_memory.shape == memory.shape


# =============================================================================
# SimpleSelectCopy Tests
# =============================================================================


class TestSimpleSelectCopy:
    """Tests for SimpleSelectCopy layer."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for simple select-copy layer."""
        return {
            "input_size": 10,
            "output_size": 10,
            "content_dim": 32,
            "num_copies": 2,
            "temperature": 1.0,
            "use_content_query": True,
        }

    @pytest.fixture
    def sample_input(self, default_config: dict) -> np.ndarray:
        """Sample input tensor."""
        batch_size = 4
        return np.random.randn(
            batch_size, default_config["input_size"], default_config["content_dim"]
        ).astype(np.float32)

    def test_instantiation_valid_config(self, default_config: dict) -> None:
        """Test layer can be instantiated with valid config."""
        layer = SimpleSelectCopy(**default_config)

        assert layer.input_size == default_config["input_size"]
        assert layer.output_size == default_config["output_size"]
        assert layer.content_dim == default_config["content_dim"]
        assert layer.num_copies == default_config["num_copies"]
        assert layer.temperature == default_config["temperature"]
        assert layer.use_content_query == default_config["use_content_query"]

    def test_invalid_input_size(self, default_config: dict) -> None:
        """Test that invalid input_size raises ValueError."""
        default_config["input_size"] = 0
        with pytest.raises(ValueError, match="input_size must be positive"):
            SimpleSelectCopy(**default_config)

    def test_invalid_output_size(self, default_config: dict) -> None:
        """Test that invalid output_size raises ValueError."""
        default_config["output_size"] = -1
        with pytest.raises(ValueError, match="output_size must be positive"):
            SimpleSelectCopy(**default_config)

    def test_invalid_content_dim(self, default_config: dict) -> None:
        """Test that invalid content_dim raises ValueError."""
        default_config["content_dim"] = 0
        with pytest.raises(ValueError, match="content_dim must be positive"):
            SimpleSelectCopy(**default_config)

    def test_invalid_num_copies(self, default_config: dict) -> None:
        """Test that invalid num_copies raises ValueError."""
        default_config["num_copies"] = 0
        with pytest.raises(ValueError, match="num_copies must be positive"):
            SimpleSelectCopy(**default_config)

    def test_invalid_temperature(self, default_config: dict) -> None:
        """Test that invalid temperature raises ValueError."""
        default_config["temperature"] = 0
        with pytest.raises(ValueError, match="temperature must be positive"):
            SimpleSelectCopy(**default_config)

    def test_forward_pass_output_shape(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test forward pass produces correct output shape."""
        layer = SimpleSelectCopy(**default_config)

        output, attention_info = layer(sample_input)

        expected_shape = (
            sample_input.shape[0],
            default_config["output_size"],
            default_config["content_dim"],
        )
        assert output.shape == expected_shape

    def test_attention_info_structure(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test attention_info has correct structure."""
        layer = SimpleSelectCopy(**default_config)

        _, attention_info = layer(sample_input)

        assert "read_weights" in attention_info
        assert "write_weights" in attention_info
        assert len(attention_info["read_weights"]) == default_config["num_copies"]
        assert len(attention_info["write_weights"]) == default_config["num_copies"]

    def test_read_weights_shape(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test read weights have correct shape."""
        layer = SimpleSelectCopy(**default_config)

        _, attention_info = layer(sample_input)

        for weights in attention_info["read_weights"]:
            expected_shape = (sample_input.shape[0], default_config["input_size"])
            assert weights.shape == expected_shape

    def test_write_weights_shape(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test write weights have correct shape."""
        layer = SimpleSelectCopy(**default_config)

        _, attention_info = layer(sample_input)

        for weights in attention_info["write_weights"]:
            expected_shape = (sample_input.shape[0], default_config["output_size"])
            assert weights.shape == expected_shape

    def test_attention_weights_are_valid_distributions(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test attention weights are valid probability distributions."""
        layer = SimpleSelectCopy(**default_config)

        _, attention_info = layer(sample_input)

        for weights in attention_info["read_weights"] + attention_info["write_weights"]:
            weights_np = ops.convert_to_numpy(weights)
            assert np.all(weights_np >= 0), "Weights should be non-negative"
            sums = np.sum(weights_np, axis=-1)
            np.testing.assert_allclose(
                sums,
                np.ones_like(sums),
                rtol=1e-5,
                atol=1e-5,
                err_msg="Weights should sum to 1",
            )

    def test_use_content_query_false(
            self,
            sample_input: np.ndarray,
    ) -> None:
        """Test with use_content_query=False (learned queries)."""
        config = {
            "input_size": 10,
            "output_size": 10,
            "content_dim": 32,
            "num_copies": 2,
            "use_content_query": False,
        }
        layer = SimpleSelectCopy(**config)

        output, attention_info = layer(sample_input)

        expected_shape = (sample_input.shape[0], config["output_size"], config["content_dim"])
        assert output.shape == expected_shape

    def test_different_input_output_sizes(self) -> None:
        """Test with different input and output sizes."""
        config = {
            "input_size": 8,
            "output_size": 16,
            "content_dim": 32,
            "num_copies": 1,
        }
        layer = SimpleSelectCopy(**config)

        inputs = np.random.randn(4, 8, 32).astype(np.float32)
        output, _ = layer(inputs)

        assert output.shape == (4, 16, 32)

    def test_temperature_effect(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test that temperature affects attention sharpness."""
        # Low temperature - sharper attention
        default_config["temperature"] = 0.1
        layer_sharp = SimpleSelectCopy(**default_config)
        _, info_sharp = layer_sharp(sample_input)

        # High temperature - softer attention
        default_config["temperature"] = 10.0
        layer_soft = SimpleSelectCopy(**default_config)
        _, info_soft = layer_soft(sample_input)

        # Sharp attention should have higher max values
        sharp_max = np.max(ops.convert_to_numpy(info_sharp["read_weights"][0]))
        soft_max = np.max(ops.convert_to_numpy(info_soft["read_weights"][0]))

        assert sharp_max > soft_max, "Lower temperature should produce sharper attention"

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape returns correct shapes."""
        layer = SimpleSelectCopy(**default_config)

        input_shape = (None, default_config["input_size"], default_config["content_dim"])
        output_shape, attention_shapes = layer.compute_output_shape(input_shape)

        expected_output = (None, default_config["output_size"], default_config["content_dim"])
        assert output_shape == expected_output

    def test_compute_output_shape_matches_actual(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test compute_output_shape matches actual output shape."""
        layer = SimpleSelectCopy(**default_config)

        computed_shape, _ = layer.compute_output_shape(sample_input.shape)
        actual_output, _ = layer(sample_input)

        # Compare ignoring batch dimension
        assert computed_shape[1:] == actual_output.shape[1:]

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        layer = SimpleSelectCopy(**default_config)
        config = layer.get_config()

        assert "input_size" in config
        assert "output_size" in config
        assert "content_dim" in config
        assert "num_copies" in config
        assert "temperature" in config
        assert "use_content_query" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config

    def test_from_config_reconstruction(
            self,
            default_config: dict,
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = SimpleSelectCopy(**default_config)
        config = original.get_config()
        reconstructed = SimpleSelectCopy.from_config(config)

        assert reconstructed.input_size == original.input_size
        assert reconstructed.output_size == original.output_size
        assert reconstructed.content_dim == original.content_dim
        assert reconstructed.num_copies == original.num_copies
        assert reconstructed.temperature == original.temperature
        assert reconstructed.use_content_query == original.use_content_query

    def test_serialization_cycle(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        # Create model wrapping the layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer = SimpleSelectCopy(**default_config)
        output, _ = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=output)

        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_simple_select_copy.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization",
        )

    def test_serialization_cycle_learned_queries(
            self,
            sample_input: np.ndarray,
    ) -> None:
        """Test serialization with use_content_query=False."""
        config = {
            "input_size": 10,
            "output_size": 10,
            "content_dim": 32,
            "num_copies": 2,
            "use_content_query": False,
        }

        inputs = keras.Input(shape=sample_input.shape[1:])
        layer = SimpleSelectCopy(**config)
        output, _ = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=output)

        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_learned_queries.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Outputs should match after serialization",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_variable_batch_size(
            self,
            default_config: dict,
            batch_size: int,
    ) -> None:
        """Test layer handles various batch sizes."""
        layer = SimpleSelectCopy(**default_config)

        inputs = np.random.randn(
            batch_size, default_config["input_size"], default_config["content_dim"]
        ).astype(np.float32)

        output, _ = layer(inputs)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("num_copies", [1, 2, 4, 8])
    def test_different_num_copies(
            self,
            default_config: dict,
            sample_input: np.ndarray,
            num_copies: int,
    ) -> None:
        """Test layer works with various num_copies values."""
        default_config["num_copies"] = num_copies
        layer = SimpleSelectCopy(**default_config)

        output, attention_info = layer(sample_input)

        assert len(attention_info["read_weights"]) == num_copies
        assert len(attention_info["write_weights"]) == num_copies

    def test_training_mode(
            self,
            default_config: dict,
            sample_input: np.ndarray,
    ) -> None:
        """Test layer behaves correctly in training mode."""
        layer = SimpleSelectCopy(**default_config)

        output_train, _ = layer(sample_input, training=True)
        output_infer, _ = layer(sample_input, training=False)

        # Without dropout, outputs should be identical
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_train),
            ops.convert_to_numpy(output_infer),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Training and inference outputs should match",
        )

    def test_with_regularizer(
            self,
            sample_input: np.ndarray,
    ) -> None:
        """Test layer works with kernel regularizer."""
        config = {
            "input_size": 10,
            "output_size": 10,
            "content_dim": 32,
            "num_copies": 1,
            "kernel_regularizer": "l2",
        }
        layer = SimpleSelectCopy(**config)

        output, _ = layer(sample_input)

        assert output.shape == (sample_input.shape[0], 10, 32)

    def test_with_custom_initializer(
            self,
            sample_input: np.ndarray,
    ) -> None:
        """Test layer works with custom initializer."""
        config = {
            "input_size": 10,
            "output_size": 10,
            "content_dim": 32,
            "num_copies": 1,
            "kernel_initializer": "he_normal",
        }
        layer = SimpleSelectCopy(**config)

        output, _ = layer(sample_input)

        assert output.shape == (sample_input.shape[0], 10, 32)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the select-copy layers."""

    def test_ntm_style_workflow(self) -> None:
        """Test a typical NTM-style read/write workflow."""
        memory_size = 32
        content_dim = 64
        controller_dim = 128
        batch_size = 4

        layer = DifferentiableSelectCopy(
            memory_size=memory_size,
            content_dim=content_dim,
            controller_dim=controller_dim,
            num_read_heads=2,
            num_write_heads=1,
        )

        # Initialize memory
        memory = np.zeros((batch_size, memory_size, content_dim), dtype=np.float32)
        controller_state = np.random.randn(batch_size, controller_dim).astype(np.float32)

        # Multiple steps
        for step in range(5):
            memory, read_content, state = layer(memory, controller_state)

            # Memory should change after writes
            assert memory.shape == (batch_size, memory_size, content_dim)

            # Update controller state based on read content (simplified)
            controller_state = np.random.randn(batch_size, controller_dim).astype(np.float32)

    def test_sequence_to_sequence_simple(self) -> None:
        """Test simple select-copy for sequence transformation."""
        input_size = 16
        output_size = 8
        content_dim = 32
        batch_size = 4

        layer = SimpleSelectCopy(
            input_size=input_size,
            output_size=output_size,
            content_dim=content_dim,
            num_copies=4,
        )

        inputs = np.random.randn(batch_size, input_size, content_dim).astype(np.float32)
        output, attention_info = layer(inputs)

        assert output.shape == (batch_size, output_size, content_dim)

    def test_model_with_multiple_layers(self) -> None:
        """Test model with multiple select-copy layers."""
        # Build a simple model with stacked SimpleSelectCopy layers
        inputs = keras.Input(shape=(10, 32))

        layer1 = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=32,
            num_copies=2,
            name="select_copy_1",
        )
        layer2 = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=32,
            num_copies=2,
            name="select_copy_2",
        )

        x, _ = layer1(inputs)
        output, _ = layer2(x)

        model = keras.Model(inputs=inputs, outputs=output)

        # Test forward pass
        test_input = np.random.randn(4, 10, 32).astype(np.float32)
        result = model(test_input)

        assert result.shape == (4, 10, 32)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "stacked_model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_result = loaded_model(test_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(result),
            ops.convert_to_numpy(loaded_result),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Stacked model outputs should match after serialization",
        )

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the layers."""
        import tensorflow as tf

        layer = SimpleSelectCopy(
            input_size=10,
            output_size=10,
            content_dim=32,
            num_copies=2,
        )

        inputs = tf.random.normal((4, 10, 32))

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output, _ = layer(inputs, training=True)
            loss = tf.reduce_mean(output ** 2)

        gradients = tape.gradient(loss, inputs)

        assert gradients is not None, "Gradients should flow through the layer"
        assert gradients.shape == inputs.shape

    def test_gradient_flow_ntm(self) -> None:
        """Test gradient flow through DifferentiableSelectCopy."""
        import tensorflow as tf

        layer = DifferentiableSelectCopy(
            memory_size=16,
            content_dim=32,
            controller_dim=64,
            num_read_heads=1,
            num_write_heads=1,
        )

        memory = tf.random.normal((4, 16, 32))
        controller_state = tf.random.normal((4, 64))

        with tf.GradientTape() as tape:
            tape.watch([memory, controller_state])
            new_memory, read_content, _ = layer(memory, controller_state, training=True)
            loss = tf.reduce_mean(new_memory ** 2) + tf.reduce_mean(read_content ** 2)

        gradients = tape.gradient(loss, [memory, controller_state])

        assert gradients[0] is not None, "Gradients should flow to memory"
        assert gradients[1] is not None, "Gradients should flow to controller state"