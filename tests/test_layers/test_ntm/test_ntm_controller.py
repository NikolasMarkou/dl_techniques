"""
Tests for Neural Turing Machine (NTM) Model Layers.

This module contains comprehensive tests for:
- NTMConfig
- NTMController
- NTMCell
- NTMLayer
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.ntm.ntm_controller import (
    NTMConfig,
    NTMController,
    NTMCell,
    NTMLayer,
)


# =============================================================================
# NTMConfig Tests
# =============================================================================


class TestNTMConfig:
    """Tests for NTMConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = NTMConfig()

        assert config.memory_size == 128
        assert config.memory_dim == 64
        assert config.num_read_heads == 1
        assert config.num_write_heads == 1
        assert config.controller_dim == 256
        assert config.controller_type == "lstm"
        assert config.shift_range == 3
        assert config.use_memory_init is True
        assert config.epsilon == 1e-6

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = NTMConfig(
            memory_size=64,
            memory_dim=32,
            num_read_heads=2,
            num_write_heads=2,
            controller_dim=128,
            controller_type="gru",
            shift_range=5,
        )

        assert config.memory_size == 64
        assert config.memory_dim == 32
        assert config.num_read_heads == 2
        assert config.num_write_heads == 2
        assert config.controller_dim == 128
        assert config.controller_type == "gru"
        assert config.shift_range == 5

    def test_invalid_memory_size(self) -> None:
        """Test that invalid memory_size raises ValueError."""
        with pytest.raises(ValueError, match="memory_size must be positive"):
            NTMConfig(memory_size=0)

    def test_invalid_memory_dim(self) -> None:
        """Test that invalid memory_dim raises ValueError."""
        with pytest.raises(ValueError, match="memory_dim must be positive"):
            NTMConfig(memory_dim=-1)

    def test_invalid_controller_type(self) -> None:
        """Test that invalid controller_type raises ValueError."""
        with pytest.raises(ValueError, match="controller_type must be"):
            NTMConfig(controller_type="transformer")

    def test_invalid_shift_range_even(self) -> None:
        """Test that even shift_range raises ValueError."""
        with pytest.raises(ValueError, match="shift_range must be a positive odd"):
            NTMConfig(shift_range=4)

    def test_invalid_shift_range_negative(self) -> None:
        """Test that negative shift_range raises ValueError."""
        with pytest.raises(ValueError, match="shift_range must be a positive odd"):
            NTMConfig(shift_range=-1)

    def test_to_dict(self) -> None:
        """Test config serialization to dict."""
        config = NTMConfig(memory_size=32, memory_dim=16)
        config_dict = config.to_dict()

        assert config_dict["memory_size"] == 32
        assert config_dict["memory_dim"] == 16
        assert "controller_type" in config_dict
        assert "shift_range" in config_dict

    def test_from_dict(self) -> None:
        """Test config deserialization from dict."""
        config_dict = {
            "memory_size": 64,
            "memory_dim": 32,
            "num_read_heads": 2,
            "num_write_heads": 1,
            "controller_dim": 128,
            "controller_type": "gru",
            "shift_range": 3,
            "use_memory_init": False,
            "epsilon": 1e-8,
        }

        config = NTMConfig.from_dict(config_dict)

        assert config.memory_size == 64
        assert config.memory_dim == 32
        assert config.controller_type == "gru"

    def test_roundtrip_serialization(self) -> None:
        """Test config survives serialization roundtrip."""
        original = NTMConfig(
            memory_size=48,
            memory_dim=24,
            controller_type="feedforward",
        )

        config_dict = original.to_dict()
        restored = NTMConfig.from_dict(config_dict)

        assert restored.memory_size == original.memory_size
        assert restored.memory_dim == original.memory_dim
        assert restored.controller_type == original.controller_type


# =============================================================================
# NTMController Tests
# =============================================================================


class TestNTMController:
    """Tests for NTMController layer."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for controller."""
        return {
            "units": 64,
            "controller_type": "lstm",
        }

    @pytest.fixture
    def sample_input(self) -> np.ndarray:
        """Sample input tensor."""
        batch_size = 4
        input_dim = 32
        return np.random.randn(batch_size, input_dim).astype(np.float32)

    def test_instantiation_lstm(self, default_config: dict) -> None:
        """Test LSTM controller instantiation."""
        controller = NTMController(**default_config)

        assert controller.units == default_config["units"]
        assert controller.controller_type == "lstm"

    def test_instantiation_gru(self) -> None:
        """Test GRU controller instantiation."""
        controller = NTMController(units=64, controller_type="gru")

        assert controller.controller_type == "gru"

    def test_instantiation_feedforward(self) -> None:
        """Test feedforward controller instantiation."""
        controller = NTMController(units=64, controller_type="feedforward")

        assert controller.controller_type == "feedforward"

    def test_invalid_units(self) -> None:
        """Test that invalid units raises ValueError."""
        with pytest.raises(ValueError, match="units must be positive"):
            NTMController(units=0)

    def test_invalid_controller_type(self) -> None:
        """Test that invalid controller_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid controller_type"):
            NTMController(units=64, controller_type="invalid")

    def test_forward_pass_lstm(
        self,
        default_config: dict,
        sample_input: np.ndarray,
    ) -> None:
        """Test LSTM forward pass."""
        controller = NTMController(**default_config)
        batch_size = sample_input.shape[0]

        states = controller.get_initial_state(batch_size)
        output, new_states = controller(sample_input, states=states)

        assert output.shape == (batch_size, default_config["units"])
        assert len(new_states) == 2  # h and c for LSTM

    def test_forward_pass_gru(self, sample_input: np.ndarray) -> None:
        """Test GRU forward pass."""
        controller = NTMController(units=64, controller_type="gru")
        batch_size = sample_input.shape[0]

        states = controller.get_initial_state(batch_size)
        output, new_states = controller(sample_input, states=states)

        assert output.shape == (batch_size, 64)
        assert len(new_states) == 1  # h only for GRU

    def test_forward_pass_feedforward(self, sample_input: np.ndarray) -> None:
        """Test feedforward forward pass."""
        controller = NTMController(units=64, controller_type="feedforward")

        output, new_states = controller(sample_input, states=None)

        assert output.shape == (sample_input.shape[0], 64)
        assert len(new_states) == 0  # No state for feedforward

    def test_get_initial_state_lstm(self, default_config: dict) -> None:
        """Test LSTM initial state generation."""
        controller = NTMController(**default_config)
        batch_size = 4

        states = controller.get_initial_state(batch_size)

        assert len(states) == 2
        assert states[0].shape == (batch_size, default_config["units"])
        assert states[1].shape == (batch_size, default_config["units"])

    def test_get_initial_state_gru(self) -> None:
        """Test GRU initial state generation."""
        controller = NTMController(units=64, controller_type="gru")
        batch_size = 4

        states = controller.get_initial_state(batch_size)

        assert len(states) == 1
        assert states[0].shape == (batch_size, 64)

    def test_get_initial_state_feedforward(self) -> None:
        """Test feedforward initial state (empty)."""
        controller = NTMController(units=64, controller_type="feedforward")

        states = controller.get_initial_state(4)

        assert len(states) == 0

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape."""
        controller = NTMController(**default_config)

        input_shape = (None, 32)
        output_shape, state_shapes = controller.compute_output_shape(input_shape)

        assert output_shape == (None, default_config["units"])
        # For LSTM default config, state_shapes should be a list of 2 shapes
        assert isinstance(state_shapes, list)
        assert len(state_shapes) == 2
        assert state_shapes[0] == (None, default_config["units"])
        assert state_shapes[1] == (None, default_config["units"])

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        controller = NTMController(**default_config)
        config = controller.get_config()

        assert "units" in config
        assert "controller_type" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config

    def test_from_config_reconstruction(
        self,
        default_config: dict,
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = NTMController(**default_config)
        config = original.get_config()
        reconstructed = NTMController.from_config(config)

        assert reconstructed.units == original.units
        assert reconstructed.controller_type == original.controller_type

    def test_serialization_cycle(
        self,
        default_config: dict,
        sample_input: np.ndarray,
    ) -> None:
        """Test full save/load cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        controller = NTMController(**default_config)
        output, _ = controller(inputs, states=None)
        model = keras.Model(inputs=inputs, outputs=output)

        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_controller.keras")
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

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
        self,
        default_config: dict,
        batch_size: int,
    ) -> None:
        """Test controller handles various batch sizes."""
        controller = NTMController(**default_config)

        inputs = np.random.randn(batch_size, 32).astype(np.float32)
        states = controller.get_initial_state(batch_size)
        output, _ = controller(inputs, states=states)

        assert output.shape[0] == batch_size


# =============================================================================
# NTMCell Tests
# =============================================================================


class TestNTMCell:
    """Tests for NTMCell layer."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for NTM cell."""
        return {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 1,
            "num_write_heads": 1,
            "controller_type": "lstm",
            "shift_range": 3,
        }

    @pytest.fixture
    def sample_input(self, default_config: dict) -> np.ndarray:
        """Sample input tensor."""
        batch_size = 4
        input_dim = 16
        return np.random.randn(batch_size, input_dim).astype(np.float32)

    def test_instantiation_valid_config(self, default_config: dict) -> None:
        """Test cell can be instantiated with valid config."""
        cell = NTMCell(**default_config)

        assert cell.memory_size == default_config["memory_size"]
        assert cell.memory_dim == default_config["memory_dim"]
        assert cell.controller_dim == default_config["controller_dim"]
        assert cell.num_read_heads == default_config["num_read_heads"]
        assert cell.num_write_heads == default_config["num_write_heads"]

    def test_invalid_memory_size(self, default_config: dict) -> None:
        """Test that invalid memory_size raises ValueError."""
        default_config["memory_size"] = 0
        with pytest.raises(ValueError, match="memory_size must be positive"):
            NTMCell(**default_config)

    def test_invalid_memory_dim(self, default_config: dict) -> None:
        """Test that invalid memory_dim raises ValueError."""
        default_config["memory_dim"] = -1
        with pytest.raises(ValueError, match="memory_dim must be positive"):
            NTMCell(**default_config)

    def test_invalid_controller_dim(self, default_config: dict) -> None:
        """Test that invalid controller_dim raises ValueError."""
        default_config["controller_dim"] = 0
        with pytest.raises(ValueError, match="controller_dim must be positive"):
            NTMCell(**default_config)

    def test_invalid_shift_range_even(self, default_config: dict) -> None:
        """Test that even shift_range raises ValueError."""
        default_config["shift_range"] = 4
        with pytest.raises(ValueError, match="shift_range must be a positive odd"):
            NTMCell(**default_config)

    def test_state_size_lstm(self, default_config: dict) -> None:
        """Test state_size property for LSTM controller."""
        cell = NTMCell(**default_config)
        state_size = cell.state_size

        # LSTM: 2 (h, c) + 1 (memory) + num_read (vectors) + num_read (weights) + num_write (weights)
        expected_count = 2 + 1 + 1 + 1 + 1
        assert len(state_size) == expected_count

    def test_state_size_gru(self, default_config: dict) -> None:
        """Test state_size property for GRU controller."""
        default_config["controller_type"] = "gru"
        cell = NTMCell(**default_config)
        state_size = cell.state_size

        # GRU: 1 (h) + 1 (memory) + num_read + num_read + num_write
        expected_count = 1 + 1 + 1 + 1 + 1
        assert len(state_size) == expected_count

    def test_state_size_feedforward(self, default_config: dict) -> None:
        """Test state_size property for feedforward controller."""
        default_config["controller_type"] = "feedforward"
        cell = NTMCell(**default_config)
        state_size = cell.state_size

        # Feedforward: 0 + 1 (memory) + num_read + num_read + num_write
        expected_count = 0 + 1 + 1 + 1 + 1
        assert len(state_size) == expected_count

    def test_output_size(self, default_config: dict) -> None:
        """Test output_size property."""
        cell = NTMCell(**default_config)

        expected = default_config["controller_dim"] + (
            default_config["num_read_heads"] * default_config["memory_dim"]
        )
        assert cell.output_size == expected

    def test_forward_pass(
        self,
        default_config: dict,
        sample_input: np.ndarray,
    ) -> None:
        """Test forward pass through cell."""
        cell = NTMCell(**default_config)
        batch_size = sample_input.shape[0]

        states = cell.get_initial_state(batch_size=batch_size)
        output, new_states = cell(sample_input, states=states)

        assert output.shape == (batch_size, cell.output_size)
        assert len(new_states) == len(states)

    def test_get_initial_state(
        self,
        default_config: dict,
        sample_input: np.ndarray,
    ) -> None:
        """Test initial state generation."""
        cell = NTMCell(**default_config)
        batch_size = sample_input.shape[0]

        states = cell.get_initial_state(batch_size=batch_size)

        assert len(states) == len(cell.state_size)

        # Check controller states
        if default_config["controller_type"] == "lstm":
            assert states[0].shape == (batch_size, default_config["controller_dim"])
            assert states[1].shape == (batch_size, default_config["controller_dim"])

    def test_multiple_read_heads(self) -> None:
        """Test cell with multiple read heads."""
        config = {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 3,
            "num_write_heads": 1,
        }
        cell = NTMCell(**config)
        batch_size = 4

        expected_output = config["controller_dim"] + (3 * config["memory_dim"])
        assert cell.output_size == expected_output

        inputs = np.random.randn(batch_size, 16).astype(np.float32)
        states = cell.get_initial_state(batch_size=batch_size)
        output, _ = cell(inputs, states=states)

        assert output.shape == (batch_size, expected_output)

    def test_multiple_write_heads(self) -> None:
        """Test cell with multiple write heads."""
        config = {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "num_read_heads": 1,
            "num_write_heads": 2,
        }
        cell = NTMCell(**config)
        batch_size = 4

        inputs = np.random.randn(batch_size, 16).astype(np.float32)
        states = cell.get_initial_state(batch_size=batch_size)
        output, new_states = cell(inputs, states=states)

        assert output.shape[0] == batch_size
        assert len(new_states) == len(states)

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape."""
        cell = NTMCell(**default_config)

        input_shape = (None, 16)
        output_shape = cell.compute_output_shape(input_shape)

        assert output_shape == (None, cell.output_size)

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        cell = NTMCell(**default_config)
        config = cell.get_config()

        assert "memory_size" in config
        assert "memory_dim" in config
        assert "controller_dim" in config
        assert "num_read_heads" in config
        assert "num_write_heads" in config
        assert "controller_type" in config
        assert "shift_range" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config
        assert "epsilon" in config

    def test_from_config_reconstruction(
        self,
        default_config: dict,
    ) -> None:
        """Test cell can be reconstructed from config."""
        original = NTMCell(**default_config)
        config = original.get_config()
        reconstructed = NTMCell.from_config(config)

        assert reconstructed.memory_size == original.memory_size
        assert reconstructed.memory_dim == original.memory_dim
        assert reconstructed.controller_dim == original.controller_dim
        assert reconstructed.controller_type == original.controller_type

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
        self,
        default_config: dict,
        batch_size: int,
    ) -> None:
        """Test cell handles various batch sizes."""
        cell = NTMCell(**default_config)

        inputs = np.random.randn(batch_size, 16).astype(np.float32)
        states = cell.get_initial_state(batch_size=batch_size)
        output, _ = cell(inputs, states=states)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("controller_type", ["lstm", "gru", "feedforward"])
    def test_different_controller_types(
        self,
        default_config: dict,
        controller_type: str,
    ) -> None:
        """Test cell works with different controller types."""
        default_config["controller_type"] = controller_type
        cell = NTMCell(**default_config)
        batch_size = 4

        inputs = np.random.randn(batch_size, 16).astype(np.float32)
        states = cell.get_initial_state(batch_size=batch_size)
        output, new_states = cell(inputs, states=states)

        assert output.shape == (batch_size, cell.output_size)


# =============================================================================
# NTMLayer Tests
# =============================================================================


class TestNTMLayer:
    """Tests for NTMLayer (complete NTM with RNN wrapper)."""

    @pytest.fixture
    def default_config(self) -> dict:
        """Default configuration for NTM layer."""
        return {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "output_dim": 10,
            "num_read_heads": 1,
            "num_write_heads": 1,
            "controller_type": "lstm",
            "shift_range": 3,
            "return_sequences": True,
            "return_state": False,
        }

    @pytest.fixture
    def sample_sequence(self, default_config: dict) -> np.ndarray:
        """Sample sequence input tensor."""
        batch_size = 4
        seq_len = 10
        input_dim = 16
        return np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

    def test_instantiation_valid_config(self, default_config: dict) -> None:
        """Test layer can be instantiated with valid config."""
        layer = NTMLayer(**default_config)

        assert layer.memory_size == default_config["memory_size"]
        assert layer.memory_dim == default_config["memory_dim"]
        assert layer.controller_dim == default_config["controller_dim"]
        assert layer.output_dim == default_config["output_dim"]

    def test_forward_pass_return_sequences(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test forward pass with return_sequences=True."""
        layer = NTMLayer(**default_config)

        output = layer(sample_sequence)

        expected_shape = (
            sample_sequence.shape[0],
            sample_sequence.shape[1],
            default_config["output_dim"],
        )
        assert output.shape == expected_shape

    def test_forward_pass_no_return_sequences(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test forward pass with return_sequences=False."""
        default_config["return_sequences"] = False
        layer = NTMLayer(**default_config)

        output = layer(sample_sequence)

        expected_shape = (
            sample_sequence.shape[0],
            default_config["output_dim"],
        )
        assert output.shape == expected_shape

    def test_forward_pass_return_state(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test forward pass with return_state=True."""
        default_config["return_state"] = True
        layer = NTMLayer(**default_config)

        output, states = layer(sample_sequence)

        assert output.shape[0] == sample_sequence.shape[0]
        assert states is not None
        assert len(states) > 0

    def test_forward_pass_no_output_projection(
        self,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test forward pass without output projection."""
        config = {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "output_dim": None,  # No projection
            "return_sequences": True,
        }
        layer = NTMLayer(**config)

        output = layer(sample_sequence)

        # Output dim should be controller_dim + num_read_heads * memory_dim
        expected_dim = config["controller_dim"] + config["memory_dim"]
        assert output.shape == (
            sample_sequence.shape[0],
            sample_sequence.shape[1],
            expected_dim,
        )

    def test_with_initial_state(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test forward pass with provided initial state."""
        layer = NTMLayer(**default_config)
        batch_size = sample_sequence.shape[0]

        # Get initial state from cell
        initial_state = layer.cell.get_initial_state(batch_size=batch_size)

        output = layer(sample_sequence, initial_state=initial_state)

        assert output.shape[0] == batch_size

    def test_compute_output_shape(self, default_config: dict) -> None:
        """Test compute_output_shape."""
        layer = NTMLayer(**default_config)

        input_shape = (None, 10, 16)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 10, default_config["output_dim"])

    def test_compute_output_shape_no_sequences(self, default_config: dict) -> None:
        """Test compute_output_shape with return_sequences=False."""
        default_config["return_sequences"] = False
        layer = NTMLayer(**default_config)

        input_shape = (None, 10, 16)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, default_config["output_dim"])

    def test_get_config_complete(self, default_config: dict) -> None:
        """Test get_config returns all constructor arguments."""
        layer = NTMLayer(**default_config)
        config = layer.get_config()

        assert "memory_size" in config
        assert "memory_dim" in config
        assert "controller_dim" in config
        assert "output_dim" in config
        assert "num_read_heads" in config
        assert "num_write_heads" in config
        assert "controller_type" in config
        assert "shift_range" in config
        assert "return_sequences" in config
        assert "return_state" in config
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config

    def test_from_config_reconstruction(
        self,
        default_config: dict,
    ) -> None:
        """Test layer can be reconstructed from config."""
        original = NTMLayer(**default_config)
        config = original.get_config()
        reconstructed = NTMLayer.from_config(config)

        assert reconstructed.memory_size == original.memory_size
        assert reconstructed.memory_dim == original.memory_dim
        assert reconstructed.controller_dim == original.controller_dim
        assert reconstructed.output_dim == original.output_dim
        assert reconstructed.return_sequences == original.return_sequences

    def test_serialization_cycle(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test full save/load cycle preserves functionality."""
        inputs = keras.Input(shape=sample_sequence.shape[1:])
        layer = NTMLayer(**default_config)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_output = model(sample_sequence)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_ntm_layer.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_sequence)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Outputs should match after serialization",
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_variable_batch_size(
        self,
        default_config: dict,
        batch_size: int,
    ) -> None:
        """Test layer handles various batch sizes."""
        layer = NTMLayer(**default_config)

        inputs = np.random.randn(batch_size, 10, 16).astype(np.float32)
        output = layer(inputs)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("seq_len", [1, 5, 20, 50])
    def test_variable_sequence_length(
        self,
        default_config: dict,
        seq_len: int,
    ) -> None:
        """Test layer handles various sequence lengths."""
        layer = NTMLayer(**default_config)
        batch_size = 4

        inputs = np.random.randn(batch_size, seq_len, 16).astype(np.float32)
        output = layer(inputs)

        assert output.shape[1] == seq_len

    @pytest.mark.parametrize("controller_type", ["lstm", "gru", "feedforward"])
    def test_different_controller_types(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
        controller_type: str,
    ) -> None:
        """Test layer works with different controller types."""
        default_config["controller_type"] = controller_type
        layer = NTMLayer(**default_config)

        output = layer(sample_sequence)

        assert output.shape[0] == sample_sequence.shape[0]

    def test_training_mode(
        self,
        default_config: dict,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test layer behaves correctly in training mode."""
        layer = NTMLayer(**default_config)

        output_train = layer(sample_sequence, training=True)
        output_infer = layer(sample_sequence, training=False)

        # Outputs should have same shape
        assert output_train.shape == output_infer.shape

    def test_multiple_heads(
        self,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test layer with multiple read and write heads."""
        config = {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "output_dim": 10,
            "num_read_heads": 3,
            "num_write_heads": 2,
        }
        layer = NTMLayer(**config)

        output = layer(sample_sequence)

        assert output.shape == (
            sample_sequence.shape[0],
            sample_sequence.shape[1],
            config["output_dim"],
        )

    def test_with_regularizer(
        self,
        sample_sequence: np.ndarray,
    ) -> None:
        """Test layer works with kernel regularizer."""
        config = {
            "memory_size": 16,
            "memory_dim": 32,
            "controller_dim": 64,
            "output_dim": 10,
            "kernel_regularizer": "l2",
        }
        layer = NTMLayer(**config)

        output = layer(sample_sequence)

        assert output.shape[0] == sample_sequence.shape[0]


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for NTM components."""

    def test_ntm_copy_task_structure(self) -> None:
        """Test NTM can be structured for copy task."""
        batch_size = 4
        seq_len = 10
        input_dim = 8

        layer = NTMLayer(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            output_dim=input_dim,  # Same as input for copy task
            num_read_heads=1,
            num_write_heads=1,
            return_sequences=True,
        )

        inputs = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        output = layer(inputs)

        # Output should have same shape as input for copy task
        assert output.shape == inputs.shape

    def test_ntm_sequence_processing(self) -> None:
        """Test NTM processes sequences correctly step by step."""
        batch_size = 2
        seq_len = 5
        input_dim = 8

        cell = NTMCell(
            memory_size=16,
            memory_dim=8,
            controller_dim=32,
            num_read_heads=1,
            num_write_heads=1,
        )

        # Process sequence step by step
        inputs = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        states = cell.get_initial_state(batch_size=batch_size)

        outputs = []
        for t in range(seq_len):
            output, states = cell(inputs[:, t, :], states=states)
            outputs.append(output)

        # All outputs should have consistent shape
        for out in outputs:
            assert out.shape == (batch_size, cell.output_size)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the NTM."""
        import tensorflow as tf

        layer = NTMLayer(
            memory_size=16,
            memory_dim=16,
            controller_dim=32,
            output_dim=8,
        )

        inputs = tf.random.normal((4, 10, 8))
        targets = tf.random.normal((4, 10, 8))

        with tf.GradientTape() as tape:
            output = layer(inputs, training=True)
            loss = tf.reduce_mean((output - targets) ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients exist for all trainable variables
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad is not None, f"No gradient for {var.name}"

    def test_model_compilation(self) -> None:
        """Test NTM layer in a compiled model."""
        inputs = keras.Input(shape=(10, 8))
        ntm_output = NTMLayer(
            memory_size=16,
            memory_dim=16,
            controller_dim=32,
            output_dim=8,
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=ntm_output)

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        # Test that model can be called
        x = np.random.randn(4, 10, 8).astype(np.float32)
        y = np.random.randn(4, 10, 8).astype(np.float32)

        # Just verify it runs without error (not training)
        model(x)
        loss = model.evaluate(x, y, verbose=0)
        assert loss > 0

    def test_stacked_ntm_layers(self) -> None:
        """Test stacking multiple NTM layers."""
        inputs = keras.Input(shape=(10, 16))

        x = NTMLayer(
            memory_size=16,
            memory_dim=16,
            controller_dim=32,
            output_dim=32,
            name="ntm_1",
        )(inputs)

        output = NTMLayer(
            memory_size=16,
            memory_dim=16,
            controller_dim=32,
            output_dim=8,
            name="ntm_2",
        )(x)

        model = keras.Model(inputs=inputs, outputs=output)

        test_input = np.random.randn(4, 10, 16).astype(np.float32)
        result = model(test_input)

        assert result.shape == (4, 10, 8)

    def test_serialization_with_multiple_heads(self) -> None:
        """Test serialization with complex configuration."""
        config = {
            "memory_size": 32,
            "memory_dim": 24,
            "controller_dim": 48,
            "output_dim": 16,
            "num_read_heads": 3,
            "num_write_heads": 2,
            "controller_type": "gru",
            "shift_range": 5,
        }

        inputs = keras.Input(shape=(10, 16))
        layer = NTMLayer(**config)
        outputs = layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = np.random.randn(4, 10, 16).astype(np.float32)
        original_output = model(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "complex_ntm.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(test_input)

        np.testing.assert_allclose(
            ops.convert_to_numpy(original_output),
            ops.convert_to_numpy(loaded_output),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Complex NTM should serialize correctly",
        )