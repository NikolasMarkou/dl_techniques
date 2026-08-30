"""
Tests for NTM baseline_ntm module.

Tests cover:
    - NTMMemory
    - NTMReadHead
    - NTMWriteHead
    - NTMController
    - NTMCell
    - NeuralTuringMachine
    - create_ntm factory
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.memory.ntm_interface import (
    AddressingMode,
    MemoryState,
    HeadState,
    NTMConfig,
    circular_convolution,
)
from dl_techniques.layers.memory.baseline_ntm import (
    NTMMemory,
    NTMReadHead,
    NTMWriteHead,
    NTMController,
    NTMCell,
    NeuralTuringMachine,
    create_ntm,
)


# ---------------------------------------------------------------------
# NTMMemory Tests
# ---------------------------------------------------------------------


class TestNTMMemory:
    """Tests for NTMMemory layer."""

    def test_init(self):
        """Test initialization."""
        memory = NTMMemory(memory_size=32, memory_dim=16)
        assert memory.memory_size == 32
        assert memory.memory_dim == 16

    def test_initialize_state(self):
        """Test state initialization."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16

        memory = NTMMemory(memory_size=memory_size, memory_dim=memory_dim)
        state = memory.initialize_state(batch_size)

        assert isinstance(state, MemoryState)
        assert ops.shape(state.memory) == (batch_size, memory_size, memory_dim)
        assert ops.shape(state.usage) == (batch_size, memory_size)

    def test_read(self):
        """Test read operation."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16

        memory_module = NTMMemory(memory_size=memory_size, memory_dim=memory_dim)
        state = memory_module.initialize_state(batch_size)

        # Set memory to known values
        state = MemoryState(
            memory=keras.random.normal((batch_size, memory_size, memory_dim), seed=42),
            usage=state.usage,
        )

        # Create read weights (one-hot at position 0)
        # Using ops.scatter which is effectively scatter_nd in Keras 3 with these args
        read_weights = ops.scatter(
            [[i, 0] for i in range(batch_size)],
            ops.ones((batch_size,)),
            (batch_size, memory_size),
        )

        read_vector = memory_module.read(state, read_weights)

        assert ops.shape(read_vector) == (batch_size, memory_dim)

        # With one-hot weights, read vector should match first memory slot
        memory_np = keras.ops.convert_to_numpy(state.memory)
        read_np = keras.ops.convert_to_numpy(read_vector)

        np.testing.assert_allclose(
            read_np,
            memory_np[:, 0, :],
            rtol=1e-5,
            atol=1e-5,
            err_msg="Read with one-hot weights should return exact slot",
        )

    def test_write(self):
        """Test write operation."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16

        memory_module = NTMMemory(memory_size=memory_size, memory_dim=memory_dim)
        state = memory_module.initialize_state(batch_size)

        # Create write weights (one-hot at position 5)
        # Using ops.scatter which is effectively scatter_nd in Keras 3 with these args
        write_weights = ops.scatter(
            [[i, 5] for i in range(batch_size)],
            ops.ones((batch_size,)),
            (batch_size, memory_size),
        )

        # Erase everything and add new content
        erase_vector = ops.ones((batch_size, memory_dim))
        add_vector = keras.random.normal((batch_size, memory_dim), seed=42)

        new_state = memory_module.write(state, write_weights, erase_vector, add_vector)

        assert isinstance(new_state, MemoryState)
        assert ops.shape(new_state.memory) == (batch_size, memory_size, memory_dim)

        # With one-hot write and full erase, slot 5 should match add_vector
        new_memory_np = keras.ops.convert_to_numpy(new_state.memory)
        add_np = keras.ops.convert_to_numpy(add_vector)

        np.testing.assert_allclose(
            new_memory_np[:, 5, :],
            add_np,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Write with one-hot weights and full erase should replace slot",
        )

    def test_serialization(self):
        """Test get_config."""
        memory = NTMMemory(memory_size=32, memory_dim=16, epsilon=1e-5)
        config = memory.get_config()

        assert config["memory_size"] == 32
        assert config["memory_dim"] == 16
        assert config["epsilon"] == 1e-5


# ---------------------------------------------------------------------
# NTMReadHead Tests
# ---------------------------------------------------------------------


class TestNTMReadHead:
    """Tests for NTMReadHead layer."""

    def test_init(self):
        """Test initialization."""
        head = NTMReadHead(
            memory_size=32,
            memory_dim=16,
            addressing_mode=AddressingMode.HYBRID,
            shift_range=3,
        )
        assert head.memory_size == 32
        assert head.memory_dim == 16
        assert head.shift_range == 3

    def test_build(self):
        """Test build creates all sub-layers."""
        head = NTMReadHead(memory_size=32, memory_dim=16)
        head.build((None, 64))

        assert head.built
        assert head.key_dense is not None
        assert head.beta_dense is not None
        assert head.gate_dense is not None
        assert head.shift_dense is not None
        assert head.gamma_dense is not None

    def test_content_addressing(self):
        """Test content-based addressing."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16

        head = NTMReadHead(memory_size=memory_size, memory_dim=memory_dim)

        key = keras.random.normal((batch_size, 1, memory_dim), seed=42)
        beta = ops.ones((batch_size, 1)) * 10.0  # High beta for sharp attention
        memory = keras.random.normal((batch_size, memory_size, memory_dim), seed=43)

        content_weights = head.content_addressing(key, beta, memory)

        assert ops.shape(content_weights) == (batch_size, memory_size)

        # Weights should sum to 1
        weights_np = keras.ops.convert_to_numpy(content_weights)
        np.testing.assert_allclose(
            np.sum(weights_np, axis=-1),
            np.ones(batch_size),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_compute_addressing(self):
        """Test full addressing computation."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16
        controller_dim = 64

        head = NTMReadHead(memory_size=memory_size, memory_dim=memory_dim)
        head.build((None, controller_dim))

        controller_output = keras.random.normal((batch_size, controller_dim), seed=42)
        memory_state = MemoryState(
            memory=keras.random.normal((batch_size, memory_size, memory_dim), seed=43)
        )
        prev_weights = ops.softmax(
            keras.random.normal((batch_size, memory_size), seed=44), axis=-1
        )

        weights, head_state = head.compute_addressing(
            controller_output, memory_state, prev_weights
        )

        assert ops.shape(weights) == (batch_size, memory_size)
        assert isinstance(head_state, HeadState)
        assert head_state.key is not None
        assert head_state.beta is not None
        assert head_state.gate is not None

    def test_serialization(self):
        """Test get_config."""
        head = NTMReadHead(
            memory_size=32,
            memory_dim=16,
            shift_range=5,
        )
        config = head.get_config()

        assert config["memory_size"] == 32
        assert config["memory_dim"] == 16
        assert config["shift_range"] == 5


# ---------------------------------------------------------------------
# NTMWriteHead Tests
# ---------------------------------------------------------------------


class TestNTMWriteHead:
    """Tests for NTMWriteHead layer."""

    def test_init(self):
        """Test initialization."""
        head = NTMWriteHead(
            memory_size=32,
            memory_dim=16,
        )
        assert head.memory_size == 32
        assert head.memory_dim == 16

    def test_build(self):
        """Test build creates all sub-layers including erase/add."""
        head = NTMWriteHead(memory_size=32, memory_dim=16)
        head.build((None, 64))

        assert head.built
        assert head.erase_dense is not None
        assert head.add_dense is not None

    def test_compute_addressing_includes_erase_add(self):
        """Test that compute_addressing returns erase and add vectors."""
        batch_size = 4
        memory_size = 32
        memory_dim = 16
        controller_dim = 64

        head = NTMWriteHead(memory_size=memory_size, memory_dim=memory_dim)
        head.build((None, controller_dim))

        controller_output = keras.random.normal((batch_size, controller_dim), seed=42)
        memory_state = MemoryState(
            memory=keras.random.normal((batch_size, memory_size, memory_dim), seed=43)
        )
        prev_weights = ops.softmax(
            keras.random.normal((batch_size, memory_size), seed=44), axis=-1
        )

        weights, head_state = head.compute_addressing(
            controller_output, memory_state, prev_weights
        )

        assert ops.shape(weights) == (batch_size, memory_size)
        assert head_state.erase_vector is not None
        assert head_state.add_vector is not None
        assert ops.shape(head_state.erase_vector) == (batch_size, memory_dim)
        assert ops.shape(head_state.add_vector) == (batch_size, memory_dim)

        # Erase vector should be in [0, 1] due to sigmoid
        erase_np = keras.ops.convert_to_numpy(head_state.erase_vector)
        assert np.all(erase_np >= 0) and np.all(erase_np <= 1)

    def test_serialization(self):
        """Test get_config."""
        head = NTMWriteHead(memory_size=32, memory_dim=16)
        config = head.get_config()

        assert config["memory_size"] == 32
        assert config["memory_dim"] == 16


# ---------------------------------------------------------------------
# NTMController Tests
# ---------------------------------------------------------------------


class TestNTMController:
    """Tests for NTMController layer."""

    def test_init_lstm(self):
        """Test LSTM controller initialization."""
        controller = NTMController(
            controller_dim=64,
            controller_type="lstm",
        )
        assert controller.controller_dim == 64
        assert controller.controller_type == "lstm"
        assert isinstance(controller.core, keras.layers.LSTMCell)

    def test_init_gru(self):
        """Test GRU controller initialization."""
        controller = NTMController(
            controller_dim=64,
            controller_type="gru",
        )
        assert controller.controller_type == "gru"
        assert isinstance(controller.core, keras.layers.GRUCell)

    def test_init_feedforward(self):
        """Test feedforward controller initialization."""
        controller = NTMController(
            controller_dim=64,
            controller_type="feedforward",
        )
        assert controller.controller_type == "feedforward"
        assert isinstance(controller.core, keras.layers.Dense)

    def test_initialize_state_lstm(self):
        """Test state initialization for LSTM."""
        batch_size = 4
        controller_dim = 64

        controller = NTMController(controller_dim=controller_dim, controller_type="lstm")
        states = controller.initialize_state(batch_size)

        assert len(states) == 2  # h and c states
        assert ops.shape(states[0]) == (batch_size, controller_dim)
        assert ops.shape(states[1]) == (batch_size, controller_dim)

    def test_initialize_state_gru(self):
        """Test state initialization for GRU."""
        batch_size = 4
        controller_dim = 64

        controller = NTMController(controller_dim=controller_dim, controller_type="gru")
        states = controller.initialize_state(batch_size)

        assert len(states) == 1
        assert ops.shape(states[0]) == (batch_size, controller_dim)

    def test_initialize_state_feedforward(self):
        """Test state initialization for feedforward (should be None)."""
        controller = NTMController(controller_dim=64, controller_type="feedforward")
        states = controller.initialize_state(4)

        assert states is None

    def test_call_lstm(self):
        """Test call with LSTM controller."""
        batch_size = 4
        input_dim = 32
        controller_dim = 64

        controller = NTMController(controller_dim=controller_dim, controller_type="lstm")
        controller.build((None, input_dim))

        inputs = keras.random.normal((batch_size, input_dim), seed=42)
        states = controller.initialize_state(batch_size)

        output, new_states = controller(inputs, state=states)

        assert ops.shape(output) == (batch_size, controller_dim)
        assert len(new_states) == 2

    def test_call_feedforward(self):
        """Test call with feedforward controller."""
        batch_size = 4
        input_dim = 32
        controller_dim = 64

        controller = NTMController(
            controller_dim=controller_dim, controller_type="feedforward"
        )
        controller.build((None, input_dim))

        inputs = keras.random.normal((batch_size, input_dim), seed=42)

        output, new_states = controller(inputs)

        assert ops.shape(output) == (batch_size, controller_dim)
        assert new_states == []

    def test_serialization(self):
        """Test get_config."""
        controller = NTMController(controller_dim=64, controller_type="lstm")
        config = controller.get_config()

        assert config["controller_dim"] == 64
        assert config["controller_type"] == "lstm"


# ---------------------------------------------------------------------
# NTMCell Tests
# ---------------------------------------------------------------------


class TestNTMCell:
    """Tests for NTMCell layer."""

    def test_init_with_config(self):
        """Test initialization with NTMConfig."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
            num_write_heads=1,
        )
        cell = NTMCell(config)

        assert cell.config.memory_size == 32
        assert cell.config.memory_dim == 16
        assert len(cell.read_heads) == 2
        assert len(cell.write_heads) == 1

    def test_init_with_dict(self):
        """Test initialization with dict config."""
        config_dict = {
            "memory_size": 32,
            "memory_dim": 16,
            "controller_dim": 64,
            "controller_type": "gru",
        }
        cell = NTMCell(config_dict)

        assert cell.config.memory_size == 32
        assert cell.config.controller_type == "gru"

    def test_state_size(self):
        """Test state_size property."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            controller_type="lstm",
            num_read_heads=2,
            num_write_heads=1,
        )
        cell = NTMCell(config)

        state_sizes = cell.state_size

        # LSTM: 2 states
        # Memory: 1 (tuple)
        # Read vectors: 2
        # Read weights: 2
        # Write weights: 1
        # Total: 2 + 1 + 2 + 2 + 1 = 8
        assert len(state_sizes) == 8

    def test_output_size(self):
        """Test output_size property."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
        )
        cell = NTMCell(config)

        # output_size = controller_dim + num_read_heads * memory_dim
        expected = 64 + 2 * 16
        assert cell.output_size == expected

    def test_get_initial_state(self):
        """Test initial state generation."""
        batch_size = 4
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            controller_type="lstm",
            num_read_heads=2,
            num_write_heads=1,
        )
        cell = NTMCell(config)

        states = cell.get_initial_state(batch_size=batch_size)

        assert len(states) == len(cell.state_size)

    def test_call(self):
        """Test single timestep call."""
        batch_size = 4
        input_dim = 32

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=1,
            num_write_heads=1,
        )
        cell = NTMCell(config)
        cell.build((None, input_dim))

        inputs = keras.random.normal((batch_size, input_dim), seed=42)
        states = cell.get_initial_state(batch_size=batch_size)

        output, new_states = cell(inputs, states)

        assert ops.shape(output) == (batch_size, cell.output_size)
        assert len(new_states) == len(states)

    def test_rnn_compatibility(self):
        """Test that cell works with keras.layers.RNN."""
        batch_size = 4
        seq_len = 10
        input_dim = 32

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        cell = NTMCell(config)

        rnn = keras.layers.RNN(cell, return_sequences=True)

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = rnn(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, cell.output_size)

    def test_serialization(self):
        """Test get_config and from_config."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
        )
        cell = NTMCell(config)

        cell_config = cell.get_config()

        assert "config" in cell_config
        assert cell_config["config"]["memory_size"] == 32
        assert cell_config["config"]["num_read_heads"] == 2

        # Reconstruct
        cell_restored = NTMCell.from_config(cell_config)

        assert cell_restored.config.memory_size == cell.config.memory_size
        assert cell_restored.config.num_read_heads == cell.config.num_read_heads

    def test_gradient_flow(self):
        """Test gradient flow through cell."""
        batch_size = 2
        input_dim = 16

        config = NTMConfig(
            memory_size=16,
            memory_dim=8,
            controller_dim=32,
        )
        cell = NTMCell(config)
        cell.build((None, input_dim))

        inputs = tf.Variable(keras.random.normal((batch_size, input_dim), seed=42))
        states = cell.get_initial_state(batch_size=batch_size)

        with tf.GradientTape() as tape:
            output, _ = cell(inputs, states)
            loss = ops.sum(output)

        grads = tape.gradient(loss, inputs)
        assert grads is not None


# ---------------------------------------------------------------------
# NeuralTuringMachine Tests
# ---------------------------------------------------------------------


class TestNeuralTuringMachine:
    """Tests for NeuralTuringMachine layer."""

    def test_init(self):
        """Test initialization."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        ntm = NeuralTuringMachine(config, output_dim=10)

        assert ntm.config.memory_size == 32
        assert ntm.output_dim == 10
        assert ntm.return_sequences is True
        assert ntm.return_state is False

    def test_init_with_dict(self):
        """Test initialization with dict config."""
        config_dict = {
            "memory_size": 32,
            "memory_dim": 16,
            "controller_dim": 64,
            "controller_type": "lstm",
            "addressing_mode": "HYBRID",
            "shift_range": 3,
            "use_memory_init": True,
            "epsilon": 1e-6,
        }
        ntm = NeuralTuringMachine(config_dict, output_dim=10)

        assert ntm.config.memory_size == 32

    def test_from_dict_drops_legacy_clip_value(self):
        """A stored config carrying the removed `clip_value` key must still load.

        `clip_value` was a declared-but-never-read field, removed by decision
        (decisions.md D-003). `NTMConfig.from_dict` does `cls(**config)`, so
        without the named-key shim an old config dict would raise TypeError.
        The key must be dropped, not absorbed as an attribute.
        """
        cfg = NTMConfig.from_dict(
            {
                "memory_size": 32,
                "memory_dim": 16,
                "controller_dim": 64,
                "addressing_mode": "HYBRID",
                "clip_value": 10.0,
                "epsilon": 1e-6,
            }
        )

        assert cfg.memory_size == 32
        assert not hasattr(cfg, "clip_value")
        assert "clip_value" not in cfg.to_dict()

    def test_from_dict_still_rejects_unknown_keys(self):
        """The shim must drop ONLY `clip_value` — a typo must stay a hard error."""
        with pytest.raises(TypeError):
            NTMConfig.from_dict({"memory_size": 32, "memmory_dim": 16})

    def test_call_return_sequences(self):
        """Test call with return_sequences=True."""
        batch_size = 4
        seq_len = 10
        input_dim = 32
        output_dim = 16

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        ntm = NeuralTuringMachine(config, output_dim=output_dim, return_sequences=True)

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, output_dim)

    def test_call_no_return_sequences(self):
        """Test call with return_sequences=False."""
        batch_size = 4
        seq_len = 10
        input_dim = 32
        output_dim = 16

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        ntm = NeuralTuringMachine(config, output_dim=output_dim, return_sequences=False)

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, output_dim)

    def test_call_return_state(self):
        """Test call with return_state=True."""
        batch_size = 4
        seq_len = 10
        input_dim = 32
        output_dim = 16

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        ntm = NeuralTuringMachine(
            config, output_dim=output_dim, return_sequences=True, return_state=True
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs, final_states = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, output_dim)
        assert isinstance(final_states, list)
        assert len(final_states) > 0

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        ntm = NeuralTuringMachine(config, output_dim=10, return_sequences=True)

        output_shape = ntm.compute_output_shape((None, 20, 32))
        assert output_shape == (None, 20, 10)

        ntm_no_seq = NeuralTuringMachine(config, output_dim=10, return_sequences=False)
        output_shape_no_seq = ntm_no_seq.compute_output_shape((None, 20, 32))
        assert output_shape_no_seq == (None, 10)

    def test_serialization(self):
        """Test get_config and from_config."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
        )
        ntm = NeuralTuringMachine(config, output_dim=10, return_sequences=False)

        ntm_config = ntm.get_config()

        assert ntm_config["output_dim"] == 10
        assert ntm_config["return_sequences"] is False
        assert "config" in ntm_config

        # Reconstruct
        ntm_restored = NeuralTuringMachine.from_config(ntm_config)

        assert ntm_restored.output_dim == ntm.output_dim
        assert ntm_restored.return_sequences == ntm.return_sequences

    def test_save_and_load(self):
        """Test model saving and loading."""
        import tempfile
        import os

        config = NTMConfig(
            memory_size=16,
            memory_dim=8,
            controller_dim=32,
        )
        ntm = NeuralTuringMachine(config, output_dim=5)

        # Build the model
        inputs = keras.Input(shape=(5, 16))
        outputs = ntm(inputs)
        model = keras.Model(inputs, outputs)

        # Generate inputs
        input_data = keras.random.normal((2, 5, 16), seed=42)
        output_before = model(input_data)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ntm_model.keras")
            model.save(model_path)

            model_loaded = keras.models.load_model(model_path)

        output_after = model_loaded(input_data)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_before),
            keras.ops.convert_to_numpy(output_after),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Loaded model should produce same output",
        )

    def test_gradient_flow(self):
        """Test gradient flow through NTM."""
        batch_size = 2
        seq_len = 5
        input_dim = 16
        output_dim = 8

        config = NTMConfig(
            memory_size=16,
            memory_dim=8,
            controller_dim=32,
        )
        ntm = NeuralTuringMachine(config, output_dim=output_dim)

        inputs = tf.Variable(
            keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        )

        with tf.GradientTape() as tape:
            outputs = ntm(inputs, training=True)
            loss = ops.sum(outputs)

        grads = tape.gradient(loss, inputs)
        assert grads is not None

    def test_training_step(self):
        """Test a simple training step."""
        batch_size = 2
        seq_len = 5
        input_dim = 16
        output_dim = 8

        config = NTMConfig(
            memory_size=16,
            memory_dim=8,
            controller_dim=32,
        )
        ntm = NeuralTuringMachine(config, output_dim=output_dim)

        # Create a simple model
        inputs = keras.Input(shape=(seq_len, input_dim))
        outputs = ntm(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        y = np.random.randn(batch_size, seq_len, output_dim).astype(np.float32)

        # Train for one step
        history = model.fit(x, y, epochs=1, verbose=0)

        assert "loss" in history.history
        assert len(history.history["loss"]) == 1


# ---------------------------------------------------------------------
# create_ntm Factory Tests
# ---------------------------------------------------------------------


class TestCreateNTM:
    """Tests for create_ntm factory function."""

    def test_default_params(self):
        """Test factory with default parameters."""
        ntm = create_ntm(output_dim=10)

        assert isinstance(ntm, NeuralTuringMachine)
        assert ntm.output_dim == 10
        assert ntm.config.memory_size == 128
        assert ntm.config.memory_dim == 64

    def test_custom_params(self):
        """Test factory with custom parameters."""
        ntm = create_ntm(
            memory_size=64,
            memory_dim=32,
            output_dim=20,
            controller_dim=128,
            controller_type="gru",
            num_read_heads=2,
            num_write_heads=2,
            shift_range=5,
            return_sequences=False,
        )

        assert ntm.config.memory_size == 64
        assert ntm.config.memory_dim == 32
        assert ntm.output_dim == 20
        assert ntm.config.controller_type == "gru"
        assert ntm.config.num_read_heads == 2
        assert ntm.return_sequences is False

    def test_factory_produces_working_model(self):
        """Test that factory produces a working model."""
        batch_size = 2
        seq_len = 5
        input_dim = 16

        ntm = create_ntm(
            memory_size=16,
            memory_dim=8,
            output_dim=4,
            controller_dim=32,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, 4)


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the complete NTM system."""

    def test_copy_task_structure(self):
        """Test NTM structure for copy task."""
        batch_size = 4
        seq_len = 10
        input_dim = 8
        output_dim = 8

        ntm = create_ntm(
            memory_size=32,
            memory_dim=16,
            output_dim=output_dim,
            controller_dim=64,
            num_read_heads=1,
            num_write_heads=1,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, output_dim)

    def test_multiple_heads(self):
        """Test NTM with multiple read/write heads."""
        batch_size = 2
        seq_len = 5
        input_dim = 16

        ntm = create_ntm(
            memory_size=32,
            memory_dim=16,
            output_dim=8,
            controller_dim=64,
            num_read_heads=3,
            num_write_heads=2,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = ntm(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, 8)

    def test_different_controller_types(self):
        """Test NTM with different controller types."""
        batch_size = 2
        seq_len = 5
        input_dim = 16

        for controller_type in ["lstm", "gru", "feedforward"]:
            ntm = create_ntm(
                memory_size=16,
                memory_dim=8,
                output_dim=4,
                controller_dim=32,
                controller_type=controller_type,
            )

            inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
            outputs = ntm(inputs)

            assert ops.shape(outputs) == (batch_size, seq_len, 4), (
                f"Failed for controller_type={controller_type}"
            )

    def test_sequential_processing(self):
        """Test that NTM processes sequences correctly."""
        batch_size = 2
        input_dim = 8
        output_dim = 8

        ntm = create_ntm(
            memory_size=16,
            memory_dim=8,
            output_dim=output_dim,
            controller_dim=32,
            return_state=True,
        )

        # Process sequence of length 5
        inputs_5 = keras.random.normal((batch_size, 5, input_dim), seed=42)
        outputs_5, states_5 = ntm(inputs_5)

        # Process sequence of length 10
        inputs_10 = keras.random.normal((batch_size, 10, input_dim), seed=43)
        outputs_10, states_10 = ntm(inputs_10)

        assert ops.shape(outputs_5) == (batch_size, 5, output_dim)
        assert ops.shape(outputs_10) == (batch_size, 10, output_dim)


# ---------------------------------------------------------------------
# Addressing Health Tests
# ---------------------------------------------------------------------


class TestAddressingHealth:
    """Tests verifying the NTM addressing mechanism is functional."""

    def test_memory_init_breaks_symmetry(self):
        """Memory initialization must produce distinct rows."""
        memory = NTMMemory(memory_size=32, memory_dim=16)
        state = memory.initialize_state(batch_size=2)
        mem_np = ops.convert_to_numpy(state.memory)

        # Rows should NOT all be identical
        row_std = np.std(mem_np[0], axis=0)  # std across 32 rows per dim
        assert np.any(row_std > 1e-6), (
            "Memory rows are identical — addressing gradients will be zero"
        )

    def test_ntm_cell_initial_memory_is_learnable(self):
        """NTMCell should have a trainable initial memory weight."""
        config = NTMConfig(
            memory_size=32, memory_dim=16, controller_dim=64,
            num_read_heads=1, num_write_heads=1, shift_range=3,
            use_memory_init=True,
        )
        cell = NTMCell(config)
        # Build the cell
        cell.build((None, 8))

        assert cell._initial_memory is not None, (
            "use_memory_init=True but no learnable initial memory created"
        )
        assert cell._initial_memory.trainable

    def test_addressing_gradients_nonzero(self):
        """All addressing parameters must receive non-zero gradients."""
        config = NTMConfig(
            memory_size=32, memory_dim=16, controller_dim=64,
            num_read_heads=1, num_write_heads=1, shift_range=3,
        )
        cell = NTMCell(config)
        rnn = keras.layers.RNN(cell, return_sequences=True)

        x = np.random.randn(2, 5, 8).astype("float32")
        x_t = ops.convert_to_tensor(x)

        with tf.GradientTape() as tape:
            y = rnn(x_t, training=True)
            loss = ops.mean(y)
        grads = tape.gradient(loss, rnn.trainable_weights)

        for w, g in zip(rnn.trainable_weights, grads):
            assert g is not None, f"Gradient is None for {w.path}"
            gnorm = float(ops.sqrt(ops.sum(ops.square(g))))
            assert gnorm > 1e-10, (
                f"Gradient is effectively zero for {w.path} (norm={gnorm:.2e})"
            )

    def test_initial_memory_gradient_flows(self):
        """The learnable initial memory weight must receive gradients."""
        config = NTMConfig(
            memory_size=32, memory_dim=16, controller_dim=64,
            num_read_heads=1, num_write_heads=1, shift_range=3,
            use_memory_init=True,
        )
        cell = NTMCell(config)
        rnn = keras.layers.RNN(cell, return_sequences=True)

        x = np.random.randn(2, 5, 8).astype("float32")
        x_t = ops.convert_to_tensor(x)

        with tf.GradientTape() as tape:
            y = rnn(x_t, training=True)
            loss = ops.mean(y)
        grads = tape.gradient(loss, rnn.trainable_weights)

        # Find the initial_memory weight
        mem_grad = None
        for w, g in zip(rnn.trainable_weights, grads):
            if "initial_memory" in w.path:
                mem_grad = g
                break

        assert mem_grad is not None, "initial_memory weight not found"
        gnorm = float(ops.sqrt(ops.sum(ops.square(mem_grad))))
        assert gnorm > 1e-8, (
            f"initial_memory gradient too small: {gnorm:.2e}"
        )


# ---------------------------------------------------------------------
# Circular Convolution Shift Direction Tests
# ---------------------------------------------------------------------


class TestCircularConvolutionDirection:
    """Delta-impulse guards pinning the DIRECTION of location-based addressing.

    Graves et al. 2014 eq. 8 defines the shift as
    ``w_tilde(i) = sum_j w(j) * s(i - j mod N)``. With ``w`` a delta impulse at
    slot 0, this reduces to ``w_tilde(i) = s(i)``: all shift mass placed on
    offset ``+1`` must land at slot ``1``.

    Shift-vector index convention (derived by reading
    ``ntm_interface.circular_convolution``: ``half_shift = shift_range // 2``
    and, for shift-vector index ``i``, ``shift_offset = i - half_shift``).
    For ``shift_range = 3``, ``half_shift = 1``, so:

    ==========  ============
    index       shift_offset
    ==========  ============
    0           -1
    1            0
    2           +1
    ==========  ============

    ``shift_range`` is deliberately 3 (never 1, where the shift is the identity
    and the guard would be vacuous) and ``memory_size`` is 8 (not a degenerate
    2- or 3-slot memory, where ``+1`` and ``-1`` alias).
    """

    MEMORY_SIZE = 8
    SHIFT_RANGE = 3

    def _delta_inputs(self, shift_index: int):
        """Build ``w = e_0`` and a one-hot shift vector at ``shift_index``."""
        weights = np.zeros((1, self.MEMORY_SIZE), dtype="float32")
        weights[0, 0] = 1.0

        shift = np.zeros((1, self.SHIFT_RANGE), dtype="float32")
        shift[0, shift_index] = 1.0

        return ops.convert_to_tensor(weights), ops.convert_to_tensor(shift)

    def test_shift_direction_positive_offset_moves_forward(self):
        """Offset +1 (index 2) must move a delta impulse from slot 0 to slot 1."""
        weights, shift = self._delta_inputs(shift_index=2)

        out = ops.convert_to_numpy(circular_convolution(weights, shift))

        assert int(np.argmax(out[0])) == 1, (
            "Shift +1 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot 1 "
            f"(Graves eq. 8); full output: {out[0]}"
        )
        assert out[0, 1] >= 0.99, (
            f"Shift +1 delivered only {out[0, 1]:.4f} mass to slot 1, "
            f"expected ~1.0; full output: {out[0]}"
        )

    def test_shift_direction_negative_offset_moves_backward(self):
        """Offset -1 (index 0) must move a delta impulse from slot 0 to slot N-1."""
        weights, shift = self._delta_inputs(shift_index=0)

        out = ops.convert_to_numpy(circular_convolution(weights, shift))
        last = self.MEMORY_SIZE - 1

        assert int(np.argmax(out[0])) == last, (
            "Shift -1 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot {last} "
            f"(Graves eq. 8); full output: {out[0]}"
        )
        assert out[0, last] >= 0.99, (
            f"Shift -1 delivered only {out[0, last]:.4f} mass to slot {last}, "
            f"expected ~1.0; full output: {out[0]}"
        )

    def test_shift_direction_zero_offset_is_identity(self):
        """Offset 0 (index 1) must leave the delta impulse at slot 0.

        This is the CONTROL: it holds under both the mirrored and the correct
        shift convention, so it proves the probe is not simply always-red.
        """
        weights, shift = self._delta_inputs(shift_index=1)

        out = ops.convert_to_numpy(circular_convolution(weights, shift))

        assert int(np.argmax(out[0])) == 0, (
            "Shift 0 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot 0; full output: {out[0]}"
        )
        assert out[0, 0] >= 0.99, (
            f"Shift 0 delivered only {out[0, 0]:.4f} mass to slot 0, "
            f"expected ~1.0; full output: {out[0]}"
        )


class TestMemoryInitSeedPaths:
    """The `memory_init_seed` must survive serialization and reach the cell's memory.

    Two defects, one concept. `NTMMemory.get_config` dropped the seed, so a
    `from_config` round-trip silently reset 7 to the default 42; and `NTMCell`
    built its `NTMMemory` without a seed, so the memory module kept 42 whatever
    its `NTMConfig` declared.

    Both guards run on the `NTMMemory` object directly, so neither depends on
    `NTMConfig.use_memory_init` -- that flag selects between the cell's learned
    `_initial_memory` weight and the cell's OWN seeded draw in
    `get_initial_state`, and `NTMMemory.initialize_state` is on neither of
    those branches. What is under test here is the seed the memory module
    carries, which is what a caller driving `BaseMemory` directly observes.
    """

    def test_get_config_round_trip_preserves_memory_init_seed(self):
        """A non-default seed must survive `get_config` -> `from_config`.

        Asserted on the RECONSTRUCTED object's attribute, not merely on the
        presence of the key, so a key emitted under the wrong name would still
        fail.
        """
        memory = NTMMemory(memory_size=8, memory_dim=4, memory_init_seed=7)
        assert memory.memory_init_seed == 7

        config = memory.get_config()
        restored = NTMMemory.from_config(config)

        assert restored.memory_init_seed == 7, (
            "get_config/from_config round-trip reset memory_init_seed to "
            f"{restored.memory_init_seed}, expected 7; emitted keys were "
            f"{sorted(config.keys())}"
        )

    def test_cell_memory_carries_the_configured_seed(self):
        """The seed an `NTMConfig` declares must reach the `NTMMemory` the cell builds."""
        config = NTMConfig(
            memory_size=8,
            memory_dim=4,
            controller_dim=6,
            memory_init_seed=7,
        )
        cell = NTMCell(config)

        assert cell.memory.memory_init_seed == 7, (
            "NTMCell built its NTMMemory with seed "
            f"{cell.memory.memory_init_seed}, but its NTMConfig declares 7"
        )

    def test_cell_memory_draw_is_seed_dependent(self):
        """The behavioural half: different configured seeds must draw different memory.

        `initialize_state` is the only place the memory module's seed reaches a
        number, so this is what "the seed reached the memory" MEANS. The
        same-seed pair is the control: it proves the difference below comes
        from the seed and not from an unseeded stateful draw.
        """
        def _memory_of(seed):
            config = NTMConfig(
                memory_size=8,
                memory_dim=4,
                controller_dim=6,
                memory_init_seed=seed,
            )
            cell = NTMCell(config)
            state = cell.memory.initialize_state(2)
            return keras.ops.convert_to_numpy(state.memory)

        first_seven = _memory_of(7)
        second_seven = _memory_of(7)
        nine_ninety_nine = _memory_of(999)

        # Control: the same seed must repeat exactly.
        np.testing.assert_allclose(first_seven, second_seven, rtol=0, atol=0)

        diff = float(np.max(np.abs(first_seven - nine_ninety_nine)))
        assert diff > 0.0, (
            "Two NTMCells configured with memory_init_seed 7 and 999 drew "
            f"IDENTICAL initial memory (max abs diff {diff}); the configured "
            "seed never reached NTMMemory"
        )


class TestCreateNTMReachesEveryConfigField:
    """`create_ntm` must be able to reach every `NTMConfig` field.

    The factory exposed 7 of `NTMConfig`'s 11 fields. `addressing_mode`,
    `use_memory_init`, `memory_init_seed` and `epsilon` were unreachable and
    silently kept their defaults, so a CONTENT-addressing NTM could not be
    built through the documented construction path at all.

    The `memory_init_seed` half of this class is ONLY checkable because the
    preceding step landed: before it, `NTMCell` built its `NTMMemory` without
    passing the configured seed down, so even a correctly widened factory
    signature would still have shown 42 on the built memory. See
    `TestMemoryInitSeedPaths`.
    """

    @staticmethod
    def _build(**kwargs):
        """Build a small NTM through the factory and run one forward pass."""
        ntm = create_ntm(
            memory_size=8,
            memory_dim=4,
            output_dim=3,
            controller_dim=6,
            num_read_heads=1,
            num_write_heads=1,
            shift_range=3,
            **kwargs,
        )
        inputs = keras.random.normal((2, 5, 4))
        ntm(inputs)
        return ntm

    def test_content_addressing_head_omits_the_location_projections(self):
        """CONTENT must change what the heads ARE, not just what they store.

        Under CONTENT the three location-addressing projections
        (`gate_dense` / `shift_dense` / `gamma_dense`) are never created, so a
        CONTENT head owns strictly fewer weights than the HYBRID head the same
        factory call builds by default. A merely-stored `addressing_mode`
        attribute cannot satisfy this: the assertion is on the built weight
        set and the parameter count, both of which a stored enum leaves
        untouched. The HYBRID build is the paired control -- without it the
        count assertion could pass against an NTM of any shape.
        """
        content = self._build(addressing_mode=AddressingMode.CONTENT)
        hybrid = self._build(addressing_mode=AddressingMode.HYBRID)

        content_head = content.ntm_cell.read_heads[0]
        hybrid_head = hybrid.ntm_cell.read_heads[0]

        assert content_head.gate_dense is None
        assert content_head.shift_dense is None
        assert content_head.gamma_dense is None
        assert hybrid_head.gate_dense is not None

        content_names = [w.path for w in content_head.weights]
        assert not any(
            token in name
            for name in content_names
            for token in ("gate", "shift", "gamma")
        ), f"a CONTENT read head owns location-addressing weights: {content_names}"

        content_params = int(sum(np.prod(w.shape) for w in content_head.weights))
        hybrid_params = int(sum(np.prod(w.shape) for w in hybrid_head.weights))
        assert content_params < hybrid_params, (
            f"CONTENT read head has {content_params} parameters and the HYBRID "
            f"control has {hybrid_params}; CONTENT did not reach the head"
        )

        # The same must hold for the write head, which builds its own projections.
        assert content.ntm_cell.write_heads[0].shift_dense is None

    def test_memory_init_seed_reaches_the_built_memory(self):
        """A non-default seed passed to the factory must reach the memory module.

        Checkable only because the preceding step landed: the cell now forwards
        its config's seed to the `NTMMemory` it builds. The behavioural half
        below is what "reached" means, and the repeat draw is its control.
        """
        seven = self._build(memory_init_seed=7)
        assert seven.ntm_cell.memory.memory_init_seed == 7, (
            "create_ntm(memory_init_seed=7) built a memory carrying "
            f"{seven.ntm_cell.memory.memory_init_seed}"
        )

        default = self._build()
        assert default.ntm_cell.memory.memory_init_seed == 42

        drawn = keras.ops.convert_to_numpy(
            seven.ntm_cell.memory.initialize_state(2).memory
        )
        again = keras.ops.convert_to_numpy(
            self._build(memory_init_seed=7).ntm_cell.memory.initialize_state(2).memory
        )
        other = keras.ops.convert_to_numpy(
            default.ntm_cell.memory.initialize_state(2).memory
        )

        # Control: the same seed repeats exactly.
        np.testing.assert_allclose(drawn, again, rtol=0, atol=0)
        assert float(np.max(np.abs(drawn - other))) > 0.0

    def test_use_memory_init_and_epsilon_reach_the_config(self):
        """The remaining two withheld fields must also arrive.

        `epsilon` is asserted on a head, which is where it is actually
        consumed (`cosine_similarity` / `sharpen_weights`), not only on the
        config dataclass.
        """
        ntm = self._build(use_memory_init=False, epsilon=1e-3)

        assert ntm.ntm_cell.config.use_memory_init is False
        assert ntm.ntm_cell.config.epsilon == 1e-3
        assert ntm.ntm_cell.read_heads[0].epsilon == 1e-3
        assert ntm.ntm_cell.write_heads[0].epsilon == 1e-3
        assert ntm.ntm_cell.memory.epsilon == 1e-3


class TestBiasInitializerReachesEveryProjection:
    """A caller's `bias_initializer` must reach EVERY Dense in a head.

    Both heads documented `bias_initializer` as "Initializer for the Dense
    biases", but each built its `key_dense` without passing it, so a
    non-default value silently missed exactly one projection per head while
    reaching every other Dense. The default is `'zeros'` and Dense's own
    default is also `'zeros'`, which is why the whole suite stayed green over
    it.

    The biases are enumerated from the BUILT weight set rather than named by
    hand, so a projection added later is covered without editing this class.
    Both addressing modes are asserted because the projection set differs by
    mode: under CONTENT `gate` / `shift` / `gamma` are never created, so a
    HYBRID-only guard could not see a CONTENT-only regression.
    """

    CONSTANT = 0.7

    @staticmethod
    def _built_head(head_cls, addressing_mode):
        """Build one head with a recognisable non-default bias initializer."""
        head = head_cls(
            memory_size=8,
            memory_dim=4,
            addressing_mode=addressing_mode,
            shift_range=3,
            bias_initializer=keras.initializers.Constant(
                TestBiasInitializerReachesEveryProjection.CONSTANT
            ),
        )
        head.build((2, 6))
        return head

    @staticmethod
    def _biases(head):
        """Map projection name -> bias values, read off the built weights."""
        return {
            w.path.split("/")[-2]: keras.ops.convert_to_numpy(w)
            for w in head.weights
            if w.path.endswith("/bias")
        }

    @pytest.mark.parametrize(
        "head_cls,addressing_mode,expected",
        [
            (
                NTMReadHead,
                AddressingMode.HYBRID,
                {"key", "beta", "gate", "shift", "gamma"},
            ),
            (NTMReadHead, AddressingMode.CONTENT, {"key", "beta"}),
            (
                NTMWriteHead,
                AddressingMode.HYBRID,
                {"key", "beta", "gate", "shift", "gamma", "erase", "add"},
            ),
            (NTMWriteHead, AddressingMode.CONTENT, {"key", "beta", "erase", "add"}),
        ],
    )
    def test_every_dense_bias_carries_the_requested_constant(
        self, head_cls, addressing_mode, expected
    ):
        """No projection may fall back to Dense's own `'zeros'` default.

        The `expected` set is asserted first so the per-bias loop can never
        pass vacuously against a head that built fewer projections than it
        should have.
        """
        head = self._built_head(head_cls, addressing_mode)
        biases = self._biases(head)

        assert set(biases) == expected, (
            f"{head_cls.__name__} under {addressing_mode} built biases "
            f"{sorted(biases)}, expected {sorted(expected)}"
        )

        missed = {
            name: float(np.max(np.abs(values - self.CONSTANT)))
            for name, values in biases.items()
            if not np.allclose(values, self.CONSTANT, rtol=0, atol=0)
        }
        assert not missed, (
            f"{head_cls.__name__} under {addressing_mode}: the caller's "
            f"bias_initializer did not reach {sorted(missed)}; "
            + ", ".join(
                f"{name} reads {biases[name].reshape(-1)[0]!r}" for name in sorted(missed)
            )
        )


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])