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

import numpy as np
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.ntm.ntm_interface import (
    AddressingMode,
    MemoryState,
    HeadState,
    NTMConfig,
)
from dl_techniques.layers.ntm.baseline_ntm import (
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
            "clip_value": 10.0,
            "epsilon": 1e-6,
        }
        ntm = NeuralTuringMachine(config_dict, output_dim=10)

        assert ntm.config.memory_size == 32

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
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])