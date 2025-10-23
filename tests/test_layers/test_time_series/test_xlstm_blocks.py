"""
Comprehensive tests for xLSTM layers.

This module provides thorough testing for all xLSTM components including:
- sLSTMCell: Scalar LSTM cell with exponential gating
- mLSTMCell: Matrix LSTM cell with matrix memory
- sLSTMLayer: RNN layer wrapping sLSTMCell
- mLSTMLayer: RNN layer wrapping mLSTMCell
- sLSTMBlock: Composite block with sLSTM
- mLSTMBlock: Composite block with mLSTM

Tests cover:
- Initialization and configuration
- Forward pass and shape validation
- Serialization and deserialization
- Gradient flow
- Training modes
- Edge cases and error handling
- State management for recurrent cells
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Any


from dl_techniques.layers.time_series.xlstm_blocks import (
    sLSTMBlock, sLSTMLayer, mLSTMLayer, mLSTMBlock, sLSTMCell, mLSTMCell)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_batch_size() -> int:
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def sample_sequence_length() -> int:
    """Standard sequence length for tests."""
    return 10


@pytest.fixture
def sample_input_dim() -> int:
    """Standard input dimension for tests."""
    return 16


@pytest.fixture
def sample_units() -> int:
    """Standard number of units for tests."""
    return 32


# =============================================================================
# sLSTMCell Tests
# =============================================================================

class TestSLSTMCell:
    """Comprehensive test suite for sLSTMCell."""

    @pytest.fixture
    def cell_config(self) -> Dict[str, Any]:
        """Standard configuration for sLSTMCell."""
        return {
            'units': 32,
            'forget_gate_activation': 'sigmoid',
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'bias_initializer': 'zeros',
        }

    @pytest.fixture
    def cell_config_exp(self) -> Dict[str, Any]:
        """Configuration with exponential forget gate."""
        return {
            'units': 32,
            'forget_gate_activation': 'exp',
        }

    def test_initialization(self, cell_config):
        """Test cell initialization."""

        cell = sLSTMCell(**cell_config)

        assert cell.units == 32
        assert cell.forget_gate_activation == 'sigmoid'
        assert not cell.built
        assert cell.state_size == [32, 32, 32, 32]  # [h, c, n, m]
        assert cell.output_size == 32

    def test_initialization_with_exp_gate(self, cell_config_exp):
        """Test initialization with exponential forget gate."""

        cell = sLSTMCell(**cell_config_exp)

        assert cell.forget_gate_activation == 'exp'

    def test_invalid_units(self):
        """Test error handling for invalid units."""
        with pytest.raises(ValueError, match="units.*must be positive"):
            sLSTMCell(units=0)

        with pytest.raises(ValueError, match="units.*must be positive"):
            sLSTMCell(units=-5)

    def test_invalid_forget_gate_activation(self):
        """Test error handling for invalid forget gate activation."""
        with pytest.raises(ValueError, match="forget_gate_activation.*must be"):
            sLSTMCell(units=32, forget_gate_activation='invalid')

    def test_single_timestep_forward_pass(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim
    ):
        """Test forward pass for a single timestep."""
        cell = sLSTMCell(**cell_config)

        # Create input and initial state
        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        # Forward pass
        output, new_state = cell(x_t, state)

        # Verify shapes
        assert output.shape == (sample_batch_size, cell.units)
        assert len(new_state) == 4  # [h, c, n, m]
        for state_tensor in new_state:
            assert state_tensor.shape == (sample_batch_size, cell.units)

        # Verify cell is built
        assert cell.built

    def test_get_initial_state(self, cell_config, sample_batch_size):
        """Test initial state generation."""
        cell = sLSTMCell(**cell_config)

        # Get initial state
        state = cell.get_initial_state(batch_size=sample_batch_size)

        # Verify state structure
        assert len(state) == 4  # [h, c, n, m]
        for state_tensor in state:
            assert state_tensor.shape == (sample_batch_size, cell.units)
            # Initial states should be zeros
            assert ops.sum(ops.abs(state_tensor)) == 0.0

    def test_state_updates(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim
    ):
        """Test that states are properly updated across timesteps."""


        cell = sLSTMCell(**cell_config)

        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        # First timestep
        output1, state1 = cell(x_t, state)

        # Second timestep with updated state
        output2, state2 = cell(x_t, state1)

        # States should be different (not equal to initial state)
        for s0, s2 in zip(state, state2):
            assert not ops.all(s0 == s2)

    def test_config_completeness(self, cell_config):
        """Test that get_config contains all __init__ parameters."""


        cell = sLSTMCell(**cell_config)
        config = cell.get_config()

        # Check all config parameters are present
        assert 'units' in config
        assert 'forget_gate_activation' in config
        assert 'kernel_initializer' in config
        assert 'recurrent_initializer' in config
        assert 'bias_initializer' in config

    def test_serialization_in_rnn(
            self,
            cell_config,
            sample_batch_size,
            sample_sequence_length,
            sample_input_dim
    ):
        """Test serialization when cell is used in RNN layer."""


        # Create RNN layer with cell
        cell = sLSTMCell(**cell_config)
        rnn_layer = keras.layers.RNN(cell, return_sequences=True)

        # Build model
        inputs = keras.Input(shape=(sample_sequence_length, sample_input_dim))
        outputs = rnn_layer(inputs)
        model = keras.Model(inputs, outputs)

        # Sample input
        sample_input = keras.random.normal(
            (sample_batch_size, sample_sequence_length, sample_input_dim)
        )

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_slstm_cell.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_gradients_flow(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim
    ):
        """Test that gradients flow through the cell."""


        cell = sLSTMCell(**cell_config)

        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        with tf.GradientTape() as tape:
            tape.watch(x_t)
            output, new_state = cell(x_t, state)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, cell.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim,
            training
    ):
        """Test behavior in different training modes."""


        cell = sLSTMCell(**cell_config)

        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        output, new_state = cell(x_t, state, training=training)

        assert output.shape == (sample_batch_size, cell.units)


# =============================================================================
# mLSTMCell Tests
# =============================================================================

class TestMLSTMCell:
    """Comprehensive test suite for mLSTMCell."""

    @pytest.fixture
    def cell_config(self) -> Dict[str, Any]:
        """Standard configuration for mLSTMCell."""
        return {
            'units': 32,
            'num_heads': 4,
            'kernel_initializer': 'glorot_uniform',
            'recurrent_initializer': 'orthogonal',
            'bias_initializer': 'zeros',
        }

    def test_initialization(self, cell_config):
        """Test cell initialization."""


        cell = mLSTMCell(**cell_config)

        assert cell.units == 32
        assert cell.num_heads == 4
        assert cell.head_dim == 8  # units / num_heads
        assert not cell.built

    def test_invalid_units_heads_ratio(self):
        """Test error when units not divisible by num_heads."""


        with pytest.raises(ValueError, match="units.*must be divisible by.*num_heads"):
            mLSTMCell(units=32, num_heads=5)

    def test_single_timestep_forward_pass(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim
    ):
        """Test forward pass for a single timestep."""


        cell = mLSTMCell(**cell_config)

        # Create input and initial state
        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        # Forward pass
        output, new_state = cell(x_t, state)

        # Verify shapes
        assert output.shape == (sample_batch_size, cell.units)
        assert cell.built

    def test_config_completeness(self, cell_config):
        """Test that get_config contains all __init__ parameters."""


        cell = mLSTMCell(**cell_config)
        config = cell.get_config()

        assert 'units' in config
        assert 'num_heads' in config
        assert 'kernel_initializer' in config
        assert 'recurrent_initializer' in config
        assert 'bias_initializer' in config

    def test_serialization_in_rnn(
            self,
            cell_config,
            sample_batch_size,
            sample_sequence_length,
            sample_input_dim
    ):
        """Test serialization when cell is used in RNN layer."""


        # Create RNN layer with cell
        cell = mLSTMCell(**cell_config)
        rnn_layer = keras.layers.RNN(cell, return_sequences=True)

        # Build model
        inputs = keras.Input(shape=(sample_sequence_length, sample_input_dim))
        outputs = rnn_layer(inputs)
        model = keras.Model(inputs, outputs)

        # Sample input
        sample_input = keras.random.normal(
            (sample_batch_size, sample_sequence_length, sample_input_dim)
        )

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_mlstm_cell.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_gradients_flow(
            self,
            cell_config,
            sample_batch_size,
            sample_input_dim
    ):
        """Test that gradients flow through the cell."""


        cell = mLSTMCell(**cell_config)

        x_t = keras.random.normal((sample_batch_size, sample_input_dim))
        state = cell.get_initial_state(batch_size=sample_batch_size)

        with tf.GradientTape() as tape:
            tape.watch(x_t)
            output, new_state = cell(x_t, state)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, cell.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0


# =============================================================================
# sLSTMLayer Tests
# =============================================================================

class TestSLSTMLayer:
    """Comprehensive test suite for sLSTMLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for sLSTMLayer."""
        return {
            'units': 64,
            'forget_gate_activation': 'sigmoid',
            'return_sequences': True,
        }

    @pytest.fixture
    def sample_input_3d(
            self,
            sample_batch_size,
            sample_sequence_length,
            sample_input_dim
    ) -> keras.KerasTensor:
        """Sample 3D input for sequence processing."""
        return keras.random.normal(
            (sample_batch_size, sample_sequence_length, sample_input_dim)
        )

    def test_initialization(self, layer_config):
        """Test layer initialization."""


        layer = sLSTMLayer(**layer_config)

        assert layer.units == 64
        assert layer.return_sequences is True
        assert not layer.built

    def test_forward_pass_return_sequences(self, layer_config, sample_input_3d):
        """Test forward pass with return_sequences=True."""


        layer = sLSTMLayer(**layer_config)

        output = layer(sample_input_3d)

        assert layer.built
        # Should return full sequence
        assert output.shape == (
            sample_input_3d.shape[0],  # batch_size
            sample_input_3d.shape[1],  # sequence_length
            layer.units
        )

    def test_forward_pass_return_last(self, sample_input_3d):
        """Test forward pass with return_sequences=False."""


        layer = sLSTMLayer(units=64, return_sequences=False)

        output = layer(sample_input_3d)

        # Should return only last timestep
        assert output.shape == (sample_input_3d.shape[0], layer.units)

    def test_serialization_cycle(self, layer_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle."""


        # Create model with layer
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = sLSTMLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_3d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_slstm_layer.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""


        layer = sLSTMLayer(**layer_config)
        config = layer.get_config()

        assert 'units' in config
        assert 'forget_gate_activation' in config
        assert 'return_sequences' in config

    def test_gradients_flow(self, layer_config, sample_input_3d):
        """Test gradient computation."""


        layer = sLSTMLayer(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = layer(sample_input_3d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input_3d, training):
        """Test behavior in different training modes."""


        layer = sLSTMLayer(**layer_config)

        output = layer(sample_input_3d, training=training)
        assert output.shape[0] == sample_input_3d.shape[0]

    def test_masking_support(self, layer_config):
        """Test that layer properly handles masking."""


        layer = sLSTMLayer(**layer_config)

        # Create input with variable length sequences
        batch_size = 4
        max_length = 10
        input_dim = 16

        # Create padded sequences
        sequences = keras.random.normal((batch_size, max_length, input_dim))

        # Create mask (first sequence has length 5, second has 7, etc.)
        mask = ops.concatenate([
            ops.ones((batch_size, 5)),
            ops.zeros((batch_size, max_length - 5))
        ], axis=1)

        output = layer(sequences, mask=mask)

        # Output should have correct shape
        if layer.return_sequences:
            assert output.shape == (batch_size, max_length, layer.units)
        else:
            assert output.shape == (batch_size, layer.units)


# =============================================================================
# mLSTMLayer Tests
# =============================================================================

class TestMLSTMLayer:
    """Comprehensive test suite for mLSTMLayer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for mLSTMLayer."""
        return {
            'units': 64,
            'num_heads': 4,
            'return_sequences': True,
        }

    @pytest.fixture
    def sample_input_3d(
            self,
            sample_batch_size,
            sample_sequence_length,
            sample_input_dim
    ) -> keras.KerasTensor:
        """Sample 3D input for sequence processing."""
        return keras.random.normal(
            (sample_batch_size, sample_sequence_length, sample_input_dim)
        )

    def test_initialization(self, layer_config):
        """Test layer initialization."""


        layer = mLSTMLayer(**layer_config)

        assert layer.units == 64
        assert layer.num_heads == 4
        assert layer.return_sequences is True
        assert not layer.built

    def test_forward_pass_return_sequences(self, layer_config, sample_input_3d):
        """Test forward pass with return_sequences=True."""


        layer = mLSTMLayer(**layer_config)

        output = layer(sample_input_3d)

        assert layer.built
        # Should return full sequence
        assert output.shape == (
            sample_input_3d.shape[0],  # batch_size
            sample_input_3d.shape[1],  # sequence_length
            layer.units
        )

    def test_serialization_cycle(self, layer_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle."""


        # Create model with layer
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = mLSTMLayer(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_3d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_mlstm_layer.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_gradients_flow(self, layer_config, sample_input_3d):
        """Test gradient computation."""


        layer = mLSTMLayer(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = layer(sample_input_3d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0


# =============================================================================
# sLSTMBlock Tests
# =============================================================================

class TestSLSTMBlock:
    """Comprehensive test suite for sLSTMBlock."""

    @pytest.fixture
    def block_config(self) -> Dict[str, Any]:
        """Standard configuration for sLSTMBlock."""
        return {
            'units': 64,
            'ffn_expansion_factor': 2,
            'ffn_type': 'swiglu',
            'normalization_type': 'layer_norm',
        }

    @pytest.fixture
    def sample_input_3d(
            self,
            sample_batch_size,
            sample_sequence_length
    ) -> keras.KerasTensor:
        """Sample 3D input matching units dimension."""
        return keras.random.normal((sample_batch_size, sample_sequence_length, 64))

    def test_initialization(self, block_config):
        """Test block initialization."""


        block = sLSTMBlock(**block_config)

        assert block.units == 64
        assert block.ffn_expansion_factor == 2
        assert not block.built

        # Check sub-layers were created
        assert hasattr(block, "slstm")
        assert hasattr(block, "norm")
        assert hasattr(block, "ffn")

    def test_forward_pass(self, block_config, sample_input_3d):
        """Test forward pass through block."""


        block = sLSTMBlock(**block_config)

        output = block(sample_input_3d)

        assert block.built
        # Output should have same shape as input (residual connection)
        assert output.shape == sample_input_3d.shape

    def test_residual_connection(self, block_config, sample_input_3d):
        """Test that residual connection is properly applied."""


        block = sLSTMBlock(**block_config)

        # Zero initialization should give near-identity mapping
        output = block(sample_input_3d)

        # The output should not be identical to input (due to transformations)
        # but should be influenced by the residual
        assert not ops.all(output == sample_input_3d)

    def test_serialization_cycle(self, block_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle."""


        # Create model with block
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = sLSTMBlock(**block_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_3d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_slstm_block.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, block_config):
        """Test that get_config contains all __init__ parameters."""


        block = sLSTMBlock(**block_config)
        config = block.get_config()

        assert 'units' in config
        assert 'ffn_expansion_factor' in config
        assert 'ffn_type' in config
        assert 'normalization_type' in config

    def test_gradients_flow(self, block_config, sample_input_3d):
        """Test gradient computation."""


        block = sLSTMBlock(**block_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = block(sample_input_3d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, block.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, block_config, sample_input_3d, training):
        """Test behavior in different training modes."""


        block = sLSTMBlock(**block_config)

        output = block(sample_input_3d, training=training)
        assert output.shape == sample_input_3d.shape

    def test_different_expansion_factors(self, sample_input_3d):
        """Test block with different expansion factors."""


        for expansion in [1, 2, 3, 4]:
            block = sLSTMBlock(
                units=64,
                ffn_expansion_factor=expansion,
                normalization_type='layer_norm',
            )

            output = block(sample_input_3d)
            assert output.shape == sample_input_3d.shape


# =============================================================================
# mLSTMBlock Tests
# =============================================================================

class TestMLSTMBlock:
    """Comprehensive test suite for mLSTMBlock."""

    @pytest.fixture
    def block_config(self) -> Dict[str, Any]:
        """Standard configuration for mLSTMBlock."""
        return {
            'units': 64,
            'expansion_factor': 2,
            'num_heads': 4,
            'conv_kernel_size': 4,
            'normalization_type': 'layer_norm',
        }

    @pytest.fixture
    def sample_input_3d(
            self,
            sample_batch_size,
            sample_sequence_length
    ) -> keras.KerasTensor:
        """Sample 3D input matching units dimension."""
        return keras.random.normal((sample_batch_size, sample_sequence_length, 64))

    def test_initialization(self, block_config):
        """Test block initialization."""


        block = mLSTMBlock(**block_config)

        assert block.units == 64
        assert block.expansion_factor == 2
        assert block.num_heads == 4
        assert block.inner_dim == 128  # units * expansion_factor
        assert not block.built

        # Check sub-layers were created
        assert block.up_proj is not None
        assert block.conv is not None
        assert block.mlstm is not None
        assert block.norm is not None
        assert block.down_proj is not None

    def test_forward_pass(self, block_config, sample_input_3d):
        """Test forward pass through block."""


        block = mLSTMBlock(**block_config)

        output = block(sample_input_3d)

        assert block.built
        # Output should have same shape as input (residual connection)
        assert output.shape == sample_input_3d.shape

    def test_serialization_cycle(self, block_config, sample_input_3d):
        """CRITICAL TEST: Full serialization cycle."""


        # Create model with block
        inputs = keras.Input(shape=sample_input_3d.shape[1:])
        outputs = mLSTMBlock(**block_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_3d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_mlstm_block.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_3d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, block_config):
        """Test that get_config contains all __init__ parameters."""


        block = mLSTMBlock(**block_config)
        config = block.get_config()

        assert 'units' in config
        assert 'expansion_factor' in config
        assert 'num_heads' in config
        assert 'conv_kernel_size' in config
        assert 'normalization_type' in config

    def test_gradients_flow(self, block_config, sample_input_3d):
        """Test gradient computation."""


        block = mLSTMBlock(**block_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_3d)
            output = block(sample_input_3d)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, block.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, block_config, sample_input_3d, training):
        """Test behavior in different training modes."""


        block = mLSTMBlock(**block_config)

        output = block(sample_input_3d, training=training)
        assert output.shape == sample_input_3d.shape

    def test_different_num_heads(self, sample_input_3d):
        """Test block with different numbers of heads."""


        # Test with different head configurations
        # (128 inner_dim = 64 units * 2 expansion must be divisible by num_heads)
        for num_heads in [1, 2, 4, 8]:
            block = mLSTMBlock(
                units=64,
                expansion_factor=2,
                num_heads=num_heads,
                conv_kernel_size=4,
                normalization_type='layer_norm',
            )

            output = block(sample_input_3d)
            assert output.shape == sample_input_3d.shape


# =============================================================================
# Integration Tests
# =============================================================================

class TestXLSTMIntegration:
    """Integration tests for xLSTM components."""

    def test_stacked_slstm_blocks(
            self,
            sample_batch_size,
            sample_sequence_length
    ):
        """Test stacking multiple sLSTM blocks."""


        units = 64
        num_blocks = 3

        # Create stacked blocks
        inputs = keras.Input(shape=(sample_sequence_length, units))
        x = inputs
        for i in range(num_blocks):
            x = sLSTMBlock(
                units=units,
                ffn_expansion_factor=2,
                name=f'slstm_block_{i}'
            )(x)

        model = keras.Model(inputs, x)

        # Test forward pass
        sample_input = keras.random.normal(
            (sample_batch_size, sample_sequence_length, units)
        )
        output = model(sample_input)

        assert output.shape == sample_input.shape

    def test_stacked_mlstm_blocks(
            self,
            sample_batch_size,
            sample_sequence_length
    ):
        """Test stacking multiple mLSTM blocks."""


        units = 64
        num_blocks = 3

        # Create stacked blocks
        inputs = keras.Input(shape=(sample_sequence_length, units))
        x = inputs
        for i in range(num_blocks):
            x = mLSTMBlock(
                units=units,
                expansion_factor=2,
                num_heads=4,
                conv_kernel_size=4,
                name=f'mlstm_block_{i}'
            )(x)

        model = keras.Model(inputs, x)

        # Test forward pass
        sample_input = keras.random.normal(
            (sample_batch_size, sample_sequence_length, units)
        )
        output = model(sample_input)

        assert output.shape == sample_input.shape

    def test_mixed_block_architecture(
            self,
            sample_batch_size,
            sample_sequence_length
    ):
        """Test mixing sLSTM and mLSTM blocks."""
        units = 64

        # Create mixed architecture
        inputs = keras.Input(shape=(sample_sequence_length, units))
        x = sLSTMBlock(units=units, ffn_expansion_factor=2)(inputs)
        x = mLSTMBlock(units=units, expansion_factor=2, num_heads=4)(x)
        x = sLSTMBlock(units=units, ffn_expansion_factor=2)(x)

        model = keras.Model(inputs, x)

        # Test forward pass
        sample_input = keras.random.normal(
            (sample_batch_size, sample_sequence_length, units)
        )
        output = model(sample_input)

        assert output.shape == sample_input.shape

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mixed_xlstm.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            loaded_output = loaded_model(sample_input)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6
            )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])