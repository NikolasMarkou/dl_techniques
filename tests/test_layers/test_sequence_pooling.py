"""
Comprehensive test suite for SequencePooling layers.

This module contains thorough tests for all pooling strategies, serialization,
masking, and edge cases to ensure robust functionality.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Dict, Any, List, Optional

from dl_techniques.layers.sequence_pooling import (
    AttentionPooling,
    WeightedPooling,
    SequencePooling
)


class TestAttentionPooling:
    """Test suite for AttentionPooling layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_dim': 64,
            'num_heads': 1,
            'dropout_rate': 0.1,
            'use_bias': True,
            'temperature': 1.0
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input tensor for testing."""
        return keras.ops.convert_to_tensor(
            np.random.randn(4, 10, 32).astype(np.float32)
        )

    @pytest.fixture
    def sample_mask(self) -> keras.KerasTensor:
        """Sample mask tensor for testing."""
        mask = np.ones((4, 10), dtype=np.float32)
        mask[0, 7:] = 0  # Mask out last 3 positions for first sample
        mask[1, 8:] = 0  # Mask out last 2 positions for second sample
        return keras.ops.convert_to_tensor(mask)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization."""
        layer = AttentionPooling(**layer_config)

        assert layer.hidden_dim == 64
        assert layer.num_heads == 1
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is True
        assert layer.temperature == 1.0
        assert not layer.built
        assert layer.attention_dense is not None
        assert layer.dropout is not None

    def test_forward_pass(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor
    ) -> None:
        """Test forward pass and building."""
        layer = AttentionPooling(**layer_config)

        # First call builds the layer
        output = layer(sample_input)

        assert layer.built
        assert output.shape == (4, 32)  # (batch_size, embed_dim)
        assert layer.context_vector is not None
        assert layer.context_vector.shape == (1, 64)

    def test_forward_pass_with_mask(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor,
            sample_mask: keras.KerasTensor
    ) -> None:
        """Test forward pass with masking."""
        layer = AttentionPooling(**layer_config)

        output = layer(sample_input, mask=sample_mask)

        assert output.shape == (4, 32)
        # Output should be valid (not NaN or Inf)
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_multi_head_attention(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multi-head attention pooling."""
        layer = AttentionPooling(
            hidden_dim=64,
            num_heads=4,
            dropout_rate=0.0
        )

        output = layer(sample_input)

        assert output.shape == (4, 32)
        assert layer.context_vector.shape == (4, 64)  # 4 heads

    def test_no_dropout(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test layer without dropout."""
        layer = AttentionPooling(
            hidden_dim=64,
            num_heads=1,
            dropout_rate=0.0
        )

        output = layer(sample_input)

        assert layer.dropout is None
        assert output.shape == (4, 32)

    def test_compute_output_shape(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """Test output shape computation."""
        layer = AttentionPooling(**layer_config)
        input_shape = (None, 10, 32)

        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 32)

    def test_serialization_cycle(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor
    ) -> None:
        """Test full serialization and deserialization cycle."""
        # Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = AttentionPooling(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # Get prediction from original
        original_prediction = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            # Verify identical outputs
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model output should match original"
            )

    def test_get_config(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """Test configuration serialization."""
        layer = AttentionPooling(**layer_config)
        config = layer.get_config()

        # Check all parameters are in config
        assert 'hidden_dim' in config
        assert 'num_heads' in config
        assert 'dropout_rate' in config
        assert 'use_bias' in config
        assert 'temperature' in config
        assert 'kernel_initializer' in config
        assert 'kernel_regularizer' in config

        # Recreate layer from config
        new_layer = AttentionPooling.from_config(config)
        assert new_layer.hidden_dim == layer.hidden_dim
        assert new_layer.num_heads == layer.num_heads


class TestWeightedPooling:
    """Test suite for WeightedPooling layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'max_seq_len': 100,
            'dropout_rate': 0.1,
            'temperature': 2.0
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input tensor for testing."""
        return keras.ops.convert_to_tensor(
            np.random.randn(4, 10, 32).astype(np.float32)
        )

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization."""
        layer = WeightedPooling(**layer_config)

        assert layer.max_seq_len == 100
        assert layer.dropout_rate == 0.1
        assert layer.temperature == 2.0
        assert not layer.built
        assert layer.dropout is not None

    def test_forward_pass(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor
    ) -> None:
        """Test forward pass and building."""
        layer = WeightedPooling(**layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape == (4, 32)
        assert layer.position_weights is not None
        assert layer.position_weights.shape == (100,)

    def test_forward_pass_with_mask(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor
    ) -> None:
        """Test forward pass with masking."""
        layer = WeightedPooling(**layer_config)
        mask = keras.ops.ones((4, 10))
        mask = ops.cast(mask, 'float32')

        output = layer(sample_input, mask=mask)

        assert output.shape == (4, 32)
        assert not ops.any(ops.isnan(output))

    def test_variable_sequence_length(
            self,
            layer_config: Dict[str, Any]
    ) -> None:
        """Test with different sequence lengths."""
        layer = WeightedPooling(**layer_config)

        # Test with sequence length 5
        input1 = keras.ops.convert_to_tensor(
            np.random.randn(2, 5, 16).astype(np.float32)
        )
        output1 = layer(input1)
        assert output1.shape == (2, 16)

        # Test with sequence length 20
        input2 = keras.ops.convert_to_tensor(
            np.random.randn(3, 20, 16).astype(np.float32)
        )
        output2 = layer(input2)
        assert output2.shape == (3, 16)

    def test_serialization_cycle(
            self,
            layer_config: Dict[str, Any],
            sample_input: keras.KerasTensor
    ) -> None:
        """Test full serialization and deserialization cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = WeightedPooling(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model output should match original"
            )


class TestSequencePooling:
    """Test suite for SequencePooling layer."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input tensor for testing."""
        return keras.ops.convert_to_tensor(
            np.random.randn(4, 10, 32).astype(np.float32)
        )

    @pytest.fixture
    def sample_mask(self) -> keras.KerasTensor:
        """Sample mask tensor for testing."""
        mask = np.ones((4, 10), dtype=np.float32)
        mask[0, 7:] = 0
        mask[1, 8:] = 0
        return keras.ops.convert_to_tensor(mask)

    # Test positional strategies

    def test_cls_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test CLS token pooling."""
        layer = SequencePooling(strategy='cls')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        # Should equal first position
        expected = sample_input[:, 0, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="CLS pooling should return first token"
        )

    def test_first_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test first token pooling."""
        layer = SequencePooling(strategy='first')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = sample_input[:, 0, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="First pooling should return first token"
        )

    def test_last_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test last token pooling."""
        layer = SequencePooling(strategy='last')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = sample_input[:, -1, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Last pooling should return last token"
        )

    def test_last_pooling_with_mask(
            self,
            sample_input: keras.KerasTensor,
            sample_mask: keras.KerasTensor
    ) -> None:
        """Test last token pooling with mask."""
        layer = SequencePooling(strategy='last')
        output = layer(sample_input, mask=sample_mask)

        assert output.shape == (4, 32)
        # First sample should return position 6 (last unmasked)
        expected_first = sample_input[0, 6, :]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output[0]),
            keras.ops.convert_to_numpy(expected_first),
            rtol=1e-6, atol=1e-6,
            err_msg="Last pooling with mask should return last unmasked token"
        )

    def test_middle_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test middle token pooling."""
        layer = SequencePooling(strategy='middle')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = sample_input[:, 5, :]  # Middle of 10 tokens
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Middle pooling should return middle token"
        )

    # Test statistical strategies

    def test_mean_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test mean pooling."""
        layer = SequencePooling(strategy='mean')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = ops.mean(sample_input, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Mean pooling should average across sequence"
        )

    def test_mean_pooling_with_mask(
            self,
            sample_input: keras.KerasTensor,
            sample_mask: keras.KerasTensor
    ) -> None:
        """Test mean pooling with mask."""
        layer = SequencePooling(strategy='mean')
        output = layer(sample_input, mask=sample_mask)

        assert output.shape == (4, 32)

        # Manually compute expected mean for first sample (7 valid tokens)
        expected_first = ops.mean(sample_input[0, :7, :], axis=0)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output[0]),
            keras.ops.convert_to_numpy(expected_first),
            rtol=1e-6, atol=1e-6,
            err_msg="Mean pooling with mask should only average valid tokens"
        )

    def test_max_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test max pooling."""
        layer = SequencePooling(strategy='max')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = ops.max(sample_input, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Max pooling should return maximum across sequence"
        )

    def test_min_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test min pooling."""
        layer = SequencePooling(strategy='min')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = ops.min(sample_input, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Min pooling should return minimum across sequence"
        )

    def test_sum_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test sum pooling."""
        layer = SequencePooling(strategy='sum')
        output = layer(sample_input)

        assert output.shape == (4, 32)
        expected = ops.sum(sample_input, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Sum pooling should sum across sequence"
        )

    # Test combined strategies

    def test_mean_max_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test mean_max combined pooling."""
        layer = SequencePooling(strategy='mean_max')
        output = layer(sample_input)

        assert output.shape == (4, 64)  # Concatenated

        mean_part = output[:, :32]
        max_part = output[:, 32:]

        expected_mean = ops.mean(sample_input, axis=1)
        expected_max = ops.max(sample_input, axis=1)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(mean_part),
            keras.ops.convert_to_numpy(expected_mean),
            rtol=1e-6, atol=1e-6,
            err_msg="First half should be mean pooling"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(max_part),
            keras.ops.convert_to_numpy(expected_max),
            rtol=1e-6, atol=1e-6,
            err_msg="Second half should be max pooling"
        )

    def test_mean_std_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test mean_std combined pooling."""
        layer = SequencePooling(strategy='mean_std')
        output = layer(sample_input)

        assert output.shape == (4, 64)

        mean_part = output[:, :32]
        expected_mean = ops.mean(sample_input, axis=1)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(mean_part),
            keras.ops.convert_to_numpy(expected_mean),
            rtol=1e-6, atol=1e-6,
            err_msg="First half should be mean pooling"
        )

    def test_mean_max_min_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test mean_max_min combined pooling."""
        layer = SequencePooling(strategy='mean_max_min')
        output = layer(sample_input)

        assert output.shape == (4, 96)  # 3x concatenated

    # Test learnable strategies

    def test_attention_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test attention pooling."""
        layer = SequencePooling(
            strategy='attention',
            attention_hidden_dim=64,
            attention_dropout=0.1
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)
        assert 'attention' in layer.learnable_components

    def test_multi_head_attention_pooling(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multi-head attention pooling."""
        layer = SequencePooling(
            strategy='multi_head_attention',
            attention_hidden_dim=64,
            attention_num_heads=4
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)
        assert 'multi_head_attention' in layer.learnable_components

    def test_weighted_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test weighted pooling."""
        layer = SequencePooling(
            strategy='weighted',
            weighted_max_seq_len=100
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)
        assert 'weighted' in layer.learnable_components

    # Test top-k strategies

    def test_top_k_mean_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test top-k mean pooling."""
        layer = SequencePooling(strategy='top_k_mean', top_k=5)
        output = layer(sample_input)

        assert output.shape == (4, 32)

    def test_top_k_max_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test top-k max pooling."""
        layer = SequencePooling(strategy='top_k_max', top_k=5)
        output = layer(sample_input)

        assert output.shape == (4, 32)

    # Test special strategies

    def test_none_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test none pooling (identity)."""
        layer = SequencePooling(strategy='none')
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(sample_input),
            rtol=1e-6, atol=1e-6,
            err_msg="None pooling should return input unchanged"
        )

    def test_flatten_pooling(self, sample_input: keras.KerasTensor) -> None:
        """Test flatten pooling."""
        layer = SequencePooling(strategy='flatten')
        output = layer(sample_input)

        assert output.shape == (4, 320)  # 10 * 32

    # Test position exclusion

    def test_exclude_positions(self, sample_input: keras.KerasTensor) -> None:
        """Test position exclusion in pooling."""
        layer = SequencePooling(
            strategy='mean',
            exclude_positions=[0, 1]  # Exclude first two positions
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)

        # Manually compute expected mean excluding first two positions
        expected = ops.mean(sample_input[:, 2:, :], axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Should exclude specified positions from pooling"
        )

    # Test multiple strategies with aggregation

    def test_multiple_strategies_concat(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multiple strategies with concatenation."""
        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='concat'
        )
        output = layer(sample_input)

        assert output.shape == (4, 64)  # Concatenated

    def test_multiple_strategies_add(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multiple strategies with addition."""
        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='add'
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)  # Same dimension

        expected = ops.mean(sample_input, axis=1) + ops.max(sample_input, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(expected),
            rtol=1e-6, atol=1e-6,
            err_msg="Add aggregation should sum strategy outputs"
        )

    def test_multiple_strategies_multiply(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multiple strategies with multiplication."""
        # Use small values to avoid numerical issues
        small_input = sample_input * 0.1

        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='multiply'
        )
        output = layer(small_input)

        assert output.shape == (4, 32)

    def test_multiple_strategies_weighted_sum(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test multiple strategies with weighted sum."""
        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='weighted_sum'
        )
        output = layer(sample_input)

        assert output.shape == (4, 32)
        assert layer.aggregation_weights is not None
        assert layer.aggregation_weights.shape == (2,)

    def test_mixed_strategies(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test mixing different types of strategies."""
        layer = SequencePooling(
            strategy=['cls', 'mean', 'attention'],
            aggregation_method='concat',
            attention_hidden_dim=32
        )
        output = layer(sample_input)

        assert output.shape == (4, 96)  # 3 * 32

    # Test compute_output_shape

    def test_compute_output_shape_single(self) -> None:
        """Test output shape computation for single strategy."""
        layer = SequencePooling(strategy='mean')
        input_shape = (None, 10, 32)

        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 32)

        # Test mean_max
        layer = SequencePooling(strategy='mean_max')
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 64)

        # Test none
        layer = SequencePooling(strategy='none')
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape

        # Test flatten
        layer = SequencePooling(strategy='flatten')
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 320)

    def test_compute_output_shape_multiple(self) -> None:
        """Test output shape computation for multiple strategies."""
        input_shape = (None, 10, 32)

        # Concat
        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='concat'
        )
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 64)

        # Add
        layer = SequencePooling(
            strategy=['mean', 'max'],
            aggregation_method='add'
        )
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 32)

    # Test serialization

    def test_serialization_simple(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test serialization with simple strategy."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = SequencePooling(strategy='mean')(inputs)
        model = keras.Model(inputs, layer_output)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Simple pooling serialization failed"
            )

    def test_serialization_learnable(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test serialization with learnable strategy."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = SequencePooling(
            strategy='attention',
            attention_hidden_dim=64,
            attention_dropout=0.0  # No dropout for deterministic test
        )(inputs)
        model = keras.Model(inputs, layer_output)

        # Train for a few steps to get non-random weights
        model.compile(optimizer='adam', loss='mse')
        dummy_target = keras.ops.zeros((4, 32))
        model.fit(sample_input, dummy_target, epochs=1, verbose=0)

        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Learnable pooling serialization failed"
            )

    def test_serialization_complex(
            self,
            sample_input: keras.KerasTensor
    ) -> None:
        """Test serialization with complex configuration."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = SequencePooling(
            strategy=['mean', 'max', 'attention', 'weighted'],
            aggregation_method='concat',
            exclude_positions=[0],
            attention_hidden_dim=32,
            attention_num_heads=2,
            weighted_max_seq_len=50,
            top_k=5,
            temperature=2.0
        )(inputs)
        model = keras.Model(inputs, layer_output)

        original_prediction = model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Complex pooling serialization failed"
            )

    def test_get_config(self) -> None:
        """Test configuration serialization completeness."""
        layer = SequencePooling(
            strategy=['mean', 'attention'],
            exclude_positions=[0, 1],
            aggregation_method='concat',
            attention_hidden_dim=128,
            attention_num_heads=4,
            attention_dropout=0.2,
            weighted_max_seq_len=256,
            top_k=20,
            temperature=0.5,
            use_bias=False
        )

        config = layer.get_config()

        # Check ALL parameters are in config
        assert config['strategy'] == ['mean', 'attention']
        assert config['exclude_positions'] == [0, 1]
        assert config['aggregation_method'] == 'concat'
        assert config['attention_hidden_dim'] == 128
        assert config['attention_num_heads'] == 4
        assert config['attention_dropout'] == 0.2
        assert config['weighted_max_seq_len'] == 256
        assert config['top_k'] == 20
        assert config['temperature'] == 0.5
        assert config['use_bias'] is False
        assert 'kernel_initializer' in config
        assert 'bias_initializer' in config
        assert 'kernel_regularizer' in config
        assert 'bias_regularizer' in config

        # Recreate from config
        new_layer = SequencePooling.from_config(config)
        assert new_layer.strategy == layer.strategy
        assert new_layer.exclude_positions == layer.exclude_positions

    # Test error cases

    def test_invalid_strategy(self) -> None:
        """Test invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            layer = SequencePooling(strategy='invalid_strategy')
            input_tensor = keras.ops.ones((2, 10, 32))
            layer(input_tensor)

    def test_invalid_aggregation(self) -> None:
        """Test invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            layer = SequencePooling(
                strategy=['mean', 'max'],
                aggregation_method='invalid_method'  # type: ignore
            )
            input_tensor = keras.ops.ones((2, 10, 32))
            layer(input_tensor)

    def test_none_with_concat(self, sample_input: keras.KerasTensor) -> None:
        """Test that 'none' strategy cannot be concatenated."""
        with pytest.raises(
                ValueError,
                match="Cannot concatenate 'none' strategy with others"
        ):
            layer = SequencePooling(
                strategy=['none', 'mean'],
                aggregation_method='concat'
            )
            layer(sample_input)


class TestIntegrationScenarios:
    """Integration tests for realistic use cases."""

    def test_bert_style_pooling(self) -> None:
        """Test BERT-style CLS token pooling."""
        # BERT output shape
        batch_size, seq_len, hidden_dim = 2, 128, 768
        bert_output = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        )

        # CLS pooling
        layer = SequencePooling(strategy='cls')
        pooled = layer(bert_output)

        assert pooled.shape == (batch_size, hidden_dim)

    def test_sentence_transformer_pooling(self) -> None:
        """Test sentence transformer mean pooling excluding CLS."""
        batch_size, seq_len, hidden_dim = 2, 128, 768
        encoder_output = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        )

        # Mean pooling excluding CLS token (position 0)
        layer = SequencePooling(
            strategy='mean',
            exclude_positions=[0]
        )
        pooled = layer(encoder_output)

        assert pooled.shape == (batch_size, hidden_dim)

    def test_vision_transformer_pooling(self) -> None:
        """Test ViT-style pooling with CLS token."""
        batch_size, num_patches, hidden_dim = 2, 197, 768  # 196 patches + 1 CLS
        vit_output = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, num_patches, hidden_dim).astype(np.float32)
        )

        # Use CLS token for classification
        layer = SequencePooling(strategy='cls')
        pooled = layer(vit_output)

        assert pooled.shape == (batch_size, hidden_dim)

    def test_multi_modal_pooling(self) -> None:
        """Test pooling for multi-modal fusion."""
        batch_size, seq_len, hidden_dim = 2, 64, 512
        modal_output = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        )

        # Rich representation with multiple pooling strategies
        layer = SequencePooling(
            strategy=['mean', 'max', 'attention'],
            aggregation_method='concat',
            attention_hidden_dim=128
        )
        pooled = layer(modal_output)

        # 3 strategies concatenated
        assert pooled.shape == (batch_size, hidden_dim * 3)

    def test_time_series_pooling(self) -> None:
        """Test pooling for time series data."""
        batch_size, time_steps, features = 4, 100, 32
        time_series = keras.ops.convert_to_tensor(
            np.random.randn(batch_size, time_steps, features).astype(np.float32)
        )

        # Weighted pooling to learn temporal importance
        layer = SequencePooling(
            strategy='weighted',
            weighted_max_seq_len=100,
            temperature=1.0
        )
        pooled = layer(time_series)

        assert pooled.shape == (batch_size, features)


# ---------------------------------------------------------------------
# F-24 — masked-position ISOLATION for `weighted` / `top_k_mean` / `top_k_max`
#
# Before this suite existed, ZERO tests in this module passed a `mask` to these
# three strategies, so the defect was completely unguarded at package level.
# Measured leak before the fix by THIS suite (seeded weights, `seq_len=6`,
# `top_k=10`, position 5 masked and perturbed), where the required movement is
# exactly 0.0:
#
#   weighted      1.129423e+00   (at every `top_k`; the defect is `softmax(0) != 0`)
#   top_k_mean    1.463992e+00   (only at `top_k=10`, i.e. `k > kept_count`)
#   top_k_max     5.990623e+00   (only at `top_k=10`, i.e. `k > kept_count`)
#
# ---------------------------------------------------------------------

# F-25's `middle`/`last` are deliberately NOT added to this list, and that was
# MEASURED, not assumed. Every suite parametrized by it is built on
# `ISO_MASKED_POS = 5` / `ISO_LIVE_POS = 0`, and a positional mode selects
# exactly ONE token: at `seq_len = 6` with position 5 masked, `middle` lands on
# index 2 and `last` on index 4, so perturbing the live control at position 0
# moves the output by exactly 0.000000e+00. Adding the two modes here produces
# 20 failures, ALL of them the suites' own vacuous-probe guards firing (plus
# `test_fully_masked_row_is_finite_and_documented`'s `-1e4` sentinel branch,
# which is a `top_k_max` fact) — manufactured noise, not coverage. F-25's
# guards live in `TestPositionalModesIsolateMaskedPositions` below, on cells
# chosen to discriminate for these modes.
ISO_LEAKY_STRATEGIES = ['weighted', 'top_k_mean', 'top_k_max']

ISO_ALL_STRATEGIES = [
    'cls', 'first', 'last', 'middle',
    'mean', 'max', 'min', 'sum',
    'mean_max', 'mean_std', 'mean_max_min',
    'attention', 'multi_head_attention', 'weighted',
    'top_k_mean', 'top_k_max',
    'none', 'flatten',
]

ISO_B, ISO_S, ISO_D = 2, 6, 8
ISO_MASKED_POS = 5      # last position; no positional strategy selects it here
ISO_LIVE_POS = 0        # perturbed for the mandatory live control

# SC-2 / I1 pin: `mask=None` AND `exclude_positions=None` outputs for all 18
# strategies, captured from the code BEFORE the F-24 fix with the fixture below
# (`_iso_layer(..., top_k=10)`, `_iso_ref_inputs()`), as
# ``(sum, sum_of_abs, first_element)`` in float64.
# A fix that moves ANY unmasked numeric moves at least the sum-of-abs.
#
# SCOPE, measured — this pin covers `exclude_positions=None` ONLY.
# `SequencePooling._apply_mask_and_exclusions` SYNTHESISES an all-ones mask
# whenever `exclude_positions` is set, so `exclude_positions` is a SECOND entry
# point into the D-002 masking fix that fires even at `mask=None`.
#
# RE-DERIVED A/B against a pristine `git worktree` at `5500c6bc`, CPU, with this
# module's own fixture at `_iso_layer(..., top_k=10, exclude_positions=[1])`,
# reported as max|elementwise diff| (the metric is named because the earlier
# draft of this note quoted three numbers that could not be reproduced):
#
#   `_iso_inputs()`      weighted 1.307e-01  top_k_mean 2.435e-01  top_k_max 2.807e-01
#   `_iso_ref_inputs()`  weighted 2.660e-01  top_k_mean 3.618e-01  top_k_max 4.558e-01
#
# In BOTH fixtures the SAME 15 of 18 strategies are bit-identical and the SAME 3
# (exactly the D-002 repaired ones) move. The movement is the fix REACHING a path
# it never reached, not a regression: pre-fix, `top_k_max` at
# `exclude_positions=[1]` was BIT-IDENTICAL to `exclude_positions=None` (the
# exclusion was ignored outright) and `top_k_mean` differed from it by only
# 2.98e-08; post-fix `top_k_max` equals `max` over the non-excluded positions
# EXACTLY (0.0) and `top_k_mean` equals `mean` over them to 5.96e-08 -- float32
# round-off in the validity divide, NOT "to the last bit".
# No shipped consumer is affected: the only two callers that pass
# `exclude_positions` are `vision_encoder.py:420`/`:424` (one gate, one kwarg)
# and `models/vit/model.py:442` -- line numbers re-grepped at write time, step
# 15; an earlier revision of this comment carried a stale `vision_encoder.py:396`
# from a previous HEAD. Both restrict the strategy to `mean`/`max`, and both
# measure exactly 0.0 movement in the A/B above. It is guarded by
# `TestExcludePositionsIsAlsoIsolated` below.
#
# THE CAPTURE/RUN SITE BELOW IS DEVICE-PINNED (`golden_reference_device`,
# `tests/conftest.py`) and the triples were captured under the same pin. Without
# it the `multi_head_attention` cell FAILS on GPU: measured on GPU 1 (RTX 4070),
# ACTUAL sum 0.6348102763295174 vs DESIRED 0.63527483121, relative 7.31e-04 --
# 731x the rtol=1e-6 below -- a pure TF32 matmul artifact amplified by
# cancellation (sum 0.635 against sum_of_abs 11.2). It passes with
# `NVIDIA_TF32_OVERRIDE=0` and on CPU.
# WHAT NOT TO DO: do not widen the tolerance to absorb it -- a real regression of
# this defect's shape is O(1e-1), and the margin is what makes the pin
# meaningful.
# WHAT IS *NOT* LOAD-BEARING HERE, measured rather than assumed: the pin belongs
# on the BUILD/RUN site ONLY. The comparison is `np.testing.assert_allclose` over
# a HOST numpy array (`_np(...)` has already left the backend), so `keras.device`
# cannot reach it. All four variants run over all 18 strategies on GPU 1, step 15:
#     build/run PINNED   | compare PINNED    -> 0 FAIL   (the shipped shape)
#     build/run PINNED   | compare unpinned  -> 0 FAIL   (identical; the pin is inert)
#     build/run unpinned | compare PINNED    -> 1 FAIL   multi_head_attention, rel 7.31e-04
#     build/run unpinned | compare unpinned  -> 1 FAIL   the SAME cell, the SAME numbers
# So a second `keras.device` around the asserts buys nothing and was removed. An
# earlier revision of this comment claimed the opposite ("pinning the capture but
# not the comparison converts one failure into a different one") on the strength
# of a CITED prior-plan measurement; executing it here refuted that for this site.
# Tolerance stays 1e-6 relative rather than 0 because the triples are stored as
# decimal literals, not bits.
I1_UNMASKED_GOLDEN = {
    'cls': (-2.4325768099e+00, 1.3625477175e+01, 1.2301533716e-03),
    'first': (-2.4325768099e+00, 1.3625477175e+01, 1.2301533716e-03),
    'last': (-4.8015104979e-01, 1.7474361904e+01, -6.4147037268e-01),
    'middle': (-9.0585865807e+00, 1.2501058457e+01, 1.5675108135e-01),
    'mean': (-4.2395858765e+00, 7.4127014056e+00, -4.5542362332e-01),
    'max': (2.4226868525e+01, 2.4226868525e+01, 1.5675108135e-01),
    'min': (-3.3379035980e+01, 3.3379035980e+01, -1.3442145586e+00),
    'sum': (-2.9677100927e+01, 5.1888909429e+01, -3.1879653931e+00),
    'mean_max': (1.9987282649e+01, 3.1639569931e+01, -4.5542362332e-01),
    'mean_std': (1.4443011433e+01, 2.6095298715e+01, -4.5542362332e-01),
    'mean_max_min': (-1.3391753331e+01, 6.5018605910e+01, -4.5542362332e-01),
    'attention': (-1.2140299166e+00, 9.4650178207e+00, -7.8558260202e-01),
    'multi_head_attention': (6.3527483121e-01, 1.1215872373e+01, -4.8161357641e-01),
    'weighted': (-4.9283401147e+00, 9.4189609364e+00, -3.5132110119e-01),
    'top_k_mean': (-4.2395857349e+00, 7.4127013832e+00, -4.5542362332e-01),
    'top_k_max': (2.4226868525e+01, 2.4226868525e+01, 1.5675108135e-01),
    'none': (-2.9677100304e+01, 1.1939868845e+02, 1.2301533716e-03),
    'flatten': (-2.9677100304e+01, 1.1939868845e+02, 1.2301533716e-03),
}

# The I1 golden values above were captured at this geometry, which differs from
# the isolation probes' geometry on purpose (a longer sequence exercises the
# `weighted` position-weight slice further).
I1_B, I1_S, I1_D = 3, 7, 8


def _np(x: Any) -> np.ndarray:
    return keras.ops.convert_to_numpy(x)


def _iso_layer(
    strategy: str,
    top_k: int = 10,
    seed: int = 1234,
    seq_len: int = ISO_S,
    exclude_positions: Optional[List[int]] = None,
) -> SequencePooling:
    """Build a `SequencePooling` and assign EVERY weight from a seeded RNG.

    Fresh Keras initialisers leave biases at zero and `WeightedPooling`'s
    position weights at a constant ``1.0``; both make a masking site far less
    observable than it is in a trained model (prior-plan D-008). These fixtures
    therefore put the layer in the state a *trained* model is in.

    `weighted` and `top_k_*` carry no bias weights at all (`WeightedPooling` has
    only `position_weights`; the top-k branches are weightless), so the
    "at least one non-zero bias" assertion is applied where biases EXIST and
    the strategy is otherwise checked to be a known biasless one. Non-vacuity
    for those strategies rests on the seeded non-uniform position weights and on
    the mandatory live control that every test below carries.
    """
    layer = SequencePooling(
        strategy=strategy,
        exclude_positions=exclude_positions,
        attention_hidden_dim=16,
        attention_num_heads=2,
        weighted_max_seq_len=32,
        top_k=top_k,
        use_bias=True,
    )
    layer.build((None, seq_len, ISO_D))

    rng = np.random.default_rng(seed)
    saw_nonzero_bias = False
    for w in layer.weights:
        shape = tuple(w.shape)
        name = w.path.split('/')[-1]
        if 'bias' in name or 'beta' in name:
            value = 0.3 + 0.2 * rng.normal(size=shape)
            saw_nonzero_bias = True
        else:
            value = 0.5 * rng.normal(size=shape) + 0.2
        w.assign(keras.ops.cast(keras.ops.convert_to_tensor(value), w.dtype))

    if strategy in ('attention', 'multi_head_attention'):
        assert saw_nonzero_bias, (
            f"Fixture is degenerate for {strategy!r}: no bias weight was assigned "
            f"a non-zero value, so a masking site downstream of a zeroed "
            f"activation would be structurally unobservable."
        )
    else:
        assert not saw_nonzero_bias, (
            f"{strategy!r} unexpectedly grew a bias weight; the fixture's "
            f"non-zero-bias assertion must now cover it too."
        )
    return layer


def _iso_inputs(seed: int = 7, batch: int = ISO_B, seq_len: int = ISO_S) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, seq_len, ISO_D)).astype('float32')


def _iso_ref_inputs() -> np.ndarray:
    """The exact inputs the `I1_UNMASKED_GOLDEN` triples were captured on."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(I1_B, I1_S, I1_D)).astype('float32')


def _iso_perturb(
    x: np.ndarray, positions, seed: int = 99, rows=None
) -> np.ndarray:
    """Return a copy of ``x`` with the given positions perturbed, per row.

    The perturbation is large and NON-UNIFORM across the embedding axis: a
    uniform offset can be cancelled by a downstream mean/normalisation and read
    as reassociation noise rather than as a real leak.
    """
    rng = np.random.default_rng(seed)
    out = np.array(x, copy=True)
    rows = range(x.shape[0]) if rows is None else rows
    for r in rows:
        for p in positions:
            out[r, p, :] += (5.0 * rng.normal(size=(x.shape[-1],))).astype(x.dtype)
    return out


def _iso_mask(masked_per_row, batch: int = ISO_B, seq_len: int = ISO_S) -> np.ndarray:
    """Build a ``(B, seq_len)`` keep-mask (1 = keep) from per-row masked positions."""
    m = np.ones((batch, seq_len), dtype='float32')
    for r, positions in enumerate(masked_per_row):
        for p in positions:
            m[r, p] = 0.0
    return m


class TestMaskedPositionIsolation:
    """F-24: `weighted` / `top_k_mean` / `top_k_max` must isolate a masked position.

    Every test carries a LIVE CONTROL — the same perturbation applied to an
    UNMASKED position must move the pooled output by a wide margin — so a test
    that passes because *nothing* moves is impossible.
    """

    @pytest.mark.parametrize("top_k", [3, 5, 10])
    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_masked_position_is_isolated(self, strategy: str, top_k: int) -> None:
        """Perturbing a masked position must leave the pooled output bit-identical.

        `top_k` is swept ACROSS the `k == kept_count` boundary deliberately.
        With `seq_len=6` and one masked position the kept count is 5, so:
        `top_k=3` selects fewer than the kept count (green before the fix too),
        `top_k=5` sits exactly on the boundary (measured 0.0 before the fix —
        the probe must not rest on this cell), and `top_k=10` clamps to
        `k = min(10, 6) = 6 > 5`, which is the cell that actually leaked.
        """
        layer = _iso_layer(strategy, top_k=top_k)
        x = _iso_inputs()
        mask = keras.ops.convert_to_tensor(_iso_mask([[ISO_MASKED_POS]] * ISO_B))

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))

        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_LIVE_POS])),
            mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe for strategy={strategy!r}, top_k={top_k}: perturbing "
            f"the UNMASKED position {ISO_LIVE_POS} moved the output by only "
            f"{live_delta:.6e}, so the isolation assertion below proves nothing."
        )

        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_MASKED_POS])),
            mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}, top_k={top_k}: a MASKED position leaked "
                f"into the pooled output by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_heterogeneous_kept_counts_are_isolated(self, strategy: str) -> None:
        """`k` is batch-GLOBAL while the kept count is PER-ROW — that asymmetry is the defect.

        Row 0 keeps 5 positions, row 1 keeps 3. Any fix that clamps `k` to the
        batch-wide minimum would make row 0's answer depend on row 1's mask; any
        fix that does not exclude invalid SELECTED positions leaks in row 1.
        """
        layer = _iso_layer(strategy, top_k=10)
        x = _iso_inputs(seed=11)
        masked_per_row = [[ISO_MASKED_POS], [3, 4, ISO_MASKED_POS]]
        mask = keras.ops.convert_to_tensor(_iso_mask(masked_per_row))

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))

        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_LIVE_POS])),
            mask=mask, training=False,
        ))
        assert float(np.max(np.abs(live - base))) > 1e-2, (
            f"Vacuous probe for strategy={strategy!r}: the live control did not move."
        )

        perturbed = np.array(x, copy=True)
        for row, positions in enumerate(masked_per_row):
            perturbed = _iso_perturb(perturbed, positions, rows=[row])
        leaked = _np(layer(
            keras.ops.convert_to_tensor(perturbed), mask=mask, training=False
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}: masked positions leaked with "
                f"heterogeneous per-row kept counts (5 and 3) by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_fully_masked_row_is_finite_and_documented(self, strategy: str) -> None:
        """A fully-masked row does NOT raise and does NOT rescue — it degenerates.

        Documented per strategy (see this plan's decisions.md D-008 and the
        source docstrings); the sibling `AttentionPooling` has no rescue path
        either, so NOT adding one here keeps the package consistent:

        * `weighted`   — every logit is the same `-1e4` sentinel, so the softmax
          is uniform and the output is the plain mean over the sequence.
        * `top_k_mean` — no selected position is valid, the denominator floors
          at 1 and the output is exactly zero.
        * `top_k_max`  — every selected embedding is replaced by the `-1e4`
          sentinel, so the output is that sentinel.

        The point of the test is that all three are FINITE (no NaN from a
        `0 * -inf`, no division by zero) and that the degenerate value is a
        deliberate, pinned choice rather than whatever fell out.
        """
        layer = _iso_layer(strategy, top_k=10)
        x = _iso_inputs(seed=13)
        mask = keras.ops.convert_to_tensor(
            _iso_mask([[ISO_MASKED_POS], list(range(ISO_S))])
        )

        out = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))
        assert np.all(np.isfinite(out)), (
            f"strategy={strategy!r}: a fully-masked row produced non-finite "
            f"output {out[1]}."
        )

        row = out[1]
        if strategy == 'weighted':
            np.testing.assert_allclose(row, x[1].mean(axis=0), rtol=1e-5, atol=1e-5)
        elif strategy == 'top_k_mean':
            np.testing.assert_allclose(row, np.zeros_like(row), rtol=0, atol=0)
        else:
            np.testing.assert_allclose(row, np.full_like(row, -1e4), rtol=1e-6, atol=0)

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_isolation_holds_under_every_dtype_policy(
        self, strategy: str, dtype_policy: str
    ) -> None:
        """The `-1e4` sentinel must stay FINITE and NaN-free at every policy.

        `mixed_float16` is the reason the sentinel is `-1e4` and not
        `layers/attention/common.py`'s `-1e9`: `float16(-1e9)` is `-inf`, and an
        `-inf` in a softmax logit or a `(1 - mask) * -inf` product produces NaN.
        The findings probe for F-24 ran float32/CPU only, so this re-runs it
        across the restore-safe `dtype_policy` fixture in
        `tests/test_layers/conftest.py`.
        """
        layer = _iso_layer(strategy, top_k=10)
        x = _iso_inputs(seed=17)
        mask = keras.ops.convert_to_tensor(_iso_mask([[ISO_MASKED_POS]] * ISO_B))

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))
        assert np.all(np.isfinite(base)), (
            f"strategy={strategy!r} under {dtype_policy}: non-finite pooled output."
        )

        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_LIVE_POS])),
            mask=mask, training=False,
        ))
        assert float(np.max(np.abs(live.astype('float64') - base.astype('float64')))) > 1e-2

        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_MASKED_POS])),
            mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=f"strategy={strategy!r} leaked under policy {dtype_policy}.",
        )

    def test_ranking_sentinel_keeps_masked_positions_out_of_the_selection(self) -> None:
        """The F-24 fix has TWO halves; this is the one the isolation tests cannot see.

        Excluding invalid SELECTED positions from the aggregation (half ii) is
        enough for isolation, so with ``k >= seq_len`` — where every position is
        selected anyway — the RANKING sentinel (half i) is unobservable and a
        dead-component probe on it comes back green.

        This test makes it observable. With ``k < seq_len`` a masked position
        competes for a slot on norm alone, and a masked position's mask-zeroed
        norm is ``0.0``, which TIES with a genuinely kept position whose
        embedding happens to be the zero vector. ``ops.top_k`` breaks ties by
        the LOWER index, so without the sentinel the masked position at index 1
        displaces the kept zero-vector position at index 3: the valid count
        falls from 5 to 4 and ``top_k_mean`` divides by the wrong denominator
        (the answer is 5/4 of the correct one) while remaining perfectly
        isolated from the masked position's actual content.
        """
        seq_len = 6
        layer = _iso_layer('top_k_mean', top_k=5, seq_len=seq_len)

        rng = np.random.default_rng(41)
        x = rng.normal(size=(1, seq_len, ISO_D)).astype('float32') + 1.5
        x[0, 3, :] = 0.0                       # a KEPT position with a zero norm
        mask = np.ones((1, seq_len), dtype='float32')
        mask[0, 1] = 0.0                       # masked, at a LOWER index (ties go here)

        out = _np(layer(
            keras.ops.convert_to_tensor(x),
            mask=keras.ops.convert_to_tensor(mask),
            training=False,
        ))

        kept = [0, 2, 3, 4, 5]
        expected = x[0, kept, :].sum(axis=0) / len(kept)
        np.testing.assert_allclose(out[0], expected, rtol=1e-6, atol=1e-6)

        # And the wrong answer must be genuinely distinguishable, not a tie.
        wrong = x[0, [0, 2, 4, 5], :].sum(axis=0) / 4
        assert float(np.max(np.abs(expected - wrong))) > 1e-2, (
            "Degenerate probe: the correct and the drop-a-kept-position answers "
            "are numerically indistinguishable at this fixture."
        )

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_all_ones_mask_matches_no_mask(self, strategy: str) -> None:
        """An all-keep mask must be equivalent to passing no mask at all.

        This is the scope pin on the fix: it must change the answer ONLY where a
        position is actually masked. It is also what makes an over-correction
        (e.g. a sentinel that biases kept positions too) visible.
        """
        layer = _iso_layer(strategy, top_k=10)
        x = keras.ops.convert_to_tensor(_iso_inputs(seed=23))

        no_mask = _np(layer(x, mask=None, training=False))
        ones_mask = _np(layer(
            x, mask=keras.ops.ones((ISO_B, ISO_S)), training=False
        ))
        np.testing.assert_allclose(ones_mask, no_mask, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------
# F-25 — the POSITIONAL modes (`cls` / `first` / `last` / `middle`)
#
# `middle` derived its index from the PADDED length (`ops.shape(inputs)[1] // 2`)
# and never consulted the mask at all; `last` consulted it only through
# `sum(mask) - 1`, which is the last index of a contiguous PREFIX and nothing
# else. `cls`/`first` return index 0 BY INTENT and are pinned, not "fixed".
#
# Cell selection here is adversarial on purpose — the enumeration below was
# re-executed at this HEAD over all 64 masks at `seq_len = 6`:
#
#   `middle`, prefix keep `0..L-1`: the current index 3 is MASKED at L=1,2,3
#     (a real leak) and KEPT-BUT-WRONG at L=4,5 (kept-middle is 2). L=5 is
#     therefore a TWO-TIER TRAP: an isolation-only assertion is GREEN against
#     broken code there. That is why `test_middle_returns_the_kept_middle_token`
#     exists ALONGSIDE the isolation test and includes L=5.
#   `last`: every contiguous-prefix mask is degenerate (`sum-1` IS the last kept
#     index), so a prefix probe proves nothing. 57 of 64 masks discriminate
#     `cur != fix`, but only 32 make the CURRENT index masked, i.e. only those
#     are isolation cells. `keep = 101101` (this plan's findings) is NOT one of
#     them — measured at HEAD: `sum-1 = 3` and position 3 is KEPT, so it is a
#     wrong-token cell, not a leak cell. It is used for the CORRECTNESS test;
#     `keep = 110011` (`sum-1 = 3`, MASKED; last kept = 5) is used for isolation.
#
# `ISO_MASKED_POS = 5` / `ISO_LIVE_POS = 0` are deliberately NOT reused: position
# 5 is invisible to `middle`, and to `last` under any prefix mask.
#
# The live control here perturbs EVERY position rather than one kept position:
# a positional mode selects exactly ONE token, and pre-fix vs post-fix that
# token is a different index, so no single-position control can be live in both
# directions. Perturbing the whole sequence is live either way, which keeps the
# control a genuine anti-vacuity check instead of a precondition that fires
# before the assertion under test.
#
# This suite is NOT device-pinned: it compares two outputs produced on the SAME
# device (following `TestExcludePositionsIsAlsoIsolated`'s stated reasoning).
# ---------------------------------------------------------------------

F25_S = 6
F25_MIDDLE_LEAK_PREFIX_LENGTHS = [2, 3]     # index 3 masked -> a real leak
F25_MIDDLE_ALL_PREFIX_LENGTHS = [2, 3, 5]   # 5 is the kept-but-wrong-token trap
F25_LAST_LEAK_MASKED = [2, 3]               # keep = 110011; `sum-1 = 3` is MASKED
F25_LAST_WRONG_TOKEN_MASKED = [1, 4]        # keep = 101101; `sum-1 = 3` is KEPT


def _f25_kept_middle(keep: np.ndarray) -> int:
    """Oracle for the shipped semantics: the middle of the KEPT positions.

    Mirrors `SequencePooling._positional_index(mode='middle')`: with `n` kept
    positions the target rank is `n // 2 + 1`, i.e. the 0-based `n // 2`-th kept
    index. A fully-masked row degenerates to index 0 (documented, D-008).
    """
    kept = [i for i, v in enumerate(keep) if v]
    return kept[len(kept) // 2] if kept else 0


def _f25_last_kept(keep: np.ndarray) -> int:
    """Oracle for `last`: the last kept index; 0 when nothing is kept."""
    kept = [i for i, v in enumerate(keep) if v]
    return kept[-1] if kept else 0


class TestPositionalModesIsolateMaskedPositions:
    """F-25: `middle` and `last` must not return a masked token.

    `cls`/`first` are NOT isolation-tested — they leak off-prefix by INTENT
    (the caller asked for index 0), so an isolation assertion for them would go
    RED against correct code. Their contract is pinned as intent instead.
    """

    @pytest.mark.parametrize("prefix_len", F25_MIDDLE_LEAK_PREFIX_LENGTHS)
    def test_middle_isolates_a_masked_position(self, prefix_len: int) -> None:
        """A prefix mask of length L <= S//2 makes index `S//2` masked."""
        layer = _iso_layer('middle', seq_len=F25_S)
        x = _iso_inputs(seed=31, seq_len=F25_S)
        masked = list(range(prefix_len, F25_S))
        keep = _iso_mask([masked] * ISO_B, seq_len=F25_S)
        mask = keras.ops.convert_to_tensor(keep)

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))

        # Live control: perturbing the WHOLE sequence must move the output,
        # both before and after the fix (see the note above this class).
        all_positions = list(range(F25_S))
        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, all_positions)),
            mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe at prefix_len={prefix_len}: perturbing every "
            f"position moved the output by only {live_delta:.6e}."
        )

        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, masked)),
            mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy='middle', prefix_len={prefix_len} (masked positions "
                f"{masked}): a MASKED position leaked into the pooled output by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("prefix_len", F25_MIDDLE_ALL_PREFIX_LENGTHS)
    def test_middle_returns_the_kept_middle_token(self, prefix_len: int) -> None:
        """Isolation and CORRECTNESS are different properties for `middle`.

        At `prefix_len = 5` the current index 3 is KEPT, so the isolation test
        above cannot see the defect — yet the kept-middle is 2. This assertion
        is what closes that gap.
        """
        layer = _iso_layer('middle', seq_len=F25_S)
        x = _iso_inputs(seed=37, seq_len=F25_S)
        masked = list(range(prefix_len, F25_S))
        keep = _iso_mask([masked] * ISO_B, seq_len=F25_S)
        expected_idx = _f25_kept_middle(keep[0])
        assert expected_idx != F25_S // 2, (
            f"Degenerate cell: prefix_len={prefix_len} has kept-middle "
            f"{expected_idx}, which equals the padded midpoint {F25_S // 2}, so "
            f"this assertion would pass against the unfixed code."
        )

        out = _np(layer(
            keras.ops.convert_to_tensor(x),
            mask=keras.ops.convert_to_tensor(keep),
            training=False,
        ))
        np.testing.assert_allclose(
            out, x[:, expected_idx, :], rtol=0, atol=0,
            err_msg=(
                f"strategy='middle', prefix_len={prefix_len}: expected the "
                f"kept-middle token at index {expected_idx}; the padded-midpoint "
                f"token at index {F25_S // 2} differs from it by "
                f"{float(np.max(np.abs(x[:, F25_S // 2, :] - x[:, expected_idx, :]))):.6e}."
            ),
        )

    def test_last_isolates_a_masked_position_under_an_interior_mask(self) -> None:
        """`keep = 110011`: `sum(mask) - 1 = 3` is MASKED; the last kept index is 5.

        Every contiguous-prefix mask is degenerate here, which is exactly why
        this cell is interior.
        """
        layer = _iso_layer('last', seq_len=F25_S)
        x = _iso_inputs(seed=41, seq_len=F25_S)
        masked = F25_LAST_LEAK_MASKED
        keep = _iso_mask([masked] * ISO_B, seq_len=F25_S)
        assert keep[0][int(keep[0].sum()) - 1] == 0.0, (
            "Degenerate cell: `sum(mask) - 1` is a KEPT position here, so the "
            "unfixed `last` returns a real token and cannot leak."
        )
        mask = keras.ops.convert_to_tensor(keep)

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))

        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, list(range(F25_S)))),
            mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe: perturbing every position moved the output by only "
            f"{live_delta:.6e}."
        )

        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, masked)),
            mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy='last', masked positions {masked}: a MASKED position "
                f"leaked into the pooled output by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    def test_last_returns_the_last_kept_index(self) -> None:
        """`keep = 101101`: `sum(mask) - 1 = 3` is KEPT but is NOT the last kept index.

        This is the `last` analogue of `middle`'s two-tier trap — the unfixed
        code returns a real, unmasked, WRONG token, so isolation alone is blind.
        """
        layer = _iso_layer('last', seq_len=F25_S)
        x = _iso_inputs(seed=43, seq_len=F25_S)
        keep = _iso_mask([F25_LAST_WRONG_TOKEN_MASKED] * ISO_B, seq_len=F25_S)
        stale_idx = int(keep[0].sum()) - 1
        expected_idx = _f25_last_kept(keep[0])
        assert keep[0][stale_idx] == 1.0 and stale_idx != expected_idx, (
            "Degenerate cell: `sum(mask) - 1` must be a KEPT position that is "
            "NOT the last kept index for this test to discriminate."
        )

        out = _np(layer(
            keras.ops.convert_to_tensor(x),
            mask=keras.ops.convert_to_tensor(keep),
            training=False,
        ))
        np.testing.assert_allclose(
            out, x[:, expected_idx, :], rtol=0, atol=0,
            err_msg=(
                f"strategy='last', masked positions "
                f"{F25_LAST_WRONG_TOKEN_MASKED}: expected the last KEPT index "
                f"{expected_idx}; the `sum(mask)-1` index {stale_idx} differs by "
                f"{float(np.max(np.abs(x[:, stale_idx, :] - x[:, expected_idx, :]))):.6e}."
            ),
        )

    @pytest.mark.parametrize("strategy", ['cls', 'first'])
    def test_cls_and_first_return_index_zero_regardless_of_mask(
        self, strategy: str
    ) -> None:
        """INTENT pin, not an isolation test.

        `cls`/`first` mean "the token at index 0". Masking position 0 does not
        make that a defect — the caller asked for index 0 — so this asserts the
        contract rather than isolation. It goes RED if anyone makes the index
        mask-aware, which is its dead-component probe.
        """
        layer = _iso_layer(strategy, seq_len=F25_S)
        x = _iso_inputs(seed=47, seq_len=F25_S)
        # Position 0 masked in row 0, an interior mask in row 1.
        keep = _iso_mask([[0], [0, 2, 4]], seq_len=F25_S)

        out = _np(layer(
            keras.ops.convert_to_tensor(x),
            mask=keras.ops.convert_to_tensor(keep),
            training=False,
        ))
        np.testing.assert_allclose(
            out, x[:, 0, :], rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}: must return index 0 regardless of the "
                f"mask, but the output differs from `x[:, 0, :]` by "
                f"{float(np.max(np.abs(out - x[:, 0, :]))):.6e}. The nearest "
                f"mask-aware answer (first KEPT index) is index 1 for row 0."
            ),
        )

    @pytest.mark.parametrize("strategy", ['cls', 'first', 'last', 'middle'])
    @pytest.mark.parametrize("seq_len", [3, 6, 7])
    def test_mask_none_is_unchanged_and_matches_an_all_ones_mask(
        self, strategy: str, seq_len: int
    ) -> None:
        """I-A's per-mode companion: the fix must be `mask=None`-NEUTRAL.

        `middle` at an all-keep mask must still land on `seq_len // 2` exactly
        (kept-middle == padded midpoint when nothing is masked), and `last` on
        `seq_len - 1`.
        """
        layer = _iso_layer(strategy, seq_len=seq_len)
        x = keras.ops.convert_to_tensor(_iso_inputs(seed=53, seq_len=seq_len))
        x_np = _np(x)
        expected_idx = {
            'cls': 0, 'first': 0, 'last': seq_len - 1, 'middle': seq_len // 2,
        }[strategy]

        no_mask = _np(layer(x, mask=None, training=False))
        np.testing.assert_allclose(
            no_mask, x_np[:, expected_idx, :], rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}, seq_len={seq_len}: the `mask=None` "
                f"answer moved off index {expected_idx}."
            ),
        )

        ones = _np(layer(
            x, mask=keras.ops.ones((ISO_B, seq_len)), training=False
        ))
        np.testing.assert_allclose(
            ones, no_mask, rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}, seq_len={seq_len}: an all-ones mask "
                f"disagrees with `mask=None` by "
                f"{float(np.max(np.abs(ones - no_mask))):.6e}."
            ),
        )

    @pytest.mark.parametrize("strategy", ['last', 'middle'])
    def test_fully_masked_row_degenerates_to_index_zero(self, strategy: str) -> None:
        """A fully-masked row does NOT rescue and does NOT raise — it returns index 0.

        Both mask-aware index expressions clamp an empty candidate set to 0 via
        `ops.maximum(ops.max(cand, axis=1), 0)`, so the returned token is itself
        masked. That is DOCUMENTED DEGENERATION, matching the in-package
        no-rescue precedent (D-008 of the F-24 plan) and matching `last`'s
        pre-existing `ops.maximum(seq_lens, 0)` clamp — not a rescue.
        """
        layer = _iso_layer(strategy, seq_len=F25_S)
        x = _iso_inputs(seed=59, seq_len=F25_S)
        # Row 0 keeps a prefix of 2; row 1 is fully masked.
        keep = _iso_mask([[2, 3, 4, 5], list(range(F25_S))], seq_len=F25_S)

        out = _np(layer(
            keras.ops.convert_to_tensor(x),
            mask=keras.ops.convert_to_tensor(keep),
            training=False,
        ))
        assert np.all(np.isfinite(out)), (
            f"strategy={strategy!r}: a fully-masked row produced non-finite "
            f"output {out[1]}."
        )
        np.testing.assert_allclose(
            out[1], x[1, 0, :], rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}: a fully-masked row must degenerate to "
                f"index 0; it differs by "
                f"{float(np.max(np.abs(out[1] - x[1, 0, :]))):.6e}."
            ),
        )


class TestUnmaskedNumericsAreFrozen:
    """I1 / SC-2: no F-24 fix may move the `mask=None` output of ANY strategy.

    Scoped to `exclude_positions=None`; `TestExcludePositionsIsAlsoIsolated`
    covers the other entry point. See the note above `I1_UNMASKED_GOLDEN`.
    """

    @pytest.mark.parametrize("strategy", ISO_ALL_STRATEGIES)
    def test_unmasked_output_matches_the_pre_fix_reference(
        self, strategy: str, golden_reference_device: str
    ) -> None:
        # The stored triples are a CPU capture, so the layer must be BUILT and
        # RUN inside the pin. The comparison below is deliberately OUTSIDE it:
        # `out` is already a host numpy array by then, and the four-variant
        # measurement in the note above the dict shows the extra wrap changes
        # nothing (0 FAIL either way, 1 FAIL either way without this one).
        with keras.device(golden_reference_device):
            layer = _iso_layer(strategy, top_k=10, seq_len=I1_S)
            x = keras.ops.convert_to_tensor(_iso_ref_inputs())
            out = _np(layer(x, mask=None, training=False)).astype('float64')

        expected_sum, expected_abs, expected_first = I1_UNMASKED_GOLDEN[strategy]
        np.testing.assert_allclose(out.sum(), expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            np.abs(out).sum(), expected_abs, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            out.ravel()[0], expected_first, rtol=1e-6, atol=1e-6
        )


class TestExcludePositionsIsAlsoIsolated:
    """`exclude_positions` is the SECOND entry point into the D-002 fix.

    `_apply_mask_and_exclusions` synthesises an all-ones mask whenever
    `exclude_positions` is non-empty, so the three formerly-leaky strategies
    reach the masked code path even when the caller passes `mask=None`. Before
    D-002 this path leaked exactly as the explicit-mask path did:
    `top_k_mean`/`top_k_max` ignored the exclusion outright (their output was
    bit-identical to the no-exclusion output), and `weighted` merely reweighted
    a still-live position.

    This suite is deliberately NOT device-pinned: it compares two outputs
    produced on the SAME device, so there is no cross-device reference to
    protect.
    """

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_excluded_position_is_isolated_at_mask_none(self, strategy: str) -> None:
        layer = _iso_layer(
            strategy, top_k=10, exclude_positions=[ISO_MASKED_POS]
        )
        x = _iso_inputs()
        base = _np(layer(keras.ops.convert_to_tensor(x), mask=None, training=False))
        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_MASKED_POS])),
            mask=None,
            training=False,
        ))

        # Live control FIRST: a probe that cannot see any movement is vacuous.
        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_LIVE_POS])),
            mask=None,
            training=False,
        ))
        assert np.abs(live - base).max() > 1e-2, (
            f"{strategy!r}: perturbing a NON-excluded position moved the output "
            f"by only {np.abs(live - base).max():.6e}; the probe is vacuous."
        )

        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}, exclude_positions=[{ISO_MASKED_POS}], "
                f"mask=None: the EXCLUDED position leaked into the pooled output "
                f"by {np.abs(leaked - base).max():.6e}; required exactly 0.0."
            ),
        )

    @pytest.mark.parametrize("strategy", ISO_LEAKY_STRATEGIES)
    def test_exclusion_and_an_explicit_mask_compose(self, strategy: str) -> None:
        """`exclude_positions` AND a caller mask, together, in ONE call.

        The two entry points MULTIPLY in `_apply_mask_and_exclusions` (the
        caller's mask replaces the synthesised all-ones one and is then scaled
        by the exclusion), so neither cell above exercises the composition:
        `test_excluded_position_is_isolated_at_mask_none` covers exclusion
        alone and `TestMaskedPositionIsolation` covers the mask alone. This is
        the configuration `models/vit` would hit if it ever passed a mask
        (`model.py:442` builds the pool with `exclude_positions=[0]`).

        Both suppressed positions are perturbed in the SAME call, so a fix that
        honoured only one of the two entry points would leak here.
        """
        mask_off = ISO_MASKED_POS - 1        # suppressed by the caller's mask
        excluded = ISO_MASKED_POS            # suppressed by `exclude_positions`
        assert mask_off != ISO_LIVE_POS and excluded != ISO_LIVE_POS

        layer = _iso_layer(strategy, top_k=10, exclude_positions=[excluded])
        x = _iso_inputs()
        mask = keras.ops.convert_to_tensor(_iso_mask([[mask_off]] * ISO_B))

        base = _np(layer(keras.ops.convert_to_tensor(x), mask=mask, training=False))
        leaked = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [mask_off, excluded])),
            mask=mask,
            training=False,
        ))

        # Live control FIRST: a probe that cannot see any movement is vacuous.
        live = _np(layer(
            keras.ops.convert_to_tensor(_iso_perturb(x, [ISO_LIVE_POS])),
            mask=mask,
            training=False,
        ))
        assert np.abs(live - base).max() > 1e-2, (
            f"{strategy!r}: perturbing a position that is neither masked nor "
            f"excluded moved the output by only {np.abs(live - base).max():.6e}; "
            f"the probe is vacuous."
        )

        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"strategy={strategy!r}, exclude_positions=[{excluded}] AND an "
                f"explicit mask suppressing position {mask_off}: perturbing both "
                f"moved the pooled output by {np.abs(leaked - base).max():.6e}; "
                f"required exactly 0.0."
            ),
        )


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])