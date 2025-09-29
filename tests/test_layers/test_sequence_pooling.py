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
from typing import Dict, Any

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


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])