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

# SC-2 / I1 pin: `mask=None` outputs for all 18 strategies, captured from the
# code BEFORE the F-24 fix with the fixture below (`_iso_layer(..., top_k=10)`,
# `_iso_ref_inputs()`), as ``(sum, sum_of_abs, first_element)`` in float64.
# A fix that moves ANY unmasked numeric moves at least the sum-of-abs.
# Tolerance is 1e-6 relative rather than 0: this module is not device-pinned and
# GPU reduction order makes bitwise equality unavailable across devices, while a
# real regression of this defect's shape is O(1e-1), five orders of magnitude up.
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
    strategy: str, top_k: int = 10, seed: int = 1234, seq_len: int = ISO_S
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


class TestUnmaskedNumericsAreFrozen:
    """I1 / SC-2: no F-24 fix may move the `mask=None` output of ANY strategy."""

    @pytest.mark.parametrize("strategy", ISO_ALL_STRATEGIES)
    def test_unmasked_output_matches_the_pre_fix_reference(self, strategy: str) -> None:
        layer = _iso_layer(strategy, top_k=10, seq_len=I1_S)
        x = keras.ops.convert_to_tensor(_iso_ref_inputs())
        out = _np(layer(x, mask=None, training=False)).astype('float64')

        expected_sum, expected_abs, expected_first = I1_UNMASKED_GOLDEN[strategy]
        np.testing.assert_allclose(out.sum(), expected_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.abs(out).sum(), expected_abs, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.ravel()[0], expected_first, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])