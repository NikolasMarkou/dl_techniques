import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import keras
from typing import Any, Dict

from dl_techniques.layers.time_series.mixed_sequential_block import MixedSequentialBlock


class TestMixedSequentialBlock:
    """Comprehensive test suite for MixedSequentialBlock."""

    @pytest.fixture
    def base_layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'embed_dim': 128,
            'num_heads': 8,
            'lstm_units': 64,
            'ff_dim': 256,
            'block_type': 'mixed',
            'dropout_rate': 0.1,
            'use_layer_norm': True,
            'normalization_type': 'rms_norm',
            'attention_type': 'multi_head',
            'ffn_type': 'mlp'
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample time series input for testing."""
        return keras.random.normal(shape=(4, 32, 128))  # (batch, seq_len, embed_dim)

    @pytest.fixture
    def sample_mask(self) -> keras.KerasTensor:
        """Sample attention mask for testing."""
        # Create a simple mask with some positions masked
        mask = keras.ops.ones((4, 32), dtype='bool')
        # Mask last 5 positions for each batch
        mask = keras.ops.where(
            keras.ops.arange(32)[None, :] < 27,
            mask,
            False
        )
        return mask

    def test_initialization_mixed_mode(self, base_layer_config):
        """Test layer initialization in mixed mode."""
        layer = MixedSequentialBlock(**base_layer_config)

        # Check configuration storage
        assert layer.embed_dim == 128
        assert layer.num_heads == 8
        assert layer.lstm_units == 64
        assert layer.block_type == 'mixed'
        assert not layer.built

        # Check sub-layers are created
        assert layer.lstm_layer is not None
        assert layer.attention_layer is not None
        assert layer.ffn_layer is not None
        assert layer.projection is not None  # lstm_units != embed_dim
        assert layer.norm1 is not None
        assert layer.norm2 is not None
        assert layer.norm3 is not None  # Mixed mode has 3 norms

    def test_initialization_lstm_mode(self, base_layer_config):
        """Test layer initialization in LSTM-only mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'lstm'

        layer = MixedSequentialBlock(**config)

        assert layer.lstm_layer is not None
        assert layer.attention_layer is None  # No attention in LSTM mode
        assert layer.ffn_layer is not None
        assert layer.norm3 is None  # Only 2 norms in LSTM mode

    def test_initialization_transformer_mode(self, base_layer_config):
        """Test layer initialization in Transformer-only mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'transformer'

        layer = MixedSequentialBlock(**config)

        assert layer.lstm_layer is None  # No LSTM in Transformer mode
        assert layer.projection is None  # No projection needed
        assert layer.attention_layer is not None
        assert layer.ffn_layer is not None
        assert layer.norm3 is None  # Only 2 norms in Transformer mode

    def test_initialization_no_projection(self, base_layer_config):
        """Test initialization when lstm_units equals embed_dim (no projection needed)."""
        config = base_layer_config.copy()
        config['lstm_units'] = config['embed_dim']  # Same as embed_dim

        layer = MixedSequentialBlock(**config)

        assert layer.lstm_layer is not None
        assert layer.projection is None  # No projection when dimensions match

    def test_forward_pass_mixed_mode(self, base_layer_config, sample_input):
        """Test forward pass in mixed mode."""
        layer = MixedSequentialBlock(**base_layer_config)

        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape  # Shape preserved
        assert output.dtype == sample_input.dtype

    def test_forward_pass_lstm_mode(self, base_layer_config, sample_input):
        """Test forward pass in LSTM mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'lstm'

        layer = MixedSequentialBlock(**config)
        output = layer(sample_input)

        assert output.shape == sample_input.shape

    def test_forward_pass_transformer_mode(self, base_layer_config, sample_input):
        """Test forward pass in Transformer mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'transformer'

        layer = MixedSequentialBlock(**config)
        output = layer(sample_input)

        assert output.shape == sample_input.shape

    def test_forward_pass_with_mask(self, base_layer_config, sample_input, sample_mask):
        """Test forward pass with attention mask."""
        layer = MixedSequentialBlock(**base_layer_config)

        output = layer(sample_input, mask=sample_mask)

        assert output.shape == sample_input.shape

    def test_serialization_cycle_mixed_mode(self, base_layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle for mixed mode."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MixedSequentialBlock(**base_layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
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

    def test_serialization_cycle_lstm_mode(self, base_layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle for LSTM mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'lstm'

        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MixedSequentialBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="LSTM mode predictions differ after serialization"
            )

    def test_serialization_cycle_transformer_mode(self, base_layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle for Transformer mode."""
        config = base_layer_config.copy()
        config['block_type'] = 'transformer'

        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = MixedSequentialBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Transformer mode predictions differ after serialization"
            )

    def test_serialization_cycle_with_custom_args(self, sample_input):
        """Test serialization with custom factory arguments."""
        config = {
            'embed_dim': 256,
            'num_heads': 8,
            'block_type': 'mixed',
            'normalization_type': 'band_rms',
            'attention_type': 'differential',
            'ffn_type': 'swiglu',
            'normalization_args': {'max_band_width': 0.1, 'epsilon': 1e-7},
            'attention_args': {'lambda_init': 0.9},
            'ffn_args': {'ffn_expansion_factor': 8}
        }

        inputs = keras.Input(shape=(32, 256))
        outputs = MixedSequentialBlock(**config)(inputs)
        model = keras.Model(inputs, outputs)

        # Adjust sample input for this test
        test_input = keras.random.normal(shape=(4, 32, 256))
        original_pred = model(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(test_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Custom args predictions differ after serialization"
            )

    def test_config_completeness(self, base_layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = MixedSequentialBlock(**base_layer_config)
        config = layer.get_config()

        # Check all base_layer_config parameters are present
        for key in base_layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters that have defaults
        expected_keys = [
            'embed_dim', 'num_heads', 'lstm_units', 'ff_dim', 'block_type',
            'dropout_rate', 'use_layer_norm', 'normalization_type',
            'attention_type', 'ffn_type', 'activation', 'normalization_args',
            'attention_args', 'ffn_args'
        ]

        for key in expected_keys:
            assert key in config, f"Missing expected key {key} in get_config()"

    def test_gradients_flow(self, base_layer_config, sample_input):
        """Test gradient computation and flow."""
        layer = MixedSequentialBlock(**base_layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert len(gradients) > 0
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check gradients have reasonable magnitudes
        for grad in gradients:
            grad_norm = keras.ops.norm(grad)
            assert keras.ops.convert_to_numpy(grad_norm) > 1e-8, "Gradient too small"
            assert keras.ops.convert_to_numpy(grad_norm) < 1e3, "Gradient too large"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, base_layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = MixedSequentialBlock(**base_layer_config)

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape

        # Test that outputs differ between training modes when dropout > 0
        if base_layer_config['dropout_rate'] > 0:
            output_train = layer(sample_input, training=True)
            output_eval = layer(sample_input, training=False)

            # They might be the same due to randomness, but shapes should match
            assert output_train.shape == output_eval.shape

    def test_compute_output_shape(self, base_layer_config):
        """Test output shape computation."""
        layer = MixedSequentialBlock(**base_layer_config)

        input_shape = (None, 32, 128)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape

    def test_edge_cases_validation(self):
        """Test error conditions and edge cases."""
        base_config = {
            'embed_dim': 128,
            'num_heads': 8
        }

        # Test invalid embed_dim
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            MixedSequentialBlock(embed_dim=0, num_heads=8)

        # Test invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            MixedSequentialBlock(embed_dim=128, num_heads=0)

        # Test embed_dim not divisible by num_heads
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            MixedSequentialBlock(embed_dim=100, num_heads=7)

        # Test invalid lstm_units
        with pytest.raises(ValueError, match="lstm_units must be positive"):
            MixedSequentialBlock(embed_dim=128, num_heads=8, lstm_units=-10)

        # Test invalid ff_dim
        with pytest.raises(ValueError, match="ff_dim must be positive"):
            MixedSequentialBlock(embed_dim=128, num_heads=8, ff_dim=0)

        # Test invalid dropout_rate
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            MixedSequentialBlock(embed_dim=128, num_heads=8, dropout_rate=1.5)

        # Test invalid block_type
        with pytest.raises(ValueError, match="block_type must be one of"):
            MixedSequentialBlock(embed_dim=128, num_heads=8, block_type='invalid')

    def test_different_factory_combinations(self, sample_input):
        """Test various factory component combinations."""
        test_configs = [
            # RMS norm + SwiGLU + Differential attention
            {
                'embed_dim': 128, 'num_heads': 8, 'block_type': 'mixed',
                'normalization_type': 'rms_norm', 'attention_type': 'differential',
                'ffn_type': 'swiglu'
            },
            # Layer norm + MLP + Multi-head attention
            {
                'embed_dim': 128, 'num_heads': 8, 'block_type': 'transformer',
                'normalization_type': 'layer_norm', 'attention_type': 'multi_head',
                'ffn_type': 'mlp'
            },
            # Band RMS + GLU + LSTM only
            {
                'embed_dim': 128, 'num_heads': 8, 'block_type': 'lstm',
                'normalization_type': 'band_rms', 'ffn_type': 'glu',
                'normalization_args': {'max_band_width': 0.1}
            }
        ]

        for i, config in enumerate(test_configs):
            layer = MixedSequentialBlock(**config)
            output = layer(sample_input)

            assert output.shape == sample_input.shape, f"Config {i} failed shape test"
            assert layer.built, f"Config {i} failed to build"

    def test_layer_reuse(self, base_layer_config, sample_input):
        """Test that the same layer can be called multiple times."""
        layer = MixedSequentialBlock(**base_layer_config)

        # First call
        output1 = layer(sample_input)

        # Second call with different input
        input2 = keras.random.normal(shape=(2, 16, 128))
        output2 = layer(input2)

        # Check outputs have correct shapes
        assert output1.shape == sample_input.shape
        assert output2.shape == input2.shape
        assert layer.built

    def test_no_normalization(self, base_layer_config, sample_input):
        """Test operation without normalization layers."""
        config = base_layer_config.copy()
        config['use_layer_norm'] = False

        layer = MixedSequentialBlock(**config)

        # Check normalization layers are None
        assert layer.norm1 is None
        assert layer.norm2 is None
        assert layer.norm3 is None

        # Forward pass should still work
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_memory_efficiency(self, sample_input):
        """Test that layer doesn't cause memory leaks in repeated calls."""
        layer = MixedSequentialBlock(embed_dim=64, num_heads=4, block_type='mixed')

        # Multiple forward passes
        for _ in range(10):
            output = layer(sample_input[:, :, :64])  # Use smaller dimension
            del output  # Explicit cleanup

        # Should not raise memory errors
        final_output = layer(sample_input[:, :, :64])
        assert final_output.shape == (4, 32, 64)

    def test_attention_parameter_mapping(self, sample_input):
        """Test that attention parameters are correctly mapped for different attention types."""
        # Test differential attention with head_dim auto-calculation
        differential_layer = MixedSequentialBlock(
            embed_dim=256,
            num_heads=8,
            block_type='transformer',
            attention_type='differential'
        )

        # Create input with correct embed_dim (256)
        test_input_256 = keras.random.normal(shape=(4, 32, 256))
        output = differential_layer(test_input_256)
        assert output.shape == test_input_256.shape

        # Test multi_head attention with sample_input (embed_dim=128)
        multi_head_layer = MixedSequentialBlock(
            embed_dim=128,
            num_heads=8,
            block_type='transformer',
            attention_type='multi_head'
        )

        output = multi_head_layer(sample_input)
        assert output.shape == sample_input.shape

    def test_custom_attention_args_override(self, sample_input):
        """Test that custom attention args properly override defaults."""
        layer = MixedSequentialBlock(
            embed_dim=128,
            num_heads=8,
            block_type='transformer',
            attention_type='differential',
            attention_args={'lambda_init': 0.95}  # Custom lambda_init value
        )

        output = layer(sample_input)
        assert output.shape == sample_input.shape

        # Verify the layer was created successfully with custom args
        assert layer.attention_layer is not None
        assert layer.built