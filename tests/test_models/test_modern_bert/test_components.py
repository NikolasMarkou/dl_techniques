import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

# Import the components to test
from dl_techniques.models.modern_bert.components import (
    ModernBertEmbeddings,
    ModernBertAttention,
    ModernBertTransformerLayer
)


class TestModernBertEmbeddings:
    """Comprehensive test suite for ModernBertEmbeddings layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'vocab_size': 1000,
            'hidden_size': 768,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'use_bias': False
        }

    @pytest.fixture
    def sample_input(self):
        """Sample input for testing."""
        batch_size, seq_len = 2, 128
        # Use keras.random.uniform with float32 then cast to int32
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_len), 0, 1000), 'int32'
        )
        token_type_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_len), 0, 2), 'int32'
        )
        return input_ids, token_type_ids

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = ModernBertEmbeddings(**layer_config)

        assert hasattr(layer, 'vocab_size')
        assert hasattr(layer, 'hidden_size')
        assert hasattr(layer, 'word_embeddings')
        assert hasattr(layer, 'token_type_embeddings')
        assert hasattr(layer, 'layer_norm')
        assert hasattr(layer, 'dropout')
        assert not layer.built

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = ModernBertEmbeddings(**layer_config)
        input_ids, token_type_ids = sample_input

        # Test forward pass
        output = layer(input_ids, token_type_ids)

        assert layer.built
        assert output.shape == (2, 128, 768)  # batch_size, seq_len, hidden_size

        # Test without token_type_ids (should default to zeros)
        output_no_types = layer(input_ids)
        assert output_no_types.shape == (2, 128, 768)

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        input_ids, token_type_ids = sample_input

        # Create model with custom layer
        inputs_ids = keras.Input(shape=(128,), dtype='int32', name='input_ids')
        inputs_types = keras.Input(shape=(128,), dtype='int32', name='token_type_ids')
        outputs = ModernBertEmbeddings(**layer_config)(inputs_ids, inputs_types)
        model = keras.Model([inputs_ids, inputs_types], outputs)

        # Get original prediction
        original_pred = model([input_ids, token_type_ids])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([input_ids, token_type_ids])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = ModernBertEmbeddings(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == layer_config[key], f"Config mismatch for {key}"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = ModernBertEmbeddings(**layer_config)
        input_ids, token_type_ids = sample_input

        with tf.GradientTape() as tape:
            tape.watch(input_ids)
            output = layer(input_ids, token_type_ids)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = ModernBertEmbeddings(**layer_config)
        input_ids, token_type_ids = sample_input

        output = layer(input_ids, token_type_ids, training=training)
        assert output.shape == (2, 128, 768)

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ModernBertEmbeddings(vocab_size=0, hidden_size=768)

        # Invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModernBertEmbeddings(vocab_size=1000, hidden_size=-1)

        # Invalid dropout rate
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between 0 and 1"):
            ModernBertEmbeddings(vocab_size=1000, hidden_size=768, hidden_dropout_prob=1.5)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = ModernBertEmbeddings(**layer_config)
        input_shape = (None, 128)  # batch_size, seq_len
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 128, 768)  # batch_size, seq_len, hidden_size


class TestModernBertAttention:
    """Comprehensive test suite for ModernBertAttention layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_size': 768,
            'num_heads': 12,
            'attention_probs_dropout_prob': 0.1,
            'use_bias': False,
            'initializer_range': 0.02,
            'is_global': True,
            'max_seq_len': 512
        }

    @pytest.fixture
    def sample_input(self):
        """Sample input for testing."""
        batch_size, seq_len, hidden_size = 2, 128, 768
        hidden_states = keras.random.normal((batch_size, seq_len, hidden_size))
        # Generate boolean attention mask
        attention_mask = keras.ops.cast(
            keras.random.uniform((batch_size, seq_len), 0, 2), 'bool'
        )
        return hidden_states, attention_mask

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = ModernBertAttention(**layer_config)

        assert hasattr(layer, 'hidden_size')
        assert hasattr(layer, 'num_heads')
        assert hasattr(layer, 'head_dim')
        assert hasattr(layer, 'mha')
        assert hasattr(layer, 'rotary_embedding')
        assert layer.head_dim == 768 // 12
        assert not layer.built

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = ModernBertAttention(**layer_config)
        hidden_states, attention_mask = sample_input

        # Test forward pass with mask
        output = layer(hidden_states, attention_mask=attention_mask)

        assert layer.built
        assert output.shape == hidden_states.shape  # Same shape as input

        # Test without attention mask
        output_no_mask = layer(hidden_states)
        assert output_no_mask.shape == hidden_states.shape

    def test_global_vs_local_attention(self):
        """Test both global and local attention modes."""
        hidden_states = keras.random.normal((2, 64, 768))

        # Global attention
        global_layer = ModernBertAttention(
            hidden_size=768,
            num_heads=12,
            is_global=True,
            max_seq_len=512
        )
        global_output = global_layer(hidden_states)
        assert global_output.shape == hidden_states.shape

        # Local attention with window
        local_layer = ModernBertAttention(
            hidden_size=768,
            num_heads=12,
            is_global=False,
            local_attention_window_size=32,
            max_seq_len=512
        )
        local_output = local_layer(hidden_states)
        assert local_output.shape == hidden_states.shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        hidden_states, attention_mask = sample_input

        # Create model with custom layer
        inputs = keras.Input(shape=(128, 768))
        mask_input = keras.Input(shape=(128,), dtype='bool')
        outputs = ModernBertAttention(**layer_config)(inputs, attention_mask=mask_input)
        model = keras.Model([inputs, mask_input], outputs)

        # Get original prediction
        original_pred = model([hidden_states, attention_mask])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([hidden_states, attention_mask])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = ModernBertAttention(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == layer_config[key], f"Config mismatch for {key}"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = ModernBertAttention(**layer_config)
        hidden_states, attention_mask = sample_input

        with tf.GradientTape() as tape:
            output = layer(hidden_states, attention_mask=attention_mask)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = ModernBertAttention(**layer_config)
        hidden_states, attention_mask = sample_input

        output = layer(hidden_states, attention_mask=attention_mask, training=training)
        assert output.shape == hidden_states.shape

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModernBertAttention(hidden_size=0, num_heads=12)

        # Invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            ModernBertAttention(hidden_size=768, num_heads=0)

        # Non-divisible hidden_size
        with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_heads"):
            ModernBertAttention(hidden_size=100, num_heads=3)

        # Invalid dropout rate
        with pytest.raises(ValueError, match="attention_probs_dropout_prob must be between 0 and 1"):
            ModernBertAttention(hidden_size=768, num_heads=12, attention_probs_dropout_prob=1.5)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = ModernBertAttention(**layer_config)
        input_shape = (None, 128, 768)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape  # Same shape as input


class TestModernBertTransformerLayer:
    """Comprehensive test suite for ModernBertTransformerLayer layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'use_bias': False,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'is_global': True
        }

    @pytest.fixture
    def sample_input(self):
        """Sample input for testing."""
        batch_size, seq_len, hidden_size = 2, 128, 768
        hidden_states = keras.random.normal((batch_size, seq_len, hidden_size))
        # Generate boolean attention mask
        attention_mask = keras.ops.cast(
            keras.random.uniform((batch_size, seq_len), 0, 2), 'bool'
        )
        return hidden_states, attention_mask

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = ModernBertTransformerLayer(**layer_config)

        assert hasattr(layer, 'hidden_size')
        assert hasattr(layer, 'num_heads')
        assert hasattr(layer, 'attention')
        assert hasattr(layer, 'ffn')
        assert hasattr(layer, 'attention_norm')
        assert hasattr(layer, 'ffn_norm')
        assert not layer.built

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building."""
        layer = ModernBertTransformerLayer(**layer_config)
        hidden_states, attention_mask = sample_input

        # Test forward pass
        output = layer(hidden_states, attention_mask=attention_mask)

        assert layer.built
        assert output.shape == hidden_states.shape  # Same shape as input

        # Test without attention mask
        output_no_mask = layer(hidden_states)
        assert output_no_mask.shape == hidden_states.shape

    def test_residual_connections(self, layer_config, sample_input):
        """Test that residual connections are working."""
        layer = ModernBertTransformerLayer(**layer_config)
        hidden_states, attention_mask = sample_input

        # Run forward pass
        output = layer(hidden_states, attention_mask=attention_mask)

        # Output should be different from input (due to transformations)
        # but should maintain similar magnitude due to residual connections
        input_norm = keras.ops.norm(keras.ops.reshape(hidden_states, [-1]))
        output_norm = keras.ops.norm(keras.ops.reshape(output, [-1]))

        # They should be in similar magnitude ranges
        ratio = keras.ops.convert_to_numpy(output_norm / input_norm)
        assert 0.5 < ratio < 2.0, f"Norm ratio {ratio} is outside the expected range (0.5, 2.0)"

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        hidden_states, attention_mask = sample_input

        # Create model with custom layer
        inputs = keras.Input(shape=(128, 768))
        mask_input = keras.Input(shape=(128,), dtype='bool')
        outputs = ModernBertTransformerLayer(**layer_config)(inputs, attention_mask=mask_input)
        model = keras.Model([inputs, mask_input], outputs)

        # Get original prediction
        original_pred = model([hidden_states, attention_mask])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([hidden_states, attention_mask])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = ModernBertTransformerLayer(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == layer_config[key], f"Config mismatch for {key}"

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        layer = ModernBertTransformerLayer(**layer_config)
        hidden_states, attention_mask = sample_input

        with tf.GradientTape() as tape:
            output = layer(hidden_states, attention_mask=attention_mask)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = ModernBertTransformerLayer(**layer_config)
        hidden_states, attention_mask = sample_input

        output = layer(hidden_states, attention_mask=attention_mask, training=training)
        assert output.shape == hidden_states.shape

    def test_global_vs_local_attention(self):
        """Test both global and local attention modes."""
        hidden_states = keras.random.normal((2, 64, 768))

        # Global attention transformer
        global_config = {
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': 3072,
            'is_global': True
        }
        global_layer = ModernBertTransformerLayer(**global_config)
        global_output = global_layer(hidden_states)
        assert global_output.shape == hidden_states.shape

        # Local attention transformer
        local_config = {
            'hidden_size': 768,
            'num_heads': 12,
            'intermediate_size': 3072,
            'is_global': False,
            'local_attention_window_size': 32
        }
        local_layer = ModernBertTransformerLayer(**local_config)
        local_output = local_layer(hidden_states)
        assert local_output.shape == hidden_states.shape

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModernBertTransformerLayer(hidden_size=0, num_heads=12, intermediate_size=3072)

        # Invalid num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            ModernBertTransformerLayer(hidden_size=768, num_heads=0, intermediate_size=3072)

        # Non-divisible hidden_size
        with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_heads"):
            ModernBertTransformerLayer(hidden_size=100, num_heads=3, intermediate_size=3072)

        # Invalid intermediate_size
        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            ModernBertTransformerLayer(hidden_size=768, num_heads=12, intermediate_size=0)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = ModernBertTransformerLayer(**layer_config)
        input_shape = (None, 128, 768)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == input_shape  # Same shape as input


class TestIntegration:
    """Integration tests for combining multiple components."""

    def test_embeddings_to_attention_pipeline(self):
        """Test that embeddings output can be fed to attention layer."""
        # Create components
        embeddings = ModernBertEmbeddings(vocab_size=1000, hidden_size=768)
        attention = ModernBertAttention(hidden_size=768, num_heads=12)

        # Generate inputs with proper casting
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 128), 0, 1000), 'int32'
        )

        # Run pipeline
        embedded = embeddings(input_ids)
        attended = attention(embedded)

        assert embedded.shape == (2, 128, 768)
        assert attended.shape == (2, 128, 768)

    def test_full_transformer_pipeline(self):
        """Test the complete pipeline: embeddings -> transformer layers."""
        # Create components
        embeddings = ModernBertEmbeddings(vocab_size=1000, hidden_size=768)
        transformer1 = ModernBertTransformerLayer(
            hidden_size=768, num_heads=12, intermediate_size=3072, is_global=True
        )
        transformer2 = ModernBertTransformerLayer(
            hidden_size=768, num_heads=12, intermediate_size=3072, is_global=False
        )

        # Generate inputs with proper casting
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 64), 0, 1000), 'int32'
        )
        attention_mask = keras.ops.cast(
            keras.random.uniform((2, 64), 0, 2), 'bool'
        )

        # Run full pipeline
        embedded = embeddings(input_ids)
        layer1_out = transformer1(embedded, attention_mask=attention_mask)
        layer2_out = transformer2(layer1_out, attention_mask=attention_mask)

        assert embedded.shape == (2, 64, 768)
        assert layer1_out.shape == (2, 64, 768)
        assert layer2_out.shape == (2, 64, 768)

    def test_pipeline_serialization(self):
        """Test serialization of a model using multiple custom components."""
        # Build a simple model using all components
        input_ids = keras.Input(shape=(128,), dtype='int32')
        attention_mask = keras.Input(shape=(128,), dtype='bool')

        # Embeddings
        embedded = ModernBertEmbeddings(vocab_size=1000, hidden_size=768)(input_ids)

        # Transformer layers
        x = ModernBertTransformerLayer(
            hidden_size=768, num_heads=12, intermediate_size=3072
        )(embedded, attention_mask=attention_mask)

        # Simple output head
        output = keras.layers.Dense(2, activation='softmax')(x[:, 0, :])  # CLS token

        model = keras.Model([input_ids, attention_mask], output)

        # Test inputs with proper casting
        test_input_ids = keras.ops.cast(
            keras.random.uniform((2, 128), 0, 1000), 'int32'
        )
        test_attention_mask = keras.ops.cast(
            keras.random.uniform((2, 128), 0, 2), 'bool'
        )

        # Get prediction
        original_pred = model([test_input_ids, test_attention_mask])

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'full_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model([test_input_ids, test_attention_mask])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Full pipeline predictions differ after serialization"
            )


# Run tests with: pytest test_modern_bert_components.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])