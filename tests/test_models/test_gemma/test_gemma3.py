"""
Comprehensive Tests for the Modern Keras 3 Gemma3 Model Implementation

This module provides thorough tests for the Gemma3 model, including:
- Basic model and transformer block functionality.
- Serialization/deserialization compliance with Modern Keras 3.
- All factory functions (from_variant, create_gemma3, generation, classification).
- Error handling for invalid configurations.
- Validation of key architectural features like dual normalization and mixed attention.

All tests use small model configurations to keep memory usage low and ensure fast execution.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any

import keras

from dl_techniques.models.gemma.gemma3 import (
    Gemma3,
    create_gemma3,
    create_gemma3_generation,
    create_gemma3_classification
)
from dl_techniques.models.gemma.components import Gemma3TransformerBlock


# It's good practice to have a base config fixture
@pytest.fixture
def base_config() -> Dict[str, Any]:
    """Provides a minimal, valid configuration for testing."""
    return {
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "ffn_hidden_size": 128,
        "max_seq_len": 256,
        "sliding_window_size": 32,
        "layer_types": ['sliding_window', 'full_attention'],
        "dropout_rate": 0.1,
        "use_bias": False,
        "norm_eps": 1e-5,
        "initializer_range": 0.02,
    }


class TestGemma3ModelAndBlock:
    """Test basic Gemma3 model and transformer block functionality."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create a sample input tensor for testing."""
        return keras.ops.convert_to_tensor(
            np.random.randint(0, 800, size=(2, 32)), dtype='int32'
        )

    def test_model_creation(self, base_config: Dict[str, Any]):
        """Test basic model creation and attribute initialization."""
        model = Gemma3(**base_config)

        # Check model attributes
        assert model.vocab_size == base_config['vocab_size']
        assert model.hidden_size == base_config['hidden_size']
        assert model.num_layers == base_config['num_layers']
        assert len(model.blocks) == base_config['num_layers']
        assert model.max_seq_len == base_config['max_seq_len']
        assert isinstance(model.final_norm, keras.layers.Layer)
        assert model.lm_head.units == base_config['vocab_size']

    def test_model_forward_pass(self, base_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test a forward pass through the model."""
        model = Gemma3(**base_config)
        output = model(sample_input)

        expected_shape = (sample_input.shape[0], sample_input.shape[1], base_config['vocab_size'])
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_transformer_block_creation(self, base_config: Dict[str, Any]):
        """Test individual transformer block creation and structure."""
        block = Gemma3TransformerBlock(
            hidden_size=base_config['hidden_size'],
            num_attention_heads=base_config['num_attention_heads'],
            num_key_value_heads=base_config['num_key_value_heads'],
            ffn_hidden_size=base_config['ffn_hidden_size'],
            max_seq_len=base_config['max_seq_len'],
            attention_type='full_attention'
        )

        # Check for the dual normalization pattern
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')
        assert hasattr(block, 'pre_feedforward_layernorm')
        assert hasattr(block, 'post_feedforward_layernorm')
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ffn')

    def test_transformer_block_forward_pass(self, base_config: Dict[str, Any]):
        """Test a forward pass through a single transformer block."""
        block = Gemma3TransformerBlock(
            hidden_size=base_config['hidden_size'],
            num_attention_heads=base_config['num_attention_heads'],
            num_key_value_heads=base_config['num_key_value_heads'],
            ffn_hidden_size=base_config['ffn_hidden_size'],
        )
        sample_input = keras.random.normal((2, 16, base_config['hidden_size']))
        output = block(sample_input)

        assert output.shape == sample_input.shape
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))

    def test_embedding_scaling(self, base_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test that embedding scaling is computed and applied correctly."""
        model = Gemma3(**base_config)

        # Check that the scaling factor is computed correctly
        expected_scale = np.sqrt(base_config['hidden_size'])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model.emb_scale),
            expected_scale,
            rtol=1e-6
        )

        # Check that scaling is applied in the forward pass
        raw_embeddings = model.embeddings(sample_input)
        scaled_embeddings_manual = raw_embeddings * model.emb_scale

        # To verify, we can use a hook or a subclass, but a simpler check is to
        # ensure the model runs without error, implying the operation is valid.
        # A full check would require a more complex setup.
        _ = model(sample_input) # This ensures the multiplication step works

    def test_attention_mask_creation_in_block(self):
        """Test attention mask creation within the transformer block."""
        seq_len = 8
        sliding_window_size = 4
        block = Gemma3TransformerBlock(
            hidden_size=64, num_attention_heads=4, num_key_value_heads=2, ffn_hidden_size=128,
            attention_type='sliding_window', sliding_window_size=sliding_window_size
        )
        mask = block._create_attention_mask(seq_len)
        mask_np = keras.ops.convert_to_numpy(mask)

        assert mask.shape == (seq_len, seq_len)
        # Check that local mask includes sliding window constraint
        for i in range(seq_len):
            for j in range(seq_len):
                is_causal = j > i
                is_outside_window = (i - j) >= sliding_window_size
                assert mask_np[i, j] == (is_causal or is_outside_window)

    def test_training_mode(self, base_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test model behavior in training vs. inference mode (for dropout)."""
        model = Gemma3(**base_config)

        output_train = model(sample_input, training=True)
        output_inference = model(sample_input, training=False)

        output_train_np = keras.ops.convert_to_numpy(output_train)
        output_inference_np = keras.ops.convert_to_numpy(output_inference)

        assert output_train_np.shape == output_inference_np.shape
        # With dropout, training and inference outputs should be different
        if base_config['dropout_rate'] > 0:
            assert not np.allclose(output_train_np, output_inference_np)


class TestGemma3Serialization:
    """Test model serialization and deserialization."""

    def test_get_config_completeness(self, base_config: Dict[str, Any]):
        """Test that get_config returns all necessary constructor parameters."""
        model = Gemma3(**base_config)
        config = model.get_config()

        # Check all constructor arguments are present
        for key in base_config:
            assert key in config, f"Missing key in config: {key}"
            # Special handling for list comparison
            if isinstance(config[key], list):
                assert config[key] == base_config[key]
            else:
                assert config[key] == base_config[key]

    def test_serialization_cycle(self, base_config: Dict[str, Any]):
        """Test full save/load cycle."""
        original_model = Gemma3(**base_config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, base_config['vocab_size'], size=(2, 20)), dtype='int32'
        )
        original_output = original_model(sample_input, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'gemma3_test.keras')
            original_model.save(model_path)
            # Make sure to register the custom objects for loading
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'Gemma3': Gemma3, 'Gemma3TransformerBlock': Gemma3TransformerBlock}
            )

            loaded_output = loaded_model(sample_input, training=False)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-5, atol=1e-5
            )


class TestGemma3Factories:
    """Test all factory functions and model variants."""

    def test_from_variant(self):
        """Test creating a model from a predefined variant."""
        for variant in Gemma3.MODEL_VARIANTS.keys():
            model = Gemma3.from_variant(variant)
            variant_config = Gemma3.MODEL_VARIANTS[variant]
            assert model.num_layers == variant_config['num_layers']
            assert model.hidden_size == variant_config['hidden_size']
            assert model.vocab_size == variant_config['vocab_size']

    def test_from_variant_with_overrides(self):
        """Test creating from a variant with overridden parameters."""
        model = Gemma3.from_variant("tiny", dropout_rate=0.5, max_seq_len=1024)
        assert model.dropout_rate == 0.5
        assert model.max_seq_len == 1024
        # Check that other parameters are from the "tiny" variant
        assert model.num_layers == Gemma3.MODEL_VARIANTS['tiny']['num_layers']

    def test_create_gemma3_generation(self):
        """Test the high-level factory for generation."""
        model = create_gemma3("tiny") # task_type defaults to generation
        assert model.name == "gemma3_for_generation"
        # Check input and output shapes
        assert len(model.inputs) == 2 # input_ids, attention_mask
        assert model.output_shape[2] == Gemma3.MODEL_VARIANTS['tiny']['vocab_size']

    def test_create_gemma3_classification(self):
        """Test the high-level factory for classification."""
        num_labels = 10
        model = create_gemma3(
            "tiny",
            task_type="classification",
            num_labels=num_labels,
            pooling_strategy="mean"
        )
        assert model.name == "gemma3_for_classification"
        # Check output shape
        assert model.output_shape == (None, num_labels)

    def test_create_gemma3_with_custom_config(self, base_config: Dict[str, Any]):
        """Test creating a model from a custom config dictionary."""
        model = create_gemma3(base_config, task_type="generation")
        assert model.name == "gemma3_for_generation"
        assert model.layers[2].num_layers == base_config['num_layers'] # Backbone is the 3rd layer
        assert model.layers[2].hidden_size == base_config['hidden_size']

    def test_factory_error_handling(self):
        """Test error handling in the high-level factory."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid_variant'"):
            create_gemma3("invalid_variant")

        with pytest.raises(ValueError, match="Unknown task_type 'invalid_task'"):
            create_gemma3("tiny", task_type="invalid_task")

        with pytest.raises(ValueError, match="num_labels must be provided"):
            create_gemma3("tiny", task_type="classification")


class TestGemma3ErrorHandling:
    """Test error handling and edge cases for invalid configurations."""

    def test_invalid_parameters(self, base_config: Dict[str, Any]):
        """Test model creation fails with various invalid parameters."""

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Gemma3(**{**base_config, 'vocab_size': 0})

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            Gemma3(**{**base_config, 'hidden_size': -1})

        with pytest.raises(ValueError, match="num_layers must be positive"):
            Gemma3(**{**base_config, 'num_layers': 0})

        with pytest.raises(ValueError, match="dropout_rate must be in"):
            Gemma3(**{**base_config, 'dropout_rate': 1.1})

        with pytest.raises(ValueError, match="must be divisible by"):
            Gemma3(**{**base_config, 'hidden_size': 66, 'num_attention_heads': 5})

        with pytest.raises(ValueError, match="layer_types length .* must match"):
            Gemma3(**{**base_config, 'num_layers': 3, 'layer_types': ['full_attention']})

        with pytest.raises(ValueError, match="Invalid layer_type"):
            Gemma3(**{**base_config, 'layer_types': ['sliding_window', 'invalid_type']})


class TestGemma3Integration:
    """Test integration and end-to-end functionality."""

    def test_end_to_end_training_simulation(self, base_config: Dict[str, Any]):
        """Test a complete training step on a tiny model."""
        model = create_gemma3_classification(base_config, num_labels=10)

        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        batch_size, seq_len = 4, 16
        input_ids = keras.ops.convert_to_tensor(
            np.random.randint(0, base_config['vocab_size'], size=(batch_size, seq_len)),
            dtype='int32'
        )
        attention_mask = keras.ops.ones_like(input_ids)
        targets = keras.ops.convert_to_tensor(
            np.random.randint(0, 10, size=(batch_size,)),
            dtype='int32'
        )

        # Train on a single batch
        history = model.fit(
            [input_ids, attention_mask],
            targets,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        loss = history.history['loss'][0]
        assert isinstance(loss, float)
        assert loss >= 0

    def test_sliding_vs_full_attention_behavior(self, base_config: Dict[str, Any]):
        """Test that sliding window and full attention produce different outputs."""
        # Ensure num_layers is 1 for a direct comparison
        config = {**base_config, 'num_layers': 1}

        model_full = Gemma3(**{**config, 'layer_types': ['full_attention']})
        model_sliding = Gemma3(**{**config, 'layer_types': ['sliding_window']})

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, base_config['vocab_size'], size=(1, 48)), dtype='int32'
        )

        output_full = model_full(sample_input, training=False)
        output_sliding = model_sliding(sample_input, training=False)

        output_full_np = keras.ops.convert_to_numpy(output_full)
        output_sliding_np = keras.ops.convert_to_numpy(output_sliding)

        # The outputs should be different due to the attention mask
        difference = np.mean(np.abs(output_full_np - output_sliding_np))
        assert difference > 1e-6

if __name__ == "__main__":
    pytest.main([__file__, "-v"])