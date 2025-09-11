"""
Comprehensive test suite for FNet model implementation.

This test suite validates the FNet model and its factory functions
with small memory footprint configurations suitable for CPU testing.
"""

import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.fnet.model import (
    FNet,
    create_fnet_for_classification,
    create_fnet_for_sequence_output,
    create_fnet,
)


class TestFNetModel:
    """Test suite for the main FNet model class."""

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Small configuration for CPU testing."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'intermediate_size': 64,
            'max_position_embeddings': 16,
            'hidden_dropout_prob': 0.1,
            'add_pooling_layer': True
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample token inputs for testing."""
        batch_size = 2
        seq_len = 16
        return {
            'input_ids': keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            'attention_mask': keras.ops.ones((batch_size, seq_len), dtype="int32"),
            'position_ids': keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)
        }

    def test_basic_model_creation(self, small_config):
        """Test basic FNet model creation."""
        model = FNet(**small_config)
        assert isinstance(model, keras.Model)
        assert model.vocab_size == 100
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert model.add_pooling_layer is True

    def test_model_forward_pass_dict_input(self, small_config, sample_inputs):
        """Test model forward pass with dictionary input."""
        model = FNet(**small_config)
        output = model(sample_inputs)

        # Should return tuple (sequence_output, pooled_output) since add_pooling_layer=True
        assert isinstance(output, tuple)
        assert len(output) == 2
        sequence_output, pooled_output = output

        # Check shapes
        assert sequence_output.shape == (2, 16, 32)  # (batch, seq_len, hidden_size)
        assert pooled_output.shape == (2, 32)        # (batch, hidden_size)

    def test_model_forward_pass_tensor_input(self, small_config):
        """Test model forward pass with tensor input."""
        model = FNet(**small_config)
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(2, 16))
        output = model(input_ids)

        assert isinstance(output, tuple)
        assert len(output) == 2
        sequence_output, pooled_output = output
        assert sequence_output.shape == (2, 16, 32)
        assert pooled_output.shape == (2, 32)

    def test_model_without_pooling(self, small_config, sample_inputs):
        """Test model without pooling layer."""
        config = small_config.copy()
        config['add_pooling_layer'] = False
        model = FNet(**config)

        output = model(sample_inputs)

        # Should return single tensor when no pooling
        assert not isinstance(output, tuple)
        assert output.shape == (2, 16, 32)

    def test_model_return_dict_format(self, small_config, sample_inputs):
        """Test model with return_dict=True."""
        model = FNet(**small_config)
        output = model(sample_inputs, return_dict=True)

        assert isinstance(output, dict)
        assert 'last_hidden_state' in output
        assert 'pooler_output' in output

        assert output['last_hidden_state'].shape == (2, 16, 32)
        assert output['pooler_output'].shape == (2, 32)

    def test_from_variant_method(self):
        """Test creating model from predefined variants."""
        # Test all available variants
        variants = ['base', 'large', 'small', 'tiny']

        for variant in variants:
            model = FNet.from_variant(variant, add_pooling_layer=False)
            assert isinstance(model, FNet)

            # Check variant-specific configurations
            expected_config = FNet.MODEL_VARIANTS[variant]
            assert model.hidden_size == expected_config['hidden_size']
            assert model.num_layers == expected_config['num_layers']

    def test_from_variant_with_overrides(self):
        """Test from_variant with configuration overrides."""
        model = FNet.from_variant(
            'tiny',
            add_pooling_layer=True,
            hidden_dropout_prob=0.2,
            normalization_type='rms_norm'
        )

        assert model.hidden_dropout_prob == 0.2
        assert model.normalization_type == 'rms_norm'
        assert model.add_pooling_layer is True

    def test_model_serialization(self, small_config, sample_inputs):
        """Test model serialization and loading."""
        model = FNet(**small_config)
        original_pred = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs)

            # Compare outputs
            if isinstance(original_pred, tuple):
                for orig, loaded in zip(original_pred, loaded_pred):
                    np.testing.assert_allclose(
                        keras.ops.convert_to_numpy(orig),
                        keras.ops.convert_to_numpy(loaded),
                        rtol=1e-6, atol=1e-6,
                        err_msg="Loaded model outputs should match original"
                    )
            else:
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_pred),
                    keras.ops.convert_to_numpy(loaded_pred),
                    rtol=1e-6, atol=1e-6,
                    err_msg="Loaded model outputs should match original"
                )

    def test_get_config_and_from_config(self, small_config):
        """Test configuration serialization."""
        model = FNet(**small_config)
        config = model.get_config()

        # Verify all expected keys are present
        expected_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'intermediate_size',
            'hidden_dropout_prob', 'max_position_embeddings', 'add_pooling_layer'
        }
        assert expected_keys.issubset(set(config.keys()))

        # Test reconstruction
        new_model = FNet.from_config(config)
        assert new_model.vocab_size == model.vocab_size
        assert new_model.hidden_size == model.hidden_size
        assert new_model.num_layers == model.num_layers

    def test_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            FNet.from_variant('invalid_variant')

    def test_invalid_input_dict(self, small_config):
        """Test error handling for invalid input dictionary."""
        model = FNet(**small_config)

        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            model({'attention_mask': keras.ops.ones((2, 16))})

    def test_gradients_flow(self, small_config, sample_inputs):
        """Test that gradients flow through the model."""
        model = FNet(**small_config)

        with tf.GradientTape() as tape:
            output = model(sample_inputs)
            # Calculate loss based on all outputs to ensure gradient path to all variables
            if isinstance(output, tuple):
                # Model has sequence_output and pooled_output
                loss = keras.ops.mean(keras.ops.square(output[0])) + keras.ops.mean(keras.ops.square(output[1]))
            else:
                # Model only has sequence_output
                loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that all gradients are not None
        assert all(g is not None for g in gradients)

        # Check that at least some gradients have meaningful magnitudes
        grad_norms = [keras.ops.convert_to_numpy(keras.ops.norm(g)) for g in gradients if g is not None]
        assert any(norm > 1e-8 for norm in grad_norms)


class TestFNetFactoryFunctions:
    """Test suite for FNet factory functions."""

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Small configuration for testing."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'intermediate_size': 64,
            'max_position_embeddings': 16,
            'hidden_dropout_prob': 0.1
        }

    def test_create_fnet_for_classification(self, small_config):
        """Test classification model creation."""
        num_labels = 5
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_classification(small_config, num_labels, max_sequence_length=seq_len)

        assert isinstance(model, keras.Model)
        assert model.name == 'fnet_for_classification'

        # Test input shapes
        assert len(model.inputs) == 3  # input_ids, attention_mask, position_ids
        assert model.inputs[0].shape == (None, seq_len)  # input_ids
        assert model.inputs[1].shape == (None, seq_len)  # attention_mask
        assert model.inputs[2].shape == (None, seq_len)  # position_ids

        # Test output shape
        assert model.output.shape == (None, num_labels)

    def test_create_fnet_for_sequence_output(self, small_config):
        """Test sequence output model creation."""
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_sequence_output(small_config, max_sequence_length=seq_len)

        assert isinstance(model, keras.Model)
        assert model.name == 'fnet_for_sequence_output'

        # Test shapes
        assert len(model.inputs) == 3
        assert model.output.shape == (None, seq_len, small_config['hidden_size'])

    def test_classification_model_forward_pass(self, small_config):
        """Test forward pass of classification model."""
        num_labels = 3
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_classification(small_config, num_labels, max_sequence_length=seq_len)

        # Create sample inputs
        batch_size = 2
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len))
        attention_mask = keras.ops.ones((batch_size, seq_len), dtype="int32")
        position_ids = keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)

        # Forward pass
        logits = model([input_ids, attention_mask, position_ids])

        assert logits.shape == (batch_size, num_labels)

    def test_sequence_model_forward_pass(self, small_config):
        """Test forward pass of sequence output model."""
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_sequence_output(small_config, max_sequence_length=seq_len)

        # Create sample inputs
        batch_size = 2
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len))
        attention_mask = keras.ops.ones((batch_size, seq_len), dtype="int32")
        position_ids = keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)

        # Forward pass
        sequence_output = model([input_ids, attention_mask, position_ids])

        assert sequence_output.shape == (batch_size, seq_len, small_config['hidden_size'])

    def test_create_fnet_convenience_function(self):
        """Test the convenience create_fnet function."""
        # Test classification
        classifier = create_fnet('tiny', num_classes=2, task_type='classification')
        assert isinstance(classifier, keras.Model)
        assert classifier.output.shape == (None, 2)

        # Test sequence output
        sequence_model = create_fnet('tiny', task_type='sequence_output')
        assert isinstance(sequence_model, keras.Model)
        # Should have sequence output shape (batch, seq_len, hidden_size)
        assert len(sequence_model.output.shape) == 3

    def test_classification_model_compilation_and_training(self, small_config):
        """Test classification model compilation and training."""
        num_labels = 2
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_classification(small_config, num_labels, max_sequence_length=seq_len)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create sample data
        batch_size = 4
        x_train = [
            keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            keras.ops.ones((batch_size, seq_len), dtype="int32"),
            keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)
        ]
        y_train = keras.random.randint(minval=0, maxval=num_labels, shape=(batch_size,))

        # Training step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert 'loss' in history.history

    def test_model_serialization_factory_functions(self, small_config):
        """Test serialization of models created by factory functions."""
        num_labels = 3
        seq_len = small_config['max_position_embeddings']
        model = create_fnet_for_classification(small_config, num_labels, max_sequence_length=seq_len)

        # Create sample inputs
        batch_size = 2
        inputs = [
            keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            keras.ops.ones((batch_size, seq_len), dtype="int32"),
            keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)
        ]

        original_pred = model(inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet_classifier.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(inputs)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded classification model should match original"
            )

class TestFNetEdgeCases:
    """Edge cases and error handling tests."""

    def test_invalid_model_parameters(self):
        """Test model creation with invalid parameters."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            FNet(vocab_size=0)

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            FNet(hidden_size=0)

        with pytest.raises(ValueError, match="num_layers must be positive"):
            FNet(num_layers=0)

        with pytest.raises(ValueError, match="intermediate_size must be positive"):
            FNet(intermediate_size=0)

        with pytest.raises(ValueError, match="hidden_dropout_prob must be between 0 and 1"):
            FNet(hidden_dropout_prob=1.5)

    def test_invalid_factory_function_parameters(self):
        """Test factory functions with invalid parameters."""
        config = {'vocab_size': 100, 'hidden_size': 32, 'num_layers': 1, 'max_position_embeddings': 16}

        with pytest.raises(ValueError, match="num_labels must be positive"):
            create_fnet_for_classification(config, num_labels=0, max_sequence_length=16)

        with pytest.raises(ValueError, match="num_classes must be provided"):
            create_fnet(variant='tiny', task_type='classification', num_classes=None)

        with pytest.raises(ValueError, match="Unknown task_type"):
            create_fnet(variant='tiny', task_type='invalid_task')

    def test_extremely_small_configurations(self):
        """Test with extremely small but valid configurations."""
        tiny_config = {
            'vocab_size': 10,
            'hidden_size': 4,
            'num_layers': 1,
            'intermediate_size': 8,
            'max_position_embeddings': 4,
            'add_pooling_layer': False
        }

        model = FNet(**tiny_config)

        # Test with tiny input
        input_ids = keras.random.randint(minval=0, maxval=10, shape=(1, 4))
        output = model(input_ids)

        assert output.shape == (1, 4, 4)  # (batch, seq_len, hidden_size)

    def test_model_with_different_input_lengths(self):
        """Test model with different sequence lengths."""
        config = {
            'vocab_size': 50,
            'hidden_size': 16,
            'num_layers': 1,
            'intermediate_size': 32,
            'max_position_embeddings': 32,
            'add_pooling_layer': True
        }

        # Test with different sequence lengths
        for seq_len in [4, 8, 16]:
            # Re-create the model for each sequence length
            # FNet bakes the sequence length into the DFT matrices upon build
            config['max_position_embeddings'] = seq_len
            model = FNet(**config)

            input_ids = keras.random.randint(minval=0, maxval=50, shape=(1, seq_len))
            sequence_output, pooled_output = model(input_ids)

            assert sequence_output.shape == (1, seq_len, 16)
            assert pooled_output.shape == (1, 16)  # Pooled output always same size

    def test_model_summary(self):
        """Test model summary functionality."""
        config = {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'intermediate_size': 64,
            'max_position_embeddings': 16
        }
        # Build the model first by calling it on some data
        model = FNet(**config)
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(1, 16))
        model(input_ids)

        # This should not raise any errors
        model.summary()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])