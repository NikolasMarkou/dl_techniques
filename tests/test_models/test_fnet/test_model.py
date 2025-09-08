"""
Comprehensive test suite for FNet model creation functions.

This test suite validates the FNet encoder and classifier model creation functions
with small memory footprint configurations suitable for CPU testing.
"""

import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.fnet.model import create_fnet_encoder, create_fnet_classifier


class TestCreateFNetEncoder:
    """Test suite for create_fnet_encoder function."""

    @pytest.fixture
    def encoder_config(self) -> Dict[str, Any]:
        """Small configuration for CPU testing."""
        return {
            'num_layers': 2,
            'hidden_dim': 32,
            'max_seq_length': 8,
            'intermediate_dim': 64,  # 2x expansion instead of 4x
            'dropout_rate': 0.1
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Small sample input for testing."""
        return keras.random.normal(shape=(2, 8, 32))  # batch=2, seq=8, hidden=32

    def test_basic_encoder_creation(self, encoder_config):
        """Test basic encoder model creation."""
        model = create_fnet_encoder(**encoder_config)
        assert isinstance(model, keras.Model)
        assert '2L_32H' in model.name
        input_shape = model.input.shape
        output_shape = model.output.shape
        assert input_shape == (None, 8, 32)
        assert output_shape == (None, 8, 32)

    def test_encoder_forward_pass(self, encoder_config, sample_input):
        """Test encoder forward pass."""
        model = create_fnet_encoder(**encoder_config)
        output = model(sample_input)
        assert output.shape == sample_input.shape
        input_np = keras.ops.convert_to_numpy(sample_input)
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.allclose(input_np, output_np, atol=1e-6)

    def test_encoder_serialization(self, encoder_config, sample_input):
        """Test encoder model serialization."""
        model = create_fnet_encoder(**encoder_config)
        original_pred = model(sample_input)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet_encoder.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )

    def test_encoder_default_intermediate_dim(self):
        """Test encoder with default intermediate_dim (4x hidden_dim)."""
        model = create_fnet_encoder(
            num_layers=1, hidden_dim=16, max_seq_length=4, dropout_rate=0.0
        )
        test_input = keras.random.normal(shape=(1, 4, 16))
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_encoder_with_additional_kwargs(self, sample_input):
        """Test encoder with additional encoder block arguments."""
        model = create_fnet_encoder(
            num_layers=2, hidden_dim=32, max_seq_length=8, dropout_rate=0.1,
            activation='relu', fourier_config={'normalize_dft': False}
        )
        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_encoder_different_layer_counts(self):
        """Test encoder with different numbers of layers."""
        test_input = keras.random.normal(shape=(1, 4, 16))
        for num_layers in [1, 2, 3]:
            model = create_fnet_encoder(
                num_layers=num_layers, hidden_dim=16, max_seq_length=4, dropout_rate=0.0
            )
            output = model(test_input)
            assert output.shape == test_input.shape
            encoder_layers = [l for l in model.layers if 'fnet_encoder_layer' in l.name]
            assert len(encoder_layers) == num_layers

    def test_encoder_compilation(self, encoder_config):
        """Test that encoder can be compiled."""
        model = create_fnet_encoder(**encoder_config)
        model.compile(optimizer='adam', loss='mse')
        assert model.compiled


class TestCreateFNetClassifier:
    """Test suite for create_fnet_classifier function."""

    @pytest.fixture
    def classifier_config(self) -> Dict[str, Any]:
        """Small configuration for CPU testing."""
        return {
            'num_classes': 5, 'num_layers': 2, 'hidden_dim': 32,
            'max_seq_length': 16, 'vocab_size': 100, 'dropout_rate': 0.1
        }

    @pytest.fixture
    def sample_token_input(self, classifier_config) -> keras.KerasTensor:
        """Sample token input for classifier."""
        return keras.random.randint(
            minval=0, maxval=classifier_config['vocab_size'],
            shape=(2, classifier_config['max_seq_length'])
        )

    def test_basic_classifier_creation(self, classifier_config):
        """Test basic classifier model creation."""
        model = create_fnet_classifier(**classifier_config)
        assert isinstance(model, keras.Model)
        assert model.name == 'fnet_classifier'
        assert model.input.shape == (None, 16)
        assert model.output.shape == (None, 5)

    def test_classifier_forward_pass(self, classifier_config, sample_token_input):
        """Test classifier forward pass."""
        model = create_fnet_classifier(**classifier_config)
        output = model(sample_token_input)
        assert output.shape == (2, 5)

    def test_classifier_serialization(self, classifier_config, sample_token_input):
        """Test classifier model serialization."""
        model = create_fnet_classifier(**classifier_config)
        original_pred = model(sample_token_input)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet_classifier.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_token_input)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )

    def test_classifier_compilation_and_training(self, classifier_config):
        """Test classifier compilation and basic training step."""
        config = classifier_config.copy()
        config.update({'num_classes': 2, 'vocab_size': 50})
        model = create_fnet_classifier(**config)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        x_train = keras.random.randint(minval=0, maxval=50, shape=(8, 16))
        y_train = keras.random.randint(minval=0, maxval=2, shape=(8,))
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert 'loss' in history.history

    def test_classifier_gradients_flow(self, classifier_config, sample_token_input):
        """Test gradient computation through classifier."""
        model = create_fnet_classifier(**classifier_config)
        with tf.GradientTape() as tape:
            output = model(sample_token_input)
            loss = keras.ops.mean(keras.ops.square(output))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)
        grad_norms = [keras.ops.convert_to_numpy(keras.ops.norm(g)) for g in gradients]
        assert any(norm > 1e-8 for norm in grad_norms)


class TestFNetModelsEdgeCases:
    """Edge cases and error handling tests."""

    def test_invalid_encoder_parameters(self):
        """Test encoder with invalid parameters."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            create_fnet_encoder(num_layers=0, hidden_dim=32, max_seq_length=8)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            create_fnet_encoder(num_layers=1, hidden_dim=0, max_seq_length=8)

    def test_invalid_classifier_parameters(self):
        """Test classifier with invalid parameters."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            create_fnet_classifier(num_classes=0, hidden_dim=32)
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            create_fnet_classifier(num_classes=2, vocab_size=0)

    def test_extremely_small_configurations(self):
        """Test with extremely small but valid configurations."""
        encoder = create_fnet_encoder(
            num_layers=1, hidden_dim=2, max_seq_length=1, intermediate_dim=4
        )
        tiny_input = keras.random.normal(shape=(1, 1, 2))
        output = encoder(tiny_input)
        assert output.shape == tiny_input.shape

        classifier = create_fnet_classifier(
            num_classes=1, num_layers=1, hidden_dim=2,
            max_seq_length=2, vocab_size=2
        )
        tiny_tokens = keras.random.randint(minval=0, maxval=2, shape=(1, 2))
        output = classifier(tiny_tokens)
        assert output.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])