"""
Comprehensive pytest test suite for the BERT Embeddings layer.

This module provides extensive testing for the Embeddings layer, ensuring it adheres
to modern Keras 3 best practices as outlined in the provided guide. Tests include:
- Initialization and parameter validation.
- Correct build process, including explicit sub-layer building.
- Forward pass functionality with different input combinations.
- Behavior in training vs. inference modes.
- Support for various normalization types.
- A full serialization and deserialization cycle to guarantee production readiness.
- Completeness of the get_config method.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.bert.components import Embeddings


class TestEmbeddingsLayer:
    """Comprehensive test suite for the Embeddings layer."""

    @pytest.fixture
    def basic_params(self) -> Dict[str, Any]:
        """Provides a standard set of parameters for creating the layer."""
        return {
            'vocab_size': 1000,
            'hidden_size': 256,
            'max_position_embeddings': 128,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'normalization_type': 'layer_norm'
        }

    @pytest.fixture
    def sample_input(self, basic_params) -> keras.KerasTensor:
        """Provides a sample input tensor for testing forward passes."""
        batch_size, seq_length = 4, 32
        return ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_params['vocab_size']
            ),
            dtype='int32'
        )

    def test_initialization(self, basic_params):
        """Test that the layer initializes correctly and creates sub-layers."""
        layer = Embeddings(**basic_params)

        # Verify all parameters are stored correctly
        for key, value in basic_params.items():
            assert getattr(layer, key) == value

        # Verify sub-layers are created (but not built)
        assert isinstance(layer.word_embeddings, keras.layers.Embedding)
        assert isinstance(layer.position_embeddings, keras.layers.Embedding)
        assert isinstance(layer.token_type_embeddings, keras.layers.Embedding)
        assert isinstance(layer.layer_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.dropout, keras.layers.Dropout)

        # The layer and its children should be unbuilt after initialization
        assert not layer.built
        assert not layer.word_embeddings.built

    def test_parameter_validation(self, basic_params):
        """Test that invalid __init__ parameters raise ValueErrors."""
        # Test invalid numerical parameters
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Embeddings(**{**basic_params, 'vocab_size': 0})
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            Embeddings(**{**basic_params, 'hidden_size': -1})
        with pytest.raises(ValueError, match="initializer_range must be positive"):
            Embeddings(**{**basic_params, 'initializer_range': 0})

        # Test invalid dropout probability
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between 0 and 1"):
            Embeddings(**{**basic_params, 'hidden_dropout_prob': 1.1})
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between 0 and 1"):
            Embeddings(**{**basic_params, 'hidden_dropout_prob': -0.1})

        # Test invalid normalization type
        with pytest.raises(ValueError, match="normalization_type must be one of"):
            Embeddings(**{**basic_params, 'normalization_type': 'invalid_norm'})

    def test_build_process(self, basic_params, sample_input):
        """Verify the explicit build method correctly builds all sub-layers."""
        layer = Embeddings(**basic_params)
        input_shape = ops.shape(sample_input)

        # Manually build the layer
        layer.build(input_shape)

        # The layer and all its sub-layers must be marked as built
        assert layer.built
        assert layer.word_embeddings.built
        assert layer.position_embeddings.built
        assert layer.token_type_embeddings.built
        assert layer.layer_norm.built
        assert layer.dropout.built

    def test_build_invalid_shape(self, basic_params):
        """Test that building with an invalid input shape raises an error."""
        layer = Embeddings(**basic_params)
        with pytest.raises(ValueError, match="Expected 2D input shape"):
            layer.build((None, 32, 128))  # Invalid 3D shape

    def test_forward_pass_input_ids_only(self, basic_params, sample_input):
        """Test forward pass with only input_ids, relying on default creation."""
        layer = Embeddings(**basic_params)
        output = layer(sample_input, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_forward_pass_with_token_type_ids(self, basic_params, sample_input):
        """Test forward pass with provided token_type_ids."""
        layer = Embeddings(**basic_params)
        token_type_ids = ops.zeros_like(sample_input, dtype='int32')
        output = layer(sample_input, token_type_ids=token_type_ids, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_forward_pass_with_position_ids(self, basic_params, sample_input):
        """Test forward pass with provided position_ids."""
        layer = Embeddings(**basic_params)
        position_ids = ops.arange(ops.shape(sample_input)[1], dtype='int32')
        position_ids = ops.broadcast_to(ops.expand_dims(position_ids, 0), ops.shape(sample_input))
        output = layer(sample_input, position_ids=position_ids, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape

    def test_training_mode(self, basic_params, sample_input):
        """Test that dropout is applied in training mode but not evaluation."""
        # With dropout enabled
        layer_with_dropout = Embeddings(**basic_params)
        output_train = layer_with_dropout(sample_input, training=True)
        output_eval = layer_with_dropout(sample_input, training=False)
        # Outputs should differ due to dropout
        assert not np.allclose(
            ops.convert_to_numpy(output_train),
            ops.convert_to_numpy(output_eval)
        )

        # With dropout disabled
        params_no_dropout = {**basic_params, 'hidden_dropout_prob': 0.0}
        layer_no_dropout = Embeddings(**params_no_dropout)
        output_train_no_dropout = layer_no_dropout(sample_input, training=True)
        output_eval_no_dropout = layer_no_dropout(sample_input, training=False)
        # Outputs should be identical
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_train_no_dropout),
            ops.convert_to_numpy(output_eval_no_dropout)
        )

    @pytest.mark.parametrize("norm_type", ['layer_norm', 'rms_norm', 'band_rms', 'batch_norm'])
    def test_normalization_types(self, basic_params, sample_input, norm_type):
        """Test that all supported normalization types are functional."""
        params = {**basic_params, 'normalization_type': norm_type}
        layer = Embeddings(**params)
        output = layer(sample_input, training=False)

        expected_shape = (*ops.shape(sample_input), basic_params['hidden_size'])
        assert output.shape == expected_shape
        assert ops.all(ops.isfinite(output))

    def test_compute_output_shape(self, basic_params, sample_input):
        """Verify the compute_output_shape method."""
        layer = Embeddings(**basic_params)
        input_shape = ops.shape(sample_input)
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (*input_shape, basic_params['hidden_size'])
        assert output_shape == expected_shape

    def test_get_config_completeness(self, basic_params):
        """Verify that get_config includes all __init__ parameters."""
        layer = Embeddings(**basic_params)
        config = layer.get_config()

        for key in basic_params:
            assert key in config, f"Key '{key}' is missing from get_config()"
            assert config[key] == basic_params[key]

    def test_serialization_cycle(self, basic_params, sample_input):
        """CRITICAL TEST: Ensure a full save and load cycle works perfectly."""
        # 1. Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:], dtype='int32')
        layer_output = Embeddings(**basic_params)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original model
        original_prediction = model(sample_input)

        # 3. Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_embeddings_model.keras')
            model.save(filepath)

            # The `custom_objects` argument is not needed due to the registration decorator
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # 4. Verify that predictions are identical
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after a save/load cycle."
            )