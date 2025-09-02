"""
Comprehensive test suite for the refined EntityGraphRefinement layer following Modern Keras 3 guidelines.
"""

import pytest
import tempfile
import os
import tensorflow as tf
from typing import Any, Dict
import numpy as np
import keras
from keras import ops

# Import the layer to test
from dl_techniques.layers.graphs.entity_graph_refinement import EntityGraphRefinement


class TestEntityGraphRefinement:
    """Comprehensive test suite for the refined EntityGraphRefinement layer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Standard configuration for basic testing."""
        # --- ADJUSTED: Removed 'extraction_method' as it's no longer a parameter ---
        return {
            'max_entities': 10,
            'entity_dim': 64,
            'num_refinement_steps': 2,
            'initial_density': 0.5,
            'attention_heads': 4,
            'dropout_rate': 0.1,
            'refinement_activation': 'gelu',
            'entity_activity_threshold': 0.1,
            'use_positional_encoding': True,
            'max_sequence_length': 100,
            'regularization_weight': 0.01,
            'activity_regularization_target': 0.1
        }

    # --- DELETED: Removed clustering_config fixture as it's obsolete ---

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return keras.random.normal(shape=(2, 20, 128))

    @pytest.fixture
    def large_sample_input(self) -> keras.KerasTensor:
        """Larger sample input for comprehensive testing."""
        return keras.random.normal(shape=(4, 50, 256))

    def test_initialization(self, basic_config):
        """Test layer initialization and attribute creation."""
        layer = EntityGraphRefinement(**basic_config)

        # Check all attributes are stored correctly
        assert layer.max_entities == basic_config['max_entities']
        assert layer.entity_dim == basic_config['entity_dim']
        assert layer.num_refinement_steps == basic_config['num_refinement_steps']
        assert layer.initial_density == basic_config['initial_density']
        # --- ADJUSTED: Removed assertion for layer.extraction_method ---
        assert layer.attention_heads == basic_config['attention_heads']
        assert layer.dropout_rate == basic_config['dropout_rate']
        assert layer.refinement_activation == basic_config['refinement_activation']
        assert layer.entity_activity_threshold == basic_config['entity_activity_threshold']
        assert layer.use_positional_encoding == basic_config['use_positional_encoding']
        assert layer.max_sequence_length == basic_config['max_sequence_length']
        assert layer.regularization_weight == basic_config['regularization_weight']
        assert layer.activity_regularization_target == basic_config['activity_regularization_target']

        # Check layer is not built initially
        assert not layer.built
        assert layer.entity_library is None
        assert layer.graph_refinement_mlp is None
        assert layer.sparsification_gate is None

    def test_forward_pass(self, basic_config, sample_input):
        """Test forward pass with the default attention extraction method."""
        # --- ADJUSTED: Renamed from test_forward_pass_attention ---
        layer = EntityGraphRefinement(**basic_config)

        # Forward pass should trigger building
        entities, graph, entity_mask = layer(sample_input)

        # Check layer is now built
        assert layer.built
        assert layer.entity_library is not None
        assert layer.graph_refinement_mlp is not None
        assert layer.sparsification_gate is not None
        assert layer.entity_extractor is not None
        # The extractor should be MultiHeadAttention
        assert isinstance(layer.entity_extractor, keras.layers.MultiHeadAttention)

        # Check output shapes
        batch_size, _seq_len, _embed_dim = sample_input.shape
        expected_entities_shape = (batch_size, basic_config['max_entities'], basic_config['entity_dim'])
        expected_graph_shape = (batch_size, basic_config['max_entities'], basic_config['max_entities'])
        expected_mask_shape = (batch_size, basic_config['max_entities'])

        assert entities.shape == expected_entities_shape
        assert graph.shape == expected_graph_shape
        assert entity_mask.shape == expected_mask_shape

    # --- DELETED: Removed test_forward_pass_clustering as it's obsolete ---

    def test_serialization_cycle(self, basic_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = EntityGraphRefinement(**basic_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original predictions
        original_entities, original_graph, original_mask = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_entities, loaded_graph, loaded_mask = loaded_model(sample_input)

            # Verify identical predictions after serialization
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_entities),
                ops.convert_to_numpy(loaded_entities),
                rtol=1e-6, atol=1e-6,
                err_msg="Entities differ after serialization"
            )
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_graph),
                ops.convert_to_numpy(loaded_graph),
                rtol=1e-6, atol=1e-6,
                err_msg="Graph differs after serialization"
            )
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_mask),
                ops.convert_to_numpy(loaded_mask),
                rtol=1e-6, atol=1e-6,
                err_msg="Mask differs after serialization"
            )

    def test_config_completeness(self, basic_config):
        """Test that get_config contains all __init__ parameters."""
        layer = EntityGraphRefinement(**basic_config)
        config = layer.get_config()

        # Check all basic_config parameters are present in get_config
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == basic_config[key], f"Value mismatch for {key}"

        # Check layer base config is included
        assert 'name' in config
        assert 'trainable' in config

    def test_config_completeness_minimal(self):
        """Test config completeness with minimal parameters."""
        # Create layer with only required parameters
        layer = EntityGraphRefinement(max_entities=5, entity_dim=32)
        config = layer.get_config()

        # --- ADJUSTED: Removed 'extraction_method' from the list of required keys ---
        required_keys = [
            'max_entities', 'entity_dim', 'num_refinement_steps', 'initial_density',
            'attention_heads', 'dropout_rate', 'refinement_activation',
            'entity_activity_threshold', 'use_positional_encoding', 'max_sequence_length',
            'regularization_weight', 'activity_regularization_target'
        ]

        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

        # Test reconstruction from config
        layer_reconstructed = EntityGraphRefinement.from_config(config)
        assert layer_reconstructed.max_entities == 5
        assert layer_reconstructed.entity_dim == 32

    def test_gradients_flow(self, basic_config, sample_input):
        """Test gradient computation flows correctly through the layer."""
        layer = EntityGraphRefinement(**basic_config)
        # Ensure input is a variable for gradient tracking
        sample_input_var = tf.Variable(sample_input)

        with tf.GradientTape() as tape:
            entities, graph, entity_mask = layer(sample_input_var, training=True)
            # Create a simple loss from all outputs
            loss = ops.mean(ops.square(entities)) + ops.mean(ops.square(graph)) + ops.mean(ops.square(entity_mask))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients exist and are not None
        assert len(gradients) > 0, "No gradients computed"
        for i, grad in enumerate(gradients):
            assert grad is not None, f"Gradient for variable {layer.trainable_variables[i].name} is None"
            assert not ops.any(ops.isnan(grad)), f"Gradient for variable {layer.trainable_variables[i].name} contains NaN"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = EntityGraphRefinement(**basic_config)
        entities, graph, entity_mask = layer(sample_input, training=training)
        assert entities.shape[0] == sample_input.shape[0]
        assert not ops.any(ops.isnan(entities)), "Entities contain NaN"
        assert not ops.any(ops.isnan(graph)), "Graph contains NaN"
        assert not ops.any(ops.isnan(entity_mask)), "Entity mask contains NaN"

    def test_training_mode_regularization(self, basic_config, sample_input):
        """Test that regularization losses are only added during training."""
        config_with_reg = basic_config.copy()
        config_with_reg['regularization_weight'] = 0.1
        layer = EntityGraphRefinement(**config_with_reg)
        _ = layer(sample_input, training=True)
        training_losses = len(layer.losses)
        layer.losses.clear() # Reset losses for next call
        _ = layer(sample_input, training=False)
        inference_losses = len(layer.losses)
        assert training_losses > 0, "No regularization losses added during training"
        assert inference_losses == 0, "Regularization losses added during inference"

    def test_value_ranges_and_constraints(self, basic_config, sample_input):
        """Test that outputs are within expected value ranges."""
        layer = EntityGraphRefinement(**basic_config)
        entities, graph, entity_mask = layer(sample_input)
        graph_np = ops.convert_to_numpy(graph)
        mask_np = ops.convert_to_numpy(entity_mask)
        assert np.all(graph_np >= -1.0) and np.all(graph_np <= 1.0), "Graph values out of range [-1, 1]"
        assert np.all(mask_np >= 0.0) and np.all(mask_np <= 1.0), "Mask values out of range [0, 1]"
        for batch_idx in range(graph_np.shape[0]):
            diagonal = np.diag(graph_np[batch_idx])
            assert np.allclose(diagonal, 0.0, atol=1e-6), f"Non-zero diagonal found in batch {batch_idx}"

    def test_positional_encoding_behavior(self, basic_config, sample_input):
        """Test positional encoding on/off behavior."""
        config_with_pos = {**basic_config, 'use_positional_encoding': True}
        layer_with_pos = EntityGraphRefinement(**config_with_pos)
        entities_with_pos, _, _ = layer_with_pos(sample_input)

        config_no_pos = {**basic_config, 'use_positional_encoding': False}
        layer_no_pos = EntityGraphRefinement(**config_no_pos)
        entities_no_pos, _, _ = layer_no_pos(sample_input)

        entities_diff = ops.mean(ops.abs(entities_with_pos - entities_no_pos))
        assert entities_diff > 1e-6, "Positional encoding should affect results"
        assert layer_with_pos.positional_encoder is not None
        assert layer_no_pos.positional_encoder is None

    # (The rest of the tests remain unchanged as they are still valid)

    def test_long_sequence_positional_clipping(self):
        """Test that long sequences are handled correctly with positional encoding."""
        config = {
            'max_entities': 5, 'entity_dim': 32, 'use_positional_encoding': True,
            'max_sequence_length': 50
        }
        layer = EntityGraphRefinement(**config)
        long_input = keras.random.normal(shape=(1, 100, 64))
        entities, graph, mask = layer(long_input)
        assert entities.shape == (1, 5, 32)
        assert graph.shape == (1, 5, 5)

    def test_different_refinement_steps(self, sample_input):
        """Test different numbers of refinement steps."""
        for num_steps in [1, 3, 5]:
            layer = EntityGraphRefinement(max_entities=8, entity_dim=32, num_refinement_steps=num_steps)
            _entities, graph, _mask = layer(sample_input)
            assert not ops.any(ops.isnan(graph)), f"NaN in graph with {num_steps} steps"

    def test_compute_output_shape(self, basic_config):
        """Test compute_output_shape method."""
        layer = EntityGraphRefinement(**basic_config)
        input_shape = (None, 50, 128)
        entities_shape, graph_shape, mask_shape = layer.compute_output_shape(input_shape)
        assert entities_shape == (None, basic_config['max_entities'], basic_config['entity_dim'])
        assert graph_shape == (None, basic_config['max_entities'], basic_config['max_entities'])
        assert mask_shape == (None, basic_config['max_entities'])

    def test_edge_cases(self):
        """Test error conditions and edge cases."""
        with pytest.raises(ValueError, match="max_entities must be positive"):
            EntityGraphRefinement(max_entities=0, entity_dim=32)
        with pytest.raises(ValueError, match="attention_heads must be positive"):
            EntityGraphRefinement(max_entities=5, entity_dim=32, attention_heads=0)
        with pytest.raises(ValueError, match="dropout_rate must be between 0 and 1"):
            EntityGraphRefinement(max_entities=5, entity_dim=32, dropout_rate=1.5)

    def test_minimal_configuration(self):
        """Test layer with minimal required configuration."""
        layer = EntityGraphRefinement(max_entities=3, entity_dim=16)
        small_input = keras.random.normal(shape=(1, 5, 32))
        entities, graph, mask = layer(small_input)
        assert entities.shape == (1, 3, 16)
        assert graph.shape == (1, 3, 3)

    def test_large_scale_configuration(self, large_sample_input):
        """Test with larger, more realistic configuration."""
        large_config = {
            'max_entities': 50, 'entity_dim': 256, 'num_refinement_steps': 4,
            'attention_heads': 16, 'initial_density': 0.8
        }
        layer = EntityGraphRefinement(**large_config)
        entities, graph, mask = layer(large_sample_input)
        batch_size = large_sample_input.shape[0]
        assert entities.shape == (batch_size, 50, 256)
        assert graph.shape == (batch_size, 50, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])