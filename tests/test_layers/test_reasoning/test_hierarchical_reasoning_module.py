import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.reasoning.hrm_reasoning_module import HierarchicalReasoningModule


class TestHierarchicalReasoningModule:
    """Comprehensive test suite for HierarchicalReasoningModule."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_layers': 2,
            'embed_dim': 32,
            'num_heads': 4,
            'ffn_expansion_factor': 4,
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'num_layers': 1,
            'embed_dim': 16,
            'num_heads': 2,
        }

    @pytest.fixture
    def sample_inputs(self) -> list:
        """Sample list of [hidden_states, input_injection] for testing."""
        hidden_states = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 8, 32)).astype(np.float32)
        )
        input_injection = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 8, 32)).astype(np.float32)
        )
        return [hidden_states, input_injection]

    @pytest.fixture
    def minimal_inputs(self) -> list:
        """Minimal sample inputs."""
        hidden_states = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 4, 16)).astype(np.float32)
        )
        input_injection = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 4, 16)).astype(np.float32)
        )
        return [hidden_states, input_injection]

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = HierarchicalReasoningModule(**layer_config)

        assert layer.num_layers == 2
        assert layer.embed_dim == 32
        assert layer.num_heads == 4
        assert layer.ffn_expansion_factor == 4
        assert not layer.built
        assert len(layer.blocks) == 2

    def test_default_configuration(self):
        """Test default HRM-optimized configuration."""
        layer = HierarchicalReasoningModule(num_layers=1, embed_dim=16, num_heads=2)

        assert layer.attention_type == 'multi_head'
        assert layer.normalization_type == 'rms_norm'
        assert layer.normalization_position == 'post'
        assert layer.ffn_type == 'swiglu'
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is False

    def test_forward_pass(self, layer_config, sample_inputs):
        """Test forward pass and building."""
        layer = HierarchicalReasoningModule(**layer_config)

        output = layer(sample_inputs)

        assert layer.built
        assert output.shape == (2, 8, 32)

    def test_input_injection_effect(self, minimal_config):
        """Test that input injection actually modifies the output."""
        layer = HierarchicalReasoningModule(**minimal_config)

        hidden = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 4, 16)).astype(np.float32)
        )
        zeros = ops.zeros_like(hidden)
        nonzeros = ops.convert_to_tensor(
            np.random.normal(0, 0.1, (2, 4, 16)).astype(np.float32)
        )

        output_no_injection = layer([hidden, zeros])
        output_with_injection = layer([hidden, nonzeros])

        assert not np.allclose(
            ops.convert_to_numpy(output_no_injection),
            ops.convert_to_numpy(output_with_injection),
            atol=1e-5
        )

    def test_different_layer_counts(self, sample_inputs):
        """Test with different numbers of transformer layers."""
        for num_layers in [1, 2, 4]:
            layer = HierarchicalReasoningModule(
                num_layers=num_layers,
                embed_dim=32,
                num_heads=4,
            )
            output = layer(sample_inputs)
            assert output.shape == (2, 8, 32)
            assert len(layer.blocks) == num_layers

    def test_configurable_architecture(self, sample_inputs):
        """Test with non-default architecture options."""
        layer = HierarchicalReasoningModule(
            num_layers=1,
            embed_dim=32,
            num_heads=4,
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            dropout_rate=0.1,
            use_bias=True,
        )

        output = layer(sample_inputs)
        assert output.shape == (2, 8, 32)

    def test_serialization_cycle(self, layer_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle."""
        input1 = keras.Input(shape=(8, 32))
        input2 = keras.Input(shape=(8, 32))
        outputs = HierarchicalReasoningModule(**layer_config)([input1, input2])
        model = keras.Model([input1, input2], outputs)

        original_pred = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_hrm_module.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-5, atol=1e-5,
                err_msg="HierarchicalReasoningModule predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = HierarchicalReasoningModule(**layer_config)
        config = layer.get_config()

        expected_keys = {
            'num_layers', 'embed_dim', 'num_heads', 'ffn_expansion_factor',
            'attention_type', 'normalization_type', 'normalization_position',
            'ffn_type', 'dropout_rate', 'use_bias',
            'kernel_initializer', 'kernel_regularizer',
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['num_layers'] == layer_config['num_layers']
        assert config['embed_dim'] == layer_config['embed_dim']
        assert config['num_heads'] == layer_config['num_heads']

    def test_config_roundtrip(self, layer_config):
        """Test config-based reconstruction."""
        layer = HierarchicalReasoningModule(**layer_config)
        config = layer.get_config()

        reconstructed = HierarchicalReasoningModule.from_config(config)

        assert reconstructed.num_layers == layer.num_layers
        assert reconstructed.embed_dim == layer.embed_dim
        assert reconstructed.num_heads == layer.num_heads
        assert reconstructed.ffn_expansion_factor == layer.ffn_expansion_factor

    def test_gradients_flow(self, layer_config, sample_inputs):
        """Test gradient computation."""
        import tensorflow as tf

        layer = HierarchicalReasoningModule(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_inputs[0])
            tape.watch(sample_inputs[1])
            output = layer(sample_inputs)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = HierarchicalReasoningModule(**layer_config)

        input_shape = [(None, 8, 32), (None, 8, 32)]
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 8, 32)

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            HierarchicalReasoningModule(num_layers=0, embed_dim=16, num_heads=2)

        with pytest.raises(ValueError, match="embed_dim must be positive"):
            HierarchicalReasoningModule(num_layers=1, embed_dim=0, num_heads=2)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            HierarchicalReasoningModule(num_layers=1, embed_dim=16, num_heads=0)

        with pytest.raises(ValueError, match="must be divisible"):
            HierarchicalReasoningModule(num_layers=1, embed_dim=15, num_heads=4)

    def test_build_input_validation(self, minimal_config):
        """Test build validates input shape."""
        layer = HierarchicalReasoningModule(**minimal_config)

        with pytest.raises(ValueError, match="list of two tensors"):
            layer.build((None, 4, 16))

        with pytest.raises(ValueError, match="must be identical"):
            layer.build([(None, 4, 16), (None, 8, 16)])

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, minimal_config, minimal_inputs, training):
        """Test behavior in different training modes."""
        layer = HierarchicalReasoningModule(**minimal_config)

        output = layer(minimal_inputs, training=training)
        assert output.shape == (2, 4, 16)
