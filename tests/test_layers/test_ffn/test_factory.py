"""
Comprehensive test suite for FFN factory module in dl_techniques framework.

This module tests the FFN factory interface, parameter validation, configuration handling,
and serialization for all supported feed-forward network types. Tests ensure robustness
and reliability of the factory system.
"""

import pytest
import tempfile
import os
import json
import tensorflow as tf
from typing import Dict, Any, List, Optional
from unittest.mock import patch

import numpy as np
import keras
from keras import ops

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import (
    create_ffn_layer,
    create_ffn_from_config,
    validate_ffn_config,
    get_ffn_info,
    FFNType,
    MLPBlock,
    SwiGLUFFN,
    LogicFFN,
    CountingFFN
)
from dl_techniques.layers.ffn.factory import FFN_REGISTRY


class TestFFNFactory:
    """Test suite for FFN factory functionality."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32, 768))

    @pytest.fixture
    def small_sample_input(self) -> keras.KerasTensor:
        """Create smaller sample input for resource-constrained tests."""
        return keras.random.normal(shape=(2, 16, 256))

    def test_get_ffn_info(self):
        """Test FFN information discovery function."""
        info = get_ffn_info()

        # Verify structure
        assert isinstance(info, dict)
        assert len(info) > 0

        # Check required FFN types are present
        required_types = ['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
        for ffn_type in required_types:
            assert ffn_type in info, f"Missing FFN type: {ffn_type}"

        # Verify information structure for each type
        for ffn_type, type_info in info.items():
            assert 'description' in type_info
            assert 'required_params' in type_info
            assert 'optional_params' in type_info
            assert 'use_case' in type_info

            # Verify required params is a list
            assert isinstance(type_info['required_params'], list)
            # Verify optional params is a dict with defaults
            assert isinstance(type_info['optional_params'], dict)

    @pytest.mark.parametrize("ffn_type", ['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp'])
    def test_create_ffn_layer_basic(self, ffn_type: str, sample_input: keras.KerasTensor):
        """Test basic layer creation for all supported types."""

        # Get type-specific parameters
        info = get_ffn_info()
        type_info = info[ffn_type]

        # Create minimal valid configuration
        config = {}

        # Add required parameters with sensible defaults
        if 'hidden_dim' in type_info['required_params']:
            config['hidden_dim'] = 512
        if 'output_dim' in type_info['required_params']:
            config['output_dim'] = 256
        if 'output_dim' in type_info['required_params']:
            config['output_dim'] = 768

        # Create layer
        layer = create_ffn_layer(ffn_type, **config)

        # Verify layer creation
        assert layer is not None
        assert hasattr(layer, 'call')

        # Test forward pass
        # Adjust input size for output_dim based layers
        if 'output_dim' in config:
            test_input = keras.random.normal(shape=(4, 32, config['output_dim']))
        else:
            test_input = keras.random.normal(shape=(4, 32, 512))  # Match hidden_dim

        output = layer(test_input)

        # Verify output shape
        expected_output_dim = config.get('output_dim', config.get('output_dim', 512))
        if ffn_type == 'swin_mlp' and 'output_dim' not in config:
             expected_output_dim = 512
        assert output.shape[-1] == expected_output_dim

    def test_create_ffn_layer_with_name(self):
        """Test layer creation with custom name."""
        layer = create_ffn_layer(
            'mlp',
            hidden_dim=256,
            output_dim=128,
            name='test_ffn_layer'
        )

        assert layer.name == 'test_ffn_layer'

    def test_validate_ffn_config_valid(self):
        """Test validation with valid configurations."""

        valid_configs = [
            ('mlp', {'hidden_dim': 512, 'output_dim': 256}),
            ('swiglu', {'output_dim': 768}),
            ('differential', {'hidden_dim': 1024, 'output_dim': 512}),
            ('glu', {'hidden_dim': 256, 'output_dim': 128}),
            ('geglu', {'hidden_dim': 512, 'output_dim': 256}),
            ('residual', {'hidden_dim': 1024, 'output_dim': 768}),
            ('swin_mlp', {'hidden_dim': 512})
        ]

        for ffn_type, config in valid_configs:
            # Should not raise exception
            validate_ffn_config(ffn_type, **config)

    def test_create_ffn_from_config(self):
        """Test configuration-based layer creation."""

        configs = [
            {
                'type': 'mlp',
                'hidden_dim': 512,
                'output_dim': 256,
                'activation': 'relu',
                'dropout_rate': 0.1,
                'name': 'test_mlp'
            },
            {
                'type': 'swiglu',
                'output_dim': 768,
                'ffn_expansion_factor': 4,
                'dropout_rate': 0.0,
                'name': 'test_swiglu'
            },
            {
                'type': 'differential',
                'hidden_dim': 1024,
                'output_dim': 512,
                'branch_activation': 'gelu',
                'name': 'test_differential'
            }
        ]

        for config in configs:
            layer = create_ffn_from_config(config)

            assert layer is not None
            assert layer.name == config['name']

            # Test forward pass
            if config['type'] == 'swiglu':
                test_input = keras.random.normal(shape=(2, 16, config['output_dim']))
            else:
                test_input = keras.random.normal(shape=(2, 16, 1024))

            output = layer(test_input)
            assert output is not None

    def test_create_ffn_from_config_missing_type(self):
        """Test error handling for missing type in config."""
        config = {'hidden_dim': 512, 'output_dim': 256}

        with pytest.raises(ValueError, match="Configuration must include 'type' key"):
            create_ffn_from_config(config)

    def test_layer_serialization_cycle(self):
        """Test complete serialization cycle for all FFN types."""

        test_configs = [
            ('mlp', {'hidden_dim': 256, 'output_dim': 128}),
            ('swiglu', {'output_dim': 512, 'ffn_expansion_factor': 4}),
            ('differential', {'hidden_dim': 256, 'output_dim': 128}),
            ('glu', {'hidden_dim': 256, 'output_dim': 128}),
            ('geglu', {'hidden_dim': 256, 'output_dim': 128}),
            ('residual', {'hidden_dim': 256, 'output_dim': 128}),
            ('swin_mlp', {'hidden_dim': 256, 'output_dim': 128})
        ]

        for ffn_type, config in test_configs:
            with self._test_single_layer_serialization(ffn_type, config):
                pass  # Context manager handles the test

    def _test_single_layer_serialization(self, ffn_type: str, config: Dict[str, Any]):
        """Context manager for testing single layer serialization."""
        return LayerSerializationTest(ffn_type, config)

    def test_parameter_override(self):
        """Test parameter override functionality."""

        # Test with custom parameters
        layer = create_ffn_layer(
            'mlp',
            hidden_dim=1024,
            output_dim=512,
            activation='swish',
            dropout_rate=0.2,
            use_bias=False,
            kernel_initializer='he_normal'
        )

        # Verify configuration is stored correctly. Activation is serialized via
        # keras.activations.serialize, which canonicalizes aliases (swish -> silu);
        # assert functional equivalence rather than the literal alias.
        config = layer.get_config()
        assert keras.activations.get(config['activation']) == keras.activations.get('swish')
        assert config['dropout_rate'] == 0.2
        assert config['use_bias'] == False
        assert config['kernel_initializer']['class_name'] == 'HeNormal'

    def test_different_activation_functions(self):
        """Test FFN creation with various activation functions."""

        activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid']

        for activation in activations:
            layer = create_ffn_layer(
                'mlp',
                hidden_dim=256,
                output_dim=128,
                activation=activation
            )

            # Test forward pass
            test_input = keras.random.normal(shape=(2, 16, 256))
            output = layer(test_input)

            assert output.shape == (2, 16, 128)

    def test_swiglu_specific_parameters(self):
        """Test SwiGLU-specific parameter handling."""

        # Test with all SwiGLU parameters
        layer = create_ffn_layer(
            'swiglu',
            output_dim=768,
            ffn_expansion_factor=8,
            ffn_multiple_of=128,
            dropout_rate=0.1
        )

        # Verify layer creation and configuration
        config = layer.get_config()
        assert config['ffn_expansion_factor'] == 8
        assert config['ffn_multiple_of'] == 128
        assert config['dropout_rate'] == 0.1

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 32, 768))
        output = layer(test_input)
        assert output.shape == (2, 32, 768)

    def test_differential_ffn_parameters(self):
        """Test DifferentialFFN-specific parameter handling."""

        layer = create_ffn_layer(
            'differential',
            hidden_dim=512,
            output_dim=256,
            branch_activation='relu',
            gate_activation='sigmoid', # Corrected from combination_activation
            dropout_rate=0.15
        )

        # Test forward pass
        test_input = keras.random.normal(shape=(2, 16, 512))
        output = layer(test_input)
        assert output.shape == (2, 16, 256)

    def test_window_attention_parameters(self):
        """Test layers that might use window-specific parameters."""

        # Test swin_mlp which might have window-related behavior
        layer = create_ffn_layer(
            'swin_mlp',
            hidden_dim=384,
            output_dim=192,
            dropout_rate=0.1,
            activation='gelu'
        )

        test_input = keras.random.normal(shape=(2, 49, 384))  # 7x7 patches
        output = layer(test_input)
        assert output.shape == (2, 49, 192)

    def test_error_handling_unknown_type(self):
        """Test error handling for unknown FFN types."""

        with pytest.raises(ValueError, match="Unknown FFN type"):
            create_ffn_layer('unknown_ffn_type', hidden_dim=512, output_dim=256)

    def test_error_handling_invalid_parameters(self):
        """Test error handling for invalid parameter values."""

        # Test negative dimensions
        with pytest.raises(ValueError):
            create_ffn_layer('mlp', hidden_dim=-512, output_dim=256)

        with pytest.raises(ValueError):
            create_ffn_layer('mlp', hidden_dim=512, output_dim=-256)

        # Test invalid dropout rate
        with pytest.raises(ValueError):
            create_ffn_layer('mlp', hidden_dim=512, output_dim=256, dropout_rate=2.0)

        # Test zero dimensions
        with pytest.raises(ValueError):
            create_ffn_layer('swiglu', output_dim=0)

    def test_config_completeness(self):
        """Test that get_config returns complete configuration."""

        layer = create_ffn_layer(
            'mlp',
            hidden_dim=512,
            output_dim=256,
            activation='gelu',
            dropout_rate=0.1,
            use_bias=True
        )

        config = layer.get_config()

        # Verify all parameters are in config
        required_keys = ['hidden_dim', 'output_dim', 'activation', 'dropout_rate', 'use_bias']
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_gradient_flow(self):
        """Test gradient computation through FFN layers."""

        test_configs = [
            ('mlp', {'hidden_dim': 128, 'output_dim': 64}),
            ('swiglu', {'output_dim': 128}),
            ('glu', {'hidden_dim': 128, 'output_dim': 64})
        ]

        for ffn_type, config in test_configs:
            layer = create_ffn_layer(ffn_type, **config)

            # Adjust input size
            if 'output_dim' in config:
                test_input = keras.random.normal(shape=(2, 8, config['output_dim']))
            else:
                test_input = keras.random.normal(shape=(2, 8, 128)) # MLP and GLU need input matching hidden_dim for this test config

            with tf.GradientTape() as tape:
                tape.watch(test_input)
                output = layer(test_input)
                loss = ops.mean(ops.square(output))

            gradients = tape.gradient(loss, layer.trainable_variables)

            # Verify gradients exist and are not None
            assert gradients is not None
            assert len(gradients) > 0
            assert all(g is not None for g in gradients)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, training: Optional[bool]):
        """Test FFN layers in different training modes."""

        layer = create_ffn_layer('mlp', hidden_dim=256, output_dim=128, dropout_rate=0.2)
        test_input = keras.random.normal(shape=(2, 16, 256))

        output = layer(test_input, training=training)
        assert output.shape == (2, 16, 128)

    def test_batch_size_invariance(self):
        """Test that layers work with different batch sizes."""

        layer = create_ffn_layer('swiglu', output_dim=512)

        batch_sizes = [1, 4, 16, 32]
        for batch_size in batch_sizes:
            test_input = keras.random.normal(shape=(batch_size, 20, 512))
            output = layer(test_input)

            assert output.shape == (batch_size, 20, 512)

    def test_activation_parameter_validation(self):
        """Test validation of activation function parameters."""

        # Valid activation strings
        valid_activations = ['relu', 'gelu', 'swish', 'tanh', 'sigmoid']
        for activation in valid_activations:
            layer = create_ffn_layer(
                'mlp',
                hidden_dim=256,
                output_dim=128,
                activation=activation
            )
            assert layer is not None

        # Invalid activation should be caught during layer creation or usage
        with pytest.raises((ValueError, AttributeError)):
            create_ffn_layer(
                'mlp',
                hidden_dim=256,
                output_dim=128,
                activation='nonexistent_activation'
            )

    def test_configuration_json_roundtrip(self):
        """Test configuration serialization/deserialization via JSON."""

        config = {
            'type': 'swiglu',
            'output_dim': 768,
            'ffn_expansion_factor': 4,
            'dropout_rate': 0.1,
            'name': 'json_test_layer'
        }

        # Convert to JSON and back
        json_str = json.dumps(config)
        loaded_config = json.loads(json_str)

        # Create layer from loaded config
        layer = create_ffn_from_config(loaded_config)

        assert layer.name == config['name']

        # Test functionality
        test_input = keras.random.normal(shape=(2, 16, 768))
        output = layer(test_input)
        assert output.shape == (2, 16, 768)

    def test_edge_case_dimensions(self):
        """Test FFN layers with edge case dimensions."""

        # Very small dimensions
        layer = create_ffn_layer('mlp', hidden_dim=2, output_dim=1)
        test_input = keras.random.normal(shape=(1, 4, 2))
        output = layer(test_input)
        assert output.shape == (1, 4, 1)

        # Large dimensions (within reasonable limits)
        layer = create_ffn_layer('swiglu', output_dim=2048)
        test_input = keras.random.normal(shape=(1, 8, 2048))
        output = layer(test_input)
        assert output.shape == (1, 8, 2048)

    def test_dropout_rate_effects(self):
        """Test dropout behavior across different rates."""

        dropout_rates = [0.0, 0.1, 0.5, 0.9]

        for rate in dropout_rates:
            layer = create_ffn_layer(
                'mlp',
                hidden_dim=256,
                output_dim=128,
                dropout_rate=rate
            )

            test_input = keras.random.normal(shape=(4, 16, 256))

            # Test in training mode
            output_train = layer(test_input, training=True)

            # Test in inference mode
            output_inference = layer(test_input, training=False)

            assert output_train.shape == output_inference.shape == (4, 16, 128)

    def test_all_ffn_types_integration(self):
        """Integration test using all FFN types in a single model."""

        # Create a model using different FFN types
        inputs = keras.Input(shape=(16, 512))

        # Stack different FFN types
        x = inputs

        ffn_layers = [
            create_ffn_layer('mlp', hidden_dim=512, output_dim=512, name='mlp_1'),
            create_ffn_layer('glu', hidden_dim=512, output_dim=512, name='glu_1'),
            create_ffn_layer('geglu', hidden_dim=512, output_dim=512, name='geglu_1'),
            create_ffn_layer('residual', hidden_dim=512, output_dim=512, name='residual_1')
        ]

        for ffn in ffn_layers:
            x = ffn(x)

        # FIX: Add a pooling layer to collapse the sequence dimension
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Final output layer
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test forward pass
        test_input = keras.random.normal(shape=(4, 16, 512))
        output = model(test_input)

        assert output.shape == (4, 10)

    def test_custom_initializers_and_regularizers(self):
        """Test FFN creation with custom initializers and regularizers."""

        layer = create_ffn_layer(
            'mlp',
            hidden_dim=256,
            output_dim=128,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        # Verify layer creation
        assert layer is not None

        # Test forward pass to ensure initializers/regularizers don't break functionality
        test_input = keras.random.normal(shape=(2, 16, 256))
        output = layer(test_input)
        assert output.shape == (2, 16, 128)

    def test_specialized_layers_direct_import(self):
        """Test direct instantiation of specialized layers."""

        # Test LogicFFN
        logic_ffn = LogicFFN(
            output_dim=256,
            logic_dim=128,
            use_bias=True
        )

        test_input = keras.random.normal(shape=(2, 16, 512))
        output = logic_ffn(test_input)
        assert output.shape == (2, 16, 256)

        # Test CountingFFN
        counting_ffn = CountingFFN(
            output_dim=256,
            count_dim=64,
            counting_scope='local'
        )

        output = counting_ffn(test_input)
        assert output.shape == (2, 16, 256)

    def test_factory_vs_direct_instantiation(self):
        """Compare factory creation vs direct instantiation."""

        # Factory creation
        factory_layer = create_ffn_layer(
            'mlp',
            hidden_dim=512,
            output_dim=256,
            activation='gelu'
        )

        # Direct instantiation
        direct_layer = MLPBlock(
            hidden_dim=512,
            output_dim=256,
            activation='gelu'
        )

        # Both should produce similar results
        test_input = keras.random.normal(shape=(2, 16, 512))

        factory_output = factory_layer(test_input)
        direct_output = direct_layer(test_input)

        assert factory_output.shape == direct_output.shape == (2, 16, 256)

    def test_factory_error_messages(self):
        """Test quality of factory error messages."""

        # Test descriptive error for missing parameters
        with pytest.raises(ValueError, match="Required parameters missing"):
            create_ffn_layer('mlp', hidden_dim=512)  # Missing output_dim

        # Test descriptive error for invalid type
        try:
            create_ffn_layer('invalid_type', hidden_dim=512)
        except ValueError as e:
            error_msg = str(e)
            assert 'Unknown FFN type' in error_msg
            assert 'invalid_type' in error_msg

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""

        layer = create_ffn_layer('mlp', hidden_dim=256, output_dim=128)

        # Test with very small values
        small_input = keras.random.normal(shape=(2, 16, 256)) * 1e-6
        output_small = layer(small_input)
        assert not ops.any(ops.isnan(output_small))
        assert not ops.any(ops.isinf(output_small))

        # Test with large values
        large_input = keras.random.normal(shape=(2, 16, 256)) * 1e3
        output_large = layer(large_input)
        assert not ops.any(ops.isnan(output_large))
        assert not ops.any(ops.isinf(output_large))

    def test_memory_efficiency(self):
        """Test memory usage with large layers."""

        # Create relatively large FFN
        layer = create_ffn_layer('swiglu', output_dim=2048, ffn_expansion_factor=4)

        # Test with moderately sized input
        test_input = keras.random.normal(shape=(8, 64, 2048))

        # Should complete without memory errors
        output = layer(test_input)
        assert output.shape == (8, 64, 2048)

    @pytest.mark.parametrize("ffn_type,config", [
        ('mlp', {'hidden_dim': 64, 'output_dim': 32, 'dropout_rate': 0.5}),
        ('swiglu', {'output_dim': 128, 'dropout_rate': 0.3}),
        ('glu', {'hidden_dim': 96, 'output_dim': 48, 'activation': 'swish'})
    ])
    def test_parameterized_layer_creation(self, ffn_type: str, config: Dict[str, Any]):
        """Parameterized test for various layer configurations."""

        layer = create_ffn_layer(ffn_type, **config)

        # Determine input dimension
        if 'output_dim' in config:
            input_dim = config['output_dim']
            expected_output_dim = config['output_dim']
        else:
            input_dim = 64 # Use a consistent input dim for non-output_dim layers
            expected_output_dim = config['output_dim']

        test_input = keras.random.normal(shape=(2, 8, input_dim))
        output = layer(test_input)

        assert output.shape == (2, 8, expected_output_dim)

    def test_layer_count_in_info(self):
        """Test that get_ffn_info returns expected number of layers."""

        info = get_ffn_info()

        # Should have exactly these factory-supported types
        expected_types = {
            'mlp',
            'swiglu',
            'differential',
            'glu', 'geglu', 'gelu_tanh', 'residual', 'swin_mlp',
            'counting', 'gated_mlp', 'power_mlp',  'orthoglu', 'logic',
            'kan', 'tversky',
            # plan_2026-06-19_2ea7a9a0: 4 new classes + 2 GLUFFN aliases
            'monarch', 'mixer', 'squared_relu', 'lowrank', 'reglu', 'bilinear'
        }
        actual_types = set(info.keys())

        assert actual_types == expected_types, f"Expected {expected_types}, got {actual_types}"

    def test_type_safety(self):
        """Test type safety of factory functions."""
        # This test checks that providing correct parameters for each type works.
        configs = {
            'mlp': {'hidden_dim': 256, 'output_dim': 128},
            'swiglu': {'output_dim': 256},
            'differential': {'hidden_dim': 256, 'output_dim': 128},
            'glu': {'hidden_dim': 256, 'output_dim': 128},
            'geglu': {'hidden_dim': 256, 'output_dim': 128},
            'residual': {'hidden_dim': 256, 'output_dim': 128},
            'swin_mlp': {'hidden_dim': 256},

            'counting': {'count_dim': 256, 'output_dim': 128},
            'gated_mlp': {'filters': 256},
            'power_mlp': {'units': 256},
            'orthoglu': {'hidden_dim': 256, 'output_dim': 128},
            'logic': {'logic_dim': 256, 'output_dim': 128},
            'kan': {'features': 128},
            'tversky': {'units': 128, 'num_features': 64}
        }
        valid_types: List[FFNType] = ['mlp',
            'swiglu',
            'differential',
            'glu', 'geglu', 'residual', 'swin_mlp',
            'counting', 'gated_mlp', 'power_mlp',  'orthoglu', 'logic',
            'kan', 'tversky'
        ]
        for ffn_type in valid_types:
            layer = create_ffn_layer(ffn_type, **configs[ffn_type])
            assert layer is not None


class LayerSerializationTest:
    """Context manager for comprehensive layer serialization testing."""

    def __init__(self, ffn_type: str, config: Dict[str, Any]):
        self.ffn_type = ffn_type
        self.config = config
        self.tmpdir = None

    def __enter__(self):
        """Set up serialization test."""
        self.tmpdir = tempfile.mkdtemp()

        try:
            # Create original layer in a model
            layer = create_ffn_layer(self.ffn_type, **self.config)

            # Determine input dimension for test
            if 'output_dim' in self.config:
                input_dim = self.config['output_dim']
            else:
                input_dim = self.config.get('hidden_dim', 512)

            # Create model
            inputs = keras.Input(shape=(16, input_dim))
            outputs = layer(inputs)
            model = keras.Model(inputs, outputs)

            # Get prediction from original
            sample_input = keras.random.normal(shape=(2, 16, input_dim))
            original_prediction = model(sample_input)

            # Save and load
            filepath = os.path.join(self.tmpdir, 'test_model.keras')
            model.save(filepath)

            lodaed_model = keras.models.load_model(filepath)
            loaded_prediction = lodaed_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Serialization failed for {self.ffn_type}"
            )

            logger.info(f"✅ Serialization test passed for {self.ffn_type}")

        except Exception as e:
            logger.error(f"❌ Serialization test failed for {self.ffn_type}: {e}")
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.tmpdir:
            import shutil
            shutil.rmtree(self.tmpdir)


class TestSpecializedFFNLayers:
    """Test suite for specialized FFN layers not covered by factory."""

    def test_logic_ffn(self):
        """Test LogicFFN layer functionality."""

        layer = LogicFFN(
            output_dim=128,
            logic_dim=64,
            use_bias=True
        )

        test_input = keras.random.normal(shape=(2, 16, 256))
        output = layer(test_input)

        assert output.shape == (2, 16, 128)

        # Test configuration
        config = layer.get_config()
        assert config['output_dim'] == 128
        assert config['logic_dim'] == 64
        assert config['use_bias'] == True

    def test_counting_ffn(self):
        """Test CountingFFN layer functionality."""

        counting_scopes = ['global', 'local', 'causal']

        for scope in counting_scopes:
            layer = CountingFFN(
                output_dim=256,
                count_dim=32,
                counting_scope=scope
            )

            test_input = keras.random.normal(shape=(2, 20, 512))
            output = layer(test_input)

            assert output.shape == (2, 20, 256)

            # Verify configuration
            config = layer.get_config()
            assert config['counting_scope'] == scope

    def test_logic_ffn_serialization(self):
        """Test LogicFFN serialization."""

        # Create model with LogicFFN
        inputs = keras.Input(shape=(16, 256))
        logic_layer = LogicFFN(output_dim=128, logic_dim=64)
        outputs = logic_layer(inputs)
        model = keras.Model(inputs, outputs)

        # Test serialization cycle
        sample_input = keras.random.normal(shape=(2, 16, 256))
        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'logic_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model(sample_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="LogicFFN serialization failed"
            )


class TestFactoryPerformance:
    """Performance and efficiency tests for FFN factory."""

    def test_factory_overhead(self):
        """Test that factory doesn't add significant overhead."""

        import time

        # Time factory creation
        start_time = time.time()
        for _ in range(100):
            layer = create_ffn_layer('mlp', hidden_dim=256, output_dim=128)
        factory_time = time.time() - start_time

        # Time direct creation
        start_time = time.time()
        for _ in range(100):
            layer = MLPBlock(hidden_dim=256, output_dim=128)
        direct_time = time.time() - start_time

        # Factory should not be more than 2x slower than direct
        assert factory_time < direct_time * 2.0

    def test_memory_usage_consistency(self):
        """Test memory usage is consistent between factory and direct creation."""

        # Create layers both ways
        factory_layer = create_ffn_layer('swiglu', output_dim=1024)
        direct_layer = SwiGLUFFN(output_dim=1024)

        # Both should have similar number of parameters
        def count_params(layer):
            test_input = keras.random.normal(shape=(1, 16, 1024))
            _ = layer(test_input)  # Build the layer
            return sum(keras.ops.convert_to_numpy(keras.ops.size(w)).item()
                      for w in layer.trainable_variables)

        factory_params = count_params(factory_layer)
        direct_params = count_params(direct_layer)

        assert factory_params == direct_params


class TestFactoryLogging:
    """Test logging functionality of FFN factory."""

    def test_info_logging(self):
        """Test that factory logs creation information."""

        with patch.object(logger, 'info') as mock_info:
            create_ffn_layer('mlp', hidden_dim=256, output_dim=128, name='test_layer')

            # Should have logged layer creation
            mock_info.assert_called()

            # Check if relevant information was logged
            logged_calls = [call.args[0] for call in mock_info.call_args_list]
            assert any('mlp' in call.lower() for call in logged_calls)

    def test_error_logging(self):
        """Test that factory logs errors appropriately."""

        with patch.object(logger, 'error') as mock_error:
            try:
                create_ffn_layer('invalid_type', hidden_dim=256)
            except ValueError:
                pass  # Expected error

            # Should have logged the error
            mock_error.assert_called()

    def test_debug_logging(self):
        """Test debug-level logging functionality."""

        # This would normally require debug level to be enabled
        with patch.object(logger, 'debug') as mock_debug:
            layer = create_ffn_layer('swiglu', output_dim=768)

            # Debug logging might be called (depends on implementation)
            # Just verify it doesn't crash
            assert layer is not None


class TestFactoryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_config_with_defaults(self):
        """Test layer creation with minimal valid configuration."""

        # SwiGLU only needs output_dim
        layer = create_ffn_layer('swiglu', output_dim=512)
        assert layer is not None

        # Should use default values for other parameters
        config = layer.get_config()
        assert 'ffn_expansion_factor' in config
        assert 'dropout_rate' in config

    def test_config_with_none_values(self):
        """Test handling of None values in configuration."""

        config = {
            'type': 'mlp',
            'hidden_dim': 256,
            'output_dim': 128,
            'activation': None,  # Should use default
            'kernel_regularizer': None  # Explicit None
        }

        layer = create_ffn_from_config(config)
        assert layer is not None

    def test_very_deep_ffn_stack(self):
        """Test creating very deep FFN stacks."""

        inputs = keras.Input(shape=(16, 256))
        x = inputs

        # Create deep stack of FFN layers
        for i in range(10):
            x = create_ffn_layer(
                'residual',  # Use residual for better gradient flow
                hidden_dim=256,
                output_dim=256,
                name=f'ffn_{i}'
            )(x)

        # FIX: Add a pooling layer to collapse the sequence dimension
        x = keras.layers.GlobalAveragePooling1D()(x)

        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs, outputs)

        # Should be able to create and run forward pass
        test_input = keras.random.normal(shape=(2, 16, 256))
        output = model(test_input)
        assert output.shape == (2, 10)


class TestKanAndTverskyFactory:
    """Coverage for the relocated KAN/Tversky FFN factory registrations."""

    # ---------- KAN ----------

    @pytest.mark.parametrize(
        "kan_kwargs,input_shape,expected_last",
        [
            ({'features': 16}, (2, 8), 16),
            ({'features': 16, 'grid_size': 8, 'spline_order': 2}, (2, 8), 16),
            # KAN supports N-D via einsum
            ({'features': 32}, (2, 4, 8), 32),
        ],
    )
    def test_kan_create_and_forward(self, kan_kwargs, input_shape, expected_last):
        layer = create_ffn_layer('kan', **kan_kwargs)
        x = keras.random.normal(shape=input_shape)
        y = layer(x)
        assert y.shape[-1] == expected_last
        assert tuple(y.shape)[:-1] == tuple(input_shape)[:-1]

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {'features': 0},
            {'features': -3},
            {'features': 16, 'grid_size': 0},
            {'features': 16, 'spline_order': -1},
            {'features': 16, 'grid_range': (2.0, -2.0)},
            {'features': 16, 'grid_range': (1.0,)},
            {'features': 16, 'epsilon': 0.0},
        ],
    )
    def test_kan_validation_rejects_bad_config(self, bad_kwargs):
        with pytest.raises(ValueError):
            validate_ffn_config('kan', **bad_kwargs)

    # ---------- Tversky ----------

    def test_tversky_create_and_forward_rank2(self):
        layer = create_ffn_layer('tversky', units=10, num_features=12)
        x = keras.random.normal(shape=(2, 32))
        y = layer(x)
        assert tuple(y.shape) == (2, 10)

    @pytest.mark.parametrize(
        "bad_kwargs",
        [
            {'units': 0, 'num_features': 4},
            {'units': 4, 'num_features': 0},
            {'units': 4, 'num_features': 4, 'intersection_reduction': 'bogus'},
            {'units': 4, 'num_features': 4, 'difference_reduction': 'bogus'},
        ],
    )
    def test_tversky_validation_rejects_bad_config(self, bad_kwargs):
        with pytest.raises(ValueError):
            validate_ffn_config('tversky', **bad_kwargs)

    def test_tversky_missing_required(self):
        with pytest.raises(ValueError, match="Required parameters missing"):
            create_ffn_layer('tversky', units=10)  # missing num_features

    def test_kan_missing_required(self):
        with pytest.raises(ValueError, match="Required parameters missing"):
            create_ffn_layer('kan')  # missing features

    def test_get_ffn_info_exposes_kan_and_tversky(self):
        info = get_ffn_info()
        assert 'kan' in info
        assert 'tversky' in info
        assert 'features' in info['kan']['required_params']
        assert 'units' in info['tversky']['required_params']
        assert 'num_features' in info['tversky']['required_params']


# ---------------------------------------------------------------------
# FFN_REGISTRY['output_dim_param'] contract guards
# ---------------------------------------------------------------------

# Written INDEPENDENTLY of len(FFN_REGISTRY) (counted from the FFNType Literal
# in ffn/factory.py). Its only job is anti-vacuity: it makes the parametrized
# guards below fail loudly if the registry is ever gutted, rather than passing
# over an empty or truncated derived list. Bump it deliberately when a genuinely
# new FFN type is registered.
_EXPECTED_FFN_TYPE_COUNT = 21

# Derived from the registry itself -- never hand-listed, so a newly-added type
# is covered by these guards on the day it lands.
_ALL_REGISTRY_FFN_TYPES = sorted(FFN_REGISTRY.keys())


class TestOutputDimParamRegistryField:
    """
    Contract guards for ``FFN_REGISTRY[t]['output_dim_param']``.

    The field names the constructor parameter that sets each FFN's OUTPUT WIDTH.
    It exists because those names are NOT uniform across types: 'output_dim' for
    most, but 'filters' (gated_mlp), 'features' (kan), 'units' (power_mlp,
    tversky), and ``None`` (mixer, which has no output-width concept). See
    ``src/dl_techniques/layers/ffn/factory.py``'s registry schema comment.
    """

    def test_registry_type_list_is_not_vacuous_for_width_param_guards(self):
        """Anti-vacuity: the derived type list must be non-empty AND the
        expected size. A ``len(x) > 0`` assertion alone would be satisfied by a
        one-entry registry, which would make every parametrized guard below
        near-meaningless."""
        assert len(_ALL_REGISTRY_FFN_TYPES) > 0, "FFN_REGISTRY is empty"
        assert len(FFN_REGISTRY) == _EXPECTED_FFN_TYPE_COUNT, (
            f"FFN_REGISTRY has {len(FFN_REGISTRY)} entries but the independently "
            f"written expected count is {_EXPECTED_FFN_TYPE_COUNT}. If a type was "
            f"deliberately added or removed, update _EXPECTED_FFN_TYPE_COUNT; do "
            f"not delete this assertion. Registry types: {_ALL_REGISTRY_FFN_TYPES}"
        )
        assert len(_ALL_REGISTRY_FFN_TYPES) == _EXPECTED_FFN_TYPE_COUNT

    @pytest.mark.parametrize('ffn_type', _ALL_REGISTRY_FFN_TYPES)
    def test_every_registry_entry_declares_its_output_width_param(self, ffn_type):
        """ASSERTION 1 (presence). A MISSING key must FAIL here -- it must never
        be allowed to fall back to a default of 'output_dim', because that
        default is exactly wrong for gated_mlp/kan/power_mlp/tversky/mixer and
        would silently reintroduce the no-op this field exists to prevent."""
        entry = FFN_REGISTRY[ffn_type]
        assert 'output_dim_param' in entry, (
            f"FFN_REGISTRY['{ffn_type}'] is missing the mandatory "
            f"'output_dim_param' key. This key is NOT optional and has NO "
            f"default: consumers must be able to distinguish 'this type's width "
            f"param is named X' from 'nobody said'. Add it, using None only for "
            f"a type whose output width equals its input width by construction."
        )

    @pytest.mark.parametrize('ffn_type', _ALL_REGISTRY_FFN_TYPES)
    def test_output_width_param_names_a_real_parameter(self, ffn_type):
        """ASSERTION 2 (validity). The value must be ``None`` or an actual
        parameter name of that same entry -- a typo would otherwise sit in the
        registry until a consumer silently dropped the width key."""
        entry = FFN_REGISTRY[ffn_type]
        if 'output_dim_param' not in entry:
            # PRESENCE is owned exclusively by
            # test_every_registry_entry_declares_its_output_width_param, which is
            # parametrized over the SAME derived type list and therefore cannot be
            # silently disabled for one type. Skipping here keeps a missing key
            # from cascading into a bare KeyError that obscures which assertion
            # actually diagnosed the defect.
            pytest.skip(
                f"'{ffn_type}' has no 'output_dim_param' -- diagnosed by "
                f"test_every_registry_entry_declares_its_output_width_param"
            )
        width_param = entry['output_dim_param']
        if width_param is None:
            return
        accepted = set(entry['required_params']) | set(entry['optional_params'])
        assert width_param in accepted, (
            f"FFN_REGISTRY['{ffn_type}']['output_dim_param'] == {width_param!r}, "
            f"which is neither in required_params {sorted(entry['required_params'])} "
            f"nor in optional_params {sorted(entry['optional_params'])}. A width "
            f"param name that does not exist on the entry cannot ever be passed "
            f"to the layer -- it is a typo, not a policy."
        )

    def test_output_width_param_names_are_heterogeneous(self):
        """
        ASSERTION 3 (anti-simplification) -- the load-bearing one.

        # DECISION plan-2026-07-30T140922-8af1028f/D-005
        This guard exists to go RED against the specific way this work could
        ship as a silent no-op: a future reader concluding the field is
        redundant with the literal string "output_dim" and "simplifying" it
        away. It is NOT redundant -- gated_mlp/kan/power_mlp/tversky name their
        output width 'filters'/'features'/'units'/'units', and mixer has no
        width param at all. See decisions.md D-005.
        """
        declared = {t: FFN_REGISTRY[t] for t in _ALL_REGISTRY_FFN_TYPES}
        undeclared = sorted(t for t, e in declared.items()
                            if 'output_dim_param' not in e)
        if undeclared:
            # PRESENCE is owned exclusively by
            # test_every_registry_entry_declares_its_output_width_param -- see the
            # note there. Do not let a missing key masquerade as a heterogeneity
            # failure; they have different remedies.
            pytest.skip(
                f"{undeclared} have no 'output_dim_param' -- diagnosed by "
                f"test_every_registry_entry_declares_its_output_width_param"
            )
        values = {t: e['output_dim_param'] for t, e in declared.items()}
        divergent = {t: v for t, v in values.items() if v != 'output_dim'}
        assert divergent, (
            "Every FFN_REGISTRY entry now declares output_dim_param == "
            "'output_dim'. That makes the field look redundant -- and it is NOT. "
            "This assertion fires because someone either normalized the values "
            "away or removed the types that diverge. The real mapping includes "
            "gated_mlp->'filters', kan->'features', power_mlp->'units', "
            "tversky->'units', mixer->None (no output-width concept). A consumer "
            "that assumes the literal string \"output_dim\" silently builds those "
            "types at the WRONG width, or drops the key entirely. Restore the "
            "per-type values; do not delete this test."
        )


# =====================================================================
# STEP 5 -- the pre-flip construction-site sweep (plan-2026-07-30T140922-8af1028f)
# =====================================================================
#
# `create_ffn_layer` is about to RAISE on a dropped caller key instead of
# warning. Assumption A-4 -- "no production construction site anywhere in
# `src/` loses a key today" -- had been verified for only 4 wrapper sites plus
# the 2 VLM sites. This block is A-4's executable re-test across EVERY
# `create_ffn_layer` / `create_ffn_from_config` call expression in `src/`.
#
# Derivation of the site list (re-run to reproduce; F-05 measured ~88% drift in
# hand-cited line numbers, so nothing here is keyed on a line number):
#
#     grep -rn "create_ffn_layer(\|create_ffn_from_config(" src/ --include=*.py
#
# The FILE set is derived at test time by `_derive_ffn_construction_files()`;
# only the per-file BUILDER is hand-written, because "how do I minimally
# instantiate the owner of this call" cannot be derived. A construction site
# added in a NEW file therefore fails
# `test_every_file_with_a_construction_site_has_a_builder` loudly, which is the
# regression this whole block exists to prevent.

import ast as _sweep_ast
import logging as _sweep_logging
from pathlib import Path as _SweepPath

_SWEEP_SRC_ROOT = _SweepPath(__file__).resolve().parents[3] / "src" / "dl_techniques"
_SWEEP_CALL_NAMES = frozenset({"create_ffn_layer", "create_ffn_from_config"})


def _derive_ffn_construction_files() -> set:
    """Every `src/dl_techniques/` file containing a real FFN construction call.

    Derived by AST, NOT by grep: an `ast.Call` node is a call EXPRESSION, so a
    docstring that merely names ``create_ffn_from_config()`` (as
    ``layers/moe/config.py`` and ``layers/moe/experts.py`` both do) and a `def`
    of the function itself are both excluded structurally rather than by
    pattern-guessing. The equivalent human-readable command is::

        grep -rn "create_ffn_layer(\\|create_ffn_from_config(" src/ --include=*.py

    which over-reports because of exactly those prose mentions.

    :return: repo-relative POSIX paths under ``src/dl_techniques/``.
    :rtype: set
    """
    found = set()
    for path in _SWEEP_SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(_SWEEP_SRC_ROOT).as_posix()
        if rel == "layers/ffn/factory.py":
            continue  # the DEFINITION site, not a construction site
        try:
            tree = _sweep_ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in _sweep_ast.walk(tree):
            if not isinstance(node, _sweep_ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in _SWEEP_CALL_NAMES:
                found.add(rel)
                break
    return found


class _SweepDropRecorder(_sweep_logging.Handler):
    """Captures `create_ffn_layer(...): dropping ...` on the **`dl`** logger.

    The logger is named `dl` (`dl_techniques/utils/logger.py`:
    `logging.getLogger("dl")`), NOT `dl_techniques`. A handler on the wrong name
    captures ZERO records while the warning sits plainly on stderr -- a silently
    blind instrument that makes this entire sweep vacuous. This trap already bit
    once during step 3 of this plan.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dropped: List[str] = []

    def emit(self, record: _sweep_logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return
        if "dropping" in message and "unsupported parameter" in message:
            self.dropped.append(message)


def _capture_ffn_drops(fn) -> List[str]:
    """Run ``fn``; return the dropped-key warnings it caused, de-duplicated."""
    handler = _SweepDropRecorder()
    dl_logger = _sweep_logging.getLogger("dl")
    dl_logger.addHandler(handler)
    try:
        fn()
    finally:
        dl_logger.removeHandler(handler)
    return sorted(set(handler.dropped))


# --- the hand-written builders -------------------------------------------
# One entry per FILE returned by `_derive_ffn_construction_files()`. Each
# builder must actually REACH the `create_ffn_layer` call: a builder that
# raises before construction reports zero drops and is vacuous, which is what
# `test_every_builder_actually_reaches_its_ffn` exists to catch.
#
# DECISION plan-2026-07-30T140922-8af1028f/D-020
# Every builder uses the site's OWN default `ffn_type` (or the literal the site
# hardcodes). Do NOT "strengthen" this into a 21-type parametrization: 4 sites --
# `layers/time_series/mixed_sequential_block.py`, `models/nam/cell.py`,
# `models/sam/image_encoder.py` and `models/tree_transformer/components.py` --
# pass `activation` unconditionally and therefore DROP it for `differential`,
# `gelu_tanh`, `squared_relu` and `swiglu`. Those cells are latent (no `src/`
# caller selects those types at those sites) but they are REAL, and turning this
# guard into a grid makes it RED without fixing them. The remedy is the one-line
# `assemble_ffn_config` adoption those sites still need, tracked separately; the
# measurement is in this plan's verification.md § "LATENT residue".
#
# This sweep's own job is narrower and is what A-4 actually claims: the
# PRODUCTION-REACHABLE surface, which is what step 6's raise can break today.

def _b_fnet_encoder_block():
    from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock
    b = FNetEncoderBlock(intermediate_dim=32)
    b.build((None, 8, 16))
    return b.ffn_layer


def _b_multimodal_fusion():
    from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion
    # BOTH call sites in this file: the per-modality FFN in the
    # cross_attention path, and the single FFN in the elementwise path.
    a = MultiModalFusion(dim=16, fusion_strategy='cross_attention')
    a.build([(None, 8, 16), (None, 8, 16)])
    b = MultiModalFusion(dim=16, fusion_strategy='multiplication')
    b.build([(None, 8, 16), (None, 8, 16)])
    return b.ffn_layers[0]


def _b_relational_graph_transformer_blocks():
    from dl_techniques.layers.graphs.relational_graph_transformer_blocks import (
        RELGTTransformerBlock,
    )
    b = RELGTTransformerBlock(
        embedding_dim=16, num_heads=2, num_global_centroids=4, ffn_dim=32
    )
    b.build([(None, 8, 16), (None, 16)])
    return b.combination_ffn


def _b_heads_nlp_factory():
    from dl_techniques.layers.heads.nlp.factory import (
        create_nlp_head, NLPTaskConfig, NLPTaskType,
    )
    cfg = NLPTaskConfig(
        name="cls", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
    )
    h = create_nlp_head(cfg, input_dim=16, use_ffn=True)
    h.build((None, 8, 16))
    return h.ffn


def _b_heads_vision_factory():
    from dl_techniques.layers.heads.vision.factory import ClassificationHead
    h = ClassificationHead(num_classes=3, hidden_dim=16, use_ffn=True)
    h.build((None, 8, 8, 16))
    return h.ffn


def _b_heads_vlm_factory():
    # Both VLM sites already carry their own dedicated 21-type grid in
    # tests/test_layers/test_heads/test_vlm.py (TestVLMFFNTypeGrid). Reached
    # here too so the derived file set stays fully covered.
    from dl_techniques.layers.heads.vlm.factory import (
        ImageTextMatchingHead, VLMTaskConfig, VLMTaskType,
    )
    head = ImageTextMatchingHead(
        task_config=VLMTaskConfig(
            name="itm", task_type=VLMTaskType.IMAGE_TEXT_MATCHING,
            hidden_size=16,
        ),
        vision_dim=16,
        text_dim=16,
        use_post_fusion_ffn=True,
    )
    # `hidden_dim` comes from `task_config.hidden_size`, and `MultiModalFusion`
    # requires every modality to already be that width.
    head.build({"vision_features": (None, 4, 16), "text_features": (None, 8, 16)})
    return head.post_fusion_ffn


def _b_moe_experts():
    from dl_techniques.layers.moe.experts import FFNExpert
    e = FFNExpert(ffn_config={'type': 'mlp', 'hidden_dim': 32, 'output_dim': 16})
    e.build((None, 16))
    return e.ffn_block


def _b_mixed_sequential_block():
    from dl_techniques.layers.time_series.mixed_sequential_block import (
        MixedSequentialBlock,
    )
    b = MixedSequentialBlock(embed_dim=16, num_heads=2, ff_dim=32)
    b.build((None, 8, 16))
    return b.ffn_layer


def _b_prism_blocks():
    from dl_techniques.layers.time_series.prism_blocks import FrequencyBandRouter
    r = FrequencyBandRouter(hidden_dim=16)
    r.build([(None, 8, 4), (None, 8, 4)])
    return r.router_mlp


def _b_xlstm_blocks():
    from dl_techniques.layers.time_series.xlstm_blocks import sLSTMBlock
    b = sLSTMBlock(units=16)
    b.build((None, 8, 16))
    return b.ffn


def _b_adaln_zero():
    from dl_techniques.layers.transformers.adaln_zero import (
        AdaLNZeroConditionalBlock,
    )
    # BOTH call sites: the hardcoded-`mlp` default branch and the generic
    # `ffn_type` branch.
    a = AdaLNZeroConditionalBlock(dim=16, num_heads=2, dim_head=8, mlp_dim=32)
    a.build([(None, 8, 16), (None, 16)])
    b = AdaLNZeroConditionalBlock(
        dim=16, num_heads=2, dim_head=8, mlp_dim=32, ffn_type='mlp',
        ffn_args={'hidden_dim': 32, 'output_dim': 16},
    )
    b.build([(None, 8, 16), (None, 16)])
    return b.mlp


def _b_free_transformer():
    from dl_techniques.layers.transformers.free_transformer import (
        FreeTransformerLayer,
    )
    # `encoder_ffn` only exists when the free-transformer path is enabled.
    b = FreeTransformerLayer(hidden_size=16, num_heads=2, intermediate_size=32,
                             use_free_transformer=True)
    b.build((None, 8, 16))
    return b.encoder_ffn


def _b_gated_linear_attention_block():
    from dl_techniques.layers.transformers.gated_linear_attention_block import (
        GatedLinearAttentionBlock,
    )
    b = GatedLinearAttentionBlock(dim=16, num_heads=2, max_seq_len=8,
                                  ffn_type='mlp',
                                  ffn_args={'hidden_dim': 32, 'output_dim': 16})
    b.build((None, 8, 16))
    return b.output_ffn


def _b_progressive_focused_transformer():
    from dl_techniques.layers.transformers.progressive_focused_transformer import (
        PFTBlock,
    )
    # The 8 hardcoded per-type branches AND the generic `else` fallback
    # (reached with a type outside the hardcoded switch).
    for ffn_type in ('mlp', 'swiglu', 'geglu', 'glu', 'swin_mlp', 'orthoglu',
                     'differential', 'residual', 'gelu_tanh'):
        b = PFTBlock(dim=16, num_heads=2, ffn_type=ffn_type)
        b.build((None, 8, 8, 16))
    return b._ffn


def _b_transformer():
    from dl_techniques.layers.transformers.transformer import TransformerLayer
    b = TransformerLayer(hidden_size=16, num_heads=2, intermediate_size=32)
    b.build((None, 8, 16))
    return b.ffn_layer


def _b_transformer_decoder():
    from dl_techniques.layers.transformers.transformer_decoder import (
        TransformerDecoderLayer,
    )
    b = TransformerDecoderLayer(hidden_size=16, num_heads=2, intermediate_size=32)
    b.build((None, 8, 16))
    return b.ffn_layer


def _b_models_detr():
    from dl_techniques.models.detr.model import DETR, DetrTransformer
    backbone = keras.Sequential([keras.layers.Conv2D(32, 1)])
    transformer = DetrTransformer(
        hidden_dim=32, num_heads=2, num_encoder_layers=1,
        num_decoder_layers=1, ffn_dim=64,
    )
    m = DETR(num_classes=3, num_queries=4, backbone=backbone,
             transformer=transformer, hidden_dim=32)
    return m.bbox_embed


def _b_models_dino_v2():
    from dl_techniques.models.dino.dino_v2 import DINOv2Block
    b = DINOv2Block(dim=16, num_heads=2)
    b.build((None, 8, 16))
    return b.ffn


def _b_models_fftnet():
    from dl_techniques.models.fftnet.model import FFTNetBlock
    b = FFTNetBlock(embed_dim=16)
    b.build((None, 8, 16))
    return b.ffn


def _b_models_gemma():
    from dl_techniques.models.gemma.components import Gemma3TransformerBlock
    b = Gemma3TransformerBlock(
        hidden_size=16, num_attention_heads=2, num_key_value_heads=2,
        ffn_hidden_size=32,
    )
    b.build((None, 8, 16))
    return b.ffn


def _b_models_nam():
    from dl_techniques.models.nam.cell import NAMCell, NAMConfig
    c = NAMCell(NAMConfig(hidden_size=16, num_heads=2, intermediate_size=32))
    c.build((None, 8, 16))
    return c.ffn


def _b_models_pw_fnet():
    from dl_techniques.models.pw_fnet.model import PW_FNet_Block
    b = PW_FNet_Block(dim=16, use_spatial_ffn=False, ffn_type='mlp')
    b.build((None, 8, 8, 16))
    return b.ffn


def _b_models_relgt():
    from dl_techniques.models.relgt.model import RELGT
    m = RELGT(output_dim=3, embedding_dim=16, num_heads=2, ffn_dim=32,
              num_global_centroids=4, num_transformer_blocks=1)
    return m.prediction_head


def _b_models_sam_image_encoder():
    from dl_techniques.models.sam.image_encoder import ViTBlock
    b = ViTBlock(dim=16, num_heads=2)
    b.build((None, 8, 8, 16))
    return b.ffn


def _b_models_sam_transformer():
    from dl_techniques.models.sam.transformer import TwoWayAttentionBlock
    b = TwoWayAttentionBlock(embedding_dim=16, num_heads=2, mlp_dim=32)
    b.build([(None, 4, 16), (None, 8, 16)])
    return b.ffn


def _b_models_prism():
    from dl_techniques.models.time_series.prism.model import PRISMModel
    m = PRISMModel(context_len=16, forecast_len=4, num_features=2, num_layers=1)
    return m.forecast_head


def _b_models_tree_transformer():
    from dl_techniques.models.tree_transformer.components import (
        TreeTransformerBlock,
    )
    b = TreeTransformerBlock(hidden_size=16, num_heads=2, intermediate_size=32)
    b.build(((None, 8, 16), (None, 8), (None, 8, 8)))
    return b.ffn


_FFN_CONSTRUCTION_SITE_BUILDERS = {
    "layers/fnet_encoder_block.py": _b_fnet_encoder_block,
    "layers/fusion/multimodal_fusion.py": _b_multimodal_fusion,
    "layers/graphs/relational_graph_transformer_blocks.py":
        _b_relational_graph_transformer_blocks,
    "layers/heads/nlp/factory.py": _b_heads_nlp_factory,
    "layers/heads/vision/factory.py": _b_heads_vision_factory,
    "layers/heads/vlm/factory.py": _b_heads_vlm_factory,
    "layers/moe/experts.py": _b_moe_experts,
    "layers/time_series/mixed_sequential_block.py": _b_mixed_sequential_block,
    "layers/time_series/prism_blocks.py": _b_prism_blocks,
    "layers/time_series/xlstm_blocks.py": _b_xlstm_blocks,
    "layers/transformers/adaln_zero.py": _b_adaln_zero,
    "layers/transformers/free_transformer.py": _b_free_transformer,
    "layers/transformers/gated_linear_attention_block.py":
        _b_gated_linear_attention_block,
    "layers/transformers/progressive_focused_transformer.py":
        _b_progressive_focused_transformer,
    "layers/transformers/transformer.py": _b_transformer,
    "layers/transformers/transformer_decoder.py": _b_transformer_decoder,
    "models/detr/model.py": _b_models_detr,
    "models/dino/dino_v2.py": _b_models_dino_v2,
    "models/fftnet/model.py": _b_models_fftnet,
    "models/gemma/components.py": _b_models_gemma,
    "models/nam/cell.py": _b_models_nam,
    "models/pw_fnet/model.py": _b_models_pw_fnet,
    "models/relgt/model.py": _b_models_relgt,
    "models/sam/image_encoder.py": _b_models_sam_image_encoder,
    "models/sam/transformer.py": _b_models_sam_transformer,
    "models/time_series/prism/model.py": _b_models_prism,
    "models/tree_transformer/components.py": _b_models_tree_transformer,
}


class TestFFNConstructionSiteSweep:
    """Zero dropped caller keys at EVERY FFN construction site in `src/`.

    This is the gate that unblocks making `create_ffn_layer` raise. Each
    builder constructs the real owning layer/model at ITS OWN defaults and the
    sweep asserts the `dl` logger saw no `dropping ... unsupported parameter`
    record.
    """

    def test_the_drop_recorder_bites(self) -> None:
        """RED-proof: a zero from a blind recorder proves nothing.

        Injects a deliberately misspelled key and requires the recorder to see
        it, then runs the identical call WITHOUT the injection as a control.
        Both halves are load-bearing: the first shows the recorder is wired to
        the right logger, the second shows it is not simply always-firing.
        """
        injected = _capture_ffn_drops(
            lambda: create_ffn_layer(
                'mlp', hidden_dim=8, output_dim=8, hiden_dim=512
            )
        )
        assert any('hiden_dim' in m for m in injected), (
            f"the recorder on the 'dl' logger saw {injected} for a deliberately "
            f"misspelled 'hiden_dim'. It is BLIND -- most likely attached to a "
            f"logger named 'dl_techniques', which does not exist. Every zero "
            f"reported by the sweep below would be vacuous."
        )
        control = _capture_ffn_drops(
            lambda: create_ffn_layer('mlp', hidden_dim=8, output_dim=8)
        )
        assert control == [], (
            f"the recorder fired {control} on a clean call; it cannot "
            f"distinguish a drop from a non-drop."
        )

    def test_every_file_with_a_construction_site_has_a_builder(self) -> None:
        """The site list is DERIVED, not hand-maintained.

        A new `create_ffn_layer` call in a file nobody thought to add here is
        the exact regression this block prevents, so it must fail loudly rather
        than be silently uncovered.
        """
        derived = _derive_ffn_construction_files()
        missing = sorted(derived - set(_FFN_CONSTRUCTION_SITE_BUILDERS))
        assert not missing, (
            f"{missing} contain an FFN construction call with no builder in "
            f"_FFN_CONSTRUCTION_SITE_BUILDERS. Add one that reaches the call at "
            f"the site's own default ffn_type -- `create_ffn_layer` RAISES on a "
            f"dropped caller key, so an unswept site is an unmeasured "
            f"construction failure."
        )

    def test_no_builder_points_at_a_vanished_site(self) -> None:
        """The inverse direction: a stale builder is a false sense of coverage."""
        derived = _derive_ffn_construction_files()
        stale = sorted(set(_FFN_CONSTRUCTION_SITE_BUILDERS) - derived)
        assert not stale, (
            f"{stale} have builders but no longer contain an FFN construction "
            f"call. Delete the builder, or fix the derivation if the call moved."
        )

    def test_the_derived_site_list_is_not_empty(self) -> None:
        """Anti-vacuity for the derivation itself.

        If `_derive_ffn_construction_files` ever returns an empty (or nearly
        empty) set -- a moved `src/` root, a broken regex -- both coverage
        assertions above pass trivially. The floor is deliberately well below
        the measured count (27 files on 2026-07-30, command in the block header)
        so ordinary refactors do not trip it.
        """
        assert len(_derive_ffn_construction_files()) >= 20

    @pytest.mark.parametrize(
        "site", sorted(_FFN_CONSTRUCTION_SITE_BUILDERS)
    )
    def test_construction_site_drops_no_caller_key(self, site: str) -> None:
        dropped = _capture_ffn_drops(_FFN_CONSTRUCTION_SITE_BUILDERS[site])
        assert dropped == [], (
            f"src/dl_techniques/{site} silently dropped a key it passed to "
            f"create_ffn_layer: {dropped}. Once the factory raises this is a "
            f"HARD construction failure. Fix the site (correct parameter name, "
            f"or route its generic conveniences through assemble_ffn_config) -- "
            f"do not soften the factory."
        )

    @pytest.mark.parametrize(
        "site", sorted(_FFN_CONSTRUCTION_SITE_BUILDERS)
    )
    def test_every_builder_actually_reaches_its_ffn(self, site: str) -> None:
        """Anti-vacuity: a builder that never constructs also reports zero drops.

        Each builder returns the FFN object its site produced. A `None` here
        means the sweep above measured nothing at all for this site.
        """
        assert _FFN_CONSTRUCTION_SITE_BUILDERS[site]() is not None, (
            f"the builder for {site} returned None, so its zero-drop result is "
            f"vacuous -- it never reached create_ffn_layer."
        )
