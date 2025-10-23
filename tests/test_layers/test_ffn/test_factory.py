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

        # Verify configuration is stored correctly
        config = layer.get_config()
        assert config['activation'] == 'swish'
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
            'glu', 'geglu', 'residual', 'swin_mlp',
            'counting', 'gated_mlp', 'power_mlp',  'orthoglu', 'logic'
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
            'logic': {'logic_dim': 256, 'output_dim': 128}
        }
        valid_types: List[FFNType] = ['mlp',
            'swiglu',
            'differential',
            'glu', 'geglu', 'residual', 'swin_mlp',
            'counting', 'gated_mlp', 'power_mlp',  'orthoglu', 'logic'
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